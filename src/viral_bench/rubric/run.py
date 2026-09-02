# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Grading one build, end to end.

The order here is load-bearing, so it is worth stating plainly:

1. **Resolve** the build and its rubric.
2. **Tier 0 gate, with no model involved.** The manifest parses strictly, setup
   exits 0, the app starts and binds, the entry URL is non-5xx, and the entry
   page renders something operable. Any failure scores **0** -- and the run
   stops there, because paying a model to grade features on an app that will not
   start is spending money to produce a number that means nothing.
3. **Reset the build's data dir.** ``AppHost`` deliberately does not, and a dirty
   ``/data`` left by an earlier pass would silently invalidate every persistence
   and multi-user item -- passing them for the wrong reason.
4. **Start** the app once and share it across passes.
5. **Three passes** of the agent loop, each in a *fresh browser context* so pass
   two cannot inherit pass one's localStorage and score a persistence item that
   the app never earned.
6. **Score, persist, tear down.**

The gate is separated from the graded items on purpose. ~27% of the corpus is
undeliverable or dead on arrival, and those builds land here, scored zero rather
than skipped -- a benchmark that silently drops the builds that failed hardest
reports the average of the survivors and calls it the average.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from viral_bench.providers import Capability, check_support, make_client, resolve
from viral_bench.rubric.checks import CheckContext, run_check
from viral_bench.rubric.grader import (
    GraderTools,
    ItemOutcome,
    merge_passes,
    run_pass,
)
from viral_bench.rubric.report import (
    build_grade_document,
    comparison_block,
    founder_block,
    new_run_id,
    rubric_run_dir,
    source_hash,
    write_grade,
)
from viral_bench.rubric.schema import Rubric, RubricItem, load_rubric
from viral_bench.rubric.score import RubricResult, score_rubric

#: Passes per build. Three, so an item can pass on a majority; the per-pass
#: answers are kept because the disagreement rate is the instrument's own noise
#: floor.
DEFAULT_PASSES = 3

#: Seconds to wait for the app to become reachable before the gate fails it.
START_TIMEOUT = 90.0


class GradeAborted(RuntimeError):
    """The build could not be graded at all -- distinct from scoring zero."""


@dataclass
class GateOutcome:
    """The Tier 0 result, decided entirely by code."""

    passed: bool
    outcomes: dict[str, ItemOutcome] = field(default_factory=dict)
    url: str = ""
    detail: str = ""


def _fail(item_id: str, reason: str) -> ItemOutcome:
    return ItemOutcome(item_id=item_id, passed=False, reason=reason)


def _pass(item_id: str, detail: str = "") -> ItemOutcome:
    return ItemOutcome(item_id=item_id, passed=True, reason=detail)


def check_manifest_strict(app_dir: Path) -> ItemOutcome:
    """G1: ``viralbench.json`` exists and parses as strict JSON.

    Strict on purpose. A leading ``#`` comment or a trailing comma is a
    documented, recurring cause of an otherwise-complete build scoring zero, and
    leniency here would grade an app the harness itself could never have run.
    """
    path = app_dir / "viralbench.json"
    if not path.is_file():
        return _fail("G1", "no viralbench.json")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _fail("G1", f"invalid JSON: {exc}")
    from viral_bench.founder.manifest import ManifestError, parse_manifest

    try:
        parse_manifest(raw, source=str(path))
    except ManifestError as exc:
        return _fail("G1", f"manifest invalid: {exc}")
    return _pass("G1", "parses strictly")


async def run_gate(
    build_id: str,
    *,
    host: Any,
    engine: Any,
    app_dir: Path,
) -> GateOutcome:
    """Tier 0, no model. Returns as soon as the build is settled as ungradeable."""
    from viral_bench.founder.apphost import AppStartFailed

    outcomes: dict[str, ItemOutcome] = {}

    outcomes["G1"] = check_manifest_strict(app_dir)
    if not outcomes["G1"].passed:
        # G2-G5 were never observed. Recorded unresolved rather than failed:
        # they are unknown, not wrong, and conflating the two would make a
        # manifest typo look like five independent defects.
        for item_id in ("G2", "G3", "G4", "G5"):
            outcomes[item_id] = ItemOutcome(item_id, None, "not reached: G1 failed")
        return GateOutcome(False, outcomes, detail=outcomes["G1"].reason)

    try:
        app = host.get(build_id, wait_timeout=START_TIMEOUT)
    except AppStartFailed as exc:
        # AppHost runs setup and start together, so a failure here is one of the
        # two. The message distinguishes them for diagnosis; the gate does not
        # need to.
        reason = str(exc)
        setup_died = "setup" in reason.lower()
        outcomes["G2"] = (
            _fail("G2", reason) if setup_died else _pass("G2", "setup completed")
        )
        outcomes["G3"] = _fail("G3", reason)
        for item_id in ("G4", "G5"):
            outcomes[item_id] = ItemOutcome(item_id, None, "not reached: app never ran")
        return GateOutcome(False, outcomes, detail=reason)
    except Exception as exc:  # noqa: BLE001 - a start failure is an outcome
        reason = f"{type(exc).__name__}: {exc}"
        outcomes["G2"] = ItemOutcome("G2", None, reason)
        outcomes["G3"] = _fail("G3", reason)
        for item_id in ("G4", "G5"):
            outcomes[item_id] = ItemOutcome(item_id, None, "not reached: app never ran")
        return GateOutcome(False, outcomes, detail=reason)

    outcomes["G2"] = _pass("G2", "setup exited 0")
    url = (app.url or "").rstrip("/")
    if not url:
        outcomes["G3"] = _fail("G3", app.ready_detail or "no URL; port never bound")
        for item_id in ("G4", "G5"):
            outcomes[item_id] = ItemOutcome(item_id, None, "not reached: no URL")
        return GateOutcome(False, outcomes, detail="app bound no port")
    outcomes["G3"] = _pass("G3", app.ready_detail or f"bound {url}")

    page = await engine.open_page()
    try:
        context = CheckContext(url=url, page=page)
        # G4: a 4xx landing page is legitimate for an auth-first app; a 5xx never
        # is, so the check is "not 5xx" rather than "200".
        status = await run_check(
            "http_status", context, {"path": "/", "max_status": 499}
        )
        outcomes["G4"] = ItemOutcome(
            "G4",
            status.passed,
            status.detail,
            observed=status.observed,
        )
        if not status.passed:
            outcomes["G5"] = ItemOutcome("G5", None, "not reached: entry URL failed")
            return GateOutcome(False, outcomes, url=url, detail=status.detail)

        await page.goto(url)
        snapshot = await page.snapshot()
        operable = [e for e in snapshot.elements if e.tag != "a" or e.label]
        fatal = [e for e in snapshot.console_errors if "pageerror" in e.lower()]
        if fatal:
            outcomes["G5"] = _fail("G5", f"uncaught error on load: {fatal[0][:160]}")
        elif not snapshot.text.strip() and not operable:
            outcomes["G5"] = _fail("G5", "entry page renders nothing operable")
        elif not operable:
            outcomes["G5"] = _fail("G5", "entry page has no operable control")
        else:
            outcomes["G5"] = _pass("G5", f"{len(operable)} operable controls")
    finally:
        with contextlib.suppress(Exception):
            await page.close()

    passed = all(outcome.passed for outcome in outcomes.values())
    return GateOutcome(passed, outcomes, url=url)


def _no_navigation(item: RubricItem) -> bool:
    """Can this item's check run before the model touches anything?

    Source greps and load-time observations can. Anything that needs the app in
    a particular state cannot, and running it early would score it against an
    untouched page.
    """
    if item.check is None:
        return False
    return item.check.name in {
        "source_absent",
        "source_present",
        "http_status",
        "network_origins",
        "no_console_errors",
        "no_failed_requests",
    }


async def deterministic_pre_pass(
    rubric: Rubric, context: CheckContext
) -> dict[str, ItemOutcome]:
    """Settle every item that needs no navigation, before spending a token."""
    outcomes: dict[str, ItemOutcome] = {}
    for item in list(rubric.scored_items) + list(rubric.penalties):
        if not _no_navigation(item):
            continue
        result = await run_check(item.check.name, context, item.check.params)
        outcomes[item.id] = ItemOutcome(
            item_id=item.id,
            passed=result.passed,
            reason=result.detail,
            observed=result.observed,
            expected=item.expect,
        )
    return outcomes


async def grade_build(  # noqa: PLR0913
    build_id: str,
    *,
    grader_model: str,
    idea_id: str = "",
    passes: int = DEFAULT_PASSES,
    client: Any = None,
    host: Any = None,
    engine: Any = None,
    container: bool = True,
    root: Path | None = None,
    write: bool = True,
) -> tuple[RubricResult, dict]:
    """Grade one build and (by default) persist the result.

    Returns the :class:`RubricResult` and the grade document. ``client``,
    ``host`` and ``engine`` are injectable so the whole pipeline can be driven by
    fakes in tests -- the same convention ``score/autorater.py`` uses, and the
    reason the loop is testable without a browser or a model.

    ``grader_model`` is a required ``provider/model`` string with no default, on
    purpose. Whoever grades a build has to say what graded it: the model id is
    written into the grade document and is the only record of which judge
    produced the number, so inheriting a hardcoded one is how a corpus ends up
    with grades nobody can attribute.
    """
    from viral_bench.founder.workspace import builds_root

    if not grader_model.strip():
        raise GradeAborted(
            "grade_build needs an explicit grader model, e.g. "
            "grader_model='anthropic/claude-...'. There is no default: run "
            "`viral-bench models` to see the providers you have keys for."
        )
    # Resolved and capability-checked before the container starts. The grader
    # drives the app through 18 tools and reads screenshots, so a provider that
    # can do neither produces a build's worth of unresolved items an hour in --
    # refusing here costs nothing and says exactly what is wrong.
    grader_spec = resolve(grader_model)
    check_support(
        grader_spec, Capability.TOOLS | Capability.IMAGES, stage="rubric grader"
    )

    base = root or builds_root()
    record_path = base / "work" / build_id / "build.json"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GradeAborted(f"no readable build record for {build_id!r}: {exc}") from exc

    resolved_idea = idea_id or str(record.get("idea_id") or "")
    if not resolved_idea:
        raise GradeAborted(f"build {build_id!r} names no idea")
    rubric = load_rubric(resolved_idea)

    app_dir = base / "work" / build_id / "app"
    run_id = new_run_id(build_id)
    run_dir = rubric_run_dir(run_id, root=base)

    owns_host = host is None
    owns_engine = engine is None
    if owns_host:
        from viral_bench.founder.apphost import AppHost

        host = AppHost(container=container)
    if owns_engine:
        from viral_bench.crowd.interaction.browser import BrowserEngine

        engine = BrowserEngine()
        await engine.start()

    # Before the app starts, not after: AppHost mounts this directory, so
    # clearing it afterwards would pull state out from under a running app.
    #
    # Resolved against `base`, NOT via workspace.reset_build_data(), which reads
    # the global builds root. Passing `root=` while wiping the real
    # builds/data/<id> would be a quiet way for a test to destroy live state.
    _reset_data_dir(base, build_id)

    tools: GraderTools | None = None
    error = ""
    try:
        gate = await run_gate(build_id, host=host, engine=engine, app_dir=app_dir)

        if gate.passed:
            per_pass: list[dict[str, ItemOutcome]] = []
            model = client or make_client(grader_spec)
            for index in range(passes):
                # A fresh context per pass. Sharing one would let pass two see
                # pass one's localStorage and pass a persistence item the app
                # never earned.
                page = await engine.open_page(
                    permissions=["clipboard-read", "clipboard-write"]
                )
                try:
                    await page.goto(gate.url)
                    tools = GraderTools(
                        page=page,
                        source=_source_toolkit(app_dir, build_id),
                        url=gate.url,
                        scratch=run_dir / "shots",
                        fixtures=_fixtures(),
                    )
                    pre = await deterministic_pre_pass(
                        rubric,
                        CheckContext(
                            url=gate.url,
                            page=page,
                            source=tools.source,
                            scratch=run_dir / "shots",
                        ),
                    )
                    remaining = [
                        item
                        for item in list(rubric.scored_items) + list(rubric.penalties)
                        if item.id not in pre
                    ]
                    agent = await run_pass(rubric, tools, model, items=remaining)
                    per_pass.append(pre | agent)
                    if index == 0 and write:
                        tools.write_transcript(run_dir / "transcript.jsonl")
                finally:
                    with contextlib.suppress(Exception):
                        await page.close()
            merged = merge_passes(per_pass)
        else:
            # A gate failure means no graded item was ever observed. They are
            # recorded unresolved so the report shows "never looked" rather than
            # 27 fabricated failures.
            merged = {}
            error = gate.detail
            passes = 0

        gate_verdicts = merge_passes([gate.outcomes])
        verdicts = gate_verdicts | merged
        result = score_rubric(
            rubric,
            verdicts,
            build_id=build_id,
            passes=max(passes, 1),
            error=error if not gate.passed else "",
        )
    finally:
        with contextlib.suppress(Exception):
            host.stop(build_id)
        if owns_engine:
            with contextlib.suppress(Exception):
                await engine.close()

    document = build_grade_document(
        rubric,
        result,
        run_id=run_id,
        grader_model=grader_model,
        graded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source_digest=source_hash(app_dir),
        brief_fingerprint=str(record.get("brief_fingerprint") or ""),
        founder=founder_block(build_id, root=base),
        comparison=comparison_block(build_id, root=base),
    )
    if write:
        write_grade(document, run_dir)
    return result, document


def _reset_data_dir(base: Path, build_id: str) -> Path:
    """Empty a build's durable data dir, under *base* rather than the global root.

    A dirty ``/data`` from an earlier pass silently invalidates every persistence
    and multi-user item -- passing them for the wrong reason, which is worse than
    failing them.
    """
    import shutil

    path = base / "data" / build_id
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fixtures() -> dict[str, Path]:
    """The sample files the grader can upload, by name.

    The same set the crowd gets, deliberately: several rubric items assert on
    exact properties of these files -- ``photo.png`` is 816 bytes at 320x240, and
    re-encoding it to JPEG makes it *bigger*, which is what turns the savings
    figure into an arithmetic-honesty trap. Grading against a different image
    would silently invalidate those items.
    """
    try:
        from viral_bench.crowd.interaction.fixtures import (
            FIXTURE_DESCRIPTIONS,
            fixture_path,
        )
        from viral_bench.rubric.fixtures import grader_fixtures

        shared = {name: fixture_path(name) for name in FIXTURE_DESCRIPTIONS}
        # Grader-only additions last: the crowd's set is never modified, only
        # extended for grading. See viral_bench.rubric.fixtures.
        return shared | grader_fixtures()
    except Exception:  # noqa: BLE001 - upload items degrade, the grade survives
        return {}


#: Source caps for grading, far above the crowd's.
#:
#: The crowd's 20 KB file cap is right for a trier skimming source as a side
#: activity, and wrong here. A ``source_absent`` item is a claim about the whole
#: tree -- "this key appears in no client-served asset" -- so a truncated read
#: turns a real finding into a silent pass. Measured before this was raised: an
#: absent-pattern check false-fired on 18 of 128 handdrawn_whiteboard builds,
#: one of which registered the pattern at byte 30k of a 61k ``app.js``.
_GRADER_MAX_FILE_BYTES = 2_000_000
_GRADER_MAX_TREE_ENTRIES = 4_000


def _source_toolkit(app_dir: Path, build_id: str):
    """A source reader over the build, or ``None`` if it cannot be opened.

    ``app_dir`` is passed explicitly. Without it the toolkit resolves the tree
    by calling ``load_build_record(build_id)``, and this used to hand it the
    *path* as the build id -- so it looked for ``builds/work/<path>/build.json``,
    raised, and was swallowed by the except below. Every ``source`` item on
    every build therefore reported "no source tree" and scored nothing: the
    universal `P4` secret-in-client-assets penalty (-10) and `P5` (-3) could not
    fire on any of the 1,000 builds, and neither could the nine authored
    ``source_present``/``source_absent`` checks.
    """
    try:
        from viral_bench.crowd.interaction.inspect import CodeInspectionToolkit

        return CodeInspectionToolkit(
            build_id,
            app_dir=str(app_dir),
            max_file_bytes=_GRADER_MAX_FILE_BYTES,
            max_tree_entries=_GRADER_MAX_TREE_ENTRIES,
        )
    except Exception:  # noqa: BLE001 - source items degrade, the grade survives
        return None
