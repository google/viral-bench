#!/usr/bin/env python3
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

"""Grade a build cohort against its rubrics: ~1,000 builds, resumable.

Defaults to cohort **r4** (``builds/cohorts/r4.json``). Earlier cohorts used a
suffix on the fleet key instead and stay reachable with ``--generation``. See
:mod:`viral_bench.rubric.corpus` for why the mechanism changed.

The RubricScore side of the comparison. Follows ``crowd_sweep.py``'s conventions
-- resume by skipping what is already on disk, bounded concurrency, a ledger for
cells killed at the wall clock, one subprocess per build so a hung container
cannot take the sweep down with it.

What it is careful about:

* **Undeliverables are graded, not skipped.** Some builds never produced a
  working app, a few percent of any cohort. They land on the Tier 0 gate and score
  0, which is the point of having a gate: a benchmark that drops the builds
  that failed hardest reports the average of the survivors and calls it the
  average. They are also nearly free -- G1 fails before a single model call.
* **The skip key includes the source hash.** ``(build_id, rubric_version,
  source_hash)``, all three already recorded in ``grade.json``. An edited app or
  a bumped rubric re-grades, while an untouched one does not, however many times the
  sweep is re-run.
* **One process per build.** ``grade_build`` drives a container and three browser
  contexts. In-process concurrency would share an event loop across all of them
  and one wedged Playwright context would stall every peer.
* **Timeouts are remembered.** A build killed at the wall clock writes no
  ``grade.json``, so nothing on disk distinguishes it from one never tried, and
  every later pass would buy it another slot-hour. Only rc=124 is suppressed,
  and only after ``--max-timeout-attempts``.

``--grader-model`` is required and has no default, because the sweep is what
writes the grader's identity into a thousand ``grade.json`` files. See
:func:`viral_bench.rubric.run.grade_build`.

Usage::

    scripts/rubric_sweep.py -m anthropic/claude-... --status
    scripts/rubric_sweep.py -m anthropic/claude-... --ideas image_compressor --limit 4
    scripts/rubric_sweep.py -m anthropic/claude-... --concurrency 24 --passes 3

Sized on a full cohort: a 3-pass grade takes ~45 min, ~15% of builds fail the
Tier 0 gate in under 2 min, and 24 workers sustain ~31 builds/hour. Keep 3 passes --
measured code-judged pass-to-pass disagreement is 5.5%, so a single pass would
put that straight into every score.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.rubric.corpus import CorpusBuild, select_builds  # noqa: E402

BUILDS = REPO / "builds"
RUBRIC_DIR = BUILDS / "rubric"
LOG_DIR = BUILDS / "rubric_logs"
TIMEOUT_LEDGER = RUBRIC_DIR / ".timeouts.json"
GRADE_FILENAME = "grade.json"

#: Wall clock for one build. 31 items x 3 passes x ~3-4 model calls, plus a
#: container start and three browser contexts. Measured on the golden set, so
#: raise it there rather than guessing here.
#: Per-build grading cap. 5400, NOT 2700 -- the old value sat on the MEDIAN of
#: the distribution it was meant to bound, and silently censored it.
#:
#: The trap is that the durations look bimodal: 2-109s and then a wall at
#: 2700.0s. That reads as "fast successes and hangs", and it is not. The fast
#: group is builds FAILING the Tier 0 gate, which grade 0 items and exit early.
#: A build that PASSES the gate runs 19-21 items x 3 passes with ~200-270 tool
#: calls, and those completed at 2312.3, 2406.4, 2573.9 and 2663.6s -- the
#: largest clearing the old cap by 36 seconds. Successes piled up against the
#: ceiling is censoring, not a workload.
#:
#: So a gate-passing build needing ~32% more work than the fastest was killed at
#: 2700.0s while still making progress -- no repetition in its transcript, last
#: tool call ``ok: true``. 89% of gate passes died that way, and each SIGKILL
#: orphaned its container, which slowed the survivors into timing out too.
#:
#: Size this from the GATE-PASSING tail only. Mixing the gate failures in makes
#: any percentile meaningless.
DEFAULT_TIMEOUT_S = 5400


def corpus_cells(cohort: str, generation: str) -> list[CorpusBuild]:
    """Every build in scope, in a stable order.

    Named cohorts live in ``builds/cohorts/<name>.json``. Earlier ones were a
    suffix on the fleet key. See :mod:`viral_bench.rubric.corpus` for why the mechanism
    changed. An untagged cohort yields nothing, so a sweep launched before the
    cohort is filled stops instead of grading some other corpus.

    INTERLEAVED BY IDEA, not alphabetical, so that any PREFIX of the sweep is a
    balanced sample. A full 3-pass grade costs ~45 minutes, so a 1,000-build
    corpus is days of work and being interrupted is the normal case, not the
    exception. In alphabetical order an interrupted sweep yields 100% of
    ``ai_room_redesign`` and ``browser_api_client`` and 0% of the other 23
    ideas -- the least useful partial available, and not reportable at all,
    because every model's score would come from two briefs. Round-robin over
    ideas means a prefix covers every idea, every model and every arm roughly
    evenly, so it can be reported as a smaller sample of the same experiment.

    This reorders nothing that matters: the same cells run, and a completed
    sweep is identical either way.
    """
    cells = select_builds(BUILDS, cohort=cohort, generation=generation)
    by_idea: dict[str, list] = {}
    for cell in cells:
        by_idea.setdefault(cell.idea_id, []).append(cell)
    for group in by_idea.values():
        group.sort(key=lambda c: (c.model, c.arm, c.build_id))
    ordered: list = []
    for rank in range(max((len(g) for g in by_idea.values()), default=0)):
        for idea in sorted(by_idea):
            group = by_idea[idea]
            if rank < len(group):
                ordered.append(group[rank])
    return ordered


def source_hash_of(app_dir: Path) -> str:
    from viral_bench.rubric.report import source_hash

    try:
        return source_hash(app_dir)
    except Exception:  # noqa: BLE001 - a build with no readable app still grades
        return ""


def existing_grades() -> dict[str, list[dict]]:
    """build_id -> the grade documents already recorded for it."""
    found: dict[str, list[dict]] = {}
    if not RUBRIC_DIR.is_dir():
        return found
    for run_dir in RUBRIC_DIR.glob("*__rubric-*"):
        grade = run_dir / GRADE_FILENAME
        if not grade.is_file():
            continue
        try:
            document = json.loads(grade.read_text())
        except (OSError, ValueError):
            continue
        found.setdefault(document.get("build_id", run_dir.name), []).append(document)
    return found


def timeouts_seen() -> Counter:
    try:
        return Counter(json.loads(TIMEOUT_LEDGER.read_text()))
    except (OSError, ValueError):
        return Counter()


def record_timeout(build_id: str) -> None:
    ledger = timeouts_seen()
    ledger[build_id] += 1
    TIMEOUT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    TIMEOUT_LEDGER.write_text(json.dumps(dict(ledger), indent=2, sort_keys=True))


def plan(
    cells: list[CorpusBuild], opts: argparse.Namespace
) -> tuple[list[CorpusBuild], Counter]:
    """Which cells still need grading, and why the others do not."""
    from viral_bench.rubric.schema import load_rubric

    grades = existing_grades()
    ledger = timeouts_seen()
    versions: dict[str, int] = {}
    todo, skipped = [], Counter()

    for cell in cells:
        if opts.ideas and cell.idea_id not in opts.ideas.split(","):
            continue
        if opts.arms and cell.arm not in opts.arms.split(","):
            continue
        if cell.build_id not in versions:
            try:
                versions[cell.idea_id] = load_rubric(cell.idea_id).rubric_version
            except Exception:  # noqa: BLE001
                skipped["no rubric"] += 1
                continue
        want_version = versions.get(cell.idea_id)
        digest = source_hash_of(BUILDS / "work" / cell.build_id / "app")

        fresh = [
            g
            for g in grades.get(cell.build_id, [])
            if g.get("rubric_version") == want_version
            and (not digest or g.get("source_hash") == digest)
        ]
        if fresh and not opts.force:
            skipped["already graded"] += 1
            continue
        if ledger.get(cell.build_id, 0) >= opts.max_timeout_attempts:
            skipped["timed out repeatedly"] += 1
            continue
        todo.append(cell)

    if opts.limit:
        todo = todo[: opts.limit]
    return todo, skipped


def grade_one(cell: CorpusBuild, opts: argparse.Namespace) -> dict:
    """Grade one build in its own process. Never raises."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{cell.build_id}.log"

    # Populate the dependency cache BEFORE grading, exactly as crowd_sweep.py
    # does. Without it the app starts with no venv and no node_modules, so any
    # build with dependencies dies at startup -- "Failed to spawn: uvicorn",
    # "from PIL import ...", "No virtual environment found" -- fails G2, and is
    # zeroed by the Tier 0 gate. Measured on the first 34 grades: 10 of them.
    #
    # That is not a measurement of the app. The crowd provisioned these same
    # builds and scored them normally, so without this the RubricScore would
    # largely be reporting "is this a zero-dependency static app", biased
    # against precisely the server-backed builds the benchmark cares most about,
    # and the comparison against ViralScore would be meaningless.
    #
    # Outside the timed subprocess on purpose, like the crowd: an install must
    # not be charged against --timeout. provision_build is idempotent and
    # memoized per process, so a cold cache costs one container once.
    if not opts.no_provision:
        try:
            from viral_bench.founder.provision import provision_build

            provision_build(cell.build_id, timeout=opts.provision_timeout)
        except Exception as exc:  # noqa: BLE001 - a cold cache must not lose a build
            print(f"    provision failed {cell.build_id}: {exc!r}", flush=True)

    code = (
        "import asyncio, sys, json;"
        "sys.path.insert(0, 'src');"
        "from viral_bench.rubric.run import grade_build;"
        f"r, d = asyncio.run(grade_build({cell.build_id!r},"
        f" grader_model={opts.grader_model!r},"
        f" passes={opts.passes}, container={not opts.no_container}));"
        "print('SCORE', json.dumps({'score': d.get('score'),"
        " 'gate': d.get('gate', {}).get('passed'),"
        " 'tiers': len(d.get('tiers', []))}))"
    )
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            done = subprocess.run(  # noqa: S603
                [sys.executable, "-c", code],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=opts.timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                check=False,
            )
            rc = done.returncode
        except subprocess.TimeoutExpired:
            rc = 124
            log.write(f"\nTIMEOUT after {opts.timeout}s\n")

    if rc == 124:
        record_timeout(cell.build_id)
    return {
        "build_id": cell.build_id,
        "idea_id": cell.idea_id,
        "arm": cell.arm,
        "rc": rc,
        "elapsed_s": round(time.monotonic() - started, 1),
        "log": str(log_path.relative_to(REPO)),
    }


def status_report(cells: list[CorpusBuild]) -> int:
    grades = existing_grades()
    ledger = timeouts_seen()
    graded = sum(1 for c in cells if c.build_id in grades)
    by_arm: Counter = Counter()
    scores = []
    for cell in cells:
        for document in grades.get(cell.build_id, []):
            by_arm[cell.arm] += 1
            if isinstance(document.get("final"), (int, float)):
                scores.append(document["final"])
            break
    print(f"builds in scope:  {len(cells)}")
    print(f"graded:           {graded} ({100 * graded / max(1, len(cells)):.1f}%)")
    print(f"undeliverable:    {sum(1 for c in cells if not c.deliverable)}")
    print(f"timeout ledger:   {len(ledger)} builds, {sum(ledger.values())} kills")
    if by_arm:
        arms = ", ".join(f"{k or '?'}={v}" for k, v in sorted(by_arm.items()))
        print(f"by arm:           {arms}")
    if scores:
        scores.sort()
        mid = scores[len(scores) // 2]
        print(
            f"scores:           n={len(scores)} min={scores[0]:.1f} "
            f"median={mid:.1f} max={scores[-1]:.1f}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        default="r4",
        help="cohort manifest under builds/cohorts (r4 and later)",
    )
    parser.add_argument(
        "--generation",
        default="",
        help="legacy fleet-key suffix selection (r3); overrides --cohort",
    )
    parser.add_argument(
        "-m",
        "--grader-model",
        required=True,
        help=(
            "the model that grades, as 'provider/model'. Required: there is no "
            "default grader, and this id is recorded in every grade.json as the "
            "only account of which judge produced the number."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-timeout-attempts", type=int, default=2)
    parser.add_argument("--ideas", default="", help="comma-separated idea filter")
    parser.add_argument("--arms", default="", help="comma-separated arm filter")
    parser.add_argument("--builds", default="", help="comma-separated build ids")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="re-grade even if fresh")
    parser.add_argument("--no-container", action="store_true")
    parser.add_argument(
        "--no-provision",
        action="store_true",
        help="skip the dependency-cache warm-up (grades apps with deps as dead)",
    )
    parser.add_argument("--provision-timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    opts = parser.parse_args(argv)

    cells = corpus_cells("" if opts.generation else opts.cohort, opts.generation)
    if not cells:
        target = opts.generation or opts.cohort
        print(f"no builds in scope for {target!r} -- is the cohort tagged yet?")
        print("  scripts/cohort.py ls        # cohorts that exist")
        print(f"  scripts/cohort.py status {opts.cohort}")
        return 1
    if opts.builds:
        wanted = set(opts.builds.split(","))
        cells = [c for c in cells if c.build_id in wanted]
    if opts.status:
        return status_report(cells)

    todo, skipped = plan(cells, opts)
    scope = opts.generation or opts.cohort
    print(f"{len(cells)} builds in {scope}; {len(todo)} to grade")
    for reason, count in sorted(skipped.items()):
        print(f"  skipped {count}: {reason}")
    if opts.dry_run or not todo:
        for cell in todo[:20]:
            print(f"  would grade {cell.build_id} ({cell.idea_id}/{cell.arm})")
        return 0

    started = datetime.now(UTC)
    done_count = 0
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=opts.concurrency) as pool:
        futures = {pool.submit(grade_one, cell, opts): cell for cell in todo}
        for future in as_completed(futures):
            outcome = future.result()
            done_count += 1
            if outcome["rc"] != 0:
                failures.append(outcome)
            mark = "ok " if outcome["rc"] == 0 else f"rc{outcome['rc']}"
            print(
                f"[{done_count}/{len(todo)}] {mark} {outcome['elapsed_s']:>7.1f}s "
                f"{outcome['build_id']}",
                flush=True,
            )

    elapsed = (datetime.now(UTC) - started).total_seconds()
    print(f"\n{done_count} graded in {elapsed / 60:.1f} min, {len(failures)} failed")
    for outcome in failures[:20]:
        print(f"  rc={outcome['rc']} {outcome['build_id']} -> {outcome['log']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
