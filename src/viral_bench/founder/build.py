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

"""Orchestrate one founder build: Design -> Build -> validate -> Ship.

``run_build`` ties the pieces together for a single idea:

1. create a fresh host workspace directory,
2. run the founder harness (opencode + Gemini) to design and build the app,
3. validate the ``viralbench.json`` manifest the agent produced,
4. *ship* the app into a single shared git store (one orphan branch per build,
   so no new repo is created per build), and
5. write a ``build.json`` record for later inspection / the runner.

The ship step is deliberately "one store, many branches": ``builds/store`` is a
single git repo, and each build becomes an orphan branch ``build/<build_id>``
whose root *is* the app. A tester or the crowd can get a clean single-app tree
with ``git clone builds/store --branch build/<build_id> --single-branch``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

from viral_bench.founder.collab import build_toolset, files_to_strip
from viral_bench.founder.harness import (
    DEFAULT_MODEL,
    FounderHarness,
    HarnessResult,
    OpenCodeHarness,
)
from viral_bench.founder.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    ManifestError,
    load_manifest,
)
from viral_bench.founder.prompts import brief_fingerprint
from viral_bench.founder.structures import (
    DEFAULT_DYNAMIC_TURNS,
    DEFAULT_ROUNDS,
    build_structure,
    normalize_agents,
)
from viral_bench.founder.trajectory import reasoning_from_dumps
from viral_bench.founder.workspace import BuildWorkspace, builds_root
from viral_bench.ideas import (
    Idea,
    IdeaValidationError,
    load_idea_file,
    load_ideas,
)

BUILD_RECORD_FILENAME = "build.json"


class BuildError(RuntimeError):
    """Raised when a build cannot be started (e.g. unknown idea)."""


def get_idea(idea_id: str) -> Idea:
    """Return the validated :class:`Idea` with ``idea_id`` from the Idea Bench."""
    ideas = {idea.idea_id: idea for idea in load_ideas()}
    if idea_id not in ideas:
        raise BuildError(f"unknown idea_id {idea_id!r}. Available: {sorted(ideas)}")
    return ideas[idea_id]


def resolve_idea(idea_ref: str) -> Idea:
    """Resolve an idea reference that is either an ``idea_id`` or a YAML path.

    - If ``idea_ref`` points at an existing ``.yaml``/``.yml`` file (or ends with
      one of those extensions), it is loaded and validated directly.
    - Otherwise it is treated as an ``idea_id`` looked up in the Idea Bench.
    """
    path = Path(idea_ref)
    looks_like_file = idea_ref.endswith((".yaml", ".yml")) or path.is_file()
    if looks_like_file:
        if not path.is_file():
            raise BuildError(f"idea file not found: {idea_ref}")
        try:
            return load_idea_file(path)
        except IdeaValidationError as exc:
            raise BuildError(f"invalid idea file {idea_ref}: {exc}") from exc
    return get_idea(idea_ref)


def generate_build_id(idea_id: str) -> str:
    """Return a unique, sortable build id: ``<idea>__<ts>__<short>``."""
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{idea_id}__{ts}__{uuid.uuid4().hex[:6]}"


@dataclass
class BuildRecord:
    """Everything known about one build (persisted as ``build.json``)."""

    build_id: str
    idea_id: str
    model: str
    created_at: str
    # ok | harness_failed | harness_timeout | manifest_missing | manifest_invalid
    status: str
    app_dir: str
    root: str
    harness_ok: bool
    # Collaboration structure metadata. Defaults describe the solo baseline and
    # keep older build.json files loadable (unknown/retired keys are ignored on
    # load, see _record_from_dict).
    structure: str = "solo"  # "solo" | "team" | "dynamic"
    # Agents the CONFIGURATION fixes: 1 for solo and for dynamic (which runs one
    # orchestrator process). In dynamic mode the interesting count -- how many
    # subagents the model chose to run -- is observed, not configured, and lives
    # in `orchestration`. Keeping the two apart matters: `build_fleet` verifies a
    # build against its arm's config, and a field the model can move is not a
    # config to verify against.
    n_agents: int = 1
    max_rounds: int = 1  # team round cap (1 for solo)
    min_rounds: int = 1  # min rounds before QA may ship (1 for solo)
    rounds_run: int = 1  # rounds executed
    shipped_early: bool = False  # team stopped because QA signalled ship
    # Whether QA exercised the running app (browser/CLI/bot) on its final
    # turn -- not merely read the code. None for solo / when unknown. A shipped build
    # with qa_verified False shipped without real runtime verification.
    qa_verified: bool | None = None
    max_turns: int = 2  # upper bound on model turns for this configuration
    min_turns: int = 2  # lower bound on model turns for this configuration
    turns_spent: int = 0  # model turns run
    roles: list[str] = field(default_factory=list)
    # Collaboration toolset. See viral_bench.founder.collab.
    collab: str = "local"
    collab_meta: dict | None = None
    # -- dynamic mode only ---------------------------------------------------
    # How many subagents the founder spawned. Zero is a legitimate and
    # interesting outcome (the model chose to work alone), which is why it is a
    # plain default rather than a sentinel: a dynamic build with 0 spawns is a
    # result, not a missing measurement.
    subagents_spawned: int = 0
    # The full observed orchestration: spawn count by subagent type, how many
    # subagents were resumed rather than re-created, peak concurrency, the agent
    # definitions the model wrote for itself, and every task call it made. None
    # for the solo/team modes, which have no such choice to record.
    orchestration: dict | None = None
    # Hash of the DESIGN + BUILD prompts this build was given -- the idea spec,
    # scope guidance, runtime notes and example manifest as the model
    # saw them. Two builds are comparable only if this matches. Without it the
    # fleet reused builds from before the web-dev pivot, whose brief said "prefer
    # plain static files" and "no backend required", as current results.
    # Empty on records written before the field existed.
    brief_fingerprint: str = ""
    # What was recorded about HOW the app was built: the model's chain of
    # thought and opencode's own session store (see _record_trajectory). Empty
    # for records written before the founder pipeline captured either.
    trajectory: dict | None = None
    phases: list[dict] = field(default_factory=list)
    manifest: dict | None = None
    manifest_error: str | None = None
    error: str | None = None  # top-line failure reason (harness or manifest)
    shipped_ref: str | None = None
    store_path: str | None = None
    #: Which named sweep this build belongs to. A LABEL, not a
    #: selection axis, and deliberately not ``replicate``.
    #:
    #: ``replicate`` answers "which independent build of this cell is this", and
    #: it is part of the fleet index key, so it cannot also mean "which sweep is
    #: this build part of": a cohort assembled from builds made at different times
    #: -- keeping some arms and rebuilding others -- has no
    #: single replicate to be, and renumbering the kept arms would fork the index.
    #:
    #: It exists so a corpus can be named. The 2026-08 run had no name, so "the
    #: 1,000 builds" had to be reconstructed from a replicate number plus four arm
    #: names plus a brief-fingerprint check every time anyone asked which builds a
    #: result covered. See ``scripts/cohort.py``.
    cohort: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _record_phases(result: HarnessResult) -> list[dict]:
    return [
        {
            "phase": p.phase,
            "role": p.role,
            "agent_index": p.agent_index,
            "turn": p.turn,
            "session_id": p.session_id,
            "returncode": p.returncode,
            "ok": p.ok,
            "timed_out": p.timed_out,
            "duration_s": round(p.duration_s, 2),
            "transcript": str(p.transcript_path),
            "stderr_tail": p.stderr_tail,
            # `_root`: measured off the stdout transcript, which opencode filters
            # to the turn's ROOT session. Correct per turn, but blind to every
            # subagent -- see _record_trajectory for the build-level total.
            "reasoning_parts_root": p.reasoning_parts,
            "reasoning_chars_root": p.reasoning_chars,
            "sessions": p.sessions_path,
            "sessions_records": p.sessions_records,
        }
        for p in result.phases
    ]


def _record_trajectory(result: HarnessResult, workspace: BuildWorkspace) -> dict:
    """Summarise what this build managed to record about HOW it was built.

    A first-class part of the build record rather than something left implicit
    in the transcript, because "the reasoning was captured" is a claim that has
    to be auditable per build. An empty trace has two quite different causes --
    the model did not think at all (Claude's ``adaptive`` mode skips easy
    turns) or capture regressed -- and only a count written down at build
    time tells them apart afterwards.

    Every figure is recorded twice, from two independent sources, because
    neither alone can be trusted:

    ``*_root``
        Summed from the per-turn stdout transcripts. opencode's printer drops
        every event whose session is not the turn's root, so this counts the
        orchestrator and none of its subagents. On a dynamic build with four
        subagents it read about 3x low -- which is why it must not be the only
        number here.
    ``*_all``
        Read from the dumped session store, which has the subagents. Computed
        ONCE for the whole build and deduped by part id, because each turn
        re-dumps its entire subtree: summing per-turn dump counts double-counts,
        and did (137 recorded against 114 records on disk).

    Keeping ``_root`` is not redundancy. ``dump_session_trace`` is best-effort
    and yields nothing on any failure, so a dump that broke and a model that did
    not think both produce ``_all == 0``. The transcript-derived count is the
    independent witness that tells those apart -- the exact ambiguity this whole
    field exists to resolve.

    The two agree on a build with no subagents, and a gap between them IS the
    subagents' share of the thinking.
    """
    dumped = reasoning_from_dumps(workspace.transcript_dir / "sessions")
    return {
        "reasoning_parts_root": sum(p.reasoning_parts for p in result.phases),
        "reasoning_chars_root": sum(p.reasoning_chars for p in result.phases),
        "turns_with_reasoning_root": sum(1 for p in result.phases if p.reasoning_parts),
        "reasoning_parts_all": dumped["parts"],
        "reasoning_chars_all": dumped["chars"],
        "reasoning_redacted_all": dumped["redacted"],
        "turns": len(result.phases),
        # Both deduped across the per-turn dumps. `records` is every distinct
        # line on disk, while `events` is the parts alone, i.e. what the exported
        # trajectory stream will hold. They are different numbers and conflating
        # them is how the double-count went unnoticed the first time.
        "session_records": dumped["records"],
        "session_events": dumped["events"],
        # Turns whose dump wrote something -- NOT a count of distinct sessions.
        # Solo design+build share one session, so two turns legitimately write
        # the same file twice, the second a superset of the first.
        "turns_dumped": sum(1 for p in result.phases if p.sessions_records),
    }


def _git_identity() -> dict[str, str]:
    """Return author/committer env for store commits (falls back to a default)."""

    def cfg(key: str) -> str:
        try:
            out = subprocess.run(
                ["git", "config", "--get", key],
                capture_output=True,
                text=True,
                check=False,
            )
            return out.stdout.strip()
        except OSError:
            return ""

    name = cfg("user.name") or "viral_bench"
    email = cfg("user.email") or "viral-bench@localhost"
    return {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }


def _run(argv: list[str], *, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(argv, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError(
            f"command failed ({' '.join(argv[:3])} ...): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def ship_to_store(app_dir: Path, build_id: str, *, message: str) -> tuple[str, Path]:
    """Commit ``app_dir`` as an orphan branch ``build/<build_id>`` in the store.

    Uses git plumbing with a temporary index so it never disturbs the store's
    working tree or HEAD -- safe to call repeatedly for many builds against the
    one shared repo. Returns ``(branch_name, store_path)``.
    """
    store = builds_root() / "store"
    git_dir = store / ".git"
    if not git_dir.exists():
        store.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "-q", "-b", "main", str(store)])

    branch = f"build/{build_id}"
    index_file = git_dir / f"tmp-index-{build_id}"

    base_env = dict(os.environ)
    base_env.update(_git_identity())
    base_env["GIT_DIR"] = str(git_dir)

    add_env = dict(base_env)
    add_env["GIT_WORK_TREE"] = str(app_dir)
    add_env["GIT_INDEX_FILE"] = str(index_file)

    try:
        _run(["git", "add", "-A"], env=add_env)
        tree = _run(["git", "write-tree"], env=add_env)
        commit = _run(["git", "commit-tree", tree, "-m", message], env=base_env)
        _run(["git", "update-ref", f"refs/heads/{branch}", commit], env=base_env)
    finally:
        index_file.unlink(missing_ok=True)

    return branch, store


def _cleanup_agent_artifacts(app_dir: Path, collab: str = "local") -> None:
    """Remove agent/build scratch so only the real app ships.

    Drops opencode/git scratch dirs and Python bytecode caches, then removes the
    collaboration files that must not ship for this ``collab`` mode (see
    :func:`viral_bench.founder.collab.files_to_strip`): local mode strips only
    coordination scratch and keeps ``DESIGN.md`` as a deliverable. A toolset that
    hosts the design outside the working directory strips ``DESIGN.md`` too, so
    the shipped tree is app-source-only.
    """
    for junk_dir in (".opencode", ".git"):
        target = app_dir / junk_dir
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
    # Bytecode caches can appear at any depth if the app was executed during the
    # build, and they are regenerated on run and must never ship.
    for pycache in app_dir.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache, ignore_errors=True)
    for name in files_to_strip(collab):
        (app_dir / name).unlink(missing_ok=True)


def run_build(
    idea_ref: str,
    *,
    model: str = DEFAULT_MODEL,
    agents: int | str = 1,
    collab: str = "local",
    rounds: int = DEFAULT_ROUNDS,
    min_rounds: int = 1,
    turns: int = DEFAULT_DYNAMIC_TURNS,
    browser_tools: bool = True,
    harness: FounderHarness | None = None,
    ship: bool = True,
) -> BuildRecord:
    """Run one full founder build and return its record.

    The build always runs on the host: the founder harness (opencode) writes the
    app into a fresh :class:`BuildWorkspace` directory. Running/testing the built
    app in a container is a separate, later step (see
    :mod:`viral_bench.founder.runtime`).

    Args:
        idea_ref: An ``idea_id`` from the Idea Bench, or a path to an idea
            ``.yaml`` file.
        model: opencode ``provider/model`` (e.g. ``google-vertex/gemini-2.0-flash``).
        agents: Founder configuration -- ``1`` (solo baseline), ``4`` (the
            round-table specialist team), or ``"dynamic"`` (one founder agent that
            chooses and orchestrates its own subagents). No other sizes.
        collab: Collaboration toolset -- ``"local"`` (shared files only) is the
            default, the only built-in, and the reproducible scored path. See
            :mod:`viral_bench.founder.collab` for the extension point.
        rounds: Hard cap on collaboration rounds for the team (ignored otherwise).
        min_rounds: Minimum rounds the team must run before QA may ship early
            (ignored otherwise). Must be ``1 <= min_rounds <= rounds``.
        turns: Hard cap on orchestrator turns in dynamic mode (ignored otherwise).
        browser_tools: Give agents a real browser so they can render and click the
            running app -- the team's Designer/QA, or (in dynamic mode) the founder
            and every subagent it spawns. Auto-disabled gracefully if the host
            lacks the browser prerequisites, and always ignored for solo.
        harness: override the founder harness (used by tests). When given, it is
            responsible for its own structure/toolset. ``agents``/``collab``/
            ``rounds`` still determine the recorded metadata.
        ship: whether to ship a healthy build into the git store.
    """
    idea = resolve_idea(idea_ref)
    # Build the team + toolset up front: this validates agents/collab/rounds and
    # provides the metadata recorded on the build (even with a test harness).
    agents = normalize_agents(agents)
    team = build_structure(
        agents, max_rounds=rounds, min_rounds=min_rounds, max_turns=turns
    )
    toolset = build_toolset(collab, n_agents=agents)
    build_id = generate_build_id(idea.idea_id)
    workspace = BuildWorkspace(build_id).create()
    harness = harness or OpenCodeHarness(
        model=model, structure=team, toolset=toolset, browser_tools=browser_tools
    )

    result = harness.run(idea, workspace)

    record = BuildRecord(
        build_id=build_id,
        idea_id=idea.idea_id,
        model=model,
        created_at=datetime.now(UTC).isoformat(),
        status="ok",
        app_dir=str(workspace.app_dir),
        root=str(workspace.root),
        harness_ok=result.ok,
        structure=team.name,
        n_agents=team.n_agents,
        max_rounds=team.max_rounds,
        min_rounds=getattr(team, "min_rounds", 1),
        rounds_run=team.rounds_run,
        shipped_early=team.shipped_early,
        qa_verified=getattr(team, "qa_verified", None),
        max_turns=team.max_turns,
        min_turns=getattr(team, "min_turns", 2),
        turns_spent=len(result.phases),
        roles=list(getattr(team, "role_keys", [])),
        collab=toolset.name,
        collab_meta=toolset.metadata(),
        brief_fingerprint=brief_fingerprint(idea),
        trajectory=_record_trajectory(result, workspace),
        phases=_record_phases(result),
    )
    orchestration = team.orchestration() if hasattr(team, "orchestration") else None
    if orchestration is not None:
        record.orchestration = orchestration
        record.subagents_spawned = int(orchestration.get("subagents_spawned", 0))

    # If the harness itself failed (timeout / preflight / crash), report the
    # real reason -- do NOT run manifest validation, whose "not found" message
    # would be a misleading downstream symptom.
    manifest: Manifest | None = None
    if not result.ok:
        failing = next((p for p in result.phases if not p.ok), None)
        # A turn SIGKILLed on the wall clock is the harness's limit biting, not the
        # model failing to build the app. Recording both as `harness_failed`
        # made a slow-but-working model indistinguishable from a broken one --
        # a fabricated capability difference. Status
        # says which, so a fleet can be audited for timeouts before its numbers
        # are believed.
        timed_out = failing is not None and failing.timed_out
        record.status = "harness_timeout" if timed_out else "harness_failed"
        record.error = (
            f"harness phase {failing.phase!r} failed (rc={failing.returncode}): "
            f"{failing.stderr_tail}".strip()
            if failing is not None
            else "harness failed"
        )
    else:
        manifest_path = workspace.app_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            record.status = "manifest_missing"
            record.manifest_error = f"{MANIFEST_FILENAME} not found at app root"
            record.error = record.manifest_error
        else:
            try:
                manifest = load_manifest(manifest_path)
                record.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except ManifestError as exc:
                record.status = "manifest_invalid"
                record.manifest_error = str(exc)
                record.error = str(exc)

    # Strip agent scratch from EVERY build, not only the ones that ship.
    #
    # The crowd now simulates every build, including ones with no valid manifest
    # -- and its code-inspection tool reads whatever is in the app directory. If
    # cleanup only ran on the ship path, agents reviewing a manifest-less build
    # would see `.opencode/`, `.git/` and the team's coordination scratch, while
    # agents reviewing a healthy build would see app source only. That is a
    # difference in what the crowd is shown that tracks which model failed to
    # ship a manifest -- i.e. a harness artifact that would read as a capability
    # difference, which is the failure mode this repo keeps rediscovering.
    if ship:
        _cleanup_agent_artifacts(workspace.app_dir, collab=toolset.name)

    # Ship only a healthy build (harness ok + valid manifest).
    if ship and result.ok and manifest is not None:
        branch, store = ship_to_store(
            workspace.app_dir,
            build_id,
            message=f"Build {build_id}: {idea.title}",
        )
        record.shipped_ref = branch
        record.store_path = str(store)

    (workspace.root / BUILD_RECORD_FILENAME).write_text(
        record.to_json(), encoding="utf-8"
    )
    return record


def _record_from_dict(data: dict) -> BuildRecord:
    """Build a :class:`BuildRecord`, ignoring unknown (e.g. legacy) keys.

    Keeps older ``build.json`` files (which may carry retired fields such as
    ``isolation``) loadable instead of crashing on an unexpected argument.
    """
    known = {f.name for f in fields(BuildRecord)}
    return BuildRecord(**{k: v for k, v in data.items() if k in known})


def load_build_record(build_id: str) -> BuildRecord:
    """Load a persisted :class:`BuildRecord` by build id."""
    path = builds_root() / "work" / build_id / BUILD_RECORD_FILENAME
    if not path.is_file():
        raise BuildError(f"no build record for {build_id!r} at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return _record_from_dict(data)


def list_builds() -> list[BuildRecord]:
    """Return all build records under ``builds/work``, newest first."""
    work = builds_root() / "work"
    if not work.is_dir():
        return []
    records: list[BuildRecord] = []
    for child in work.iterdir():
        record_path = child / BUILD_RECORD_FILENAME
        if record_path.is_file():
            try:
                records.append(
                    _record_from_dict(
                        json.loads(record_path.read_text(encoding="utf-8"))
                    )
                )
            except (json.JSONDecodeError, TypeError):
                continue
    return sorted(records, key=lambda r: r.created_at, reverse=True)
