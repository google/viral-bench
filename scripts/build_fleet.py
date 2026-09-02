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

"""Build the frozen ViralBench fleet: every idea x every founder model, once.

The fleet is the benchmark's capital: 25 ideas x 2 models x a 4-agent team build
is hours of wall clock and hundreds of dollars, and everything downstream (crowd,
score, sweep) re-reads it for free. So this driver exists to make the expensive
part happen exactly once, under one recorded config, and to be safe to re-run.

Three properties matter more than speed:

* **Identical config.** Every build runs the same command --
  ``found <idea> --model <M> --agents 4 --collab local`` with the shipped
  defaults. The config is recorded in ``builds/fleet.json`` and re-checked
  against each ``build.json``; a build that ran under anything else is not in
  the fleet.
* **Isolation.** Builds run concurrently inside private network namespaces (see
  ``scripts/netns_run.sh``), because a team build serves the app it is writing
  and the model picks the port. Sharing a loopback across 8 concurrent builds
  means QA can review the wrong app.
* **Resumable.** Re-running only builds the (idea, model) cells that do not yet
  have a healthy build. Nothing under ``builds/`` is ever deleted or rewritten.

Usage::

    scripts/build_fleet.py --concurrency 8            # build everything missing
    scripts/build_fleet.py --limit 4                  # pilot a few cells first
    scripts/build_fleet.py --status                   # report, build nothing
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.founder.prompts import brief_fingerprint  # noqa: E402
from viral_bench.ideas import load_ideas  # noqa: E402

FLEET_PATH = REPO / "builds" / "fleet.json"
#: Guards the read-modify-write of the fleet index so two build_fleet processes
#: (e.g. one per quota lineage) cannot clobber each other's entries. See
#: ``persist``.
FLEET_LOCK_PATH = REPO / "builds" / ".fleet.json.lock"
LOG_DIR = REPO / "builds" / "fleet_logs"
#: Per-cell opencode state, so concurrent builds do not share one SQLite file.
OPENCODE_STATE_ROOT = REPO / "builds" / "opencode_state"
NETNS = REPO / "scripts" / "netns_run.sh"

#: The two founder models under comparison. A and B are deliberately labels, not
#: vendor names: the same fleet machinery has to work across providers, which it
#: does -- ``--models openai/gpt-5-mini,anthropic/claude-sonnet-4-5`` runs a
#: cross-provider fleet, because the provider follows from the model id (see
#: viral_bench.founder.models).
#:
#: Empty by default, like every other model setting: pass ``--models`` or set
#: them in config. There is nothing sensible to default to.
MODEL_A = ""
MODEL_B = ""

#: The founder configurations under comparison, by name.
#:
#: This used to be a single frozen ``FOUNDER_CONFIG``, so the fleet's only axis
#: was the model. That made the obvious question unaskable: does a 4-agent team
#: build more viral apps than one agent, and does letting the founder pick its
#: own team add anything over a fixed round table? Those are claims the benchmark
#: exists to test, and answering them by hand-editing a constant between runs
#: produces two fleets that are not comparable.
#:
#: Every entry is verified against each build record, so a build produced under
#: one shape is never silently counted toward another's results.
STRUCTURES: dict[str, dict[str, object]] = {
    # One agent: design -> build. The baseline.
    "solo": {
        "agents": 1,
        "collab": "local",
        "rounds": 1,
        "min_rounds": 1,
        "structure": "solo",
        "browser_tools": True,
    },
    # Four specialists round-tabling in a shared working directory.
    #
    # The 90-minute fleet default is a solo-arm number and truncates this arm:
    # four agents over three rounds is several times one agent's work. Measured
    # while the default was still in force: nearly every build failure in this
    # arm landed at EXACTLY 5400.0s, and it fell almost entirely on the slowest
    # founder rather than being spread across the models. That is our wall clock
    # being recorded as a model that cannot ship, which is the fabricated
    # capability difference of docs/crowd_bugs.md T0.1. Sized like the dynamic
    # arm's: generous enough not to bind, bounded enough that a wedged cell does
    # not hold a slot forever.
    #
    # min_rounds 1 -> 2. At a floor of 1 the round-table did not round-table: the
    # overwhelming majority of builds ran exactly ONE round, and they are genuine
    # early ships rather than truncated builds -- they carry shipped_early and
    # qa_verified true, and mean rounds run was barely above 1. So the arm billed
    # as "4 specialists x 3 rounds" was really 4 specialists x 4 turns, and a
    # team-vs-solo comparison drawn from it could not speak to whether iteration
    # helps -- the mechanism under test barely ran. A floor of 2 forces the second
    # round; the cap stays 3 so a team that wants a third can still take it.
    #
    # Note this necessarily invalidates any team corpus built under the old floor:
    # config_matches() compares min_rounds, so those builds no longer match this
    # shape. That is the intended semantics -- they are a different experiment --
    # but it does mean a re-run rebuilds them.
    "team": {
        "agents": 4,
        "collab": "local",
        "rounds": 3,
        "min_rounds": 2,
        "structure": "team",
        "browser_tools": True,
        "timeout_s": 14400,
    },
    # One founder agent that picks its own team: it decides whether to delegate,
    # to whom, how many at a time, and may define its own subagents. The other
    # two arms both encode one human's answer to "how should a founding team be
    # organised?"; this one asks the model instead, so it measures orchestration
    # as well as coding.
    "dynamic": {
        "agents": "dynamic",
        "collab": "local",
        "turns": 3,
        "structure": "dynamic",
        "browser_tools": True,
        # This arm needs its own wall clock, well past the 90-minute fleet
        # default. A team turn is ONE model turn; a dynamic turn can contain the
        # whole team's work, and a model that fans out runs its subagents inside
        # the single process being timed. Measured on Claude Opus 5, which
        # designed itself a five-subagent team on an *easy* idea: a single
        # orchestrator turn ran ~95 minutes. At the fleet default that build is
        # SIGKILLed at 90 and recorded harness_timeout -- our limit biting the
        # model that orchestrated hardest, which is the fabricated capability
        # difference of docs/crowd_bugs.md T0.1 in its purest form.
        #
        # Raised 4h -> 6h when the brief started telling the founder how to earn
        # a second turn (prompts.py `_delegation_brief`), since that is the only
        # way it can use an agent it defined for itself. Two effects stack on the
        # wall clock and 4h covered neither. A model that takes the second turn
        # roughly doubles its runtime, and Opus 5's SINGLE turn was already ~101
        # minutes measured, so two turns is ~3.4h with nothing spare. And this
        # arm is now built at higher concurrency, which does not fail builds:
        # opencode absorbs provider rate limits inside its own retry loop, so
        # quota pressure arrives as slower turns rather than errors. Both
        # convert a working build into harness_timeout at a tight cap -- the
        # T0.1 artifact again, landing hardest on the models that orchestrate
        # most, which is exactly the signal this arm exists to measure. 6h still
        # bounds a wedged cell.
        "timeout_s": 21600,
    },
}

#: Default structure. Kept as the existing fleet's shape so the builds already
#: indexed stay valid and are never rebuilt.
DEFAULT_STRUCTURE = "team"

#: Back-compat alias for the original single-config name.
FOUNDER_CONFIG: dict[str, object] = STRUCTURES[DEFAULT_STRUCTURE]

_BUILD_LINE = re.compile(r"^Build:\s+(\S+)", re.MULTILINE)

#: idea_id -> Idea, for fingerprinting the current brief.
_IDEA_BY_ID = {idea.idea_id: idea for idea in load_ideas()}

#: Minimum gap between launching two opencode processes.
#:
#: opencode keeps its session state in a shared SQLite database under
#: ~/.local/share/opencode. Ten of them starting in the same instant contend on
#: it, and one loses: "Error: Unexpected error / database is locked", four
#: seconds in, recorded as a harness failure on whichever cell happened to be
#: unlucky. That is a coin flip deciding a model's build outcome, so the starts
#: are spaced instead. Costs 18 s across a 50-build fleet.
_START_GAP_S = 2.0
_start_gate = threading.Lock()
_last_start = 0.0


def _stagger_start() -> None:
    """Block until at least ``_START_GAP_S`` has passed since the last launch."""
    global _last_start
    with _start_gate:
        wait = _START_GAP_S - (time.monotonic() - _last_start)
        if wait > 0:
            time.sleep(wait)
        _last_start = time.monotonic()


@dataclass(frozen=True)
class Cell:
    """One (idea, model) fleet cell."""

    idea_id: str
    model: str
    #: Which independent build of this (idea, model) pair. Replicate 1 is the
    #: original fleet; 2+ re-build the SAME brief under the SAME config, which is
    #: how build-to-build variance gets measured. Founder models are sampled at
    #: temperature, so a re-run is a genuine second draw, not a cache hit.
    replicate: int = 1
    #: Which founder configuration built it (see :data:`STRUCTURES`).
    structure: str = DEFAULT_STRUCTURE

    @property
    def config(self) -> dict[str, object]:
        return STRUCTURES[self.structure]

    @property
    def key(self) -> str:
        # Replicate 1 of the default structure keeps its original un-suffixed key
        # so the existing fleet index stays valid and is never rebuilt.
        suffix = "" if self.replicate == 1 else f"::r{self.replicate}"
        struct = "" if self.structure == DEFAULT_STRUCTURE else f"::{self.structure}"
        return f"{self.idea_id}::{self.model}{struct}{suffix}"


def build_record(build_id: str) -> dict | None:
    """Return the persisted build.json for ``build_id`` (None if unreadable)."""
    path = REPO / "builds" / "work" / build_id / "build.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def config_matches(record: dict, structure: str = DEFAULT_STRUCTURE) -> bool:
    """True if ``record`` was produced under the named founder configuration.

    Checked per structure, not against one global config: a solo build and a team
    build of the same idea are different experiments, and counting one as the
    other would quietly mix the arms of the comparison.

    The dynamic arm is verified on its own terms. Its team size and turn count
    are OUTCOMES -- the model decides how many subagents to run, and stops as
    soon as it declares the build complete -- so checking them the way the fixed
    arms are checked would reject every build whose model finished early or
    delegated at all. What is fixed, and therefore what is verified, is the
    structure, the toolset and the turn cap it was given.
    """
    cfg = STRUCTURES[structure]
    if cfg["structure"] == "dynamic":
        return (
            record.get("structure") == "dynamic"
            and record.get("collab") == cfg["collab"]
            and record.get("max_turns") == cfg["turns"]
        )
    return (
        record.get("structure") == cfg["structure"]
        and record.get("n_agents") == cfg["agents"]
        and record.get("collab") == cfg["collab"]
        and record.get("max_rounds") == cfg["rounds"]
        and record.get("min_rounds") == cfg["min_rounds"]
    )


def load_fleet() -> dict:
    """Load the fleet index, or an empty one."""
    try:
        data = json.loads(FLEET_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"config": dict(FOUNDER_CONFIG), "models": {}, "entries": {}}
    data.setdefault("config", dict(FOUNDER_CONFIG))
    data.setdefault("entries", {})
    return data


def _short_model(model: str) -> str:
    """Strip ANY provider prefix so 'google-vertex/x' and 'x' compare equal.

    Two deliberate details, both of which decide whether a resume rebuilds work
    that is already done:

    * Applied to BOTH sides of the comparison below, not just the build record.
      A cell may legitimately be named either way -- ``--models
      gemini-2.0-flash`` or the fully-qualified
      ``google-vertex-anthropic/gemini-2.0-flash`` -- and comparing a
      stripped record against a qualified cell would never match.
    * Strips any prefix, not just providers we currently know. This fleet already
      contains builds recorded under the legacy ``google/`` prefix (11 of them,
      from before the move to ``google-vertex``). Being strict about known
      providers would leave those unmatched and silently rebuild them, which
      costs real money and reruns a cell that already has a result. No Vertex
      model id contains a slash, so splitting on the last one is safe.
    """
    return model.rsplit("/", 1)[-1]


def cell_ok(fleet: dict, cell: Cell) -> bool:
    """True if this cell already has a healthy build under the frozen config."""
    entry = fleet["entries"].get(cell.key)
    if not entry or not entry.get("build_id"):
        return False
    record = build_record(entry["build_id"])
    if record is None:
        return False
    return (
        record.get("status") == "ok"
        and _short_model(record.get("model", "")) == _short_model(cell.model)
        and record.get("idea_id") == cell.idea_id
        and config_matches(record, cell.structure)
        and brief_matches(record, cell.idea_id)
    )


def brief_matches(record: dict, idea_id: str) -> bool:
    """True if the build was given the CURRENT brief for this idea.

    Matching only the founder shape is not enough. When the corpus and prompts
    changed in the web-dev pivot, 12 builds from six days earlier still counted
    as current results -- they had been told to "strongly prefer plain static
    files" and that "no backend required", the opposite of what the bench now
    asks. Reusing them as one arm of a structure comparison would have measured
    the prompt rewrite and reported it as a difference between founder teams.

    A record with no fingerprint predates the field, which means it predates the
    pivot, so it is stale by definition.
    """
    stored = record.get("brief_fingerprint") or ""
    if not stored:
        return False
    idea = _IDEA_BY_ID.get(idea_id)
    return idea is not None and stored == brief_fingerprint(idea)


#: Substrings in a failed build's output that identify OUR infrastructure
#: failing, not the model failing to build the app.
#:
#: opencode keeps its session state in a shared SQLite database under
#: ~/.local/share/opencode, and concurrent builds contend on it. When one loses,
#: opencode dies with an internal error and the build is recorded
#: ``harness_failed`` -- a status deliberately NOT auto-retried, on the reasoning
#: that it is "a real (if rare) model outcome". For a lost database lock it is
#: nothing of the kind: it is a coin flip on our side being written down as the
#: model's inability to build an app, which is precisely the fabricated
#: capability difference docs/crowd_bugs.md T0.1 warns about. Observed live:
#: db_schema_designer/solo died in 44s with "Failed to execute statement".
_INFRA_FAILURE_SIGNS = (
    "failed to execute statement",
    "database is locked",
    "sqlite_busy",
    "unexpected error / database",
    "econnreset",
    "socket hang up",
)

#: Substrings identifying a PROVIDER REFUSAL: the model's output was blocked by a
#: safety filter before it could finish.
#:
#: A third category, and it needs to be, because the two existing ones both give
#: the wrong answer for it.
#:
#: It is not ``harness_infra``. That status means "ours, so retry it", and
#: retrying a refusal hands the affected model extra draws that no other cell
#: gets -- the same control violation that keeps ``manifest_missing`` off the
#: auto-retry list.
#:
#: It is not ``manifest_missing`` either, which is a MODEL result and is correctly
#: scored at the floor: the model was asked to found an app, ran its turns, and
#: shipped nothing usable. A refusal says nothing of the sort. Observed live,
#: typing_speed_test/dynamic/gemini-2.0-flash died in 108s to "Output blocked by
#: content filtering policy" -- on a TYPING SPEED TEST, from a model that built
#: the other 24 ideas fine. Flooring that would confound provider POLICY with
#: model CAPABILITY, and asymmetrically: whichever provider filters hardest loses
#: score, exactly as the Developer-API-vs-Vertex gap once made a rate limit look
#: like a quality difference.
#:
#: So it is terminal AND unscored: the cell is not retried, and it leaves the
#: denominator rather than sitting at the floor. Like every other exclusion here
#: it has to be reported, not merely dropped -- see FleetCorpus.refusals().
_REFUSAL_SIGNS = (
    "output blocked by content filtering policy",
    "blocked by content filtering",
    "content_filter",
    "response was blocked",
)


def _is_infra_failure(stdout: str, stderr: str) -> str:
    """Return the matching infrastructure signature, or "" if none."""
    blob = f"{stdout}\n{stderr}".lower()
    for sign in _INFRA_FAILURE_SIGNS:
        if sign in blob:
            return sign
    return ""


def _is_refusal(stdout: str, stderr: str) -> str:
    """Return the matching provider-refusal signature, or "" if none."""
    blob = f"{stdout}\n{stderr}".lower()
    for sign in _REFUSAL_SIGNS:
        if sign in blob:
            return sign
    return ""


def run_cell(cell: Cell, *, timeout_s: float) -> dict:
    """Run one founder build in its own network namespace; return its entry."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = (
        LOG_DIR
        / f"{cell.idea_id}__{cell.model}__{cell.structure}__r{cell.replicate}.log"
    )
    cfg = cell.config
    argv = [
        str(NETNS),
        "uv",
        "run",
        "viral-bench",
        "found",
        cell.idea_id,
        "--model",
        cell.model,
        "--agents",
        str(cfg["agents"]),
        "--collab",
        str(cfg["collab"]),
    ]
    if cfg["structure"] == "dynamic":
        argv += ["--turns", str(cfg["turns"])]
    # --rounds/--min-rounds only apply to the multi-agent team; passing them for
    # a solo build would be rejected by the CLI.
    elif int(cfg["agents"]) > 1:
        argv += ["--rounds", str(cfg["rounds"]), "--min-rounds", str(cfg["min_rounds"])]
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.local' / 'bin'}:{env.get('PATH', '')}"
    # Give this build its OWN opencode state directory.
    #
    # opencode keeps every session it has ever run in one SQLite database under
    # $XDG_DATA_HOME/opencode. On this machine that file had grown to 41 GB, and
    # concurrent builds contend on it: the loser dies with "Failed to execute
    # statement" and the cell is recorded as a build failure. Measured live at
    # concurrency 8, that was 3 of the first 14 cells (21%) -- a coin flip on our
    # side being written down as the model's inability to build an app.
    #
    # Isolating the directory removes the contention rather than retrying it, and
    # keeps the shared database from growing without bound. Nothing else lives
    # there: opencode's credentials are not in this directory, so a private one
    # costs nothing.
    cell_state = OPENCODE_STATE_ROOT / f"{cell.key.replace('::', '__')}"
    shutil.rmtree(cell_state, ignore_errors=True)
    cell_state.mkdir(parents=True, exist_ok=True)
    env["XDG_DATA_HOME"] = str(cell_state)
    # A structure may need a different wall-clock cap than the fleet default.
    # The dynamic arm does: one of its turns can contain a whole fan-out of
    # subagent runs, so a cap sized for a team turn would SIGKILL working builds
    # and record them as harness_timeout -- our limit biting, dressed up as the
    # model being unable to build an app.
    cell_timeout = float(cfg.get("timeout_s") or timeout_s)
    _stagger_start()
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=cell_timeout,
        )
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        rc = 124
    elapsed = time.time() - started
    log_path.write_text(
        f"$ {' '.join(argv)}\nrc={rc} elapsed={elapsed:.0f}s\n"
        f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n",
        encoding="utf-8",
    )

    match = _BUILD_LINE.search(stdout or "")
    build_id = match.group(1) if match else ""
    record = build_record(build_id) if build_id else None

    status = (record or {}).get("status", "no_record")
    if status == "ok":
        # Only discard a successful cell's opencode state; a failed one still
        # holds the session transcript that explains why.
        shutil.rmtree(cell_state, ignore_errors=True)
    infra = ""
    if status in ("harness_failed", "no_record"):
        # Refusal first: it is the more specific reading of the same output, and
        # the two lead to opposite handling (terminal + unscored vs retry).
        infra = _is_refusal(stdout or "", stderr or "")
        if infra:
            status = "provider_refusal"
        else:
            infra = _is_infra_failure(stdout or "", stderr or "")
            if infra:
                # Retryable, and labelled so the retry is visible rather than
                # looking like a model that failed and was quietly given a
                # mulligan.
                status = "harness_infra"

    return {
        "idea_id": cell.idea_id,
        "model": cell.model,
        "replicate": cell.replicate,
        "build_id": build_id,
        "returncode": rc,
        "elapsed_s": round(elapsed, 1),
        "status": status,
        "infra_signature": infra or None,
        "rounds_run": (record or {}).get("rounds_run"),
        "turns_spent": (record or {}).get("turns_spent"),
        "shipped_early": (record or {}).get("shipped_early"),
        "qa_verified": (record or {}).get("qa_verified"),
        "error": (record or {}).get("error"),
        "finished_at": datetime.now(UTC).isoformat(),
        "log": str(log_path.relative_to(REPO)),
    }


def cell_attempted(fleet: dict, cell: Cell) -> bool:
    """True if this cell has already been attempted at all (ok or not).

    Attempts are counted, not just successes, because retrying is a control
    violation waiting to happen: quietly re-running only the cells one model
    failed gives that model N attempts against the other's one, and the fleet
    stops being a fair comparison. A retry must be asked for by name (--retry)
    and is recorded in ``attempts``.
    """
    return cell.key in fleet["entries"]


def _recorded_before(entry: dict, cutoff: str | None) -> bool:
    """Was this cell's stored result recorded before ``cutoff``?

    No cutoff means no bound, so every named retry qualifies. An entry with no
    ``finished_at`` predates the field and so predates any cutoff worth naming.
    Timestamps are ISO-8601 UTC and compared as strings, which orders correctly
    for that format and cannot raise on a malformed value the way parsing can --
    this decides whether a build runs, so it must not throw.
    """
    if not cutoff:
        return True
    return str(entry.get("finished_at") or "") < cutoff


def pending_cells(
    fleet: dict,
    models: list[str],
    retry: frozenset[str] = frozenset(),
    replicate: int = 1,
    structures: tuple[str, ...] = (DEFAULT_STRUCTURE,),
    retry_before: str | None = None,
) -> list[Cell]:
    """Fleet cells that still need a build, ordered to spread load.

    Cells are interleaved by idea and alternated by model so that (a) concurrent
    builds are always of *different* ideas, and (b) an interrupted fleet is
    balanced across models rather than complete for one and empty for the other.

    A cell is pending if it has never been attempted, or if its recorded status
    is in ``retry`` -- which the caller must name explicitly. ``driver_error``,
    ``no_record`` and ``harness_timeout`` are always retryable: those are OUR
    failures, not the model's, and leaving them in place would report a harness
    bug as a model capability (docs/crowd_bugs.md T0.1).

    ``harness_timeout`` qualifies on exactly that rule -- it means our wall-clock
    backstop killed a turn that was still working, so the cell holds no evidence
    about the model at all. It used to be indistinguishable from
    ``harness_failed``, which is a real (if rare) model outcome and so is
    deliberately NOT auto-retried.

    ``retry_before`` bounds the NAMED retries to results recorded before an
    instant, which is what makes "rebuild this arm, one fresh draw per cell"
    survive a restart. Rebuilding a whole arm may legitimately re-attempt a
    model-attributable status such as ``manifest_missing``, because every cell
    gets exactly one new draw and none is singled out. Unbounded, that stops
    being true the moment the driver is restarted -- a cell whose fresh draw
    already landed and re-failed is simply offered again, so the models that fail
    this way collect extra draws while the models that succeed collect none, and
    the arm silently becomes best-of-N for the weakest founders. Pass the instant
    the rebuild started; the always-ours statuses are deliberately not bounded by
    it, since those hold no evidence about any model at all.
    """
    always_retry = {"driver_error", "no_record", "harness_timeout", "harness_infra"}
    ideas = sorted(idea.idea_id for idea in load_ideas())
    ordered: list[Cell] = []
    for idea_id in ideas:
        for model in models:
            for structure in structures:
                ordered.append(Cell(idea_id, model, replicate, structure))
    out: list[Cell] = []
    for cell in ordered:
        if cell_ok(fleet, cell):
            continue
        if not cell_attempted(fleet, cell):
            out.append(cell)
            continue
        entry = fleet["entries"].get(cell.key) or {}
        status = entry.get("status", "")
        if status in always_retry:
            out.append(cell)
            continue
        if status in retry and _recorded_before(entry, retry_before):
            out.append(cell)
            continue
        # A cell whose BUILD succeeded but whose brief or founder config has
        # since moved is not a failure to retry -- it is a result that no longer
        # describes the current experiment, and it must be rebuilt or that arm
        # stays permanently short. Without this it was skipped forever: not `ok`
        # (so it never counted) and not a failure status (so it was never
        # re-run), which would have left the pre-pivot arm empty while the report
        # said only "attempted".
        if status == "ok":
            record = build_record(entry.get("build_id") or "")
            if record is None or not (
                config_matches(record, cell.structure)
                and brief_matches(record, cell.idea_id)
            ):
                out.append(cell)
    return out


def report(
    fleet: dict,
    models: list[str],
    rep: int = 1,
    structures: tuple[str, ...] = (DEFAULT_STRUCTURE,),
    retry: frozenset[str] = frozenset(),
    retry_before: str | None = None,
) -> str:
    """Render the fleet's state for the same ``--retry`` the caller will build with.

    ``retry`` is not decoration. Without it the queue line answered a different
    question from the one the operator asked: ``--status --retry
    manifest_invalid,manifest_missing`` reported a pending count that left out
    every named-retry cell the build it was previewing would have run. Two
    consequences, both live: an operator cannot confirm the queue before
    committing hours to it, and a driver that derives its stop condition from
    this number declares an arm COMPLETE while those cells are still unbuilt.
    """
    ideas = sorted(idea.idea_id for idea in load_ideas())
    lines = [
        f"fleet replicate {rep}: {len(ideas)} ideas x {len(models)} models "
        f"x {len(structures)} structures ({', '.join(structures)})"
    ]
    for structure in structures:
        for model in models:
            cells = [Cell(i, model, rep, structure) for i in ideas]
            ok = sum(1 for c in cells if cell_ok(fleet, c))
            tried = sum(1 for c in cells if c.key in fleet["entries"])
            lines.append(
                f"  {structure:<6} {model:<20} "
                f"ok={ok:>2}/{len(ideas)}  attempted={tried}"
            )

    def _why(c: Cell) -> str:
        """Why a cell that was attempted is not usable."""
        entry = fleet["entries"].get(c.key) or {}
        status = entry.get("status", "?")
        if status != "ok":
            return status
        # The build itself succeeded, so it is the brief or the shape that moved.
        record = build_record(entry.get("build_id") or "")
        if record is None:
            return "record missing"
        if not config_matches(record, c.structure):
            return "built under a different founder config"
        if not brief_matches(record, c.idea_id):
            return "STALE BRIEF (idea/prompts changed since it was built)"
        return status

    attempted_not_ok = [
        f"{c.idea_id}[{c.model}/{c.structure}]: {_why(c)}"
        for structure in structures
        for m in models
        for c in (Cell(i, m, rep, structure) for i in ideas)
        if cell_attempted(fleet, c) and not cell_ok(fleet, c)
    ]
    if attempted_not_ok:
        lines.append(f"  attempted but not ok ({len(attempted_not_ok)}):")
        for item in attempted_not_ok:
            lines.append(f"    ! {item}")
    missing = pending_cells(
        fleet,
        models,
        retry,
        replicate=rep,
        structures=structures,
        retry_before=retry_before,
    )
    # "never attempted" was the old label and it was never quite true -- a cell
    # whose build succeeded under a since-changed founder config is queued here
    # too, and it was very much attempted. Under --retry it became actively
    # misleading. "to build" is what the number has always meant.
    suffix = f", incl. --retry {','.join(sorted(retry))}" if retry else ""
    if retry and retry_before:
        suffix += f" recorded before {retry_before}"
    lines.append(f"  pending (to build{suffix}): {len(missing)}")
    for cell in missing[:10]:
        entry = fleet["entries"].get(cell.key, {})
        why = entry.get("status", "never built")
        lines.append(f"    - {cell.idea_id} [{cell.model}/{cell.structure}] ({why})")
    if len(missing) > 10:
        lines.append(f"    ... and {len(missing) - 10} more")
    return "\n".join(lines)


def reclassify_infra(dry_run: bool = False) -> list[tuple[str, str]]:
    """Relabel stored cells whose failure we NOW recognise as ours.

    A cell's status is decided once, when it is built. Add a signature to
    ``_INFRA_FAILURE_SIGNS`` afterwards and every cell already condemned under the
    old list stays condemned, because ``harness_failed`` is deliberately not
    auto-retried -- so the fix silently never reaches the builds it was written
    for. This applies the CURRENT classifier to what is already on disk.

    Only ever relaxes a verdict in our own direction (harness_failed / no_record
    -> harness_infra), never the reverse, so it cannot launder a genuine model
    failure into a free retry.
    """
    changed: list[tuple[str, str]] = []
    with open(FLEET_LOCK_PATH, "w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        fleet = load_fleet()
        # harness_infra is included as a SOURCE state, not just a target: a
        # refusal was briefly classified that way before it got its own status,
        # and leaving those behind would auto-retry the exact cells that must not
        # be retried.
        for key, entry in fleet.get("entries", {}).items():
            if entry.get("status") not in (
                "harness_failed",
                "no_record",
                "harness_infra",
            ):
                continue
            error = entry.get("error") or ""
            sign = _is_refusal(error, "")
            target = "provider_refusal"
            if not sign:
                if entry.get("status") == "harness_infra":
                    continue  # already correct
                sign = _is_infra_failure(error, "")
                target = "harness_infra"
            if not sign:
                continue
            changed.append((key, f"{entry.get('status')} -> {target}: {sign}"))
            if not dry_run:
                entry["status"] = target
                entry["infra_signature"] = sign
        if changed and not dry_run:
            tmp = FLEET_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(fleet, indent=2), encoding="utf-8")
            tmp.replace(FLEET_PATH)
    return changed


def merge_fleet_entries(
    entries: dict,
    *,
    meta: dict | None = None,
    structures: dict | None = None,
) -> None:
    """Merge exactly ``entries`` into the on-disk index, atomically and locked.

    Takes the entries to write rather than a whole fleet snapshot, and that
    signature is the fix. Two things have to be true at once for a shared index:
    the write must not tear (an exclusive lock plus ``os.replace``), and it must
    not carry stale keys the caller does not own.

    Only the first was ever true. The old code re-read the index under the lock
    and then layered the caller's entire startup snapshot -- every key it had
    ever seen -- over it, so every write faithfully restored the OTHER arm's keys
    to whatever they were when this process started. Observed by polling two keys
    every 10s while two arms were built side by side: one persist put
    ``form_builder[…/solo]`` back by 7 days while restoring
    ``group_scheduling_poll[…/team]`` to today, and the next swapped which was
    current. Both arms silently reverted each other's results for hours, and
    neither could see it, because each only ever read back its own writes.

    The lock was never the problem, so no amount of locking was going to fix it.
    A stale payload written atomically is still a lost update. Passing only the
    finished keys makes the whole class of bug unrepresentable: there is nothing
    here to write that the caller does not own.
    """
    FLEET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FLEET_LOCK_PATH, "w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            merged = load_fleet()
            merged.setdefault("entries", {}).update(entries)
            for key, value in (meta or {}).items():
                merged[key] = value
            if structures:
                merged["structures"] = {
                    **(merged.get("structures") or {}),
                    **structures,
                }
            merged["updated_at"] = datetime.now(UTC).isoformat()
            tmp = FLEET_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            os.replace(tmp, FLEET_PATH)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=0, help="build at most N cells (0 = all)"
    )
    parser.add_argument(
        "--models",
        required=True,
        help="comma-separated '<provider>/<model>' pair to compare, "
        "e.g. openai/gpt-5-mini,anthropic/claude-sonnet-4-5",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0 * 60,
        help="per-build wall clock cap. Sized from the observed distribution "
        "(solo ~260s, team 1100-1800s) with a 3x margin, NOT set generously "
        "'to be safe': a hung build holds a fleet slot for the whole cap, and at "
        "the previous 4h default two hung team cells blocked the fleet for 83 "
        "minutes after writing nothing to disk for 80 of them. Killing a slow "
        "build is cheap now that a timeout is recorded as harness_timeout and "
        "auto-retried, so it costs a rebuild rather than a fabricated failure.",
    )
    parser.add_argument("--status", action="store_true", help="report, build nothing")
    parser.add_argument(
        "--reclassify-infra",
        action="store_true",
        help="re-run the infra classifier over stored entries, relabelling any "
        "harness_failed/no_record whose error now matches as harness_infra (which "
        "is auto-retryable). Needed whenever a signature is added: a verdict is "
        "written at build time, so a cell condemned before the harness learned to "
        "recognise its failure stays condemned forever.",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="which independent build of each (idea, model) to produce. 1 is the "
        "original fleet; 2+ re-build the SAME briefs under the SAME config, which "
        "is how build-to-build variance gets measured. Founder models are sampled "
        "at temperature, so a re-run is a genuine second draw.",
    )
    parser.add_argument(
        "--retry",
        default="",
        help="comma-separated build statuses to re-attempt (e.g. harness_failed). "
        "Retrying only one model's failures biases the comparison -- say why in "
        "the iteration log whenever you use this.",
    )
    parser.add_argument(
        "--retry-before",
        default="",
        help="only re-attempt a --retry status whose result was recorded before "
        "this ISO-8601 instant. Use it whenever a whole-arm rebuild retries a "
        "model-attributable status: it makes 'one fresh draw per cell' hold "
        "across a restart, instead of re-offering cells whose fresh draw has "
        "already landed and failed -- which quietly turns the arm into "
        "best-of-N for the weakest founders.",
    )
    parser.add_argument(
        "--structures",
        default=DEFAULT_STRUCTURE,
        help="comma-separated founder configurations to build: "
        + ",".join(STRUCTURES)
        + f" (default: {DEFAULT_STRUCTURE}). Each is a separate arm of the "
        "comparison and is verified against every build record, so builds made "
        "under one shape are never counted toward another's results.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.reclassify_infra:
        changed = reclassify_infra()
        for key, what in changed:
            print(f"  {key}\n      {what}")
        print(f"{len(changed)} cell(s) relabelled")
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    structures = tuple(s.strip() for s in args.structures.split(",") if s.strip())
    unknown = [s for s in structures if s not in STRUCTURES]
    if unknown:
        print(
            f"error: unknown structure(s) {', '.join(unknown)}; "
            f"known: {', '.join(STRUCTURES)}",
            file=sys.stderr,
        )
        return 2
    fleet = load_fleet()
    fleet["models"] = {"A": MODEL_A, "B": MODEL_B}
    # Record every structure this index has ever built under, not just the
    # current run's, so fleet.json documents the whole comparison.
    known = dict(fleet.get("structures") or {})
    known.update({name: dict(STRUCTURES[name]) for name in structures})
    fleet["structures"] = known
    fleet["config"] = dict(FOUNDER_CONFIG)  # back-compat: the default arm

    retry = frozenset(s.strip() for s in args.retry.split(",") if s.strip())
    retry_before = args.retry_before or None

    if args.status:
        print(report(fleet, models, args.replicate, structures, retry, retry_before))
        return 0

    todo = pending_cells(
        fleet,
        models,
        retry,
        replicate=args.replicate,
        structures=structures,
        retry_before=retry_before,
    )
    if args.limit:
        todo = todo[: args.limit]
    print(report(fleet, models, args.replicate, structures, retry, retry_before))
    print(f"\nbuilding {len(todo)} cells at concurrency {args.concurrency}")
    if args.dry_run or not todo:
        for cell in todo:
            print(f"  would build {cell.idea_id} [{cell.model}/{cell.structure}]")
        return 0

    # netns_run.sh is not optional: without it every build shares one loopback and
    # one /proc, which is how build B's QA ends up reviewing build A's app. It
    # exits 127 per cell when these are missing and run_cell records that as
    # "no_record", so a whole fleet would grind through producing nothing. Check
    # once, here, rather than 50 times, slowly.
    missing = [tool for tool in ("unshare", "pasta") if shutil.which(tool) is None]
    if missing:
        print(
            f"error: {', '.join(missing)} not on PATH, so {NETNS.name} cannot "
            f"isolate a build (Debian/Ubuntu: apt install util-linux passt). "
            f"Building without isolation would let concurrent builds share port "
            f"8000 and kill each other's app servers.",
            file=sys.stderr,
        )
        return 2

    fleet.setdefault("started_at", datetime.now(UTC).isoformat())
    lock = threading.Lock()
    #: Keys THIS process has itself finished. Only these may be written back --
    #: see persist().
    completed_keys: set[str] = set()

    def persist() -> None:
        """Write back exactly the cells THIS process has finished.

        Deliberately does NOT take ``lock``: it is called from inside the
        completion handler's critical section, and ``threading.Lock`` is not
        reentrant, so re-acquiring here would deadlock the pool on its very
        first finished cell.
        """
        mine = {
            key: fleet["entries"][key]
            for key in completed_keys
            if key in fleet["entries"]
        }
        merge_fleet_entries(
            mine,
            meta={
                k: fleet[k] for k in ("models", "config", "started_at") if k in fleet
            },
            structures=fleet.get("structures") or {},
        )

    persist()
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(run_cell, cell, timeout_s=args.timeout): cell for cell in todo
        }
        for future in as_completed(futures):
            cell = futures[future]
            try:
                entry = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad cell must not stop 49
                entry = {
                    "idea_id": cell.idea_id,
                    "model": cell.model,
                    "build_id": "",
                    "status": "driver_error",
                    "error": repr(exc),
                    "finished_at": datetime.now(UTC).isoformat(),
                }
            done += 1
            with lock:
                prior = fleet["entries"].get(cell.key) or {}
                entry["attempts"] = int(prior.get("attempts", 0)) + 1
                fleet["entries"][cell.key] = entry
                completed_keys.add(cell.key)
                fleet["updated_at"] = datetime.now(UTC).isoformat()
                persist()
            print(
                f"[{done}/{len(todo)}] {entry['status']:<16} "
                f"{cell.idea_id} [{cell.model}/{cell.structure}] "
                f"rounds={entry.get('rounds_run')} {entry.get('elapsed_s')}s",
                flush=True,
            )

    print()
    print(report(fleet, models, args.replicate, structures, retry, retry_before))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
