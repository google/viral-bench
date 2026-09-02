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

"""Run the crowd over the frozen fleet: every healthy build, several seeds.

This is the loop's inner cycle. The fleet is capital spent once. This is the
part that is cheap enough to redo whenever the architecture changes -- a run is
1-3 minutes and re-scoring afterwards is free.

What it is careful about:

* **The negative control runs in the same sweep as the real apps.** A control
  measured under last week's crowd proves nothing about this week's. It is
  included by default, with the same config and the same seeds.
* **Resumable by (build, seed).** Existing runs of the current crowd
  architecture are detected and skipped, so widening a sweep only pays for the
  cells it adds.
* **Failures are recorded, not retried into existence.** A run that fails is
  reported, and the scoring stage counts it as unscorable rather than dropping
  it.
* **A cell killed at the wall clock is remembered.** It writes no
  ``run_summary.json``, so nothing else on disk distinguishes it from a cell
  never tried, and every later pass used to buy it another slot-hour. The
  ledger under ``builds/crowd/.timeouts.json`` closes that hole. Only rc=124 is
  suppressed, and only after ``--max-timeout-attempts``. Every other failure
  stays retryable, because those are the ones a retry recovers.
* **The autorater runs here, not later.** Its three dimensions carry 15% of the
  active profile, and NOT ONE stored run had an ``autorating.json``,
  so that 15% was being silently renormalised away and the "hybrid" score was
  deterministic in every number ever reported. Rating is 3 cheap LLM calls per
  run against artifacts already on disk, so it belongs in the sweep that
  produces them.

Usage::

    scripts/crowd_sweep.py --seeds 3 --concurrency 8
    scripts/crowd_sweep.py --ideas quick_notes_app,markdown_slides --seeds 2
    scripts/crowd_sweep.py --status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.crowd.sim_defaults import (  # noqa: E402
    CROWD_ARCH_VERSION,
    DEFAULT_AGENTS,
    DEFAULT_ROUNDS,
    DEFAULT_TRIERS,
)
from viral_bench.founder.prompts import brief_fingerprint  # noqa: E402
from viral_bench.founder.provision import provision_build  # noqa: E402
from viral_bench.founder.runner import (  # noqa: E402
    reclaim_trash,
    sweep_run_dirs,
)
from viral_bench.ideas import load_ideas  # noqa: E402
from viral_bench.score.fleet import (  # noqa: E402
    CURRENT_FLEET,
    apply_fleet_status,
    fleet_replicates,
    load_builds,
    structure_name,
)

_IDEAS = {i.idea_id: i for i in load_ideas()}

BUILDS = REPO / "builds"
CROWD_DIR = BUILDS / "crowd"
LOG_DIR = BUILDS / "sweep_logs"
#: Cells killed at the wall clock, per architecture. See ``timeouts_seen``.
TIMEOUT_LEDGER = CROWD_DIR / ".timeouts.json"
#: Both controls, swept alongside the fleet in the same conditions.
#:
#: The broken one calibrates the floor. The working full-stack one calibrates
#: the CONTRACT: it is deliberately correct -- migrations run, one agent's write
#: is visible to another, state survives a restart -- so if the crowd stops
#: reporting persistence or multi-user visibility on THIS build, the harness has
#: broken, not the fleet. Measured on it under arch v10: work_survived 30/30,
#: saw_other_users 30/30. It is not a ceiling: it is a plain memo board and the
#: crowd correctly finds it dull (adoption 0.13), which is itself the useful
#: demonstration that "works" and "wanted" are separate measurements here.
CONTROL_BUILDS = (
    "quick_notes_app__20260728-000000__brokn0",
    "quick_notes_app__20260806-000000__fsctl0",
)


@dataclass(frozen=True)
class Cell:
    build_id: str
    seed: int


def existing_runs() -> dict[tuple[str, int], list[Path]]:
    """Map (build_id, seed) -> HEALTHY run dirs recorded for THIS architecture.

    A run that failed its health check is not coverage: it produced no usable
    verdicts, so the cell still needs measuring. Re-running writes a new
    timestamped directory and deletes nothing, so the failure stays on disk and
    stays counted in the scorable-run fraction: the loop is told what it cost,
    not only what it got.
    """
    out: dict[tuple[str, int], list[Path]] = {}
    if not CROWD_DIR.is_dir():
        return out
    for summary_path in CROWD_DIR.glob("*/run_summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(data.get("crowd_arch_version", "0")) != CROWD_ARCH_VERSION:
            continue
        if not data.get("ok"):
            continue
        key = (
            data.get("build_id", ""),
            int((data.get("config") or {}).get("seed", 0) or 0),
        )
        out.setdefault(key, []).append(summary_path.parent)
    return out


def failed_attempts() -> dict[tuple[str, int], int]:
    """Map (build_id, seed) -> how many previous attempts left nothing usable.

    ``run_cell`` creates its output directory before the run starts and only
    writes ``run_summary.json`` if the run completes, so a directory without one
    is the fossil of an attempt that died -- at the wall clock, or in a crash.
    Counting those needs no ledger and no bookkeeping: the evidence is already on
    disk, and it survives a ledger that has been cleared by hand.

    This is a *scheduling* signal, not a suppression one. Nothing is skipped
    because of it. See ``plan_cells`` for what it is used for.
    """
    out: dict[tuple[str, int], int] = {}
    if not CROWD_DIR.is_dir():
        return out
    for run_dir in CROWD_DIR.glob("*__crowd-*"):
        if not run_dir.is_dir() or (run_dir / "run_summary.json").exists():
            continue
        base, _, tail = run_dir.name.rpartition("__crowd-")
        seed_part = tail.rpartition("-s")[2]
        if not base or not seed_part.isdigit():
            continue
        key = (base, int(seed_part))
        out[key] = out.get(key, 0) + 1
    return out


#: Serialises the read-modify-write of the ledger across sweep threads.
_LEDGER_LOCK = threading.Lock()


def timeouts_seen() -> dict[tuple[str, int], int]:
    """Map (build_id, seed) -> how many times that cell hit the wall clock.

    A cell killed at ``--timeout`` writes NO ``run_summary.json``, so
    ``existing_runs`` cannot see that it was ever attempted and the resume check
    offers it up again on the next pass as though it were untouched. At
    concurrency 3 each of those re-offers costs a full slot-hour and returns
    nothing: the retry pass that discovers this is the single largest block of
    unproductive time in a sweep.

    This ledger is the missing record. It is keyed by architecture for the same
    reason ``existing_runs`` filters on one -- bumping ``CROWD_ARCH_VERSION``
    retires the old measurements, and a cell that timed out under the previous
    crowd deserves a fresh attempt under this one.
    """
    try:
        raw = json.loads(TIMEOUT_LEDGER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[tuple[str, int], int] = {}
    for key, count in (raw.get(CROWD_ARCH_VERSION) or {}).items():
        build_id, _, seed = key.rpartition("::")
        if not build_id or not seed.isdigit():
            continue
        out[(build_id, int(seed))] = int(count)
    return out


def record_timeout(cell: Cell) -> None:
    """Add one to this cell's timeout count, atomically.

    Only ``rc == 124`` reaches here. A cell that failed any other way is left
    out on purpose: those are the transient failures -- a 429 storm, a harness
    error -- that the driver's retry pass exists to recover, and they must stay
    retryable.
    """
    with _LEDGER_LOCK:
        try:
            raw = json.loads(TIMEOUT_LEDGER.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        arch = raw.setdefault(CROWD_ARCH_VERSION, {})
        key = f"{cell.build_id}::{cell.seed}"
        arch[key] = int(arch.get(key, 0)) + 1
        TIMEOUT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        tmp = TIMEOUT_LEDGER.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(TIMEOUT_LEDGER)


def plan_cells(
    build_ids: list[str],
    *,
    seeds: int,
    have: dict[tuple[str, int], list[Path]],
    timed_out: dict[tuple[str, int], int],
    max_timeout_attempts: int,
    tried: dict[tuple[str, int], int] | None = None,
) -> tuple[list[Cell], list[Cell]]:
    """Split the grid into cells worth running and cells given up on.

    Seed-major order: every build gets seed 0 before any build gets seed 1, so
    an interrupted sweep is a complete low-seed corpus rather than a few
    over-measured builds and many with nothing.

    Within a seed, cells are ordered by how many previous attempts died without
    producing a summary (``tried``), fewest first. Build ids begin with the idea
    name, so the natural order is alphabetical -- and the apps that hang the
    browser harness cluster at the top of the alphabet (``ai_room_redesign``,
    ``browser_api_client``, ``collaborative_table``, ``image_compressor``). That
    put the worst cells at the head of every pass: in one pass at concurrency
    10, five of the ten slots were held by ``image_compressor`` runs that had
    written nothing for 36 minutes, and the pass produced ZERO completions in its
    first 39 minutes while hundreds of healthy cells waited behind them.

    Ordering costs nothing and drops nothing -- the same cells run either way. It
    only stops a known-bad cell from holding a slot in front of a healthy one, so
    an interrupted sweep has scored as much of the grid as its time allowed.
    """
    tried = tried or {}
    todo: list[Cell] = []
    skipped: list[Cell] = []
    for seed in range(seeds):
        block: list[Cell] = []
        for build_id in build_ids:
            key = (build_id, seed)
            if have.get(key):
                continue
            if (
                max_timeout_attempts > 0
                and timed_out.get(key, 0) >= max_timeout_attempts
            ):
                skipped.append(Cell(build_id, seed))
                continue
            block.append(Cell(build_id, seed))
        block.sort(key=lambda c: (tried.get((c.build_id, c.seed), 0), c.build_id))
        todo.extend(block)
    return todo, skipped


def run_cell(cell: Cell, opts: argparse.Namespace) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Warm this build's dependency cache BEFORE the timed run.
    #
    # A build is materialized by `git clone` of its shipped branch, so it
    # legitimately arrives without the node_modules/.venv its own .gitignore
    # excludes. Without this the crowd is the first thing to discover that the
    # app cannot start, which is how dozens of builds produced nothing at all on
    # an earlier sweep, each after holding a slot for the full wall clock.
    #
    # Outside the subprocess on purpose: an install must not be charged against
    # --timeout. Idempotent and memoized per process, so it costs one container
    # per build for a whole sweep rather than one per cell.
    # getattr, not attribute access: run_cell is called directly by tests and by
    # ad-hoc drivers that build a Namespace by hand, and a new flag must not turn
    # those into AttributeError.
    if not getattr(opts, "no_provision", False):
        try:
            provision_build(
                cell.build_id, timeout=getattr(opts, "provision_timeout", 900.0)
            )
        except Exception as exc:  # noqa: BLE001 - a cold cache must not lose a cell
            print(f"    provision failed {cell.build_id}: {exc!r}", flush=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = CROWD_DIR / f"{cell.build_id}__crowd-{stamp}-s{cell.seed}"
    argv = [
        "uv",
        "run",
        "viral-bench",
        "crowd-run",
        cell.build_id,
        "--agents",
        str(opts.agents),
        "--triers",
        str(opts.triers),
        "--rounds",
        str(opts.rounds),
        "--seed",
        str(cell.seed),
        "--out",
        str(out_dir),
    ]
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.local' / 'bin'}:{env.get('PATH', '')}"
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=opts.timeout,
        )
        rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        record_timeout(cell)
    elapsed = time.time() - started
    (LOG_DIR / f"{cell.build_id}__s{cell.seed}.log").write_text(
        f"$ {' '.join(argv)}\nrc={rc} elapsed={elapsed:.0f}s\n{stdout}\n"
        f"--- stderr (tail) ---\n" + "\n".join((stderr or "").splitlines()[-40:]),
        encoding="utf-8",
    )
    summary_path = out_dir / "run_summary.json"
    summary = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    verdicts = (summary.get("verdicts") or {}).get("interviews") or {}
    return {
        "build_id": cell.build_id,
        "seed": cell.seed,
        "rc": rc,
        "ok": bool(summary.get("ok")),
        "elapsed_s": round(elapsed, 1),
        "interviews": verdicts.get("n"),
        "out_dir": str(out_dir),
    }


def autorate_missing(runs: list[Path], *, concurrency: int) -> tuple[int, int]:
    """Rate every run in ``runs`` that has no ``autorating.json`` yet.

    Separate from the run loop so a resumed sweep repairs earlier runs too: a
    cell whose simulation succeeded but whose rating failed would otherwise be
    skipped forever by the (build, seed) resume check, and the run would be
    scored under a profile it does not satisfy.
    """
    from viral_bench.score.autorater import rate_pack
    from viral_bench.score.evidence import build_evidence_pack

    todo = [d for d in runs if not (d / "autorating.json").is_file()]
    if not todo:
        return (0, 0)
    print(f"autorating {len(todo)} runs at concurrency {concurrency}")

    def _rate(run_dir: Path) -> bool:
        try:
            rating = rate_pack(build_evidence_pack(run_dir))
        except Exception as exc:  # noqa: BLE001 - one bad rating must not stop all
            print(f"  autorate FAILED {run_dir.name}: {exc!r}", flush=True)
            return False
        if not rating.ok:
            print(f"  autorate empty {run_dir.name}: {rating.errors[:2]}", flush=True)
            return False
        (run_dir / "autorating.json").write_text(
            json.dumps(rating.as_dict(), indent=2), encoding="utf-8"
        )
        return True

    done = ok = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_rate, d): d for d in todo}
        for future in as_completed(futures):
            done += 1
            ok += 1 if future.result() else 0
            if done % 25 == 0:
                print(f"  rated {done}/{len(todo)}", flush=True)
    print(f"autorated {ok}/{done} runs")
    return (ok, done)


def healthy_run_dirs(build_ids: set[str]) -> list[Path]:
    """Current-architecture runs that completed, for the given builds."""
    out: list[Path] = []
    for summary_path in sorted(CROWD_DIR.glob("*/run_summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(data.get("crowd_arch_version", "0")) != CROWD_ARCH_VERSION:
            continue
        if not data.get("ok") or data.get("build_id") not in build_ids:
            continue
        out.append(summary_path.parent)
    return out


def build_coverage_line(
    build_ids: list[str], have: dict[tuple[str, int], list[Path]], seeds: int
) -> str:
    """One line of BUILD-level coverage: how many builds, at what depth.

    The grain matters more than the number. A sweep reporting "N builds x 2
    seeds, ~2N runs on disk" sounds complete while dozens of builds have nothing
    at all, and that is what happened: a results document claimed "complete, all
    models x all pipelines" while a whole tail of builds had never scored.
    Runs-on-disk cannot see a hole. Builds-covered can only see holes.
    """
    depths = [
        len({s for (b, s) in have if b == bid and have[(b, s)]}) for bid in build_ids
    ]
    covered = sum(1 for d in depths if d)
    at_target = sum(1 for d in depths if d >= seeds)
    zero = [bid for bid, d in zip(build_ids, depths, strict=True) if not d]
    line = (
        f"builds covered: {covered}/{len(build_ids)} "
        f"({at_target} at the full {seeds} seed(s))"
    )
    if zero:
        line += f"; NO runs for {len(zero)}"
        # Named only once the list is short enough to act on. At the start of a
        # sweep every build is uncovered and printing 250 ids each poll buries the
        # rest of the status. Near the end the names ARE the actionable content,
        # and "99.2% covered" is not.
        if len(zero) <= 25:
            line += ": " + ", ".join(zero)
    return line


def _brief_is_current(build_id: str) -> bool:
    """True if this build was given the brief the corpus currently defines.

    A build with no recorded fingerprint predates the field, and therefore
    predates the web-dev pivot, so it is stale by definition.
    """
    try:
        record = json.loads(
            (BUILDS / "work" / build_id / "build.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    stored = record.get("brief_fingerprint") or ""
    idea = _IDEAS.get(record.get("idea_id"))
    return bool(stored) and idea is not None and stored == brief_fingerprint(idea)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    # Crowd shape comes from config/crowd.yaml, not from a second set of
    # defaults here. It used to be hardcoded 30/8/4, so the sweep silently ran a
    # different instrument from the one the config described, and editing the
    # config changed nothing about the corpus that gets scored.
    parser.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    parser.add_argument(
        "--triers", type=int, default=DEFAULT_TRIERS, help="-1 means every agent"
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    # The ceiling here is the Gemini Developer API quota, not the host: six
    # concurrent 30-agent runs produced a sustained 429 storm that wiped every
    # interview in three of them. Each run already fans out internally
    # (semaphore 12), so 4 runs is ~48 in-flight requests.
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="per-run wall clock cap. A timed-out run yields NOTHING -- no "
        "interviews, no verdicts, no partial credit -- so the cap must sit well "
        "clear of the tail, not near it. At 1800s it was only 1.2x the longest "
        "successful run (1550s) and lost 2 of 64. The tail grew when "
        "interaction.max_steps started taking effect at 40: the median trial is "
        "unchanged at 5 steps but the longest went from 13 to 40, so thorough "
        "triers now explore far more before finishing.",
    )
    parser.add_argument(
        "--max-timeout-attempts",
        type=int,
        default=1,
        help="stop offering a cell once it has hit --timeout this many times "
        "under the current architecture (default: %(default)s; 0 means never "
        "stop). A killed cell writes no run_summary.json, so without this the "
        "resume check cannot tell it apart from one never tried, and every pass "
        "spends another slot-hour on it. Measured recurrence is 3/3. This only "
        "suppresses rc=124: transient failures stay retryable, so the driver's "
        "retry pass keeps doing the job it exists for.",
    )
    parser.add_argument(
        "--autorate-concurrency",
        type=int,
        default=12,
        # NOT tied to --concurrency. That one is pinned to 3 because each crowd
        # run holds `semaphore` (12) requests open, so 3 runs is ~36 in flight
        # and 6 was measured to stall completely. The autorater runs after the
        # simulations are done, one request at a time per worker, against
        # artifacts already on disk -- it is nowhere near that ceiling. If it
        # does get throttled, ratings merely slow down: failures are caught per
        # run and repaired by the next pass.
        help="workers for the autorater pass (default: %(default)s).",
    )
    parser.add_argument(
        "--structures",
        default="",
        help="comma-separated founder structures to sweep (solo,team,dynamic). "
        "Empty means all. Filtering by --ideas is NOT a substitute: an idea "
        "exists in every arm, so --ideas queues every structure of it.",
    )
    # CURRENT_FLEET names the one experiment under test, deliberately: fleet.json
    # is append-only and holds several eras that must never be pooled. But until
    # now a NEW arm could not be swept at all without editing that constant --
    # i.e. editing scoring code in order to run a sweep, which is an easy way to
    # redefine "the fleet" by accident while meaning to do something else. These
    # flags make the choice explicit and per-invocation. They only ever select a
    # different named arm, and they cannot pool two.
    parser.add_argument(
        "--fleet-structure",
        default=CURRENT_FLEET.structure,
        help="which founder arm IS the fleet for this sweep (default: "
        "%(default)s, from score.fleet.CURRENT_FLEET).",
    )
    parser.add_argument(
        "--fleet-replicate",
        type=int,
        default=CURRENT_FLEET.replicate,
        help="which fleet replicate is under test (default: %(default)s).",
    )
    parser.add_argument("--ideas", default="", help="comma-separated idea filter")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument(
        "--no-autorate",
        action="store_true",
        # The doubled %% is load-bearing: argparse `%`-interpolates every help
        # string to expand %(default)s, so a bare "15% of" is read as the %o
        # conversion and `--help` dies with a TypeError instead of printing.
        help="skip the autorater pass. The active profile weights its three "
        "dimensions, so skipping it renormalises 15%% of the score away -- do it "
        "only for a wiring smoke, never for a scored sweep.",
    )
    parser.add_argument(
        "--no-provision",
        action="store_true",
        help="do not warm each build's dependency cache before its first run. "
        "Only for reproducing pre-cache behaviour; a scored sweep wants it on, "
        "or the crowd becomes the thing that discovers an app cannot start.",
    )
    parser.add_argument(
        "--provision-timeout",
        type=float,
        default=900.0,
        help="wall clock for one build's dependency install (default: %(default)s)",
    )
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)

    spec = replace(
        CURRENT_FLEET,
        structure=args.fleet_structure,
        replicate=args.fleet_replicate,
    )
    builds = apply_fleet_status(load_builds(BUILDS), BUILDS)
    replicates = fleet_replicates(BUILDS)
    for bid, rep in replicates.items():
        if bid in builds:
            builds[bid] = replace(builds[bid], replicate=rep)
    # Restrict to THE FLEET UNDER TEST, and to builds given the current brief.
    # fleet.json accumulates every entry the index has ever held -- builds across
    # several model pairs, founder structures and replicates -- so sweeping "the
    # fleet" would spend hours of crowd time on apps that are not part of this
    # experiment and would then mix them into the comparison.
    fleet = {
        bid
        for bid, b in builds.items()
        if bid in replicates and spec.wants(b) and _brief_is_current(bid)
    }
    # EVERY build in the fleet, including ones whose manifest is missing or
    # malformed. Those used to be skipped, so a model that failed to ship a
    # launch contract vanished from the denominator instead of being marked
    # down. They are simulated like anything else now, and the crowd finds
    # nothing to run and scores them at the floor.
    #
    # The ONE exception is a build a provider refused to produce, which is
    # excluded rather than floored -- see FleetBuild.UNSCORABLE_STATUSES. Running
    # the crowd over it would spend three slots manufacturing the floor score that
    # the exclusion exists to avoid, and the scoring layer would then discard the
    # result anyway.
    want_structures = {x.strip() for x in args.structures.split(",") if x.strip()}
    wanted = [
        b
        for bid, b in sorted(builds.items())
        if bid in fleet
        and not b.unscorable
        and (not args.ideas or b.idea_id in args.ideas.split(","))
        and (not want_structures or structure_name(b.config) in want_structures)
    ]
    excluded = sorted(bid for bid, b in builds.items() if bid in fleet and b.unscorable)
    if not args.no_control:
        wanted += [builds[b] for b in CONTROL_BUILDS if b in builds]

    have = existing_runs()
    # `skipped` is counted and reported, never silently dropped: an unscored
    # cell still has to show up somewhere, or the sweep looks complete when it
    # is merely finished.
    todo, skipped = plan_cells(
        [b.build_id for b in wanted],
        seeds=args.seeds,
        have=have,
        timed_out=timeouts_seen(),
        max_timeout_attempts=args.max_timeout_attempts,
        tried=failed_attempts(),
    )

    print(
        f"crowd sweep: arch v{CROWD_ARCH_VERSION}, fleet "
        f"{spec.model_a} vs {spec.model_b} "
        f"[{spec.structure} r{spec.replicate}]"
    )
    print(
        f"  {len(wanted)} builds x "
        f"{args.seeds} seeds, n={args.agents} triers={args.triers} "
        f"rounds={args.rounds}"
    )
    print(f"  already on disk: {sum(len(v) for v in have.values())} runs")
    print(f"  to run: {len(todo)}")
    print("  " + build_coverage_line([b.build_id for b in wanted], have, args.seeds))
    if excluded:
        print(
            f"  excluded (provider refusal, unscored not floored): "
            f"{len(excluded)} -> {', '.join(excluded)}"
        )
    if skipped:
        print(
            f"  skipped: {len(skipped)} cells already killed at the wall clock "
            f"{args.max_timeout_attempts}x under arch v{CROWD_ARCH_VERSION} "
            f"(--max-timeout-attempts 0 to retry them)"
        )
    if args.status:
        return 0
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        return 0

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_cell, c, args): c for c in todo}
        for future in as_completed(futures):
            cell = futures[future]
            try:
                res = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad cell must not stop all
                res = {
                    "build_id": cell.build_id,
                    "seed": cell.seed,
                    "ok": False,
                    "rc": -1,
                    "elapsed_s": 0,
                    "interviews": None,
                    "error": repr(exc),
                }
            done += 1
            failed += 0 if res["ok"] else 1
            print(
                f"[{done}/{len(todo)}] {'ok  ' if res['ok'] else 'FAIL'} "
                f"{res['build_id'][:46]:<46} s{res['seed']} "
                f"n_int={res.get('interviews')} {res.get('elapsed_s')}s",
                flush=True,
            )
            # Housekeeping, because nothing else in a fresh checkout does it.
            #
            # Every app restart clones the whole app tree, and only the CURRENT
            # clone is retired when a session closes -- a crash, a kill, or a
            # start that raised before the session was cached leaks one. They are
            # ~188,600 files apiece. Left unattended they pile up in the
            # thousands, and crowd throughput collapsed from ~100 runs/hour to
            # roughly 2.
            #
            # `sweep_run_dirs` existed for a long time with NO callers, which is
            # exactly why that happened. Both calls are cheap: retiring is a
            # rename, and reclaiming is explicitly time-budgeted so a long delete
            # never stalls the sweep. Every 25 cells keeps `builds/runs` flat
            # without making this the hot path.
            if done % 25 == 0:
                retired = sweep_run_dirs(older_than_hours=1.0)
                freed = reclaim_trash(budget_s=20.0)
                if retired or freed:
                    print(
                        f"    housekeeping: retired {retired} leaked clone(s), "
                        f"reclaimed {freed}",
                        flush=True,
                    )
    print(f"\n{done - failed}/{done} runs ok")

    if not args.no_autorate:
        autorate_missing(
            healthy_run_dirs({b.build_id for b in wanted}),
            concurrency=max(2, args.autorate_concurrency),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
