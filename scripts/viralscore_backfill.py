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

"""Compute the ViralScore for every stored crowd run. No model spend.

The comparison needs both instruments on the same builds, and only 4 of 5,130
crowd runs had a ``score.json`` -- the score has been recomputed ad hoc at
analysis time and never persisted. ``score_run()`` is deterministic given the
stored signals plus a stored autorating, and 2,859 ``autorating.json`` files are
already on disk, so this is pure arithmetic over artifacts that already exist.

A run with no stored autorating is still scored: ``score_run`` re-normalises the
autorater weights away and records that it did, so an unrated run is not
penalised for being unrated. Which runs were rated is reported, because two runs
scored under the "same" profile can otherwise be using different formulas
without saying so.

Usage::

    scripts/viralscore_backfill.py --status
    scripts/viralscore_backfill.py --generation r3
    scripts/viralscore_backfill.py --all --concurrency 16
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BUILDS = REPO / "builds"
CROWD = BUILDS / "crowd"
SCORE_FILENAME = "score.json"
AUTORATING_FILENAME = "autorating.json"


def r3_build_ids(generation: str) -> set[str]:
    entries = json.loads((BUILDS / "fleet.json").read_text())["entries"]
    return {
        entry["build_id"]
        for key, entry in entries.items()
        if key.endswith(f"::{generation}")
    }


def run_dirs(build_ids: set[str] | None) -> list[Path]:
    if not CROWD.is_dir():
        return []
    found = []
    for entry in sorted(CROWD.iterdir()):
        if not entry.is_dir() or "__crowd-" not in entry.name:
            continue
        if build_ids is not None:
            build_id = entry.name.split("__crowd-")[0]
            if build_id not in build_ids:
                continue
        found.append(entry)
    return found


def score_one(run_dir: Path) -> dict:
    """Score one crowd run and persist it. Never raises."""
    sys.path.insert(0, str(REPO / "src"))
    from viral_bench.score.report import write_score
    from viral_bench.score.signals import extract_signals
    from viral_bench.score.viralscore import score_run

    outcome = {"dir": run_dir.name, "ok": False, "rated": False, "score": None}
    try:
        signals = extract_signals(run_dir)
    except Exception as exc:  # noqa: BLE001 - an unreadable run is data
        outcome["error"] = f"signals: {type(exc).__name__}: {exc}"
        return outcome

    autorating = None
    rating_path = run_dir / AUTORATING_FILENAME
    if rating_path.is_file():
        try:
            from viral_bench.score.autorater import AutoRating

            # from_dict, not AutoRating(**payload): the constructor would leave
            # `dimensions` as plain dicts and score_run would then fail on
            # `.score`. from_dict also degrades a truncated file to ok=False, so
            # the run is scored deterministically and counted as unrated rather
            # than taking the backfill down.
            autorating = AutoRating.from_dict(json.loads(rating_path.read_text()))
        except Exception:  # noqa: BLE001 - an unrated run still scores
            autorating = None
    outcome["rated"] = autorating is not None and getattr(autorating, "ok", False)

    try:
        result = score_run(signals, autorating=autorating)
        write_score(result, run_dir)
    except Exception as exc:  # noqa: BLE001
        outcome["error"] = f"score: {type(exc).__name__}: {exc}"
        return outcome

    outcome["ok"] = True
    outcome["score"] = result.score
    outcome["scorable"] = result.scorable
    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", default="r3")
    parser.add_argument("--all", action="store_true", help="every generation")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="rewrite existing")
    parser.add_argument("--status", action="store_true")
    opts = parser.parse_args(argv)

    wanted = None if opts.all else r3_build_ids(opts.generation)
    dirs = run_dirs(wanted)
    have = [d for d in dirs if (d / SCORE_FILENAME).is_file()]
    rated = [d for d in dirs if (d / AUTORATING_FILENAME).is_file()]

    if opts.status:
        print(f"crowd runs in scope: {len(dirs)}")
        print(f"  with score.json:   {len(have)}")
        print(f"  with autorating:   {len(rated)}")
        return 0

    todo = dirs if opts.force else [d for d in dirs if d not in set(have)]
    print(f"{len(dirs)} runs in scope, {len(todo)} to score")
    if not todo:
        return 0

    tally: Counter = Counter()
    scores: list[float] = []
    errors: list[dict] = []
    with ProcessPoolExecutor(max_workers=opts.concurrency) as pool:
        futures = {pool.submit(score_one, d): d for d in todo}
        for n, future in enumerate(as_completed(futures), 1):
            outcome = future.result()
            tally["ok" if outcome["ok"] else "failed"] += 1
            tally["rated"] += bool(outcome["rated"])
            if outcome.get("score") is not None:
                scores.append(outcome["score"])
            elif outcome["ok"]:
                tally["unscorable"] += 1
            else:
                errors.append(outcome)
            if n % 250 == 0:
                print(f"  {n}/{len(todo)}", flush=True)

    print(f"\nscored {tally['ok']}, failed {tally['failed']}")
    print(f"  with a stored autorating: {tally['rated']}")
    print(f"  unscorable (gate/blockers): {tally['unscorable']}")
    if scores:
        scores.sort()
        mid = scores[len(scores) // 2]
        print(
            f"  score: n={len(scores)} min={scores[0]:.1f} "
            f"median={mid:.1f} max={scores[-1]:.1f}"
        )
    for outcome in errors[:10]:
        print(f"  ! {outcome['dir']}: {outcome.get('error')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
