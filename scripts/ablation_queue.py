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

"""Run a queue of crowd ablations back to back, one variable at a time.

Ablations are the only honest way to attribute a change in the instrument to a
change in the architecture, and they are slow -- each arm is a fresh crowd run
over a matched subset of the frozen fleet, about an hour. Running them one at a
time by hand wastes most of a day in gaps, and running them concurrently makes
every arm contend for the same API quota, which is the one resource that decides
throughput here. So they queue.

Each entry names ONE variable and its value; everything else is the shipped
default. The variant name becomes the run's architecture tag
(``<version>+<variant>``), which is what keeps an arm out of the main corpus.

Usage::

    scripts/ablation_queue.py --list
    scripts/ablation_queue.py --only rounds2,rounds5
    scripts/ablation_queue.py                # everything not already on disk
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.crowd.sim_defaults import CROWD_ARCH_VERSION  # noqa: E402

#: variant name -> extra argv for scripts/crowd_ablation.py. One knob each.
QUEUE: dict[str, list[str]] = {
    # -- who is in the crowd, and who touches the app ------------------------
    "triers8": ["--triers", "8"],
    "triers15": ["--triers", "15"],
    "agents15": ["--agents", "15"],
    "agents50": ["--agents", "50"],
    "latecomers8": ["--latecomers", "8"],
    # -- how long the crowd runs ---------------------------------------------
    "rounds1": ["--rounds", "1"],
    "rounds2": ["--rounds", "2"],
    "rounds5": ["--rounds", "5"],
    # -- how hard the crowd has to try before judging ------------------------
    "depth3": ["--min-interactions", "3"],
    "depth5": ["--min-interactions", "5"],
    # -- how much rope a trial gets -------------------------------------------
    "steps20": ["--trial-max-steps", "20"],
    "steps80": ["--trial-max-steps", "80"],
    # -- crowd size, finer grained --------------------------------------------
    "agents20": ["--agents", "20"],
    "agents40": ["--agents", "40"],
    # -- the crowd's own model -------------------------------------------------
    # The crowd's own model, compared at a crowd shape 3.7-flash can sustain.
    # At 30 agents it loses 11-18% of turns to rate limits even alone at
    # semaphore 4, and a throttled turn is recorded as an agent choosing to do
    # nothing -- so the only honest comparison holds the shape fixed at 15 and
    # varies the model against the existing agents15 arm.
    "gemini-2.0-flash": ["--model", "gemini-2.0-flash", "--semaphore", "8"],
    "gemini-2.0-flashn15": [
        "--model",
        "gemini-2.0-flash",
        "--agents",
        "15",
        "--semaphore",
        "6",
    ],
    "flash35n15": [
        "--model",
        "gemini-2.0-flash",
        "--agents",
        "15",
        "--semaphore",
        "6",
    ],
    # -- what the crowd can do -----------------------------------------------
    "noreload": ["--disable-tools", "reload_page"],
    "noshots": ["--disable-tools", "screenshot"],
    "noselect": ["--disable-tools", "select_option"],
    "nolook": ["--disable-tools", "look"],
    "noenvnotice": ["--no-env-notice"],
    # -- the feed and the graph ----------------------------------------------
    "feed40": ["--feed-max-posts", "40"],
    "feed10": ["--feed-max-posts", "10"],
    "nopeers": ["--follow-peers", "0"],
    "peers6": ["--follow-peers", "6"],
    "twitterrec": ["--recsys", "twitter"],
    # -- how the crowd is sampled --------------------------------------------
    "temp0": ["--temperature", "0.0"],
    "temp1": ["--temperature", "1.0"],
}


def done_variants() -> set[str]:
    """Variants that already have healthy runs on disk."""
    out: set[str] = set()
    for path in (REPO / "builds" / "ablation").glob("*/run_summary.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        arch = str(data.get("crowd_arch_version", ""))
        if data.get("ok") and arch.startswith(f"{CROWD_ARCH_VERSION}+"):
            out.add(arch.split("+", 1)[1])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="comma-separated variant names")
    parser.add_argument("--builds", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--redo", action="store_true", help="rerun finished arms")
    args = parser.parse_args(argv)

    wanted = [v.strip() for v in args.only.split(",") if v.strip()] or list(QUEUE)
    unknown = [v for v in wanted if v not in QUEUE]
    if unknown:
        print(f"unknown variant(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    if args.list:
        for name in wanted:
            print(f"{name:<14} {' '.join(QUEUE[name])}")
        return 0

    already = set() if args.redo else done_variants()
    todo = [v for v in wanted if v not in already]
    print(f"queue: {len(todo)} arms to run ({len(already)} already on disk)")
    for i, name in enumerate(todo, 1):
        started = time.time()
        print(f"\n=== [{i}/{len(todo)}] {name}: {' '.join(QUEUE[name])}", flush=True)
        proc = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "-u",
                str(REPO / "scripts" / "crowd_ablation.py"),
                "--variant",
                name,
                "--builds",
                str(args.builds),
                "--seeds",
                str(args.seeds),
                "--concurrency",
                str(args.concurrency),
                *QUEUE[name],
            ],
            cwd=REPO,
        )
        print(
            f"=== {name} finished rc={proc.returncode} "
            f"in {(time.time() - started) / 60:.0f} min",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
