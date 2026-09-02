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

"""Run the ViralScore calibration sweep (E1 reliability + E2 discrimination).

Scores the same apps repeatedly at several crowd sizes and reports, per size,
how much of the score's variance is real differences between apps rather than
run-to-run noise. That answers the two open questions: how many crowd agents a
scored run needs, and whether the score can rank apps at all.

Usage:
    uv run python scripts/calibrate_score.py --sizes 15 30 50 --seeds 0 1 2
    uv run python scripts/calibrate_score.py --analyze-only   # re-score, no runs

Because scoring is a pure function of stored artifacts, ``--analyze-only``
re-derives every number from runs already on disk, for free.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viral_bench.crowd.launch import run_crowd_sim  # noqa: E402
from viral_bench.score.calibrate import (  # noqa: E402
    compare_weightings,
    component_reliability,
    group_runs_by_agents,
    reliability,
)
from viral_bench.score.viralscore import ScoreWeights  # noqa: E402

#: The calibration set: the same idea built three ways, so differences in score
#: are differences in the BUILD, not the idea. Spans the range the benchmark has
#: to separate -- a strong team build, a weak solo build, and a broken app.
DEFAULT_APPS = {
    "strong (4-agent 3.6-flash)": "quick_notes_app__20260723-214453__606bb1",
    "weak (solo 3.5-flash-lite)": "quick_notes_app__20260728-220614__efac3a",
    "broken (control)": "quick_notes_app__20260728-000000__brokn0",
}

CALIBRATION_DIR = Path("builds/calibration")


def _run_one(build_id: str, agents: int, seed: int, recsys: str) -> str | None:
    out = (
        CALIBRATION_DIR
        / f"{build_id}__n{agents}__seed{seed}__{time.strftime('%H%M%S')}"
    )
    try:
        result = run_crowd_sim(
            build_id,
            agents=agents,
            triers=max(3, round(agents * 0.24)),  # ~24% hands-on, scales with crowd
            seed=seed,
            recsys=recsys,
            out_dir=out,
            stream=False,
        )
    except Exception as exc:  # noqa: BLE001 - one failed run must not stop the sweep
        print(f"    ! run failed ({type(exc).__name__}: {exc})")
        return None
    if not result.ok:
        print(f"    ! run not ok: {result.error}")
    return str(out)


def sweep(
    apps: dict[str, str], sizes: list[int], seeds: list[int], recsys: str
) -> dict:
    runs: dict[str, list[str]] = {label: [] for label in apps}
    total = len(apps) * len(sizes) * len(seeds)
    done = 0
    for size in sizes:
        for seed in seeds:
            for label, build_id in apps.items():
                done += 1
                print(f"[{done}/{total}] {label} n={size} seed={seed} ...", flush=True)
                started = time.time()
                path = _run_one(build_id, size, seed, recsys)
                if path:
                    runs[label].append(path)
                print(f"    done in {time.time() - started:.0f}s", flush=True)
    return runs


def discover_runs(apps: dict[str, str]) -> dict[str, list[str]]:
    """Find calibration runs already on disk for these builds."""
    runs: dict[str, list[str]] = {}
    if not CALIBRATION_DIR.is_dir():
        return runs
    for label, build_id in apps.items():
        found = [
            str(d)
            for d in sorted(CALIBRATION_DIR.glob(f"{build_id}__n*"))
            if (d / "run_summary.json").is_file()
        ]
        if found:
            runs[label] = found
    return runs


def analyze(runs: dict[str, list[str]]) -> None:
    grouped = group_runs_by_agents(runs)
    print("\n" + "=" * 72)
    print("ViralScore reliability (rho >= 0.8 is the bar for ranking models)")
    print("=" * 72)
    summary = {}
    for size in sorted(grouped):
        rel = reliability(grouped[size], n_agents=size)
        print(rel.render())
        summary[size] = {
            "within_sd": rel.within_sd,
            "between_sd": rel.between_sd,
            "rho": rel.rho,
            "apps": {k: {"mean": v.mean, "sd": v.sd} for k, v in rel.apps.items()},
        }
    comps = component_reliability(runs)
    print("\n" + "=" * 72)
    print("per-component signal vs noise (does each component earn its weight?)")
    print("=" * 72)
    print(f"  {'component':<16}{'within SD':<12}{'between SD':<12}{'signal/noise'}")
    for name, stats in sorted(
        comps.items(), key=lambda kv: -(kv[1]["signal_to_noise"] or 0)
    ):
        print(
            f"  {name:<16}{stats['within_sd']!s:<12}"
            f"{stats['between_sd']!s:<12}{stats['signal_to_noise']}"
        )
    summary["components"] = comps

    # Re-score the SAME stored runs under alternative weightings. Free, because
    # scoring never re-simulates. Keeps the 65/35 judgement/behaviour split and
    # only moves weight between judgement components.
    candidates = {
        "v1.1 (shipped)": ScoreWeights(),
        "craft-heavy": ScoreWeights(
            adoption=0.20, advocacy=0.15, craft=0.30, amplification=0.35
        ),
        "craft-lifted": ScoreWeights(
            adoption=0.20, advocacy=0.20, craft=0.25, amplification=0.35
        ),
        "judgement-only": ScoreWeights(
            adoption=0.30, advocacy=0.30, craft=0.40, amplification=0.0
        ),
    }
    print("\n" + "=" * 72)
    print("alternative weightings, re-scored offline (rho >= 0.8 is the bar)")
    print("=" * 72)
    for name, stats in compare_weightings(runs, candidates).items():
        print(
            f"  {name:<18} rho={stats['rho']!s:<8} within={stats['within_sd']!s:<8}"
            f" between={stats['between_sd']}"
        )
        summary.setdefault("weightings", {})[name] = stats

    out = CALIBRATION_DIR / "reliability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=int, nargs="+", default=[15, 30, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--recsys", default="twitter")
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="re-score runs already on disk (free)",
    )
    p.add_argument(
        "--app",
        action="append",
        default=None,
        metavar="LABEL=BUILD_ID",
        help="override the calibration set (repeatable)",
    )
    args = p.parse_args(argv)

    apps = dict(DEFAULT_APPS)
    if args.app:
        apps = {}
        for item in args.app:
            label, _, build_id = item.partition("=")
            apps[label] = build_id

    for label, build_id in apps.items():
        print(f"calibration app: {label:<30} {build_id}")

    runs = (
        discover_runs(apps)
        if args.analyze_only
        else sweep(apps, args.sizes, args.seeds, args.recsys)
    )
    if args.analyze_only:
        # Include any runs we discovered even if the sweep never ran.
        for label in apps:
            runs.setdefault(label, [])
    if not any(runs.values()):
        print("no scorable calibration runs found", file=sys.stderr)
        return 1
    analyze(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
