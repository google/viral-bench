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

"""Measure how far apart the ViralScore puts two founder models.

This is the number the benchmark exists to produce: not "is model A good" but
"is model A measurably different from model B, by more than the noise". Cohen's
d expresses the gap in units of run-to-run spread, so 0.8+ means a difference a
leaderboard can legitimately report.

    # compare two founder models over every run of every idea they built
    uv run python scripts/discriminate.py \\
        --model "anthropic/claude-sonnet-4-5=quick_notes_app__*,markdown_slides__*" \\
        --model "gemini-2.0-flash=quick_notes_app__*,markdown_slides__*"

    # which scoring configuration separates them most, over the whole test set?
    uv run python scripts/discriminate.py --model ... --model ... --sweep-profiles

Everything here re-scores runs already on disk, so exploring scoring
configurations is free. Never re-run a crowd simulation to try a weighting.

The autorater's dimensions are included whenever a run has an ``autorating.json``
next to it (written by ``viral-bench score --autorate``). The output states how
many runs were rated: with none, profiles that weight the autorater silently
collapse to their deterministic part, and a change to the rater cannot move any
number on this page.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viral_bench.score import (  # noqa: E402
    ScoreWeights,
    component_separation,
    redundancy_panel,
    score_runs_by_model,
    separation,
    sweep_profiles,
)
from viral_bench.score.autorater import AutoRating  # noqa: E402

SEARCH_DIRS = ("builds/calibration", "builds/crowd")


def _runs_for(spec: str) -> list[str]:
    """Every crowd run directory matching one comma-separated build spec.

    Each element may be a literal build id or a glob (``quick_notes_app__*``),
    so one model's whole slice of the test set fits in a single ``--model``.
    """
    out: list[str] = []
    for pattern in (p.strip() for p in spec.split(",") if p.strip()):
        for root in SEARCH_DIRS:
            out += glob.glob(f"{root}/{pattern}__*")
    seen = sorted(set(out))
    return [d for d in seen if Path(d, "run_summary.json").is_file()]


def _load_ratings(runs: dict[str, list[str]]) -> tuple[dict[str, AutoRating], int]:
    """Read every persisted ``autorating.json`` and report how many were found.

    Rating is an expensive LLM call already paid for at score time, so it is
    read back rather than repeated. The count is returned because scoring an
    unrated run is not an error -- its autorater weight is renormalised away --
    which makes a corpus with no ratings look healthy while quietly measuring
    something else.
    """
    ratings: dict[str, AutoRating] = {}
    total = 0
    for dirs in runs.values():
        for d in dirs:
            total += 1
            path = Path(d, "autorating.json")
            if not path.is_file():
                continue
            try:
                rating = AutoRating.from_dict(json.loads(path.read_text("utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
            if rating.ok:
                ratings[str(d)] = rating
    return ratings, total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        action="append",
        metavar="LABEL=BUILD_IDS",
        help=(
            "a founder model and its builds: one build id, several separated by "
            "commas, or globs (repeat the flag for each model)"
        ),
    )
    p.add_argument(
        "--runs",
        metavar="BUILD_IDS",
        help=(
            "a model-agnostic pool of builds (comma-separated ids or globs). "
            "--redundancy asks about the score's internal structure rather than "
            "about any model, so it takes this instead of --model"
        ),
    )
    p.add_argument("--profile", default=None, help="weight profile to score with")
    p.add_argument(
        "--sweep-profiles",
        action="store_true",
        help="re-score under every profile in config/score.yaml, ranked by separation",
    )
    p.add_argument(
        "--redundancy",
        action="store_true",
        help=(
            "measure whether the components are N measurements or one measured "
            "N times: keep-one / drop-one Spearman plus the correlation matrix"
        ),
    )
    p.add_argument(
        "--redundancy-include-gated",
        action="store_true",
        help=(
            "keep validity-gated runs in the redundancy panel. Off by default: "
            "the gate multiplies every component at once, so including gated "
            "runs manufactures agreement between otherwise unrelated terms"
        ),
    )
    p.add_argument("--json", action="store_true", help="emit JSON")
    args = p.parse_args(argv)

    if not args.model and not args.runs:
        p.error("need --model (separation) or --runs (--redundancy on its own)")

    runs: dict[str, list[str]] = {}
    for spec in args.model or ():
        label, _, build_spec = spec.partition("=")
        found = _runs_for(build_spec)
        if not found:
            print(f"WARNING: no crowd runs found for {build_spec!r}", file=sys.stderr)
        # Accumulate: one model's test-set slice is usually many --model flags
        # or many comma-separated builds, and overwriting would silently drop
        # every idea but the last.
        existing = runs.setdefault(label, [])
        existing += [d for d in found if d not in existing]

    pool: list[str] = []
    if args.runs:
        pool = _runs_for(args.runs)
        if not pool:
            print(f"WARNING: no crowd runs found for {args.runs!r}", file=sys.stderr)

    ratings, n_runs = _load_ratings(runs)
    if pool:
        pool_ratings, _ = _load_ratings({"pool": pool})
        ratings.update(pool_ratings)
        # Denominator is the union: --runs and --model may overlap, and the
        # point of the line is "what fraction of what I scored was rated".
        n_runs = len({d for ds in runs.values() for d in ds} | set(pool))
    weights = ScoreWeights.from_profile(args.profile)
    scored = score_runs_by_model(runs, weights, ratings)
    # Separation needs two models. The redundancy panel is about the score's
    # own structure and needs none, so a --runs-only invocation is valid.
    if runs and len(scored) < 2:
        print(
            "ERROR: need scorable runs for at least two models. Unscorable runs "
            "(e.g. missing interviews) are skipped, not counted as zero.",
            file=sys.stderr,
        )
        return 2

    report: dict = {
        "profile": weights.profile,
        "n_runs": n_runs,
        "n_rated": len(ratings),
        "models": {},
    }
    print(f"profile: {weights.profile}")
    print(
        f"autorater: {len(ratings)} of {n_runs} runs rated"
        + (
            ""
            if ratings
            else "  ! no autorating.json found -- autorater weights are being "
            "renormalised away, so rater changes cannot move these numbers"
        )
    )
    for model, s in scored.items():
        # Report the denominator: a run that failed to score is dropped, not
        # zeroed, so the count it was dropped from has to be visible.
        found = len(runs.get(model, []))
        print(
            f"  {model:<34} mean={s.mean:<8} sd={s.sd:<8} "
            f"scored={len(s.all_scores)} of {found} runs ideas={len(s.by_idea)}"
        )
        report["models"][model] = {
            "mean": s.mean,
            "sd": s.sd,
            "n_runs": len(s.all_scores),
            "n_found": found,
        }

    if scored:
        names = sorted(scored)
        print("\n=== SEPARATION (Cohen's d: 0.2 small / 0.5 medium / 0.8 large) ===")
        report["separations"] = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                sep = separation(scored[a], scored[b])
                print(f"  {sep.render()}")
                report["separations"].append(sep.__dict__)

        print("\n=== WHICH COMPONENTS SEPARATE THE MODELS? ===")
        print(f"  {'component':<16}{'max |d|':<10}per-model mean")
        comps = component_separation(runs, ratings)
        for name, st in sorted(
            comps.items(), key=lambda kv: -(kv[1]["max_abs_cohens_d"] or 0)
        ):
            d = str(st["max_abs_cohens_d"])
            print(f"  {name:<16}{d:<10}{st['per_model_mean']}")
        report["components"] = comps

    if args.redundancy:
        # A component that separates models (the table above) can still be
        # buying nothing, if another component separates them the same way. The
        # two questions are independent and both have to be asked.
        dirs = pool or sorted({d for ds in runs.values() for d in ds})
        print("\n=== ARE THE COMPONENTS INDEPENDENT? ===")
        panel = redundancy_panel(
            dirs,
            weights,
            ratings,
            clean_only=not args.redundancy_include_gated,
        )
        print(panel.render())
        report["redundancy"] = asdict(panel)

    if args.sweep_profiles:
        print("\n=== SCORING CONFIGURATIONS, RANKED BY SEPARATION ===")
        print(f"  {'profile':<22}{'max |d|':<10}means")
        rows = sweep_profiles(runs, ratings=ratings)
        for row in rows:
            if row.get("error"):
                print(f"  {row['profile']:<22}{'--':<10}! {row['error']}")
                continue
            print(
                f"  {row['profile']:<22}{str(row['max_abs_cohens_d']):<10}"
                f"{row.get('means')}"
            )
        report["profile_sweep"] = rows

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
