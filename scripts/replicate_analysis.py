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

"""Is a measured model gap real, or did one model get luckier builds?

A founder model is sampled at temperature, so building the same brief twice
produces two different apps. Any comparison built on **one** app per brief is
therefore reporting a mixture of "this model is better" and "this build came out
well", with no way to tell how much of each. The only way to separate them is to
build the fleet more than once and compare.

Given a corpus with two or more replicates, this reports:

1. **Variance decomposition** -- crowd noise (same app, different crowds) against
   build noise (different apps, same brief and model). If build noise dominates,
   per-brief results are not interpretable no matter how many crowd seeds you run.
2. **The model gap per replicate, and pooled** -- the replication check.
3. **Winner flips** -- briefs that changed sides between replicates.
4. **Build delivery** -- how often each model shipped a launchable app, with a
   Wilson interval (the normal approximation reports a zero-width interval at
   0 failures, which is nonsense).

Everything is read from artifacts already on disk. No LLM calls, no re-running.
The statistics themselves live in :mod:`viral_bench.score.fleet`, where they are
tested. This is a thin renderer over them.

    scripts/replicate_analysis.py
    scripts/replicate_analysis.py --model-a claude-x --model-b gemini-y
    scripts/replicate_analysis.py --profile v2_hybrid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.score.fleet import (  # noqa: E402
    delivery_stats,
    load_corpus,
    paired_gap,
    variance_components,
    winner_flips,
)
from viral_bench.score.viralscore import ScoreWeights  # noqa: E402

DEFAULT_A = "gemini-2.0-flash"
DEFAULT_B = "gemini-2.0-flash"


def _ci(low: float | None, high: float | None) -> str:
    """Render a confidence interval, or say plainly that there isn't one.

    ``PairedGap.ci_low``/``ci_high`` are declared ``float | None`` and are None
    whenever there were too few paired ideas to bootstrap -- which happens
    routinely on a partial fleet, and happened here the moment brief
    fingerprinting retired the pre-pivot builds. The printer formatted them
    unconditionally and died with "unsupported format string passed to
    NoneType.__format__", taking down the whole analysis over a missing interval
    on one line.
    """
    if low is None or high is None:
        return "CI [n/a: too few paired ideas]"
    return f"CI [{low:+.2f}, {high:+.2f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", default=DEFAULT_A)
    parser.add_argument("--model-b", default=DEFAULT_B)
    parser.add_argument("--profile", default=None, help="scoring profile to use")
    parser.add_argument("--repo", default=str(REPO))
    parser.add_argument(
        "--all-eras",
        action="store_true",
        help="include builds made under an older brief. Off by default: "
        "fleet.json accumulates every entry it has ever held, and pooling eras "
        "compares two models through two different sets of instructions. Turn "
        "it on to analyse a historical fleet on its own terms.",
    )
    args = parser.parse_args(argv)

    weights = ScoreWeights.from_profile(args.profile) if args.profile else None
    corpus = load_corpus(args.repo, weights, current_brief_only=not args.all_eras)
    a, b = args.model_a, args.model_b
    reps = corpus.replicates()

    print(f"profile: {corpus.profile}   crowd arch: v{corpus.arch_version}")
    print(f"A = {a}")
    print(f"B = {b}")
    print(f"replicates present: {reps or 'none'}")
    if len(reps) < 2:
        print(
            "\nNeed at least two replicates to say anything about build variance.\n"
            "Build another with:  scripts/build_fleet.py --replicate 2"
        )
        return 1

    # A typo in a model name is the likeliest way to run this wrong, and it would
    # otherwise surface as a formatting crash on an empty statistic.
    present = sorted({b.model for b in corpus.fleet_builds()})
    missing = [m for m in (a, b) if m not in present]
    if missing:
        print(f"\nno builds in this corpus for: {', '.join(missing)}")
        print(f"models present: {', '.join(present) or 'none'}")
        return 1
    if paired_gap(corpus, a, b).n_ideas == 0:
        print(f"\nno brief was built by BOTH {a} and {b}; nothing to compare")
        return 1

    # ------------------------------------------------------------- variance
    v = variance_components(corpus)

    # Every one of these is Optional and is None on a thin corpus, which is the
    # normal state of a fleet mid-sweep. Formatting them unconditionally is the
    # same crash as the confidence interval above, and _num keeps one missing
    # statistic from taking down the whole analysis.
    def _num(value: float | None, width: int = 5) -> str:
        return "  n/a" if value is None else f"{value:{width}.2f}"

    print("\n=== VARIANCE DECOMPOSITION (ViralScore points)")
    print(f"  crowd noise  sigma_crowd = {_num(v.crowd_sd)}   same app, other crowds")
    print(f"  build noise  sigma_build = {_num(v.build_sd)}   other app, same brief")
    print(f"  ratio build/crowd        = {_num(v.ratio)}")
    print(
        f"  spread within a cell: median {v.median_abs_diff}, max {v.max_abs_diff}"
        f" (over {v.n_cells} cells built more than once)"
    )
    if v.ratio and v.ratio > 1:
        print(
            "  -> which app the model produced matters MORE than which crowd\n"
            "     judged it; a single build of a single brief is an anecdote."
        )

    # ----------------------------------------------------------------- gaps
    print("\n=== MODEL GAP, BY REPLICATE")
    for rep in reps:
        g = paired_gap(corpus.restrict(replicate=rep), a, b)
        print(
            f"  replicate {rep}: gap {g.gap:+6.2f}  "
            f"{_ci(g.ci_low, g.ci_high)}  "
            f"A wins {g.wins_a}/{g.n_ideas}"
        )
    pooled = paired_gap(corpus, a, b)
    print(
        f"  POOLED:      gap {pooled.gap:+6.2f}  "
        f"{_ci(pooled.ci_low, pooled.ci_high)}  "
        f"A wins {pooled.wins_a}/{pooled.n_ideas}"
    )

    # --------------------------------------------------------------- flips
    flips = winner_flips(corpus, a, b)
    print(f"\n=== WINNER FLIPS: {len(flips)}/{pooled.n_ideas} briefs changed sides")
    for flip in flips:
        detail = "  ".join(
            f"r{r} {d:+7.1f}" for r, d in sorted(flip.per_replicate.items())
        )
        print(f"  {flip.idea_id:<30}{detail}   swing {flip.swing:.1f}")

    first = paired_gap(corpus.restrict(replicate=reps[0]), a, b).per_idea
    losses = sorted(idea for idea, delta in first.items() if delta < 0)
    if losses:
        later = {
            r: paired_gap(corpus.restrict(replicate=r), a, b).per_idea for r in reps[1:]
        }
        print(f"\n  briefs B won in replicate {reps[0]} -- did they hold up?")
        for idea in losses:
            others = [later[r].get(idea) for r in reps[1:]]
            held = all(o is not None and o < 0 for o in others)
            shown = "  ".join(f"{o:+.1f}" if o is not None else "n/a" for o in others)
            print(
                f"    {idea:<30} r{reps[0]} {first[idea]:+7.1f}  later {shown:<12} "
                f"{'replicates' if held else 'does NOT replicate'}"
            )

    # ------------------------------------------------------------ delivery
    print("\n=== BUILD DELIVERY (manifest failures: the model's own contract)")
    for model in (a, b):
        d = delivery_stats(corpus, model)
        per = "  ".join(
            f"r{r}: {k}/{n}" for r, (k, n) in sorted(d.per_replicate.items())
        )
        lo, hi = d.failure_ci
        print(
            f"  {model:<20} {d.manifest_failures}/{d.attempted} = "
            f"{d.failure_rate:.0%}  95% CI [{lo:.0%}, {hi:.0%}]   {per}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
