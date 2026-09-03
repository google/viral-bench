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

"""Compare founder STRUCTURES over the fleet: solo vs a four-specialist team.

`replicate_analysis.py` answers "is model A better than model B". This answers a
different question the benchmark is also supposed to settle: does putting four
specialists round a table in a shared working directory build more viral apps
than one agent working alone?

The comparison is PAIRED on the idea. Ideas differ enormously in how viral their
apps can be -- the corpus spans a 2048 clone and a collaborative database -- so
comparing arm means over different idea subsets would mostly measure which ideas
each arm happened to get. Every comparison here is over ideas all arms built.

Three things it reports and why each is needed:

* **Paired gaps with a bootstrap CI.** A gap without an interval cannot be told
  apart from noise, and the within-build seed noise on this instrument is real
  (sd ~3.5 points).
* **The validity gate separately.** A structure that ships more *working* apps is
  a different claim from one that ships more *liked* apps, and averaging a score
  over broken builds conflates them.
* **Build cost.** Four agents cost roughly four times the tokens and far more
  wall clock. A structure that wins by 2 points for 5x the spend is a finding,
  not a recommendation, and the table should make that visible rather than
  leaving it to be discovered later.

Usage::

    scripts/structure_analysis.py
    scripts/structure_analysis.py --profile default
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.score.fleet import load_corpus  # noqa: E402
from viral_bench.score.viralscore import ScoreWeights  # noqa: E402

BUILDS = REPO / "builds"

#: Founder shapes, keyed as in scripts/build_fleet.py STRUCTURES.
STRUCTURE_LABELS = {
    "solo": "solo (1 agent)",
    "team": "team (4, shared dir)",
}


def structure_of(config: dict) -> str | None:
    """Which named structure produced this build, from its recorded config."""
    agents = config.get("n_agents")
    collab = config.get("collab")
    if agents == 1:
        return "solo"
    if agents == 4 and collab == "local":
        return "team"
    return None


def _bootstrap_ci(diffs: list[float], *, n: int = 5000, seed: int = 0):
    """Percentile bootstrap over paired differences."""
    import random

    if len(diffs) < 3:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(statistics.mean(sample))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile", default="", help="score profile (default: the active one)"
    )
    args = ap.parse_args(argv)

    weights = ScoreWeights.from_profile(args.profile) if args.profile else None
    corpus = load_corpus(REPO, weights, current_brief_only=True)

    # score and validity per (structure, idea), averaged over seeds
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    valid: dict[str, list[bool]] = defaultdict(list)
    cost: dict[str, list[float]] = defaultdict(list)

    for run in corpus.fleet_runs(scorable_only=False):
        build = corpus.builds.get(run.build_id)
        if build is None or build.is_control:
            continue
        struct = structure_of(build.config)
        if struct is None:
            continue
        valid[struct].append(bool(run.scorable and run.score is not None))
        if run.score is not None:
            scores[struct][build.idea_id].append(run.score)

    for build in corpus.fleet_builds():
        if build.is_control:
            continue
        struct = structure_of(build.config)
        if struct and build.build_seconds:
            cost[struct].append(build.build_seconds)

    present = [s for s in ("solo", "team") if scores.get(s)]
    if not present:
        print("no scored runs for any structure yet")
        return 1

    print("=== PER-STRUCTURE SUMMARY")
    print(
        f"  {'structure':22} {'ideas':>6} {'mean':>7} {'sd':>6} "
        f"{'scorable':>9} {'build':>9}"
    )
    for s in present:
        per_idea = {i: statistics.mean(v) for i, v in scores[s].items()}
        vals = list(per_idea.values())
        v = valid[s]
        c = cost[s]
        print(
            f"  {STRUCTURE_LABELS[s]:22} {len(vals):>6} {statistics.mean(vals):>7.1f} "
            f"{(statistics.pstdev(vals) if len(vals) > 1 else 0):>6.1f} "
            f"{100 * sum(v) / max(len(v), 1):>8.0f}% "
            f"{(statistics.median(c) / 60 if c else 0):>8.1f}m"
        )

    # Two yardsticks, printed BEFORE the gaps so neither can be skipped. They
    # answer different questions and confusing them is easy.
    #
    # ABSOLUTE noise: how much a build's score moves when a DIFFERENT crowd
    # judges it. Each seed samples 30 personas from a pool of 150, so seeds get
    # different crowds and the level swings a lot: measured here, one arm's mean
    # moved by nearly 8 points across four seeds (sd 3.6). Any comparison of
    # absolute scores between differently-seeded runs must clear this.
    #
    # PAIRED noise: how much a GAP moves. Far smaller, because the same crowd
    # judges every arm within a seed, so a generous crowd lifts all of them and
    # the difference is unaffected. Measured: gap sd ~2.0-2.9 against an absolute
    # sd of 3.6. This is the yardstick for everything below, and it is the reason
    # every arm must be run on the IDENTICAL seed set -- the cancellation is what
    # buys the sensitivity.
    from viral_bench.score.fleet import self_separation

    by_seed_arm: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    for run in corpus.fleet_runs():
        if run.score is not None:
            by_seed_arm[(run.seed, run.structure)][run.idea_id] = run.score
    seeds = sorted({k[0] for k in by_seed_arm})

    print("\n=== NOISE YARDSTICKS")
    levels = [
        statistics.mean(by_seed_arm[(sd, present[0])].values())
        for sd in seeds
        if by_seed_arm.get((sd, present[0]))
    ]
    if len(levels) > 1:
        print(
            f"  absolute level across {len(levels)} seeds: "
            f"sd {statistics.stdev(levels):.2f} "
            f"(range {max(levels) - min(levels):.1f}) -- a DIFFERENT crowd judging"
        )
    models = {b.model.rsplit("/", 1)[-1] for b in corpus.fleet_builds()}
    for m in sorted(models):
        null = self_separation(corpus, m)
        if null.gap is None:
            continue
        ci = (
            "CI [n/a]"
            if null.ci_low is None
            else f"CI [{null.ci_low:+.1f}, {null.ci_high:+.1f}]"
        )
        print(
            f"  same model, seeds split in half: {null.gap:+.1f}  {ci}  "
            f"n={null.n_ideas}"
        )
        print("     ^ this is ABSOLUTE noise, not the floor for a paired gap below")

    print("\n=== PAIRED GAPS (only ideas BOTH arms built)")
    print("  a gap whose interval spans 0 is not distinguishable from noise,")
    print("  seed-sd is how much the GAP itself moves between seeds")
    for i, a in enumerate(present):
        for b in present[i + 1 :]:
            shared = sorted(set(scores[a]) & set(scores[b]))
            if not shared:
                print(f"  {a} vs {b}: no shared ideas")
                continue
            diffs = [
                statistics.mean(scores[a][idea]) - statistics.mean(scores[b][idea])
                for idea in shared
            ]
            gap = statistics.mean(diffs)
            lo, hi = _bootstrap_ci(diffs)
            wins = sum(1 for d in diffs if d > 0)
            ci = (
                "CI [n/a: too few pairs]"
                if lo is None
                else f"CI [{lo:+.1f}, {hi:+.1f}]"
            )
            sig = ""
            if lo is not None:
                sig = (
                    "  SIGNIFICANT" if (lo > 0 or hi < 0) else "  (not distinguishable)"
                )
            # How much this gap itself moves between seeds -- the honest
            # floor for a paired comparison.
            per_seed = []
            for sd in seeds:
                A = by_seed_arm.get((sd, a), {})
                B = by_seed_arm.get((sd, b), {})
                common = set(A) & set(B)
                if common:
                    per_seed.append(statistics.mean([A[i] - B[i] for i in common]))
            seed_sd = (
                f"  seed-sd {statistics.stdev(per_seed):.1f}"
                if len(per_seed) > 1
                else ""
            )
            print(
                f"  {a:5} - {b:5}: {gap:+6.1f}  {ci}  "
                f"{a} wins {wins}/{len(shared)}{seed_sd}{sig}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
