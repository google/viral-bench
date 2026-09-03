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

"""The deliverable: one row per founder model, over the frozen fleet.

Every model in the sweep built the same 25 briefs under one identical recorded
founder configuration (one agent, ``found <idea> --model M --agents 1``), and was
then judged by the same crowd instrument at the same three seeds. So the columns
below differ only by the model that wrote the code.

What each column is, and why it is here:

* **built** -- how many of the 25 briefs produced a healthy build at all.
  Delivery is a capability, and a model that cannot ship a launch contract
  should be visible before any score is discussed.
* **ViralScore** -- the headline, mean over that model's runs, with the spread
  across its 25 briefs. The score is defined in ``config/score.yaml``
  (``v4_earned``). Re-scoring is free and every model is re-scored together.
* **vs Claude Opus 5** -- the paired per-brief difference against the strongest
  model, which cancels brief difficulty exactly. This is the only fair way to
  compare two models over a corpus where briefs differ wildly in difficulty.
* **noise** -- pooled seed-to-seed SD inside that model's cells. It is the
  yardstick every difference has to clear, and it is reported next to the
  difference rather than in a footnote.
* **craft / adoption / persistence** -- the three components that carry the most
  weight, so a reader can see *why* a model scored where it did rather than
  taking the composite on trust.

The founder *structure* is an axis in its own right. ``solo`` (one agent),
``team`` (four specialists in a shared directory) and ``dynamic`` (one
orchestrator that picks its own team) are three different answers to "how should
a founding team be organised?", and a build made under one shape must never be
counted toward another's results -- so a table is always drawn for exactly one
structure. ``--matrix`` draws the cross-structure view by loading each arm
separately and putting them side by side.

Usage::

    scripts/model_table.py                        # markdown, for the results doc
    scripts/model_table.py --structure dynamic    # one arm
    scripts/model_table.py --matrix               # models x structures
    scripts/model_table.py --csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.crowd.sim_defaults import CROWD_ARCH_VERSION  # noqa: E402
from viral_bench.ideas import load_ideas  # noqa: E402
from viral_bench.score.fleet import (  # noqa: E402
    CURRENT_FLEET,
    FleetCorpus,
    load_corpus,
)
from viral_bench.score.viralscore import SCORE_VERSION, ScoreWeights  # noqa: E402

#: Every founder arm, in the order a reader should meet them: increasing
#: delegation, with the model-chosen shape last.
STRUCTURES = ("solo", "team", "dynamic")

#: Display names for models a table should present prettily. Empty by default:
#: which models a run covers is not knowable here, and a stale hard-coded roster
#: is worse than none. Add entries for your own run, or let the fallback stand.
PRETTY: dict[str, str] = {}

#: The model every other is differenced against. Empty means "the first model
#: found in the corpus", which is the right behaviour for a fleet this script
#: did not define.
REFERENCE = ""


def pretty(model: str) -> str:
    """A human label for a model id: the configured one, else the id itself."""
    return PRETTY.get(model, model)


def cells_by_model(corpus: FleetCorpus) -> dict[str, dict[str, list]]:
    out: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for run in corpus.fleet_runs():
        out[run.model][run.idea_id].append(run)
    return out


def paired_delta(
    cells: dict[str, dict[str, list]], model: str, reference: str
) -> tuple[float | None, int, int]:
    """Mean per-brief (model − reference), and how many briefs it spans."""
    mine, theirs = cells.get(model, {}), cells.get(reference, {})
    shared = sorted(set(mine) & set(theirs))
    deltas = [
        statistics.fmean(r.score for r in mine[i])
        - statistics.fmean(r.score for r in theirs[i])
        for i in shared
    ]
    if not deltas:
        return (None, 0, 0)
    return (statistics.fmean(deltas), len(deltas), sum(1 for d in deltas if d > 0))


def build_rows(corpus: FleetCorpus) -> list[dict]:
    cells = cells_by_model(corpus)
    builds = corpus.fleet_builds()
    n_ideas = len({i.idea_id for i in load_ideas()})
    rows = []
    for model in CURRENT_FLEET.short_models:
        runs = [r for r in corpus.fleet_runs() if r.model == model]
        if not runs:
            continue
        mine = [b for b in builds if b.model == model]
        scores = [r.score for r in runs]
        per_cell = [statistics.fmean(r.score for r in v) for v in cells[model].values()]
        withins = [
            statistics.pstdev([r.score for r in v])
            for v in cells[model].values()
            if len(v) > 1
        ]
        delta, n_paired, wins = paired_delta(cells, model, REFERENCE)

        def comp(name: str, runs: list = runs) -> float | None:
            vals = [
                r.components[name] for r in runs if r.components.get(name) is not None
            ]
            return statistics.fmean(vals) if vals else None

        rows.append(
            {
                "model": PRETTY.get(model, model),
                "id": model,
                "built": sum(1 for b in mine if b.ok),
                "attempted": len(mine) or n_ideas,
                "runs": len(runs),
                "score": statistics.fmean(scores),
                "brief_sd": statistics.pstdev(per_cell) if len(per_cell) > 1 else 0.0,
                "noise": statistics.fmean(withins) if withins else None,
                "delta": delta,
                "n_paired": n_paired,
                "wins": wins,
                "craft": comp("craft"),
                "adoption": comp("adoption"),
                "persistence": comp("persistence"),
                "minutes": (
                    statistics.fmean(
                        [(b.build_seconds or 0) / 60 for b in mine if b.ok]
                    )
                    if any(b.ok for b in mine)
                    else None
                ),
            }
        )
    return sorted(rows, key=lambda r: -r["score"])


def _f(value, spec=".1f", missing="--"):
    return missing if value is None else format(value, spec)


def arm_spec(structure: str, replicate: int):
    """The fleet spec for one founder arm.

    Only the arm selector moves, and the model list stays exactly as CURRENT_FLEET
    declares it, so two arms are always compared over the same models.
    """
    return replace(CURRENT_FLEET, structure=structure, replicate=replicate)


def load_arm(structure: str, replicate: int) -> list[dict]:
    """Rows for one arm, or [] when nothing of it has been scored yet."""
    return build_rows(load_corpus(REPO, spec=arm_spec(structure, replicate)))


def resolve_arm(
    structure: str, preferred: int, strict: bool = False
) -> tuple[list[dict], int]:
    """Rows for an arm plus the replicate they came from.

    Arms do not all live at the same replicate. solo and team were built as
    replicate 2, while the dynamic arm was rebuilt as replicate 1 after its founder
    brief changed, because a brief change does NOT retire existing builds --
    ``brief_fingerprint`` hashes only ``design_prompt`` + ``build_prompt``, and
    the dynamic brief is a separate function outside that hash. Pinning the
    matrix to one replicate silently drops whichever arm is not on it, so each
    arm reports the replicate it has runs for, and the caller is told
    which one that was.

    ``strict`` turns that fallback off, and a sweep in flight is exactly when it
    is needed. The fallback was written for a corpus whose arms sat at r1 and r2
    under the SAME founder code, where borrowing a neighbouring replicate cost
    nothing. It is unsafe the moment a replicate marks a new *era*: asking for
    the arm at replicate N while only N-1 has been scored quietly draws the old
    era's numbers into a table captioned as the new one, and the arms most likely
    to lag are the slowest ones -- so the mixing lands on whichever pipeline took
    longest, not at random. The per-arm summary underneath does disclose the
    replicate it used, but by then the headline matrix has already been read.
    """
    candidates = (
        (preferred,) if strict else (preferred, *(r for r in (2, 1) if r != preferred))
    )
    for rep in candidates:
        rows = load_arm(structure, rep)
        if rows:
            return rows, rep
    return [], preferred


#: Build-level coverage a table must reach before it may be called complete.
#:
#: Not 100%: some builds are unscorable (a founder that shipped no manifest has
#: nothing to run, and that is a result, not a gap). 99% is tight enough that
#: the coverage hole an early pass reported as "complete" would have failed it
#: loudly.
COVERAGE_FLOOR = 0.99


def _cohort_note(replicate: int) -> str:
    """Name the cohort this table covers, when the builds agree on one.

    A table that says only "replicate 3" does not identify its corpus: a cohort
    can keep some arms and rebuild others, so two tables at the same replicate
    can cover different builds. Reported as a caption rather than used as a
    filter -- the cohort is a label, and the arms are still selected by
    --structure/--replicate.
    """
    from viral_bench.score.fleet import fleet_replicates, load_builds

    builds = load_builds(REPO / "builds")
    at_rep = [
        builds[b]
        for b, rep in fleet_replicates(REPO / "builds").items()
        if rep == replicate and b in builds
    ]
    names = {b.cohort for b in at_rep if b.cohort}
    if not names:
        return ""
    tagged = sum(1 for b in at_rep if b.cohort)
    label = names.pop() if len(names) == 1 else f"MIXED {sorted(names)}"
    if tagged == len(at_rep):
        return f", cohort {label}"
    # Say so rather than implying the whole corpus is in it. Half-tagged is the
    # normal mid-sweep state: a cohort tags the kept arms before the rebuilt ones
    # exist -- and a caption that hid it would be a completeness claim at the
    # wrong grain, which is the exact failure the coverage work exists to stop.
    return f", cohort {label} (PARTIAL: {tagged}/{len(at_rep)} builds tagged)"


def arm_coverage(structure: str, replicate: int):
    """Build-level coverage for one arm."""
    corpus = load_corpus(REPO, spec=arm_spec(structure, replicate))
    return corpus.build_coverage(structure=structure)


def print_coverage_caveat(coverage: dict) -> None:
    """State what the table is missing, in builds, by name.

    THE HEADLINE USED TO SAY ``Models scored 10/10`` AND MEAN IT while 26 of
    1,000 builds had never scored at all. Both facts were true: a model's mean is
    taken over the runs that exist, so a missing build leaves the denominator
    instead of showing up as a hole, and the claim was checked at the only grain
    that could not see it. Any completeness statement here now names its gaps or
    is not made.
    """
    holes = {s: c for s, c in coverage.items() if not c.complete}
    if not holes:
        print()
        print(
            "> Build-level coverage is complete in every arm shown: every fleet "
            "build has at least one scored run."
        )
        return
    worst = min(c.fraction for c in holes.values())
    label = "INCOMPLETE" if worst < COVERAGE_FLOOR else "Note"
    print()
    print(f"> **{label}: not every build is scored.**")
    for arm, cov in sorted(holes.items()):
        missing = cov.builds - cov.covered
        named = ", ".join(f"`{b}`" for b in cov.uncovered_ids[:6])
        more = f" and {missing - 6} more" if missing > 6 else ""
        extra = (
            f" ({cov.app_start_failed} could not be started by the harness)"
            if cov.app_start_failed
            else ""
        )
        print(
            f"> - **{arm}**: {missing} of {cov.builds} unscored{extra}: {named}{more}"
        )


def print_matrix(replicate: int, strict: bool = False) -> None:
    """Models down, founder structures across.

    Each arm is loaded on its own spec, so a build made under one shape can
    never leak into another's column.
    """
    resolved = {s: resolve_arm(s, replicate, strict=strict) for s in STRUCTURES}
    reps = {s: rep for s, (rows, rep) in resolved.items() if rows}
    arms = {s: {r["id"]: r for r in rows} for s, (rows, _) in resolved.items()}
    present = [s for s in STRUCTURES if arms[s]]
    if not present:
        print(
            f"no scored runs in any founder arm at replicate {replicate}"
            if strict
            else "no scored runs in any founder arm yet"
        )
        return
    # Say it ABOVE the table, not in the summary underneath. A matrix whose
    # columns come from different replicates is comparing eras as well as
    # pipelines, and a reader who takes the headline at face value has already
    # been misled by the time they reach the per-arm rows.
    if len(set(reps.values())) > 1:
        spread = ", ".join(f"{s}=r{reps[s]}" for s in present)
        print(f"> **WARNING: mixed replicates** ({spread}).")
        print(
            "> These columns are not a like-for-like comparison -- they were "
            "built at different times, potentially under different founder code."
        )
        print("> Re-run with `--strict-replicate` to draw one replicate only.")
        print()
    if len(present) < len(STRUCTURES):
        missing = ", ".join(s for s in STRUCTURES if s not in present)
        print(f"> **Note:** no scored runs for: {missing}.")
        print()
    order = sorted(
        {m for s in present for m in arms[s]},
        key=lambda m: (
            -statistics.fmean([arms[s][m]["score"] for s in present if m in arms[s]])
        ),
    )
    head = " | ".join(s for s in present)
    print(f"| Model | {head} | mean |")
    print("|---" * (len(present) + 2) + "|")
    for m in order:
        vals = [arms[s].get(m, {}).get("score") for s in present]
        got = [v for v in vals if v is not None]
        cells = " | ".join(_f(v) for v in vals)
        mean = _f(statistics.fmean(got)) if got else "--"
        print(f"| {PRETTY.get(m, m)} | {cells} | **{mean}** |")
    print()
    print(
        "| Arm | Replicate | Models | Builds scored | Seeds min/med | Runs | "
        "Mean ViralScore |"
    )
    print("|---|---|---|---|---|---|---|")
    coverage = {s: arm_coverage(s, reps[s]) for s in present}
    for s in present:
        rows = list(arms[s].values())
        cov = coverage[s]
        depths = [d for d, n in cov.seed_histogram.items() for _ in range(n) if d]
        seeds = f"{min(depths)}/{statistics.median(depths):.0f}" if depths else "--"
        print(
            f"| {s} | r{reps[s]} | {len(rows)}/10 | "
            f"{cov.covered}/{cov.builds} ({100 * cov.fraction:.1f}%) | {seeds} | "
            f"{sum(r['runs'] for r in rows)} | "
            f"**{_f(statistics.fmean([r['score'] for r in rows]))}** |"
        )
    print_coverage_caveat(coverage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument(
        "--structure",
        default=CURRENT_FLEET.structure,
        choices=STRUCTURES,
        help="which founder arm to draw (default: %(default)s)",
    )
    parser.add_argument("--replicate", type=int, default=CURRENT_FLEET.replicate)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="draw every founder arm side by side instead of one table",
    )
    parser.add_argument(
        "--strict-replicate",
        action="store_true",
        help="draw ONLY --replicate, never falling back to a neighbouring one. "
        "Use this for any table reporting a single sweep: without it an arm "
        "that is not yet scored is filled in from an older replicate, which "
        "silently mixes eras in the headline matrix.",
    )
    args = parser.parse_args(argv)

    # Say which score produced these numbers, in the output itself. A table
    # captioned only "ViralScore" is not reproducible: the weights have moved six
    # times, every stored run records the profile it was scored under, and a
    # reader comparing two documents has no other way to know whether they used
    # the same definition.
    weights = ScoreWeights.from_profile()
    print(
        f"<!-- score profile: {weights.profile}, "
        f"score_version {SCORE_VERSION}, crowd arch v{CROWD_ARCH_VERSION}"
        f"{_cohort_note(args.replicate)} -->"
    )

    if args.matrix:
        print_matrix(args.replicate, strict=args.strict_replicate)
        return 0

    corpus = load_corpus(REPO, spec=arm_spec(args.structure, args.replicate))
    rows = build_rows(corpus)
    if not rows:
        print(f"no scored runs for any model in the {args.structure} arm yet")
        return 0

    if args.csv:
        cols = list(rows[0])
        print(",".join(cols))
        for r in rows:
            print(",".join("" if r[c] is None else str(r[c]) for c in cols))
        return 0

    print("| # | Model | Built | ViralScore | vs Opus 5 | Wins | Craft | Adoption |")
    print("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        delta = (
            "reference"
            if r["id"] == REFERENCE
            else f"{_f(r['delta'], '+.1f')} ({r['n_paired']} briefs)"
        )
        wins = "--" if r["id"] == REFERENCE else f"{r['wins']}/{r['n_paired']}"
        print(
            f"| {i} | **{r['model']}** | {r['built']}/{r['attempted']} | "
            f"**{_f(r['score'])}** ± {_f(r['brief_sd'])} | {delta} | {wins} | "
            f"{_f(r['craft'], '.2f')} | {_f(r['adoption'], '.2f')} |"
        )
    print()
    print("Supporting detail:")
    print()
    print("| Model | Runs | Seed noise | Persistence | Build min |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['model']} | {r['runs']} | {_f(r['noise'], '.2f')} | "
            f"{_f(r['persistence'], '.2f')} | {_f(r['minutes'])} |"
        )
    print_coverage_caveat(
        {args.structure: corpus.build_coverage(structure=args.structure)}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
