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

"""Price the validity gate: is a deterministic override worth what it costs?

The validity gate is the only part of the ViralScore that is not the crowd's
opinion. It multiplies a finished score by a fixed number because a container
probe said the app does not run. That is a strong claim to hard-code, and the
two multipliers it uses (0.2 and 0.6) date from the first ViralScore commit
and were never fitted to anything -- the ``gates:`` block in config/score.yaml
has said "Sweep it" ever since.

This is the sweep. It re-prices every stored run under a grid of gate policies
and multipliers and ranks them on the project's standing rule -- **null, then
discrimination, then control margin, with the model gap read off last** -- so a
policy cannot win by widening the gap.

The policies:

* ``hard``       -- shipped. Multiplier whenever verification failed.
* ``off``        -- no gate. The score is whatever the crowd said.
* ``evidence_k`` -- multiplier only when FEWER THAN k agents got the app working
  first-hand. The probe stops overriding the crowd and becomes a tie-breaker for
  runs where the crowd has no evidence of its own. Motivated by a measured false
  positive: ``collaborative_table``/``gemini-2.0-flash`` seed 2 scored the
  HIGHEST adoption (0.67) and craft (0.78) of its three seeds, and the gate cut
  it from 76.0 to 15.2 -- the only cell in 243 where the gate flaps across seeds.
* ``graded``     -- multiplier scales continuously with the fraction of the crowd
  that reached the app, so evidence buys back credit smoothly rather than at a
  cliff.

Usage::

    scripts/gate_sweep.py                     # both profiles, full grid
    scripts/gate_sweep.py --profile v5_advocacy
    scripts/gate_sweep.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from viral_bench.score.fleet import (  # noqa: E402
    CURRENT_FLEET,
    FleetCorpus,
    control_separation,
    paired_gap,
    self_separation,
    within_cell_sd,
)
from viral_bench.score.signals import extract_signals  # noqa: E402
from viral_bench.score.viralscore import ScoreWeights  # noqa: E402

#: Multipliers to price for the broken-app tier. 1.0 is "no penalty" and is
#: included so the grid contains its own null hypothesis.
BROKEN_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)

#: Evidence thresholds for ``evidence_k``: how many agents must have got the app
#: working before their first-hand experience outranks the container probe.
EVIDENCE_GRID = (1, 2, 3, 5, 10)


def gate_for(policy: str, mult: float, k: int, sig_row: dict) -> float:
    """The multiplier a policy applies to one run."""
    if sig_row["claims"] is not False:  # verified, or never checked
        return 1.0
    dead = sig_row["dead"]
    base = mult if dead else 1.0 - (1.0 - mult) * 0.5  # self-check tier is softer
    if policy == "off":
        return 1.0
    if policy == "hard":
        return base
    if policy == "evidence":
        return 1.0 if sig_row["valid_trials"] >= k else base
    if policy == "graded":
        reach = sig_row["reach"]
        return base + (1.0 - base) * reach
    raise ValueError(policy)


def load(profile: str) -> tuple[FleetCorpus, dict]:
    """Corpus scored under ``profile``, plus the gate inputs for every run."""
    from viral_bench.score.fleet import load_corpus

    weights = ScoreWeights.from_profile(profile)
    corpus = load_corpus(REPO, weights=weights, spec=CURRENT_FLEET)
    rows: dict[str, dict] = {}
    for run in corpus.runs:
        if run.score is None:
            continue
        try:
            sig = extract_signals(Path(run.crowd_dir))
        except (FileNotFoundError, ValueError, KeyError):
            continue
        exposed = sig.exposed_agents or 0
        rows[run.crowd_dir] = {
            "claims": sig.does_what_it_claims,
            "dead": sig.builds is False or sig.runs is False,
            "valid_trials": sig.n_valid_trials,
            "reach": (sig.n_valid_trials / exposed) if exposed else 0.0,
            "raw": run.score / run.gate if run.gate else run.score,
        }
    return corpus, rows


def regate(corpus: FleetCorpus, rows: dict, policy: str, mult: float, k: int):
    """A copy of the corpus rescored under one gate policy."""
    out = []
    for run in corpus.runs:
        row = rows.get(run.crowd_dir)
        if row is None:
            out.append(run)
            continue
        g = gate_for(policy, mult, k, row)
        out.append(replace(run, score=round(row["raw"] * g, 1), gate=g))
    return replace(corpus, runs=out)


def floored_cells(corpus: FleetCorpus, eps: float = 0.05) -> int:
    """Cells whose every run sits on the floor, so the cell has no variance.

    A gate floor of exactly 0.0 sends dead runs to a hard zero. Zero-variance
    cells drag the pooled within-cell SD down and shrink the seed-parity null
    towards nothing, which shows up as a better instrument when in fact the
    scale has stopped resolving anything down there. This counts the cells
    that would do it.
    """
    n = 0
    for runs in corpus.cells().values():
        if len(runs) > 1 and all((r.score or 0.0) <= eps for r in runs):
            n += 1
    return n


def stats_for(corpus: FleetCorpus) -> dict:
    """The four numbers a scoring candidate is judged on, in priority order."""
    a, b = CURRENT_FLEET.model_a, CURRENT_FLEET.model_b
    gap = paired_gap(corpus, a, b)
    na, nb = self_separation(corpus, a), self_separation(corpus, b)
    sep = control_separation(corpus)
    noise = within_cell_sd(corpus)
    means = [statistics.fmean(r.score for r in v) for v in corpus.cells().values() if v]
    # Whichever model separates from itself worst, with ITS bootstrap interval.
    worst = max((na, nb), key=lambda p: abs(p.gap or 0.0))
    return {
        "floored_cells": floored_cells(corpus),
        "null": round(abs(worst.gap or 0.0), 2),
        "null_lo": worst.ci_low,
        "null_hi": worst.ci_high,
        "noise": noise,
        "ratio": (
            round(statistics.pstdev(means) / noise, 2)
            if noise and len(means) > 1
            else None
        ),
        "margin": (
            round(sep.working_median - sep.control_mean, 1)
            if sep.working_median is not None and sep.control_mean is not None
            else None
        ),
        "control": sep.control_mean,
        "gap": gap.gap,
        "below": len(sep.below_control),
    }


def rank_key(s: dict) -> tuple:
    """The standing selection rule: null, then discrimination. Gap is NOT here.

    ``margin`` is deliberately absent. The broken control builds, runs, passes
    its own self-check and is reached by all 30 agents -- it is a useless app,
    not a dead one -- so no gate policy touches its score and the control margin
    is constant across the whole grid. Ranking on a constant is noise.
    """
    return (s["null"], -(s["ratio_alive"] or 0))


def tied_on_null(results: list[dict]) -> list[dict]:
    """Candidates whose null cannot be distinguished from the best one.

    The null is a bootstrap statistic over 25 briefs, so ranking 64 candidates on
    its point estimate to two decimals is how you overfit an ablation. Anything
    whose interval covers the best point estimate is treated as tied, and the
    tie is broken on discrimination instead.
    """
    best = min(results, key=lambda s: s["null"])["null"]
    out = []
    for s in results:
        lo, hi = s.get("null_lo"), s.get("null_hi")
        span = max(abs(lo or 0.0), abs(hi or 0.0))
        if s["null"] <= best + 1e-9 or span >= best:
            out.append(s)
    return out


def sweep(profile: str) -> list[dict]:
    corpus, rows = load(profile)
    fleet = {r.crowd_dir for r in corpus.fleet_runs()}
    g = [rows[d] for d in fleet if d in rows and rows[d]["claims"] is False]
    spared = sum(1 for row in g if row["valid_trials"] >= 1)
    print(
        f"\n## {profile}  (fleet: {len(fleet)} runs, {len(g)} marked broken by the "
        f"probe, {spared} of those with hands-on evidence)\n"
    )

    candidates: list[tuple[str, str, float, int]] = [("off", "off", 1.0, 0)]
    for m in BROKEN_GRID:
        candidates.append((f"hard m={m}", "hard", m, 0))
    for m in BROKEN_GRID:
        for k in EVIDENCE_GRID:
            candidates.append((f"evidence k={k} m={m}", "evidence", m, k))
    for m in BROKEN_GRID:
        candidates.append((f"graded m={m}", "graded", m, 0))

    results = []
    for label, policy, mult, k in candidates:
        rg = regate(corpus, rows, policy, mult, k)
        s = stats_for(rg)
        # Re-price on cells the candidate has NOT flattened onto its floor. If a
        # candidate only looks good with those cells in, its gain is the floor
        # collapsing, not the instrument resolving better -- which is exactly
        # how m=0.0 first won this sweep.
        alive = {
            key
            for key, runs in rg.cells().items()
            if any((r.score or 0.0) > 0.05 for r in runs)
        }
        fleet_ids = {r.crowd_dir for r in rg.fleet_runs()}
        sub = replace(
            rg,
            runs=[
                r
                for r in rg.runs
                if r.crowd_dir not in fleet_ids
                or (r.idea_id, r.model, r.structure) in alive
            ],
        )
        held = stats_for(sub)
        s.update(
            label=label,
            policy=policy,
            mult=mult,
            k=k,
            profile=profile,
            null_alive=held["null"],
            ratio_alive=held["ratio"],
        )
        results.append(s)
    return results


def report(results: list[dict], top: int = 12) -> None:
    ranked = sorted(tied_on_null(results), key=lambda s: -(s["ratio_alive"] or 0))
    print("  candidates tied with the best null, ranked by discrimination:\n")
    print(
        f"  {'candidate':22}{'null':>7}{'ratio':>7}{'ratio*':>8}"
        f"{'floored':>9}{'noise':>7}{'gap':>8}"
    )
    for s in ranked[:top]:
        print(
            f"  {s['label']:22}{s['null']:7.2f}{(s['ratio'] or 0):7.2f}"
            f"{(s['ratio_alive'] or 0):8.2f}{s['floored_cells']:9d}"
            f"{(s['noise'] or 0):7.2f}{(s['gap'] or 0):+8.1f}"
        )
    shipped = next(s for s in results if s["label"] == "hard m=0.2")
    off = next(s for s in results if s["label"] == "off")
    print(
        f"\n  shipped (hard m=0.2) ranks {ranked.index(shipped) + 1} of {len(ranked)}; "
        f"no gate ranks {ranked.index(off) + 1}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", action="append", default=None)
    ap.add_argument("--json", default="")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    profiles = args.profile or ["v4_earned", "v5_advocacy"]

    everything = []
    for profile in profiles:
        results = sweep(profile)
        everything.extend(results)
        report(results, args.top)

    if len(profiles) > 1:
        print("\n## Candidates that rank top-5 under BOTH profiles\n")
        per = {}
        for profile in profiles:
            ranked = sorted(
                (s for s in everything if s["profile"] == profile), key=rank_key
            )
            for i, s in enumerate(ranked):
                per.setdefault(s["label"], []).append(i + 1)
        both = {k: v for k, v in per.items() if len(v) == len(profiles) and max(v) <= 5}
        for label, ranks in sorted(both.items(), key=lambda kv: sum(kv[1])):
            print(f"  {label:22} ranks {ranks}")
        if not both:
            print("  (none)")

    if args.json:
        Path(args.json).write_text(json.dumps(everything, indent=1, default=str))
        print(f"\nraw results -> {args.json}")


if __name__ == "__main__":
    main()
