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

"""Is the model ranking an artifact of the ViralScore weights?

The challenge this answers is the obvious one to put to any composite score:
*"can't you just tweak the weights until your favourite model wins?"* It is a
fair question and it deserves a measurement rather than an assurance.

Scoring is a pure function of stored artifacts, and every run on disk already
carries its raw component values in [0,1] plus its gate multiplier. So a
weighting can be re-priced without re-simulating anything::

    score = 100 * gate * (sum_k w_k c_k) / (sum_k w_k)     over MEASURED k

That makes the whole weight simplex searchable. Five analyses:

**A. Exhaustive bound.** On runs where every component was measured the
per-model mean is *linear* in the weight vector, so its extrema over the simplex
are attained at vertices -- weight 1.0 on a single component. Evaluating the
vertices therefore gives each model's best and worst achievable score over
*every weighting that exists*, as a bound rather than a sample. Pairs whose
order survives that bound cannot be reordered by any weighting at all.

**B. Monte Carlo.** Uniform draws over the simplex, plus sparse
(corner-seeking) draws that concentrate weight on one or two components, plus a
family constrained to the shipped 0.85/0.15 deterministic-to-autorater split.
Reports how often each model takes first place and how far the ranking moves.

**C. Adversarial search.** For each model in turn, hill-climb the weights to
maximise *that model's* score, and separately to maximise its rank. This is the
literal form of the challenge: someone actively trying to promote a model.

**D. The gate.** The validity gate is not a weight but it is the strongest
single lever in the score, so the ranking is re-derived with it disabled.

**E. How the weights were set.** Per-component discrimination (between-cell
spread over within-cell noise), the component correlation matrix, and -- the
part that actually answers the accusation -- the weighting that *maximises* the
reported model gap, shown next to the shipped one so the difference between
"chosen for the construct" and "chosen for the headline" is a number.

Usage::

    scripts/score_robustness.py                    # full study, markdown to stdout
    scripts/score_robustness.py --draws 20000      # smaller Monte Carlo
    scripts/score_robustness.py --json out.json    # also dump raw results
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from model_table import PRETTY, REFERENCE  # noqa: E402

from viral_bench.score.fleet import (  # noqa: E402
    CURRENT_FLEET,
    FleetSpec,
    load_corpus,
)
from viral_bench.score.viralscore import ScoreWeights  # noqa: E402

#: Component order used by every matrix here. Mirrors ``ScoreWeights.as_dict()``
#: (deterministic terms first, then the autorater dimensions).
COMPONENTS = (
    "adoption",
    "advocacy",
    "craft",
    "amplification",
    "advocacy_spread",
    "reception",
    "persistence",
    "cascade",
    "substance",
    "severity",
    "word_of_mouth",
)


#: The last hand-tuned profile, kept as the comparison baseline for section F.
#: Once the flat profile is active, SHIPPED() returns the flat vector, so the
#: "did tuning matter" question needs the tuned weighting named explicitly.
TUNED_BASELINE = "v6_witnessed"


def equal_sixths() -> np.ndarray:
    """Flat weighting: six parts, the autorater counted as ONE of them.

    The rater is a single judgement measured over three dimensions, so it takes
    one sixth in total rather than three sixths -- otherwise "equal" would
    quietly hand half the score to the LLM.
    """
    rater = {"substance", "severity", "word_of_mouth"}
    live = [
        n for n in COMPONENTS if n not in rater and SHIPPED()[COMPONENTS.index(n)] > 0
    ]
    parts = len(live) + 1
    w = np.zeros(len(COMPONENTS))
    for name in live:
        w[COMPONENTS.index(name)] = 1.0 / parts
    for name in rater:
        w[COMPONENTS.index(name)] = (1.0 / parts) / len(rater)
    return w


def SHIPPED(profile: str | None = None) -> np.ndarray:  # noqa: N802
    """The active profile's weights, in ``COMPONENTS`` order.

    Read from config rather than hardcoded: this script hardcoded the v4_earned
    vector once, and when the active profile moved to v5 the only thing that
    caught it was the validation below refusing to reconcile.
    """
    w = ScoreWeights.from_profile(profile).as_dict()
    return np.array([float(w.get(name, 0.0)) for name in COMPONENTS])


#: Tier assignment as reported, so tier stability can be checked directly.
#:
#: Cut where the data has a gap far larger than the noise floor, not at round
#: numbers. On r4 (pooled, 3 seeds, noise 3.71) the model means run
#: 68.9 64.8 63.9 61.0 59.9 58.0 | 41.9 40.3 38.6 | 17.6, so the only two
#: defensible boundaries are the 16.1- and 21.0-point drops. Inside the top
#: group every neighbouring gap is 4.1 or less and most are inside noise, so
#: splitting it further would assert an ordering this sweep cannot resolve.
#:
#: Expected quality tier per model, used only to report how often the score's
#: ranking agrees with a prior expectation.
#:
#: This map is DATA and it goes stale -- ours did, silently. An earlier cohort's
#: ranking put two models in the wrong order relative to a later one, and
#: scoring the later cohort against the stale map reported ~97% "cross-tier
#: compliance" for what were really tier reassignments. It ships EMPTY for that
#: reason: fill it in for your own models, and re-derive it on any new cohort
#: before quoting a cross-tier number.
TIERS: dict[str, int] = {}


def specs_for(replicate: int, arms: tuple[str, ...]) -> list:
    """One FleetSpec per founder arm, all at the same replicate.

    ``FleetSpec`` names a single ``structure`` on purpose -- pooling two founder
    shapes into one scoring cell is the bug it exists to prevent. Pooling them
    for a WEIGHTING study is a different question: the weights are global, so
    the honest test is whether the ranking they produce survives over the whole
    sweep rather than inside one arm. Each arm is therefore loaded under its own
    spec, still keyed by its own structure, and only the run lists are combined.
    """
    return [
        FleetSpec(
            model_a=CURRENT_FLEET.model_a,
            model_b=CURRENT_FLEET.model_b,
            structure=arm,
            replicate=replicate,
            extra_models=CURRENT_FLEET.extra_models,
        )
        for arm in arms
    ]


class Corpus:
    """The stored corpus as matrices, so any weighting can be priced instantly."""

    def __init__(self, spec=CURRENT_FLEET, *, specs=None, seeds=None):
        specs = list(specs) if specs else [spec]
        runs: list = []
        corpus = None
        for one in specs:
            loaded = load_corpus(REPO, spec=one)
            corpus = corpus or loaded
            runs.extend(r for r in loaded.fleet_runs() if r.score is not None)
        if seeds is not None:
            keep = set(seeds)
            runs = [r for r in runs if r.seed in keep]
        runs.sort(key=lambda r: (r.model, r.idea_id, r.seed, r.crowd_dir))
        self.runs = runs
        self.corpus = corpus

        n, k = len(runs), len(COMPONENTS)
        self.values = np.zeros((n, k))
        self.measured = np.zeros((n, k))
        for i, run in enumerate(runs):
            for j, name in enumerate(COMPONENTS):
                v = run.components.get(name)
                if v is not None:
                    self.values[i, j] = float(v)
                    self.measured[i, j] = 1.0
        self.gate = np.array([r.gate for r in runs])
        self.models = [r.model for r in runs]
        self.model_ids = sorted({r.model for r in runs})
        self.rows_by_model = {
            m: np.array([i for i, r in enumerate(runs) if r.model == m])
            for m in self.model_ids
        }
        # (idea, model, structure) cells, and the seed-parity split used by the null.
        # Structure is part of the key. Without it, pooling founder arms folds a
        # real between-arm difference into "within-cell noise" -- measured here
        # as within-cell SD 20.2 against a true seed-to-seed ~4, which would in
        # turn crush the discrimination ratio to nothing. Same bug, and same
        # fix, as FleetCorpus.cells().
        cells: dict[tuple[str, str, str], list[int]] = {}
        for i, r in enumerate(runs):
            cells.setdefault((r.idea_id, r.model, r.structure), []).append(i)
        self.cells = cells
        self.complete = self.measured.all(axis=1)

    def score(self, weights: np.ndarray, *, gate: bool = True) -> np.ndarray:
        """Per-run scores for one or many weightings -> (n_runs, n_weightings)."""
        w = np.atleast_2d(weights)
        num = self.values @ w.T
        den = self.measured @ w.T
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        g = self.gate[:, None] if gate else 1.0
        return 100.0 * g * raw

    def model_scores(self, weights: np.ndarray, *, gate: bool = True) -> np.ndarray:
        """Mean score per model -> (n_models, n_weightings). Matches model_table."""
        s = self.score(weights, gate=gate)
        out = np.empty((len(self.model_ids), s.shape[1]))
        for m, model in enumerate(self.model_ids):
            rows = self.rows_by_model[model]
            with np.errstate(invalid="ignore"):
                out[m] = np.nanmean(s[rows], axis=0)
        return out

    def cell_means(self, weights: np.ndarray, *, gate: bool = True) -> dict:
        s = self.score(weights, gate=gate)
        return {key: np.nanmean(s[idx], axis=0) for key, idx in self.cells.items()}


# --------------------------------------------------------------------------- #
# Instrument statistics, vectorised over weightings
# --------------------------------------------------------------------------- #


def instrument_stats(c: Corpus, weights: np.ndarray, *, gate: bool = True) -> dict:
    """Gap, null, noise and discrimination for a single weighting."""
    s = c.score(weights, gate=gate)[:, 0]
    means = {key: float(np.nanmean(s[idx])) for key, idx in c.cells.items()}

    # Paired gap, Opus 5 minus the Gemini A/B partner. Pairing is on
    # (brief, arm) rather than brief alone: when several founder arms are
    # pooled, the same brief appears once per arm and pairing on the brief
    # would compare a solo build against a team one.
    a, b = REFERENCE, CURRENT_FLEET.model_a
    keys = sorted(
        {(i, st) for (i, m, st) in means if m == a}
        & {(i, st) for (i, m, st) in means if m == b}
    )
    deltas = [means[(i, a, st)] - means[(i, b, st)] for i, st in keys]
    gap = statistics.fmean(deltas) if deltas else None

    # within-cell (seed) noise, pooled as fleet.within_cell_sd does
    variances, cell_list = [], []
    for idx in c.cells.values():
        vals = [v for v in s[idx] if not np.isnan(v)]
        if len(vals) > 1:
            variances.append(statistics.variance(vals))
        cell_list.append(statistics.fmean(vals) if vals else float("nan"))
    within = (sum(variances) / len(variances)) ** 0.5 if variances else None
    finite = [v for v in cell_list if not np.isnan(v)]
    between = statistics.stdev(finite) if len(finite) > 1 else None

    # null: one model against itself, seeds split by parity
    nulls = {}
    for model in (REFERENCE, CURRENT_FLEET.model_a):
        per_idea = []
        for (_idea, m, _st), idx in c.cells.items():
            if m != model or len(idx) < 2:
                continue
            vals = s[np.array(idx)]
            first, second = vals[0::2], vals[1::2]
            if len(first) and len(second):
                per_idea.append(float(np.nanmean(first) - np.nanmean(second)))
        nulls[model] = statistics.fmean(per_idea) if per_idea else None

    return {
        "gap": gap,
        "within_sd": within,
        "between_sd": between,
        "ratio": (between / within) if within else None,
        "null_a": nulls.get(REFERENCE),
        "null_b": nulls.get(CURRENT_FLEET.model_a),
        "worst_null": max(abs(v) for v in nulls.values() if v is not None),
    }


# --------------------------------------------------------------------------- #
# A. Exhaustive vertex bound
# --------------------------------------------------------------------------- #


def live_mask(live_only: bool) -> np.ndarray:
    """Which components the weight simplex is allowed to touch.

    ``reception`` and ``cascade`` carry weight 0.00 because they were MEASURED
    to be degenerate -- reception has a between/within ratio near 1 and points
    at the weaker model, cascade has no variance at all. A weighting that loads
    them is not an alternative opinion about what matters, it is a weighting
    that reads a near-constant, so the study reports the simplex both with and
    without them.
    """
    keep = np.ones(len(COMPONENTS), dtype=bool)
    if live_only:
        keep = SHIPPED() > 0
    return keep


def vertex_bound(
    c: Corpus, *, complete_only: bool = True, live_only: bool = False
) -> dict:
    """Best and worst achievable score per model over the WHOLE simplex.

    On complete-case runs the per-model mean is linear in the weights, so the
    extrema sit at the simplex vertices and evaluating them is a bound, not a
    sample.
    """
    rows = np.where(c.complete)[0] if complete_only else np.arange(len(c.runs))
    keep = live_mask(live_only)
    verts = np.eye(len(COMPONENTS))[keep]
    s = c.score(verts)[rows]
    models = [c.models[i] for i in rows]

    out = {}
    for model in c.model_ids:
        sel = np.array([i for i, m in enumerate(models) if m == model])
        if not len(sel):
            continue
        with np.errstate(invalid="ignore"):
            per_vertex = np.nanmean(s[sel], axis=0)
        shipped = float(
            np.nanmean(
                c.score(SHIPPED())[np.array([i for i in rows if c.models[i] == model])]
            )
        )
        names = [n for n, k in zip(COMPONENTS, keep, strict=True) if k]
        out[model] = {
            "min": float(np.nanmin(per_vertex)),
            "max": float(np.nanmax(per_vertex)),
            "shipped": shipped,
            "argmax": names[int(np.nanargmax(per_vertex))],
            "argmin": names[int(np.nanargmin(per_vertex))],
            "n_runs": int(len(sel)),
        }
    return out


def dominance(c: Corpus, bounds: dict, *, live_only: bool = False) -> dict:
    """Pairs whose order NO weighting can flip, decided exactly.

    On complete-case runs each model's mean is linear in the weight vector, so
    ``B(w) - A(w) = w . d`` where ``d_k = B(e_k) - A(e_k)`` is the difference at
    vertex ``k``. The maximum of a linear form over the simplex is its largest
    coefficient, so B can overtake A for *some* weighting if and only if it
    beats A at *some vertex*. Comparing marginal best-case to marginal
    worst-case would be wrong: those are attained at different weightings.
    """
    rows = np.where(c.complete)[0]
    keep = live_mask(live_only)
    names = [n for n, k in zip(COMPONENTS, keep, strict=True) if k]
    verts = np.eye(len(COMPONENTS))[keep]
    s = c.score(verts)[rows]
    models = [c.models[i] for i in rows]
    per_vertex = {}
    for model in c.model_ids:
        sel = np.array([i for i, m in enumerate(models) if m == model])
        with np.errstate(invalid="ignore"):
            per_vertex[model] = np.nanmean(s[sel], axis=0)

    order = sorted(bounds, key=lambda m: -bounds[m]["shipped"])
    fixed, flippable = [], []
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            d = per_vertex[b] - per_vertex[a]  # positive => b overtakes a there
            if np.nanmax(d) < 0:
                fixed.append((a, b))
            else:
                flippable.append(
                    (a, b, names[int(np.nanargmax(d))], float(np.nanmax(d)))
                )
    # A model that loses to someone at every vertex can never place first.
    never_first = []
    for m in order:
        beaten_everywhere = any(
            np.nanmin(per_vertex[m] - per_vertex[o]) < 0
            and np.nanmax(per_vertex[m] - per_vertex[o]) < 0
            for o in order
            if o != m
        )
        if beaten_everywhere:
            never_first.append(m)
    return {
        "order": order,
        "fixed": fixed,
        "flippable": flippable,
        "never_first": never_first,
        "per_vertex": {m: v.tolist() for m, v in per_vertex.items()},
    }


# --------------------------------------------------------------------------- #
# B. Monte Carlo over the simplex
# --------------------------------------------------------------------------- #


def sample_weights(rng, n: int, alpha: float, *, live_only: bool = False) -> np.ndarray:
    """Dirichlet draws over the simplex. ``live_only`` excludes retired terms."""
    k = len(COMPONENTS)
    w = rng.dirichlet(np.full(k, alpha), size=n)
    if live_only:  # zero the two retired components and renormalise
        for name in ("reception", "cascade"):
            w[:, COMPONENTS.index(name)] = 0.0
        w /= w.sum(axis=1, keepdims=True)
    return w


def monte_carlo(c: Corpus, weights: np.ndarray, chunk: int = 4000) -> dict:
    """Rank statistics over many weightings."""
    n_models = len(c.model_ids)
    firsts = np.zeros(n_models, dtype=int)
    top2 = np.zeros(n_models, dtype=int)
    ranks = np.zeros((n_models, len(weights)), dtype=np.int16)
    shipped_rank = np.argsort(np.argsort(-c.model_scores(SHIPPED())[:, 0]))

    at = 0
    for start in range(0, len(weights), chunk):
        block = weights[start : start + chunk]
        ms = c.model_scores(block)
        order = np.argsort(np.argsort(-ms, axis=0), axis=0)
        ranks[:, at : at + block.shape[0]] = order
        firsts += (order == 0).sum(axis=1)
        top2 += (order <= 1).sum(axis=1)
        at += block.shape[0]

    taus = kendall_tau(ranks, shipped_rank)
    pair_stable = pairwise_stability(ranks)
    return {
        "n": len(weights),
        "p_first": {c.model_ids[i]: firsts[i] / len(weights) for i in range(n_models)},
        "p_top2": {c.model_ids[i]: top2[i] / len(weights) for i in range(n_models)},
        "median_rank": {
            c.model_ids[i]: float(np.median(ranks[i]) + 1) for i in range(n_models)
        },
        "rank_range": {
            c.model_ids[i]: (int(ranks[i].min() + 1), int(ranks[i].max() + 1))
            for i in range(n_models)
        },
        "tau_mean": float(taus.mean()),
        "tau_p05": float(np.percentile(taus, 5)),
        "tau_min": float(taus.min()),
        "pairs_stable_95": pair_stable["stable_95"],
        "pairs_total": pair_stable["total"],
        "tier_violations": tier_violations(c, ranks),
    }


def kendall_tau(ranks: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Kendall tau-a of each column of ``ranks`` against ``reference``."""
    n = ranks.shape[0]
    iu = np.triu_indices(n, 1)
    ref_sign = np.sign(reference[iu[0]] - reference[iu[1]])[:, None]
    cur_sign = np.sign(ranks[iu[0], :] - ranks[iu[1], :])
    concordant = (ref_sign == cur_sign).sum(axis=0)
    total = len(iu[0])
    return (2.0 * concordant - total) / total


def pairwise_stability(ranks: np.ndarray) -> dict:
    n = ranks.shape[0]
    iu = np.triu_indices(n, 1)
    agree = (ranks[iu[0], :] < ranks[iu[1], :]).mean(axis=1)
    keep = np.maximum(agree, 1 - agree)
    return {"total": len(agree), "stable_95": int((keep >= 0.95).sum())}


def tier_violations(c: Corpus, ranks: np.ndarray) -> dict:
    """How often a lower tier outranks a higher one."""
    tiers = np.array([TIERS.get(m, 9) for m in c.model_ids])
    n = len(c.model_ids)
    iu = np.triu_indices(n, 1)
    cross = tiers[iu[0]] != tiers[iu[1]]
    higher_first = np.where(
        (tiers[iu[0]] < tiers[iu[1]])[:, None],
        ranks[iu[0], :] < ranks[iu[1], :],
        ranks[iu[1], :] < ranks[iu[0], :],
    )
    cross_pairs = higher_first[cross]
    return {
        "cross_tier_pairs": int(cross.sum()),
        "obeyed_rate": float(cross_pairs.mean()),
        "always_obeyed": int((cross_pairs.all(axis=1)).sum()),
    }


# --------------------------------------------------------------------------- #
# C. Adversarial search
# --------------------------------------------------------------------------- #


def promote(
    c: Corpus, model: str, rng, *, restarts: int = 12, steps: int = 400
) -> dict:
    """Hill-climb the weights to push one model as high as it can go."""
    idx = c.model_ids.index(model)
    best_rank, best_score, best_w = 99, -1.0, None
    for _ in range(restarts):
        w = rng.dirichlet(np.full(len(COMPONENTS), 0.5))
        ms = c.model_scores(w[None, :])[:, 0]
        cur_rank = int(np.argsort(np.argsort(-ms))[idx])
        cur_margin = ms[idx] - np.max(np.delete(ms, idx))
        for _ in range(steps):
            step = rng.dirichlet(np.full(len(COMPONENTS), 0.5))
            eta = rng.uniform(0.02, 0.4)
            cand = (1 - eta) * w + eta * step
            cand /= cand.sum()
            ms = c.model_scores(cand[None, :])[:, 0]
            rank = int(np.argsort(np.argsort(-ms))[idx])
            margin = ms[idx] - np.max(np.delete(ms, idx))
            if (rank, -margin) < (cur_rank, -cur_margin):
                w, cur_rank, cur_margin = cand, rank, margin
        if (cur_rank, -cur_margin) < (best_rank, -best_score):
            best_rank, best_score, best_w = cur_rank, cur_margin, w
    ms = c.model_scores(best_w[None, :])[:, 0]
    return {
        "model": model,
        "best_rank": best_rank + 1,
        "margin_to_first": float(best_score),
        "score": float(ms[idx]),
        "winner": c.model_ids[int(np.argmax(ms))],
        "weights": {
            COMPONENTS[i]: round(float(best_w[i]), 3) for i in range(len(COMPONENTS))
        },
    }


# --------------------------------------------------------------------------- #
# E. How the weights were set
# --------------------------------------------------------------------------- #


def component_discrimination(c: Corpus) -> dict:
    """Between-cell spread over within-cell noise, per component.

    This is the empirical basis for the weight ordering: a component whose
    app-to-app spread is large relative to its run-to-run noise can rank apps;
    one whose ratio is near 1 is a near-constant dressed up as a signal.
    """
    out = {}
    for j, name in enumerate(COMPONENTS):
        vals = np.where(c.measured[:, j] > 0, c.values[:, j], np.nan)
        cell_means, withins = [], []
        for idx in c.cells.values():
            v = vals[np.array(idx)]
            v = v[~np.isnan(v)]
            if len(v):
                cell_means.append(float(v.mean()))
            if len(v) > 1:
                withins.append(statistics.variance(v.tolist()))
        if len(cell_means) < 2 or not withins:
            continue
        within = (sum(withins) / len(withins)) ** 0.5
        between = statistics.stdev(cell_means)
        out[name] = {
            "between_sd": round(between, 4),
            "within_sd": round(within, 4),
            "ratio": round(between / within, 2) if within else None,
            "mean": round(float(np.nanmean(vals)), 3),
            "coverage": round(float(c.measured[:, j].mean()), 3),
        }
    return out


def component_correlations(c: Corpus) -> dict:
    vals = np.where(c.measured > 0, c.values, np.nan)
    live = [i for i, n in enumerate(COMPONENTS) if SHIPPED()[i] > 0]
    names = [COMPONENTS[i] for i in live]
    m = np.full((len(live), len(live)), np.nan)
    for a in range(len(live)):
        for b in range(len(live)):
            x, y = vals[:, live[a]], vals[:, live[b]]
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() > 2:
                m[a, b] = np.corrcoef(x[ok], y[ok])[0, 1]
    return {"names": names, "matrix": m.round(2).tolist()}


def maximise_gap(c: Corpus, rng, *, restarts: int = 10, steps: int = 300) -> dict:
    """The weighting that makes the reported model gap as large as possible."""
    best, best_w = -1e9, None
    for _ in range(restarts):
        w = rng.dirichlet(np.full(len(COMPONENTS), 0.5))
        cur = instrument_stats(c, w[None, :])["gap"]
        for _ in range(steps):
            step = rng.dirichlet(np.full(len(COMPONENTS), 0.5))
            eta = rng.uniform(0.02, 0.4)
            cand = (1 - eta) * w + eta * step
            cand /= cand.sum()
            val = instrument_stats(c, cand[None, :])["gap"]
            if val is not None and val > cur:
                w, cur = cand, val
        if cur > best:
            best, best_w = cur, w
    stats = instrument_stats(c, best_w[None, :])
    return {
        "weights": {
            COMPONENTS[i]: round(float(best_w[i]), 3) for i in range(len(COMPONENTS))
        },
        **stats,
    }


# --------------------------------------------------------------------------- #
# Validation + report
# --------------------------------------------------------------------------- #


def validate(c: Corpus) -> None:
    """The numpy re-implementation must reproduce the library exactly."""
    lib = {r.crowd_dir: r.score for r in c.runs}
    mine = c.score(SHIPPED())[:, 0]
    diffs = [
        abs(lib[r.crowd_dir] - mine[i])
        for i, r in enumerate(c.runs)
        if lib[r.crowd_dir] is not None and not np.isnan(mine[i])
    ]
    worst = max(diffs)
    if worst > 0.06:  # library rounds to 1dp
        raise SystemExit(f"numpy scorer disagrees with viralscore by {worst:.4f}")
    print(f"<!-- validated against viralscore.score_run: max |diff| = {worst:.4f} -->")


def fmt(value, spec=".1f", missing="--"):
    return missing if value is None else format(value, spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--draws", type=int, default=50000, help="Monte Carlo draws per family"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument(
        "--replicate",
        type=int,
        default=None,
        help="fleet replicate to study (default: whatever CURRENT_FLEET names)",
    )
    ap.add_argument(
        "--arms",
        type=str,
        default="",
        help="comma-separated founder arms to POOL, e.g. solo,team,dynamic",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="comma-separated crowd seeds to keep, e.g. 0,1",
    )
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or None
    if arms:
        replicate = (
            args.replicate if args.replicate is not None else CURRENT_FLEET.replicate
        )
        c = Corpus(specs=specs_for(replicate, arms), seeds=seeds)
        scope = f"replicate {replicate}, arms {'+'.join(arms)}"
        if seeds:
            scope += f", seeds {','.join(str(s) for s in seeds)}"
    else:
        c = Corpus(seeds=seeds)
        scope = f"replicate {CURRENT_FLEET.replicate}, arm {CURRENT_FLEET.structure}"
    print(f"<!-- corpus scope: {scope} -->")
    validate(c)
    results: dict = {"n_runs": len(c.runs), "n_models": len(c.model_ids)}

    print(
        f"\n# ViralScore robustness — {len(c.runs)} runs, {len(c.model_ids)} models\n"
    )

    # --- A ---------------------------------------------------------------- #
    print("## A. Exhaustive bound over the whole weight simplex\n")
    print(f"Complete-case runs: {int(c.complete.sum())} of {len(c.runs)}\n")
    results["dominance"] = {}
    for label, live in (("all 10 components", False), ("live components only", True)):
        bounds = vertex_bound(c, live_only=live)
        dom = dominance(c, bounds, live_only=live)
        results.setdefault("vertex_bounds", {})[label] = bounds
        results["dominance"][label] = {
            "fixed": dom["fixed"],
            "flippable": [(a, b, comp, amt) for a, b, comp, amt in dom["flippable"]],
            "never_first": dom["never_first"],
        }
        n_pairs = len(dom["fixed"]) + len(dom["flippable"])
        print(f"### Simplex over {label}\n")
        if not live:
            print("| model | worst possible | shipped | best possible | best via |")
            print("|---|---|---|---|---|")
            for m in dom["order"]:
                b = bounds[m]
                print(
                    f"| {PRETTY.get(m, m)} | {b['min']:.1f} | {b['shipped']:.1f} "
                    f"| {b['max']:.1f} | {b['argmax']} |"
                )
            print()
        print(
            f"Order-invariant pairs (no weighting can flip): "
            f"**{len(dom['fixed'])} of {n_pairs}**"
        )
        print(
            f"Models no weighting can place first: "
            f"**{len(dom['never_first'])} of {len(dom['order'])}**"
            + (
                " — " + ", ".join(PRETTY.get(m, m) for m in dom["never_first"])
                if dom["never_first"]
                else ""
            )
        )
        if dom["flippable"]:
            byc = {}
            for *_, comp, _amt in dom["flippable"]:
                byc[comp] = byc.get(comp, 0) + 1
            share = ", ".join(
                f"{k} {v}" for k, v in sorted(byc.items(), key=lambda kv: -kv[1])
            )
            print(f"Reorderable pairs by the component that does it: {share}")
        print()

    # --- B ---------------------------------------------------------------- #
    families = {
        "uniform simplex": sample_weights(rng, args.draws, 1.0),
        "sparse / corner-seeking": sample_weights(rng, args.draws, 0.3),
        "live components only": sample_weights(rng, args.draws, 1.0, live_only=True),
    }
    results["monte_carlo"] = {}
    print("## B. Monte Carlo over the simplex\n")
    for name, w in families.items():
        mc = monte_carlo(c, w)
        results["monte_carlo"][name] = mc
        print(f"### {name} (n={mc['n']:,})\n")
        print(
            f"- Kendall tau vs shipped ranking: mean **{mc['tau_mean']:.3f}**, "
            f"5th pct {mc['tau_p05']:.3f}, min {mc['tau_min']:.3f}"
        )
        print(
            f"- Pairwise orderings stable in >=95% of weightings: "
            f"**{mc['pairs_stable_95']} of {mc['pairs_total']}**"
        )
        tv = mc["tier_violations"]
        print(
            f"- Cross-tier orderings obeyed: **{tv['obeyed_rate'] * 100:.2f}%** "
            f"({tv['always_obeyed']} of {tv['cross_tier_pairs']} never violated)\n"
        )
        print("| model | P(rank 1) | median rank | rank range |")
        print("|---|---|---|---|")
        for m in sorted(c.model_ids, key=lambda x: -mc["p_first"][x]):
            lo, hi = mc["rank_range"][m]
            print(
                f"| {PRETTY.get(m, m)} | {mc['p_first'][m] * 100:.2f}% "
                f"| {mc['median_rank'][m]:.0f} | {lo}–{hi} |"
            )
        print()

    # --- C ---------------------------------------------------------------- #
    print("## C. Adversarial search: actively trying to promote each model\n")
    print("| model | best rank achievable | its score there | who still beats it |")
    print("|---|---|---|---|")
    results["adversarial"] = {}
    for m in c.model_ids:
        r = promote(c, m, rng)
        results["adversarial"][m] = r
        winner = (
            "— (it wins)"
            if r["best_rank"] == 1
            else PRETTY.get(r["winner"], r["winner"])
        )
        print(
            f"| {PRETTY.get(m, m)} | {r['best_rank']} | {r['score']:.1f} | {winner} |"
        )
    print()

    # --- D ---------------------------------------------------------------- #
    print("## D. With the validity gate disabled\n")
    ms_gate = c.model_scores(SHIPPED())[:, 0]
    ms_nogate = c.model_scores(SHIPPED(), gate=False)[:, 0]
    order_g = np.argsort(np.argsort(-ms_gate))
    order_n = np.argsort(np.argsort(-ms_nogate))
    results["gate_off"] = {
        c.model_ids[i]: {
            "with_gate": float(ms_gate[i]),
            "no_gate": float(ms_nogate[i]),
            "rank_with": int(order_g[i] + 1),
            "rank_without": int(order_n[i] + 1),
        }
        for i in range(len(c.model_ids))
    }
    print("| model | with gate | gate disabled | rank with | rank without |")
    print("|---|---|---|---|---|")
    for i in sorted(range(len(c.model_ids)), key=lambda i: order_g[i]):
        m = c.model_ids[i]
        print(
            f"| {PRETTY.get(m, m)} | {ms_gate[i]:.1f} | {ms_nogate[i]:.1f} "
            f"| {order_g[i] + 1} | {order_n[i] + 1} |"
        )
    moved = int((order_g != order_n).sum())
    print(
        f"\nModels whose rank changes when the gate is removed: "
        f"**{moved} of {len(c.model_ids)}**\n"
    )

    # --- E ---------------------------------------------------------------- #
    print("## E. How the weights were set\n")
    disc = component_discrimination(c)
    results["discrimination"] = disc
    print("### E1. Per-component discrimination (between-cell SD / within-cell SD)\n")
    print("| component | shipped weight | between SD | within SD | ratio | coverage |")
    print("|---|---|---|---|---|---|")
    for name in sorted(disc, key=lambda n: -(disc[n]["ratio"] or 0)):
        w = SHIPPED()[COMPONENTS.index(name)]
        d = disc[name]
        print(
            f"| {name} | {w:.2f} | {d['between_sd']:.3f} | {d['within_sd']:.3f} "
            f"| **{d['ratio']}** | {d['coverage'] * 100:.0f}% |"
        )
    print()

    corr = component_correlations(c)
    results["correlations"] = corr
    print("### E2. Component correlations (weighted components only)\n")
    print("| | " + " | ".join(corr["names"]) + " |")
    print("|---" * (len(corr["names"]) + 1) + "|")
    for i, name in enumerate(corr["names"]):
        row = " | ".join("--" if np.isnan(v) else f"{v:.2f}" for v in corr["matrix"][i])
        print(f"| **{name}** | {row} |")
    print()

    shipped_stats = instrument_stats(c, SHIPPED()[None, :])
    gapmax = maximise_gap(c, rng)
    results["shipped_stats"], results["gap_max"] = shipped_stats, gapmax
    print("### E3. The weighting that maximises the reported gap\n")
    print("| | reported gap | worst null | seed noise | discrimination |")
    print("|---|---|---|---|---|")
    active = ScoreWeights.from_profile().profile
    print(
        f"| **shipped ({active})** | {fmt(shipped_stats['gap'])} "
        f"| {fmt(shipped_stats['worst_null'], '.2f')} "
        f"| {fmt(shipped_stats['within_sd'], '.2f')} "
        f"| {fmt(shipped_stats['ratio'], '.2f')} |"
    )
    print(
        f"| gap-maximising | {fmt(gapmax['gap'])} | {fmt(gapmax['worst_null'], '.2f')} "
        f"| {fmt(gapmax['within_sd'], '.2f')} | {fmt(gapmax['ratio'], '.2f')} |"
    )
    print(f"\nGap-maximising weights: `{gapmax['weights']}`\n")

    # -- F. flat weights ---------------------------------------------------- #
    # The question this answers is not "which weighting scores best" but "does
    # the weighting need defending at all". If a flat weighting reproduces the
    # tuned one, the tuning is not load-bearing and the simplest defensible
    # scheme is the one to ship.
    # The baseline here is the TUNED profile by name, not SHIPPED(): once
    # v7_equal is active, SHIPPED() is the flat vector and the comparison would
    # be against itself.
    flat = equal_sixths()
    tuned_w = SHIPPED(TUNED_BASELINE)
    tuned_stats = instrument_stats(c, tuned_w[None, :])
    flat_stats = instrument_stats(c, flat[None, :])
    ship_models = c.model_scores(tuned_w[None, :])[:, 0]
    flat_models = c.model_scores(flat[None, :])[:, 0]
    order_ship = np.argsort(-ship_models)
    order_flat = np.argsort(-flat_models)
    moved = sum(
        1
        for rank, idx in enumerate(order_ship)
        if rank != int(np.where(order_flat == idx)[0][0])
    )
    pairs = kept = 0
    for x in range(len(ship_models)):
        for y in range(x + 1, len(ship_models)):
            pairs += 1
            if (ship_models[x] > ship_models[y]) == (flat_models[x] > flat_models[y]):
                kept += 1
    results["equal_weight"] = {
        "stats": flat_stats,
        "ranks_moved": moved,
        "pairs_kept": kept,
        "pairs": pairs,
        "model_scores": {
            c.model_ids[i]: {
                "shipped": float(ship_models[i]),
                "equal": float(flat_models[i]),
            }
            for i in range(len(c.model_ids))
        },
    }
    print("## F. Equal weights -- does the tuning carry any load?\n")
    print("Six components, the autorater counted as one, 1/6 each.\n")
    print("| | reported gap | worst null | seed noise | discrimination |")
    print("|---|---|---|---|---|")
    print(
        f"| tuned ({TUNED_BASELINE}) | {fmt(tuned_stats['gap'])} "
        f"| {fmt(tuned_stats['worst_null'], '.2f')} "
        f"| {fmt(tuned_stats['within_sd'], '.2f')} "
        f"| {fmt(tuned_stats['ratio'], '.2f')} |"
    )
    print(
        f"| equal sixths | {fmt(flat_stats['gap'])} "
        f"| {fmt(flat_stats['worst_null'], '.2f')} "
        f"| {fmt(flat_stats['within_sd'], '.2f')} "
        f"| {fmt(flat_stats['ratio'], '.2f')} |"
    )
    print()
    print("| model | shipped | equal sixths | delta |")
    print("|---|---|---|---|")
    for i in order_flat:
        d = flat_models[i] - ship_models[i]
        print(
            f"| {PRETTY.get(c.model_ids[i], c.model_ids[i])} | {ship_models[i]:.1f} "
            f"| {flat_models[i]:.1f} | {d:+.1f} |"
        )
    print(
        f"\nModel ranks that change: **{moved} of {len(ship_models)}**. "
        f"Pairwise orderings preserved: **{kept} of {pairs}**.\n"
    )

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=str))
        print(f"<!-- raw results written to {args.json} -->")


if __name__ == "__main__":
    main()
