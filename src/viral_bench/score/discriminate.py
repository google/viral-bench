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

"""Measure how well the ViralScore separates founder models.

This is the benchmark's actual purpose. Reliability says the score is stable;
*discrimination* says it can tell one founder model from another, which is the
signal the Gemini team would act on. A perfectly stable score that gives every
model 60 is useless.

The headline statistic is **Cohen's d**, the gap between two models' mean scores
in units of their pooled run-to-run spread:

    d = (mean_A - mean_B) / pooled_sd

Rules of thumb: 0.2 small, 0.5 medium, 0.8 large. d is the right measure here
because it answers the question a leaderboard needs -- "is this gap bigger than
the noise?" -- rather than "is the gap non-zero", which any large enough sample
will eventually say yes to.

**On finding a scoring configuration that separates.** Weights are adjustable so
you can find the factors that genuinely separate models, and re-scoring stored
runs is free. :func:`sweep_profiles` re-scores the whole corpus under every
profile in ``config/score.yaml`` at once and ranks them by effect size, so the
question "which scoring configuration makes these two models look most
different" is one command over the entire test set.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from viral_bench.score.signals import extract_signals
from viral_bench.score.viralscore import ScoreWeights, score_run


@dataclass
class ModelScores:
    """Every scored run for one founder model, grouped by idea."""

    model: str
    by_idea: dict[str, list[float]] = field(default_factory=dict)

    def add(self, idea: str, score: float) -> None:
        self.by_idea.setdefault(idea, []).append(score)

    @property
    def all_scores(self) -> list[float]:
        return [s for scores in self.by_idea.values() for s in scores]

    @property
    def mean(self) -> float | None:
        vals = self.all_scores
        return round(statistics.fmean(vals), 2) if vals else None

    @property
    def sd(self) -> float | None:
        vals = self.all_scores
        return round(statistics.stdev(vals), 2) if len(vals) > 1 else None


@dataclass
class Separation:
    """How far apart two models are, relative to the noise."""

    model_a: str
    model_b: str
    mean_a: float | None
    mean_b: float | None
    gap: float | None
    pooled_sd: float | None
    cohens_d: float | None
    n_ideas_shared: int = 0
    paired_gap: float | None = None

    @property
    def verdict(self) -> str:
        d = abs(self.cohens_d or 0.0)
        if d >= 0.8:
            return "large"
        if d >= 0.5:
            return "medium"
        if d >= 0.2:
            return "small"
        return "negligible"

    def render(self) -> str:
        return (
            f"{self.model_a} ({self.mean_a}) vs {self.model_b} ({self.mean_b}): "
            f"gap={self.gap} pooled_sd={self.pooled_sd} d={self.cohens_d} "
            f"[{self.verdict}]"
            + (
                f" | paired over {self.n_ideas_shared} shared ideas: {self.paired_gap}"
                if self.paired_gap is not None
                else ""
            )
        )


def _pooled_sd(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    return pooled**0.5 if pooled > 0 else None


def separation(a: ModelScores, b: ModelScores) -> Separation:
    """Effect size between two founder models."""
    sa, sb = a.all_scores, b.all_scores
    mean_a = statistics.fmean(sa) if sa else None
    mean_b = statistics.fmean(sb) if sb else None
    gap = (mean_a - mean_b) if (mean_a is not None and mean_b is not None) else None
    pooled = _pooled_sd(sa, sb)
    d = (gap / pooled) if (gap is not None and pooled) else None

    # Paired comparison over ideas both models built. Idea difficulty is the
    # largest nuisance factor -- a hard idea drags every model down -- so
    # differencing within an idea removes it and is the more sensitive test.
    shared = sorted(set(a.by_idea) & set(b.by_idea))
    paired = None
    if shared:
        diffs = [
            statistics.fmean(a.by_idea[i]) - statistics.fmean(b.by_idea[i])
            for i in shared
        ]
        paired = round(statistics.fmean(diffs), 2)

    return Separation(
        model_a=a.model,
        model_b=b.model,
        mean_a=round(mean_a, 2) if mean_a is not None else None,
        mean_b=round(mean_b, 2) if mean_b is not None else None,
        gap=round(gap, 2) if gap is not None else None,
        pooled_sd=round(pooled, 2) if pooled else None,
        cohens_d=round(d, 2) if d is not None else None,
        n_ideas_shared=len(shared),
        paired_gap=paired,
    )


def score_runs_by_model(
    runs: dict[str, list[str | Path]],
    weights: ScoreWeights | None = None,
    ratings: dict[str, object] | None = None,
) -> dict[str, ModelScores]:
    """Score runs grouped as ``{founder_model: [crowd_run_dir, ...]}``.

    Unscorable runs are skipped rather than counted as zero -- a run whose
    evidence was lost says nothing about the model that built the app.
    """
    ratings = ratings or {}
    out: dict[str, ModelScores] = {}
    for model, dirs in runs.items():
        bucket = ModelScores(model=model)
        for d in dirs:
            try:
                sig = extract_signals(d)
            except (FileNotFoundError, ValueError):
                continue
            result = score_run(sig, weights, ratings.get(str(d)))
            if result.score is None:
                continue
            idea = sig.build_id.split("__")[0] if sig.build_id else str(d)
            bucket.add(idea, result.score)
        if bucket.all_scores:
            out[model] = bucket
    return out


def component_separation(
    runs: dict[str, list[str | Path]],
    ratings: dict[str, object] | None = None,
) -> dict[str, dict]:
    """Per-component effect size: which factors actually separate models?

    This is the dashboard for the search. A component with a large |d| carries
    signal about model quality and deserves weight; one near zero is dead weight
    in the composite no matter how intuitive it sounds.

    Pass ``ratings`` to include the autorater's dimensions. Without them only
    the deterministic components appear, which would hide whether a change to
    the autorater bought any separation at all.
    """
    ratings = ratings or {}
    per_component: dict[str, dict[str, list[float]]] = {}
    for model, dirs in runs.items():
        for d in dirs:
            try:
                result = score_run(extract_signals(d), None, ratings.get(str(d)))
            except (FileNotFoundError, ValueError):
                continue
            for name, value in result.components.items():
                if value is not None:
                    per_component.setdefault(name, {}).setdefault(model, []).append(
                        value
                    )

    out: dict[str, dict] = {}
    for name, by_model in per_component.items():
        best = None
        for m1, m2 in itertools.combinations(sorted(by_model), 2):
            a, b = by_model[m1], by_model[m2]
            pooled = _pooled_sd(a, b)
            if not pooled:
                continue
            d = (statistics.fmean(a) - statistics.fmean(b)) / pooled
            if best is None or abs(d) > abs(best):
                best = d
        out[name] = {
            "per_model_mean": {
                m: round(statistics.fmean(v), 3) for m, v in by_model.items()
            },
            "max_abs_cohens_d": round(abs(best), 2) if best is not None else None,
        }
    return out


def _rank(values: list[float]) -> list[float]:
    """Ranks with ties averaged, so a plateau does not fake an ordering."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float | None:
    # strict: these are two readings of the SAME runs in the same order, so a
    # length mismatch means the caller mispaired them, and every number below
    # would then be a plausible lie rather than an error.
    pairs = [
        (x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None
    ]
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if not sx or not sy:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (sx * sy)


def _spearman(a: list[float], b: list[float]) -> float | None:
    pairs = [
        (x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None
    ]
    if len(pairs) < 3:
        return None
    return _pearson(
        _rank([p[0] for p in pairs]),
        _rank([p[1] for p in pairs]),
    )


def _composite(
    components: dict,
    weights: ScoreWeights,
    gate: float,
    *,
    keep: set[str] | None = None,
    drop: frozenset[str] = frozenset(),
) -> float | None:
    """The score recomputed from a subset of components, renormalised.

    Mirrors :func:`viral_bench.score.viralscore._weighted` exactly, including
    the renormalisation over measured components, so a keep-one number is
    directly comparable to the full score rather than to a differently-scaled
    thing.
    """
    w = weights.as_dict()
    pairs = [
        (components[k], w[k])
        for k in w
        if components.get(k) is not None
        and w[k] > 0
        and k not in drop
        and (keep is None or k in keep)
    ]
    total = sum(weight for _, weight in pairs)
    if not pairs or total <= 0:
        return None
    return 100.0 * gate * sum(v * weight for v, weight in pairs) / total


@dataclass
class Redundancy:
    """How much independent information the components actually carry.

    A composite of N components is only worth N components if removing one
    changes the answer. When every term alone reproduces the ranking, the score
    is one latent factor measured N times, and tuning the weights between them
    cannot move anything -- which makes a weight sweep look like it is
    converging when it is really just reporting noise.
    """

    n_runs: int
    n_excluded_gated: int
    components: list[str]
    correlations: dict[str, dict[str, float | None]]
    keep_one: dict[str, float | None]
    drop_one: dict[str, float | None]

    @property
    def worst_keep_one(self) -> tuple[str, float] | None:
        """The single component that best impersonates the whole score."""
        rated = [(k, v) for k, v in self.keep_one.items() if v is not None]
        return max(rated, key=lambda kv: kv[1]) if rated else None

    def render(self) -> str:
        out: list[str] = []
        out.append(
            f"REDUNDANCY over {self.n_runs} runs "
            f"({self.n_excluded_gated} gated runs excluded)"
        )
        if self.n_runs < 3:
            out.append("  too few runs to measure")
            return "\n".join(out)

        out.append("")
        out.append("  keep ONE component, discard the rest (Spearman vs full score):")
        for name in self.components:
            value = self.keep_one.get(name)
            out.append(
                f"    {name:<16} {'n/a' if value is None else format(value, '.4f')}"
            )
        worst = self.worst_keep_one
        if worst:
            out.append(
                f"    -> `{worst[0]}` alone reproduces the ranking at "
                f"rho={worst[1]:.4f}"
            )

        out.append("")
        out.append("  drop ONE component (Spearman vs full score):")
        for name in self.components:
            value = self.drop_one.get(name)
            out.append(
                f"    {name:<16} {'n/a' if value is None else format(value, '.4f')}"
            )

        out.append("")
        out.append("  component correlations (Pearson):")
        out.append("    " + " " * 16 + "".join(f"{n[:6]:>8}" for n in self.components))
        for row in self.components:
            cells = ""
            for col in self.components:
                value = self.correlations.get(row, {}).get(col)
                cells += f"{'':>8}" if value is None else f"{value:>8.2f}"
            out.append(f"    {row:<16}{cells}")
        return "\n".join(out)


def redundancy_panel(
    dirs: list[str | Path],
    weights: ScoreWeights | None = None,
    ratings: dict[str, object] | None = None,
    *,
    clean_only: bool = True,
) -> Redundancy:
    """Measure whether the components are N measurements or one, N times.

    Three views of the same question:

    * **keep-one** -- score each run on a single component and rank-correlate
      against the full score. A component near 1.0 *is* the score.
    * **drop-one** -- remove one component and re-rank. Near 1.0 means the
      component is buying nothing.
    * **correlations** -- which pairs are the same measurement wearing two
      names.

    ``clean_only`` (default on) drops runs whose validity gate fired. This is
    not cosmetic: the gate is a common multiplier applied to every component at
    once, so on a corpus where 18% of runs are gated it manufactures agreement
    between terms that are otherwise unrelated. Measured on the stored corpus
    the keep-one figures barely move (0.93-0.97 gated-in vs 0.90-0.97 clean),
    which is the point -- the redundancy is real and not a gate artifact, and
    the clean cut is what proves it rather than assuming it.
    """
    weights = weights or ScoreWeights.from_profile()
    ratings = ratings or {}

    per_run: list[tuple[dict, float]] = []
    n_gated = 0
    for d in dirs:
        try:
            result = score_run(extract_signals(d), weights, ratings.get(str(d)))
        except (FileNotFoundError, ValueError):
            continue
        if result.score is None:
            continue
        if clean_only and result.gate != 1.0:
            n_gated += 1
            continue
        per_run.append((result.components, result.gate))

    weighted = weights.as_dict()
    names = [
        n
        for n in weighted
        if weighted[n] > 0 and any(comp.get(n) is not None for comp, _ in per_run)
    ]

    full = [_composite(comp, weights, gate) for comp, gate in per_run]
    keep_one = {
        n: _spearman(full, [_composite(c, weights, g, keep={n}) for c, g in per_run])
        for n in names
    }
    drop_one = {
        n: _spearman(
            full, [_composite(c, weights, g, drop=frozenset({n})) for c, g in per_run]
        )
        for n in names
    }
    columns = {n: [comp.get(n) for comp, _ in per_run] for n in names}
    correlations = {
        row: {col: _pearson(columns[row], columns[col]) for col in names}
        for row in names
    }

    return Redundancy(
        n_runs=len(per_run),
        n_excluded_gated=n_gated,
        components=names,
        correlations=correlations,
        keep_one=keep_one,
        drop_one=drop_one,
    )


def sweep_profiles(
    runs: dict[str, list[str | Path]],
    profiles: list[str] | dict[str, ScoreWeights] | None = None,
    ratings: dict[str, object] | None = None,
) -> list[dict]:
    """Re-score the whole corpus under every weight profile, ranked by separation.

    This is the scoring-configuration search: which combination of weights makes
    the founder models look most different, measured over every run supplied.
    Re-scoring is a pure function of stored artifacts, so sweeping the full
    profile list costs nothing but CPU.

    ``profiles`` defaults to every profile in ``config/score.yaml``. Pass a list
    of names to restrict it, or a ``{name: ScoreWeights}`` mapping to try
    candidate weightings that are not in the config at all.

    Returns one row per profile, sorted by the largest absolute effect size it
    achieves over any model pair. A profile whose weights are malformed (they
    must sum to 1.0) yields a row carrying its ``error`` rather than aborting
    the sweep -- profiles are hand-edited between iterations and one typo must
    not cost the whole comparison.
    """
    from viral_bench import config as _config

    if profiles is None:
        profiles = _config.score_profile_names()
    explicit = profiles if isinstance(profiles, dict) else None

    rows: list[dict] = []
    for name in profiles:
        row: dict = {"profile": name, "separations": [], "max_abs_cohens_d": None}
        try:
            weights = (
                explicit[name]
                if explicit is not None
                else ScoreWeights.from_profile(name)
            )
            weights.validate()
            scored = score_runs_by_model(runs, weights, ratings)
        except (ValueError, TypeError) as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue

        row["weights"] = weights.as_dict()
        row["means"] = {m: s.mean for m, s in scored.items()}
        best = None
        for a, b in itertools.combinations(sorted(scored), 2):
            sep = separation(scored[a], scored[b])
            row["separations"].append(sep.__dict__)
            d = sep.cohens_d
            if d is not None and (best is None or abs(d) > abs(best)):
                best = d
        row["max_abs_cohens_d"] = round(abs(best), 2) if best is not None else None
        rows.append(row)

    # Best separator first; profiles that produced no number sink to the bottom
    # rather than being dropped, so a broken profile stays visible.
    rows.sort(
        key=lambda r: (r["max_abs_cohens_d"] is None, -(r["max_abs_cohens_d"] or 0))
    )
    return rows
