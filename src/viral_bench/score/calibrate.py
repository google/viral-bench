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

"""Measure whether the ViralScore is reliable and discriminating.

A benchmark metric is only worth publishing if you can show it is stable. This
module computes, over a set of apps each scored several times:

* **within-app SD** -- score the same app twice, how far apart are the numbers?
  This is the noise floor, and it is what decides how big a gap between two
  models is real.
* **between-app SD** -- how far apart do different apps land? This is the
  signal.
* **reliability** ``rho = var_between / (var_between + var_within)`` -- the
  fraction of the score's variance that is real differences between apps.
  ``rho >= 0.8`` is the bar for ranking models on a leaderboard.

Used to answer the two questions the plan left open: how many crowd agents a
scored run needs, and whether the weights earn their keep. Because scoring is a
pure function of stored artifacts, re-weighting is free -- only the crowd runs
cost anything.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

from viral_bench.score.signals import extract_signals
from viral_bench.score.viralscore import ScoreWeights, score_run


@dataclass
class AppScores:
    """Every score recorded for one app at one crowd size."""

    label: str
    scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float | None:
        return round(statistics.fmean(self.scores), 2) if self.scores else None

    @property
    def sd(self) -> float | None:
        return round(statistics.stdev(self.scores), 2) if len(self.scores) > 1 else None

    @property
    def spread(self) -> float | None:
        return round(max(self.scores) - min(self.scores), 2) if self.scores else None


@dataclass
class Reliability:
    """The reliability of the score across a set of apps at one crowd size."""

    n_agents: int | None
    apps: dict[str, AppScores]

    @property
    def within_sd(self) -> float | None:
        """Pooled within-app SD: the run-to-run noise floor."""
        sds = [a.sd for a in self.apps.values() if a.sd is not None]
        return round(statistics.fmean(sds), 2) if sds else None

    @property
    def between_sd(self) -> float | None:
        """SD of the per-app means: the signal the benchmark is after."""
        means = [a.mean for a in self.apps.values() if a.mean is not None]
        return round(statistics.stdev(means), 2) if len(means) > 1 else None

    @property
    def rho(self) -> float | None:
        """Reliability in [0,1]. >= 0.8 is publishable for ranking."""
        within, between = self.within_sd, self.between_sd
        if within is None or between is None:
            return None
        denom = between**2 + within**2
        return round(between**2 / denom, 3) if denom else None

    def render(self) -> str:
        head = f"crowd size {self.n_agents if self.n_agents else '?'}"
        lines = [f"=== {head} ==="]
        for label, app in sorted(self.apps.items()):
            lines.append(
                f"  {label:<28} mean={app.mean!s:<7} sd={app.sd!s:<7} "
                f"spread={app.spread!s:<7} runs={len(app.scores)}"
            )
        lines.append(
            f"  -> within-app SD (noise) = {self.within_sd}, "
            f"between-app SD (signal) = {self.between_sd}, reliability rho = {self.rho}"
        )
        return "\n".join(lines)


def score_dirs(
    dirs: list[str | Path], weights: ScoreWeights | None = None
) -> list[tuple[str, float]]:
    """Score each crowd run directory, skipping unscorable runs."""
    out: list[tuple[str, float]] = []
    for d in dirs:
        try:
            result = score_run(extract_signals(d), weights)
        except (FileNotFoundError, ValueError):
            continue
        if result.score is not None:
            out.append((result.build_id, result.score))
    return out


def reliability(
    runs_by_app: dict[str, list[str | Path]],
    *,
    n_agents: int | None = None,
    weights: ScoreWeights | None = None,
) -> Reliability:
    """Compute reliability from ``{app_label: [crowd_run_dir, ...]}``."""
    apps: dict[str, AppScores] = {}
    for label, dirs in runs_by_app.items():
        scores = [s for _, s in score_dirs(dirs, weights)]
        if scores:
            apps[label] = AppScores(label=label, scores=scores)
    return Reliability(n_agents=n_agents, apps=apps)


def component_reliability(
    runs_by_app: dict[str, list[str | Path]],
) -> dict[str, dict]:
    """Per-component within/between spread -- which parts carry signal vs noise.

    The composite can only be as stable as the components feeding it, so this
    says *where* instability comes from and whether a component earns its weight.
    A component whose between-app spread is small relative to its run-to-run
    spread is contributing noise to the score, not information.
    """
    per_component: dict[str, dict[str, list[float]]] = {}
    for label, dirs in runs_by_app.items():
        for d in dirs:
            try:
                result = score_run(extract_signals(d))
            except (FileNotFoundError, ValueError):
                continue
            for name, value in result.components.items():
                if value is None:
                    continue
                per_component.setdefault(name, {}).setdefault(label, []).append(value)

    out: dict[str, dict] = {}
    for name, by_app in per_component.items():
        withins = [statistics.stdev(vals) for vals in by_app.values() if len(vals) > 1]
        means = [statistics.fmean(vals) for vals in by_app.values() if vals]
        within = statistics.fmean(withins) if withins else None
        between = statistics.stdev(means) if len(means) > 1 else None
        out[name] = {
            "within_sd": round(within, 4) if within is not None else None,
            "between_sd": round(between, 4) if between is not None else None,
            "signal_to_noise": (
                round(between / within, 2)
                if within and between is not None and within > 0
                else None
            ),
            "per_app_mean": {
                k: round(statistics.fmean(v), 3) for k, v in by_app.items()
            },
        }
    return out


def compare_weightings(
    runs_by_app: dict[str, list[str | Path]],
    candidates: dict[str, ScoreWeights],
) -> dict[str, dict]:
    """Score the same stored runs under several weightings and compare rho.

    Free: nothing is re-simulated, only re-scored. This is what lets the weights
    be chosen from evidence rather than taste -- though maximising rho alone
    would drift the metric toward whatever is easiest to measure, so a candidate
    must also remain a defensible definition of *virality*.
    """
    out: dict[str, dict] = {}
    for name, weights in candidates.items():
        rel = reliability(runs_by_app, weights=weights)
        out[name] = {
            "rho": rel.rho,
            "within_sd": rel.within_sd,
            "between_sd": rel.between_sd,
            "app_means": {k: v.mean for k, v in rel.apps.items()},
        }
    return out


def group_runs_by_agents(
    runs_by_app: dict[str, list[str | Path]],
) -> dict[int, dict[str, list[str | Path]]]:
    """Regroup runs by the crowd size each was run at."""
    grouped: dict[int, dict[str, list[str | Path]]] = {}
    for label, dirs in runs_by_app.items():
        for d in dirs:
            try:
                sig = extract_signals(d)
            except (FileNotFoundError, ValueError):
                continue
            grouped.setdefault(sig.actual_agents, {}).setdefault(label, []).append(d)
    return grouped
