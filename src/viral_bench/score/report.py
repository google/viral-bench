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

"""Render and persist a ViralScore.

Writes ``score.json`` next to the crowd run (stamped with ``score_version`` so a
leaderboard never mixes numbers from different formula versions) and renders a
human-readable breakdown that shows *why* an app scored what it did -- the
component contributions, the facet diagnostics, and anything that makes the run
less trustworthy.
"""

from __future__ import annotations

import json
from pathlib import Path

from viral_bench.score.viralscore import ViralScoreResult

SCORE_FILENAME = "score.json"


def write_score(result: ViralScoreResult, crowd_dir: str | Path | None = None) -> Path:
    """Persist ``score.json`` into the crowd run directory; return its path."""
    target = Path(crowd_dir or result.crowd_dir) / SCORE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return target


def _bar(value: float | None, width: int = 20) -> str:
    if value is None:
        return "-" * width
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def render_score(result: ViralScoreResult) -> str:
    """A readable breakdown of one scored run."""
    lines: list[str] = []
    head = f"ViralScore: {result.score if result.scorable else 'UNSCORABLE'}"
    if result.scorable:
        head += " / 100"
        if result.ci_low is not None:
            head += f"   (95% CI {result.ci_low}-{result.ci_high})"
    lines.append(head)
    lines.append(f"build: {result.build_id}")
    lines.append(f"score_version: {result.score_version}")
    if result.gate != 1.0:
        lines.append(
            f"!! VALIDITY GATE APPLIED (x{result.gate}): the app does not do what "
            "it claims"
        )
    lines.append("")

    lines.append("components (0-1, weighted):")
    for name, weight in result.weights.items():
        value = result.components.get(name)
        shown = "unmeasured" if value is None else f"{value:.3f}"
        note = ""
        if weight == 0:
            note = "  [measured, unweighted in this version]"
        contrib = (
            f"{100 * weight * value:5.1f} pts"
            if value is not None and weight
            else "  -"
        )
        lines.append(
            f"  {name:<15} w={weight:<5.2f} {shown:>10}  {_bar(value)} {contrib}{note}"
        )

    diag = result.diagnostics
    lines.append("")
    lines.append("evidence:")
    lines.append(
        f"  interviews: {diag.get('n_interviews')} agents"
        f" | hands-on trials: {diag.get('n_valid_trials')} valid"
        f" | exposed: {diag.get('exposed_agents')}"
    )
    if diag.get("facets"):
        facets = "  ".join(f"{k}={v}" for k, v in sorted(diag["facets"].items()))
        lines.append(f"  craft facets (0-10): {facets}")
    if diag.get("audience_fit_rate") is not None:
        lines.append(
            f"  audience fit: {diag['audience_fit_rate']:.0%} say it's aimed at them"
            f" | in-audience adoption: {diag.get('resonance_adoption')}"
            f" advocacy: {diag.get('resonance_advocacy')}"
        )
    if diag.get("delight_mean") is not None:
        lines.append(
            f"  crowd delight: mean {diag['delight_mean']} "
            f"(sd {diag.get('delight_stdev')})"
        )

    if result.confidence:
        lines.append("")
        lines.append("confidence warnings:")
        lines += [f"  - {w}" for w in result.confidence]
    return "\n".join(lines)
