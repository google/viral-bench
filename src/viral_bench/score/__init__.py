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

"""Stage 4: turn a crowd run into a quantitative ViralScore.

The crowd simulation produces the raw social signal, and this package turns it into
one comparable 0-100 number per built app, which is what the benchmark's
leaderboard ranks frontier models by.

Scoring is a **pure, offline function of persisted artifacts** and is stamped
with a ``score_version``. Nothing here re-runs a simulation, so changing the
formula means re-scoring stored runs rather than paying for the crowd again --
which is what makes calibrating the weights affordable.
"""

from viral_bench.score.discriminate import (
    ModelScores,
    Redundancy,
    Separation,
    component_separation,
    redundancy_panel,
    score_runs_by_model,
    separation,
    sweep_profiles,
)
from viral_bench.score.evidence import build_evidence_pack, write_evidence_pack
from viral_bench.score.report import render_score, write_score
from viral_bench.score.signals import RunSignals, extract_signals, find_crowd_runs
from viral_bench.score.spread import (
    SPREAD_WEIGHTS,
    SpreadSignals,
    extract_spread,
    spread_score,
)
from viral_bench.score.viralscore import (
    SCORE_VERSION,
    ScoreWeights,
    ViralScoreResult,
    compute_components,
    score_run,
)

__all__ = [
    "SCORE_VERSION",
    "SPREAD_WEIGHTS",
    "ModelScores",
    "Redundancy",
    "Separation",
    "SpreadSignals",
    "build_evidence_pack",
    "component_separation",
    "extract_spread",
    "redundancy_panel",
    "score_runs_by_model",
    "separation",
    "spread_score",
    "sweep_profiles",
    "write_evidence_pack",
    "RunSignals",
    "ScoreWeights",
    "ViralScoreResult",
    "compute_components",
    "extract_signals",
    "find_crowd_runs",
    "render_score",
    "score_run",
    "write_score",
]
