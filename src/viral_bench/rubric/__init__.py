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

"""The RubricScore: a second, independent scoring track.

The ViralScore asks *would simulated users adopt this app*. This asks the
question a benchmark normally asks -- *does it do what the brief said* -- so the
two can be compared. They are deliberately independent: no rubric result is an
input to any ViralScore component, and nothing here imports :mod:`viral_bench.score`.

The design decision that makes this a benchmark rather than a second opinion is
**the model gathers, the code judges**. An item either carries a ``check``
block, in which case a deterministic primitive in :mod:`~viral_bench.rubric.checks`
decides pass/fail and the grader model's only job is navigating the app into the
state the check needs. Or it does not, in which case the model judges and must
cite the id of a tool call, and the harness reads the value out of *its own*
record of that call rather than out of the model's prose. Across the 25 shipped
rubrics that puts ~91% of the available points beyond the model's discretion.

Grading needs a live app, so unlike :mod:`viral_bench.score` it is not a pure
function of stored artifacts and re-grading is not free. The per-item verdicts
are therefore persisted in full, so the *aggregation* stays re-runnable even
though the grading is not.
"""

from __future__ import annotations

#: Bumped whenever the scoring formula, the tier structure or the universal
#: sections change, so a comparison never mixes two definitions. Distinct from
#: the ``rubric_version`` inside each rubric file, which versions the items.
RUBRIC_SCORE_VERSION = "1.0"

__all__ = ["RUBRIC_SCORE_VERSION"]
