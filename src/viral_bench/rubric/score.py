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

"""Turn per-item verdicts into a RubricScore.

The arithmetic, in full::

    if any Tier 0 gate item failed:   score = 0
    base      = 100 * points_earned / points_applicable
    penalties = max(PENALTY_CAP, sum of fired penalties)
    score     = max(0, base + penalties)

Three properties are deliberate.

**The gate zeroes rather than deducts.** A unit-test suite does not award
partial credit when the binary fails to build, and ~27% of the corpus is
undeliverable or dead on arrival. Awarding those builds feature points for code
nobody can execute is the easiest way to make the whole track non-credible. Gate
item results are still recorded individually, for diagnosis.

**Scoring normalises over *applicable* points.** An idea whose brief never asks
for persistence marks the universal R1 not-applicable, so its denominator is 92
rather than 100. Normalising keeps that idea comparable with the others instead
of capping it at 92.

**A verdict of None is a failure, not a skip.** "Could not determine" scores
zero, and is counted separately from "checked and failed" so instrument faults
stay visible in :attr:`RubricResult.unresolved` rather than hiding inside the
score. This is the same rule the crowd work arrived at the hard way.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import median

from viral_bench.rubric import RUBRIC_SCORE_VERSION
from viral_bench.rubric.schema import PENALTY_CAP, TIER_LABELS, Rubric, RubricItem


@dataclass
class ItemVerdict:
    """One item's outcome, across every grading pass.

    ``passes`` holds one entry per pass: True, False, or None where the pass
    could not resolve it. ``passed`` is the majority verdict. ``observed`` and
    ``expected`` are kept as strings so the report renders them side by side
    without the viewer needing to know the primitive's return type.
    """

    item_id: str
    passes: list[bool | None] = field(default_factory=list)
    reason: str = ""
    expected: str = ""
    observed: str = ""
    evidence: list[str] = field(default_factory=list)
    #: The harness overruled the model's stated verdict for this item. A high
    #: rate means the model is not navigating reliably and the transcript should
    #: not be trusted even where the code decides.
    harness_override: bool = False

    @property
    def passed(self) -> bool:
        """Majority verdict. Ties and all-unresolved resolve to False."""
        yes = sum(1 for value in self.passes if value is True)
        return yes * 2 > len(self.passes) if self.passes else False

    @property
    def unresolved(self) -> bool:
        """No pass could determine an answer -- an instrument fault, not a fail."""
        return bool(self.passes) and all(value is None for value in self.passes)

    @property
    def disagreement(self) -> bool:
        return len({value for value in self.passes}) > 1


def _fired_total(item: RubricItem, verdict: ItemVerdict) -> int:
    """Points a fired penalty contributes, respecting its own ``max_total``."""
    if not verdict.passed:
        return 0
    if item.max_total is not None:
        return max(item.max_total, item.points)
    return item.points


@dataclass
class RubricResult:
    """A graded build: the number, how it was built, and how much to trust it."""

    build_id: str
    idea_id: str
    rubric_version: str
    score_version: str = RUBRIC_SCORE_VERSION
    score: float = 0.0
    points_earned: int = 0
    points_applicable: int = 0
    base: float = 0.0
    penalty_total: int = 0
    penalty_capped: bool = False
    floor_applied: bool = False
    gate_zeroed: bool = False
    gate_failures: list[str] = field(default_factory=list)
    verdicts: dict[str, ItemVerdict] = field(default_factory=dict)
    not_applicable: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    passes: int = 1
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def reliability(self) -> dict:
        """The track's own noise floor, split by how each item is decided.

        Reported per grade and aggregated over a sweep. Without it, comparing a
        RubricScore gap against a ViralScore gap is not interpretable: a gap
        smaller than the instrument's own disagreement is not a finding.
        """
        by_method: dict[str, list[bool]] = {}
        overrides = 0
        for item_id, verdict in self.verdicts.items():
            method = self._methods.get(item_id, "agent")
            by_method.setdefault(method, []).append(verdict.disagreement)
            overrides += 1 if verdict.harness_override else 0
        total = sum(len(values) for values in by_method.values())
        rates = {
            method: round(sum(values) / len(values), 4)
            for method, values in sorted(by_method.items())
            if values
        }
        code = [
            flag
            for method, values in by_method.items()
            if method in ("probe", "assert")
            for flag in values
        ]
        agent = [
            flag
            for method, values in by_method.items()
            if method not in ("probe", "assert")
            for flag in values
        ]
        return {
            "items_total": total,
            "items_disagreeing": sum(
                1 for verdict in self.verdicts.values() if verdict.disagreement
            ),
            "by_method": rates,
            "code_disagreement": round(sum(code) / len(code), 4) if code else 0.0,
            "agent_disagreement": round(sum(agent) / len(agent), 4) if agent else 0.0,
            "override_rate": round(overrides / total, 4) if total else 0.0,
            "unresolved": len(self.unresolved),
        }

    #: item_id -> method, populated by :func:`score_rubric` so ``reliability``
    #: can split by how each item was decided without re-reading the rubric.
    _methods: dict[str, str] = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict:
        data = asdict(self)
        data.pop("_methods", None)
        data["verdicts"] = {
            item_id: asdict(verdict)
            | {
                "passed": verdict.passed,
                "disagreement": verdict.disagreement,
                "unresolved": verdict.unresolved,
            }
            for item_id, verdict in self.verdicts.items()
        }
        data["ok"] = self.ok
        data["reliability"] = self.reliability()
        return data


def score_rubric(
    rubric: Rubric,
    verdicts: dict[str, ItemVerdict],
    *,
    build_id: str,
    passes: int = 1,
    error: str = "",
) -> RubricResult:
    """Aggregate per-item verdicts into a :class:`RubricResult`.

    A pure function of the verdicts, so re-aggregating a stored grade is free
    even though re-grading is not.
    """
    result = RubricResult(
        build_id=build_id,
        idea_id=rubric.idea_id,
        rubric_version=rubric.rubric_version,
        verdicts=verdicts,
        not_applicable=dict(rubric.not_applicable),
        passes=passes,
        error=error,
    )
    result._methods = {item.id: item.method for item in rubric.items_by_id().values()}

    # The gate first: it decides whether anything else counts.
    for item in rubric.gate:
        verdict = verdicts.get(item.id)
        if verdict is None or not verdict.passed:
            result.gate_failures.append(item.id)

    for item in rubric.scored_items:
        verdict = verdicts.get(item.id)
        result.points_applicable += item.points
        if verdict is None:
            result.unresolved.append(item.id)
            continue
        if verdict.unresolved:
            result.unresolved.append(item.id)
        if verdict.passed:
            result.points_earned += item.points

    for item in rubric.penalties:
        verdict = verdicts.get(item.id)
        if verdict is not None:
            result.penalty_total += _fired_total(item, verdict)

    if result.penalty_total < PENALTY_CAP:
        result.penalty_total = PENALTY_CAP
        result.penalty_capped = True

    if result.points_applicable:
        result.base = round(100.0 * result.points_earned / result.points_applicable, 1)

    if result.gate_failures:
        result.gate_zeroed = True
        result.score = 0.0
        return result

    raw = result.base + result.penalty_total
    result.floor_applied = raw < 0
    result.score = round(max(0.0, raw), 1)
    return result


def tier_breakdown(rubric: Rubric, result: RubricResult) -> list[dict]:
    """Per-tier earned/available, for the report and the viewer.

    Kept here rather than in the renderer so the numbers in the UI can never
    disagree with the numbers in the score.
    """
    tiers = []
    for number, items in ((1, rubric.tier1), (2, rubric.tier2), (3, rubric.tier3)):
        rows = []
        earned = 0
        for item in items:
            verdict = result.verdicts.get(item.id)
            passed = bool(verdict and verdict.passed)
            earned += item.points if passed else 0
            rows.append(
                {
                    "id": item.id,
                    "text": item.text,
                    "points": item.points,
                    "earned": item.points if passed else 0,
                    "method": item.method,
                    "expect": item.expect,
                    "note": item.note,
                    "passed": passed,
                    # No verdict at all means the item was never observed, which
                    # is the definition of unresolved and not of failure. The
                    # gate is where this bites: a failed Tier 0 stops the run
                    # before a single item is graded, and rendering those rows as
                    # FAIL turned one manifest typo into twenty-seven defects in
                    # the report -- precisely the collapse the three-verdict
                    # split exists to prevent. The stored `unresolved` list had
                    # them right the whole time; only the per-item rows disagreed.
                    "unresolved": bool(verdict.unresolved) if verdict else True,
                    "disagreement": bool(verdict and verdict.disagreement),
                    "harness_override": bool(verdict and verdict.harness_override),
                    "observed": verdict.observed if verdict else "",
                    "reason": verdict.reason if verdict else "",
                    "evidence": list(verdict.evidence) if verdict else [],
                    "passes": list(verdict.passes) if verdict else [],
                }
            )
        tiers.append(
            {
                "tier": number,
                "label": TIER_LABELS[number],
                "points": sum(item.points for item in items),
                "earned": earned,
                "items": rows,
            }
        )
    return tiers


def median_score(scores: list[float]) -> float | None:
    """Median of several grades of the same build. None when there are none."""
    values = [value for value in scores if value is not None]
    return round(median(values), 1) if values else None
