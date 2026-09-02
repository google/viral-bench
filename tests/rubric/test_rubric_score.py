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

"""Tests for the RubricScore arithmetic.

The scoring function is pure, so everything here runs offline with fabricated
verdicts. The cases that matter are the ones where a plausible-looking
implementation would be wrong: a failed gate zeroing the score, a not-applicable
item leaving the denominator, an unresolved item scoring zero rather than being
skipped, and the penalty cap.
"""

from __future__ import annotations

import pytest

from viral_bench.rubric.schema import PENALTY_CAP, Rubric, RubricItem
from viral_bench.rubric.score import ItemVerdict, score_rubric, tier_breakdown


def item(item_id, points, tier, method="assert", **kwargs):
    return RubricItem(
        id=item_id,
        text=f"{item_id} text",
        points=points,
        method=method,
        tier=tier,
        **kwargs,
    )


def make_rubric(*, not_applicable=None, penalties=None):
    """A rubric summing to the real totals: 40 / 25 / 35."""
    return Rubric(
        idea_id="demo",
        rubric_version="1",
        tier1=(item("S1", 25, 1), item("S2", 15, 1)),
        tier2=(item("F1", 25, 2, method="probe"),),
        tier3=() if not_applicable else (item("R1", 35, 3),),
        gate=(item("G1", 0, 0, method="probe"),),
        penalties=tuple(penalties or (item("A1", -8, -1),)),
        not_applicable=not_applicable or {},
    )


def verdicts(**flags):
    return {
        key: ItemVerdict(item_id=key, passes=[value]) for key, value in flags.items()
    }


def test_a_perfect_build_scores_100():
    rubric = make_rubric()
    result = score_rubric(
        rubric,
        verdicts(G1=True, S1=True, S2=True, F1=True, R1=True, A1=False),
        build_id="b",
        passes=1,
    )
    assert result.points_earned == 100
    assert result.points_applicable == 100
    assert result.score == 100.0
    assert not result.gate_zeroed


def test_a_failed_gate_zeroes_everything_and_records_why():
    rubric = make_rubric()
    result = score_rubric(
        rubric,
        verdicts(G1=False, S1=True, S2=True, F1=True, R1=True),
        build_id="b",
        passes=1,
    )
    assert result.score == 0.0
    assert result.gate_zeroed
    assert result.gate_failures == ["G1"]
    # The earned points are still recorded, so a diagnosis is possible.
    assert result.points_earned == 100


def test_a_missing_gate_verdict_fails_the_gate():
    """Missing evidence is FAIL, never PASS -- including for the gate itself."""
    rubric = make_rubric()
    result = score_rubric(rubric, verdicts(S1=True), build_id="b")
    assert result.gate_zeroed and result.gate_failures == ["G1"]


def test_not_applicable_leaves_the_denominator():
    rubric = make_rubric(not_applicable={"R1": "the brief never asks for it"})
    result = score_rubric(
        rubric, verdicts(G1=True, S1=True, S2=True, F1=True, A1=False), build_id="b"
    )
    assert result.points_applicable == 65, "R1's 35 points must not count against it"
    assert result.score == 100.0, "a full score on 65 applicable points is still 100"


def test_normalisation_makes_two_ideas_comparable():
    """The 92-point ideas and the 100-point ideas must land on the same scale."""
    full = score_rubric(
        make_rubric(),
        verdicts(G1=True, S1=True, S2=False, F1=True, R1=True, A1=False),
        build_id="b",
    )
    reduced = score_rubric(
        make_rubric(not_applicable={"R1": "n/a"}),
        verdicts(G1=True, S1=True, S2=False, F1=True, A1=False),
        build_id="b",
    )
    assert full.base == pytest.approx(85.0)
    assert reduced.base == pytest.approx(76.9, abs=0.1)
    # Same items failed; the scores differ only because the denominators do.
    assert full.score > reduced.score


def test_an_unresolved_item_scores_zero_and_is_flagged():
    rubric = make_rubric()
    unresolved = ItemVerdict(item_id="S1", passes=[None, None, None])
    given = verdicts(G1=True, S2=True, F1=True, R1=True) | {"S1": unresolved}
    result = score_rubric(rubric, given, build_id="b", passes=3)

    assert "S1" in result.unresolved
    assert result.points_earned == 75, "an unresolved item earns nothing"
    assert result.reliability()["unresolved"] == 1


def test_penalties_subtract_and_are_capped():
    heavy = [item("A1", -10, -1), item("A2", -10, -1), item("A3", -10, -1)]
    rubric = make_rubric(penalties=heavy)
    result = score_rubric(
        rubric,
        verdicts(
            G1=True, S1=True, S2=True, F1=True, R1=True, A1=True, A2=True, A3=True
        ),
        build_id="b",
    )
    assert result.penalty_total == PENALTY_CAP
    assert result.penalty_capped
    assert result.score == 75.0


def test_a_penalty_respects_its_own_max_total():
    rubric = make_rubric(penalties=[item("A1", -5, -1, max_total=-10)])
    result = score_rubric(
        rubric,
        verdicts(G1=True, S1=True, S2=True, F1=True, R1=True, A1=True),
        build_id="b",
    )
    assert result.penalty_total == -5


def test_the_score_floors_at_zero_rather_than_going_negative():
    rubric = make_rubric(penalties=[item("A1", -25, -1)])
    result = score_rubric(
        rubric,
        verdicts(G1=True, S1=False, S2=False, F1=False, R1=True, A1=True),
        build_id="b",
    )
    assert result.base == 35.0
    assert result.penalty_total == -25
    assert result.score == 10.0

    worse = score_rubric(
        rubric,
        verdicts(G1=True, S1=False, S2=False, F1=False, R1=False, A1=True),
        build_id="b",
    )
    assert worse.score == 0.0
    assert worse.floor_applied


# ------------------------------------------------------------ majority verdict


@pytest.mark.parametrize(
    ("passes", "expected"),
    [
        ([True, True, True], True),
        ([True, True, False], True),
        ([True, False, False], False),
        ([False, False, False], False),
        ([True, None, True], True),
        ([True, None, False], False),  # a tie is not a pass
        ([None, None, None], False),
        ([], False),
    ],
)
def test_majority_of_three_passes(passes, expected):
    assert ItemVerdict(item_id="S1", passes=passes).passed is expected


def test_disagreement_and_unresolved_are_distinct():
    assert ItemVerdict(item_id="x", passes=[True, False, True]).disagreement
    assert not ItemVerdict(item_id="x", passes=[True, True, True]).disagreement
    assert ItemVerdict(item_id="x", passes=[None, None, None]).unresolved
    assert not ItemVerdict(item_id="x", passes=[None, True, True]).unresolved


def test_reliability_splits_code_judged_from_agent_judged():
    rubric = make_rubric()
    given = {
        "G1": ItemVerdict(item_id="G1", passes=[True, True, True]),
        "S1": ItemVerdict(
            item_id="S1", passes=[True, False, True]
        ),  # assert, disagrees
        "S2": ItemVerdict(item_id="S2", passes=[True, True, True]),
        "F1": ItemVerdict(item_id="F1", passes=[True, True, True]),  # probe
        "R1": ItemVerdict(
            item_id="R1", passes=[True, True, True], harness_override=True
        ),
        "A1": ItemVerdict(item_id="A1", passes=[False, False, False]),
    }
    stats = score_rubric(rubric, given, build_id="b", passes=3).reliability()

    assert stats["items_total"] == 6
    assert stats["items_disagreeing"] == 1
    assert stats["code_disagreement"] > 0
    assert stats["override_rate"] == pytest.approx(1 / 6, abs=0.01)


# ------------------------------------------------------------------- rendering


def test_tier_breakdown_matches_the_score():
    rubric = make_rubric()
    result = score_rubric(
        rubric,
        verdicts(G1=True, S1=True, S2=False, F1=True, R1=True, A1=False),
        build_id="b",
    )
    tiers = tier_breakdown(rubric, result)

    assert [tier["earned"] for tier in tiers] == [25, 25, 35]
    assert sum(tier["earned"] for tier in tiers) == result.points_earned
    assert sum(tier["points"] for tier in tiers) == result.points_applicable
    failed = next(row for row in tiers[0]["items"] if row["id"] == "S2")
    assert failed["passed"] is False and failed["earned"] == 0


def test_as_dict_is_json_shaped_and_carries_the_derived_fields():
    rubric = make_rubric()
    result = score_rubric(
        rubric,
        verdicts(G1=True, S1=True, S2=True, F1=True, R1=True),
        build_id="b",
        passes=1,
    )
    data = result.as_dict()

    assert data["ok"] is True
    assert data["verdicts"]["S1"]["passed"] is True
    assert "reliability" in data
    assert "_methods" not in data
