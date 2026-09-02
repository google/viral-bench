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

"""Tests for the grader loop, driven by a scripted client -- no model, no app.

The cases that matter here are the ones the anti-fabrication rule exists for. A
model that claims a pass and cites nothing, or cites a call it made while
grading a *different* item, must not be able to earn the point. Those are not
hypothetical failure modes: 36% of the crowd's triers filed verdicts on apps
they never touched, and the fix that worked was making evidence a recorded
artifact rather than a claim.

The other half is the division of labour. Where an item carries a ``check:``,
the harness decides and the model's stated verdict is advisory -- so a
disagreement is recorded as an override rather than silently resolved either
way.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from viral_bench.rubric.checks import CheckResult, check, registry
from viral_bench.rubric.grader import (
    GraderTools,
    ItemOutcome,
    merge_passes,
    run_pass,
)
from viral_bench.rubric.schema import Check, Rubric, RubricItem
from viral_bench.rubric.score import score_rubric


class Reply:
    def __init__(self, tool_calls=None, text=""):
        self.text = text
        self.tool_calls = tool_calls or []
        self.stop_reason = "stop"
        self.usage = {}


class Call:
    def __init__(self, name, args, call_id="c1"):
        self.id = call_id
        self.name = name
        self.args = args


class ScriptedClient:
    """Replays a fixed list of replies, so a loop test is deterministic."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def generate(self, messages, *, tools=None, system=""):
        self.seen.append(messages)
        if not self.replies:
            return Reply([Call("report", {"verdict": "unknown"})])
        return self.replies.pop(0)


class FakePage:
    """Just enough PageHandle for the loop: it records what it was asked to do."""

    def __init__(self, snapshot_text="hello"):
        self.snapshot_text = snapshot_text
        self.clicks = []
        self._requests = []

    def requests(self, *, since=0):
        return self._requests[since:]

    async def snapshot(self):
        class Snap:
            url = "http://localhost:8000/"
            title = "Demo"
            text = self.snapshot_text
            elements = ()
            console_errors = ()

        return Snap()

    async def click(self, target):
        self.clicks.append(target)

    async def evaluate(self, js):
        return {"js": js}


def tools_for(page=None):
    return GraderTools(page=page or FakePage(), url="http://localhost:8000")


def item(item_id, points, tier, *, method="agent", check_block=None):
    return RubricItem(
        id=item_id,
        text=f"{item_id} claim",
        points=points,
        method=method,
        tier=tier,
        check=check_block,
    )


def rubric_with(*items):
    """A rubric holding exactly the items under test, totals not enforced."""
    return Rubric(
        idea_id="demo",
        rubric_version="1",
        tier1=tuple(items),
        tier2=(),
        tier3=(),
        gate=(),
        penalties=(),
    )


def one_pass(rubric, client, tools, items=None):
    return asyncio.run(run_pass(rubric, tools, client, items=items))


# --------------------------------------------------------------- evidence


def test_a_pass_citing_a_real_call_is_kept():
    target = item("S1", 10, 1)
    client = ScriptedClient(
        [
            Reply([Call("look", {})]),
            Reply([Call("report", {"verdict": "pass", "evidence": ["tc_001"]})]),
        ]
    )
    tools = tools_for()
    outcomes = one_pass(rubric_with(target), client, tools)
    assert outcomes["S1"].passed is True
    assert outcomes["S1"].evidence == ["tc_001"]
    assert not outcomes["S1"].harness_override


def test_a_pass_citing_nothing_is_recorded_fail():
    """The anti-fabrication rule, in its simplest form."""
    target = item("S1", 10, 1)
    client = ScriptedClient([Reply([Call("report", {"verdict": "pass"})])])
    outcomes = one_pass(rubric_with(target), client, tools_for())
    assert outcomes["S1"].passed is False
    assert "cites no recorded tool call" in outcomes["S1"].reason
    # The model said pass and the harness said otherwise -- that is an override,
    # and it has to be visible or the disagreement rate understates the problem.
    assert outcomes["S1"].harness_override is True


def test_a_pass_citing_an_unknown_id_is_recorded_fail():
    target = item("S1", 10, 1)
    client = ScriptedClient(
        [
            Reply([Call("look", {})]),
            Reply([Call("report", {"verdict": "pass", "evidence": ["tc_999"]})]),
        ]
    )
    outcomes = one_pass(rubric_with(target), client, tools_for())
    assert outcomes["S1"].passed is False
    assert "tc_999" in outcomes["S1"].reason


def test_evidence_from_a_different_item_does_not_count():
    """Citing last item's call is the shortcut a model under step pressure takes."""
    first, second = item("S1", 10, 1), item("S2", 10, 1)
    client = ScriptedClient(
        [
            # S1: makes a real call, reports honestly.
            Reply([Call("look", {})]),
            Reply([Call("report", {"verdict": "pass", "evidence": ["tc_001"]})]),
            # S2: touches nothing, cites S1's call.
            Reply([Call("report", {"verdict": "pass", "evidence": ["tc_001"]})]),
        ]
    )
    outcomes = one_pass(rubric_with(first, second), client, tools_for())
    assert outcomes["S1"].passed is True
    assert outcomes["S2"].passed is False
    assert "none from this item" in outcomes["S2"].reason


def test_a_claimed_fail_without_evidence_stays_a_fail():
    """The rule only ever costs points that were never evidenced."""
    target = item("S1", 10, 1)
    client = ScriptedClient([Reply([Call("report", {"verdict": "fail"})])])
    outcomes = one_pass(rubric_with(target), client, tools_for())
    assert outcomes["S1"].passed is False
    assert outcomes["S1"].harness_override is False


def test_unknown_is_recorded_as_unresolved_not_as_a_fail():
    target = item("S1", 10, 1)
    client = ScriptedClient(
        [Reply([Call("report", {"verdict": "unknown", "note": "app never loaded"})])]
    )
    outcomes = one_pass(rubric_with(target), client, tools_for())
    assert outcomes["S1"].passed is None
    assert "never loaded" in outcomes["S1"].reason


def test_evidence_given_as_prose_still_binds():
    """Models cite ids in a sentence as often as in a list; accept both."""
    target = item("S1", 10, 1)
    client = ScriptedClient(
        [
            Reply([Call("look", {})]),
            Reply([Call("report", {"verdict": "pass", "evidence": "see tc_001"})]),
        ]
    )
    outcomes = one_pass(rubric_with(target), client, tools_for())
    assert outcomes["S1"].passed is True


# ------------------------------------------------------- code-judged items


@pytest.fixture
def always_true_check():
    name = "_test_always_true"
    saved = registry()

    @check(name)
    def _fn(ctx, **params):
        return CheckResult.yes(observed="42")

    yield name
    # Restore, so a registered test primitive cannot leak into another test.
    from viral_bench.rubric import checks as checks_module

    checks_module._REGISTRY.clear()
    checks_module._REGISTRY.update(saved)


def test_the_harness_overrules_the_model_on_a_checked_item(always_true_check):
    target = item("F1", 10, 2, method="assert", check_block=Check(always_true_check))
    client = ScriptedClient(
        [
            Reply([Call("look", {})]),
            Reply([Call("report", {"verdict": "fail"})]),
        ]
    )
    outcomes = one_pass(rubric_with(target), client, tools_for())
    assert outcomes["F1"].passed is True  # the code decided
    assert outcomes["F1"].model_verdict is False
    assert outcomes["F1"].harness_override is True
    assert outcomes["F1"].observed == "42"


def test_a_checked_item_needs_no_citation_from_the_model(always_true_check):
    """The model did not decide it, so its evidence discipline is not the gate."""
    target = item("F1", 10, 2, method="assert", check_block=Check(always_true_check))
    client = ScriptedClient(
        [
            Reply([Call("look", {})]),
            Reply([Call("report", {"verdict": "pass"})]),
        ]
    )
    outcomes = one_pass(rubric_with(target), client, tools_for())
    assert outcomes["F1"].passed is True
    assert outcomes["F1"].evidence == ["tc_001"]  # the harness's own log


def test_a_missing_primitive_is_unresolved_not_a_fail():
    target = item("F1", 10, 2, method="assert", check_block=Check("no_such_check"))
    client = ScriptedClient([Reply([Call("report", {"verdict": "pass"})])])
    outcomes = one_pass(rubric_with(target), client, tools_for())
    assert outcomes["F1"].passed is None
    assert "no such check primitive" in outcomes["F1"].reason


# ----------------------------------------------------------------- the loop


def test_the_step_budget_stops_a_model_that_never_reports():
    """Budget exhaustion is an instrument fault, so it is unresolved, not FAIL.

    Both earn zero. The difference is that unresolved is counted where it can be
    seen; recording it as FAIL would let a grader that keeps running out of steps
    produce low scores that read as real findings about the app.
    """
    target = item("S1", 10, 1)
    client = ScriptedClient([Reply([Call("look", {})]) for _ in range(50)])
    tools = tools_for()
    outcomes = asyncio.run(run_pass(rubric_with(target), tools, client, items=[target]))
    assert outcomes["S1"].passed is None
    assert "no verdict within 14 steps" in outcomes["S1"].reason
    assert outcomes["S1"].steps <= 14


def test_a_tool_failure_is_data_not_a_crash():
    class Exploding(FakePage):
        async def click(self, target):
            raise RuntimeError("element detached")

    target = item("S1", 10, 1)
    client = ScriptedClient(
        [
            Reply([Call("click", {"target": "Save"})]),
            Reply([Call("report", {"verdict": "fail", "evidence": ["tc_001"]})]),
        ]
    )
    tools = tools_for(Exploding())
    outcomes = one_pass(rubric_with(target), client, tools)
    assert outcomes["S1"].passed is False
    assert tools.log[0].ok is False
    assert "element detached" in tools.log[0].result


def test_the_transcript_records_every_call_with_its_item(tmp_path):
    first, second = item("S1", 10, 1), item("S2", 10, 1)
    client = ScriptedClient(
        [
            Reply([Call("look", {})]),
            Reply([Call("report", {"verdict": "pass", "evidence": ["tc_001"]})]),
            Reply([Call("click", {"target": "Go"})]),
            Reply([Call("report", {"verdict": "pass", "evidence": ["tc_002"]})]),
        ]
    )
    tools = tools_for()
    one_pass(rubric_with(first, second), client, tools)
    path = tools.write_transcript(tmp_path / "transcript.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["item_id"] for row in rows] == ["S1", "S2"]
    assert [row["name"] for row in rows] == ["look", "click"]
    assert [row["id"] for row in rows] == ["tc_001", "tc_002"]


def test_an_unknown_tool_name_does_not_end_the_grade():
    target = item("S1", 10, 1)
    client = ScriptedClient(
        [
            Reply([Call("teleport", {})]),
            Reply([Call("report", {"verdict": "fail", "evidence": ["tc_001"]})]),
        ]
    )
    tools = tools_for()
    outcomes = one_pass(rubric_with(target), client, tools)
    assert outcomes["S1"].passed is False
    assert "no such tool" in tools.log[0].result


# ------------------------------------------------------------ merging passes


def test_an_item_passes_on_two_of_three():
    merged = merge_passes(
        [
            {"S1": ItemOutcome("S1", True)},
            {"S1": ItemOutcome("S1", False)},
            {"S1": ItemOutcome("S1", True)},
        ]
    )
    assert merged["S1"].passes == [True, False, True]
    assert merged["S1"].passed is True
    assert merged["S1"].disagreement is True


def test_one_of_three_is_not_enough():
    merged = merge_passes(
        [
            {"S1": ItemOutcome("S1", True)},
            {"S1": ItemOutcome("S1", False)},
            {"S1": ItemOutcome("S1", False)},
        ]
    )
    assert merged["S1"].passed is False


def test_an_override_in_any_pass_is_carried_forward():
    """A transcript that needed overruling once is suspect for the whole item."""
    merged = merge_passes(
        [
            {"S1": ItemOutcome("S1", True)},
            {"S1": ItemOutcome("S1", True, harness_override=True)},
            {"S1": ItemOutcome("S1", True)},
        ]
    )
    assert merged["S1"].harness_override is True


def test_all_unresolved_stays_unresolved_through_scoring():
    merged = merge_passes([{"R1": ItemOutcome("R1", None)} for _ in range(3)])
    assert merged["R1"].unresolved is True
    rubric = rubric_with(item("R1", 10, 1))
    result = score_rubric(rubric, merged, build_id="b", passes=3)
    assert result.points_earned == 0
    assert result.unresolved == ["R1"]


def test_evidence_from_every_pass_is_kept():
    merged = merge_passes(
        [
            {"S1": ItemOutcome("S1", True, evidence=["tc_001"])},
            {"S1": ItemOutcome("S1", True, evidence=["tc_001", "tc_007"])},
        ]
    )
    assert merged["S1"].evidence == ["tc_001", "tc_007"]
