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

"""Unit tests for the interaction trace (pure dataclasses; no app needed)."""

from __future__ import annotations

import json

from viral_bench.crowd.interaction.trace import (
    InteractionTrace,
    TrialVerdict,
)


def test_record_appends_indexed_steps() -> None:
    trace = InteractionTrace(build_id="b", app_type="client-app")
    s0 = trace.record("click", args={"target": "x"}, summary="ok")
    s1 = trace.record("click", args={"target": "y"}, summary="ok")
    assert (s0.index, s1.index) == (0, 1)
    assert trace.n_steps == 2


def test_had_effect_distinguishes_active_from_passive() -> None:
    trace = InteractionTrace(build_id="b", app_type="client-app")
    trace.record("open", summary="opened")
    trace.record("look", summary="looked")
    assert trace.had_effect() is False
    trace.record("click", args={"target": "Go"}, summary="clicked", ok=True)
    assert trace.had_effect() is True


def test_had_effect_ignores_failed_actions() -> None:
    trace = InteractionTrace(build_id="b", app_type="client-app")
    trace.record("click", args={"target": "Nope"}, summary="failed", ok=False)
    assert trace.had_effect() is False


def test_errors_are_flattened_across_steps() -> None:
    trace = InteractionTrace(build_id="b", app_type="client-app")
    trace.record("open", summary="o", errors=("console.error: boom",))
    trace.record("click", summary="c", errors=("pageerror: bang",))
    assert trace.errors == ["console.error: boom", "pageerror: bang"]


def test_verdict_clamps_delight() -> None:
    assert TrialVerdict(True, True, 99, "").delight == 10
    assert TrialVerdict(False, False, -5, "").delight == 0


def test_finish_records_verdict_and_end_time() -> None:
    trace = InteractionTrace(build_id="b", app_type="client-app")
    trace.record("click", summary="clicked")
    trace.finish(TrialVerdict(True, False, 7, "solid"))
    assert trace.ended_at is not None
    assert trace.verdict is not None
    assert trace.verdict.delight == 7


def test_jsonl_lines_carry_build_and_are_valid_json() -> None:
    trace = InteractionTrace(build_id="build-123", app_type="client-app")
    trace.record("click", args={"target": "Go"}, summary="ok")
    lines = trace.jsonl_lines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["build_id"] == "build-123"
    assert row["action"] == "click"
    assert row["index"] == 0


def test_to_json_roundtrips_structure() -> None:
    trace = InteractionTrace(build_id="b", app_type="client-app")
    trace.target_url = "http://localhost:1/"
    trace.record("open", summary="opened")
    trace.finish(TrialVerdict(True, True, 8, "great"))
    data = json.loads(trace.to_json())
    assert data["build_id"] == "b"
    assert data["target_url"] == "http://localhost:1/"
    assert data["verdict"]["would_share"] is True
    assert len(data["steps"]) == 1


def test_render_includes_degraded_banner_and_verdict() -> None:
    trace = InteractionTrace(build_id="b", app_type="client-app", degraded=True)
    trace.target_url = "http://x/"
    trace.record("open", summary="opened the page")
    trace.finish(TrialVerdict(False, False, 2, "meh"))
    rendered = trace.render()
    assert "DEGRADED" in rendered
    assert "opened the page" in rendered
    assert "delight=2/10" in rendered
