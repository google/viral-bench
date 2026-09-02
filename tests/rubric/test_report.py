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

"""Tests for the grade document and its storage.

The document is the contract between the scorer and the viewer, and the viewer's
readers never import ``viral_bench`` (``viz/README.md:452-456``). So the tests
that matter are the ones that pin *self-description*: every number the header
shows, and every string it renders, has to be in the file. A test that only
checked round-tripping would pass while the viewer had to reach back into
``ideas/rubrics/`` to render anything.
"""

from __future__ import annotations

import json

from viral_bench.rubric.report import (
    GRADE_KIND,
    build_grade_document,
    comparison_block,
    founder_block,
    new_run_id,
    read_grade,
    render_grade,
    rubric_runs_for_build,
    source_hash,
    write_grade,
)
from viral_bench.rubric.schema import Rubric, RubricItem
from viral_bench.rubric.score import ItemVerdict, score_rubric


def item(item_id, points, tier, method="assert", **kwargs):
    return RubricItem(
        id=item_id,
        text=f"{item_id} claim text",
        points=points,
        method=method,
        tier=tier,
        **kwargs,
    )


def make_rubric(not_applicable=None):
    return Rubric(
        idea_id="demo",
        rubric_version="1",
        tier1=(item("S1", 25, 1), item("S2", 15, 1)),
        tier2=(item("F1", 25, 2, method="probe"),),
        tier3=() if not_applicable else (item("R1", 35, 3),),
        gate=(item("G1", 0, 0, method="probe"),),
        penalties=(item("A1", -8, -1),),
        not_applicable=not_applicable or {},
    )


def graded(**flags):
    verdicts = {
        key: ItemVerdict(item_id=key, passes=[value]) for key, value in flags.items()
    }
    return make_rubric(), verdicts


def document_for(**flags):
    rubric, verdicts = graded(**flags)
    result = score_rubric(rubric, verdicts, build_id="b1", passes=1)
    return build_grade_document(
        rubric,
        result,
        run_id="b1__rubric-20260101-000000",
        grader_model="openai/gpt-test",
        founder={"on_disk": False, "build_id": "b1"},
    )


# ------------------------------------------------------- self-description


def test_the_document_carries_item_text_not_just_ids():
    """The viewer cannot look the wording up, so it has to be in the file."""
    doc = document_for(G1=True, S1=True, S2=False, F1=True, R1=True, A1=False)
    texts = [row["text"] for tier in doc["tiers"] for row in tier["items"]]
    assert "S1 claim text" in texts
    assert all(text for text in texts)


def test_the_document_carries_tier_labels_and_points():
    doc = document_for(G1=True, S1=True, S2=True, F1=True, R1=True, A1=False)
    labels = {tier["tier"]: tier["label"] for tier in doc["tiers"]}
    assert labels == {
        1: "Success criteria",
        2: "Core features",
        3: "Robustness and craft",
    }
    assert [tier["points"] for tier in doc["tiers"]] == [40, 25, 35]


def test_every_number_the_header_shows_is_precomputed():
    """`math` exists so the viewer never re-derives the score and disagrees."""
    doc = document_for(G1=True, S1=True, S2=False, F1=True, R1=True, A1=True)
    math = doc["math"]
    assert math["points_earned"] == 85
    assert math["points_applicable"] == 100
    assert math["base"] == 85.0
    assert math["penalty_total"] == -8
    assert doc["score"] == 77.0
    # The arithmetic in the file has to agree with the score in the file.
    assert doc["score"] == round(math["base"] + math["penalty_total"], 1)


def test_a_failed_gate_is_visible_as_a_zero_with_its_reason():
    doc = document_for(G1=False, S1=True, S2=True, F1=True, R1=True, A1=False)
    assert doc["score"] == 0
    assert doc["math"]["gate_zeroed"] is True
    assert doc["gate"]["passed"] is False
    assert doc["gate"]["failures"] == ["G1"]
    # The tier rows survive, so a dead build can still be diagnosed.
    assert doc["tiers"][0]["items"]


def test_not_applicable_items_are_listed_with_their_reason():
    rubric = make_rubric({"R1": "the brief never asks for persistence"})
    verdicts = {
        key: ItemVerdict(item_id=key, passes=[True]) for key in ("G1", "S1", "S2", "F1")
    }
    result = score_rubric(rubric, verdicts, build_id="b1", passes=1)
    doc = build_grade_document(rubric, result, run_id="r", grader_model="m", founder={})
    assert doc["not_applicable"] == [
        {"id": "R1", "reason": "the brief never asks for persistence"}
    ]
    # And the denominator shrank rather than the score being capped at 65.
    assert doc["math"]["points_applicable"] == 65
    assert doc["score"] == 100.0


def test_penalties_record_whether_they_fired():
    doc = document_for(G1=True, S1=True, S2=True, F1=True, R1=True, A1=True)
    penalty = doc["penalties"][0]
    assert penalty["id"] == "A1"
    assert penalty["fired"] is True
    assert penalty["points"] == -8
    assert penalty["text"]


def test_reliability_travels_with_the_grade():
    """A gap smaller than the instrument's own noise is not a finding."""
    doc = document_for(G1=True, S1=True, S2=True, F1=True, R1=True, A1=False)
    assert "items_total" in doc["reliability"]
    assert "override_rate" in doc["reliability"]


def test_the_document_is_json_serialisable():
    doc = document_for(G1=True, S1=True, S2=True, F1=True, R1=True, A1=False)
    assert json.loads(json.dumps(doc))["kind"] == GRADE_KIND


# ------------------------------------------------------------- storage


def test_a_grade_round_trips_through_disk(tmp_path):
    doc = document_for(G1=True, S1=True, S2=False, F1=True, R1=True, A1=False)
    run_dir = tmp_path / "rubric" / "b1__rubric-20260101-000000"
    write_grade(doc, run_dir)
    loaded = read_grade(run_dir)
    assert loaded["score"] == doc["score"]
    assert loaded["tiers"][0]["items"][0]["text"] == "S1 claim text"


def test_writing_fills_in_the_paths_block(tmp_path):
    doc = document_for(G1=True, S1=True, S2=True, F1=True, R1=True, A1=False)
    run_dir = tmp_path / "b1__rubric-20260101-000000"
    write_grade(doc, run_dir)
    paths = read_grade(run_dir)["paths"]
    assert paths["grade"].endswith("grade.json")
    assert paths["transcript"].endswith("transcript.jsonl")
    assert paths["shots"].endswith("shots")


def test_a_truncated_grade_reads_as_none_rather_than_raising(tmp_path):
    """Sweeps get killed; half a file must not take the reader down."""
    run_dir = tmp_path / "b1__rubric-20260101-000000"
    run_dir.mkdir(parents=True)
    (run_dir / "grade.json").write_text('{"score": 4')
    assert read_grade(run_dir) is None


def test_a_build_can_be_graded_more_than_once(tmp_path):
    doc = document_for(G1=True, S1=True, S2=True, F1=True, R1=True, A1=False)
    for stamp in ("20260101-000000", "20260202-000000"):
        write_grade(doc, tmp_path / "rubric" / f"b1__rubric-{stamp}")
    found = rubric_runs_for_build("b1", root=tmp_path)
    assert len(found) == 2
    # Newest first, so a viewer's default pick is the current grade.
    assert found[0].name.endswith("20260202-000000")


def test_run_ids_follow_the_prefix_glob_convention():
    run_id = new_run_id("some_idea__20260101-000000__abc123")
    assert run_id.startswith("some_idea__20260101-000000__abc123__rubric-")


# --------------------------------------------------------- context blocks


def test_source_hash_changes_when_source_changes(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "index.html").write_text("<h1>one</h1>")
    first = source_hash(app)
    (app / "index.html").write_text("<h1>two</h1>")
    assert source_hash(app) != first


def test_source_hash_ignores_generated_directories(tmp_path):
    """Otherwise a grade would invalidate itself by running the app once."""
    app = tmp_path / "app"
    (app / "node_modules" / "x").mkdir(parents=True)
    (app / "index.html").write_text("<h1>one</h1>")
    before = source_hash(app)
    (app / "node_modules" / "x" / "junk.js").write_text("whatever")
    assert source_hash(app) == before


def test_source_hash_notices_a_rename(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "a.html").write_text("same bytes")
    before = source_hash(app)
    (app / "a.html").rename(app / "b.html")
    assert source_hash(app) != before


def test_founder_block_reports_not_on_disk_for_a_pruned_build(tmp_path):
    block = founder_block("gone", root=tmp_path)
    assert block == {"on_disk": False, "build_id": "gone"}


def test_founder_block_resolves_the_arm_and_the_model(tmp_path):
    work = tmp_path / "work" / "b1"
    work.mkdir(parents=True)
    (work / "build.json").write_text(
        json.dumps(
            {
                "idea_id": "demo",
                "structure": "team",
                "collab": "local",
                "model": "publishers/anthropic/claude-test@default",
                "status": "shipped",
                "manifest": {"title": "Demo App"},
            }
        )
    )
    block = founder_block("b1", root=tmp_path)
    assert block["arm"] == "team"
    assert block["arm_label"] == "4-agent team"
    assert block["model_short"] == "claude-test"
    assert block["app_title"] == "Demo App"


def test_comparison_is_none_when_the_build_was_never_crowd_scored(tmp_path):
    assert comparison_block("b1", root=tmp_path) is None


def test_comparison_averages_every_crowd_score_for_the_build(tmp_path):
    for stamp, score in (("20260101-000000", 40.0), ("20260102-000000", 50.0)):
        run = tmp_path / "crowd" / f"b1__crowd-{stamp}"
        run.mkdir(parents=True)
        (run / "score.json").write_text(json.dumps({"score": score, "scorable": True}))
    block = comparison_block("b1", root=tmp_path)
    assert block == {
        "viral_score_mean": 45.0,
        "viral_score_min": 40.0,
        "viral_score_max": 50.0,
        "crowd_runs": 2,
    }


def test_comparison_ignores_unscorable_runs(tmp_path):
    run = tmp_path / "crowd" / "b1__crowd-20260101-000000"
    run.mkdir(parents=True)
    (run / "score.json").write_text(json.dumps({"score": 0, "scorable": False}))
    assert comparison_block("b1", root=tmp_path) is None


# --------------------------------------------------------------- rendering


def test_the_render_shows_the_arithmetic_not_just_the_number():
    doc = document_for(G1=True, S1=True, S2=False, F1=True, R1=True, A1=True)
    text = render_grade(doc)
    assert "RubricScore: 77.0 / 100" in text
    assert "85/100 pts" in text
    assert "-8 penalties" in text
    assert "S2 claim text" in text


def test_the_render_calls_out_a_gate_failure():
    doc = document_for(G1=False, S1=True, S2=True, F1=True, R1=True, A1=False)
    text = render_grade(doc)
    assert "GATE FAILED" in text
    assert "RubricScore: 0" in text


def test_the_render_survives_a_document_with_missing_sections():
    """Renderers run over old files; a KeyError here would hide every grade."""
    assert render_grade({"score": 12.0}) is not None
