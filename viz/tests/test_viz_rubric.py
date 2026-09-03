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

"""Tests for the rubric grade reader and its routes.

Two properties carry most of the weight here, because getting either wrong makes
the viewer lie about a score rather than merely look wrong.

**A failed gate zeroes the score and leaves everything else unresolved, not
failed.** 54 of the 1,000 builds in the sweep are undeliverable and land exactly
here. Rendering them as "27 items failed" would describe one manifest typo as
twenty-seven defects.

**Not-applicable items leave the denominator.** An idea whose brief never asks
for persistence is scored out of 92, not capped at 92 out of 100 -- and a viewer
that recomputed it the obvious way would quietly disagree with the scorer.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import serve as serve_mod  # noqa: E402
from core import export as export_mod  # noqa: E402
from core import rubric as rubric_mod  # noqa: E402
from core.apphost import AppLauncher  # noqa: E402
from core.paths import (  # noqa: E402
    index_rubric_grades,
    rubric_grades_for_build,
    rubric_run_dir,
)
from vizfixtures import builds, make_rubric_run  # noqa: E402,F401

GRADE_ID = "idea__solo__rubric-20260101-000000"
GATE_ID = "idea__gated__rubric-20260101-000000"


@pytest.fixture
def graded(builds):  # noqa: F811 - pytest fixture injection
    make_rubric_run(builds, GRADE_ID)
    return builds


@pytest.fixture
def rubric_server(graded):
    make_rubric_run(graded, GATE_ID, gate_failed=True)
    serve_mod.Handler.store = serve_mod.Store(graded)
    serve_mod.Handler.launcher = AppLauncher(graded)
    serve_mod.Handler.verbose = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve_mod.Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.05)
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get_json(base: str, path: str):
    with urllib.request.urlopen(f"{base}{path}", timeout=10) as response:
        return json.loads(response.read())


def expect_status(base: str, path: str, status: int):
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=10) as response:
            assert response.status == status
    except urllib.error.HTTPError as exc:
        assert exc.code == status, f"{path}: wanted {status}, got {exc.code}"


# ---------------------------------------------------------------- discovery


def test_a_grade_directory_is_found_by_id(graded):
    assert rubric_run_dir(graded, GRADE_ID) is not None


def test_a_directory_without_grade_json_is_not_a_grade(graded, tmp_path):
    (graded / "rubric" / "not_a_grade__rubric-20260101-000000").mkdir(parents=True)
    assert rubric_run_dir(graded, "not_a_grade__rubric-20260101-000000") is None


def test_grades_for_a_build_are_found_by_prefix(graded):
    found = rubric_grades_for_build(graded, "idea__solo")
    assert [p.name for p in found] == [GRADE_ID]


def test_the_index_carries_both_scores_and_their_difference(graded):
    rows = index_rubric_grades(graded)
    assert len(rows) == 1
    row = rows[0]
    assert row["score"] == 62.0
    assert row["viral_score"] == 41.9
    # The delta is the entire reason the second track exists, so the picker
    # carries it rather than making you open each grade to work it out.
    assert row["delta"] == 20.1
    assert row["arm"] == "solo"


# ---------------------------------------------------------------- reading


def test_load_grade_returns_the_document_verbatim(graded):
    grade = rubric_mod.load_grade(rubric_run_dir(graded, GRADE_ID))
    assert grade["score"] == 62.0
    assert grade["math"]["points_earned"] == 64
    assert grade["idea_id"] == "demo_idea"


def test_the_grade_is_self_describing(graded):
    """The viewer never imports viral_bench, so the wording must be in the file."""
    grade = rubric_mod.load_grade(rubric_run_dir(graded, GRADE_ID))
    texts = [i["text"] for t in grade["tiers"] for i in t["items"]]
    assert all(texts)
    assert [t["label"] for t in grade["tiers"]] == [
        "Success criteria",
        "Core features",
        "Robustness and craft",
    ]


def test_not_applicable_items_leave_the_denominator(graded):
    grade = rubric_mod.load_grade(rubric_run_dir(graded, GRADE_ID))
    assert grade["math"]["points_applicable"] == 92
    assert grade["not_applicable"][0]["id"] == "R1"


def test_a_failed_gate_zeroes_the_score_and_unresolves_the_rest(graded):
    make_rubric_run(graded, GATE_ID, gate_failed=True)
    grade = rubric_mod.load_grade(rubric_run_dir(graded, GATE_ID))
    assert grade["score"] == 0.0
    assert grade["math"]["gate_zeroed"] is True
    assert grade["gate"]["passed"] is False
    items = [i for t in grade["tiers"] for i in t["items"]]
    # Unresolved, not failed. One typo is one defect, not twenty-seven.
    assert all(i["passed"] is None and i["unresolved"] for i in items)
    assert all(i["earned"] == 0 for i in items)


def test_a_missing_grade_reads_as_none(graded, tmp_path):
    assert rubric_mod.load_grade(tmp_path) is None


def test_a_truncated_grade_reads_as_none(graded, tmp_path):
    (tmp_path / "grade.json").write_text('{"score": 6')
    assert rubric_mod.load_grade(tmp_path) is None


# ---------------------------------------------------------------- transcript


def test_the_corrupt_final_line_is_skipped_not_fatal(graded):
    """A killed sweep leaves half a line; it must cost one row, not the grade."""
    page = rubric_mod.load_transcript(rubric_run_dir(graded, GRADE_ID))
    assert page["total"] == 6
    assert {c["id"] for c in page["calls"]} == {
        "tc_004",
        "tc_007",
        "tc_009",
        "tc_011",
        "tc_013",
        "tc_014",
    }


def test_the_transcript_can_be_filtered_to_one_item(graded):
    page = rubric_mod.load_transcript(rubric_run_dir(graded, GRADE_ID), item_id="R3")
    assert page["total"] == 2
    assert all(c["item_id"] == "R3" for c in page["calls"])


def test_the_transcript_pages(graded):
    page = rubric_mod.load_transcript(
        rubric_run_dir(graded, GRADE_ID), offset=2, limit=2
    )
    assert len(page["calls"]) == 2
    assert page["total"] == 6


def test_a_failed_tool_call_is_marked_not_dropped(graded):
    page = rubric_mod.load_transcript(rubric_run_dir(graded, GRADE_ID))
    failed = [c for c in page["calls"] if not c["ok"]]
    assert len(failed) == 1
    assert failed[0]["id"] == "tc_014"


def test_evidence_is_indexed_from_the_harness_log_not_the_claim(graded):
    """The index is built from recorded calls, so it shows evidence that exists."""
    grade = rubric_mod.load_grade(rubric_run_dir(graded, GRADE_ID))
    assert grade["evidence_index"]["S2"] == ["tc_007"]
    assert grade["evidence_index"]["R3"] == ["tc_013", "tc_014"]


def test_find_item_reaches_tiers_penalties_and_the_gate(graded):
    grade = rubric_mod.load_grade(rubric_run_dir(graded, GRADE_ID))
    assert rubric_mod.find_item(grade, "S1")["kind"] == "item"
    assert rubric_mod.find_item(grade, "A1")["kind"] == "penalty"
    assert rubric_mod.find_item(grade, "G1")["kind"] == "gate"
    assert rubric_mod.find_item(grade, "nope") is None


# ---------------------------------------------------------------- export


def test_the_bundle_carries_the_grade_and_the_transcript(graded):
    bundle = export_mod.rubric_bundle(rubric_run_dir(graded, GRADE_ID))
    assert bundle["kind"] == "viral_bench.rubric_grade"
    assert bundle["grade"]["score"] == 62.0
    assert len(bundle["transcript"]) == 6


def test_the_zip_holds_the_grade_the_log_and_the_shots(graded):
    import io
    import zipfile

    blob = export_mod.rubric_zip(rubric_run_dir(graded, GRADE_ID))
    names = zipfile.ZipFile(io.BytesIO(blob)).namelist()
    assert f"{GRADE_ID}/grade.json" in names
    assert f"{GRADE_ID}/transcript.jsonl" in names
    assert f"{GRADE_ID}/shots/shot_001.png" in names


# ---------------------------------------------------------------- routes


def test_the_rubric_page_is_served(rubric_server):
    with urllib.request.urlopen(f"{rubric_server}/rubric", timeout=10) as response:
        body = response.read()
    assert response.status == 200
    assert b"ViralBench" in body


def test_the_grade_endpoint_returns_the_document(rubric_server):
    data = get_json(rubric_server, f"/api/rubric/{GRADE_ID}")
    assert data["score"] == 62.0
    assert data["run_id"] == GRADE_ID


def test_the_picker_lists_grades(rubric_server):
    data = get_json(rubric_server, "/api/rubric/grades?limit=10")
    assert data["matched"] == 2
    assert {g["run_id"] for g in data["grades"]} == {GRADE_ID, GATE_ID}


def test_the_picker_can_filter_to_gate_failures(rubric_server):
    data = get_json(rubric_server, "/api/rubric/grades?gated=1")
    assert [g["run_id"] for g in data["grades"]] == [GATE_ID]


def test_the_picker_can_filter_to_divergences(rubric_server):
    """The interesting grades are the ones the two tracks disagree about."""
    data = get_json(rubric_server, "/api/rubric/grades?diverged=1")
    assert {g["run_id"] for g in data["grades"]} == {GRADE_ID, GATE_ID}


def test_the_picker_filters_by_build(rubric_server):
    data = get_json(rubric_server, "/api/rubric/grades?build_id=idea__solo")
    assert [g["run_id"] for g in data["grades"]] == [GRADE_ID]


def test_the_transcript_endpoint_pages(rubric_server):
    data = get_json(rubric_server, f"/api/rubric/{GRADE_ID}/transcript?limit=3")
    assert data["total"] == 6
    assert len(data["calls"]) == 3


def test_a_screenshot_is_served_from_the_grade_dir(rubric_server):
    with urllib.request.urlopen(
        f"{rubric_server}/api/rubric/{GRADE_ID}/shot/shot_001.png", timeout=10
    ) as response:
        assert response.status == 200
        assert response.read().startswith(b"\x89PNG")


def test_the_download_endpoints_set_a_filename(rubric_server):
    for suffix, kind in (("", "application/json"), ("?format=zip", "application/zip")):
        with urllib.request.urlopen(
            f"{rubric_server}/api/rubric/{GRADE_ID}/download{suffix}", timeout=10
        ) as response:
            assert response.status == 200
            assert kind in response.headers["Content-Type"]
            assert GRADE_ID in response.headers["Content-Disposition"]


def test_an_unknown_grade_is_404(rubric_server):
    expect_status(rubric_server, "/api/rubric/no_such_grade", 404)


def test_an_unknown_sub_endpoint_is_404(rubric_server):
    expect_status(rubric_server, f"/api/rubric/{GRADE_ID}/nope", 404)


@pytest.mark.parametrize(
    "attempt",
    [
        "/api/rubric/..%2f..%2fetc",
        "/api/rubric/..",
        "/api/rubric/a%2f..%2fb",
    ],
)
def test_a_grade_id_cannot_climb_out_of_the_builds_tree(rubric_server, attempt):
    try:
        with urllib.request.urlopen(f"{rubric_server}{attempt}", timeout=10) as resp:
            assert resp.status in (400, 404)
    except urllib.error.HTTPError as exc:
        assert exc.code in (400, 404)


def test_a_screenshot_name_cannot_climb_out(rubric_server):
    expect_status(
        rubric_server, f"/api/rubric/{GRADE_ID}/shot/..%2f..%2fgrade.json", 404
    )


def test_a_grade_written_after_startup_is_found_immediately(rubric_server, graded):
    """Discovery is a stat on a path, not a cached index, so a sweep's output
    shows up while the sweep is still running."""
    late = "idea__late__rubric-20260101-999999"
    make_rubric_run(graded, late)
    data = get_json(rubric_server, f"/api/rubric/{late}")
    assert data["run_id"] == late
