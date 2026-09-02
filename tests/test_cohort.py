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

"""Tests for naming a set of builds.

A cohort exists because the first full sweep had none: "the whole sweep" was
reconstructible only as "replicate 3, these arms, brief fingerprint current",
which had to be restated and re-verified every time anyone asked what a number
covered. It also does not survive the next step -- r4 keeps the solo and dynamic
arms and rebuilds team, so its members span two build eras and no replicate
number describes them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _cohort_module(tmp_path, monkeypatch):
    """Load scripts/cohort.py pointed at a throwaway builds tree."""
    spec = importlib.util.spec_from_file_location(
        "cohort_under_test", REPO / "scripts" / "cohort.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cohort_under_test"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "BUILDS", tmp_path)
    monkeypatch.setattr(mod, "COHORTS", tmp_path / "cohorts")
    return mod


def _fleet(tmp_path, cells: list[tuple[str, str, str, int]]) -> None:
    """Write a builds tree: (build_id, idea, structure, replicate) per cell."""
    entries = {}
    for build_id, idea, structure, replicate in cells:
        work = tmp_path / "work" / build_id
        work.mkdir(parents=True, exist_ok=True)
        (work / "build.json").write_text(
            json.dumps(
                {
                    "build_id": build_id,
                    "idea_id": idea,
                    "model": "m1",
                    "status": "ok",
                    "structure": structure,
                    "n_agents": 1 if structure == "solo" else 4,
                    "collab": "local",
                }
            ),
            encoding="utf-8",
        )
        entries[f"{idea}::m1::{structure}"] = {
            "build_id": build_id,
            "replicate": replicate,
        }
    (tmp_path / "fleet.json").write_text(
        json.dumps({"entries": entries}), encoding="utf-8"
    )


def test_tagging_stamps_the_builds_and_writes_a_manifest(tmp_path, monkeypatch):
    mod = _cohort_module(tmp_path, monkeypatch)
    _fleet(
        tmp_path,
        [
            ("b_solo", "i1", "solo", 3),
            ("b_dyn", "i1", "dynamic", 3),
            ("b_team", "i1", "team", 3),
        ],
    )

    assert mod.main(["tag", "r4", "--structures", "solo,dynamic"]) == 0

    for build_id, expected in (("b_solo", "r4"), ("b_dyn", "r4"), ("b_team", None)):
        rec = json.loads((tmp_path / "work" / build_id / "build.json").read_text())
        assert rec.get("cohort") == expected, build_id

    manifest = json.loads((tmp_path / "cohorts" / "r4.json").read_text())
    assert manifest["count"] == 2
    assert manifest["by_arm"] == {"dynamic": 1, "solo": 1}
    assert set(manifest["members"]) == {"b_solo", "b_dyn"}


def test_a_cohort_is_filled_in_stages_without_losing_the_first_half(
    tmp_path, monkeypatch
):
    """r4 tags the kept arms today and the rebuilt ones days later.

    A manifest that overwrote rather than merged would drop whichever half was
    written first -- which is the whole corpus, for the arm that finished first.
    """
    mod = _cohort_module(tmp_path, monkeypatch)
    _fleet(
        tmp_path,
        [
            ("b_solo", "i1", "solo", 3),
            ("b_dyn", "i1", "dynamic", 3),
            ("b_team", "i1", "team", 3),
        ],
    )

    mod.main(["tag", "r4", "--structures", "solo"])
    mod.main(["tag", "r4", "--structures", "dynamic,team"])

    manifest = json.loads((tmp_path / "cohorts" / "r4.json").read_text())
    assert set(manifest["members"]) == {"b_solo", "b_dyn", "b_team"}
    assert manifest["by_arm"] == {"dynamic": 1, "solo": 1, "team": 1}
    assert len(manifest["criteria"]) == 2, "each tagging pass is recorded"


def test_tagging_is_idempotent(tmp_path, monkeypatch):
    mod = _cohort_module(tmp_path, monkeypatch)
    _fleet(tmp_path, [("b_solo", "i1", "solo", 3)])
    mod.main(["tag", "r4", "--structures", "solo"])
    first = (tmp_path / "work" / "b_solo" / "build.json").read_text()
    mod.main(["tag", "r4", "--structures", "solo"])
    assert (tmp_path / "work" / "b_solo" / "build.json").read_text() == first


def test_a_dry_run_changes_nothing(tmp_path, monkeypatch):
    mod = _cohort_module(tmp_path, monkeypatch)
    _fleet(tmp_path, [("b_solo", "i1", "solo", 3)])
    mod.main(["tag", "r4", "--structures", "solo", "--dry-run"])
    rec = json.loads((tmp_path / "work" / "b_solo" / "build.json").read_text())
    assert "cohort" not in rec
    assert not (tmp_path / "cohorts").exists()


def test_stamping_preserves_fields_this_checkout_does_not_know(tmp_path, monkeypatch):
    """Rewrite the JSON, never round-trip it through BuildRecord.

    A record carrying a field added by a newer checkout would be silently dropped
    by a dataclass round trip, and build.json is the only copy.
    """
    mod = _cohort_module(tmp_path, monkeypatch)
    _fleet(tmp_path, [("b_solo", "i1", "solo", 3)])
    path = tmp_path / "work" / "b_solo" / "build.json"
    data = json.loads(path.read_text())
    data["some_future_field"] = {"keep": "me"}
    path.write_text(json.dumps(data))

    mod.main(["tag", "r4", "--structures", "solo"])

    rec = json.loads(path.read_text())
    assert rec["cohort"] == "r4"
    assert rec["some_future_field"] == {"keep": "me"}


def test_the_build_record_round_trips_a_cohort() -> None:
    from viral_bench.founder.build import BuildRecord, _record_from_dict

    rec = BuildRecord(
        build_id="b",
        idea_id="i",
        model="m",
        created_at="now",
        status="ok",
        app_dir="/a",
        root="/r",
        harness_ok=True,
        cohort="r4",
    )
    assert _record_from_dict(json.loads(rec.to_json())).cohort == "r4"


def test_a_scored_run_carries_the_cohort_from_its_summary() -> None:
    """So "which runs belong to the final sweep" is answerable from the run."""
    from viral_bench.score.fleet import ScoredRun

    assert (
        ScoredRun(
            crowd_dir="/tmp/x",
            build_id="b",
            idea_id="i",
            model="m",
            seed=0,
            n_agents=30,
            score=50.0,
            cohort="r4",
        ).cohort
        == "r4"
    )


@pytest.mark.parametrize("name", ["r4", "final-2026-08"])
def test_status_reports_a_partly_filled_cohort_as_partly_filled(
    tmp_path, monkeypatch, name, capsys
):
    mod = _cohort_module(tmp_path, monkeypatch)
    _fleet(tmp_path, [("b_solo", "i1", "solo", 3)])
    mod.main(["tag", name, "--structures", "solo"])
    mod.main(["status", name, "--expect-per-arm", "1"])
    out = capsys.readouterr().out
    assert "team      0  <-- expected 1" in out
    assert "every member is stamped and readable" in out
