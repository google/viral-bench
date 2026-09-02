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

"""Which builds a rubric sweep is about, across both selection eras."""

from __future__ import annotations

import json

from viral_bench.rubric.corpus import cohort_members, select_builds


def _fleet(tmp_path, entries):
    (tmp_path / "fleet.json").write_text(json.dumps({"entries": entries}))
    return tmp_path


def _cohort(tmp_path, name, members):
    (tmp_path / "cohorts").mkdir(exist_ok=True)
    (tmp_path / "cohorts" / f"{name}.json").write_text(
        json.dumps({"cohort": name, "members": members})
    )


def test_r4_is_selected_from_its_cohort_manifest(tmp_path):
    """A cohort spans two build eras (solo/dynamic kept, team rebuilt), so no
    property of a fleet key describes it. The manifest is the definition."""
    _fleet(
        tmp_path,
        {
            "notes::m1::solo::r3": {
                "build_id": "notes__1",
                "idea_id": "notes",
                "model": "m1",
                "status": "ok",
            },
            "wiki::m2::team::r4": {
                "build_id": "wiki__2",
                "idea_id": "wiki",
                "model": "m2",
                "status": "ok",
            },
        },
    )
    _cohort(tmp_path, "r4", {"notes__1": "solo", "wiki__2": "team"})

    found = select_builds(tmp_path, cohort="r4")
    assert [b.build_id for b in found] == ["notes__1", "wiki__2"]
    # The arm comes from the manifest, not the key: notes__1's key says r3.
    assert {b.build_id: b.arm for b in found} == {"notes__1": "solo", "wiki__2": "team"}
    assert {b.idea_id for b in found} == {"notes", "wiki"}


def test_an_untagged_cohort_selects_nothing_rather_than_some_other_corpus(tmp_path):
    """The dangerous failure is grading the wrong corpus and reporting it as the
    right one. Empty stops the sweep, and a fallback would not."""
    _fleet(
        tmp_path,
        {"notes::m1::solo::r3": {"build_id": "notes__1", "idea_id": "notes"}},
    )
    assert select_builds(tmp_path, cohort="r4") == []
    assert select_builds(tmp_path) == []
    assert cohort_members("r4", tmp_path) == {}


def test_the_r3_generation_path_still_works(tmp_path):
    _fleet(
        tmp_path,
        {
            "notes::m1::solo::r3": {
                "build_id": "notes__1",
                "idea_id": "notes",
                "model": "m1",
                "status": "ok",
            },
            "notes::m1::solo::r2": {
                "build_id": "notes__0",
                "idea_id": "notes",
                "model": "m1",
                "status": "ok",
            },
        },
    )
    found = select_builds(tmp_path, generation="r3")
    assert [b.build_id for b in found] == ["notes__1"]
    assert found[0].arm == "solo"


def test_undeliverable_builds_are_in_scope(tmp_path):
    """They land on the Tier 0 gate and score 0. Dropping them would report the
    average of the survivors and call it the average."""
    _fleet(
        tmp_path,
        {
            "a::m::solo::r4": {
                "build_id": "a__1",
                "idea_id": "a",
                "status": "manifest_missing",
            }
        },
    )
    _cohort(tmp_path, "r4", {"a__1": "solo"})
    found = select_builds(tmp_path, cohort="r4")
    assert len(found) == 1
    assert found[0].deliverable is False


def test_an_interrupted_rubric_sweep_yields_a_balanced_sample(tmp_path, monkeypatch):
    """The queue is interleaved by idea, so any PREFIX covers the whole corpus.

    A full 3-pass grade costs ~45 minutes, so a 1,000-build sweep is days of work
    and being interrupted is the normal case. Under the previous alphabetical
    order a prefix was 100% of the first idea or two and 0% of the other 23 --
    every model's score would come from two briefs, which is not reportable at
    any size. Round-robin over ideas makes a prefix a smaller sample of the same
    experiment instead of a different, useless one.
    """
    import importlib.util
    import pathlib
    import sys

    repo = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "rubric_sweep", repo / "scripts" / "rubric_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["rubric_sweep"] = module
    spec.loader.exec_module(module)

    from viral_bench.rubric.corpus import CorpusBuild

    ideas = [f"idea_{i:02d}" for i in range(10)]
    models = ["m1", "m2", "m3"]
    corpus = [
        CorpusBuild(f"{idea}__{model}", idea, model, "solo", "ok")
        for idea in ideas
        for model in models
    ]
    monkeypatch.setattr(module, "select_builds", lambda *a, **k: list(corpus))

    ordered = module.corpus_cells("r4", "")

    assert len(ordered) == len(corpus), "reorders, never drops"
    assert {c.build_id for c in ordered} == {c.build_id for c in corpus}
    # The first pass over the queue touches every idea exactly once, so a
    # prefix of any length is spread across the corpus rather than concentrated.
    first_round = ordered[: len(ideas)]
    assert len({c.idea_id for c in first_round}) == len(ideas)
    # And a third of the way in, every idea is represented roughly equally --
    # the property that makes a partial sweep reportable.
    third = ordered[: len(corpus) // 3]
    counts = {i: sum(1 for c in third if c.idea_id == i) for i in ideas}
    assert max(counts.values()) - min(counts.values()) <= 1, counts
