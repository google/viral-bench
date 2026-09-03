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

"""Tests for the rubric loader/validator.

Three kinds:

1. Unit tests over fabricated rubric files (fast, exhaustive, including
   mutation checks -- deliberately corrupt a rubric and confirm it is refused).
2. A test that loads EVERY real ``ideas/rubrics/*.yaml`` so a malformed rubric
   added later fails CI, mirroring ``tests/test_ideas.py``.
3. The leak guarantee: nothing under ``ideas/rubrics/`` may reach the founder.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from viral_bench.founder.prompts import render_idea
from viral_bench.ideas import load_ideas
from viral_bench.rubric.schema import (
    PENALTY_CAP,
    TIER1_POINTS,
    TIER2_POINTS,
    TIER3_POINTS,
    RubricError,
    available_rubrics,
    load_rubric,
    load_universal,
)

UNIVERSAL = """\
rubric_version: 1
gate:
  - {id: G1, text: "manifest parses", method: probe}
tier3:
  - {id: R1, text: "state survives a reload", points: 35, method: assert}
penalties:
  - {id: P1, text: "a dead control", points: -5, method: agent}
"""

IDEA = """\
rubric_version: 1
idea_id: demo
tier1:
  - {id: S1, text: "the thing works", points: 40, method: assert, expect: "ok"}
tier2:
  - {id: F1, text: "the other thing works", points: 25, method: probe}
penalties:
  - {id: A1, text: "lies about it", points: -8, method: assert}
"""


def write_rubrics(root: Path, *, universal: str = UNIVERSAL, idea: str = IDEA) -> Path:
    directory = root / "ideas" / "rubrics"
    directory.mkdir(parents=True)
    (directory / "_universal.yaml").write_text(universal, encoding="utf-8")
    (directory / "demo.yaml").write_text(idea, encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _clear_universal_cache():
    """``load_universal`` is lru_cached on the root, so tests must not share it."""
    load_universal.cache_clear()
    yield
    load_universal.cache_clear()


# ---------------------------------------------------------------- happy path


def test_loads_and_merges_universal_sections(tmp_path):
    root = write_rubrics(tmp_path)
    rubric = load_rubric("demo", root)

    assert rubric.idea_id == "demo"
    assert [item.id for item in rubric.tier1] == ["S1"]
    assert [item.id for item in rubric.tier3] == ["R1"], "universal tier3 must merge in"
    assert [item.id for item in rubric.gate] == ["G1"]
    assert [item.id for item in rubric.penalties] == ["A1", "P1"], "app penalties first"
    assert rubric.points_applicable == TIER1_POINTS + TIER2_POINTS + TIER3_POINTS


def test_not_applicable_removes_the_item_and_its_points(tmp_path):
    idea = IDEA + textwrap.dedent(
        """\
        universal_overrides:
          R1:
            applicable: false
            reason: the brief never asks for persistence
        """
    )
    root = write_rubrics(tmp_path, idea=idea)
    rubric = load_rubric("demo", root)

    assert [item.id for item in rubric.tier3] == []
    assert rubric.points_applicable == TIER1_POINTS + TIER2_POINTS
    assert rubric.not_applicable == {"R1": "the brief never asks for persistence"}


def test_check_block_parses_both_shapes(tmp_path):
    idea = IDEA.replace(
        '  - {id: F1, text: "the other thing works", points: 25, method: probe}',
        textwrap.dedent(
            """\
              - id: F1
                text: the other thing works
                points: 25
                method: assert
                check: {name: http_status, path: /, expect: 200}
            """
        ).rstrip(),
    )
    root = write_rubrics(tmp_path, idea=idea)
    rubric = load_rubric("demo", root)

    check = rubric.items_by_id()["F1"].check
    assert check is not None
    assert check.name == "http_status"
    assert check.params == {"path": "/", "expect": 200}
    assert rubric.items_by_id()["F1"].code_judged


def test_probe_items_are_code_judged_without_a_check(tmp_path):
    root = write_rubrics(tmp_path)
    assert load_rubric("demo", root).items_by_id()["F1"].code_judged


# ------------------------------------------------------------ mutation checks


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda s: s.replace("points: 40", "points: 39"), "tier1 must sum to 40"),
        (lambda s: s.replace("points: 25", "points: 26"), "tier2 must sum to 25"),
        (lambda s: s.replace("points: -8", "points: 8"), "must carry negative points"),
        (lambda s: s.replace("id: S1", "id: F1"), "duplicate item id"),
        (
            lambda s: s.replace("method: assert", "method: vibes"),
            "'method' must be one of",
        ),
        (
            lambda s: s.replace('text: "the thing works"', 'text: ""'),
            "non-empty 'text'",
        ),
        (lambda s: s.replace("idea_id: demo", "idea_id: other"), "declares idea_id"),
        (lambda s: s.replace("points: 40", 'points: "forty"'), "must be an integer"),
    ],
)
def test_a_corrupt_rubric_is_refused(tmp_path, mutation, message):
    root = write_rubrics(tmp_path, idea=mutation(IDEA))
    with pytest.raises(RubricError, match=message):
        load_rubric("demo", root)


def test_universal_tier3_total_is_enforced(tmp_path):
    root = write_rubrics(
        tmp_path, universal=UNIVERSAL.replace("points: 35", "points: 34")
    )
    with pytest.raises(RubricError, match="tier3 must sum to 35"):
        load_rubric("demo", root)


def test_an_override_naming_an_unknown_item_is_refused(tmp_path):
    idea = IDEA + "universal_overrides:\n  R9: {applicable: false, reason: nope}\n"
    root = write_rubrics(tmp_path, idea=idea)
    with pytest.raises(RubricError, match="unknown item"):
        load_rubric("demo", root)


def test_marking_an_item_not_applicable_requires_a_reason(tmp_path):
    idea = IDEA + "universal_overrides:\n  R1: {applicable: false}\n"
    root = write_rubrics(tmp_path, idea=idea)
    with pytest.raises(RubricError, match="requires a 'reason'"):
        load_rubric("demo", root)


def test_a_missing_rubric_is_an_error_not_a_default(tmp_path):
    root = write_rubrics(tmp_path)
    with pytest.raises(RubricError, match="no rubric for idea"):
        load_rubric("absent", root)


# ------------------------------------------------------------- the real files


def test_every_idea_has_a_rubric():
    ideas = {idea.idea_id for idea in load_ideas()}
    rubrics = set(available_rubrics())
    assert ideas == rubrics, f"missing {ideas - rubrics}, orphaned {rubrics - ideas}"


@pytest.mark.parametrize("idea_id", available_rubrics())
def test_real_rubric_is_valid(idea_id):
    rubric = load_rubric(idea_id)

    assert sum(item.points for item in rubric.tier1) == TIER1_POINTS
    assert sum(item.points for item in rubric.tier2) == TIER2_POINTS
    assert (
        rubric.points_applicable
        in (
            TIER1_POINTS + TIER2_POINTS,
            TIER1_POINTS + TIER2_POINTS + TIER3_POINTS,
        )
        or rubric.points_applicable > 0
    )
    assert rubric.penalties, "a rubric with no anti-pattern penalties is incomplete"
    assert all(item.points < 0 for item in rubric.penalties)
    # A rubric that could lose more than the cap is fine, but one that cannot reach
    # it means the cap is doing nothing and the weighting deserves a second look.
    assert sum(item.points for item in rubric.penalties) <= PENALTY_CAP


def test_rubric_content_never_reaches_the_founder_prompt():
    """The whole reason rubrics live in a subdirectory rather than on ``Idea``.

    Tested against the *actual* rubric text rather than a hand-written needle
    list: a brief may legitimately share vocabulary with its rubric (the CodeShot
    brief does say "clipboard"), so the question is whether a rubric's own
    wording -- its item text and its expected values -- shows up in the prompt.
    """
    for idea in load_ideas():
        rendered = render_idea(idea)
        rubric = load_rubric(idea.idea_id)
        for item in rubric.items_by_id().values():
            assert item.text not in rendered, f"{idea.idea_id}:{item.id} text leaked"
            if item.expect:
                assert item.expect not in rendered, (
                    f"{idea.idea_id}:{item.id} expect leaked"
                )


def test_the_idea_loader_does_not_pick_up_rubric_files():
    assert len(load_ideas()) == 25
    assert not any("rubric" in idea.idea_id for idea in load_ideas())
