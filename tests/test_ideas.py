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

"""Tests for the Idea Bench loader/validator (Stage 1).

Two kinds of tests:

1. Schema unit tests using small in-memory dicts (fast, exhaustive).
2. A test that loads EVERY real ``ideas/*.yaml`` file so a malformed idea
   added in a PR fails CI.
"""

from __future__ import annotations

import copy

import pytest

from viral_bench.ideas import (
    IdeaValidationError,
    load_ideas,
    parse_idea,
)

# A minimal valid spec used as a baseline; individual tests mutate a copy.
VALID = {
    "idea_id": "test_idea",
    "title": "Test Idea",
    "pitch": "A one-line pitch.",
    "problem": "A problem worth solving.",
    "target_user": "Someone specific.",
    "core_features": ["feature one"],
    "success_criteria": "It works.",
    "allowed_scope": "client-app",
    "difficulty": "easy",
}


def test_parse_minimal_valid_idea() -> None:
    idea = parse_idea(copy.deepcopy(VALID), source="test")
    assert idea.idea_id == "test_idea"
    assert idea.core_features == ("feature one",)
    assert idea.ground_truth is None


def test_parse_with_ground_truth() -> None:
    raw = copy.deepcopy(VALID)
    raw["ground_truth"] = {
        "source": "Twitter/X",
        "metric": "github_stars",
        "value": 12000,
    }
    idea = parse_idea(raw, source="test")
    assert idea.ground_truth is not None
    assert idea.ground_truth.value == 12000.0


@pytest.mark.parametrize("field", list(VALID.keys()))
def test_missing_required_field_raises(field: str) -> None:
    raw = copy.deepcopy(VALID)
    del raw[field]
    with pytest.raises(IdeaValidationError, match="missing required field"):
        parse_idea(raw, source="test")


def test_invalid_scope_raises() -> None:
    raw = copy.deepcopy(VALID)
    raw["allowed_scope"] = "mobile-app"
    with pytest.raises(IdeaValidationError, match="allowed_scope"):
        parse_idea(raw, source="test")


def test_invalid_difficulty_raises() -> None:
    raw = copy.deepcopy(VALID)
    raw["difficulty"] = "trivial"
    with pytest.raises(IdeaValidationError, match="difficulty"):
        parse_idea(raw, source="test")


def test_empty_core_features_raises() -> None:
    raw = copy.deepcopy(VALID)
    raw["core_features"] = []
    with pytest.raises(IdeaValidationError, match="core_features"):
        parse_idea(raw, source="test")


def test_ground_truth_missing_field_raises() -> None:
    raw = copy.deepcopy(VALID)
    raw["ground_truth"] = {"source": "Product Hunt", "metric": "github_stars"}
    with pytest.raises(IdeaValidationError, match="ground_truth"):
        parse_idea(raw, source="test")


def test_ground_truth_non_numeric_value_raises() -> None:
    raw = copy.deepcopy(VALID)
    raw["ground_truth"] = {"source": "X", "metric": "stars", "value": "lots"}
    with pytest.raises(IdeaValidationError, match="value"):
        parse_idea(raw, source="test")


# --- Tests against the real ideas/ directory ---------------------------------


def test_all_real_ideas_are_valid() -> None:
    """Every committed idea spec must parse and validate."""
    ideas = load_ideas()
    assert len(ideas) >= 1, "expected at least one idea spec in ideas/"
    # idea_ids must be unique (load_ideas enforces this, but assert explicitly).
    ids = [i.idea_id for i in ideas]
    assert len(ids) == len(set(ids))
