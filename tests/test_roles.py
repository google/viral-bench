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

"""Tests for the founder roles (solo + the four specialists)."""

from __future__ import annotations

import pytest

from viral_bench.founder.roles import (
    ALL_RESPONSIBILITIES,
    QA,
    TEAM_SIZE,
    roles_for,
)


def test_solo_is_single_founder_owning_everything() -> None:
    team = roles_for(1)
    assert len(team) == 1
    assert team[0].key == "founder"
    assert set(team[0].responsibilities) == set(ALL_RESPONSIBILITIES)


def test_team_has_four_specialists_in_order() -> None:
    keys = [r.key for r in roles_for(TEAM_SIZE)]
    assert keys == ["architect", "implementer", "designer", "qa_finisher"]


def test_team_responsibility_union_is_the_full_brief_once() -> None:
    owned: list = []
    for role in roles_for(TEAM_SIZE):
        owned.extend(role.responsibilities)
    assert set(owned) == set(ALL_RESPONSIBILITIES)
    assert len(owned) == len(ALL_RESPONSIBILITIES)  # no responsibility duplicated


def test_last_team_role_owns_qa() -> None:
    assert roles_for(TEAM_SIZE)[-1].owns_qa


@pytest.mark.parametrize("bad", [0, 2, 3, 5, -1, 100])
def test_unsupported_sizes_raise(bad: int) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        roles_for(bad)


def test_owns_qa_reflects_responsibilities() -> None:
    architect = roles_for(TEAM_SIZE)[0]
    assert QA not in architect.responsibilities
    assert not architect.owns_qa


# -- differentiated capabilities (what "levels up" each specialist) ---------- #


def test_solo_founder_has_no_specialist_tooling() -> None:
    founder = roles_for(1)[0]
    assert founder.system_prompt == ""
    assert founder.skills == ()
    assert founder.permissions == {}
    assert founder.temperature is None
    assert not founder.wants_browser


def test_each_specialist_has_a_persona_and_temperature() -> None:
    for role in roles_for(TEAM_SIZE):
        assert role.system_prompt, f"{role.key} missing system prompt"
        assert role.temperature is not None, f"{role.key} missing temperature"


def test_web_research_is_role_specific() -> None:
    by_key = {r.key: r for r in roles_for(TEAM_SIZE)}
    # Architect + Designer research the web; Implementer + QA stay focused (no web).
    assert by_key["architect"].permissions.get("webfetch") == "allow"
    assert by_key["designer"].permissions.get("webfetch") == "allow"
    assert by_key["implementer"].permissions.get("webfetch") == "deny"
    assert by_key["qa_finisher"].permissions.get("webfetch") == "deny"


def test_temperature_ordering_matches_role_character() -> None:
    by_key = {r.key: r for r in roles_for(TEAM_SIZE)}
    # QA rigorous (coldest) < Implementer precise < Architect < Designer creative.
    assert (
        by_key["qa_finisher"].temperature
        < by_key["implementer"].temperature
        < by_key["architect"].temperature
        < by_key["designer"].temperature
    )


def test_browser_roles_are_designer_and_qa() -> None:
    by_key = {r.key: r for r in roles_for(TEAM_SIZE)}
    assert by_key["designer"].wants_browser
    assert by_key["qa_finisher"].wants_browser
    assert not by_key["architect"].wants_browser
    assert not by_key["implementer"].wants_browser


def test_every_specialist_has_at_least_one_skill() -> None:
    for role in roles_for(TEAM_SIZE):
        assert role.skills, f"{role.key} has no skills"


def test_qa_has_live_app_testing_and_release_checklist() -> None:
    qa = {r.key: r for r in roles_for(TEAM_SIZE)}["qa_finisher"]
    assert "release-checklist" in qa.skills
    assert "live-app-testing" in qa.skills  # can actually run + drive the app
