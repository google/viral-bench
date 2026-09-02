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

"""Tests for crowd prompts + launch-post text (mostly oasis/camel-free)."""

from __future__ import annotations

import pytest

from viral_bench.crowd.sim.personas import load_personas
from viral_bench.crowd.sim.prompts import (
    PROFILE_KEYS,
    RATING_RUBRIC,
    build_profile,
    interview_prompt,
    launch_post_text,
)


def _persona():
    return load_personas()[0]


def test_build_profile_keys_match_template_placeholders() -> None:
    # The profile dict keys must exactly match what the template expects.
    profile = build_profile(_persona(), "trier")
    assert set(profile.keys()) == set(PROFILE_KEYS)


def test_trier_and_reactor_missions_differ() -> None:
    trier = build_profile(_persona(), "trier")
    reactor = build_profile(_persona(), "reactor")
    assert trier["mission"] != reactor["mission"]
    assert "ACTUALLY TRIES IT" in trier["mission"]
    assert "AUDIENCE" in reactor["mission"]


def test_profile_carries_persona_fields() -> None:
    p = _persona()
    profile = build_profile(p, "reactor")
    assert profile["name"] == p.name
    assert profile["interests"] == p.interests
    assert profile["skepticism"] == p.skepticism


def test_launch_post_text_includes_title() -> None:
    text = launch_post_text("TileMerge", "a sliding puzzle", pitch="merge tiles to win")
    assert "TileMerge" in text
    assert "merge tiles to win" in text
    assert "try it" in text.lower()


def test_launch_post_text_without_pitch() -> None:
    text = launch_post_text("MyApp", "does a thing")
    assert "MyApp" in text
    assert "does a thing" in text


def test_interview_prompt_asks_use_and_share() -> None:
    prompt = interview_prompt()
    assert "would_use" in prompt and "would_share" in prompt


def test_interview_prompt_is_structured_and_scored() -> None:
    # Structured, parseable reply (yes/no + a 0-10 score) so the run summary can
    # aggregate a real distribution, not free text.
    prompt = interview_prompt()
    assert "score:" in prompt
    assert "0-10" in prompt
    # carries the shared calibration rubric
    assert "HOW TO JUDGE" in prompt


def test_rating_rubric_anchors_full_scale_and_is_injected() -> None:
    # The rubric must anchor the whole 0-10 range so the crowd discriminates
    # instead of rating everything 9-10.
    for anchor in ("0-2", "3-4", "5-6", "7-8", "9-10"):
        assert anchor in RATING_RUBRIC
    # and it is actually injected into the hands-on trier mission
    trier = build_profile(_persona(), "trier")
    assert "HOW TO JUDGE" in trier["mission"]


def test_reactor_mission_has_no_finish_trial_rubric() -> None:
    # Reactors don't run finish_trial; the hands-on rating guide belongs to triers.
    reactor = build_profile(_persona(), "reactor")
    assert "HOW TO JUDGE" not in reactor["mission"]


def test_system_template_matches_profile_keys() -> None:
    # system_template needs camel (TextPrompt); skip if not installed.
    pytest.importorskip("camel.prompts", reason="camel-ai not installed")
    from viral_bench.crowd.sim.prompts import system_template

    template = system_template()
    assert set(template.key_words) == set(PROFILE_KEYS)


def test_only_a_server_backed_app_gets_an_account_and_other_users() -> None:
    """A client-side page has no accounts and no other users.

    Telling an agent to sign up and hunt for other people's data on a static
    page invents a failure the app was never asked to avoid -- and 17 of the 25
    briefs are client-side.
    """
    p = _persona()
    client = build_profile(p, "trier", "client-app")["mission"]
    server = build_profile(p, "trier", "full-stack-app")["mission"]
    assert "MAKE AN ACCOUNT" not in client
    assert "YOU ARE NOT ALONE" not in client
    assert "MAKE AN ACCOUNT" in server
    # The identity is the persona's own, so 30 agents are 30 distinct users.
    assert p.username in server
    assert f"{p.username}@example.com" in server


def test_every_trier_is_told_to_publish_its_own_post() -> None:
    """88% of comments hung off the launch post, so the feed was flat.

    A take buried in the founder's thread is not content anyone can find,
    repost or rank -- and with no posts to rank, exposure cannot be earned.
    """
    mission = build_profile(_persona(), "trier")["mission"]
    assert "create_post" in mission
    assert "reload_page" in mission
