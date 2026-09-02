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

"""Tests for the config loader (viral_bench.config)."""

from __future__ import annotations

from viral_bench import config


def test_the_tracked_config_ships_no_model_for_any_stage() -> None:
    """A checked-in default model is a provider account chosen on the user's
    behalf, so every stage ships empty and ``viral-bench init`` fills them in.

    Read straight out of the tracked files rather than through
    :func:`config.stage_model`, which would also see a developer's own
    (gitignored) ``config/local.yaml`` and pass for the wrong reason.
    """
    assert config._get("founder.yaml", ("model", "id"), "unset") == ""
    assert config._get("founder.yaml", ("grader", "model"), "unset") == ""
    assert config._get("crowd.yaml", ("simulation", "model", "id"), "unset") == ""
    assert config._get("score.yaml", ("autorater", "model"), "unset") == ""


def test_a_configured_model_keeps_its_provider_prefix(monkeypatch) -> None:
    # The prefix is the whole selector -- it decides which API is called and
    # which credential is read -- so it must survive the config round trip
    # rather than being split off as a "provider" field somewhere.
    monkeypatch.setattr(
        config, "_load", lambda name: {"model": {"id": "openai/gpt-test"}}
    )
    assert config.founder_model_id("fallback") == "openai/gpt-test"


def test_an_unset_stage_falls_back_to_the_callers_default() -> None:
    # Empty in YAML means "not chosen yet", never a literal empty model id, so
    # callers that supply a default still get theirs.
    assert config.founder_model_id("fallback") == "fallback"
    assert config.crowd_model_id("fallback") == "fallback"
    # And with no default, empty is the honest answer.
    assert config.crowd_model_id() == ""


def test_vertex_target_from_yaml() -> None:
    # The location is a real, loaded value; the project ships empty because
    # ViralBench presumes no cloud account. Vertex is one optional provider
    # among many now, not the path everything goes through.
    assert config.vertex_location("fallback") == "global"
    assert config._get("founder.yaml", ("vertex", "project"), "unset") == ""


def test_crowd_knobs_from_yaml() -> None:
    assert config.crowd_agents(0) == 30
    # -1 is not a fallback here: it is the shipped value, and it means "every
    # agent in the crowd tries the app first-hand".
    assert config.crowd_triers(0) == -1
    assert config.crowd_rounds(0) == 3
    assert config.crowd_feed("max_rec_posts", 0) == 20


def test_missing_file_falls_back_to_default() -> None:
    # A file that does not exist must never raise -- always the code default.
    assert config._get("does_not_exist.yaml", ("a", "b"), "dflt") == "dflt"


def test_missing_key_falls_back_to_default() -> None:
    assert config._get("founder.yaml", ("nope", "missing"), 42) == 42


def test_as_int_is_defensive() -> None:
    assert config._as_int("7", 0) == 7
    assert config._as_int("notanint", 3) == 3
    assert config._as_int(None, 5) == 5


def test_code_defaults_reflect_config() -> None:
    # The wired constants should equal the YAML-driven values, not a literal
    # copy of them. `DEFAULT_MODEL` now lives on the harness (the founder's
    # model is a config question, not a registry question) and both model
    # constants are whatever the config says -- including "" when nothing has
    # been chosen, which is the shipped state.
    from viral_bench.crowd import sim_defaults
    from viral_bench.founder import harness

    assert harness.DEFAULT_MODEL == config.founder_model_id("")
    assert sim_defaults.DEFAULT_MODEL == config.crowd_model_id("")
    assert sim_defaults.DEFAULT_AGENTS == 30


# -- no artificial token ceilings -------------------------------------------- #
#
# These caps have been reintroduced twice (2048, then 8192), each time as a guess
# nobody could evaluate because truncation was invisible. They are gone now, and
# these tests exist so they cannot come back silently. If you are deliberately
# adding a cap, change the config -- do not just edit the constant.


def test_crowd_sends_no_output_token_cap_by_default() -> None:
    from viral_bench.crowd import sim_defaults
    from viral_bench.crowd.sim.model import CROWD_MAX_TOKENS

    assert config.crowd_max_tokens(None) is None
    assert sim_defaults.DEFAULT_MAX_TOKENS is None
    assert CROWD_MAX_TOKENS is None


def test_crowd_max_tokens_knob_is_live_not_decorative() -> None:
    # The whole point of the crowd.yaml contract ("EVERY knob in this file
    # actually takes effect"): a set value must survive to the caller. This used
    # to load into sim_defaults.DEFAULT_MAX_TOKENS, which nothing imported.
    assert config.crowd_max_tokens(4096) == 4096
    # Non-positive and null both mean "no cap", never a literal 0.
    assert config.crowd_max_tokens(0) is None


def test_founder_timeouts_are_loaded_from_yaml() -> None:
    # founder.yaml `timeouts_seconds:` was dead config: the real values were
    # hardcoded in harness.py, so raising them here did nothing at all.
    from viral_bench.founder import harness

    assert harness._DESIGN_TIMEOUT_S == config.founder_timeout("design", None)
    assert harness._BUILD_TIMEOUT_S == config.founder_timeout("build", None)
    assert harness._TEAM_TURN_TIMEOUT_S == config.founder_timeout("team_turn", None)
    # A backstop, not a budget: it must be far above a real turn, and nullable.
    assert harness._BUILD_TIMEOUT_S is None or harness._BUILD_TIMEOUT_S >= 3600
    assert config.founder_timeout("design", None) is not None  # sanity: key exists
    assert config.founder_timeout("nonexistent", None) is None


def test_founder_collaboration_block_is_loaded() -> None:
    # Also dead config until now, which let founder.yaml claim
    # `browser_tools: false` while the code defaulted to True.
    assert config.founder_collaboration("agents", -1) == 1
    assert config.founder_collaboration("rounds", -1) == 3
    assert config.founder_collaboration("browser_tools", None) is True


def test_score_minimums_are_loaded_from_yaml() -> None:
    # score.yaml documented these as "NOT YET WIRED" for a long time.
    from viral_bench.score.viralscore import CALIBRATED_CROWD_SIZE, MIN_INTERVIEWS

    assert MIN_INTERVIEWS == config.score_minimum("interviews", -1)
    # 30, not the discredited 50 the YAML used to carry (README, docs).
    assert CALIBRATED_CROWD_SIZE == 30
