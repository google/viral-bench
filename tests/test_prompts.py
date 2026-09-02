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

"""Tests for founder prompt construction."""

from __future__ import annotations

from viral_bench.founder.manifest import MANIFEST_FILENAME
from viral_bench.founder.prompts import (
    DYNAMIC_DONE_SIGNAL,
    RUNTIME_NOTES,
    SHIP_SIGNAL,
    brief_fingerprint,
    build_prompt,
    design_prompt,
    dynamic_continue_prompt,
    dynamic_founder_prompt,
    render_idea,
    team_turn_prompt,
)
from viral_bench.founder.roles import roles_for
from viral_bench.ideas import Idea, load_ideas

CLIENT_IDEA = Idea(
    idea_id="sliding_tile_game",
    title="TileMerge",
    pitch="Slide and merge numbered tiles.",
    problem="People want a quick browser game.",
    target_user="Casual players.",
    core_features=("4x4 grid", "merge tiles", "score"),
    success_criteria="Play a full game with correct merges.",
    allowed_scope="client-app",
    difficulty="easy",
)

# The other half of the scope enum. Both scopes are web apps now, so this is what
# "a different scope" looks like -- the per-scope guidance must still differ.
FULL_STACK_IDEA = Idea(
    idea_id="team_wiki",
    title="TeamWiki",
    pitch="A shared wiki a team can actually edit together.",
    problem="Notes get lost in chat.",
    target_user="Small teams.",
    core_features=("accounts", "shared pages", "history"),
    success_criteria="Two users edit the same page and both see it.",
    allowed_scope="full-stack-app",
    difficulty="medium",
)


def test_render_idea_includes_core_fields() -> None:
    text = render_idea(CLIENT_IDEA)
    assert "TileMerge" in text
    assert "4x4 grid" in text
    assert "client-app" in text
    assert "Play a full game" in text


# -- solo prompts (unchanged baseline) --------------------------------------- #


def test_design_prompt_mentions_design_md_and_runtime() -> None:
    prompt = design_prompt(CLIENT_IDEA)
    assert "DESIGN.md" in prompt
    assert "run/test runtime" in prompt
    assert "TileMerge" in prompt
    # design phase should not ask for code yet
    assert "only" in prompt.lower()


def test_runtime_notes_name_the_provider_neutral_llm_triple() -> None:
    """A built app gets an endpoint, a key and a model id -- never a vendor.

    The brief used to name one vendor's variable and one specific model, so
    every generated app hard-coded the provider the bench happened to be run
    against and could not be re-run against another. The three variables here
    are the ones the runtime actually injects (viral_bench.founder.appenv), so
    they are asserted from that module rather than retyped -- a rename on one
    side without the other silently produces apps that read nothing.
    """
    from viral_bench.founder.appenv import (
        APP_API_KEY_VAR,
        APP_BASE_URL_VAR,
        APP_MODEL_VAR,
    )

    for var in (APP_BASE_URL_VAR, APP_API_KEY_VAR, APP_MODEL_VAR):
        assert var in RUNTIME_NOTES
    assert "OpenAI-compatible" in RUNTIME_NOTES
    # No vendor's own key variable, which is what apps used to be told to read.
    assert "GEMINI_API_KEY" not in RUNTIME_NOTES
    # The key stays server-side (a static page cannot read the runtime env), and
    # the app must still start without one.
    assert "backend" in RUNTIME_NOTES.lower()
    assert "localStorage" in RUNTIME_NOTES  # explicit "don't do this"
    assert "WITHOUT a key" in RUNTIME_NOTES


def test_manifest_rules_require_real_feature_in_manual_steps() -> None:
    # test.manual drives what a crowd trier actually runs, so it must showcase
    # the live feature; a mock/demo mode is only a labelled fallback.
    prompt = build_prompt(CLIENT_IDEA)
    assert "test.manual" in prompt
    assert "REAL headline feature" in prompt
    assert "fallback" in prompt


def test_build_prompt_requires_manifest_and_scope_guidance() -> None:
    prompt = build_prompt(CLIENT_IDEA)
    assert MANIFEST_FILENAME in prompt
    assert "CLIENT-APP" in prompt
    assert '"app_type": "client-app"' in prompt  # example embedded
    assert 'MUST be "client-app"' in prompt


def test_build_prompt_scope_specific_for_full_stack() -> None:
    # Both scopes are web, so the guidance must still say something *different*
    # for a full-stack app -- the durable-state contract is the whole difference.
    prompt = build_prompt(FULL_STACK_IDEA)
    assert "FULL-STACK-APP" in prompt
    assert "VIRALBENCH_DATA_DIR" in prompt
    assert '"app_type": "full-stack-app"' in prompt
    assert MANIFEST_FILENAME in prompt
    assert "CLIENT-APP" not in prompt


# -- team round-table prompts ------------------------------------------------ #


def _teammates() -> list[tuple[str, str]]:
    return [("Architect", "Architecture"), ("Implementer", "Core implementation")]


def test_team_turn_prompt_is_role_and_round_aware() -> None:
    architect = roles_for(4)[0]
    prompt = team_turn_prompt(
        CLIENT_IDEA,
        architect,
        round_index=1,
        max_rounds=3,
        agent_index=1,
        n_agents=4,
        teammates=_teammates(),
        is_first_turn=True,
        collaboration_brief="COLLAB-BRIEF",
    )
    assert "Architect" in prompt
    assert "agent 1 of 4" in prompt
    assert "round 1 of at most 3" in prompt
    assert "FIRST turn" in prompt
    # collaboration medium is injected verbatim from the toolset
    assert "COLLAB-BRIEF" in prompt
    # teammates roster present
    assert "Implementer" in prompt


def test_team_turn_prompt_resuming_language_when_not_first() -> None:
    impl = roles_for(4)[1]
    prompt = team_turn_prompt(
        CLIENT_IDEA,
        impl,
        round_index=2,
        max_rounds=3,
        agent_index=2,
        n_agents=4,
        teammates=_teammates(),
        is_first_turn=False,
        collaboration_brief="x",
    )
    assert "resuming your OWN context" in prompt
    assert "FIRST turn" not in prompt


def test_non_qa_turn_has_no_manifest_or_ship_signal() -> None:
    impl = roles_for(4)[1]  # implementer, does not own QA
    prompt = team_turn_prompt(
        CLIENT_IDEA,
        impl,
        round_index=1,
        max_rounds=3,
        agent_index=2,
        n_agents=4,
        teammates=_teammates(),
        is_first_turn=True,
        collaboration_brief="x",
    )
    assert SHIP_SIGNAL not in prompt
    assert MANIFEST_FILENAME not in prompt
    assert "coordinate through the collaboration channel" in prompt


def test_qa_turn_has_manifest_contract_and_ship_signal() -> None:
    qa = roles_for(4)[-1]  # qa_finisher owns QA
    prompt = team_turn_prompt(
        CLIENT_IDEA,
        qa,
        round_index=3,
        max_rounds=3,
        agent_index=4,
        n_agents=4,
        teammates=_teammates(),
        is_first_turn=False,
        collaboration_brief="x",
    )
    assert SHIP_SIGNAL in prompt
    assert MANIFEST_FILENAME in prompt
    assert "REQUIRED DELIVERABLES" in prompt
    assert 'MUST be "client-app"' in prompt


def test_qa_holds_ship_below_min_rounds() -> None:
    qa = roles_for(4)[-1]
    prompt = team_turn_prompt(
        CLIENT_IDEA,
        qa,
        round_index=1,
        max_rounds=4,
        min_rounds=3,
        agent_index=4,
        n_agents=4,
        teammates=_teammates(),
        is_first_turn=True,
        collaboration_brief="x",
    )
    # below the floor: no ship token offered, told to keep improving
    assert "DO NOT SHIP YET" in prompt
    assert SHIP_SIGNAL not in prompt


def test_qa_may_ship_at_min_rounds() -> None:
    qa = roles_for(4)[-1]
    prompt = team_turn_prompt(
        CLIENT_IDEA,
        qa,
        round_index=3,
        max_rounds=4,
        min_rounds=3,
        agent_index=4,
        n_agents=4,
        teammates=_teammates(),
        is_first_turn=False,
        collaboration_brief="x",
    )
    assert "SHIP DECISION" in prompt
    assert SHIP_SIGNAL in prompt


def test_team_turn_prompt_includes_scope_guidance() -> None:
    qa = roles_for(4)[-1]
    prompt = team_turn_prompt(
        FULL_STACK_IDEA,
        qa,
        round_index=1,
        max_rounds=2,
        agent_index=4,
        n_agents=4,
        teammates=_teammates(),
        is_first_turn=True,
        collaboration_brief="x",
    )
    assert "FULL-STACK-APP" in prompt


# -- brief fingerprint: the corpus depends on this not moving ---------------- #


#: Fingerprints of four real ideas, captured before the dynamic mode was added.
#:
#: `brief_fingerprint` keys every build on disk: `build_fleet` refuses to count a
#: build whose fingerprint no longer matches its idea, so a change to the shared
#: prompt blocks (RUNTIME_NOTES, the scope guidance, the deliverables block, the
#: solo design/build prompts) silently retires 400+ builds and hundreds of
#: dollars of fleet. That is sometimes the right thing to do -- the web-dev pivot
#: did exactly that on purpose -- but it must never happen as a side effect of
#: adding a founder mode, so these are pinned.
#:
#: If you changed a prompt DELIBERATELY: re-capture these, and expect to rebuild
#: the fleet.
#:
#: Re-captured when the runtime notes stopped naming one vendor's key variable
#: and one specific model, and started handing apps the provider-neutral
#: VIRALBENCH_APP_LLM_* triple instead. That is a deliberate change to a shared
#: block: every build made under the old brief was told to call a model this
#: bench no longer presumes anyone has, so those builds are genuinely not
#: comparable with new ones and retiring them is the point.
PINNED_FINGERPRINTS = {
    "ai_room_redesign": "f1c8ab847da3da6c",
    "browser_api_client": "cea44007bea6b777",
    "code_screenshot_studio": "872ac77abaabe621",
    "collaborative_table": "c59e1755b55582e2",
}


def test_brief_fingerprints_have_not_moved() -> None:
    ideas = {idea.idea_id: idea for idea in load_ideas()}
    got = {
        name: brief_fingerprint(ideas[name])
        for name in PINNED_FINGERPRINTS
        if name in ideas
    }
    assert got == PINNED_FINGERPRINTS


# -- dynamic orchestrator prompts -------------------------------------------- #


def test_dynamic_prompt_carries_the_shared_contract() -> None:
    prompt = dynamic_founder_prompt(
        CLIENT_IDEA, app_dir="/w/app", agents_dir="/w/.opencode/agents", max_turns=3
    )
    assert MANIFEST_FILENAME in prompt
    assert RUNTIME_NOTES in prompt
    assert render_idea(CLIENT_IDEA) in prompt
    assert DYNAMIC_DONE_SIGNAL in prompt


def test_dynamic_prompt_states_the_delegation_mechanics_truthfully() -> None:
    """Each of these is a live-verified property of opencode 1.17.14. A brief
    that promises a capability the harness lacks makes the model fight the tool
    and reads out as bad orchestration."""
    prompt = dynamic_founder_prompt(
        CLIENT_IDEA, app_dir="/w/app", agents_dir="/w/.opencode/agents", max_turns=3
    )
    assert "/w/.opencode/agents/<name>.md" in prompt
    assert "CONCURRENTLY" in prompt  # several task calls in one message
    assert "task_id" in prompt  # a subagent can be resumed
    assert "NEXT turn" in prompt  # self-defined agents load next turn


def test_dynamic_prompt_prescribes_no_process() -> None:
    """Guards the one thing that would quietly turn this back into the team arm.

    Only the role names are banned outright. "roles" and "rounds" DO appear -- in
    the sentence saying there are none -- and that sentence is the load-bearing
    part of the brief, so it is asserted rather than forbidden.
    """
    prompt = dynamic_founder_prompt(
        FULL_STACK_IDEA, app_dir="/w/app", agents_dir="/a", max_turns=3
    ).lower()
    for banned in ("architect", "implementer", "designer", "qa & finisher"):
        assert banned not in prompt
    assert "no assigned roles, no phases, no rounds" in " ".join(prompt.split())


def test_dynamic_prompt_uses_scope_guidance() -> None:
    prompt = dynamic_founder_prompt(
        FULL_STACK_IDEA, app_dir="/w/app", agents_dir="/a", max_turns=3
    )
    assert "FULL-STACK-APP" in prompt


def test_dynamic_continue_prompt_states_the_gap_and_nothing_else() -> None:
    """The nudge is the one place the harness could start doing the
    orchestrating; it must stay a statement of the contract, not advice."""
    text = dynamic_continue_prompt(
        turn_index=2, max_turns=3, gaps=["There is no `viralbench.json` yet."]
    )
    assert "viralbench.json" in text
    assert "turn 2 of at most 3" in text
    assert DYNAMIC_DONE_SIGNAL in text
    lowered = text.lower()
    for banned in ("subagent", "delegate", "browser", "test", "architect"):
        assert banned not in lowered


def test_dynamic_continue_prompt_warns_on_the_last_turn() -> None:
    assert "LAST turn" in dynamic_continue_prompt(turn_index=3, max_turns=3, gaps=[])
    assert "LAST turn" not in dynamic_continue_prompt(
        turn_index=2, max_turns=3, gaps=[]
    )


def test_dynamic_prompt_pins_the_app_root() -> None:
    """The only mode that names a path outside the app dir must also name the
    app dir. Without it, a live build wrote its manifest one level up on all
    three turns and was recorded `manifest_missing`."""
    prompt = dynamic_founder_prompt(
        CLIENT_IDEA, app_dir="/w/app", agents_dir="/w/.opencode/agents", max_turns=3
    )
    assert "`/w/app` is the app" in prompt
    assert MANIFEST_FILENAME in prompt
