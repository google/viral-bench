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

"""Build the OASIS ``AgentGraph`` for one app's crowd trial.

This is where the phase-1 app-interaction toolkit meets OASIS. Each crowd member
becomes a ``SocialAgent`` whose tool set depends on its tier:

* **triers** get the full :class:`AppInteractionToolkit` (drive the app in a
  browser / CLI / bot) *and* the read-only :class:`CodeInspectionToolkit`, so they
  can form a first-hand opinion, plus the platform's social actions;
* **reactors** get only the cheap :class:`CodeInspectionToolkit` plus the social
  actions -- they judge from the announcement, the discussion, and a code skim.

``max_iteration`` is set high enough for triers to chain a full trial (open -> use
-> ``finish_trial``) and then react in a single turn (OASIS's per-step tool-call
budget). A dedicated **founder** account (agent 0) posts the launch and never
gets app tools. The function also returns a **follow plan** (who should follow
whom) that the simulation seeds so the launch and the early-adopters' takes
actually reach feeds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from oasis import ActionType, AgentGraph, SocialAgent, UserInfo

from viral_bench.crowd.interaction.inspect import CodeInspectionToolkit
from viral_bench.crowd.interaction.toolkit import AppInteractionToolkit
from viral_bench.crowd.sim.personas import CrowdSelection, Persona
from viral_bench.crowd.sim.prompts import build_profile, system_template
from viral_bench.crowd.sim.turns import record_budget_exhausted
from viral_bench.crowd.sim_defaults import (
    DEFAULT_MAX_ITERATION_REACTOR,
    DEFAULT_MAX_ITERATION_TRIER,
    DEFAULT_TRIAL_MAX_STEPS,
)

# The viral-relevant action set the crowd can take. Includes the amplification
# actions (repost/quote), discussion (comment/like-comment), the follow signal,
# negatives (dislike/report), discovery (search/refresh), and opting out.
VIRAL_ACTIONS: list[ActionType] = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.QUOTE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.FOLLOW,
    ActionType.DISLIKE_POST,
    ActionType.REPORT_POST,
    ActionType.SEARCH_POSTS,
    ActionType.REFRESH,
    ActionType.DO_NOTHING,
]

# The founder only announces + replies; it never rates content (anti-gaming is
# also enforced by the platform's allow_self_rating=False).
FOUNDER_ACTIONS: list[ActionType] = [
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.DO_NOTHING,
]

FOUNDER_AGENT_ID = 0

#: Default app scope when a build shipped no manifest to declare one.
_WEB = "client-app"

_LOG = logging.getLogger("viral_bench.crowd.agents")


class BudgetAwareSocialAgent(SocialAgent):
    """A ``SocialAgent`` that says so when it runs out of tool-call budget.

    CAMEL enforces ``max_iteration`` by simply ``break``-ing out of its
    tool-calling loop (``chat_agent.py``), with no exception, no flag and no log
    line -- the agent just stops mid-turn and the run looks normal. That silence
    is the whole reason the trier-starvation defect went unnoticed for so long:
    the only symptom was an agent that happened to say nothing, which is also
    what a genuinely unimpressed agent looks like.

    Those two must never be confused. "Had nothing to say" is the signal a
    virality benchmark exists to measure; "was cut off mid-turn" is an
    instrument fault that deletes first-hand opinion from exactly the runs where
    triers engaged most -- i.e. the best apps. The budget arithmetic in
    :mod:`viral_bench.crowd.sim_defaults` is designed so this cannot happen; this
    class is what verifies that at runtime rather than assuming it.
    """

    async def astep(self, *args, **kwargs):  # type: ignore[override]
        response = await super().astep(*args, **kwargs)
        limit = getattr(self, "max_iteration", None)
        if not limit:
            return response
        info = getattr(response, "info", None)
        if not isinstance(info, dict):  # an error, not a response -- nothing to check
            return response
        # CAMEL breaks once iteration_count >= max_iteration, and every iteration
        # runs at least one tool call, so reaching the limit in calls means the
        # turn ended at the ceiling rather than because the agent was finished.
        if len(info.get("tool_calls") or []) >= limit:
            record_budget_exhausted()
            _LOG.warning(
                "agent %s used its entire %s-call budget in one turn, so the "
                "turn was cut off rather than completed. Raise "
                "simulation.social_headroom in config/crowd.yaml -- a truncated "
                "trier is a lost opinion, not a quiet one.",
                getattr(self, "social_agent_id", "?"),
                limit,
            )
        return response


@dataclass
class CrowdAgents:
    """The built agent graph plus everything the simulation needs to drive it."""

    agent_graph: AgentGraph
    founder_id: int
    trier_ids: list[int]
    reactor_ids: list[int]
    # agent_id -> the trier's app-interaction toolkit (owns its trace + browser
    # context; the simulation harvests traces from these and closes them).
    trier_toolkits: dict[int, AppInteractionToolkit]
    persona_by_id: dict[int, Persona]
    tier_by_id: dict[int, str]
    #: Agents holding the app tools who were told NOT to use them unless the
    #: feed convinces them. They are the only measurement of virality this
    #: simulation EARNS rather than computes.
    latecomer_ids: list[int] = field(default_factory=list)
    # (follower_id, followee_id) edges to seed at round 0.
    follow_plan: list[tuple[int, int]] = field(default_factory=list)

    @property
    def crowd_ids(self) -> list[int]:
        return [*self.trier_ids, *self.latecomer_ids, *self.reactor_ids]

    @property
    def hands_on_ids(self) -> list[int]:
        """Everyone who HAS the app tools, whether or not they were told to use them."""
        return [*self.trier_ids, *self.latecomer_ids]


def _founder_agent(app_title: str, *, model, graph: AgentGraph) -> SocialAgent:
    from camel.prompts import TextPrompt

    template = TextPrompt(
        "You are {name}, the maker who just launched {app} on this platform. "
        "You are proud but honest, and you want people to actually try it. "
        "Announce it clearly and reply warmly to people who engage."
    )
    user_info = UserInfo(
        user_name="founder",
        name="The Founder",
        description=f"Maker. Just shipped {app_title}.",
        profile={"name": "The Founder", "app": app_title},
        recsys_type="twitter",
    )
    return SocialAgent(
        agent_id=FOUNDER_AGENT_ID,
        user_info=user_info,
        user_info_template=template,
        model=model,
        agent_graph=graph,
        available_actions=FOUNDER_ACTIONS,
        max_iteration=2,
    )


def persona_bio(persona: Persona) -> str:
    """The public one-liner a crowd member's account shows.

    ``UserInfo.description`` is what OASIS signs an agent up with, and it is what
    lands in ``user.bio``. We never set it, so ``bio`` was NULL in 1,197 of 1,197
    stored rows -- which broke two things at once. The interest-based recommender
    scores candidate posts by cosine similarity between the post and the reader's
    **bio**, so with every bio empty there was nothing to personalise on and every
    agent got an identical feed in 45 of 45 runs. And an account with no bio is an
    account nobody can size up, which is half of deciding whether to trust a
    stranger's recommendation.
    """
    return f"{persona.archetype}. Into {persona.interests}."


def _crowd_agent(
    agent_id: int,
    persona: Persona,
    tier: str,
    *,
    model,
    tools: list,
    max_iteration: int,
    graph: AgentGraph,
    recsys_type: str = "twitter",
    app_type: str = "client-app",
) -> SocialAgent:
    # recsys_type was hard-coded here, so --recsys selected a platform-level
    # recommender while every agent still declared "twitter". All 45 stored runs
    # used twitter regardless of the flag, and the twhin-bert path advertised by
    # the CLI has never once executed.
    user_info = UserInfo(
        user_name=persona.username,
        name=persona.name,
        description=persona_bio(persona),
        profile=build_profile(persona, tier, app_type),
        recsys_type=recsys_type,
    )
    return BudgetAwareSocialAgent(
        agent_id=agent_id,
        user_info=user_info,
        user_info_template=system_template(),
        model=model,
        agent_graph=graph,
        available_actions=VIRAL_ACTIONS,
        tools=tools,
        max_iteration=max_iteration,
    )


def _follow_plan(
    trier_ids: list[int],
    reactor_ids: list[int],
    persona_by_id: dict[int, Persona],
    *,
    top_influencers: int = 3,
    peers_each: int = 3,
) -> list[tuple[int, int]]:
    """Seed a follow graph with hubs AND interest neighbourhoods.

    Three edge classes, each earning its place:

    * **everyone follows the founder**, so the launch reaches every feed;
    * **everyone follows the crowd's biggest accounts**, so a take from a hub
      propagates -- this is the word-of-mouth path;
    * **everyone follows a few people with overlapping interests**, so the graph
      has neighbourhoods instead of being one star. Without this the only route
      between two crowd members is through the founder, and an app cannot spread
      through a community because there are no communities.

    Tier-agnostic on purpose. The previous version wired hub edges only from
    reactors to triers, which was fine when a quarter of the crowd were triers
    and degenerate the moment everyone is one: with no reactors it produced a
    pure star on the founder and not a single crowd-to-crowd edge.

    Deterministic: interest overlap is Jaccard over the persona's interest list,
    ties broken by influence and then username.
    """
    edges: list[tuple[int, int]] = []
    crowd = [*trier_ids, *reactor_ids]
    for agent_id in crowd:
        edges.append((agent_id, FOUNDER_AGENT_ID))

    influencers = sorted(
        crowd, key=lambda i: (-persona_by_id[i].influence, persona_by_id[i].username)
    )[:top_influencers]
    interests = {i: set(persona_by_id[i].interest_list) for i in crowd}
    for agent_id in crowd:
        for inf_id in influencers:
            if agent_id != inf_id:
                edges.append((agent_id, inf_id))
        mine = interests[agent_id]

        def _rank(other: int, mine: set[str] = mine) -> tuple:
            theirs = interests[other]
            union = mine | theirs
            jaccard = len(mine & theirs) / len(union) if union else 0.0
            return (
                -jaccard,
                -persona_by_id[other].influence,
                persona_by_id[other].username,
            )

        peers = [i for i in crowd if i != agent_id and i not in influencers]
        edges.extend((agent_id, p) for p in sorted(peers, key=_rank)[:peers_each])
    return edges


def build_crowd_agents(
    build_id: str,
    selection: CrowdSelection,
    *,
    model,
    app_host,
    app_title: str,
    browser_engine=None,
    container: bool = True,
    max_iteration_trier: int = DEFAULT_MAX_ITERATION_TRIER,
    max_iteration_reactor: int = DEFAULT_MAX_ITERATION_REACTOR,
    recsys_type: str = "twitter",
    trial_max_steps: int = DEFAULT_TRIAL_MAX_STEPS,
    start_wait: float = 90.0,
    app_type: str | None = None,
    min_interactions: int = 1,
    disabled_tools: tuple[str, ...] = (),
    env_notice: bool = True,
    follow_peers: int = 3,
) -> CrowdAgents:
    """Construct the founder + crowd ``AgentGraph`` for ``build_id``.

    Args:
        build_id: The app under test.
        selection: The chosen crowd (triers + reactors).
        model: Shared CAMEL model backend for all agents.
        app_host: Shared :class:`~viral_bench.founder.apphost.AppHost` (one running
            app instance, many agents).
        app_title: The app's display title (for the founder prompt).
        browser_engine: Shared browser for web-app trials (triers only).
        container: Run the app in a container (crowd default).
        max_iteration_trier: Per-step tool-call budget for triers. This is ONE
            budget covering both the hands-on trial and the social actions that
            follow it, and the trial runs first -- so it must be at least
            ``trial_max_steps`` plus enough headroom for the trier to still
            speak. Under the original budget of 10 it was not: triers spending
            >=10 calls on the app averaged 0.05 social actions and 37 of 39 were
            silenced, which for a virality benchmark inverts the measurement.
            The default is now derived (``trial_max_steps + social_headroom``,
            see :mod:`viral_bench.crowd.sim_defaults`) so a maximal trial cannot
            starve the voice; do not pass a bare number here without preserving
            that margin.
        max_iteration_reactor: Per-step tool-call budget for reactors. They run
            no trial, so this is purely social and deliberately unrelated.
        trial_max_steps: Cap on interaction steps within a single trier's trial.
        start_wait: How long to wait for a web app to actually answer HTTP
            before giving up. A cold ``uv run`` boot can take far longer than a
            TCP connect, and handing out the URL early is how triers ended up
            reviewing connection-reset pages.

    Returns:
        A :class:`CrowdAgents` bundle.
    """
    graph = AgentGraph()
    graph.add_agent(_founder_agent(app_title, model=model, graph=graph))

    trier_ids: list[int] = []
    latecomer_ids: list[int] = []
    reactor_ids: list[int] = []
    trier_toolkits: dict[int, AppInteractionToolkit] = {}
    persona_by_id: dict[int, Persona] = {}
    tier_by_id: dict[int, str] = {}

    next_id = FOUNDER_AGENT_ID + 1
    for persona in selection.all:
        tier = selection.tier_of(persona)
        persona_by_id[next_id] = persona
        tier_by_id[next_id] = tier

        if tier in ("trier", "latecomer"):
            toolkit = AppInteractionToolkit(
                build_id,
                app_host=app_host,
                container=container,
                browser_engine=browser_engine,
                max_steps=trial_max_steps,
                start_wait=start_wait,
                app_type=app_type,
                # A latecomer holds the tools and cannot use them until it says,
                # in a turn with no tool in reach, that the feed convinced it.
                locked=(tier == "latecomer"),
                min_interactions=min_interactions,
                disabled_tools=disabled_tools,
                env_notice=env_notice,
            )
            trier_toolkits[next_id] = toolkit
            tools = [
                *toolkit.as_camel_tools(),
                *CodeInspectionToolkit(build_id).as_camel_tools(),
            ]
            agent = _crowd_agent(
                next_id,
                persona,
                tier,
                model=model,
                tools=tools,
                max_iteration=max_iteration_trier,
                graph=graph,
                recsys_type=recsys_type,
                app_type=app_type or _WEB,
            )
            (trier_ids if tier == "trier" else latecomer_ids).append(next_id)
        else:
            tools = CodeInspectionToolkit(build_id).as_camel_tools()
            agent = _crowd_agent(
                next_id,
                persona,
                tier,
                model=model,
                tools=tools,
                max_iteration=max_iteration_reactor,
                graph=graph,
                recsys_type=recsys_type,
                app_type=app_type or _WEB,
            )
            reactor_ids.append(next_id)

        graph.add_agent(agent)
        next_id += 1

    follow_plan = _follow_plan(
        [*trier_ids, *latecomer_ids],
        reactor_ids,
        persona_by_id,
        peers_each=follow_peers,
    )
    return CrowdAgents(
        agent_graph=graph,
        founder_id=FOUNDER_AGENT_ID,
        trier_ids=trier_ids,
        latecomer_ids=latecomer_ids,
        reactor_ids=reactor_ids,
        trier_toolkits=trier_toolkits,
        persona_by_id=persona_by_id,
        tier_by_id=tier_by_id,
        follow_plan=follow_plan,
    )
