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

"""Run one app's crowd trial: seed the launch, run rounds, emit artifacts.

This is the orchestrator. For a single build it: starts a shared running instance
of the app, builds the founder + crowd agents, seeds a follow graph and the launch
post, runs a few rounds where triers try the app and everyone reacts, optionally
interviews the crowd, and writes the run's artifacts (the OASIS SQLite DB, the
triers' interaction traces, an action log, and a run summary).

Each call is fully self-contained and isolated -- a fresh database, a fresh agent
graph with fresh memory, and a fresh app instance -- so scoring one app never
leaks into another ("wipe & repeat"). Turning these artifacts into a ViralScore is
a later stage. This module's job is to produce a rich, well-recorded social run.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from oasis import ActionType, LLMAction, ManualAction

from viral_bench.crowd.interaction.browser import (
    BrowserConfig,
    BrowserEngine,
    browser_available,
)
from viral_bench.crowd.interaction.session import manifest_or_none
from viral_bench.crowd.sim.agents import build_crowd_agents
from viral_bench.crowd.sim.model import crowd_model, crowd_transport, dummy_model
from viral_bench.crowd.sim.personas import load_personas, select_crowd
from viral_bench.crowd.sim.platform import PlatformConfig, make_env
from viral_bench.crowd.sim.prompts import (
    consider_prompt,
    interview_prompt,
    launch_post_text,
)
from viral_bench.crowd.sim.trajectory import (
    collect_trajectories,
    write_trajectories,
)
from viral_bench.crowd.sim.turns import reset_turn_stats, turn_stats
from viral_bench.crowd.sim.verdicts import (
    assess_run_health,
    consideration_decisions,
    conversion_stats,
    expected_rounds,
    interview_verdicts,
    trier_verdicts,
    undeliverable_validity,
)
from viral_bench.crowd.sim_defaults import (
    CROWD_ARCH_VERSION,
    DEFAULT_ENV_NOTICE,
    DEFAULT_FOLLOW_PEERS,
    DEFAULT_MAX_ITERATION_REACTOR,
    DEFAULT_MAX_ITERATION_TRIER,
    DEFAULT_MIN_INTERACTIONS,
    DEFAULT_MODEL,
    DEFAULT_START_WAIT,
    DEFAULT_TRIAL_MAX_STEPS,
)
from viral_bench.founder.apphost import AppHost
from viral_bench.founder.build import get_idea, load_build_record
from viral_bench.founder.workspace import reset_build_data

_WEB = "client-app"
_FULL_STACK = "full-stack-app"
_LOG = logging.getLogger("viral_bench.crowd.simulation")


def arch_version(variant: str = "") -> str:
    """The architecture version to stamp on a run, with any variant suffix."""
    return f"{CROWD_ARCH_VERSION}+{variant}" if variant else CROWD_ARCH_VERSION


@dataclass
class SimulationConfig:
    """Everything that defines one crowd run (frozen into the run summary)."""

    build_id: str
    out_dir: str
    n_agents: int = 8
    n_triers: int = 5
    #: Agents who hold the app tools but were told not to use them unless the
    #: feed convinces them. Their conversion rate is the only virality number
    #: this simulation earns instead of computing (see personas.CrowdSelection).
    n_latecomers: int = 0
    rounds: int = 4
    #: ``provider/model``, from config/crowd.yaml or --model. Empty when nothing
    #: is configured, which fails when the model is built with a message naming
    #: what to set -- there is no model to fall back to and guessing one would
    #: bill an account nobody chose.
    model_id: str = DEFAULT_MODEL
    #: Crowd sampling temperature. Higher gives livelier, more varied
    #: social behaviour. Lower makes the measurement more repeatable.
    #: Swept during score calibration to find the reliability sweet spot.
    temperature: float = 0.7
    #: Output-token cap per LLM call. ``None`` sends no cap, so the model's own
    #: maximum applies -- which also leaves deliberation untruncated, since
    #: recent Gemini charges thinking tokens against this same budget.
    max_tokens: int | None = None
    recsys_type: str = "twhin-bert"
    semaphore: int = 4
    seed: int = 0
    container: bool = True
    interview: bool = True
    no_llm: bool = False  # scripted wiring smoke: ManualActions only, no LLM spend
    personas_file: str | None = None
    #: Defaults come from sim_defaults so config/crowd.yaml reaches them,
    #: and so the trier budget stays TIED to trial_max_steps rather than being a
    #: second hand-picked number that can silently fall below it (see
    #: sim_defaults.DEFAULT_SOCIAL_HEADROOM).
    max_iteration_trier: int = DEFAULT_MAX_ITERATION_TRIER
    max_iteration_reactor: int = DEFAULT_MAX_ITERATION_REACTOR
    trial_max_steps: int = DEFAULT_TRIAL_MAX_STEPS
    start_wait: float = DEFAULT_START_WAIT
    #: Run the verify_code validity gate (builds / runs / does_what_it_claims)
    #: and record it in the run summary. The ViralScore caps a broken app's
    #: score, so a scored run needs this. The --no-llm wiring smokes can skip it.
    validity_gate: bool = True
    #: How hard the crowd model is allowed to think per call ("low" | "high" |
    #: None for the model's own default). recent Gemini thinks by default, which is
    #: a 13x latency difference on a trivial call and the dominant cost of a
    #: sweep -- and how much deliberation a good judge needs is a real question
    #: about the instrument.
    thinking_level: str | None = None
    #: Successful app interactions required before a verdict is accepted
    #: without push-back (see sim_defaults.DEFAULT_MIN_INTERACTIONS).
    min_interactions: int = DEFAULT_MIN_INTERACTIONS
    #: Verbs withheld from the crowd, by tool name (an ablation lever).
    disabled_tools: tuple[str, ...] = ()
    #: Tell agents which API keys the environment provides, on open.
    env_notice: bool = DEFAULT_ENV_NOTICE
    #: Interest-neighbours each agent follows (0 = hub-and-founder only).
    follow_peers: int = DEFAULT_FOLLOW_PEERS
    #: Ranked feed size, which is also OASIS's personalisation threshold.
    feed_max_posts: int = 0
    #: Name of a deliberate instrument variation (an ablation), appended to the
    #: recorded crowd architecture version as ``11+triers8``.
    #:
    #: An ablation IS a different instrument, so its runs must never pool with
    #: the main corpus -- and the corpus already filters on the architecture
    #: version, so tagging the version is the whole mechanism. Without this,
    #: running "the same sweep with 8 triers" to settle an argument would
    #: silently contaminate the sweep the argument is about.
    variant: str = ""


@dataclass
class SimulationResult:
    """Outcome + pointers to the artifacts a run produced."""

    build_id: str
    out_dir: str
    db_path: str
    summary_path: str
    ok: bool
    app_type: str = ""
    n_agents: int = 0
    n_triers: int = 0
    rounds_run: int = 0
    launch_post_id: int | None = None
    engagement: dict = field(default_factory=dict)
    trace_paths: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    error: str | None = None
    error_traceback: str | None = None
    #: How the interview was collected {mode, interviewed, failed}.
    interview_stats: dict | None = None
    #: All-agent reasoning trails (the agentic autorater's input).
    trajectories_path: str | None = None
    #: The founder shipped no usable ``viralbench.json``, so there was nothing
    #: for the crowd to launch. Recorded so a floor score can be traced back to
    #: a delivery failure rather than to an app users merely disliked.
    undeliverable: bool = False
    #: Which named sweep the build under test belongs to.
    #:
    #: Copied from the build record at run time, for the same reason
    #: ``crowd_arch_version`` and ``crowd_transport`` are stamped: record what
    #: produced a number, at the time, because it cannot be recovered afterwards.
    #: Without it, "which runs belong to the final sweep" is answerable only by
    #: joining every run back to a build list that may since have moved.
    cohort: str = ""
    #: The app is documented and its code is there, but no instance of it could
    #: be started, so no agent ever saw it.
    #:
    #: A DISTINCT outcome from both ``undeliverable`` and a disliked app, and it
    #: had to become one. This state used to write nothing at all: every agent
    #: trial re-attempted the start, the run produced no ``run_summary.json``, and
    #: it died at the wall clock -- so 26 of 1,000 builds on one sweep were
    #: indistinguishable from cells never tried and silently left the denominator.
    #: Worse, the usual cause was the harness's own packaging (a clone
    #: materialized without its dependencies), so a harness bug read as a model
    #: that ships broken apps.
    app_start_failed: bool = False
    #: Why, in the container's own words.
    app_start_detail: str | None = None
    #: Who is at fault: ``"harness"`` or ``"app"`` (``""`` when it started fine).
    #:
    #: This decides the SCORING, so it is recorded rather than re-derived later
    #: from text that may not survive. An app the harness shipped without its
    #: dependencies is excluded -- scoring a model down for the harness's
    #: packaging manufactures a capability difference. An app that will not
    #: start because its own source has a syntax error is FLOORED like any other
    #: broken app: the crowd did everything right, nobody could use the thing,
    #: and noticing that is what the benchmark is for.
    app_start_fault: str = ""
    #: {builds, runs, does_what_it_claims, detail} from the validity gate, or
    #: None when it was skipped. A broken app cannot be viral, so the score
    #: stage reads this before trusting any engagement.
    validity: dict | None = None
    #: Model turn accounting: how many turns the crowd model could not complete,
    #: and how many of those were rate limits. A skipped turn looks exactly like
    #: an agent choosing to do nothing.
    turn_stats: dict = field(default_factory=dict)
    #: Why this run is (or is not) usable: ``{"ok": bool, "failures": [...]}``.
    #: ``ok`` used to mean only "the orchestrator did not raise", so runs that
    #: lost half their rounds and every single interview still reported success
    #: and exited 0.
    health: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass
class BuildUnderTest:
    """Everything the crowd needs to present one build, deliverable or not.

    ``manifest is None`` means the founder shipped no usable ``viralbench.json``
    -- the file that says how to install, start and test the app. Such builds
    used to be dropped from the crowd stage entirely, so a model that failed to
    ship a launch contract disappeared from the denominator instead of being
    marked down, while the other model's bad-but-runnable apps stayed in and
    dragged its average down. Now they are simulated like anything else and the
    crowd discovers what a real user would: there is nothing to run.
    """

    record: object
    manifest: object | None
    app_type: str
    title: str
    summary: str
    pitch: str | None

    @property
    def undeliverable(self) -> bool:
        return self.manifest is None


def _load_build(build_id: str) -> BuildUnderTest:
    record = load_build_record(build_id)
    manifest = manifest_or_none(build_id)
    idea = None
    try:
        idea = get_idea(record.idea_id)
    except Exception:  # noqa: BLE001 - idea metadata is a nicety, not a contract
        idea = None

    if manifest is not None:
        return BuildUnderTest(
            record=record,
            manifest=manifest,
            app_type=manifest.app_type,
            title=manifest.title,
            summary=manifest.summary,
            pitch=getattr(idea, "pitch", None),
        )

    # No manifest -> no title, no summary and no declared app type from the
    # founder. Fall back to the IDEA's own metadata, which the harness supplies
    # and which is identical for both models on a given idea. That is
    # deliberately generous: the build gets a clean pitch it did not write, so
    # this biases against finding a difference rather than for one.
    return BuildUnderTest(
        record=record,
        manifest=None,
        app_type=getattr(idea, "allowed_scope", None) or "client-app",
        title=getattr(idea, "name", None) or record.idea_id,
        summary=getattr(idea, "pitch", "") or "",
        pitch=getattr(idea, "pitch", None),
    )


def _round_action_log(db_path: str) -> list[dict]:
    """Read the OASIS action trace into a compact, attributable list of rows."""
    rows: list[dict] = []
    try:
        con = sqlite3.connect(db_path)
        cur = con.execute(
            "SELECT user_id, created_at, action, info FROM trace ORDER BY created_at"
        )
        for user_id, created_at, action, info in cur.fetchall():
            rows.append(
                {
                    "user_id": user_id,
                    "created_at": created_at,
                    "action": action,
                    "info": info,
                }
            )
        con.close()
    except sqlite3.Error:
        pass
    return rows


def _engagement_snapshot(db_path: str, launch_post_id: int | None) -> dict:
    """Read the raw engagement signals scoring will later build on."""
    snap: dict = {}
    try:
        con = sqlite3.connect(db_path)

        def scalar(sql: str, *args) -> int:
            return con.execute(sql, args).fetchone()[0]

        snap["posts"] = scalar("SELECT COUNT(*) FROM post")
        snap["likes"] = scalar("SELECT COUNT(*) FROM like")
        snap["dislikes"] = scalar("SELECT COUNT(*) FROM dislike")
        snap["comments"] = scalar("SELECT COUNT(*) FROM comment")
        snap["reposts"] = scalar(
            "SELECT COUNT(*) FROM post WHERE original_post_id IS NOT NULL"
        )
        snap["follows"] = scalar("SELECT COUNT(*) FROM follow")
        snap["reports"] = scalar("SELECT COUNT(*) FROM report")

        # Exposure + per-capita participation. Raw counts scale with crowd size
        # and are heavy-tailed, so the ViralScore reads DISTINCT-ACTOR fractions
        # of the exposed audience instead -- bounded in [0,1] and directly
        # comparable between an 8-agent dev run and a 50-agent scored run.
        snap["reach"] = {
            "exposed_agents": scalar("SELECT COUNT(DISTINCT user_id) FROM rec"),
            "impressions": scalar("SELECT COUNT(*) FROM rec"),
            "actors_liked": scalar("SELECT COUNT(DISTINCT user_id) FROM like"),
            "actors_reposted": scalar(
                "SELECT COUNT(DISTINCT user_id) FROM post "
                "WHERE original_post_id IS NOT NULL"
            ),
            "actors_commented": scalar("SELECT COUNT(DISTINCT user_id) FROM comment"),
            "actors_negative": scalar(
                "SELECT COUNT(DISTINCT user_id) FROM ("
                "  SELECT user_id FROM dislike UNION SELECT user_id FROM report"
                ")"
            ),
        }
        # Cascade shape: does the conversation live only on the launch post, or
        # does derived content earn its own engagement (true multi-generation
        # spread)? Reported for calibration, and weighted only once non-degenerate.
        secondary = scalar(
            "SELECT COALESCE(SUM(num_likes + num_shares + num_dislikes), 0) "
            "FROM post WHERE original_post_id IS NOT NULL"
        ) + scalar(
            "SELECT COUNT(*) FROM comment WHERE post_id != ?", launch_post_id or -1
        )
        primary = scalar(
            "SELECT COALESCE(SUM(num_likes + num_shares + num_dislikes), 0) "
            "FROM post WHERE original_post_id IS NULL"
        ) + scalar(
            "SELECT COUNT(*) FROM comment WHERE post_id = ?", launch_post_id or -1
        )
        late = scalar("SELECT COUNT(*) FROM trace WHERE created_at > 1")
        total_acts = scalar("SELECT COUNT(*) FROM trace")
        snap["cascade"] = {
            "primary_engagement": primary,
            "secondary_engagement": secondary,
            "secondary_share": round(secondary / (primary + secondary), 4)
            if (primary + secondary)
            else 0.0,
            "late_action_share": round(late / total_acts, 4) if total_acts else 0.0,
        }
        # Exposure shape: does the recommender rank at all, or does every
        # agent see an identical feed? Two things depend on this. A frozen rec
        # table predicted total interview loss with 9/9 precision and recall in
        # the stored corpus, so it is a live health signal. And "a better app
        # earns more reach" is the mechanism virality is supposed to run on --
        # if every feed is identical there is no such mechanism, and no amount
        # of scoring rework will find one.
        try:
            feeds = con.execute(
                "SELECT user_id, GROUP_CONCAT(post_id) FROM rec GROUP BY user_id"
            ).fetchall()
            sizes = [len((f[1] or "").split(",")) if f[1] else 0 for f in feeds]
            distinct_feeds = len({f[1] for f in feeds})
            snap["exposure"] = {
                "agents_with_feed": len(feeds),
                "distinct_feeds": distinct_feeds,
                # 1.0 means every agent saw exactly the same posts: the
                # recommender is a pass-through and reach cannot be earned.
                "uniform": bool(feeds) and distinct_feeds == 1,
                "min_feed": min(sizes) if sizes else 0,
                "max_feed": max(sizes) if sizes else 0,
                "mean_feed": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
            }
        except sqlite3.Error:
            snap["exposure"] = {"error": "rec table unreadable"}

        if launch_post_id is not None:
            row = con.execute(
                "SELECT num_likes, num_dislikes, num_shares, num_reports "
                "FROM post WHERE post_id = ?",
                (launch_post_id,),
            ).fetchone()
            if row:
                snap["launch_post"] = {
                    "post_id": launch_post_id,
                    "num_likes": row[0],
                    "num_dislikes": row[1],
                    "num_shares": row[2],
                    "num_reports": row[3],
                    "num_comments": scalar(
                        "SELECT COUNT(*) FROM comment WHERE post_id = ?",
                        launch_post_id,
                    ),
                }
        con.close()
    except sqlite3.Error:
        pass
    return snap


#: How ``verify_code`` words a failure to START, as opposed to a failure to
#: BEHAVE. The two are the same ``runs=False`` and must not be treated alike: an
#: app that starts and serves a 500 is the model's result, while an app that never
#: starts is usually the harness's packaging and carries no information about
#: the model.
_GATE_START_FAILURE = "start failed:"


def _start_fault(detail: str | None) -> str:
    """``"harness"`` or ``"app"`` for a start failure, from its own error text."""
    from viral_bench.founder.runtime import HARNESS_CLASSES, classify_start_failure

    return (
        "harness" if classify_start_failure(detail or "") in HARNESS_CLASSES else "app"
    )


def _gate_says_it_never_started(validity: dict | None) -> bool:
    if not isinstance(validity, dict) or validity.get("gate_errored"):
        return False
    return validity.get("runs") is False and _GATE_START_FAILURE in str(
        validity.get("detail", "")
    )


async def _run_validity_gate(config: SimulationConfig) -> dict:
    """Run ``verify_code`` off-thread and return its dict (never raises).

    A gate that errors out must not abort the crowd run: the failure is recorded
    so the scoring stage can treat the run as unverified rather than silently
    assuming the app works.
    """
    import asyncio

    from viral_bench.founder.verify import verify_code

    try:
        res = await asyncio.to_thread(
            verify_code, config.build_id, container=config.container
        )
        return res.as_dict()
    except Exception as exc:  # noqa: BLE001 - gate failure must not kill the run
        # None, not False. False means "the app was checked and is dead", which
        # fires the 0.2x broken-app multiplier and prints "app failed to build or
        # start" -- a verdict about the app, from an error in the harness.
        # None is the sentinel validity_gate() already treats as unverified: no
        # discount, surfaced as a confidence warning instead.
        return {
            "build_id": config.build_id,
            "builds": None,
            "runs": None,
            "does_what_it_claims": None,
            "gate_errored": True,
            "detail": f"validity gate errored: {type(exc).__name__}: {exc}",
        }


async def _safe_step(env, actions: dict, label: str) -> bool:
    """Run one ``env.step`` and report success instead of propagating.

    OASIS intermittently raises on a malformed agent action (observed:
    ``KeyError: 'post_id'`` while rendering a feed containing quote posts). At
    30-50 agents a scored run is expensive, so one bad action must not throw
    away everything simulated so far.
    """
    try:
        await env.step(actions)
        return True
    except Exception as exc:  # noqa: BLE001 - a faulty step must not abort the run
        _LOG.warning(
            "%s failed (%s: %s); continuing with the signal collected so far.",
            label,
            type(exc).__name__,
            str(exc).splitlines()[0][:160],
        )
        return False


def _answered_agents(db_path: str, crowd) -> set[int]:
    """Agent ids whose interview reply PARSED into a verdict."""
    return {
        row["agent_id"]
        for row in (interview_verdicts(db_path, crowd).get("per_agent") or [])
    }


async def _run_interview(env, crowd, db_path: str, *, repairs: int = 2) -> dict:
    """Interview the whole crowd, then re-ask whoever did not answer.

    The interview is where every agent states, independently, whether it would
    use and share the app -- the primary input to the ViralScore. A step that
    "succeeds" is not an answer: a turn the model failed becomes a synthetic
    "(no response)", which parses to nothing and is dropped. Under load those
    losses concentrate, and they are not random -- they land hardest on the runs
    that generated the most discussion, i.e. the best apps.

    So the batch is followed by up to ``repairs`` targeted passes over exactly
    the agents with no parseable verdict. Re-asking is safe: the aggregation
    keeps the first *successful* parse per agent, so a repaired answer replaces
    nothing and a duplicate cannot double-weight a persona.
    """
    prompt = interview_prompt()

    def _action():
        return ManualAction(
            action_type=ActionType.INTERVIEW, action_args={"prompt": prompt}
        )

    stats: dict = {"mode": "batch", "agents_asked": len(crowd.crowd_ids)}
    batch = {env.agent_graph.get_agent(aid): _action() for aid in crowd.crowd_ids}
    if not await _safe_step(env, batch, "interview (batch)"):
        asked = failed = 0
        for aid in crowd.crowd_ids:
            one = {env.agent_graph.get_agent(aid): _action()}
            if await _safe_step(env, one, f"interview (agent {aid})"):
                asked += 1
            else:
                failed += 1
        _LOG.warning(
            "interview fell back to per-agent: %d steps ok, %d failed.", asked, failed
        )
        stats = {"mode": "per_agent", "agents_asked": asked, "steps_failed": failed}

    # Repair pass: ask again, but only the agents whose reply did not parse.
    repaired: list[int] = []
    for attempt in range(repairs):
        missing = [
            a for a in crowd.crowd_ids if a not in _answered_agents(db_path, crowd)
        ]
        if not missing:
            break
        _LOG.warning(
            "interview repair pass %d: re-asking %d agents with no parseable "
            "verdict (%s)",
            attempt + 1,
            len(missing),
            missing[:10],
        )
        await _safe_step(
            env,
            {env.agent_graph.get_agent(aid): _action() for aid in missing},
            f"interview (repair {attempt + 1})",
        )
        repaired.extend(missing)
    stats["repair_passes"] = len({a for a in repaired})
    stats["steps_failed"] = stats.get("steps_failed", 0)
    return stats


async def _run_consideration(env, crowd, db_path: str) -> dict[int, dict]:
    """Ask the latecomers whether the feed earned a click, then unlock the yeses.

    The decision is taken in its own turn, with no app tool reachable, because a
    model handed a tool uses it: told in its system prompt to hold off unless
    convinced, 8 of 8 latecomers opened the app anyway. Separating the decision
    from the ability is what makes the answer cost something to give -- and the
    resulting conversion rate is the only measurement here that virality has to
    earn rather than be assigned.
    """
    prompt = consider_prompt()
    await _safe_step(
        env,
        {
            env.agent_graph.get_agent(aid): ManualAction(
                action_type=ActionType.INTERVIEW, action_args={"prompt": prompt}
            )
            for aid in crowd.latecomer_ids
        },
        "consideration",
    )
    decisions = consideration_decisions(db_path)
    unlocked = 0
    for aid in crowd.latecomer_ids:
        if (decisions.get(aid) or {}).get("decided"):
            toolkit = crowd.trier_toolkits.get(aid)
            if toolkit is not None:
                toolkit.unlock()
                unlocked += 1
    _LOG.info(
        "consideration: %d of %d latecomers decided to try the app",
        unlocked,
        len(crowd.latecomer_ids),
    )
    return decisions


async def _seed_follows(env, crowd) -> None:
    """Seed the follow graph so the launch + early takes reach feeds."""
    by_follower: dict[int, list] = {}
    for follower_id, followee_id in crowd.follow_plan:
        by_follower.setdefault(follower_id, []).append(
            ManualAction(
                action_type=ActionType.FOLLOW,
                action_args={"followee_id": followee_id},
            )
        )
    actions = {
        env.agent_graph.get_agent(fid): acts for fid, acts in by_follower.items()
    }
    if actions:
        await env.step(actions)


async def _seed_launch(env, crowd, text: str) -> None:
    founder = env.agent_graph.get_agent(crowd.founder_id)
    await env.step(
        {
            founder: ManualAction(
                action_type=ActionType.CREATE_POST, action_args={"content": text}
            )
        }
    )


def _write_traces(out_dir: Path, crowd) -> list[str]:
    """Dump each trier's interaction trace (the first-hand evidence)."""
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for agent_id, toolkit in crowd.trier_toolkits.items():
        trace = toolkit.trace
        path = traces_dir / f"agent_{agent_id}.json"
        payload = {
            "agent_id": agent_id,
            "persona": crowd.persona_by_id[agent_id].username,
            "trace": trace.to_dict(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths.append(str(path))
    return paths


async def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run one crowd trial end-to-end and return pointers to its artifacts."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(out_dir / "simulation.db")
    summary_path = str(out_dir / "run_summary.json")
    if Path(db_path).exists():
        Path(db_path).unlink()  # fresh DB per run (wipe & repeat)

    result = SimulationResult(
        build_id=config.build_id,
        out_dir=str(out_dir),
        db_path=db_path,
        summary_path=summary_path,
        ok=False,
    )

    reset_turn_stats()
    build = _load_build(config.build_id)
    result.app_type = build.app_type
    result.undeliverable = build.undeliverable
    result.cohort = getattr(build.record, "cohort", "") or ""

    # Validity gate FIRST: does the app build, run, and do what it
    # claims? A broken app cannot legitimately be viral, so the ViralScore caps
    # its score -- but only if this evidence exists. Runs in its own ephemeral
    # container, so it never touches the crowd's shared instance.
    if build.undeliverable:
        result.validity = undeliverable_validity(config.build_id)
    elif config.validity_gate and not config.no_llm:
        result.validity = await _run_validity_gate(config)
        # The gate already tried to start this app in a container, so it is the
        # earliest -- and often the ONLY -- place the failure is visible. The
        # AppHost signal below is better evidence when it exists, but it only
        # exists if some agent opened the app, and a run where nobody did
        # would otherwise record a dead app as a merely unpopular one.
        if _gate_says_it_never_started(result.validity):
            result.app_start_failed = True
            result.app_start_detail = str(result.validity.get("detail", ""))[:2000]
            result.app_start_fault = _start_fault(result.app_start_detail)

    # Shared running app instance + shared browser (for web-app triers). An
    # undeliverable build has nothing to serve and nothing to render, so neither
    # is started -- spinning up a browser for an app that cannot exist burns
    # 30 seconds per run for nothing.
    #
    # Clear the app's durable state first. State must accumulate *within* a run
    # -- that is the whole multi-user signal, one agent seeing what another wrote
    # -- but must not leak *between* runs, which would make two runs of the same
    # build incomparable. The AppHost's lifetime is exactly one crowd run, so
    # this is the right seam for that reset.
    app_host = None
    if not build.undeliverable:
        reset_build_data(config.build_id)
        app_host = AppHost(container=config.container)
    browser_engine = None
    if not build.undeliverable and browser_available():
        browser_engine = BrowserEngine(BrowserConfig())

    personas = load_personas(config.personas_file)
    selection = select_crowd(
        personas,
        n_agents=config.n_agents,
        n_triers=config.n_triers,
        n_latecomers=config.n_latecomers,
        seed=config.seed,
    )
    model = (
        dummy_model()
        if config.no_llm
        else crowd_model(
            config.model_id,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            thinking_level=config.thinking_level,
        )
    )

    crowd = build_crowd_agents(
        config.build_id,
        selection,
        model=model,
        app_host=app_host,
        app_title=build.title,
        app_type=build.app_type,
        browser_engine=browser_engine,
        container=config.container,
        max_iteration_trier=config.max_iteration_trier,
        max_iteration_reactor=config.max_iteration_reactor,
        trial_max_steps=config.trial_max_steps,
        start_wait=config.start_wait,
        recsys_type=config.recsys_type,
        min_interactions=config.min_interactions,
        disabled_tools=config.disabled_tools,
        env_notice=config.env_notice,
        follow_peers=config.follow_peers,
    )

    platform_cfg = PlatformConfig(
        recsys_type=config.recsys_type, semaphore=config.semaphore
    )
    if config.feed_max_posts:
        platform_cfg.max_rec_post_len = config.feed_max_posts
    env = make_env(crowd.agent_graph, db_path, platform_cfg)

    round_log: list[dict] = []
    started = time.time()
    trajectories: list[dict] = []
    decisions: dict[int, dict] = {}
    try:
        await env.reset()

        # Round 0: seed the follow graph, then the founder's launch post (id 1).
        await _seed_follows(env, crowd)
        await _seed_launch(
            env, crowd, launch_post_text(build.title, build.summary, build.pitch)
        )
        result.launch_post_id = 1

        rounds_run = 0
        decisions: dict[int, dict] = {}
        for rnd in range(1, config.rounds + 1):
            if config.no_llm:
                actions = _scripted_round(env, crowd, rnd)
            else:
                # Round 1: only the triers, who discover and try the app.
                # Latecomers deliberately sit it out -- they have nothing to
                # react to yet, and the point of the tier is that the feed has
                # to reach them. Later rounds: everyone.
                active = crowd.trier_ids if rnd == 1 else crowd.crowd_ids
                actions = {
                    env.agent_graph.get_agent(aid): LLMAction() for aid in active
                }
            if not actions:
                continue
            # A round that blows up (OASIS can raise on a malformed agent action)
            # must not discard the rounds already simulated -- scored runs are
            # expensive and the remaining signal is still worth collecting.
            ok = await _safe_step(env, actions, f"round {rnd}")
            if ok:
                rounds_run += 1
            round_log.append({"round": rnd, "active_agents": len(actions), "ok": ok})
            # After the first-hand round there is finally something in the feed
            # worth being convinced by, so this is where the latecomers decide.
            if rnd == 1 and crowd.latecomer_ids and not config.no_llm:
                decisions = await _run_consideration(env, crowd, db_path)
        result.rounds_run = rounds_run

        # Optional non-perturbing measurement interview. This is the PRIMARY
        # scoring signal (every agent, independently judged), so it degrades to
        # per-agent rather than losing the whole crowd's verdicts to one fault.
        if config.interview and not config.no_llm:
            result.interview_stats = await _run_interview(env, crowd, db_path)

        # Capture every agent's reasoning BEFORE the env is closed: it lives
        # only in agent memory until now, and it is what the agentic autorater
        # reads. Best-effort -- capture must never cost a completed run.
        try:
            trajectories = collect_trajectories(env, crowd)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("trajectory capture failed: %s", exc)

        await env.close()
        # Not `ok = True`: the orchestrator finishing says nothing about whether
        # the run collected the data it exists to collect. _assess_health decides,
        # once the verdicts are in.
    except Exception as exc:  # noqa: BLE001 - record + still emit artifacts
        result.error = f"{type(exc).__name__}: {exc}"
        # Keep the traceback: a scored run that fails with only an exception
        # string is undiagnosable after the fact, and these runs are expensive.
        result.error_traceback = traceback.format_exc()
        _LOG.error("crowd run failed: %s\n%s", result.error, result.error_traceback)
        try:
            await env.close()
        except Exception:  # noqa: BLE001
            pass
    finally:
        result.trace_paths = _write_traces(out_dir, crowd)
        if trajectories:
            result.trajectories_path = write_trajectories(out_dir, trajectories)
        for toolkit in crowd.trier_toolkits.values():
            try:
                await toolkit.close()
            except Exception:  # noqa: BLE001
                pass
        if browser_engine is not None:
            try:
                await browser_engine.close()
            except Exception:  # noqa: BLE001
                pass
        # None for an undeliverable build: there was never an app to host.
        if app_host is not None:
            # Read the host's verdict BEFORE closing it. The AppHost gives up on a
            # build after MAX_START_ATTEMPTS and remembers why. That memory is the
            # only place the reason exists, since the container log lives in a
            # clone that close() is about to retire to builds/.trash.
            detail = app_host.unstartable(config.build_id)
            if detail:
                result.app_start_failed = True
                result.app_start_detail = detail[:2000]
                result.app_start_fault = _start_fault(detail)
            app_host.close()

    result.duration_s = round(time.time() - started, 1)
    result.engagement = _engagement_snapshot(db_path, result.launch_post_id)
    result.n_agents = len(crowd.crowd_ids)
    result.n_triers = len(crowd.trier_ids)

    # Compute the verdicts before deciding whether the run succeeded: losing the
    # crowd's own answers is the single most important failure this can have, and
    # it cannot be judged from the exception channel alone.
    verdicts = {
        "triers": trier_verdicts(crowd),
        "interviews": interview_verdicts(result.db_path, crowd),
        "conversion": conversion_stats(crowd, decisions),
    }
    result.turn_stats = turn_stats()
    result.health = assess_run_health(
        error=result.error,
        rounds_run=result.rounds_run,
        rounds_expected=expected_rounds(
            no_llm=config.no_llm,
            configured_rounds=config.rounds,
            round_log=round_log,
        ),
        round_log=round_log,
        verdicts=verdicts,
        interview_expected=config.interview and not config.no_llm,
        turn_stats=result.turn_stats,
    )
    result.ok = result.health["ok"]

    _write_summary(
        summary_path,
        config,
        result,
        selection,
        crowd,
        round_log,
        actions=_round_action_log(db_path),
        verdicts=verdicts,
    )
    # Also emit the append-only action log alongside the summary.
    (out_dir / "actions.jsonl").write_text(
        "\n".join(
            json.dumps(r, separators=(",", ":")) for r in _round_action_log(db_path)
        ),
        encoding="utf-8",
    )
    return result


def _scripted_round(env, crowd, rnd: int) -> dict:
    """No-LLM wiring smoke: deterministic ManualActions on the launch post.

    Round 1 = triers like + comment, round 2 = reactors like, and later rounds are
    skipped (empty) -- each agent acts once, so the platform's like de-dup never
    fires. This validates the platform + artifact wiring without any LLM spend.
    """
    launch = {"post_id": 1}
    if rnd == 1:
        return {
            env.agent_graph.get_agent(aid): [
                ManualAction(action_type=ActionType.LIKE_POST, action_args=launch),
                ManualAction(
                    action_type=ActionType.CREATE_COMMENT,
                    action_args={"post_id": 1, "content": "tried it, pretty neat"},
                ),
            ]
            for aid in crowd.trier_ids
        }
    if rnd == 2:
        return {
            env.agent_graph.get_agent(aid): ManualAction(
                action_type=ActionType.LIKE_POST, action_args=launch
            )
            for aid in crowd.reactor_ids
        }
    return {}


def _tier_skepticism_counts(selection) -> dict:
    """How the hands-on triers split across skepticism (evidence-quality check)."""
    counts: dict[str, int] = {}
    for p in selection.triers:
        counts[p.skepticism] = counts.get(p.skepticism, 0) + 1
    return counts


def _write_summary(
    path: str, config, result, selection, crowd, round_log, *, actions, verdicts
) -> None:
    summary = {
        "build_id": config.build_id,
        # Which INSTRUMENT produced this run. Two runs with the same config dict
        # can still be incomparable if the prompts, tiers or feed changed between
        # them, and that difference is invisible after the fact unless it is
        # stamped here at the time.
        "crowd_arch_version": arch_version(config.variant),
        # Which PROVIDER carried the model calls. Not cosmetic: the Gemini
        # Developer API rejected ~51% of this workload as RESOURCE_EXHAUSTED and
        # gemini_native's six rate-limit retries absorbed it, so the loss showed
        # up only as agent turns that silently did not happen -- a median ~18-21
        # skipped turns per run against 0 on Vertex, and roughly half the
        # comments per run. Comments and reposts feed advocacy_spread, 15% of the
        # active profile, so the two eras do not measure the same thing.
        #
        # Nothing on disk recorded which surface produced a run, so the two eras
        # were indistinguishable after the fact and were pooled -- and because
        # the transport happened to line up with the founder arm, that pooling
        # would have reported an API-surface difference as a pipeline result.
        # Stamping it here is the same contract crowd_arch_version has: record
        # what produced the run, at the time, because it cannot be recovered.
        #
        # Now that the crowd can run on any provider this records the provider id
        # rather than one of two surfaces. Runs from before that carry "vertex"
        # or "developer_api", which are today's "google-vertex" and "google".
        "crowd_transport": crowd_transport(config.model_id),
        "cohort": result.cohort,
        "app_start_fault": result.app_start_fault,
        "undeliverable": result.undeliverable,
        # Recorded even when false, so "this run was measured and the app started"
        # is a positive statement on disk rather than the absence of a key.
        "app_start_failed": result.app_start_failed,
        "app_start_detail": result.app_start_detail,
        "config": asdict(config),
        "ok": result.ok,
        "health": result.health,
        "error": result.error,
        "app_type": result.app_type,
        "duration_s": result.duration_s,
        "rounds_run": result.rounds_run,
        "launch_post_id": result.launch_post_id,
        "engagement": result.engagement,
        # Validity gate (builds/runs/does_what_it_claims) -- the ViralScore caps
        # a broken app. None means it was skipped, i.e. the run is unverified.
        "validity": result.validity,
        "interview_stats": result.interview_stats,
        "turn_stats": result.turn_stats,
        # Crowd integrity: a run whose crowd was silently clamped below the
        # requested size is not comparable, and the score stage flags it.
        "crowd_integrity": {
            "requested_n_agents": selection.requested_n_agents,
            "actual_n_agents": len(selection.all),
            "persona_pool_size": selection.pool_size,
            "clamped": selection.clamped,
            "n_triers": len(selection.triers),
            "trier_skepticism": _tier_skepticism_counts(selection),
        },
        # Aggregated crowd verdicts -- the discriminating signal the scoring stage
        # reads (raw distributions, with no composite score computed here). Computed
        # by the caller, because run health depends on them.
        "verdicts": verdicts,
        "n_actions_logged": len(actions),
        "crowd": [
            {
                "agent_id": aid,
                "username": crowd.persona_by_id[aid].username,
                "archetype": crowd.persona_by_id[aid].archetype,
                "tier": crowd.tier_by_id[aid],
                "influence": crowd.persona_by_id[aid].influence,
            }
            for aid in crowd.crowd_ids
        ],
        "rounds": round_log,
        "trace_paths": result.trace_paths,
    }
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
