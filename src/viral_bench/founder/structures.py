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

"""Collaboration structures: how a founder configuration is driven, turn by turn.

A collaboration structure decides the *sequence* of opencode turns for one build
-- who acts, in what order, resuming which session -- while the
:class:`~viral_bench.founder.harness.OpenCodeRunner` primitive it is handed knows
how to execute a single turn. There are three, matching the three founder
configurations:

* :class:`SoloPipeline` (``agents=1``) -- the original single-agent baseline:
  one Design turn then one Build turn, in one opencode session. Byte-for-byte the
  pre-team pipeline.
* :class:`RoundTableTeam` (``agents=4``) -- four specialists collaborating over
  multiple **rounds**. This is the real "founder as a team": it is *not*
  flattenable to one agent, because

  1. **each specialist keeps its own context.** An agent's first turn starts a
     fresh opencode session; the structure captures that session id and, on every
     later round, resumes *that* session (``--session``). So the Architect always
     reasons with the Architect's memory even though the Implementer, Designer,
     and QA ran in between.
  2. **they iterate back and forth over rounds.** Each round every specialist
     acts once (Architect -> Implementer -> Designer -> QA), reacting to what the
     others changed, so QA's findings in round *k* are addressed by the others in
     round *k+1*.
  3. **it terminates sensibly.** A hard cap of ``max_rounds`` bounds the work, and
     QA can end the build early by emitting the ship signal once the app is truly
     ready. QA always acts last, and its turn contract requires a complete,
     runnable app + valid manifest -- so whether the team ships early or hits the
     cap, the final turn leaves something shippable.

* :class:`DynamicOrchestrator` (``agents="dynamic"``) -- ONE founder agent that
  decides its own team. It is handed the idea, the deliverable contract and
  opencode's delegation machinery (the ``task`` tool, resumable subagents, and a
  directory it can write its own agent definitions into), and nothing else: no
  roles, no phases, no rounds, no division of labour. Whether it works alone,
  runs six specialists it invented, or changes its mind halfway is its call, and
  that call is part of what the mode measures.

  Both other structures encode a human's answer to "how should a founding team be
  organised?" -- a fixed relay of four personas, decided in advance and identical
  for every idea and every model. That bottlenecks a strong model into a shape it
  did not choose and cannot leave. This mode removes the answer and keeps the
  question, so a model is measured on orchestrating an agentic process as well as
  on writing code.

  The only thing the harness insists on is *finishing*: a turn that ends with the
  deliverables incomplete gets a deliberately content-free nudge (see
  :func:`~viral_bench.founder.prompts.dynamic_continue_prompt`) and the same
  session resumes, up to ``max_turns``. That is a completion backstop, not a
  process -- without it, a model that stops after planning would be recorded as
  unable to build an app, which is a harness artifact reported as a capability
  difference (docs/crowd_bugs.md T0.1).

How the team *collaborates* during turns (shared local files vs Google Workspace)
is decided by the collaboration toolset (see :mod:`viral_bench.founder.collab`),
which the structure asks for per-turn env and a collaboration brief.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from viral_bench.founder.collab import CollaborationToolset, LocalToolset
from viral_bench.founder.harness import (
    HarnessResult,
    OpenCodeRunner,
    concurrent_spawn_peak,
    read_assistant_text,
    read_task_spawns,
    read_tool_calls,
    session_tree,
)
from viral_bench.founder.manifest import (
    ALLOWED_APP_TYPES,
    MANIFEST_FILENAME,
    ManifestError,
    load_manifest,
)
from viral_bench.founder.prompts import (
    DYNAMIC_DONE_SIGNAL,
    SHIP_SIGNAL,
    build_prompt,
    design_prompt,
    dynamic_continue_prompt,
    dynamic_founder_prompt,
    team_turn_prompt,
)
from viral_bench.founder.roles import TEAM_SIZE, VALID_TEAM_SIZES, Role, roles_for
from viral_bench.founder.workspace import BuildWorkspace
from viral_bench.ideas import Idea

__all__ = [
    "CollaborationStructure",
    "SoloPipeline",
    "RoundTableTeam",
    "DynamicOrchestrator",
    "StructureError",
    "build_structure",
    "DEFAULT_ROUNDS",
    "DEFAULT_DYNAMIC_TURNS",
    "DYNAMIC",
    "AGENTS_DIRNAME",
]

#: Default hard cap on collaboration rounds for the team (each round = 4 turns).
DEFAULT_ROUNDS = 3

#: The value of ``agents`` that selects :class:`DynamicOrchestrator`. A string
#: rather than a number because the number is precisely what the model decides.
DYNAMIC = "dynamic"

#: Default hard cap on orchestrator turns in dynamic mode.
#:
#: Three is the point where the affordances line up rather than an arbitrary
#: budget: a subagent the founder defines for itself only loads on the turn AFTER
#: it is written (opencode reads agent definitions at turn start), so a cap of 1
#: would advertise a capability the model cannot reach and a cap of 2 would let it
#: use a self-defined team exactly once. Three gives write -> use -> revise, and a
#: model that finishes sooner simply says so and stops.
DEFAULT_DYNAMIC_TURNS = 3

#: Where a dynamic founder writes subagent definitions, relative to the workspace
#: root. Deliberately NOT inside ``app/``: opencode treats any directory holding
#: ``.opencode`` as an instance directory and installs ~60 MB of plugin
#: dependencies beside it, which inside the app would be 60 MB of node_modules
#: sitting in the tree we ship and diff. At the workspace root it lands next to
#: the team's skills, outside everything that ships.
AGENTS_DIRNAME = "agents"


class StructureError(ValueError):
    """Raised when a collaboration structure cannot be constructed."""


@runtime_checkable
class CollaborationStructure(Protocol):
    """Anything that can drive a founder build's turns.

    Implementations expose lightweight metadata (so the build record and CLI can
    describe the configuration) plus a :meth:`run` that executes the turn sequence
    using the supplied runner and (for the team) collaboration toolset.
    """

    name: str
    n_agents: int
    roles: list[Role]
    max_rounds: int
    min_rounds: int
    max_turns: int
    min_turns: int
    rounds_run: int
    shipped_early: bool
    qa_verified: bool | None

    def run(
        self,
        idea: Idea,
        workspace: BuildWorkspace,
        runner: OpenCodeRunner,
        toolset: CollaborationToolset | None = None,
    ) -> HarnessResult: ...


class SoloPipeline:
    """The solo founder baseline: one Design turn, then one Build turn.

    Uses the plain :func:`~viral_bench.founder.prompts.design_prompt` /
    :func:`~viral_bench.founder.prompts.build_prompt` and the default opencode
    agent (no specialist tooling), so this path is identical to the pre-team
    pipeline. The build turn ``--continue``s the design session so the founder
    keeps its own context across the two turns.
    """

    name = "solo"

    def __init__(self, n_agents: int = 1) -> None:
        if n_agents != 1:
            raise StructureError(f"SoloPipeline is agents=1 only, got {n_agents!r}")
        self.n_agents = 1
        self.roles: list[Role] = roles_for(1)
        self.max_rounds = 1
        self.min_rounds = 1
        self.max_turns = 2
        self.min_turns = 2
        self.rounds_run = 0
        self.shipped_early = False
        self.qa_verified: bool | None = None  # solo has no separate QA gate

    @property
    def role_keys(self) -> list[str]:
        return [role.key for role in self.roles]

    def describe(self) -> str:
        return "solo founder: design -> build (2 turns)"

    def run(
        self,
        idea: Idea,
        workspace: BuildWorkspace,
        runner: OpenCodeRunner,
        toolset: CollaborationToolset | None = None,
    ) -> HarnessResult:
        result = HarnessResult(model=runner.model)
        self.rounds_run = 1

        design = runner.run_turn(
            design_prompt(idea),
            workspace=workspace,
            phase="design",
            turn="design",
            continue_session=False,
        )
        result.phases.append(design)
        if not design.ok:
            return result

        build = runner.run_turn(
            build_prompt(idea),
            workspace=workspace,
            phase="build",
            turn="build",
            continue_session=True,
        )
        result.phases.append(build)
        return result


class RoundTableTeam:
    """Four specialists collaborating over rounds, each keeping its own context.

    See the module docstring for the design. The four roles come from
    :func:`~viral_bench.founder.roles.roles_for` (Architect, Implementer, UX &
    Virality Designer, QA & Finisher); the last owns QA & Finish.
    """

    name = "team"

    def __init__(self, max_rounds: int = DEFAULT_ROUNDS, min_rounds: int = 1) -> None:
        if max_rounds < 1:
            raise StructureError(f"max_rounds must be >= 1, got {max_rounds!r}")
        if not 1 <= min_rounds <= max_rounds:
            raise StructureError(
                f"min_rounds must be between 1 and max_rounds ({max_rounds}), "
                f"got {min_rounds!r}"
            )
        self.n_agents = TEAM_SIZE
        self.roles: list[Role] = roles_for(TEAM_SIZE)
        self.max_rounds = max_rounds
        self.min_rounds = min_rounds
        self.max_turns = TEAM_SIZE * max_rounds
        self.min_turns = TEAM_SIZE * min_rounds
        self.rounds_run = 0
        self.shipped_early = False
        # Whether QA actually exercised the running app on its final turn (browser
        # for a web app, or running the real command for a CLI/bot). None until QA
        # first acts. Early ship is gated on this being True.
        self.qa_verified: bool | None = None

    @property
    def role_keys(self) -> list[str]:
        return [role.key for role in self.roles]

    def describe(self) -> str:
        relay = ", ".join(role.title for role in self.roles)
        rounds = (
            f"{self.max_rounds}"
            if self.min_rounds == self.max_rounds
            else f"{self.min_rounds}-{self.max_rounds}"
        )
        turns = (
            f"{self.max_turns}"
            if self.min_turns == self.max_turns
            else f"{self.min_turns}-{self.max_turns}"
        )
        return (
            f"round-table team: {self.n_agents} specialists x {rounds} rounds "
            f"({turns} turns) [{relay}]"
        )

    def _teammates(self, agent_index: int) -> list[tuple[str, str]]:
        """``(title, responsibility label)`` for every role except ``agent_index``."""
        out: list[tuple[str, str]] = []
        for j, role in enumerate(self.roles, start=1):
            if j == agent_index:
                continue
            labels = ", ".join(r.label for r in role.responsibilities)
            out.append((role.title, labels))
        return out

    def run(
        self,
        idea: Idea,
        workspace: BuildWorkspace,
        runner: OpenCodeRunner,
        toolset: CollaborationToolset | None = None,
    ) -> HarnessResult:
        """Run rounds of the team until QA ships or the round cap is hit.

        Stops early (returning the phases so far) if any turn fails -- a broken
        collaboration cannot produce a complete app.
        """
        if toolset is None:
            toolset = LocalToolset()
        result = HarnessResult(model=runner.model)
        sessions: dict[int, str] = {}  # agent_index -> its own opencode session id
        self.rounds_run = 0
        self.shipped_early = False
        self.qa_verified = None

        toolset.prepare(workspace.build_id, self.roles)
        try:
            for rnd in range(1, self.max_rounds + 1):
                self.rounds_run = rnd
                qa_shipped = False

                for agent_index, role in enumerate(self.roles, start=1):
                    extra_env = toolset.turn_env(agent_index)
                    brief = toolset.collaboration_brief(agent_index)
                    is_first_turn = agent_index not in sessions

                    prompt = team_turn_prompt(
                        idea,
                        role,
                        round_index=rnd,
                        max_rounds=self.max_rounds,
                        min_rounds=self.min_rounds,
                        agent_index=agent_index,
                        n_agents=self.n_agents,
                        teammates=self._teammates(agent_index),
                        is_first_turn=is_first_turn,
                        collaboration_brief=brief,
                    )
                    turn = runner.run_turn(
                        prompt,
                        workspace=workspace,
                        phase=f"r{rnd}_a{agent_index}_{role.key}",
                        turn="team",
                        continue_session=False,
                        role=role.key,
                        agent_index=agent_index,
                        extra_env=extra_env,
                        agent=role.key,
                        session_id=sessions.get(agent_index, ""),
                    )
                    result.phases.append(turn)

                    # Remember this agent's session so it resumes its OWN context
                    # next round (regardless of who ran in between).
                    if turn.session_id:
                        sessions[agent_index] = turn.session_id
                    if not turn.ok:
                        return result

                    # Only QA (which acts last) can end the build early, and only
                    # once the minimum number of rounds is done AND QA actually
                    # exercised the running app this turn (a browser interaction for
                    # a web app, or running the real command for a CLI/bot) -- a
                    # ship signal from a QA turn that never ran the app is ignored.
                    if role.owns_qa:
                        self.qa_verified = _turn_has_test_evidence(
                            turn.transcript_path, workspace.app_dir
                        )
                        if (
                            rnd >= self.min_rounds
                            and _turn_signals_ship(turn.transcript_path)
                            and self.qa_verified
                        ):
                            qa_shipped = True

                if qa_shipped:
                    self.shipped_early = True
                    break
        finally:
            toolset.cleanup()

        return result


class DynamicOrchestrator:
    """One founder agent that chooses and runs its own subagents.

    See the module docstring for why this exists. Mechanically it is the simplest
    structure of the three -- one session, resumed turn after turn -- because all
    the interesting structure is decided by the model at run time rather than
    here. What this class owns is only:

    1. **the brief**, once, on turn 1 (:func:`dynamic_founder_prompt`);
    2. **a completion backstop** -- if a turn ends with the deliverables missing,
       resume the same session with a content-free nudge naming only the gap, up
       to ``max_turns``;
    3. **telemetry** -- every ``task`` call the founder made, so a sweep can ask
       what shape of team each model actually built for itself.

    Note what is *not* here: no evidence gate on the completion signal, unlike
    :class:`RoundTableTeam`, which ignores QA's ship signal unless QA was seen
    driving the running app. That check reads the turn's own transcript, and a
    dynamic founder that delegates verification leaves that evidence in a CHILD
    session the parent transcript never shows. Gating on it would mark exactly
    the models that delegate well as unverified -- a harness artifact dressed up
    as a capability difference. Verification is recorded, not enforced; the
    container check and the crowd remain the real judges of whether an app works.
    """

    name = "dynamic"

    def __init__(self, max_turns: int = DEFAULT_DYNAMIC_TURNS) -> None:
        if max_turns < 1:
            raise StructureError(f"max_turns must be >= 1, got {max_turns!r}")
        # One opencode process, one session: the count of *agents* is the model's
        # business and is reported as observed telemetry, never as configuration.
        self.n_agents = 1
        self.roles: list[Role] = []
        self.max_rounds = max_turns
        self.min_rounds = 1
        self.max_turns = max_turns
        self.min_turns = 1
        self.rounds_run = 0
        self.shipped_early = False
        self.qa_verified: bool | None = None
        #: Everything observed about how the founder orchestrated (see
        #: :meth:`orchestration`). Populated during :meth:`run`.
        self.spawns: list[dict] = []
        self.done_signalled = False
        self.self_authored_agents: list[str] = []
        self.peak_concurrent = 0
        #: Shape of the whole session tree under the founder, which is the only
        #: place NESTED delegation shows up (a subagent that delegated in turn).
        self.tree: dict = {"sessions": 0, "max_depth": 0, "titles": []}

    @property
    def role_keys(self) -> list[str]:
        return []

    def describe(self) -> str:
        return (
            f"dynamic orchestrator: 1 founder agent choosing its own subagents "
            f"(<= {self.max_turns} turns)"
        )

    def orchestration(self) -> dict:
        """Observed orchestration, recorded on the build for later analysis."""
        by_type: dict[str, int] = {}
        for spawn in self.spawns:
            key = spawn.get("subagent_type") or "?"
            by_type[key] = by_type.get(key, 0) + 1
        resumed = sum(1 for s in self.spawns if s.get("resumed_task_id"))
        failed = sum(1 for s in self.spawns if s.get("status") == "error")
        # Which self-defined agents the founder actually RAN, as opposed to
        # merely wrote. The two came apart in practice: a model authored two
        # specialists, finished inside one turn, and spawned only `general` --
        # because opencode loads agent definitions at turn start, so an agent is
        # unusable until the turn after it is written. Recording the names and
        # the spawn types separately left that invisible unless you cross-read
        # two fields by eye, and "the model designed a team it never used" is a
        # different result from "the model chose stock agents".
        authored_used = sorted(set(self.self_authored_agents) & set(by_type))
        return {
            "subagents_spawned": len(self.spawns),
            "subagent_types": by_type,
            "resumed_subagents": resumed,
            "failed_spawns": failed,
            "peak_concurrent_subagents": self.peak_concurrent,
            # From opencode's session store, so it counts subagents a SUBAGENT
            # spawned too; `subagents_spawned` above counts only the founder's
            # own task calls. A gap between the two is nested delegation.
            "session_tree": self.tree,
            "max_delegation_depth": self.tree.get("max_depth", 0),
            "self_authored_agents": sorted(self.self_authored_agents),
            "authored_agents_used": authored_used,
            "done_signalled": self.done_signalled,
            "orchestrator_turns": self.rounds_run,
            "spawns": self.spawns,
        }

    def run(
        self,
        idea: Idea,
        workspace: BuildWorkspace,
        runner: OpenCodeRunner,
        toolset: CollaborationToolset | None = None,
    ) -> HarnessResult:
        """Run the founder until it declares the build done or the cap is hit."""
        result = HarnessResult(model=runner.model)
        self.rounds_run = 0
        self.spawns = []
        self.done_signalled = False
        self.self_authored_agents = []
        self.peak_concurrent = 0
        self.tree = {"sessions": 0, "max_depth": 0, "titles": []}

        agents_dir = dynamic_agents_dir(workspace)
        # Created up front so the path in the brief is real. A model told to write
        # into a directory that does not exist wastes a tool call discovering that.
        agents_dir.mkdir(parents=True, exist_ok=True)

        session_id = ""
        for turn_index in range(1, self.max_turns + 1):
            self.rounds_run = turn_index
            if turn_index == 1:
                prompt = dynamic_founder_prompt(
                    idea,
                    app_dir=str(workspace.app_dir),
                    agents_dir=str(agents_dir),
                    max_turns=self.max_turns,
                )
            else:
                prompt = dynamic_continue_prompt(
                    turn_index=turn_index,
                    max_turns=self.max_turns,
                    gaps=_dynamic_gaps(workspace.app_dir, self.done_signalled),
                )

            turn = runner.run_turn(
                prompt,
                workspace=workspace,
                phase=f"t{turn_index}_founder",
                turn="dynamic",
                continue_session=False,
                role="founder",
                agent_index=1,
                session_id=session_id,
            )
            result.phases.append(turn)
            if turn.session_id:
                session_id = turn.session_id

            # Record what happened even on a failed turn: a turn that died having
            # already spawned eight subagents is a very different failure from one
            # that died before delegating anything, and the record should say so.
            spawns = read_task_spawns(turn.transcript_path)
            self.spawns.extend(spawns)
            self.peak_concurrent = max(
                self.peak_concurrent, concurrent_spawn_peak(spawns)
            )
            self.self_authored_agents = _authored_agent_names(agents_dir)
            if session_id:
                self.tree = session_tree(session_id)
            if DYNAMIC_DONE_SIGNAL in read_assistant_text(turn.transcript_path):
                self.done_signalled = True

            if not turn.ok:
                return result
            if self.done_signalled and not _dynamic_gaps(workspace.app_dir, True):
                # Declared finished AND the contract is actually satisfied.
                self.shipped_early = turn_index < self.max_turns
                break

        return result


def dynamic_agents_dir(workspace: BuildWorkspace) -> Path:
    """Directory a dynamic founder writes its own subagent definitions into.

    Sits beside the team's skills under the workspace's ``.opencode`` -- inside
    the git worktree (so opencode discovers it) and outside ``app/`` (so it never
    ships and never drags opencode's plugin ``node_modules`` into the built app).
    """
    return Path(workspace.root) / ".opencode" / AGENTS_DIRNAME


def _authored_agent_names(agents_dir: Path) -> list[str]:
    """Names of the subagents the founder defined for itself, if any."""
    try:
        return sorted(p.stem for p in agents_dir.glob("*.md") if p.is_file())
    except OSError:
        return []


def _dynamic_gaps(app_dir, done_signalled: bool) -> list[str]:
    """What still stands between this build and a complete deliverable.

    Only ever states facts about the contract every founder mode shares -- is
    there a manifest, does it parse, did you say you were finished. It must not
    grow into build advice: the nudge that carries these lines is the one place
    the harness could accidentally start doing the orchestrating.
    """
    gaps: list[str] = []
    path = Path(app_dir) / MANIFEST_FILENAME
    if not path.is_file():
        # Name the exact path rather than "the app root". A model that wrote the
        # manifest one directory up reads "there is no manifest" as a claim it
        # can see is false, argues with it, and never moves the file; the path
        # makes the mismatch obvious without telling it what to do about it.
        gaps.append(
            f"There is no manifest at `{path}` (that exact path is what is checked)."
        )
    else:
        try:
            load_manifest(path)
        except ManifestError as exc:
            gaps.append(f"`{MANIFEST_FILENAME}` is not valid: {exc}")
    if not gaps and not done_signalled:
        gaps.append(
            "The deliverables are in place but you did not declare the build "
            "complete, so it is treated as unfinished."
        )
    return gaps


def _turn_signals_ship(transcript_path) -> bool:
    """True if the QA agent emitted the ship signal in this turn's output."""
    return SHIP_SIGNAL in read_assistant_text(transcript_path)


def _manifest_run_hints(app_dir) -> tuple[str, str]:
    """Return ``(app_type, run_command)`` from the app's manifest (best-effort)."""
    try:
        data = json.loads(
            (Path(app_dir) / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return "", ""
    if not isinstance(data, dict):
        return "", ""
    app_type = str(data.get("app_type") or "")
    run = data.get("run")
    cmd = str(run.get("command") or "") if isinstance(run, dict) else ""
    return app_type, cmd


def _ran_command(bash_cmd: str, target: str) -> bool:
    """True if ``bash_cmd`` appears to execute ``target`` (the manifest command)."""
    b = " ".join(bash_cmd.split())
    t = " ".join(target.split())
    if not t:
        return False
    if t in b:
        return True
    entrypoints: list[str] = []
    tokens = t.split()
    for i, tok in enumerate(tokens):
        if tok == "-m" and i + 1 < len(tokens):
            entrypoints.append(tokens[i + 1])
        elif ("." in tok or "/" in tok) and not tok.startswith("-"):
            entrypoints.append(tok)
    return any(ep in b for ep in entrypoints)


def _turn_has_test_evidence(transcript_path, app_dir) -> bool:
    """True if QA's turn actually exercised the running app, not just read it.

    Evidence is a real browser interaction (a ``browser_*`` tool call) or a shell
    run of the app's manifest ``run.command``. For a single-page-app the browser is
    the only real check *when a browser is available on the host* (a server started
    via bash does not prove the UI renders); when no browser is available the gate
    degrades to accepting a run of the server so a build is never hard-blocked.
    """
    calls = read_tool_calls(transcript_path)
    if any(name.startswith("browser_") for name, _ in calls):
        return True
    app_type, run_cmd = _manifest_run_hints(app_dir)
    if app_type in ALLOWED_APP_TYPES:
        from viral_bench.founder.opencode_agents import browser_prereqs_ok

        if browser_prereqs_ok():
            # A browser was available but unused -> not real UI verification.
            return False
    if not run_cmd:
        return False
    return any(
        name == "bash"
        and isinstance(inp.get("command"), str)
        and _ran_command(inp["command"], run_cmd)
        for name, inp in calls
    )


def normalize_agents(agents: int | str) -> int | str:
    """Coerce a CLI/config ``agents`` value to ``1``, ``TEAM_SIZE`` or ``"dynamic"``.

    The selector is a number for the two fixed-size configurations and the string
    ``"dynamic"`` for the one whose size the model decides, so it arrives from
    argparse, YAML and ``build_fleet`` in both shapes. Normalising in one place
    keeps every caller from re-deciding whether ``"4"`` is a team.

    Raises:
        StructureError: If the value names no supported configuration.
    """
    if isinstance(agents, str):
        text = agents.strip().lower()
        if text == DYNAMIC:
            return DYNAMIC
        try:
            agents = int(text)
        except ValueError:
            raise StructureError(
                f"agents must be one of {list(VALID_TEAM_SIZES)} or {DYNAMIC!r}, "
                f"got {agents!r}"
            ) from None
    if agents in VALID_TEAM_SIZES:
        return int(agents)
    raise StructureError(
        f"agents must be one of {list(VALID_TEAM_SIZES)} or {DYNAMIC!r} "
        f"(1 = solo founder, {TEAM_SIZE} = the team, {DYNAMIC!r} = one founder "
        f"choosing its own subagents), got {agents!r}"
    )


def build_structure(
    n_agents: int | str,
    *,
    max_rounds: int = DEFAULT_ROUNDS,
    min_rounds: int = 1,
    max_turns: int = DEFAULT_DYNAMIC_TURNS,
) -> CollaborationStructure:
    """Construct the collaboration structure for a founder configuration.

    Args:
        n_agents: ``1`` (solo), :data:`~viral_bench.founder.roles.TEAM_SIZE` (the
            four-agent round-table team), or :data:`DYNAMIC` (one founder agent
            that chooses its own subagents).
        max_rounds: Hard round cap for the team (ignored for solo/dynamic).
        min_rounds: Minimum rounds the team must run before QA may ship early
            (ignored for solo/dynamic). Must be ``1 <= min_rounds <= max_rounds``.
        max_turns: Hard cap on orchestrator turns for the dynamic founder
            (ignored for solo/team).

    Raises:
        StructureError: If ``n_agents`` is not a supported configuration or
            ``max_rounds``/``min_rounds``/``max_turns`` is invalid.
    """
    selected = normalize_agents(n_agents)
    if selected == DYNAMIC:
        return DynamicOrchestrator(max_turns=max_turns)
    if selected == 1:
        return SoloPipeline(1)
    return RoundTableTeam(max_rounds=max_rounds, min_rounds=min_rounds)
