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

"""Tests for the founder collaboration structures.

A fake runner records each turn and writes a minimal opencode-style transcript so
no real opencode/Gemini calls happen. It mints a per-agent session id on the first
turn (like real opencode) so we can verify each specialist resumes its OWN session
across rounds, and can emit the ship signal on demand to test early termination.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from viral_bench.founder.harness import HarnessResult, PhaseResult
from viral_bench.founder.prompts import (
    DYNAMIC_DONE_SIGNAL,
    SHIP_SIGNAL,
    build_prompt,
    design_prompt,
)
from viral_bench.founder.structures import (
    DEFAULT_DYNAMIC_TURNS,
    DEFAULT_ROUNDS,
    DYNAMIC,
    CollaborationStructure,
    DynamicOrchestrator,
    RoundTableTeam,
    SoloPipeline,
    StructureError,
    _ran_command,
    _turn_has_test_evidence,
    build_structure,
    dynamic_agents_dir,
    normalize_agents,
)
from viral_bench.ideas import Idea

IDEA = Idea(
    idea_id="sliding_tile_game",
    title="TileMerge",
    pitch="Slide and merge.",
    problem="p",
    target_user="u",
    core_features=("a",),
    success_criteria="s",
    allowed_scope="client-app",
    difficulty="easy",
)

_ROUND_RE = re.compile(r"^r(\d+)_")


class FakeWorkspace:
    """Minimal stand-in for BuildWorkspace."""

    def __init__(self, root: Path) -> None:
        self.build_id = "b-test"
        self.root = root
        self.app_dir = root / "app"


class FakeRunner:
    """Records each turn; mints per-agent sessions; can emit the ship signal.

    Args:
        tmp: directory to write per-turn transcripts into.
        returncodes: return codes to hand out in call order (default all 0).
        ship_on_round: if set, QA (owns-qa role) emits the ship signal on that
            round, so the structure should terminate early.
        qa_runs_app: if True (default), QA turns include a ``browser_*`` tool call
            so they count as real runtime evidence for the ship-evidence gate. Set
            False to simulate a QA that only read the code.
    """

    model = "google/fake-model"

    def __init__(
        self, tmp: Path, *, returncodes=None, ship_on_round=None, qa_runs_app=True
    ) -> None:
        self.tmp = tmp
        self.returncodes = returncodes or []
        self.ship_on_round = ship_on_round
        self.qa_runs_app = qa_runs_app
        self.calls: list[dict] = []

    def timeout_for(self, turn: str) -> float:
        return 1.0

    def run_turn(
        self,
        prompt: str,
        *,
        workspace,
        phase: str,
        turn: str,
        continue_session: bool,
        timeout_s=None,
        role: str = "",
        agent_index: int = 0,
        extra_env=None,
        agent: str = "",
        session_id: str = "",
    ) -> PhaseResult:
        idx = len(self.calls)
        rc = self.returncodes[idx] if idx < len(self.returncodes) else 0
        # Mint a stable per-agent session id on first turn; reuse thereafter (as
        # real opencode does when handed --session).
        sid = session_id or f"ses-{agent_index}"

        # Assistant output; QA emits the ship token on the configured round.
        text = f"work by {role or 'solo'}"
        m = _ROUND_RE.match(phase)
        rnd = int(m.group(1)) if m else 0
        is_qa = role == "qa_finisher"
        if is_qa and self.ship_on_round is not None and rnd == self.ship_on_round:
            text += f"\nAll criteria pass.\n{SHIP_SIGNAL}"

        events = [{"type": "text", "part": {"type": "text", "text": text}}]
        # A QA turn that "ran the app" emits a browser tool call, which the
        # ship-evidence gate looks for (real runtime verification, not just reading).
        if is_qa and self.qa_runs_app:
            events.append(
                {
                    "type": "tool_use",
                    "part": {
                        "type": "tool",
                        "tool": "browser_navigate",
                        "state": {"input": {"url": "http://localhost:8000/"}},
                    },
                }
            )
        transcript = self.tmp / f"{phase}.json"
        transcript.write_text(
            "\n".join(json.dumps(e) for e in events), encoding="utf-8"
        )

        self.calls.append(
            {
                "prompt": prompt,
                "phase": phase,
                "turn": turn,
                "continue_session": continue_session,
                "role": role,
                "agent_index": agent_index,
                "agent": agent,
                "session_id_in": session_id,
                "extra_env": extra_env or {},
            }
        )
        return PhaseResult(
            phase=phase,
            returncode=rc,
            transcript_path=transcript,
            duration_s=0.1,
            role=role,
            agent_index=agent_index,
            turn=turn,
            session_id=sid,
        )


# -- solo -------------------------------------------------------------------- #


def test_solo_runs_design_then_build(tmp_path) -> None:
    runner = FakeRunner(tmp_path)
    result = SoloPipeline(1).run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    assert isinstance(result, HarnessResult)
    assert [c["phase"] for c in runner.calls] == ["design", "build"]
    # exact solo prompts, and build --continues the design session
    assert runner.calls[0]["prompt"] == design_prompt(IDEA)
    assert runner.calls[1]["prompt"] == build_prompt(IDEA)
    assert runner.calls[0]["continue_session"] is False
    assert runner.calls[1]["continue_session"] is True
    # solo uses the default agent (no --agent selection)
    assert runner.calls[0]["agent"] == ""


def test_solo_rejects_non_one() -> None:
    with pytest.raises(StructureError):
        SoloPipeline(4)


# -- team round-table -------------------------------------------------------- #


def test_team_runs_four_turns_per_round(tmp_path) -> None:
    runner = FakeRunner(tmp_path)
    team = RoundTableTeam(max_rounds=2)
    result = team.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    assert result.ok
    assert len(runner.calls) == 8  # 4 agents x 2 rounds
    assert team.rounds_run == 2
    assert team.shipped_early is False
    # phase labels encode round + agent + role
    assert [c["phase"] for c in runner.calls] == [
        "r1_a1_architect",
        "r1_a2_implementer",
        "r1_a3_designer",
        "r1_a4_qa_finisher",
        "r2_a1_architect",
        "r2_a2_implementer",
        "r2_a3_designer",
        "r2_a4_qa_finisher",
    ]
    # every team turn selects the specialist's own agent and is a "team" turn
    for c in runner.calls:
        assert c["agent"] == c["role"]
        assert c["turn"] == "team"


def test_each_agent_resumes_its_own_session_across_rounds(tmp_path) -> None:
    runner = FakeRunner(tmp_path)
    RoundTableTeam(max_rounds=3).run(
        IDEA, workspace=FakeWorkspace(tmp_path), runner=runner
    )
    # For each agent: first turn starts fresh (no session in), later rounds resume
    # THAT agent's own captured session id -- never a teammate's.
    for agent_index in range(1, 5):
        mine = [c for c in runner.calls if c["agent_index"] == agent_index]
        assert len(mine) == 3
        assert mine[0]["session_id_in"] == ""  # first turn: fresh
        assert mine[1]["session_id_in"] == f"ses-{agent_index}"
        assert mine[2]["session_id_in"] == f"ses-{agent_index}"
        # team turns never use --continue (they target a specific session)
        assert all(c["continue_session"] is False for c in mine)


def test_team_stops_early_when_qa_ships(tmp_path) -> None:
    runner = FakeRunner(tmp_path, ship_on_round=2)
    team = RoundTableTeam(max_rounds=5)
    team.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    # ships at the end of round 2 -> 8 turns, not the full 20
    assert len(runner.calls) == 8
    assert team.rounds_run == 2
    assert team.shipped_early is True


def test_no_early_ship_runs_to_the_cap(tmp_path) -> None:
    runner = FakeRunner(tmp_path)  # QA never ships
    team = RoundTableTeam(max_rounds=3)
    team.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    assert len(runner.calls) == 12
    assert team.rounds_run == 3
    assert team.shipped_early is False
    # QA acted last in the final round -> app is left in QA's hands
    assert runner.calls[-1]["role"] == "qa_finisher"


def test_ship_signal_from_non_qa_is_ignored(tmp_path) -> None:
    # Force the architect (not QA) to emit the token; it must NOT end the build.
    class ArchShips(FakeRunner):
        def run_turn(self, prompt, **kw):
            res = super().run_turn(prompt, **kw)
            if kw.get("role") == "architect":
                res.transcript_path.write_text(
                    json.dumps(
                        {"type": "text", "part": {"type": "text", "text": SHIP_SIGNAL}}
                    )
                )
            return res

    runner = ArchShips(tmp_path)
    team = RoundTableTeam(max_rounds=2)
    team.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    assert team.shipped_early is False
    assert len(runner.calls) == 8


def test_team_aborts_on_failed_turn(tmp_path) -> None:
    # third turn (designer, round 1) fails -> stop immediately
    runner = FakeRunner(tmp_path, returncodes=[0, 0, 1])
    team = RoundTableTeam(max_rounds=3)
    result = team.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    assert not result.ok
    assert len(runner.calls) == 3
    assert result.phases[-1].role == "designer"


def test_team_prompt_is_role_specific_and_round_aware(tmp_path) -> None:
    runner = FakeRunner(tmp_path)
    RoundTableTeam(max_rounds=2).run(
        IDEA, workspace=FakeWorkspace(tmp_path), runner=runner
    )
    first = runner.calls[0]["prompt"]
    assert "Architect" in first
    assert "round 1 of at most 2" in first
    assert "FIRST turn" in first  # architect's first turn
    # QA's prompt carries the ship contract
    qa_first = runner.calls[3]["prompt"]
    assert SHIP_SIGNAL in qa_first
    # second round: architect resumes (not its first turn)
    arch_round2 = runner.calls[4]["prompt"]
    assert "resuming your OWN context" in arch_round2


# -- min_rounds floor -------------------------------------------------------- #


def test_min_rounds_allows_ship_at_the_floor(tmp_path) -> None:
    # QA ships in round 3, and min_rounds is 3 -> honored: stop at 12 turns.
    runner = FakeRunner(tmp_path, ship_on_round=3)
    team = RoundTableTeam(max_rounds=4, min_rounds=3)
    team.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    assert team.rounds_run == 3
    assert team.shipped_early is True
    assert len(runner.calls) == 12


def test_min_rounds_ignores_ship_below_floor(tmp_path) -> None:
    # QA emits the ship signal in round 2, but min_rounds is 3 -> ignored, so the
    # team keeps going to the cap (no later ship signal).
    runner = FakeRunner(tmp_path, ship_on_round=2)
    team = RoundTableTeam(max_rounds=4, min_rounds=3)
    team.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    assert team.shipped_early is False
    assert team.rounds_run == 4
    assert len(runner.calls) == 16


def test_min_rounds_prompt_holds_ship_then_allows(tmp_path) -> None:
    runner = FakeRunner(tmp_path)
    team = RoundTableTeam(max_rounds=4, min_rounds=3)
    team.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    # QA is agent 4; its turns are calls index 3, 7, 11, 15 (rounds 1..4)
    qa_r1 = runner.calls[3]["prompt"]
    qa_r3 = runner.calls[11]["prompt"]
    # below the floor: QA is told NOT to ship (no ship token in the prompt)
    assert "DO NOT SHIP YET" in qa_r1
    assert SHIP_SIGNAL not in qa_r1
    # at the floor: the real ship decision is offered
    assert "SHIP DECISION" in qa_r3
    assert SHIP_SIGNAL in qa_r3


def test_team_rejects_min_rounds_above_max() -> None:
    with pytest.raises(StructureError):
        RoundTableTeam(max_rounds=2, min_rounds=3)


def test_min_turns_reflects_floor() -> None:
    team = RoundTableTeam(max_rounds=4, min_rounds=3)
    assert team.min_turns == 12
    assert team.max_turns == 16


# -- ship-evidence gate + qa_verified ---------------------------------------- #


def test_early_ship_requires_signal_and_evidence(tmp_path) -> None:
    # QA both ships AND exercises the app -> honored at the floor.
    runner = FakeRunner(tmp_path, ship_on_round=1, qa_runs_app=True)
    team = RoundTableTeam(max_rounds=3, min_rounds=1)
    team.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    assert team.shipped_early is True
    assert team.qa_verified is True
    assert len(runner.calls) == 4  # shipped at end of round 1


def test_ship_signal_without_evidence_is_ignored(tmp_path) -> None:
    # QA emits the ship token but never ran the app -> gate blocks early ship.
    runner = FakeRunner(tmp_path, ship_on_round=2, qa_runs_app=False)
    team = RoundTableTeam(max_rounds=3)
    team.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    assert team.shipped_early is False
    assert team.qa_verified is False
    assert len(runner.calls) == 12  # ran to the cap despite the token


def test_qa_verified_reflects_final_qa_turn(tmp_path) -> None:
    # No ship, but QA exercises the app each round -> qa_verified True.
    runner = FakeRunner(tmp_path, qa_runs_app=True)
    team = RoundTableTeam(max_rounds=2)
    team.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=runner)
    assert team.qa_verified is True
    other = FakeRunner(tmp_path, qa_runs_app=False)
    team2 = RoundTableTeam(max_rounds=2)
    team2.run(IDEA, workspace=FakeWorkspace(tmp_path), runner=other)
    assert team2.qa_verified is False


def _write_transcript(path: Path, *, text="", tools=()):
    events = []
    if text:
        events.append({"type": "text", "part": {"type": "text", "text": text}})
    for name, inp in tools:
        events.append(
            {
                "type": "tool_use",
                "part": {"type": "tool", "tool": name, "state": {"input": inp}},
            }
        )
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return path


def test_evidence_detects_browser_call(tmp_path) -> None:
    t = _write_transcript(
        tmp_path / "t.json", tools=[("browser_navigate", {"url": "http://x/"})]
    )
    assert _turn_has_test_evidence(t, tmp_path / "app") is True


def test_evidence_false_when_qa_only_reads(tmp_path, monkeypatch) -> None:
    # Browser forced unavailable so this asserts what it claims to: reading and
    # grepping the source is not evidence *even on the lenient path*. With a
    # browser available every web app short-circuits to False anyway, which
    # would make the read/grep calls incidental to the result.
    monkeypatch.setattr(
        "viral_bench.founder.opencode_agents.browser_prereqs_ok", lambda: False
    )
    app = tmp_path / "app"
    app.mkdir()
    (app / "viralbench.json").write_text(
        json.dumps({"app_type": "client-app", "run": {"command": "python3 main.py"}})
    )
    t = _write_transcript(
        tmp_path / "t.json", tools=[("read", {"filePath": "main.py"}), ("grep", {})]
    )
    assert _turn_has_test_evidence(t, app) is False


def test_web_app_requires_browser_when_browser_available(tmp_path, monkeypatch) -> None:
    # Every app is a web app, so merely starting the server via bash is NOT
    # evidence when a browser is available -- the UI must actually be rendered.
    monkeypatch.setattr(
        "viral_bench.founder.opencode_agents.browser_prereqs_ok", lambda: True
    )
    app = tmp_path / "app"
    app.mkdir()
    (app / "viralbench.json").write_text(
        json.dumps({"app_type": "client-app", "run": {"command": "python3 server.py"}})
    )
    t = _write_transcript(
        tmp_path / "t.json", tools=[("bash", {"command": "python3 server.py &"})]
    )
    assert _turn_has_test_evidence(t, app) is False


def test_full_stack_app_requires_browser_too(tmp_path, monkeypatch) -> None:
    """The browser gate keys off the whole scope enum, not one member of it.

    It used to fire only for ``single-page-app``; if it had stayed keyed to a
    single value, a ``full-stack-app`` would silently pass QA on a bash run.
    """
    monkeypatch.setattr(
        "viral_bench.founder.opencode_agents.browser_prereqs_ok", lambda: True
    )
    app = tmp_path / "app"
    app.mkdir()
    (app / "viralbench.json").write_text(
        json.dumps(
            {"app_type": "full-stack-app", "run": {"command": "python3 server.py"}}
        )
    )
    t = _write_transcript(
        tmp_path / "t.json", tools=[("bash", {"command": "python3 server.py &"})]
    )
    assert _turn_has_test_evidence(t, app) is False


def test_web_app_bash_run_ok_when_no_browser(tmp_path, monkeypatch) -> None:
    # Graceful fallback: with no browser on the host, running the server counts.
    # (This absorbs the old cli-manifest case -- with cli gone, "a bash run of
    # the manifest command is evidence" only survives on this no-browser path.)
    monkeypatch.setattr(
        "viral_bench.founder.opencode_agents.browser_prereqs_ok", lambda: False
    )
    app = tmp_path / "app"
    app.mkdir()
    (app / "viralbench.json").write_text(
        json.dumps({"app_type": "client-app", "run": {"command": "python3 server.py"}})
    )
    t = _write_transcript(
        tmp_path / "t.json", tools=[("bash", {"command": "python3 server.py &"})]
    )
    assert _turn_has_test_evidence(t, app) is True


def test_ran_command_matches_entrypoint() -> None:
    assert _ran_command("python3 server.py 8000", "python3 server.py")
    assert _ran_command(
        "cd app && python3 -m http.server 8000", "python3 -m http.server"
    )
    assert _ran_command("node ./index.js", "node index.js")
    assert not _ran_command("cat server.txt", "python3 server.py")
    assert not _ran_command("ls -la", "python3 server.py")


# -- toolset threading ------------------------------------------------------- #


class FakeToolset:
    """Records lifecycle calls; injects per-agent env + a collaboration brief."""

    name = "fake"

    def __init__(self) -> None:
        self.prepared = None
        self.cleaned = False

    def prepare(self, build_id, roles) -> None:
        self.prepared = (build_id, [r.key for r in roles])

    def turn_env(self, agent_index):
        return {"COLLAB_TOKEN": f"tok-{agent_index}"}

    def collaboration_brief(self, agent_index):
        return f"BRIEF-FOR-a{agent_index}"

    def metadata(self):
        return {"collab": "fake"}

    def cleanup(self):
        self.cleaned = True


def test_toolset_env_and_brief_threaded_into_turns(tmp_path) -> None:
    runner = FakeRunner(tmp_path)
    tools = FakeToolset()
    RoundTableTeam(max_rounds=1).run(
        IDEA, workspace=FakeWorkspace(tmp_path), runner=runner, toolset=tools
    )
    assert tools.prepared == (
        "b-test",
        ["architect", "implementer", "designer", "qa_finisher"],
    )
    assert tools.cleaned is True
    for c in runner.calls:
        assert c["extra_env"] == {"COLLAB_TOKEN": f"tok-{c['agent_index']}"}
        assert f"BRIEF-FOR-a{c['agent_index']}" in c["prompt"]


def test_toolset_cleanup_runs_even_on_failure(tmp_path) -> None:
    runner = FakeRunner(tmp_path, returncodes=[1])
    tools = FakeToolset()
    RoundTableTeam(max_rounds=2).run(
        IDEA, workspace=FakeWorkspace(tmp_path), runner=runner, toolset=tools
    )
    assert tools.cleaned is True  # cleanup is in a finally


# -- build_structure factory ------------------------------------------------- #


def test_build_structure_solo() -> None:
    s = build_structure(1)
    assert isinstance(s, SoloPipeline)
    assert isinstance(s, CollaborationStructure)
    assert s.name == "solo"


def test_build_structure_team() -> None:
    s = build_structure(4, max_rounds=4, min_rounds=3)
    assert isinstance(s, RoundTableTeam)
    assert s.n_agents == 4
    assert s.max_rounds == 4
    assert s.min_rounds == 3
    assert s.max_turns == 16
    assert s.min_turns == 12


def test_build_structure_default_rounds() -> None:
    s = build_structure(4)
    assert s.max_rounds == DEFAULT_ROUNDS
    assert s.min_rounds == 1  # default floor: QA may ship after round 1


def test_build_structure_rejects_min_above_max() -> None:
    with pytest.raises(StructureError):
        build_structure(4, max_rounds=3, min_rounds=5)


@pytest.mark.parametrize("bad", [0, 2, 3, 5])
def test_build_structure_rejects_unsupported_sizes(bad: int) -> None:
    with pytest.raises(StructureError):
        build_structure(bad)


def test_team_rejects_zero_rounds() -> None:
    with pytest.raises(StructureError):
        RoundTableTeam(max_rounds=0)


def test_describe_mentions_rounds_and_specialists() -> None:
    text = RoundTableTeam(max_rounds=3).describe()
    assert "3 rounds" in text
    assert "12 turns" in text
    assert "Architect" in text


def test_describe_shows_min_max_range() -> None:
    text = RoundTableTeam(max_rounds=4, min_rounds=3).describe()
    assert "3-4 rounds" in text
    assert "12-16 turns" in text


# -- dynamic orchestrator ---------------------------------------------------- #


VALID_MANIFEST = {
    "app_type": "client-app",
    "title": "TileMerge",
    "summary": "A sliding tile puzzle.",
    "setup": [],
    "run": {"command": "python3 -m http.server 8000", "port": 8000, "url": "u"},
    "test": {"manual": ["play it"], "smoke": "true"},
}


class DynamicFakeRunner:
    """Drives the dynamic orchestrator without opencode.

    Models the three things the structure actually reacts to: whether the turn
    emitted the done token, whether a manifest exists by then, and what ``task``
    calls the transcript contains.

    Args:
        tmp: directory for per-turn transcripts.
        done_on: 1-based turn on which the founder emits the done signal (None =
            never, so the structure should run to the cap).
        manifest_on: 1-based turn on which a VALID manifest appears (None =
            never; ``"invalid"`` writes unparseable JSON on turn 1 instead).
        spawns_per_turn: how many ``task`` calls each turn records.
        returncodes: return codes in call order (default all 0).
    """

    model = "google/fake-model"

    def __init__(
        self,
        tmp: Path,
        *,
        done_on: int | None = None,
        manifest_on: int | None = None,
        spawns_per_turn: int = 0,
        returncodes=None,
    ) -> None:
        self.tmp = tmp
        self.done_on = done_on
        self.manifest_on = manifest_on
        self.spawns_per_turn = spawns_per_turn
        self.returncodes = returncodes or []
        self.calls: list[dict] = []

    def timeout_for(self, turn: str) -> float:
        return 1.0

    def run_turn(
        self,
        prompt: str,
        *,
        workspace,
        phase: str,
        turn: str,
        continue_session: bool,
        timeout_s=None,
        role: str = "",
        agent_index: int = 0,
        extra_env=None,
        agent: str = "",
        session_id: str = "",
    ) -> PhaseResult:
        n = len(self.calls) + 1
        rc = self.returncodes[n - 1] if n - 1 < len(self.returncodes) else 0
        events: list[dict] = [
            {"type": "text", "part": {"type": "text", "text": f"turn {n}"}}
        ]
        for i in range(self.spawns_per_turn):
            events.append(
                {
                    "type": "tool",
                    "part": {
                        "type": "tool",
                        "tool": "task",
                        "state": {
                            "status": "completed",
                            "input": {
                                "subagent_type": "general",
                                "description": f"job {i}",
                                "prompt": "do it",
                            },
                            "metadata": {
                                "sessionId": f"ses-kid-{n}-{i}",
                                "parentSessionId": "ses-main",
                                "model": {
                                    "providerID": "google-vertex",
                                    "modelID": "fake",
                                },
                            },
                            # Deliberately overlapping, so peak concurrency > 1.
                            "time": {"start": 1000, "end": 2000},
                        },
                    },
                }
            )
        manifest_path = workspace.app_dir / "viralbench.json"
        if self.manifest_on == "invalid" and n == 1:
            manifest_path.write_text("{not json", encoding="utf-8")
        elif isinstance(self.manifest_on, int) and n >= self.manifest_on:
            manifest_path.write_text(json.dumps(VALID_MANIFEST), encoding="utf-8")
        if self.done_on is not None and n >= self.done_on:
            events.append(
                {"type": "text", "part": {"type": "text", "text": DYNAMIC_DONE_SIGNAL}}
            )

        transcript = self.tmp / f"{phase}.json"
        transcript.write_text(
            "\n".join(json.dumps(e) for e in events), encoding="utf-8"
        )
        self.calls.append(
            {
                "prompt": prompt,
                "phase": phase,
                "turn": turn,
                "agent": agent,
                "session_id_in": session_id,
                "role": role,
            }
        )
        return PhaseResult(
            phase=phase,
            returncode=rc,
            transcript_path=transcript,
            duration_s=0.1,
            role=role,
            agent_index=agent_index,
            turn=turn,
            session_id="ses-main",
        )


def _dyn_workspace(tmp_path) -> FakeWorkspace:
    ws = FakeWorkspace(tmp_path)
    ws.app_dir.mkdir(parents=True, exist_ok=True)
    return ws


def test_dynamic_stops_when_founder_declares_done_and_manifest_is_valid(
    tmp_path,
) -> None:
    runner = DynamicFakeRunner(tmp_path, done_on=1, manifest_on=1)
    s = DynamicOrchestrator(max_turns=3)
    result = s.run(IDEA, _dyn_workspace(tmp_path), runner)
    assert result.ok
    assert len(runner.calls) == 1
    assert s.rounds_run == 1
    assert s.done_signalled is True
    assert s.shipped_early is True


def test_dynamic_keeps_going_when_the_manifest_is_missing(tmp_path) -> None:
    """The done token alone must not end a build with no deliverable.

    A model that says it is finished when it is not is exactly the case the
    completion backstop exists for -- otherwise its word ends the build and the
    idea is recorded as unbuildable.
    """
    runner = DynamicFakeRunner(tmp_path, done_on=1, manifest_on=3)
    s = DynamicOrchestrator(max_turns=3)
    s.run(IDEA, _dyn_workspace(tmp_path), runner)
    assert len(runner.calls) == 3
    assert "no manifest at" in runner.calls[1]["prompt"]
    # names the exact path checked, so a misplaced file is visibly misplaced
    assert "viralbench.json`" in runner.calls[1]["prompt"]


def test_dynamic_names_an_invalid_manifest_as_the_gap(tmp_path) -> None:
    runner = DynamicFakeRunner(tmp_path, done_on=1, manifest_on="invalid")
    DynamicOrchestrator(max_turns=2).run(IDEA, _dyn_workspace(tmp_path), runner)
    assert "is not valid" in runner.calls[1]["prompt"]


def test_dynamic_nudges_a_silent_founder_that_left_a_valid_manifest(tmp_path) -> None:
    """Deliverables present but no declaration reads as a truncated turn."""
    runner = DynamicFakeRunner(tmp_path, done_on=None, manifest_on=1)
    s = DynamicOrchestrator(max_turns=2)
    s.run(IDEA, _dyn_workspace(tmp_path), runner)
    assert len(runner.calls) == 2
    assert "did not declare the build complete" in runner.calls[1]["prompt"]
    assert s.done_signalled is False


def test_dynamic_resumes_its_own_session_across_turns(tmp_path) -> None:
    runner = DynamicFakeRunner(tmp_path)
    DynamicOrchestrator(max_turns=3).run(IDEA, _dyn_workspace(tmp_path), runner)
    assert [c["session_id_in"] for c in runner.calls] == ["", "ses-main", "ses-main"]
    # No --agent: the dynamic founder is stock opencode, not a bench persona.
    assert {c["agent"] for c in runner.calls} == {""}
    assert {c["turn"] for c in runner.calls} == {"dynamic"}


def test_dynamic_stops_on_a_failed_turn(tmp_path) -> None:
    runner = DynamicFakeRunner(tmp_path, returncodes=[0, 1, 0])
    s = DynamicOrchestrator(max_turns=3)
    result = s.run(IDEA, _dyn_workspace(tmp_path), runner)
    assert result.ok is False
    assert len(runner.calls) == 2


def test_dynamic_records_the_orchestration_it_observed(tmp_path) -> None:
    runner = DynamicFakeRunner(tmp_path, done_on=2, manifest_on=2, spawns_per_turn=2)
    s = DynamicOrchestrator(max_turns=3)
    s.run(IDEA, _dyn_workspace(tmp_path), runner)
    orch = s.orchestration()
    assert orch["subagents_spawned"] == 4  # 2 turns x 2 spawns
    assert orch["subagent_types"] == {"general": 4}
    assert orch["peak_concurrent_subagents"] == 2  # overlapping within a turn
    assert orch["orchestrator_turns"] == 2
    assert orch["done_signalled"] is True
    assert len(orch["spawns"]) == 4


def test_dynamic_records_spawns_from_a_turn_that_then_failed(tmp_path) -> None:
    """A build that died having already delegated is not the same as one that
    died before delegating; the record has to be able to tell them apart."""
    runner = DynamicFakeRunner(tmp_path, spawns_per_turn=3, returncodes=[1])
    s = DynamicOrchestrator(max_turns=3)
    s.run(IDEA, _dyn_workspace(tmp_path), runner)
    assert s.orchestration()["subagents_spawned"] == 3


def test_dynamic_reports_agents_the_founder_defined_for_itself(tmp_path) -> None:
    ws = _dyn_workspace(tmp_path)
    agents_dir = dynamic_agents_dir(ws)

    class AuthoringRunner(DynamicFakeRunner):
        def run_turn(self, *a, **kw):
            agents_dir.mkdir(parents=True, exist_ok=True)
            (agents_dir / "growth-hacker.md").write_text("---\n---\n", encoding="utf-8")
            return super().run_turn(*a, **kw)

    s = DynamicOrchestrator(max_turns=1)
    s.run(IDEA, ws, AuthoringRunner(tmp_path))
    assert s.orchestration()["self_authored_agents"] == ["growth-hacker"]


def test_dynamic_agents_dir_is_outside_the_app(tmp_path) -> None:
    """It must not land in app/: opencode installs ~60 MB of plugin deps beside
    any .opencode it finds, and app/ is the tree that ships."""
    ws = _dyn_workspace(tmp_path)
    agents_dir = dynamic_agents_dir(ws)
    assert ws.app_dir not in agents_dir.parents
    assert agents_dir.parts[-2:] == (".opencode", "agents")


def test_dynamic_creates_the_agents_dir_before_the_first_turn(tmp_path) -> None:
    ws = _dyn_workspace(tmp_path)
    DynamicOrchestrator(max_turns=1).run(IDEA, ws, DynamicFakeRunner(tmp_path))
    assert dynamic_agents_dir(ws).is_dir()


def test_dynamic_first_prompt_imposes_no_team_shape(tmp_path) -> None:
    """The whole point of the mode: the brief must not smuggle in a process.

    Guards against the natural drift of "helpfully" re-adding the four roles or a
    round structure to the dynamic brief, which would quietly turn it back into
    the team arm with extra steps.
    """
    runner = DynamicFakeRunner(tmp_path, done_on=1, manifest_on=1)
    DynamicOrchestrator(max_turns=3).run(IDEA, _dyn_workspace(tmp_path), runner)
    brief = runner.calls[0]["prompt"].lower()
    for banned in ("architect", "implementer", "qa & finisher", "round-table"):
        assert banned not in brief
    # ...while still carrying the contract every mode shares.
    assert "viralbench.json" in brief
    assert DYNAMIC_DONE_SIGNAL in runner.calls[0]["prompt"]


def test_build_structure_dynamic() -> None:
    s = build_structure(DYNAMIC, max_turns=5)
    assert isinstance(s, DynamicOrchestrator)
    assert s.name == "dynamic"
    assert s.max_turns == 5
    # One configured agent (the orchestrator process); the rest is the model's
    # choice and is reported as telemetry, never as configuration.
    assert s.n_agents == 1
    assert s.roles == []


def test_build_structure_default_dynamic_turns() -> None:
    assert build_structure(DYNAMIC).max_turns == DEFAULT_DYNAMIC_TURNS


def test_dynamic_rejects_zero_turns() -> None:
    with pytest.raises(StructureError):
        DynamicOrchestrator(max_turns=0)


@pytest.mark.parametrize(
    "given,expected",
    [
        ("1", 1),
        ("4", 4),
        ("dynamic", "dynamic"),
        ("DYNAMIC", "dynamic"),
        (1, 1),
        (4, 4),
    ],
)
def test_normalize_agents_accepts_both_shapes(given, expected) -> None:
    assert normalize_agents(given) == expected


@pytest.mark.parametrize("bad", ["", "0", "2", "solo", "team", 3])
def test_normalize_agents_rejects_everything_else(bad) -> None:
    with pytest.raises(StructureError):
        normalize_agents(bad)
