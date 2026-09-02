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

"""Tests for the opencode founder harness (with a fake opencode subprocess).

Every model id here is fake, and every test names one explicitly. There is no
default model any more -- a fresh checkout configures none, and the harness
refuses to run without one (:func:`require_model`) rather than picking a
provider to bill on the user's behalf -- so a harness constructed with no model
cannot even build its opencode config.

The two Vertex providers are used for most cases because they authenticate
ambiently, which keeps these tests free of credentials. The keyed path is
covered in ``tests/test_providers.py``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from viral_bench.founder.harness import (
    NO_REAP_ENV,
    HarnessError,
    OpenCodeHarness,
    OpenCodeRunner,
    _diagnose,
    _parse_session_id,
    _thinking_options,
    concurrent_spawn_peak,
    dump_session_trace,
    read_assistant_text,
    read_reasoning,
    read_task_spawns,
    read_tool_calls,
    reap_workspace_processes,
    require_model,
    session_tree,
)
from viral_bench.founder.structures import DynamicOrchestrator, RoundTableTeam
from viral_bench.founder.workspace import BuildWorkspace
from viral_bench.ideas import Idea

#: A Gemini-shaped model on Vertex, and a Claude-shaped one. Both providers use
#: ambient cloud credentials, so no key is needed to build a config for them.
MODEL = "google-vertex/gemini-test"
CLAUDE = "google-vertex-anthropic/claude-test"

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

# A minimal opencode JSON transcript line carrying a session id + assistant text.
_STDOUT = json.dumps(
    {"type": "text", "sessionID": "ses_abc", "part": {"type": "text", "text": "hi"}}
)


def _workspace(tmp_path) -> BuildWorkspace:
    return BuildWorkspace("b1", root=tmp_path / "b1").create()


def test_a_run_with_no_model_configured_says_how_to_pick_one() -> None:
    """There is no default model anywhere, so "" has to be a loud failure.

    It reaches here from an unconfigured checkout (`config/founder.yaml` ships
    every stage empty), and the only useful thing to do with it is name the two
    commands that fix it. Guessing a provider would bill an account the user
    never chose.
    """
    with pytest.raises(HarnessError) as excinfo:
        require_model("")
    message = str(excinfo.value)
    assert "viral-bench init" in message
    assert "--model" in message
    # A model that IS set passes straight through, unvalidated -- resolution is
    # the provider layer's job, not this function's.
    assert require_model(MODEL) == MODEL


def test_build_env_scopes_the_cloud_project_to_the_child(monkeypatch, tmp_path) -> None:
    # An ambient-auth provider needs its cloud project in the environment, and
    # it goes to the opencode child ONLY -- never exported into this process --
    # so a co-resident tool relying on the ambient project is unaffected and a
    # sweep can run two arms against two providers at once.
    monkeypatch.setenv("VERTEX_PROJECT", "project-test")
    runner = OpenCodeRunner(MODEL, binary="/bin/true", preflight=False)
    env = runner._build_env(tmp_path / "opencode.json")
    assert env["GOOGLE_CLOUD_PROJECT"] == "project-test"
    assert env["GOOGLE_CLOUD_QUOTA_PROJECT"] == "project-test"
    assert os.environ.get("GOOGLE_CLOUD_PROJECT") != "project-test"
    # Ambient credentials, so no API key is ever put in the child's environment.
    assert "GOOGLE_GENERATIVE_AI_API_KEY" not in env


def test_config_json_pins_model_and_disables_share() -> None:
    h = OpenCodeHarness(model=MODEL, binary="/bin/true")
    config = json.loads(h._config_json())
    assert config["model"] == MODEL
    assert config["small_model"] == MODEL
    assert config["share"] == "disabled"
    assert "gemini-test" in config["provider"]["google-vertex"]["models"]
    # solo path defines no custom agents
    assert "agent" not in config


def test_config_json_follows_the_model_to_a_partner_provider() -> None:
    """A Claude model must configure google-vertex-anthropic, not google-vertex.

    The provider block tells opencode which surface to call, so deriving it from
    the model string (rather than a hardcoded constant) is the whole mechanism
    behind swapping the founder model across providers.
    """
    h = OpenCodeHarness(model=CLAUDE, binary="/bin/true")
    config = json.loads(h._config_json())
    assert config["model"] == CLAUDE
    # The founder model does its own bookkeeping calls: swapping in a second,
    # cheaper small model would make the cross-provider comparison uneven.
    assert config["small_model"] == CLAUDE
    assert "claude-test" in config["provider"]["google-vertex-anthropic"]["models"]
    assert "google-vertex" not in config["provider"]


def test_subprocess_env_pins_location_for_both_providers() -> None:
    """opencode's Anthropic loader falls back to us-central1, which serves no Claude 5.

    It resolves location as GOOGLE_VERTEX_LOCATION -> GOOGLE_CLOUD_LOCATION ->
    VERTEX_LOCATION -> "us-central1". Leaving any unset risks a not-found that
    reads like a bad model id rather than a bad location, so all are set.
    """
    from viral_bench.founder.vertex import vertex_location, vertex_subprocess_env

    env = vertex_subprocess_env()
    location = vertex_location()
    assert location
    for var in ("GOOGLE_VERTEX_LOCATION", "GOOGLE_CLOUD_LOCATION", "VERTEX_LOCATION"):
        assert env[var] == location
    assert env["GOOGLE_VERTEX_PROJECT"] == env["GOOGLE_CLOUD_PROJECT"]


def test_team_config_defines_specialist_agents() -> None:
    h = OpenCodeHarness(
        model=MODEL,
        binary="/bin/true",
        structure=RoundTableTeam(2),
    )
    config = json.loads(h._config_json())
    assert set(config["agent"]) == {
        "architect",
        "implementer",
        "designer",
        "qa_finisher",
    }
    # web research differentiated at the agent level
    assert config["agent"]["architect"]["permission"]["webfetch"] == "allow"
    assert config["agent"]["implementer"]["permission"]["webfetch"] == "deny"


def test_binary_not_found_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        "viral_bench.founder.harness.find_opencode_binary", lambda: None
    )
    h = OpenCodeHarness(MODEL)
    with pytest.raises(HarnessError, match="opencode binary not found"):
        h.binary()


def _fake_run_factory(returncodes, stdout=_STDOUT):
    """Return a fake subprocess.run that yields the given return codes in order.

    Only opencode turns are faked. git passes through to the real subprocess and
    is not counted. A workspace created while this patch is installed sets up its
    own repo (``BuildWorkspace.init_git_boundary``), which is not a turn.
    """
    calls = []
    real_run = subprocess.run

    def fake_run(argv, **kwargs):
        if argv and str(argv[0]) == "git":
            return real_run(argv, **kwargs)
        calls.append(argv)
        rc = returncodes[len(calls) - 1]
        return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr="")

    return fake_run, calls


# -- solo path --------------------------------------------------------------- #


def test_run_records_two_phases_on_success(monkeypatch, tmp_path) -> None:
    fake_run, calls = _fake_run_factory([0, 0])
    monkeypatch.setattr(subprocess, "run", fake_run)

    workspace = _workspace(tmp_path)
    h = OpenCodeHarness(MODEL, binary="/bin/true", preflight=False)
    result = h.run(IDEA, workspace)

    assert result.ok
    assert [p.phase for p in result.phases] == ["design", "build"]
    # session id captured from the transcript
    assert result.phases[0].session_id == "ses_abc"
    # opencode.json written outside the app dir
    assert (workspace.app_dir.parent / "opencode.json").is_file()
    # transcripts captured
    assert (workspace.transcript_dir / "design.json").is_file()
    assert (workspace.transcript_dir / "build.json").is_file()
    # logs routed to stderr for capture, and the build phase used --continue
    assert "--print-logs" in calls[0]
    assert "--continue" in calls[1]
    assert "--continue" not in calls[0]
    # solo path never selects a named agent
    assert "--agent" not in calls[0]


def test_run_stops_after_failed_design(monkeypatch, tmp_path) -> None:
    fake_run, calls = _fake_run_factory([1, 0])
    monkeypatch.setattr(subprocess, "run", fake_run)

    workspace = _workspace(tmp_path)
    h = OpenCodeHarness(MODEL, binary="/bin/true", preflight=False)
    result = h.run(IDEA, workspace)

    assert not result.ok
    assert [p.phase for p in result.phases] == ["design"]
    assert len(calls) == 1


def test_run_records_preflight_failure(monkeypatch, tmp_path) -> None:
    """A model the account cannot call must cost one second, not a whole build.

    Without the preflight, opencode spends the entire design timeout inside its
    own retry loop and the run is then recorded as a *build* failure -- a setup
    problem laundered into evidence about the model.
    """
    # Built before the patch so the workspace's own git setup is not captured.
    workspace = _workspace(tmp_path)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))

    def boom(model_id):
        raise HarnessError("preflight failed: high demand")

    monkeypatch.setattr("viral_bench.founder.harness.preflight", boom)

    result = OpenCodeHarness(MODEL, binary="/bin/true").run(IDEA, workspace)

    assert not result.ok
    assert [p.phase for p in result.phases] == ["preflight"]
    assert "high demand" in result.phases[0].stderr_tail
    assert calls == []  # opencode never launched


# -- team path --------------------------------------------------------------- #


def test_team_harness_runs_one_agent_per_role_per_round(monkeypatch, tmp_path) -> None:
    fake_run, calls = _fake_run_factory([0, 0, 0, 0])
    monkeypatch.setattr(subprocess, "run", fake_run)

    workspace = _workspace(tmp_path)
    h = OpenCodeHarness(
        MODEL,
        binary="/bin/true",
        preflight=False,
        structure=RoundTableTeam(max_rounds=1),
    )
    result = h.run(IDEA, workspace)

    assert result.ok
    assert len(result.phases) == 4  # 4 specialists x 1 round
    assert [p.role for p in result.phases] == [
        "architect",
        "implementer",
        "designer",
        "qa_finisher",
    ]
    # each turn selects that specialist's own agent via --agent
    for role, argv in zip(
        ["architect", "implementer", "designer", "qa_finisher"], calls, strict=True
    ):
        assert "--agent" in argv
        assert argv[argv.index("--agent") + 1] == role
    # per-agent, per-round transcript files (no collisions)
    assert (workspace.transcript_dir / "r1_a1_architect.json").is_file()
    assert (workspace.transcript_dir / "r1_a4_qa_finisher.json").is_file()
    # skills installed into the private (non-shipping) .opencode/skills dir
    assert (workspace.root / ".opencode" / "skills" / "virality-playbook").is_dir()


# -- run_turn session/agent wiring ------------------------------------------- #


def test_run_turn_passes_session_and_agent(monkeypatch, tmp_path) -> None:
    fake_run, calls = _fake_run_factory([0])
    monkeypatch.setattr(subprocess, "run", fake_run)

    workspace = _workspace(tmp_path)
    runner = OpenCodeRunner(MODEL, binary="/bin/true", preflight=False)
    assert runner.prepare(workspace) is None
    res = runner.run_turn(
        "hello",
        workspace=workspace,
        phase="r2_a1_architect",
        turn="team",
        continue_session=False,
        agent="architect",
        session_id="ses_prev",
    )
    argv = calls[0]
    assert "--agent" in argv and argv[argv.index("--agent") + 1] == "architect"
    assert "--session" in argv and argv[argv.index("--session") + 1] == "ses_prev"
    # --session takes precedence: no --continue
    assert "--continue" not in argv
    # session id echoed from the transcript
    assert res.session_id == "ses_abc"


def test_run_turn_requires_prepare(tmp_path) -> None:
    runner = OpenCodeRunner(MODEL, binary="/bin/true", preflight=False)
    with pytest.raises(HarnessError, match="prepare"):
        runner.run_turn(
            "hi",
            workspace=_workspace(tmp_path),
            phase="design",
            turn="design",
            continue_session=False,
        )


# -- transcript parsing helpers ---------------------------------------------- #


def test_parse_session_id() -> None:
    assert _parse_session_id(_STDOUT) == "ses_abc"
    assert _parse_session_id("not json\n" + _STDOUT) == "ses_abc"
    assert _parse_session_id("") == ""


def test_read_assistant_text_only_text_events(tmp_path) -> None:
    p = tmp_path / "t.json"
    p.write_text(
        "\n".join(
            [
                json.dumps({"type": "step_start", "part": {"type": "step-start"}}),
                json.dumps({"type": "text", "part": {"type": "text", "text": "one"}}),
                json.dumps({"type": "tool_use", "part": {"type": "tool"}}),
                json.dumps({"type": "text", "part": {"type": "text", "text": "two"}}),
            ]
        )
    )
    assert read_assistant_text(p) == "one\ntwo"


def test_read_assistant_text_missing_file(tmp_path) -> None:
    assert read_assistant_text(tmp_path / "nope.json") == ""


def test_diagnose_overload_and_auth() -> None:
    assert "overload" in _diagnose("This model is experiencing high demand").lower()
    assert "overload" in _diagnose("Error 503 UNAVAILABLE").lower()
    # An auth failure is not retryable, so the hint points at the two commands
    # that show which credential is in play.
    auth = _diagnose("API key not valid (403)")
    assert "viral-bench models" in auth
    assert "viral-bench doctor" in auth


def test_diagnose_translates_a_missing_entitlement() -> None:
    """A provider reports a model the account was never entitled to as a
    not-found, which reads like a typo in the model id. The fix is an opt-in on
    the account, not a retry and not a spelling check, so the hint has to say so
    -- otherwise the next hour goes into re-reading the model string."""
    hint = _diagnose(
        "Model `some-model-test` was not found or your project does not have "
        "access to it."
    )
    assert "not enabled on the account" in hint
    assert "catalogue" in hint


def test_read_tool_calls_parses_tool_events(tmp_path) -> None:
    p = tmp_path / "t.json"
    p.write_text(
        "\n".join(
            [
                json.dumps({"type": "text", "part": {"type": "text", "text": "hi"}}),
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "type": "tool",
                            "tool": "browser_navigate",
                            "state": {"input": {"url": "http://x/"}},
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_use",
                        "part": {
                            "type": "tool",
                            "tool": "bash",
                            "state": {"input": {"command": "ls"}},
                        },
                    }
                ),
            ]
        )
    )
    calls = read_tool_calls(p)
    assert [name for name, _ in calls] == ["browser_navigate", "bash"]
    assert calls[0][1]["url"] == "http://x/"
    assert calls[1][1]["command"] == "ls"


def test_read_tool_calls_missing_file(tmp_path) -> None:
    assert read_tool_calls(tmp_path / "nope.json") == []


def test_harness_autodetects_browser(monkeypatch) -> None:
    import viral_bench.founder.opencode_agents as oa

    monkeypatch.setattr(oa, "browser_prereqs_ok", lambda: False)
    h = OpenCodeHarness(
        MODEL,
        structure=RoundTableTeam(max_rounds=1),
        preflight=False,
        browser_tools=True,
    )
    assert h.browser_tools is False  # gracefully disabled when prereqs missing

    monkeypatch.setattr(oa, "browser_prereqs_ok", lambda: True)
    h2 = OpenCodeHarness(
        MODEL,
        structure=RoundTableTeam(max_rounds=1),
        preflight=False,
        browser_tools=True,
    )
    assert h2.browser_tools is True
    assert _diagnose("some unrelated error") == ""


# -- git containment (see BuildWorkspace.init_git_boundary) ------------------- #


def test_build_env_ceilings_git_below_the_checkout(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path / "checkout" / "builds"))
    monkeypatch.delenv("GIT_CEILING_DIRECTORIES", raising=False)

    runner = OpenCodeRunner(MODEL, binary="/bin/true", preflight=False)
    env = runner._build_env(tmp_path / "opencode.json")

    # git must not be able to walk up into the checkout looking for a repo.
    assert env["GIT_CEILING_DIRECTORIES"] == str(tmp_path / "checkout")


def test_build_env_keeps_an_inherited_ceiling(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path / "checkout" / "builds"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", "/outer")

    runner = OpenCodeRunner(MODEL, binary="/bin/true", preflight=False)
    env = runner._build_env(tmp_path / "opencode.json")

    assert env["GIT_CEILING_DIRECTORIES"] == f"{tmp_path / 'checkout'}:/outer"


def test_run_turn_launches_opencode_in_the_app_dir(monkeypatch, tmp_path) -> None:
    """Otherwise opencode inherits the caller's cwd (the repo root) and stray
    shell commands run against the ViralBench checkout instead of the built app."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    # Built before the patch so the workspace's own git setup is not captured.
    workspace = _workspace(tmp_path)
    seen: list[dict] = []

    def fake_run(argv, **kwargs):
        seen.append(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout=_STDOUT, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    runner = OpenCodeRunner(MODEL, binary="/bin/true", preflight=False)
    assert runner.prepare(workspace) is None
    runner.run_turn(
        "hi",
        workspace=workspace,
        phase="design",
        turn="design",
        continue_session=False,
    )

    assert seen[0]["cwd"] == str(workspace.app_dir)


def test_prompt_goes_on_stdin_not_the_command_line(tmp_path, monkeypatch) -> None:
    """A prompt on argv is a prompt in /proc/<pid>/cmdline, and that is lethal.

    The client-app guidance contains the literal string
    "python3 -m http.server". A QA agent that starts the app that way and cleans
    up with `pkill -f "python3 -m http.server"` matched the opencode process
    running its own turn. Four fleet builds died that way, three of them one
    model's -- a fabricated capability difference.
    """
    from viral_bench.founder.harness import OpenCodeRunner
    from viral_bench.founder.workspace import BuildWorkspace

    captured: dict = {}

    class _Proc:
        stdout = '{"sessionID": "ses_x"}'
        stderr = ""
        returncode = 0

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["input"] = kwargs.get("input")
        return _Proc()

    monkeypatch.setattr("subprocess.run", fake_run)
    workspace = BuildWorkspace("b1", root=tmp_path / "ws").create()
    runner = OpenCodeRunner(MODEL, binary="/bin/true", preflight=False)
    assert runner.prepare(workspace) is None

    prompt = "serve it with python3 -m http.server on port 8000"
    result = runner.run_turn(
        prompt,
        workspace=workspace,
        phase="r1_a4_qa_finisher",
        turn="team",
        continue_session=False,
    )
    assert result.returncode == 0
    assert captured["input"] == prompt
    assert prompt not in captured["argv"]
    assert not any("http.server" in str(a) for a in captured["argv"])


# -- reaping app servers agents leave running --------------------------------- #

_LINUX_ONLY = pytest.mark.skipif(
    not Path("/proc").is_dir(), reason="reaping reads /proc (Linux only)"
)


def _spawn_orphan_in(cwd) -> int:
    """Start a detached `sleep` in ``cwd``, orphaned like an agent's app server.

    Faithful to what opencode does: its bash tool spawns detached, so a
    backgrounded server ends up in its OWN session (pgid == sid == the shell's
    pid, unrelated to opencode's group) and is reparented away when the shell
    exits. That is why the reaper matches on cwd and not on the process group.
    """
    # The redirect matters: without it the backgrounded child inherits the
    # captured stdout pipe and subprocess.run blocks until the child exits.
    proc = subprocess.run(
        ["sh", "-c", "sleep 120 >/dev/null 2>&1 & echo $!"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        start_new_session=True,
    )
    return int(proc.stdout.strip())


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@_LINUX_ONLY
def test_reaper_kills_a_server_left_running_in_the_workspace(tmp_path) -> None:
    workspace = BuildWorkspace("b1", root=tmp_path / "ws").create()
    pid = _spawn_orphan_in(workspace.app_dir)
    try:
        assert _is_alive(pid)
        reaped = reap_workspace_processes(workspace.root, grace_s=5.0)
        assert pid in [p for p, _ in reaped]
        assert "sleep 120" in dict(reaped)[pid]
        assert not _is_alive(pid)
    finally:
        if _is_alive(pid):
            os.kill(pid, signal.SIGKILL)


@_LINUX_ONLY
def test_reaper_leaves_processes_outside_the_workspace_alone(tmp_path) -> None:
    """Scoping is the whole point: one build must never reap another's server."""
    mine = BuildWorkspace("b1", root=tmp_path / "mine").create()
    theirs = BuildWorkspace("b2", root=tmp_path / "theirs").create()
    pid = _spawn_orphan_in(theirs.app_dir)
    try:
        assert reap_workspace_processes(mine.root, grace_s=1.0) == []
        assert _is_alive(pid)
    finally:
        os.kill(pid, signal.SIGKILL)


@_LINUX_ONLY
def test_reaper_can_be_disabled_for_hand_debugging(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(NO_REAP_ENV, "1")
    workspace = BuildWorkspace("b1", root=tmp_path / "ws").create()
    pid = _spawn_orphan_in(workspace.app_dir)
    try:
        assert reap_workspace_processes(workspace.root, grace_s=1.0) == []
        assert _is_alive(pid)
    finally:
        os.kill(pid, signal.SIGKILL)


@_LINUX_ONLY
def test_run_turn_reaps_and_records_what_the_agent_left_behind(
    tmp_path, monkeypatch
) -> None:
    fake_run, _calls = _fake_run_factory([0])
    workspace = BuildWorkspace("b1", root=tmp_path / "ws").create()
    runner = OpenCodeRunner(MODEL, binary="/bin/true", preflight=False)
    assert runner.prepare(workspace) is None
    pid = _spawn_orphan_in(workspace.app_dir)
    # Patch only around the turn, so the spawn above uses the real subprocess.
    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        runner.run_turn(
            "go",
            workspace=workspace,
            phase="r1_a4_qa_finisher",
            turn="team",
            continue_session=False,
        )
        assert not _is_alive(pid)
        log = (workspace.root / "reaped.log").read_text(encoding="utf-8")
        assert "r1_a4_qa_finisher" in log
        assert str(pid) in log
        assert "sleep 120" in log
    finally:
        if _is_alive(pid):
            os.kill(pid, signal.SIGKILL)


# -- task/subagent telemetry (dynamic mode) ---------------------------------- #


def _task_event(
    *, sub="general", sid="ses-kid", start=None, end=None, status="completed", **extra
) -> str:
    state: dict = {
        "status": status,
        "input": {"subagent_type": sub, "description": "d", "prompt": "p", **extra},
        "metadata": {
            "sessionId": sid,
            "parentSessionId": "ses-main",
            "model": {"providerID": "google-vertex", "modelID": "gemini-test"},
        },
    }
    if start is not None:
        state["time"] = {"start": start, "end": end}
    return json.dumps(
        {"type": "tool", "part": {"type": "tool", "tool": "task", "state": state}}
    )


def test_read_task_spawns_extracts_the_orchestration(tmp_path) -> None:
    path = tmp_path / "t.json"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "text", "part": {"type": "text", "text": "hi"}}),
                _task_event(sub="general", sid="ses-a", start=1000, end=5000),
                # a resumed subagent carries the task_id it is continuing
                _task_event(
                    sub="explore", sid="ses-b", start=2000, end=3000, task_id="ses-a"
                ),
                # a non-task tool must not be picked up
                json.dumps(
                    {
                        "type": "tool",
                        "part": {
                            "type": "tool",
                            "tool": "bash",
                            "state": {"input": {"command": "ls"}},
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    spawns = read_task_spawns(path)
    assert [s["subagent_type"] for s in spawns] == ["general", "explore"]
    assert spawns[0]["session_id"] == "ses-a"
    assert spawns[0]["model"] == "google-vertex/gemini-test"
    assert spawns[0]["resumed_task_id"] == ""
    assert spawns[1]["resumed_task_id"] == "ses-a"


def test_read_task_spawns_tolerates_a_missing_transcript(tmp_path) -> None:
    assert read_task_spawns(tmp_path / "nope.json") == []


def test_concurrent_spawn_peak_counts_overlap(tmp_path) -> None:
    overlapping = [
        {"start_ms": 0, "end_ms": 100},
        {"start_ms": 50, "end_ms": 150},
        {"start_ms": 60, "end_ms": 70},
    ]
    assert concurrent_spawn_peak(overlapping) == 3


def test_concurrent_spawn_peak_treats_abutting_tasks_as_sequential() -> None:
    """Six subagents run one at a time is a different orchestration from six at
    once, and back-to-back tasks must not be scored as the latter."""
    sequential = [
        {"start_ms": 0, "end_ms": 100},
        {"start_ms": 100, "end_ms": 200},
        {"start_ms": 200, "end_ms": 300},
    ]
    assert concurrent_spawn_peak(sequential) == 1


def test_concurrent_spawn_peak_skips_untimed_spawns() -> None:
    assert concurrent_spawn_peak([{"start_ms": None, "end_ms": None}]) == 0


def test_dynamic_turns_get_their_own_timeout() -> None:
    runner = OpenCodeRunner(
        MODEL,
        design_timeout_s=1,
        build_timeout_s=2,
        team_turn_timeout_s=3,
        dynamic_turn_timeout_s=4,
        preflight=False,
    )
    assert runner.timeout_for("design") == 1
    assert runner.timeout_for("build") == 2
    assert runner.timeout_for("team") == 3
    assert runner.timeout_for("dynamic") == 4


def test_dynamic_harness_ships_no_bench_agents_or_skills() -> None:
    """Dynamic mode must not hand the model a team the harness designed.

    The team arm defines four opencode agents and six skills, and if any of that
    leaked into the dynamic config the mode would be measuring the harness's own
    founder process again, with the labels filed off.
    """
    harness = OpenCodeHarness(
        MODEL,
        structure=DynamicOrchestrator(max_turns=2),
        browser_tools=False,
        preflight=False,
    )
    config = json.loads(harness._config_json())
    assert "agent" not in config
    assert harness._runner._skills == []
    # Delegation is the one capability the whole mode rests on.
    assert config["permission"]["task"] == "allow"


def test_dynamic_harness_gives_the_browser_to_every_agent(monkeypatch) -> None:
    """Team mode disables browser tools globally and re-enables them per role.
    Here nobody knows which agent will want to look at the app, so it is on for
    all of them -- including subagents the model invents."""
    import viral_bench.founder.opencode_agents as oa

    monkeypatch.setattr(oa, "browser_prereqs_ok", lambda *a, **k: True)
    harness = OpenCodeHarness(
        MODEL,
        structure=DynamicOrchestrator(max_turns=2),
        browser_tools=True,
        preflight=False,
    )
    config = json.loads(harness._config_json())
    assert config["tools"] == {"browser_*": True}
    assert config["mcp"]["browser"]["enabled"] is True


def test_session_tree_walks_children_and_grandchildren(tmp_path, monkeypatch) -> None:
    """Nested delegation is invisible in the parent transcript, and this is the
    only place it shows up."""
    import sqlite3

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    db = tmp_path / "opencode" / "opencode.db"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE session (id TEXT, parent_id TEXT, title TEXT)")
    con.executemany(
        "INSERT INTO session VALUES (?, ?, ?)",
        [
            ("root", None, "founder"),
            ("kid1", "root", "build the backend"),
            ("kid2", "root", "build the UI"),
            ("grandkid", "kid1", "write the migrations"),
            ("unrelated", None, "another build entirely"),
        ],
    )
    con.commit()
    con.close()

    tree = session_tree("root")
    assert tree["sessions"] == 3  # descendants only, and nothing unrelated
    assert tree["max_depth"] == 2  # a subagent delegated in turn
    assert "write the migrations" in tree["titles"]


def test_session_tree_is_best_effort(tmp_path, monkeypatch) -> None:
    """opencode's session store is internal with no compatibility promise, so a
    schema change must cost telemetry, never a build."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert session_tree("root") == {"sessions": 0, "max_depth": 0, "titles": []}
    assert session_tree("") == {"sessions": 0, "max_depth": 0, "titles": []}

    db = tmp_path / "opencode" / "opencode.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_text("not a database", encoding="utf-8")
    assert session_tree("root")["sessions"] == 0


# -- trajectory capture: thinking + the session store ------------------------ #


def _thinking_stdout(*texts: str) -> str:
    """A JSON transcript carrying reasoning events alongside the assistant text."""
    lines = [
        json.dumps(
            {
                "type": "reasoning",
                "sessionID": "ses_abc",
                "part": {"type": "reasoning", "text": text},
            }
        )
        for text in texts
    ]
    lines.append(_STDOUT)
    return "\n".join(lines)


def test_turn_asks_opencode_for_thinking(monkeypatch, tmp_path) -> None:
    """The whole original bug in one assertion: opencode's `--thinking` defaults
    to OFF, so every build was billed for reasoning tokens and then discarded
    the reasoning."""
    fake_run, calls = _fake_run_factory([0, 0])
    monkeypatch.setattr(subprocess, "run", fake_run)

    OpenCodeHarness(MODEL, binary="/bin/true", preflight=False).run(
        IDEA, _workspace(tmp_path)
    )
    assert "--thinking" in calls[0]
    assert "--thinking" in calls[1]


def test_thinking_capture_can_be_switched_off(monkeypatch, tmp_path) -> None:
    fake_run, calls = _fake_run_factory([0, 0])
    monkeypatch.setattr(subprocess, "run", fake_run)

    OpenCodeHarness(
        MODEL, binary="/bin/true", preflight=False, capture_thinking=False
    ).run(IDEA, _workspace(tmp_path))
    assert "--thinking" not in calls[0]


def test_claude_asks_for_summarized_thinking_but_gemini_does_not() -> None:
    """Claude returns an ENCRYPTED thinking block -- signature present, text
    empty -- unless the request asks for a summary. Gemini needs nothing, and
    must keep the request it always had."""
    claude = OpenCodeRunner(CLAUDE, binary="/bin/true", preflight=False)
    options = json.loads(claude._config_json())["provider"]["google-vertex-anthropic"][
        "models"
    ]["claude-test"]["options"]
    assert options["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert options["effort"] == "high"

    gemini = OpenCodeRunner(MODEL, binary="/bin/true", preflight=False)
    entry = json.loads(gemini._config_json())["provider"]["google-vertex"]["models"][
        "gemini-test"
    ]
    assert "options" not in entry


def test_the_thinking_shape_follows_the_transport_not_the_provider() -> None:
    """Both providers that speak the Anthropic wire format need the same option.

    Keying this off the transport rather than the provider id is what makes a
    second Anthropic-speaking provider work with no change here -- and it is
    also why the direct and the cloud-hosted Claude surfaces cannot drift apart.

    The shape itself is load-bearing: `thinking.type: enabled` is what opencode's
    own `--variant high` resolves to, and the Claude surfaces answer HTTP 400
    for it, so asking for it would fail the build outright rather than capture
    anything.
    """
    for model in (CLAUDE, "anthropic/claude-test"):
        options = _thinking_options(model)
        assert options["thinking"]["type"] == "adaptive"
        assert options["thinking"]["display"] == "summarized"
        assert "enabled" not in json.dumps(options)
    # A model on any other transport gets no options at all, so its request is
    # byte-identical to what it was before thinking capture existed.
    assert _thinking_options(MODEL) == {}
    assert _thinking_options("openai/gpt-test") == {}


def test_read_reasoning_returns_the_chain_of_thought(tmp_path) -> None:
    path = tmp_path / "t.json"
    path.write_text(_thinking_stdout("first thought", "second thought"), "utf-8")
    assert read_reasoning(path) == "first thought\nsecond thought"
    assert read_reasoning(tmp_path / "missing.json") == ""


def test_reasoning_is_kept_out_of_the_assistant_text(tmp_path) -> None:
    """Ship/done signals are matched against read_assistant_text. A model that
    only CONSIDERS shipping in its thinking must not end its own build early --
    that would penalise models which think out loud."""
    path = tmp_path / "t.json"
    path.write_text(_thinking_stdout("maybe I should emit READY_TO_SHIP now"), "utf-8")
    assert "READY_TO_SHIP" not in read_assistant_text(path)
    assert "READY_TO_SHIP" in read_reasoning(path)


def test_phase_result_counts_reasoning(monkeypatch, tmp_path) -> None:
    fake_run, _ = _fake_run_factory([0, 0], stdout=_thinking_stdout("aaa", "bbbb"))
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = OpenCodeHarness(MODEL, binary="/bin/true", preflight=False).run(
        IDEA, _workspace(tmp_path)
    )
    assert result.phases[0].reasoning_parts == 2
    assert result.phases[0].reasoning_chars == len("aaa\nbbbb")


def _store(tmp_path, monkeypatch, *, rows=None):
    """Build a fake opencode session store shaped like the real one."""
    import sqlite3

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    db = tmp_path / "opencode" / "opencode.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE session (id TEXT, parent_id TEXT, title TEXT, agent TEXT, "
        "model TEXT, cost REAL, tokens_input INT, tokens_output INT, "
        "tokens_reasoning INT, tokens_cache_read INT, tokens_cache_write INT, "
        "time_created INT)"
    )
    con.execute(
        "CREATE TABLE message (id TEXT, session_id TEXT, time_created INT, data TEXT)"
    )
    con.execute(
        "CREATE TABLE part (id TEXT, message_id TEXT, session_id TEXT, "
        "time_created INT, data TEXT)"
    )
    con.executemany(
        "INSERT INTO session VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows
        or [
            ("root", None, "founder", "build", "m", 0.5, 10, 20, 30, 0, 0, 1),
            ("kid", "root", "the subagent", "general", "m", 0.25, 5, 6, 7, 0, 0, 2),
            ("other", None, "a different build", "build", "m", 9.0, 1, 1, 1, 0, 0, 3),
        ],
    )
    con.executemany(
        "INSERT INTO message VALUES (?,?,?,?)",
        [
            ("m_user", "root", 1, json.dumps({"role": "user", "agent": "build"})),
            ("m_ai", "root", 2, json.dumps({"role": "assistant", "agent": "build"})),
            ("m_kid", "kid", 3, json.dumps({"role": "assistant", "agent": "general"})),
            ("m_other", "other", 4, json.dumps({"role": "assistant"})),
        ],
    )
    con.executemany(
        "INSERT INTO part VALUES (?,?,?,?,?)",
        [
            ("p1", "m_user", "root", 1, json.dumps({"type": "text", "text": "PROMPT"})),
            ("p2", "m_ai", "root", 2, json.dumps({"type": "reasoning", "text": "hmm"})),
            ("p3", "m_kid", "kid", 3, json.dumps({"type": "text", "text": "SUBAGENT"})),
            ("p4", "m_other", "other", 4, json.dumps({"type": "text", "text": "NOPE"})),
        ],
    )
    con.commit()
    con.close()
    return db


def test_dump_session_trace_captures_prompts_and_subagents(
    tmp_path, monkeypatch
) -> None:
    """The three things stdout never has: the prompt that was sent, the
    subagent's own session, and the file patches."""
    _store(tmp_path, monkeypatch)
    dest = tmp_path / "out" / "root.jsonl"

    assert dump_session_trace("root", dest) > 0

    records = [json.loads(line) for line in dest.read_text("utf-8").splitlines()]
    texts = [r["part"].get("text") for r in records if r.get("record") == "part"]
    assert "PROMPT" in texts  # the input, absent from every stdout transcript
    assert "SUBAGENT" in texts  # a child session, filtered out of stdout
    assert "hmm" in texts
    # A user prompt is attributable as one, so an SFT consumer can pair it up.
    prompt = next(r for r in records if r.get("part", {}).get("text") == "PROMPT")
    assert prompt["role"] == "user"


def test_dump_session_trace_never_dumps_unrelated_sessions(
    tmp_path, monkeypatch
) -> None:
    """Under the fleet the store is private per cell, but a bare `viral-bench
    found` shares the global one -- dumping that would copy out every unrelated
    session on the machine."""
    _store(tmp_path, monkeypatch)
    dest = tmp_path / "out" / "root.jsonl"
    dump_session_trace("root", dest)

    body = dest.read_text("utf-8")
    assert "NOPE" not in body
    assert "a different build" not in body


def test_dump_session_trace_is_best_effort(tmp_path, monkeypatch) -> None:
    """Telemetry must never fail a build: the store is opencode's internal
    format with no compatibility promise."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert dump_session_trace("root", tmp_path / "a.jsonl") == 0
    assert dump_session_trace("", tmp_path / "b.jsonl") == 0

    db = tmp_path / "opencode" / "opencode.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_text("not a database", encoding="utf-8")
    assert dump_session_trace("root", tmp_path / "c.jsonl") == 0


def test_dump_session_trace_survives_a_cycle(tmp_path, monkeypatch) -> None:
    """A malformed store must not hang a build."""
    _store(
        tmp_path,
        monkeypatch,
        rows=[
            ("root", "kid", "founder", "a", "m", 0, 0, 0, 0, 0, 0, 1),
            ("kid", "root", "child", "a", "m", 0, 0, 0, 0, 0, 0, 2),
        ],
    )
    assert dump_session_trace("root", tmp_path / "cyc.jsonl") > 0


def test_turn_dumps_the_session_store(monkeypatch, tmp_path) -> None:
    _store(tmp_path / "store", monkeypatch)
    fake_run, _ = _fake_run_factory(
        [0, 0],
        stdout=json.dumps(
            {"type": "text", "sessionID": "root", "part": {"type": "text", "text": "x"}}
        ),
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    workspace = _workspace(tmp_path)
    result = OpenCodeHarness(MODEL, binary="/bin/true", preflight=False).run(
        IDEA, workspace
    )

    dumped = workspace.transcript_dir / "sessions" / "root.jsonl"
    assert dumped.is_file()
    assert result.phases[0].sessions_records > 0
    assert result.phases[0].sessions_path == str(dumped)


def test_timed_out_turn_still_dumps_what_it_had(monkeypatch, tmp_path) -> None:
    """The wall-clock backstop SIGKILLs a wedged turn, and that turn's reasoning
    is exactly the evidence for why it wedged. Dumping only on the happy path
    would lose it."""
    _store(tmp_path / "store", monkeypatch)

    real_run = subprocess.run

    def fake_run(argv, **kwargs):
        if argv and str(argv[0]) == "git":
            return real_run(argv, **kwargs)
        raise subprocess.TimeoutExpired(
            argv,
            1.0,
            output=json.dumps(
                {
                    "type": "text",
                    "sessionID": "root",
                    "part": {"type": "text", "text": "partial"},
                }
            ),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    workspace = _workspace(tmp_path)
    result = OpenCodeHarness(MODEL, binary="/bin/true", preflight=False).run(
        IDEA, workspace
    )

    assert result.phases[0].timed_out
    assert (workspace.transcript_dir / "sessions" / "root.jsonl").is_file()
    assert result.phases[0].sessions_records > 0
