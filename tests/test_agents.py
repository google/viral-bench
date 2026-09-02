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

"""Tests for translating roles into opencode agent config."""

from __future__ import annotations

import viral_bench.founder.opencode_agents as oa
from viral_bench.founder.opencode_agents import (
    BROWSER_MCP_NAME,
    DEFAULT_BROWSER_COMMAND,
    browser_prereqs_ok,
    build_agents_config,
    build_dynamic_config,
    skills_for_roles,
)
from viral_bench.founder.roles import roles_for

MODEL = "google-vertex/gemini-test"


def test_one_agent_per_role_keyed_by_role_key() -> None:
    cfg = build_agents_config(roles_for(4), model=MODEL)
    assert list(cfg["agent"]) == [
        "architect",
        "implementer",
        "designer",
        "qa_finisher",
    ]


def test_each_agent_has_required_opencode_fields() -> None:
    cfg = build_agents_config(roles_for(4), model=MODEL)
    for entry in cfg["agent"].values():
        assert entry["description"]  # required by opencode
        assert entry["mode"] == "primary"
        assert entry["model"] == MODEL
        assert entry["prompt"]  # role persona system prompt
        assert "permission" in entry


def test_web_permissions_are_differentiated() -> None:
    agents = build_agents_config(roles_for(4), model=MODEL)["agent"]
    assert agents["architect"]["permission"]["webfetch"] == "allow"
    assert agents["designer"]["permission"]["webfetch"] == "allow"
    assert agents["implementer"]["permission"]["webfetch"] == "deny"
    assert agents["qa_finisher"]["permission"]["webfetch"] == "deny"


def test_skills_are_gated_per_role() -> None:
    agents = build_agents_config(roles_for(4), model=MODEL)["agent"]
    arch_skill = agents["architect"]["permission"]["skill"]
    # deny-all first, then allow only this role's own skills
    assert arch_skill["*"] == "deny"
    assert arch_skill["runtime-architecture"] == "allow"
    # the architect cannot load the designer's skill
    assert "virality-playbook" not in arch_skill
    designer_skill = agents["designer"]["permission"]["skill"]
    assert designer_skill["virality-playbook"] == "allow"
    assert designer_skill["ux-polish"] == "allow"


def test_temperature_is_carried_through() -> None:
    agents = build_agents_config(roles_for(4), model=MODEL)["agent"]
    assert agents["designer"]["temperature"] == 0.6
    assert agents["qa_finisher"]["temperature"] == 0.1


def test_browser_tools_off_by_default() -> None:
    cfg = build_agents_config(roles_for(4), model=MODEL)
    assert "mcp" not in cfg
    for entry in cfg["agent"].values():
        assert "tools" not in entry or f"{BROWSER_MCP_NAME}_*" not in entry.get(
            "tools", {}
        )


def test_browser_tools_enabled_only_for_designer_and_qa() -> None:
    cfg = build_agents_config(roles_for(4), model=MODEL, browser_tools=True)
    # MCP server declared, disabled globally, enabled per wanting-role
    assert BROWSER_MCP_NAME in cfg["mcp"]
    assert cfg["tools"][f"{BROWSER_MCP_NAME}_*"] is False
    agents = cfg["agent"]
    assert agents["designer"]["tools"][f"{BROWSER_MCP_NAME}_*"] is True
    assert agents["qa_finisher"]["tools"][f"{BROWSER_MCP_NAME}_*"] is True
    assert "tools" not in agents["architect"]
    assert "tools" not in agents["implementer"]


def test_skills_for_roles_is_sorted_and_deduped() -> None:
    names = skills_for_roles(roles_for(4))
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert "virality-playbook" in names
    assert "runtime-architecture" in names
    assert "live-app-testing" in names  # QA's live-testing skill is installed


def test_qa_gets_live_app_testing_skill_gated() -> None:
    agents = build_agents_config(roles_for(4), model=MODEL)["agent"]
    qa_skill = agents["qa_finisher"]["permission"]["skill"]
    assert qa_skill["*"] == "deny"
    assert qa_skill["live-app-testing"] == "allow"
    assert qa_skill["release-checklist"] == "allow"
    # gated: another role cannot load QA's live-testing skill
    assert "live-app-testing" not in agents["architect"]["permission"]["skill"]


def test_browser_command_is_node_launched_not_npx() -> None:
    # Vendored MCP launched via node, so the runtime never needs npx/npm.
    assert DEFAULT_BROWSER_COMMAND[0] == "node"
    assert not any("npx" in part for part in DEFAULT_BROWSER_COMMAND)


def test_build_agents_config_threads_browser_command() -> None:
    cmd = ("node", "/x/cli.js", "--headless")
    cfg = build_agents_config(
        roles_for(4), model=MODEL, browser_tools=True, browser_command=cmd
    )
    assert cfg["mcp"][BROWSER_MCP_NAME]["command"] == list(cmd)


def test_browser_prereqs_ok(monkeypatch, tmp_path) -> None:
    entry = tmp_path / "cli.js"
    entry.write_text("// mcp")
    cmd = ("node", str(entry))
    monkeypatch.setattr(
        oa.shutil, "which", lambda x: "/usr/bin/node" if x == "node" else None
    )
    monkeypatch.setattr(oa, "_browser_available", lambda: True)
    assert browser_prereqs_ok(cmd) is True
    # no browser present -> not ok
    monkeypatch.setattr(oa, "_browser_available", lambda: False)
    assert browser_prereqs_ok(cmd) is False
    # vendored entry missing -> not ok
    monkeypatch.setattr(oa, "_browser_available", lambda: True)
    assert browser_prereqs_ok(("node", str(tmp_path / "missing.js"))) is False


def test_dynamic_config_defines_no_agents() -> None:
    """Dynamic mode's specialists are the model's, not the harness's."""
    cfg = build_dynamic_config(browser_tools=False)
    assert "agent" not in cfg
    assert cfg["permission"]["task"] == "allow"
    assert "mcp" not in cfg


def test_dynamic_config_enables_the_browser_for_everyone() -> None:
    cfg = build_dynamic_config(browser_tools=True, browser_command=("node", "/x.js"))
    # Inverted vs the team config, which disables globally and re-enables per role.
    assert cfg["tools"] == {"browser_*": True}
    assert cfg["mcp"][BROWSER_MCP_NAME]["command"] == ["node", "/x.js"]


def test_dynamic_config_preapproves_every_permission() -> None:
    """`--auto` approves prompts for the PRIMARY agent only.

    A subagent that trips a permission (measured: `general` writing under /tmp,
    gated by `external_directory: ask`) raises a prompt with nobody to answer it
    and the build hangs until the wall-clock backstop records `harness_timeout`.
    In a mode built on delegation that is the common path, not an edge case.
    """
    perm = build_dynamic_config(browser_tools=False)["permission"]
    assert perm["external_directory"] == "allow"
    # A catch-all so a permission nobody thought of cannot wedge a subagent, with
    # "*" FIRST because opencode resolves these last-match-wins.
    assert next(iter(perm)) == "*"
    assert perm["*"] == "allow"
    # `question` is the sole exception: it waits for a human, and there is none.
    assert perm["question"] == "deny"
    assert set(perm.values()) == {"allow", "deny"}
