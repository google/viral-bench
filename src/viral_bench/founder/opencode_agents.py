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

"""Translate founder roles into opencode agent configuration.

opencode lets you define named **agents**, each with its own system prompt,
sampling temperature, tool **permissions**, and (via MCP) extra tools -- selected
per run with ``opencode run --agent <name>`` (see
https://opencode.ai/docs/agents/). This module turns the founder
:class:`~viral_bench.founder.roles.Role` specs into that config so each
specialist runs as a distinct, "levelled-up" agent rather than a
generic build agent:

* each role becomes an ``agent`` entry keyed by ``role.key`` (the value passed to
  ``--agent``),
* its ``permission`` map encodes differentiated tool access (e.g. web research on
  for the Architect/Designer, off for the Implementer/QA),
* its ``permission.skill`` map gates on-demand skills to that role alone, and
* when browser tooling is enabled, the roles that want it get a real browser MCP
  (so the Designer/QA can render and click the running app), disabled for
  everyone else.

Everything here is a pure function of the roles + options, so it is easy to unit
test without launching opencode.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from viral_bench.founder.roles import Role

__all__ = [
    "BROWSER_MCP_NAME",
    "DEFAULT_BROWSER_COMMAND",
    "browser_prereqs_ok",
    "build_agents_config",
    "build_dynamic_config",
    "skills_for_roles",
]

#: Name of the optional browser MCP server. Its tools are exposed as ``browser_*``.
BROWSER_MCP_NAME = "browser"

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Entry point of the browser MCP, vendored into the repo so the runtime only needs
#: ``node`` (no ``npx``/``npm`` at run time). Produced by ``tooling/browser-mcp``.
VENDORED_MCP_ENTRY = (
    _REPO_ROOT
    / "tooling"
    / "browser-mcp"
    / "node_modules"
    / "@playwright"
    / "mcp"
    / "cli.js"
)

#: System Chrome/Chromium binaries the MCP's ``chrome`` channel can drive. The
#: system browser channel is used rather than Playwright's bundled Chromium, so
#: Playwright's exact browser revision need not be matched at run time.
_CHROME_BINARIES: tuple[str, ...] = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
    "microsoft-edge",
)

#: Default command for the browser MCP: launch the vendored Playwright MCP with the
#: system ``node`` against a headless system Chrome. No ``npx`` needed at run time.
DEFAULT_BROWSER_COMMAND: tuple[str, ...] = (
    "node",
    str(VENDORED_MCP_ENTRY),
    "--headless",
    "--browser",
    "chrome",
    "--isolated",
)


def _browser_available() -> bool:
    """True if a Chrome/Chromium the MCP's ``chrome`` channel can drive is on PATH."""
    return any(shutil.which(exe) for exe in _CHROME_BINARIES)


def browser_prereqs_ok(command: tuple[str, ...] = DEFAULT_BROWSER_COMMAND) -> bool:
    """True if the browser MCP in ``command`` can launch on this host.

    Checks the launcher (``node``) is available, any referenced ``.js`` entry
    exists (the vendored MCP), and a Chromium/Chrome is present. Used by the
    harness to auto-disable the browser gracefully when it cannot run, so a build
    never fails merely because the optional browser tooling is missing.
    """
    launcher = command[0] if command else ""
    if not launcher or (shutil.which(launcher) is None and not Path(launcher).exists()):
        return False
    for arg in command[1:]:
        if arg.endswith(".js") and not Path(arg).is_file():
            return False
    return _browser_available()


# Baseline tool access every specialist gets, which roles override (e.g. deny web).
# ``--auto`` auto-approves anything not denied, but the baseline is spelled out so
# behaviour is explicit and stable regardless of the --auto default.
_BASE_PERMISSION: dict[str, object] = {
    "read": "allow",
    "glob": "allow",
    "grep": "allow",
    "list": "allow",
    "lsp": "allow",
    "edit": "allow",
    "bash": "allow",
    "todowrite": "allow",
    "webfetch": "allow",
    "websearch": "allow",
}


def _skill_permission(role: Role) -> dict[str, str]:
    """Gate skills to this role: deny all, then allow the role's own skills.

    ``*`` is listed first so the later, specific allows win (opencode resolves
    permission patterns with "last matching rule wins").
    """
    perm: dict[str, str] = {"*": "deny"}
    for name in role.skills:
        perm[name] = "allow"
    return perm


def _browser_tools_key() -> str:
    """Glob matching every tool from the browser MCP server."""
    return f"{BROWSER_MCP_NAME}_*"


def _agent_entry(role: Role, *, model: str, browser_tools: bool) -> dict:
    """Build one opencode ``agent`` entry for ``role``."""
    permission: dict[str, object] = {**_BASE_PERMISSION, **role.permissions}
    permission["skill"] = _skill_permission(role)

    entry: dict[str, object] = {
        "description": f"{role.title}: {role.mission}.",
        "mode": "primary",
        "model": model,
        "prompt": role.system_prompt,
        "permission": permission,
    }
    if role.temperature is not None:
        entry["temperature"] = role.temperature

    # Browser MCP: disabled globally (see build_agents_config), enabled only for
    # the roles that want it, and only when browser tooling is on for the build.
    tools: dict[str, bool] = {}
    if browser_tools and role.wants_browser:
        tools[_browser_tools_key()] = True
    if tools:
        entry["tools"] = tools
    return entry


def skills_for_roles(roles: list[Role]) -> list[str]:
    """Return the sorted, de-duplicated skill names used by ``roles``."""
    names: set[str] = set()
    for role in roles:
        names.update(role.skills)
    return sorted(names)


def build_dynamic_config(
    *,
    browser_tools: bool = False,
    browser_command: tuple[str, ...] = DEFAULT_BROWSER_COMMAND,
) -> dict:
    """Config for the dynamic founder: stock opencode, delegation switched on.

    Deliberately defines **no agents at all**. The dynamic mode runs the default
    ``build`` agent with no ``--agent`` flag, because every custom agent defined
    here is another decision taken away from the model -- and the model can define its
    own agents at run time anyway (see
    :data:`~viral_bench.founder.structures.AGENTS_DIRNAME`). What it configures is
    only the two things that would otherwise silently limit delegation:

    * **every permission pre-approved, explicitly.** This is not belt-and-braces,
      it is load-bearing, and it cost a wedged build to find out: ``--auto``
      approves permission prompts for the PRIMARY agent only. A *subagent* that
      trips one -- measured: ``general`` writing a scratch file under ``/tmp``,
      which opencode gates behind ``external_directory: ask`` -- raises a prompt
      with nobody to answer it, and the whole build sits there until the
      wall-clock backstop kills it and records ``harness_timeout``. In a mode
      built on delegation that is not an edge case, but the common path. So the
      map opens with ``"*": "allow"`` (later, more specific rules still win) and
      names ``external_directory`` and ``task`` outright. Anything less pre-approves
      only the prompts that happened to be thought of in advance.
    * the browser MCP, enabled **globally** rather than per-role. The team mode
      disables browser tools globally and re-enables them for the two roles that
      asked for one. Here nobody knows in advance which agent will want to look at
      the app, so every agent -- the founder and any subagent it spawns -- gets
      the browser. Without that, a dynamic build would be measured against team
      builds whose QA could see the rendered page while it could not.

    Returns a fragment ready to merge into the base opencode config.
    """
    fragment: dict[str, object] = {
        "permission": {
            # "*" first: opencode resolves permission patterns last-match-wins,
            # so the specific entries after it still take effect.
            "*": "allow",
            **_BASE_PERMISSION,
            "task": "allow",
            # Subagents write scratch outside the worktree constantly (/tmp pid
            # files, previews). Left at its default `ask`, that deadlocks the run.
            "external_directory": "allow",
            # The one thing NOT allowed, for the same reason everything else is:
            # there is no human in a benchmark run. `question` exists to ask the
            # user something and then wait, which for an unattended subagent is
            # the permission deadlock again wearing a different hat. Denied, so a
            # model that reaches for it gets an error and carries on instead of
            # burning the wall-clock budget waiting for an answer nobody will
            # give. It costs no real capability: there is nobody to reply.
            "question": "deny",
        }
    }
    if browser_tools:
        fragment["mcp"] = {
            BROWSER_MCP_NAME: {
                "type": "local",
                "command": list(browser_command),
                "enabled": True,
            }
        }
        # Note the inversion vs build_agents_config: enabled for everyone, not
        # disabled globally and re-enabled per role.
        fragment["tools"] = {_browser_tools_key(): True}
    return fragment


def build_agents_config(
    roles: list[Role],
    *,
    model: str,
    browser_tools: bool = False,
    browser_command: tuple[str, ...] = DEFAULT_BROWSER_COMMAND,
) -> dict:
    """Return opencode config fragments defining one agent per role.

    Args:
        roles: The specialist roles (the four-agent team). The solo founder does
            not use custom agents, so callers should not pass it here.
        model: The opencode ``provider/model`` each agent runs.
        browser_tools: If true, wire an optional browser MCP and enable it for the
            roles whose :attr:`~viral_bench.founder.roles.Role.wants_browser` is
            set (Designer, QA), disabled for everyone else.
        browser_command: Command used to launch the browser MCP server.

    Returns:
        A dict with an ``"agent"`` mapping (and, when ``browser_tools`` is on,
        ``"mcp"`` + a global ``"tools"`` disable) ready to merge into the base
        opencode config.
    """
    agents = {
        role.key: _agent_entry(role, model=model, browser_tools=browser_tools)
        for role in roles
    }
    fragment: dict[str, object] = {"agent": agents}

    if browser_tools and any(role.wants_browser for role in roles):
        fragment["mcp"] = {
            BROWSER_MCP_NAME: {
                "type": "local",
                "command": list(browser_command),
                "enabled": True,
            }
        }
        # Disable browser tools globally. Each agent re-enables if it wants them.
        fragment["tools"] = {_browser_tools_key(): False}

    return fragment
