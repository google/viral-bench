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

"""Collaboration toolsets: the medium a founder team collaborates *through*.

A collaboration *toolset* is orthogonal to the collaboration *structure* (see
:mod:`viral_bench.founder.structures`): the structure decides the sequence of
turns (who acts, in which round), the toolset decides *how teammates talk to each
other* during those turns. They compose freely.

One toolset ships today, used only by the four-agent
:class:`~viral_bench.founder.structures.RoundTableTeam` (the solo founder has no
one to collaborate with):

* :class:`LocalToolset` (default, ``--collab local``) -- teammates collaborate
  through the shared working directory: the design lives in ``DESIGN.md`` and
  short notes in ``TEAM_NOTES.md`` alongside the code.

:class:`CollaborationToolset` is the extension point. A toolset that routes
collaboration through some external surface (a chat service, a shared document
store, email) only has to satisfy that Protocol and register itself in
:data:`TOOLSETS`, and nothing in the structures or the harness needs to change. Such
a toolset may want :func:`files_to_strip` to remove ``DESIGN.md`` too, so the
working directory stays app-source-only when the design lives elsewhere.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from viral_bench.founder.roles import Role

__all__ = [
    "CollaborationToolset",
    "LocalToolset",
    "CollabError",
    "build_toolset",
    "TOOLSETS",
    "DESIGN_FILE",
    "SCRATCH_FILES",
    "files_to_strip",
]

#: The genuine design deliverable kept in local mode (crowd can read it).
DESIGN_FILE = "DESIGN.md"

#: Pure coordination scratch -- never part of the shipped app in any mode.
SCRATCH_FILES: tuple[str, ...] = ("TEAM_NOTES.md", "HANDOFF.md")


def files_to_strip(collab: str) -> tuple[str, ...]:
    """Return the non-source files to remove from the app before shipping.

    Local mode keeps ``DESIGN.md`` (a real deliverable) and strips only scratch.
    A toolset that hosts the design outside the working directory should strip
    ``DESIGN.md`` as well, so the shipped app stays source-only.
    """
    if collab != "local":
        return (DESIGN_FILE, *SCRATCH_FILES)
    return SCRATCH_FILES


class CollabError(RuntimeError):
    """Raised when a collaboration toolset cannot be constructed or prepared."""


@runtime_checkable
class CollaborationToolset(Protocol):
    """Grants each agent turn a way to collaborate with its teammates.

    The structure calls :meth:`prepare` once per build, then for every turn reads
    :meth:`turn_env` (env merged into that opencode turn) and
    :meth:`collaboration_brief` (medium-specific instructions appended to that
    turn's prompt). :meth:`metadata` is recorded on the build, and
    :meth:`cleanup` runs after the build.
    """

    name: str

    def prepare(self, build_id: str, roles: list[Role]) -> None: ...
    def turn_env(self, agent_index: int) -> dict[str, str]: ...
    def collaboration_brief(self, agent_index: int) -> str: ...
    def metadata(self) -> dict: ...
    def cleanup(self) -> None: ...


class LocalToolset:
    """Default toolset: collaboration through the shared working directory."""

    name = "local"

    def prepare(self, build_id: str, roles: list[Role]) -> None:
        self._roles = list(roles)

    def turn_env(self, agent_index: int) -> dict[str, str]:
        return {}

    def collaboration_brief(self, agent_index: int) -> str:
        scratch = SCRATCH_FILES[0]
        return f"""\
Collaborate through the shared working directory -- it is your only channel:
- Keep the evolving design and decisions in `{DESIGN_FILE}`.
- Leave short notes for your teammates in `{scratch}`: what you changed, what
  still needs doing, and any decisions or gotchas. Read it before you act.
- Read your teammates' code to see the current state before changing it.
`{DESIGN_FILE}` ships with the app; `{scratch}` is scratch and is removed before
shipping."""

    def metadata(self) -> dict:
        return {"collab": self.name}

    def cleanup(self) -> None:
        return None


#: Registry of available toolsets (names accepted by ``--collab``).
TOOLSETS = ("local",)


def build_toolset(name: str, *, n_agents: int) -> CollaborationToolset:
    """Construct a collaboration toolset by name, validating requirements early.

    Args:
        name: currently only ``"local"``.
        n_agents: Team size. Unused by the local toolset, but part of the
            signature because a toolset may legitimately be team-only.

    Raises:
        CollabError: If ``name`` is not a known toolset.
    """
    if name == "local":
        return LocalToolset()
    raise CollabError(f"unknown collab toolset {name!r}. Available: {list(TOOLSETS)}")
