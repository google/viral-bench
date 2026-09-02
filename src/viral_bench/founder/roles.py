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

"""Specialist roles for the founder team, with real, differentiated capabilities.

There are exactly **two** founder configurations (see
:mod:`viral_bench.founder.structures`):

* **Solo** (``roles_for(1)``) -- a single ``founder`` role owning every
  responsibility. Its prompts are byte-for-byte the original single-agent
  Design -> Build baseline; it does *not* get specialist tooling, so the solo
  path stays identical to the pre-team pipeline.
* **Team** (``roles_for(4)``) -- four specialists who collaborate over many
  rounds (:class:`~viral_bench.founder.structures.RoundTableTeam`):

  * **Architect** -- sets the tech stack + architecture that runs cleanly in the
    runtime and defines the single viral "wow".
  * **Implementer** -- builds the core features and the main loop.
  * **UX & Virality Designer** -- polish + the concrete shareable hook.
  * **QA & Finisher** -- verifies every success criterion and ships.

Each specialist is more than a persona: it maps to a distinct **opencode agent**
with its own system prompt, sampling ``temperature``, tool **permissions**, and
on-demand **skills** (see :mod:`viral_bench.founder.skills` and
:mod:`viral_bench.founder.opencode_agents`). That is what "levels up" a role
beyond a generic build agent -- e.g. the Architect gets web research tools and an
architecture skill; the Implementer runs focused with code-intelligence and no
web; the Designer gets creative sampling, design research, and (optionally) a
real browser to *see* the app; QA runs rigorous and low-temperature with a
release-verification skill.

The QA & Finisher role always sorts **last** so the final turn of any team run
leaves a complete, runnable app with a valid manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Responsibility",
    "Role",
    "ARCHITECTURE",
    "IMPLEMENTATION",
    "UX_VIRALITY",
    "QA",
    "ALL_RESPONSIBILITIES",
    "roles_for",
    "TEAM_SIZE",
    "VALID_TEAM_SIZES",
]

#: The one non-trivial team size (a full founder team). Solo is the other mode.
TEAM_SIZE = 4

#: The only two founder configurations we support: solo, or the full team.
VALID_TEAM_SIZES: tuple[int, ...] = (1, TEAM_SIZE)


@dataclass(frozen=True)
class Responsibility:
    """One of the fixed responsibilities a founder configuration must cover.

    Attributes:
        key: Stable identifier (e.g. ``"architecture"``).
        label: Human-readable label used in prompts and records.
        focus: What this responsibility is accountable for (injected into the
            owning specialist's turn prompts).
    """

    key: str
    label: str
    focus: str


ARCHITECTURE = Responsibility(
    key="architecture",
    label="Architecture & product direction",
    focus=(
        "the tech stack and architecture that will run cleanly in the run/test "
        "runtime, the core user experience, and the single viral 'wow' that makes "
        "people want to share this app"
    ),
)

IMPLEMENTATION = Responsibility(
    key="implementation",
    label="Core implementation",
    focus=(
        "implementing the core features and the main app loop so the success "
        "criteria are genuinely met"
    ),
)

UX_VIRALITY = Responsibility(
    key="ux_virality",
    label="UX polish & virality",
    focus=(
        "making the app delightful and genuinely shareable -- the polish and the "
        "concrete viral hook (e.g. a shareable link or a copy-to-clipboard result "
        "card) that drives organic spread"
    ),
)

QA = Responsibility(
    key="qa",
    label="QA & finish",
    focus=(
        "verifying the app meets every success criterion, fixing bugs, and "
        "guaranteeing it runs from a clean checkout with the exact command in the "
        "manifest"
    ),
)

#: The fixed union of responsibilities, in canonical order (A, I, D, Q).
ALL_RESPONSIBILITIES: tuple[Responsibility, ...] = (
    ARCHITECTURE,
    IMPLEMENTATION,
    UX_VIRALITY,
    QA,
)


@dataclass(frozen=True)
class Role:
    """One specialist's role, persona, and differentiated opencode capabilities.

    Attributes:
        key: Stable identifier used in transcripts/records and as the opencode
            agent name (e.g. ``"architect"``).
        title: Human-readable role title (e.g. ``"Architect"``).
        mission: One-line persona statement injected into the agent's prompts.
        responsibilities: The subset of :data:`ALL_RESPONSIBILITIES` this role
            owns (for the team, exactly one each; the union covers the full set).
        system_prompt: Stable opencode agent system prompt -- the persona plus
            guidance on how to use this role's special tools/skills. Empty for
            the solo founder (which uses opencode's default build agent).
        temperature: Sampling temperature for this role's opencode agent (lower =
            more rigorous, higher = more creative). ``None`` uses model defaults.
        permissions: opencode ``permission`` map for this role's agent -- the
            differentiated tool access (e.g. web research on/off).
        skills: Names of the on-demand skills this role may load (see
            :mod:`viral_bench.founder.skills`); gated to this role only.
        wants_browser: Whether this role benefits from the optional browser tool
            (so it can render/click the running app). Enabled only when browser
            tooling is turned on for the build.
    """

    key: str
    title: str
    mission: str
    responsibilities: tuple[Responsibility, ...]
    system_prompt: str = ""
    temperature: float | None = None
    permissions: dict = field(default_factory=dict)
    skills: tuple[str, ...] = ()
    wants_browser: bool = False

    @property
    def owns_qa(self) -> bool:
        """True if this role owns the QA & Finish responsibility."""
        return QA in self.responsibilities


# --------------------------------------------------------------------------- #
# Permission building blocks. ``--auto`` auto-approves anything not explicitly
# denied, so we only need to spell out what a role is *denied* (to keep it
# focused) or explicitly *allowed*. Web research is the main lever we toggle.
# --------------------------------------------------------------------------- #

# Deny outbound web tools -- keeps a role focused on the code in front of it.
_NO_WEB = {"webfetch": "deny", "websearch": "deny"}
# Allow outbound web research (libraries, APIs, design inspiration).
_WEB_RESEARCH = {"webfetch": "allow", "websearch": "allow"}


# -- The solo founder (baseline; no specialist tooling) ---------------------- #

_FOUNDER = Role(
    key="founder",
    title="Founder",
    mission="a startup founder-engineer building a real, shippable app end to end",
    responsibilities=ALL_RESPONSIBILITIES,
)


# -- The four team specialists ---------------------------------------------- #

_ARCHITECT = Role(
    key="architect",
    title="Architect",
    mission=(
        "the founding architect who sets the technical direction and the "
        "product's viral hook"
    ),
    responsibilities=(ARCHITECTURE,),
    system_prompt=(
        "You are the founding Architect on a small startup team. You decide the "
        "tech stack, the architecture, and the single viral 'wow', then scaffold "
        "the skeleton the others build on. You are equipped with web research "
        "tools (webfetch/websearch) and the `runtime-architecture` skill -- use "
        "them to choose a stack that runs cleanly in the target runtime and to "
        "de-risk decisions before your teammates depend on them. Bias toward "
        "simple, reliable choices and leave clear seams for the Implementer, "
        "Designer, and QA."
    ),
    temperature=0.4,
    permissions=_WEB_RESEARCH,
    skills=("runtime-architecture",),
)

_IMPLEMENTER = Role(
    key="implementer",
    title="Implementer",
    mission=(
        "the founding engineer who implements the core features to meet the "
        "success criteria"
    ),
    responsibilities=(IMPLEMENTATION,),
    system_prompt=(
        "You are the founding Implementer -- the fastest, most precise coder on "
        "the team. You turn the Architect's design into a working core loop that "
        "meets the success criteria. You run focused: use code-intelligence (LSP), "
        "grep, and the `core-loop-implementation` skill, and do NOT browse the web "
        "-- build with what the Architect chose. Prefer small, verifiable "
        "increments and keep the app runnable at all times."
    ),
    temperature=0.2,
    permissions=_NO_WEB,
    skills=("core-loop-implementation",),
)

_DESIGNER = Role(
    key="designer",
    title="UX & Virality Designer",
    mission="the product designer who makes the app delightful and genuinely shareable",
    responsibilities=(UX_VIRALITY,),
    system_prompt=(
        "You are the founding UX & Virality Designer. You make the app delightful "
        "and genuinely shareable -- the polish and the concrete viral hook that "
        "drives organic spread. You think in real user experience: when the "
        "browser tool is available, actually open the running app, look at it, and "
        "iterate on what you see; otherwise reason carefully from the markup and "
        "styles. Use web research for design inspiration and the `virality-playbook` "
        "and `ux-polish` skills for proven shareable mechanics. Be bold and "
        "creative, but never break what works."
    ),
    temperature=0.6,
    permissions=_WEB_RESEARCH,
    skills=("virality-playbook", "ux-polish"),
    wants_browser=True,
)

_QA_FINISHER = Role(
    key="qa_finisher",
    title="QA & Finisher",
    mission="the QA engineer who verifies, hardens, and ships the product",
    responsibilities=(QA,),
    system_prompt=(
        "You are the founding QA & Finisher -- the last line before ship. You "
        "verify every success criterion by ACTUALLY running and using the app, not "
        "by reading it: load the `live-app-testing` skill and drive the running app "
        "end to end -- open it in the browser and click/type through the core loop "
        "for a web app, or run the real command for a CLI/bot -- watching the DOM, "
        "console, and output for what a real user would hit. Then fix what is "
        "broken and guarantee it launches from a clean checkout with the exact "
        "manifest command. Also use the `release-checklist` skill. Run rigorous and "
        "skeptical (low temperature); do not browse the web. Only emit the ship "
        "signal once you have exercised the running app this turn and it genuinely "
        "works -- the signal is not honoured without that evidence."
    ),
    temperature=0.1,
    permissions=_NO_WEB,
    skills=("release-checklist", "live-app-testing"),
    wants_browser=True,
)


#: The two supported configurations, keyed by size. The team's last role owns QA.
_TEAMS: dict[int, tuple[Role, ...]] = {
    1: (_FOUNDER,),
    TEAM_SIZE: (_ARCHITECT, _IMPLEMENTER, _DESIGNER, _QA_FINISHER),
}


def roles_for(n_agents: int) -> list[Role]:
    """Return the ordered roles for a founder configuration of ``n_agents``.

    Args:
        n_agents: Either ``1`` (solo founder) or :data:`TEAM_SIZE` (the full
            four-specialist team). No other sizes are supported.

    Returns:
        The roles in canonical order. For the team, the final role always owns
        QA & Finish.

    Raises:
        ValueError: If ``n_agents`` is not one of :data:`VALID_TEAM_SIZES`.
    """
    if n_agents not in _TEAMS:
        raise ValueError(
            f"n_agents must be one of {VALID_TEAM_SIZES} "
            f"(solo founder or the {TEAM_SIZE}-agent team), got {n_agents!r}"
        )
    return list(_TEAMS[n_agents])
