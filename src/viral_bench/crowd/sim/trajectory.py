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

"""Capture what every crowd agent actually did and why.

The deterministic score reads aggregate numbers; an agentic autorater needs the
*reasoning* behind them -- whether praise was substantive or reflexive, whether a
criticism was fatal or cosmetic, whether the crowd genuinely convinced each
other. None of that survives in a rate or a mean.

Two problems this fixes:

* Only the hands-on triers had a stored trajectory. In a 50-agent run that is 12
  agents; the other 38 -- the ones whose reaction actually constitutes "did this
  spread" -- left nothing but rows in a database.
* Even the trier trace stored tool calls and observations, never the agent's own
  reasoning, which is the part a rater most needs.

OASIS keeps each agent's full message history on ``SocialAgent.memory``, so this
is a matter of persisting what already exists. We keep it asymmetric on purpose:
triers get their full step-by-step trial, reactors get their reasoning and public
content without the feed dumps that dominate their context. Feed dumps are the
bulk of the tokens and the least informative part -- they are the *stimulus*, not
the judgement, and they are already recoverable from the database.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

_LOG = logging.getLogger("viral_bench.crowd.trajectory")

#: Roles whose content is the agent's own reasoning/output rather than the
#: environment's prompt to it.
_AGENT_ROLES = {"assistant", "ai", "model"}

#: A feed dump looks like a system/user turn full of serialised posts. We keep a
#: short prefix for context and drop the rest.
_STIMULUS_CLIP = 400

#: Clip for an agent's own reasoning. This is CAPTURE, not rendering: whatever it
#: drops is gone from trajectories.json permanently, and that file is the agentic
#: rater's window into why the crowd reacted the way it did.
#:
#: Raised 2000 -> 32000 as headroom, not as a bug fix. Measured over the 58,887
#: stored reasoning entries the old 2000 never actually bound (longest: 1,402),
#: so nothing in the corpus was lost. But it sat *below* the 4000 that
#: score/evidence.py applies to the same text downstream, which is backwards for
#: a capture-time clip -- the irreversible one should never be the tighter of the
#: two. Now the only thing it guards against is a runaway turn dumping an entire
#: feed into the file, which is all a capture clip should ever do.
_REASONING_CLIP = 32000


@dataclass
class AgentTrajectory:
    """One agent's reasoning trail through the simulation."""

    agent_id: int
    username: str
    tier: str
    influence: int = 0
    reasoning: list[str] = field(default_factory=list)
    stimulus_sample: str = ""
    n_messages: int = 0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "username": self.username,
            "tier": self.tier,
            "influence": self.influence,
            "n_messages": self.n_messages,
            "reasoning": self.reasoning,
            "stimulus_sample": self.stimulus_sample,
        }


def _message_parts(record) -> tuple[str, str]:
    """Best-effort ``(role, content)`` from a CAMEL memory record."""
    for attr in ("memory_record", "record"):
        record = getattr(record, attr, record)
    msg = getattr(record, "message", record)
    role = (
        getattr(record, "role_at_backend", None)
        or getattr(msg, "role_name", None)
        or getattr(msg, "role", None)
        or ""
    )
    role = getattr(role, "value", role)
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        role = msg.get("role", role)
        content = msg.get("content")
    return str(role).lower(), str(content or "")


def extract_trajectory(agent, *, agent_id, username, tier, influence=0, full=False):
    """Pull one agent's reasoning out of its OASIS/CAMEL memory.

    ``full`` keeps every agent message (used for triers, whose step-by-step trial
    is the richest first-hand evidence); otherwise only the agent's own outputs
    are kept, which is where the judgement lives.
    """
    traj = AgentTrajectory(
        agent_id=agent_id, username=username, tier=tier, influence=influence
    )
    try:
        records = agent.memory.retrieve()
    except Exception as exc:  # noqa: BLE001 - never let capture break a run
        _LOG.debug("no memory for agent %s: %s", agent_id, exc)
        return traj

    traj.n_messages = len(records)
    for record in records:
        role, content = _message_parts(record)
        if not content:
            continue
        if role in _AGENT_ROLES:
            traj.reasoning.append(content[:_REASONING_CLIP])
        elif full and not traj.stimulus_sample:
            traj.stimulus_sample = content[:_STIMULUS_CLIP]
    if not traj.stimulus_sample:
        for record in records:
            role, content = _message_parts(record)
            if content and role not in _AGENT_ROLES:
                traj.stimulus_sample = content[:_STIMULUS_CLIP]
                break
    return traj


def collect_trajectories(env, crowd) -> list[dict]:
    """Capture every crowd agent's trajectory (triers full, reactors compact)."""
    out: list[dict] = []
    for agent_id in crowd.crowd_ids:
        persona = crowd.persona_by_id.get(agent_id)
        try:
            agent = env.agent_graph.get_agent(agent_id)
        except Exception as exc:  # noqa: BLE001 - capture is best-effort
            _LOG.debug("no agent %s in graph: %s", agent_id, exc)
            continue
        tier = crowd.tier_by_id.get(agent_id, "?")
        traj = extract_trajectory(
            agent,
            agent_id=agent_id,
            username=persona.username if persona else str(agent_id),
            tier=tier,
            influence=persona.influence if persona else 0,
            full=(tier == "trier"),
        )
        out.append(traj.to_dict())
    return out


def write_trajectories(out_dir: str | Path, trajectories: list[dict]) -> str:
    """Persist all-agent trajectories next to the run's other artifacts."""
    path = Path(out_dir) / "trajectories.json"
    path.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
    return str(path)
