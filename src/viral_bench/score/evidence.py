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

"""Assemble one autorater-ready evidence pack from a crowd run.

The evidence for how an app landed is scattered across four places: aggregate
numbers in ``run_summary.json``, the social thread in ``simulation.db``, the
hands-on trials in ``traces/``, and the agents' reasoning in
``trajectories.json``. A rater -- human or model -- should not have to join
those by hand, and an LLM cannot read them all at once anyway.

This produces a single ordered, budgeted bundle: the hard metrics first, then
the first-hand trials, then the discussion, then the crowd's verdicts with their
reasons. Every item carries an id (``agent_id`` / ``post_id``) so a rating can
cite the evidence it relied on and a human can check it.

Two deliberate properties:

* **Blinded.** The founder model that built the app never appears. A rater that
  knows which model it is judging can rate the model's reputation instead of the
  artifact, which would be fatal for a benchmark whose purpose is to compare
  models.
* **Budgeted and deterministic.** Content is truncated and ordered by a fixed
  rule (influence, then id), never by sampling, so the same run always produces
  the same pack and the same pack always fits a context window.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

#: Per-field clips. Generous enough to preserve an argument, tight enough that a
#: 50-agent run still fits comfortably in one prompt.
_POST_CLIP = 2000
_COMMENT_CLIP = 1500
_REASONING_CLIP = 4000
_NOTES_CLIP = 2000

#: How many of each kind of item to include, taken in a deterministic order.
#: Raised substantially: this pack is the agentic rater's entire view of the
#: run, and the old caps threw away most of it. _MAX_COMMENTS=40 in particular
#: took the FIRST 40 by id, which deterministically discarded the late-round
#: discussion -- the only place word-of-mouth could possibly show up.
_MAX_POSTS = 200
_MAX_COMMENTS = 400
_MAX_TRIALS = 60
_MAX_VERDICTS = 200

#: How many of a trial's actions to render. What an agent DID is the strongest
#: evidence in the pack that a trial was real rather than imagined, so this
#: should not be the thing that silently drops.
#:
#: It was an inline ``[:12]``, which bound: measured over the 4,970 stored
#: trials, 109 of them (2.2%) took more than 12 actions, topping out at 41 -- and
#: a long action list is the signature of a thorough trial, i.e. precisely the
#: evidence worth keeping. 200 clears every trial in the corpus.
_MAX_TRIAL_ACTIONS = 200


@dataclass
class EvidencePack:
    """Everything a rater needs about one crowd run, in one object."""

    build_id: str
    app_type: str
    crowd_dir: str
    metrics: dict = field(default_factory=dict)
    trials: list[dict] = field(default_factory=list)
    discussion: list[dict] = field(default_factory=list)
    verdicts: list[dict] = field(default_factory=list)
    integrity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "build_id": self.build_id,
            "app_type": self.app_type,
            "crowd_dir": self.crowd_dir,
            "metrics": self.metrics,
            "trials": self.trials,
            "discussion": self.discussion,
            "verdicts": self.verdicts,
            "integrity": self.integrity,
        }

    def to_prompt(self) -> str:
        """Render the pack as the text an LLM rater is shown."""
        out: list[str] = []
        out.append(f"APP TYPE: {self.app_type}")
        out.append("")
        out.append("== HARD METRICS (from the simulation database) ==")
        for key, value in self.metrics.items():
            out.append(f"  {key}: {value}")

        out.append("")
        out.append(
            f"== FIRST-HAND TRIALS ({len(self.trials)}) -- agents who actually "
            "used the app =="
        )
        for t in self.trials:
            out.append(
                f"  [agent {t['agent_id']} @{t['username']}] verdict:"
                f" would_use={t.get('would_use')} would_share={t.get('would_share')}"
                f" delight={t.get('delight')} craft={t.get('craft')}"
            )
            if t.get("facets"):
                facets = " ".join(f"{k}={v}" for k, v in sorted(t["facets"].items()))
                out.append(f"      facets: {facets}")
            # What the agent verified rather than felt. A rater weighing whether
            # praise was earned needs to know the app kept the work and showed
            # other people's, which is the difference between a real product and
            # a convincing demo.
            checks = [
                f"{name}={t[name]}"
                for name in ("work_survived", "saw_other_users")
                if t.get(name) is not None
            ]
            if checks:
                out.append(f"      verified: {' '.join(checks)}")
            if t.get("actions"):
                shown = t["actions"][:_MAX_TRIAL_ACTIONS]
                out.append(f"      did: {', '.join(shown)}")
            if t.get("notes"):
                out.append(f"      said: {t['notes']}")
            for r in t.get("reasoning", []):
                out.append(f"      reasoned: {r}")

        out.append("")
        out.append(f"== PUBLIC DISCUSSION ({len(self.discussion)} items) ==")
        for d in self.discussion:
            kind = d["kind"].upper()
            out.append(f"  [{kind} {d['id']} by @{d['username']}] {d['content']}")

        out.append("")
        out.append(f"== CROWD VERDICTS ({len(self.verdicts)}) ==")
        for v in self.verdicts:
            out.append(
                f"  [agent {v['agent_id']} @{v['username']} ({v.get('tier')})]"
                f" would_use={v.get('would_use')} would_share={v.get('would_share')}"
                f" score={v.get('delight')} for_me={v.get('for_me')}"
            )
            if v.get("why"):
                out.append(f"      why: {v['why']}")
        return "\n".join(out)


def _clip(text: str | None, limit: int) -> str:
    return " ".join((text or "").split())[:limit]


def _load(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_evidence_pack(crowd_dir: str | Path) -> EvidencePack:
    """Assemble the evidence pack for one crowd run directory."""
    crowd_dir = Path(crowd_dir)
    summary = _load(crowd_dir / "run_summary.json") or {}
    trajectories = _load(crowd_dir / "trajectories.json") or []
    reasoning_by_agent = {
        t["agent_id"]: [_clip(r, _REASONING_CLIP) for r in t.get("reasoning", [])]
        for t in trajectories
        if isinstance(t, dict) and "agent_id" in t
    }
    names = {c["agent_id"]: c["username"] for c in summary.get("crowd", [])}
    influence = {c["agent_id"]: c.get("influence", 0) for c in summary.get("crowd", [])}

    verdicts = summary.get("verdicts") or {}
    interviews = verdicts.get("interviews") or {}
    triers = verdicts.get("triers") or {}
    engagement = summary.get("engagement") or {}
    reach = engagement.get("reach") or {}

    exposed = reach.get("exposed_agents") or 0

    # Scale context ONLY -- deliberately no score components.
    #
    # This block is rendered first in the pack, so whatever is here anchors the
    # rater before it reads a single word of actual evidence. It used to lead
    # with adoption_rate, craft_mean and app_builds_and_runs: precisely the
    # deterministic components the rater's dimensions are supposed to be
    # INDEPENDENT of. An independent judge primed with the answer is not an
    # independent judge, and its 0.15 weight then buys a paraphrase of the
    # formula rather than the judgement the formula cannot make.
    #
    # What stays is raw scale -- how big the crowd was and how much was said --
    # which the rater needs to interpret volume but cannot read a verdict off.
    metrics = {
        "crowd_size": len(summary.get("crowd", [])),
        "agents_exposed": exposed,
        "posts": engagement.get("posts"),
        "comments": engagement.get("comments"),
        "reposts": engagement.get("reposts"),
        "likes": engagement.get("likes"),
        "negative_reactions": (engagement.get("dislikes") or 0)
        + (engagement.get("reports") or 0),
    }

    # -- first-hand trials, richest evidence first ---------------------------
    trials: list[dict] = []
    for row in (triers.get("per_agent") or [])[:_MAX_TRIALS]:
        aid = row.get("agent_id")
        trial = {
            "agent_id": aid,
            "username": row.get("username") or names.get(aid, str(aid)),
            "would_use": row.get("would_use"),
            "would_share": row.get("would_share"),
            "delight": row.get("delight"),
            "craft": row.get("craft"),
            "degraded": row.get("degraded"),
            "work_survived": row.get("work_survived"),
            "saw_other_users": row.get("saw_other_users"),
            "facets": {
                k: row[k]
                for k in ("functionality", "usability", "design", "simplicity")
                if row.get(k) is not None
            },
            # Was [:2]. 20 of 25 agents had 3 or more reasoning entries, so
            # the slice silently dropped most of what the rater is meant to
            # read.
            "reasoning": reasoning_by_agent.get(aid, []),
        }
        trials.append(trial)

    # Step lists + notes come from the per-trier trace files.
    traces_dir = crowd_dir / "traces"
    if traces_dir.is_dir():
        by_id = {t["agent_id"]: t for t in trials}
        for tf in sorted(traces_dir.glob("agent_*.json")):
            data = _load(tf) or {}
            aid = data.get("agent_id")
            if aid not in by_id:
                continue
            trace = data.get("trace") or {}
            by_id[aid]["actions"] = [
                s.get("action") for s in trace.get("steps", []) if s.get("action")
            ]
            verdict = trace.get("verdict") or {}
            by_id[aid]["notes"] = _clip(verdict.get("notes"), _NOTES_CLIP)

    # -- public discussion ---------------------------------------------------
    discussion: list[dict] = []
    db = crowd_dir / "simulation.db"
    if db.is_file():
        try:
            con = sqlite3.connect(str(db))
            for pid, uid, content, orig in con.execute(
                "SELECT post_id, user_id, content, original_post_id FROM post "
                "ORDER BY post_id LIMIT ?",
                (_MAX_POSTS,),
            ):
                text = _clip(content, _POST_CLIP)
                if text:
                    discussion.append(
                        {
                            "kind": "repost" if orig else "post",
                            "id": pid,
                            "agent_id": uid,
                            "username": names.get(uid, "founder" if uid == 0 else uid),
                            "content": text,
                        }
                    )
            for cid, uid, content in con.execute(
                # Newest last, but selected from the END: taking the first N
                # by id drops the late-round conversation, which is exactly
                # where persuasion between agents would appear.
                "SELECT comment_id, user_id, content FROM ("
                "  SELECT comment_id, user_id, content FROM comment "
                "  ORDER BY comment_id DESC LIMIT ?"
                ") ORDER BY comment_id",
                (_MAX_COMMENTS,),
            ):
                text = _clip(content, _COMMENT_CLIP)
                if text:
                    discussion.append(
                        {
                            "kind": "comment",
                            "id": cid,
                            "agent_id": uid,
                            "username": names.get(uid, uid),
                            "content": text,
                        }
                    )
            con.close()
        except sqlite3.Error:
            pass

    # -- crowd verdicts, most influential first (deterministic) --------------
    rows = list(interviews.get("per_agent") or [])
    rows.sort(
        key=lambda r: (-influence.get(r.get("agent_id"), 0), r.get("agent_id", 0))
    )
    crowd_verdicts = [
        {
            "agent_id": r.get("agent_id"),
            "username": r.get("username"),
            "tier": r.get("tier"),
            "would_use": r.get("would_use"),
            "would_share": r.get("would_share"),
            "delight": r.get("delight"),
            "for_me": r.get("for_me"),
            "why": _clip(r.get("why"), _NOTES_CLIP),
            # Reactor reasoning was captured and then thrown away: the pack only
            # read reasoning for triers, who are a quarter of the crowd at n=50,
            # so ~71% of the reasoning collected never reached the rater that
            # exists to read it.
            "reasoning": reasoning_by_agent.get(r.get("agent_id"), []),
        }
        for r in rows[:_MAX_VERDICTS]
    ]

    return EvidencePack(
        build_id=summary.get("build_id", ""),
        app_type=summary.get("app_type", ""),
        crowd_dir=str(crowd_dir),
        metrics=metrics,
        trials=trials,
        discussion=discussion,
        verdicts=crowd_verdicts,
        integrity={
            "run_ok": summary.get("ok"),
            "rounds_ok": all(r.get("ok", True) for r in summary.get("rounds", [])),
            "n_interviews": interviews.get("n", 0),
            "n_trials": triers.get("n", 0),
            "clamped": (summary.get("crowd_integrity") or {}).get("clamped"),
        },
    )


def write_evidence_pack(crowd_dir: str | Path) -> str:
    """Build and persist ``evidence.json`` next to the run."""
    pack = build_evidence_pack(crowd_dir)
    path = Path(crowd_dir) / "evidence.json"
    path.write_text(json.dumps(pack.to_dict(), indent=2), encoding="utf-8")
    return str(path)
