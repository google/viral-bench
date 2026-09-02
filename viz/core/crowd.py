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

"""Read one crowd simulation run into a social graph, a timeline, and app trials.

A crowd run is an OASIS social simulation pointed at a built app. Its artifacts
split cleanly in two, and so does this module:

*The social side* -- ``simulation.db`` (users, posts, follows, likes, comments, and
the recommender's materialised feed) plus ``actions.jsonl`` (every action, in order).
Time here is a **discrete integer clock**, not a wall clock: ``0`` is sign-ups and
the seeded follow graph, ``1`` is the founder's launch post, round *N* happens at
``N + 1``, and the last step is the exit interview. That mapping is what makes
playback possible.

*The app-trial side* -- ``traces/agent_<N>.json``, one per agent that got hands on
the app: every click, every keystroke, the page state that came back, and the
screenshots. This is the record of an agent using the thing.

Two details cost real time to discover and are worth stating plainly. ``info`` in
``actions.jsonl`` is **double-encoded JSON** -- a JSON string inside a JSON object.
And absolute paths recorded inside these files (``out_dir``, ``db_path``,
``trace_paths``) come from whichever machine produced the run, frequently a
``/tmp`` sweep directory that no longer exists, so everything must be resolved
relative to the directory being read.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .paths import read_json

#: Post ids referenced by a feed are cheap. The rendered feed bodies are not. A
#: 30-agent run records ~90 ``refresh`` actions, each carrying up to 20 fully
#: rendered posts with their comment threads. Keeping ids in the main payload and
#: fetching bodies on demand is the difference between a 300 KB and a 12 MB response.
FEED_PREVIEW = 20

#: Trial step summaries embed a full accessibility tree. Fine to read one at a
#: time, ruinous to ship 30 agents x 40 steps of them at once.
STEP_SUMMARY_CHARS = 20000


def _connect(db_path: Path) -> sqlite3.Connection | None:
    """Open the simulation database strictly read-only."""
    if not db_path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str) -> list[dict]:
    try:
        return [dict(r) for r in conn.execute(sql)]
    except sqlite3.Error:
        return []


def _decode_info(raw) -> dict:
    """Unwrap ``actions.jsonl``'s double-encoded ``info`` payload."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        value = json.loads(raw)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _int(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# The social side
# --------------------------------------------------------------------------


def _load_actions(run_dir: Path) -> tuple[list[dict], dict[int, dict[int, list[int]]]]:
    """Return ``(actions, feeds)`` from ``actions.jsonl``.

    ``feeds[user_id][timestep]`` is the list of post ids that agent was shown at
    that step -- the only *time-resolved* exposure record in the run. The ``rec``
    table holds a final snapshot with no timestep, so it cannot answer "what did
    this agent see in round 2", which is exactly the question a playback UI asks.
    """
    path = run_dir / "actions.jsonl"
    actions: list[dict] = []
    feeds: dict[int, dict[int, list[int]]] = {}
    if not path.is_file():
        return actions, feeds

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            user_id = _int(row.get("user_id"), -1)
            step = _int(row.get("created_at"))
            name = row.get("action") or ""
            info = _decode_info(row.get("info"))

            if name == "refresh":
                posts = info.get("posts") or []
                ids = [_int(p.get("post_id")) for p in posts if isinstance(p, dict)]
                feeds.setdefault(user_id, {})[step] = ids[:FEED_PREVIEW]
                actions.append(
                    {"t": step, "user_id": user_id, "action": name, "n_posts": len(ids)}
                )
                continue

            entry = {"t": step, "user_id": user_id, "action": name}
            if name == "sign_up":
                entry["username"] = info.get("user_name")
            elif name == "follow":
                entry["follow_id"] = info.get("follow_id")
            elif name == "create_post":
                entry["post_id"] = _int(info.get("post_id"))
                entry["text"] = (info.get("content") or "")[:400]
            elif name in ("repost", "quote_post"):
                entry["source_post"] = _int(
                    info.get("reposted_id") or info.get("quoted_id")
                )
                entry["post_id"] = _int(info.get("new_post_id"))
            elif name in ("like_post", "dislike_post", "report_post", "unlike_post"):
                entry["post_id"] = _int(info.get("post_id"))
            elif name == "create_comment":
                entry["comment_id"] = _int(info.get("comment_id"))
                entry["text"] = (info.get("content") or "")[:400]
            elif name == "like_comment":
                entry["comment_id"] = _int(info.get("comment_id"))
            elif name == "search_posts":
                entry["query"] = info.get("query")
            elif name == "interview":
                entry["response"] = (info.get("response") or "")[:2000]
            actions.append(entry)

    actions.sort(key=lambda a: (a["t"], a["user_id"]))
    return actions, feeds


def _load_social(run_dir: Path) -> dict:
    """Users, posts, comments, follows and the final recommender feed."""
    conn = _connect(run_dir / "simulation.db")
    if conn is None:
        return {"users": [], "posts": [], "comments": [], "follows": [], "rec": {}}
    try:
        users = _rows(conn, "SELECT * FROM user ORDER BY user_id")
        posts = _rows(conn, "SELECT * FROM post ORDER BY post_id")
        comments = _rows(conn, "SELECT * FROM comment ORDER BY comment_id")
        follows = _rows(conn, "SELECT * FROM follow ORDER BY follow_id")
        likes = _rows(conn, "SELECT * FROM like ORDER BY like_id")
        rec_rows = _rows(conn, "SELECT user_id, post_id FROM rec")
    finally:
        conn.close()

    rec: dict[int, list[int]] = {}
    for row in rec_rows:
        rec.setdefault(_int(row["user_id"]), []).append(_int(row["post_id"]))

    by_post: dict[int, list[dict]] = {}
    for comment in comments:
        by_post.setdefault(_int(comment.get("post_id")), []).append(
            {
                "comment_id": _int(comment.get("comment_id")),
                "user_id": _int(comment.get("user_id")),
                "content": comment.get("content") or "",
                "t": _int(comment.get("created_at")),
                "likes": _int(comment.get("num_likes")),
                "dislikes": _int(comment.get("num_dislikes")),
            }
        )

    post_rows = [
        {
            "post_id": _int(p.get("post_id")),
            "user_id": _int(p.get("user_id")),
            "original_post_id": p.get("original_post_id"),
            "content": p.get("content") or "",
            "quote_content": p.get("quote_content"),
            "t": _int(p.get("created_at")),
            "likes": _int(p.get("num_likes")),
            "dislikes": _int(p.get("num_dislikes")),
            "shares": _int(p.get("num_shares")),
            "reports": _int(p.get("num_reports")),
            "comments": by_post.get(_int(p.get("post_id")), []),
            # A repost carries no text of its own, while a quote does. Distinguishing
            # them matters -- one is amplification, the other is commentary.
            "kind": (
                "repost"
                if p.get("original_post_id") and not p.get("quote_content")
                else "quote"
                if p.get("original_post_id")
                else "post"
            ),
        }
        for p in posts
    ]

    return {
        "users": users,
        "posts": post_rows,
        "comments": comments,
        "follows": [
            {
                "source": _int(f.get("follower_id")),
                "target": _int(f.get("followee_id")),
                "t": _int(f.get("created_at")),
            }
            for f in follows
        ],
        "likes": [
            {
                "user_id": _int(x.get("user_id")),
                "post_id": _int(x.get("post_id")),
                "t": _int(x.get("created_at")),
            }
            for x in likes
        ],
        "rec": rec,
    }


def _agent_roster(summary: dict, social: dict, run_dir: Path) -> list[dict]:
    """Fuse everything known about each agent into one row.

    Five sources describe the same agent under the same id: the cast list in
    ``run_summary.crowd[]``, the sign-up row in the ``user`` table, the trial
    verdict, the interview answer, and the reasoning in ``trajectories.json``.
    Agent 0 is always the founder and is never in the cast list.
    """
    verdicts = summary.get("verdicts") or {}
    trial_by_id = {
        _int(v.get("agent_id")): v
        for v in (verdicts.get("triers") or {}).get("per_agent", [])
    }
    interview_by_id = {
        _int(v.get("agent_id")): v
        for v in (verdicts.get("interviews") or {}).get("per_agent", [])
    }
    cast_by_id = {_int(c.get("agent_id")): c for c in (summary.get("crowd") or [])}
    user_by_id = {_int(u.get("user_id")): u for u in social["users"]}

    trajectories = read_json(run_dir / "trajectories.json", []) or []
    traj_by_id = {
        _int(t.get("agent_id")): t for t in trajectories if isinstance(t, dict)
    }

    trace_dir = run_dir / "traces"
    trace_ids = set()
    if trace_dir.is_dir():
        for path in trace_dir.glob("agent_*.json"):
            try:
                trace_ids.add(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue

    posted = {}
    for post in social["posts"]:
        posted[post["user_id"]] = posted.get(post["user_id"], 0) + 1

    ids = sorted(
        set(user_by_id) | set(cast_by_id) | set(trial_by_id) | set(interview_by_id)
    )
    roster = []
    for agent_id in ids:
        user = user_by_id.get(agent_id, {})
        cast = cast_by_id.get(agent_id, {})
        trial = trial_by_id.get(agent_id) or {}
        interview = interview_by_id.get(agent_id) or {}
        traj = traj_by_id.get(agent_id) or {}
        roster.append(
            {
                "id": agent_id,
                "username": user.get("user_name")
                or cast.get("username")
                or f"agent_{agent_id}",
                "name": user.get("name") or "",
                "bio": user.get("bio") or "",
                "archetype": cast.get("archetype") or "",
                "tier": "founder" if agent_id == 0 else (cast.get("tier") or ""),
                "influence": cast.get("influence"),
                "followers": _int(user.get("num_followers")),
                "followings": _int(user.get("num_followings")),
                "posts": posted.get(agent_id, 0),
                "has_trace": agent_id in trace_ids,
                "trial": {
                    "would_use": trial.get("would_use"),
                    "would_share": trial.get("would_share"),
                    "delight": trial.get("delight"),
                    "functionality": trial.get("functionality"),
                    "usability": trial.get("usability"),
                    "design": trial.get("design"),
                    "simplicity": trial.get("simplicity"),
                    "craft": trial.get("craft"),
                    "finished": trial.get("finished"),
                    "degraded": trial.get("degraded"),
                    "app_reachable": trial.get("app_reachable"),
                    "had_effect": trial.get("had_effect"),
                    "n_steps": trial.get("n_steps"),
                    "work_survived": trial.get("work_survived"),
                    "saw_other_users": trial.get("saw_other_users"),
                }
                if trial
                else None,
                "interview": {
                    "would_use": interview.get("would_use"),
                    "would_share": interview.get("would_share"),
                    "delight": interview.get("delight"),
                    "for_me": interview.get("for_me"),
                    "why": interview.get("why") or "",
                }
                if interview
                else None,
                # The crowd's reasoning IS recorded, unlike the founder's. This is
                # the agent narrating what it did and why.
                "reasoning": [str(r) for r in (traj.get("reasoning") or [])][:40],
                "n_messages": traj.get("n_messages"),
                "persona_prompt": (traj.get("stimulus_sample") or "")[:4000],
            }
        )
    return roster


def _founder_build(run_dir: Path, build_id: str) -> dict:
    """What the founder side of this run was: which pipeline, which model.

    A crowd run only records the build id it scored. Everything about *how* that
    app came to exist -- the arm, the model, whether it shipped -- lives in the
    founder build record, and without it a crowd run is a score with no subject.
    It also removes a real misreading: the model named in a run's own config is
    the CROWD's model, not the one that built the app, and the two are different
    by design.

    Resolved by walking up from the run directory rather than taking a builds root
    as an argument, since every crowd-shaped set (crowd, ablation, calibration,
    smoke) sits directly under it. Returns ``on_disk: False`` rather than raising
    when the build has been pruned -- crowd runs outlive their builds.
    """
    empty = {"on_disk": False, "build_id": build_id}
    if not build_id:
        return empty
    record = read_json(run_dir.parent.parent / "work" / build_id / "build.json")
    if not isinstance(record, dict):
        return empty

    structure = record.get("structure") or ""
    collab = record.get("collab") or ""
    arm = {
        ("solo", "local"): "solo",
        ("team", "local"): "team",
        ("dynamic", "local"): "dynamic",
    }.get((structure, collab), structure or "unknown")
    label = {
        "solo": "single agent",
        "team": "4-agent team (local)",
        "dynamic": "dynamic orchestrator",
    }.get(arm, arm)

    trajectory = record.get("trajectory") or {}
    # `_all` counts the subagents, and the bare key is what builds recorded before the
    # counts were split by source.
    reasoning = (
        trajectory.get("reasoning_chars_all") or trajectory.get("reasoning_chars") or 0
    )
    model = str(record.get("model") or "")
    return {
        "on_disk": True,
        "build_id": build_id,
        "idea_id": record.get("idea_id") or "",
        "arm": arm,
        "arm_label": label,
        "structure": structure,
        "collab": collab,
        "n_agents": record.get("n_agents"),
        "model": model,
        "model_short": model.rsplit("/", 1)[-1].split("@", 1)[0],
        "status": record.get("status") or "",
        "turns_spent": record.get("turns_spent"),
        "rounds_run": record.get("rounds_run"),
        "shipped_early": record.get("shipped_early"),
        "qa_verified": record.get("qa_verified"),
        "subagents_spawned": record.get("subagents_spawned") or 0,
        "reasoning_chars": int(reasoning or 0),
        "app_title": (record.get("manifest") or {}).get("title") or "",
        "created_at": record.get("created_at") or "",
    }


def load_run(run_dir: Path) -> dict | None:
    """Assemble a whole crowd run: roster, graph, posts, timeline, ratings."""
    summary = read_json(run_dir / "run_summary.json")
    if not isinstance(summary, dict):
        return None

    social = _load_social(run_dir)
    actions, feeds = _load_actions(run_dir)
    roster = _agent_roster(summary, social, run_dir)

    steps = sorted({a["t"] for a in actions} | {p["t"] for p in social["posts"]})
    rounds_run = _int(summary.get("rounds_run"))

    # 0 = sign-ups + seeded follows, 1 = launch post, round N = N+1, last = interview.
    def label_for(step: int) -> str:
        if step == 0:
            return "Sign-ups & follow graph"
        if step == 1:
            return "Launch post"
        if 2 <= step <= rounds_run + 1:
            return f"Round {step - 1}"
        return "Interview"

    timeline = []
    for step in steps:
        at_step = [a for a in actions if a["t"] == step]
        counts: dict[str, int] = {}
        for action in at_step:
            counts[action["action"]] = counts.get(action["action"], 0) + 1
        timeline.append(
            {
                "t": step,
                "label": label_for(step),
                "n_actions": len(at_step),
                "actors": len({a["user_id"] for a in at_step}),
                "counts": counts,
            }
        )

    build_id = summary.get("build_id") or ""
    return {
        "run_id": run_dir.name,
        "build_id": build_id,
        # Which pipeline and which model produced the app being scored here.
        "founder": _founder_build(run_dir, build_id),
        "ok": bool(summary.get("ok")),
        "app_type": summary.get("app_type") or "",
        "arch_version": str(summary.get("crowd_arch_version") or ""),
        "undeliverable": bool(summary.get("undeliverable")),
        "config": summary.get("config") or {},
        "engagement": summary.get("engagement") or {},
        "verdicts": summary.get("verdicts") or {},
        "health": summary.get("health") or {},
        "validity": summary.get("validity") or {},
        "integrity": summary.get("crowd_integrity") or {},
        "turn_stats": summary.get("turn_stats") or {},
        "interview_stats": summary.get("interview_stats") or {},
        "rounds": summary.get("rounds") or [],
        "duration_s": summary.get("duration_s"),
        "launch_post_id": _int(summary.get("launch_post_id"), 1),
        "agents": roster,
        "posts": social["posts"],
        "follows": social["follows"],
        "likes": social["likes"],
        "rec": {str(k): v for k, v in social["rec"].items()},
        "feeds": {
            str(k): {str(t): ids for t, ids in v.items()} for k, v in feeds.items()
        },
        "actions": actions,
        "timeline": timeline,
        "autorating": read_json(run_dir / "autorating.json", {}) or {},
        "paths": {
            "run_dir": str(run_dir),
            "summary": str(run_dir / "run_summary.json"),
            "actions": str(run_dir / "actions.jsonl"),
            "db": str(run_dir / "simulation.db"),
            "traces": str(run_dir / "traces"),
        },
    }


# --------------------------------------------------------------------------
# The app-trial side
# --------------------------------------------------------------------------


def load_trial(run_dir: Path, agent_id: int) -> dict | None:
    """One agent's hands-on trial of the app: every step, in order, with screenshots.

    Unlike the social clock, these steps carry real wall-clock timestamps
    (``ts``, ``duration_s``), so a replay can run at the pace the agent
    worked at.
    """
    path = run_dir / "traces" / f"agent_{agent_id}.json"
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    trace = payload.get("trace") or {}

    steps = []
    for step in trace.get("steps") or []:
        if not isinstance(step, dict):
            continue
        summary = str(step.get("summary") or "")
        shot = step.get("screenshot")
        steps.append(
            {
                "index": step.get("index"),
                "action": step.get("action") or "",
                "args": step.get("args") or {},
                "summary": summary[:STEP_SUMMARY_CHARS],
                "summary_chars": len(summary),
                "ok": step.get("ok"),
                "errors": step.get("errors") or [],
                "ts": step.get("ts"),
                "duration_s": step.get("duration_s"),
                # Screenshots live in a temp dir by absolute path. Hand the UI a
                # basename and let the server resolve it, so a run copied between
                # machines still renders whatever images survived.
                "screenshot": Path(shot).name if shot else None,
                "screenshot_path": shot or None,
                "screenshot_exists": bool(shot) and Path(shot).is_file(),
            }
        )

    return {
        "agent_id": payload.get("agent_id", agent_id),
        "persona": payload.get("persona") or "",
        "build_id": trace.get("build_id") or "",
        "app_type": trace.get("app_type") or "",
        "started_at": trace.get("started_at"),
        "ended_at": trace.get("ended_at"),
        "degraded": trace.get("degraded"),
        "app_reachable": trace.get("app_reachable"),
        "target_url": trace.get("target_url"),
        "verdict": trace.get("verdict"),
        "steps": steps,
        "path": str(path),
    }


def load_feed(run_dir: Path, agent_id: int, step: int) -> list[dict]:
    """The rendered feed an agent saw at one timestep, bodies included.

    Read straight from ``actions.jsonl`` rather than held in memory: these payloads
    are the single largest thing in a run and are only ever wanted one at a time.
    """
    path = run_dir / "actions.jsonl"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("action") != "refresh":
                continue
            if (
                _int(row.get("user_id"), -1) != agent_id
                or _int(row.get("created_at")) != step
            ):
                continue
            posts = _decode_info(row.get("info")).get("posts") or []
            return [p for p in posts if isinstance(p, dict)]
    return []
