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

"""Synthetic artifacts shaped like the real ones, shared by both test modules.

The shapes here are copied from artifacts on disk: the NDJSON event envelope, the
double-encoded ``info`` string in ``actions.jsonl``, and the OASIS table layout.
Building them by hand rather than pointing at a real builds tree keeps the tests
true regardless of which machine they run on.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: A Playwright MCP result, in the three-section markdown shape the real ones use.
BROWSER_RESULT = """### Ran Playwright code
```js
await page.click();
```

### Page state
- Page URL: http://localhost:8000/
- Page Title: Demo
- Page Snapshot:
```yaml
- button [ref=e5]
```
"""


def event(kind: str, timestamp: int, part: dict, session: str = "ses_a") -> str:
    return json.dumps(
        {"type": kind, "timestamp": timestamp, "sessionID": session, "part": part}
    )


def tool_event(
    timestamp: int, tool: str, *, status="completed", session="ses_a", **extra
) -> str:
    state = {
        "status": status,
        "input": extra.pop("input", {}),
        "time": {"start": timestamp - 5, "end": timestamp},
    }
    state.update(extra)
    return event(
        "tool_use",
        timestamp,
        {
            "id": f"prt_{tool}_{timestamp}",
            "type": "tool",
            "tool": tool,
            "callID": f"c{timestamp}",
            "messageID": "msg_1",
            "state": state,
        },
        session,
    )


def step_event(timestamp: int, *, cost=0.01, reasoning=0, session="ses_a") -> str:
    return event(
        "step_finish",
        timestamp,
        {
            "id": f"prt_step_{timestamp}",
            "type": "step-finish",
            "messageID": "msg_1",
            "reason": "tool-calls",
            "cost": cost,
            "tokens": {
                "total": 100,
                "input": 10,
                "output": 20,
                "reasoning": reasoning,
                "cache": {"read": 5, "write": 1},
            },
        },
        session,
    )


def text_event(timestamp: int, text: str, session="ses_a") -> str:
    return event(
        "text", timestamp, {"type": "text", "messageID": "msg_1", "text": text}, session
    )


def write_build(
    root: Path, build_id: str, record: dict, transcripts: dict[str, str]
) -> Path:
    build_dir = root / "work" / build_id
    (build_dir / "transcript").mkdir(parents=True, exist_ok=True)
    (build_dir / "app").mkdir(parents=True, exist_ok=True)
    record.setdefault("build_id", build_id)
    (build_dir / "build.json").write_text(json.dumps(record), encoding="utf-8")
    for name, body in transcripts.items():
        (build_dir / "transcript" / f"{name}.json").write_text(body, encoding="utf-8")
    return build_dir


@pytest.fixture
def builds(tmp_path: Path) -> Path:
    """A builds tree holding one build of every structure the corpus contains."""
    root = tmp_path / "builds"

    write_build(
        root,
        "idea__solo",
        {
            "idea_id": "idea",
            "model": "google-vertex/gemini-test",
            "structure": "solo",
            "n_agents": 1,
            "collab": "local",
            "status": "ok",
            "created_at": "2026-08-16T04:07:24+00:00",
            "roles": ["founder"],
            "rounds_run": 1,
            "turns_spent": 2,
            "qa_verified": None,
            "phases": [
                {
                    "phase": "design",
                    "role": "",
                    "agent_index": 0,
                    "turn": "design",
                    "session_id": "ses_a",
                    "returncode": 0,
                    "ok": True,
                    "duration_s": 10.0,
                },
                {
                    "phase": "build",
                    "role": "",
                    "agent_index": 0,
                    "turn": "build",
                    "session_id": "ses_a",
                    "returncode": 0,
                    "ok": True,
                    "duration_s": 20.0,
                },
            ],
        },
        {
            "design": "\n".join([text_event(1000, "plan"), step_event(1010)]),
            "build": "\n".join(
                [
                    tool_event(
                        2000,
                        "write",
                        input={"filePath": "/x/app/main.py", "content": "hi"},
                        metadata={
                            "filediff": {
                                "file": "main.py",
                                "patch": "@@\n+hi",
                                "additions": 1,
                                "deletions": 0,
                            }
                        },
                    ),
                    step_event(2010, cost=0.5),
                ]
            ),
        },
    )

    write_build(
        root,
        "idea__team",
        {
            "idea_id": "idea",
            "model": "google-vertex/gemini-test",
            "structure": "team",
            "n_agents": 4,
            "collab": "local",
            "status": "ok",
            "created_at": "2026-08-01T11:49:00+00:00",
            "roles": ["architect", "implementer", "designer", "qa_finisher"],
            "rounds_run": 2,
            "turns_spent": 8,
            "shipped_early": True,
            "qa_verified": True,
            "phases": [
                {
                    "phase": f"r{rnd}_a{i}_{role}",
                    "role": role,
                    "agent_index": i,
                    "turn": "team",
                    "session_id": f"ses_{i}",
                    "returncode": 0,
                    "ok": True,
                    "duration_s": 5.0,
                }
                for rnd in (1, 2)
                for i, role in enumerate(
                    ["architect", "implementer", "designer", "qa_finisher"], 1
                )
            ],
        },
        {
            f"r{rnd}_a{i}_{role}": "\n".join(
                [
                    tool_event(
                        1000 * rnd + 100 * i,
                        "bash",
                        input={"command": f"echo {role}"},
                        session=f"ses_{i}",
                    ),
                    step_event(
                        1000 * rnd + 100 * i + 1, reasoning=7, session=f"ses_{i}"
                    ),
                ]
            )
            for rnd in (1, 2)
            for i, role in enumerate(
                ["architect", "implementer", "designer", "qa_finisher"], 1
            )
        },
    )

    write_build(
        root,
        "idea__dynamic",
        {
            "idea_id": "idea",
            "model": "claude",
            "structure": "dynamic",
            "n_agents": 1,
            "collab": "local",
            "status": "ok",
            "created_at": "2026-08-17T20:09:00+00:00",
            "roles": [],
            "rounds_run": 1,
            "turns_spent": 1,
            "subagents_spawned": 2,
            "orchestration": {
                "subagents_spawned": 2,
                "peak_concurrent_subagents": 2,
                "spawns": [
                    {
                        "subagent_type": "general",
                        "description": "qa",
                        "status": "completed",
                        "session_id": "ses_c1",
                        "start_ms": 100,
                        "end_ms": 400,
                        "prompt": "do qa",
                    },
                    {
                        "subagent_type": "general",
                        "description": "polish",
                        "status": "completed",
                        "session_id": "ses_c2",
                        "start_ms": 200,
                        "end_ms": 500,
                        "prompt": "polish",
                    },
                ],
            },
            "phases": [
                {
                    "phase": "t1_founder",
                    "role": "founder",
                    "agent_index": 1,
                    "turn": "dynamic",
                    "session_id": "ses_a",
                    "returncode": 0,
                    "ok": True,
                    "duration_s": 9.0,
                }
            ],
        },
        {
            "t1_founder": "\n".join(
                [
                    tool_event(
                        100,
                        "task",
                        input={"subagent_type": "general", "description": "qa"},
                        metadata={"sessionId": "ses_c1"},
                    ),
                    tool_event(
                        300,
                        "browser_browser_click",
                        input={"element": "Start button", "ref": "e5"},
                        output=BROWSER_RESULT,
                    ),
                ]
            )
        },
    )

    # Retired relay mode, and a turn that was killed mid-write.
    write_build(
        root,
        "idea__legacy",
        {
            "idea_id": "idea",
            "model": "m",
            "structure": "specialist",
            "n_agents": 2,
            "status": "ok",
            "created_at": "2026-07-13T21:30:00+00:00",
            "turn_budget": 4,
            "phases": [
                {
                    "phase": "a1_architect.design",
                    "role": "architect",
                    "agent_index": 1,
                    "turn": "design",
                    "session_id": "s",
                    "returncode": 0,
                    "ok": True,
                    "duration_s": 1.0,
                },
                {
                    "phase": "a2_builder.build",
                    "role": "builder",
                    "agent_index": 2,
                    "turn": "build",
                    "session_id": "s",
                    "returncode": -15,
                    "ok": False,
                    "duration_s": 1.0,
                },
            ],
        },
        {
            "a1_architect.design": "",  # a failed turn writes nothing at all
            "a2_builder.build": tool_event(10, "bash", input={"command": "x"})
            + '\n{"type":"tool_use","time',
        },
    )
    return root


def make_crowd_run(root: Path, run_id: str, *, app_type="client-app") -> Path:
    run_dir = root / "crowd" / run_id
    (run_dir / "traces").mkdir(parents=True, exist_ok=True)

    summary = {
        "build_id": run_id.split("__crowd-")[0],
        "crowd_arch_version": "12",
        "ok": True,
        "app_type": app_type,
        "duration_s": 61.0,
        "rounds_run": 2,
        "launch_post_id": 1,
        "config": {
            "n_agents": 2,
            "rounds": 2,
            "model_id": "gemini",
            "recsys_type": "twhin-bert",
            "seed": 0,
        },
        "engagement": {
            "posts": 2,
            "likes": 1,
            "comments": 1,
            "reposts": 1,
            "follows": 2,
        },
        "verdicts": {
            "triers": {
                "n": 1,
                "would_use_rate": 1.0,
                "delight_mean": 8.0,
                "per_agent": [
                    {
                        "agent_id": 1,
                        "username": "maya",
                        "would_use": True,
                        "would_share": True,
                        "delight": 8,
                        "craft": 8.5,
                        "n_steps": 3,
                        "app_reachable": True,
                    }
                ],
            },
            # Deliberately different figures from the triers above: the two
            # passes are separate questions, and a fixture where they agree
            # cannot catch the UI reading one and labelling it the other.
            "interviews": {
                "n": 2,
                "expected": 2,
                "would_use_rate": 0.5,
                "delight_mean": 6.0,
                "per_agent": [
                    {
                        "agent_id": 1,
                        "username": "maya",
                        "tier": "trier",
                        "would_use": True,
                        "delight": 8,
                        "for_me": True,
                        "why": "it is good",
                    },
                    {
                        "agent_id": 2,
                        "username": "onlooker",
                        "tier": "lurker",
                        "would_use": False,
                        "delight": 4,
                        "for_me": False,
                        "why": "never opened it",
                    },
                ],
            },
        },
        "crowd": [
            {
                "agent_id": 1,
                "username": "maya",
                "archetype": "builder",
                "tier": "trier",
                "influence": 3,
            },
            {
                "agent_id": 2,
                "username": "sam",
                "archetype": "student",
                "tier": "reactor",
                "influence": 1,
            },
        ],
        "rounds": [
            {"round": 1, "active_agents": 2, "ok": True},
            {"round": 2, "active_agents": 2, "ok": True},
        ],
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    # info is a JSON string inside the JSON object -- the double encoding is real.
    lines = [
        {
            "user_id": 0,
            "created_at": 0,
            "action": "sign_up",
            "info": json.dumps({"user_name": "founder"}),
        },
        {
            "user_id": 1,
            "created_at": 0,
            "action": "follow",
            "info": json.dumps({"follow_id": 1}),
        },
        {
            "user_id": 0,
            "created_at": 1,
            "action": "create_post",
            "info": json.dumps({"content": "launched", "post_id": 1}),
        },
        {
            "user_id": 1,
            "created_at": 2,
            "action": "refresh",
            "info": json.dumps(
                {
                    "posts": [
                        {
                            "post_id": 1,
                            "user_id": 0,
                            "content": "launched",
                            "author": "@founder",
                        }
                    ]
                }
            ),
        },
        {
            "user_id": 1,
            "created_at": 2,
            "action": "repost",
            "info": json.dumps({"reposted_id": 1, "new_post_id": 2}),
        },
        {
            "user_id": 2,
            "created_at": 3,
            "action": "create_comment",
            "info": json.dumps({"content": "nice", "comment_id": 1}),
        },
        "{ this line is corrupt",
    ]
    (run_dir / "actions.jsonl").write_text(
        "\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines),
        encoding="utf-8",
    )

    conn = sqlite3.connect(run_dir / "simulation.db")
    conn.executescript(
        """
        CREATE TABLE user (
            user_id INTEGER PRIMARY KEY, agent_id INTEGER, user_name TEXT,
            name TEXT, bio TEXT, created_at DATETIME,
            num_followings INTEGER, num_followers INTEGER);
        CREATE TABLE post (
            post_id INTEGER PRIMARY KEY, user_id INTEGER, original_post_id INTEGER,
            content TEXT, quote_content TEXT, created_at DATETIME,
            num_likes INTEGER, num_dislikes INTEGER,
            num_shares INTEGER, num_reports INTEGER);
        CREATE TABLE follow (
            follow_id INTEGER PRIMARY KEY, follower_id INTEGER,
            followee_id INTEGER, created_at DATETIME);
        CREATE TABLE comment (
            comment_id INTEGER PRIMARY KEY, post_id INTEGER, user_id INTEGER,
            content TEXT, created_at DATETIME,
            num_likes INTEGER, num_dislikes INTEGER);
        CREATE TABLE "like" (
            like_id INTEGER PRIMARY KEY, user_id INTEGER,
            post_id INTEGER, created_at DATETIME);
        CREATE TABLE rec (user_id INTEGER, post_id INTEGER);
        INSERT INTO user VALUES
            (0,0,'founder','The Founder','maker',0,0,2),
            (1,1,'maya','Maya','builds things',0,1,0),
            (2,2,'sam','Sam','learns',0,1,0);
        INSERT INTO post VALUES
            (1,0,NULL,'launched',NULL,1,1,0,1,0),
            (2,1,1,'',NULL,2,0,0,0,0);
        INSERT INTO follow VALUES (1,1,0,0),(2,2,0,0);
        INSERT INTO comment VALUES (1,1,2,'nice',3,0,0);
        INSERT INTO "like" VALUES (1,2,1,2);
        INSERT INTO rec VALUES (1,1),(2,1);
        """
    )
    conn.commit()
    conn.close()

    shot = run_dir / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "traces" / "agent_1.json").write_text(
        json.dumps(
            {
                "agent_id": 1,
                "persona": "maya",
                "trace": {
                    "build_id": summary["build_id"],
                    "app_type": app_type,
                    "started_at": 1.0,
                    "ended_at": 9.0,
                    "degraded": False,
                    "app_reachable": True,
                    "target_url": "http://localhost:1234/",
                    "verdict": {"would_use": True, "delight": 8, "craft": 8.5},
                    "steps": [
                        {
                            "index": 0,
                            "action": "open",
                            "args": {"url": "http://localhost:1234/"},
                            "summary": "Opened",
                            "ok": True,
                            "errors": [],
                            "screenshot": None,
                            "ts": 1.0,
                            "duration_s": 0.2,
                        },
                        {
                            "index": 1,
                            "action": "click",
                            "args": {"target": "Start"},
                            "summary": "Clicked Start",
                            "ok": True,
                            "errors": [],
                            "screenshot": str(shot),
                            "ts": 4.0,
                            "duration_s": 0.4,
                        },
                        {
                            "index": 2,
                            "action": "finish",
                            "args": {"would_use": True, "delight": 8},
                            "summary": "done",
                            "ok": True,
                            "errors": [],
                            "screenshot": None,
                            "ts": 9.0,
                            "duration_s": 0.0,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trajectories.json").write_text(
        json.dumps(
            [
                {
                    "agent_id": 1,
                    "username": "maya",
                    "tier": "trier",
                    "n_messages": 4,
                    "reasoning": ["I tried the app and it worked."],
                    "stimulus_sample": "You are Maya.",
                },
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "autorating.json").write_text(
        json.dumps(
            {
                "build_id": summary["build_id"],
                "model": "gemini",
                "repeats": 3,
                "dimensions": {
                    "substance": {
                        "score": 8,
                        "samples": [8, 8, 8],
                        "spread": 0,
                        "evidence": ["agent 1", "POST 1"],
                        "reason": "real bug found",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


# --------------------------------------------------------------------------
# Rubric grades
# --------------------------------------------------------------------------


def make_rubric_run(
    root: Path, run_id: str, *, idea_id="demo_idea", gate_failed=False
) -> Path:
    """One RubricScore grade, shaped like the real thing.

    Deliberately not a clean pass. A fixture where everything succeeds cannot
    catch a viewer that renders failure states wrong, and every one of these is a
    case the real corpus produces:

    * a failed Tier 1 item, with expected and observed differing;
    * an item that disagreed 2-of-3 across passes, so the reliability figures
      have something to report;
    * a ``harness_override`` -- the code overruled the model's stated verdict;
    * a not-applicable universal item, so the denominator is 92 and not 100;
    * a fired penalty;
    * a corrupt final transcript line, because a killed sweep leaves one.
    """
    run_dir = root / "rubric" / run_id
    (run_dir / "shots").mkdir(parents=True, exist_ok=True)
    build_id = run_id.split("__rubric-")[0]

    grade = {
        "kind": "viral_bench.rubric_grade",
        "version": 1,
        "rubric_version": "1",
        "score_version": "1.0",
        "run_id": run_id,
        "build_id": build_id,
        "idea_id": idea_id,
        "graded_at": "2026-01-01T00:00:00+00:00",
        "grader_model": "claude-test@default",
        "passes": 3,
        "brief_fingerprint": "era-3",
        "source_hash": "abc123def456",
        "ok": True,
        "error": None,
        "founder": {
            "on_disk": True,
            "build_id": build_id,
            "idea_id": idea_id,
            "arm": "solo",
            "arm_label": "single agent",
            "model": "publishers/anthropic/claude-test@default",
            "model_short": "claude-test",
            "status": "shipped",
            "app_title": "Demo App",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        "score": 62.0,
        "math": {
            "points_earned": 64,
            "points_applicable": 92,
            "base": 69.6,
            "penalty_total": -8,
            "penalty_capped": False,
            "floor_applied": False,
            "gate_zeroed": False,
        },
        "gate": {
            "passed": True,
            "label": "Deliverability gate",
            "failures": [],
            "items": [
                {
                    "id": f"G{n}",
                    "text": f"Gate item {n}.",
                    "method": "probe",
                    "passed": True,
                    "detail": "ok",
                    "observed": "",
                    "evidence": [],
                    "passes": [True],
                }
                for n in range(1, 6)
            ],
        },
        "tiers": [
            {
                "tier": 1,
                "label": "Success criteria",
                "points": 40,
                "earned": 24,
                "items": [
                    {
                        "id": "S1",
                        "text": "The export button produces a real PDF.",
                        "points": 16,
                        "earned": 16,
                        "method": "assert",
                        "expect": "a file starting %PDF-",
                        "note": "",
                        "passed": True,
                        "unresolved": False,
                        "disagreement": False,
                        "harness_override": False,
                        "observed": "application/pdf, 24kB, 3 pages",
                        "reason": "",
                        "evidence": ["tc_004"],
                        "passes": [True, True, True],
                    },
                    {
                        "id": "S2",
                        "text": "The savings figure is arithmetically correct.",
                        "points": 16,
                        "earned": 0,
                        "method": "assert",
                        "expect": "(before-after)/before, sign included",
                        "note": "",
                        "passed": False,
                        "unresolved": False,
                        "disagreement": False,
                        "harness_override": True,
                        "observed": "SAVED 166.9% larger",
                        "reason": "sign stripped; the file grew",
                        "evidence": ["tc_007"],
                        "passes": [False, False, False],
                    },
                    {
                        "id": "S3",
                        "text": "Keyboard alone can drive the primary flow.",
                        "points": 8,
                        "earned": 8,
                        "method": "agent",
                        "expect": "",
                        "note": "",
                        "passed": True,
                        "unresolved": False,
                        "disagreement": True,
                        "harness_override": False,
                        "observed": "reached submit via Tab x4",
                        "reason": "one pass could not find the focus ring",
                        "evidence": ["tc_009"],
                        "passes": [True, False, True],
                    },
                ],
            },
            {
                "tier": 2,
                "label": "Core features",
                "points": 25,
                "earned": 25,
                "items": [
                    {
                        "id": "F1",
                        "text": "Items can be created and deleted.",
                        "points": 25,
                        "earned": 25,
                        "method": "assert",
                        "expect": "count returns to zero",
                        "note": "",
                        "passed": True,
                        "unresolved": False,
                        "disagreement": False,
                        "harness_override": False,
                        "observed": "3 created, 3 deleted",
                        "reason": "",
                        "evidence": ["tc_011"],
                        "passes": [True, True, True],
                    }
                ],
            },
            {
                "tier": 3,
                "label": "Robustness and craft",
                "points": 27,
                "earned": 15,
                "items": [
                    {
                        "id": "R2",
                        "text": "No uncaught console errors during the flow.",
                        "points": 15,
                        "earned": 15,
                        "method": "assert",
                        "expect": "zero pageerrors",
                        "note": "",
                        "passed": True,
                        "unresolved": False,
                        "disagreement": False,
                        "harness_override": False,
                        "observed": "0 errors",
                        "reason": "",
                        "evidence": [],
                        "passes": [True, True, True],
                    },
                    {
                        "id": "R3",
                        "text": "Every request is same-origin.",
                        "points": 12,
                        "earned": 0,
                        "method": "assert",
                        "expect": "no third-party host",
                        "note": "",
                        "passed": False,
                        "unresolved": False,
                        "disagreement": False,
                        "harness_override": False,
                        "observed": "cdn.jsdelivr.net",
                        "reason": "loads a CDN behind an 'Offline' badge",
                        "evidence": ["tc_013"],
                        "passes": [False, False, False],
                    },
                ],
            },
        ],
        "penalties": [
            {
                "id": "A1",
                "text": "Reports a number it did not compute.",
                "points": -8,
                "max_total": None,
                "method": "assert",
                "fired": True,
                "reason": "savings figure fabricated",
                "observed": "166.9%",
                "evidence": ["tc_007"],
                "passes": [True, True, True],
                "harness_override": False,
            },
            {
                "id": "A2",
                "text": "Ships demo content as if it were the user's.",
                "points": -6,
                "max_total": None,
                "method": "assert",
                "fired": False,
                "reason": "",
                "observed": "empty on first load",
                "evidence": [],
                "passes": [False, False, False],
                "harness_override": False,
            },
        ],
        "not_applicable": [
            {"id": "R1", "reason": "the brief never asks for persistence"}
        ],
        "unresolved": [],
        "reliability": {
            "items_total": 6,
            "items_disagreeing": 1,
            "by_method": {"agent": 0.5, "assert": 0.0},
            "code_disagreement": 0.0,
            "agent_disagreement": 0.5,
            "override_rate": 0.1667,
            "unresolved": 0,
        },
        "comparison": {
            "viral_score_mean": 41.9,
            "viral_score_min": 38.0,
            "viral_score_max": 45.8,
            "crowd_runs": 3,
        },
        "paths": {
            "run_dir": str(run_dir),
            "grade": str(run_dir / "grade.json"),
            "transcript": str(run_dir / "transcript.jsonl"),
            "shots": str(run_dir / "shots"),
        },
    }
    if gate_failed:
        # The undeliverable shape: the manifest never parsed, so nothing below the
        # gate was ever observed. Those items are unresolved, NOT failed -- one
        # typo is one defect, and rendering it as 27 would misdescribe the build.
        grade["gate"]["passed"] = False
        grade["gate"]["failures"] = ["G1"]
        grade["gate"]["items"][0].update(
            {
                "passed": False,
                "detail": "invalid JSON: leading comment on line 1",
                "passes": [False],
            }
        )
        for item in grade["gate"]["items"][1:]:
            item.update(
                {"passed": None, "detail": "not reached: G1 failed", "passes": [None]}
            )
        grade["score"] = 0.0
        grade["math"].update(
            {"gate_zeroed": True, "points_earned": 0, "base": 0.0, "penalty_total": 0}
        )
        for tier in grade["tiers"]:
            tier["earned"] = 0
            for item in tier["items"]:
                item.update(
                    {
                        "passed": None,
                        "unresolved": True,
                        "earned": 0,
                        "observed": "",
                        "reason": "not observed: the gate failed",
                        "passes": [None],
                        "harness_override": False,
                        "disagreement": False,
                    }
                )
        for penalty in grade["penalties"]:
            penalty.update({"fired": False, "passes": [None]})
        grade["unresolved"] = [i["id"] for t in grade["tiers"] for i in t["items"]]
        # Nothing was measured, so nothing can have disagreed. Leaving the
        # healthy-run reliability figures here would have the page claim an item
        # disagreed across passes on a build that was never opened.
        grade["reliability"].update(
            {
                "items_disagreeing": 0,
                "by_method": {},
                "code_disagreement": 0.0,
                "agent_disagreement": 0.0,
                "override_rate": 0.0,
                "unresolved": len(grade["unresolved"]),
            }
        )

    (run_dir / "grade.json").write_text(json.dumps(grade, indent=2), encoding="utf-8")

    calls = [
        {
            "id": "tc_004",
            "name": "click",
            "args": {"target": "Export PDF"},
            "result": "clicked 'Export PDF'",
            "ok": True,
            "item_id": "S1",
        },
        {
            "id": "tc_007",
            "name": "look",
            "args": {},
            "result": "text: SAVED 166.9% larger",
            "ok": True,
            "item_id": "S2",
        },
        {
            "id": "tc_009",
            "name": "press_key",
            "args": {"key": "Tab"},
            "result": "pressed Tab",
            "ok": True,
            "item_id": "S3",
        },
        {
            "id": "tc_011",
            "name": "evaluate",
            "args": {"js": "items.length"},
            "result": "0",
            "ok": True,
            "item_id": "F1",
        },
        {
            "id": "tc_013",
            "name": "look",
            "args": {},
            "result": "request to cdn.jsdelivr.net",
            "ok": True,
            "item_id": "R3",
        },
        {
            "id": "tc_014",
            "name": "click",
            "args": {"target": "Ghost"},
            "result": "click failed: element not found",
            "ok": False,
            "item_id": "R3",
        },
    ]
    lines = [json.dumps(c) for c in calls]
    # A killed sweep leaves a truncated final line, so the reader must skip it
    # rather than losing the whole transcript.
    lines.append('{ "id": "tc_015", "name": "look"')
    (run_dir / "transcript.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (run_dir / "shots" / "shot_001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return run_dir


# --------------------------------------------------------------------------
# Session-store dumps
# --------------------------------------------------------------------------


def session_row(
    session_id, parent="", *, title="", agent="build", model="", ts=0, **tokens
):
    """One ``record: "session"`` line, in the harness's dump shape.

    ``model`` on a session row is the raw store column -- a JSON *string*, not the
    ``provider/model`` form the part rows use. Faithful here on purpose: the reader
    has to normalise it, and a fixture that pre-normalised would hide that.
    """
    return json.dumps(
        {
            "record": "session",
            "session_id": session_id,
            "parent_session_id": parent,
            "title": title,
            "agent": agent,
            "model": model
            or json.dumps({"id": "gemini-test", "providerID": "google-vertex"}),
            "cost": tokens.pop("cost", 0.5),
            "tokens": {
                "input": tokens.get("input", 10),
                "output": tokens.get("output", 20),
                "reasoning": tokens.get("reasoning", 30),
                "cache_read": 0,
                "cache_write": 0,
            },
            "time_created": ts,
            "order": 0,
        }
    )


def part_row(part_id, session_id, part, *, role="assistant", ts=0, message="msg_1"):
    return json.dumps(
        {
            "record": "part",
            "part_id": part_id,
            "message_id": message,
            "session_id": session_id,
            "role": role,
            "agent": "build",
            "model": "google-vertex/gemini-test",
            "time_created": ts,
            "part": part,
        }
    )


def write_dump(build_dir: Path, name: str, lines: list[str]) -> Path:
    dump = build_dir / "transcript" / "sessions" / f"{name}.jsonl"
    dump.parent.mkdir(parents=True, exist_ok=True)
    dump.write_text("\n".join(lines), encoding="utf-8")
    return dump


@pytest.fixture
def traced(tmp_path: Path) -> Path:
    """A build whose session store was dumped.

    Carries a prompt, thinking in all three states, a subagent and a patch.
    """
    root = tmp_path / "builds"
    build_dir = write_build(
        root,
        "idea__traced",
        {
            "idea_id": "idea",
            "model": "google-vertex-anthropic/claude-test@default",
            "structure": "dynamic",
            "n_agents": 1,
            "collab": "local",
            "status": "ok",
            "created_at": "2026-08-22T05:00:00+00:00",
            "roles": [],
            "rounds_run": 1,
            "turns_spent": 1,
            "subagents_spawned": 1,
            "trajectory": {
                "reasoning_parts": 3,
                "reasoning_chars": 17,
                "turns_with_reasoning": 1,
                "turns": 1,
                "session_records": 9,
                "turns_dumped": 1,
            },
            "phases": [
                {
                    "phase": "t1_founder",
                    "role": "founder",
                    "agent_index": 1,
                    "turn": "dynamic",
                    "session_id": "ses_root",
                    "returncode": 0,
                    "ok": True,
                    "duration_s": 12.0,
                    "reasoning_parts": 3,
                    "reasoning_chars": 17,
                    "sessions_records": 9,
                }
            ],
        },
        {"t1_founder": "\n".join([text_event(1000, "shipped"), step_event(1200)])},
    )

    write_dump(
        build_dir,
        "ses_root",
        [
            session_row("ses_root", "", title="RoughDraw", ts=990),
            session_row(
                "ses_kid", "ses_root", title="the subagent", agent="general", ts=1050
            ),
            part_row(
                "p1",
                "ses_root",
                {"type": "text", "text": "THE PROMPT"},
                role="user",
                ts=1000,
            ),
            part_row(
                "p2", "ses_root", {"type": "reasoning", "text": "thinking"}, ts=1010
            ),
            # Empty text plus provider metadata: it thought, we may not read it.
            part_row(
                "p3",
                "ses_root",
                {
                    "type": "reasoning",
                    "text": "",
                    "metadata": {"anthropic": {"signature": "x"}},
                },
                ts=1020,
            ),
            # Empty with no metadata at all: it genuinely did not think.
            part_row("p4", "ses_root", {"type": "reasoning", "text": ""}, ts=1030),
            part_row(
                "p5",
                "ses_root",
                {
                    "type": "tool",
                    "tool": "bash",
                    "state": {
                        "status": "completed",
                        "input": {"command": "ls"},
                        "output": "app/",
                        "time": {"start": 1035, "end": 1040},
                    },
                },
                ts=1040,
            ),
            part_row("p6", "ses_kid", {"type": "reasoning", "text": "sub"}, ts=1050),
            part_row(
                "p7", "ses_kid", {"type": "text", "text": "SUBAGENT OUTPUT"}, ts=1060
            ),
            part_row(
                "p8",
                "ses_root",
                {"type": "patch", "hash": "abc123", "files": ["/x/app/index.html"]},
                ts=1070,
            ),
            part_row("p9", "ses_root", {"type": "step-start"}, ts=1080),
        ],
    )
    return root
