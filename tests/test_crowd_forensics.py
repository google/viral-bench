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

"""Tests for the sweep forensics reader.

The forensics script is how the loop reads 150 runs end to end, so its
arithmetic has to be pinned down: a report that quietly miscounts "agents who
operated the app" would send the whole iteration after the wrong thing.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from viral_bench.crowd.sim_defaults import CROWD_ARCH_VERSION

REPO = Path(__file__).resolve().parents[1]


def _forensics():
    spec = importlib.util.spec_from_file_location(
        "crowd_forensics", REPO / "scripts" / "crowd_forensics.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["crowd_forensics"] = module
    spec.loader.exec_module(module)
    return module


def _write_build(root: Path, build_id: str, **record) -> None:
    d = root / "work" / build_id
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "build_id": build_id,
        "idea_id": "i1",
        "model": "model-a",
        "status": "ok",
        "n_agents": 1,
    }
    payload.update(record)
    (d / "build.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_run(
    root: Path,
    build_id: str,
    seed: int,
    *,
    steps: list[dict] | None = None,
    posts: list[tuple[int, int, str]] = (),
    comments: list[tuple[int, int, str]] = (),
    arch: str = CROWD_ARCH_VERSION,
) -> Path:
    d = root / "crowd" / f"{build_id}__crowd-{seed}"
    (d / "traces").mkdir(parents=True, exist_ok=True)
    (d / "run_summary.json").write_text(
        json.dumps(
            {
                "build_id": build_id,
                "crowd_arch_version": arch,
                "ok": True,
                "app_type": "client-app",
                "duration_s": 100.0,
                "launch_post_id": 1,
                "config": {"seed": seed, "n_agents": 3},
                "health": {"ok": True, "failures": []},
                "turn_stats": {"turns": 10, "skipped": 1, "skipped_rate_limited": 1},
                "engagement": {
                    "reach": {
                        "exposed_agents": 3,
                        "actors_liked": 1,
                        "actors_reposted": 0,
                        "actors_commented": 2,
                        "actors_negative": 0,
                    },
                    "exposure": {"uniform": True},
                },
                "verdicts": {
                    "interviews": {
                        "n": 2,
                        "would_use_rate": 0.5,
                        "delight_mean": 5.0,
                        "loss_rate": 0.0,
                        "per_agent": [
                            {
                                "agent_id": 1,
                                "tier": "trier",
                                "delight": 7,
                                "would_use": True,
                                "why": "it worked",
                            },
                            {
                                "agent_id": 2,
                                "tier": "reactor",
                                "delight": 3,
                                "would_use": False,
                                "why": "meh",
                            },
                        ],
                    },
                    "triers": {
                        "craft_mean": 6.0,
                        "per_agent": [
                            {
                                "agent_id": 1,
                                "craft": 6.0,
                                "functionality": 6,
                                "usability": 6,
                                "design": 6,
                                "simplicity": 6,
                            }
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    if steps is not None:
        (d / "traces" / "agent_1.json").write_text(
            json.dumps(
                {
                    "agent_id": 1,
                    "persona": "p1",
                    "trace": {
                        "steps": steps,
                        "app_reachable": True,
                        "verdict": {"would_use": True},
                    },
                }
            ),
            encoding="utf-8",
        )
    con = sqlite3.connect(d / "simulation.db")
    con.execute(
        "CREATE TABLE post (post_id INTEGER, user_id INTEGER, "
        "original_post_id INTEGER, content TEXT)"
    )
    con.execute(
        "CREATE TABLE comment (comment_id INTEGER, post_id INTEGER, "
        "user_id INTEGER, content TEXT)"
    )
    con.execute("CREATE TABLE trace (user_id INTEGER, created_at INTEGER, action TEXT)")
    for pid, uid, content in posts:
        con.execute(
            "INSERT INTO post VALUES (?,?,NULL,?)",
            (pid, uid, content),
        )
    for cid, pid, content in comments:
        con.execute("INSERT INTO comment VALUES (?,?,?,?)", (cid, pid, 5, content))
    con.execute("INSERT INTO trace VALUES (1, 2, 'create_comment')")
    con.commit()
    con.close()
    return d


def test_a_trial_that_only_reads_the_page_does_not_count_as_using_the_app(tmp_path):
    """`look` and `screenshot` are observation, and the distinction is the point.

    36% of triers once filed a verdict without a single successful click, and
    they rated apps a full point higher than triers who operated one.
    A report that counted `look` as use would have hidden that entirely.
    """
    forensics = _forensics()
    root = tmp_path / "builds"
    _write_build(root, "b1")
    _write_run(
        root,
        "b1",
        0,
        steps=[
            {"action": "open", "ok": True, "summary": "page"},
            {"action": "look", "ok": True, "summary": "page"},
            {"action": "screenshot", "ok": True, "summary": "shot"},
            {"action": "finish", "ok": True, "summary": "done"},
        ],
    )
    runs = forensics.load_runs(root, arch=CROWD_ARCH_VERSION)
    text = "\n".join(forensics.trials_section(runs))
    assert "actually operated the app 0%" in text


def test_a_failed_click_does_not_count_as_using_the_app(tmp_path):
    forensics = _forensics()
    root = tmp_path / "builds"
    _write_build(root, "b1")
    _write_run(
        root,
        "b1",
        0,
        steps=[
            {"action": "open", "ok": True, "summary": "page"},
            {"action": "click", "ok": False, "summary": "could not click 'Go'"},
            {"action": "finish", "ok": True, "summary": "done"},
        ],
    )
    runs = forensics.load_runs(root, arch=CROWD_ARCH_VERSION)
    text = "\n".join(forensics.trials_section(runs))
    assert "actually operated the app 0%" in text
    assert "click" in text


def test_a_successful_click_counts(tmp_path):
    forensics = _forensics()
    root = tmp_path / "builds"
    _write_build(root, "b1")
    _write_run(
        root,
        "b1",
        0,
        steps=[
            {"action": "open", "ok": True, "summary": "page"},
            {"action": "click", "ok": True, "summary": "clicked Go"},
        ],
    )
    runs = forensics.load_runs(root, arch=CROWD_ARCH_VERSION)
    assert "actually operated the app 100%" in "\n".join(forensics.trials_section(runs))


def test_comments_away_from_the_launch_post_are_reported_as_conversation(tmp_path):
    forensics = _forensics()
    root = tmp_path / "builds"
    _write_build(root, "b1")
    _write_run(
        root,
        "b1",
        0,
        steps=[],
        posts=[(1, 0, "launch"), (2, 4, "my take @ana")],
        comments=[(1, 1, "on the launch"), (2, 2, "@bob replying to your take")],
    )
    runs = forensics.load_runs(root, arch=CROWD_ARCH_VERSION)
    text = "\n".join(forensics.social_section(runs))
    assert "50% hang off the launch post" in text
    # Two of the four texts (2 posts + 2 comments) carry an @handle.
    assert "50% name another agent" in text
    assert "byte-identical feed" in text


def test_runs_from_another_architecture_are_not_read(tmp_path):
    forensics = _forensics()
    root = tmp_path / "builds"
    _write_build(root, "b1")
    _write_run(root, "b1", 0, steps=[], arch="0")
    assert forensics.load_runs(root, arch=CROWD_ARCH_VERSION) == []


def test_auc_is_one_when_every_positive_beats_every_negative():
    forensics = _forensics()
    assert forensics._auc([3.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert forensics._auc([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.5)
    assert forensics._auc([], [1.0]) is None


# -- the ablation driver -----------------------------------------------------


def _ablation():
    spec = importlib.util.spec_from_file_location(
        "crowd_ablation", REPO / "scripts" / "crowd_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["crowd_ablation"] = module
    spec.loader.exec_module(module)
    return module


def test_an_ablation_subset_is_paired_never_a_slice(monkeypatch):
    """A paired statistic over unpaired cells is not a paired statistic.

    Slicing the build list takes whole ideas of one model and none of the other
    whenever the count is odd, so the subset has to be chosen by IDEA.
    """
    ablation = _ablation()

    class _B:
        def __init__(self, idea, model):
            self.idea_id = idea
            self.model = model
            self.build_id = f"{idea}__{model}"

    builds = [_B(i, m) for i in ("a", "b", "c") for m in ("A", "B")]
    monkeypatch.setattr(ablation, "fleet_builds", lambda: builds)
    chosen = ablation.subset(2)
    assert {b.idea_id for b in chosen} == {"a", "b"}
    assert sorted(b.model for b in chosen) == ["A", "A", "B", "B"]
