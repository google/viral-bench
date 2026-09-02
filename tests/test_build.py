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

"""Tests for the build orchestrator + ship-to-store (with a fake harness)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from viral_bench.founder.build import (
    BuildError,
    _record_trajectory,
    generate_build_id,
    list_builds,
    load_build_record,
    resolve_idea,
    run_build,
)
from viral_bench.founder.collab import CollabError
from viral_bench.founder.harness import HarnessResult, PhaseResult
from viral_bench.founder.structures import StructureError
from viral_bench.founder.workspace import BuildWorkspace
from viral_bench.ideas import ideas_dir

MANIFEST = {
    "app_type": "client-app",
    "title": "TileMerge",
    "summary": "A sliding tile puzzle.",
    "setup": [],
    "run": {"command": "python3 -m http.server 8000", "port": 8000},
    "test": {"manual": ["play it"], "smoke": "test -f index.html"},
}


class FakeHarness:
    """A harness that writes canned files and reports success/failure."""

    def __init__(self, files: dict[str, str], *, ok: bool = True) -> None:
        self.files = files
        self.ok = ok

    def run(self, idea, workspace: BuildWorkspace) -> HarnessResult:
        for name, content in self.files.items():
            (workspace.app_dir / name).write_text(content, encoding="utf-8")
        rc = 0 if self.ok else 1
        return HarnessResult(
            model="fake",
            phases=[
                PhaseResult("design", 0, workspace.app_dir / "t", 1.0),
                PhaseResult("build", rc, workspace.app_dir / "t", 1.0),
            ],
        )


@pytest.fixture
def builds_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    return tmp_path


def _good_files() -> dict[str, str]:
    return {
        "index.html": "<!doctype html><title>TileMerge</title>",
        "README.md": "# TileMerge",
        "viralbench.json": json.dumps(MANIFEST),
    }


def test_generate_build_id_shape() -> None:
    bid = generate_build_id("my_idea")
    assert bid.startswith("my_idea__")
    assert len(bid.split("__")) == 3


def test_unknown_idea_raises(builds_dir) -> None:
    with pytest.raises(BuildError, match="unknown idea_id"):
        run_build("does_not_exist", harness=FakeHarness({}))


def test_successful_build_ships_orphan_branch(builds_dir) -> None:
    harness = FakeHarness(_good_files())
    record = run_build("sliding_tile_game", harness=harness)

    assert record.status == "ok"
    assert record.harness_ok
    assert record.shipped_ref == f"build/{record.build_id}"

    # build.json persisted
    reloaded = load_build_record(record.build_id)
    assert reloaded.build_id == record.build_id
    assert reloaded.manifest["app_type"] == "client-app"

    # store is ONE git repo with an orphan branch whose root is the app
    store = builds_dir / "store"
    assert (store / ".git").is_dir()
    names = subprocess.run(
        ["git", "-C", str(store), "ls-tree", "--name-only", record.shipped_ref],
        capture_output=True,
        text=True,
    ).stdout.split()
    assert "index.html" in names
    assert "viralbench.json" in names


def test_multiple_builds_share_one_store(builds_dir) -> None:
    r1 = run_build("sliding_tile_game", harness=FakeHarness(_good_files()))
    r2 = run_build("quick_notes_app", harness=FakeHarness(_good_files()))

    store = builds_dir / "store"
    branches = subprocess.run(
        ["git", "-C", str(store), "branch", "--format=%(refname:short)"],
        capture_output=True,
        text=True,
    ).stdout.split()
    assert r1.shipped_ref in branches
    assert r2.shipped_ref in branches
    # exactly one repo (one .git) for all builds
    assert (store / ".git").is_dir()
    assert len(list_builds()) == 2


def test_missing_manifest_not_shipped(builds_dir) -> None:
    files = {"index.html": "<title>x</title>"}  # no viralbench.json
    record = run_build("sliding_tile_game", harness=FakeHarness(files))
    assert record.status == "manifest_missing"
    assert record.shipped_ref is None


def test_invalid_manifest_not_shipped(builds_dir) -> None:
    files = _good_files()
    files["viralbench.json"] = json.dumps({"app_type": "client-app"})
    record = run_build("sliding_tile_game", harness=FakeHarness(files))
    assert record.status == "manifest_invalid"
    assert record.shipped_ref is None


def test_harness_failure_not_shipped(builds_dir) -> None:
    record = run_build(
        "sliding_tile_game", harness=FakeHarness(_good_files(), ok=False)
    )
    assert record.status == "harness_failed"
    assert record.shipped_ref is None


def test_a_timed_out_turn_is_not_recorded_as_a_build_failure(builds_dir) -> None:
    """A wall-clock kill is OUR limit biting, not the model failing.

    Both used to land as ``harness_failed``, which made a slow-but-working model
    indistinguishable from one that could not ship an app -- a fabricated
    capability difference (docs/crowd_bugs.md T0.1). ``harness_timeout`` is a
    distinct status, and scripts/build_fleet.py auto-retries it for that reason.
    """

    class TimeoutHarness(FakeHarness):
        def run(self, idea, workspace: BuildWorkspace) -> HarnessResult:
            for name, content in self.files.items():
                (workspace.app_dir / name).write_text(content, encoding="utf-8")
            return HarnessResult(
                model="fake",
                phases=[
                    PhaseResult("design", 0, workspace.app_dir / "t", 1.0),
                    PhaseResult(
                        "build", 124, workspace.app_dir / "t", 1.0, timed_out=True
                    ),
                ],
            )

    record = run_build("sliding_tile_game", harness=TimeoutHarness(_good_files()))
    assert record.status == "harness_timeout"
    assert record.shipped_ref is None
    # The flag survives into the record, so a fleet can be audited for timeouts.
    assert record.phases[-1]["timed_out"] is True


def test_resolve_idea_by_id() -> None:
    idea = resolve_idea("sliding_tile_game")
    assert idea.idea_id == "sliding_tile_game"


def test_resolve_idea_by_path() -> None:
    path = ideas_dir() / "sliding_tile_game.yaml"
    idea = resolve_idea(str(path))
    assert idea.idea_id == "sliding_tile_game"


def test_resolve_idea_unknown_id_raises() -> None:
    with pytest.raises(BuildError, match="unknown idea_id"):
        resolve_idea("nope_not_a_real_idea")


def test_resolve_idea_missing_file_raises() -> None:
    with pytest.raises(BuildError, match="not found"):
        resolve_idea("does/not/exist.yaml")


def test_run_build_accepts_yaml_path(builds_dir) -> None:
    path = ideas_dir() / "sliding_tile_game.yaml"
    record = run_build(str(path), harness=FakeHarness(_good_files()))
    assert record.idea_id == "sliding_tile_game"
    assert record.status == "ok"


# -- collaboration metadata -------------------------------------------------- #


def test_solo_build_records_solo_metadata(builds_dir) -> None:
    record = run_build("sliding_tile_game", harness=FakeHarness(_good_files()))
    assert record.structure == "solo"
    assert record.n_agents == 1
    assert record.max_rounds == 1
    assert record.max_turns == 2
    assert record.roles == ["founder"]
    assert record.collab == "local"
    assert record.collab_meta == {"collab": "local"}
    # metadata round-trips through build.json
    reloaded = load_build_record(record.build_id)
    assert reloaded.roles == ["founder"]
    assert reloaded.structure == "solo"


def test_team_build_records_team_metadata(builds_dir) -> None:
    record = run_build(
        "sliding_tile_game",
        agents=4,
        rounds=4,
        min_rounds=3,
        harness=FakeHarness(_good_files()),
    )
    assert record.structure == "team"
    assert record.n_agents == 4
    assert record.max_rounds == 4
    assert record.min_rounds == 3
    assert record.max_turns == 16
    assert record.min_turns == 12
    assert record.roles == ["architect", "implementer", "designer", "qa_finisher"]
    reloaded = load_build_record(record.build_id)
    assert reloaded.max_rounds == 4
    assert reloaded.min_rounds == 3


def test_run_build_rejects_min_rounds_above_max(builds_dir) -> None:
    with pytest.raises(StructureError):
        run_build(
            "sliding_tile_game",
            agents=4,
            rounds=2,
            min_rounds=3,
            harness=FakeHarness(_good_files()),
        )


@pytest.mark.parametrize("bad", [0, 2, 3, 5])
def test_run_build_rejects_unsupported_agent_counts(builds_dir, bad: int) -> None:
    with pytest.raises(StructureError):
        run_build("sliding_tile_game", agents=bad, harness=FakeHarness(_good_files()))


def test_unknown_collab_is_rejected(builds_dir) -> None:
    # The toolset is constructed up front, so a name nothing implements fails
    # here rather than after a long build.
    with pytest.raises(CollabError):
        run_build(
            "sliding_tile_game",
            agents=4,
            collab="carrier-pigeon",
            harness=FakeHarness(_good_files()),
        )


# -- ship-time cleanup ------------------------------------------------------- #


class HarnessWithPycache:
    """Writes good files plus __pycache__ scratch (top-level and nested)."""

    def run(self, idea, workspace: BuildWorkspace) -> HarnessResult:
        app = workspace.app_dir
        for name, content in _good_files().items():
            (app / name).write_text(content, encoding="utf-8")
        (app / "__pycache__").mkdir()
        (app / "__pycache__" / "server.cpython-312.pyc").write_text("x")
        pkg = app / "pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__pycache__").mkdir()
        (pkg / "__pycache__" / "__init__.cpython-312.pyc").write_text("x")
        return HarnessResult(
            model="fake",
            phases=[
                PhaseResult("design", 0, app / "t", 1.0),
                PhaseResult("build", 0, app / "t", 1.0),
            ],
        )


def test_pycache_stripped_on_ship(builds_dir) -> None:
    record = run_build("sliding_tile_game", harness=HarnessWithPycache())
    assert record.status == "ok"
    app = Path(record.app_dir)
    # bytecode caches are stripped from the work tree at every depth
    assert not (app / "__pycache__").exists()
    assert not (app / "pkg" / "__pycache__").exists()
    # ...and never reach the shipped tree, while real package files remain
    tree = subprocess.run(
        [
            "git",
            "-C",
            str(builds_dir / "store"),
            "ls-tree",
            "-r",
            "--name-only",
            record.shipped_ref,
        ],
        capture_output=True,
        text=True,
    ).stdout.split()
    assert not any("__pycache__" in p for p in tree)
    assert "pkg/__init__.py" in tree


def _shipped_names(builds_dir, ref: str) -> list[str]:
    return subprocess.run(
        ["git", "-C", str(builds_dir / "store"), "ls-tree", "--name-only", ref],
        capture_output=True,
        text=True,
    ).stdout.split()


def test_local_team_strips_scratch_keeps_design(builds_dir) -> None:
    files = _good_files()
    files["DESIGN.md"] = "# design"
    files["TEAM_NOTES.md"] = "notes for the team"
    files["HANDOFF.md"] = "legacy handoff"
    record = run_build("sliding_tile_game", agents=4, harness=FakeHarness(files))
    assert record.status == "ok"
    app = Path(record.app_dir)
    # coordination scratch is removed from the work tree and never ships...
    assert not (app / "TEAM_NOTES.md").exists()
    assert not (app / "HANDOFF.md").exists()
    names = _shipped_names(builds_dir, record.shipped_ref)
    assert "TEAM_NOTES.md" not in names
    assert "HANDOFF.md" not in names
    # ...but DESIGN.md is a real local-mode deliverable and is kept.
    assert "DESIGN.md" in names


# -- dynamic mode ------------------------------------------------------------ #


class DynamicFakeHarness:
    """A harness that writes a valid app and reports a chosen orchestration."""

    def __init__(self, spawns: int = 3) -> None:
        self.spawns = spawns

    def run(self, idea, workspace: BuildWorkspace) -> HarnessResult:
        (workspace.app_dir / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
        (workspace.app_dir / "viralbench.json").write_text(
            json.dumps(MANIFEST), encoding="utf-8"
        )
        return HarnessResult(
            model="fake",
            phases=[PhaseResult("t1_founder", 0, workspace.app_dir / "t", 1.0)],
        )


def test_dynamic_build_records_the_arm_and_its_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    record = run_build(
        "sliding_tile_game", agents="dynamic", turns=4, harness=DynamicFakeHarness()
    )
    assert record.structure == "dynamic"
    # One configured agent (the orchestrator process). The subagent count is an
    # outcome and lives in `orchestration`, never in the config fields the fleet
    # verifies an arm against.
    assert record.n_agents == 1
    assert record.max_turns == 4
    assert record.roles == []
    assert record.orchestration is not None
    assert record.orchestration["orchestrator_turns"] >= 0
    assert record.subagents_spawned == record.orchestration["subagents_spawned"]
    # Round-trips through build.json for the sweep to read later.
    reloaded = load_build_record(record.build_id)
    assert reloaded.structure == "dynamic"
    assert reloaded.orchestration == record.orchestration


def test_non_dynamic_builds_carry_no_orchestration(tmp_path, monkeypatch):
    """solo/team have no such choice to record, and a stray empty dict would
    read as 'this model orchestrated nothing'."""
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    record = run_build(
        "sliding_tile_game",
        agents=1,
        harness=FakeHarness({"viralbench.json": json.dumps(MANIFEST)}),
    )
    assert record.orchestration is None
    assert record.subagents_spawned == 0


def test_unknown_agent_selector_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    with pytest.raises(StructureError):
        run_build("sliding_tile_game", agents="swarm", harness=DynamicFakeHarness())


# -- trajectory accounting: root transcript vs the full session store -------- #


def _phase(name, *, chars, parts, records) -> PhaseResult:
    return PhaseResult(
        phase=name,
        returncode=0,
        transcript_path=Path("/nonexistent"),
        duration_s=1.0,
        reasoning_parts=parts,
        reasoning_chars=chars,
        sessions_records=records,
        sessions_path=f"/tmp/{name}.jsonl",
    )


def _dump_sessions(workspace: BuildWorkspace, records) -> None:
    path = workspace.transcript_dir / "sessions" / "root.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _reasoning_part(part_id, session, text, ts):
    return {
        "record": "part",
        "part_id": part_id,
        "message_id": "m" + part_id,
        "session_id": session,
        "role": "assistant",
        "time_created": ts,
        "part": {"type": "reasoning", "text": text},
    }


def test_trajectory_counts_subagent_reasoning_the_transcript_cannot_see(
    builds_dir,
) -> None:
    """opencode's stdout printer drops every event below the root session, so a
    transcript-derived total omits every subagent -- measured ~3x low on a
    dynamic build with four of them. The record has to carry both."""
    workspace = BuildWorkspace("b_sub").create()
    _dump_sessions(
        workspace,
        [
            {"record": "session", "session_id": "root", "parent_session_id": ""},
            {"record": "session", "session_id": "kid", "parent_session_id": "root"},
            _reasoning_part("p1", "root", "orchestrator", 1),
            _reasoning_part("p2", "kid", "the subagent thought a great deal", 2),
        ],
    )
    result = HarnessResult(
        model="m", phases=[_phase("t1", chars=len("orchestrator"), parts=1, records=4)]
    )

    traj = _record_trajectory(result, workspace)

    assert traj["reasoning_chars_root"] == len("orchestrator")
    assert traj["reasoning_chars_all"] == len("orchestrator") + len(
        "the subagent thought a great deal"
    )
    assert traj["reasoning_chars_all"] > traj["reasoning_chars_root"]
    assert traj["reasoning_parts_all"] == 2


def test_trajectory_counts_agree_when_nothing_was_delegated(builds_dir) -> None:
    """The two sources are only allowed to diverge because of subagents."""
    workspace = BuildWorkspace("b_solo").create()
    _dump_sessions(
        workspace,
        [
            {"record": "session", "session_id": "root", "parent_session_id": ""},
            _reasoning_part("p1", "root", "just me", 1),
        ],
    )
    result = HarnessResult(
        model="m", phases=[_phase("t1", chars=len("just me"), parts=1, records=2)]
    )

    traj = _record_trajectory(result, workspace)
    assert traj["reasoning_chars_all"] == traj["reasoning_chars_root"]


def test_trajectory_session_records_are_deduped_not_summed(builds_dir) -> None:
    """Each turn re-dumps its whole subtree, so summing per-turn counts reports
    more records than exist -- measured 137 recorded against 114 on disk."""
    workspace = BuildWorkspace("b_dedup").create()
    _dump_sessions(
        workspace,
        [
            {"record": "session", "session_id": "root", "parent_session_id": ""},
            _reasoning_part("p1", "root", "design", 1),
            _reasoning_part("p2", "root", "build", 2),
        ],
    )
    result = HarnessResult(
        model="m",
        phases=[
            _phase("design", chars=6, parts=1, records=1),
            _phase("build", chars=5, parts=1, records=2),
        ],
    )

    traj = _record_trajectory(result, workspace)
    assert traj["session_events"] == 2  # parts, deduped -- not 1 + 2
    assert traj["turns_dumped"] == 2


def test_trajectory_keeps_the_root_count_when_the_dump_fails(builds_dir) -> None:
    """dump_session_trace is best-effort and yields nothing on any failure. If
    the dump were the only source, a broken dump and a model that did not think
    would be indistinguishable -- the exact ambiguity this field resolves."""
    workspace = BuildWorkspace("b_nodump").create()
    result = HarnessResult(
        model="m", phases=[_phase("t1", chars=42, parts=3, records=0)]
    )

    traj = _record_trajectory(result, workspace)
    assert traj["reasoning_chars_all"] == 0
    assert traj["reasoning_chars_root"] == 42
    assert traj["turns_with_reasoning_root"] == 1
