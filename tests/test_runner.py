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

"""Tests for the run/test orchestration layer (materialize + AppSession)."""

from __future__ import annotations

import json

import pytest

from viral_bench.founder.build import run_build
from viral_bench.founder.harness import HarnessResult, PhaseResult
from viral_bench.founder.runner import (
    AppSession,
    materialize_build,
    open_session,
    runs_root,
)
from viral_bench.founder.runtime import ContainerRuntime, LocalRuntime

MANIFEST = {
    "app_type": "client-app",
    "title": "TileMerge",
    "summary": "A sliding tile puzzle.",
    "setup": [],
    "run": {"command": "python3 -m http.server 8000", "port": 8000},
    "test": {"manual": ["play it"], "smoke": "test -f index.html"},
}


class FakeHarness:
    def __init__(self, files: dict[str, str], *, ok: bool = True) -> None:
        self.files = files
        self.ok = ok

    def run(self, idea, workspace) -> HarnessResult:
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


def _files() -> dict[str, str]:
    return {
        "index.html": "<!doctype html><title>TileMerge</title>",
        "README.md": "# TileMerge",
        "viralbench.json": json.dumps(MANIFEST),
    }


@pytest.fixture
def builds_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    return tmp_path


def test_materialize_clones_shipped_app(builds_dir) -> None:
    record = run_build("sliding_tile_game", harness=FakeHarness(_files()))
    run_dir = materialize_build(record.build_id)
    assert (run_dir / "app" / "index.html").is_file()
    assert (run_dir / "app" / "viralbench.json").is_file()
    assert run_dir.parent == runs_root()


def test_materialize_is_isolated_per_call(builds_dir) -> None:
    record = run_build("sliding_tile_game", harness=FakeHarness(_files()))
    a = materialize_build(record.build_id)
    b = materialize_build(record.build_id)
    assert a != b  # each session gets its own throwaway copy (parallel-safe)


def test_materialize_falls_back_to_worktree_when_unshipped(builds_dir) -> None:
    record = run_build("sliding_tile_game", harness=FakeHarness(_files()), ship=False)
    assert record.shipped_ref is None
    run_dir = materialize_build(record.build_id)
    assert (run_dir / "app" / "index.html").is_file()


def test_open_session_host_runtime_and_smoke(builds_dir) -> None:
    record = run_build("sliding_tile_game", harness=FakeHarness(_files()))
    session = open_session(record.build_id)
    try:
        assert isinstance(session, AppSession)
        assert isinstance(session.runtime, LocalRuntime)
        assert session.app_dir.is_dir()
        smoke = session.smoke()
        assert smoke is not None and smoke.returncode == 0
    finally:
        session.close()
    assert not session.run_dir.exists()  # close() cleans up the throwaway copy


def test_open_session_container_injects_the_apps_own_llm_credential(
    builds_dir, monkeypatch
) -> None:
    """The app gets the ``app`` stage's endpoint/key/model, and nothing else.

    Built apps are untrusted, model-generated code, so the credential the
    pipeline itself runs on must never reach one. The variables are
    provider-neutral (viral_bench.founder.appenv), so an app has one code path
    whichever provider the benchmark is being run against.
    """
    from viral_bench import config
    from viral_bench.founder.appenv import (
        APP_API_KEY_VAR,
        APP_BASE_URL_VAR,
        APP_MODEL_VAR,
    )

    # Constructing the container session needs no runtime; only run/start do.
    # Point the .env resolver at a nonexistent file so the real repo .env is not
    # read; drive inputs purely via the environment for determinism.
    monkeypatch.setenv("VIRAL_BENCH_ENV_FILE", str(builds_dir / "none.env"))
    monkeypatch.setattr(
        config,
        "stage_model",
        lambda stage, default="": "custom/app-model-test" if stage == "app" else "",
    )
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://app-llm.test/v1")
    monkeypatch.setenv("CUSTOM_API_KEY", "app-key")
    monkeypatch.setenv("OPENAI_API_KEY", "pipeline-secret")

    record = run_build("sliding_tile_game", harness=FakeHarness(_files()))
    session = open_session(record.build_id, container=True)
    try:
        assert isinstance(session.runtime, ContainerRuntime)
        assert session.runtime.env == {
            APP_BASE_URL_VAR: "http://app-llm.test/v1",
            APP_API_KEY_VAR: "app-key",
            APP_MODEL_VAR: "app-model-test",
        }
        assert "pipeline-secret" not in session.runtime.env.values()
    finally:
        session.close()


def test_open_session_container_env_map_override(builds_dir, monkeypatch) -> None:
    from viral_bench import config

    monkeypatch.setenv("VIRAL_BENCH_ENV_FILE", str(builds_dir / "none.env"))
    monkeypatch.setattr(config, "stage_model", lambda stage, default="": "")
    monkeypatch.setenv("MY_HOST_KEY", "v")
    record = run_build("sliding_tile_game", harness=FakeHarness(_files()))
    session = open_session(
        record.build_id, container=True, env_map={"APP_KEY": "MY_HOST_KEY"}
    )
    try:
        assert isinstance(session.runtime, ContainerRuntime)
        assert session.runtime.env == {"APP_KEY": "v"}
    finally:
        session.close()


def test_sweep_run_dirs_removes_stale_leaves_fresh(monkeypatch, tmp_path) -> None:
    """Leaked clones get collected; a concurrent run's live clone does not.

    Sessions delete their own run dir on close, but a crash or a start that
    raised leaks it. This checkout had 1,412 such directories (36 GB) before a
    sweeper existed.
    """
    import os
    import time as _time

    from viral_bench.founder.runner import runs_root, sweep_run_dirs

    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    root = runs_root()
    root.mkdir(parents=True)

    stale = root / "build__run-old"
    stale.mkdir()
    (stale / "app").mkdir()
    old = _time.time() - (24 * 3600)
    os.utime(stale, (old, old))

    live = root / "build__run-live"
    live.mkdir()

    removed = sweep_run_dirs(older_than_hours=6.0)
    assert removed == 1
    assert not stale.exists()
    assert live.exists(), "a clone in use by a concurrent run must survive"
