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

"""Tests for the shared running-app instance pool (host mode)."""

from __future__ import annotations

import json
import socket

import pytest

from viral_bench.founder.apphost import AppHost
from viral_bench.founder.build import run_build
from viral_bench.founder.harness import HarnessResult, PhaseResult
from viral_bench.founder.runtime import AppRuntimeError


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _manifest(port: int) -> dict:
    """A trivial web app that stays up and serves.

    This used to be a port-less ``bot`` ("sleep 120") because the pool's
    start/reuse/restart/teardown logic does not care what the app does. It has
    to serve for real now: every app is a web app, and ``LocalRuntime.start``
    will not hand out a URL until the declared port answers HTTP.
    """
    return {
        "app_type": "client-app",
        "title": "Sleeper",
        "summary": "Stays up so the crowd can share one instance.",
        "run": {
            "command": f"python3 -m http.server {port}",
            "port": port,
            "url": f"http://localhost:{port}/",
        },
        "test": {"manual": ["ping it"]},
    }


class FakeHarness:
    def __init__(self, port: int) -> None:
        self.port = port

    def run(self, idea, workspace) -> HarnessResult:
        (workspace.app_dir / "viralbench.json").write_text(
            json.dumps(_manifest(self.port))
        )
        (workspace.app_dir / "index.html").write_text("<title>Sleeper</title>\n")
        return HarnessResult(
            model="fake",
            phases=[
                PhaseResult("design", 0, workspace.app_dir / "t", 1.0),
                PhaseResult("build", 0, workspace.app_dir / "t", 1.0),
            ],
        )


@pytest.fixture
def build_id(monkeypatch, tmp_path):
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    record = run_build("sliding_tile_game", harness=FakeHarness(_free_port()))
    assert record.status == "ok"
    return record.build_id


def test_get_starts_once_and_shares(build_id) -> None:
    host = AppHost(container=False)
    try:
        app1 = host.get(build_id)
        app2 = host.get(build_id)
        assert app1 is app2  # same shared instance, not a new one per call
        assert app1.is_running()
        assert host.running_builds() == [build_id]
    finally:
        host.close()
    assert not app1.is_running()


def test_get_restarts_dead_instance(build_id) -> None:
    host = AppHost(container=False)
    try:
        app1 = host.get(build_id)
        app1.stop()
        assert not app1.is_running()
        app2 = host.get(build_id)
        assert app2 is not app1
        assert app2.is_running()
    finally:
        host.close()


def test_context_manager_tears_down(build_id) -> None:
    with AppHost(container=False) as host:
        app = host.get(build_id)
        assert app.is_running()
    assert not app.is_running()
    assert host.running_builds() == []


def test_apphost_threads_env_map_to_open_session(monkeypatch) -> None:
    """AppHost forwards its env_map into every session it opens for the crowd."""
    captured: dict = {}

    def fake_open_session(build_id, **kwargs):
        captured["build_id"] = build_id
        captured["kwargs"] = kwargs
        raise RuntimeError("halt after capture")

    monkeypatch.setattr("viral_bench.founder.apphost.open_session", fake_open_session)
    host = AppHost(container=True, env_map={"APP_KEY": "HOST_KEY"})
    with pytest.raises(RuntimeError, match="halt after capture"):
        host.get("b1")
    assert captured["kwargs"]["env_map"] == {"APP_KEY": "HOST_KEY"}
    assert captured["kwargs"]["container"] is True


# -- an app that cannot start must settle, not be retried forever ------------ #


def test_a_failing_start_is_retried_a_bounded_number_of_times(monkeypatch) -> None:
    """After MAX_START_ATTEMPTS the host stops trying and says why.

    Unbounded retry is what turned one bad build into a lost slot-hour. Every
    agent trial called ``get``, every call re-materialized the app (~188,600
    files for a node build), and the run was finally culled at the 3600s wall
    clock having written no ``run_summary.json`` at all -- so the cell looked
    untried and was re-offered on the next pass. Measured on a full sweep: 15 to
    34 attempts per failing build.
    """
    from viral_bench.founder.apphost import MAX_START_ATTEMPTS, AppStartFailed

    attempts = {"n": 0}

    def fake_open_session(build_id, **kwargs):
        attempts["n"] += 1
        raise AppRuntimeError("app container exited immediately; No module named x")

    monkeypatch.setattr("viral_bench.founder.apphost.open_session", fake_open_session)
    host = AppHost(container=True)

    for _ in range(MAX_START_ATTEMPTS - 1):
        with pytest.raises(AppRuntimeError):
            host.get("b1")
    # The attempt that exhausts the budget settles the build.
    with pytest.raises(AppStartFailed):
        host.get("b1")
    assert attempts["n"] == MAX_START_ATTEMPTS

    # Every later caller is refused WITHOUT another attempt -- that is the whole
    # point, since the next thirty agents would otherwise each pay for one.
    for _ in range(5):
        with pytest.raises(AppStartFailed):
            host.get("b1")
    assert attempts["n"] == MAX_START_ATTEMPTS

    assert "No module named x" in (host.unstartable("b1") or "")
    assert host.unstartable("other-build") is None


def test_a_failed_start_retires_its_clone(monkeypatch) -> None:
    """The clone made for a failed attempt must not be left behind.

    A build that cannot start is exactly the one that generates the most clones,
    and they are the largest thing on disk: leaked trees piled up in the thousands
    over one sweep and crowd throughput fell from ~100 runs/hour to roughly 2.
    """
    closed: list[str] = []

    class FakeSession:
        def __init__(self, build_id: str) -> None:
            self.build_id = build_id

        def setup(self, **_kwargs):
            return []

        def start(self, **_kwargs):
            raise AppRuntimeError("never became reachable")

        def close(self):
            closed.append(self.build_id)

    monkeypatch.setattr(
        "viral_bench.founder.apphost.open_session",
        lambda build_id, **kwargs: FakeSession(build_id),
    )
    host = AppHost(container=True)
    with pytest.raises(AppRuntimeError):
        host.get("b1")
    assert closed == ["b1"]
