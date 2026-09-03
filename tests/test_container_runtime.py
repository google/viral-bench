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

"""Tests for the container app runtime (ContainerRuntime).

These exercise a real rootless container and are skipped unless a container
runtime is available *and* the test image is already present locally (so CI
never has to pull). Locally, ``busybox`` is enough to prove the mechanics:
mount-write persistence, one-shot exec, dynamic-port web serving, and teardown.
Override the image with ``VIRAL_BENCH_TEST_IMAGE``.
"""

from __future__ import annotations

import json
import os
import socket
import urllib.request
from pathlib import Path

import pytest

from viral_bench.founder.apphost import AppHost
from viral_bench.founder.build import run_build
from viral_bench.founder.harness import HarnessResult, PhaseResult
from viral_bench.founder.manifest import Manifest, RunSpec
from viral_bench.founder.manifest import TestSpec as _TestSpec
from viral_bench.founder.runtime import ContainerRuntime, reap_orphans
from viral_bench.founder.verify import try_app, verify_code

IMAGE = os.environ.get("VIRAL_BENCH_TEST_IMAGE", "docker.io/library/busybox:latest")
_READY = (
    ContainerRuntime.available()
    and ContainerRuntime(Path("."), image=IMAGE).image_exists()
)

pytestmark = pytest.mark.skipif(
    not _READY, reason=f"container runtime + local image {IMAGE!r} required"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _rt(app_dir: Path, **kw) -> ContainerRuntime:
    return ContainerRuntime(app_dir, image=IMAGE, memory="256m", cpus="1.0", **kw)


def test_setup_writes_persist_to_host_mount(tmp_path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    manifest = Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        run=RunSpec(command="true"),
        setup=("touch created.txt",),
        test=_TestSpec(smoke="test -f created.txt"),
    )
    rt = _rt(app_dir)
    rt.setup(manifest)
    # The install ran in an ephemeral container but wrote through the mount.
    assert (app_dir / "created.txt").is_file()
    assert rt.smoke(manifest).returncode == 0


def test_exec_returns_stdout(tmp_path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    proc = _rt(app_dir).exec("echo hello")
    assert proc.returncode == 0
    assert proc.stdout.strip() == "hello"


def test_start_web_app_gets_dynamic_host_port(tmp_path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "index.html").write_text("<title>hi</title>")
    manifest = Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        # busybox httpd serves the cwd (/work) in the foreground.
        run=RunSpec(command="httpd -f -p 8000", port=8000),
    )
    rt = _rt(app_dir)
    app = rt.start(manifest, wait_timeout=20.0)
    try:
        assert app.container is not None
        assert rt.is_running(app)
        # URL points at the dynamically-assigned host port, not 8000.
        assert app.url and app.url.startswith("http://localhost:")
        assert ":8000/" not in app.url
        with urllib.request.urlopen(app.url, timeout=5) as resp:
            assert resp.status == 200
            assert b"hi" in resp.read()
    finally:
        rt.stop(app)
    assert not rt.is_running(app)


def test_reap_orphans_spares_live_containers_but_removes_leaks(tmp_path) -> None:
    """Reaping must not kill an app another run is still using.

    This test previously asserted the opposite -- that a freshly started,
    running container was reaped -- which encoded a real bug: the sweep driver
    runs four simulations concurrently, so the first to finish would force-remove
    the other three's live app containers and their agents would suddenly be
    reviewing a dead port. A container now counts as an orphan only once it has
    stopped, or once it has been running far longer than any trial.
    """
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    manifest = Manifest(
        app_type="client-app", title="T", summary="S", run=RunSpec(command="sleep 120")
    )
    rt = _rt(app_dir)
    app = rt.start(manifest)
    try:
        assert rt.is_running(app)

        # A fresh, running container belongs to somebody -- leave it alone.
        reap_orphans()
        assert rt.is_running(app), "a live container was reaped out from under a run"

        # One running longer than any plausible trial is leaked.
        assert reap_orphans(running_grace_seconds=0) >= 1
        assert not rt.is_running(app)
    finally:
        rt.stop(app)  # idempotent, already reaped


class _BusyboxHarness:
    """Ships a web app served by busybox httpd, for the container crowd path."""

    def __init__(self, port: int) -> None:
        self.port = port

    def run(self, idea, workspace) -> HarnessResult:
        (workspace.app_dir / "index.html").write_text("<title>TileMerge</title>")
        (workspace.app_dir / "viralbench.json").write_text(
            json.dumps(
                {
                    "app_type": "client-app",
                    "title": "TileMerge",
                    "summary": "A sliding tile puzzle.",
                    "run": {"command": f"httpd -f -p {self.port}", "port": self.port},
                    "test": {"manual": ["play"], "smoke": "test -f index.html"},
                }
            )
        )
        return HarnessResult(
            model="fake",
            phases=[
                PhaseResult("design", 0, workspace.app_dir / "t", 1.0),
                PhaseResult("build", 0, workspace.app_dir / "t", 1.0),
            ],
        )


def test_crowd_path_verify_and_try_in_container(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    record = run_build("sliding_tile_game", harness=_BusyboxHarness(_free_port()))
    assert record.status == "ok"

    # Validity gate runs entirely inside an ephemeral container.
    result = verify_code(record.build_id, container=True, image=IMAGE)
    assert result.builds and result.runs and result.does_what_it_claims

    # Delight probe drives the shared container instance like a user (browser if
    # available, else a static-HTTP fallback). Either way it sees the rendered
    # title.
    host = AppHost(container=True, image=IMAGE)
    try:
        tried = try_app(record.build_id, host)
        assert tried.ok
        assert "TileMerge" in tried.observation
    finally:
        host.close()


def test_data_dir_is_mounted_and_shared_by_setup_and_run(tmp_path) -> None:
    """setup, smoke and start must all see ONE data directory.

    A one-shot (setup/smoke) runs in its own ephemeral ``--rm`` container. Before
    ``/data`` was mounted, a migration run in setup wrote into that container's
    throwaway layer, so the server started against an empty database -- the app
    looked broken and the cause was invisible.
    """
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    rt = ContainerRuntime(app_dir, image=IMAGE, data_dir=data_dir)
    manifest = Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        run=RunSpec(command="true"),
        setup=("echo migrated > /data/schema.txt",),
        test=_TestSpec(smoke="test -f /data/schema.txt"),
    )

    rt.setup(manifest)
    # Written through the mount, so it is visible on the host...
    assert (data_dir / "schema.txt").read_text().strip() == "migrated"
    # ...and to a LATER, separate one-shot container.
    smoke = rt.smoke(manifest)
    assert smoke is not None and smoke.returncode == 0


def test_data_dir_absent_by_default(tmp_path) -> None:
    """No data_dir means no /data mount -- the previous behaviour, unchanged."""
    rt = ContainerRuntime(tmp_path, image=IMAGE)
    assert not any("/data" in arg for arg in rt._common_args())
