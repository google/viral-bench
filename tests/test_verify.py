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

"""Tests for the crowd-facing verify_code / try_app checks (host mode)."""

from __future__ import annotations

import json
import socket

import pytest

from viral_bench.founder.apphost import AppHost
from viral_bench.founder.build import run_build
from viral_bench.founder.harness import HarnessResult, PhaseResult
from viral_bench.founder.verify import try_app, verify_code


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _web_manifest(port: int, *, setup=(), smoke="test -f index.html") -> dict:
    return {
        "app_type": "client-app",
        "title": "TileMerge",
        "summary": "A sliding tile puzzle.",
        "setup": list(setup),
        "run": {
            "command": f"python3 -m http.server {port}",
            "port": port,
            "url": f"http://localhost:{port}/",
        },
        "test": {"manual": ["play"], "smoke": smoke},
    }


def _status_manifest(port: int, status: int) -> dict:
    """A manifest whose server answers `status` on every path.

    Used to check what the gate does with an app that starts and then serves
    errors -- the shape that slipped through as a healthy build.
    """
    code = (
        "import http.server as h;"
        f"C={status};"
        "H=type('H',(h.BaseHTTPRequestHandler,),{"
        "'do_GET':lambda s:(s.send_response(C),s.send_header('Content-Length','2'),"
        "s.end_headers(),s.wfile.write(b'hi')),"
        "'log_message':lambda s,*a:None});"
        f"h.ThreadingHTTPServer(('0.0.0.0',{port}),H).serve_forever()"
    )
    return {
        "app_type": "client-app",
        "title": "TileMerge",
        "summary": "A sliding tile puzzle.",
        "setup": [],
        "run": {
            "command": f'python3 -c "{code}"',
            "port": port,
            "url": f"http://localhost:{port}/",
        },
        "test": {"manual": ["play"], "smoke": "test -f index.html"},
    }


class ManifestHarness:
    """Fake harness that ships a caller-supplied manifest + index.html."""

    def __init__(self, manifest: dict, *, index: bool = True) -> None:
        self.manifest = manifest
        self.index = index

    def run(self, idea, workspace) -> HarnessResult:
        if self.index:
            (workspace.app_dir / "index.html").write_text("<title>TileMerge</title>")
        (workspace.app_dir / "viralbench.json").write_text(json.dumps(self.manifest))
        return HarnessResult(
            model="fake",
            phases=[
                PhaseResult("design", 0, workspace.app_dir / "t", 1.0),
                PhaseResult("build", 0, workspace.app_dir / "t", 1.0),
            ],
        )


@pytest.fixture
def builds_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    return tmp_path


def _make_build(port: int, **kw) -> str:
    harness = ManifestHarness(_web_manifest(port, **kw))
    record = run_build("sliding_tile_game", harness=harness)
    assert record.status == "ok"
    return record.build_id


def test_verify_code_healthy_app(builds_dir) -> None:
    build_id = _make_build(_free_port())
    result = verify_code(build_id, container=False)
    assert result.builds is True
    assert result.runs is True
    assert result.does_what_it_claims is True


def test_verify_code_flags_failed_setup(builds_dir) -> None:
    build_id = _make_build(_free_port(), setup=("exit 7",))
    result = verify_code(build_id, container=False)
    assert result.builds is False
    assert result.runs is False
    assert result.does_what_it_claims is False


def test_verify_code_flags_failed_smoke(builds_dir) -> None:
    # Smoke references a file that does not exist -> unhealthy.
    build_id = _make_build(_free_port(), smoke="test -f nope.txt")
    result = verify_code(build_id, container=False)
    assert result.builds is True
    assert result.does_what_it_claims is False


def test_smoke_may_probe_the_running_app(builds_dir) -> None:
    """A smoke command that talks to the app's own port must be able to pass.

    The gate used to run smoke BEFORE starting the app, so this class of health
    check, the only kind that proves the app serves, failed by
    construction. Measured over the stored corpus, network-dependent smoke
    commands passed 0 times out of 45 while local ones passed 372/450.
    """
    port = _free_port()
    build_id = _make_build(
        port,
        smoke=f"python3 -c \"import urllib.request as u; u.urlopen('http://localhost:{port}/')\"",
    )
    result = verify_code(build_id, container=False)
    assert result.runs is True
    assert result.does_what_it_claims is True, result.detail


def test_app_state_survives_a_restart(builds_dir) -> None:
    """State written to the data dir outlives the throwaway run directory.

    The crowd shares one instance per build and re-materializes a fresh clone
    whenever it dies, so a database kept under the app dir is wiped mid-run.
    """
    from viral_bench.founder.runner import open_session
    from viral_bench.founder.workspace import build_data_dir, reset_build_data

    build_id = _make_build(_free_port())
    reset_build_data(build_id)

    session = open_session(build_id, container=False)
    assert session.data_dir is not None
    (session.data_dir / "app.db").write_text("rows")
    run_dir = session.run_dir
    session.close()

    assert not run_dir.exists(), "the run dir is disposable and should be gone"
    assert (build_data_dir(build_id) / "app.db").read_text() == "rows"

    # A fresh session sees the same state.
    session2 = open_session(build_id, container=False)
    try:
        assert (session2.data_dir / "app.db").read_text() == "rows"
    finally:
        session2.close()

    # ...and a new crowd run starts clean.
    reset_build_data(build_id)
    assert not (build_data_dir(build_id) / "app.db").exists()


def test_try_app_uses_running_web_app(builds_dir) -> None:
    # try_app now drives the app like a user (real browser when available, else a
    # static-HTTP fallback). Either way the rendered title is observed.
    build_id = _make_build(_free_port())
    host = AppHost(container=False)
    try:
        result = try_app(build_id, host)
        assert result.ok is True
        assert "TileMerge" in result.observation
        assert result.url is not None
    finally:
        host.close()


def test_http_probe_budget_is_wall_clock_not_attempt_count() -> None:
    """The readiness probe must outlast a slow port-forwarder.

    It used to retry 12 times 0.3s apart. That sounds generous but a refused
    connection fails instantly, so the real budget was ~3.6 seconds -- short
    enough to lose the race under a concurrent sweep. This probe alone decides
    `runs`, which gates the score: on the stored corpus one such false negative
    moved a build from 55.2 to 13.2 between two seeds of the SAME app, while the
    crowd reached that app in 8 of 8 trials and the in-container smoke passed.
    """
    import time as _time

    from viral_bench.founder.verify import _http_probe

    # Nothing is listening, so every attempt fails fast, but the probe must still
    # spend its wall-clock budget rather than returning almost immediately.
    port = _free_port()
    started = _time.monotonic()
    responded, status, detail = _http_probe(
        f"http://127.0.0.1:{port}/", deadline_s=2.0, delay=0.05
    )
    elapsed = _time.monotonic() - started

    assert responded is False and status is None
    assert elapsed >= 1.5, f"gave up after {elapsed:.2f}s of a 2s budget"
    assert elapsed < 8.0, f"overshot the budget badly ({elapsed:.2f}s)"
    assert "2s" in detail or "error" in detail


def test_http_probe_returns_as_soon_as_the_app_answers() -> None:
    """Patience must not cost latency when the app is already up."""
    import http.server
    import socketserver
    import threading
    import time as _time

    from viral_bench.founder.verify import _http_probe

    class _H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    srv = socketserver.TCPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        started = _time.monotonic()
        responded, status, _ = _http_probe(
            f"http://127.0.0.1:{srv.server_address[1]}/", deadline_s=30.0
        )
        elapsed = _time.monotonic() - started
        assert responded is True and status == 200
        assert elapsed < 2.0, f"a healthy app took {elapsed:.2f}s to verify"
    finally:
        srv.shutdown()


def test_a_500_on_the_entry_page_is_not_a_running_app(builds_dir) -> None:
    """An app whose landing page errors does not work, however healthy its /healthz.

    Found by the crowd, not the gate. One build served HTTP 500 on `/` while
    `/healthz` returned 200, so the gate passed it `runs=True` AND
    `does_what_it_claims=True` -- and its smoke test passed too, because the
    model had pointed the smoke command at /healthz. Three agents then wrote
    glowing 7/10 reviews of it without interacting at all. Six apps in a single
    25-build sweep were serving errors the gate could not see.
    """
    port = _free_port()
    harness = ManifestHarness(_status_manifest(port, 500))
    build_id = run_build("sliding_tile_game", harness=harness).build_id
    result = verify_code(build_id, container=False)
    assert result.runs is False
    assert result.does_what_it_claims is False
    assert "500" in result.detail


def test_a_401_landing_page_still_counts_as_running(builds_dir) -> None:
    """Auth-first is a legitimate design, so 4xx must NOT fail the gate.

    A full-stack app may reasonably answer 401 at `/` and let the crowd sign up.
    Only 5xx is never a design decision.
    """
    port = _free_port()
    harness = ManifestHarness(_status_manifest(port, 401))
    build_id = run_build("sliding_tile_game", harness=harness).build_id
    result = verify_code(build_id, container=False)
    assert result.runs is True
