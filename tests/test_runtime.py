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

"""Tests for the host app runtime (LocalRuntime)."""

from __future__ import annotations

import socket
import urllib.request

import pytest

from viral_bench.founder.manifest import Manifest, RunSpec
from viral_bench.founder.manifest import TestSpec as _TestSpec
from viral_bench.founder.runtime import (
    AppRuntimeError,
    ContainerRuntime,
    LocalRuntime,
    _wait_for_http,
    _wait_for_port,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_setup_runs_commands(tmp_path) -> None:
    rt = LocalRuntime(tmp_path)
    manifest = Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        run=RunSpec(command="true"),
        setup=("touch created.txt",),
    )
    rt.setup(manifest)
    assert (tmp_path / "created.txt").is_file()


def test_setup_failure_raises(tmp_path) -> None:
    rt = LocalRuntime(tmp_path)
    manifest = Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        run=RunSpec(command="true"),
        setup=("exit 3",),
    )
    with pytest.raises(AppRuntimeError, match="setup step failed"):
        rt.setup(manifest)


def test_smoke_pass_and_none(tmp_path) -> None:
    (tmp_path / "index.html").write_text("x")
    rt = LocalRuntime(tmp_path)
    passing = Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        run=RunSpec(command="true"),
        test=_TestSpec(smoke="test -f index.html"),
    )
    assert rt.smoke(passing).returncode == 0

    no_smoke = Manifest(
        app_type="client-app", title="T", summary="S", run=RunSpec(command="true")
    )
    assert rt.smoke(no_smoke) is None


def test_exec_returns_stdout(tmp_path) -> None:
    rt = LocalRuntime(tmp_path)
    proc = rt.exec("echo hello")
    assert proc.returncode == 0
    assert proc.stdout.strip() == "hello"


def test_start_stop_no_port(tmp_path) -> None:
    # A port-less manifest is a malformed web app (verify.py scores it
    # runs=False now), but the runtime itself still starts the process and
    # skips the readiness probe -- there is nothing to probe.
    rt = LocalRuntime(tmp_path)
    manifest = Manifest(
        app_type="client-app", title="T", summary="S", run=RunSpec(command="sleep 30")
    )
    app = rt.start(manifest)
    assert app.is_running()
    assert app.url is None
    app.stop()
    assert not app.is_running()


def test_start_web_app_serves(tmp_path) -> None:
    (tmp_path / "index.html").write_text("<title>hi</title>")
    port = _free_port()
    rt = LocalRuntime(tmp_path)
    manifest = Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        run=RunSpec(command=f"python3 -m http.server {port}", port=port),
    )
    app = rt.start(manifest, wait_timeout=10.0)
    try:
        assert app.url == f"http://localhost:{port}/"
        with urllib.request.urlopen(app.url, timeout=5) as resp:
            assert resp.status == 200
    finally:
        app.stop()
    assert not app.is_running()


# --- Container env-var injection (argv construction, no real container) ------


def test_container_forwards_allowlisted_env_when_set(tmp_path, monkeypatch) -> None:
    """A same-name allowlisted var present in the host env is passed via ``-e``."""
    monkeypatch.setenv("SOME_HOST_VAR", "secret-test-value")
    rt = ContainerRuntime(tmp_path, env_allowlist=("SOME_HOST_VAR",))
    args = rt._common_args()
    assert "-e" in args
    assert "SOME_HOST_VAR=secret-test-value" in args


def test_container_skips_allowlisted_env_when_unset(tmp_path, monkeypatch) -> None:
    """An allowlisted var that is not set is silently skipped (no leak, no -e)."""
    monkeypatch.delenv("SOME_HOST_VAR", raising=False)
    rt = ContainerRuntime(tmp_path, env_allowlist=("SOME_HOST_VAR",))
    args = rt._common_args()
    assert not any(a.startswith("SOME_HOST_VAR=") for a in args)


def test_container_injects_explicit_env(tmp_path) -> None:
    """Explicit env values (how the crowd path injects the mapped key) appear."""
    rt = ContainerRuntime(tmp_path, env={"GEMINI_API_KEY": "mapped-founder-value"})
    args = rt._common_args()
    assert "GEMINI_API_KEY=mapped-founder-value" in args


def test_container_runtime_defaults_are_neutral(tmp_path) -> None:
    """The low-level runtime stays generic, and the crowd entry points opt in."""
    rt = ContainerRuntime(tmp_path)
    assert rt.env_allowlist == ()
    assert rt.env == {}


# -- readiness probe -------------------------------------------------------
#
# Regression cover for the bug that let crowd triers review connection-reset
# pages as if they were the product: a TCP connect is not readiness evidence,
# because a rootless port forwarder (and a plain listening socket) accepts long
# before anything is serving HTTP.


def test_wait_for_http_rejects_a_tcp_only_listener() -> None:
    """The exact bug: TCP accepts, nothing serves. Must NOT report ready."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)  # accepts connections, never answers
        port = sock.getsockname()[1]

        assert _wait_for_port("127.0.0.1", port, timeout=1.0) is True

        ready, detail = _wait_for_http("127.0.0.1", port, timeout=1.0)
        assert ready is False
        assert "no HTTP response" in detail


def test_wait_for_http_accepts_any_http_status(tmp_path) -> None:
    """A 404 is a live server with an unhappy route, not a readiness failure."""
    import http.server
    import threading

    class _NotFound(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_error(404)

        def log_message(self, *a):  # keep test output clean
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _NotFound)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        ready, detail = _wait_for_http("127.0.0.1", server.server_port, timeout=5.0)
        assert ready is True
        assert "404" in detail
    finally:
        server.shutdown()


def test_local_runtime_refuses_to_hand_out_an_unreachable_url(tmp_path) -> None:
    """An app that binds a port but never serves must fail loudly, not silently.

    Previously this returned a RunningApp with a URL pointing at a dead server,
    and every trier that opened it wrote a confident review of an error page.
    """
    port = _free_port()
    rt = LocalRuntime(tmp_path)
    manifest = Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        # Binds the port and holds it open without ever speaking HTTP.
        run=RunSpec(
            command=(
                'python3 -c "import socket,time;'
                "s=socket.socket();s.setsockopt(socket.SOL_SOCKET,"
                "socket.SO_REUSEADDR,1);"
                f"s.bind(('127.0.0.1',{port}));s.listen(1);time.sleep(30)\""
            ),
            port=port,
        ),
    )
    with pytest.raises(AppRuntimeError, match="never became reachable"):
        rt.start(manifest, wait_timeout=2.0)


def test_local_runtime_refuses_to_adopt_a_server_it_did_not_start(tmp_path) -> None:
    """The readiness probe must not accept a stranger already on the port.

    Agents leak app servers, and 74 of 126 shipped manifests declare port 8000.
    Without this check the probe takes its HTTP 200 from the leftover server, the
    URL is handed out, and QA (or a crowd trier) reviews the wrong application
    while every other signal looks perfect.
    """
    import http.server
    import threading

    port = _free_port()
    squatter = http.server.HTTPServer(
        ("127.0.0.1", port), http.server.SimpleHTTPRequestHandler
    )
    threading.Thread(target=squatter.serve_forever, daemon=True).start()
    try:
        rt = LocalRuntime(tmp_path)
        manifest = Manifest(
            app_type="client-app",
            title="T",
            summary="S",
            # Would look healthy: it exits at once, so only the squatter answers.
            run=RunSpec(command="true", port=port),
        )
        with pytest.raises(AppRuntimeError, match="already in use"):
            rt.start(manifest, wait_timeout=2.0)
    finally:
        squatter.shutdown()
        squatter.server_close()


def test_local_runtime_records_a_launcher_that_backgrounded_itself(tmp_path) -> None:
    """A self-backgrounding run.command is legitimate: serve it, but say so.

    ``python3 -m http.server &`` exits the launcher shell immediately, so treating
    "launcher gone but the port answers" as a hijack would break real apps. It is
    recorded in ready_detail instead of raising.
    """
    import subprocess

    port = _free_port()
    (tmp_path / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    rt = LocalRuntime(tmp_path)
    manifest = Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        run=RunSpec(command=f"python3 -m http.server {port} &", port=port),
    )
    try:
        app = rt.start(manifest, wait_timeout=15.0)
        assert app.url is not None
        assert "launcher process exited" in app.ready_detail
    finally:
        # The launcher shell is gone, so the orphan cannot be killed by group.
        subprocess.run(f"fuser -k {port}/tcp", shell=True, capture_output=True)


def test_local_runtime_exports_data_dir(tmp_path) -> None:
    """Host runs get the data directory by env, since they have no mount ns.

    Apps read ``VIRALBENCH_DATA_DIR`` in both runtimes, so one line of app code
    works in a container (where it is /data) and on the host.
    """
    data_dir = tmp_path / "state"
    rt = LocalRuntime(tmp_path, data_dir=data_dir)
    manifest = Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        run=RunSpec(command="true"),
        setup=('printf "%s" "$VIRALBENCH_DATA_DIR" > seen.txt',),
    )
    rt.setup(manifest)
    assert (tmp_path / "seen.txt").read_text() == str(data_dir)


# -- both backends take extra env the same way -------------------------------- #
#
# The two store it under different names (LocalRuntime.extra_env is merged over
# the host env; ContainerRuntime.env becomes explicit -e flags), so a caller
# that touched an attribute directly worked on the host and raised
# AttributeError in a container. `serve-build --container` did exactly that and
# was dead on arrival while plain `serve-build` was fine.


def test_local_runtime_takes_extra_env(tmp_path) -> None:
    rt = LocalRuntime(tmp_path, env={"KEEP": "1"})
    rt.add_env({"PORT": "8123"})
    assert rt.extra_env == {"KEEP": "1", "PORT": "8123"}
    assert rt._env()["PORT"] == "8123"


def test_container_runtime_takes_extra_env(tmp_path) -> None:
    rt = ContainerRuntime(tmp_path, env={"KEEP": "1"})
    rt.add_env({"PORT": "8123"})
    assert rt.env == {"KEEP": "1", "PORT": "8123"}


def test_both_runtimes_satisfy_the_same_env_call(tmp_path) -> None:
    """What serve-build actually does, against both backends."""
    for rt in (LocalRuntime(tmp_path), ContainerRuntime(tmp_path)):
        rt.add_env({"PORT": "9001"})
