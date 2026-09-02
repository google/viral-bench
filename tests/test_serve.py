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

"""Tests for the local serving front end (``viral-bench serve-build``).

The behaviour under test is the one that cost real debugging time: a browser must
not be able to reuse one build's assets when the next build is served on the same
port. That means (a) every response says no-store, (b) conditional requests never
reach the app so it can never answer 304, and (c) apps can be moved off the port
they hard-code so several can run at once.
"""

from __future__ import annotations

import http.client
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from viral_bench.founder.manifest import Manifest, RunSpec
from viral_bench.founder.manifest import TestSpec as _TestSpec  # not a test class
from viral_bench.founder.serve import NoCacheProxy, free_port, rebind

# --------------------------------------------------------------------------- #
# rebind
# --------------------------------------------------------------------------- #


def _manifest(**run_kwargs) -> Manifest:
    run = RunSpec(
        command="uvicorn main:app --host 0.0.0.0 --port 8000",
        port=8000,
        url="http://localhost:8000/",
        **run_kwargs,
    )
    return Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        run=run,
        test=_TestSpec(smoke="curl -fsS http://localhost:8000/healthz"),
    )


def test_rebind_moves_command_url_and_smoke():
    out = rebind(_manifest(), 9123)
    assert out.run.port == 9123
    assert out.run.command == "uvicorn main:app --host 0.0.0.0 --port 9123"
    assert out.run.url == "http://localhost:9123/"
    assert out.test.smoke == "curl -fsS http://localhost:9123/healthz"


def test_rebind_leaves_unrelated_numbers_alone():
    """Digit-boundary matching: 8000 moves, 18000 and 80001 do not."""
    m = Manifest(
        app_type="client-app",
        title="T",
        summary="S",
        run=RunSpec(command="serve --mem 18000 --port 8000 --id 80001", port=8000),
    )
    out = rebind(m, 9999)
    assert out.run.command == "serve --mem 18000 --port 9999 --id 80001"


def test_rebind_is_a_noop_without_a_port():
    m = Manifest(
        app_type="client-app", title="T", summary="S", run=RunSpec(command="./run.sh")
    )
    assert rebind(m, 9123) is m


def test_rebind_does_not_mutate_the_original():
    original = _manifest()
    rebind(original, 9123)
    assert original.run.port == 8000
    assert original.run.command.endswith("8000")


def test_free_port_is_bindable():
    import socket

    port = free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))  # must not raise


# --------------------------------------------------------------------------- #
# NoCacheProxy
# --------------------------------------------------------------------------- #


class _Upstream(BaseHTTPRequestHandler):
    """A stand-in app that caches aggressively and honours conditional GETs."""

    protocol_version = "HTTP/1.1"
    seen_headers: list[dict[str, str]] = []

    def log_message(self, *a):
        pass

    def _record(self):
        type(self).seen_headers.append({k.lower(): v for k, v in self.headers.items()})

    def do_GET(self):
        self._record()
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header(
                "Location", f"http://localhost:{self.server.server_address[1]}/moved"
            )
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/stream":
            # No Content-Length: the streaming shape.
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"chunk{i}\n".encode())
                self.wfile.flush()
            self.close_connection = True
            return
        if self.headers.get("If-None-Match") == '"v1"':
            self.send_response(304)
            self.send_header("ETag", '"v1"')
            self.end_headers()
            return
        body = b"hello from upstream"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=31536000")
        self.send_header("ETag", '"v1"')
        self.send_header("Last-Modified", "Wed, 01 Jan 2020 00:00:00 GMT")
        self.send_header("Expires", "Thu, 01 Jan 2099 00:00:00 GMT")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self._record()
        length = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(length)
        body = json.dumps({"echo": payload.decode()}).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def proxied():
    _Upstream.seen_headers = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    upstream.daemon_threads = True
    up_port = upstream.server_address[1]
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    proxy = NoCacheProxy(
        port=free_port(), upstream_port=up_port, host="127.0.0.1"
    ).start()
    try:
        yield proxy, up_port
    finally:
        proxy.stop()
        upstream.shutdown()
        upstream.server_close()


def _get(proxy: NoCacheProxy, path="/", headers=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy.port}{path}", headers=headers or {}
    )
    return urllib.request.urlopen(req, timeout=10)


def test_proxy_relays_the_body(proxied):
    proxy, _ = proxied
    assert _get(proxy).read() == b"hello from upstream"


def test_proxy_replaces_every_caching_header(proxied):
    """The whole point: nothing the browser could cache on survives."""
    proxy, _ = proxied
    resp = _get(proxy)
    assert (
        resp.headers["Cache-Control"]
        == "no-store, no-cache, must-revalidate, max-age=0"
    )
    assert resp.headers["Pragma"] == "no-cache"
    assert resp.headers["Expires"] == "0"
    assert resp.headers.get("ETag") is None
    assert resp.headers.get("Last-Modified") is None


def test_proxy_strips_conditional_headers_so_upstream_cannot_304(proxied):
    """A browser sending If-None-Match must still get a full 200 body."""
    proxy, _ = proxied
    resp = _get(proxy, headers={"If-None-Match": '"v1"'})
    assert resp.status == 200
    assert resp.read() == b"hello from upstream"
    assert "if-none-match" not in _Upstream.seen_headers[-1]


def test_proxy_relays_post_bodies(proxied):
    proxy, _ = proxied
    req = urllib.request.Request(
        f"http://127.0.0.1:{proxy.port}/api",
        data=b"payload",
        headers={"Content-Type": "text/plain"},
    )
    resp = urllib.request.urlopen(req, timeout=10)
    assert resp.status == 201
    assert json.loads(resp.read())["echo"] == "payload"


def test_proxy_rewrites_redirects_to_the_public_port(proxied):
    proxy, up_port = proxied
    conn = http.client.HTTPConnection("127.0.0.1", proxy.port, timeout=10)
    conn.request("GET", "/redirect")
    resp = conn.getresponse()
    assert resp.status == 302
    assert resp.getheader("Location") == f"http://localhost:{proxy.port}/moved"
    assert str(up_port) not in resp.getheader("Location")
    conn.close()


def test_proxy_relays_a_streaming_response(proxied):
    proxy, _ = proxied
    assert _get(proxy, "/stream").read() == b"chunk0\nchunk1\nchunk2\n"


def test_proxy_reports_a_dead_app_as_502(proxied):
    """A build that died must render an explanation, not a dropped connection."""
    proxy, _ = proxied
    dead = NoCacheProxy(
        port=free_port(), upstream_port=free_port(), host="127.0.0.1"
    ).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get(dead)
        assert excinfo.value.code == 502
        assert b"cannot reach the app" in excinfo.value.read()
    finally:
        dead.stop()


def test_proxy_serves_concurrent_requests(proxied):
    """A browser opens several connections at once, and one must not block the rest."""
    proxy, _ = proxied
    results: list[bytes] = []
    lock = threading.Lock()

    def hit():
        body = _get(proxy).read()
        with lock:
            results.append(body)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert results == [b"hello from upstream"] * 8
