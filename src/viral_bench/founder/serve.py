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

"""Serve one built app locally, on a port you choose, with browser caching off.

Two problems make "run the app and open it" the wrong thing to hand a human
tester, and this module exists to solve both.

**Caching across builds.** Every build is a different app, but they are served
from the same handful of URLs: of the 25 apps in the current corpus, five serve
``/app.js`` and ``/style.css`` and three serve ``/static/app.js``. Several ship a
hand-rolled ``http.server`` that sends ``Last-Modified`` with no ``Cache-Control``
and no ``ETag``, which puts Chrome into *heuristic* caching -- roughly 10% of the
file's age, so a build a few days old is "fresh" for hours. Test build A then
build B on the same port and the browser reuses A's JS, CSS and even A's HTML
document, without ever asking the server. The result looks exactly like a broken
build: correct-ish markup, no styling, and dead buttons (markup wired with inline
``onclick="app.doThing()"`` throws ``app is not defined`` against the wrong
bundle). Diagnosing that as a model failure is a real and expensive mistake, so
:class:`NoCacheProxy` fronts the app and strips the whole revalidation story out
of every response.

**Port collisions.** 74 of 126 shipped manifests declare port 8000, so apps
cannot be run side by side as they are. :func:`rebind` moves an app onto a free
internal port, leaving the port *you* asked for to the proxy. Serve as many
builds at once as you like, each on its own port.

Typical use::

    uv run viral-bench serve-build <build_id> --port 8003

The app itself is untouched: it still runs under the normal
:class:`~viral_bench.founder.runner.AppSession` machinery, from a throwaway clone,
against its durable data dir. This is a *testing* front end, not a second runtime
-- the crowd path does not go through here (it gives every container its own
random host port, so it never had the caching problem in the first place).
"""

from __future__ import annotations

import http.client
import re
import socket
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from viral_bench.founder.manifest import Manifest

__all__ = [
    "DEFAULT_PORT",
    "NoCacheProxy",
    "free_port",
    "rebind",
]

#: Where a served app lands unless you say otherwise.
DEFAULT_PORT = 8000

# Hop-by-hop headers (RFC 7230 6.1): meaningful to one connection only, so they
# must not be relayed to the other side.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

# Dropped from the REQUEST. Without the conditional headers the app can never
# answer 304, which is what stops the browser reusing another build's body. The
# encoding header is dropped so only identity bytes are relayed and gzip framing
# never has to be reasoned about.
_DROP_REQUEST = _HOP_BY_HOP | {
    "accept-encoding",
    "if-match",
    "if-modified-since",
    "if-none-match",
    "if-range",
    "if-unmodified-since",
}

# Dropped from the RESPONSE: every header a browser could use to decide it
# already has this URL. Replaced wholesale by _NO_CACHE below.
_DROP_RESPONSE = _HOP_BY_HOP | {
    "age",
    "cache-control",
    "etag",
    "expires",
    "last-modified",
    "pragma",
}

_NO_CACHE = (
    ("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"),
    ("Pragma", "no-cache"),
    ("Expires", "0"),
)


def free_port() -> int:
    """Return a port that is free right now, chosen by the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def rebind(manifest: Manifest, port: int) -> Manifest:
    """Return a copy of ``manifest`` that runs the app on ``port``.

    Rewrites the declared port wherever it appears in the run command, the URL
    and the smoke command, matching on digit boundaries so ``8000`` in
    ``--port 8000`` moves but ``8000`` inside an unrelated number does not. Apps
    that instead read ``$PORT`` are handled by the caller exporting it. Between
    the two, every stack in the corpus lands on the right port.

    A manifest with no declared port (nothing to serve) is returned unchanged.
    """
    run = manifest.run
    if run.port is None or run.port == port:
        return manifest

    old = re.compile(rf"(?<!\d){run.port}(?!\d)")
    new = str(port)
    return replace(
        manifest,
        run=replace(
            run,
            command=old.sub(new, run.command),
            port=port,
            url=old.sub(new, run.url) if run.url else run.url,
        ),
        test=replace(
            manifest.test,
            smoke=old.sub(new, manifest.test.smoke) if manifest.test.smoke else None,
        ),
    )


class _Handler(BaseHTTPRequestHandler):
    """Relays one request upstream and strips caching from the response."""

    protocol_version = "HTTP/1.1"

    # Injected by NoCacheProxy via a subclass.
    upstream_host = "127.0.0.1"
    upstream_port = 0
    public_port = 0
    timeout_s = 300.0

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002 - stdlib signature
        # The app's own log is the interesting one, and this would double it.
        pass

    def _relay(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        headers = {
            k: v for k, v in self.headers.items() if k.lower() not in _DROP_REQUEST
        }
        headers["Accept-Encoding"] = "identity"
        headers["Connection"] = "close"

        try:
            conn = http.client.HTTPConnection(
                self.upstream_host, self.upstream_port, timeout=self.timeout_s
            )
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except OSError as exc:
            # The app died or never came up. Say so in the browser rather than
            # dropping the connection, which renders as an unexplained failure.
            self._fail(502, f"cannot reach the app on port {self.upstream_port}: {exc}")
            return

        try:
            self._send_response(resp)
        finally:
            conn.close()

    def _send_response(self, resp: http.client.HTTPResponse) -> None:
        declared = resp.getheader("Content-Length")
        self.send_response(resp.status, resp.reason)

        for key, value in resp.getheaders():
            if key.lower() in _DROP_RESPONSE or key.lower() == "content-length":
                continue
            if key.lower() == "location":
                value = self._rewrite_location(value)
            self.send_header(key, value)
        for key, value in _NO_CACHE:
            self.send_header(key, value)

        if declared is not None:
            self.send_header("Content-Length", declared)
            self.end_headers()
            self._pump(resp)
            return

        # No length upstream => a streaming response (an LLM call, SSE). Relaying
        # it verbatim and closing at the end frames it correctly with no
        # re-chunking, and keeps tokens flowing to the browser as they arrive.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self._pump(resp)

    def _pump(self, resp: http.client.HTTPResponse) -> None:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # Browser navigated away mid-response, so nothing to salvage.
                self.close_connection = True
                return

    def _rewrite_location(self, value: str) -> str:
        """Point redirects at the port the browser is talking to."""
        for host in ("localhost", "127.0.0.1", "0.0.0.0"):
            value = value.replace(
                f"http://{host}:{self.upstream_port}",
                f"http://{host}:{self.public_port}",
            )
        return value

    def _fail(self, status: int, message: str) -> None:
        payload = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in _NO_CACHE:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    # Every method a tester's browser (or curl) might use.
    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _relay
    do_HEAD = do_OPTIONS = _relay


class NoCacheProxy:
    """A threaded HTTP front end that forbids the browser to cache anything.

    Binds ``host:port`` and relays to ``upstream_port`` on loopback. Threaded on
    purpose: a browser opens several connections at once, and a single-threaded
    proxy would serialise every asset behind the slowest one.
    """

    def __init__(
        self,
        *,
        port: int = DEFAULT_PORT,
        upstream_port: int,
        host: str = "0.0.0.0",
        timeout_s: float = 300.0,
    ) -> None:
        self.host = host
        self.port = port
        self.upstream_port = upstream_port

        handler = type(
            "_BoundHandler",
            (_Handler,),
            {
                "upstream_port": upstream_port,
                "public_port": port,
                "timeout_s": timeout_s,
            },
        )
        self._server = ThreadingHTTPServer((host, port), handler)
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"

    def start(self) -> NoCacheProxy:
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="vb-serve-proxy", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> NoCacheProxy:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
