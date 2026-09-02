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

"""Run a built app so a human can use the thing the crowd agents used.

Watching a replay of an agent clicking through an app answers "what did it do".
Standing the app up next to the replay answers "was it right", and that is the
question a reviewer actually has. This module is a self-contained port of the
benchmark's ``serve-build`` command, carrying over the two problems it exists to
solve, because both bite immediately otherwise.

**Browser caching across builds.** Every build is a different app, but they are
served from the same handful of URLs -- most ship ``/app.js`` and ``/style.css``,
and many hand-roll an ``http.server`` that sends ``Last-Modified`` with no
``Cache-Control`` and no ``ETag``. That puts Chrome into heuristic caching, so
testing build A and then build B on the same port silently reuses A's JavaScript
against B's markup. It looks exactly like a broken build: right-ish layout, no
styling, dead buttons. :class:`NoCacheProxy` fronts the app and strips the whole
revalidation story out of every response.

**Port collisions.** Most shipped manifests declare port 8000, so apps cannot run
side by side as written. :func:`rebind_command` moves the app to a private port and
leaves the public one to the proxy, so any number of builds can be open at once.

The app runs from a throwaway copy under ``viz/cache/runs``; the original build
tree is never written to, and never executed in place.
"""

from __future__ import annotations

import http.client
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .paths import assert_writable, cache_dir, founder_paths, read_json

_NO_CACHE = (
    ("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"),
    ("Pragma", "no-cache"),
    ("Expires", "0"),
)
_DROP_REQUEST = {
    "host",
    "accept-encoding",
    "connection",
    "keep-alive",
    "proxy-connection",
}
_DROP_RESPONSE = {
    "transfer-encoding",
    "connection",
    "keep-alive",
    "content-encoding",
    "etag",
    "last-modified",
    "cache-control",
    "pragma",
    "expires",
    "age",
    # send_response() already emits Server and Date. Relaying the upstream's copies
    # as well produces a response carrying two of each, which Chrome treats as
    # malformed and renders as an empty frame -- indistinguishable, to whoever is
    # reviewing, from a build that does not work.
    "server",
    "date",
    # The viewer shows the app in a pane beside its replay, and the app is served
    # on its own port, so any framing restriction the app declares would block
    # exactly the thing this proxy exists to enable. Dropped here rather than
    # rewritten; the app is a throwaway local copy on loopback.
    "x-frame-options",
}

SETUP_TIMEOUT = 600.0
START_TIMEOUT = 90.0


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def rebind_command(command: str, declared: int | None, port: int) -> str:
    """Rewrite a declared port inside a run command.

    Matched on digit boundaries so ``8000`` in ``--port 8000`` moves while ``8000``
    inside an unrelated number does not. Apps that read ``$PORT`` instead are
    covered by the caller exporting it; between the two, every stack in the corpus
    lands on the right port.
    """
    if not declared or declared == port:
        return command
    return re.sub(rf"(?<!\d){declared}(?!\d)", str(port), command)


# --------------------------------------------------------------------------
# The proxy
# --------------------------------------------------------------------------


def _strip_frame_ancestors(policy: str) -> str:
    """Remove a ``frame-ancestors`` directive from a CSP, keeping the rest."""
    kept = [
        part.strip()
        for part in policy.split(";")
        if part.strip() and not part.strip().lower().startswith("frame-ancestors")
    ]
    return "; ".join(kept)


class _Handler(BaseHTTPRequestHandler):
    """Relay one request upstream and strip caching from the response."""

    protocol_version = "HTTP/1.1"
    upstream_port = 0
    public_port = 0
    timeout_s = 300.0

    def log_message(self, fmt, *args) -> None:  # noqa: A002 - stdlib signature
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
                "127.0.0.1", self.upstream_port, timeout=self.timeout_s
            )
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
        except OSError as exc:
            self._fail(502, f"cannot reach the app on port {self.upstream_port}: {exc}")
            return
        try:
            self._send(response)
        finally:
            conn.close()

    def _send(self, response) -> None:
        declared = response.getheader("Content-Length")
        self.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            low = key.lower()
            if low in _DROP_RESPONSE or low == "content-length":
                continue
            if low == "location":
                for host in ("localhost", "127.0.0.1", "0.0.0.0"):
                    value = value.replace(
                        f"http://{host}:{self.upstream_port}",
                        f"http://{host}:{self.public_port}",
                    )
            if low == "content-security-policy":
                # Same reason as x-frame-options: a frame-ancestors directive
                # would stop the app rendering in the viewer's pane. Only that
                # directive is removed; the rest of the policy still applies.
                value = _strip_frame_ancestors(value)
                if not value:
                    continue
            self.send_header(key, value)
        for key, value in _NO_CACHE:
            self.send_header(key, value)
        if declared is not None:
            self.send_header("Content-Length", declared)
            self.end_headers()
        else:
            # No upstream length means a streaming response; relay verbatim and
            # close at the end rather than re-chunking it.
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
        self._pump(response)

    def _pump(self, response) -> None:
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
                return

    def _fail(self, status: int, message: str) -> None:
        payload = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in _NO_CACHE:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = _relay
    do_HEAD = do_OPTIONS = _relay


class NoCacheProxy:
    """Threaded front end that forbids the browser to cache anything.

    Threaded because a browser opens several connections at once; single-threaded
    would serialise every asset behind the slowest request.
    """

    def __init__(self, *, port: int, upstream_port: int):
        handler = type(
            "BoundHandler",
            (_Handler,),
            {"upstream_port": upstream_port, "public_port": port},
        )
        self._server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self._server.daemon_threads = True
        self.port = port
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------


@dataclass
class AppInstance:
    """One running build, addressable at ``url``."""

    build_id: str
    session_id: str
    run_dir: Path
    app_dir: Path
    public_port: int
    internal_port: int
    url: str
    command: str
    app_type: str
    title: str
    started_at: float
    #: Set when the app ignored the port we gave it and bound its own.
    claimed_port: int | None = None
    process: subprocess.Popen | None = None
    proxy: NoCacheProxy | None = None
    log_path: Path | None = None
    setup_log: list[str] = field(default_factory=list)

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "build_id": self.build_id,
            "url": self.url,
            "public_port": self.public_port,
            "internal_port": self.internal_port,
            "command": self.command,
            "app_type": self.app_type,
            "title": self.title,
            "alive": self.alive(),
            "uptime_s": round(time.time() - self.started_at, 1),
            "app_dir": str(self.app_dir),
            "log": str(self.log_path) if self.log_path else None,
            "setup_log": self.setup_log[-40:],
        }


class AppLauncher:
    """Starts, tracks and stops built apps for interactive testing."""

    def __init__(self, builds_root: Path):
        self.builds_root = builds_root
        self._instances: dict[str, AppInstance] = {}
        #: Ports held by apps that hard-code them and cannot be rebound.
        self._claimed_ports: set[int] = set()
        self._lock = threading.Lock()

    def list(self) -> list[dict]:
        with self._lock:
            return [i.as_dict() for i in self._instances.values()]

    def get(self, session_id: str) -> AppInstance | None:
        with self._lock:
            return self._instances.get(session_id)

    def for_build(self, build_id: str) -> AppInstance | None:
        with self._lock:
            for instance in self._instances.values():
                if instance.build_id == build_id and instance.alive():
                    return instance
        return None

    def launch(
        self, build_id: str, *, run_setup: bool = True, port: int | None = None
    ) -> dict:
        """Materialise, set up and start one build. Returns a status dict.

        ``port`` pins the public port instead of taking whatever is free, which is
        what makes an app reachable over a fixed tunnel. The app's own port stays
        arbitrary either way -- only the proxy in front of it is pinned.

        Never raises for an app-side failure: a build that will not start is a
        finding, not a crash, and the log belongs in front of the user.
        """
        existing = self.for_build(build_id)
        if existing and (port is None or existing.public_port == port):
            return {"ok": True, "reused": True, **existing.as_dict()}
        if existing:
            self.stop(existing.session_id)
        if port is not None and port_open(port):
            return {
                "ok": False,
                "error": f"port {port} is already in use by something else",
            }

        paths = founder_paths(self.builds_root, build_id)
        if not paths.app_dir.is_dir():
            return {"ok": False, "error": f"no app directory for {build_id}"}
        manifest = read_json(paths.app_dir / "viralbench.json")
        if not isinstance(manifest, dict):
            return {
                "ok": False,
                "error": f"{build_id} has no valid viralbench.json (undeliverable)",
            }

        run = manifest.get("run") or {}
        command = str(run.get("command") or "").strip()
        app_type = str(manifest.get("app_type") or "")
        if not command:
            return {
                "ok": False,
                "error": f"{build_id} declares no run command (app_type={app_type})",
            }

        session_id = f"{build_id}__{uuid.uuid4().hex[:8]}"
        run_dir = cache_dir("runs", session_id)
        app_dst = run_dir / "app"
        try:
            # Copy rather than run in place: the build tree is read-only to us, and
            # an app that writes next to its source would dirty it.
            shutil.copytree(paths.app_dir, app_dst, symlinks=True, dirs_exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": f"could not materialise app: {exc}"}

        declared = run.get("port") if isinstance(run.get("port"), int) else None
        internal = free_port()
        public = port if port is not None else free_port()
        cwd = app_dst / (run.get("cwd") or ".")
        bound = rebind_command(command, declared, internal)

        # Whether an app honours the port we give it cannot be read off its run
        # command: the port may be rewritten in the command, taken from $PORT
        # inside the app's own source, or hard-coded and ignored. So do not
        # guess -- start it, then see which port it actually opened.
        #
        # `declared` is only accepted as that port when no other live instance is
        # already using it. Two apps that both hard-code 8000 would otherwise
        # both look healthy while the second is proxied to the first one's
        # server, which presents as a build that shipped someone else's app.
        with self._lock:
            declared_free = (
                declared is not None
                and declared not in self._claimed_ports
                and not port_open(declared)
            )

        env = dict(os.environ)
        env["PORT"] = str(internal)
        # State the app owns lives outside the throwaway copy, so a restart does
        # not silently wipe a database the user just populated.
        data_dir = cache_dir("appdata", build_id)
        env["VIRALBENCH_DATA_DIR"] = str(data_dir)

        log_path = assert_writable(run_dir / "app.log")
        setup_log: list[str] = []
        if run_setup:
            for step in manifest.get("setup") or []:
                step = str(step).strip()
                if not step:
                    continue
                setup_log.append(f"$ {step}")
                try:
                    proc = subprocess.run(
                        step,
                        shell=True,
                        cwd=cwd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=SETUP_TIMEOUT,
                    )
                    tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-1500:]
                    setup_log.append(f"-> exit {proc.returncode}")
                    if tail.strip():
                        setup_log.extend(tail.strip().splitlines()[-20:])
                except (subprocess.SubprocessError, OSError) as exc:
                    # A failed setup step is worth reporting but not fatal: most
                    # builds ship a prebuilt dist/ and start fine without it.
                    setup_log.append(f"-> failed: {exc}")

        try:
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                bound,
                shell=True,
                cwd=cwd,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            return {
                "ok": False,
                "error": f"could not start app: {exc}",
                "setup_log": setup_log,
            }

        # Wait for whichever port it really opened: the one we asked for, or the
        # one it insisted on.
        candidates = [internal] + ([declared] if declared_free else [])
        deadline = time.time() + START_TIMEOUT
        while time.time() < deadline:
            opened = next((c for c in candidates if port_open(c)), None)
            if opened is not None:
                internal = opened
                if opened == declared:
                    with self._lock:
                        self._claimed_ports.add(declared)
                break
            if process.poll() is not None:
                return {
                    "ok": False,
                    "error": (
                        f"the app exited with code {process.returncode} before serving"
                    ),
                    "setup_log": setup_log,
                    "log": _tail(log_path),
                }
            time.sleep(0.25)
        else:
            process.terminate()
            return {
                "ok": False,
                "error": (
                    f"the app did not open port {internal} within {START_TIMEOUT:.0f}s"
                ),
                "setup_log": setup_log,
                "log": _tail(log_path),
            }

        try:
            proxy = NoCacheProxy(port=public, upstream_port=internal)
            proxy.start()
        except OSError as exc:
            _kill_process(process)
            return {"ok": False, "error": f"could not bind port {public}: {exc}"}

        instance = AppInstance(
            build_id=build_id,
            session_id=session_id,
            run_dir=run_dir,
            app_dir=app_dst,
            public_port=public,
            internal_port=internal,
            url=f"http://localhost:{public}/",
            command=bound,
            app_type=app_type,
            title=str(manifest.get("title") or build_id),
            started_at=time.time(),
            claimed_port=declared if internal == declared else None,
            process=process,
            proxy=proxy,
            log_path=log_path,
            setup_log=setup_log,
        )
        with self._lock:
            self._instances[session_id] = instance
        return {"ok": True, "reused": False, **instance.as_dict()}

    def stop(self, session_id: str) -> dict:
        with self._lock:
            instance = self._instances.pop(session_id, None)
        if instance is None:
            return {"ok": False, "error": "no such session"}
        if instance.claimed_port is not None:
            with self._lock:
                self._claimed_ports.discard(instance.claimed_port)
        _shutdown(instance)
        return {"ok": True, "session_id": session_id}

    def stop_all(self) -> None:
        with self._lock:
            instances = list(self._instances.values())
            self._instances.clear()
            self._claimed_ports.clear()
        for instance in instances:
            _shutdown(instance)


def _shutdown(instance: AppInstance) -> None:
    if instance.proxy:
        try:
            instance.proxy.stop()
        except OSError:
            pass
    _kill_process(instance.process)
    shutil.rmtree(instance.run_dir, ignore_errors=True)


def _kill_process(process: subprocess.Popen | None) -> None:
    if process and process.poll() is None:
        try:
            # The app was started in its own session so a shell wrapper's children
            # die with it; killing only the shell would leave the server holding
            # the port.
            os.killpg(os.getpgid(process.pid), 15)
        except (ProcessLookupError, PermissionError, OSError):
            process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), 9)
            except (ProcessLookupError, PermissionError, OSError):
                process.kill()


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""
