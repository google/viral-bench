#!/usr/bin/env python3
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

"""ViralBench trajectory viewer -- one server, two views.

    python3 viz/serve.py                # both UIs on http://localhost:8770
    python3 viz/serve.py --open founder # ...and open one of them in a browser

``/founder`` replays a founder build: every turn, tool call and file edit on a
timeline you can scrub, for all three run structures (single agent, 4-agent local,
dynamic orchestrator).

``/crowd`` replays a crowd simulation: the social graph forming, the launch post
spreading, and each agent's hands-on trial of the app -- with the real screenshots,
and a button to run the app yourself.

Deliberately dependency-free. This viewer reads a live benchmark checkout whose
Python environment changes underneath it, so it uses nothing but the standard
library and vanilla JavaScript: no framework, no build step, no lockfile to keep in
sync. The one hard rule is enforced in ``core.paths.assert_writable`` -- the builds
tree is read-only, and everything this process writes lands in ``viz/cache/``.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import crowd as crowd_mod  # noqa: E402
from core import export as export_mod  # noqa: E402
from core import founder as founder_mod  # noqa: E402
from core import rubric as rubric_mod  # noqa: E402
from core.apphost import AppLauncher  # noqa: E402
from core.cache import rescue_run, resolve_shot  # noqa: E402
from core.paths import (  # noqa: E402
    crowd_run_dir,
    default_builds_root,
    founder_paths,
    index_builds,
    index_crowd_runs,
    index_rubric_grades,
    rubric_run_dir,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 8770

#: Build ids and run ids are user input that becomes a path segment. They are
#: generated as ``<idea>__<utc>__<hex>`` (plus ``__crowd-...`` suffixes), so this
#: character class covers every real one and excludes anything that could climb
#: out of the builds tree.
SAFE_ID = re.compile(r"^[A-Za-z0-9._\-]+$")


def safe_id(value: str) -> str | None:
    value = unquote(value or "")
    return value if value and SAFE_ID.match(value) and ".." not in value else None


class Store:
    """Indexes and parsed trajectories, cached and invalidated by file mtime."""

    def __init__(self, builds_root: Path):
        self.builds_root = builds_root
        self._lock = threading.Lock()
        self._builds: list[dict] | None = None
        self._builds_at = 0.0
        self._crowd: list[dict] | None = None
        self._crowd_at = 0.0
        self._trajectories: dict[str, tuple[str, dict]] = {}
        self._runs: dict[str, tuple[str, dict]] = {}
        self._rubric: list[dict] | None = None
        self._rubric_at = 0.0
        self._grades: dict[str, tuple[str, dict]] = {}
        self.build_ttl = 60.0
        self.crowd_ttl = 300.0
        # Shorter than the crowd's: grades arrive in bursts during a sweep, and
        # the whole point of watching one is seeing results land.
        self.rubric_ttl = 120.0

    def builds(self, refresh: bool = False) -> list[dict]:
        with self._lock:
            stale = (
                refresh
                or self._builds is None
                or time.time() - self._builds_at > self.build_ttl
            )
        if stale:
            rows = [b.as_dict() for b in index_builds(self.builds_root)]
            with self._lock:
                self._builds, self._builds_at = rows, time.time()
        with self._lock:
            return self._builds or []

    def crowd_runs(self, refresh: bool = False) -> list[dict]:
        with self._lock:
            stale = (
                refresh
                or self._crowd is None
                or time.time() - self._crowd_at > self.crowd_ttl
            )
        if stale:
            rows = index_crowd_runs(self.builds_root)
            with self._lock:
                self._crowd, self._crowd_at = rows, time.time()
        with self._lock:
            return self._crowd or []

    def _signature(self, paths) -> str:
        """Cheap fingerprint of a build's inputs, so a re-run invalidates the cache."""
        parts = []
        for path in [paths.build_json, *sorted(paths.transcript_dir.glob("*.json"))]:
            try:
                stat = path.stat()
                parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
            except OSError:
                continue
        return "|".join(parts)

    def trajectory(self, build_id: str) -> dict | None:
        paths = founder_paths(self.builds_root, build_id)
        signature = self._signature(paths)
        cached = self._trajectories.get(build_id)
        if cached and cached[0] == signature:
            return cached[1]
        data = founder_mod.load_trajectory(self.builds_root, build_id)
        if data is not None:
            # Bounded so a long browsing session cannot grow without limit. The
            # payloads are ~0.2-1.4 MB each.
            if len(self._trajectories) > 24:
                self._trajectories.clear()
            self._trajectories[build_id] = (signature, data)
        return data

    def run(self, run_id: str) -> tuple[Path, dict] | None:
        run_dir = crowd_run_dir(self.builds_root, run_id)
        if run_dir is None:
            return None
        try:
            signature = str((run_dir / "run_summary.json").stat().st_mtime_ns)
        except OSError:
            signature = "0"
        cached = self._runs.get(run_id)
        if cached and cached[0] == signature:
            return run_dir, cached[1]
        data = crowd_mod.load_run(run_dir)
        if data is None:
            return None
        if len(self._runs) > 24:
            self._runs.clear()
        self._runs[run_id] = (signature, data)
        return run_dir, data

    def rubric_grades(self, refresh: bool = False) -> list[dict]:
        with self._lock:
            stale = (
                refresh
                or self._rubric is None
                or time.time() - self._rubric_at > self.rubric_ttl
            )
        if stale:
            rows = index_rubric_grades(self.builds_root)
            with self._lock:
                self._rubric, self._rubric_at = rows, time.time()
        with self._lock:
            return self._rubric or []

    def grade(self, run_id: str) -> tuple[Path, dict] | None:
        run_dir = rubric_run_dir(self.builds_root, run_id)
        if run_dir is None:
            return None
        try:
            signature = str((run_dir / "grade.json").stat().st_mtime_ns)
        except OSError:
            signature = "0"
        cached = self._grades.get(run_id)
        if cached and cached[0] == signature:
            return run_dir, cached[1]
        data = rubric_mod.load_grade(run_dir)
        if data is None:
            return None
        if len(self._grades) > 24:
            self._grades.clear()
        self._grades[run_id] = (signature, data)
        return run_dir, data


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ViralBenchViewer/1.0"

    store: Store
    launcher: AppLauncher
    verbose = False

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args) -> None:  # noqa: A002 - stdlib signature
        if self.verbose:
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def _send(
        self, status: int, body: bytes, content_type: str, extra: dict | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Everything here is generated from files that keep changing, and a cached
        # timeline showing a build's previous state would be misleading.
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def fail(self, status: int, message: str) -> None:
        self.json({"error": message}, status)

    def _query(self) -> dict:
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8")) or {}
        except (ValueError, UnicodeDecodeError):
            return {}

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        self._route("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._route("POST")

    def _route(self, method: str) -> None:
        path = urlparse(self.path).path
        try:
            if method == "GET" and self._serve_page(path):
                return
            if path.startswith("/api/"):
                self._api(method, path)
                return
            self.fail(404, f"no route for {path}")
        except BrokenPipeError:
            pass
        except Exception:  # noqa: BLE001 - a viewer must not die on one bad build
            traceback.print_exc()
            self.fail(500, "internal error, see server log")

    def _serve_page(self, path: str) -> bool:
        pages = {
            "/": "index.html",
            "/founder": "founder.html",
            "/crowd": "crowd.html",
            "/rubric": "rubric.html",
        }
        if path in pages:
            self._static(STATIC_DIR / pages[path])
            return True
        if path.startswith("/static/"):
            relative = path[len("/static/") :]
            target = (STATIC_DIR / relative).resolve()
            if STATIC_DIR.resolve() in target.parents and target.is_file():
                self._static(target)
            else:
                self.fail(404, "not found")
            return True
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return True
        return False

    def _static(self, target: Path) -> None:
        if not target.is_file():
            self.fail(404, f"missing asset {target.name}")
            return
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if kind.startswith("text/") or kind in (
            "application/javascript",
            "application/json",
        ):
            kind += "; charset=utf-8"
        self._send(200, target.read_bytes(), kind)

    def _api(self, method: str, path: str) -> None:
        parts = [p for p in path.split("/") if p][1:]  # drop "api"
        query = self._query()

        if not parts:
            return self.fail(404, "no endpoint")

        head = parts[0]

        if head == "meta" and method == "GET":
            return self.json(
                {
                    "builds_root": str(self.store.builds_root),
                    "cache": str(Path(__file__).resolve().parent / "cache"),
                    "thinking_note": founder_mod.THINKING_NOTE,
                }
            )

        if head == "builds" and method == "GET":
            return self.json(self._filter_builds(query))

        if head == "founder":
            return self._founder(method, parts[1:], query)

        if head == "crowd":
            return self._crowd(method, parts[1:], query)

        if head == "rubric":
            return self._rubric(method, parts[1:], query)

        if head == "shot" and method == "GET" and len(parts) == 2:
            return self._shot(parts[1])

        if head == "app":
            return self._app(method, parts[1:])

        return self.fail(404, f"no endpoint /api/{'/'.join(parts)}")

    # -- endpoints ---------------------------------------------------------

    def _filter_builds(self, query: dict) -> dict:
        rows = self.store.builds(refresh=query.get("refresh") == "1")
        needle = (query.get("q") or "").strip().lower()
        mode = query.get("mode") or ""
        status = query.get("status") or ""
        only_crowd = query.get("crowd") == "1"

        filtered = []
        for row in rows:
            if mode and row["mode"] != mode:
                continue
            if status and row["status"] != status:
                continue
            if only_crowd and not row["crowd_runs"]:
                continue
            if (
                needle
                and needle
                not in (
                    f"{row['build_id']} {row['idea_id']} {row['model']} "
                    f"{row['mode_label']} "
                    f"{row['app_title'] or ''}"
                ).lower()
            ):
                continue
            filtered.append(row)

        limit = min(int(query.get("limit") or 400), 2000)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["mode"]] = counts.get(row["mode"], 0) + 1
        return {
            "total": len(rows),
            "matched": len(filtered),
            "modes": counts,
            "builds": filtered[:limit],
        }

    def _founder(self, method: str, rest: list[str], query: dict) -> None:
        if method != "GET" or not rest:
            return self.fail(404, "unknown founder endpoint")
        build_id = safe_id(rest[0])
        if not build_id:
            return self.fail(400, "bad build id")
        tail = rest[1:]

        if not tail:
            data = self.store.trajectory(build_id)
            if data is None:
                return self.fail(404, f"no build {build_id}")
            data = dict(data)
            data["crowd_runs"] = [
                r for r in self.store.crowd_runs() if r["build_id"] == build_id
            ]
            return self.json(data)

        if tail[0] == "event":
            # Two addressing schemes, because there are two sources: a traced
            # build names events by part_id in the session dumps, an untraced one
            # by transcript file and line.
            part_id = safe_id(query.get("part") or "")
            if part_id:
                body = founder_mod.load_event_body(
                    self.store.builds_root, build_id, part_id=part_id
                )
                return self.json(body) if body else self.fail(404, "no such event")
            phase = safe_id(query.get("phase") or "")
            try:
                line = int(query.get("line"))
            except (TypeError, ValueError):
                return self.fail(400, "line must be an integer")
            if not phase:
                return self.fail(400, "bad phase")
            body = founder_mod.load_event_body(
                self.store.builds_root, build_id, phase=phase, line=line
            )
            return self.json(body) if body else self.fail(404, "no such event")

        if tail[0] == "shot" and len(tail) == 2:
            name = Path(unquote(tail[1])).name
            target = (
                founder_paths(self.store.builds_root, build_id).screenshots_dir / name
            )
            if not target.is_file():
                return self.fail(404, "screenshot not found")
            return self._send(200, target.read_bytes(), "image/png")

        if tail[0] == "download":
            fmt = query.get("format") or "json"
            if fmt == "zip":
                blob = export_mod.founder_zip(self.store.builds_root, build_id)
                if blob is None:
                    return self.fail(404, "no such build")
                return self._send(
                    200,
                    blob,
                    "application/zip",
                    {"Content-Disposition": f'attachment; filename="{build_id}.zip"'},
                )
            bundle = export_mod.founder_bundle(self.store.builds_root, build_id)
            if bundle is None:
                return self.fail(404, "no such build")
            body = json.dumps(bundle, indent=2, ensure_ascii=False, default=str).encode(
                "utf-8"
            )
            return self._send(
                200,
                body,
                "application/json; charset=utf-8",
                {
                    "Content-Disposition": (
                        f'attachment; filename="{build_id}.trajectory.json"'
                    )
                },
            )

        return self.fail(404, "unknown founder endpoint")

    def _crowd(self, method: str, rest: list[str], query: dict) -> None:
        if not rest:
            return self.fail(404, "unknown crowd endpoint")

        if rest[0] == "runs" and method == "GET":
            rows = self.store.crowd_runs(refresh=query.get("refresh") == "1")
            build_id = query.get("build_id")
            needle = (query.get("q") or "").strip().lower()
            if build_id:
                rows = [r for r in rows if r["build_id"] == build_id]
            if needle:
                rows = [
                    r
                    for r in rows
                    if needle in f"{r['run_id']} {r['build_id']} {r['model']}".lower()
                ]
            if query.get("ok") == "1":
                rows = [r for r in rows if r["ok"]]
            limit = min(int(query.get("limit") or 300), 3000)
            return self.json({"matched": len(rows), "runs": rows[:limit]})

        run_id = safe_id(rest[0])
        if not run_id:
            return self.fail(400, "bad run id")
        loaded = self.store.run(run_id)
        if loaded is None:
            return self.fail(404, f"no crowd run {run_id}")
        run_dir, data = loaded
        tail = rest[1:]

        if not tail and method == "GET":
            return self.json(data)

        if tail and tail[0] == "trial" and len(tail) == 2 and method == "GET":
            try:
                agent_id = int(tail[1])
            except ValueError:
                return self.fail(400, "bad agent id")
            trial = crowd_mod.load_trial(run_dir, agent_id)
            return (
                self.json(trial) if trial else self.fail(404, "no trial for that agent")
            )

        if tail and tail[0] == "feed" and len(tail) == 3 and method == "GET":
            try:
                agent_id, step = int(tail[1]), int(tail[2])
            except ValueError:
                return self.fail(400, "bad agent id or step")
            return self.json(
                {
                    "agent_id": agent_id,
                    "t": step,
                    "posts": crowd_mod.load_feed(run_dir, agent_id, step),
                }
            )

        if tail and tail[0] == "rescue" and method == "POST":
            return self.json(rescue_run(run_dir))

        if tail and tail[0] == "download" and method == "GET":
            if (query.get("format") or "json") == "zip":
                blob = export_mod.crowd_zip(run_dir)
                if blob is None:
                    return self.fail(404, "no such run")
                return self._send(
                    200,
                    blob,
                    "application/zip",
                    {"Content-Disposition": f'attachment; filename="{run_id}.zip"'},
                )
            bundle = export_mod.crowd_bundle(run_dir)
            if bundle is None:
                return self.fail(404, "no such run")
            body = json.dumps(bundle, indent=2, ensure_ascii=False, default=str).encode(
                "utf-8"
            )
            return self._send(
                200,
                body,
                "application/json; charset=utf-8",
                {
                    "Content-Disposition": (
                        f'attachment; filename="{run_id}.trajectory.json"'
                    )
                },
            )

        return self.fail(404, "unknown crowd endpoint")

    def _rubric(self, method: str, rest: list[str], query: dict) -> None:
        if not rest:
            return self.fail(404, "unknown rubric endpoint")

        if rest[0] == "grades" and method == "GET":
            rows = self.store.rubric_grades(refresh=query.get("refresh") == "1")
            build_id = query.get("build_id")
            needle = (query.get("q") or "").strip().lower()
            if build_id:
                rows = [r for r in rows if r["build_id"] == build_id]
            if needle:
                rows = [
                    r
                    for r in rows
                    if needle
                    in f"{r['run_id']} {r['build_id']} {r['idea_id']} "
                    f"{r['model_short']}".lower()
                ]
            if query.get("gated") == "1":
                rows = [r for r in rows if r["gate_zeroed"]]
            if query.get("diverged") == "1":
                # Only grades where the two tracks disagree. This is the
                # question the whole second track exists to answer, so it gets a
                # filter rather than making you open grades one at a time.
                rows = [
                    r for r in rows if r["delta"] is not None and abs(r["delta"]) >= 20
                ]
            limit = min(int(query.get("limit") or 300), 3000)
            return self.json({"matched": len(rows), "grades": rows[:limit]})

        run_id = safe_id(rest[0])
        if not run_id:
            return self.fail(400, "bad grade id")
        loaded = self.store.grade(run_id)
        if loaded is None:
            return self.fail(404, f"no rubric grade {run_id}")
        run_dir, data = loaded
        tail = rest[1:]

        if not tail and method == "GET":
            return self.json(data)

        if tail[0] == "transcript" and method == "GET":
            return self.json(
                rubric_mod.load_transcript(
                    run_dir,
                    offset=int(query.get("offset") or 0),
                    limit=min(int(query.get("limit") or 200), 2000),
                    item_id=(query.get("item_id") or "").strip(),
                )
            )

        if tail[0] == "shot" and len(tail) == 2 and method == "GET":
            # Resolved inside the grade's own shots/ dir rather than through the
            # crowd's /tmp rescue path: rubric evidence is written beside the
            # grade and never goes missing the way a crowd screenshot can.
            name = Path(unquote(tail[1])).name
            target = run_dir / "shots" / name
            if not target.is_file():
                return self.fail(404, f"no such screenshot {name}")
            kind = mimetypes.guess_type(target.name)[0] or "image/png"
            return self._send(200, target.read_bytes(), kind)

        if tail[0] == "download" and method == "GET":
            if query.get("format") == "zip":
                blob = export_mod.rubric_zip(run_dir)
                if blob is None:
                    return self.fail(404, "nothing to export")
                return self._send(
                    200,
                    blob,
                    "application/zip",
                    {"Content-Disposition": f'attachment; filename="{run_id}.zip"'},
                )
            bundle = export_mod.rubric_bundle(run_dir)
            if bundle is None:
                return self.fail(404, "nothing to export")
            body = json.dumps(bundle, indent=2, ensure_ascii=False, default=str).encode(
                "utf-8"
            )
            return self._send(
                200,
                body,
                "application/json; charset=utf-8",
                {
                    "Content-Disposition": (
                        f'attachment; filename="{run_id}.grade.json"'
                    )
                },
            )

        return self.fail(404, "unknown rubric endpoint")

    def _shot(self, name: str) -> None:
        resolved = resolve_shot(unquote(name))
        if resolved is None:
            return self.fail(
                404, "screenshot not available (it lived in /tmp and is gone)"
            )
        kind = mimetypes.guess_type(resolved.name)[0] or "image/png"
        self._send(200, resolved.read_bytes(), kind)

    def _app(self, method: str, rest: list[str]) -> None:
        if method == "GET" and not rest:
            return self.json({"running": self.launcher.list()})
        if method == "POST" and rest and rest[0] == "launch":
            body = self._body()
            build_id = safe_id(str(body.get("build_id") or ""))
            if not build_id:
                return self.fail(400, "bad build id")
            try:
                port = int(body["port"]) if body.get("port") else None
            except (TypeError, ValueError):
                return self.fail(400, "port must be an integer")
            return self.json(self.launcher.launch(build_id, port=port))
        if method == "POST" and rest and rest[0] == "stop":
            session_id = str(self._body().get("session_id") or "")
            return self.json(self.launcher.stop(session_id))
        return self.fail(404, "unknown app endpoint")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("VIZ_PORT", DEFAULT_PORT))
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--builds-root",
        type=Path,
        default=None,
        help=(
            "builds/ tree to read (read-only). Defaults to this repo's builds/, "
            "or $VIRAL_BENCH_BUILDS_DIR when set."
        ),
    )
    parser.add_argument(
        "--open",
        dest="open_view",
        choices=("founder", "crowd", "rubric", "home"),
        default=None,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log every request"
    )
    args = parser.parse_args(argv)

    builds_root = (args.builds_root or default_builds_root()).resolve()
    if not (builds_root / "work").is_dir():
        print(
            f"warning: {builds_root} has no work/ subdirectory -- "
            "the build list will be empty",
            file=sys.stderr,
        )

    Handler.store = Store(builds_root)
    Handler.launcher = AppLauncher(builds_root)
    Handler.verbose = args.verbose

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    base = f"http://{args.host}:{args.port}"

    print("ViralBench trajectory viewer")
    print(f"  builds     {builds_root}  (read-only)")
    print(f"  founder    {base}/founder")
    print(f"  crowd      {base}/crowd")
    print(f"  rubric     {base}/rubric")
    print(f"  home       {base}/")
    print("Ctrl-C to stop.")

    if args.open_view:
        target = base if args.open_view == "home" else f"{base}/{args.open_view}"
        threading.Timer(0.6, lambda: webbrowser.open(target)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        Handler.launcher.stop_all()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
