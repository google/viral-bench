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

"""Create a WORKING full-stack control app that exercises the server-side contract.

This is the positive counterpart to ``make_control_build.py`` (which builds a
deliberately broken app to calibrate the score's floor). This one is deliberately
*correct*, and exists to answer a different question: **can the harness run,
persist, and observe a real server-side app?**

It is the acceptance test for the web-dev pivot. Specifically it exercises, in one
build, every part of the contract a full-stack app depends on:

* a ``setup`` step that runs a migration BEFORE the app starts, writing to the same
  data directory the server later reads (``_run_oneshot`` composes the same mount
  args as ``start``, so this should hold -- this app proves it),
* a database that must SURVIVE an app restart (``AppSession.close`` deletes the
  run dir, so a DB living under the app dir does not),
* a multi-user flow where one agent's write is visible to another agent's read,
  which is the whole reason the crowd shares one instance per build, and
* a UI that is legible to a TEXT-ONLY agent: the crowd perceives an app as
  ``innerText`` + an ARIA snapshot and never sees pixels, so every control here is
  a labelled form element and every piece of state is rendered as text.

Deliberately stdlib-only (``ThreadingHTTPServer`` + ``sqlite3``). A control that
needed ``uv add fastapi`` would fail whenever the network hiccuped, and would
conflate "can the harness run a server app?" with "did the install work?". Those
are separate questions and this script answers only the first.

Where the database lives is the point of the whole exercise: the app writes to
``$VIRALBENCH_DATA_DIR`` (default ``/data``). Before the persistent-volume work
lands, that path is an ordinary directory *inside* the container, so it vanishes on
restart and the persistence check below FAILS -- correctly. Once ``/data`` is a real
bind mount, the same app passes with no code change. That A/B is the test.

Run:  uv run python scripts/make_fullstack_control.py
Then: uv run python scripts/make_fullstack_control.py --verify
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viral_bench.founder.build import builds_root  # noqa: E402

BUILD_ID = "quick_notes_app__20260806-000000__fsctl0"

MIGRATE_PY = '''"""Create the schema.

Runs as the manifest's setup step, before the app starts.
"""

import os
import sqlite3
from pathlib import Path

DATA_DIR = Path(os.environ.get("VIRALBENCH_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    pw_hash  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memos (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token   TEXT PRIMARY KEY,
    username TEXT NOT NULL
);
"""


def main() -> int:
    con = sqlite3.connect(DB_PATH)
    # WAL lets many readers run alongside a writer, which matters because the
    # crowd hits one shared instance with up to 8 agents at once.
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    con.commit()
    con.close()
    print(f"migrated {DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

SERVER_PY = '''"""QuickMemo -- a tiny multi-user memo board.

Positive control for ViralBench.

Everything a text-only agent needs is rendered as labelled text: no canvas, no
images, no icon-only buttons.
"""

import hashlib
import html
import http.cookies
import os
import secrets
import sqlite3
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DATA_DIR = Path(os.environ.get("VIRALBENCH_DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.db"
PORT = int(os.environ.get("PORT", "8000"))


def db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def hash_pw(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 100_000
    ).hex()


def page(body: str, user: str | None) -> bytes:
    who = (
        f'<p>Signed in as <strong>{html.escape(user)}</strong>. '
        f'<a href="/logout">Sign out</a></p>'
        if user
        else "<p>Not signed in.</p>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>QuickMemo</title></head>
<body>
<h1>QuickMemo</h1>
<p>A shared memo board. Anything you post is visible to everyone.</p>
{who}
{body}
</body></html>""".encode()


def render_memos(con) -> str:
    rows = con.execute(
        "SELECT author, body, created_at FROM memos ORDER BY id DESC LIMIT 50"
    ).fetchall()
    if not rows:
        return "<h2>All memos</h2><p>No memos yet. Be the first to post one.</p>"
    items = "".join(
        f"<li><strong>{html.escape(r['author'])}</strong> wrote: "
        f"{html.escape(r['body'])} <em>({html.escape(r['created_at'])})</em></li>"
        for r in rows
    )
    return f"<h2>All memos ({len(rows)})</h2><ul>{items}</ul>"


AUTH_FORM = """
<h2>Sign in or create an account</h2>
<form method="POST" action="/register">
  <label for="r_user">Username</label>
  <input id="r_user" name="username" type="text" required>
  <label for="r_pass">Password</label>
  <input id="r_pass" name="password" type="password" required>
  <button type="submit">Create account</button>
</form>
<form method="POST" action="/login">
  <label for="l_user">Username</label>
  <input id="l_user" name="username" type="text" required>
  <label for="l_pass">Password</label>
  <input id="l_pass" name="password" type="password" required>
  <button type="submit">Sign in</button>
</form>
"""

POST_FORM = """
<h2>Write a memo</h2>
<form method="POST" action="/post">
  <label for="body">Memo text</label>
  <input id="body" name="body" type="text" required>
  <button type="submit">Post memo</button>
</form>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter logs
        pass

    def _user(self) -> str | None:
        raw = self.headers.get("Cookie")
        if not raw:
            return None
        cookie = http.cookies.SimpleCookie(raw)
        morsel = cookie.get("session")
        if morsel is None:
            return None
        con = db()
        row = con.execute(
            "SELECT username FROM sessions WHERE token = ?", (morsel.value,)
        ).fetchone()
        con.close()
        return row["username"] if row else None

    def _send(self, body: bytes, status: int = 200, cookie: str | None = None):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, to: str, cookie: str | None = None):
        self.send_response(303)
        self.send_header("Location", to)
        self.send_header("Content-Length", "0")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()

    def _form(self) -> dict:
        # Tolerate a missing/short Content-Length rather than blocking forever.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length).decode() if length > 0 else ""
        return {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}

    def do_GET(self):
        if self.path == "/healthz":
            self._send(b"ok")
            return
        if self.path == "/logout":
            self._redirect("/", cookie="session=; Max-Age=0; Path=/")
            return
        user = self._user()
        con = db()
        memos = render_memos(con)
        con.close()
        body = (POST_FORM if user else AUTH_FORM) + memos
        self._send(page(body, user))

    def do_POST(self):
        form = self._form()
        if self.path == "/register":
            username = (form.get("username") or "").strip()
            password = form.get("password") or ""
            if not username or not password:
                self._send(
                    page("<p>Username and password are required.</p>", None), 400
                )
                return
            salt = "quickmemo"
            con = db()
            try:
                con.execute(
                    "INSERT INTO users (username, pw_hash) VALUES (?, ?)",
                    (username, hash_pw(password, salt)),
                )
                con.commit()
            except sqlite3.IntegrityError:
                con.close()
                self._send(page("<p>That username is taken.</p>", None), 409)
                return
            token = secrets.token_hex(16)
            con.execute(
                "INSERT INTO sessions (token, username) VALUES (?, ?)",
                (token, username),
            )
            con.commit()
            con.close()
            self._redirect("/", cookie=f"session={token}; Path=/; HttpOnly")
            return

        if self.path == "/login":
            username = (form.get("username") or "").strip()
            password = form.get("password") or ""
            con = db()
            row = con.execute(
                "SELECT pw_hash FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row is None or row["pw_hash"] != hash_pw(password, "quickmemo"):
                con.close()
                self._send(page("<p>Wrong username or password.</p>", None), 401)
                return
            token = secrets.token_hex(16)
            con.execute(
                "INSERT INTO sessions (token, username) VALUES (?, ?)",
                (token, username),
            )
            con.commit()
            con.close()
            self._redirect("/", cookie=f"session={token}; Path=/; HttpOnly")
            return

        if self.path == "/post":
            user = self._user()
            if user is None:
                self._send(page("<p>Sign in to post a memo.</p>", None), 401)
                return
            body = (form.get("body") or "").strip()
            if body:
                con = db()
                con.execute(
                    "INSERT INTO memos (author, body, created_at) VALUES (?, ?, ?)",
                    (user, body, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")),
                )
                con.commit()
                con.close()
            self._redirect("/")
            return

        self._send(page("<p>Not found.</p>", None), 404)


if __name__ == "__main__":
    # ThreadingHTTPServer, never the single-threaded HTTPServer: a browser opens
    # several connections at once and the crowd runs up to 8 agents in parallel.
    # 0.0.0.0, never localhost: the container publishes the port to the host.
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
'''

README = """# QuickMemo

A shared memo board. Create an account, post a memo, and see everyone else's.

Positive control for ViralBench: a real server-side app with a database, accounts,
and a multi-user flow, using only the Python standard library.

- `python3 migrate.py` creates the schema (runs as the manifest `setup` step)
- `python3 server.py` serves on port 8000
- State lives in `$VIRALBENCH_DATA_DIR` (default `/data`) so it survives a restart
"""

MANIFEST = {
    "app_type": "full-stack-app",
    "title": "QuickMemo",
    "summary": "A shared multi-user memo board with accounts and a SQLite database.",
    "setup": ["python3 migrate.py"],
    "run": {
        "command": "python3 server.py",
        "cwd": ".",
        "port": 8000,
        "url": "http://localhost:8000/",
    },
    "test": {
        "manual": [
            "Open the URL.",
            "Create an account with a username and password.",
            "Post a memo, then confirm it appears under 'All memos'.",
            "Open the app as a second user and confirm the first user's"
            " memo is visible.",
        ],
        # A real health check against the running server. This is only passable
        # because the gate now starts the app BEFORE smoking it, and runs the
        # command inside the app's own container -- previously a smoke test that
        # touched the app's port failed 100% of the time (0/45 across the corpus).
        "smoke": "curl -fsS http://localhost:8000/healthz",
    },
    "notes": (
        "Full-stack: ThreadingHTTPServer + sqlite3, no third-party dependencies. "
        "Database lives under $VIRALBENCH_DATA_DIR (default /data) so it survives "
        "an app restart; everything under the app dir is deleted when the session "
        "closes."
    ),
}


def create() -> Path:
    root = builds_root() / "work" / BUILD_ID
    app = root / "app"
    app.mkdir(parents=True, exist_ok=True)

    (app / "migrate.py").write_text(MIGRATE_PY, encoding="utf-8")
    (app / "server.py").write_text(SERVER_PY, encoding="utf-8")
    (app / "README.md").write_text(README, encoding="utf-8")
    (app / "viralbench.json").write_text(
        json.dumps(MANIFEST, indent=2), encoding="utf-8"
    )

    (root / "build.json").write_text(
        json.dumps(
            {
                "build_id": BUILD_ID,
                "idea_id": "quick_notes_app",
                "model": "control/fullstack-ok",
                "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "status": "ok",
                "app_dir": str(app.resolve()),
                "root": str(root.resolve()),
                "harness_ok": True,
                "structure": "solo",
                "n_agents": 1,
                "max_rounds": 1,
                "min_rounds": 1,
                "rounds_run": 1,
                "shipped_early": False,
                "qa_verified": True,
                "max_turns": 2,
                "min_turns": 2,
                "turns_spent": 2,
                "roles": ["founder"],
                "collab": "local",
                "collab_meta": {"collab": "local"},
                "phases": [],
                "manifest": MANIFEST,
                "manifest_error": None,
                "error": None,
                "shipped_ref": None,
                "store_path": None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    app = create()
    print(f"created full-stack control build: {BUILD_ID}")
    print(f"  app: {app}")
    print("  verify it with: uv run python scripts/verify_fullstack_control.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
