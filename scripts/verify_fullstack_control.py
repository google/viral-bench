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

"""Acceptance test for the server-side app contract, run against the control build.

This is the check that the web-dev pivot rests on. It is deliberately an
end-to-end test through the real ``AppHost``/``AppSession`` machinery rather than
a unit test, because every failure it is designed to catch lives in the seams
between those pieces, not inside any one of them.

Four properties, in the order they broke historically:

1. **setup -> run share one database.** A migration runs in an ephemeral ``--rm``
   container, while the server runs in a different, long-lived one. Before ``/data`` was
   mounted, the schema the migration created vanished with the setup container and
   the app started against an empty file.
2. **State survives an app restart.** ``AppSession.close`` deletes the run dir, and
   the crowd's shared ``AppHost`` re-materializes a fresh clone whenever an
   instance dies. A database kept under the app dir is wiped by that, silently,
   in the middle of a run.
3. **One agent's write is visible to another agent.** The crowd shares a single
   instance per build precisely so this holds, and it is the entire reason a
   full-stack idea is worth benchmarking.
4. **State does NOT leak between crowd runs.** Accumulating across runs would make
   two runs of the same build incomparable.

Run:  uv run python scripts/verify_fullstack_control.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viral_bench.founder.apphost import AppHost  # noqa: E402
from viral_bench.founder.workspace import (  # noqa: E402
    build_data_dir,
    reset_build_data,
)

BUILD_ID = "quick_notes_app__20260806-000000__fsctl0"

_PASS = "PASS"
_FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((_PASS if ok else _FAIL, name, detail))
    print(f"  [{_PASS if ok else _FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep the 303 itself instead of following it.

    The session cookie is set ON the redirect response, and urllib's default handler
    follows it transparently and hands back the final GET, whose headers carry no
    Set-Cookie. Without this the login flow looks broken when it is not.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def post(url: str, fields: dict, cookie: str | None = None) -> tuple[int, str, str]:
    """POST a form. Returns (status, body, session-cookie-if-any)."""
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    if cookie:
        req.add_header("Cookie", f"session={cookie}")
    try:
        with _opener.open(req, timeout=15) as resp:
            body = resp.read().decode()
            set_cookie = resp.headers.get("Set-Cookie") or ""
            status = resp.status
    except urllib.error.HTTPError as exc:
        # A 3xx surfaces here because the opener refuses to follow it, which is success.
        status = exc.code
        body = exc.read().decode()
        set_cookie = exc.headers.get("Set-Cookie") or ""
    token = ""
    if "session=" in set_cookie:
        token = set_cookie.split("session=", 1)[1].split(";", 1)[0]
    return status, body, token


def get(url: str, cookie: str | None = None) -> str:
    req = urllib.request.Request(url)
    if cookie:
        req.add_header("Cookie", f"session={cookie}")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode()


def main() -> int:
    print(f"Acceptance test: {BUILD_ID}\n")

    # A crowd run begins by clearing the build's durable state.
    reset_build_data(BUILD_ID)
    data_dir = build_data_dir(BUILD_ID)
    print(f"data dir: {data_dir}\n")

    host = AppHost(container=True)
    try:
        print("1. setup -> run share one database")
        app = host.get(BUILD_ID)
        url = app.url
        check("app started and published a URL", bool(url), str(url))
        body = get(url)
        check(
            "server answers without a schema error",
            "QuickMemo" in body and "no such table" not in body,
        )
        check(
            "migration from the setup step is visible to the server",
            (data_dir / "app.db").exists(),
            f"{(data_dir / 'app.db').exists()=}",
        )

        print("\n2. one agent writes")
        status, _, alice = post(
            f"{url}register", {"username": "alice", "password": "pw"}
        )
        check("agent A registered", status in (200, 303) and bool(alice))
        post(f"{url}post", {"body": "memo from alice"}, cookie=alice)
        check("agent A's memo is on the page", "memo from alice" in get(url))

        print("\n3. a DIFFERENT agent sees it")
        status, _, bob = post(f"{url}register", {"username": "bob", "password": "pw"})
        check("agent B registered", status in (200, 303) and bool(bob))
        bob_view = get(url, cookie=bob)
        check(
            "agent B sees agent A's memo",
            "memo from alice" in bob_view,
            "this is the multi-user signal",
        )
        check(
            "agent B is a distinct identity",
            alice != bob and "bob" in bob_view,
        )

        print("\n4. state survives the restart the crowd actually performs")
        # host.stop() closes the session -- which deletes the run dir -- and the
        # next get() re-materializes a fresh clone. This is the exact path taken
        # when an instance dies mid-run.
        host.stop(BUILD_ID)
        app2 = host.get(BUILD_ID)
        url2 = app2.url
        check("app came back up", bool(url2), str(url2))
        after = get(url2)
        check(
            "agent A's memo survived the restart",
            "memo from alice" in after,
            "a DB under the app dir would have been wiped here",
        )
        # A new container, not the old one still running: the host port
        # is assigned dynamically per container, so it must differ.
        check(
            "the instance really was replaced, not reused",
            url2 != url,
            f"{url} -> {url2}",
        )

        print("\n5. state does NOT leak into the next crowd run")
        host.stop(BUILD_ID)
        reset_build_data(BUILD_ID)
        app3 = host.get(BUILD_ID)
        fresh = get(app3.url)
        check(
            "next run starts with an empty board",
            "memo from alice" not in fresh and "No memos yet" in fresh,
        )
    finally:
        host.close()

    failed = [r for r in results if r[0] == _FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("\nFAILED:")
        for _, name, detail in failed:
            print(f"  - {name}" + (f" ({detail})" if detail else ""))
        return 1
    print("Server-side app contract holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
