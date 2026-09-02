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

"""End-to-end checks against a real server on a real port.

The parser tests cover the data; this covers the wiring -- routes, status codes,
content types, the download headers, and the two places where a URL segment is
user input that becomes a filesystem path.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import serve as serve_mod  # noqa: E402
from core.apphost import AppLauncher  # noqa: E402
from vizfixtures import make_crowd_run  # noqa: E402


@pytest.fixture
def server(builds):  # noqa: F811 - pytest fixture injection
    make_crowd_run(builds, "idea__solo__crowd-20260101-000000-s0")

    serve_mod.Handler.store = serve_mod.Store(builds)
    serve_mod.Handler.launcher = AppLauncher(builds)
    serve_mod.Handler.verbose = False

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve_mod.Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    # The socket is already bound and listening by the time serve_forever starts,
    # so a request can go out immediately; this only yields the GIL.
    time.sleep(0.05)
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(base: str, path: str):
    with urllib.request.urlopen(f"{base}{path}", timeout=10) as response:
        return response.status, response.headers, response.read()


def get_json(base: str, path: str):
    status, _, body = get(base, path)
    return status, json.loads(body)


def expect_status(base: str, path: str) -> int:
    try:
        return get(base, path)[0]
    except urllib.error.HTTPError as exc:
        return exc.code


@pytest.mark.parametrize("path", ["/", "/founder", "/crowd", "/rubric"])
def test_pages_render(server, path):
    status, headers, body = get(server, path)
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"ViralBench" in body


@pytest.mark.parametrize(
    "path",
    [
        "/static/css/app.css",
        "/static/js/common.js",
        "/static/js/founder.js",
        "/static/js/crowd.js",
        "/static/js/rubric.js",
    ],
)
def test_static_assets_are_served(server, path):
    status, _, body = get(server, path)
    assert status == 200 and body


def test_meta_reports_the_builds_root(server):
    status, payload = get_json(server, "/api/meta")
    assert status == 200
    assert "redacted" in payload["thinking_note"]


def test_build_index_and_filters(server):
    status, payload = get_json(server, "/api/builds")
    assert status == 200
    assert payload["total"] == 4
    assert payload["modes"]["team"] == 1

    _, filtered = get_json(server, "/api/builds?mode=team")
    assert [b["build_id"] for b in filtered["builds"]] == ["idea__team"]

    _, searched = get_json(server, "/api/builds?q=dynamic")
    assert searched["matched"] == 1


def test_founder_trajectory_endpoint(server):
    status, payload = get_json(server, "/api/founder/idea__team")
    assert status == 200
    assert payload["mode"] == "team"
    assert len(payload["lanes"]) == 4
    assert payload["events"]


def test_founder_event_body_endpoint(server):
    _, trajectory = get_json(server, "/api/founder/idea__solo")
    write = next(e for e in trajectory["events"] if e.get("tool") == "write")
    _, body = get_json(server, f"/api/founder/idea__solo/event?part={write['part_id']}")
    assert body["part"]["tool"] == "write"


def test_founder_downloads(server):
    status, headers, body = get(server, "/api/founder/idea__team/download")
    assert status == 200
    assert "idea__team.trajectory.json" in headers["Content-Disposition"]
    assert json.loads(body)["kind"] == "viral_bench.founder_trajectory"

    status, headers, blob = get(server, "/api/founder/idea__team/download?format=zip")
    assert status == 200
    assert headers["Content-Type"] == "application/zip"
    assert blob[:2] == b"PK"


def test_crowd_endpoints(server):
    run_id = "idea__solo__crowd-20260101-000000-s0"
    status, listed = get_json(server, "/api/crowd/runs")
    assert status == 200 and listed["matched"] == 1

    _, run = get_json(server, f"/api/crowd/{run_id}")
    assert run["app_type"] == "client-app"
    assert len(run["agents"]) == 3

    _, trial = get_json(server, f"/api/crowd/{run_id}/trial/1")
    assert len(trial["steps"]) == 3

    _, feed = get_json(server, f"/api/crowd/{run_id}/feed/1/2")
    assert feed["posts"][0]["post_id"] == 1

    status, _, blob = get(server, f"/api/crowd/{run_id}/download?format=zip")
    assert status == 200 and blob[:2] == b"PK"


def test_a_build_links_to_its_crowd_runs(server):
    _, payload = get_json(server, "/api/founder/idea__solo")
    assert [r["run_id"] for r in payload["crowd_runs"]] == [
        "idea__solo__crowd-20260101-000000-s0"
    ]


def test_unknown_ids_are_404_not_500(server):
    assert expect_status(server, "/api/founder/nope__nope") == 404
    assert expect_status(server, "/api/crowd/nope__nope") == 404
    assert expect_status(server, "/api/nonsense") == 404


@pytest.mark.parametrize(
    "nasty",
    [
        "/api/founder/..%2F..%2Fetc",
        "/api/founder/idea__solo/shot/..%2F..%2F..%2Fetc%2Fpasswd",
        "/static/../serve.py",
        "/api/shot/..%2F..%2Fetc%2Fpasswd",
    ],
)
def test_path_traversal_is_refused(server, nasty):
    """Ids and filenames arrive from the URL and end up joined onto a path."""
    assert expect_status(server, nasty) in (400, 404)


def test_app_launch_rejects_a_build_with_no_manifest(server):
    request = urllib.request.Request(
        f"{server}/api/app/launch",
        data=json.dumps({"build_id": "idea__solo"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read())
    # The fixture build ships no viralbench.json, so this is the honest answer
    # rather than a traceback.
    assert payload["ok"] is False
    assert "viralbench.json" in payload["error"]


# --------------------------------------------------------------------------
# live discovery
# --------------------------------------------------------------------------
#
# The viewer is pointed at a builds tree that an active benchmark keeps writing
# into, so "will tomorrow's run show up?" is a property of the product, not an
# implementation detail. These pin it.


def test_a_build_written_after_startup_is_found_by_id_immediately(server, builds):
    """No index refresh, no TTL wait, no restart.

    Lookup by id resolves straight to a path and stats it, so a run that finished
    one second ago opens. If this ever routed through the cached index instead,
    a fresh build would 404 for up to a minute and look lost.
    """
    from vizfixtures import write_build

    assert expect_status(server, "/api/founder/idea__brandnew") == 404
    write_build(
        builds,
        "idea__brandnew",
        {
            "idea_id": "idea",
            "model": "google-vertex/gemini-9-future",
            "structure": "solo",
            "collab": "local",
            "status": "ok",
            "created_at": "2027-01-01T00:00:00+00:00",
        },
        {},
    )
    status, payload = get_json(server, "/api/founder/idea__brandnew")
    assert status == 200
    assert payload["build_id"] == "idea__brandnew"
    assert payload["record"]["model"] == "google-vertex/gemini-9-future"


def test_a_crowd_run_written_after_startup_is_found_by_id_immediately(server, builds):
    run_id = "idea__solo__crowd-20270101-000000-s0"
    assert expect_status(server, f"/api/crowd/{run_id}") == 404
    make_crowd_run(builds, run_id)
    status, payload = get_json(server, f"/api/crowd/{run_id}")
    assert status == 200
    assert payload["run_id"] == run_id


def test_new_artifacts_enter_the_browsable_index_on_refresh(server, builds):
    """The index is a TTL cache over a directory scan, so browsing catches up
    on its own; `?refresh=1` is the impatient path and the one the UI's reload
    button uses."""
    from vizfixtures import write_build

    _, before = get_json(server, "/api/builds")
    write_build(
        builds,
        "idea__later",
        {
            "idea_id": "idea",
            "model": "google-vertex/gemini-9-future",
            "structure": "dynamic",
            "collab": "local",
            "status": "ok",
            "created_at": "2027-01-01T00:00:00+00:00",
        },
        {},
    )
    make_crowd_run(builds, "idea__later__crowd-20270101-000000-s0")

    _, stale = get_json(server, "/api/builds")
    assert stale["total"] == before["total"], "cached index, as designed"

    _, fresh = get_json(server, "/api/builds?refresh=1")
    assert fresh["total"] == before["total"] + 1
    ids = [b["build_id"] for b in fresh["builds"]]
    assert "idea__later" in ids
    # Newest first, and the new build's crowd run was counted in the same pass.
    assert ids[0] == "idea__later"
    row = next(b for b in fresh["builds"] if b["build_id"] == "idea__later")
    assert row["crowd_runs"] == 1

    _, runs = get_json(server, "/api/crowd/runs?refresh=1&q=idea__later")
    assert [r["run_id"] for r in runs["runs"]] == [
        "idea__later__crowd-20270101-000000-s0"
    ]


def test_an_unknown_future_structure_is_listed_not_dropped(server, builds):
    """A build recorded by a pipeline arm that does not exist yet still appears.

    `classify_mode` falls through to a descriptive label rather than raising or
    filtering, so adding a fifth arm to the benchmark does not make its builds
    invisible here until the viewer is taught about it.
    """
    from vizfixtures import write_build

    write_build(
        builds,
        "idea__swarm",
        {
            "idea_id": "idea",
            "model": "google-vertex/gemini-9-future",
            "structure": "swarm",
            "collab": "local",
            "status": "ok",
            "created_at": "2027-06-01T00:00:00+00:00",
        },
        {},
    )
    _, payload = get_json(server, "/api/builds?refresh=1")
    row = next(b for b in payload["builds"] if b["build_id"] == "idea__swarm")
    assert row["mode"] == "other" and row["mode_label"] == "swarm"
    assert get_json(server, "/api/founder/idea__swarm")[0] == 200
