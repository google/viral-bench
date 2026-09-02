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

"""Shared fixtures for the crowd interaction tests.

These build *real* tiny apps through the same fake-harness path the founder tests
use, so the interaction layer is exercised end-to-end against actual running apps
in host mode -- no mocks of the thing under test.

Every app here is a web app, because the bench is: the scope enum is
``{client-app, full-stack-app}`` and both are served over HTTP and driven in a
browser. The former ``cli`` and ``bot`` fixtures are gone with the app types they
built.
"""

from __future__ import annotations

import json
import socket

import pytest

from viral_bench.founder.build import run_build
from viral_bench.founder.harness import HarnessResult, PhaseResult


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FilesHarness:
    """A fake founder harness that ships caller-supplied files + a manifest.

    ``manifest=None`` writes no manifest at all and ``raw_manifest`` writes a
    literal string, so tests can reproduce the two ways a real build arrives
    undeliverable: the file is missing, or it is there and does not parse.
    """

    def __init__(
        self,
        files: dict[str, str],
        manifest: dict | None,
        raw_manifest: str | None = None,
    ) -> None:
        self.files = files
        self.manifest = manifest
        self.raw_manifest = raw_manifest

    def run(self, idea, workspace) -> HarnessResult:
        for name, content in self.files.items():
            path = workspace.app_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        if self.raw_manifest is not None:
            (workspace.app_dir / "viralbench.json").write_text(
                self.raw_manifest, encoding="utf-8"
            )
        elif self.manifest is not None:
            (workspace.app_dir / "viralbench.json").write_text(
                json.dumps(self.manifest), encoding="utf-8"
            )
        return HarnessResult(
            model="fake",
            phases=[PhaseResult("build", 0, workspace.app_dir / "t", 1.0)],
        )


def build_app(files: dict[str, str], manifest: dict) -> str:
    record = run_build("sliding_tile_game", harness=FilesHarness(files, manifest))
    assert record.status == "ok", record.error
    return record.build_id


def build_undeliverable(
    files: dict[str, str] | None = None, raw_manifest: str | None = None
) -> str:
    """A build with real app files but no usable ``viralbench.json``."""
    record = run_build(
        "sliding_tile_game",
        harness=FilesHarness(
            files or {"index.html": "<h1>hi</h1>"}, None, raw_manifest=raw_manifest
        ),
    )
    assert record.status in {"manifest_missing", "manifest_invalid"}, record.status
    return record.build_id


# -- app builders -----------------------------------------------------------

_SPA_HTML = """<!doctype html><html><head><title>Playground</title></head><body>
<h1>Playground</h1>
<p>Count: <span id="count">0</span></p>
<button id="inc">Increment</button>
<input id="name" placeholder="your name">
<button id="greet">Greet</button>
<p id="greeting"></p>
<script src="app.js"></script>
</body></html>"""

_SPA_JS = """
let c = 0;
document.getElementById('inc').onclick = () => {
  c++; document.getElementById('count').textContent = String(c);
  console.log('count', c);
};
document.getElementById('greet').onclick = () => {
  const n = document.getElementById('name').value;
  document.getElementById('greeting').textContent = 'Hello ' + n + '!';
};
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowUp') {
    c++; document.getElementById('count').textContent = String(c);
  }
});
"""


def make_web_manifest(port: int, app_type: str = "client-app") -> dict:
    return {
        "app_type": app_type,
        "title": "Playground",
        "summary": "A tiny interactive playground.",
        "run": {
            "command": f"python3 -m http.server {port}",
            "port": port,
            "url": f"http://localhost:{port}/",
        },
        "test": {"manual": ["click Increment"], "smoke": "test -f index.html"},
    }


@pytest.fixture
def builds_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def web_build(builds_dir) -> str:
    """A running ``client-app``: static files served over HTTP."""
    return build_app(
        {"index.html": _SPA_HTML, "app.js": _SPA_JS},
        make_web_manifest(free_port()),
    )


@pytest.fixture
def full_stack_build(builds_dir) -> str:
    """The same app declared as a ``full-stack-app``.

    The scope survives as a label for analysis, not as a code path -- this
    fixture exists so tests can prove the two scopes are driven identically.
    """
    return build_app(
        {"index.html": _SPA_HTML, "app.js": _SPA_JS},
        make_web_manifest(free_port(), app_type="full-stack-app"),
    )


@pytest.fixture
def undeliverable_build(builds_dir) -> str:
    """A build with real app files and NO ``viralbench.json``."""
    return build_undeliverable()


@pytest.fixture
def invalid_manifest_build(builds_dir) -> str:
    """A build whose manifest is present but does not parse.

    Reproduces the commonest real failure: an unescaped double quote inside a
    JSON string value, which three of one model's seven failed builds hit.
    """
    return build_undeliverable(
        raw_manifest='{"app_type": "client-app", "run": {"command": "echo "hi" there"}}'
    )
