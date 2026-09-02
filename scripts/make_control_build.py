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

"""Create the deliberately BROKEN control app used to calibrate the ViralScore.

Every app the crowd had scored until now was a working build by a strong model,
so there was no evidence the score could recognise a bad one -- and a benchmark
that cannot identify failure cannot rank success. This generates a negative
control: an app that installs, starts, and passes its smoke test, but whose
JavaScript throws on load so nothing on the page actually works.

That combination is deliberate. It is exactly the case the coarse ``verify_code``
gate cannot catch (the server starts, ``test -f index.html`` passes), so it tests
whether the *crowd* -- which opens the app in a real browser -- notices. It does:
the agents diagnose the ReferenceError by name and score it near the floor.

Run:  uv run python scripts/make_control_build.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from viral_bench.founder.build import builds_root  # noqa: E402

BUILD_ID = "quick_notes_app__20260728-000000__brokn0"

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>QuickNotes</title></head>
<body>
  <h1>QuickNotes</h1>
  <div id="app">Loading your notes...</div>
  <script src="app.js"></script>
</body></html>
"""

# The first statement throws, so the page never renders and no control works.
APP_JS = """// Notes app bootstrap
const notes = JSON.parse(window.__PRELOADED_NOTES__);  // ReferenceError: undefined
document.getElementById('app').innerHTML = notes.map(n => n.text).join('');
"""

README = (
    "# QuickNotes\n\nA quick markdown notes app. Open index.html to start taking "
    "notes, tag them, and search.\n"
)

MANIFEST = {
    "app_type": "single-page-app",
    "title": "QuickNotes",
    "summary": (
        "A lightweight spot to jot down quick markdown notes and find them later."
    ),
    "setup": [],
    "run": {
        "command": "python3 -m http.server 8000",
        "cwd": ".",
        "port": 8000,
        "url": "http://localhost:8000/",
    },
    "test": {
        "manual": [
            "Open the URL.",
            "Type a note and save it.",
            "Search your notes.",
        ],
        "smoke": "test -f index.html",
    },
    "notes": "Pure static files.",
}


def main() -> int:
    root = builds_root() / "work" / BUILD_ID
    app = root / "app"
    app.mkdir(parents=True, exist_ok=True)

    (app / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (app / "app.js").write_text(APP_JS, encoding="utf-8")
    (app / "README.md").write_text(README, encoding="utf-8")
    (app / "viralbench.json").write_text(
        json.dumps(MANIFEST, indent=2), encoding="utf-8"
    )

    (root / "build.json").write_text(
        json.dumps(
            {
                "build_id": BUILD_ID,
                "idea_id": "quick_notes_app",
                "model": "control/broken",
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
                "qa_verified": False,
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
    print(f"created control build: {BUILD_ID}")
    print(f"  app: {app}")
    print(
        "  score it with: viral-bench crowd-run "
        f"{BUILD_ID} && viral-bench score {BUILD_ID}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
