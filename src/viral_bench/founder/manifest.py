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

"""The run/test manifest a founder-built app must emit (``viralbench.json``).

Because the founder agent is free to choose its own tech stack, the pipeline
cannot guess how to run or test the app it produced. Instead, every build must
drop a small, machine-readable manifest at the app root describing:

* what kind of app it is (``client-app`` | ``full-stack-app``),
* how to install its dependencies (``setup``),
* how to run it (``run``: a command, plus a port/URL for web apps), and
* how to test/interact with it (``test``: manual steps + an optional automated
  smoke command).

This manifest is the contract the :mod:`viral_bench.founder.runner` (and, later,
the OASIS crowd's ``verify_code`` / ``try_app`` tools) consume to actually
exercise the app. Keeping it tiny and validated means a malformed build fails
loudly instead of silently producing something no one can run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from viral_bench.ideas import ALLOWED_SCOPES

# The founder must write the manifest to this filename at the app root.
MANIFEST_FILENAME = "viralbench.json"

# App types ARE the Idea Bench's ``allowed_scope`` enum, imported rather than
# restated: the build prompt tells the model `app_type` must equal the idea's
# scope, so two hand-maintained copies of the same three strings could drift into
# a contradiction that nothing would catch.
ALLOWED_APP_TYPES = ALLOWED_SCOPES


class ManifestError(ValueError):
    """Raised when an app's ``viralbench.json`` does not match the schema."""


@dataclass(frozen=True)
class RunSpec:
    """How to start the app.

    Attributes:
        command: Shell command that starts the app, run from ``cwd``.
        cwd: Directory to run ``command`` in, relative to the app root.
        port: TCP port the app listens on (web apps); ``None`` otherwise.
        url: URL to open once running (web apps); ``None`` otherwise.
    """

    command: str
    cwd: str = "."
    port: int | None = None
    url: str | None = None


@dataclass(frozen=True)
class TestSpec:
    """How to test/interact with the app.

    Attributes:
        manual: Human/agent-readable steps to try the app.
        smoke: Optional command that exits 0 iff the app is healthy.
    """

    manual: tuple[str, ...] = ()
    smoke: str | None = None


@dataclass(frozen=True)
class Manifest:
    """A validated ``viralbench.json`` describing how to run and test an app."""

    app_type: str
    title: str
    summary: str
    run: RunSpec
    setup: tuple[str, ...] = ()
    test: TestSpec = field(default_factory=TestSpec)
    notes: str | None = None


def _require_str(raw: dict, key: str, *, source: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{source}: '{key}' must be a non-empty string")
    return value


def _optional_str_list(raw: dict, key: str, *, source: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if value in (None, []):
        return ()
    if not isinstance(value, list) or not all(
        isinstance(v, str) and v.strip() for v in value
    ):
        raise ManifestError(f"{source}: '{key}' must be a list of non-empty strings")
    return tuple(value)


def _parse_run(raw: object, *, source: str) -> RunSpec:
    if not isinstance(raw, dict):
        raise ManifestError(f"{source}: 'run' must be a mapping")
    command = _require_str(raw, "command", source=f"{source}.run")

    cwd = raw.get("cwd", ".")
    if not isinstance(cwd, str) or not cwd.strip():
        raise ManifestError(f"{source}: 'run.cwd' must be a non-empty string")

    port = raw.get("port")
    if port is not None:
        if not isinstance(port, int) or isinstance(port, bool):
            raise ManifestError(f"{source}: 'run.port' must be an integer")
        if not (1 <= port <= 65535):
            raise ManifestError(f"{source}: 'run.port' must be in 1..65535")

    url = raw.get("url")
    if url is not None and (not isinstance(url, str) or not url.strip()):
        raise ManifestError(f"{source}: 'run.url' must be a non-empty string")

    return RunSpec(command=command, cwd=cwd, port=port, url=url)


def _parse_test(raw: object, *, source: str) -> TestSpec:
    if raw is None:
        return TestSpec()
    if not isinstance(raw, dict):
        raise ManifestError(f"{source}: 'test' must be a mapping")
    manual = _optional_str_list(raw, "manual", source=f"{source}.test")
    smoke = raw.get("smoke")
    if smoke is not None and (not isinstance(smoke, str) or not smoke.strip()):
        raise ManifestError(f"{source}: 'test.smoke' must be a non-empty string")
    return TestSpec(manual=manual, smoke=smoke)


def parse_manifest(raw: object, *, source: str = "viralbench.json") -> Manifest:
    """Validate a raw mapping and return a :class:`Manifest`.

    Args:
        raw: Object parsed from ``viralbench.json`` (expected to be a mapping).
        source: Label used in error messages (usually the file path).

    Raises:
        ManifestError: If any field is missing or has the wrong type/value.
    """
    if not isinstance(raw, dict):
        raise ManifestError(f"{source}: top-level JSON must be a mapping")

    app_type = _require_str(raw, "app_type", source=source)
    if app_type not in ALLOWED_APP_TYPES:
        raise ManifestError(
            f"{source}: 'app_type' must be one of {sorted(ALLOWED_APP_TYPES)}, "
            f"got {app_type!r}"
        )

    if "run" not in raw:
        raise ManifestError(f"{source}: missing required field 'run'")

    return Manifest(
        app_type=app_type,
        title=_require_str(raw, "title", source=source),
        summary=_require_str(raw, "summary", source=source),
        run=_parse_run(raw["run"], source=source),
        setup=_optional_str_list(raw, "setup", source=source),
        test=_parse_test(raw.get("test"), source=source),
        notes=raw.get("notes") if isinstance(raw.get("notes"), str) else None,
    )


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate a single ``viralbench.json`` file."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"{path}: manifest not found") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: invalid JSON ({exc})") from exc
    return parse_manifest(raw, source=str(path))


def example_manifest_json(app_type: str = "client-app") -> str:
    """Return a filled-in example manifest, used in the founder's build prompt.

    The example is intentionally concrete (not just a schema) so the model has a
    clear target to imitate for the given ``app_type``.

    Treat this as load-bearing, not illustrative. Measured over 126 shipped
    builds, 58% used the example's port, 28% copied its smoke string verbatim,
    24% reused its "Pure static files" prose and 18% shipped its run command
    byte-identically. Whatever this function shows is what the fleet builds, so a
    detail that is wrong here is wrong everywhere.
    """
    examples: dict[str, dict] = {
        "client-app": {
            "app_type": "client-app",
            "title": "TileMerge",
            "summary": "A 2048-style sliding tile puzzle that runs in the browser.",
            "setup": [],
            "run": {
                "command": "python3 -m http.server 8000 --bind 0.0.0.0",
                "cwd": ".",
                "port": 8000,
                "url": "http://localhost:8000/",
            },
            "test": {
                "manual": [
                    "Open the URL in a browser.",
                    "Use arrow keys to slide tiles; equal tiles merge.",
                    "Confirm the score increases and 'best' persists on reload.",
                    "Play until game-over, then click Restart.",
                ],
                "smoke": "curl -fsS http://localhost:8000/",
            },
            "notes": "Static files; state kept in localStorage. No backend.",
        },
        "full-stack-app": {
            "app_type": "full-stack-app",
            "title": "QuickMemo",
            "summary": "A shared memo board with accounts and a public timeline.",
            "setup": ["uv sync", "uv run python migrate.py"],
            "run": {
                "command": "uv run uvicorn app.main:app --host 0.0.0.0 --port 8000",
                "cwd": ".",
                "port": 8000,
                "url": "http://localhost:8000/",
            },
            "test": {
                "manual": [
                    "Open the URL and create an account.",
                    "Post a public memo and confirm it appears on the timeline.",
                    "Sign in as a SECOND user; confirm the first user's public memo"
                    " is visible and their private one is not.",
                    "Restart the app and confirm both accounts and memos survive.",
                ],
                "smoke": "curl -fsS http://localhost:8000/healthz",
            },
            "notes": "SQLite at $VIRALBENCH_DATA_DIR/app.db (i.e. /data/app.db) so "
            "state survives a restart; schema created by the setup step; WAL mode "
            "for concurrent readers.",
        },
    }
    chosen = examples.get(app_type, examples["client-app"])
    return json.dumps(chosen, indent=2)
