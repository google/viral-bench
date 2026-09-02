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

"""Rescue crowd screenshots out of ``/tmp`` before something deletes them.

The crowd's browser never had a screenshot directory configured, so every image an
agent captured landed in ``tempfile.gettempdir()/viralbench-shots`` -- currently
34,570 files, 3.7 GB, referenced from trace steps by absolute path. They are the
only visual record of an agent using the app, and ``/tmp`` will not keep them.

So: serve from wherever the file still is, and copy each one into ``viz/cache`` the
first time it is viewed. A run you have looked at once keeps its pictures. Nothing
is written outside the cache, and nothing is deleted from ``/tmp``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .paths import assert_writable, cache_dir

SHOT_DIRNAME = "viralbench-shots"


def source_dirs() -> list[Path]:
    """Places a crowd screenshot may still be found, in search order."""
    return [Path(tempfile.gettempdir()) / SHOT_DIRNAME, Path("/tmp") / SHOT_DIRNAME]


def cached_shot(name: str) -> Path:
    return cache_dir("shots") / name


def resolve_shot(name: str, *, rescue: bool = True) -> Path | None:
    """Find one screenshot by basename, copying it into the cache when found.

    Only a bare filename is accepted -- these names come out of JSON that records
    absolute paths from other machines, and joining an untrusted path against a
    directory is how a viewer turns into a file-disclosure bug.
    """
    name = Path(name).name
    if not name or name.startswith("."):
        return None

    cached = cached_shot(name)
    if cached.is_file():
        return cached

    for base in source_dirs():
        candidate = base / name
        if candidate.is_file():
            if rescue:
                try:
                    shutil.copy2(candidate, assert_writable(cached))
                    return cached
                except OSError:
                    pass
            return candidate
    return None


def rescue_run(run_dir: Path) -> dict:
    """Copy every screenshot a run references into the cache.

    Used to pin the curated demo runs so their images survive a reboot, and
    available from the UI for any run worth keeping.
    """
    import json

    trace_dir = run_dir / "traces"
    if not trace_dir.is_dir():
        return {"referenced": 0, "rescued": 0, "missing": 0}

    referenced = rescued = missing = 0
    for path in sorted(trace_dir.glob("agent_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            continue
        for step in (payload.get("trace") or {}).get("steps") or []:
            shot = step.get("screenshot") if isinstance(step, dict) else None
            if not shot:
                continue
            referenced += 1
            name = Path(shot).name
            if cached_shot(name).is_file():
                rescued += 1
                continue
            source = Path(shot)
            if not source.is_file():
                found = resolve_shot(name, rescue=False)
                source = found if found else source
            if source.is_file():
                try:
                    shutil.copy2(source, assert_writable(cached_shot(name)))
                    rescued += 1
                except OSError:
                    missing += 1
            else:
                missing += 1
    return {"referenced": referenced, "rescued": rescued, "missing": missing}
