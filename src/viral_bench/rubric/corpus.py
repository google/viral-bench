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

"""Which builds a rubric sweep is about.

Two eras of selection exist and both have to keep working.

**r3** was identified by a suffix on the fleet key -- ``<idea>::<model>::<arm>::r3``
-- so the generation was a property of the index entry.

**r4** is a cohort: ``builds/cohorts/r4.json`` lists its members explicitly and
each ``build.json`` carries ``"cohort": "r4"``. That is the better mechanism,
because a cohort can span two build eras -- some arms kept from the previous
generation, others rebuilt -- so no single property of a fleet key can describe
it. A label over an explicit member list can.

Selecting by cohort also makes the sweep safe against a cohort still being
filled: it grades exactly the builds the manifest names today, and grading it
again after more are tagged picks up only the new ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusBuild:
    """One build in scope, with the facts a sweep needs to plan and report."""

    build_id: str
    idea_id: str
    model: str
    arm: str
    status: str

    @property
    def deliverable(self) -> bool:
        """Whether the founder produced an app at all.

        Undeliverable builds are still graded: they land on the Tier 0 gate and
        score 0, which is the point of the gate. A benchmark that drops the
        builds that failed hardest reports the average of the survivors.
        """
        return self.status == "ok"


def _fleet_entries(builds_root: Path) -> dict:
    try:
        return json.loads((builds_root / "fleet.json").read_text())["entries"]
    except (OSError, ValueError, KeyError):
        return {}


def _by_build_id(entries: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key, entry in entries.items():
        build_id = entry.get("build_id")
        if not build_id:
            continue
        # Later keys win only if the earlier one carried no status, so a
        # re-tagged build does not lose its verdict to an empty duplicate.
        if build_id not in out or not out[build_id].get("status"):
            out[build_id] = {**entry, "_key": key}
    return out


def cohort_members(name: str, builds_root: Path) -> dict[str, str]:
    """``build_id -> arm`` for a named cohort, empty if it has no manifest."""
    path = builds_root / "cohorts" / f"{name}.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    members = manifest.get("members")
    return dict(members) if isinstance(members, dict) else {}


def select_builds(
    builds_root: Path,
    *,
    cohort: str = "",
    generation: str = "",
) -> list[CorpusBuild]:
    """The builds a sweep should grade, in a stable order.

    Pass ``cohort`` for r4 and later, ``generation`` for the r3 fleet-key era.
    Passing neither, or a cohort with no manifest yet, yields an empty list --
    deliberately, so a sweep launched against a cohort that has not been tagged
    stops rather than silently grading some other corpus.
    """
    entries = _fleet_entries(builds_root)
    indexed = _by_build_id(entries)
    found: list[CorpusBuild] = []

    if cohort:
        for build_id, arm in sorted(cohort_members(cohort, builds_root).items()):
            entry = indexed.get(build_id, {})
            found.append(
                CorpusBuild(
                    build_id=build_id,
                    idea_id=str(entry.get("idea_id") or build_id.split("__")[0]),
                    model=str(entry.get("model") or ""),
                    arm=str(arm or ""),
                    status=str(entry.get("status") or ""),
                )
            )
        return found

    if generation:
        for key, entry in sorted(entries.items()):
            if not key.endswith(f"::{generation}"):
                continue
            parts = key.split("::")
            found.append(
                CorpusBuild(
                    build_id=str(entry.get("build_id") or ""),
                    idea_id=str(entry.get("idea_id") or ""),
                    model=str(entry.get("model") or ""),
                    arm=parts[2] if len(parts) > 3 else "",
                    status=str(entry.get("status") or ""),
                )
            )
    return found
