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

"""Bundle a build's or a run's JSON into one downloadable file.

The raw artifacts already live in the build folder -- ``builds/work/<id>/`` for the
founder side, ``builds/crowd/<id>__crowd-*/`` for the crowd -- and the UI shows
those paths so they can be copied straight off disk. This module is the other
route: one download that carries everything, so a trajectory can be handed to
someone who does not have the builds tree.

Founder bundles include the transcripts as parsed event arrays rather than raw
NDJSON, because that is the form anything downstream wants, and base64
screenshot attachments are dropped -- they can be 93% of a transcript's bytes and
the images are exported separately.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from .crowd import load_run, load_trial
from .founder import iter_events, load_trajectory
from .paths import founder_paths, read_json
from .trace import load_trace


def _strip_attachments(event: dict) -> dict:
    """Replace inline base64 image payloads with a note about where the file is."""
    state = (event.get("part") or {}).get("state")
    if isinstance(state, dict) and state.get("attachments"):
        state["attachments"] = [
            {
                "mime": a.get("mime"),
                "inline_bytes": len(a.get("url") or ""),
                "note": "base64 omitted from export, see screenshots/",
            }
            for a in state["attachments"]
        ]
    return event


def trajectory_bundle(builds_root: Path, build_id: str) -> dict | None:
    """The portable trajectory: manifest plus ordered event stream.

    Emits the shape ``viral-bench trajectory`` writes -- ``trajectory.json`` and
    ``events.jsonl``, schema version 1 -- so a bundle can be produced from the
    viewer without the CLI, and anything already consuming that format reads this
    one unchanged.
    """
    record = read_json(founder_paths(builds_root, build_id).build_json)
    if not isinstance(record, dict):
        return None
    traced = load_trace(builds_root, build_id, record)
    manifest = dict(traced["manifest"])
    manifest.pop("bundle_dir", None)
    # Strip the reader's own bookkeeping so the stream matches the schema exactly.
    events = []
    for event in traced["events"]:
        clean = {
            k: v for k, v in event.items() if k not in ("part_id", "unknown_session")
        }
        events.append(clean)
    return {"manifest": manifest, "events": events}


def founder_bundle(builds_root: Path, build_id: str) -> dict | None:
    """Everything about one founder build, as a single JSON document."""
    paths = founder_paths(builds_root, build_id)
    record = read_json(paths.build_json)
    if not isinstance(record, dict):
        return None
    trajectory = load_trajectory(builds_root, build_id) or {}

    transcripts = {}
    if paths.transcript_dir.is_dir():
        for path in sorted(paths.transcript_dir.glob("*.json")):
            transcripts[path.stem] = [
                _strip_attachments(e) for _, e in iter_events(path)
            ]

    portable = trajectory_bundle(builds_root, build_id) or {
        "manifest": {},
        "events": [],
    }
    return {
        "kind": "viral_bench.founder_trajectory",
        "version": 1,
        "build_id": build_id,
        "exported_from": str(paths.root),
        "build": record,
        # The same manifest + stream `viral-bench trajectory` writes, so this
        # single file is a superset of the portable bundle rather than a rival.
        "trajectory": portable["manifest"],
        "events": portable["events"],
        "mode": trajectory.get("mode"),
        "mode_label": trajectory.get("mode_label"),
        "lanes": trajectory.get("lanes"),
        "totals": trajectory.get("totals"),
        "span": trajectory.get("span"),
        "tool_counts": trajectory.get("tool_counts"),
        "spawns": trajectory.get("spawns"),
        "authored_agents": trajectory.get("authored_agents"),
        "thinking_note": trajectory.get("thinking_note"),
        "timeline": trajectory.get("events"),
        "transcripts": transcripts,
    }


def founder_zip(builds_root: Path, build_id: str) -> bytes | None:
    """A zip of the raw artifacts exactly as they sit in the build folder."""
    paths = founder_paths(builds_root, build_id)
    if not paths.build_json.is_file():
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(paths.build_json, f"{build_id}/build.json")
        for path in (
            sorted(paths.transcript_dir.glob("*"))
            if paths.transcript_dir.is_dir()
            else []
        ):
            if path.is_file():
                archive.write(path, f"{build_id}/transcript/{path.name}")
        if paths.screenshots_dir.is_dir():
            for path in sorted(paths.screenshots_dir.glob("*.png")):
                archive.write(path, f"{build_id}/screenshots/{path.name}")
        if paths.agents_dir.is_dir():
            for path in sorted(paths.agents_dir.glob("*.md")):
                archive.write(path, f"{build_id}/authored_agents/{path.name}")
        portable = trajectory_bundle(builds_root, build_id)
        if portable:
            # The canonical pair, named and shaped exactly as the CLI writes them.
            archive.writestr(
                f"{build_id}/trajectory.json",
                json.dumps(portable["manifest"], indent=2, ensure_ascii=False),
            )
            archive.writestr(
                f"{build_id}/events.jsonl",
                "".join(
                    json.dumps(e, ensure_ascii=False) + "\n" for e in portable["events"]
                ),
            )
        bundle = founder_bundle(builds_root, build_id)
        if bundle:
            archive.writestr(
                f"{build_id}/build_bundle.json",
                json.dumps(bundle, indent=2, ensure_ascii=False),
            )
    return buffer.getvalue()


def crowd_bundle(run_dir: Path) -> dict | None:
    """One crowd run as a single JSON document, trials included."""
    run = load_run(run_dir)
    if run is None:
        return None
    trials = {}
    for agent in run["agents"]:
        if not agent.get("has_trace"):
            continue
        trial = load_trial(run_dir, agent["id"])
        if trial:
            trials[str(agent["id"])] = trial
    return {
        "kind": "viral_bench.crowd_trajectory",
        "version": 1,
        "run_id": run["run_id"],
        "build_id": run["build_id"],
        "exported_from": str(run_dir),
        "run": run,
        "trials": trials,
        "trajectories": read_json(run_dir / "trajectories.json", []),
        "result": read_json(run_dir / "result.json", {}),
    }


def crowd_zip(run_dir: Path) -> bytes | None:
    """A zip of the run directory's JSON plus whatever screenshots resolved."""
    from .cache import resolve_shot

    if not (run_dir / "run_summary.json").is_file():
        return None
    name = run_dir.name
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in (
            "run_summary.json",
            "result.json",
            "trajectories.json",
            "autorating.json",
            "actions.jsonl",
        ):
            path = run_dir / filename
            if path.is_file():
                archive.write(path, f"{name}/{filename}")
        trace_dir = run_dir / "traces"
        if trace_dir.is_dir():
            for path in sorted(trace_dir.glob("agent_*.json")):
                archive.write(path, f"{name}/traces/{path.name}")
        bundle = crowd_bundle(run_dir)
        if bundle:
            archive.writestr(
                f"{name}/trajectory.json",
                json.dumps(bundle, indent=2, ensure_ascii=False),
            )
            seen = set()
            for trial in bundle["trials"].values():
                for step in trial["steps"]:
                    shot = step.get("screenshot")
                    if not shot or shot in seen:
                        continue
                    seen.add(shot)
                    resolved = resolve_shot(shot)
                    if resolved and resolved.is_file():
                        archive.write(resolved, f"{name}/screenshots/{shot}")
    return buffer.getvalue()


def rubric_bundle(run_dir: Path) -> dict | None:
    """One rubric grade as a single JSON document, transcript included.

    The grade is already self-describing, so unlike the crowd bundle this adds
    little beyond the transcript -- which is the part the API otherwise serves
    lazily, and the part that makes a verdict auditable away from this machine.
    """
    from .rubric import load_grade, load_transcript

    grade = load_grade(run_dir)
    if grade is None:
        return None
    transcript = load_transcript(run_dir, offset=0, limit=100_000)
    return {
        "kind": "viral_bench.rubric_grade",
        "version": 1,
        "run_id": grade["run_id"],
        "build_id": grade.get("build_id") or "",
        "exported_from": str(run_dir),
        "grade": grade,
        "transcript": transcript.get("calls") or [],
    }


def rubric_zip(run_dir: Path) -> bytes | None:
    """A zip of the grade directory: the document, the log and the screenshots."""
    if not (run_dir / "grade.json").is_file():
        return None
    name = run_dir.name
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in ("grade.json", "transcript.jsonl"):
            path = run_dir / filename
            if path.is_file():
                archive.write(path, f"{name}/{filename}")
        shots = run_dir / "shots"
        if shots.is_dir():
            for path in sorted(shots.iterdir()):
                if path.is_file():
                    archive.write(path, f"{name}/shots/{path.name}")
        bundle = rubric_bundle(run_dir)
        if bundle:
            archive.writestr(
                f"{name}/grade_bundle.json",
                json.dumps(bundle, indent=2, ensure_ascii=False, default=str),
            )
    return buffer.getvalue()
