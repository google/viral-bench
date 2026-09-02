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

"""Assemble one founder build's full trajectory into a portable bundle.

A build record says *what* a model shipped. This says *how* it got there: the
prompts it was given, what it thought, what it ran, what it wrote, and -- in the
dynamic arm -- every subagent it decided to create and what each of those did.
That is the artifact a trajectory viewer renders and a supervised-finetuning set
is built from, and neither can be reconstructed from the shipped app.

**Where the pieces come from, and why it takes two sources.**

``opencode run --format json`` writes one JSON event per line to stdout, and the
harness keeps that per turn as ``transcript/<phase>.json``. It is the record of
what the *root* agent emitted, and with ``--thinking`` on it now includes the
model's reasoning. But three things are missing from it by construction:

* the prompt. It goes to opencode on stdin and is never echoed back as an event,
  so the stdout transcript has the model's answers and not the questions.
* every subagent. opencode's printer drops any event whose ``sessionID`` is not
  the root session, so a dynamic founder's delegated work -- the thing that arm
  exists to measure -- leaves no trace in stdout at all.
* file patches, which are never printed.

All three are in opencode's own SQLite session store, which the harness dumps
per turn to ``transcript/sessions/<session_id>.jsonl`` (see
:func:`~viral_bench.founder.harness.dump_session_trace`). So the dump is the
primary source here and the stdout transcript is the fallback: a build made
before the dump existed still exports, only without prompts or subagents.

**The output.** A directory (or zip) holding:

``trajectory.json``   manifest: schema version, the build's identity and arm, the
                      session tree, per-turn index, and token/cost rollups.
``events.jsonl``      one time-ordered record per event, each self-contained --
                      session, depth, agent, role, model, type, payload -- so it
                      streams without holding the file in memory.

Both are stable, documented shapes owned by this repo, deliberately not
opencode's internal one: the store is undocumented and free to change, and a
downstream consumer should not be pinned to it.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from viral_bench.founder.workspace import builds_root

__all__ = [
    "SCHEMA_VERSION",
    "TrajectoryError",
    "build_trajectory",
    "export_trajectory",
    "iter_events",
    "reasoning_from_dumps",
]

#: Bump when the shape of ``trajectory.json`` / ``events.jsonl`` changes in a way
#: a reader must notice. Written into every manifest so a consumer can refuse a
#: bundle it does not understand rather than misreading it.
SCHEMA_VERSION = 1

#: Part types carried through, in the order they matter to a reader. Anything
#: else opencode invents later is passed through untouched rather than dropped --
#: an unknown part is still evidence.
_TEXTUAL_PARTS = ("reasoning", "text")


class TrajectoryError(RuntimeError):
    """Raised when a build's trajectory cannot be assembled."""


@dataclass
class Trajectory:
    """One build's manifest plus its ordered event stream."""

    manifest: dict[str, Any]
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def build_id(self) -> str:
        return str(self.manifest.get("build_id", ""))


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _iter_jsonl(path: Path) -> Iterator[dict]:
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _tool_payload(part: dict) -> dict:
    """Flatten an opencode ``tool`` part into a stable, readable shape."""
    state = part.get("state") if isinstance(part.get("state"), dict) else {}
    times = state.get("time") if isinstance(state.get("time"), dict) else {}
    output = state.get("output")
    return {
        "name": part.get("tool") or "",
        "status": state.get("status") or "",
        "input": state.get("input") if isinstance(state.get("input"), dict) else {},
        "output": output if isinstance(output, str) else "",
        "error": str(state.get("error") or ""),
        "metadata": (
            state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
        ),
        "start_ms": times.get("start"),
        "end_ms": times.get("end"),
    }


def _event_from_part(record: dict, *, depth: int, seq: int) -> dict | None:
    """Turn one dumped store ``part`` record into a trajectory event."""
    part = record.get("part")
    if not isinstance(part, dict):
        return None
    kind = str(part.get("type") or "")
    event: dict[str, Any] = {
        "seq": seq,
        "ts": record.get("time_created"),
        "session_id": record.get("session_id", ""),
        "depth": depth,
        "message_id": record.get("message_id", ""),
        "role": record.get("role", ""),
        "agent": record.get("agent", ""),
        "model": record.get("model", ""),
        "type": kind,
    }
    if kind in _TEXTUAL_PARTS:
        text = part.get("text")
        event["text"] = text if isinstance(text, str) else ""
        # Anthropic returns an encrypted thinking block when the request did not
        # ask for a summary: signature present, text empty. Flag it rather than
        # emitting a silently blank thought, so a consumer can tell "did not
        # think" from "thought, but it could not be read".
        metadata = part.get("metadata")
        if kind == "reasoning" and not event["text"] and isinstance(metadata, dict):
            event["redacted"] = bool(metadata)
    elif kind == "tool":
        event["tool"] = _tool_payload(part)
    elif kind == "patch":
        event["patch"] = {
            key: part.get(key) for key in ("hash", "files") if key in part
        }
    else:
        # step-start / step-finish / anything opencode adds later.
        event["payload"] = {
            k: v for k, v in part.items() if k not in ("id", "sessionID", "messageID")
        }
    return event


def reasoning_from_dumps(session_dir: Path) -> dict:
    """Deduped reasoning totals across every session dumped for one build.

    The count a build record needs, and the reason it cannot be taken from the
    stdout transcript: opencode's printer drops every event below the root
    session, so a transcript-derived total silently omits every subagent. On a
    dynamic build with four subagents that ran about 3x low.

    Deliberately implemented on top of :func:`_events_from_sessions` rather than
    counting the JSONL directly. Every turn re-dumps its whole session subtree,
    so the same part appears in several files and a naive sum double-counts --
    the solo build turn's dump already contains the design turn's reasoning.
    That dedup (by ``part_id``) exists once, here, and nowhere else.

    Returns zeros when nothing was dumped, which a caller must not read as "the
    model did not think": :func:`~viral_bench.founder.harness.dump_session_trace`
    is best-effort and returns nothing on any failure. That is exactly why the
    build record keeps the transcript-derived count alongside this one.
    """
    empty = {"parts": 0, "chars": 0, "redacted": 0, "records": 0, "events": 0}
    if not session_dir.is_dir():
        return empty
    paths = sorted(session_dir.glob("*.jsonl"))
    if not paths:
        return empty
    _, events = _events_from_sessions(paths)
    reasoning = [e for e in events if e.get("type") == "reasoning"]

    # Distinct lines across the dumps, by (kind, id). `events` counts only the
    # part records -- what the exported stream will hold -- so it is not the
    # same number as "how much is in the dump", and conflating the two is how
    # the double-count got missed the first time.
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for record in _iter_jsonl(path):
            kind = str(record.get("record") or "")
            key = record.get(
                {"session": "session_id", "message": "message_id"}.get(kind, "part_id")
            )
            seen.add((kind, str(key)))

    return {
        "parts": sum(1 for e in reasoning if (e.get("text") or "").strip()),
        "chars": sum(len(e.get("text") or "") for e in reasoning),
        "redacted": sum(1 for e in reasoning if e.get("redacted")),
        "records": len(seen),
        "events": len(events),
    }


def _events_from_sessions(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Read dumped session stores into (session records, ordered events)."""
    sessions: dict[str, dict] = {}
    parts: list[dict] = []
    for path in paths:
        for record in _iter_jsonl(path):
            kind = record.get("record")
            if kind == "session":
                sessions.setdefault(str(record.get("session_id") or ""), record)
            elif kind == "part":
                parts.append(record)

    # Depth is what makes a delegating build readable: 0 is the founder, 1 a
    # subagent it spawned, 2 a subagent of that. Derived from parent links so it
    # survives sessions arriving from several per-turn dumps.
    depth_of: dict[str, int] = {}

    def depth(session_id: str, guard: int = 0) -> int:
        if session_id in depth_of:
            return depth_of[session_id]
        record = sessions.get(session_id)
        parent = str((record or {}).get("parent_session_id") or "")
        # guard: a malformed store must not recurse forever.
        value = 0 if not parent or guard > 32 else depth(parent, guard + 1) + 1
        depth_of[session_id] = value
        return value

    seen: set[str] = set()
    events: list[dict] = []
    parts.sort(key=lambda r: (r.get("time_created") or 0, str(r.get("part_id") or "")))
    for record in parts:
        part_id = str(record.get("part_id") or "")
        # Each turn re-dumps its whole subtree, so the same part legitimately
        # appears in several files, so keep the first.
        if part_id and part_id in seen:
            continue
        seen.add(part_id)
        event = _event_from_part(
            record, depth=depth(str(record.get("session_id") or "")), seq=len(events)
        )
        if event is not None:
            events.append(event)

    session_list = []
    for session_id, record in sessions.items():
        session_list.append(
            {
                "session_id": session_id,
                "parent_session_id": record.get("parent_session_id", ""),
                "title": record.get("title", ""),
                "agent": record.get("agent", ""),
                "model": record.get("model", ""),
                "depth": depth(session_id),
                "cost": record.get("cost"),
                "tokens": record.get("tokens", {}),
            }
        )
    session_list.sort(key=lambda s: (s["depth"], s["session_id"]))
    return session_list, events


def _events_from_transcripts(phases: list[dict]) -> list[dict]:
    """Fallback: rebuild events from the stdout transcripts alone.

    Used for a build made before the session dump existed. It recovers the root
    agent's reasoning, text and tool calls -- but not the prompts and not any
    subagent, because those were never written to stdout. The manifest says so
    via ``source: "transcript"`` so nobody mistakes a partial trace for a full
    one.
    """
    events: list[dict] = []
    for phase in phases:
        path = Path(str(phase.get("transcript") or ""))
        if not path.is_file():
            continue
        for record in _iter_jsonl(path):
            kind = str(record.get("type") or "")
            part = record.get("part")
            if not isinstance(part, dict):
                continue
            event: dict[str, Any] = {
                "seq": len(events),
                "ts": record.get("timestamp"),
                "session_id": record.get("sessionID", ""),
                "depth": 0,
                "message_id": part.get("messageID", ""),
                "role": "assistant",
                "agent": str(phase.get("role") or ""),
                "model": "",
                "type": str(part.get("type") or kind),
            }
            if event["type"] in _TEXTUAL_PARTS:
                text = part.get("text")
                event["text"] = text if isinstance(text, str) else ""
            elif event["type"] == "tool":
                event["tool"] = _tool_payload(part)
            else:
                event["payload"] = {
                    k: v
                    for k, v in part.items()
                    if k not in ("id", "sessionID", "messageID")
                }
            events.append(event)
    return events


def _totals(events: list[dict], sessions: list[dict]) -> dict:
    """Rollups a reader wants before deciding whether to open the stream."""
    by_type: dict[str, int] = {}
    reasoning_chars = 0
    redacted = 0
    for event in events:
        by_type[event["type"]] = by_type.get(event["type"], 0) + 1
        if event["type"] == "reasoning":
            reasoning_chars += len(event.get("text") or "")
            redacted += 1 if event.get("redacted") else 0

    def total(key: str) -> int:
        return sum(
            int((s.get("tokens") or {}).get(key) or 0)
            for s in sessions
            if isinstance(s.get("tokens"), dict)
        )

    return {
        "events": len(events),
        "events_by_type": by_type,
        "reasoning_chars": reasoning_chars,
        "reasoning_redacted": redacted,
        "sessions": len(sessions),
        "max_depth": max((s["depth"] for s in sessions), default=0),
        "tokens": {
            "input": total("input"),
            "output": total("output"),
            "reasoning": total("reasoning"),
        },
        "cost": round(
            sum(float(s.get("cost") or 0) for s in sessions),
            6,
        ),
    }


def build_trajectory(build_id: str, *, root: Path | None = None) -> Trajectory:
    """Assemble the trajectory for ``build_id`` from its workspace.

    Reads the dumped session stores when present, and falls back to the stdout
    transcripts otherwise. Raises :class:`TrajectoryError` only when there is no
    build to read at all -- a build with nothing captured yields an empty stream
    with a manifest that says why, which is more useful than an exception.
    """
    work = (root or builds_root() / "work") / build_id
    record_path = work / "build.json"
    if not record_path.is_file():
        raise TrajectoryError(f"no build record for {build_id!r} at {record_path}")
    record = _load_json(record_path)

    phases = record.get("phases") if isinstance(record.get("phases"), list) else []
    session_files = sorted((work / "transcript" / "sessions").glob("*.jsonl"))

    if session_files:
        sessions, events = _events_from_sessions(session_files)
        source = "session_store"
    else:
        sessions, events = [], _events_from_transcripts(phases)
        source = "transcript"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": record.get("build_id", build_id),
        "idea_id": record.get("idea_id", ""),
        "model": record.get("model", ""),
        "created_at": record.get("created_at", ""),
        "status": record.get("status", ""),
        "brief_fingerprint": record.get("brief_fingerprint", ""),
        # The arm this build belongs to. A trajectory is only comparable with
        # another from the same arm, so it travels with the bundle.
        "arm": {
            "structure": record.get("structure", ""),
            "n_agents": record.get("n_agents"),
            "max_rounds": record.get("max_rounds"),
            "rounds_run": record.get("rounds_run"),
            "max_turns": record.get("max_turns"),
            "turns_spent": record.get("turns_spent"),
            "collab": record.get("collab", ""),
            "roles": record.get("roles", []),
            "subagents_spawned": record.get("subagents_spawned", 0),
        },
        "source": source,
        "capture": record.get("trajectory") or {},
        "sessions": sessions,
        "turns": [
            {
                "phase": p.get("phase", ""),
                "role": p.get("role", ""),
                "turn": p.get("turn", ""),
                "agent_index": p.get("agent_index", 0),
                "session_id": p.get("session_id", ""),
                "ok": p.get("ok"),
                "timed_out": p.get("timed_out"),
                "duration_s": p.get("duration_s"),
                # Root session only -- opencode's stdout printer cannot see a
                # subagent. Suffixed so this never reads as a per-turn share of
                # `totals.reasoning_chars`, which counts every session: summing
                # these to less than the total is correct, not a discrepancy.
                # The unsuffixed fallback reads the handful of builds recorded
                # after capture landed but before the suffix did. Without it
                # they would report a confident zero instead of what they have.
                "reasoning_parts_root": p.get(
                    "reasoning_parts_root", p.get("reasoning_parts", 0)
                ),
                "reasoning_chars_root": p.get(
                    "reasoning_chars_root", p.get("reasoning_chars", 0)
                ),
            }
            for p in phases
            if isinstance(p, dict)
        ],
        "totals": _totals(events, sessions),
    }
    return Trajectory(manifest=manifest, events=events)


def iter_events(build_id: str, *, root: Path | None = None) -> Iterator[dict]:
    """Yield one build's trajectory events without materialising a bundle."""
    yield from build_trajectory(build_id, root=root).events


def export_trajectory(
    build_id: str,
    dest: Path,
    *,
    root: Path | None = None,
    as_zip: bool = False,
) -> Path:
    """Write ``build_id``'s trajectory bundle to ``dest``. Returns the path.

    With ``as_zip`` the bundle is a single ``.zip`` holding the same two files,
    which is what makes it a thing you can hand to someone.
    """
    trajectory = build_trajectory(build_id, root=root)
    manifest = json.dumps(trajectory.manifest, indent=2, ensure_ascii=False)
    events = "".join(
        json.dumps(event, ensure_ascii=False) + "\n" for event in trajectory.events
    )

    if as_zip:
        dest = dest if dest.suffix == ".zip" else dest.with_suffix(".zip")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"{build_id}/trajectory.json", manifest)
            archive.writestr(f"{build_id}/events.jsonl", events)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "trajectory.json").write_text(manifest, encoding="utf-8")
    (dest / "events.jsonl").write_text(events, encoding="utf-8")
    return dest
