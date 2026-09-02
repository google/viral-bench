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

"""Read what a founder build *thought*, not just what it emitted.

The benchmark now records three things per turn that the stdout transcript never
had: the model's chain of thought, the prompts it was given, and every subagent it
delegated to. This module turns those into one ordered event stream.

**Where it reads from, in precedence order.**

1. An exported bundle -- ``trajectory.json`` + ``events.jsonl``, written by
   ``viral-bench trajectory``. Authoritative when present, and used as-is.
2. ``transcript/sessions/<session_id>.jsonl`` -- opencode's own session store,
   dumped per turn by the harness. This is the normal path: it is written
   automatically on every build and is a strict superset of stdout, because it
   carries the prompts (never echoed to stdout), every subagent session (opencode's
   printer drops any event below the root session), and file patches (never
   printed at all).
3. ``transcript/<phase>.json`` -- the stdout NDJSON. The fallback for builds made
   before the dump existed. A bundle built this way is labelled ``source:
   "transcript"`` and is missing prompts, subagents, patches, and all token/cost
   figures, so the UI has to say so rather than render zeros as facts.

**This mirrors ``viral_bench.founder.trajectory`` deliberately.** That module owns
``SCHEMA_VERSION`` and the shapes; this is a reader for the same shapes that does
not import it, because the viewer has to open a build recorded by any revision of
the benchmark without being pinned to the one currently checked out. The semantics
copied here and worth not "improving": dedup by ``part_id`` keeping the first
occurrence (each turn re-dumps the whole subtree, so parts repeat across files);
a single global sort by ``(time_created, part_id)`` across every session before
``seq`` is assigned; and the reasoning tri-state below.

**Reasoning is tri-state and the states mean different things.** Non-empty
``text`` is a readable thought. Empty ``text`` with ``redacted: true`` means the
model thought and the provider encrypted it -- Vertex returns Claude's thinking
that way unless the request asks for a summary. Empty with no flag means it did not
think on that step. Collapsing those three into "no reasoning" is how a capture
regression gets mistaken for a quiet model.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .paths import founder_paths, read_json

#: The shape this reader emits. Matches ``viral_bench.founder.trajectory``; a
#: bundle declaring anything else is surfaced to the UI rather than reinterpreted.
SCHEMA_VERSION = 1

#: Part types that carry prose in ``part.text``.
_TEXTUAL_PARTS = ("reasoning", "text")

#: Depth walk cut-off. A malformed store with a parent cycle must degrade, not hang.
_MAX_DEPTH = 32


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield dict records from a JSONL file, skipping anything unreadable."""
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
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def _model_name(value: Any) -> str:
    """Normalise opencode's two ways of naming a model.

    Part records carry ``"google-vertex/gemini-2.0-flash"``, but session rows carry
    the raw column, which is a JSON *string* holding ``{"id", "providerID",
    "variant"}``. Showing that blob in a session tree would be unreadable.
    """
    if not value:
        return ""
    text = str(value)
    if not text.startswith("{"):
        return text
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    if not isinstance(parsed, dict):
        return text
    provider, model = parsed.get("providerID") or "", parsed.get("id") or ""
    return f"{provider}/{model}" if provider and model else (model or text)


def _tool_payload(part: dict) -> dict:
    """Flatten an opencode ``tool`` part into the eight keys the schema promises."""
    state = part.get("state") if isinstance(part.get("state"), dict) else {}
    times = state.get("time") if isinstance(state.get("time"), dict) else {}
    output = state.get("output")
    metadata = state.get("metadata")
    args = state.get("input")
    return {
        "name": part.get("tool") or "",
        "status": state.get("status") or "",
        "input": args if isinstance(args, dict) else {},
        "output": output if isinstance(output, str) else "",
        "error": str(state.get("error") or ""),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "start_ms": times.get("start"),
        "end_ms": times.get("end"),
    }


def _event_from_part(record: dict) -> dict | None:
    """Turn one stored ``part`` record into an event, or None if unusable."""
    part = record.get("part")
    if not isinstance(part, dict):
        return None

    event: dict[str, Any] = {
        "ts": record.get("time_created"),
        "session_id": str(record.get("session_id") or ""),
        "message_id": str(record.get("message_id") or ""),
        "role": str(record.get("role") or ""),
        "agent": str(record.get("agent") or ""),
        "model": _model_name(record.get("model")),
        "type": str(part.get("type") or ""),
        "part_id": str(record.get("part_id") or ""),
    }
    kind = event["type"]

    if kind in _TEXTUAL_PARTS:
        text = part.get("text")
        event["text"] = text if isinstance(text, str) else ""
        if kind == "reasoning" and not event["text"]:
            metadata = part.get("metadata")
            if isinstance(metadata, dict):
                # A signature with no text is the provider saying "it thought,
                # you may not read it". Distinct from "it did not think".
                event["redacted"] = bool(metadata)
    elif kind == "tool":
        event["tool"] = _tool_payload(part)
    elif kind == "patch":
        event["patch"] = {key: part[key] for key in ("hash", "files") if key in part}
    else:
        event["payload"] = {
            key: value
            for key, value in part.items()
            if key not in ("id", "sessionID", "messageID")
        }
    return event


def _session_depths(sessions: dict[str, dict]) -> dict[str, int]:
    """Depth of every session: 0 for a root, +1 per delegation level."""
    memo: dict[str, int] = {}

    def depth_of(session_id: str, guard: int = 0) -> int:
        if session_id in memo:
            return memo[session_id]
        if guard > _MAX_DEPTH:
            return 0
        parent = (sessions.get(session_id) or {}).get("parent_session_id") or ""
        value = (
            0 if not parent or parent == session_id else depth_of(parent, guard + 1) + 1
        )
        memo[session_id] = value
        return value

    return {session_id: depth_of(session_id) for session_id in sessions}


def _read_session_dumps(paths) -> tuple[dict[str, dict], list[dict]]:
    """Collect sessions and de-duplicated part events from every per-turn dump."""
    sessions: dict[str, dict] = {}
    events: list[dict] = []
    seen: set[str] = set()

    dump_dir = paths.transcript_dir / "sessions"
    if not dump_dir.is_dir():
        return sessions, events

    for path in sorted(dump_dir.glob("*.jsonl")):
        for record in _iter_jsonl(path):
            kind = record.get("record")
            if kind == "session":
                session_id = str(record.get("session_id") or "")
                # Each turn re-dumps the whole subtree; the first sighting wins.
                sessions.setdefault(
                    session_id,
                    {
                        "session_id": session_id,
                        "parent_session_id": str(record.get("parent_session_id") or ""),
                        "title": str(record.get("title") or ""),
                        "agent": str(record.get("agent") or ""),
                        "model": _model_name(record.get("model")),
                        "cost": record.get("cost"),
                        "tokens": record.get("tokens")
                        if isinstance(record.get("tokens"), dict)
                        else {},
                    },
                )
            elif kind == "part":
                part_id = str(record.get("part_id") or "")
                if part_id and part_id in seen:
                    continue
                if part_id:
                    seen.add(part_id)
                event = _event_from_part(record)
                if event is not None:
                    events.append(event)
    return sessions, events


def _events_from_transcripts(paths, phases: list[dict]) -> list[dict]:
    """Fallback: rebuild what we can from the stdout NDJSON.

    Everything here is the root agent's own output, so role is assistant and depth
    is 0 by construction. There are no prompts, no subagents and no patches to find.
    """
    events: list[dict] = []
    for phase in phases:
        name = str(phase.get("phase") or "")
        path = paths.transcript_dir / f"{name}.json"
        if not path.is_file():
            recorded = phase.get("transcript")
            path = Path(recorded) if recorded else path
        if not path.is_file():
            continue
        role_label = str(phase.get("role") or "") or "founder"
        for record in _iter_jsonl(path):
            part = record.get("part")
            if not isinstance(part, dict):
                continue
            event = _event_from_part(
                {
                    "time_created": record.get("timestamp"),
                    "session_id": record.get("sessionID"),
                    "message_id": part.get("messageID"),
                    "role": "assistant",
                    "agent": role_label,
                    "model": "",
                    "part_id": part.get("id"),
                    "part": part,
                }
            )
            if event is not None:
                event["phase"] = name
                events.append(event)
    return events


def _totals(events: list[dict], sessions: list[dict]) -> dict:
    by_type: dict[str, int] = {}
    reasoning_chars = 0
    redacted = 0
    for event in events:
        by_type[event["type"]] = by_type.get(event["type"], 0) + 1
        if event["type"] == "reasoning":
            reasoning_chars += len(event.get("text") or "")
            if event.get("redacted"):
                redacted += 1

    tokens = {"input": 0, "output": 0, "reasoning": 0}
    cost = 0.0
    for session in sessions:
        counts = session.get("tokens") or {}
        for key in tokens:
            try:
                tokens[key] += int(counts.get(key) or 0)
            except (TypeError, ValueError):
                pass
        try:
            cost += float(session.get("cost") or 0.0)
        except (TypeError, ValueError):
            pass

    return {
        "events": len(events),
        "events_by_type": by_type,
        "reasoning_chars": reasoning_chars,
        "reasoning_redacted": redacted,
        "sessions": len(sessions),
        "max_depth": max((s.get("depth", 0) for s in sessions), default=0),
        "tokens": tokens,
        "cost": round(cost, 6),
    }


def _load_bundle(bundle_dir: Path) -> dict | None:
    """Load an exported ``trajectory.json`` + ``events.jsonl`` pair."""
    manifest = read_json(bundle_dir / "trajectory.json")
    if not isinstance(manifest, dict):
        return None
    events = list(_iter_jsonl(bundle_dir / "events.jsonl"))
    return {"manifest": manifest, "events": events, "bundle_dir": str(bundle_dir)}


def bundle_dir_for(builds_root: Path, build_id: str) -> Path:
    """Where ``viral-bench trajectory`` writes a build's bundle by default."""
    return builds_root / "trajectories" / build_id


def load_trace(builds_root: Path, build_id: str, record: dict) -> dict:
    """Return ``{manifest, events}`` for one build, from the best source available.

    ``record`` is the parsed ``build.json``; it supplies the arm description and
    the harness's own capture counts, neither of which is in the event stream.
    Never raises -- a build with nothing recorded returns an empty stream with
    ``source: "transcript"``, which is the honest description of that state.
    """
    paths = founder_paths(builds_root, build_id)
    phases = [p for p in (record.get("phases") or []) if isinstance(p, dict)]

    exported = _load_bundle(bundle_dir_for(builds_root, build_id))
    if exported and exported["events"]:
        manifest = exported["manifest"]
        manifest.setdefault("source", "session_store")
        manifest["bundle_dir"] = exported["bundle_dir"]
        return {"manifest": manifest, "events": exported["events"]}

    sessions_by_id, events = _read_session_dumps(paths)
    if events or sessions_by_id:
        source = "session_store"
        depths = _session_depths(sessions_by_id)
        # One global order across every session, so a subagent's work interleaves
        # with its parent's exactly where it happened.
        events.sort(key=lambda e: (e.get("ts") or 0, e.get("part_id") or ""))
        for event in events:
            event["depth"] = depths.get(event["session_id"], 0)
            # A part whose session has no row cannot be placed in the tree. Saying
            # so beats silently drawing it as a root-level event.
            event["unknown_session"] = event["session_id"] not in sessions_by_id
    else:
        source = "transcript"
        events = _events_from_transcripts(paths, phases)
        for event in events:
            event["depth"] = 0
            event["unknown_session"] = False

    for index, event in enumerate(events):
        event["seq"] = index

    session_list = [
        {**session, "depth": _session_depths(sessions_by_id).get(sid, 0)}
        for sid, session in sessions_by_id.items()
    ]
    session_list.sort(key=lambda s: (s["depth"], s["session_id"]))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": record.get("build_id") or build_id,
        "idea_id": record.get("idea_id") or "",
        "model": record.get("model") or "",
        "created_at": record.get("created_at") or "",
        "status": record.get("status") or "",
        "brief_fingerprint": record.get("brief_fingerprint") or "",
        "arm": {
            "structure": record.get("structure") or "",
            "n_agents": record.get("n_agents"),
            "max_rounds": record.get("max_rounds"),
            "rounds_run": record.get("rounds_run"),
            "max_turns": record.get("max_turns"),
            "turns_spent": record.get("turns_spent"),
            "collab": record.get("collab") or "",
            "roles": record.get("roles") or [],
            "subagents_spawned": record.get("subagents_spawned") or 0,
        },
        "source": source,
        # The harness's own per-build tally, written at build time. Kept separate
        # from the counts derived here: if the two disagree, that is a finding.
        "capture": record.get("trajectory") or {},
        "sessions": session_list,
        "turns": [
            {
                "phase": str(phase.get("phase") or ""),
                "role": str(phase.get("role") or ""),
                "turn": str(phase.get("turn") or ""),
                "agent_index": phase.get("agent_index") or 0,
                "session_id": str(phase.get("session_id") or ""),
                "ok": phase.get("ok"),
                "timed_out": phase.get("timed_out"),
                "duration_s": phase.get("duration_s"),
                # `_root` since the counts were split by source; the bare names
                # are what builds recorded between capture landing and the split.
                "reasoning_parts_root": phase.get("reasoning_parts_root")
                or phase.get("reasoning_parts")
                or 0,
                "reasoning_chars_root": phase.get("reasoning_chars_root")
                or phase.get("reasoning_chars")
                or 0,
            }
            for phase in phases
        ],
        "totals": _totals(events, session_list),
    }
    return {"manifest": manifest, "events": events}


def _capture_int(capture: dict, *names: str) -> int:
    """First of ``names`` present in the harness's capture block, as an int.

    The build record's field names changed once, when the counts were split by
    source: ``reasoning_chars`` became ``reasoning_chars_root`` (the stdout
    transcript) alongside ``reasoning_chars_all`` (the session dumps). Builds
    exist on disk from before the split, during it, and after, so every read
    here takes the newest name it can find and falls back rather than reporting
    a confident zero for a build that recorded plenty.
    """
    for name in names:
        value = capture.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def capture_summary(manifest: dict, events: list[dict]) -> dict:
    """Was the thinking captured, how much of it, and whose?

    Three numbers, from three places, none of them merged:

    ``stream_chars``
        Counted here, off the event stream.
    ``harness_all``
        The harness's own build-time count over the session dumps. Should equal
        ``stream_chars`` -- same source, same dedup, different code. A gap means
        one of the two readers is wrong, which is worth surfacing.
    ``harness_root``
        The harness's count over the stdout transcript, which opencode filters to
        each turn's root session. On a build that delegated it is *supposed* to
        be lower: the difference is the subagents' share of the thinking, and
        reporting it as a discrepancy would be reporting the architecture as a
        bug.
    """
    totals = manifest.get("totals") or {}
    capture = manifest.get("capture") or {}

    stream_chars = int(totals.get("reasoning_chars") or 0)
    harness_all = _capture_int(capture, "reasoning_chars_all")
    harness_root = _capture_int(capture, "reasoning_chars_root", "reasoning_chars")
    # Only a same-source pair can disagree. Comparing the dump-derived stream
    # against the transcript-derived count would flag every delegating build.
    disagrees = bool(harness_all and stream_chars and harness_all != stream_chars)
    team_chars = max(stream_chars - harness_root, 0) if harness_root else 0

    return {
        "source": manifest.get("source") or "transcript",
        "has_thinking": bool(stream_chars or harness_all or harness_root),
        "stream_chars": stream_chars,
        "harness_all": harness_all,
        "harness_root": harness_root,
        # How much of the thinking happened below the root session. Zero on a
        # build that never delegated, which is most of them.
        "team_chars": team_chars,
        "disagrees": disagrees,
        "redacted": int(totals.get("reasoning_redacted") or 0),
        "prompts": sum(
            1 for e in events if e.get("type") == "text" and e.get("role") == "user"
        ),
        "patches": int((totals.get("events_by_type") or {}).get("patch") or 0),
        "sessions": int(totals.get("sessions") or 0),
        "max_depth": int(totals.get("max_depth") or 0),
    }
