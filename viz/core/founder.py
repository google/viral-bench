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

"""Read a founder build's transcripts into something a timeline can render.

A transcript is the raw stdout of ``opencode run --format json``: newline-delimited
JSON, one event per line, four event types (``step_start``, ``tool_use``,
``step_finish``, ``text``) plus a rare ``error``. One file per turn, named after the
phase, and the files of a build are disjoint in time because the agents run
serially. So the global timeline is simple to build: read every phase, sort by
``timestamp``.

Three things shape this module.

**Bodies are enormous and mostly unwanted.** The largest single transcript in the
corpus is 5.26 MB, of which 93% is four base64 screenshots. Sending that to a
browser to render a timeline is absurd, so :func:`load_trajectory` emits a compact
event per line -- previews only -- and records the ``(phase, line)`` each event came
from. :func:`load_event_body` seeks back to that exact line when the user clicks.

**Nothing here may assume the current schema.** The corpus spans retired modes
(``specialist`` relay, a pre-``structure`` record), killed turns that left a
truncated final line, and turns that produced a 0-byte file. Every one of those is
a build somebody may want to look at, so parsing is per-line and forgiving.

**Thinking is not in the data.** Verified across the whole corpus: there is no
reasoning or thinking part type. What exists is a per-step ``tokens.reasoning``
count and an opaque encrypted ``metadata.*.thoughtSignature`` blob on tool calls.
:func:`load_trajectory` surfaces both, per message, so the UI can show an honest
proxy -- and the UI says so in as many words. See ``THINKING_NOTE``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .paths import FounderPaths, classify_mode, founder_paths, read_json
from .trace import capture_summary, load_trace

#: Shown verbatim in the UI wherever thinking is displayed. The honesty matters:
#: a reader who assumes these are the model's thoughts would draw wrong conclusions
#: about what the agent considered.
THINKING_NOTE = (
    "Reasoning is read from opencode's own session store, which the harness dumps "
    "per turn. It is the model's own chain of thought, not a proxy. Three states "
    "are distinguished and mean different things: readable text, then 'redacted', "
    "where the provider returned the thought encrypted (Vertex does this for Claude "
    "unless the request asks for a summary) and the model did think, and absent. "
    "Measured in characters, never tokens -- Vertex Anthropic reports "
    "tokens_reasoning as 0 while returning thousands of characters. Builds recorded "
    "before the capture existed are labelled 'partial trace' and have none of this."
)

PREVIEW_CHARS = 700
INPUT_PREVIEW_CHARS = 400

#: Tool-name prefix -> the visual family the UI groups it under. Browser tools are
#: double-prefixed (MCP server ``browser`` + tool ``browser_*``) which is why the
#: check is on ``browser_``, not an exact name.
_TOOL_GROUPS = (
    ("browser_", "browser"),
    ("bash", "shell"),
    ("edit", "edit"),
    ("write", "edit"),
    ("patch", "edit"),
    ("read", "read"),
    ("glob", "search"),
    ("grep", "search"),
    ("list", "search"),
    ("lsp", "search"),
    ("task", "task"),
    ("skill", "skill"),
    ("todowrite", "todo"),
    ("webfetch", "web"),
    ("websearch", "web"),
)

_PHASE_TEAM = re.compile(r"^r(\d+)_a(\d+)_(.+)$")
_PHASE_DYNAMIC = re.compile(r"^t(\d+)_founder$")
_PHASE_LEGACY = re.compile(r"^a(\d+)_(.+?)\.(design|build)$")

#: Playwright MCP results are one markdown string with fixed ``### `` sections.
#: Splitting them beats dumping the raw blob: the a11y snapshot alone can be 40 KB
#: and is only occasionally what you want to read.
_SECTION_RE = re.compile(r"^### (.+)$", re.MULTILINE)
_SCREENSHOT_PATH_RE = re.compile(r"saved it as (\S+\.(?:png|jpe?g))", re.IGNORECASE)


def tool_group(name: str) -> str:
    for prefix, group in _TOOL_GROUPS:
        if name.startswith(prefix):
            return group
    return "other"


def _clip(text: str | None, limit: int) -> tuple[str, int, bool]:
    """Return ``(preview, full_length, truncated)``."""
    if not text:
        return "", 0, False
    if len(text) <= limit:
        return text, len(text), False
    return text[:limit], len(text), True


def iter_events(path: Path) -> Iterator[tuple[int, dict]]:
    """Yield ``(line_number, event)`` for a transcript, skipping unparseable lines.

    A turn killed mid-write leaves a partial final line, and the harness itself parses
    these files line-by-line with the same tolerance.
    """
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for number, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield number, event


# --------------------------------------------------------------------------
# Phase / lane structure
# --------------------------------------------------------------------------


@dataclass
class Lane:
    """One actor in the swimlane: a specialist, the solo founder, an orchestrator."""

    key: str
    label: str
    role: str
    agent_index: int
    kind: str = "agent"  # agent | subagent
    session_id: str = ""
    session_ids: list[str] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)


def _phase_shape(phase: str, record_turn: str) -> tuple[int, int, str]:
    """Return ``(round, agent_index, role)`` inferred from a phase label.

    ``build.json`` carries these fields too, but not on every legacy record, and the
    phase name is the one thing that has always been correct -- it is the transcript
    filename stem.
    """
    match = _PHASE_TEAM.match(phase)
    if match:
        return int(match.group(1)), int(match.group(2)), match.group(3)
    match = _PHASE_DYNAMIC.match(phase)
    if match:
        return int(match.group(1)), 1, "founder"
    match = _PHASE_LEGACY.match(phase)
    if match:
        return 1, int(match.group(1)), match.group(2)
    if phase in ("design", "build"):
        return 1, 1, "founder"
    return 1, 1, record_turn or "founder"


def _pretty_role(role: str) -> str:
    return role.replace("_", " ").title()


def build_lanes(mode: str, phases: list[dict]) -> list[Lane]:
    """Group phases into the actors that produced them.

    The grouping is what makes each mode legible, so it differs by mode on purpose:

    * ``solo`` -- one actor with two turns. Two lanes would imply two agents.
    * ``team`` -- one lane per specialist, rounds running left to right. The
      session id repeats down a lane, which is the visible proof that a specialist
      resumes its own session rather than being re-rolled each round.
    * ``dynamic`` -- one orchestrator lane. The subagents it invented get their own
      lanes elsewhere, from spawn intervals that do overlap.
    """
    lanes: dict[str, Lane] = {}
    for phase in phases:
        if mode in ("team", "legacy") and phase["agent_index"]:
            key = f"a{phase['agent_index']}"
            label = _pretty_role(phase["role"] or f"Agent {phase['agent_index']}")
        else:
            key = "founder"
            label = "Founder" if mode != "dynamic" else "Orchestrator"
        lane = lanes.get(key)
        if lane is None:
            lane = Lane(
                key=key,
                label=label,
                role=phase["role"] or "",
                agent_index=phase["agent_index"] or 1,
            )
            lanes[key] = lane
        lane.phases.append(phase["phase"])
        sid = phase.get("session_id") or ""
        if sid and sid not in lane.session_ids:
            lane.session_ids.append(sid)
        if sid and not lane.session_id:
            lane.session_id = sid
    return [lanes[k] for k in sorted(lanes, key=lambda k: (len(k), k))]


# --------------------------------------------------------------------------
# Event normalisation
# --------------------------------------------------------------------------


def split_browser_result(output: str) -> dict:
    """Split a Playwright MCP result into its ``### `` sections.

    Returns the named sections plus any fenced code blocks pulled out of them, so
    the UI can show the JS that ran and keep the accessibility snapshot behind a
    disclosure instead of flooding the feed with it.
    """
    if not output:
        return {}
    matches = list(_SECTION_RE.finditer(output))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
        sections[match.group(1).strip()] = output[match.end() : end].strip()

    result: dict = {}
    code = sections.get("Ran Playwright code", "")
    if code:
        fenced = re.search(r"```(?:js|javascript)?\n(.*?)```", code, re.DOTALL)
        result["code"] = (fenced.group(1) if fenced else code).strip()
    state = sections.get("Page state", "")
    if state:
        url = re.search(r"^- Page URL:\s*(.+)$", state, re.MULTILINE)
        title = re.search(r"^- Page Title:\s*(.+)$", state, re.MULTILINE)
        snapshot = re.search(r"```yaml\n(.*?)```", state, re.DOTALL)
        if url:
            result["url"] = url.group(1).strip()
        if title:
            result["title"] = title.group(1).strip()
        if snapshot:
            snap, length, truncated = _clip(snapshot.group(1).strip(), 4000)
            result["snapshot"] = snap
            result["snapshot_chars"] = length
            result["snapshot_truncated"] = truncated
    for name in ("Result", "New console messages", "Open tabs"):
        if sections.get(name):
            body, length, truncated = _clip(sections[name], 1500)
            key = name.lower().replace(" ", "_")
            result[key] = body
            result[f"{key}_chars"] = length
    return result


def _diff_stats(metadata: dict) -> dict | None:
    filediff = metadata.get("filediff")
    if not isinstance(filediff, dict):
        return None
    return {
        "file": filediff.get("file"),
        "additions": filediff.get("additions"),
        "deletions": filediff.get("deletions"),
        "has_patch": bool(filediff.get("patch")),
    }


def _input_summary(tool: str, args: dict) -> str:
    """One line that says what the call did, per tool.

    Generic argument dumps make a 400-call feed unreadable. The point of a feed is
    that you can skim it and see the shape of the work.
    """
    if not isinstance(args, dict):
        return ""
    if tool == "bash":
        return str(args.get("command") or "")
    if tool in ("read", "write", "edit"):
        return str(args.get("filePath") or "")
    if tool in ("glob", "grep"):
        return str(args.get("pattern") or "")
    if tool == "skill":
        return str(args.get("name") or "")
    if tool == "task":
        return str(args.get("description") or args.get("subagent_type") or "")
    if tool == "todowrite":
        todos = args.get("todos") or []
        return f"{len(todos)} item(s)"
    if tool.startswith("browser_"):
        for key in ("element", "url", "key", "text", "ref", "selector", "filename"):
            if args.get(key):
                return f"{key}={args[key]}"
        # browser_evaluate carries the JS under `function`, and it is the only
        # thing that says what the call was for.
        code = args.get("function") or args.get("expression")
        if code:
            return " ".join(str(code).split())[:160]
        return ""
    for key in ("query", "url", "prompt"):
        if args.get(key):
            return str(args[key])
    return ""


def _normalise_tool(payload: dict) -> dict:
    """Turn a schema-conformant ``tool`` payload into what the feed renders.

    Input is the eight-key flattening the trajectory schema defines --
    ``name/status/input/output/error/metadata/start_ms/end_ms`` -- not opencode's
    raw part, so one code path serves a build read from the session store and one
    read from an exported bundle.
    """
    metadata = payload.get("metadata") or {}
    args = payload.get("input") or {}
    tool = payload.get("name") or "?"
    output = payload.get("output") or ""
    group = tool_group(tool)

    start = payload.get("start_ms")
    end = payload.get("end_ms")

    preview, out_chars, out_truncated = _clip(output, PREVIEW_CHARS)
    arg_text, arg_chars, arg_truncated = _clip(
        json.dumps(args, ensure_ascii=False, indent=2) if args else "",
        INPUT_PREVIEW_CHARS,
    )

    entry: dict = {
        "tool": tool,
        "group": group,
        "status": payload.get("status") or "unknown",
        "summary": _input_summary(tool, args),
        "start_ms": start,
        "end_ms": end,
        "duration_ms": (end - start)
        if isinstance(start, int) and isinstance(end, int)
        else None,
        "output_preview": preview,
        "output_chars": out_chars,
        "output_truncated": out_truncated,
        "input_preview": arg_text,
        "input_chars": arg_chars,
        "input_truncated": arg_truncated,
    }
    if payload.get("error"):
        entry["error"], _, _ = _clip(str(payload["error"]), 800)

    if group == "edit":
        stats = _diff_stats(metadata)
        if stats:
            entry["diff"] = stats
        entry["file"] = args.get("filePath")
    elif group == "shell":
        entry["exit"] = metadata.get("exit")
    elif group == "browser":
        parsed = split_browser_result(output)
        if parsed:
            entry["browser"] = parsed
        shot = _SCREENSHOT_PATH_RE.search(output)
        if shot:
            entry["screenshot_path"] = shot.group(1)
    elif group == "task":
        entry["subagent_type"] = args.get("subagent_type")
        entry["child_session"] = metadata.get("sessionId")
        entry["parent_session"] = metadata.get("parentSessionId")
        model = metadata.get("model") or {}
        entry["subagent_model"] = model.get("modelID")
    elif group == "todo":
        entry["todos"] = metadata.get("todos") or args.get("todos") or []
    return entry


# --------------------------------------------------------------------------
# Trajectory assembly
# --------------------------------------------------------------------------


def _phase_rollup(path: Path) -> dict:
    """Per-turn cost, tokens and time span, read from the stdout transcript.

    The event stream itself comes from the session store (see :mod:`core.trace`),
    but these numbers do not: ``step_finish`` carries the per-step cost and token
    counts, and the store's own totals are per *session*, which is the wrong unit
    when four specialists share a build or one session spans two turns.

    The time span is the other reason this pass exists. The dump is organised by
    session, so a solo build's two turns land in one file with nothing marking
    where ``design`` ends and ``build`` begins. The stdout transcripts are per
    turn, so their first and last timestamps give the window each event can be
    attributed to.
    """
    rollup = {
        "cost": 0.0,
        "tokens_input": 0,
        "tokens_output": 0,
        "tokens_reasoning": 0,
        "tokens_total": 0,
        "cache_read": 0,
        "cache_write": 0,
        "steps": 0,
        "tools": 0,
        "tool_errors": 0,
        "texts": 0,
        "reasoning": 0,
        "errors": 0,
        "first_ms": None,
        "last_ms": None,
    }

    for _line_no, raw in iter_events(path):
        kind = raw.get("type")
        timestamp = raw.get("timestamp")
        if isinstance(timestamp, int):
            if rollup["first_ms"] is None:
                rollup["first_ms"] = timestamp
            rollup["last_ms"] = timestamp

        if kind == "error":
            rollup["errors"] += 1
            continue

        part = raw.get("part") or {}
        if kind == "text":
            rollup["texts"] += 1
        elif kind == "reasoning":
            rollup["reasoning"] += 1
        elif kind == "tool_use":
            rollup["tools"] += 1
            if ((part.get("state") or {}).get("status")) == "error":
                rollup["tool_errors"] += 1
        elif kind == "step_finish":
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            rollup["cost"] += float(part.get("cost") or 0.0)
            rollup["steps"] += 1
            rollup["tokens_input"] += int(tokens.get("input") or 0)
            rollup["tokens_output"] += int(tokens.get("output") or 0)
            rollup["tokens_reasoning"] += int(tokens.get("reasoning") or 0)
            rollup["tokens_total"] += int(tokens.get("total") or 0)
            rollup["cache_read"] += int(cache.get("read") or 0)
            rollup["cache_write"] += int(cache.get("write") or 0)

    return rollup


def _phase_of(timestamp, windows: list[tuple[str, int, int]]) -> str:
    """Which turn an event belongs to, by wall-clock time.

    Turns of one build never overlap -- the agents run serially -- so a timestamp
    inside a window identifies its turn unambiguously, including for a subagent
    event that has no turn of its own but ran inside its parent's.

    Events in the *gap* between two turns belong to the turn that is starting, not
    the one that has ended: the previous turn's window closes on its last output,
    and the thing that reliably happens in that gap is the next turn's prompt being
    sent. Crediting the build prompt to the design turn is the bug this rule fixes.
    """
    if not windows or not isinstance(timestamp, int):
        return windows[0][0] if windows else ""
    for name, first, last in windows:
        if first <= timestamp <= last:
            return name
    for name, first, _last in windows:
        if timestamp < first:
            return name
    return windows[-1][0]


#: Trace part types the feed renders as their own row. Everything else -- the
#: ``step-start`` bookkeeping, and anything opencode invents later -- is folded
#: into ``other`` rather than dropped.
_UI_KINDS = {
    "reasoning": "reasoning",
    "text": "text",
    "tool": "tool",
    "patch": "patch",
    "step-finish": "step",
}


def _ui_events(
    trace_events: list[dict], windows: list[tuple[str, int, int]]
) -> list[dict]:
    """Compact the trace stream into what the timeline renders.

    Previews only: the largest single transcript in the corpus is 5 MB and a
    reasoning block runs to thousands of characters, so full bodies are fetched on
    click via ``part_id``.
    """
    out: list[dict] = []
    for event in trace_events:
        kind = _UI_KINDS.get(event.get("type"), "other")
        base = {
            "t": event.get("ts"),
            "seq": event.get("seq"),
            "part_id": event.get("part_id") or "",
            "session": event.get("session_id") or "",
            "depth": event.get("depth") or 0,
            "msg": event.get("message_id") or "",
            "agent": event.get("agent") or "",
            "role": event.get("role") or "",
            "phase": _phase_of(event.get("ts"), windows),
        }
        if event.get("unknown_session"):
            base["unknown_session"] = True

        if kind == "reasoning":
            text, chars, truncated = _clip(event.get("text") or "", 4000)
            out.append(
                {
                    **base,
                    "kind": "reasoning",
                    "text": text,
                    "chars": chars,
                    "truncated": truncated,
                    # Empty text plus this flag means the provider encrypted the
                    # thought. Not the same as not thinking, and the UI says so.
                    "redacted": bool(event.get("redacted")),
                }
            )
        elif kind == "text":
            # A prompt is a text part with role user, since there is no prompt type.
            is_prompt = event.get("role") == "user"
            text, chars, truncated = _clip(
                event.get("text") or "", 8000 if is_prompt else 4000
            )
            out.append(
                {
                    **base,
                    "kind": "prompt" if is_prompt else "text",
                    "text": text,
                    "chars": chars,
                    "truncated": truncated,
                }
            )
        elif kind == "tool":
            out.append(
                {**base, "kind": "tool", **_normalise_tool(event.get("tool") or {})}
            )
        elif kind == "patch":
            patch = event.get("patch") or {}
            files = patch.get("files") or []
            out.append(
                {
                    **base,
                    "kind": "patch",
                    "hash": patch.get("hash") or "",
                    "files": [str(f) for f in files][:60],
                    "n_files": len(files),
                }
            )
        elif kind == "step":
            payload = event.get("payload") or {}
            tokens = payload.get("tokens") or {}
            cache = tokens.get("cache") or {}
            out.append(
                {
                    **base,
                    "kind": "step",
                    "reason": payload.get("reason"),
                    "cost": float(payload.get("cost") or 0.0),
                    "snapshot": payload.get("snapshot"),
                    "tokens": {
                        "input": tokens.get("input") or 0,
                        "output": tokens.get("output") or 0,
                        "reasoning": tokens.get("reasoning") or 0,
                        "total": tokens.get("total") or 0,
                        "cache_read": cache.get("read") or 0,
                        "cache_write": cache.get("write") or 0,
                    },
                }
            )
        elif event.get("type") == "step-start":
            continue
        else:
            preview, chars, truncated = _clip(
                json.dumps(event.get("payload") or {}, ensure_ascii=False)[:2000], 2000
            )
            out.append(
                {
                    **base,
                    "kind": "other",
                    "type": event.get("type") or "?",
                    "preview": preview,
                    "chars": chars,
                    "truncated": truncated,
                }
            )
    for index, event in enumerate(out):
        event["i"] = index
    return out


def _spawn_lanes(record: dict, events: list[dict]) -> list[dict]:
    """Subagent intervals for dynamic mode, from build.json or the transcript.

    ``build.json.orchestration.spawns`` is the richer source but is absent on the
    older dynamic builds, so the transcript's own ``task`` calls are the fallback.
    These intervals do overlap -- that overlap is the only place the
    orchestrator's parallelism is visible, and it is the point of the mode.
    """
    orchestration = record.get("orchestration") or {}
    spawns = orchestration.get("spawns") or []
    rows: list[dict] = []
    for index, spawn in enumerate(spawns):
        rows.append(
            {
                "index": index,
                "label": spawn.get("description") or f"subagent {index + 1}",
                "subagent_type": spawn.get("subagent_type") or "",
                "status": spawn.get("status") or "",
                "error": spawn.get("error") or "",
                "session_id": spawn.get("session_id") or "",
                "parent_session_id": spawn.get("parent_session_id") or "",
                "resumed": bool(spawn.get("resumed_task_id")),
                "model": spawn.get("model") or "",
                "prompt_chars": spawn.get("prompt_chars"),
                "prompt": (spawn.get("prompt") or "")[:2000],
                "start_ms": spawn.get("start_ms"),
                "end_ms": spawn.get("end_ms"),
            }
        )
    if rows:
        return rows
    for index, event in enumerate(e for e in events if e.get("group") == "task"):
        rows.append(
            {
                "index": index,
                "label": event.get("summary") or f"subagent {index + 1}",
                "subagent_type": event.get("subagent_type") or "",
                "status": event.get("status") or "",
                "error": event.get("error") or "",
                "session_id": event.get("child_session") or "",
                "parent_session_id": event.get("parent_session") or "",
                "resumed": False,
                "model": event.get("subagent_model") or "",
                "prompt_chars": event.get("input_chars"),
                "prompt": event.get("input_preview") or "",
                "start_ms": event.get("start_ms"),
                "end_ms": event.get("end_ms"),
            }
        )
    return rows


def _authored_agents(paths: FounderPaths) -> list[dict]:
    """Read the agent definitions a dynamic founder wrote for itself."""
    if not paths.agents_dir.is_dir():
        return []
    out = []
    for path in sorted(paths.agents_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        description, mode, temperature = "", "", None
        if text.startswith("---"):
            _, _, rest = text.partition("---")
            front, _, body = rest.partition("---")
            for line in front.splitlines():
                key, sep, value = line.partition(":")
                if not sep:
                    continue
                key, value = key.strip(), value.strip()
                if key == "description":
                    description = value
                elif key == "mode":
                    mode = value
                elif key == "temperature":
                    try:
                        temperature = float(value)
                    except ValueError:
                        pass
            text = body
        out.append(
            {
                "name": path.stem,
                "description": description,
                "mode": mode,
                "temperature": temperature,
                "body": text.strip()[:4000],
            }
        )
    return out


def load_trajectory(builds_root: Path, build_id: str) -> dict | None:
    """Assemble the full, compact trajectory for one build.

    Returns ``None`` when the build id does not resolve. Everything else -- missing
    transcripts, empty phases, failed turns -- is represented in the result rather
    than raised, because a failed build is exactly the kind you want to inspect.
    """
    paths = founder_paths(builds_root, build_id)
    record = read_json(paths.build_json)
    if not isinstance(record, dict):
        return None

    mode, mode_label = classify_mode(record)
    raw_phases = record.get("phases") or []

    # A build killed before build.json was written still has transcripts on disk.
    if not raw_phases and paths.transcript_dir.is_dir():
        raw_phases = [
            {"phase": p.stem, "transcript": str(p)}
            for p in sorted(paths.transcript_dir.glob("*.json"))
        ]

    phases: list[dict] = []
    windows: list[tuple[str, int, int]] = []

    for order, entry in enumerate(raw_phases):
        phase_id = entry.get("phase") or f"phase{order}"
        # Transcript paths inside build.json are absolute paths from the machine
        # that produced the run. Resolve by name against the directory being
        # read, and only fall back to the recorded path.
        candidate = paths.transcript_dir / f"{phase_id}.json"
        if not candidate.is_file():
            recorded = entry.get("transcript")
            candidate = Path(recorded) if recorded else candidate
        rnd, agent_index, role = _phase_shape(phase_id, entry.get("turn") or "")

        rollup = _phase_rollup(candidate) if candidate.is_file() else {}
        if rollup.get("first_ms") is not None:
            windows.append((phase_id, rollup["first_ms"], rollup["last_ms"]))

        phases.append(
            {
                "phase": phase_id,
                "order": order,
                "role": entry.get("role") or role,
                "agent_index": entry.get("agent_index") or agent_index,
                "round": rnd,
                "turn": entry.get("turn") or "",
                "session_id": entry.get("session_id") or "",
                "returncode": entry.get("returncode"),
                "ok": entry.get("ok"),
                "timed_out": entry.get("timed_out"),
                "duration_s": entry.get("duration_s"),
                "transcript": str(candidate),
                "transcript_bytes": candidate.stat().st_size
                if candidate.is_file()
                else 0,
                "missing": not candidate.is_file(),
                "stderr_tail": (entry.get("stderr_tail") or "")[-4000:],
                # What the harness counted for this turn at build time, off the
                # stdout transcript -- so root-session only, blind to subagents.
                # `_root` since the counts were split by source. The bare names
                # are what builds recorded before the split.
                "reasoning_parts": entry.get("reasoning_parts_root")
                or entry.get("reasoning_parts")
                or 0,
                "reasoning_chars": entry.get("reasoning_chars_root")
                or entry.get("reasoning_chars")
                or 0,
                "sessions_records": entry.get("sessions_records") or 0,
                "rollup": rollup,
            }
        )

    windows.sort(key=lambda w: w[1])

    # The event stream comes from the session store when it exists: it is a strict
    # superset of stdout, carrying the prompts, the subagents and the file patches
    # that never reach the transcript at all.
    traced = load_trace(builds_root, build_id, record)
    trace_manifest = traced["manifest"]
    events = _ui_events(traced["events"], windows)

    lanes = build_lanes(mode, phases)
    lane_of_phase = {p: lane.key for lane in lanes for p in lane.phases}
    # A delegated session has no turn of its own, so it gets a lane of its own
    # rather than being folded into whichever turn it happened to run inside.
    sub_lanes: dict[str, Lane] = {}
    sessions_by_id = {s["session_id"]: s for s in trace_manifest.get("sessions", [])}
    for event in events:
        if event.get("depth", 0) > 0:
            session_id = event.get("session") or ""
            lane = sub_lanes.get(session_id)
            if lane is None:
                info = sessions_by_id.get(session_id) or {}
                label = info.get("title") or info.get("agent") or session_id[:12]
                lane = Lane(
                    key=f"s:{session_id}",
                    label=label[:44],
                    role=info.get("agent") or "subagent",
                    agent_index=0,
                    kind="subagent",
                    session_id=session_id,
                    session_ids=[session_id],
                )
                sub_lanes[session_id] = lane
            event["lane"] = lane.key
        else:
            event["lane"] = lane_of_phase.get(
                event.get("phase"), lanes[0].key if lanes else "founder"
            )

    for lane in sub_lanes.values():
        lane.phases = sorted({e["phase"] for e in events if e["lane"] == lane.key})
    lanes.extend(sorted(sub_lanes.values(), key=lambda entry: entry.label))

    kinds: dict[str, int] = {}
    for event in events:
        kinds[event["kind"]] = kinds.get(event["kind"], 0) + 1

    totals = {
        "cost": round(sum(p["rollup"].get("cost", 0.0) for p in phases), 4),
        "tools": kinds.get("tool", 0),
        "tool_errors": sum(1 for e in events if e.get("status") == "error"),
        "texts": kinds.get("text", 0),
        "prompts": kinds.get("prompt", 0),
        "reasoning": kinds.get("reasoning", 0),
        "patches": kinds.get("patch", 0),
        "steps": kinds.get("step", 0),
        "errors": sum(p["rollup"].get("errors", 0) for p in phases),
        "tokens_total": sum(p["rollup"].get("tokens_total", 0) for p in phases),
        "tokens_output": sum(p["rollup"].get("tokens_output", 0) for p in phases),
        # Kept, but never the headline: Vertex Anthropic reports this as 0 while
        # returning thousands of characters of thought, so a Claude arm would read
        # as "did not think". reasoning_chars is the number to trust.
        "tokens_reasoning": sum(p["rollup"].get("tokens_reasoning", 0) for p in phases),
        "reasoning_chars": int(
            (trace_manifest.get("totals") or {}).get("reasoning_chars") or 0
        ),
        "reasoning_redacted": int(
            (trace_manifest.get("totals") or {}).get("reasoning_redacted") or 0
        ),
        # Counted by depth, not by "sessions minus one": a team build has four
        # INDEPENDENT root sessions, one per specialist, and none of them is a
        # subagent of the others.
        "subagent_sessions": sum(
            1 for s in trace_manifest.get("sessions", []) if s.get("depth", 0) > 0
        ),
        "root_sessions": sum(
            1 for s in trace_manifest.get("sessions", []) if not s.get("depth")
        ),
        "max_depth": int((trace_manifest.get("totals") or {}).get("max_depth") or 0),
        "events": len(events),
        "duration_s": round(sum(float(p.get("duration_s") or 0) for p in phases), 1),
    }

    stamps = [e["t"] for e in events if isinstance(e.get("t"), int)]
    span = (
        {"start_ms": min(stamps), "end_ms": max(stamps)}
        if stamps
        else {"start_ms": None, "end_ms": None}
    )

    tool_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    files_touched: dict[str, int] = {}
    for event in events:
        if event.get("kind") != "tool":
            continue
        tool_counts[event["tool"]] = tool_counts.get(event["tool"], 0) + 1
        group_counts[event["group"]] = group_counts.get(event["group"], 0) + 1
        target = event.get("file")
        if target:
            name = Path(target).name
            files_touched[name] = files_touched.get(name, 0) + 1

    manifest = record.get("manifest") or {}
    summary_record = {k: v for k, v in record.items() if k != "phases"}

    return {
        "build_id": build_id,
        "mode": mode,
        "mode_label": mode_label,
        "record": summary_record,
        "manifest": manifest,
        "orchestration": record.get("orchestration") or {},
        "phases": phases,
        "lanes": [lane.__dict__ for lane in lanes],
        "events": events,
        "totals": totals,
        "span": span,
        "tool_counts": tool_counts,
        "group_counts": group_counts,
        "files_touched": files_touched,
        "spawns": _spawn_lanes(record, events) if mode == "dynamic" else [],
        "authored_agents": _authored_agents(paths) if mode == "dynamic" else [],
        "trace": {
            "source": trace_manifest.get("source"),
            "schema_version": trace_manifest.get("schema_version"),
            "sessions": trace_manifest.get("sessions", []),
            "totals": trace_manifest.get("totals", {}),
            "capture": trace_manifest.get("capture", {}),
            "summary": capture_summary(trace_manifest, traced["events"]),
            "bundle_dir": trace_manifest.get("bundle_dir"),
        },
        "screenshots": sorted(p.name for p in paths.screenshots_dir.glob("*.png"))
        if paths.screenshots_dir.is_dir()
        else [],
        "thinking_note": THINKING_NOTE,
        "paths": {
            "root": str(paths.root),
            "build_json": str(paths.build_json),
            "transcript_dir": str(paths.transcript_dir),
            "app_dir": str(paths.app_dir),
        },
    }


def load_event_body(
    builds_root: Path, build_id: str, part_id: str = "", phase: str = "", line: int = -1
) -> dict | None:
    """Re-read one event in full, for the detail pane.

    The timeline ships previews, because a reasoning block runs to thousands of
    characters and a Playwright result to tens of thousands. This is the
    click-through, and it has two lookups because there are two sources: a
    ``part_id`` seeks the session dumps, a ``phase``+``line`` seeks the stdout
    transcript for builds recorded before the dumps existed.

    Base64 attachment payloads are replaced by a marker either way -- the bytes
    are already on disk as a file, and a 5 MB data URL in a JSON response helps
    nobody.
    """
    paths = founder_paths(builds_root, build_id)

    if part_id:
        dump_dir = paths.transcript_dir / "sessions"
        if dump_dir.is_dir():
            for dump in sorted(dump_dir.glob("*.jsonl")):
                for record in _iter_dump(dump):
                    if record.get("record") != "part":
                        continue
                    if str(record.get("part_id") or "") != part_id:
                        continue
                    return {
                        "build_id": build_id,
                        "part_id": part_id,
                        "session_id": record.get("session_id"),
                        "message_id": record.get("message_id"),
                        "role": record.get("role"),
                        "agent": record.get("agent"),
                        "timestamp": record.get("time_created"),
                        "part": _strip_attachments(record.get("part") or {}),
                    }
        # No dump: the build predates it, but its stdout parts carry the same ids,
        # so the same address still resolves.
        for path in sorted(paths.transcript_dir.glob("*.json")):
            for _number, event in iter_events(path):
                part = event.get("part") or {}
                if str(part.get("id") or "") != part_id:
                    continue
                return {
                    "build_id": build_id,
                    "part_id": part_id,
                    "phase": path.stem,
                    "session_id": event.get("sessionID"),
                    "message_id": part.get("messageID"),
                    "timestamp": event.get("timestamp"),
                    "part": _strip_attachments(part),
                }
        return None

    path = paths.transcript_dir / f"{phase}.json"
    if not path.is_file() or line < 0:
        return None
    for number, event in iter_events(path):
        if number != line:
            continue
        return {
            "build_id": build_id,
            "phase": phase,
            "line": line,
            "type": event.get("type"),
            "timestamp": event.get("timestamp"),
            "session_id": event.get("sessionID"),
            "part": _strip_attachments(event.get("part") or {}),
            "error": event.get("error"),
        }
    return None


def _iter_dump(path: Path) -> Iterator[dict]:
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def _strip_attachments(part: dict) -> dict:
    state = part.get("state")
    if isinstance(state, dict) and state.get("attachments"):
        state["attachments"] = [
            {
                "type": a.get("mime") or a.get("type"),
                "inline_bytes": len(a.get("url") or ""),
                "note": "base64 payload omitted, use the on-disk screenshot",
            }
            for a in state["attachments"]
        ]
    return part
