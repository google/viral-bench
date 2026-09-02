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

"""The agent loop: the model gathers, the code judges.

Every rubric item resolves one of two ways, and the difference is the whole
design:

**It has a ``check:`` block.** The harness runs a deterministic primitive and
decides. The model's only job is *navigation* -- getting the app into the state
the item's ``setup`` prose describes. It never reports the value, so it cannot
misreport it.

**It has no ``check:``.** The model judges, and must cite the ``tool_call_id`` of
the call that proves it. **The harness then reads the value out of its own log of
that call, not out of the model's message.** A verdict citing no call, or a call
whose recorded result contradicts it, is recorded FAIL.

That second rule is structural rather than exhortative, and it is the direct
descendant of what the crowd work learned: 36% of triers were filing verdicts on
apps they had never touched, and the fix that worked was making the evidence a
recorded artifact rather than a claim.

Everything the model does flows through :class:`GraderTools`, which logs each
call and its result to ``transcript.jsonl`` before the model ever sees the
result. The log is therefore the harness's own record, written first, and the
model's message is checked against it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from viral_bench.rubric.checks import CheckContext, CheckResult, run_check
from viral_bench.rubric.schema import Rubric, RubricItem
from viral_bench.rubric.score import ItemVerdict

#: Tool calls one item may make before the harness moves on. An item that cannot
#: be settled in this many steps is an item the model is lost on, and letting it
#: wander costs the rest of the build's grade.
MAX_STEPS_PER_ITEM = 14

#: How many items to hand the model at once. Batching cuts token cost sharply,
#: but a long batch degrades navigation, so this stays small.
ITEMS_PER_BATCH = 1

#: Verdict tokens the model may emit. Anything else is treated as unresolved
#: rather than guessed at.
_VERDICTS = {"pass": True, "fail": False, "unknown": None}

_TOOL_ID = re.compile(r"\btc_\d+\b")


class GraderError(RuntimeError):
    """The grader could not run at all -- distinct from a build scoring badly."""


@dataclass
class ToolRecord:
    """One logged tool call: what was asked, and what actually came back."""

    id: str
    name: str
    args: dict
    result: str
    ok: bool = True
    item_id: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "args": self.args,
            "result": self.result,
            "ok": self.ok,
            "item_id": self.item_id,
        }


#: The model's tool surface. Anthropic tool-schema shape, which is the canonical
#: one :mod:`viral_bench.providers` translates out of, so one list serves every
#: provider. Deliberately close to the crowd's toolkit -- the grader should be
#: able to do what a determined user could do, and no more, or items calibrated
#: against what the crowd can reach would be graded against a different machine.
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "look",
        "description": (
            "Describe the current page: title, visible text, interactive "
            "controls, and any console errors since the last look."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "click",
        "description": "Click a control, named the way `look` describes it.",
        "input_schema": {
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
    },
    {
        "name": "type_text",
        "description": "Type text into a field, replacing what is there.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["target", "text"],
        },
    },
    {
        "name": "press_key",
        "description": "Press a key, e.g. Enter, Tab, ArrowDown, Control+b.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["key"],
        },
    },
    {
        "name": "select_option",
        "description": "Choose a value in a select control.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["target", "value"],
        },
    },
    {
        "name": "upload_file",
        "description": (
            "Upload one of the named test fixtures into a file input. "
            "Use `list_fixtures` to see what is available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixture": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["fixture"],
        },
    },
    {
        "name": "reload_page",
        "description": "Reload the current page, keeping cookies and storage.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "goto",
        "description": "Navigate to a path on the app, e.g. /settings.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "evaluate",
        "description": (
            "Run a JavaScript expression in the page and return its value as "
            "JSON. Use for reading state the UI does not show."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"js": {"type": "string"}},
            "required": ["js"],
        },
    },
    {
        "name": "screenshot",
        "description": "Capture the page as evidence; returns a shot id.",
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string"}},
        },
    },
    {
        "name": "list_files",
        "description": "List the app's source files.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_file",
        "description": "Read one source file by relative path.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "grep",
        "description": "Search the app's source for a regular expression.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "drag",
        "description": (
            "Press, move and release the mouse across an element. The only way "
            "to interact with a <canvas>: drawing, moving a shape, resizing and "
            "panning are all this one gesture. 'from_x'/'from_y' are offsets "
            "inside the element's own box (default: its centre), and 'dx'/'dy' "
            "are how far to travel from there. Real pointer events are emitted, "
            "so an app listening for pointerdown sees a genuine gesture."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "dx": {"type": "integer"},
                "dy": {"type": "integer"},
                "from_x": {"type": "number"},
                "from_y": {"type": "number"},
                "steps": {"type": "integer"},
            },
            "required": ["target"],
        },
    },
    {
        "name": "capture",
        "description": (
            "Record a labelled snapshot of the app's current visible state. "
            "Use it when an item asks whether a value changed: capture "
            "'before', perform the action, then capture 'after'. The harness "
            "takes and stores the snapshot itself -- you choose the moment, "
            "not the contents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
    },
    {
        "name": "report",
        "description": (
            "Record your verdict for the current item and move on. For a "
            "judged item you MUST cite the ids of the tool calls that prove "
            "it; a verdict with no cited evidence is recorded as a failure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "fail", "unknown"]},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "note": {"type": "string"},
            },
            "required": ["verdict"],
        },
    },
]

SYSTEM_PROMPT = """\
You are grading a web application against one rubric item at a time. You are not
a user and not a reviewer writing prose: you are an instrument.

Rules that are enforced by the harness, not merely requested:

1. Judge ONLY the item you are given. Do not comment on anything else.
2. Before reporting, actually exercise the app with the tools. A verdict that
   cites no tool call is recorded as a FAIL regardless of what you write.
3. When you report, cite the ids of the tool calls that prove your verdict (they
   look like tc_007). The harness reads the recorded result of those calls; it
   does not read your description of them. Describing a result you did not
   observe will therefore not help you.
4. Some items are decided by code after you finish navigating. For those, your
   job is only to reach the state the item describes, then report `unknown` if
   you cannot tell -- the harness decides.
5. `unknown` is a legitimate answer when the app is too broken to tell. Guessing
   is not.

Be economical. You have a small step budget per item.
"""


class GraderTools:
    """The model's hands, and the harness's log of what they did.

    Every call is recorded *before* its result is handed back, so the transcript
    is the harness's own account rather than the model's. Evidence binding reads
    from here.
    """

    def __init__(
        self,
        *,
        page: Any,
        source: Any = None,
        url: str = "",
        scratch: Path | None = None,
        fixtures: dict[str, Path] | None = None,
    ) -> None:
        self.page = page
        self.source = source
        self.url = url.rstrip("/")
        self.scratch = scratch or Path(".")
        self.fixtures = fixtures or {}
        self.log: list[ToolRecord] = []
        #: label -> {"text": ..., "full": ...}, consumed by `value_changes`.
        #:
        #: Two views, because one is a trap. "full" includes form values, so a
        #: snapshot taken after the grader types into a search box differs from
        #: the one before it *whether or not the app did anything* -- which would
        #: award a "search filters as you type" item to a build whose search
        #: filters nothing. "text" is the rendered result only.
        self.captures: dict[str, dict[str, str]] = {}
        self.shots: list[str] = []
        self.downloads: list[dict] = []
        self.current_item = ""
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return f"tc_{self._counter:03d}"

    def record(self, name: str, args: dict, result: str, *, ok: bool = True) -> str:
        """Append one call to the log and return its id."""
        entry = ToolRecord(
            id=self.next_id(),
            name=name,
            args=dict(args),
            result=result,
            ok=ok,
            item_id=self.current_item,
        )
        self.log.append(entry)
        return entry.id

    def find(self, call_id: str) -> ToolRecord | None:
        for entry in self.log:
            if entry.id == call_id:
                return entry
        return None

    def calls_for(self, item_id: str) -> list[ToolRecord]:
        return [entry for entry in self.log if entry.item_id == item_id]

    async def dispatch(self, name: str, args: dict) -> tuple[str, bool]:
        """Run one tool. Never raises: a failure is a result the model can read."""
        try:
            handler = getattr(self, f"_do_{name}", None)
            if handler is None:
                return f"no such tool: {name}", False
            result = handler(**args)
            if hasattr(result, "__await__"):
                result = await result
            return str(result), True
        except TypeError as exc:
            return f"{name}: bad arguments ({exc})", False
        except Exception as exc:  # noqa: BLE001 - a tool failure is data
            return f"{name} failed: {type(exc).__name__}: {exc}", False

    # -- the tools themselves ------------------------------------------------

    async def _do_look(self) -> str:
        snapshot = await self.page.snapshot()
        controls = "\n".join(f"  - {e.describe()}" for e in snapshot.elements[:60])
        errors = "\n".join(f"  ! {e}" for e in snapshot.console_errors[:10])
        parts = [
            f"url: {snapshot.url}",
            f"title: {snapshot.title}",
            f"text:\n{snapshot.text[:4000]}",
            f"controls:\n{controls or '  (none)'}",
        ]
        if errors:
            parts.append(f"console errors:\n{errors}")
        return "\n".join(parts)

    async def _do_click(self, target: str) -> str:
        await self.page.click(target)
        return f"clicked {target!r}"

    async def _do_type_text(self, target: str, text: str) -> str:
        await self.page.fill(target, text)
        return f"typed {len(text)} chars into {target!r}"

    async def _do_drag(
        self,
        target: str,
        dx: int = 0,
        dy: int = 0,
        from_x: float | None = None,
        from_y: float | None = None,
        steps: int = 16,
    ) -> str:
        return await self.page.drag(
            target, dx=dx, dy=dy, from_x=from_x, from_y=from_y, steps=steps
        )

    async def _do_press_key(self, key: str, target: str | None = None) -> str:
        await self.page.press(key, target=target)
        return f"pressed {key}"

    async def _do_select_option(self, target: str, value: str) -> str:
        await self.page.select_option(target, value)
        return f"selected {value!r} in {target!r}"

    async def _do_upload_file(self, fixture: str, target: str = "") -> str:
        path = self.fixtures.get(fixture)
        if path is None:
            return f"no such fixture {fixture!r}; have: {', '.join(self.fixtures)}"
        await self.page.set_files(target or None, [str(path)])
        return f"uploaded {fixture}"

    async def _do_reload_page(self) -> str:
        await self.page.goto(self.page.page.url)
        return "reloaded"

    async def _do_goto(self, path: str) -> str:
        target = path if path.startswith("http") else f"{self.url}/{path.lstrip('/')}"
        await self.page.goto(target)
        return f"navigated to {target}"

    async def _do_evaluate(self, js: str) -> str:
        value = await self.page.evaluate(js)
        try:
            return json.dumps(value)[:4000]
        except (TypeError, ValueError):
            return str(value)[:4000]

    async def _do_screenshot(self, note: str = "") -> str:
        path = await self.page.screenshot(label=note or "shot")
        self.shots.append(path)
        return f"captured {path}"

    async def _do_capture(self, label: str) -> str:
        from viral_bench.rubric.checks import USER_VISIBLE_JS

        # A third view: a style fingerprint. A theme switch changes no text at
        # all, so a text-only snapshot cannot see it -- and because "full"
        # includes select.value, comparing that would report a change on exactly
        # the builds whose theme control does nothing.
        #
        # Keyed by selector, and covering the largest visible blocks as well as
        # the page anchors. The anchors alone falsely failed any build that
        # repaints a scoped element -- a code card, an editor pane -- rather than
        # the whole page, which is the common shape for a themed preview. Blocks
        # are keyed by tag/id/class rather than by position, so a repaint that
        # also reflows does not read as every key having changed.
        style_js = """() => {
          const fp = e => { const c = getComputedStyle(e);
            return [c.backgroundColor, c.color, c.fontFamily,
                    c.borderColor].join('|'); };
          const out = {};
          for (const sel of ['body', ':root', 'main', 'section']) {
            const e = document.querySelector(sel);
            if (e) out[sel] = fp(e);
          }
          const blocks = [...(document.body ? document.body.querySelectorAll('*') : [])]
            .map(e => [e, e.getBoundingClientRect()])
            .filter(([e, r]) => r.width * r.height > 5000
                    && getComputedStyle(e).visibility !== 'hidden')
            .sort((a, b) => b[1].width * b[1].height - a[1].width * a[1].height)
            .slice(0, 12);
          for (const [e] of blocks) {
            const cls = (e.className && typeof e.className === 'string')
              ? '.' + e.className.trim().split(/\\s+/).slice(0, 3).join('.') : '';
            let key = e.tagName.toLowerCase() + (e.id ? '#' + e.id : '') + cls;
            let n = 1; while (key in out) { key = key + '~' + (++n); }
            out[key] = fp(e);
          }
          return JSON.stringify(out, Object.keys(out).sort());
        }"""
        # A fourth view: what is actually drawn on the canvases. For the 125 of
        # 127 whiteboard builds that render their scene to a <canvas>, the DOM
        # says nothing at all -- text, style and form state are identical before
        # and after a stroke. Downsampled to a 32x32 grid so the fingerprint is
        # stable against antialiasing, and reported with an ink ratio so "the
        # canvas has content" is separable from "the canvas changed".
        canvas_js = """() => {
          const out = [];
          for (const c of document.querySelectorAll('canvas')) {
            if (!c.width || !c.height) { out.push('empty'); continue; }
            try {
              const s = document.createElement('canvas');
              s.width = 32; s.height = 32;
              const g = s.getContext('2d');
              g.drawImage(c, 0, 0, 32, 32);
              const d = g.getImageData(0, 0, 32, 32).data;
              const freq = {};
              let cells = '';
              for (let i = 0; i < d.length; i += 4) {
                const q = [d[i] >> 5, d[i+1] >> 5, d[i+2] >> 5,
                           d[i+3] > 8 ? 1 : 0].join(',');
                freq[q] = (freq[q] || 0) + 1;
                cells += (d[i+3] > 8 ? (d[i] + d[i+1] + d[i+2]) >> 7 : 0).toString(32);
              }
              let top = 0;
              for (const k in freq) { if (freq[k] > top) top = freq[k]; }
              const ink = 1 - top / 1024;
              out.push(ink.toFixed(3) + ':' + cells);
            } catch (e) { out.push('tainted'); }
          }
          return JSON.stringify(out);
        }"""
        views = {}
        for name, expression in (
            ("text", "() => document.body ? document.body.innerText : ''"),
            ("full", USER_VISIBLE_JS),
            ("style", style_js),
            ("canvas", canvas_js),
        ):
            value = await self.page.evaluate(expression)
            views[name] = (
                value if isinstance(value, str) else json.dumps(value, default=str)
            )
        self.captures[str(label)] = views
        return f"captured {label!r} ({len(views['text'])} chars rendered)"

    def _do_list_files(self) -> str:
        return self.source.list_files() if self.source else "no source available"

    def _do_read_file(self, path: str) -> str:
        return self.source.read_file(path) if self.source else "no source available"

    def _do_grep(self, pattern: str) -> str:
        return self.source.grep(pattern) if self.source else "no source available"

    def write_transcript(self, path: Path) -> Path:
        """Persist the log as JSONL, one call per line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self.log:
                handle.write(json.dumps(entry.as_dict()) + "\n")
        return path


@dataclass
class ItemOutcome:
    """One item, one pass: what the model said and what the harness concluded."""

    item_id: str
    passed: bool | None
    reason: str = ""
    observed: str = ""
    expected: str = ""
    evidence: list[str] = field(default_factory=list)
    model_verdict: bool | None = None
    harness_override: bool = False
    steps: int = 0


def _parse_report(args: dict) -> tuple[bool | None, list[str], str]:
    verdict = _VERDICTS.get(str(args.get("verdict") or "").strip().lower())
    raw = args.get("evidence") or []
    if isinstance(raw, str):
        cited = _TOOL_ID.findall(raw)
    else:
        cited = [str(value) for value in raw]
    return verdict, cited, str(args.get("note") or "").strip()


def _bind_evidence(
    tools: GraderTools, item_id: str, cited: Sequence[str]
) -> tuple[list[ToolRecord], list[str]]:
    """Resolve cited ids against the harness's own log.

    Only calls made *while grading this item* count. Citing an id from an earlier
    item is not evidence that this one holds, and it is exactly the shortcut a
    model under step pressure reaches for.
    """
    valid, unknown = [], []
    own = {entry.id for entry in tools.calls_for(item_id)}
    for call_id in cited:
        entry = tools.find(call_id)
        if entry is None or entry.id not in own:
            unknown.append(call_id)
            continue
        valid.append(entry)
    return valid, unknown


def _item_prompt(item: RubricItem, rubric: Rubric) -> str:
    parts = [
        f"Item {item.id} (tier {item.tier}, {item.points} points)",
        f"Claim to test: {item.text}",
    ]
    if item.expect:
        parts.append(f"Expected: {item.expect}")
    if item.setup:
        parts.append(f"Do this first: {item.setup}")
    if item.note:
        parts.append(f"Note: {item.note}")
    if item.check is not None:
        parts.append(
            "This item is decided by code once you finish. Your job is to put "
            "the app into the state described above, then call `report`. Your "
            "verdict is advisory here; the harness has the last word."
        )
    else:
        parts.append(
            "You decide this one. Exercise it, then call `report` citing the "
            "tool call ids that prove your verdict."
        )
    parts.append(f"(App under test: {rubric.idea_id})")
    return "\n".join(parts)


async def grade_item(
    item: RubricItem,
    rubric: Rubric,
    tools: GraderTools,
    client: Any,
    *,
    max_steps: int = MAX_STEPS_PER_ITEM,
) -> ItemOutcome:
    """Navigate, then settle one item. The heart of the loop."""
    tools.current_item = item.id
    tools.captures.clear()
    mark = len(tools.page.requests()) if tools.page is not None else 0
    # Downloads are marked, not cleared: the transcript keeps the whole session's
    # files for evidence, while the check sees only what this item produced.
    download_mark = len(tools.downloads)
    messages: list[dict] = [{"role": "user", "content": _item_prompt(item, rubric)}]

    model_verdict: bool | None = None
    cited: list[str] = []
    note = ""
    steps = 0
    reported = False

    while steps < max_steps:
        reply = client.generate(messages, tools=TOOL_SCHEMAS, system=SYSTEM_PROMPT)
        if not reply.tool_calls:
            # No call and no report: nudge once, then give up on the item rather
            # than letting it consume the budget in prose.
            messages.append({"role": "assistant", "content": reply.text or "(silence)"})
            messages.append(
                {
                    "role": "user",
                    "content": "Use a tool, or call `report` with your verdict.",
                }
            )
            steps += 1
            continue

        assistant_blocks: list[dict] = []
        if reply.text:
            assistant_blocks.append({"type": "text", "text": reply.text})
        result_blocks: list[dict] = []
        done = False

        for call in reply.tool_calls:
            assistant_blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.args,
                }
            )
            if call.name == "report":
                model_verdict, cited, note = _parse_report(call.args)
                done = reported = True
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": "recorded",
                    }
                )
                continue
            output, ok = await tools.dispatch(call.name, call.args)
            # Log first, then show the model -- the record is the harness's, and
            # it exists whether or not the model ever mentions this call again.
            call_id = tools.record(call.name, call.args, output, ok=ok)
            result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": f"[{call_id}] {output}"[:8000],
                    "is_error": not ok,
                }
            )
            steps += 1

        messages.append({"role": "assistant", "content": assistant_blocks})
        messages.append({"role": "user", "content": result_blocks})
        if done:
            break

    if not reported:
        # Out of budget with no verdict. Recorded unresolved rather than FAIL:
        # the grader gave up, which is not evidence the app lacks the feature,
        # and burying that in the score would make a flaky grader look like a
        # finding. It still earns nothing -- it is just counted where it can be
        # seen.
        note = f"no verdict within {max_steps} steps"

    return await _settle(
        item,
        tools,
        mark,
        download_mark,
        model_verdict=model_verdict,
        cited=cited,
        note=note,
        steps=steps,
    )


async def _settle(
    item: RubricItem,
    tools: GraderTools,
    mark: int,
    download_mark: int = 0,
    *,
    model_verdict: bool | None,
    cited: Sequence[str],
    note: str,
    steps: int,
) -> ItemOutcome:
    """Decide the item: code where there is a check, bound evidence where not."""
    if item.check is not None:
        context = CheckContext(
            url=tools.url,
            page=tools.page,
            source=tools.source,
            scratch=tools.scratch,
            downloads=list(tools.downloads[download_mark:]),
            request_mark=mark,
            captures=dict(tools.captures),
        )
        result: CheckResult = await run_check(
            item.check.name, context, item.check.params
        )
        override = model_verdict is not None and result.passed != model_verdict
        return ItemOutcome(
            item_id=item.id,
            passed=result.passed,
            reason=result.detail or note,
            observed=result.observed,
            expected=item.expect,
            # A code-judged item's evidence is the harness's own call log for it,
            # not the model's citation -- the model did not decide this one.
            evidence=[entry.id for entry in tools.calls_for(item.id)],
            model_verdict=model_verdict,
            harness_override=override,
            steps=steps,
        )

    valid, unknown = _bind_evidence(tools, item.id, cited)
    if model_verdict is None:
        return ItemOutcome(
            item_id=item.id,
            passed=None,
            reason=note or "the grader could not determine this",
            expected=item.expect,
            evidence=[entry.id for entry in valid],
            model_verdict=None,
            steps=steps,
        )
    if not valid:
        # The anti-fabrication rule. A pass claimed without a call the harness
        # recorded is not a pass; a claimed fail is still a fail, so this only
        # ever costs points that were never evidenced.
        detail = "verdict cites no recorded tool call"
        if unknown:
            detail += f" (cited {', '.join(unknown[:4])}, none from this item)"
        return ItemOutcome(
            item_id=item.id,
            passed=False,
            reason=f"{detail}; recorded FAIL",
            expected=item.expect,
            evidence=[],
            model_verdict=model_verdict,
            harness_override=model_verdict is True,
            steps=steps,
        )
    return ItemOutcome(
        item_id=item.id,
        passed=model_verdict,
        reason=note,
        observed="; ".join(entry.result[:200] for entry in valid[:2]),
        expected=item.expect,
        evidence=[entry.id for entry in valid],
        model_verdict=model_verdict,
        steps=steps,
    )


async def run_pass(
    rubric: Rubric,
    tools: GraderTools,
    client: Any,
    *,
    items: Sequence[RubricItem] | None = None,
    max_steps: int = MAX_STEPS_PER_ITEM,
) -> dict[str, ItemOutcome]:
    """One full grading pass over the rubric's items."""
    walk = list(items) if items is not None else list(_gradeable(rubric))
    outcomes: dict[str, ItemOutcome] = {}
    for item in walk:
        outcomes[item.id] = await grade_item(
            item, rubric, tools, client, max_steps=max_steps
        )
    return outcomes


def _gradeable(rubric: Rubric) -> list[RubricItem]:
    """Every item the agent walks, gate first so a dead app fails fast."""
    return list(rubric.gate) + list(rubric.scored_items) + list(rubric.penalties)


def merge_passes(passes: Sequence[dict[str, ItemOutcome]]) -> dict[str, ItemVerdict]:
    """Fold N passes into one verdict per item; an item passes on a majority.

    The per-pass answers are kept, not just the majority, because the
    disagreement rate *is* the instrument's noise floor -- and a RubricScore gap
    smaller than that floor is not a finding.
    """
    verdicts: dict[str, ItemVerdict] = {}
    for outcomes in passes:
        for item_id, outcome in outcomes.items():
            verdict = verdicts.get(item_id)
            if verdict is None:
                verdict = ItemVerdict(item_id=item_id)
                verdicts[item_id] = verdict
            verdict.passes.append(outcome.passed)
            # Keep the first pass's narrative, and any later override flag: an
            # override in ANY pass is a reason to distrust the transcript.
            if not verdict.reason and outcome.reason:
                verdict.reason = outcome.reason
            if not verdict.observed and outcome.observed:
                verdict.observed = outcome.observed
            if not verdict.expected and outcome.expected:
                verdict.expected = outcome.expected
            for call_id in outcome.evidence:
                if call_id not in verdict.evidence:
                    verdict.evidence.append(call_id)
            verdict.harness_override = (
                verdict.harness_override or outcome.harness_override
            )
    return verdicts
