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

"""The founder agent harness: drives opencode to design + build an app.

This module splits into two layers:

* :class:`OpenCodeRunner` -- the reusable *primitive* that runs ONE opencode turn
  (``opencode run``) in a build workspace, backed by a Vertex AI model garden
  model -- Gemini or a partner model such as Claude (Application Default
  provider named by ``--model``). It owns the opencode wiring
  (binary, config, env, model preflight) and captures the raw JSON transcript of
  each turn for debugging/cost analysis, but knows nothing about how many turns a
  build takes.
* :class:`OpenCodeHarness` -- a thin adapter implementing :class:`FounderHarness`
  that pairs a runner with a *collaboration structure* (see
  :mod:`viral_bench.founder.structures`) and delegates the sequence of turns to
  it. The default structure is a single-agent specialist pipeline, i.e. the
  original Design -> Build baseline.

The harness is deliberately hidden behind the :class:`FounderHarness` protocol so
the rest of the pipeline never depends on opencode specifics (mirroring the
design doc's harness-agnostic ``FounderHarness`` interface).

opencode wiring notes:
- Model access goes through Vertex AI model garden, authenticated via Application
  Default Credentials (no API key). The opencode provider follows from the model:
  ``google-vertex`` for Gemini, ``google-vertex-anthropic`` for Claude (see
  :mod:`viral_bench.founder.models`). We scope the Vertex project + quota project
  to the opencode child-process env only (see :mod:`viral_bench.founder.vertex`),
  so this never disturbs a co-resident Cloud Code that relies on the ambient
  ``GOOGLE_CLOUD_PROJECT``.
- Config is passed via ``OPENCODE_CONFIG`` pointing at a file we write OUTSIDE
  the app directory, so the shipped app stays clean. It pins the model, pins the
  small model to the same Vertex model (so opencode never calls its default
  hosted small model), and disables session sharing so nothing leaves the machine.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from viral_bench import config as _config
from viral_bench.founder.models import (
    REQUIRED,
    UnknownModelError,
    resolve_model,
)
from viral_bench.founder.workspace import BuildWorkspace, builds_root
from viral_bench.ideas import Idea
from viral_bench.providers import (
    MissingCredentialError,
    ModelError,
    UnsupportedCapabilityError,
    check_support,
    make_client,
)
from viral_bench.providers.opencode import opencode_provider

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The configured founder model, or ``""`` when none is set.
#:
#: Empty is the shipped state and an intentional one: ViralBench presumes no
#: provider, so there is nothing honest to default to. Callers that reach a run
#: with an empty model must say so with :func:`require_model` rather than
#: silently picking someone's API to bill.
DEFAULT_MODEL = _config.founder_model_id("")


# Where opencode's standalone binary commonly lands (npm user-prefix, curl, etc).
_BINARY_CANDIDATES = (
    "opencode",
    str(Path.home() / ".npm-global" / "bin" / "opencode"),
    str(Path.home() / ".local" / "bin" / "opencode"),
    str(Path.home() / ".opencode" / "bin" / "opencode"),
)


#: Wall-clock backstop per model turn, from config/founder.yaml
#: ``timeouts_seconds`` (``None`` there, or here, means no limit).
#:
#: These exist to break a deadlock -- an opencode child wedged on a hung tool
#: call or a dead socket -- NOT to budget compute. That distinction matters
#: because a turn killed part-way returns rc=124 and is recorded as a failed
#: build: set them too tight and the harness converts "this model thinks for
#: longer" into "this model cannot ship an app", which is a fabricated
#: capability difference (docs/crowd_bugs.md T0.1). The previous
#: 1200/2400/2700 were tight enough to do exactly that on heavy team turns.
_DESIGN_TIMEOUT_S = _config.founder_timeout("design", 5400.0)
_BUILD_TIMEOUT_S = _config.founder_timeout("build", 10800.0)
_TEAM_TURN_TIMEOUT_S = _config.founder_timeout("team_turn", 10800.0)
#: One dynamic orchestrator turn is the largest unit of work in the bench: it may
#: contain an arbitrary number of subagent runs, sequential or concurrent, all
#: inside the single opencode process we are timing. A team turn is one model's
#: turn; this can be a whole team's round. Sized accordingly.
_DYNAMIC_TURN_TIMEOUT_S = _config.founder_timeout("dynamic_turn", 14400.0)

#: Capture the model's chain of thought, and opencode's own session store, for
#: every build. See ``trajectory:`` in config/founder.yaml for what each buys and
#: why neither changes what the model does.
_CAPTURE_THINKING = _config.founder_trajectory("thinking", True)
_CAPTURE_SESSIONS = _config.founder_trajectory("sessions", True)


class HarnessError(RuntimeError):
    """Raised when the founder harness cannot run (setup or execution error)."""


def require_model(model: str) -> str:
    """Return ``model``, or raise if no model has been chosen.

    Raises:
        HarnessError: with the two commands that fix it.
    """
    if model:
        return model
    raise HarnessError(
        "no founder model set. ViralBench ships no default provider -- pick "
        "one:\n"
        "  viral-bench init                 # walk through it interactively\n"
        "  viral-bench found --model openai/gpt-5-mini <idea>\n"
        "Run `viral-bench models` to see which providers you have keys for."
    )


@dataclass(frozen=True)
class PhaseResult:
    """Outcome of a single opencode turn (design or build).

    ``role``, ``agent_index`` and ``turn`` are populated for multi-agent
    specialist builds so the transcript can be attributed to a specific agent;
    they default to empty/zero for the solo (N=1) path.
    """

    phase: str
    returncode: int
    transcript_path: Path
    duration_s: float
    stderr_tail: str = ""
    role: str = ""
    agent_index: int = 0
    turn: str = ""  # "design" | "build" | "team"
    session_id: str = ""  # opencode session id (so an agent can resume its own)
    #: The turn was SIGKILLed by the wall-clock backstop rather than exiting on
    #: its own. Recorded separately because it is NOT evidence about the model's
    #: ability to build the app -- it is evidence about our timeout. Collapsing
    #: the two lets a slow-but-working model be scored as a broken one.
    timed_out: bool = False
    #: How much chain of thought this turn recorded, and where the full opencode
    #: session store for it was dumped. Counted rather than inlined so a build
    #: record stays readable; the text itself is in the transcript and the dump.
    #: Zero reasoning is a real result (Claude's `adaptive` thinking skips easy
    #: turns) -- and it is also what a capture regression looks like, so the
    #: numbers are recorded per turn rather than only totalled per build.
    reasoning_parts: int = 0
    reasoning_chars: int = 0
    sessions_path: str = ""
    sessions_records: int = 0

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class HarnessResult:
    """Aggregate outcome of a full Design -> Build run."""

    model: str
    phases: list[PhaseResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.phases) and all(p.ok for p in self.phases)


class FounderHarness(Protocol):
    """Anything that can run the founder's Design -> Build loop on the host."""

    def run(self, idea: Idea, workspace: BuildWorkspace) -> HarnessResult: ...


def find_opencode_binary() -> str | None:
    """Return the path to the opencode binary, or ``None`` if not found."""
    for candidate in _BINARY_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


# Markers that indicate a transient/capacity or auth problem worth calling out.
_OVERLOAD_MARKERS = (
    "high demand",
    "overloaded",
    "resource_exhausted",
    "unavailable",
    "rate limit",
    "quota",
    " 429",
    " 503",
)
_AUTH_MARKERS = ("api key", "unauthenticated", "permission denied", "403", "401")
#: What a provider says when the account exists but this particular model was
#: never enabled on it. It reads like a typo in the model name, so it is worth
#: translating -- the fix is an entitlement, not a retry.
_ENTITLEMENT_MARKERS = (
    "does not have access to it",
    "was not found or your project",
    "model_not_found",
    "does not exist or you do not have access",
)


def _diagnose(text: str) -> str:
    """Return a short, actionable hint for common failure signatures."""
    low = text.lower()
    if any(m in low for m in _ENTITLEMENT_MARKERS):
        return (
            "HINT: the credential is valid but this model is not enabled on the "
            "account. Check the id against the provider's own catalogue, and "
            "that your plan or project includes it -- several providers gate "
            "their largest models behind a separate opt-in."
        )
    if any(m in low for m in _OVERLOAD_MARKERS):
        return (
            "HINT: the provider returned overload/quota errors (transient). "
            "Retry later, lower --semaphore, or try a different --model. Quotas "
            "are often shared across a whole model lineage, and long-context "
            "team turns exhaust input tokens-per-minute well before "
            "requests-per-minute."
        )
    if any(m in low for m in _AUTH_MARKERS):
        return (
            "HINT: authentication or permission error. Run `viral-bench models` "
            "to see which providers have a credential, and `viral-bench doctor` "
            "to check the one you named end to end."
        )
    return ""


def _opencode_log_tail(n: int = 25) -> str:
    """Return the last ``n`` lines of opencode's own log (best-effort).

    Used as a fallback when opencode writes nothing to stdout/stderr (e.g. it
    was killed mid-retry), so a failure is never completely silent.
    """
    log = Path.home() / ".local" / "share" / "opencode" / "log" / "opencode.log"
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def preflight(model: str) -> None:
    """Quick ping to the provider so auth/entitlement failures surface in ~1s.

    Without this, a model the account cannot call burns the full design timeout
    inside opencode's retry loop and then reports as a build failure -- i.e. a
    setup problem laundered into evidence about the model.

    The probe itself belongs to the provider (see
    :meth:`viral_bench.providers.client.Adapter.ping`), which uses a free
    endpoint where one exists and a one-token generation otherwise.

    Args:
        model: A ``provider/model`` string.

    Raises:
        HarnessError: on any clear failure, with an actionable hint.
    """
    try:
        spec = resolve_model(require_model(model))
    except UnknownModelError as exc:
        raise HarnessError(str(exc)) from exc

    try:
        check_support(spec, REQUIRED, stage="founder")
    except UnsupportedCapabilityError as exc:
        raise HarnessError(str(exc)) from exc

    try:
        make_client(spec).ping()
    except MissingCredentialError as exc:
        # Not reachable-but-refused: simply not configured. The message already
        # says which variable to set and where to get a key.
        raise HarnessError(str(exc)) from exc
    except ModelError as exc:
        detail = str(exc).strip().splitlines()
        first = detail[0][:400] if detail else type(exc).__name__
        hint = _diagnose(first)
        msg = f"preflight failed for {spec.qualified!r}: {first}"
        raise HarnessError(f"{msg}\n{hint}" if hint else msg) from exc


def _thinking_options(model: str) -> dict:
    """Provider options that make this model's thinking come back *readable*.

    ``--thinking`` decides whether opencode PRINTS reasoning; this decides
    whether there is any reasoning text to print. The two providers differ, and
    the difference is not cosmetic:

    * **Gemini** needs nothing. opencode already sets
      ``thinkingConfig.includeThoughts``, plus ``thinkingLevel: high`` for a
      reasoning-capable one, for any model whose metadata says it can reason --
      so the thought summaries arrive on their own. Returning ``{}`` here keeps
      the request byte-identical to what opencode would have sent anyway.
    * **Claude** returns a thinking block that is real but ENCRYPTED unless the
      request asks for a summary: ``signature`` present, ``text`` empty. That is
      the worst possible failure mode, because the trace looks captured. Adding
      ``display: "summarized"`` is what turns it into prose. Measured on
      gemini-2.0-flash: identical request, ``adaptive`` alone gave 0
      characters, ``adaptive`` + ``summarized`` gave 290.

    Note what is deliberately NOT used here: opencode's own ``--variant high``.
    For every Claude model in our registry it resolves to
    ``thinking: {type: "enabled", budgetTokens: N}``, and Vertex rejects that
    outright -- ``"thinking.type.enabled" is not supported for this model``,
    HTTP 400. Verified against all four of gemini-2.0-flash, sonnet-5, opus-4-8 and
    opus-4-7. The obvious knob would have failed every Claude build rather than
    capturing anything, which is why the shape is spelled out here instead.

    ``effort`` rides along because ``adaptive`` lets the model decide whether to
    think at all; without it a Claude build's easy turns record no reasoning.
    """
    from viral_bench.founder.models import transport_for  # noqa: PLC0415

    if transport_for(model) != "anthropic":
        return {}
    return {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "effort": "high",
    }


def _parse_session_id(transcript_text: str) -> str:
    """Extract opencode's ``sessionID`` from a JSON turn transcript.

    ``opencode run --format json`` emits one JSON event per line, each carrying a
    ``sessionID``. We read it from the first parseable line so a specialist can
    resume *its own* session on a later turn via ``--session`` (the mechanism that
    lets each agent keep an independent context across interleaved rounds).
    Returns ``""`` if no id can be found.
    """
    for line in transcript_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            sid = event.get("sessionID") or event.get("sessionId")
            if sid:
                return str(sid)
    return ""


def _read_part_text(transcript_path: Path, event_type: str) -> list[str]:
    """Return the ``part.text`` of every ``event_type`` event in a transcript."""
    try:
        raw = transcript_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    chunks: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == event_type:
            part = event.get("part")
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
    return chunks


def read_assistant_text(transcript_path: Path) -> str:
    """Return the assistant's emitted text from a JSON turn transcript.

    Only ``type == "text"`` events carry the model's own output (the user prompt
    is passed on argv and is never echoed as a text event), so this is safe to
    scan for a control token like the QA ship signal without matching the prompt.
    Returns ``""`` if the transcript is missing or unparseable.

    Reasoning is deliberately NOT included, even now that we capture it. The
    control tokens are matched against this text: QA's ship signal
    (``structures._turn_signals_ship``) and the dynamic founder's done signal.
    A model that merely *considers* shipping -- "the app looks complete, I could
    emit READY_TO_SHIP now, but let me check the manifest first" -- would end its
    own build a turn early if its thinking counted as its answer. That is a
    harness artifact that would hit the models which think out loud hardest,
    i.e. a fabricated capability difference (docs/crowd_bugs.md T0.1). Use
    :func:`read_reasoning` when you want the thinking.
    """
    return "\n".join(_read_part_text(transcript_path, "text"))


def read_reasoning(transcript_path: Path) -> str:
    """Return the model's chain of thought from a JSON turn transcript.

    ``opencode run --thinking --format json`` emits the model's thoughts as
    ``type == "reasoning"`` events, in order, alongside the text and tool events.
    Empty when the turn recorded no reasoning -- which is a real outcome, not
    only a failure: Claude's ``adaptive`` thinking declines to think on easy
    turns, and a build run before ``trajectory.thinking`` was switched on has
    none at all.

    Never feed this to a control-token check; see :func:`read_assistant_text`.
    """
    return "\n".join(t for t in _read_part_text(transcript_path, "reasoning") if t)


def read_tool_calls(transcript_path: Path) -> list[tuple[str, dict]]:
    """Return ``(tool_name, input)`` for each tool call in a JSON turn transcript.

    ``opencode run --format json`` emits one JSON event per line; a tool call is a
    ``part`` of ``type == "tool"`` carrying the tool ``name`` and its ``input``.
    Used to check whether a turn actually *exercised* the app (e.g. a ``browser_*``
    interaction or a ``bash`` run of the app), not just reasoned about it. Returns
    ``[]`` if the transcript is missing or unparseable.
    """
    try:
        raw = transcript_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    calls: list[tuple[str, dict]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        part = event.get("part")
        if not isinstance(part, dict) or part.get("type") != "tool":
            continue
        name = part.get("tool")
        if not isinstance(name, str):
            continue
        state = part.get("state")
        inp = state.get("input") if isinstance(state, dict) else None
        calls.append((name, inp if isinstance(inp, dict) else {}))
    return calls


def read_task_spawns(transcript_path: Path) -> list[dict]:
    """Return one record per subagent the agent spawned in this turn.

    The ``task`` tool is how an opencode agent delegates: each call starts a child
    session that runs autonomously and returns a single message. For the dynamic
    founder mode that choice -- how many subagents, of what type, with what
    prompts, run in parallel or one after another -- IS the thing under test, so
    it has to be recorded rather than inferred from wall clock.

    Everything below was confirmed against opencode 1.17.14's own event stream:
    a ``task`` part carries ``input`` (``subagent_type``/``description``/
    ``prompt``, plus ``task_id`` when the agent is resuming a subagent it already
    used) and ``state.metadata`` (the child ``sessionId``, the ``parentSessionId``
    and the ``model`` the child actually ran on). ``time.start``/``time.end`` are
    epoch milliseconds, and they genuinely overlap when several tasks are issued
    in one message -- which is what makes "did this model parallelise?" an
    answerable question.

    Returns ``[]`` if the transcript is missing or has no task calls.
    """
    records: list[dict] = []
    try:
        raw = transcript_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return records
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        part = event.get("part")
        if not isinstance(part, dict) or part.get("type") != "tool":
            continue
        if part.get("tool") != "task":
            continue
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        inp = state.get("input") if isinstance(state.get("input"), dict) else {}
        meta = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
        times = state.get("time") if isinstance(state.get("time"), dict) else {}
        model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
        records.append(
            {
                "subagent_type": str(inp.get("subagent_type") or ""),
                "description": str(inp.get("description") or "")[:200],
                # The instruction the orchestrator wrote is the clearest evidence
                # of how it split the work, but a full prompt can be thousands of
                # characters and there may be dozens of them -- clip it.
                "prompt": str(inp.get("prompt") or "")[:1000],
                "prompt_chars": len(str(inp.get("prompt") or "")),
                # Present only when the orchestrator resumed a subagent instead of
                # spawning a fresh one, i.e. kept a persistent teammate.
                "resumed_task_id": str(inp.get("task_id") or ""),
                "status": str(state.get("status") or ""),
                "error": str(state.get("error") or "")[:300],
                "session_id": str(meta.get("sessionId") or ""),
                "parent_session_id": str(meta.get("parentSessionId") or ""),
                "model": (
                    f"{model.get('providerID')}/{model.get('modelID')}"
                    if model.get("modelID")
                    else ""
                ),
                "start_ms": times.get("start"),
                "end_ms": times.get("end"),
            }
        )
    return records


def concurrent_spawn_peak(spawns: list[dict]) -> int:
    """Peak number of subagents running at the same instant in one turn.

    A sweep-level question the raw count cannot answer: two models may each spawn
    six subagents, one strictly in sequence and the other three at a time, and
    those are very different orchestrations. Computed by sweeping the start/end
    intervals; spawns missing a timestamp are skipped rather than guessed at.
    """
    events: list[tuple[int, int]] = []
    for spawn in spawns:
        start, end = spawn.get("start_ms"), spawn.get("end_ms")
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            continue
        events.append((start, 1))
        events.append((end, -1))
    if not events:
        return 0
    # Close before open at an identical timestamp: two tasks that merely abut are
    # sequential, and counting them as concurrent would overstate parallelism.
    events.sort(key=lambda e: (e[0], e[1]))
    peak = live = 0
    for _, delta in events:
        live += delta
        peak = max(peak, live)
    return peak


def _opencode_db_path() -> Path:
    """Where opencode keeps its session database for THIS process's env."""
    root = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(root) / "opencode" / "opencode.db"


def session_tree(root_session_id: str) -> dict:
    """Best-effort shape of the subagent session tree under ``root_session_id``.

    The parent transcript records every ``task`` call the ORCHESTRATOR made, but
    a subagent can delegate too, and that grandchild appears only in the child's
    own session. "Six subagents" and "six subagents, two of which ran teams of
    their own" are different orchestrations, and only the session tree separates
    them.

    Read straight from opencode's SQLite session store, which is scoped by
    ``XDG_DATA_HOME`` (the fleet gives every build its own). Strictly
    best-effort: that store is internal with no compatibility promise, so any
    failure -- missing file, renamed table, locked database -- returns empty
    rather than affecting the build. ``max_depth`` of 1 means the founder
    delegated but no subagent did.

    Returns ``{"sessions": int, "max_depth": int, "titles": [...]}``, counting
    descendants only, not the root.
    """
    empty: dict = {"sessions": 0, "max_depth": 0, "titles": []}
    if not root_session_id:
        return empty
    db = _opencode_db_path()
    if not db.is_file():
        return empty
    try:
        import sqlite3

        # Read-only and never-create: this must not perturb a store opencode may
        # still be writing to.
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)
        try:
            rows = list(con.execute("SELECT id, parent_id, title FROM session"))
        finally:
            con.close()
    except Exception:  # noqa: BLE001 - telemetry must never fail a build
        return empty

    children: dict[str, list[tuple[str, str]]] = {}
    for sid, parent, title in rows:
        if parent:
            children.setdefault(str(parent), []).append((str(sid), str(title or "")))

    titles: list[str] = []
    max_depth = 0
    frontier = [(root_session_id, 0)]
    seen = {root_session_id}
    while frontier:
        node, depth = frontier.pop()
        for child_id, title in children.get(node, ()):
            if child_id in seen:  # a cycle would otherwise hang the walk
                continue
            seen.add(child_id)
            titles.append(title[:120])
            max_depth = max(max_depth, depth + 1)
            frontier.append((child_id, depth + 1))
    return {"sessions": len(titles), "max_depth": max_depth, "titles": titles}


def _descendant_session_ids(con, root_session_id: str) -> list[str]:
    """Return ``root_session_id`` followed by every session beneath it.

    Walks ``session.parent_id`` breadth-first over the indexed column, so this
    stays cheap even against the shared store (which has reached 49 GB on this
    machine). Cycle-guarded: a malformed store must not hang a build.
    """
    ordered = [root_session_id]
    seen = {root_session_id}
    frontier = [root_session_id]
    while frontier:
        parent = frontier.pop()
        rows = con.execute(
            "SELECT id FROM session WHERE parent_id = ?", (parent,)
        ).fetchall()
        for (child,) in rows:
            child = str(child)
            if child in seen:
                continue
            seen.add(child)
            ordered.append(child)
            frontier.append(child)
    return ordered


def dump_session_trace(root_session_id: str, dest: Path) -> int:
    """Write opencode's own record of ``root_session_id`` to ``dest`` as JSONL.

    The JSON transcript we capture from stdout is not the whole build, and the
    gaps are exactly the parts worth keeping:

    * **The prompts are missing.** ``run_turn`` sends the prompt on stdin and
      opencode never echoes it as an event, so the transcript records what the
      model *said* and not what it was *asked*. A trajectory without its inputs
      cannot train anything.
    * **Subagents are missing.** ``opencode run``'s printer skips every event
      whose ``sessionID`` is not the root session, so in dynamic mode -- the arm
      whose whole point is delegation -- none of the delegated work is recorded.
      Only the store has the child sessions.
    * **File patches are missing.** ``patch`` parts never reach stdout at all.

    All three live in the session store, which is why this reads the store
    rather than post-processing the transcript. It has to run DURING the build:
    ``scripts/build_fleet.py`` deletes the per-cell store the moment a cell
    succeeds, so a dump that waited until afterwards would find nothing for
    every build that worked.

    Scoped deliberately to this build's own session subtree, never "dump the
    database". Under the fleet the store is private to one cell, but a bare
    ``viral-bench found`` shares the global one, and dumping that would copy out
    every unrelated session on the machine.

    Best-effort, exactly like :func:`session_tree`: the store is opencode's
    internal format with no compatibility promise, so any failure writes nothing
    and returns 0 rather than failing a build over telemetry.

    Returns the number of records written (0 if nothing could be read).
    """
    if not root_session_id:
        return 0
    db = _opencode_db_path()
    if not db.is_file():
        return 0
    try:
        import sqlite3

        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10.0)
        try:
            session_ids = _descendant_session_ids(con, root_session_id)
            records = _collect_session_records(con, session_ids)
        finally:
            con.close()
    except Exception:  # noqa: BLE001 - telemetry must never fail a build
        return 0

    if not records:
        return 0
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return 0
    return len(records)


def _collect_session_records(con, session_ids: list[str]) -> list[dict]:
    """Build the ordered session/message/part records for ``session_ids``."""
    depth_of = {sid: index for index, sid in enumerate(session_ids)}
    records: list[dict] = []

    placeholders = ",".join("?" * len(session_ids))
    sessions = con.execute(
        f"SELECT id, parent_id, title, agent, model, cost, tokens_input, "  # noqa: S608
        f"tokens_output, tokens_reasoning, tokens_cache_read, tokens_cache_write, "
        f"time_created FROM session WHERE id IN ({placeholders})",
        session_ids,
    ).fetchall()
    for row in sessions:
        records.append(
            {
                "record": "session",
                "session_id": str(row[0]),
                "parent_session_id": str(row[1] or ""),
                "title": row[2] or "",
                "agent": row[3] or "",
                "model": row[4] or "",
                "cost": row[5],
                "tokens": {
                    "input": row[6],
                    "output": row[7],
                    "reasoning": row[8],
                    "cache_read": row[9],
                    "cache_write": row[10],
                },
                "time_created": row[11],
                "order": depth_of.get(str(row[0]), 0),
            }
        )

    # role/agent/model live on the message, and every part needs them to be
    # interpretable on its own -- an SFT consumer reading one line must be able
    # to tell a user prompt from a model's answer without holding the file in
    # memory.
    message_meta: dict[str, dict] = {}
    for mid, sid, created, data in con.execute(
        f"SELECT id, session_id, time_created, data FROM message "  # noqa: S608
        f"WHERE session_id IN ({placeholders})",
        session_ids,
    ):
        try:
            payload = json.loads(data) if data else {}
        except (json.JSONDecodeError, TypeError):
            payload = {}
        model = payload.get("model") if isinstance(payload.get("model"), dict) else {}
        meta = {
            "role": str(payload.get("role") or ""),
            "agent": str(payload.get("agent") or ""),
            "model": (
                f"{model.get('providerID')}/{model.get('modelID')}"
                if model.get("modelID")
                else ""
            ),
            "session_id": str(sid),
            "time_created": created,
        }
        message_meta[str(mid)] = meta
        records.append({"record": "message", "message_id": str(mid), **meta})

    for pid, mid, sid, created, data in con.execute(
        f"SELECT id, message_id, session_id, time_created, data FROM part "  # noqa: S608
        f"WHERE session_id IN ({placeholders})",
        session_ids,
    ):
        try:
            payload = json.loads(data) if data else {}
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        meta = message_meta.get(str(mid), {})
        records.append(
            {
                "record": "part",
                "part_id": str(pid),
                "message_id": str(mid),
                "session_id": str(sid),
                "role": meta.get("role", ""),
                "agent": meta.get("agent", ""),
                "model": meta.get("model", ""),
                "time_created": created,
                "part": payload,
            }
        )

    records.sort(key=lambda r: (r.get("time_created") or 0, r.get("record") or ""))
    return records


#: Set to "1" to leave agent-started processes running after a turn. Escape hatch
#: for debugging a build by hand; never set it for a scored run.
NO_REAP_ENV = "VIRAL_BENCH_NO_REAP"


def _processes_with_cwd_under(root: Path, exclude: frozenset[int]) -> list[tuple]:
    """Return ``(pid, cmdline)`` for live processes whose cwd is inside ``root``.

    Linux only -- it reads ``/proc/<pid>/cwd``. Returns ``[]`` elsewhere rather
    than pretending, so a non-Linux host degrades to today's behaviour instead of
    failing.
    """
    proc_dir = Path("/proc")
    if not proc_dir.is_dir():
        return []
    try:
        target = root.resolve()
    except OSError:
        return []
    found: list[tuple] = []
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in exclude:
            continue
        try:
            cwd = Path(os.readlink(entry / "cwd")).resolve()
        except OSError:
            # Exited between listdir and readlink, or another user's process.
            continue
        if cwd != target and target not in cwd.parents:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            raw = b""
        cmd = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        found.append((pid, cmd or f"pid {pid}"))
    return found


def reap_workspace_processes(root: Path, *, grace_s: float = 5.0) -> list[tuple]:
    """Kill processes an agent left running inside a build workspace.

    Agents start the app they are building and routinely do not stop it. Measured
    on this repo: a ``python3 server.py`` outlived its build by 36 minutes, and a
    three-round build hit ``Address already in use`` six times and ran eight
    port-probing commands fighting infrastructure it had leaked itself. The real
    hazard is across builds -- build B's QA binds and fails, ``curl`` still
    answers 200 from build A's server, and QA verifies and ships B while looking
    at A. That is docs/crowd_bugs.md T0.1 in a new costume: a harness race that
    hits one model more than the other reads out as a capability difference.

    **Why cwd and not the process group.** The obvious implementation -- run
    opencode with ``start_new_session=True`` and ``killpg`` the group afterwards
    -- does not work, and it took a live probe to find that out. opencode's bash
    tool spawns detached: a leaked server measured here had ``pgid == sid ==``
    its own shell's pid, nothing to do with opencode's group, reparented to
    systemd. ``killpg`` on the turn's group reaches none of it. What *does*
    identify it is its working directory: opencode runs with ``cwd`` set to the
    build's ``app_dir``, so anything an agent starts inherits a cwd inside this
    build's workspace. That is precise (it cannot touch another build, which is
    the whole point) and attributable.

    This is a backstop, not a substitute for isolation. ``scripts/netns_run.sh``
    is still the real fix for the fleet -- a private PID namespace means the
    kernel reaps the survivors and ``pkill`` cannot cross builds. This exists for
    the bare-host path (``viral-bench found``, ``scripts/smoke_founder.py``),
    which has no namespace at all.

    Call it only once the turn's opencode process has exited; anything still
    running in the workspace at that point is by definition something the agent
    left behind.

    Returns the ``(pid, cmdline)`` pairs it killed, so a caller can record them --
    a leak nobody counts is a leak nobody fixes.
    """
    if os.environ.get(NO_REAP_ENV) == "1":
        return []
    exclude = frozenset({os.getpid(), os.getppid()})
    leaked = _processes_with_cwd_under(root, exclude)
    if not leaked:
        return []

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    for pid, _ in leaked:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not any(alive(pid) for pid, _ in leaked):
            break
        time.sleep(0.1)
    for pid, _ in leaked:
        if alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    return leaked


class OpenCodeRunner:
    """Reusable primitive: run ONE opencode turn in a build workspace.

    Owns the opencode wiring (binary, model, config, env, model preflight, and
    transcript capture) but knows nothing about *how many* turns a build takes or
    who runs them -- that is a collaboration structure's job (see
    :mod:`viral_bench.founder.structures`). Call :meth:`prepare` once per build
    (writes the opencode config and pings the model), then :meth:`run_turn` for
    each turn in the sequence.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        binary: str | None = None,
        design_timeout_s: float | None = _DESIGN_TIMEOUT_S,
        build_timeout_s: float | None = _BUILD_TIMEOUT_S,
        team_turn_timeout_s: float | None = _TEAM_TURN_TIMEOUT_S,
        dynamic_turn_timeout_s: float | None = _DYNAMIC_TURN_TIMEOUT_S,
        env_file: Path | None = None,
        preflight: bool = True,
        config_extra: dict | None = None,
        skills: list[str] | None = None,
        capture_thinking: bool = _CAPTURE_THINKING,
        capture_sessions: bool = _CAPTURE_SESSIONS,
    ) -> None:
        self.model = model
        self._binary = binary
        self.design_timeout_s = design_timeout_s
        self.build_timeout_s = build_timeout_s
        self.team_turn_timeout_s = team_turn_timeout_s
        self.dynamic_turn_timeout_s = dynamic_turn_timeout_s
        self.env_file = env_file if env_file is not None else _REPO_ROOT / ".env"
        # Ping the model before a long run so overload/auth fail in ~1s, not
        # after the full design timeout. Tests disable this.
        self.preflight = preflight
        # Extra opencode config merged into the per-build config (e.g. per-role
        # `agent` definitions + browser MCP), and the skills to install for them.
        self._config_extra = config_extra or {}
        self._skills = skills or []
        # Record the model's reasoning, and opencode's full session store, for
        # every turn (config/founder.yaml `trajectory:`).
        self.capture_thinking = capture_thinking
        self.capture_sessions = capture_sessions
        self._env: dict[str, str] | None = None

    # -- setup helpers -------------------------------------------------------

    def binary(self) -> str:
        found = self._binary or find_opencode_binary()
        if not found:
            raise HarnessError(
                "opencode binary not found. Install it (e.g. `npm i -g "
                "opencode-ai`) and ensure it is on PATH."
            )
        return found

    def _config_json(self) -> str:
        """opencode config: pin model + Vertex small model, disable sharing.

        Any :attr:`_config_extra` (e.g. per-role ``agent`` definitions and an
        optional browser MCP built by :mod:`viral_bench.founder.opencode_agents`)
        is merged in at the top level, so team builds run specialised, capability-
        differentiated agents while the solo path stays the plain baseline config.
        """
        target = self._opencode_target()
        config = {
            "$schema": "https://opencode.ai/config.json",
            "share": "disabled",
            "model": target.model,
            "small_model": target.model,
            # `small_model` matters more than it looks: left unset, opencode uses
            # its own hosted default for side tasks like session titles, which
            # would put a second, unasked-for provider in the loop.
            "provider": target.provider_block,
        }
        # Top-level keys in _config_extra (agent/mcp/tools) never collide with the
        # base keys above, so a shallow merge is safe and predictable.
        config.update(self._config_extra)
        return json.dumps(config, indent=2)

    def _opencode_target(self):
        """Resolve this build's model into opencode config + child environment.

        Cached: the credential lookup and (for ambient auth) the token mint are
        not free, and this is read once per turn.
        """
        if getattr(self, "_target", None) is None:
            spec = resolve_model(self.model)
            self._target = opencode_provider(
                spec, thinking_options=_thinking_options(self.model)
            )
        return self._target

    def _build_env(self, config_path: Path) -> dict[str, str]:
        env = dict(os.environ)
        # Provider credentials go to THIS opencode child only -- never exported
        # into our own process -- so a sweep can run two arms against two
        # providers at once, and an ambient cloud project set by the surrounding
        # environment cannot leak into a build that named a different one.
        env.update(self._opencode_target().env)
        env["OPENCODE_CONFIG"] = str(config_path)
        env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
        # Backstop for the workspace's own repo (BuildWorkspace.init_git_boundary):
        # halt git's upward search below the ViralBench checkout, so a stray `git`
        # cannot reach our history even if that repo is missing or was deleted
        # mid-build. Prepend rather than overwrite so a caller's ceiling survives.
        ceiling = str(builds_root().parent)
        inherited = env.get("GIT_CEILING_DIRECTORIES")
        env["GIT_CEILING_DIRECTORIES"] = (
            f"{ceiling}:{inherited}" if inherited else ceiling
        )
        return env

    def timeout_for(self, turn: str) -> float | None:
        """Return the configured timeout for a turn kind, None for no limit.

        ``"build"``, ``"team"`` (a collaborative team turn, which both plans and
        edits) and ``"dynamic"`` (an orchestrator turn, which may contain a whole
        fan-out of subagent runs) get the longer budgets; anything else uses the
        design budget.
        """
        if turn == "build":
            return self.build_timeout_s
        if turn == "team":
            return self.team_turn_timeout_s
        if turn == "dynamic":
            return self.dynamic_turn_timeout_s
        return self.design_timeout_s

    def prepare(self, workspace: BuildWorkspace) -> PhaseResult | None:
        """Write the opencode config and (optionally) preflight the model.

        Returns a ``"preflight"`` :class:`PhaseResult` if setup fails -- so the
        caller can abort before launching opencode -- or ``None`` on success, at
        which point the turn env is ready for :meth:`run_turn`.
        """
        config_path = workspace.app_dir.parent / "opencode.json"
        config_path.write_text(self._config_json(), encoding="utf-8")
        transcript_dir = workspace.app_dir.parent / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        # Install this build's role skills into a private .opencode/skills dir at
        # the workspace root. opencode discovers skills by walking up from --dir
        # (app_dir) to the git worktree, so they are visible to the agents but sit
        # OUTSIDE app_dir and therefore never ship inside the built app.
        if self._skills:
            from viral_bench.founder.skills import write_skills

            write_skills(workspace.root / ".opencode" / "skills", self._skills)

        # Fail fast on a missing credential or an overloaded/unauthorized model
        # rather than burning the full design timeout inside opencode's retries.
        try:
            if self.preflight:
                preflight(self.model)
        except HarnessError as exc:
            path = transcript_dir / "preflight.log"
            path.write_text(str(exc), encoding="utf-8")
            return PhaseResult(
                phase="preflight",
                returncode=1,
                transcript_path=path,
                duration_s=0.0,
                stderr_tail=str(exc)[:800],
            )

        self._env = self._build_env(config_path)
        return None

    # -- execution -----------------------------------------------------------

    def run_turn(
        self,
        prompt: str,
        *,
        workspace: BuildWorkspace,
        phase: str,
        turn: str,
        continue_session: bool,
        timeout_s: float | None = None,
        role: str = "",
        agent_index: int = 0,
        extra_env: dict[str, str] | None = None,
        agent: str = "",
        session_id: str = "",
    ) -> PhaseResult:
        """Run one opencode turn and capture its transcript.

        Args:
            prompt: The message to send to opencode.
            workspace: The build workspace (opencode runs with ``--dir app_dir``).
            phase: Transcript label (also the transcript filename stem). For the
                solo path this is ``"design"``/``"build"``; team builds use a
                per-agent, per-round label like ``"r2_a3_designer"``.
            turn: The turn kind, ``"design"``/``"build"``/``"team"`` (chooses the
                default timeout and is recorded on the result).
            continue_session: If true, pass ``--continue`` so this turn resumes the
                most recent session (used to link the solo build turn to its own
                design turn); ignored when ``session_id`` is given.
            timeout_s: Override the timeout; defaults to :meth:`timeout_for`.
            role: Role key, recorded on the result (team builds only).
            agent_index: 1-based agent position, recorded on the result.
            extra_env: Per-turn env overrides merged over the prepared env (used by
                a collaboration toolset to inject, e.g., an OTA token + PATH for
                this specific agent's turn).
            agent: opencode agent name to run as (``--agent``); empty uses the
                default build agent. Team turns pass the specialist's role key.
            session_id: Resume this exact opencode session (``--session``) so a
                specialist keeps its OWN context across interleaved rounds. Empty
                starts a fresh session (or continues the last, per
                ``continue_session``). The turn's actual session id is parsed from
                the transcript and returned on the result.
        """
        if self._env is None:
            raise HarnessError(
                "OpenCodeRunner.run_turn called before prepare() -- no env is ready."
            )
        import subprocess  # local import keeps module import cheap/testable

        if timeout_s is None:
            timeout_s = self.timeout_for(turn)

        env = self._env if not extra_env else {**self._env, **extra_env}

        argv = [
            self.binary(),
            "run",
            "--dir",
            str(workspace.app_dir),
            "--model",
            self.model,
            "--format",
            "json",
            "--auto",
            "--print-logs",  # route opencode's logs to stderr so we capture them
        ]
        if self.capture_thinking:
            # Keep the model's chain of thought. opencode already ASKS the
            # provider for thoughts and we already pay for the tokens; this flag
            # only decides whether they are printed, and it defaults to off --
            # which is why every build before this silently discarded its
            # reasoning while being billed 18-32k thinking tokens a session.
            # Display-only: it changes no request, so a build with it on is
            # comparable to one without.
            argv.append("--thinking")
        if agent:
            argv += ["--agent", agent]
        # Resuming a specific session takes precedence over --continue: it lets an
        # agent reload its own context regardless of which teammate ran last.
        if session_id:
            argv += ["--session", session_id]
        elif continue_session:
            argv.append("--continue")
        # The prompt goes on STDIN, never on argv. `opencode run` reads its
        # message from stdin when no positional message is given, and the model
        # receives byte-identical text either way -- but a prompt on argv is a
        # prompt in /proc/<pid>/cmdline, and that turned the founder's own
        # instructions into a landmine.
        #
        # Our single-page-app guidance contains the literal string
        # "python3 -m http.server". A QA agent that starts the app that way and
        # then tidies up with `pkill -f "python3 -m http.server"` -- a completely
        # ordinary thing to do -- matched the opencode process running its own
        # turn and killed it. Measured on this fleet: four builds died with
        # rc=-15 mid-QA-turn, three of them one model's. A model that happens to
        # prefer `pkill -f` over `kill $PID` would have been recorded as less
        # able to finish a build, which is a fabricated capability difference of
        # exactly the kind in docs/crowd_bugs.md T0.1.

        transcript_dir = workspace.app_dir.parent / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = transcript_dir / f"{phase}.json"

        start = time.monotonic()
        timed_out = False
        try:
            proc = subprocess.run(
                argv,
                env=env,
                input=prompt,
                # Without this opencode inherits *our* cwd -- the repo root --
                # so any command the agent runs before it cds lands in the
                # ViralBench checkout rather than the app it is building.
                # `--dir` only scopes opencode's own tools, not the shell.
                cwd=str(workspace.app_dir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            # Preserve whatever was captured before the kill (do NOT drop stderr).
            timed_out = True
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            returncode = 124
        duration = time.monotonic() - start

        # opencode has exited, so anything still running inside this build's
        # workspace is something the agent started and did not stop -- usually the
        # app's own server, still holding its port. Kill it before the next turn
        # (or the next build) has to fight it. See reap_workspace_processes.
        reaped = reap_workspace_processes(workspace.root)
        if reaped:
            with (workspace.root / "reaped.log").open("a", encoding="utf-8") as log:
                for pid, cmd in reaped:
                    log.write(f"{phase}\t{pid}\t{cmd}\n")

        transcript_path.write_text(stdout or "", encoding="utf-8")
        (transcript_dir / f"{phase}.stderr.log").write_text(
            stderr or "", encoding="utf-8"
        )

        # Recover the session id opencode used this turn (empty if it never got
        # far enough to emit one) so the caller can resume this exact context.
        emitted_session = _parse_session_id(stdout or "") or session_id

        # Never fail silently: if opencode emitted nothing to stderr (e.g. killed
        # mid-retry), fall back to its own log file so the real cause surfaces.
        tail_source = (stderr or "").strip()
        if not tail_source and returncode != 0:
            tail_source = _opencode_log_tail()
        parts: list[str] = []
        if timed_out:
            parts.append(f"timed out after {(timeout_s or 0):.0f}s")
        if returncode != 0:
            hint = _diagnose(tail_source)
            if hint:
                parts.append(hint)
        if tail_source:
            parts.append("\n".join(tail_source.splitlines()[-20:]))
        stderr_tail = "\n".join(parts).strip()

        reasoning = read_reasoning(transcript_path)
        reasoning_parts = sum(
            1 for chunk in _read_part_text(transcript_path, "reasoning") if chunk
        )

        # Dump opencode's own store for this turn's session subtree, now, while
        # it still exists: the fleet deletes a cell's store as soon as the cell
        # succeeds. Re-dumped every turn rather than once at the end, so a build
        # that the wall-clock backstop SIGKILLs mid-turn still keeps every turn
        # that completed. Rewriting one file per session keeps that idempotent.
        sessions_path = ""
        sessions_records = 0
        if self.capture_sessions and emitted_session:
            dest = transcript_dir / "sessions" / f"{emitted_session}.jsonl"
            sessions_records = dump_session_trace(emitted_session, dest)
            if sessions_records:
                sessions_path = str(dest)

        return PhaseResult(
            phase=phase,
            returncode=returncode,
            transcript_path=transcript_path,
            duration_s=duration,
            stderr_tail=stderr_tail,
            role=role,
            agent_index=agent_index,
            turn=turn,
            session_id=emitted_session,
            timed_out=timed_out,
            reasoning_parts=reasoning_parts,
            reasoning_chars=len(reasoning),
            sessions_path=sessions_path,
            sessions_records=sessions_records,
        )


class OpenCodeHarness:
    """Founder harness backed by opencode + a Vertex model garden model.

    Thin adapter implementing :class:`FounderHarness`: it owns an
    :class:`OpenCodeRunner` (the per-turn opencode primitive) and delegates the
    *sequence* of turns to a collaboration structure. When ``structure`` is not
    given it defaults to the solo pipeline -- the original Design -> Build
    baseline. An optional collaboration ``toolset`` (local by default) decides how
    agents collaborate during their turns.

    For a team structure the harness turns each specialist role into a distinct,
    capability-differentiated opencode agent (own prompt, temperature,
    permissions, gated skills, and -- when ``browser_tools`` is on -- a real
    browser); the solo path gets none of that and stays byte-for-byte the
    baseline. The dynamic structure gets a third shape: no bench-authored agents
    or skills at all, delegation explicitly allowed, and the browser available to
    every agent -- the specialists in that build are the ones the model writes.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        binary: str | None = None,
        design_timeout_s: float | None = _DESIGN_TIMEOUT_S,
        build_timeout_s: float | None = _BUILD_TIMEOUT_S,
        team_turn_timeout_s: float | None = _TEAM_TURN_TIMEOUT_S,
        dynamic_turn_timeout_s: float | None = _DYNAMIC_TURN_TIMEOUT_S,
        env_file: Path | None = None,
        preflight: bool = True,
        structure: object | None = None,
        toolset: object | None = None,
        browser_tools: bool = True,
        capture_thinking: bool = _CAPTURE_THINKING,
        capture_sessions: bool = _CAPTURE_SESSIONS,
    ) -> None:
        if structure is None:
            # Local import avoids a module-level import cycle (structures imports
            # this module for the runner primitive).
            from viral_bench.founder.structures import SoloPipeline

            structure = SoloPipeline(1)
        self.structure = structure
        self.toolset = toolset

        # Three shapes of opencode config, one per founder mode:
        #   solo     the plain baseline -- no custom agents, no skills.
        #   team     one specialised agent per role, with role-gated skills.
        #   dynamic  stock agent, delegation allowed, browser for everyone, and
        #            deliberately NO bench-authored agents or skills, so the only
        #            specialists in the build are ones the model invented.
        config_extra: dict = {}
        skills: list[str] = []
        roles = list(getattr(structure, "roles", []))
        if getattr(structure, "name", "") == "dynamic":
            from viral_bench.founder.opencode_agents import (
                browser_prereqs_ok,
                build_dynamic_config,
            )

            browser_tools = bool(browser_tools) and browser_prereqs_ok()
            config_extra = build_dynamic_config(browser_tools=browser_tools)
        elif getattr(structure, "n_agents", 1) > 1 and roles:
            from viral_bench.founder.opencode_agents import (
                browser_prereqs_ok,
                build_agents_config,
                skills_for_roles,
            )

            # Auto-detect: only wire the browser MCP when it can actually launch on
            # this host. Otherwise degrade gracefully (the Designer/QA reason from
            # markup) instead of failing the build over missing browser tooling.
            browser_tools = bool(browser_tools) and browser_prereqs_ok()
            config_extra = build_agents_config(
                roles, model=model, browser_tools=browser_tools
            )
            skills = skills_for_roles(roles)
        self.browser_tools = browser_tools

        self._runner = OpenCodeRunner(
            model,
            binary=binary,
            design_timeout_s=design_timeout_s,
            build_timeout_s=build_timeout_s,
            team_turn_timeout_s=team_turn_timeout_s,
            dynamic_turn_timeout_s=dynamic_turn_timeout_s,
            env_file=env_file,
            preflight=preflight,
            config_extra=config_extra,
            skills=skills,
            capture_thinking=capture_thinking,
            capture_sessions=capture_sessions,
        )

    # -- backwards-compatible delegation to the runner -----------------------

    @property
    def model(self) -> str:
        return self._runner.model

    def binary(self) -> str:
        return self._runner.binary()

    def _config_json(self) -> str:
        return self._runner._config_json()

    # -- execution -----------------------------------------------------------

    def run(self, idea: Idea, workspace: BuildWorkspace) -> HarnessResult:
        """Design + build ``idea`` in ``workspace`` via the collaboration structure."""
        setup_error = self._runner.prepare(workspace)
        if setup_error is not None:
            return HarnessResult(model=self._runner.model, phases=[setup_error])
        return self.structure.run(idea, workspace, self._runner, toolset=self.toolset)
