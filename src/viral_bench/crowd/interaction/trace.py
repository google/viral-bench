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

"""The structured record of one agent's hands-on trial of one app.

Every interaction the crowd has with an app -- each click, command, or chat turn,
and the observation it produced -- is appended to an
:class:`InteractionTrace` as an :class:`InteractionStep`. The trace serves three
jobs at once:

* **evidence for scoring**: the delight/adoption signal the scoring stage derives
  starts from what the agent actually did and saw (:meth:`InteractionTrace.render`
  feeds a compact version back to the deciding LLM);
* **an audit trail**: it proves the agent genuinely ran the app rather than
  hallucinating a verdict (the crowd analogue of the founder's ``qa_verified``
  ship-evidence gate);
* **a replayable log**: :meth:`InteractionTrace.jsonl_lines` emits one JSON object
  per step (round/action/args/result/success), the append-only action-log shape
  the design doc reuses from MiroFish for dashboards and post-hoc metrics.

The module is deliberately dependency-free (pure dataclasses + json) so it can be
imported anywhere -- tests, the CLI, or the eventual OASIS crowd loop -- without
pulling in Playwright, containers, or an agent framework.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

# Keep any single serialized field from blowing up a log line / an LLM context.
_MAX_SUMMARY = 4000
_MAX_ERROR = 500


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars]"


@dataclass
class InteractionStep:
    """One action an agent took against an app, plus what it observed.

    Attributes:
        index: 0-based position of this step within its trace.
        action: The verb performed, e.g. ``"open"``, ``"look"``, ``"click"``,
            ``"type"``, ``"press"``, ``"run_command"``, ``"send_message"``,
            ``"screenshot"``, or ``"finish"``.
        args: The action's arguments (target, text, command, ...), JSON-able.
        summary: A short, LLM-readable description of what happened / was seen.
        ok: Whether the action itself completed (a command running to a non-zero
            exit still ``ok=True``; a driver error / timeout is ``ok=False``).
        errors: Runtime errors observed as a side effect (browser console errors,
            page errors, stderr), each clipped for logging.
        screenshot: Path to a screenshot captured for this step, if any.
        ts: Epoch seconds when the step completed.
        duration_s: Wall-clock seconds the action took.
    """

    index: int
    action: str
    args: dict = field(default_factory=dict)
    summary: str = ""
    ok: bool = True
    errors: tuple[str, ...] = ()
    screenshot: str | None = None
    ts: float = field(default_factory=time.time)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["summary"] = _clip(self.summary, _MAX_SUMMARY)
        data["errors"] = [_clip(e, _MAX_ERROR) for e in self.errors]
        return data


@dataclass
class TrialVerdict:
    """The agent's own self-report at the end of a trial.

    This is the natural handoff into the scoring stage: rather than scoring the
    raw trace blind, the deciding agent states, having actually used the app,
    whether it would use and share it and how delightful it was.

    The facet ratings turn this from a single-item judgement into a short scale.
    One item is a noisy instrument; averaging several facets of the same
    underlying "is this any good" judgement raises reliability substantially
    (Spearman-Brown), and the facets double as diagnostics -- they say *why* an
    app scored badly, which a lone delight number cannot. They are optional, so
    a malformed tool call still records the core verdict and traces recorded
    before the facets existed still load.

    Attributes:
        would_use: Would this agent adopt the app?
        would_share: Would this agent share/repost it to others?
        delight: Subjective delight, ``0`` (bad) .. ``10`` (loved it).
        notes: Free-text justification grounded in what the agent observed.
        functionality: Does it actually work and do what it claims, ``0``..``10``.
        usability: Could you figure it out and get the job done, ``0``..``10``.
        design: Visual and interaction craft, ``0``..``10``.
        simplicity: Focused and friction-free vs bloated, ``0``..``10``.
    """

    would_use: bool
    would_share: bool
    delight: int
    notes: str = ""
    functionality: int | None = None
    usability: int | None = None
    design: int | None = None
    simplicity: int | None = None
    #: Did the agent's own work survive a page reload? ``None`` = never checked.
    #: Recorded and reported, deliberately NOT scored yet: the honest order is to
    #: find out how often an app really keeps anything before deciding what that
    #: is worth. Until arch v10 no agent could even ask -- there was no reload
    #: verb -- so a page that forgets everything on refresh and a database-backed
    #: one were indistinguishable to this benchmark.
    work_survived: bool | None = None
    #: Could the agent see anything another crowd member made in the same shared
    #: instance? ``None`` = never checked / not applicable. It is the property a
    #: multi-user app exists for, and no run has ever tested it.
    saw_other_users: bool | None = None

    #: Facets averaged into :attr:`craft`. ``delight`` is included: it is the
    #: overall item of the same scale.
    FACET_FIELDS = ("functionality", "usability", "design", "simplicity", "delight")

    def __post_init__(self) -> None:
        # Clamp defensively: these values come from an LLM tool call.
        self.delight = max(0, min(10, int(self.delight)))
        for name in ("functionality", "usability", "design", "simplicity"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, max(0, min(10, int(value))))

    @property
    def craft(self) -> float | None:
        """Mean of the supplied facet ratings, ``0``..``10`` (None if none given).

        This multi-item mean -- not the single delight score -- is the hands-on
        quality signal the ViralScore reads.
        """
        values = [
            v for v in (getattr(self, f) for f in self.FACET_FIELDS) if v is not None
        ]
        return round(sum(values) / len(values), 2) if values else None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["craft"] = self.craft
        return data


@dataclass
class InteractionTrace:
    """The full record of one trial: every step, plus the final verdict.

    Attributes:
        build_id: The build that was tried.
        app_type: ``single-page-app`` | ``cli`` | ``bot``.
        steps: Ordered actions + observations.
        started_at: Epoch seconds when the trial opened.
        ended_at: Epoch seconds when it closed (``None`` while open).
        degraded: True if the trial ran without full fidelity -- e.g. a web app
            observed over static HTTP because no browser was available, or a
            navigation that never reached the app. A degraded trace is still
            useful but must not be treated as a real hands-on verification.
        app_reachable: Did the agent ever get the app to WORK? ``True`` once a
            page genuinely loaded (web), the app's own entrypoint exited 0 (cli),
            or the bot answered (bot); ``False`` once contact was attempted and
            failed; ``None`` before any attempt. Separate from ``degraded``
            because "I saw the app through a narrow window" and "I never saw the
            app at all" are different claims, and only the second voids a verdict.

            This used to be **web-only**, which left 48% of the bench (9 cli + 3
            bot ideas) with no reachability check at all. Measured consequence on
            `local_document_chat`: every single command the triers ran exited 1
            with a traceback, and 7 of 8 of them still filed craft 8.0-8.8 and
            "I would use and share this", describing features -- "grounded
            citations", "Privacy Audit Cards" -- reconstructed from reading the
            source and narrated as first-hand use. That is docs/crowd_bugs.md
            T0.1 happening again in the half of the corpus that was not watched.
        target_url: For web apps, the URL that was driven.
        verdict: The agent's self-report, if it finished the trial.
    """

    build_id: str
    app_type: str
    steps: list[InteractionStep] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    degraded: bool = False
    app_reachable: bool | None = None
    target_url: str | None = None
    verdict: TrialVerdict | None = None

    # -- building -----------------------------------------------------------

    def record(
        self,
        action: str,
        *,
        args: dict | None = None,
        summary: str = "",
        ok: bool = True,
        errors: tuple[str, ...] = (),
        screenshot: str | None = None,
        duration_s: float = 0.0,
    ) -> InteractionStep:
        """Append a step and return it.

        Convention, so the two channels stop contradicting each other:
        ``ok`` answers "did this action do what it was asked to do", and
        ``errors`` collects everything observed to be wrong -- action failures
        *and* page-level noise like console errors. A step with ``ok=False``
        must therefore always carry at least one entry in ``errors``; callers
        that omit it get a generic one synthesised from ``summary`` rather than
        an empty tuple. Previously every failed step had an empty ``errors``
        while every populated ``errors`` sat on a successful step, so filtering
        on either one found the wrong set.
        """
        errors = tuple(errors)
        if not ok and not errors:
            errors = (summary or f"{action} failed",)
        step = InteractionStep(
            index=len(self.steps),
            action=action,
            args=dict(args or {}),
            summary=summary,
            ok=ok,
            errors=errors,
            screenshot=screenshot,
            duration_s=duration_s,
        )
        self.steps.append(step)
        return step

    def finish(self, verdict: TrialVerdict | None = None) -> None:
        """Mark the trial closed, optionally recording the agent's verdict."""
        if verdict is not None:
            self.verdict = verdict
        self.ended_at = time.time()

    # -- introspection ------------------------------------------------------

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def errors(self) -> list[str]:
        """Every runtime error observed across all steps (flattened)."""
        out: list[str] = []
        for step in self.steps:
            out.extend(step.errors)
        return out

    def had_effect(self) -> bool:
        """True if at least one non-observational action succeeded.

        Distinguishes a real hands-on trial (clicked/typed/ran/chatted) from one
        that only opened and looked -- used to tell "genuinely exercised" from
        "merely loaded", the crowd analogue of the ship-evidence gate.
        """
        passive = {"open", "look", "observe", "screenshot", "finish"}
        return any(s.ok for s in self.steps if s.action not in passive)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "build_id": self.build_id,
            "app_type": self.app_type,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "degraded": self.degraded,
            "app_reachable": self.app_reachable,
            "target_url": self.target_url,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def jsonl_lines(self) -> list[str]:
        """One compact JSON object per step (the append-only action log).

        Each line carries the build id and app type so lines from many trials can
        be concatenated into a single run-level log and still be attributable.
        """
        lines: list[str] = []
        for step in self.steps:
            row = {
                "build_id": self.build_id,
                "app_type": self.app_type,
                **step.to_dict(),
            }
            lines.append(json.dumps(row, separators=(",", ":")))
        return lines

    def render(self, *, max_steps: int = 40, max_summary: int = 500) -> str:
        """A compact, human/LLM-readable transcript of the trial.

        This is what gets fed back to the deciding agent (and printed by the CLI):
        a numbered list of ``action(args) -> summary`` lines, plus a header and a
        trailing error/verdict summary.
        """
        header = f"Trial of {self.build_id} ({self.app_type})"
        if self.target_url:
            header += f" at {self.target_url}"
        if self.degraded:
            header += "  [DEGRADED: no browser; static HTML only]"
        lines = [header]

        shown = self.steps[-max_steps:]
        if len(self.steps) > len(shown):
            lines.append(f"... ({len(self.steps) - len(shown)} earlier steps omitted)")
        for step in shown:
            arg_str = ", ".join(f"{k}={v!r}" for k, v in step.args.items())
            status = "" if step.ok else " [FAILED]"
            line = f"  {step.index}. {step.action}({arg_str}){status}"
            if step.summary:
                line += f" -> {_clip(step.summary, max_summary)}"
            lines.append(line)
            for err in step.errors:
                lines.append(f"       ! {_clip(err, 200)}")

        if self.verdict:
            v = self.verdict
            lines.append(
                f"Verdict: would_use={v.would_use} would_share={v.would_share} "
                f"delight={v.delight}/10 -- {v.notes}"
            )
        return "\n".join(lines)
