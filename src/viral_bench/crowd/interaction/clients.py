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

"""Human-like app clients: one interface for using a running app.

A crowd agent should not care how an app is wired -- it should *use the
thing* and see what happens. Each client here wraps one running app behind a
small, human-shaped API and returns a uniform :class:`Observation` (what the
agent now sees) for every action, while recording the action to the trial's
:class:`~viral_bench.crowd.interaction.trace.InteractionTrace`.

* :class:`WebAppClient` -- drive the app in a real browser (open, look, click,
  type, press, upload, screenshot), the fix for the "a curl 200 is not a test"
  problem: it runs the page's JS and reports the *rendered* DOM.
* :class:`StaticWebAppClient` -- the graceful fallback when no browser is
  available: fetch the HTML over HTTP and report it, clearly flagged as degraded
  (it cannot click or see JS-rendered state).
* :class:`UndeliverableAppClient` -- the founder shipped nothing runnable, so
  every verb reports that there is no app, which is what a real user would find.

Both real app types (``client-app`` and ``full-stack-app``) are web apps and use
the same client: the scope changes what the app must DO, not how it is driven.
"""

from __future__ import annotations

import html
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from viral_bench.crowd.interaction.browser import PageHandle, PageSnapshot
from viral_bench.crowd.interaction.fixtures import FIXTURE_DESCRIPTIONS, fixture_path
from viral_bench.crowd.interaction.trace import InteractionTrace

# How much of the app an agent is allowed to see.
#
# These were far too tight. The rendered-text clip alone truncated 100% of
# snapshots of the working app (688 of 688), so the crowd was judging a product
# through an 800-character keyhole and then rating its "design" and
# "simplicity". Starving the observation does not make the measurement cheaper,
# it makes it uninformative -- and uninformative measurements are exactly what a
# benchmark that must separate two good models cannot afford.
_MAX_OBSERVATION = 24000
#: Visible page text shown to the agent per snapshot.
_MAX_PAGE_TEXT = 6000
#: Accessibility tree depth shown to the agent.
_MAX_ARIA = 6000
#: Interactive controls enumerated per snapshot (82% of snapshots hit 20).
_MAX_CONTROLS = 40
#: Console/page errors surfaced per snapshot.
_MAX_CONSOLE_ERRORS = 12
#: Captured stdout for a CLI trial (33% of runs hit 1500).
#: Captured reply for a bot trial (23% of runs hit 1500).


def _clip(text: str, limit: int) -> str:
    """Truncate to ``limit``, saying so when it happens.

    Silent truncation is worse than a short observation: the agent cannot tell it
    is looking at a fragment, so it confidently rates a page it only partly saw.
    Every clip in the observation path is marked.
    """
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated {len(text) - limit} more characters]"


def live_feature_notice() -> str:
    """Tell the trier to exercise the app's real (credentialed) feature.

    Apps that need an API key are given one by the runtime (see
    :mod:`viral_bench.founder.appenv`), but they usually also ship an offline
    demo path -- ``--mock``, a sample dataset, a canned reply -- and their manifest
    often advertises *that* as the easy way to try it. A trier who only runs the
    mock never exercises the app's headline feature, so a build whose real feature
    is broken scores the same as one that works. Surfacing the injected
    credentials in the usage text keeps the crowd honest.

    Returns an empty string when the runtime injected no credentials (then the
    offline path is the only thing to judge).
    """
    from viral_bench.founder.appenv import resolve_app_env

    names = sorted(resolve_app_env())
    if not names:
        return ""
    joined = ", ".join(names)
    return (
        f"NOTE: this environment provides {joined}, so the app's real "
        "AI/network-backed features work here. Test the REAL feature, not just an "
        "offline/mock/demo mode -- if it also offers a mock, that is a fallback, "
        "not the thing to judge. Judge the live output you actually get back."
    )


@dataclass
class Observation:
    """What an agent perceives after an action, uniform across app types.

    ``summary`` is the LLM-facing rendering (what a tool returns to the agent).
    The structured fields are kept for scoring and the trace.
    """

    ok: bool
    app_type: str
    summary: str
    title: str | None = None
    url: str | None = None
    text: str = ""
    elements: list[str] = field(default_factory=list)
    console_errors: tuple[str, ...] = ()
    screenshot: str | None = None
    degraded: bool = False
    # Backend extras (exit code, argv, reply latency, ...), JSON-able.
    raw: dict = field(default_factory=dict)

    def render(self) -> str:
        return self.summary


@runtime_checkable
class AppClient(Protocol):
    """Common surface every app client exposes (used for cleanup + typing)."""

    app_type: str
    trace: InteractionTrace

    async def observe(self) -> Observation: ...

    async def close(self) -> None: ...


# ----------------------------------------------------------------------------
# Undeliverable (the founder shipped no way to launch the app)
# ----------------------------------------------------------------------------


#: What every action on an undeliverable app reports back. Stated as an
#: observation about the delivery, not a judgement about the code, because the
#: agent can still go and read the source -- and should be free to conclude
#: whatever it likes about what it finds there.
UNDELIVERABLE_NOTICE = (
    "This app cannot be launched. The founder shipped no valid "
    "`viralbench.json`, which is the file that says how to install, start and "
    "test the app -- so there is no documented way to run it, and nothing here "
    "to click, type into or execute. You can still read the source if you want "
    "to form a view, but you have not used this app and cannot report that you "
    "did."
)


class UndeliverableAppClient:
    """The client for a build that has no runnable contract.

    A build whose ``viralbench.json`` is missing or malformed used to be dropped
    from the crowd stage entirely -- it was never simulated, so it never scored,
    so a model that failed to ship a launch contract vanished from the
    denominator instead of being marked down. Seven of one model's 25 builds
    disappeared that way while the other model's *bad but runnable* apps stayed
    in and dragged its average down, which biased the comparison toward the model
    that delivered less.

    So the app is presented to the crowd anyway, and the crowd discovers what a
    real user would discover: there is no way to start it. Every action returns
    the same notice, ``app_reachable`` is False from the outset (so no trial here
    can ever count as hands-on craft evidence), and the score falls out of the
    same machinery that already handles apps which build but do not run -- no
    imputed default, no special case in the scorer.
    """

    def __init__(self, build_id: str, app_type: str, trace: InteractionTrace) -> None:
        self.build_id = build_id
        self.app_type = app_type
        self.trace = trace
        # Not "this could not be established to work" -- it has been
        # established that the app cannot be started at all. Set before any
        # action, so even a trial that calls nothing is correctly marked.
        self.trace.app_reachable = False

    def _report(self, action: str, args: dict | None = None) -> Observation:
        self.trace.record(
            action, args=args or {}, summary=UNDELIVERABLE_NOTICE, ok=False
        )
        return Observation(
            ok=False,
            app_type=self.app_type,
            summary=UNDELIVERABLE_NOTICE,
            degraded=True,
        )

    async def observe(self) -> Observation:
        return self._report("look")

    async def open(self) -> Observation:
        return self._report("open")

    async def reload(self) -> Observation:
        return self._report("reload")

    async def start(self) -> Observation:
        return self._report("start")

    async def click(self, target: str) -> Observation:
        return self._report("click", {"target": target})

    async def select_option(self, target: str, value: str) -> Observation:
        return self._report("select", {"target": target, "value": value})

    async def type_text(self, target: str, text: str) -> Observation:
        return self._report("type", {"target": target, "text": text})

    async def press_key(self, key: str, *, target: str | None = None) -> Observation:
        return self._report("press", {"key": key, "target": target})

    async def screenshot(self, note: str = "") -> Observation:
        return self._report("screenshot", {"note": note})

    async def close(self) -> None:
        return None


# ----------------------------------------------------------------------------
# Web (single-page app)
# ----------------------------------------------------------------------------


#: Text the browser itself renders when a navigation failed. If any of these is
#: on screen, the agent is looking at the browser's apology page, not the app.
_BROWSER_ERROR_TEXT = (
    "this site can't be reached",
    "this site can\u2019t be reached",
    "the connection was reset",
    "err_connection_refused",
    "err_connection_reset",
    "err_empty_response",
    "err_socket_not_connected",
    "err_address_unreachable",
    "unable to connect",
)


def _unreachable_reason(snap: PageSnapshot, expected_url: str | None) -> str | None:
    """Why this snapshot is an error page rather than the app, or ``None``.

    A failed ``goto`` raises and is handled directly, but a *subsequent* ``look``
    or ``click`` snapshots whatever the browser is showing -- which, after
    a failed navigation, is Chrome's "This site can't be reached" interstitial.
    That snapshot used to be recorded as a successful observation, so a trier
    could open nothing, look at nothing, and still file a four-facet craft
    verdict. 36 of 314 stored trials did exactly that.
    """
    url = (snap.url or "").strip()
    if url.startswith(("chrome-error://", "edge-error://")):
        return f"browser error page ({url})"
    if expected_url and url in {"about:blank", ""}:
        return f"navigation never landed (url={url or 'empty'})"
    lowered = (snap.text or "").lower()
    for marker in _BROWSER_ERROR_TEXT:
        if marker in lowered:
            return f"browser error page (text: {marker!r})"
    for err in snap.console_errors:
        low = err.lower()
        # ONLY the document navigation. A failed sub-resource means some part of
        # the page did not load, which is a fact about the app worth reporting
        # (it is still in ``console_errors``, which the agent reads) -- it is not
        # evidence that the agent never reached the product. Treating the two as
        # the same voided 16% of the first 660 trials of the v10 sweep, every one
        # of them for a live-sync stream, a web font with no network, a
        # cancelled XHR or a revoked blob URL.
        if low.startswith("navigation-failed") and "net::err" in low:
            return f"the page itself failed to load ({err})"
    return None


def _web_summary(snap: PageSnapshot, *, note: str = "") -> str:
    """Render a page snapshot the way an agent would describe what it sees."""
    lines: list[str] = []
    if note:
        lines.append(note)
    lines.append(f"Title: {snap.title or '(none)'}  |  URL: {snap.url}")
    if snap.text.strip():
        lines.append(f"Visible text: {_clip(snap.text.strip(), _MAX_PAGE_TEXT)}")
    if snap.elements:
        controls = ", ".join(e.describe() for e in snap.elements[:_MAX_CONTROLS])
        lines.append(f"Controls you can use: {controls}")
    if snap.aria.strip():
        lines.append(f"Accessibility tree:\n{_clip(snap.aria.strip(), _MAX_ARIA)}")
    if snap.console_errors:
        lines.append(
            "Console/page errors: "
            + " | ".join(snap.console_errors[:_MAX_CONSOLE_ERRORS])
        )
    return "\n".join(lines)[:_MAX_OBSERVATION]


class WebAppClient:
    """Drive a single-page app in a real headless browser."""

    app_type = "client-app"

    def __init__(
        self,
        url: str,
        page: PageHandle,
        trace: InteractionTrace,
        *,
        engine=None,
        owns_engine: bool = False,
        session=None,
        env_notice: bool = True,
    ) -> None:
        self.url = url
        self.page = page
        self.trace = trace
        #: Whether to tell the agent which API keys the environment provides.
        #: An ablation lever: the notice fires on every app, including the many
        #: that have no AI feature to test.
        self.env_notice = env_notice
        self._engine = engine
        self._owns_engine = owns_engine
        # A standalone web trial (no shared AppHost) owns the app server session
        # it started and must tear it down. A shared instance is left running.
        self._session = session

    async def _snapshot_obs(
        self, action: str, args: dict, *, note: str = "", screenshot: str | None = None
    ) -> Observation:
        started = time.monotonic()
        snap = await self.page.snapshot()

        # Looking at the browser's error page is not observing the app. Without
        # this the interstitial is recorded ok=True and every downstream check
        # believes the trial saw the product.
        unreachable = _unreachable_reason(snap, self.url)
        if unreachable is not None:
            self.trace.app_reachable = False
            self.trace.degraded = True
            summary = (
                f"{action} did not reach the app: {unreachable}. "
                "You are looking at the browser's error page, not the product -- "
                "do not describe or rate it as if it were the app."
            )
            self.trace.record(
                action,
                args=args,
                summary=summary,
                ok=False,
                errors=(unreachable, *snap.console_errors),
                duration_s=time.monotonic() - started,
            )
            return Observation(
                ok=False,
                app_type=self.app_type,
                summary=summary,
                url=snap.url,
                console_errors=snap.console_errors,
                degraded=True,
            )

        self.trace.app_reachable = True
        obs = Observation(
            ok=True,
            app_type=self.app_type,
            summary=_web_summary(snap, note=note),
            title=snap.title,
            url=snap.url,
            text=snap.text,
            elements=[e.describe() for e in snap.elements],
            console_errors=snap.console_errors,
            screenshot=screenshot,
        )
        self.trace.record(
            action,
            args=args,
            summary=obs.summary,
            ok=True,
            errors=snap.console_errors,
            screenshot=screenshot,
            duration_s=time.monotonic() - started,
        )
        return obs

    def _error_obs(self, action: str, args: dict, exc: Exception) -> Observation:
        summary = f"{action} failed: {exc}"
        self.trace.record(
            action, args=args, summary=summary, ok=False, errors=(str(exc),)
        )
        return Observation(
            ok=False, app_type=self.app_type, summary=summary, url=self.url
        )

    async def open(self) -> Observation:
        try:
            await self.page.goto(self.url)
        except Exception as exc:  # noqa: BLE001
            # The app was never reached. Mark the whole trial, not merely this
            # step: anything the agent says afterwards is about the source tree
            # or its imagination, never about a running product.
            self.trace.app_reachable = False
            self.trace.degraded = True
            return self._error_obs("open", {"url": self.url}, exc)
        note = f"Opened {self.url}"
        notice = live_feature_notice() if self.env_notice else ""
        if notice:
            note = f"{note}\n{notice}"
        return await self._snapshot_obs("open", {"url": self.url}, note=note)

    async def observe(self) -> Observation:
        return await self._snapshot_obs("look", {})

    async def reload(self) -> Observation:
        """Re-navigate to the app, the way a user hitting refresh would.

        This is the only way to ask the question a full-stack app exists to
        answer: **did my work survive?** An app that keeps everything in a
        JavaScript variable and one that persists to a database look identical
        until the page reloads, and no agent in any run has ever been able to
        tell them apart, because there was no verb for it. 8 of the 25 briefs
        here are explicitly full-stack, so a third of the fleet was judged on a
        property nobody could observe.
        """
        try:
            await self.page.goto(self.url)
        except Exception as exc:  # noqa: BLE001
            return self._error_obs("reload", {"url": self.url}, exc)
        return await self._snapshot_obs(
            "reload",
            {"url": self.url},
            note=(
                "Reloaded the page. Whatever you created that is still here was "
                "really saved; whatever is missing was only ever in the "
                "browser's memory."
            ),
        )

    async def click(self, target: str) -> Observation:
        try:
            await self.page.click(target)
        except Exception as exc:  # noqa: BLE001
            return self._error_obs("click", {"target": target}, exc)
        return await self._snapshot_obs(
            "click", {"target": target}, note=f"Clicked {target!r}."
        )

    async def type_text(self, target: str, text: str) -> Observation:
        try:
            await self.page.fill(target, text)
        except Exception as exc:  # noqa: BLE001
            return self._error_obs("type", {"target": target, "text": text}, exc)
        return await self._snapshot_obs(
            "type",
            {"target": target, "text": text},
            note=f"Typed {text!r} into {target!r}.",
        )

    async def press_key(self, key: str, *, target: str | None = None) -> Observation:
        try:
            await self.page.press(key, target=target)
        except Exception as exc:  # noqa: BLE001
            return self._error_obs("press", {"key": key, "target": target}, exc)
        return await self._snapshot_obs(
            "press", {"key": key, "target": target}, note=f"Pressed {key!r}."
        )

    async def select_option(self, target: str, value: str) -> Observation:
        try:
            await self.page.select_option(target, value)
        except Exception as exc:  # noqa: BLE001
            return self._error_obs("select", {"target": target, "value": value}, exc)
        return await self._snapshot_obs(
            "select",
            {"target": target, "value": value},
            note=f"Selected {value!r} in {target!r}.",
        )

    async def upload_file(self, fixture: str, target: str | None = None) -> Observation:
        """Attach one of the bundled sample files to a file input."""
        try:
            path = str(fixture_path(fixture))
        except KeyError as exc:
            return self._error_obs("upload", {"fixture": fixture}, exc)
        try:
            await self.page.set_files(target, [path])
        except Exception as exc:  # noqa: BLE001
            return self._error_obs(
                "upload", {"fixture": fixture, "target": target}, exc
            )
        return await self._snapshot_obs(
            "upload",
            {"fixture": fixture, "target": target},
            note=f"Uploaded {fixture} ({FIXTURE_DESCRIPTIONS[fixture]}).",
        )

    async def screenshot(self, note: str = "") -> Observation:
        try:
            path = await self.page.screenshot(label=note or "shot")
        except Exception as exc:  # noqa: BLE001
            return self._error_obs("screenshot", {"note": note}, exc)
        return await self._snapshot_obs(
            "screenshot",
            {"note": note},
            note=f"Saved screenshot to {path}",
            screenshot=path,
        )

    async def close(self) -> None:
        try:
            await self.page.close()
        finally:
            if self._owns_engine and self._engine is not None:
                await self._engine.close()
            if self._session is not None:
                try:
                    self._session.close()
                except Exception:  # noqa: BLE001
                    pass


class StaticWebAppClient:
    """Degraded web client: HTTP-only, no JS/DOM. The graceful fallback.

    Used when no browser is available. It can fetch and report the served HTML
    but cannot execute JavaScript or interact, so every trial is flagged
    ``degraded`` -- honest about the fact that it did not truly exercise the UI.
    """

    app_type = "client-app"

    def __init__(
        self,
        url: str,
        trace: InteractionTrace,
        *,
        timeout: float = 10.0,
        session=None,
    ) -> None:
        self.url = url
        self.trace = trace
        self._timeout = timeout
        self._session = session
        self.trace.degraded = True

    def _fetch(self) -> tuple[bool, str, str]:
        try:
            with urllib.request.urlopen(self.url, timeout=self._timeout) as resp:
                body = resp.read(_MAX_OBSERVATION * 2).decode("utf-8", "replace")
                return True, str(resp.status), body
        except urllib.error.HTTPError as exc:
            return True, str(exc.code), ""
        except Exception as exc:  # noqa: BLE001
            return False, "0", str(exc)

    @staticmethod
    def _extract_title(body: str) -> str | None:
        m = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
        return html.unescape(m.group(1).strip()) if m else None

    @staticmethod
    def _to_text(body: str) -> str:
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", body)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        return html.unescape(re.sub(r"\s+", " ", text)).strip()

    async def _observe(self, action: str, args: dict, *, note: str = "") -> Observation:
        ok, status, body = self._fetch()
        title = self._extract_title(body) if ok else None
        text = self._to_text(body) if ok else ""
        summary_parts = [note] if note else []
        summary_parts.append(
            f"[degraded/no-browser] GET {self.url} -> HTTP {status}. "
            "JavaScript did NOT run, so this is the raw served HTML, not the "
            "rendered UI a user sees."
        )
        if title:
            summary_parts.append(f"Title: {title}")
        if text:
            summary_parts.append(f"HTML text: {_clip(text, _MAX_PAGE_TEXT)}")
        summary = "\n".join(summary_parts)[:_MAX_OBSERVATION]
        self.trace.record(action, args=args, summary=summary, ok=ok)
        return Observation(
            ok=ok,
            app_type=self.app_type,
            summary=summary,
            title=title,
            url=self.url,
            text=text,
            degraded=True,
            raw={"http_status": status},
        )

    async def open(self) -> Observation:
        return await self._observe("open", {"url": self.url}, note=f"Opened {self.url}")

    async def observe(self) -> Observation:
        return await self._observe("look", {})

    async def _cannot(self, action: str, args: dict) -> Observation:
        summary = (
            f"Cannot {action} without a browser (this trial is running in "
            "degraded static-HTTP mode). Install Playwright + a browser to "
            "interact with the UI."
        )
        self.trace.record(action, args=args, summary=summary, ok=False)
        return Observation(
            ok=False,
            app_type=self.app_type,
            summary=summary,
            url=self.url,
            degraded=True,
        )

    async def click(self, target: str) -> Observation:
        return await self._cannot("click", {"target": target})

    async def type_text(self, target: str, text: str) -> Observation:
        return await self._cannot("type", {"target": target, "text": text})

    async def press_key(self, key: str, *, target: str | None = None) -> Observation:
        return await self._cannot("press", {"key": key, "target": target})

    async def screenshot(self, note: str = "") -> Observation:
        return await self._cannot("screenshot", {"note": note})

    async def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                pass


# ----------------------------------------------------------------------------
# CLI
