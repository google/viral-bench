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

"""Drive a single-page app in a real (headless) browser, the Python way.

The founder's QA gets a browser through a vendored Node Playwright *MCP* wired to
its opencode agent. The crowd's agents are CAMEL/Python agents, so they need the
*same capability* as plain Python: this module is that -- a thin async wrapper
over `Playwright for Python <https://playwright.dev/python/>`_ that launches a
headless browser, opens one isolated page per trial, and exposes the handful of
human verbs an agent needs (navigate, snapshot the accessible DOM, click, type,
press keys, screenshot) while capturing console/page/network errors.

Two deliberate choices mirror the founder side so the whole repo drives apps the
same way:

* **system Chrome, not a bundled download.** The launch uses
  ``channel="chrome"`` against the host's Chrome/Chromium (falling back to
  Playwright's bundled Chromium only if that fails), so there is no per-machine
  browser-revision download to manage -- exactly the founder MCP's posture.
* **graceful absence.** :func:`browser_available` lets callers detect up front
  whether a real browser can run. When it cannot, the trial degrades to a static
  HTTP observation (see :mod:`~viral_bench.crowd.interaction.session`) instead of
  failing the simulation.

Everything here is ``async`` because the OASIS crowd loop is async and Playwright's
sync API cannot run inside a running event loop. Non-async callers (the CLI, the
tests) drive it with ``asyncio.run``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from viral_bench.founder.workspace import builds_root

# System Chrome/Chromium binaries the ``chrome`` channel can drive. Mirrors the
# list the founder browser MCP uses (viral_bench.founder.opencode_agents) so both
# sides agree on "is there a browser here".
_CHROME_BINARIES: tuple[str, ...] = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
    "microsoft-edge",
)

# Interactive elements surfaced in the structured digest so an agent can refer
# to controls the way a person would ("the Roll button", "the name field").
_INTERACTIVE_JS = r"""
() => {
  const sel = 'a[href],button,input,select,textarea,summary,'
    + '[role=button],[role=link],[role=tab],[role=checkbox],[role=radio],'
    + '[role=menuitem],[onclick],[contenteditable=""],[contenteditable=true]';
  const seen = [];
  for (const e of document.querySelectorAll(sel)) {
    const r = e.getBoundingClientRect();
    const style = window.getComputedStyle(e);
    const visible = r.width > 0 && r.height > 0
      && style.visibility !== 'hidden' && style.display !== 'none';
    if (!visible) continue;
    const label = (
      e.getAttribute('aria-label') || e.innerText || e.value
      || e.getAttribute('placeholder') || e.getAttribute('title')
      || e.getAttribute('name') || ''
    ).trim().replace(/\s+/g, ' ').slice(0, 120);
    seen.push({
      tag: e.tagName.toLowerCase(),
      type: e.getAttribute('type') || '',
      role: e.getAttribute('role') || '',
      label: label,
    });
    if (seen.length >= 120) break;
  }
  return seen;
}
"""


def playwright_installed() -> bool:
    """True if the ``playwright`` Python package is importable."""
    return importlib.util.find_spec("playwright") is not None


def system_chrome() -> str | None:
    """Return the first system Chrome/Chromium binary on PATH, if any."""
    for exe in _CHROME_BINARIES:
        if shutil.which(exe):
            return exe
    return None


def browser_available() -> bool:
    """True if a real browser can be driven on this host.

    Requires the ``playwright`` package *and* a system Chrome/Chromium for the
    ``chrome`` channel. This is the reliable, no-download path, and callers use
    it to decide between a real browser trial and the static-HTTP fallback.
    """
    return playwright_installed() and system_chrome() is not None


def _browser_default(key: str, fallback):
    """Read one ``browser:`` value from crowd.yaml, falling back to the code."""
    try:
        from viral_bench import config as _config

        return _config.crowd_browser(key, fallback)
    except Exception:  # noqa: BLE001 - config must never break the browser
        return fallback


@dataclass
class BrowserConfig:
    """How the crowd's browser launches and behaves.

    Defaults come from ``config/crowd.yaml`` ``browser:``. That block used to be
    documentation only -- it described the code defaults without setting them.
    """

    headless: bool = field(
        default_factory=lambda: bool(_browser_default("headless", True))
    )
    # Prefer the system Chrome channel (no bundled-browser download). ``None``
    # uses Playwright's bundled Chromium.
    channel: str | None = field(
        default_factory=lambda: _browser_default("channel", "chrome")
    )
    nav_timeout_ms: int = field(
        default_factory=lambda: int(_browser_default("nav_timeout_ms", 30000))
    )
    # 10s, not 5s: 90 of 147 failed steps in the stored corpus were
    # `Locator.click: Timeout 5000ms`, and a single-page app can still be
    # hydrating right after it starts answering HTTP.
    action_timeout_ms: int = field(
        default_factory=lambda: int(_browser_default("action_timeout_ms", 10000))
    )
    viewport_width: int = 1280
    viewport_height: int = 800
    # Where screenshots are written. ``builds/screenshots`` is used when unset.
    screenshot_dir: Path | None = None


@dataclass
class ElementInfo:
    """One visible, interactive control, as an agent would describe it."""

    tag: str
    label: str
    role: str = ""
    type: str = ""

    def describe(self) -> str:
        kind = self.role or (f"{self.tag}:{self.type}" if self.type else self.tag)
        return f'{kind} "{self.label}"' if self.label else kind


@dataclass
class PageSnapshot:
    """A point-in-time view of the page, roughly what a user perceives."""

    url: str
    title: str
    text: str
    aria: str
    elements: list[ElementInfo] = field(default_factory=list)
    console_errors: tuple[str, ...] = ()


class BrowserError(RuntimeError):
    """Raised when the browser cannot launch or an interaction cannot proceed."""


class BrowserEngine:
    """Owns one Playwright browser process and hands out isolated pages.

    A crowd shares a single engine (one browser) across many agents, giving each
    trial its own :class:`PageHandle` (a fresh browser *context*, so cookies and
    storage never leak between agents). Cheap to fan out, because the expensive
    browser process is started once.
    """

    def __init__(self, config: BrowserConfig | None = None) -> None:
        self.config = config or BrowserConfig()
        self._pw = None
        self._browser = None

    async def start(self) -> None:
        if self._browser is not None:
            return
        if not playwright_installed():
            raise BrowserError("playwright is not installed")
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        cfg = self.config
        try:
            if cfg.channel:
                self._browser = await self._pw.chromium.launch(
                    headless=cfg.headless, channel=cfg.channel
                )
            else:
                self._browser = await self._pw.chromium.launch(headless=cfg.headless)
        except Exception as exc:  # noqa: BLE001 - fall back to bundled chromium
            try:
                self._browser = await self._pw.chromium.launch(headless=cfg.headless)
            except Exception as exc2:  # noqa: BLE001
                await self._stop_pw()
                raise BrowserError(
                    f"could not launch a browser (channel={cfg.channel!r}: {exc}; "
                    f"bundled: {exc2})"
                ) from exc2

    async def open_page(self, *, permissions: list[str] | None = None) -> PageHandle:
        """Open a fresh, isolated page (its own context).

        ``permissions`` is granted for the page's own origin. The crowd passes
        nothing. The rubric grader asks for clipboard access, because reading the
        clipboard is the only way to tell a real copy-to-clipboard from a button
        that merely shows a "Copied!" toast.
        """
        await self.start()
        cfg = self.config
        context = await self._browser.new_context(
            viewport={"width": cfg.viewport_width, "height": cfg.viewport_height},
            ignore_https_errors=True,
            accept_downloads=True,
        )
        if permissions:
            try:
                await context.grant_permissions(permissions)
            except Exception:  # noqa: BLE001 - an unsupported permission must not
                pass  # sink the whole page. The check that needs it fails loudly
        context.set_default_timeout(cfg.action_timeout_ms)
        context.set_default_navigation_timeout(cfg.nav_timeout_ms)
        # A missing favicon is a near-universal, benign 404 that would otherwise
        # show up as a console error and unfairly count against every app. Serve
        # an empty one so only real app errors are captured.
        await context.route(
            "**/favicon.ico",
            lambda route: route.fulfill(
                status=200, body=b"", content_type="image/x-icon"
            ),
        )
        page = await context.new_page()
        handle = PageHandle(context, page, cfg)
        handle._wire()
        return handle

    async def _stop_pw(self) -> None:
        if self._pw is not None:
            try:
                await self._pw.stop()
            finally:
                self._pw = None

    async def close(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            finally:
                self._browser = None
        await self._stop_pw()

    async def __aenter__(self) -> BrowserEngine:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()


#: Roles tried when resolving a bare name to a control.
_ROLES = (
    "button",
    "link",
    "textbox",
    "checkbox",
    "radio",
    "combobox",
    "tab",
    "menuitem",
    "option",
    "switch",
    "slider",
    "heading",
)

#: The form ElementInfo.describe() renders, e.g. `button "Save note"` or
#: `input:text "Email"`. The agent sees this and copies it back as a target, so
#: the resolver has to understand it.
_DESCRIBED = re.compile(r'^([A-Za-z][\w:-]*)\s+"(.*)"$', re.S)


def _looks_like_css(target: str) -> bool:
    """True if this is plausibly a CSS selector rather than a human label.

    Deliberately strict. The old test was `starts with # . [ or contains >`,
    which classified ordinary accessibility labels -- "Next >", ".env settings"
    -- as CSS and sent them to a parser that rejected them. A parse error tells
    the agent nothing about the page, but "not found" at least tells it to look.
    """
    t = target.strip()
    if not t or '"' in t or "'" in t:
        return False
    if t.startswith(("#", ".", "[")) and " " not in t:
        return True
    return bool(re.fullmatch(r"[a-zA-Z][\w-]*(\[[^\]]+\]|[.#][\w-]+)+", t))


class PageHandle:
    """One isolated page an agent drives: the low-level browser verbs.

    Higher layers (``WebAppClient``) add trace recording and the
    :class:`~viral_bench.crowd.interaction.clients.Observation` shaping. This
    class talks to Playwright and returns plain data.
    """

    def __init__(self, context, page, config: BrowserConfig) -> None:
        self._context = context
        self.page = page
        self.config = config
        self._console: list[str] = []
        self._drained = 0
        #: Every request the page issued, in order. Recorded unconditionally:
        #: it costs a dict per request, and the rubric's offline/privacy checks
        #: cannot be reconstructed after the fact -- an app that phoned home
        #: leaves no other trace once the page is closed.
        self._requests: list[dict] = []
        #: Downloads seen while :meth:`capture_downloads` is active.
        self._downloads: list[dict] = []
        self._download_dir: Path | None = None

    # -- event wiring -------------------------------------------------------

    def _wire(self) -> None:
        def on_console(msg) -> None:
            if msg.type in ("error", "warning"):
                self._console.append(f"console.{msg.type}: {msg.text}"[:500])

        def on_pageerror(err) -> None:
            self._console.append(f"pageerror: {err}"[:500])

        def on_requestfailed(req) -> None:
            failure = getattr(req, "failure", None)
            # Label the DOCUMENT navigation separately from every other request.
            #
            # Downstream, ANY "requestfailed ... net::ERR" was read as "the main
            # request failed", which marked the whole trial degraded and struck
            # its craft rating out of the score. Measured over the first 660
            # trials of this sweep that voided **106 of them (16%)**, and not one
            # was the app failing to load. They were: a Server-Sent Events
            # stream aborted by navigating away (a collaborative app punished
            # for having live sync), a Google Fonts request that cannot resolve
            # because the container has no external network, an in-flight XHR
            # cancelled by a reload, and revoked blob: URLs from a download.
            # Every one of those is a marker of a MORE capable app, so the bias
            # ran against exactly the builds the bench should reward:
            # image_compressor lost 30 of 30 trials, collaborative_table 23/60.
            #
            # net::ERR_ABORTED is doubly not an error: it is the normal outcome
            # for any request still open when the page navigates.
            try:
                is_doc = bool(req.is_navigation_request()) and req.resource_type in (
                    "document",
                    "",
                )
            except Exception:  # noqa: BLE001 - a stale request object must not throw
                is_doc = False
            kind = "navigation-failed" if is_doc else "subresource-failed"
            self._console.append(f"{kind}: {req.url} ({failure})"[:500])

        def on_request(req) -> None:
            try:
                self._requests.append(
                    {
                        "url": req.url,
                        "method": req.method,
                        "resource_type": req.resource_type,
                        "body_size": len(req.post_data_buffer or b""),
                    }
                )
            except Exception:  # noqa: BLE001 - a stale request must not throw
                pass

        async def on_download(download) -> None:
            # Playwright deletes a download when its page closes, so it has to
            # be persisted the moment it arrives rather than at read time. The
            # listener is async because ``Download.save_as`` is a coroutine in
            # the async API -- calling it synchronously saves nothing at all,
            # silently, which would turn "the export works" into a false pass.
            if self._download_dir is None:
                return
            try:
                name = (
                    Path(download.suggested_filename or "download").name or "download"
                )
                target = self._download_dir / f"{len(self._downloads):02d}-{name}"
                await download.save_as(str(target))
                self._downloads.append(
                    {
                        "name": name,
                        "path": str(target),
                        "size": target.stat().st_size if target.is_file() else 0,
                        "url": download.url,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self._downloads.append(
                    {"name": "", "path": "", "size": 0, "error": str(exc)}
                )

        self.page.on("console", on_console)
        self.page.on("pageerror", on_pageerror)
        self.page.on("requestfailed", on_requestfailed)
        self.page.on("request", on_request)
        self.page.on("download", on_download)

    def drain_console(self) -> list[str]:
        """Return console/page/network errors seen since the last drain."""
        new = self._console[self._drained :]
        self._drained = len(self._console)
        return list(new)

    # -- evidence capture (used by the rubric grader) -----------------------

    def requests(self, *, since: int = 0) -> list[dict]:
        """Requests issued since index ``since``.

        Pass ``len(page.requests())`` before an action to mark a phase
        boundary, then read back only what that action caused.
        """
        return list(self._requests[since:])

    @contextlib.contextmanager
    def capture_downloads(self, directory: Path):
        """Persist any download that fires inside the block.

        Playwright hands a download over as a temporary file that vanishes with
        the page, so a check like "export produced a real PNG" has to save the
        bytes as they arrive. Yields the list that fills up.
        """
        directory.mkdir(parents=True, exist_ok=True)
        previous_dir, previous = self._download_dir, self._downloads
        self._download_dir, self._downloads = directory, []
        try:
            yield self._downloads
        finally:
            self._download_dir = previous_dir
            captured, self._downloads = self._downloads, previous
            self._downloads.extend(captured)

    async def evaluate(self, expression: str):
        """Evaluate JavaScript in the page and return the result.

        The backing store for the ``value_equals`` / ``dom_count`` family of
        checks. The *comparison* stays in Python: this only fetches.
        """
        try:
            return await self.page.evaluate(expression)
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"could not evaluate: {exc}") from exc

    async def read_clipboard(self) -> str:
        """Read the clipboard, the only way to tell a real copy from a toast.

        Requires the ``clipboard-read`` permission, which the grader's context
        grants. A crowd context does not, and gets an empty string.
        """
        try:
            return (
                await self.page.evaluate("() => navigator.clipboard.readText()") or ""
            )
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"could not read the clipboard: {exc}") from exc

    # -- navigation & observation ------------------------------------------

    async def goto(self, url: str) -> None:
        try:
            await self.page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"could not open {url}: {exc}") from exc

    async def snapshot(
        self, *, max_text: int = 12000, max_aria: int = 12000
    ) -> PageSnapshot:
        """Capture what a user would perceive: title, text, controls, errors."""
        url = self.page.url
        try:
            title = await self.page.title()
        except Exception:  # noqa: BLE001
            title = ""
        try:
            text = (await self.page.inner_text("body"))[:max_text]
        except Exception:  # noqa: BLE001
            text = ""
        try:
            aria = (await self.page.locator("body").aria_snapshot())[:max_aria]
        except Exception:  # noqa: BLE001
            aria = ""
        elements: list[ElementInfo] = []
        try:
            for raw in await self.page.evaluate(_INTERACTIVE_JS):
                elements.append(
                    ElementInfo(
                        tag=raw.get("tag", ""),
                        label=raw.get("label", ""),
                        role=raw.get("role", ""),
                        type=raw.get("type", ""),
                    )
                )
        except Exception:  # noqa: BLE001
            pass
        return PageSnapshot(
            url=url,
            title=title,
            text=text,
            aria=aria,
            elements=elements,
            console_errors=tuple(self.drain_console()),
        )

    # -- interaction --------------------------------------------------------

    async def _locator(self, target: str):
        """Resolve a human target string to a Playwright locator.

        Tries, in order: an explicit Playwright engine prefix (``role=``/``text=``
        /``css=``/``xpath=``), a raw CSS selector, then human strategies (by role
        name, visible text, placeholder, label). Returns the first locator that
        matches at least one element, and falls back to a text locator so the
        eventual action raises an informative "not found" rather than a guess.
        """
        page = self.page
        t = target.strip()
        if t.startswith(("role=", "text=", "css=", "xpath=", "id=")):
            return page.locator(t).first

        candidates = []
        # `button "Save note"` -- exactly the form describe() shows the agent.
        m = _DESCRIBED.match(t)
        name = m.group(2) if m else t
        if m:
            role = m.group(1).split(":", 1)[0]
            candidates.append(page.get_by_role(role, name=name, exact=False))
        elif _looks_like_css(t):
            candidates.append(page.locator(t))
        for role in _ROLES:
            candidates.append(page.get_by_role(role, name=name, exact=False))
        candidates.append(page.get_by_text(name, exact=False))
        candidates.append(page.get_by_placeholder(name))
        candidates.append(page.get_by_label(name))
        candidates.append(page.get_by_title(name))

        # Prefer a VISIBLE match over merely a present one.
        #
        # Matching on count() alone is what made clicking unreliable: 109 of 136
        # click timeouts across the solo fleet were "element is not visible".
        # A modern app keeps its whole auth modal in the DOM and toggles it, so
        # `Sign In` resolves to the hidden copy inside the closed modal, and
        # Playwright then blocks the full 10s actionability timeout waiting for
        # something that will never appear -- while the visible header button
        # with the same name sits one candidate further down the list.
        #
        # Two passes rather than one filtered pass: a first-choice strategy that
        # matched only hidden elements should lose to a later strategy that
        # matched a visible one, but if NOTHING is visible the original
        # behaviour still applies (act on the hidden node and let the action report a
        # real, informative failure).
        fallback = None
        for loc in candidates:
            try:
                if await loc.count() == 0:
                    continue
            except Exception:  # noqa: BLE001 - a bad strategy is not fatal
                continue
            if fallback is None:
                fallback = loc.first
            try:
                visible = loc.locator("visible=true")
                if await visible.count() > 0:
                    return visible.first
            except Exception:  # noqa: BLE001 - some locators reject chaining
                continue
        if fallback is not None:
            return fallback
        # Nothing matched. Return a text locator so the action fails with
        # "element not found", which the agent can react to, rather than a
        # selector parse error, which tells it nothing about the page.
        return page.get_by_text(name, exact=False).first

    async def click(self, target: str) -> None:
        """Click a control, the way a person would.

        Falls back once when the direct click is blocked by something a human
        would not be blocked by. Two cases, both measured on real builds:

        * **Intercepted.** A styled ``<label>`` sits over the real checkbox or
          radio, so Playwright refuses -- another element would receive the
          click. A person clicks the label and the control toggles, so the retry
          targets the label, then dispatches the event directly.
        * **Covered by a transient.** A toast or overlay is mid-animation.

        The retry is deliberately narrow: it only runs after a normal click has
        already failed, so an unclickable control still reports a failure rather
        than being forced. On the build used to validate this, these two cases
        were 13 of 32 remaining click failures.
        """
        loc = await self._locator(target)
        # The FIRST attempt gets a short budget, not the full one. A click that
        # is going to succeed does so in milliseconds. The full timeout is only
        # ever spent by a click that is blocked, and blocked is exactly the case
        # the fallbacks below handle. Spending it twice turned a recoverable
        # click into a 20-second stall.
        probe_ms = min(2500, self.config.action_timeout_ms)
        try:
            await loc.click(timeout=probe_ms)
            return
        except Exception as exc:  # noqa: BLE001
            first_error = exc

        # A person clicks the visible label and the browser routes it to the input.
        try:
            handle = await loc.element_handle(timeout=1000)
            if handle is not None:
                label = await handle.evaluate_handle(
                    "el => el.closest('label') || "
                    "(el.id && document.querySelector(`label[for='${el.id}']`))"
                )
                as_element = label.as_element() if label else None
                if as_element is not None:
                    await as_element.click(timeout=probe_ms)
                    return
        except Exception:  # noqa: BLE001 - fall through to the direct dispatch
            pass

        # Last resort: fire the click on the element itself. This bypasses the
        # actionability checks, so it is only reachable once a real click has
        # failed -- it recovers an overlay-covered control without silently
        # turning "unclickable" into "clicked".
        try:
            await loc.dispatch_event("click", timeout=probe_ms)
            return
        except Exception:  # noqa: BLE001
            pass
        raise BrowserError(
            f"could not click {target!r}: {first_error}"
        ) from first_error

    async def fill(self, target: str, text: str) -> None:
        loc = await self._locator(target)
        try:
            await loc.fill(text, timeout=self.config.action_timeout_ms)
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"could not type into {target!r}: {exc}") from exc

    async def press(self, key: str, *, target: str | None = None) -> None:
        try:
            if target is not None:
                loc = await self._locator(target)
                await loc.press(key, timeout=self.config.action_timeout_ms)
            else:
                await self.page.keyboard.press(key)
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"could not press {key!r}: {exc}") from exc

    async def drag(
        self,
        target: str,
        *,
        dx: int = 0,
        dy: int = 0,
        from_x: float | None = None,
        from_y: float | None = None,
        steps: int = 16,
    ) -> str:
        """Press, move and release across an element -- the canvas gesture.

        Without this a whole class of build is ungradeable rather than badly
        graded. Measured on the corpus: 125 of 127 handdrawn_whiteboard builds
        render their scene into a ``<canvas>``, where drawing, moving, resizing
        and panning are all one primitive gesture and none of them is reachable
        by click, fill or press. An item about drawing had no honest check, so
        its points fell to the model on every build.

        Coordinates are offsets *inside* the target's box, so a caller aims at
        "a third of the way across the canvas" without knowing where on the
        screen the canvas landed. Playwright's mouse emits real pointer events
        in Chromium, so apps listening for ``pointerdown`` see a genuine
        gesture rather than a synthesized one that bypasses their handlers.

        ``steps`` matters: an app that draws a freehand stroke records the
        pointermove path, and a single jump from start to end produces one
        straight segment or nothing at all.
        """
        loc = await self._locator(target)
        try:
            box = await loc.bounding_box(timeout=self.config.action_timeout_ms)
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"could not locate {target!r} to drag: {exc}") from exc
        if not box or not box.get("width") or not box.get("height"):
            raise BrowserError(f"{target!r} has no visible box to drag across")

        start_x = box["x"] + (box["width"] / 2 if from_x is None else from_x)
        start_y = box["y"] + (box["height"] / 2 if from_y is None else from_y)
        end_x, end_y = start_x + dx, start_y + dy
        try:
            mouse = self.page.mouse
            await mouse.move(start_x, start_y)
            await mouse.down()
            for i in range(1, max(1, steps) + 1):
                frac = i / max(1, steps)
                await mouse.move(
                    start_x + (end_x - start_x) * frac,
                    start_y + (end_y - start_y) * frac,
                )
            await mouse.up()
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"could not drag across {target!r}: {exc}") from exc
        return (
            f"dragged ({start_x:.0f},{start_y:.0f}) -> ({end_x:.0f},{end_y:.0f}) "
            f"in {steps} steps"
        )

    async def select_option(self, target: str, value: str) -> None:
        loc = await self._locator(target)
        try:
            await loc.select_option(value, timeout=self.config.action_timeout_ms)
        except Exception as exc:  # noqa: BLE001
            # A custom dropdown (a styled div, not a <select>) is a perfectly
            # normal thing to ship, and it is the majority of the select
            # failures observed. Say what to do instead, rather than leaving the
            # agent to conclude the control is broken.
            hint = (
                " -- this is not a real <select>, so open it with click() and "
                "then click the option you want"
                if "not a <select>" in str(exc)
                else ""
            )
            raise BrowserError(
                f"could not select {value!r} in {target!r}: {exc}{hint}"
            ) from exc

    async def set_files(self, target: str | None, paths: list[str]) -> None:
        """Attach local files to a file input.

        Without this an agent cannot get past the first step of any app whose
        premise is "upload a photo / PDF / screenshot" -- it can see the control
        and has no way to satisfy it. ``target`` may be ``None``, in which case
        the first file input on the page is used, because file inputs are
        frequently visually hidden behind a styled label and have no accessible
        name for the agent to aim at.
        """
        # Always drive the real <input type=file>, never the thing the agent
        # named. Upload UIs are a styled label, button or drop-zone in front of
        # an input that is display:none -- so `set_input_files` on the named
        # element either finds no input at all or blocks the full actionability
        # timeout waiting for a deliberately hidden one to become visible. That
        # is why uploading failed 70% of the time (7 of 10 attempts) across the
        # solo fleet, on the apps whose whole premise is uploading a file.
        #
        # `target` still narrows the search when a page has several uploads: an
        # input inside or near the named element is preferred, falling back to
        # the page's first file input.
        inputs = self.page.locator("input[type=file]")
        loc = inputs.first
        if target:
            try:
                named = await self._locator(target)
                scoped = named.locator("input[type=file]")
                if await scoped.count() > 0:
                    loc = scoped.first
            except Exception:  # noqa: BLE001 - fall back to the page-level input
                pass
        try:
            if await inputs.count() == 0:
                raise BrowserError(
                    "this page has no file input, so there is nothing to upload to"
                )
            # A file input is legitimately invisible, so waiting for
            # actionability is wrong here by construction.
            await loc.set_input_files(
                paths, timeout=self.config.action_timeout_ms, no_wait_after=True
            )
        except BrowserError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(
                f"could not attach {paths!r} to {target or 'the file input'}: {exc}"
            ) from exc

    async def wait_for_text(self, text: str, *, timeout_ms: int | None = None) -> bool:
        try:
            await self.page.get_by_text(text, exact=False).first.wait_for(
                timeout=timeout_ms or self.config.action_timeout_ms
            )
            return True
        except Exception:  # noqa: BLE001
            return False

    async def screenshot(self, *, label: str = "shot") -> str:
        directory = self.config.screenshot_dir
        if directory is None:
            # NOT the system temp dir. `/tmp` is a tmpfs on the sweep box, so
            # every screenshot taken there is held in RAM until reboot, and
            # nothing prunes them: 55,947 files / 6.0 GB had accumulated part
            # way through one build phase, on a machine whose sweep had already
            # been killed seven times by memory pressure.
            # `config/crowd.yaml` ships `screenshot_dir: null`,
            # so this fallback is not an edge case -- it is the path every
            # screenshot takes. builds/ is gitignored, on real disk, and is
            # already where every other run artifact goes.
            directory = builds_root() / "screenshots"
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
        path = directory / f"{safe}-{int(time.time() * 1000)}.png"
        try:
            await self.page.screenshot(path=str(path), full_page=False)
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"could not screenshot: {exc}") from exc
        return str(path)

    async def close(self) -> None:
        try:
            await self._context.close()
        except Exception:  # noqa: BLE001
            pass
