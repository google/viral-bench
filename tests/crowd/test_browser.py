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

"""Tests for the Playwright browser engine (skipped when no browser is present).

These drive ``page.set_content`` directly (no server needed) to keep the browser
layer's own behavior -- snapshot, click-by-name, fill, key press, console capture,
screenshots -- under focused test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from viral_bench.crowd.interaction.browser import (
    _DESCRIBED,
    BrowserConfig,
    BrowserEngine,
    ElementInfo,
    _looks_like_css,
    browser_available,
)

pytestmark = pytest.mark.skipif(
    not browser_available(), reason="no system browser + playwright available"
)

HTML = (
    "<!doctype html><html><head><title>T</title></head><body>"
    "<h1>Hi there</h1>"
    "<button id='go' onclick=\"document.getElementById('o').textContent='clicked'\">"
    "Go</button>"
    "<input placeholder='your name'>"
    "<p id='o'>idle</p>"
    "<script>"
    "console.error('boom');"
    "document.addEventListener('keydown', (e) => {"
    "document.getElementById('o').textContent = 'key:' + e.key; });"
    "</script>"
    "</body></html>"
)


def _run(coro):
    return asyncio.run(coro)


async def _page(engine: BrowserEngine):
    handle = await engine.open_page()
    await handle.page.set_content(HTML)
    return handle


def test_snapshot_reports_title_text_and_controls() -> None:
    async def inner():
        async with BrowserEngine() as engine:
            handle = await _page(engine)
            snap = await handle.snapshot()
            assert snap.title == "T"
            assert "Hi there" in snap.text
            labels = [e.label for e in snap.elements]
            assert "Go" in labels
            assert 'button "Go"' in snap.aria
            await handle.close()

    _run(inner())


def test_click_by_accessible_name_changes_dom() -> None:
    async def inner():
        async with BrowserEngine() as engine:
            handle = await _page(engine)
            await handle.click("Go")
            snap = await handle.snapshot()
            assert "clicked" in snap.text
            await handle.close()

    _run(inner())


def test_fill_by_placeholder() -> None:
    async def inner():
        async with BrowserEngine() as engine:
            handle = await _page(engine)
            await handle.fill("your name", "Koa")
            value = await handle.page.locator("input").input_value()
            assert value == "Koa"
            await handle.close()

    _run(inner())


def test_press_key_dispatches_to_page() -> None:
    async def inner():
        async with BrowserEngine() as engine:
            handle = await _page(engine)
            await handle.press("Enter")
            snap = await handle.snapshot()
            assert "key:Enter" in snap.text
            await handle.close()

    _run(inner())


def test_console_errors_are_captured() -> None:
    async def inner():
        async with BrowserEngine() as engine:
            handle = await _page(engine)
            await handle.page.wait_for_timeout(150)  # let the console event fire
            snap = await handle.snapshot()
            assert any("boom" in e for e in snap.console_errors)
            await handle.close()

    _run(inner())


def test_screenshot_writes_a_file(tmp_path) -> None:
    async def inner():
        config = BrowserConfig(screenshot_dir=tmp_path)
        async with BrowserEngine(config) as engine:
            handle = await _page(engine)
            path = await handle.screenshot(label="shot")
            assert Path(path).is_file()
            assert Path(path).stat().st_size > 0
            await handle.close()

    _run(inner())


def test_unconfigured_screenshots_land_on_disk_not_in_tmpfs(tmp_path, monkeypatch):
    """The default screenshot directory must be builds/, never the temp dir.

    `/tmp` is a tmpfs on the sweep box, so a screenshot written there is held in
    RAM until reboot and nothing ever prunes it: 55,947 files / 6.0 GB had
    accumulated by the end of one build phase, on a machine whose sweep had been
    killed repeatedly by memory exhaustion. And this is not a rare path:
    `config/crowd.yaml` ships `screenshot_dir: null`, so every screenshot the
    crowd takes goes through this fallback.
    """
    import tempfile

    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path / "builds"))
    assert BrowserConfig().screenshot_dir is None, "the fallback is what is tested"

    async def inner():
        async with BrowserEngine() as engine:
            handle = await _page(engine)
            path = Path(await handle.screenshot(label="shot"))
            assert path.is_file() and path.stat().st_size > 0
            assert path.parent == tmp_path / "builds" / "screenshots"
            # The specific regression. Asserting "not under gettempdir()" would
            # be vacuous, because pytest's own tmp_path lives there too -- so
            # name the directory that filled with RAM.
            assert path.parent != Path(tempfile.gettempdir()) / "viralbench-shots"
            await handle.close()

    _run(inner())


def test_missing_target_click_raises_browser_error() -> None:
    from viral_bench.crowd.interaction.browser import BrowserError

    async def inner():
        async with BrowserEngine() as engine:
            handle = await _page(engine)
            with pytest.raises(BrowserError):
                await handle.click("NoSuchButtonAnywhere")
            await handle.close()

    _run(inner())


# -- target resolution -------------------------------------------------------
#
# Selector handling caused 104 of 147 failed interaction steps. The largest
# cause: describe() shows the agent `button "Save note"`, the agent copies that
# string back, and nothing parsed it -- it fell through to
# page.locator('button "Save note"'), which is not valid CSS. The step then
# died with a parse error instead of a missing-element error.


def test_described_form_is_parsed() -> None:
    """The exact string format the agent is shown must be understood."""
    m = _DESCRIBED.match('button "Save note"')
    assert m and m.group(1) == "button" and m.group(2) == "Save note"


def test_described_form_survives_angle_brackets_in_the_label() -> None:
    m = _DESCRIBED.match('a "Next >"')
    assert m and m.group(2) == "Next >"


def test_typed_input_describe_form_is_parsed() -> None:
    m = _DESCRIBED.match('input:text "Email"')
    assert m and m.group(1) == "input:text"


def test_real_css_selectors_are_recognised() -> None:
    for sel in ("#app", ".btn-primary", "div.card"):
        assert _looks_like_css(sel), sel


def test_human_labels_are_not_mistaken_for_css() -> None:
    """The old heuristic sent these to a CSS parser, which rejected them."""
    for label in ("Next >", ".env settings", "Save note", 'button "Save"'):
        assert not _looks_like_css(label), label


def test_element_describe_round_trips_through_the_matcher() -> None:
    """Whatever describe() emits, _locator must be able to parse back."""
    for info in (
        ElementInfo(tag="button", label="Save note", role="button", type=""),
        ElementInfo(tag="input", label="Email", role="", type="text"),
        ElementInfo(tag="a", label="Next >", role="link", type=""),
    ):
        described = info.describe()
        assert _DESCRIBED.match(described), f"describe() emitted {described!r}"
