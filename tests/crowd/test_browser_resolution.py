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

"""Element resolution against the DOM shapes real generated apps use.

These are regression tests for the two interaction failures measured across the
25 solo builds: clicks timing out 9.3% of the time (109 of 136 timeouts were
"element is not visible") and uploads failing 70% of the time on exactly the
apps whose premise is uploading a file.

Both come from the same wrong assumption -- that an element present in the DOM
is an element you can act on. Modern apps keep closed modals mounted and hide
file inputs behind styled labels, so presence and actionability are different
questions.
"""

from __future__ import annotations

import http.server
import socketserver
import threading

import pytest

pytest.importorskip("playwright", reason="needs the crowd env (.venv-crowd)")

# importorskip is not enough: the main venv has the playwright PACKAGE but no
# installed browser, so these would fail there while passing in .venv-crowd.
import asyncio  # noqa: E402

# A closed modal holding a hidden duplicate of a visible control, plus an upload
# UI whose real input is display:none. Both patterns came from real builds.
PAGE = b"""<!doctype html><html><body>
<div id="modal" style="display:none">
  <button aria-label="Sign In" onclick="go('modal')">Sign In</button>
</div>
<header><button aria-label="Sign In" onclick="go('header')">Sign In</button></header>
<label for="pick" id="dz">Select Image File</label>
<input type="file" id="pick" style="display:none">
<div id="out">nothing yet</div>
<script>
function go(w){ document.getElementById('out').textContent = 'clicked: ' + w; }
document.getElementById('pick').addEventListener('change', e => {
  document.getElementById('out').textContent = 'uploaded: ' + e.target.files[0].name;
});
</script></body></html>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(PAGE)))
        self.end_headers()
        self.wfile.write(PAGE)


@pytest.fixture
def page_url():
    srv = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/"
    finally:
        srv.shutdown()


def test_click_prefers_the_visible_match_over_a_hidden_duplicate(page_url):
    """A hidden copy in a closed modal must not swallow the click.

    Resolution used to take the first strategy whose count() was non-zero, so
    `Sign In` landed on the hidden modal button and Playwright blocked the full
    actionability timeout waiting for it to appear.
    """

    async def _run():
        from viral_bench.crowd.interaction.browser import BrowserConfig, BrowserEngine

        engine = BrowserEngine(BrowserConfig())
        await engine.start()
        try:
            page = await engine.open_page()
            await page.goto(page_url)
            await page.click("Sign In")
            text = (await page.snapshot()).text
            assert "clicked: header" in text, "clicked the hidden modal copy"
        finally:
            await engine.close()

    asyncio.run(_run())


def test_upload_drives_the_hidden_file_input(page_url):
    """Upload must work through a display:none input behind a styled label.

    That is how nearly every real upload UI is built, and it is why
    uploading failed on 7 of 10 attempts in the fleet.
    """

    async def _run():
        import asyncio

        from viral_bench.crowd.interaction.browser import BrowserConfig, BrowserEngine
        from viral_bench.crowd.interaction.fixtures import fixture_path

        engine = BrowserEngine(BrowserConfig())
        await engine.start()
        try:
            page = await engine.open_page()
            await page.goto(page_url)
            # Named by the label the agent can see, not the input.
            await page.set_files("Select Image File", [str(fixture_path("photo.png"))])
            await asyncio.sleep(0.3)
            assert "uploaded: photo.png" in (await page.snapshot()).text
        finally:
            await engine.close()

    asyncio.run(_run())


def test_upload_on_a_page_with_no_file_input_says_so(page_url):
    """The agent must learn the page has no upload, not hit a raw timeout."""

    async def _run():
        from viral_bench.crowd.interaction.browser import (
            BrowserConfig,
            BrowserEngine,
            BrowserError,
        )
        from viral_bench.crowd.interaction.fixtures import fixture_path

        engine = BrowserEngine(BrowserConfig())
        await engine.start()
        try:
            page = await engine.open_page()
            await page.goto("data:text/html,<button>nothing here</button>")
            with pytest.raises(BrowserError, match="no file input"):
                await page.set_files(None, [str(fixture_path("photo.png"))])
        finally:
            await engine.close()

    asyncio.run(_run())


CONTROLS = b"""<!doctype html><html><head><style>
.sw input{position:absolute;opacity:0;width:0;height:0}
.sw span{display:inline-block;padding:6px 14px;background:#ddd}
#veil{position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:9}
</style></head><body>
<label class="sw"><input type="checkbox" id="pub" aria-label="Toggle Public">
<span>Public</span></label>
<button id="under" onclick="say('under-overlay')">Save Note</button>
<div id="veil"></div><div id="out">nothing yet</div>
<script>
function say(w){document.getElementById('out').textContent='clicked: '+w;}
document.getElementById('pub').addEventListener('change',e=>say('checkbox='+e.target.checked));
</script></body></html>"""


class _ControlsHandler(_Handler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(CONTROLS)))
        self.end_headers()
        self.wfile.write(CONTROLS)


@pytest.fixture
def controls_url():
    srv = socketserver.TCPServer(("127.0.0.1", 0), _ControlsHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/"
    finally:
        srv.shutdown()


def test_click_recovers_a_label_wrapped_control_and_an_overlay(controls_url):
    """The two shapes that survived the visibility fix.

    On the validation build these were 13 and 7 of 32 residual click failures.
    A styled <label> over a visually-hidden checkbox is how nearly every toggle
    switch is built, and a person clicks the label. A transient overlay blocks a
    click a person would make a moment later. Both must recover,
    and quickly -- the first attempt uses a short budget so a blocked click does
    not spend the full timeout before the fallback even starts.
    """
    import asyncio
    import time

    from viral_bench.crowd.interaction.browser import BrowserConfig, BrowserEngine

    async def _run():
        engine = BrowserEngine(BrowserConfig())
        await engine.start()
        try:
            page = await engine.open_page()
            await page.goto(controls_url)

            started = time.monotonic()
            await page.click("Toggle Public")
            assert "checkbox=true" in (await page.snapshot()).text
            assert time.monotonic() - started < 9.0, "label fallback was too slow"

            started = time.monotonic()
            await page.click("Save Note")
            assert "under-overlay" in (await page.snapshot()).text
            assert time.monotonic() - started < 9.0, "overlay fallback was too slow"
        finally:
            await engine.close()

    asyncio.run(_run())
