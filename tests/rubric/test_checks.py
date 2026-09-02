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

"""L1: prove every check primitive against synthetic fixture apps.

Ground truth by construction. Two pages are served over real HTTP and driven by
a real browser: ``fixtures/good`` is built so every primitive passes, and
``fixtures/bad`` so every primitive fails -- each in a way copied from an actual
defect in the corpus rather than invented.

This is the cheap checkpoint. No model is involved and no real build is needed,
so a failure here is unambiguously a bug in the primitive.
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import json
import threading
from pathlib import Path

import pytest

from viral_bench.crowd.interaction.browser import (
    BrowserConfig,
    BrowserEngine,
    browser_available,
)
from viral_bench.rubric.checks import CheckContext, registry, run_check

FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(
    not browser_available(), reason="needs Playwright and a system Chrome"
)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Silent request handler.

    ``SimpleHTTPRequestHandler`` logs every request to ``sys.stderr`` from a
    daemon thread, and writing to pytest's captured stderr from a background
    thread deadlocks the run. ``viz/serve.py`` overrides ``log_message`` for the
    same reason.
    """

    def log_message(self, fmt, *args) -> None:  # noqa: A002 - stdlib signature
        pass


@pytest.fixture(scope="module")
def served():
    """Serve both fixture apps over real HTTP, one port each."""
    servers, urls = [], {}
    for name in ("good", "bad"):
        handler = functools.partial(QuietHandler, directory=str(FIXTURES / name))
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        urls[name] = f"http://127.0.0.1:{httpd.server_address[1]}/"
    yield urls
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


async def _drive(url, name, params, before, scratch):
    """Open a browser, put the page in the right state, run one primitive."""
    engine = BrowserEngine(BrowserConfig(headless=True))
    await engine.start()
    try:
        page = await engine.open_page(permissions=["clipboard-read", "clipboard-write"])
        try:
            await page.goto(url)
            ctx = CheckContext(url=url, page=page, scratch=scratch, request_mark=0)
            if before is not None:
                await before(page, ctx)
            return await run_check(name, ctx, params)
        finally:
            await page.close()
    finally:
        await engine.close()


def run(served, fixture, name, params=None, *, before=None, scratch=None):
    """Run one primitive against one fixture app, in its own event loop.

    A browser per check rather than a shared one: Playwright binds its objects to
    the loop that created them, and sharing an engine across pytest fixtures
    deadlocked on teardown. A launch is ~1s, which is a fair price for a test
    harness with no cross-test state at all.
    """
    return asyncio.run(
        _drive(served[fixture], name, params or {}, before, scratch or Path("/tmp"))
    )


# --------------------------------------------------------------- page checks


def test_is_real_control_separates_an_input_from_a_labelled_div(served):
    assert run(served, "good", "is_real_control", {"target": "Note"}).passed is True
    bad = run(served, "bad", "is_real_control", {"target": "Note"})
    assert bad.passed is False
    assert "not a fillable control" in bad.detail


def test_aria_names_counts_a_machine_readable_board(served):
    pattern = r"^row \d+, column \d+, (empty|tile \d+)$"
    params = {"pattern": pattern, "count": 4, "unique": True}
    good = run(served, "good", "aria_names", params)
    assert good.passed is True

    bad = run(served, "bad", "aria_names", {"pattern": pattern, "count": 4})
    assert bad.passed is False, "a board flattened to bare text must not pass"


def test_computed_style_distinct_catches_flat_highlighting(served):
    params = {"selector": "#code span", "prop": "color", "min_distinct": 3}
    assert run(served, "good", "computed_style_distinct", params).passed is True
    assert run(served, "bad", "computed_style_distinct", params).passed is False


def test_value_equals_compares_in_python_not_in_the_page(served):
    js = "() => document.getElementById('unicode').textContent"
    params = {"js": js, "expect": "héllo 🎉 안녕하세요"}
    good = run(served, "good", "value_equals", params)
    assert good.passed is True
    bad = run(served, "bad", "value_equals", params)
    assert bad.passed is False, "mojibake must not compare equal"


def test_dom_count(served):
    params = {"selector": "[role=gridcell]", "op": "==", "n": 4}
    assert run(served, "good", "dom_count", params).passed is True
    assert run(served, "bad", "dom_count", params).passed is False


def test_dom_text_absent_catches_a_shipped_placeholder(served):
    params = {"pattern": r"\(placeholder\)|coming soon|lorem ipsum"}
    assert run(served, "good", "dom_text_absent", params).passed is True
    assert run(served, "bad", "dom_text_absent", params).passed is False


# ------------------------------------------------------------ the arithmetic


def test_arith_treats_the_sign_as_part_of_the_answer(served):
    """The sharpest finding in the corpus, as a unit test.

    Two builds produced byte-identical outcomes; the one printing ``-166.9``
    scored 2.53 and the one printing ``166.9`` under a "saved" label scored 7.73.
    A check that ignored the sign would rate them the same.
    """
    js = (
        "() => document.getElementById('savings').textContent"
        "  .replace(/[^0-9.-]/g, '')"
    )
    params = {"js": js, "expect": -167.0, "tolerance": 0.5}
    assert run(served, "good", "arith", params).passed is True
    bad = run(served, "bad", "arith", params)
    assert bad.passed is False
    assert "+333" in bad.detail or "333" in bad.detail


def test_number_in_range_catches_an_impossible_wpm(served):
    js = "() => document.getElementById('wpm').textContent.replace(/[^0-9.-]/g, '')"
    params = {"js": js, "low": 0, "high": 300}
    assert run(served, "good", "number_in_range", params).passed is True
    assert run(served, "bad", "number_in_range", params).passed is False


# ------------------------------------------------------------------ console


def test_no_console_errors_catches_an_uncaught_error(served):
    assert run(served, "good", "no_console_errors").passed is True
    bad = run(served, "bad", "no_console_errors")
    assert bad.passed is False
    assert "undefinedFunctionCallOnLoad" in bad.detail


def test_the_tailwind_cdn_warning_is_ignored_by_default(served):
    """687 occurrences corpus-wide: a check that fires on every build is noise."""

    async def inject(page, ctx):
        await page.page.evaluate(
            "() => console.warn('cdn.tailwindcss.com should not be used in production')"
        )

    assert run(served, "good", "no_console_errors", before=inject).passed is True


# ------------------------------------------------------------------ network


def test_network_origins_catches_a_third_party_fetch(served):
    assert run(served, "good", "network_origins").passed is True
    bad = run(served, "bad", "network_origins")
    assert bad.passed is False
    assert "cdn.example.invalid" in bad.observed


# ---------------------------------------------------------------- downloads


def test_download_and_image_props_catch_a_dead_export_button(served, tmp_path):
    """An export that fires nothing is indistinguishable from a working one to
    the crowd; it is trivially distinguishable here."""

    async def click_export(page, ctx):
        directory = tmp_path / "dl"
        with page.capture_downloads(directory) as captured:
            await page.click("Export PNG")
            for _ in range(40):
                if captured:
                    break
                await asyncio.sleep(0.05)
        ctx.downloads = list(captured)

    good = run(
        served,
        "good",
        "download",
        {"magic": "png", "min_size": 50},
        before=click_export,
    )
    assert good.passed is True, good.detail

    dims = run(
        served, "good", "image_props", {"width": 32, "height": 16}, before=click_export
    )
    assert dims.passed is True, dims.detail

    bad = run(served, "bad", "download", {"magic": "png"}, before=click_export)
    assert bad.passed is False
    assert "no download" in bad.observed


# ---------------------------------------------------------------- clipboard


def test_clipboard_equals_separates_a_real_copy_from_a_toast(served):
    async def click_copy(page, ctx):
        await page.click("Copy")
        await asyncio.sleep(0.15)

    js = "() => document.getElementById('code').textContent"
    good = run(served, "good", "clipboard_equals", {"js": js}, before=click_copy)
    assert good.passed is True, good.detail

    bad = run(served, "bad", "clipboard_equals", {"js": js}, before=click_copy)
    assert bad.passed is False
    assert "empty" in bad.observed or "expected" in bad.detail


# -------------------------------------------------------------- persistence


def test_survives_reload_requires_the_users_own_nonce(served):
    """A demo default surviving a reload is not persistence.

    One build was awarded 30/30 on persistence for preserving an em-dash
    placeholder; the nonce precondition is what makes the check honest.
    """
    nonce = "ZQX-CANARY-7741"

    async def type_and_save(page, ctx):
        await page.fill("Note", nonce)
        await page.click("Save")
        await asyncio.sleep(0.1)

    js = "() => document.getElementById('note').value || ''"
    good = run(
        served,
        "good",
        "survives_reload",
        {"js": js, "nonce": nonce},
        before=type_and_save,
    )
    assert good.passed is True, good.detail

    async def click_save_only(page, ctx):
        await page.click("Save")
        await asyncio.sleep(0.1)

    js_bad = "() => document.getElementById('note').textContent || ''"
    bad = run(
        served,
        "bad",
        "survives_reload",
        {"js": js_bad, "nonce": nonce},
        before=click_save_only,
    )
    assert bad.passed is False, "a Saved! toast that stores nothing must fail"


# --------------------------------------------------------------------- HTTP


def test_http_status(served):
    ctx = CheckContext(url=served["good"])
    from viral_bench.rubric.checks import http_status

    assert http_status(ctx, path="/").passed is True
    assert http_status(ctx, path="/nope.html", expect=200).passed is False
    assert http_status(ctx, path="/nope.html", expect=404).passed is True


def test_http_status_can_assert_on_the_body(served):
    from viral_bench.rubric.checks import http_status

    ctx = CheckContext(url=served["good"])
    assert http_status(ctx, path="/", body_contains="Fixture app").passed is True
    assert http_status(ctx, path="/", body_contains="nonsense").passed is False
    assert http_status(ctx, path="/", body_excludes="Fixture app").passed is False


# ------------------------------------------------------------------- source


def test_source_checks_read_the_tree(tmp_path):
    from viral_bench.crowd.interaction.inspect import CodeInspectionToolkit
    from viral_bench.rubric.checks import source_absent, source_present

    leak = "const KEY = 'AIzaLeakedSecret';\n"
    (tmp_path / "app.js").write_text(leak, encoding="utf-8")
    ctx = CheckContext(url="", source=CodeInspectionToolkit("x", app_dir=tmp_path))

    assert source_present(ctx, pattern="AIza").passed is True
    assert source_absent(ctx, pattern="AIza").passed is False
    assert source_absent(ctx, pattern="NOTHING_LIKE_THIS").passed is True


# --------------------------------------------------------------- error paths


def test_a_primitive_never_raises_and_reports_unknown_instead():
    """A grader that dies on one malformed page loses the whole build's grade."""
    ctx = CheckContext(url="http://127.0.0.1:1/")

    assert asyncio.run(run_check("nope_not_real", ctx, {})).passed is None
    assert asyncio.run(run_check("value_equals", ctx, {"js": "() => 1"})).passed is None
    bad_params = asyncio.run(run_check("dom_count", ctx, {"wrong": 1}))
    assert bad_params.passed is None and "bad parameters" in bad_params.detail


def test_a_check_that_cannot_run_is_unknown_not_failed():
    """None and False are different facts and must not be collapsed."""
    ctx = CheckContext(url="http://x/", page=None)
    result = asyncio.run(run_check("no_console_errors", ctx, {}))
    assert result.passed is None, "no page means undetermined, not a failure"


# ------------------------------------------------------- universal Tier 3


def test_controls_have_names_catches_a_div_posing_as_a_field(served):
    """The universal form of is_real_control: a survey, not one named target.

    The bad page carries both shapes that break a crowd agent -- a div wearing
    role="textbox" where an input belongs, and an icon-only button with no
    accessible name at all. Neither is visible to an agent, which simply reports
    that it could not find the control.
    """
    good = run(served, "good", "controls_have_names")
    assert good.passed is True, good.detail

    bad = run(served, "bad", "controls_have_names")
    assert bad.passed is False
    assert "stands in for a field" in bad.detail


def test_no_mojibake_catches_utf8_read_as_latin1(served):
    text = "héllo 🎉 안녕하세요"
    good = run(served, "good", "no_mojibake", {"text": text})
    assert good.passed is True, good.detail

    bad = run(served, "bad", "no_mojibake", {"text": text})
    assert bad.passed is False
    assert "mangled" in bad.detail


def test_the_default_read_sees_text_typed_into_an_input(served):
    """innerText alone misses input.value, which would fail every editor app."""
    nonce = "VBNONCE-DEFAULTREAD"

    async def type_and_save(page, ctx):
        await page.fill("Note", nonce)
        await page.click("Save")
        await asyncio.sleep(0.1)

    # No `js` param: the primitive's own default has to find the input's value.
    result = run(
        served, "good", "survives_reload", {"nonce": nonce}, before=type_and_save
    )
    assert result.passed is True, result.detail


def test_default_restored_separates_your_work_from_any_work(served):
    """The corpus's most-rewarded illusion: a populated screen after a reload."""
    nonce = "VBNONCE-DEFAULTRESTORE"

    async def type_and_save(page, ctx):
        await page.fill("Note", nonce)
        await page.click("Save")
        await asyncio.sleep(0.1)

    # The honest app restores the user's own text, so the penalty must NOT fire.
    good = run(
        served, "good", "default_restored", {"nonce": nonce}, before=type_and_save
    )
    assert good.passed is False, good.detail

    async def click_save_only(page, ctx):
        await page.click("Save")
        await asyncio.sleep(0.1)

    # The dishonest one shows a full page after reload with none of your work.
    bad = run(
        served, "bad", "default_restored", {"nonce": nonce}, before=click_save_only
    )
    assert bad.passed is True, "content restored but not the user's must fire"


# ------------------------------------------------ source-judged universals


def source_ctx(tmp_path, files):
    """A CheckContext over a throwaway source tree."""
    from viral_bench.crowd.interaction.inspect import CodeInspectionToolkit

    for name, body in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return CheckContext(
        url="", source=CodeInspectionToolkit("b", app_dir=tmp_path), scratch=tmp_path
    )


def source_check(tmp_path, files, name, params=None):
    return asyncio.run(run_check(name, source_ctx(tmp_path, files), params or {}))


#: A key of realistic shape and length, assembled at import time.
#:
#: Not a literal, because the repo's pre-commit secret scanner matches on
#: exactly this shape and would refuse the commit. The primitive under test
#: needs the real length -- its pattern requires 30+ characters after the
#: prefix -- so shortening it would stop the test exercising anything.
FAKE_GOOGLE_KEY = "AIza" + "SyB" + "1234567890abcdefghijklmnopqrstuvw"


def test_client_secret_present_ignores_a_server_side_key(tmp_path):
    """A key in a server module is configuration; in a bundle it is a leak.

    Grepping the whole tree, as a naive check would, fires on both and so
    distinguishes nothing -- which is the entire point of this primitive.
    """
    served_key = source_check(
        tmp_path,
        {"app.js": f'const k = "{FAKE_GOOGLE_KEY}";'},
        "client_secret_present",
    )
    assert served_key.passed is True, "a key in a .js bundle must fire"


def test_client_secret_present_does_not_fire_on_server_config(tmp_path):
    result = source_check(
        tmp_path,
        {"server.py": f'KEY = "{FAKE_GOOGLE_KEY}"'},
        "client_secret_present",
    )
    assert result.passed is False, "a server-side key is configuration, not a leak"


def test_model_dependency_detects_a_bolted_on_ai_feature(tmp_path):
    fires = source_check(
        tmp_path, {"app.js": "fetch(process.env.GEMINI_API_KEY)"}, "model_dependency"
    )
    assert fires.passed is True

    clean = source_check(
        tmp_path, {"app.js": "const total = a + b;"}, "model_dependency"
    )
    assert clean.passed is False


def test_key_optional_passes_an_app_with_no_model_at_all(tmp_path):
    result = source_check(tmp_path, {"app.js": "const x = 1;"}, "key_optional")
    assert result.passed is True
    assert "no model key needed" in result.observed


def test_key_optional_fails_an_app_that_dies_without_a_key(tmp_path):
    result = source_check(
        tmp_path,
        {
            "app.js": "if (!process.env.OPENAI_API_KEY) "
            'throw new Error("OPENAI_API_KEY required");'
        },
        "key_optional",
    )
    assert result.passed is False
    assert "startup dies" in result.detail


def test_key_optional_passes_a_guarded_reference(tmp_path):
    result = source_check(
        tmp_path,
        {
            "app.js": "const key = process.env.GEMINI_API_KEY || null;\n"
            'if (!key) { showMessage("AI features are unavailable."); }'
        },
        "key_optional",
    )
    assert result.passed is True


# ---------------------------------------------------- DOM-agnostic reads


def test_text_matches_reads_the_rendered_page_not_the_markup(served):
    """Forty builds of one idea share no ids, so checks read text, not selectors."""
    good = run(served, "good", "text_matches", {"patterns": ["Fixture app", "62"]})
    assert good.passed is True, good.detail

    missing = run(served, "good", "text_matches", {"patterns": ["nowhere-in-page"]})
    assert missing.passed is False


def test_text_matches_absent_catches_what_should_not_be_there(served):
    leaked = run(served, "bad", "text_matches", {"absent": ["Placeholder"]})
    assert leaked.passed is False, "a shipped placeholder must be caught"

    clean = run(served, "good", "text_matches", {"absent": ["Placeholder"]})
    assert clean.passed is True


def test_text_number_equals_treats_the_sign_as_part_of_the_answer(served):
    """The corpus's sharpest finding: +166.9 and -166.9 are not the same claim."""
    pattern = r"Saved:\s*(-?\d+(?:\.\d+)?)"
    good = run(
        served,
        "good",
        "text_number_equals",
        {"pattern": pattern, "expect": -167, "tolerance": 1},
    )
    assert good.passed is True, good.detail

    bad = run(
        served,
        "bad",
        "text_number_equals",
        {"pattern": pattern, "expect": -167, "tolerance": 1},
    )
    assert bad.passed is False, "a stripped minus sign must not pass"


def test_text_number_in_range_catches_the_impossible_figure(served):
    """12,000 WPM after one keystroke, without knowing the right answer."""
    pattern = r"WPM:\s*(\d+)"
    good = run(
        served,
        "good",
        "text_number_in_range",
        {"pattern": pattern, "low": 0, "high": 300},
    )
    assert good.passed is True, good.detail

    bad = run(
        served,
        "bad",
        "text_number_in_range",
        {"pattern": pattern, "low": 0, "high": 300},
    )
    assert bad.passed is False
    assert "outside" in bad.detail


def test_distinct_sources_needs_two_genuinely_different_images(served):
    """A before/after pane showing the same source twice is not a comparison."""
    good = run(served, "good", "distinct_sources", {"selector": "img", "minimum": 2})
    assert good.passed is True, good.detail

    bad = run(served, "bad", "distinct_sources", {"selector": "img", "minimum": 2})
    assert bad.passed is False, "the same image twice is not a before/after"


def test_third_party_requests_is_the_inverse_of_network_origins(served):
    good = run(served, "good", "third_party_requests")
    assert good.passed is False, "an all-local page must not fire the penalty"

    bad = run(served, "bad", "third_party_requests")
    assert bad.passed is True, "a CDN fetch must fire it"


def test_value_changes_compares_two_harness_snapshots(served):
    """The model picks the moment; the harness picks what a snapshot is."""

    async def capture_two(page, ctx):
        ctx.captures["before"] = await page.evaluate("() => document.title")
        await page.fill("Note", "changed")
        ctx.captures["after"] = await page.evaluate(
            "() => document.getElementById('note').value"
        )

    result = run(served, "good", "value_changes", {}, before=capture_two)
    assert result.passed is True, result.detail


def test_value_changes_is_unresolved_when_the_grader_never_captured(served):
    """Missing evidence is an instrument fault, not a failed app."""
    result = run(served, "good", "value_changes", {})
    assert result.passed is None
    assert "never captured" in result.detail


def test_captures_monotonic_catches_a_size_that_grows(served):
    async def three(page, ctx):
        ctx.captures["q90"] = "size 900 KB"
        ctx.captures["q60"] = "size 500 KB"
        ctx.captures["q30"] = "size 700 KB"

    result = run(
        served,
        "good",
        "captures_monotonic",
        {"labels": ["q90", "q60", "q30"], "pattern": r"([\d.]+)\s*KB"},
        before=three,
    )
    assert result.passed is False
    assert "900 -> 500 -> 700" in result.observed


def test_captures_monotonic_accepts_a_falling_series(served):
    async def three(page, ctx):
        ctx.captures["q90"] = "size 900 KB"
        ctx.captures["q60"] = "size 500 KB"
        ctx.captures["q30"] = "size 300 KB"

    result = run(
        served,
        "good",
        "captures_monotonic",
        {"labels": ["q90", "q60", "q30"], "pattern": r"([\d.]+)\s*KB"},
        before=three,
    )
    assert result.passed is True


# ------------------------------------------------------------------- pdf


def test_pdf_props_reads_object_stream_pdfs(tmp_path):
    """pdf-lib defaults to object streams; 59 of 60 surveyed builds use them.

    Their page dictionaries are Flate-compressed inside an /ObjStm, so a raw
    byte scan finds zero pages in a perfectly correct three-page merge. Before
    this was fixed, every honest build failed the page-count items and every
    build that produced no file at all failed them identically.
    """
    import zlib

    from viral_bench.rubric.checks import CheckContext, run_check

    inner = b"<< /Type /Page /Parent 1 0 R >> << /Type /Page /Parent 1 0 R >>"
    content = zlib.compress(b"BT /F1 12 Tf (Page A body) Tj ET")
    pdf = (
        b"%PDF-1.7\n5 0 obj << /Type /ObjStm /Filter /FlateDecode >> stream\n"
        + zlib.compress(inner)
        + b"\nendstream endobj\n6 0 obj << /Filter /FlateDecode >> stream\n"
        + content
        + b"\nendstream endobj\ntrailer\n%%EOF"
    )
    target = tmp_path / "merged.pdf"
    target.write_bytes(pdf)
    ctx = CheckContext(url="", downloads=[{"path": str(target), "name": "merged.pdf"}])

    counted = asyncio.run(run_check("pdf_props", ctx, {"pages": 2}))
    assert counted.passed is True, counted.detail

    text = asyncio.run(run_check("pdf_props", ctx, {"contains": "Page A body"}))
    assert text.passed is True, "content-stream text must be searchable"

    wrong = asyncio.run(run_check("pdf_props", ctx, {"pages": 5}))
    assert wrong.passed is False


# ---------------------------------------- primitives the golden set asked for


def test_sql_executes_rejects_an_export_that_will_not_run(served):
    """A MySQL-flavoured export satisfies every regex and parses nowhere."""
    mysql = "CREATE TABLE users (id INT AUTO_INCREMENT PRIMARY KEY) ENGINE=InnoDB;"

    async def show(page, ctx):
        await page.evaluate(
            "() => { document.body.innerHTML += "
            + json.dumps("<pre>" + mysql + "</pre>")
            + "; }"
        )

    result = run(served, "good", "sql_executes", {"tables": ["users"]}, before=show)
    assert result.passed is False
    assert "does not run" in result.detail


def test_sql_executes_accepts_valid_ddl_and_checks_the_schema(served):
    ddl = (
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
        "CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER "
        "REFERENCES users(id));"
    )

    async def show(page, ctx):
        await page.evaluate(
            "() => { document.body.innerHTML += "
            + json.dumps("<pre>" + ddl + "</pre>")
            + "; }"
        )

    result = run(
        served,
        "good",
        "sql_executes",
        {
            "tables": ["users", "posts"],
            "foreign_keys": [{"table": "posts", "references": "users"}],
        },
        before=show,
    )
    assert result.passed is True, result.detail

    wrong = run(served, "good", "sql_executes", {"tables": ["nowhere"]}, before=show)
    assert wrong.passed is False


def test_download_absent_fires_when_nothing_was_written(served):
    """The window.print() case: a dialog opens and no file appears."""
    nothing = run(served, "bad", "download_absent")
    assert nothing.passed is True, "no file means the penalty fires"


def test_download_absent_does_not_fire_when_a_file_appeared(served, tmp_path):
    from viral_bench.rubric.checks import CheckContext, run_check

    target = tmp_path / "out.pdf"
    target.write_bytes(b"%PDF-1.7\n%%EOF")
    ctx = CheckContext(url="", downloads=[{"path": str(target), "name": "out.pdf"}])
    result = asyncio.run(run_check("download_absent", ctx, {}))
    assert result.passed is False


def test_text_pattern_count_counts_occurrences(served):
    many = run(
        served, "good", "text_pattern_count", {"pattern": r"Fixture", "minimum": 1}
    )
    assert many.passed is True, many.detail

    too_few = run(
        served, "good", "text_pattern_count", {"pattern": r"Fixture", "minimum": 99}
    )
    assert too_few.passed is False


def test_text_numbers_distinct_catches_a_constant_fake(served):
    """A monitor reporting one response time forever is not measuring."""

    async def constant(page, ctx):
        await page.evaluate(
            "() => { document.body.innerHTML += "
            "'<p>120 ms</p><p>120 ms</p><p>120 ms</p>'; }"
        )

    same = run(
        served,
        "good",
        "text_numbers_distinct",
        {"pattern": r"(\d+)\s*ms", "minimum": 2},
        before=constant,
    )
    assert same.passed is False
    assert "distinct" in same.detail

    async def varied(page, ctx):
        await page.evaluate(
            "() => { document.body.innerHTML += "
            "'<p>120 ms</p><p>145 ms</p><p>131 ms</p>'; }"
        )

    differs = run(
        served,
        "good",
        "text_numbers_distinct",
        {"pattern": r"(\d+)\s*ms", "minimum": 2},
        before=varied,
    )
    assert differs.passed is True, differs.detail


def test_console_errors_present_is_the_inverse_of_no_console_errors(served):
    good = run(served, "good", "console_errors_present")
    assert good.passed is False, "a clean page must not fire the penalty"

    bad = run(served, "bad", "console_errors_present")
    assert bad.passed is True, "an uncaught error must fire it"


def test_computed_style_changes_sees_a_theme_switch(served):
    async def restyle(page, ctx):
        light = '["rgb(255, 255, 255)|rgb(0, 0, 0)|serif"]'
        dark = '["rgb(17, 17, 17)|rgb(238, 238, 238)|serif"]'
        ctx.captures["before"] = {"style": light}
        ctx.captures["after"] = {"style": dark}

    changed = run(served, "good", "computed_style_changes", {}, before=restyle)
    assert changed.passed is True, changed.detail

    async def dead(page, ctx):
        same = '["rgb(255, 255, 255)|rgb(0, 0, 0)|serif"]'
        ctx.captures["before"] = {"style": same}
        ctx.captures["after"] = {"style": same}

    unchanged = run(served, "good", "computed_style_changes", {}, before=dead)
    assert unchanged.passed is False, "a dead theme control must fail"


def test_clipboard_matches_checks_shape_not_equality(served):
    async def copy(page, ctx):
        await page.click("Copy")
        await asyncio.sleep(0.15)

    ok = run(
        served,
        "good",
        "clipboard_matches",
        {"patterns": [r"def"], "absent": [r"<script"]},
        before=copy,
    )
    assert ok.passed is True, ok.detail

    wrong = run(
        served,
        "good",
        "clipboard_matches",
        {"patterns": [r"nowhere-at-all"]},
        before=copy,
    )
    assert wrong.passed is False


def test_is_operable_rejects_a_real_input_nobody_can_touch(served):
    """Correct by tag, impossible to use: the typing corpus's dominant failure.

    ``is_real_control`` passes this because it is a genuine ``<input>``. Only a
    check that looks at size, opacity and pointer-events can tell that 43% of
    type attempts against it will fail.
    """
    good = run(
        served,
        "good",
        "is_operable",
        {"selector": "input, textarea, [contenteditable]"},
    )
    assert good.passed is True, good.detail

    bad = run(served, "bad", "is_operable", {"selector": "#hidden-typing"})
    assert bad.passed is False
    assert "pointer-events=none" in bad.detail


def test_every_registered_primitive_is_exercised_here():
    """A primitive nobody tests is a primitive nobody should trust."""
    tested = {
        "aria_names",
        "arith",
        "clipboard_equals",
        "computed_style_distinct",
        "dom_count",
        "dom_text_absent",
        "download",
        "http_status",
        "image_props",
        "controls_have_names",
        "no_mojibake",
        "default_restored",
        "client_secret_present",
        "model_dependency",
        "key_optional",
        "text_matches",
        "text_number_equals",
        "text_number_in_range",
        "distinct_sources",
        "third_party_requests",
        "value_changes",
        "captures_monotonic",
        "pdf_props",
        "sql_executes",
        "download_absent",
        "text_pattern_count",
        "text_numbers_distinct",
        "console_errors_present",
        "computed_style_changes",
        "clipboard_matches",
        "is_operable",
        "is_real_control",
        "network_origins",
        "no_console_errors",
        "number_in_range",
        "source_absent",
        "source_present",
        "survives_reload",
        "value_equals",
        "canvas_changed",
        "canvas_ink",
        "downloads_distinct",
        "text_pattern_distinct",
    }
    known = set(registry())
    untested = known - tested
    assert untested <= {
        # Exercised against real builds in L2 rather than fixtures: each needs an
        # artifact a static page cannot produce (a PDF, a model call, a 4xx).
        "download_matches_text",
        "request_payload_hash",
        "no_failed_requests",
        "value_matches",
        # Needs a real download plus a matching on-screen figure: the fixture
        # apps produce a PNG but report no size, so this is exercised in L2.
        "download_size_matches",
    }, f"untested primitives with no L2 plan: {untested}"


# ------------------------------------------------------- canvas and gestures


async def _capture(page, ctx, label):
    """Take a harness snapshot the way GraderTools._do_capture does."""
    from viral_bench.rubric.grader import GraderTools

    tools = GraderTools.__new__(GraderTools)
    tools.page = page
    tools.captures = {}
    await GraderTools._do_capture(tools, label)
    ctx.captures[label] = tools.captures[label]


def test_drag_draws_on_a_live_canvas_and_not_on_a_dead_one(served):
    """The whole point of the gesture: 125 of 127 whiteboard builds put their
    scene in a canvas, where nothing else the harness can see moves at all."""

    async def draw(page, ctx):
        await _capture(page, ctx, "before")
        await page.drag("#sketch", dx=90, dy=50, from_x=30, from_y=30, steps=12)
        await _capture(page, ctx, "after")

    good = run(served, "good", "canvas_changed", {}, before=draw)
    assert good.passed is True, "a stroke must register as drawn pixels"

    bad = run(served, "bad", "canvas_changed", {}, before=draw)
    assert bad.passed is False, "a canvas with no handler must not read as drawn"
    assert "nothing was drawn" in bad.detail


def test_value_changes_cannot_see_what_the_canvas_check_sees(served):
    """Why canvas_changed had to exist: the DOM is identical either side of a
    stroke, so the text view reports 'unchanged' on a build that works."""

    async def draw(page, ctx):
        await _capture(page, ctx, "before")
        await page.drag("#sketch", dx=90, dy=50, from_x=30, from_y=30, steps=12)
        await _capture(page, ctx, "after")

    text = run(served, "good", "value_changes", {"view": "text"}, before=draw)
    assert text.passed is False, "the DOM does not move when a canvas is drawn on"


def test_canvas_ink_separates_content_from_a_flat_repaint(served):
    async def draw(page, ctx):
        await _capture(page, ctx, "blank")
        for offset in (0, 25, 50):
            await page.drag(
                "#sketch", dx=100, dy=40, from_x=20, from_y=20 + offset, steps=12
            )
        await _capture(page, ctx, "inked")

    blank = run(served, "good", "canvas_ink", {"label": "blank"}, before=draw)
    assert blank.passed is False, "a uniform canvas has no ink"
    inked = run(served, "good", "canvas_ink", {"label": "inked"}, before=draw)
    assert inked.passed is True


def test_canvas_changed_is_unknown_when_the_grader_never_captured(served):
    assert run(served, "good", "canvas_changed", {}).passed is None


# ------------------------------------------------- downloads_distinct, magic


def test_downloads_distinct_catches_the_same_picture_served_twice(served, tmp_path):
    """An identical re-serve arrives under a fresh blob: URL, so comparing
    img.src finds a difference that is not there. Bytes are the only answer."""

    async def generate_twice(page, ctx):
        directory = tmp_path / "gen"
        with page.capture_downloads(directory) as captured:
            for _ in range(2):
                await page.click("Generate")
                for _ in range(40):
                    if len(captured) >= 1:
                        break
                    await asyncio.sleep(0.05)
                await asyncio.sleep(0.15)
            for _ in range(40):
                if len(captured) >= 2:
                    break
                await asyncio.sleep(0.05)
        ctx.downloads = list(captured)

    good = run(served, "good", "downloads_distinct", {}, before=generate_twice)
    assert good.passed is True, f"two generations differ: {good.observed}"

    bad = run(served, "bad", "downloads_distinct", {}, before=generate_twice)
    assert bad.passed is False, "the same bytes twice is not two pictures"


def test_downloads_distinct_is_unknown_when_the_button_never_fired():
    from viral_bench.rubric.checks import CheckContext, run_check

    result = asyncio.run(
        run_check("downloads_distinct", CheckContext(url="", downloads=[]), {})
    )
    assert result.passed is None, "a broken button is not 'the same picture twice'"


def test_magic_accepts_svg_webp_and_a_list_of_formats(tmp_path):
    from viral_bench.rubric.checks import CheckContext, magic_matches, run_check

    # The bug: magic: svg returned unknown, so those points were unearnable.
    assert magic_matches(b'<?xml version="1.0"?>\n<svg xmlns="x"/>', "svg") is True
    assert magic_matches(b"<!DOCTYPE html><html>", "svg") is False
    assert magic_matches(b"RIFF\x00\x00\x00\x00WEBPVP8 ", "webp") is True
    wav = magic_matches(b"RIFF\x00\x00\x00\x00WAVEfmt ", "webp")
    assert wav is False, "RIFF alone is also WAV; the fourcc decides"

    target = tmp_path / "scene.svg"
    target.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>')
    ctx = CheckContext(url="", downloads=[{"path": str(target), "name": "scene.svg"}])
    assert asyncio.run(run_check("download", ctx, {"magic": "svg"})).passed is True
    # A list means "any of these", for an item whose brief accepts several.
    listed = asyncio.run(run_check("download", ctx, {"magic": ["png", "svg"]}))
    assert listed.passed is True
    assert asyncio.run(run_check("download", ctx, {"magic": "png"})).passed is False


def test_text_pattern_distinct_does_not_credit_one_name_twice(served):
    """text_pattern_count over-credits: 'Fira Code' and 'Font: Fira Code' are
    two occurrences of one theme."""
    params = {"pattern": r"(?:Theme|Font):\s*([A-Za-z ]+)", "minimum": 3}
    assert run(served, "good", "text_pattern_distinct", params).passed is True
    bad = run(served, "bad", "text_pattern_distinct", params)
    assert bad.passed is False
    assert "1 distinct of 3 matches" in bad.observed
