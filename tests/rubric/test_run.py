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

"""Tests for the Tier 0 gate and the end-to-end pipeline.

The gate is the most consequential code in the track: it decides whether a build
scores 0 or gets graded at all, and ~27% of the corpus lands on it. So the cases
here are the ones where being wrong is expensive in a specific direction --
zeroing a build that actually works, or grading features on an app that never
started.

Everything runs offline. The app host, browser and model are all injected fakes,
which is the point of them being injectable.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading

import pytest

from viral_bench.rubric.checks import CheckContext, CheckResult, check, registry
from viral_bench.rubric.grader import ItemOutcome
from viral_bench.rubric.run import (
    GradeAborted,
    check_manifest_strict,
    deterministic_pre_pass,
    grade_build,
    run_gate,
)
from viral_bench.rubric.schema import Check, Rubric, RubricItem

MANIFEST = {
    "app_type": "client-app",
    "title": "Demo",
    "summary": "A demo app.",
    "run": {"command": "python -m http.server 8000", "port": 8000},
}

#: The grader model these tests name. ``grade_build`` requires one and has no
#: default, but every test here injects a fake client, so the id only has to
#: resolve to a registered provider -- it is never called, and naming a real
#: model would suggest the suite depends on that model existing.
GRADER_MODEL = "openai/gpt-test"


class FakeApp:
    def __init__(self, url="http://localhost:8000", ready="HTTP 200"):
        self.url = url
        self.ready_detail = ready


class FakeHost:
    """Stands in for AppHost: either yields an app or refuses to start one."""

    def __init__(self, app=None, error=None):
        self.app = app if app is not None else FakeApp()
        self.error = error
        self.stopped = []

    def get(self, build_id, *, wait_timeout=90.0):
        if self.error is not None:
            raise self.error
        return self.app

    def stop(self, build_id):
        self.stopped.append(build_id)


class FakeElement:
    def __init__(self, tag="button", label="Save"):
        self.tag = tag
        self.label = label

    def describe(self):
        return f'{self.tag} "{self.label}"'


class FakeSnapshot:
    def __init__(self, text="Hello", elements=(), errors=()):
        self.url = "http://localhost:8000/"
        self.title = "Demo"
        self.text = text
        self.elements = list(elements)
        self.console_errors = list(errors)


class FakePage:
    def __init__(self, snapshot=None):
        self._snapshot = snapshot or FakeSnapshot(elements=[FakeElement()])
        self.closed = False

    def requests(self, *, since=0):
        return []

    async def snapshot(self):
        return self._snapshot

    async def goto(self, url):
        pass

    async def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self, page=None):
        self.page = page or FakePage()

    async def open_page(self, *, permissions=None):
        return self.page

    async def close(self):
        pass


@pytest.fixture
def app_dir(tmp_path):
    directory = tmp_path / "app"
    directory.mkdir()
    (directory / "viralbench.json").write_text(json.dumps(MANIFEST))
    return directory


@pytest.fixture(scope="module")
def served():
    """A real HTTP server, because G4 makes a real request.

    Faking the fetch would test the fake. The gate's whole job is to find out
    whether something is actually listening, so the test has to give it
    something that actually listens.
    """

    class Quiet(http.server.BaseHTTPRequestHandler):
        status = 200

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's interface
            self.send_response(type(self).status)
            self.end_headers()
            self.wfile.write(b"<h1>Demo</h1>")

        def log_message(self, *args):
            """Silenced: logging from this thread deadlocks pytest's capture."""

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", Quiet
    httpd.shutdown()


def gate(app_dir, *, host=None, engine=None, url=None):
    if host is None:
        host = FakeHost(app=FakeApp(url=url)) if url else FakeHost()
    return asyncio.run(
        run_gate(
            "b1",
            host=host,
            engine=engine or FakeEngine(),
            app_dir=app_dir,
        )
    )


# ------------------------------------------------------------------ G1


def test_a_missing_manifest_fails_g1(tmp_path):
    outcome = check_manifest_strict(tmp_path)
    assert outcome.passed is False
    assert "no viralbench.json" in outcome.reason


def test_a_leading_comment_fails_g1_strictly(tmp_path):
    """The documented, recurring cause of a complete build scoring zero."""
    (tmp_path / "viralbench.json").write_text("# how to run\n" + json.dumps(MANIFEST))
    assert check_manifest_strict(tmp_path).passed is False


def test_a_trailing_comma_fails_g1_strictly(tmp_path):
    (tmp_path / "viralbench.json").write_text('{"app_type": "client-app",}')
    assert check_manifest_strict(tmp_path).passed is False


def test_a_manifest_missing_run_fails_g1(tmp_path):
    (tmp_path / "viralbench.json").write_text(json.dumps({"app_type": "client-app"}))
    outcome = check_manifest_strict(tmp_path)
    assert outcome.passed is False
    assert "manifest invalid" in outcome.reason


def test_a_valid_manifest_passes_g1(app_dir):
    assert check_manifest_strict(app_dir).passed is True


def test_a_g1_failure_leaves_the_rest_unresolved_not_failed(tmp_path):
    """A manifest typo is one defect, not five."""
    result = gate(tmp_path)
    assert result.passed is False
    assert result.outcomes["G1"].passed is False
    for item_id in ("G2", "G3", "G4", "G5"):
        assert result.outcomes[item_id].passed is None


# ---------------------------------------------------------------- G2/G3


def test_a_build_that_never_starts_fails_g3(app_dir):
    from viral_bench.founder.apphost import AppStartFailed

    result = gate(app_dir, host=FakeHost(error=AppStartFailed("port never bound")))
    assert result.passed is False
    assert result.outcomes["G3"].passed is False
    assert result.outcomes["G2"].passed is True  # setup itself was fine
    assert result.outcomes["G4"].passed is None


def test_a_setup_death_is_attributed_to_g2(app_dir):
    from viral_bench.founder.apphost import AppStartFailed

    result = gate(app_dir, host=FakeHost(error=AppStartFailed("setup: tsc: not found")))
    assert result.outcomes["G2"].passed is False
    assert "tsc: not found" in result.outcomes["G2"].reason


def test_an_app_that_binds_no_port_fails_g3(app_dir):
    result = gate(app_dir, host=FakeHost(app=FakeApp(url="", ready="no port")))
    assert result.outcomes["G3"].passed is False
    assert result.passed is False


def test_an_unexpected_start_error_does_not_crash_the_grade(app_dir):
    result = gate(app_dir, host=FakeHost(error=OSError("podman is not running")))
    assert result.passed is False
    assert result.outcomes["G3"].passed is False


# ------------------------------------------------------------------ G5


def test_an_uncaught_page_error_fails_g5(app_dir, served):
    """The dead-on-arrival class: a template compile error that renders nothing."""
    url, _ = served
    page = FakePage(
        FakeSnapshot(
            text="",
            elements=[FakeElement()],
            errors=["pageerror: Invalid or unexpected token"],
        )
    )
    result = gate(app_dir, engine=FakeEngine(page), url=url)
    assert result.outcomes["G5"].passed is False
    assert "uncaught error" in result.outcomes["G5"].reason
    assert result.passed is False


def test_a_page_with_no_operable_control_fails_g5(app_dir, served):
    url, _ = served
    page = FakePage(FakeSnapshot(text="Coming soon", elements=[]))
    result = gate(app_dir, engine=FakeEngine(page), url=url)
    assert result.outcomes["G5"].passed is False


def test_a_rendering_page_passes_the_whole_gate(app_dir, served):
    url, _ = served
    result = gate(app_dir, url=url)
    assert result.passed is True
    assert all(outcome.passed for outcome in result.outcomes.values())


def test_the_gate_closes_the_page_it_opened(app_dir, served):
    url, _ = served
    page = FakePage()
    gate(app_dir, engine=FakeEngine(page), url=url)
    assert page.closed is True


# ------------------------------------------------------------------ G4


def test_a_url_that_refuses_the_connection_fails_g4_definitely(app_dir):
    """Not "could not tell": nothing was listening, and that is knowable."""
    result = gate(app_dir, url="http://127.0.0.1:1")
    assert result.outcomes["G4"].passed is False
    assert result.outcomes["G5"].passed is None  # never reached
    assert result.passed is False


def test_a_5xx_entry_page_fails_g4(app_dir, served):
    url, handler = served
    handler.status = 500
    try:
        result = gate(app_dir, url=url)
    finally:
        handler.status = 200
    assert result.outcomes["G4"].passed is False
    assert result.passed is False


def test_a_4xx_entry_page_is_legitimate_for_an_auth_first_app(app_dir, served):
    """A login wall is a design, not a defect. Only a 5xx is never legitimate."""
    url, handler = served
    handler.status = 401
    try:
        result = gate(app_dir, url=url)
    finally:
        handler.status = 200
    assert result.outcomes["G4"].passed is True
    assert result.passed is True


# ------------------------------------------------------- deterministic pre-pass


@pytest.fixture
def counting_check():
    """A primitive that records how often it ran."""
    name = "_test_counting"
    saved = registry()
    calls = []

    @check(name)
    def _fn(ctx, **params):
        calls.append(params)
        return CheckResult.yes(observed="seen")

    yield name, calls
    from viral_bench.rubric import checks as checks_module

    checks_module._REGISTRY.clear()
    checks_module._REGISTRY.update(saved)


def item(item_id, points, tier, *, check_block=None, method="assert"):
    return RubricItem(
        id=item_id,
        text=f"{item_id} claim",
        points=points,
        method=method,
        tier=tier,
        check=check_block,
    )


def rubric_with(*items):
    return Rubric(
        idea_id="demo",
        rubric_version="1",
        tier1=tuple(items),
        tier2=(),
        tier3=(),
        gate=(),
        penalties=(),
    )


def test_the_pre_pass_settles_source_items_before_any_navigation():
    rubric = rubric_with(
        item("S1", 10, 1, check_block=Check("source_absent", {"pattern": "TODO"})),
    )
    context = CheckContext(url="http://localhost:8000", source=None)
    outcomes = asyncio.run(deterministic_pre_pass(rubric, context))
    assert "S1" in outcomes


def test_the_pre_pass_skips_items_that_need_the_app_in_a_state(counting_check):
    """Running these early would score them against an untouched page."""
    name, calls = counting_check
    rubric = rubric_with(item("S1", 10, 1, check_block=Check(name)))
    context = CheckContext(url="http://localhost:8000")
    outcomes = asyncio.run(deterministic_pre_pass(rubric, context))
    assert outcomes == {}
    assert calls == []


def test_the_pre_pass_ignores_items_with_no_check():
    rubric = rubric_with(item("S1", 10, 1, method="agent"))
    context = CheckContext(url="http://localhost:8000")
    assert asyncio.run(deterministic_pre_pass(rubric, context)) == {}


# ------------------------------------------------------------- gate scoring


def test_a_failed_gate_zeroes_the_score_but_keeps_the_items():
    """Undeliverable builds are scored, not skipped -- otherwise the benchmark
    reports the average of the survivors and calls it the average."""
    from viral_bench.rubric.grader import merge_passes
    from viral_bench.rubric.score import score_rubric

    rubric = Rubric(
        idea_id="demo",
        rubric_version="1",
        tier1=(item("S1", 40, 1),),
        tier2=(item("F1", 25, 2),),
        tier3=(item("R1", 35, 3),),
        gate=(item("G1", 0, 0, method="probe"),),
        penalties=(),
    )
    verdicts = merge_passes([{"G1": ItemOutcome("G1", False, "no viralbench.json")}])
    result = score_rubric(rubric, verdicts, build_id="b1", passes=1, error="gate")
    assert result.score == 0.0
    assert result.gate_zeroed is True
    assert result.gate_failures == ["G1"]
    # The graded items are still enumerated, so the failure can be diagnosed.
    assert result.points_applicable == 100


# --------------------------------------------------------- end to end


def make_build(root, build_id="b1", idea_id="ai_room_redesign", manifest=None):
    """A minimal on-disk build: a record and an app dir with a manifest."""
    work = root / "work" / build_id
    (work / "app").mkdir(parents=True)
    (work / "app" / "viralbench.json").write_text(
        json.dumps(manifest if manifest is not None else MANIFEST)
    )
    (work / "app" / "index.html").write_text("<h1>Demo</h1>")
    (work / "build.json").write_text(
        json.dumps(
            {
                "idea_id": idea_id,
                "structure": "solo",
                "collab": "local",
                "model": "publishers/anthropic/claude-test@default",
                "status": "shipped",
                "brief_fingerprint": "era-3",
                "manifest": {"title": "Demo"},
            }
        )
    )
    return build_id


class SilentClient:
    """A model that reports `unknown` for everything, spending nothing.

    Enough to prove the wiring: the pipeline has to walk every item, run the
    code-judged ones, score and persist, without a real model in the loop.
    """

    def __init__(self):
        self.calls = 0

    def generate(self, messages, *, tools=None, system=""):
        self.calls += 1

        class Reply:
            text = ""
            stop_reason = "stop"
            usage = {}
            tool_calls = [
                type(
                    "C",
                    (),
                    {"id": "c1", "name": "report", "args": {"verdict": "unknown"}},
                )()
            ]

        return Reply()


def test_grade_build_runs_end_to_end_and_persists(tmp_path, served):
    url, _ = served
    build_id = make_build(tmp_path)
    client = SilentClient()
    result, document = asyncio.run(
        grade_build(
            build_id,
            grader_model=GRADER_MODEL,
            client=client,
            host=FakeHost(app=FakeApp(url=url)),
            engine=FakeEngine(),
            root=tmp_path,
            passes=1,
        )
    )
    assert result.gate_zeroed is False
    assert document["kind"] == "viral_bench.rubric_grade"
    assert document["build_id"] == build_id
    assert document["idea_id"] == "ai_room_redesign"
    # Persisted where the viewer will look for it.
    written = list((tmp_path / "rubric").glob(f"{build_id}__rubric-*/grade.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["score"] == result.score


def test_grade_build_zeroes_an_undeliverable_build_without_calling_the_model(tmp_path):
    """The whole point of gating first: no model spend on an app that cannot run."""
    build_id = make_build(tmp_path, manifest={"nope": True})
    client = SilentClient()
    result, document = asyncio.run(
        grade_build(
            build_id,
            grader_model=GRADER_MODEL,
            client=client,
            host=FakeHost(),
            engine=FakeEngine(),
            root=tmp_path,
            passes=3,
        )
    )
    assert result.score == 0.0
    assert result.gate_zeroed is True
    assert client.calls == 0
    assert document["gate"]["passed"] is False


def test_grade_build_refuses_a_build_it_cannot_resolve(tmp_path):
    with pytest.raises(GradeAborted):
        asyncio.run(
            grade_build(
                "nope",
                grader_model=GRADER_MODEL,
                host=FakeHost(),
                engine=FakeEngine(),
                root=tmp_path,
            )
        )


def test_grade_build_refuses_to_grade_with_no_model_named(tmp_path):
    """There is no default grader. Whoever grades a build has to say what did.

    The model id is written into the grade document and is the only record of
    which judge produced the number, so a default would silently attribute a
    corpus to whatever was hardcoded when it was graded.
    """
    build_id = make_build(tmp_path)
    with pytest.raises(GradeAborted, match="explicit grader model"):
        asyncio.run(
            grade_build(
                build_id,
                grader_model="",
                client=SilentClient(),
                host=FakeHost(),
                engine=FakeEngine(),
                root=tmp_path,
            )
        )


def test_grade_build_refuses_a_provider_that_cannot_see(tmp_path):
    """Up front, before the container starts -- not 40 minutes into the grade.

    Design is a scored dimension and the grader takes screenshots to judge it, so
    a text-only provider does not degrade gracefully: it produces a build's worth
    of items marked unresolved for a reason nothing in the grade explains.
    """
    from viral_bench.providers import UnsupportedCapabilityError

    build_id = make_build(tmp_path)
    with pytest.raises(UnsupportedCapabilityError, match="rubric grader"):
        asyncio.run(
            grade_build(
                build_id,
                # Registered, tool-capable, and cannot take an image.
                grader_model="ollama/llava-test",
                client=SilentClient(),
                host=FakeHost(),
                engine=FakeEngine(),
                root=tmp_path,
            )
        )


def test_grade_build_resets_only_its_own_root(tmp_path, served):
    """Passing `root=` must not reach the real builds/data tree."""
    url, _ = served
    build_id = make_build(tmp_path)
    stale = tmp_path / "data" / build_id
    stale.mkdir(parents=True)
    (stale / "old.sqlite").write_text("previous pass")
    asyncio.run(
        grade_build(
            build_id,
            grader_model=GRADER_MODEL,
            client=SilentClient(),
            host=FakeHost(app=FakeApp(url=url)),
            engine=FakeEngine(),
            root=tmp_path,
            passes=1,
            write=False,
        )
    )
    assert stale.is_dir()
    assert not (stale / "old.sqlite").exists()


def test_grade_build_stops_the_app_it_started(tmp_path, served):
    url, _ = served
    build_id = make_build(tmp_path)
    host = FakeHost(app=FakeApp(url=url))
    asyncio.run(
        grade_build(
            build_id,
            grader_model=GRADER_MODEL,
            client=SilentClient(),
            host=host,
            engine=FakeEngine(),
            root=tmp_path,
            passes=1,
            write=False,
        )
    )
    assert host.stopped == [build_id]


def test_source_toolkit_resolves_the_app_dir_it_was_given(tmp_path):
    """Regression: the toolkit used to be handed the app PATH as the build id.

    ``CodeInspectionToolkit`` resolves its tree from ``load_build_record(build_id)``
    unless ``app_dir`` is passed, so the old call looked for
    ``builds/work/<path>/build.json``, raised, and was swallowed. Every `source`
    item on every build reported "no source tree" -- including the universal P4
    secret-in-client-assets penalty, which could therefore never fire.
    """
    from viral_bench.rubric.run import _source_toolkit

    app_dir = tmp_path / "work" / "demo__20260101-000000__abc123" / "app"
    app_dir.mkdir(parents=True)
    (app_dir / "app.js").write_text("const KEY = 'AIzaSyTOTALLYNOTAREALKEYVALUE00';")

    toolkit = _source_toolkit(app_dir, "demo__20260101-000000__abc123")
    assert toolkit is not None, "a readable app dir must yield a toolkit"
    assert "app.js" in toolkit.list_files()
    assert "AIzaSy" in toolkit.grep("AIza")


def test_source_toolkit_reads_past_the_crowd_20kb_cap(tmp_path):
    """The grader raises the caps; the crowd's would truncate the answer.

    36 of 39 handdrawn_whiteboard builds ship a client file over 20 KB, so a
    claim about the whole tree made against the first 20 KB is a claim about a
    fifth of the average bundle.
    """
    from viral_bench.rubric.run import _source_toolkit

    app_dir = tmp_path / "work" / "demo__20260101-000000__abc123" / "app"
    app_dir.mkdir(parents=True)
    padding = "// filler\n" * 4000  # ~40 KB before the interesting line
    (app_dir / "app.js").write_text(padding + "canvas.addEventListener('mousedown')")

    toolkit = _source_toolkit(app_dir, "demo__20260101-000000__abc123")
    assert "mousedown" in toolkit.grep("mousedown"), "handler past 20 KB must be seen"


def test_the_rubric_client_is_a_shim_over_the_provider_layer():
    """One transport, not two -- and no default grader model hiding in it.

    ``rubric.client`` used to own a Claude client and a Gemini one. They are the
    provider layer's adapters now, and this module is kept only so existing
    imports resolve; a second copy drifting back is the failure worth catching.
    """
    from viral_bench import providers
    from viral_bench.rubric import client as shim

    assert shim.Reply is providers.Reply
    assert shim.ToolCall is providers.ToolCall
    assert shim.ModelError is providers.ModelError
    assert shim.make_client is providers.make_client
    assert not hasattr(shim, "DEFAULT_GRADER_MODEL")


def test_a_read_timeout_is_retried_not_fatal(monkeypatch):
    """Regression: a socket read timeout used to abort the whole build.

    ``urlopen`` wraps a *connect* timeout in ``URLError``, which the retry loop
    already handled. A timeout while reading the response body raises a bare
    ``TimeoutError`` out of ``ssl.read`` instead, which bypassed the loop
    entirely -- spending a ~50-minute grade on a transient hiccup. Under a
    concurrent sweep it is not rare: it lost one build of the first twelve.

    The transport now lives in :mod:`viral_bench.providers`, so this asserts it
    through the client the grader actually builds -- ``rubric.client.make_client``
    on the Vertex Claude path, which is the same request this test always made.
    """
    import json as _json
    from contextlib import contextmanager

    from viral_bench.founder import vertex as vertex_module
    from viral_bench.providers import anthropic as anthropic_module
    from viral_bench.providers import client as provider_client
    from viral_bench.rubric.client import make_client

    monkeypatch.setattr(provider_client.time, "sleep", lambda _: None)
    monkeypatch.setattr(vertex_module, "vertex_project", lambda: "p")
    monkeypatch.setattr(vertex_module, "vertex_location", lambda: "global")
    monkeypatch.setattr(vertex_module, "vertex_api_host", lambda _loc: "host")
    monkeypatch.setattr(vertex_module, "vertex_access_token", lambda: "t")

    calls = {"n": 0}
    body = _json.dumps({"content": [{"type": "text", "text": "recovered"}]})

    class _Response:
        def read(self):
            return body.encode()

    @contextmanager
    def fake_urlopen(_request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("The read operation timed out")
        yield _Response()

    monkeypatch.setattr(anthropic_module.urllib.request, "urlopen", fake_urlopen)

    client = make_client("google-vertex-anthropic/claude-test")
    reply = client.generate([{"role": "user", "content": "hi"}])
    assert reply.text == "recovered"
    assert calls["n"] == 2, "the timeout must cost one retry, not the whole build"
