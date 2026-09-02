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

"""Tests for the agent-facing toolkit, try_app, and the CAMEL adapter."""

from __future__ import annotations

import asyncio

import pytest

from viral_bench.crowd.interaction import (
    AppInteractionToolkit,
    browser_available,
    try_app,
)

_needs_browser = pytest.mark.skipif(
    not browser_available(), reason="no system browser + playwright available"
)


def _run(coro):
    return asyncio.run(coro)


# -- the tool set ------------------------------------------------------------


def test_tools_are_the_web_set(web_build) -> None:
    """Every app is a web app, so there is exactly one tool set.

    Asserted as equality rather than containment: the terminal/chat verbs
    (``run_command``, ``show_usage``, ``send_message``) went away with the ``cli``
    and ``bot`` app types, and an agent handed a verb it cannot use is a
    measurement bug -- so an accidental re-addition must fail here.
    """
    names = {
        t.__name__ for t in AppInteractionToolkit(web_build, container=False).tools()
    }
    assert names == {
        "open_app",
        "look",
        "click",
        "type_text",
        "select_option",
        "press_key",
        "upload_file",
        "reload_page",
        "screenshot",
        "finish_trial",
    }


def test_full_stack_app_gets_the_same_tools(full_stack_build) -> None:
    """Scope is a label for analysis, not a different tool surface."""
    names = {
        t.__name__
        for t in AppInteractionToolkit(full_stack_build, container=False).tools()
    }
    assert names == {
        "open_app",
        "look",
        "click",
        "type_text",
        "select_option",
        "press_key",
        "upload_file",
        "reload_page",
        "screenshot",
        "finish_trial",
    }


# -- finish_trial + verdict -------------------------------------------------


def test_finish_trial_records_verdict(web_build) -> None:
    async def inner():
        toolkit = AppInteractionToolkit(web_build, container=False)
        try:
            # An agent that never opened the app is pushed back once, and the
            # second call always goes through (see the never-opened guard).
            first = await toolkit.finish_trial(True, True, 9, "loved it")
            assert "have not opened this app" in first
            assert toolkit.trace.verdict is None
            msg = await toolkit.finish_trial(True, True, 9, "loved it")
            assert "delight=9" in msg
            assert toolkit.trace.verdict is not None
            assert toolkit.trace.verdict.would_use is True
            assert toolkit.trace.verdict.delight == 9
        finally:
            await toolkit.close()

    _run(inner())


def test_finish_trial_is_idempotent_first_verdict_wins(web_build) -> None:
    # An agent re-activated in a later round often re-calls finish_trial with a
    # contentless "already did this" note. That must NOT overwrite the real,
    # evidence-grounded verdict (it was even flipping would_share).
    async def inner():
        toolkit = AppInteractionToolkit(web_build, container=False)
        try:
            await toolkit.finish_trial(True, True, 8, "grounded first verdict")
            await toolkit.finish_trial(True, True, 8, "grounded first verdict")
            msg = await toolkit.finish_trial(False, False, 2, "already did this")
            assert "already recorded" in msg.lower()
            v = toolkit.trace.verdict
            assert v.would_use is True and v.would_share is True
            assert v.delight == 8
            assert v.notes == "grounded first verdict"
            # exactly one finish step in the trace
            assert sum(s.action == "finish" for s in toolkit.trace.steps) == 1
        finally:
            await toolkit.close()

    _run(inner())


# -- step budget ------------------------------------------------------------


def test_max_steps_budget_stops_further_actions(web_build) -> None:
    async def inner():
        # Opening the trial already records the first observation, so a budget of
        # 2 leaves room for exactly one further action. Asserted on the budget
        # notice rather than on the click's own result so this holds whether the
        # trial got a real browser or the static fallback.
        toolkit = AppInteractionToolkit(web_build, container=False, max_steps=2)
        try:
            first = await toolkit.click("Increment")
            assert "budget" not in first.lower()
            second = await toolkit.click("Increment")
            assert "budget" in second.lower()
        finally:
            await toolkit.close()

    _run(inner())


# -- try_app: scripted + default -------------------------------------------


def test_run_script_unknown_action_is_recorded_failed(web_build) -> None:
    async def inner():
        toolkit = AppInteractionToolkit(web_build, container=False)
        try:
            await toolkit.run_script([{"action": "no_such_tool"}])
            assert any(
                s.action == "no_such_tool" and not s.ok for s in toolkit.trace.steps
            )
        finally:
            await toolkit.close()

    _run(inner())


@_needs_browser
def test_try_app_web_default_interacts(web_build) -> None:
    trace = _run(try_app(web_build, container=False))
    assert trace.app_type == "client-app"
    assert trace.degraded is False
    assert any(s.action == "click" and s.ok for s in trace.steps)
    assert trace.had_effect() is True


@_needs_browser
def test_try_app_web_scripted(web_build) -> None:
    trace = _run(
        try_app(
            web_build,
            container=False,
            script=[
                {"action": "click", "target": "Increment"},
                {"action": "click", "target": "Increment"},
                {"action": "look"},
            ],
        )
    )
    assert any("Count: 2" in s.summary for s in trace.steps)


# -- CAMEL adapter ----------------------------------------------------------


def test_as_camel_tools_wraps_each_tool(web_build) -> None:
    pytest.importorskip("camel.toolkits", reason="camel-ai not installed (crowd extra)")
    toolkit = AppInteractionToolkit(web_build, container=False)
    camel_tools = toolkit.as_camel_tools()
    assert len(camel_tools) == len(toolkit.tools())


# -- craft facets are one rubric for the whole fleet ------------------------


def test_facet_definitions_are_one_rubric_for_every_app(web_build, full_stack_build):
    """Craft must be one quantity across the bench, and now it is one rubric.

    This assertion inverted with the pivot. The facet guide used to be
    per-app-type precisely because a web-shaped wording ("how it looks and
    feels") read as inapplicable to a CLI, and an LLM asked to rate an
    inapplicable facet omitted the whole optional set: 0 of 25 stored cli+bot
    trials rated any facet against 259 of 293 web trials. With every idea now a
    web app that hazard is gone, so the honest thing is a single definition --
    and this test now guards the opposite property, that ``app_type`` cannot
    change the rubric and silently make two builds' craft incomparable.
    """
    from viral_bench.crowd.interaction.toolkit import finish_trial_doc

    client_doc = finish_trial_doc("client-app")
    full_stack_doc = finish_trial_doc("full-stack-app")

    assert client_doc == full_stack_doc
    assert "Visual and interaction craft" in client_doc
    for doc in (client_doc, full_stack_doc):
        assert "ALL FOUR facet scores are REQUIRED" in doc
        for facet in ("functionality", "usability", "design", "simplicity"):
            assert f"{facet}:" in doc

    # The doc an agent reads off the exported tool is that same rubric.
    for build in (web_build, full_stack_build):
        toolkit = AppInteractionToolkit(build, container=False)
        finish = next(t for t in toolkit.tools() if t.__name__ == "finish_trial")
        assert (finish.__doc__ or "") == client_doc


def test_facet_tool_wrapper_still_records_through_the_real_method(web_build):
    """The exported wrapper must change only the description, not the data."""

    async def go() -> None:
        toolkit = AppInteractionToolkit(web_build, container=False)
        finish = next(t for t in toolkit.tools() if t.__name__ == "finish_trial")
        await finish(True, False, 7, "solid", 8, 7, 6, 9)  # refused: never opened
        await finish(True, False, 7, "solid", 8, 7, 6, 9)
        verdict = toolkit.trace.verdict
        assert verdict is not None
        assert (verdict.functionality, verdict.usability) == (8, 7)
        assert (verdict.design, verdict.simplicity) == (6, 9)
        assert verdict.craft == pytest.approx((8 + 7 + 6 + 9 + 7) / 5)

    asyncio.run(go())


# -- run-command parsing -----------------------------------------------------


def test_finish_trial_tells_an_agent_it_never_got_the_app_working(web_build):
    async def go():
        toolkit = AppInteractionToolkit(web_build, container=False)
        toolkit._trace.app_reachable = False
        await toolkit.finish_trial(True, True, 9, "loved it", 9, 9, 9, 9)
        msg = await toolkit.finish_trial(True, True, 9, "loved it", 9, 9, 9, 9)
        assert "never got this app to actually work" in msg
        # The verdict is still recorded -- the crowd is marked, not censored.
        assert toolkit.trace.verdict is not None
        assert toolkit.trace.verdict.would_use is True

    asyncio.run(go())


# -- undeliverable builds are simulated, not skipped -------------------------


def test_undeliverable_build_is_never_reachable_and_says_why(undeliverable_build):
    """A build with no runnable manifest still gets a trial, and the trial fails.

    These used to be dropped from the crowd stage entirely, so a model that
    failed to ship a launch contract vanished from the denominator instead of
    being marked down -- while the other model's bad-but-runnable apps stayed in
    and lowered its average.
    """

    from viral_bench.crowd.interaction.clients import UNDELIVERABLE_NOTICE

    build_id = undeliverable_build

    async def go() -> None:
        toolkit = AppInteractionToolkit(
            build_id, container=False, app_type="client-app"
        )
        assert toolkit.undeliverable is True
        out = await toolkit.open_app()
        assert UNDELIVERABLE_NOTICE in out
        assert toolkit.trace.app_reachable is False
        # Every surface reports the same thing rather than raising.
        assert UNDELIVERABLE_NOTICE in await toolkit.look()
        assert UNDELIVERABLE_NOTICE in await toolkit.click("Increment")
        # The verdict is still recorded -- the crowd is marked, not censored.
        await toolkit.finish_trial(False, False, 1, "could not run it")
        assert toolkit.trace.verdict is not None
        assert toolkit.trace.app_reachable is False

    asyncio.run(go())


def test_invalid_manifest_is_undeliverable_too(invalid_manifest_build):
    """Three of the seven real failures were an unescaped quote in a JSON string."""
    build_id = invalid_manifest_build

    async def go() -> None:
        toolkit = AppInteractionToolkit(
            build_id, container=False, app_type="client-app"
        )
        assert toolkit.undeliverable is True
        assert toolkit.trace.app_reachable is not True

    asyncio.run(go())


def _toolkit_with_trace(
    *, reachable: bool, steps: list[tuple[str, str]], min_interactions: int = 1
):
    """A toolkit whose trace already holds the given (action, summary) steps."""
    from viral_bench.crowd.interaction.toolkit import AppInteractionToolkit
    from viral_bench.crowd.interaction.trace import InteractionTrace

    trace = InteractionTrace(build_id="b", app_type="client-app")
    trace.app_reachable = reachable
    for action, summary in steps:
        trace.record(action=action, args={}, summary=summary)
    tk = AppInteractionToolkit.__new__(AppInteractionToolkit)
    tk._trace = trace
    tk._finished = False
    tk._nudged = False
    tk._locked = False
    tk._min_interactions = min_interactions
    tk._disabled = frozenset()
    return tk


def test_a_trier_that_only_read_the_page_has_not_interacted() -> None:
    """Looking is not using.

    Measured over 792 solo-fleet trials, 284 triers (35.9%) recorded a verdict
    without one successful click, type, upload or keypress -- 218 never even
    attempted one. Their verdicts ran delight 5.94 / would_use 0.55 against
    7.05 / 0.81 for triers who interacted 3-5 times, so a large slice of the
    hands-on signal was a reading-comprehension signal.
    """
    tk = _toolkit_with_trace(
        reachable=True,
        steps=[("open", "opened"), ("look", "saw stuff"), ("screenshot", "shot")],
    )
    assert tk._did_interact() is False

    tk = _toolkit_with_trace(
        reachable=True, steps=[("open", "opened"), ("click", "Clicked 'Start'.")]
    )
    assert tk._did_interact() is True


def test_a_failed_interaction_does_not_count_as_using_the_app() -> None:
    """A click that timed out taught the agent nothing about the product."""
    tk = _toolkit_with_trace(
        reachable=True,
        steps=[("click", "click failed: could not click 'Sign In': Timeout")],
    )
    assert tk._did_interact() is False


def test_unreachable_app_is_never_nudged() -> None:
    """An app that will not load cannot be operated.

    "I tried and it was broken" is a real verdict and must still be recordable
    on the first call, so the nudge is gated on reachability.
    """
    tk = _toolkit_with_trace(reachable=False, steps=[("open", "connection refused")])
    assert tk._app_is_reachable() is False


# -- the never-opened guard --------------------------------------------------


def test_a_verdict_filed_without_opening_the_app_is_pushed_back_once(web_build):
    """Reading the source is not using the app, and it never was.

    32 of 352 stored trials went straight from being handed the tools to filing
    a verdict, with confident craft ratings reconstructed from the source tree
    ("the SQLite persistence works robustly") for apps that would not start.
    The previous guard could not catch them: it only fired when the app was
    reachable, which is exactly what a trial that opened nothing is not.
    """

    async def go():
        toolkit = AppInteractionToolkit(web_build, container=False)
        first = await toolkit.finish_trial(False, False, 2, "looks broken")
        assert "have not opened this app" in first
        assert toolkit.trace.verdict is None
        # The push-back happens once: "I could not get it to start" must stay
        # recordable, or the crowd cannot report a dead app at all.
        second = await toolkit.finish_trial(False, False, 2, "could not start it")
        assert toolkit.trace.verdict is not None
        assert toolkit.trace.verdict.delight == 2
        assert "have not opened this app" not in second

    asyncio.run(go())


@_needs_browser
def test_reload_reports_whether_the_work_survived(web_build):
    """The verb that makes persistence observable at all."""

    async def go():
        toolkit = AppInteractionToolkit(web_build, container=False)
        try:
            await toolkit.open_app()
            out = await toolkit.reload_page()
            assert "really saved" in out or "Reloaded" in out
            assert any(s.action == "reload" for s in toolkit.trace.steps)
        finally:
            await toolkit.close()

    asyncio.run(go())


def test_the_interaction_floor_is_a_count_not_a_boolean():
    """One click satisfied the guard, and one click is what agents did.

    On client-side apps 41% of trials perform exactly ONE substantive action and
    the median is 2, against five steps of ceremony. Whether that is enough is an
    empirical question, so the floor has to be a number that can be moved and
    measured rather than a hard-coded "at least one".
    """
    steps = [("open", "opened"), ("click", "Clicked 'Start'."), ("look", "saw")]
    assert _toolkit_with_trace(
        reachable=True, steps=steps, min_interactions=1
    )._did_interact()
    assert not _toolkit_with_trace(
        reachable=True, steps=steps, min_interactions=3
    )._did_interact()
    deep = steps + [("type", "Typed 'x'."), ("select", "Selected 'y'.")]
    assert _toolkit_with_trace(
        reachable=True, steps=deep, min_interactions=3
    )._did_interact()


def test_a_withheld_tool_is_not_offered_and_finish_trial_always_is(web_build):
    """Removing a verb and re-measuring is the only way to price it."""
    from viral_bench.crowd.interaction.toolkit import AppInteractionToolkit

    tk = AppInteractionToolkit(
        web_build, container=False, disabled_tools=("reload_page", "screenshot")
    )
    names = {t.__name__ for t in tk.tools()}
    assert "reload_page" not in names and "screenshot" not in names
    assert "finish_trial" in names and "open_app" in names
