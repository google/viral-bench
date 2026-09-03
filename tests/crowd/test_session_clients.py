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

"""Integration tests: open_trial + the web clients against real running apps."""

from __future__ import annotations

import asyncio

import pytest

from viral_bench.crowd.interaction import browser_available, open_trial
from viral_bench.crowd.interaction.clients import (
    StaticWebAppClient,
    WebAppClient,
)

_needs_browser = pytest.mark.skipif(
    not browser_available(), reason="no system browser + playwright available"
)


def _run(coro):
    return asyncio.run(coro)


# -- credentials notice -----------------------------------------------------


def test_no_live_feature_notice_without_credentials(monkeypatch) -> None:
    # No injected credentials -> no misleading notice (offline IS the app).
    from viral_bench.crowd.interaction.clients import live_feature_notice

    monkeypatch.setattr(
        "viral_bench.founder.appenv.resolve_app_env", lambda *a, **k: {}
    )
    assert live_feature_notice() == ""


def test_the_notice_names_whatever_the_runtime_actually_injected(monkeypatch) -> None:
    """The names are read from the environment, never spelled out here.

    The notice exists to stop a trier reviewing an app's mock mode when its real
    feature is live. It used to name one vendor's key, so against any other
    provider it would have named a variable the app does not read -- and a trier
    told to "test the real feature" with the wrong variable named is worse off
    than one told nothing.
    """
    from viral_bench.crowd.interaction.clients import live_feature_notice
    from viral_bench.founder.appenv import APP_API_KEY_VAR, APP_BASE_URL_VAR

    monkeypatch.setattr(
        "viral_bench.founder.appenv.resolve_app_env",
        lambda *a, **k: {APP_BASE_URL_VAR: "http://llm.test/v1", APP_API_KEY_VAR: "k"},
    )
    notice = live_feature_notice()
    assert APP_BASE_URL_VAR in notice
    assert APP_API_KEY_VAR in notice
    assert "REAL feature" in notice


# -- Web (real browser) -----------------------------------------------------


@_needs_browser
def test_web_trial_click_mutates_dom(web_build) -> None:
    async def inner():
        client = await open_trial(web_build, container=False)
        assert isinstance(client, WebAppClient)
        try:
            await client.open()
            after = await client.click("Increment")
            assert after.ok
            assert "Count: 1" in after.text
            assert after.degraded is False
        finally:
            await client.close()

    _run(inner())


@_needs_browser
def test_web_trial_type_and_press(web_build) -> None:
    async def inner():
        client = await open_trial(web_build, container=False)
        try:
            await client.open()
            await client.type_text("your name", "Koa")
            greeted = await client.click("Greet")
            assert "Hello Koa!" in greeted.text
            pressed = await client.press_key("ArrowUp")
            assert "Count: 1" in pressed.text
        finally:
            await client.close()

    _run(inner())


@_needs_browser
def test_full_stack_app_is_opened_the_same_way(full_stack_build) -> None:
    """Both scopes are web apps, so both get the same client and the same verbs.

    ``open_trial`` used to branch on app type. Now the scope is only a label, so
    a ``full-stack-app`` must be driven by exactly the same browser client as a
    ``client-app``. Asserted with a real click so this is parity of behaviour,
    not merely of class name.
    """

    async def inner():
        client = await open_trial(full_stack_build, container=False)
        assert isinstance(client, WebAppClient)
        try:
            await client.open()
            after = await client.click("Increment")
            assert after.ok
            assert "Count: 1" in after.text
            assert after.app_type == "client-app"
        finally:
            await client.close()

    _run(inner())


# -- Web (static fallback) --------------------------------------------------


def test_web_static_fallback_is_degraded(web_build) -> None:
    async def inner():
        client = await open_trial(web_build, container=False, use_browser=False)
        assert isinstance(client, StaticWebAppClient)
        try:
            obs = await client.open()
            assert obs.degraded is True
            assert obs.title == "Playground"  # served HTML title is still readable
            # Cannot truly interact without a browser.
            clicked = await client.click("Increment")
            assert clicked.ok is False
            assert clicked.degraded is True
        finally:
            await client.close()

    _run(inner())


def test_trace_is_shared_and_records_steps(web_build) -> None:
    """One trace threads through the client, whichever client you get."""

    async def inner():
        client = await open_trial(web_build, container=False, use_browser=False)
        try:
            await client.open()
            await client.observe()
            assert client.trace.n_steps >= 2
            assert client.trace.app_type == "client-app"
        finally:
            await client.close()

    _run(inner())
