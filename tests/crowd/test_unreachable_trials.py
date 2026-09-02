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

"""Regression cover: a trial that never reached the app must not rate it.

36 of 314 stored trier traces had every ``open`` fail with a connection error,
yet all 36 recorded ``degraded: false`` and emitted a full four-facet craft
verdict describing UI that never rendered. All 36 were in one build, so the
harness race showed up downstream as a difference between founder models.
"""

from __future__ import annotations

import asyncio

from viral_bench.crowd.interaction.clients import WebAppClient, _unreachable_reason
from viral_bench.crowd.interaction.trace import InteractionTrace
from viral_bench.score.signals import RunSignals  # noqa: F401  (schema anchor)


class _Snap:
    def __init__(self, *, url="", title=None, text="", console_errors=()):
        self.url = url
        self.title = title
        self.text = text
        self.elements = []
        self.aria = ""
        self.console_errors = tuple(console_errors)


class _Page:
    """A page whose navigation always fails, like a server that isn't up yet."""

    def __init__(self, snap):
        self._snap = snap

    async def goto(self, url):
        raise RuntimeError(f"Page.goto: net::ERR_CONNECTION_RESET at {url}")

    async def snapshot(self):
        return self._snap


def _run(coro):
    return asyncio.run(coro)


# -- detection ---------------------------------------------------------------


def test_detects_chrome_error_interstitial() -> None:
    snap = _Snap(
        url="about:blank",
        text="This site can't be reached\n\nThe connection was reset.",
    )
    assert _unreachable_reason(snap, "http://localhost:8000/") is not None


def test_detects_failed_main_request_from_console() -> None:
    snap = _Snap(
        url="http://localhost:8000/",
        text="",
        console_errors=(
            "navigation-failed: http://localhost:8000/ (net::ERR_CONNECTION_RESET)",
        ),
    )
    assert _unreachable_reason(snap, "http://localhost:8000/") is not None


def test_a_failed_SUBRESOURCE_does_not_void_the_trial() -> None:
    """A live-sync stream, a web font and a revoked blob are not a dead app.

    Any failed request used to read as "the main request failed", which marked
    the trial degraded and struck its craft rating out of the score. Over the
    first 660 trials of the v10 sweep that voided 106 of them (16%) and not one
    was the app failing to load -- they were Server-Sent Events aborted by
    navigating away, fonts.gstatic.com with no external network, cancelled XHRs
    and revoked blob: URLs. All four are markers of a MORE capable app, so the
    bias ran against the builds the bench exists to reward: image_compressor
    lost 30 of 30 trials this way, collaborative_table 23 of 60.
    """
    for url, err in (
        ("http://localhost:8000/api/tables/t1/stream", "net::ERR_ABORTED"),
        ("https://fonts.gstatic.com/s/inter/v20/x.woff2", "net::ERR_NAME_NOT_RESOLVED"),
        ("blob:http://localhost:8000/7d46a6dd", "net::ERR_FILE_NOT_FOUND"),
        ("http://localhost:8000/api/doc", "net::ERR_ABORTED"),
    ):
        snap = _Snap(
            url="http://localhost:8000/",
            title="Gridwork",
            text="Shared Database Grid",
            console_errors=(f"subresource-failed: {url} ({err})",),
        )
        assert _unreachable_reason(snap, "http://localhost:8000/") is None, url


def test_a_real_page_is_not_flagged() -> None:
    snap = _Snap(url="http://localhost:8000/", title="QuickMemo", text="Your notes")
    assert _unreachable_reason(snap, "http://localhost:8000/") is None


# -- the trial marks itself ---------------------------------------------------


def test_failed_open_marks_trial_unreachable_and_degraded() -> None:
    trace = InteractionTrace(build_id="b", app_type="client-app")
    client = WebAppClient("http://localhost:8000/", _Page(_Snap()), trace)

    obs = _run(client.open())

    assert obs.ok is False
    assert trace.app_reachable is False
    assert trace.degraded is True
    # ok=False must always carry a reason (the two channels used to be disjoint)
    assert trace.steps[0].errors


def test_looking_at_the_error_page_is_not_a_successful_observation() -> None:
    """The step after a failed open used to be recorded ok=True."""
    snap = _Snap(url="about:blank", text="This site can't be reached")
    trace = InteractionTrace(build_id="b", app_type="client-app")
    client = WebAppClient("http://localhost:8000/", _Page(snap), trace)

    obs = _run(client._snapshot_obs("look", {}))

    assert obs.ok is False
    assert trace.app_reachable is False
    assert trace.degraded is True
    assert trace.steps[-1].errors
    assert "not the product" in obs.summary


def test_ok_false_always_populates_errors() -> None:
    trace = InteractionTrace(build_id="b", app_type="client-app")
    step = trace.record("run", summary="it blew up", ok=False)
    assert step.errors == ("it blew up",)
