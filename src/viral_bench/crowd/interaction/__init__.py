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

"""Give a crowd agent hands to drive a founder-built app like a human would.

The problem this solves is the "blind AI" one the design doc calls out: a `curl`
that returns 200 is *not* a test -- it fetches a single-page app's HTML but never
runs the JavaScript, so the DOM a real user sees is invisible and an LLM defaults
to rubber-stamping. To judge whether an app is any good, a crowd agent has to
actually *use* it.

This subpackage provides that, per app type, behind one small surface:

* :mod:`~viral_bench.crowd.interaction.browser` -- an async Playwright engine that
  drives a **single-page app** in a real (headless) browser: navigate, snapshot
  the accessible DOM, click, type, press keys, screenshot, and read the console.
* :mod:`~viral_bench.crowd.interaction.clients` -- the human-like
  :class:`~viral_bench.crowd.interaction.clients.AppClient` interface with one
  implementation per app type (``WebAppClient`` / ``CliAppClient`` /
  ``BotAppClient``), each returning a structured
  :class:`~viral_bench.crowd.interaction.clients.Observation`.
* :mod:`~viral_bench.crowd.interaction.session` -- ``open_trial(...)``: pick the
  right client for a build (reusing the shared app instance / container runtime)
  and guarantee cleanup.
* :mod:`~viral_bench.crowd.interaction.toolkit` -- wrap all of the above as
  agent-callable tools (``AppInteractionToolkit``), ready to hand to an OASIS
  ``SocialAgent(tools=...)``, plus ``try_app(...)`` for a scripted trial.
* :mod:`~viral_bench.crowd.interaction.trace` -- the structured
  :class:`~viral_bench.crowd.interaction.trace.InteractionTrace` every trial
  emits: the evidence the scoring stage turns into a reaction, and the audit
  trail proving the agent really ran the app.
"""

from __future__ import annotations

from viral_bench.crowd.interaction.browser import (
    BrowserConfig,
    BrowserEngine,
    browser_available,
)
from viral_bench.crowd.interaction.clients import (
    AppClient,
    Observation,
    StaticWebAppClient,
    WebAppClient,
)
from viral_bench.crowd.interaction.inspect import CodeInspectionToolkit
from viral_bench.crowd.interaction.session import TrialError, open_trial, peek_manifest
from viral_bench.crowd.interaction.toolkit import AppInteractionToolkit, try_app
from viral_bench.crowd.interaction.trace import (
    InteractionStep,
    InteractionTrace,
    TrialVerdict,
)

__all__ = [
    # trace / evidence
    "InteractionStep",
    "InteractionTrace",
    "TrialVerdict",
    # browser engine
    "BrowserConfig",
    "BrowserEngine",
    "browser_available",
    # clients
    "AppClient",
    "Observation",
    "WebAppClient",
    "StaticWebAppClient",
    # session + toolkit (the agent-facing surface)
    "TrialError",
    "open_trial",
    "peek_manifest",
    "AppInteractionToolkit",
    "try_app",
    # code inspection (cheap, read-only; for reactor agents)
    "CodeInspectionToolkit",
]
