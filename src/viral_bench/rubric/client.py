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

"""Where the rubric grader's model client used to live.

The grader was the first stage that needed a real tool-calling loop, so it grew
its own pair of clients: Claude over Vertex ``rawPredict``, and a Gemini one
beside it so "does the grader's identity change the verdict?" stayed a config
flag rather than a rewrite. :mod:`viral_bench.providers` now does all of that for
every provider ViralBench knows about, and it speaks the message format this
module defined, so nothing changed shape on the way across:

* ``ClaudeVertexClient`` is
  :class:`~viral_bench.providers.anthropic.AnthropicAdapter` under the provider
  id ``google-vertex-anthropic`` -- same URL, same ADC bearer token, same
  ``x-goog-user-project`` header, same retry-on-read-timeout.
* ``GeminiClient`` is :class:`~viral_bench.providers.google_genai.GoogleAdapter`,
  which additionally round-trips Gemini's thought signatures that the grader's
  own translation dropped.

The module survives so that ``from viral_bench.rubric.client import ...`` keeps
resolving for existing callers and tests. New code should import from
:mod:`viral_bench.providers` directly.

One thing did NOT come across: the default grader model. Grading with whichever
model happened to be hardcoded here is how a benchmark ends up quietly grading a
model with itself, and the provider layer ships no default for the same reason.
The grader model is now named explicitly at the call site -- see
:func:`viral_bench.rubric.run.grade_build`.
"""

from __future__ import annotations

from viral_bench.providers import ModelError, Reply, ToolCall, make_client

__all__ = ["ModelError", "Reply", "ToolCall", "make_client"]
