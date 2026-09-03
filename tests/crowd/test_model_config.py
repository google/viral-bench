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

"""What the crowd backend sends to Gemini.

Specifically: no artificial output-token ceiling. This has been reintroduced
twice -- 2048, then 8192 -- each time as a number nobody could evaluate, because
``finish_reason`` was hard-coded to ``"stop"`` and a truncated reply was
indistinguishable from a complete one.

Two things make a cap here worse than it looks. The crowd's job is rich,
differentiated reactions, so a ceiling trades away the signal the benchmark
exists to measure. And recent Gemini charges *thinking* tokens against
``max_output_tokens``, so any figure is a combined think-plus-answer budget: a
long deliberation surfaces as a reply cut off mid-sentence.

The backend subclasses CAMEL's ``GeminiModel``, and camel-ai is deliberately not
in this project's lock -- it cannot be resolved alongside the Gemini SDK, and
lives in ``.venv-crowd`` instead. So this module is skipped, not collected into
an error, in the environment a plain ``uv sync`` produces.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "camel", reason="needs camel-ai, which lives in .venv-crowd, not this env"
)

from viral_bench.crowd.sim.gemini_native import GeminiNativeModel  # noqa: E402
from viral_bench.providers import resolve  # noqa: E402

#: The keyed Gemini surface. The backend now takes a resolved ModelSpec rather
#: than a bare model id, because which of the two Google surfaces to call is a
#: property of the provider named in the config -- it used to be an environment
#: flag that defaulted to Vertex whatever the user had configured.
_SPEC = resolve("google/gemini-lite-test")


def _config(**model_config):
    backend = GeminiNativeModel(
        _SPEC,
        model_config_dict=model_config,
        api_key="not-a-real-key",
    )
    return backend._to_config(None, "system text")


def test_no_output_token_cap_is_sent_by_default() -> None:
    # This is the assertion that matters: what reaches the API, not what a
    # constant says. crowd_model() passes no max_tokens, so none is set.
    assert _config(temperature=0.7).max_output_tokens is None


def test_an_explicit_cap_is_still_honoured() -> None:
    # The knob has to be real, not merely switched off -- config/crowd.yaml
    # `simulation.model.max_tokens` must be able to impose one deliberately.
    assert _config(temperature=0.7, max_tokens=1234).max_output_tokens == 1234


def test_context_window_is_not_tied_to_the_output_cap() -> None:
    """``token_limit`` is the INPUT window and must stay large.

    CAMEL's ``BaseModelBackend.token_limit`` returns ``max_tokens`` when set, and
    ChatAgent sizes its context window from it -- which would slice a
    ``function_call`` turn away from its preceding turn and make Gemini reject
    the request. The override reports the model's real ~1M input window.
    """
    backend = GeminiNativeModel(
        _SPEC,
        model_config_dict={"temperature": 0.7, "max_tokens": 512},
        api_key="not-a-real-key",
    )
    assert backend.token_limit == 1_048_576


def test_trier_budget_cannot_be_starved_by_a_thorough_trial() -> None:
    """A trier's social turn must survive a maximal hands-on trial.

    A trier spends ONE OASIS step both driving the app and posting about it, out
    of a single ``max_iteration``. The trial runs first, so the two compete and
    the app always wins. Under the original budget of 10 this was severe: triers
    spending >=10 calls on the app averaged 0.05 social actions and 37 of 39 were
    silenced -- for a virality benchmark, the agents with first-hand experience
    were exactly the ones who never got to speak.

    It stopped reproducing once the budget was raised to 30, but by luck rather
    than design: a trial may use up to ``trial_max_steps`` (24), leaving 6. This
    pins the arithmetic so the guarantee is structural.
    """
    from pathlib import Path

    from viral_bench.crowd.sim_defaults import (
        DEFAULT_MAX_ITERATION_TRIER,
        DEFAULT_SOCIAL_HEADROOM,
        DEFAULT_TRIAL_MAX_STEPS,
    )

    assert (
        DEFAULT_MAX_ITERATION_TRIER == DEFAULT_TRIAL_MAX_STEPS + DEFAULT_SOCIAL_HEADROOM
    )
    # Worst case measured over 600 triers on arch v8 was 25 total calls, and the
    # headroom alone must comfortably exceed the social half of that.
    assert DEFAULT_SOCIAL_HEADROOM >= 12

    # SimulationConfig lives behind the OASIS import, which is only installed in
    # the crowd venv, and the invariant itself is in sim_defaults and checked above.
    pytest.importorskip("oasis", reason="SimulationConfig needs the crowd env")
    from viral_bench.crowd.sim.simulation import SimulationConfig

    cfg = SimulationConfig(build_id="b", out_dir=Path("/tmp/vb-invariant"))
    assert cfg.max_iteration_trier - cfg.trial_max_steps >= 12, (
        "a maximal trial would leave a trier too few calls to say anything"
    )
    # Reactors run no trial, so their budget is unrelated and must not be tied.
    assert cfg.max_iteration_reactor < cfg.max_iteration_trier
