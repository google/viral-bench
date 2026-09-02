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

"""Tests for choosing the founder's model, now that there is no roster.

``viral_bench.founder.models`` used to carry a curated registry -- a list of
model ids, their families, their Vertex locations -- and every one of those
entries was a claim that went stale. It is now a thin, opinion-free view over
:mod:`viral_bench.providers`: resolve a ``provider/model`` string, say which
provider and transport serve it, and refuse anything that names no provider.

Two properties are worth pinning, and they are what most of this file is about.
The registry must not creep back (a benchmark that ships a roster implies the
models on it are the ones worth testing), and the founder's model must stay
independent of the crowd's and the built apps' -- see the invariant at the
bottom, which is the reason this file exists at all.

Every model id here is deliberately fake. A real one in a test is a model this
repo appears to endorse, and it dates the moment the provider retires it.
"""

from __future__ import annotations

import pytest

from viral_bench import config
from viral_bench.founder import models
from viral_bench.founder.models import (
    REQUIRED,
    UnknownModelError,
    describe_models,
    is_supported,
    normalize_model_id,
    provider_for,
    resolve_model,
    transport_for,
)
from viral_bench.providers import PROVIDERS, Capability, UnknownProviderError

FAKE_OPENAI = "openai/gpt-test"
FAKE_ANTHROPIC = "anthropic/claude-test"
FAKE_GEMINI = "google/gemini-test"
FAKE_VERTEX = "google-vertex/gemini-test"
FAKE_VERTEX_CLAUDE = "google-vertex-anthropic/claude-test"


# -- the roster is gone, and must stay gone ---------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "FOUNDER_MODELS",
        "FounderModel",
        "VERTEX_PROVIDERS",
        "DEFAULT_PROVIDER",
        "PROVIDER",
        "CLAUDE_LOCATIONS",
        "DEFAULT_MODEL",
        "available_model_ids",
        "available_providers",
        "split_model",
        "family_for",
        "get_model",
        "is_registered",
        "to_opencode_model",
        "check_location",
    ],
)
def test_the_curated_registry_is_not_reintroduced(name: str) -> None:
    """Each of these encoded a fact with a shelf life.

    A hard-coded roster is out of date the week after it is published, a
    ``DEFAULT_MODEL`` picks someone's API to bill on a user's behalf, and
    ``check_location``/``CLAUDE_LOCATIONS`` pinned one cloud's regional
    availability into a benchmark that is now provider-agnostic. They were
    deleted together, and re-adding any one of them re-opens the same problem, so
    the absence is asserted rather than assumed.
    """
    assert not hasattr(models, name)


def test_unknown_model_error_is_the_provider_layer_error() -> None:
    """Old call sites catch ``UnknownModelError`` while the provider layer raises
    ``UnknownProviderError``. They have to be the same class or the alias is a
    silent hole in every ``except`` in the CLI."""
    assert UnknownModelError is UnknownProviderError


# -- resolution -------------------------------------------------------------- #


def test_an_unheard_of_model_id_resolves_under_a_known_provider() -> None:
    """The escape hatch: benchmark a model released this morning, no code edit.

    Nothing validates the model id itself -- only the provider prefix -- because
    the provider's own catalogue is always more current than anything shipped
    here.
    """
    spec = resolve_model("openai/some-model-nobody-has-heard-of")
    assert spec.provider.id == "openai"
    assert spec.model == "some-model-nobody-has-heard-of"
    assert spec.qualified == "openai/some-model-nobody-has-heard-of"


def test_a_model_id_containing_slashes_survives_resolution() -> None:
    """Split on the FIRST slash only. No provider id contains one and plenty of
    model ids do, so splitting on the last would send an OpenRouter-style id to
    a provider that does not exist."""
    spec = resolve_model("openrouter/some-org/some-model-test")
    assert spec.provider.id == "openrouter"
    assert spec.model == "some-org/some-model-test"


@pytest.mark.parametrize(
    "bad",
    [
        "gpt-test",  # a bare id: no provider to bill
        "claude-test",
        "nosuchvendor/some-model",  # a provider with no entry
        "openai/",  # a provider with no model
        "",  # nothing at all
    ],
)
def test_a_model_that_names_no_provider_is_refused(bad: str) -> None:
    with pytest.raises(UnknownModelError):
        resolve_model(bad)
    assert not is_supported(bad)


def test_is_supported_accepts_any_prefixed_id() -> None:
    assert is_supported(FAKE_OPENAI)
    assert is_supported(FAKE_VERTEX_CLAUDE)


def test_provider_for_names_the_provider_that_will_be_billed() -> None:
    assert provider_for(FAKE_OPENAI) == "openai"
    assert provider_for(FAKE_VERTEX_CLAUDE) == "google-vertex-anthropic"


def test_transport_is_reported_so_call_sites_need_not_know_the_provider() -> None:
    """Branching on the wire protocol rather than the provider id is what makes
    adding a provider that speaks an existing protocol a registry entry and no
    code -- the twelve OpenAI-compatible providers all land on one adapter."""
    assert transport_for(FAKE_OPENAI) == "openai_compat"
    assert transport_for(FAKE_ANTHROPIC) == "anthropic"
    assert transport_for(FAKE_GEMINI) == "google"
    # The two Vertex surfaces differ in wire format, not only in name.
    assert transport_for(FAKE_VERTEX) == "google"
    assert transport_for(FAKE_VERTEX_CLAUDE) == "anthropic"


def test_normalize_strips_only_prefixes_we_recognise() -> None:
    """Never raises, so it is safe on arbitrary strings -- including the retired
    prefixes carried by historical build records."""
    assert normalize_model_id(FAKE_VERTEX) == "gemini-test"
    assert normalize_model_id("gemini-test") == "gemini-test"
    assert normalize_model_id("retired-provider/gemini-test") == (
        "retired-provider/gemini-test"
    )


# -- what the founder stage insists on --------------------------------------- #


def test_the_founder_requires_tool_calling_and_nothing_else() -> None:
    """The founder drives opencode, which cannot work without tools. It does not
    need vision or JSON mode, and requiring either would rule out perfectly good
    coding models for no benefit."""
    assert REQUIRED == Capability.TOOLS


def test_describe_models_lists_providers_and_names_no_model_id() -> None:
    """``viral-bench models`` is the discovery path, so it has to list every
    provider and its credential state -- and must NOT print model ids, which is
    the roster arriving by the back door."""
    text = describe_models()
    for provider_id in PROVIDERS:
        assert provider_id in text
    assert "no default model" in text.lower()
    assert "viral-bench models --check" in text


# -- the invariant the whole comparison depends on --------------------------- #


def _stage_config(monkeypatch, **files: dict) -> None:
    """Pretend ``config/`` holds exactly ``files`` (name -> parsed YAML)."""
    monkeypatch.setattr(config, "_load", lambda name: files.get(name, {}))


def test_swapping_the_founder_model_cannot_move_the_yardstick(monkeypatch) -> None:
    """Each stage's model is configured, and resolved, on its own.

    The founder model is the thing under test. The crowd that rates the apps,
    the grader that inspects them and the model the apps themselves call are the
    measuring instrument, and they are held fixed across a comparison. If a
    founder swap could drag any of them along, every cross-model result would be
    two variables moving at once, read out as one capability difference.

    The separation used to be structural -- different constants in different
    modules. It is now a layering in :func:`viral_bench.config.stage_model`, so
    it is worth checking that the layering does keep the stages apart
    rather than collapsing them onto one value.
    """
    _stage_config(
        monkeypatch,
        **{
            "founder.yaml": {"model": {"id": FAKE_OPENAI}},
            "crowd.yaml": {"simulation": {"model": {"id": FAKE_GEMINI}}},
            "score.yaml": {"autorater": {"model": FAKE_ANTHROPIC}},
        },
    )

    assert config.stage_model("founder") == FAKE_OPENAI
    assert config.stage_model("crowd") == FAKE_GEMINI
    assert config.stage_model("autorater") == FAKE_ANTHROPIC
    # Four distinct stages, four independent answers. The app stage is unset
    # here and stays unset rather than inheriting the founder's.
    assert config.stage_model("app") == ""
    assert config.founder_model_id("") == FAKE_OPENAI
    assert config.crowd_model_id("") == FAKE_GEMINI


def test_a_users_own_choice_overrides_the_tracked_config_per_stage(monkeypatch) -> None:
    """``config/local.yaml`` (written by ``viral-bench init``, gitignored) wins.

    Per stage, not wholesale: overriding the founder must leave the crowd on
    whatever the tracked config says, or upgrading the model under test would
    silently re-point the instrument measuring it.
    """
    _stage_config(
        monkeypatch,
        **{
            "local.yaml": {"models": {"founder": FAKE_ANTHROPIC}},
            "founder.yaml": {"model": {"id": FAKE_OPENAI}},
            "crowd.yaml": {"simulation": {"model": {"id": FAKE_GEMINI}}},
        },
    )
    assert config.stage_model("founder") == FAKE_ANTHROPIC
    assert config.stage_model("crowd") == FAKE_GEMINI
