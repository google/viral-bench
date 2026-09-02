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

"""Tests for the provider layer: one interface over every backend.

ViralBench used to reach three SDKs three ways from four stages. This package
replaced that with a registry, one client interface and one retry loop, and the
things worth testing are the decisions that layer makes on a caller's behalf:
what it refuses, what it retries, and what it hands to the founder's opencode
child.

Nothing here touches the network, and nothing needs a real credential -- every
provider used is either ambient-auth, keyless, or given a fake key by the test.
Model ids are fake throughout: a real one in a test is a model this repo appears
to endorse, and it dates the moment the provider retires it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# WORKAROUND, not a dependency: importing `viral_bench.providers` first in a
# fresh interpreter raises ImportError. providers/credentials.py reaches for
# viral_bench.founder.appenv, which runs the founder package __init__, which
# imports back into viral_bench.providers before it has finished initialising.
# Importing the founder package first breaks the cycle. Reported to be fixed in
# src; remove this line once it is.
import viral_bench.founder  # noqa: F401
from viral_bench.providers import (
    FULL,
    PROVIDERS,
    Capability,
    MissingCredentialError,
    ModelError,
    UnknownProviderError,
    UnsupportedCapabilityError,
    check_support,
    credentials,
    make_client,
    resolve,
)
from viral_bench.providers import client as client_module
from viral_bench.providers.client import Adapter, Reply
from viral_bench.providers.errors import (
    AuthError,
    BadRequestError,
    RateLimitError,
    TransientError,
    backoff_seconds,
    classify,
    from_status,
)
from viral_bench.providers.opencode import OPENCODE_KEY_ENV, opencode_provider


@pytest.fixture(autouse=True)
def no_env_file(tmp_path, monkeypatch):
    """Never read the developer's own repo ``.env``.

    Otherwise whether a provider "has a credential" depends on whose machine the
    suite is running on, which is the least useful kind of flake.
    """
    monkeypatch.setenv("VIRAL_BENCH_ENV_FILE", str(tmp_path / "absent.env"))
    for provider in PROVIDERS.values():
        if provider.key_env:
            monkeypatch.delenv(provider.key_env, raising=False)
        monkeypatch.delenv(credentials.base_url_env(provider), raising=False)


# -- resolution: a provider prefix is mandatory ------------------------------ #


def test_a_bare_model_id_is_refused_rather_than_guessed() -> None:
    """The single most important refusal in the package.

    With no default provider there is nothing sensible to guess, and a guess
    would silently bill whichever account happened to have a key set. The error
    therefore has to teach the fix rather than just say no.
    """
    with pytest.raises(UnknownProviderError) as excinfo:
        resolve("gpt-test")
    message = str(excinfo.value)
    assert "<provider>/<model>" in message
    assert "viral-bench models" in message
    # And it lists what the prefixes can be, so the fix needs no second lookup.
    assert "openai" in message


def test_an_unregistered_provider_is_named_in_the_error() -> None:
    with pytest.raises(UnknownProviderError) as excinfo:
        resolve("nosuchvendor/some-model")
    assert "nosuchvendor" in str(excinfo.value)


def test_a_provider_prefix_with_no_model_is_refused() -> None:
    with pytest.raises(UnknownProviderError):
        resolve("openai/")


def test_resolution_splits_on_the_first_slash_only() -> None:
    """No provider id contains a slash and plenty of model ids do."""
    spec = resolve("openrouter/some-org/some-model-test")
    assert spec.provider.id == "openrouter"
    assert spec.model == "some-org/some-model-test"
    assert spec.qualified == "openrouter/some-org/some-model-test"


# -- registry invariants ----------------------------------------------------- #


def test_every_provider_can_actually_be_dispatched() -> None:
    """A registry entry naming a transport nobody implements is a run that dies
    at the first call, having already been accepted by every earlier check."""
    for provider in PROVIDERS.values():
        assert provider.transport in ("openai_compat", "anthropic", "google")
        # The hint is what a user sees when their credential is missing, so an
        # entry without one turns a fixable error into a dead end.
        assert provider.signup_hint


def test_every_provider_is_reachable_without_being_asked_for_a_base_url() -> None:
    """A provider with neither a default base URL nor ambient auth cannot be
    used until the user finds the right ``<ID>_BASE_URL`` -- acceptable for the
    deliberately-blank ``custom`` entry, and a bug for anything else."""
    for provider in PROVIDERS.values():
        if provider.id == "custom":
            continue
        assert provider.base_url or provider.ambient_auth, provider.id


def test_the_founders_opencode_child_can_be_given_every_keyed_credential() -> None:
    """opencode reads a per-provider variable, and the names are not always ours.

    A provider opencode knows, that authenticates with a key, but that has no
    entry in the opencode key map, would launch the founder child with no
    credential at all -- and fail as an auth error against the model rather than
    as the mapping gap it is.
    """
    for provider in PROVIDERS.values():
        if provider.ambient_auth or not provider.opencode_id or not provider.key_env:
            continue
        assert provider.opencode_id in OPENCODE_KEY_ENV, provider.id
        # And it must be the SAME variable we resolve the credential from.
        assert OPENCODE_KEY_ENV[provider.opencode_id] == provider.key_env


# -- capability checking ----------------------------------------------------- #


def test_a_provider_missing_a_capability_is_refused_before_the_run() -> None:
    """The alternative is finding out three hours into a sweep.

    A crowd whose model cannot see the screenshots it was sent still produces
    numbers -- it scores a design dimension it never observed -- so this has to
    fail up front, and the message has to name the missing capability and the
    provider rather than just failing.
    """
    spec = resolve("groq/model-test")  # tools + json, no image input
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        check_support(spec, Capability.IMAGES, stage="crowd")
    message = str(excinfo.value)
    assert "images" in message
    assert "crowd" in message
    assert spec.provider.name in message


def test_every_missing_capability_is_named_not_just_the_first() -> None:
    """Fixing one and rediscovering the next is two round trips for no reason."""
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        check_support(
            resolve("ollama/model-test"),  # tools only
            Capability.IMAGES | Capability.JSON_MODE,
            stage="autorater",
        )
    message = str(excinfo.value)
    assert "images" in message
    assert "json_mode" in message


def test_a_capable_provider_passes_silently() -> None:
    check_support(resolve("openai/gpt-test"), FULL, stage="founder")


# -- credentials -------------------------------------------------------------- #


def test_a_missing_key_says_which_variable_and_where_to_get_one() -> None:
    provider = PROVIDERS["openai"]
    with pytest.raises(MissingCredentialError) as excinfo:
        credentials.resolve(provider)
    message = str(excinfo.value)
    assert provider.key_env in message
    assert provider.signup_hint in message
    assert "viral-bench init" in message


def test_a_base_url_override_is_named_per_provider() -> None:
    """The escape hatch for a proxy or a gateway. Hyphens become underscores so
    the name is a legal shell variable."""
    assert credentials.base_url_env(PROVIDERS["google-vertex"]) == (
        "GOOGLE_VERTEX_BASE_URL"
    )
    assert credentials.base_url_env(PROVIDERS["openai"]) == "OPENAI_BASE_URL"


def test_an_override_wins_over_the_providers_default(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://gateway.test/v1")
    resolved = credentials.resolve(PROVIDERS["openai"])
    assert resolved.base_url == "http://gateway.test/v1"
    assert resolved.api_key == "key-test"


def test_providers_that_need_no_key_report_usable(monkeypatch) -> None:
    # Ambient cloud credentials, and a local server, are both usable with
    # nothing set -- reporting them as MISSING would send a user hunting for a
    # key that does not exist.
    assert credentials.has_credential(PROVIDERS["google-vertex"]) is True
    assert credentials.has_credential(PROVIDERS["ollama"]) is True
    assert credentials.has_credential(PROVIDERS["openai"]) is False
    monkeypatch.setenv("OPENAI_API_KEY", "key-test")
    assert credentials.has_credential(PROVIDERS["openai"]) is True


# -- the opencode hand-off (the founder stage) -------------------------------- #


def test_a_keyed_provider_opencode_knows_is_named_and_given_its_key(
    monkeypatch,
) -> None:
    """The common case: name the provider, set its variable, declare the model.

    The model entry is declared even though opencode knows the provider, because
    its bundled catalogue may not list this exact id yet -- and without an entry
    opencode has no context limit and cannot compact a long session.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "key-test")
    target = opencode_provider(resolve("openai/gpt-test"))

    assert target.model == "openai/gpt-test"
    assert target.provider_block == {
        "openai": {"models": {"gpt-test": {"name": "gpt-test"}}}
    }
    # The credential goes to the child's environment only, never to the config
    # file we write into the build workspace.
    assert target.env == {"OPENAI_API_KEY": "key-test"}
    assert "apiKey" not in str(target.provider_block)
    # A provider opencode has a built-in for must NOT be synthesised.
    assert "npm" not in target.provider_block["openai"]


def test_opencode_is_told_the_id_it_knows_the_provider_by(monkeypatch) -> None:
    """Ours and opencode's ids are not always the same string, and a mismatch
    would ask opencode for a provider it has never heard of."""
    monkeypatch.setenv("TOGETHER_API_KEY", "key-test")
    target = opencode_provider(resolve("together/model-test"))
    assert target.model == "togetherai/model-test"
    assert set(target.provider_block) == {"togetherai"}


def test_a_provider_opencode_does_not_know_is_synthesised(monkeypatch) -> None:
    """How a local Ollama, a vLLM server or a private gateway works at all.

    opencode has no built-in for these, so we emit an OpenAI-compatible provider
    block pointing at the configured base URL. Without it the founder stage --
    the one place ViralBench does not make the model call itself -- would support
    strictly fewer providers than every other stage.
    """
    target = opencode_provider(resolve("ollama/model-test"))

    block = target.provider_block["ollama"]
    assert block["npm"] == "@ai-sdk/openai-compatible"
    assert block["options"]["baseURL"] == PROVIDERS["ollama"].base_url
    assert block["models"] == {"model-test": {"name": "model-test"}}
    # A synthesised provider keeps OUR id, since opencode has no name for it.
    assert target.model == "ollama/model-test"
    # A local server needs no key, so none is invented.
    assert "apiKey" not in block["options"]
    assert target.env == {}


def test_a_synthesised_provider_carries_its_key_in_the_config(monkeypatch) -> None:
    """There is no environment variable opencode would read for a provider it
    does not know, so the key has to travel in the provider block instead."""
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://gateway.test/v1")
    monkeypatch.setenv("CUSTOM_API_KEY", "key-test")
    target = opencode_provider(resolve("custom/model-test"))

    options = target.provider_block["custom"]["options"]
    assert options == {"baseURL": "http://gateway.test/v1", "apiKey": "key-test"}


def test_a_base_url_is_only_overridden_when_the_user_set_one(monkeypatch) -> None:
    """Otherwise we would pin a provider's endpoint to whatever it was when this
    registry entry was written, and opencode keeps its own copy current."""
    monkeypatch.setenv("OPENAI_API_KEY", "key-test")
    assert (
        "options"
        not in opencode_provider(resolve("openai/gpt-test")).provider_block["openai"]
    )

    monkeypatch.setenv("OPENAI_BASE_URL", "http://gateway.test/v1")
    overridden = opencode_provider(resolve("openai/gpt-test"))
    assert overridden.provider_block["openai"]["options"] == {
        "baseURL": "http://gateway.test/v1"
    }


def test_an_ambient_provider_gets_a_cloud_environment_and_no_key(monkeypatch) -> None:
    monkeypatch.setenv("VERTEX_PROJECT", "project-test")
    target = opencode_provider(resolve("google-vertex/gemini-test"))

    assert target.env["GOOGLE_CLOUD_PROJECT"] == "project-test"
    assert not any(key.endswith("API_KEY") for key in target.env)
    assert target.provider_block == {
        "google-vertex": {"models": {"gemini-test": {"name": "gemini-test"}}}
    }


def test_thinking_options_ride_along_on_the_model_entry(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-test")
    target = opencode_provider(
        resolve("anthropic/claude-test"), thinking_options={"effort": "high"}
    )
    entry = target.provider_block["anthropic"]["models"]["claude-test"]
    assert entry["options"] == {"effort": "high"}


def test_a_provider_with_no_credential_refuses_to_build_a_founder_config() -> None:
    """Better here than as an opaque auth failure inside the opencode child."""
    with pytest.raises(MissingCredentialError):
        opencode_provider(resolve("openai/gpt-test"))


# -- the shared retry loop ---------------------------------------------------- #


class _FakeAdapter(Adapter):
    """An adapter whose ``_call`` does whatever the test says."""

    def __init__(self, spec, **kwargs) -> None:
        super().__init__(spec, **kwargs)
        self.calls: list[dict] = []
        self.outcomes: list = []

    def _call(self, messages, *, tools=None, system="") -> Reply:
        self.calls.append({"messages": messages, "max_tokens": self.max_tokens})
        outcome = self.outcomes.pop(0) if self.outcomes else Reply(text="ok")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def no_sleep(monkeypatch):
    """Record backoff waits instead of taking them."""
    slept: list[float] = []
    monkeypatch.setattr(
        client_module, "time", SimpleNamespace(sleep=lambda s: slept.append(s))
    )
    return slept


def _adapter(**kwargs) -> _FakeAdapter:
    # A keyless local provider, so no credential is needed to construct one.
    return _FakeAdapter(resolve("ollama/model-test"), **kwargs)


def test_a_transient_failure_does_not_cost_the_run(no_sleep) -> None:
    """A grade or a crowd turn is expensive to redo, so one hiccup must not end
    it. The loop lives here rather than in each adapter because it used to be
    three different backoff curves in three files."""
    adapter = _adapter()
    adapter.outcomes = [TransientError("overloaded"), Reply(text="second time")]

    assert adapter.generate([{"role": "user", "content": "hi"}]).text == "second time"
    assert len(adapter.calls) == 2
    assert len(no_sleep) == 1


def test_a_malformed_request_is_not_retried(no_sleep) -> None:
    """Sending the same broken request three more times changes nothing except
    how long the user waits for the error."""
    adapter = _adapter()
    adapter.outcomes = [BadRequestError("bad tool schema")]

    with pytest.raises(BadRequestError):
        adapter.generate([{"role": "user", "content": "hi"}])
    assert len(adapter.calls) == 1
    assert no_sleep == []


def test_a_persistent_failure_surfaces_rather_than_hanging(no_sleep) -> None:
    adapter = _adapter()
    adapter.outcomes = [TransientError("overloaded")] * client_module.MAX_ATTEMPTS

    with pytest.raises(ModelError) as excinfo:
        adapter.generate([{"role": "user", "content": "hi"}])
    assert len(adapter.calls) == client_module.MAX_ATTEMPTS
    # The model that failed and the last thing it said are both in the message.
    assert "ollama/model-test" in str(excinfo.value)
    assert "overloaded" in str(excinfo.value)


def test_ping_is_a_one_token_call_on_the_same_model() -> None:
    """Preflight has to prove the model is callable without being worth
    skipping, so it must stay this cheap -- and it must not disturb the
    caller's own configuration."""
    adapter = _adapter(max_tokens=4096)
    adapter.ping()
    # The probe is a separate instance, so the real client keeps its ceiling.
    assert adapter.max_tokens == 4096


def test_rate_limits_back_off_harder_than_ordinary_failures() -> None:
    """A 429 is the server asking for room; anything else is usually one bad
    connection. Jitter matters too: a sweep runs many workers, and a fixed delay
    has them retry in lockstep and reproduce the burst."""
    limited = [backoff_seconds(3, RateLimitError("429")) for _ in range(20)]
    ordinary = [backoff_seconds(3, TransientError("reset")) for _ in range(20)]

    assert min(limited) > max(ordinary)
    assert len(set(limited)) > 1  # jittered, not a constant


# -- error classification ----------------------------------------------------- #


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, RateLimitError),
        (503, TransientError),
        (401, AuthError),
        (403, AuthError),
        (400, BadRequestError),
    ],
)
def test_a_status_code_decides_whether_to_retry(status: int, expected: type) -> None:
    """The crowd used to string-match Google and Anthropic phrasings, so an
    OpenAI 429 read as fatal and aborted a run a two-second backoff would have
    saved. Classification belongs to the status code where one exists."""
    assert isinstance(from_status(status, "detail"), expected)


def test_an_sdk_exception_carrying_a_status_is_classified_by_it() -> None:
    exc = RuntimeError("service unavailable")
    exc.status_code = 503
    assert isinstance(classify(exc), TransientError)


def test_text_is_only_the_fallback_when_there_is_no_status() -> None:
    assert isinstance(classify(RuntimeError("invalid api key")), AuthError)
    assert isinstance(classify(TimeoutError("took too long")), TransientError)


# -- dispatch ------------------------------------------------------------------ #


@pytest.mark.parametrize(
    ("model", "adapter_name"),
    [
        ("openai/gpt-test", "OpenAICompatAdapter"),
        ("anthropic/claude-test", "AnthropicAdapter"),
        ("google-vertex-anthropic/claude-test", "AnthropicAdapter"),
        ("google-vertex/gemini-test", "GoogleAdapter"),
    ],
)
def test_the_transport_chooses_the_adapter(
    monkeypatch, model: str, adapter_name: str
) -> None:
    """Two providers on one wire format share one adapter -- that is what makes
    the long tail of providers cost a registry line each rather than a module."""
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "key-test")
    assert type(make_client(model)).__name__ == adapter_name
