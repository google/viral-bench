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

"""What a provider is, and how a ``provider/model`` string resolves to one.

ViralBench calls a model from four places -- the founder that builds the app, the
crowd that uses it, the rubric grader, and the autorater -- and each of them used
to reach a different SDK a different way. This module is the single place that
knows what providers exist, how to authenticate to each, and which of them can do
the things the benchmark actually needs.

There is deliberately **no default provider**. ViralBench ships no model and
presumes no account: you bring a key, name it once, and every stage uses it. See
:mod:`viral_bench.providers.credentials` for where a key is read from, and
``viral-bench init`` for the interactive way to write one down.

Adding a provider
-----------------
Most providers speak the OpenAI wire format, so adding one is usually a single
entry in :data:`PROVIDERS` naming its base URL and key variable -- no new code.
Only a provider with a genuinely different wire format (or a different auth
model) needs an adapter of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto


class Capability(Flag):
    """What a model can do, checked before a stage is allowed to use it.

    These are not decoration. Every flag here is load-bearing for at least one
    stage, and a model missing one produces a confusing mid-run failure rather
    than an obvious one -- so :func:`viral_bench.providers.client.check_support`
    refuses the pairing up front instead.
    """

    #: Multi-turn function/tool calling. The crowd drives a real browser through
    #: tools and the grader runs an 18-tool inspection loop; neither degrades to
    #: a plain-text model.
    TOOLS = auto()
    #: Image (PNG) input. Design is a scored dimension and the grader takes
    #: screenshots -- a model that cannot see cannot judge it.
    IMAGES = auto()
    #: Constrained JSON output. The autorater parses its own replies.
    JSON_MODE = auto()
    #: A reasoning/thinking budget knob. Optional everywhere; absence just means
    #: the knob is ignored.
    THINKING = auto()

    NONE = 0


#: What a general-purpose frontier model is assumed to do.
FULL = Capability.TOOLS | Capability.IMAGES | Capability.JSON_MODE


@dataclass(frozen=True)
class ProviderSpec:
    """One model provider: how to reach it, and what it can do."""

    #: The id used in a ``provider/model`` string, e.g. ``openai``.
    id: str
    #: Human-readable name for CLI output.
    name: str
    #: Which adapter speaks to it: ``openai_compat``, ``anthropic`` or ``google``.
    transport: str
    #: Environment variable holding the API key. Empty means the provider needs
    #: no key (a local server), or uses ambient cloud credentials.
    key_env: str = ""
    #: Default API base URL. May be overridden per provider by ``<ID>_BASE_URL``.
    base_url: str = ""
    #: The provider id opencode knows this provider by, for the founder stage.
    #: Empty means opencode has no built-in for it and we must synthesise an
    #: OpenAI-compatible provider block instead.
    opencode_id: str = ""
    #: What models from this provider can generally do. A specific model may do
    #: less; this is the ceiling used for up-front checking.
    capabilities: Capability = FULL
    #: True when authentication is ambient (cloud application-default
    #: credentials) rather than an API key.
    ambient_auth: bool = False
    #: Extra environment variables the founder's opencode child needs.
    extra_env: dict[str, str] = field(default_factory=dict)
    #: One-line hint shown when the credential is missing.
    signup_hint: str = ""


#: Every provider ViralBench knows how to talk to.
#:
#: Ordering is meaningful only for display. The OpenAI-compatible entries differ
#: from each other by base URL and key variable alone -- that is the point of the
#: wire format, and why the long tail costs one line each.
PROVIDERS: dict[str, ProviderSpec] = {
    p.id: p
    for p in (
        ProviderSpec(
            id="openai",
            name="OpenAI",
            transport="openai_compat",
            key_env="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1",
            opencode_id="openai",
            capabilities=FULL | Capability.THINKING,
            signup_hint="https://platform.openai.com/api-keys",
        ),
        ProviderSpec(
            id="anthropic",
            name="Anthropic",
            transport="anthropic",
            key_env="ANTHROPIC_API_KEY",
            base_url="https://api.anthropic.com",
            opencode_id="anthropic",
            capabilities=FULL | Capability.THINKING,
            signup_hint="https://console.anthropic.com/settings/keys",
        ),
        ProviderSpec(
            id="google",
            name="Google Gemini API",
            transport="google",
            key_env="GEMINI_API_KEY",
            # Used only for the OpenAI-compatible view handed to built apps;
            # the adapter itself goes through google-genai.
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            opencode_id="google",
            capabilities=FULL | Capability.THINKING,
            signup_hint="https://aistudio.google.com/apikey",
        ),
        ProviderSpec(
            id="openrouter",
            name="OpenRouter",
            transport="openai_compat",
            key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            opencode_id="openrouter",
            signup_hint="https://openrouter.ai/keys",
        ),
        ProviderSpec(
            id="groq",
            name="Groq",
            transport="openai_compat",
            key_env="GROQ_API_KEY",
            base_url="https://api.groq.com/openai/v1",
            opencode_id="groq",
            capabilities=Capability.TOOLS | Capability.JSON_MODE,
            signup_hint="https://console.groq.com/keys",
        ),
        ProviderSpec(
            id="together",
            name="Together AI",
            transport="openai_compat",
            key_env="TOGETHER_API_KEY",
            base_url="https://api.together.xyz/v1",
            opencode_id="togetherai",
            capabilities=Capability.TOOLS | Capability.JSON_MODE,
            signup_hint="https://api.together.ai/settings/api-keys",
        ),
        ProviderSpec(
            id="fireworks",
            name="Fireworks AI",
            transport="openai_compat",
            key_env="FIREWORKS_API_KEY",
            base_url="https://api.fireworks.ai/inference/v1",
            opencode_id="fireworks-ai",
            capabilities=Capability.TOOLS | Capability.JSON_MODE,
            signup_hint="https://fireworks.ai/account/api-keys",
        ),
        ProviderSpec(
            id="deepseek",
            name="DeepSeek",
            transport="openai_compat",
            key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            opencode_id="deepseek",
            capabilities=Capability.TOOLS | Capability.JSON_MODE,
            signup_hint="https://platform.deepseek.com/api_keys",
        ),
        ProviderSpec(
            id="xai",
            name="xAI",
            transport="openai_compat",
            key_env="XAI_API_KEY",
            base_url="https://api.x.ai/v1",
            opencode_id="xai",
            signup_hint="https://console.x.ai",
        ),
        ProviderSpec(
            id="mistral",
            name="Mistral",
            transport="openai_compat",
            key_env="MISTRAL_API_KEY",
            base_url="https://api.mistral.ai/v1",
            opencode_id="mistral",
            capabilities=Capability.TOOLS | Capability.JSON_MODE,
            signup_hint="https://console.mistral.ai/api-keys",
        ),
        ProviderSpec(
            id="ollama",
            name="Ollama (local)",
            transport="openai_compat",
            key_env="",  # a local server needs no key
            base_url="http://localhost:11434/v1",
            opencode_id="",  # synthesised as an openai-compatible block
            capabilities=Capability.TOOLS,
            signup_hint="https://ollama.com/download, then `ollama serve`",
        ),
        ProviderSpec(
            id="google-vertex",
            name="Google Vertex AI (Gemini)",
            transport="google",
            key_env="",
            opencode_id="google-vertex",
            capabilities=FULL | Capability.THINKING,
            ambient_auth=True,
            signup_hint="gcloud auth application-default login",
        ),
        ProviderSpec(
            id="google-vertex-anthropic",
            name="Google Vertex AI (Claude)",
            transport="anthropic",
            key_env="",
            opencode_id="google-vertex-anthropic",
            capabilities=FULL | Capability.THINKING,
            ambient_auth=True,
            signup_hint="gcloud auth application-default login",
        ),
        ProviderSpec(
            id="custom",
            name="Custom OpenAI-compatible endpoint",
            transport="openai_compat",
            key_env="CUSTOM_API_KEY",
            base_url="",  # must be supplied via CUSTOM_BASE_URL
            opencode_id="",
            capabilities=Capability.TOOLS,
            signup_hint="set CUSTOM_BASE_URL to any OpenAI-compatible /v1 endpoint",
        ),
    )
}


class UnknownProviderError(ValueError):
    """Raised when a model string names a provider we have no entry for."""


@dataclass(frozen=True)
class ModelSpec:
    """A fully resolved model: which provider serves it, and under what id."""

    provider: ProviderSpec
    #: The bare model id as the provider's API expects it.
    model: str

    @property
    def qualified(self) -> str:
        """The canonical ``provider/model`` string."""
        return f"{self.provider.id}/{self.model}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.qualified


def resolve(spec: str) -> ModelSpec:
    """Resolve a ``provider/model`` string.

    The split is on the FIRST ``/`` only: no provider id contains one, and model
    ids frequently do (``openrouter/meta-llama/llama-3.3-70b-instruct``).

    A bare model id with no provider prefix is rejected on purpose. Guessing
    would silently bill the wrong account, and with no default provider there is
    nothing sensible to guess.

    Raises:
        UnknownProviderError: if the string has no provider prefix, or names a
            provider that is not registered.
    """
    provider_id, sep, model = spec.partition("/")
    if not sep or not model:
        raise UnknownProviderError(
            f"{spec!r} is missing a provider prefix. Use '<provider>/<model>', "
            f"e.g. 'openai/gpt-5-mini'. Known providers: {', '.join(PROVIDERS)}. "
            "Run `viral-bench models` to see which ones you have keys for."
        )
    provider = PROVIDERS.get(provider_id)
    if provider is None:
        raise UnknownProviderError(
            f"unknown provider {provider_id!r} in {spec!r}. "
            f"Known providers: {', '.join(PROVIDERS)}."
        )
    return ModelSpec(provider=provider, model=model)


def provider_ids() -> list[str]:
    """Registered provider ids, in registry order."""
    return list(PROVIDERS)
