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

"""The LLM backend that powers the crowd agents (CAMEL -> any provider).

The crowd is the dominant compute cost of the whole benchmark -- many agents,
several rounds, per app -- so its model wants to be cheap and fast. Which model
that is, is not for ViralBench to decide: it ships no model and presumes no
account. You name one as ``provider/model`` and every stage uses it, the crowd
included. See :mod:`viral_bench.providers` for the registry.

Two backends sit behind :func:`crowd_model`, chosen by the resolved provider:

* **Gemini** (``google/...`` and ``google-vertex/...``) keeps its own backend,
  :class:`~viral_bench.crowd.sim.gemini_native.GeminiNativeModel`. recent Gemini
  multi-turn tool calling needs ``thought_signature`` round-tripping, and Vertex
  needs several request-shape workarounds that cost 54,522 agent turns to find.
  None of that generalises, and none of it may be lost.
* **Everything else** goes through
  :class:`~viral_bench.crowd.sim.unified_model.UnifiedModel`, which delegates to
  ``providers.make_client()`` and therefore reaches OpenAI, Anthropic,
  OpenRouter, Groq, a local Ollama, or anything else in the registry.

There is no default model and no default provider. There used to be
(``gemini-2.0-flash``), and with it a ``CROWD_USE_VERTEX`` flag that
defaulted ON -- so a new user with a perfectly good ``GEMINI_API_KEY`` got an
application-default-credentials failure naming a Google cloud project, from a
flag they had no way to know existed. Vertex is now a provider id you
choose (``google-vertex/...``) rather than a hidden default you must opt out of.

The crowd still reads its OWN Gemini key first (``GEMINI_API_KEY_CROWD``, then
``GEMINI_API_KEY``) so its spend can be budgeted apart from the founder's. Both
are resolved from the environment or the repo ``.env`` via the shared
:mod:`viral_bench.founder.appenv` reader.
"""

from __future__ import annotations

from viral_bench.crowd.sim_defaults import DEFAULT_MAX_TOKENS
from viral_bench.founder.appenv import read_key
from viral_bench.providers import (
    Capability,
    ModelSpec,
    UnknownProviderError,
    check_support,
    resolve,
)

#: Output-token cap for crowd calls. ``None`` means no cap is sent, so the
#: model's own maximum applies. Set ``simulation.model.max_tokens`` in
#: config/crowd.yaml to impose one deliberately.
#:
#: History: 2048 -> 8192 -> uncapped. Neither number was ever chosen from
#: evidence, because finish_reason was hard-coded to "stop" and a truncated
#: reply was indistinguishable from a complete one. It is reported honestly now
#: (see gemini_native._finish_reason), and on recent Gemini thinking tokens are
#: charged against this same budget -- so any ceiling here silently caps
#: deliberation as well as output.
CROWD_MAX_TOKENS = DEFAULT_MAX_TOKENS

#: Host env var names checked, in order, for the crowd's Gemini key.
CROWD_KEY_NAMES = ("GEMINI_API_KEY_CROWD", "GEMINI_API_KEY")

#: What the crowd needs from whatever model it is pointed at. It drives a real
#: browser through tools and its percept of an app is a PNG screenshot -- a
#: model missing either does not degrade gracefully, it silently rates a design
#: it never saw. Checked when the backend is built, so an incapable provider is
#: refused before a 40-minute run starts rather than three hours into a sweep.
CROWD_CAPABILITIES = Capability.TOOLS | Capability.IMAGES

#: Handed to CAMEL's ``GeminiModel`` base, which validates that *some* key is
#: present, on the Vertex path where there is none (ADC carries the auth).
#: gemini_native replaces the client outright, so it is never used.
_ADC_PLACEHOLDER_KEY = "vertex-adc-no-key"


class CrowdModelError(RuntimeError):
    """Raised when the crowd model backend cannot be constructed."""


def resolve_crowd_key() -> str | None:
    """Return the crowd's Gemini API key from the env / repo .env, or None."""
    for name in CROWD_KEY_NAMES:
        value = read_key(name)
        if value:
            return value
    return None


def resolve_crowd_model(model_id: str) -> ModelSpec:
    """Resolve the crowd's ``provider/model`` string into a :class:`ModelSpec`.

    Raises:
        CrowdModelError: if nothing is configured, or the string does not name a
            registered provider. Both messages say what to run next: there is no
            model to guess at, and guessing would bill an account the user never
            chose.
    """
    named = (model_id or "").strip()
    if not named:
        raise CrowdModelError(
            "no crowd model is configured. The crowd needs one named as "
            "'<provider>/<model>' -- e.g. 'openai/gpt-5-mini', "
            "'anthropic/claude-sonnet-4-5' or 'google/gemini-2.0-flash'. "
            "Set it as `simulation.model.id` in config/crowd.yaml or pass "
            "--model, and run `viral-bench init` to pick a provider and store "
            "its key."
        )
    try:
        return resolve(named)
    except UnknownProviderError as exc:
        raise CrowdModelError(
            f"the crowd cannot use {named!r}: {exc} "
            "Run `viral-bench init` to pick a provider and store its key."
        ) from exc


def crowd_transport(model_id: str) -> str:
    """Which provider carried a run's model calls, for the run summary.

    Returns the provider id (``google-vertex``, ``openai``, ...), or
    ``"unresolved"`` when the run never got as far as building a model.

    Runs recorded before the provider layer landed carry ``"vertex"`` or
    ``"developer_api"`` here instead. Those are the same distinction under the
    older two-surface vocabulary, and map to ``google-vertex`` and ``google``.
    """
    try:
        return resolve_crowd_model(model_id).provider.id
    except CrowdModelError:
        return "unresolved"


def crowd_model(
    model_id: str,
    *,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = CROWD_MAX_TOKENS,
    thinking_level: str | None = None,
):
    """Create a CAMEL model backend for the crowd agents.

    Args:
        model_id: A ``provider/model`` string, e.g. ``openai/gpt-5-mini`` or
            ``google-vertex/gemini-2.0-flash``. Required: there is no
            default model and a bare model id is rejected, because guessing the
            provider would silently bill an account nobody chose.
        api_key: Explicit key for the Gemini Developer API path. When omitted,
            resolved from ``GEMINI_API_KEY_CROWD`` then ``GEMINI_API_KEY``.
            Every other provider resolves its own key through
            :mod:`viral_bench.providers.credentials`, and the Vertex path needs
            none at all (ADC carries the auth).
        temperature: Sampling temperature (some spread gives persona variety).
        max_tokens: Optional output cap. ``None`` (the default) sends no cap.
        thinking_level: How hard the model may think per call, where the
            provider has such a knob.

    Returns:
        A ``camel.models`` backend usable as a ``SocialAgent(model=...)``.

    Raises:
        CrowdModelError: If no model is configured, the provider is unknown, the
            Gemini Developer API path was chosen without a key, or the crowd
            environment is missing its dependencies.
        UnsupportedCapabilityError: If the provider cannot do tool calling and
            image input, which the crowd cannot work without.
    """
    spec = resolve_crowd_model(model_id)
    check_support(spec, CROWD_CAPABILITIES, stage="crowd")

    config: dict[str, object] = {"temperature": temperature}
    if thinking_level:
        config["thinking_level"] = thinking_level
    if max_tokens is not None:
        config["max_tokens"] = max_tokens

    if spec.provider.transport == "google":
        return _gemini_backend(spec, config, api_key)
    try:
        from viral_bench.crowd.sim.unified_model import UnifiedModel
    except ImportError as exc:  # pragma: no cover - only in the crowd env
        raise CrowdModelError(
            "camel-ai is required for the crowd model; run inside the crowd env."
        ) from exc
    return UnifiedModel(spec, model_config_dict=config)


def _gemini_backend(spec: ModelSpec, config: dict[str, object], api_key: str | None):
    """Build the Gemini-only backend, which is not interchangeable with the rest.

    Keeping Gemini on its own transport is not tidiness: CAMEL's built-in
    ``GeminiModel`` goes through the OpenAI-compatibility endpoint, which cannot
    round-trip the ``thought_signature`` recent Gemini requires, and every crowd
    agent's SECOND tool call then fails with an HTTP 400. See
    :mod:`viral_bench.crowd.sim.gemini_native` for that and for the Vertex
    request-shape workarounds it carries.
    """
    key = api_key or resolve_crowd_key()
    if not spec.provider.ambient_auth and not key:
        raise CrowdModelError(
            f"{spec.qualified} authenticates with an API key and none is set. "
            "Put GEMINI_API_KEY_CROWD (or GEMINI_API_KEY) in the environment or "
            "the repo .env -- the crowd reads its own key first so its spend can "
            "be budgeted apart from the founder's -- or name "
            f"'google-vertex/{spec.model}' instead to authenticate with "
            "application-default credentials. `viral-bench init` walks you "
            "through either one."
        )
    try:
        from viral_bench.crowd.sim.gemini_native import GeminiNativeModel
    except ImportError as exc:  # pragma: no cover - only in the crowd env
        raise CrowdModelError(
            "camel-ai + google-genai are required for the crowd model; run inside "
            "the crowd env."
        ) from exc
    return GeminiNativeModel(
        spec,
        model_config_dict=config,
        api_key=key or _ADC_PLACEHOLDER_KEY,
    )


def dummy_model(model_id: str = "no-llm-placeholder"):
    """A constructible-but-never-called model backend for no-LLM smoke runs.

    OASIS ``SocialAgent`` needs a model to construct, but the ``--no-llm`` wiring
    smoke drives everything with scripted ``ManualAction``s and never invokes the
    LLM, so a placeholder key is fine (and avoids requiring a real key merely to
    validate the platform wiring). The model id is a placeholder for the same
    reason -- it names no real model, and naming one here would reintroduce
    exactly the default this module exists without.
    """
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    return ModelFactory.create(
        model_platform=ModelPlatformType.GEMINI,
        model_type=model_id,
        api_key="dummy-key-not-used",
    )
