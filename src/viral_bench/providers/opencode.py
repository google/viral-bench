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

"""Teaching the founder's opencode child which provider to use.

The founder stage is the one place ViralBench does not make the model call
itself: it drives the `opencode <https://opencode.ai>`_ CLI, and opencode makes
the call. That looked like the hardest part of supporting arbitrary providers and
turned out to be the easiest -- opencode already speaks 75+ providers through the
AI SDK and models.dev, so the work here is emitting the right provider block and
the right environment, not writing an agent loop.

Three shapes come out of :func:`opencode_provider`:

* a provider opencode knows and that authenticates with a key -- name it and set
  its key variable;
* a provider opencode knows that uses ambient cloud credentials (Vertex) -- name
  it and set the cloud environment;
* anything else -- synthesise an OpenAI-compatible provider block pointing at the
  configured base URL, which is how a local Ollama or a private gateway works
  with no opencode support at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from viral_bench.providers import credentials
from viral_bench.providers.spec import ModelSpec

#: The environment variable opencode reads for each provider it knows natively.
#: opencode also accepts credentials from its own auth.json, but ViralBench is
#: non-interactive, so it always passes them explicitly.
OPENCODE_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "togetherai": "TOGETHER_API_KEY",
    "fireworks-ai": "FIREWORKS_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


@dataclass
class OpencodeTarget:
    """Everything the harness needs to point opencode at one model."""

    #: The ``provider/model`` string opencode itself uses. Not necessarily ours:
    #: opencode calls Together "togetherai", and a synthesised provider keeps our
    #: own id.
    model: str
    #: The ``provider`` section to merge into the per-build opencode config.
    provider_block: dict
    #: Environment for the opencode child only -- never exported into this
    #: process, so a sweep can run two arms against two providers at once.
    env: dict[str, str] = field(default_factory=dict)


def opencode_provider(
    spec: ModelSpec, *, thinking_options: dict | None = None
) -> OpencodeTarget:
    """Build the opencode config and environment for ``spec``.

    Args:
        spec: The resolved model to run the founder on.
        thinking_options: Provider-specific model options (e.g. a reasoning
            budget) to attach to the model entry, or ``None``.

    Raises:
        MissingCredentialError: if the provider needs a credential and has none.
    """
    provider = spec.provider
    credential = credentials.resolve(provider)

    # Declare the model explicitly even for a provider opencode knows: its
    # bundled models.dev catalogue may not list this exact id yet, and without an
    # entry opencode has no context/output limits and cannot compact a long
    # session.
    model_entry: dict = {"name": spec.model}
    if thinking_options:
        model_entry["options"] = thinking_options

    opencode_id = provider.opencode_id
    env: dict[str, str] = {}

    if provider.ambient_auth:
        # Vertex: opencode uses application-default credentials, and needs the
        # project and location in its environment rather than in the config.
        from viral_bench.founder.vertex import vertex_subprocess_env  # noqa: PLC0415

        env.update(vertex_subprocess_env())
        block = {opencode_id: {"models": {spec.model: model_entry}}}
        return OpencodeTarget(f"{opencode_id}/{spec.model}", block, env)

    if opencode_id:
        key_env = OPENCODE_KEY_ENV.get(opencode_id)
        if key_env and credential.api_key:
            env[key_env] = credential.api_key
        options: dict = {}
        # Only override the base URL when the user actually set one; otherwise
        # let opencode use the provider's own default, which it keeps current.
        if credential.base_url and credential.base_url != provider.base_url:
            options["baseURL"] = credential.base_url
        entry: dict = {"models": {spec.model: model_entry}}
        if options:
            entry["options"] = options
        return OpencodeTarget(f"{opencode_id}/{spec.model}", {opencode_id: entry}, env)

    # No opencode built-in: synthesise an OpenAI-compatible provider. This is how
    # Ollama, vLLM, LM Studio and any private gateway work here.
    options = {"baseURL": credential.base_url}
    if credential.api_key:
        options["apiKey"] = credential.api_key
    block = {
        provider.id: {
            "npm": "@ai-sdk/openai-compatible",
            "name": provider.name,
            "options": options,
            "models": {spec.model: model_entry},
        }
    }
    return OpencodeTarget(f"{provider.id}/{spec.model}", block, env)
