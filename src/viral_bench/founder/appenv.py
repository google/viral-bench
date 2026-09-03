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

"""Resolve the environment variables handed to a founder-built app at run time.

A founder-built app may need to call a model when the crowd (or a human)
*uses* it. That credential is kept strictly separate from how the pipeline itself
authenticates, for one reason: built apps are untrusted, model-generated code,
and must never see the key the benchmark runs on.

So the app is given its own, and it is given it in a **provider-neutral** shape:

    VIRALBENCH_APP_LLM_BASE_URL   an OpenAI-compatible endpoint
    VIRALBENCH_APP_LLM_API_KEY    the key for it
    VIRALBENCH_APP_LLM_MODEL      the model id to name

Every provider ViralBench supports is reachable through an OpenAI-compatible
endpoint, so a generated app has exactly ONE code path regardless of which
provider the harness itself is using. That matters more than it sounds: the app
is written by a model that has to get this right first time, with no chance to
debug, and "use the OpenAI SDK against this base URL" is the single most
widely-known way to call an LLM there is.

Which model the apps get is configured as the ``app`` stage (``viral-bench
init``). Hold it FIXED across a comparison: it is part of the environment the
apps are measured in, not part of what is being measured.

Values are read from the process environment first, then from the repo ``.env``,
so a new key can be added to ``.env`` with no exports. Point
``VIRAL_BENCH_ENV_FILE`` at a different file to override (used by tests).
"""

from __future__ import annotations

from pathlib import Path

from viral_bench.env import env_file, read_key

__all__ = [
    "APP_API_KEY_VAR",
    "APP_BASE_URL_VAR",
    "APP_MODEL_VAR",
    "DEFAULT_ENV_MAP",
    "app_llm_env",
    "env_file",
    "read_key",
    "resolve_app_env",
]

#: The container-visible names a built app reads. Deliberately prefixed and
#: provider-neutral: an app that hard-codes one vendor's variable name stops
#: working the moment the benchmark is run against another provider.
APP_BASE_URL_VAR = "VIRALBENCH_APP_LLM_BASE_URL"
APP_API_KEY_VAR = "VIRALBENCH_APP_LLM_API_KEY"
APP_MODEL_VAR = "VIRALBENCH_APP_LLM_MODEL"

#: Extra ``{container_var: host_var}`` pass-throughs, for anything a specific
#: deployment wants apps to see. Empty by default, since the LLM triple above is
#: resolved from the provider layer rather than from a name mapping.
DEFAULT_ENV_MAP: dict[str, str] = {}


def app_llm_env(*, env_file_path: Path | None = None) -> dict[str, str]:
    """Resolve the provider-neutral LLM triple for a built app.

    Returns ``{}`` when no ``app`` model is configured, or when its provider has
    no credential -- a missing value is a silent no-op rather than an injected
    empty string, because an app that sees a blank key behaves differently from
    one that sees none.

    A caveat worth knowing when reading results: every built app is required to
    start and pass its health check WITHOUT a key, degrading the model-backed
    feature gracefully. That means a missing or exhausted key does not look like
    a failure -- the app answers from its fallback and reads as perfectly
    healthy, while the crowd rates canned output instead of the real feature.
    Check ``viral-bench doctor`` before a scored run rather than after it.
    """
    from viral_bench import config as _config  # noqa: PLC0415 - avoids a cycle
    from viral_bench.providers import UnknownProviderError, resolve  # noqa: PLC0415
    from viral_bench.providers.credentials import (  # noqa: PLC0415
        MissingCredentialError,
    )
    from viral_bench.providers.credentials import resolve as resolve_credential

    configured = _config.stage_model("app")
    if not configured:
        return {}
    try:
        spec = resolve(configured)
        credential = resolve_credential(spec.provider)
    except (UnknownProviderError, MissingCredentialError):
        return {}
    if not credential.base_url:
        return {}
    return {
        APP_BASE_URL_VAR: credential.base_url,
        APP_API_KEY_VAR: credential.api_key,
        APP_MODEL_VAR: spec.model,
    }


def resolve_app_env(
    env_map: dict[str, str] | None = None,
    *,
    env_file_path: Path | None = None,
) -> dict[str, str]:
    """Resolve ``{container_var: value}`` to inject into an app container.

    The provider-neutral LLM triple is always resolved (see :func:`app_llm_env`).
    ``env_map`` adds plain ``{container_var: host_var}`` pass-throughs on top,
    and a mapping whose host var is unset is skipped.
    """
    resolved = app_llm_env(env_file_path=env_file_path)
    mapping = DEFAULT_ENV_MAP if env_map is None else env_map
    for container_name, host_name in mapping.items():
        val = read_key(host_name, env_file_path=env_file_path)
        if val:
            resolved[container_name] = val
    return resolved
