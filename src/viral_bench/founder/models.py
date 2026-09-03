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

"""Choosing the model the founder builds with.

The founder is the only stage whose model is meant to vary -- it is the thing
under test. The crowd that uses the app, the grader that inspects it and the
autorater that rates it are configured separately and held fixed across a
comparison, so swapping the founder model cannot move the yardstick it is
measured against. ``tests/test_models.py`` asserts that separation.

There is no curated list of model ids here, and that is deliberate. A benchmark
that ships a hard-coded roster is out of date the week after it is published, and
it quietly implies the models on the list are the ones worth testing. Instead you
name any ``provider/model`` string for any provider in
:data:`viral_bench.providers.PROVIDERS`:

    viral-bench found --model openai/gpt-5-mini
    viral-bench found --model anthropic/claude-sonnet-4-5
    viral-bench found --model openrouter/meta-llama/llama-3.3-70b-instruct
    viral-bench found --model ollama/qwen3-coder

``viral-bench models`` lists the providers, says which ones you have credentials
for, and points at each provider's own catalogue for the ids it serves. That
catalogue is always more current than anything this file could hold.

What the founder needs from a model
-----------------------------------
Tool calling, and enough context to hold a growing codebase. The founder stage
runs through the `opencode <https://opencode.ai>`_ CLI, so in practice any model
opencode can drive will work. See :mod:`viral_bench.providers.opencode` for how a
provider is passed through, including providers opencode has no built-in for.
"""

from __future__ import annotations

from viral_bench.providers import (
    PROVIDERS,
    Capability,
    ModelSpec,
    UnknownProviderError,
    has_credential,
)
from viral_bench.providers import resolve as _resolve

#: Backwards-compatible alias: callers catch this, the provider layer raises it.
UnknownModelError = UnknownProviderError

#: What the founder stage cannot do without.
REQUIRED = Capability.TOOLS


def resolve_model(model: str) -> ModelSpec:
    """Resolve a ``--model`` string to the provider that serves it.

    Raises:
        UnknownModelError: if the string has no provider prefix, or names a
            provider that is not registered.
    """
    return _resolve(model)


def is_supported(model: str) -> bool:
    """True if ``model`` resolves to a registered provider."""
    try:
        _resolve(model)
    except UnknownModelError:
        return False
    return True


def provider_for(model: str) -> str:
    """Return the provider id serving ``model``."""
    return _resolve(model).provider.id


def transport_for(model: str) -> str:
    """Return the wire protocol serving ``model``.

    One of ``openai_compat``, ``anthropic`` or ``google``. Call sites needing a
    provider-specific request shape branch on this rather than on the provider
    id, so adding a provider that speaks an existing protocol needs no change.
    """
    return _resolve(model).provider.transport


def normalize_model_id(model: str) -> str:
    """Return the bare model id, dropping a recognised ``provider/`` prefix.

    Never raises: an unrecognised prefix is left in place. That makes it right
    for the live path (where the provider has already been resolved) and WRONG for
    comparing historical build records, which carry retired prefixes -- see
    ``scripts/build_fleet.py:_short_model``, which strips unconditionally for
    exactly that reason.
    """
    provider, sep, bare = model.partition("/")
    if sep and provider in PROVIDERS and bare:
        return bare
    return model


def describe_models() -> str:
    """A human-readable provider table for ``viral-bench models``."""
    lines = [
        "ViralBench has no default model. Pass --model <provider>/<model>.",
        "",
        f"{'PROVIDER':<26} {'CREDENTIAL':<12} HOW TO GET ONE",
    ]
    for provider in PROVIDERS.values():
        if provider.ambient_auth:
            state = "cloud auth"
        elif not provider.key_env:
            state = "none needed"
        else:
            state = "found" if has_credential(provider) else "MISSING"
        lines.append(f"{provider.id:<26} {state:<12} {provider.signup_hint}")
    lines.extend(
        [
            "",
            "The model ids each provider serves are listed in its own catalogue --",
            "always more current than anything shipped here. Any id that provider",
            "accepts works, e.g. `--model openai/gpt-5-mini`.",
            "",
            "Check that a specific model is callable:",
            "  viral-bench models --check <provider>/<model>",
        ]
    )
    return "\n".join(lines)
