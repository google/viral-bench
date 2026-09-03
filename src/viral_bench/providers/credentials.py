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

"""Where a provider's credential comes from, and how to say it is missing.

One resolution order, used by every stage:

    process environment  >  the repo's .env file

That is deliberately shallow. A credential is not the kind of thing that should
come from a YAML file checked into a repo, so ``config/`` has no say in it -- it
picks *which model* to use, never *which key*.

Base URLs follow the same order, under ``<PROVIDER>_BASE_URL`` with the provider
id upper-cased and hyphens turned into underscores (``OPENROUTER_BASE_URL``,
``GOOGLE_VERTEX_BASE_URL``). That is the escape hatch for a proxy, a gateway, or
a self-hosted server that speaks someone else's wire format.
"""

from __future__ import annotations

from dataclasses import dataclass

from viral_bench.env import read_key
from viral_bench.providers.spec import ProviderSpec


class MissingCredentialError(RuntimeError):
    """A provider was selected but its credential is not set anywhere."""


@dataclass(frozen=True)
class Credential:
    """A resolved credential for one provider."""

    #: The API key, or ``""`` when the provider authenticates some other way.
    api_key: str
    #: The base URL to call, already accounting for any override.
    base_url: str
    #: True when the provider uses ambient cloud credentials and there is no key.
    ambient: bool


def base_url_env(provider: ProviderSpec) -> str:
    """The environment variable that overrides this provider's base URL."""
    return provider.id.upper().replace("-", "_") + "_BASE_URL"


def has_credential(provider: ProviderSpec) -> bool:
    """True if this provider could be used right now, without raising.

    Ambient-auth providers report ``True`` here without probing the cloud: a
    real check costs a network round trip, which belongs in ``viral-bench
    doctor``, not in the listing that decides what to show in a menu.
    """
    if provider.ambient_auth:
        return True
    if not provider.key_env:
        return True  # a local server, e.g. Ollama
    return bool(read_key(provider.key_env))


def resolve(provider: ProviderSpec) -> Credential:
    """Resolve the credential and base URL for ``provider``.

    Raises:
        MissingCredentialError: if the provider needs a key and none is set, or
            needs a base URL and none is configured.
    """
    base_url = read_key(base_url_env(provider)) or provider.base_url

    if provider.ambient_auth:
        return Credential(api_key="", base_url=base_url, ambient=True)

    api_key = read_key(provider.key_env) if provider.key_env else ""
    if provider.key_env and not api_key:
        raise MissingCredentialError(
            f"{provider.name} needs {provider.key_env}, which is not set in the "
            f"environment or in .env.\n"
            f"  Get a key: {provider.signup_hint}\n"
            f"  Then either `export {provider.key_env}=...` or add it to .env "
            f"(run `viral-bench init` to be walked through it)."
        )
    if not base_url:
        raise MissingCredentialError(
            f"{provider.name} has no base URL. Set {base_url_env(provider)} to "
            f"the endpoint to call. {provider.signup_hint}"
        )
    return Credential(api_key=api_key or "", base_url=base_url, ambient=False)
