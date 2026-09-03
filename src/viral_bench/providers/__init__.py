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

"""Model providers: one interface over every backend ViralBench can call.

ViralBench ships **no model and no default provider**. You bring a key for
whichever provider you want to benchmark, name it once, and all four model
stages -- founder, crowd, rubric grader, autorater -- go through this package.

    from viral_bench.providers import make_client, resolve

    spec = resolve("openai/gpt-5-mini")
    client = make_client(spec)
    reply = client.generate([{"role": "user", "content": "hello"}])

See :mod:`viral_bench.providers.spec` for the registry and the canonical message
format, :mod:`viral_bench.providers.credentials` for where keys are read from,
and :mod:`viral_bench.providers.opencode` for how the founder stage passes a
provider through to the opencode CLI.
"""

from viral_bench.providers.client import (
    MAX_ATTEMPTS,
    Adapter,
    LLMClient,
    Reply,
    ToolCall,
    UnsupportedCapabilityError,
    check_support,
    make_client,
)
from viral_bench.providers.credentials import (
    Credential,
    MissingCredentialError,
    has_credential,
)
from viral_bench.providers.errors import (
    AuthError,
    BadRequestError,
    ModelError,
    RateLimitError,
    TransientError,
    backoff_seconds,
    classify,
)
from viral_bench.providers.spec import (
    FULL,
    PROVIDERS,
    Capability,
    ModelSpec,
    ProviderSpec,
    UnknownProviderError,
    provider_ids,
    resolve,
)

__all__ = [
    # spec
    "PROVIDERS",
    "FULL",
    "Capability",
    "ModelSpec",
    "ProviderSpec",
    "UnknownProviderError",
    "provider_ids",
    "resolve",
    # client
    "MAX_ATTEMPTS",
    "Adapter",
    "LLMClient",
    "Reply",
    "ToolCall",
    "UnsupportedCapabilityError",
    "check_support",
    "make_client",
    # credentials
    "Credential",
    "MissingCredentialError",
    "has_credential",
    # errors
    "AuthError",
    "BadRequestError",
    "ModelError",
    "RateLimitError",
    "TransientError",
    "backoff_seconds",
    "classify",
]
