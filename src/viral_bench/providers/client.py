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

"""The one interface every stage uses to call a model.

``generate(messages, tools=..., system=...) -> Reply``. That is the whole
contract, and it is the shape the rubric grader already spoke -- this module
generalises it so the crowd, the autorater and the grader all share one client
instead of three.

The canonical message format
----------------------------
Messages use the **Anthropic content-block shape**, and adapters translate to and
from it:

    {"role": "user", "content": "plain text"}
    {"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "image", "source": {"type": "base64",
                                     "media_type": "image/png",
                                     "data": "<b64>"}},
        {"type": "tool_result", "tool_use_id": "tc_000", "content": "..."},
    ]}
    {"role": "assistant", "content": [
        {"type": "text", "text": "..."},
        {"type": "tool_use", "id": "tc_000", "name": "click", "input": {...}},
    ]}

Tools are Anthropic-shaped too: ``{"name", "description", "input_schema"}``.

That shape was chosen because it is the most expressive of the three -- it can
carry a tool call, its result and an image in one turn without inventing a
convention -- so translation is lossy in only one direction, and never on the way
in.

Opaque reasoning metadata
-------------------------
Some providers attach per-tool-call state that must be handed straight back on
the next turn or the conversation is rejected (recent Gemini ``thought_signature``,
Anthropic's thinking-block signature). Normalising those away is the single
easiest way to break multi-turn tool calling, so :class:`ToolCall` carries an
opaque ``metadata`` dict that the layer never inspects and always round-trips.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from viral_bench.providers import credentials
from viral_bench.providers.errors import (
    BadRequestError,
    ModelError,
    backoff_seconds,
)
from viral_bench.providers.spec import Capability, ModelSpec, resolve

#: Attempts per call, including the first. A grade or a crowd turn is expensive
#: to redo, so a transient failure must not cost the run. A persistent one should
#: still surface rather than hang forever.
MAX_ATTEMPTS = 4


@dataclass
class ToolCall:
    """One tool the model asked for."""

    id: str
    name: str
    args: dict
    #: Provider-opaque state to hand back verbatim with this call's result.
    #: Never inspected here. See the module docstring.
    metadata: dict = field(default_factory=dict)


@dataclass
class Reply:
    """One model turn: prose, plus any tool calls it wants run."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """What every adapter provides."""

    #: The ``provider/model`` string this client was built for.
    model_id: str

    def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        system: str = "",
    ) -> Reply: ...


class Adapter:
    """Shared behaviour for adapters: config, retries, error classification.

    An adapter implements :meth:`_call` and inherits the retry loop. Keeping the
    loop here rather than in each adapter is what makes backoff behaviour the
    same everywhere -- it was previously three different curves in three files.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        temperature: float | None = None,
        max_tokens: int = 8192,
        timeout_s: float = 180.0,
        thinking: str | int | None = None,
        json_mode: bool = False,
    ) -> None:
        # `max_tokens=0` means "send no ceiling and let the provider decide".
        # The autorater wants exactly that: a truncated rating is a silently
        # wrong score, which is worse than a slow one.
        self.spec = spec
        self.model = spec.model
        self.model_id = spec.qualified
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.thinking = thinking
        self.json_mode = json_mode
        self.credential = credentials.resolve(spec.provider)

    def generate(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        system: str = "",
    ) -> Reply:
        last: ModelError | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self._call(messages, tools=tools, system=system)
            except BadRequestError:
                # The request is malformed, so sending it again changes nothing.
                raise
            except ModelError as exc:
                last = exc
                if attempt < MAX_ATTEMPTS - 1:
                    time.sleep(backoff_seconds(attempt, exc))
        raise ModelError(
            f"{self.model_id}: gave up after {MAX_ATTEMPTS} attempts: {last}"
        ) from last

    def _call(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        system: str = "",
    ) -> Reply:
        raise NotImplementedError

    def ping(self) -> None:
        """Prove the model is reachable and callable, or raise.

        Called before a long stage starts. Without it, a model the account
        cannot use burns the whole design timeout inside a retry loop somewhere
        and then reports as a *build failure* -- a setup problem laundered into
        evidence about the model.

        The default is a one-token generation, which works everywhere and costs
        almost nothing. An adapter with a free reachability endpoint should
        override this and use it.

        Raises:
            ModelError: with the provider's own message, unmodified.
        """
        probe = self.__class__(
            self.spec, max_tokens=1, timeout_s=min(self.timeout_s, 30.0)
        )
        probe._call([{"role": "user", "content": "ping"}])


class UnsupportedCapabilityError(RuntimeError):
    """A stage needs something the chosen provider cannot do."""


def check_support(spec: ModelSpec, needed: Capability, *, stage: str) -> None:
    """Refuse a provider that cannot do what ``stage`` requires.

    Called before a run starts. The alternative is discovering three hours into
    a sweep that the crowd's model cannot see the screenshots it was sent, and
    scoring a whole arm against a design dimension it never observed.

    Raises:
        UnsupportedCapabilityError: if any needed capability is absent.
    """
    missing = needed & ~spec.provider.capabilities
    if not missing:
        return
    names = ", ".join(flag.name.lower() for flag in Capability if flag & missing)
    raise UnsupportedCapabilityError(
        f"the {stage} stage needs {names}, which {spec.provider.name} "
        f"({spec.qualified}) does not support. Choose another provider for this "
        f"stage, or run `viral-bench models` to see what each one can do."
    )


def make_client(model: str | ModelSpec, **kwargs) -> LLMClient:
    """Build a client for a ``provider/model`` string.

    Adapters are imported lazily: the Google adapter pulls in ``google-genai``
    and the OpenAI-compatible one pulls in ``openai``, and a user who brought an
    Anthropic key should not need either installed.
    """
    spec = model if isinstance(model, ModelSpec) else resolve(model)
    transport = spec.provider.transport

    if transport == "openai_compat":
        from viral_bench.providers.openai_compat import OpenAICompatAdapter

        return OpenAICompatAdapter(spec, **kwargs)
    if transport == "anthropic":
        from viral_bench.providers.anthropic import AnthropicAdapter

        return AnthropicAdapter(spec, **kwargs)
    if transport == "google":
        from viral_bench.providers.google_genai import GoogleAdapter

        return GoogleAdapter(spec, **kwargs)
    raise ValueError(  # pragma: no cover - guarded by the registry
        f"provider {spec.provider.id!r} names unknown transport {transport!r}"
    )
