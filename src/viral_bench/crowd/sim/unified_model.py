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

"""A CAMEL model backend for every provider that is not Google.

The crowd used to be a Gemini crowd: one backend, one SDK, one account. This is
the other half of making it provider-agnostic -- it takes whatever
:func:`viral_bench.providers.make_client` returns (OpenAI, Anthropic, OpenRouter,
Groq, Together, a local Ollama, ...) and presents it to CAMEL as an ordinary
``BaseModelBackend``. Gemini keeps its own backend,
:mod:`viral_bench.crowd.sim.gemini_native`, because ``thought_signature``
round-tripping and the Vertex request-shape workarounds it carries are worth far
more than the symmetry of deleting it.

Three translations happen here, in this order:

1. **CAMEL's OpenAI messages -> the canonical content-block shape.**
   :func:`from_openai_messages` and :func:`from_openai_tools` are the exact
   inverse of ``providers.openai_compat.to_openai_messages`` /
   ``to_openai_tools``. Writing them as a matched pair rather than as a third
   independent translator is the whole point, and the round trip through both is
   pinned by a test so the pair cannot silently drift into two dialects.
2. **Screenshot markers -> real image parts.** A tool result reaches us as a
   string carrying ``[[VB_IMAGE:/path.png]]`` (see
   :mod:`viral_bench.crowd.interaction.imagery`), under the same newest-wins
   budget the Gemini path uses. The crowd's design verdicts -- 22% of the score --
   are worthless from a model that cannot see.
3. **The provider's Reply -> an OpenAI ChatCompletion**, which is what CAMEL and
   OASIS consume.

Retries are NOT repeated here. :class:`viral_bench.providers.client.Adapter` owns
the retry loop precisely so that backoff is identical for every stage, and
wrapping it in a second loop would multiply the attempts rather than add
patience. What this class does own is the crowd's own requirement: a turn that
fails after all of that must be RECORDED and SKIPPED, never raised, or one
agent's bad minute aborts the whole OASIS round.
"""

from __future__ import annotations

import base64
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from camel.models.base_model import BaseModelBackend
from camel.types import ModelType
from camel.utils import BaseTokenCounter, OpenAITokenCounter
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

from viral_bench.crowd.interaction.imagery import split_images
from viral_bench.crowd.sim.turns import (
    TURN_STATS,
    empty_completion,
    keep_newest_images,
    record_skip,
)
from viral_bench.providers import ModelSpec, Reply, make_client
from viral_bench.providers.errors import AuthError, ModelError, classify

_LOG = logging.getLogger("viral_bench.crowd.unified_model")

#: The INPUT context window reported to CAMEL. See :meth:`UnifiedModel.token_limit`
#: for why it is one number rather than a per-provider table.
_CONTEXT_WINDOW_TOKENS = 1_048_576

#: Stop reasons from every provider we speak to, mapped onto the OpenAI values
#: CAMEL's ``ChatCompletion`` accepts. An unmapped value must never reach pydantic:
#: ``Choice.finish_reason`` is a Literal, so an honest passthrough of Anthropic's
#: ``end_turn`` would raise a ValidationError and cost the very turn it reports on.
_FINISH_REASONS = {
    "stop": "stop",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "eos": "stop",
    "length": "length",
    "max_tokens": "length",
    "model_length": "length",
    "content_filter": "content_filter",
    "refusal": "content_filter",
    "tool_calls": "tool_calls",
    "tool_use": "tool_calls",
    "function_call": "tool_calls",
}


def from_openai_tools(tools: list[dict] | None) -> list[dict] | None:
    """Translate OpenAI function tools into the canonical Anthropic shape.

    The inverse of ``providers.openai_compat.to_openai_tools``. CAMEL hands tools
    over already wrapped as ``{"type": "function", "function": {...}}``, but the
    bare inner form turns up too, so both are accepted -- the same leniency
    ``gemini_native._tool_config`` needs for the same reason.
    """
    if not tools:
        return None
    return [
        {
            "name": (tool.get("function", tool))["name"],
            "description": tool.get("function", tool).get("description", ""),
            "input_schema": tool.get("function", tool).get("parameters")
            or {"type": "object", "properties": {}},
        }
        for tool in tools
    ]


def _image_block(path: Path, media_type: str = "image/png") -> dict | None:
    """A canonical base64 image block for ``path``, or None if it cannot be read.

    A screenshot swept away by disk cleanup between the turn that took it and a
    later turn that replays the conversation must never break a run.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def _data_url_block(url: str) -> dict:
    """Turn an OpenAI ``image_url`` data URL back into a canonical image block."""
    header, _, payload = url.partition(",")
    media_type = "image/png"
    if header.startswith("data:"):
        media_type = header[len("data:") :].split(";", 1)[0] or media_type
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": payload},
    }


def _content_blocks(content) -> list[dict]:
    """Canonical text/image blocks for one OpenAI message's ``content`` field."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict] = []
    for part in content or []:
        kind = part.get("type")
        if kind == "text" and part.get("text"):
            blocks.append({"type": "text", "text": part["text"]})
        elif kind == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                blocks.append(_data_url_block(url))
    return blocks


def from_openai_messages(
    messages: list[dict], metadata_by_id: dict[str, dict] | None = None
) -> tuple[list[dict], str]:
    """Translate CAMEL's OpenAI messages into canonical messages + system text.

    The inverse of ``providers.openai_compat.to_openai_messages``, which splits
    one canonical turn into its ``role: "tool"`` messages followed by whatever
    prose and images remain. This puts them back together: a run of tool results,
    plus any user content immediately after it, becomes ONE canonical user turn.

    Screenshot markers in a tool result are replaced by real image blocks, subject
    to the newest-wins budget. Unlike the Gemini path, an image may stay in the
    same turn as the ``tool_result`` it came from: the split there works around a
    Vertex validator that rejects that exact shape, and neither the OpenAI nor the
    Anthropic wire format has the problem.

    Args:
        messages: CAMEL's OpenAI-shaped conversation.
        metadata_by_id: Provider-opaque state to re-attach to a tool call when it
            reappears in history, keyed by tool-call id. See
            :attr:`UnifiedModel._meta_by_id`.
    """
    keep = keep_newest_images(messages)
    metadata_by_id = metadata_by_id or {}
    system_parts: list[str] = []
    canonical: list[dict] = []
    pending: list[dict] = []  # blocks accumulating into the next user turn

    def flush() -> None:
        if pending:
            canonical.append({"role": "user", "content": list(pending)})
            pending.clear()

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")

        if role == "system":
            if content:
                system_parts.append(content)
            continue

        if role == "tool":
            text = content if isinstance(content, str) else json.dumps(content)
            text, images = split_images(text)
            # No ``name`` key here, even though the canonical shape tolerates one
            # for the Gemini adapter's benefit: Anthropic rejects unknown fields
            # inside a content block outright, and this backend never routes to
            # Gemini.
            pending.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id", ""),
                    "content": text,
                }
            )
            for path in images:
                if str(path) not in keep:
                    continue
                block = _image_block(path)
                if block is not None:
                    pending.append(block)
            continue

        if role == "assistant":
            flush()
            blocks = _content_blocks(content)
            for call in message.get("tool_calls") or []:
                blocks.append(_tool_use_block(call, metadata_by_id))
            if blocks:
                canonical.append({"role": "assistant", "content": blocks})
            continue

        pending.extend(_content_blocks(content))

    flush()
    return canonical, "\n\n".join(system_parts)


def _tool_use_block(call: dict, metadata_by_id: dict[str, dict]) -> dict:
    """One assistant tool call, canonical, with its opaque state re-attached."""
    fn = call.get("function", {})
    args = fn.get("arguments") or "{}"
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            # A model can emit malformed JSON arguments. Replaying the call with
            # empty arguments still preserves the turn's structure, which is what
            # the next request needs; dropping the block instead orphans the tool
            # result that answers it.
            args = {}
    cid = call.get("id", "")
    block = {
        "type": "tool_use",
        "id": cid,
        "name": fn.get("name", ""),
        "input": args,
    }
    # Some providers attach per-call state that must be handed straight back or
    # the next turn is rejected -- Gemini's thought_signature, Anthropic's
    # thinking-block signature. ``providers.client`` says this is round-tripped
    # verbatim and never inspected, and ``providers.google_genai`` reads it from
    # exactly this key. It is absent for every provider this backend serves today
    # (neither adapter fills it unless extended thinking is on), which is the only
    # reason it is safe to pass through: the Anthropic adapter posts these blocks
    # as-is, so an unknown field here would be a 400 rather than an ignored key.
    metadata = metadata_by_id.get(cid)
    if metadata:
        block["metadata"] = metadata
    return block


def _finish_reason(reply: Reply) -> str:
    """Translate the provider's real stop reason into OpenAI's vocabulary.

    Truncation and safety blocks have to stay distinguishable from a clean stop.
    The Gemini path hard-coded "stop" for years, so a reply cut off at the token
    limit looked exactly like a complete one and nobody could tell whether the cap
    was ever binding. The same mistake is available here for free; it is not made.
    """
    mapped = _FINISH_REASONS.get(str(reply.stop_reason or "").strip().lower())
    if mapped in ("length", "content_filter"):
        return mapped
    return "tool_calls" if reply.tool_calls else (mapped or "stop")


class UnifiedModel(BaseModelBackend):
    """A CAMEL backend over any provider in the registry except Google."""

    def __init__(
        self,
        spec: ModelSpec,
        model_config_dict: dict[str, Any] | None = None,
        token_counter: BaseTokenCounter | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
    ) -> None:
        config = dict(model_config_dict or {})
        super().__init__(
            spec.model,
            config,
            api_key=None,  # credentials are the provider layer's job, not CAMEL's
            url=None,
            token_counter=token_counter,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.spec = spec
        client_kwargs: dict[str, Any] = {"temperature": config.get("temperature")}
        # The crowd's "no output cap" means *send no cap*, and on Gemini that is
        # exactly what happens. It cannot be honoured here: Anthropic requires
        # max_tokens and rejects a value above the model's own ceiling, so there
        # is no number that means "unlimited". When the crowd sets no cap the
        # provider layer's default applies instead -- a real difference between
        # the two paths, and worth knowing before comparing arms across them.
        if config.get("max_tokens"):
            client_kwargs["max_tokens"] = config["max_tokens"]
        if config.get("thinking_level"):
            client_kwargs["thinking"] = config["thinking_level"]
        if timeout:
            client_kwargs["timeout_s"] = timeout
        self._client = make_client(spec, **client_kwargs)
        # tool_call_id -> the provider-opaque metadata that came back with it, so
        # it can be handed back when that call reappears in history. See
        # :func:`_tool_use_block`.
        self._meta_by_id: dict[str, dict] = {}

    @property
    def token_counter(self) -> BaseTokenCounter:
        """Tokens are counted with OpenAI's tokenizer, as CAMEL does elsewhere.

        It is the wrong tokenizer for most providers here, and that is acceptable:
        the count decides when to trim context, not what anything costs, and
        CAMEL's own ``OpenAICompatibleModel`` makes the same assumption for the
        same reason.
        """
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(ModelType.GPT_4O_MINI)
        return self._token_counter

    @property
    def token_limit(self) -> int:
        """The INPUT context window, decoupled from the ``max_tokens`` output cap.

        CAMEL's ``BaseModelBackend.token_limit`` returns ``max_tokens`` when set,
        and ChatAgent uses that value to size its context window. The crowd sets
        ``max_tokens`` only to cap *output* (short posts), so inheriting it as the
        context limit makes ChatAgent truncate multi-turn history aggressively --
        which slices a tool call away from the result that answers it and gets the
        whole request rejected.

        One large number rather than a per-provider table, because the trade is
        asymmetric. Over-reporting means we never trim, and a genuinely over-long
        request comes back as a 400 that lands in ``skipped_bad_request`` and says
        so in the run summary. Under-reporting silently mangles tool history and
        loses the turn with no evidence of why.
        """
        return _CONTEXT_WINDOW_TOKENS

    # -- BaseModelBackend hooks --------------------------------------------

    def _run(self, messages, response_format=None, tools=None) -> ChatCompletion:
        canonical, system = from_openai_messages(messages, self._meta_by_id)
        TURN_STATS["turns"] += 1
        try:
            reply = self._client.generate(
                canonical, tools=from_openai_tools(tools), system=system
            )
        except Exception as exc:  # noqa: BLE001 - classified in _failed
            return self._failed(exc)
        return self._to_completion(reply)

    async def _arun(self, messages, response_format=None, tools=None) -> ChatCompletion:
        """Async is the sync call: the provider clients have no async surface.

        OASIS runs a round's agents concurrently, so this is called from an event
        loop and a blocking HTTPS request here serialises that round. It is still
        the right shape for now -- ``providers`` is deliberately one synchronous
        interface, and the crowd's real concurrency limit is its semaphore -- but
        this is the seam to put a thread executor behind if a sweep ever shows the
        loop is what is binding.
        """
        return self._run(messages, response_format, tools)

    def _failed(self, exc: BaseException) -> ChatCompletion:
        """Record a turn the provider could not complete, and skip it.

        ``Adapter.generate`` retries transient failures itself and then raises a
        plain ``ModelError`` naming the attempt count, chaining the last real
        failure as ``__cause__``. That cause is the one carrying the status code,
        so it -- not the wrapper -- decides which ``TURN_STATS`` bucket this lands
        in. Without unwrapping it every exhausted retry would read as
        ``skipped_other``, and the summary could not tell throttling from a bug,
        which is the distinction the whole by-cause tally exists to make.

        Only the bare base class is unwrapped: every deliberate classification is
        a subclass, so a ``TransientError`` an adapter raised on purpose is taken
        at its word rather than second-guessed from whatever it chained.
        """
        underlying = exc
        if type(exc) is ModelError and exc.__cause__ is not None:
            underlying = exc.__cause__
        error = classify(underlying)
        if isinstance(error, AuthError):
            # Not one app's fault: a bad key would silently zero out every agent's
            # engagement, and the run would read as an app nobody wanted to use.
            raise error from exc
        record_skip(error)
        _LOG.warning(
            "crowd agent turn failed (%s); skipping this agent's action for the "
            "round so the run can continue.",
            str(error).splitlines()[0][:200] or "unknown error",
        )
        return empty_completion(str(self.model_type))

    def _to_completion(self, reply: Reply) -> ChatCompletion:
        tool_calls: list[ChatCompletionMessageToolCall] = []
        for index, call in enumerate(reply.tool_calls):
            cid = call.id or f"call_{uuid.uuid4().hex[:24]}"
            self._meta_by_id[cid] = dict(call.metadata or {})
            tool_calls.append(
                ChatCompletionMessageToolCall(
                    id=cid,
                    type="function",
                    function=Function(
                        name=call.name or f"tool_{index}",
                        arguments=json.dumps(call.args or {}),
                    ),
                )
            )
        finish = _finish_reason(reply)
        if finish == "length":
            _LOG.warning(
                "crowd model response hit the output token limit and was "
                "truncated (raise or null out simulation.model.max_tokens in "
                "config/crowd.yaml)"
            )
        usage = reply.usage or {}
        prompt_tokens = int(usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or 0)
        message = ChatCompletionMessage(
            role="assistant",
            content=reply.text or None,
            tool_calls=tool_calls or None,
        )
        return ChatCompletion(
            id="chatcmpl-" + uuid.uuid4().hex[:24],
            choices=[Choice(index=0, message=message, finish_reason=finish)],
            created=int(time.time()),
            model=str(self.model_type),
            object="chat.completion",
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
