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

"""A CAMEL model backend for Gemini via the native ``google-genai`` SDK.

Why this exists: CAMEL 0.2.78's built-in ``GeminiModel`` talks to Gemini through
the OpenAI-compatibility endpoint, which cannot round-trip the ``thought_signature``
that recent Gemini requires for multi-turn function calling. In practice that breaks
*every* OASIS crowd agent -- not just the ones with app-interaction tools, because
OASIS's own social actions (like/repost/comment/...) are also tool calls -- the
moment an agent makes a second tool call it gets an HTTP 400
("Function call is missing a thought_signature").

This backend swaps the transport to the native ``google-genai`` SDK, which
preserves thought signatures across turns, while keeping CAMEL's
``BaseModelBackend`` contract (it subclasses ``GeminiModel`` for its config/token
handling and overrides only ``_run``/``_arun``). It translates between CAMEL's
OpenAI-style messages/tools and genai's ``Content``/``FunctionDeclaration`` format,
and persists each tool call's ``thought_signature`` (keyed by the tool-call id it
hands back to CAMEL) so it can re-attach it when that call reappears in history.

It serves the two Google provider ids -- ``google`` (Developer API, key auth) and
``google-vertex`` (Vertex AI, application-default credentials). Which of the two
is in play is read off the resolved :class:`~viral_bench.providers.ModelSpec`,
not off an environment flag: a ``CROWD_USE_VERTEX`` opt-OUT used to default the
crowd onto Vertex, so a user holding a perfectly good ``GEMINI_API_KEY`` failed
with an ADC error that never mentioned the flag responsible. Every other provider
goes through :mod:`viral_bench.crowd.sim.unified_model`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

from camel.models.gemini_model import GeminiModel
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
from viral_bench.providers import ModelSpec
from viral_bench.providers.errors import (
    AuthError,
    ModelError,
    RateLimitError,
    TransientError,
    backoff_seconds,
    classify,
)

_LOG = logging.getLogger("viral_bench.crowd.gemini_native")

#: Filler for the user turn appended when a conversation would otherwise end on
#: a model turn, which Vertex refuses outright. Deliberately contentless: the
#: agent's actual instruction is already in the history, and anything with
#: meaning here would be an extra prompt that the Developer API path never sent.
_CONTINUE_TEXT = "Continue."


def _retry_delay(
    error: ModelError, attempt: int, *, resilient: int, limited: int
) -> float | None:
    """Seconds to wait before retrying, or ``None`` to give up on this turn.

    Which kind of failure this is used to be decided by three lists of substrings
    matched against the exception text -- transient, rate-limited, fatal-auth --
    written against Google's phrasings. They were right for Google and wrong for
    everyone else, and even for Google they read a message rather than the status
    code sitting on the exception. ``google.genai`` errors carry ``.code``, which
    :func:`viral_bench.providers.errors.classify` uses first, falling back to text
    only when there is none.

    A 429 from a quota is not a 0.8-second problem: the old fixed schedule
    (0.8 s, 1.6 s, give up) meant that under any sustained throttle every agent
    in the round was skipped, and the run reported an app nobody engaged with.
    Rate limits therefore keep a much larger attempt budget, and
    :func:`viral_bench.providers.errors.backoff_seconds` gives them the
    exponential curve while other transients keep the short linear one.
    """
    budget = limited if isinstance(error, RateLimitError) else resilient
    if attempt < budget and isinstance(error, TransientError):
        return backoff_seconds(attempt, error)
    return None


#: Gemini finish reasons mapped to the OpenAI vocabulary CAMEL expects.
_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "BLOCKLIST": "content_filter",
    "SPII": "content_filter",
    "MALFORMED_FUNCTION_CALL": "stop",
    "OTHER": "stop",
}


def _finish_reason(resp, has_tool_calls: bool) -> str:
    """Translate the model's real finish reason into OpenAI's vocabulary.

    Truncation and safety blocks have to be distinguishable from a clean stop.
    Hard-coding "stop" made both invisible: a reply cut off at the token limit
    was indistinguishable from a complete one, so the corpus could not answer
    whether the cap was binding.
    """
    candidates = getattr(resp, "candidates", None) or []
    raw = getattr(candidates[0], "finish_reason", None) if candidates else None
    name = getattr(raw, "name", None) or (str(raw) if raw is not None else "")
    name = name.rsplit(".", 1)[-1].upper()
    mapped = _FINISH_REASONS.get(name)
    if mapped == "length" or mapped == "content_filter":
        return mapped
    return "tool_calls" if has_tool_calls else (mapped or "stop")


class GeminiNativeModel(GeminiModel):
    """Gemini backend using google-genai (round-trips ``thought_signature``)."""

    def __init__(
        self,
        spec: ModelSpec,
        model_config_dict: dict[str, Any] | None = None,
        api_key: str | None = None,
        url: str | None = None,
        token_counter=None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            spec.model,
            model_config_dict,
            api_key,
            url,
            token_counter,
            timeout,
            max_retries,
            **kwargs,
        )
        self.spec = spec
        if spec.provider.ambient_auth:
            # Same model and parameters, different transport and quota pool: the
            # Developer API throttles this workload at ~10-12 concurrent, Vertex
            # does not. Measured on the replicate-3 sweep, crowd throughput on
            # the Developer API plateaued at ~11 runs/h with the concurrency knee
            # at 10-12, and at 16 and 20 a pass returned ZERO completed runs; the
            # same model on Vertex ran 10, 30 and 60 concurrent calls at 100% OK
            # with p50 latency flat at 0.7-0.8 s. That is a reason to CHOOSE
            # `google-vertex/...` for a sweep, which is now what naming it means
            # -- it is no longer a default the Developer API user must opt out of.
            #
            # vertex_genai_client() also forces plain TLS, which the Vertex path
            # needs on corp machines (its mTLS path fails with a missing-pyOpenSSL
            # error).
            from viral_bench.founder.vertex import vertex_genai_client

            self._genai = vertex_genai_client()
        else:
            from google import genai

            key = api_key or os.environ.get("GEMINI_API_KEY")
            self._genai = genai.Client(api_key=key)
        # tool_call_id -> the thought_signature / function name from the genai
        # response that produced it, so we can re-attach on the next turn.
        self._sig_by_id: dict[str, bytes | None] = {}
        self._name_by_id: dict[str, str] = {}
        # Bounded retries for transient failures before a turn is skipped, so one
        # agent's hiccup never aborts the whole OASIS round (see _empty_completion).
        self._resilient_retries = 2
        #: Rate limits get their own, much longer budget: 6 attempts with
        #: exponential backoff is up to ~60 s of waiting, which is the right
        #: order for a quota window. Skipping instead turns throttling into
        #: measured indifference.
        self._rate_limit_retries = 6

    @property
    def token_limit(self) -> int:
        """The INPUT context window, decoupled from the ``max_tokens`` output cap.

        CAMEL's ``BaseModelBackend.token_limit`` returns ``max_tokens`` when set,
        and ChatAgent uses that value to size its context window. The crowd sets
        ``max_tokens`` only to cap *output* (short posts), so inheriting it as the
        context limit makes ChatAgent truncate multi-turn history aggressively --
        which can slice a ``function_call`` turn away from its preceding user /
        function-response turn and make Gemini reject the request ("function call
        turn comes immediately after a user turn..."). Report the model's real
        (~1M-token) input window instead so tool-calling history stays intact.
        """
        return 1_048_576

    # -- OpenAI -> genai ----------------------------------------------------

    @staticmethod
    def _split_system(messages) -> tuple[str, list]:
        system_parts, convo = [], []
        for m in messages:
            if m.get("role") == "system":
                if m.get("content"):
                    system_parts.append(m["content"])
            else:
                convo.append(m)
        return "\n\n".join(system_parts), convo

    def _tool_config(self, tools):
        from google.genai import types

        decls = []
        for tool in tools or []:
            fn = tool.get("function", tool)
            decls.append(
                types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters_json_schema=fn.get("parameters")
                    or {"type": "object", "properties": {}},
                )
            )
        return [types.Tool(function_declarations=decls)] if decls else None

    def _to_config(self, tools, system_text: str):
        from google.genai import types

        cfg = self.model_config_dict or {}
        kwargs: dict[str, Any] = {
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            )
        }
        if system_text:
            kwargs["system_instruction"] = system_text
        if cfg.get("temperature") is not None:
            kwargs["temperature"] = cfg["temperature"]
        if cfg.get("max_tokens"):
            kwargs["max_output_tokens"] = cfg["max_tokens"]
        # How hard the crowd is allowed to think before answering.
        #
        # recent Gemini spends thinking tokens by default, which is why a trivial
        # call to 3.7-flash takes 7.9 s against flash-lite's 0.6 s. For a crowd
        # of 30 agents making ~460 calls a run that is the difference between a
        # sweep finishing overnight and not finishing, so it has to be a knob --
        # and "how much deliberation does a good judge need" is a real question
        # about the instrument, not just a cost dial.
        level = cfg.get("thinking_level")
        if level:
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
            except Exception:  # noqa: BLE001 - older SDK / model without the field
                pass
        tool_cfg = self._tool_config(tools)
        if tool_cfg:
            kwargs["tools"] = tool_cfg
        return types.GenerateContentConfig(**kwargs)

    def _to_contents(self, convo):
        from google.genai import types

        # Decide which screenshots survive the attachment budget BEFORE building
        # any Content: the parts list is copied into the pydantic model, so
        # appending to it afterwards would silently do nothing.
        #
        # The conversation is replayed on every turn, so attaching every shot an
        # agent ever took would grow request size quadratically over a trial.
        # Newest wins; older ones degrade to the text that described them, which
        # keeps the transcript coherent.
        keep = keep_newest_images(convo)

        contents = []
        for m in convo:
            role = m.get("role")
            if role == "user":
                contents.append(
                    types.Content(
                        role="user", parts=[types.Part(text=m.get("content") or "")]
                    )
                )
            elif role == "assistant":
                tool_calls = m.get("tool_calls") or []
                if tool_calls:
                    parts = []
                    for tc in tool_calls:
                        fn = tc["function"]
                        cid = tc.get("id", "")
                        args = fn.get("arguments") or "{}"
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        part = types.Part(
                            function_call=types.FunctionCall(name=fn["name"], args=args)
                        )
                        sig = self._sig_by_id.get(cid)
                        if sig is not None:
                            part.thought_signature = sig
                        parts.append(part)
                    contents.append(types.Content(role="model", parts=parts))
                elif m.get("content"):
                    contents.append(
                        types.Content(
                            role="model", parts=[types.Part(text=m["content"])]
                        )
                    )
            elif role == "tool":
                cid = m.get("tool_call_id", "")
                name = self._name_by_id.get(cid, m.get("name", "tool"))
                content = m.get("content")
                result = content if isinstance(content, str) else json.dumps(content)
                result, images = split_images(result)
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=name, response={"result": result}
                                )
                            )
                        ],
                    )
                )
                # Screenshots go in their OWN user turn, never appended to the
                # function_response turn above.
                #
                # THIS IS THE 400. Vertex rejects a final content whose parts are
                # [function_response, image] with "Requests ending with a model
                # turn are not supported" -- a message that names the wrong
                # problem, since that content's role is "user". Reproduced
                # directly against gemini-2.0-flash: [fr, img] as the last
                # content fails, the same content followed by any other turn
                # passes, and fr alone followed by a user turn of images passes
                # with one image and with three.
                #
                # That misleading text is why the earlier fix missed. It guarded
                # `contents[-1].role == "model"`, which is never true here, so the
                # guard could not fire -- and the bug survived it. Measured after
                # that fix landed: 54,522 rejections in builds/sweep_logs, still
                # 26,131 of them two days later, against 10,697 rate limits. It
                # was the single largest cause of lost agent turns, and every one
                # costs that agent its action for the round.
                #
                # A separate content is preferred over merely reordering the parts
                # (image first also passes) because a function_response content
                # carrying only function_response parts is the documented shape;
                # the ordering workaround depends on a quirk of the validator that
                # produced this error message in the first place.
                shots = []
                for path in images:
                    if str(path) not in keep:
                        continue
                    try:
                        shots.append(
                            types.Part.from_bytes(
                                data=path.read_bytes(), mime_type="image/png"
                            )
                        )
                    except OSError:
                        # A swept-away screenshot must never break a run.
                        continue
                if shots:
                    contents.append(types.Content(role="user", parts=shots))

        # Vertex rejects a request whose final turn genuinely IS the model's,
        # with "400 INVALID_ARGUMENT: Requests ending with a model turn are not
        # supported." The Developer API accepts the same request and simply
        # continues, so this only became reachable when the crowd changed
        # transport -- nothing about the conversation itself changed.
        #
        # It is not a rare edge, because a failed turn is recorded as an
        # assistant message (turns.SKIPPED_TURN_CONTENT, which has to stay
        # non-empty for CAMEL). So one failure leaves the NEXT request ending
        # on a model turn, which fails for this reason, which appends another
        # assistant message: the agent cascades into silence.
        #
        # This guard is correct and still needed, but it was never the whole
        # story: Vertex emits the SAME message for a trailing
        # [function_response, image] user turn, which this cannot see. See the
        # tool branch above -- that shape was the large majority of the loss.
        #
        # The appended turn carries no instruction -- it exists to satisfy the
        # transport, not to steer the agent -- so the request stays semantically
        # the one the Developer API was already answering.
        if contents and getattr(contents[-1], "role", None) == "model":
            contents.append(
                types.Content(role="user", parts=[types.Part(text=_CONTINUE_TEXT)])
            )

        return contents

    # -- genai -> OpenAI ----------------------------------------------------

    def _to_completion(self, resp) -> ChatCompletion:
        candidates = getattr(resp, "candidates", None) or []
        cand = candidates[0] if candidates else None
        tool_calls: list[ChatCompletionMessageToolCall] = []
        text_parts: list[str] = []
        if cand is not None and cand.content and cand.content.parts:
            for part in cand.content.parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    cid = "call_" + uuid.uuid4().hex[:24]
                    self._sig_by_id[cid] = getattr(part, "thought_signature", None)
                    self._name_by_id[cid] = fc.name
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=cid,
                            type="function",
                            function=Function(
                                name=fc.name,
                                arguments=json.dumps(dict(fc.args or {})),
                            ),
                        )
                    )
                elif getattr(part, "text", None):
                    text_parts.append(part.text)

        message = ChatCompletionMessage(
            role="assistant",
            content="".join(text_parts) or None,
            tool_calls=tool_calls or None,
        )
        # Report the real reason, not a flattering one. This was hard-coded to
        # "stop", so a reply cut off at max_tokens looked exactly like a complete
        # one -- which is why nobody could tell whether the token cap was ever
        # binding, and why raising it was guesswork.
        finish = _finish_reason(resp, bool(tool_calls))
        if finish == "length":
            _LOG.warning(
                "crowd model response hit the output token limit and was "
                "truncated (raise or null out simulation.model.max_tokens in "
                "config/crowd.yaml)"
            )
        usage = None
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage = CompletionUsage(
                prompt_tokens=getattr(um, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(um, "candidates_token_count", 0) or 0,
                total_tokens=getattr(um, "total_token_count", 0) or 0,
            )
        return ChatCompletion(
            id="chatcmpl-" + uuid.uuid4().hex[:24],
            choices=[Choice(index=0, message=message, finish_reason=finish)],
            created=int(time.time()),
            model=str(self.model_type),
            object="chat.completion",
            usage=usage,
        )

    # -- BaseModelBackend hooks --------------------------------------------

    def _run(self, messages, response_format=None, tools=None) -> ChatCompletion:
        system_text, convo = self._split_system(messages)
        contents = self._to_contents(convo)
        config = self._to_config(tools, system_text)
        last_error: ModelError | None = None
        TURN_STATS["turns"] += 1
        attempt = 0
        while True:
            try:
                resp = self._genai.models.generate_content(
                    model=str(self.model_type), contents=contents, config=config
                )
                return self._to_completion(resp)
            except Exception as exc:  # noqa: BLE001 - classified below
                last_error = classify(exc)
                if isinstance(last_error, AuthError):
                    raise
                delay = _retry_delay(
                    last_error,
                    attempt,
                    resilient=self._resilient_retries,
                    limited=self._rate_limit_retries,
                )
                if delay is None:
                    break
                time.sleep(delay)
                attempt += 1
        return self._skip_turn(last_error, contents)

    async def _arun(self, messages, response_format=None, tools=None) -> ChatCompletion:
        system_text, convo = self._split_system(messages)
        contents = self._to_contents(convo)
        config = self._to_config(tools, system_text)
        last_error: ModelError | None = None
        TURN_STATS["turns"] += 1
        attempt = 0
        while True:
            try:
                resp = await self._genai.aio.models.generate_content(
                    model=str(self.model_type), contents=contents, config=config
                )
                return self._to_completion(resp)
            except Exception as exc:  # noqa: BLE001 - classified below
                last_error = classify(exc)
                # Auth/config errors are not one app's fault: fail the run fast
                # rather than silently zero-out every agent's engagement.
                if isinstance(last_error, AuthError):
                    raise
                delay = _retry_delay(
                    last_error,
                    attempt,
                    resilient=self._resilient_retries,
                    limited=self._rate_limit_retries,
                )
                if delay is None:
                    break
                await asyncio.sleep(delay)
                attempt += 1
        return self._skip_turn(last_error, contents)

    def _skip_turn(
        self, error: ModelError | None, contents: list | None = None
    ) -> ChatCompletion:
        record_skip(error)
        detail = str(error).splitlines()[0][:200] if error else "unknown error"
        _LOG.warning(
            "crowd agent turn failed (%s); skipping this agent's action for the "
            "round so the run can continue.",
            detail,
        )
        _dump_rejected_shape(error, contents)
        return empty_completion(str(self.model_type))


#: Where to record the shape of a request Vertex refused. Unset = record nothing.
CROWD_SHAPE_DUMP_ENV = "CROWD_DUMP_REJECTED_SHAPE"


def _dump_rejected_shape(exc: Exception | None, contents: list | None) -> None:
    """Append the role sequence of a rejected request, if asked to.

    DIAGNOSTIC ONLY, and off unless ``CROWD_DUMP_REJECTED_SHAPE`` names a file,
    so it cannot perturb a sweep in progress.

    It exists for one specific open bug. Vertex rejects roughly **3.2% of all
    agent turns** with ``400 ... Requests ending with a model turn are not
    supported`` -- 12,200 of 383,275 turns measured over 773 runs, with only 30
    attributable to rate limiting. Each one costs that agent its action for the
    round, so the loss is real but silent: it lands in ``skipped``, not in
    ``skipped_rate_limited``, and nothing on disk says which conversation shape
    provoked it.

    What makes it puzzling, and why this dump is the way in: ``_to_contents``
    already appends a trailing user turn whenever the last content is the
    model's, and that guard demonstrably fires for both shapes that should
    produce this error (a conversation ending in an assistant message, and one
    ending in a tool result). So the request being rejected is a shape we have
    not identified, and it cannot be reasoned out from the code alone.

    Recording the role sequence -- not the content, which is large and
    sensitive -- is enough to identify it. Turn it on for a handful of runs
    OUTSIDE a scoring sweep:

        CROWD_DUMP_REJECTED_SHAPE=/tmp/shapes.jsonl viral-bench crowd-run ...

    then look for what the last few roles have in common. Fix it BETWEEN sweeps,
    never during one: it changes what the crowd asks the model, so runs from
    before and after are not comparable.
    """
    path = os.environ.get(CROWD_SHAPE_DUMP_ENV)
    if not path or contents is None:
        return
    if "model turn" not in str(exc or "").lower():
        return
    try:
        roles = [getattr(c, "role", "?") for c in contents]
        parts = [
            [
                type(p).__name__ if getattr(p, "text", None) is None else "text"
                for p in (getattr(c, "parts", None) or [])
            ]
            for c in contents[-3:]
        ]
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"n": len(roles), "roles": roles, "tail_parts": parts})
                + "\n"
            )
    except Exception:  # noqa: BLE001 - a diagnostic must never break a run
        pass
