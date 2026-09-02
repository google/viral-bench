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

"""The universal adapter: anything that speaks the OpenAI chat-completions API.

This one adapter covers OpenAI itself plus OpenRouter, Groq, Together, Fireworks,
DeepSeek, xAI, Mistral, Ollama, vLLM, LM Studio and any private gateway -- they
differ only by base URL and key variable, which is why the long tail of providers
costs a registry line each rather than a module each.

Translation is between the canonical Anthropic content-block shape (see
:mod:`viral_bench.providers.client`) and OpenAI's flatter one. The awkward part is
tool results: Anthropic carries them as blocks inside a user turn, OpenAI wants a
separate message with ``role: "tool"``, so one canonical turn can expand into
several OpenAI messages.
"""

from __future__ import annotations

import json

from viral_bench.providers.client import Adapter, Reply, ToolCall
from viral_bench.providers.errors import classify, from_status


def _content_blocks(content) -> list[dict]:
    """Normalise a message's content to a block list."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content or [])


def to_openai_messages(messages: list[dict], system: str = "") -> list[dict]:
    """Translate canonical messages into OpenAI chat-completions messages."""
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})

    for message in messages:
        role = message.get("role", "user")
        blocks = _content_blocks(message.get("content"))

        # Tool results become their own `role: "tool"` messages and must come
        # before any remaining prose from the same canonical turn, because
        # OpenAI requires every tool_call_id to be answered before new content.
        for block in blocks:
            if block.get("type") == "tool_result":
                body = block.get("content")
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": body
                        if isinstance(body, str)
                        else json.dumps(body, default=str),
                    }
                )

        parts: list[dict] = []
        tool_calls: list[dict] = []
        for block in blocks:
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif kind == "image":
                source = block.get("source") or {}
                media = source.get("media_type", "image/png")
                data = source.get("data", "")
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{data}"},
                    }
                )
            elif kind == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    }
                )

        if not parts and not tool_calls:
            continue
        entry: dict = {"role": role}
        if parts:
            # A single text part is sent as a bare string: some OpenAI-compatible
            # servers (notably older local ones) reject the array form.
            entry["content"] = (
                parts[0]["text"]
                if len(parts) == 1 and parts[0]["type"] == "text"
                else parts
            )
        else:
            entry["content"] = None
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.append(entry)
    return out


def to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
    """Translate Anthropic-shaped tool schemas into OpenAI function tools."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema")
                or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


class OpenAICompatAdapter(Adapter):
    """Chat completions over any OpenAI-compatible endpoint."""

    def _client(self):
        if getattr(self, "_cached", None) is None:
            try:
                from openai import OpenAI  # noqa: PLC0415 - optional heavy import
            except ImportError as exc:  # pragma: no cover - install-time problem
                raise RuntimeError(
                    "the openai package is required for OpenAI-compatible "
                    "providers: `uv add openai`"
                ) from exc
            self._cached = OpenAI(
                api_key=self.credential.api_key or "not-needed",
                base_url=self.credential.base_url,
                timeout=self.timeout_s,
                max_retries=0,  # the Adapter owns retries, so backoff is uniform
            )
        return self._cached

    def _call(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        system: str = "",
    ) -> Reply:
        payload: dict = {
            "model": self.model,
            "messages": to_openai_messages(messages, system),
        }
        if self.max_tokens:
            payload["max_completion_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        converted = to_openai_tools(tools)
        if converted:
            payload["tools"] = converted
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = self._client().chat.completions.create(**payload)
        except Exception as exc:  # noqa: BLE001 - re-raised as a ModelError
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                raise from_status(status, str(exc)) from exc
            raise classify(exc) from exc

        return self._parse(response)

    @staticmethod
    def _parse(response) -> Reply:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return Reply(stop_reason="empty")
        message = choices[0].message
        calls = []
        for index, call in enumerate(getattr(message, "tool_calls", None) or []):
            raw_args = getattr(call.function, "arguments", "") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                # A model can emit malformed JSON arguments. Pass the raw string
                # through rather than dropping the call: the tool layer produces
                # a far more useful error than a silently missing action.
                args = {"_raw": raw_args}
            calls.append(
                ToolCall(
                    id=str(getattr(call, "id", "") or f"tc_{index:03d}"),
                    name=str(getattr(call.function, "name", "")),
                    args=args if isinstance(args, dict) else {"_value": args},
                )
            )
        usage = getattr(response, "usage", None)
        return Reply(
            text=(getattr(message, "content", None) or "").strip(),
            tool_calls=calls,
            stop_reason=str(getattr(choices[0], "finish_reason", "") or ""),
            usage={
                "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            },
        )
