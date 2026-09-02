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

"""Gemini, over the Developer API (key) or Vertex AI (application credentials).

One adapter, two auth modes, chosen by which provider id was named:

* ``google/<model>`` -- the Gemini Developer API with ``GEMINI_API_KEY``.
* ``google-vertex/<model>`` -- Vertex AI with application-default credentials,
  for people already billing a cloud project.

Neither is a default. Both are ordinary entries in the provider registry, and a
ViralBench run that never names one never imports ``google-genai`` at all.

Two Gemini-specific details this adapter exists to preserve:

* **Thought signatures.** recent Gemini attaches an opaque ``thought_signature`` to
  a function call that must be handed back with the result, or the next turn is
  rejected. It rides in :attr:`ToolCall.metadata` and is replayed on the way out.
* **Automatic function calling is off.** The caller owns the tool loop -- the
  crowd and the grader both need to see and record each call -- so the SDK's
  helpful habit of running tools itself is disabled.
"""

from __future__ import annotations

import base64

from viral_bench.providers.client import Adapter, Reply, ToolCall
from viral_bench.providers.errors import classify

#: Gemini names the model turn "model", not "assistant".
_ROLE = {"assistant": "model", "user": "user"}


def to_gemini_contents(messages: list[dict]) -> list[dict]:
    """Translate canonical messages into Gemini ``contents``.

    Tool results become ``function_response`` parts and stay in the user turn,
    which is where Gemini wants them -- unlike OpenAI, no message splitting is
    needed here.
    """
    contents: list[dict] = []
    for message in messages:
        role = _ROLE.get(message.get("role", "user"), "user")
        content = message.get("content")
        if isinstance(content, str):
            if content:
                contents.append({"role": role, "parts": [{"text": content}]})
            continue

        parts: list[dict] = []
        for block in content or []:
            kind = block.get("type")
            if kind == "text" and block.get("text"):
                parts.append({"text": block["text"]})
            elif kind == "image":
                source = block.get("source") or {}
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": source.get("media_type", "image/png"),
                            "data": base64.b64decode(source.get("data", "")),
                        }
                    }
                )
            elif kind == "tool_use":
                part: dict = {
                    "function_call": {
                        "name": block.get("name", ""),
                        "args": block.get("input") or {},
                    }
                }
                signature = (block.get("metadata") or {}).get("thought_signature")
                if signature:
                    part["thought_signature"] = signature
                parts.append(part)
            elif kind == "tool_result":
                body = block.get("content")
                parts.append(
                    {
                        "function_response": {
                            "name": block.get("name") or block.get("tool_use_id", ""),
                            "response": body
                            if isinstance(body, dict)
                            else {"result": body},
                        }
                    }
                )
        if parts:
            contents.append({"role": role, "parts": parts})
    return contents


class GoogleAdapter(Adapter):
    """Gemini via ``google-genai``, keyed or on Vertex."""

    def _client(self):
        if getattr(self, "_cached", None) is None:
            if self.spec.provider.ambient_auth:
                from viral_bench.founder.vertex import (
                    vertex_genai_client,  # noqa: PLC0415
                )

                self._cached = vertex_genai_client()
            else:
                try:
                    from google import genai  # noqa: PLC0415 - optional heavy import
                except ImportError as exc:  # pragma: no cover - install problem
                    raise RuntimeError(
                        "the google-genai package is required for Gemini: "
                        "`uv add google-genai`"
                    ) from exc
                self._cached = genai.Client(api_key=self.credential.api_key)
        return self._cached

    def _config(self, tools: list[dict] | None, system: str):
        from google.genai import types  # noqa: PLC0415 - optional heavy import

        declarations = [
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=tool.get("input_schema")
                or {"type": "object", "properties": {}},
            )
            for tool in (tools or [])
        ]
        config: dict = {
            "system_instruction": system or None,
            "tools": [types.Tool(function_declarations=declarations)]
            if declarations
            else None,
            # The caller owns the tool loop; see the module docstring.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if self.temperature is not None:
            config["temperature"] = self.temperature
        if self.max_tokens:
            config["max_output_tokens"] = self.max_tokens
        if self.json_mode:
            config["response_mime_type"] = "application/json"
        if self.thinking:
            config["thinking_config"] = types.ThinkingConfig(
                thinking_level=str(self.thinking)
            )
        return types.GenerateContentConfig(**config)

    def ping(self) -> None:
        """Use ``count_tokens``: free, non-mutating, and still proves auth."""
        try:
            self._client().models.count_tokens(model=self.model, contents="ping")
        except Exception as exc:  # noqa: BLE001 - re-raised as a ModelError
            raise classify(exc) from exc

    def _call(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        system: str = "",
    ) -> Reply:
        try:
            response = self._client().models.generate_content(
                model=self.model,
                contents=to_gemini_contents(messages),
                config=self._config(tools, system),
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a ModelError
            raise classify(exc) from exc
        return self._parse(response)

    @staticmethod
    def _parse(response) -> Reply:
        text_parts, calls = [], []
        stop_reason = ""
        for candidate in getattr(response, "candidates", None) or []:
            stop_reason = str(getattr(candidate, "finish_reason", "") or stop_reason)
            for part in getattr(candidate.content, "parts", None) or []:
                call = getattr(part, "function_call", None)
                if call is not None:
                    metadata = {}
                    signature = getattr(part, "thought_signature", None)
                    if signature:
                        metadata["thought_signature"] = signature
                    calls.append(
                        ToolCall(
                            id=str(getattr(call, "id", "") or f"tc_{len(calls):03d}"),
                            name=str(call.name or ""),
                            args=dict(call.args or {}),
                            metadata=metadata,
                        )
                    )
                elif getattr(part, "text", None):
                    text_parts.append(part.text)
        usage = getattr(response, "usage_metadata", None)
        return Reply(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            stop_reason=stop_reason,
            usage={
                "input_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0)
                if usage
                else 0,
            },
        )
