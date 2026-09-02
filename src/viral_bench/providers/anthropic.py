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

"""Claude, natively -- over the public API or over Vertex ``rawPredict``.

The canonical message shape *is* Anthropic's, so this adapter does almost no
translation. What it does do is speak two transports behind one class:

* ``anthropic/<model>`` -- the public Messages API, authenticated with a key.
* ``google-vertex-anthropic/<model>`` -- the same request body posted to Vertex's
  ``rawPredict`` endpoint with an application-default-credentials bearer token
  and an ``anthropic_version`` field instead of a key.

Both are plain HTTPS POSTs of the same JSON, so they share one code path and the
``anthropic`` SDK is not a dependency. That is deliberate: it keeps the crowd's
isolated environment small, and it means a user who never touches Claude installs
nothing extra.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from viral_bench.providers.client import Adapter, Reply, ToolCall
from viral_bench.providers.errors import TransientError, from_status

#: Pinned by Anthropic's Vertex integration; not the model version.
VERTEX_ANTHROPIC_VERSION = "vertex-2023-10-16"

#: Public Messages API version header.
ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicAdapter(Adapter):
    """Anthropic Messages, over the public API or Vertex."""

    @property
    def _via_vertex(self) -> bool:
        return self.spec.provider.id == "google-vertex-anthropic"

    def _endpoint(self) -> tuple[str, dict[str, str]]:
        """Return ``(url, headers)`` for the configured transport."""
        if self._via_vertex:
            from viral_bench.founder.vertex import (  # noqa: PLC0415
                vertex_access_token,
                vertex_api_host,
                vertex_location,
                vertex_project,
            )

            project, location = vertex_project(), vertex_location()
            # Registry ids carry an "@default" suffix that the REST path does not.
            bare = self.model.split("@")[0]
            url = (
                f"https://{vertex_api_host(location)}/v1/projects/{project}"
                f"/locations/{location}/publishers/anthropic/models/{bare}:rawPredict"
            )
            return url, {
                "Authorization": f"Bearer {vertex_access_token()}",
                "Content-Type": "application/json; charset=utf-8",
                "x-goog-user-project": project,
            }

        base = self.credential.base_url.rstrip("/")
        return f"{base}/v1/messages", {
            "x-api-key": self.credential.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "Content-Type": "application/json; charset=utf-8",
        }

    @staticmethod
    def _clean(messages: list[dict]) -> list[dict]:
        """Drop the canonical-shape-only keys Anthropic's API rejects.

        A ``tool_use`` block carries our opaque ``metadata`` dict so a provider
        that needs per-call state can round-trip it. Anthropic is strict about
        unknown keys, so it is lifted back into the ``signature`` field it came
        from and the dict itself is removed.
        """
        cleaned = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                cleaned.append(message)
                continue
            blocks = []
            for block in content or []:
                if block.get("type") != "tool_use" or "metadata" not in block:
                    blocks.append(block)
                    continue
                copied = {k: v for k, v in block.items() if k != "metadata"}
                signature = (block.get("metadata") or {}).get("signature")
                if signature:
                    copied["signature"] = signature
                blocks.append(copied)
            cleaned.append({**message, "content": blocks})
        return cleaned

    def _body(
        self, messages: list[dict], tools: list[dict] | None, system: str
    ) -> bytes:
        # Anthropic requires max_tokens; 0 ("no ceiling") becomes its maximum
        # rather than being omitted, which the API would reject.
        payload: dict = {
            "max_tokens": self.max_tokens or 64000,
            "messages": self._clean(messages),
        }
        if self._via_vertex:
            # Vertex takes the model from the URL and the version from the body;
            # the public API is the other way round.
            payload["anthropic_version"] = VERTEX_ANTHROPIC_VERSION
        else:
            payload["model"] = self.model
        # Some Claude models reject `temperature` outright rather than ignoring
        # it, so it is only sent when explicitly asked for. Determinism where it
        # matters comes from repeated sampling, not from this knob.
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
        if isinstance(self.thinking, int) and self.thinking > 0:
            payload["thinking"] = {"type": "enabled", "budget_tokens": self.thinking}
        return json.dumps(payload).encode("utf-8")

    def _call(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        system: str = "",
    ) -> Reply:
        url, headers = self._endpoint()
        request = urllib.request.Request(  # noqa: S310 - fixed https endpoint
            url,
            data=self._body(messages, tools, system),
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                raw = json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                parsed = json.loads(exc.read().decode("utf-8", "replace"))
                detail = str(parsed.get("error", {}).get("message", "")).strip()
            except Exception:  # noqa: BLE001 - the body may not be JSON at all
                detail = ""
            raise from_status(exc.code, detail or str(exc.reason)) from exc
        except urllib.error.URLError as exc:
            raise TransientError(f"could not reach {url}: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            # A read timeout is NOT a URLError: `urlopen` wraps a *connect*
            # timeout, but a timeout while reading the response body surfaces as
            # a bare TimeoutError out of ssl.read. Left unconverted it sails past
            # the retry loop and aborts a run that a short wait would have saved.
            raise TransientError(
                f"read timed out after {self.timeout_s}s: {exc}"
            ) from exc
        return self._parse(raw)

    @staticmethod
    def _parse(raw: dict) -> Reply:
        text_parts, calls = [], []
        for block in raw.get("content") or []:
            kind = block.get("type")
            if kind == "text":
                text_parts.append(str(block.get("text", "")))
            elif kind == "tool_use":
                metadata = {}
                # Extended thinking attaches a signature that must be replayed
                # with the tool result or the next turn is rejected.
                if block.get("signature"):
                    metadata["signature"] = block["signature"]
                calls.append(
                    ToolCall(
                        id=str(block.get("id") or f"tc_{len(calls):03d}"),
                        name=str(block.get("name") or ""),
                        args=dict(block.get("input") or {}),
                        metadata=metadata,
                    )
                )
        return Reply(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            stop_reason=str(raw.get("stop_reason") or ""),
            usage=dict(raw.get("usage") or {}),
        )
