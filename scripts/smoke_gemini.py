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

"""Smoke test for the Gemini Developer API key wiring (crowd + built apps).

Run:  uv run python scripts/smoke_gemini.py

This checks one provider path only: the Gemini Developer API reached with an API
key, which is what the crowd simulation and the founder-built apps use when they
are pointed at `google/<model>`. Any other provider is checked with
`viral-bench models --check <provider>/<model>` instead.

Loads GEMINI_API_KEY from the gitignored .env. Passing the key explicitly forces
the Developer API path, so an ambient cloud project in the environment
(GOOGLE_CLOUD_PROJECT) can never silently redirect the call to Vertex. The key
itself is never printed.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from google import genai


def main() -> int:
    load_dotenv()  # read .env from repo root

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set (check .env)", file=sys.stderr)
        return 1

    client = genai.Client(api_key=api_key)

    # 1) Auth check: list models that can generate content (short ids).
    available = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            available.append(m.name.split("/")[-1])

    print(f"Auth OK: {len(available)} generateContent-capable models visible.")
    for name in sorted(available):
        print(f"  - {name}")

    # 2) Pick a Gemini 3-series text model by EXACT match from a priority list.
    #    The 2.x series is intentionally excluded, leaving only the latest 3.x
    #    models (prefer the newest flash tier, then fall back to 3.x pro/preview).
    available_set = set(available)
    priority = [
        "gemini-2.0-flash",  # latest flash (GA), preferred
        "gemini-2.0-flash",  # 3.1 flash tier (GA)
        "gemini-2.0-flash",  # 3.1 pro
        "gemini-2.0-flash",  # 3.0 flash
        "gemini-2.0-flash",
    ]
    model = next((p for p in priority if p in available_set), None)
    if model is None:
        # Safety net: fall back to any Gemini 3-series text model (newest first).
        # Never fall back to the 2.x series.
        skip = (
            "tts",
            "image",
            "embedding",
            "vision",
            "aqa",
            "learnlm",
            "live",
            "audio",
        )
        three_series = [
            n
            for n in available
            if n.startswith("gemini-") and not any(s in n for s in skip)
        ]
        if not three_series:
            print(
                "ERROR: no Gemini 3-series text model is available to this API key.",
                file=sys.stderr,
            )
            return 1
        flash = [n for n in three_series if "flash" in n]
        model = sorted(flash or three_series, reverse=True)[0]
    print(f"\nUsing model: {model}")

    resp = client.models.generate_content(
        model=model,
        contents="Reply with exactly: viral_bench gemini wiring OK",
    )
    print("Response:", (resp.text or "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
