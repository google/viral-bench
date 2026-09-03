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

"""Carrying screenshots from a tool result to a multimodal model.

The crowd used to be blind. An agent's entire percept of an app was
``document.body.innerText`` plus an ARIA snapshot: the ``screenshot`` tool wrote a
PNG to disk for the human evidence trail and returned a *sentence about* the
screenshot, and every part the Gemini backend built was ``Part(text=...)``. No
image ever reached a model.

That is a measurement problem, not a missing nicety. Craft is 22% of the
ViralScore and its ``design`` facet asks how an app "looks and feels" -- and
canvas-rendered and image-output apps were being scored on that facet at the same
rate as plain-DOM apps, from a percept containing zero pixels. For a benchmark
about virality it is worse still, because visual appeal is a primary reason the
apps in question went viral at all.

Tools must return ``str`` (the agent framework requires it), so the image travels
as a marker inside that string and the model transport swaps it for a real image
part. Keeping the protocol here, rather than in the transport, means the tool
layer and the backend agree on one definition instead of two regexes.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "IMAGE_MARKER_RE",
    "mark_image",
    "split_images",
]

#: ``[[VB_IMAGE:/abs/path.png]]`` -- deliberately unlikely to occur in app text.
IMAGE_MARKER_RE = re.compile(r"\[\[VB_IMAGE:(?P<path>[^\]]+)\]\]\s*")


def mark_image(text: str, path: str | Path | None) -> str:
    """Tag ``text`` as carrying the image at ``path`` (no-op without a path)."""
    if not path:
        return text
    return f"[[VB_IMAGE:{path}]]{text}"


def split_images(text: str) -> tuple[str, list[Path]]:
    """Return ``text`` with markers removed, plus the image paths they named.

    Paths that no longer exist are dropped rather than raising: a screenshot can
    be swept away by disk cleanup between the turn that took it and the turn that
    replays the conversation, and losing an image must never fail a run.
    """
    paths: list[Path] = []
    for match in IMAGE_MARKER_RE.finditer(text):
        candidate = Path(match.group("path"))
        if candidate.is_file():
            paths.append(candidate)
    return IMAGE_MARKER_RE.sub("", text), paths
