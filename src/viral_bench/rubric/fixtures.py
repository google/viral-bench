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

"""Sample files the grader can upload, beyond the ones the crowd has.

**Additive, deliberately.** The crowd's set in
:mod:`viral_bench.crowd.interaction.fixtures` is left exactly as it is: adding a
file there would give this era's crowd agents an upload option every earlier run
lacked, and cross-era ViralScore comparisons would quietly stop meaning anything.
Same discipline as ``CodeInspectionToolkit.grader_tools``.

Only one file so far, and it exists because an item needs it. The crowd's
``photo.png`` is 320x240 and 816 bytes, which is small enough that re-encoding it
to JPEG makes it *bigger* -- excellent as an arithmetic-honesty trap, useless for
"at quality 50 the output is genuinely smaller". That claim needs a photograph
big enough to actually compress.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

#: Grader-only fixtures, described the way the grader is told about them.
GRADER_FIXTURE_DESCRIPTIONS: dict[str, str] = {
    "photo_large.png": (
        "a large photographic image (1024x768, noisy gradient) that genuinely "
        "compresses"
    ),
}


def fixture_dir() -> Path:
    path = Path(__file__).resolve().parents[3] / "data" / "fixtures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def _large_photo(path: Path, width: int = 1024, height: int = 768) -> None:
    """Write a large RGB PNG that behaves like a photograph under compression.

    Photograph-like matters. A flat or geometric image compresses so well that
    every build looks good; real photographic noise is what makes a quality
    setting show its effect. The pattern is a deterministic pseudo-random
    gradient, so the file is byte-identical on every machine and the grade is
    reproducible.
    """
    raw = bytearray()
    seed = 12345
    for y in range(height):
        raw.append(0)  # filter type 0
        for x in range(width):
            # A cheap LCG, mixed with a smooth gradient: locally noisy, globally
            # structured, which is what a JPEG encoder sees in a real photo.
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            noise = (seed >> 16) & 0x1F
            raw.append((x * 255 // width + noise) & 0xFF)
            raw.append((y * 255 // height + noise) & 0xFF)
            raw.append(((x + y) * 127 // (width + height) + noise) & 0xFF)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)


_BUILDERS = {"photo_large.png": _large_photo}


def grader_fixture_path(name: str) -> Path:
    """Path to a grader-only fixture, generating it on first use."""
    if name not in _BUILDERS:
        raise KeyError(f"unknown grader fixture {name!r}")
    path = fixture_dir() / name
    if not path.is_file():
        _BUILDERS[name](path)
    return path


def grader_fixtures() -> dict[str, Path]:
    """Every grader-only fixture, by name."""
    return {name: grader_fixture_path(name) for name in _BUILDERS}
