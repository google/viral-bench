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

"""Sample files a crowd agent can upload into an app under test.

A large share of viral web apps open with "drop in a photo / PDF / screenshot".
Until now the crowd had no way to satisfy that: there was no upload action at
all, so those apps were structurally un-enterable and an agent's review described
the landing page rather than the product. Giving agents an upload verb is only
half the fix -- they also need something to upload, and it has to be something
the app can plausibly process.

The fixtures are GENERATED, not checked in as binaries, for three reasons: the
repo stays free of opaque blobs, there is no licensing question about the sample
content, and each file is tiny and deterministic so a run is reproducible. They
are written once into a cache directory and reused.

Everything here is stdlib-only on purpose. Pillow or reportlab would give nicer
samples, but the crowd venv should not grow a dependency just to make a 1 KB PNG.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

__all__ = ["FIXTURE_DESCRIPTIONS", "fixture_path", "fixture_dir", "available_fixtures"]

#: What each fixture is, in the words the agent is shown. Keep these accurate --
#: an agent told it is uploading "a photo of a room" will judge the result
#: against that expectation.
FIXTURE_DESCRIPTIONS: dict[str, str] = {
    "photo.png": "a small photograph-like image (a room interior, 320x240)",
    "screenshot.png": "a screenshot of a simple web UI with a header and a button",
    "document.pdf": "a one-page PDF containing a short paragraph of text",
    "data.csv": "a small CSV table of 5 rows with headers",
    "notes.md": "a short Markdown document with a heading and a list",
    "config.json": "a small JSON object with nested fields",
}


def fixture_dir() -> Path:
    """Directory holding the generated fixtures (created on first use)."""
    path = Path(__file__).resolve().parents[3].parent / "data" / "fixtures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _png(path: Path, width: int, height: int, palette: str) -> None:
    """Write a minimal, valid PNG with a simple generated pattern."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type 0 for each scanline
        for x in range(width):
            if palette == "room":
                # Warm horizontal bands: floor, wall, ceiling.
                if y > height * 0.72:
                    rgb = (122, 96, 72)
                elif y < height * 0.18:
                    rgb = (238, 236, 230)
                else:
                    rgb = (206, 198, 184)
                if abs(x - width // 3) < 3 and y > height * 0.3:
                    rgb = (90, 70, 55)  # a doorframe, so it is not a flat field
            else:
                # A "UI": light page, dark header bar, a button rectangle.
                rgb = (247, 248, 250)
                if y < height * 0.14:
                    rgb = (38, 44, 56)
                elif (
                    height * 0.45 < y < height * 0.58 and width * 0.1 < x < width * 0.4
                ):
                    rgb = (52, 120, 246)
            raw.extend(rgb)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def _pdf(path: Path, text: str) -> None:
    """Write a minimal one-page PDF (hand-built; no dependency needed)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
        + f"startxref\n{xref_at}\n%%EOF\n".encode()
    )
    path.write_bytes(bytes(out))


def _ensure(name: str) -> Path:
    path = fixture_dir() / name
    if path.exists() and path.stat().st_size > 0:
        return path
    if name == "photo.png":
        _png(path, 320, 240, "room")
    elif name == "screenshot.png":
        _png(path, 400, 300, "ui")
    elif name == "document.pdf":
        _pdf(path, "ViralBench sample document. The quick brown fox jumps over it.")
    elif name == "data.csv":
        path.write_text(
            "name,role,city,signups,active\n"
            "Ada,engineer,London,412,true\n"
            "Grace,admiral,New York,377,true\n"
            "Alan,researcher,Cambridge,290,false\n"
            "Katherine,mathematician,Hampton,455,true\n"
            "Edsger,professor,Austin,198,false\n",
            encoding="utf-8",
        )
    elif name == "notes.md":
        path.write_text(
            "# Project notes\n\n"
            "A short sample document.\n\n"
            "- First item\n- Second item\n- Third item\n\n"
            "## Next steps\n\nShip it.\n",
            encoding="utf-8",
        )
    elif name == "config.json":
        path.write_text(
            '{\n  "name": "sample",\n  "version": 2,\n'
            '  "features": {"search": true, "export": false},\n'
            '  "tags": ["alpha", "beta"]\n}\n',
            encoding="utf-8",
        )
    else:
        raise KeyError(f"unknown fixture: {name!r}")
    return path


def fixture_path(name: str) -> Path:
    """Return the path to a named fixture, generating it if needed."""
    if name not in FIXTURE_DESCRIPTIONS:
        raise KeyError(
            f"unknown fixture {name!r}; available: {sorted(FIXTURE_DESCRIPTIONS)}"
        )
    return _ensure(name)


def available_fixtures() -> str:
    """A one-line-per-fixture catalogue, for an agent-facing tool docstring."""
    return "\n".join(
        f"  - {name}: {desc}" for name, desc in sorted(FIXTURE_DESCRIPTIONS.items())
    )
