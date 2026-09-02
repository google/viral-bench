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

"""Tests for the upload fixtures the crowd attaches to apps under test.

These exist because a whole class of viral app opens with "drop in a photo /
PDF / screenshot", and the crowd previously had no upload action at all -- so
those apps could not be entered and the resulting review described the landing
page rather than the product.
"""

from __future__ import annotations

import struct
import zlib

from viral_bench.crowd.interaction.fixtures import (
    FIXTURE_DESCRIPTIONS,
    available_fixtures,
    fixture_path,
)


def test_every_advertised_fixture_can_be_produced() -> None:
    """The catalogue an agent is shown must match what actually exists.

    The docstring the agent reads is generated from FIXTURE_DESCRIPTIONS, so a
    name listed there and missing on disk becomes an agent asking for a file the
    tool cannot supply.
    """
    for name in FIXTURE_DESCRIPTIONS:
        path = fixture_path(name)
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_generated_png_is_structurally_valid() -> None:
    """A malformed PNG would fail inside the app, not in our code.

    Checked by inflating IDAT and confirming the pixel count matches the header
    rather than just asserting the file is non-empty.
    """
    data = fixture_path("photo.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (320, 240)

    idat = data.index(b"IDAT")
    length = struct.unpack(">I", data[idat - 4 : idat])[0]
    raw = zlib.decompress(data[idat + 4 : idat + 4 + length])
    # One filter byte per scanline, then 3 bytes per pixel.
    assert len(raw) == height * (1 + width * 3)


def test_generated_pdf_has_a_page() -> None:
    data = fixture_path("document.pdf").read_bytes()
    assert data.startswith(b"%PDF-")
    assert b"/Type /Page" in data
    assert data.rstrip().endswith(b"%%EOF")


def test_unknown_fixture_is_rejected() -> None:
    """An agent naming a file that does not exist gets a clear error, not a crash."""
    import pytest

    with pytest.raises(KeyError):
        fixture_path("definitely-not-a-fixture.xyz")


def test_catalogue_lists_every_fixture() -> None:
    listing = available_fixtures()
    for name, description in FIXTURE_DESCRIPTIONS.items():
        assert name in listing
        assert description in listing


def test_upload_tool_is_offered_to_web_agents() -> None:
    """A tool the agent is never handed might as well not exist."""
    from viral_bench.crowd.interaction.toolkit import AppInteractionToolkit

    doc = AppInteractionToolkit.upload_file.__doc__ or ""
    assert "{fixtures}" not in doc, "the fixture catalogue was never substituted in"
    assert "photo.png" in doc
