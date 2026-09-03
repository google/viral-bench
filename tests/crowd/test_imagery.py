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

"""Tests for getting screenshots in front of the crowd's model.

The crowd was text-only: screenshots were written to disk for the human evidence
trail and every part sent to Gemini was ``Part(text=...)``. Craft is 22% of the
ViralScore and its ``design`` facet asks how an app looks, so canvas and
image-output apps were being rated on a percept with no pixels in it.
"""

from __future__ import annotations

import pytest

from viral_bench.crowd.interaction.imagery import mark_image, split_images


def test_marker_round_trips(tmp_path) -> None:
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\nnot-really-a-png")

    marked = mark_image("Saved screenshot to shot.png", shot)
    assert str(shot) in marked

    text, images = split_images(marked)
    assert text == "Saved screenshot to shot.png"
    assert images == [shot]


def test_marker_is_stripped_even_when_the_file_is_gone(tmp_path) -> None:
    """A swept-away screenshot must degrade to text, not leak a marker.

    Run directories are garbage-collected, so a shot can vanish between the turn
    that took it and a later turn that replays the conversation. The agent should
    then see a clean sentence, never ``[[VB_IMAGE:...]]``.
    """
    missing = tmp_path / "gone.png"
    text, images = split_images(mark_image("Saved a screenshot.", missing))
    assert images == []
    assert "VB_IMAGE" not in text
    assert text == "Saved a screenshot."


def test_no_path_is_a_no_op() -> None:
    assert mark_image("plain text", None) == "plain text"
    text, images = split_images("plain text")
    assert (text, images) == ("plain text", [])


def test_tool_result_becomes_an_image_part(tmp_path, monkeypatch) -> None:
    """The transport must turn the marker into a real image part.

    This is the actual fix: without it the marker is stripped and the model still
    sees only words about a picture.
    """
    pytest.importorskip(
        "camel.models",
        reason="needs camel-ai, which lives in .venv-crowd, not this env",
    )
    from viral_bench.crowd.sim.gemini_native import GeminiNativeModel

    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    backend = GeminiNativeModel.__new__(GeminiNativeModel)
    backend._sig_by_id = {}
    backend._name_by_id = {"call-1": "screenshot"}

    convo = [
        {"role": "user", "content": "try the app"},
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "screenshot",
            "content": mark_image("Saved screenshot to shot.png", shot),
        },
    ]
    contents = backend._to_contents(convo)

    # The tool result and its screenshot are two turns, not one: Vertex rejects a
    # trailing [function_response, image] content. See
    # test_a_screenshot_never_shares_a_turn_with_its_tool_result.
    result_parts, image_parts = contents[-2].parts, contents[-1].parts
    assert result_parts[0].function_response is not None
    assert len(image_parts) == 1, "the screenshot never became an image part"
    assert image_parts[0].inline_data.mime_type == "image/png"
    assert image_parts[0].inline_data.data == shot.read_bytes()

    # And the marker must not survive into the text the model reads.
    response = result_parts[0].function_response.response["result"]
    assert "VB_IMAGE" not in response


def test_image_attachments_are_capped(tmp_path) -> None:
    """History is replayed every turn, so attachments must be bounded.

    Without a cap, a trial that takes N screenshots sends O(N^2) image bytes over
    its lifetime.
    """
    pytest.importorskip(
        "camel.models",
        reason="needs camel-ai, which lives in .venv-crowd, not this env",
    )
    from viral_bench.crowd.sim.gemini_native import GeminiNativeModel
    from viral_bench.crowd.sim.turns import MAX_IMAGES_PER_REQUEST

    backend = GeminiNativeModel.__new__(GeminiNativeModel)
    backend._sig_by_id = {}
    backend._name_by_id = {}

    convo = []
    total = MAX_IMAGES_PER_REQUEST + 3
    for i in range(total):
        shot = tmp_path / f"shot{i}.png"
        shot.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i]) * 16)
        convo.append(
            {
                "role": "tool",
                "tool_call_id": f"c{i}",
                "name": "screenshot",
                "content": mark_image(f"shot {i}", shot),
            }
        )

    contents = backend._to_contents(convo)
    attached = [
        p
        for c in contents
        for p in c.parts
        if getattr(p, "inline_data", None) is not None
    ]
    assert len(attached) == MAX_IMAGES_PER_REQUEST

    # The newest shots are the ones kept.
    newest = (tmp_path / f"shot{total - 1}.png").read_bytes()
    assert any(p.inline_data.data == newest for p in attached)


# -- the 400 that ate 1.7% of every run's agent turns ------------------------ #


def test_a_screenshot_never_shares_a_turn_with_its_tool_result(tmp_path) -> None:
    """An image part must never sit in the same Content as a function_response.

    THIS IS THE BUG THAT COST THE MOST. Vertex rejects a final content whose
    parts are ``[function_response, image]`` with

        400 INVALID_ARGUMENT: Requests ending with a model turn are not supported

    which names the wrong problem -- that content's role is ``user``. Verified
    against the crowd model: the shape fails as the last content, passes
    when any turn follows it, and passes when the image is moved into its own
    user turn (with one image and with three).

    Because the message says "model turn", the guard written for it checked
    ``contents[-1].role == "model"`` and could never fire. It shipped, and the
    rejections carried on: 54,522 across the sweep logs, 26,131 of them two days
    AFTER that fix, against 10,697 rate limits. Every one costs an agent its
    action for the round and is recorded as an agent that chose to do nothing.
    """
    pytest.importorskip(
        "camel.models",
        reason="needs camel-ai, which lives in .venv-crowd, not this env",
    )
    from viral_bench.crowd.sim.gemini_native import GeminiNativeModel

    backend = GeminiNativeModel.__new__(GeminiNativeModel)
    backend._sig_by_id = {}
    backend._name_by_id = {}

    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    convo = [
        {"role": "user", "content": "look at it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "function": {"name": "screenshot", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "screenshot",
            "content": mark_image("took a shot", shot),
        },
    ]

    contents = backend._to_contents(convo)

    for content in contents:
        kinds = {
            "fn" if getattr(p, "function_response", None) is not None else "other"
            for p in content.parts
        }
        assert kinds in ({"fn"}, {"other"}), (
            "a function_response part must not share a Content with anything else"
        )

    # The image still reaches the model, in its own user turn, after the result.
    assert any(
        getattr(p, "inline_data", None) is not None for c in contents for p in c.parts
    )
    assert contents[-1].role == "user"


class _ApiError(Exception):
    """An SDK exception shaped like the real ones: a status code on the object.

    ``google.genai`` puts it on ``.code`` and ``openai`` on ``.status_code``, and
    :func:`viral_bench.providers.errors.classify` reads either. Reading the code
    rather than the message is the point: the buckets used to be decided by
    matching Google's phrasings, so an OpenAI 429 worded differently fell into the
    unattributable remainder.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def test_skipped_turns_record_why_not_just_how_many() -> None:
    """A lost turn must be attributable from the run summary alone.

    Only rate limiting was ever broken out, so every other cause pooled into an
    unattributable remainder -- and that remainder was 84% one fixable bug.
    Recovering it meant grepping 1,600 sweep logs, because the reason lived only
    in a WARNING line that the stall reaper's SIGKILL discards with the buffer.
    """
    from viral_bench.crowd.sim.turns import skip_reason

    assert skip_reason(_ApiError(429, "RESOURCE_EXHAUSTED")) == "skipped_rate_limited"
    assert (
        skip_reason(_ApiError(400, "INVALID_ARGUMENT. Requests ending with a model"))
        == "skipped_bad_request"
    )
    assert skip_reason(_ApiError(503, "UNAVAILABLE")) == "skipped_unavailable"
    assert skip_reason(Exception("boom")) == "skipped_other"
    assert skip_reason(None) == "skipped_other"

    # An exception carrying no status at all still falls back to its text, and
    # there a rate limit is only recognisable as a generic transient. That is the
    # price of not string-matching vendor phrasings, and it is why an adapter
    # should raise `from_status` rather than a bare exception.
    assert skip_reason(Exception("429 RESOURCE_EXHAUSTED")) == "skipped_unavailable"


def test_the_newest_screenshots_are_the_ones_that_survive(tmp_path) -> None:
    """The attachment budget is shared by both backends, so it is pinned here.

    History is replayed every turn, so without a cap a trial that takes N
    screenshots sends O(N^2) image bytes over its lifetime. Which N survive is
    decided across the whole conversation rather than per message: it is the
    newest shots that have to reach the model, because those are the ones the
    agent is talking about.
    """
    from viral_bench.crowd.sim.turns import MAX_IMAGES_PER_REQUEST, keep_newest_images

    shots = []
    convo = []
    for i in range(MAX_IMAGES_PER_REQUEST + 3):
        shot = tmp_path / f"shot{i}.png"
        shot.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i]) * 16)
        shots.append(shot)
        convo.append(
            {
                "role": "tool",
                "tool_call_id": f"c{i}",
                "content": mark_image(f"shot {i}", shot),
            }
        )

    keep = keep_newest_images(convo)
    assert len(keep) == MAX_IMAGES_PER_REQUEST
    assert keep == {str(s) for s in shots[-MAX_IMAGES_PER_REQUEST:]}
