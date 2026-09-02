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

"""The crowd can run on any provider, and says so plainly when it cannot.

Two things are being pinned here.

**No default, anywhere.** The crowd used to default to one specific Gemini model
over Vertex, and the Vertex path was an opt-OUT flag (``CROWD_USE_VERTEX``) that
defaulted ON. Between them, a new user holding a perfectly good ``GEMINI_API_KEY``
got an application-default-credentials failure naming a Google cloud project --
from a flag they had no way to know existed, for a model they never chose. Every
test below that asserts an error is asserting that the error now says what to do.

**The translation pair stays a pair.** CAMEL speaks OpenAI's message shape and the
provider layer speaks the canonical content-block shape.
``providers.openai_compat`` owns one direction and
``crowd.sim.unified_model`` owns the other, and the only thing keeping them from
drifting into two dialects is that the round trip through both is checked.

The ``UnifiedModel`` half needs ``camel`` + ``openai``, which live in the crowd
env, so those tests skip elsewhere. Everything about routing and refusal runs
anywhere, because it must happen before a backend is imported -- refusing a
40-minute run is worth nothing if it costs a heavyweight import to find out.
"""

from __future__ import annotations

import inspect

import pytest

from viral_bench.crowd.sim import model as crowd_model_module
from viral_bench.crowd.sim.model import (
    CROWD_CAPABILITIES,
    CrowdModelError,
    crowd_model,
    crowd_transport,
    resolve_crowd_model,
)
from viral_bench.providers import (
    Capability,
    UnsupportedCapabilityError,
    check_support,
    resolve,
)


@pytest.fixture
def no_keys(monkeypatch, tmp_path):
    """A machine with no credentials of any kind configured."""
    monkeypatch.setenv("VIRAL_BENCH_ENV_FILE", str(tmp_path / "absent.env"))
    for name in ("GEMINI_API_KEY_CROWD", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# -- no default model, no default provider ----------------------------------- #


def test_there_is_no_default_crowd_model() -> None:
    """``crowd_model()`` cannot be called without naming one."""
    parameter = inspect.signature(crowd_model).parameters["model_id"]
    assert parameter.default is inspect.Parameter.empty


def test_an_unconfigured_crowd_says_what_to_run() -> None:
    with pytest.raises(CrowdModelError) as excinfo:
        crowd_model("")
    message = str(excinfo.value)
    assert "viral-bench init" in message
    assert "config/crowd.yaml" in message


def test_a_bare_model_id_is_rejected_rather_than_guessed() -> None:
    """A bare id names no provider, and guessing one bills someone.

    A bare id is what the crowd used to default to, so it is the shape most
    likely to be left behind in an old config file.
    """
    with pytest.raises(CrowdModelError) as excinfo:
        crowd_model("gemini-lite-test")
    assert "viral-bench init" in str(excinfo.value)


def test_an_unknown_provider_is_named_in_the_error() -> None:
    with pytest.raises(CrowdModelError) as excinfo:
        crowd_model("nosuchvendor/some-model")
    assert "nosuchvendor" in str(excinfo.value)


def test_the_vertex_opt_out_flag_is_gone(monkeypatch) -> None:
    """``CROWD_USE_VERTEX`` must not exist, and must not be readable by accident.

    It defaulted ON, so it silently overrode a perfectly good API key. Which of
    the two Google surfaces to call is now decided by the provider id alone, and
    setting the old flag either way must change nothing.
    """
    assert not hasattr(crowd_model_module, "CROWD_VERTEX_ENV")
    assert not hasattr(crowd_model_module, "CROWD_DEFAULT_MODEL")
    assert not hasattr(crowd_model_module, "crowd_uses_vertex")

    for value in ("0", "1"):
        monkeypatch.setenv("CROWD_USE_VERTEX", value)
        vertex = resolve_crowd_model("google-vertex/gemini-lite-test")
        keyed = resolve_crowd_model("google/gemini-lite-test")
        assert vertex.provider.ambient_auth is True
        assert keyed.provider.ambient_auth is False


def test_the_keyed_gemini_path_names_the_key_it_wants(no_keys) -> None:
    """The failure the opt-out flag used to hide, now said out loud."""
    with pytest.raises(CrowdModelError) as excinfo:
        crowd_model("google/gemini-lite-test")
    message = str(excinfo.value)
    assert "GEMINI_API_KEY_CROWD" in message
    # And it points at the other way of authenticating, which is the thing that
    # used to happen silently.
    assert "google-vertex/gemini-lite-test" in message


# -- capability checking ----------------------------------------------------- #


def test_a_provider_that_cannot_see_is_refused_up_front() -> None:
    """The crowd sends PNG screenshots, so a text-only provider must not start a run.

    Discovering this three hours into a sweep means scoring a whole arm against
    ``design`` -- 22% of the score -- from a percept the model never received.
    """
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        crowd_model("ollama/llama-test")
    assert "images" in str(excinfo.value)


def test_a_full_provider_clears_the_check() -> None:
    assert CROWD_CAPABILITIES == Capability.TOOLS | Capability.IMAGES
    for qualified in ("openai/gpt-test", "anthropic/claude-sonnet-test"):
        check_support(resolve(qualified), CROWD_CAPABILITIES, stage="crowd")


# -- what the run summary records -------------------------------------------- #


def test_the_run_records_which_provider_carried_it() -> None:
    """A run summary has to say which surface produced it, or eras get pooled."""
    assert crowd_transport("google-vertex/gemini-lite-test") == "google-vertex"
    assert crowd_transport("openai/gpt-test") == "openai"
    assert crowd_transport("") == "unresolved"


# -- OpenAI <-> canonical, in both directions -------------------------------- #

_CAMEL_CONVERSATION = [
    {"role": "system", "content": "You are agent 7."},
    {"role": "user", "content": "Go and try the app."},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "open_app", "arguments": '{"url": "/"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "opened"},
    {"role": "assistant", "content": "Neat little thing."},
]


def test_the_openai_translation_pair_round_trips() -> None:
    """``to_openai_messages(from_openai_messages(x)) == x`` for a real conversation.

    This is what makes the second translator safe to have at all: it is the
    inverse of the first, not a second opinion about what the shapes mean. A
    change to either that is not mirrored in the other shows up here rather than
    as a provider rejecting a turn mid-sweep.
    """
    pytest.importorskip("camel", reason="needs the crowd extra (uv sync --extra crowd)")
    from viral_bench.crowd.sim.unified_model import from_openai_messages
    from viral_bench.providers.openai_compat import to_openai_messages

    canonical, system = from_openai_messages(_CAMEL_CONVERSATION)
    assert system == "You are agent 7."
    assert to_openai_messages(canonical, system) == _CAMEL_CONVERSATION


def test_the_tool_schema_pair_round_trips() -> None:
    pytest.importorskip("camel", reason="needs the crowd extra (uv sync --extra crowd)")
    from viral_bench.crowd.sim.unified_model import from_openai_tools
    from viral_bench.providers.openai_compat import to_openai_tools

    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "click",
                "description": "Click an element.",
                "parameters": {
                    "type": "object",
                    "properties": {"selector": {"type": "string"}},
                },
            },
        }
    ]
    canonical = from_openai_tools(openai_tools)
    assert canonical == [
        {
            "name": "click",
            "description": "Click an element.",
            "input_schema": openai_tools[0]["function"]["parameters"],
        }
    ]
    assert to_openai_tools(canonical) == openai_tools


def test_a_bare_function_schema_is_accepted_too() -> None:
    """OASIS builds some schemas unwrapped, so both forms have to work."""
    pytest.importorskip("camel", reason="needs the crowd extra (uv sync --extra crowd)")
    from viral_bench.crowd.sim.unified_model import from_openai_tools

    assert from_openai_tools([{"name": "refresh"}]) == [
        {
            "name": "refresh",
            "description": "",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]


def test_a_screenshot_reaches_the_model_as_an_image(tmp_path) -> None:
    """The percept has to contain pixels, not a sentence about pixels.

    Craft is 22% of the ViralScore and its ``design`` facet asks how an app looks.
    The Gemini backend learned this the expensive way, and the universal one must
    not have to learn it again.
    """
    pytest.importorskip("camel", reason="needs the crowd extra (uv sync --extra crowd)")
    import base64

    from viral_bench.crowd.interaction.imagery import mark_image
    from viral_bench.crowd.sim.unified_model import from_openai_messages

    shot = tmp_path / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x07" * 32)
    canonical, _ = from_openai_messages(
        [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": mark_image("Saved screenshot to shot.png", shot),
            }
        ]
    )

    blocks = canonical[0]["content"]
    assert blocks[0]["type"] == "tool_result"
    # The marker must not survive into the text the model reads.
    assert "VB_IMAGE" not in blocks[0]["content"]
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert base64.b64decode(blocks[1]["source"]["data"]) == shot.read_bytes()


# -- a lost turn is recorded, not raised ------------------------------------- #


def test_a_failed_turn_is_skipped_and_attributed_by_cause() -> None:
    """One agent's bad minute must not abort the round, and must be countable.

    ``Adapter.generate`` retries and then raises a bare ``ModelError`` naming the
    attempt count, chaining the real failure. Reading the wrapper alone would file
    every exhausted retry under ``skipped_other`` and lose the one distinction the
    by-cause tally exists to make: the request, or the provider's capacity.
    """
    pytest.importorskip("camel", reason="needs the crowd extra (uv sync --extra crowd)")
    from viral_bench.crowd.sim.turns import TURN_STATS, reset_turn_stats
    from viral_bench.crowd.sim.unified_model import UnifiedModel
    from viral_bench.providers.errors import ModelError, RateLimitError

    backend = UnifiedModel.__new__(UnifiedModel)
    backend.model_type = "openai/gpt-test"
    backend._meta_by_id = {}

    reset_turn_stats()
    try:
        exhausted = ModelError("gave up after 4 attempts")
        exhausted.__cause__ = RateLimitError("HTTP 429: slow down")
        completion = backend._failed(exhausted)

        assert TURN_STATS["skipped"] == 1
        assert TURN_STATS["skipped_rate_limited"] == 1
        # Non-empty content is load-bearing: CAMEL drops an empty-content choice
        # and OASIS's interview path then indexes into zero output messages.
        assert completion.choices[0].message.content
    finally:
        reset_turn_stats()


def test_a_bad_key_still_stops_the_run() -> None:
    """Auth failures are not one app's fault, and must not read as indifference."""
    pytest.importorskip("camel", reason="needs the crowd extra (uv sync --extra crowd)")
    from viral_bench.crowd.sim.turns import TURN_STATS, reset_turn_stats
    from viral_bench.crowd.sim.unified_model import UnifiedModel
    from viral_bench.providers.errors import AuthError

    backend = UnifiedModel.__new__(UnifiedModel)
    backend.model_type = "openai/gpt-test"
    backend._meta_by_id = {}

    reset_turn_stats()
    try:
        with pytest.raises(AuthError):
            backend._failed(AuthError("HTTP 401: bad key"))
        assert TURN_STATS["skipped"] == 0
    finally:
        reset_turn_stats()
