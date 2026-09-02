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

"""What every crowd model backend shares, whichever provider it talks to.

A crowd turn is the unit of measurement here: one agent, one step, one model
call. Three things about a turn have to behave identically on the Gemini path
(:mod:`viral_bench.crowd.sim.gemini_native`) and on the universal one
(:mod:`viral_bench.crowd.sim.unified_model`), or a run stops being comparable
with the runs beside it:

* how many turns were lost, and to what;
* what a lost turn returns, so the round survives it;
* how many screenshots one request is allowed to carry.

They live here rather than in either backend because the second backend would
otherwise have had to import the first one -- and because the reasoning below was
paid for in sweeps, not derived, so it must not fork into two versions that drift.

Nothing here imports ``camel`` or ``openai`` at module scope: the accounting is
plain arithmetic that the main 3.12 process (and its test suite) can read without
the crowd extra installed. Only :func:`empty_completion` needs the OpenAI response
types, and it reaches for them when called.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from viral_bench.crowd.interaction.imagery import split_images
from viral_bench.providers.errors import (
    BadRequestError,
    RateLimitError,
    TransientError,
    classify,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai.types.chat import ChatCompletion

#: Process-wide tally of turns the model could not complete. A skipped turn is
#: recorded by OASIS as an agent choosing to do nothing, which is indistinguishable
#: from genuine indifference -- so a throttled run reads as an unengaging app. The
#: run summary must be able to say how much of its silence was ours.
#:
#: ``budget_exhausted`` is that same failure by another route: CAMEL enforces
#: ``max_iteration`` by breaking out of its tool-call loop with no exception and
#: no log line, so an agent cut off mid-turn is likewise recorded as one that
#: simply had nothing to say. See ``BudgetAwareSocialAgent`` in sim/agents.py.
#:
#: Skips are counted BY CAUSE, not just in total. Only rate limiting used to be
#: broken out, so every other reason pooled into an unattributable remainder --
#: and that remainder turned out to be 84% one fixable bug (a malformed
#: [function_response, image] turn, see ``gemini_native._to_contents``). Recovering
#: that took grepping 1,600 sweep logs, because the reason lived only in a WARNING
#: line that a SIGKILL discards along with Python's buffer. The summary should be
#: able to answer "why did this run lose turns" from its own JSON.
TURN_STATS: dict[str, int] = {
    "turns": 0,
    "skipped": 0,
    "skipped_rate_limited": 0,
    "skipped_bad_request": 0,
    "skipped_unavailable": 0,
    "skipped_other": 0,
    "budget_exhausted": 0,
}

#: Content used for a skipped turn. It MUST be non-empty: CAMEL drops an
#: empty-content choice, yielding zero ``output_messages``, and OASIS's interview
#: path then does ``output_messages[0]`` and raises IndexError. That turned the
#: skip shim -- which exists to survive a transient failure -- into permanent loss
#: of that agent's interview, the primary signal. Measured: 11 of 50 agents lost
#: their interview this way in one run.
SKIPPED_TURN_CONTENT = "(no response)"

#: How many screenshots to attach to any one request. The conversation is
#: replayed every turn, so this bounds an otherwise quadratic growth in request
#: size over a trial. Newest shots win.
#:
#: 3 -> 8, because 3 was binding: measured over the 4,970 stored trials, 88 of
#: them (1.77%) took more than 3 screenshots, topping out at 6. In those the
#: earliest shots degraded to the text that described them -- and "design" is 22%
#: of the score and is answerable only by an agent that can actually see. 8 keeps
#: the quadratic guard (the concern was unbounded growth, not six images) while
#: clearing every trial in the corpus.
MAX_IMAGES_PER_REQUEST = 8


def keep_newest_images(messages: list[dict]) -> set[str]:
    """Which screenshot paths survive :data:`MAX_IMAGES_PER_REQUEST` this request.

    Decided over the whole conversation BEFORE any request body is built, for two
    reasons. The Gemini path copies its parts list into a pydantic model, so
    appending to it afterwards silently does nothing. And the choice is inherently
    global: it is the newest ``MAX_IMAGES_PER_REQUEST`` shots in the conversation
    that are kept, not the first ones encountered. Older shots degrade to the text
    that described them, which keeps the transcript coherent.

    Args:
        messages: CAMEL's OpenAI-shaped messages. Only ``role: "tool"`` results
            carry screenshot markers.
    """
    keep: set[str] = set()
    budget = MAX_IMAGES_PER_REQUEST
    for message in reversed(messages):
        if budget <= 0:
            break
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for path in split_images(content)[1]:
            if budget <= 0:
                break
            keep.add(str(path))
            budget -= 1
    return keep


def record_budget_exhausted() -> None:
    """Note that one agent turn ended at its ``max_iteration`` ceiling."""
    TURN_STATS["budget_exhausted"] += 1


def reset_turn_stats() -> None:
    for key in TURN_STATS:
        TURN_STATS[key] = 0


def turn_stats() -> dict:
    """Turn accounting for this process, plus the skip rate."""
    turns = TURN_STATS["turns"]
    return {
        **TURN_STATS,
        "skip_rate": round(TURN_STATS["skipped"] / turns, 4) if turns else 0.0,
    }


def skip_reason(error: BaseException | None) -> str:
    """Which ``TURN_STATS`` bucket a lost turn belongs in.

    Coarse on purpose -- the question it answers is "is this our request or their
    capacity", which decides whether a skip is a bug to fix or weather to ride out.

    The buckets used to be decided by string-matching the exception text against a
    list of Google phrasings, which read an OpenAI 429 as an unclassifiable
    "other". :func:`viral_bench.providers.errors.classify` reads the status code
    the SDK already carries, so the same 429 lands in the same bucket whoever sent
    it. An exception with no status at all still falls back to text, and there a
    rate limit is only distinguishable as a generic transient -- which is why an
    adapter should raise ``from_status`` rather than a bare exception.
    """
    if error is None:
        return "skipped_other"
    classified = classify(error)
    if isinstance(classified, RateLimitError):
        return "skipped_rate_limited"
    if isinstance(classified, BadRequestError):
        return "skipped_bad_request"
    if isinstance(classified, TransientError):
        return "skipped_unavailable"
    return "skipped_other"


def record_skip(error: BaseException | None) -> str:
    """Count one lost turn against its cause, and return the bucket used."""
    bucket = skip_reason(error)
    TURN_STATS["skipped"] += 1
    TURN_STATS[bucket] += 1
    return bucket


def empty_completion(model: str) -> ChatCompletion:
    """A valid 'did nothing this turn' response.

    Returned when an agent's turn fails after retries, so OASIS records that one
    agent took no action rather than the exception propagating out of ``env.step``
    and aborting the entire round (and the whole run).
    """
    from openai.types.chat import (  # noqa: PLC0415 - crowd-env-only import
        ChatCompletion,
        ChatCompletionMessage,
    )
    from openai.types.chat.chat_completion import Choice  # noqa: PLC0415

    message = ChatCompletionMessage(
        role="assistant", content=SKIPPED_TURN_CONTENT, tool_calls=None
    )
    return ChatCompletion(
        id="chatcmpl-" + uuid.uuid4().hex[:24],
        choices=[Choice(index=0, message=message, finish_reason="stop")],
        created=int(time.time()),
        model=model,
        object="chat.completion",
        usage=None,
    )
