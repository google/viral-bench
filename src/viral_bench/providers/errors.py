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

"""Classifying a model failure into something a caller can act on.

The crowd used to decide whether a failure was worth retrying by string-matching
the exception text against a list of Google and Anthropic phrasings. That works
for exactly the two providers it was written against: an OpenAI 429 arrives with
different wording and would be read as fatal, aborting a run that a two-second
backoff would have saved.

So classification is the adapter's job -- each one knows its own error shape and
returns one of these -- and only the generic fallback still looks at text.
"""

from __future__ import annotations

import re


class ModelError(RuntimeError):
    """The model could not be reached, or refused, after retries."""


class TransientError(ModelError):
    """A hiccup: overloaded, unavailable, timed out. Retry with backoff."""


class RateLimitError(TransientError):
    """Quota or rate limit. Retry, but back off harder and for longer."""


class BadRequestError(ModelError):
    """The request itself is wrong. Retrying sends the same broken request."""


class AuthError(ModelError):
    """Bad or missing credentials. No amount of retrying fixes this."""


#: HTTP statuses that are worth another attempt.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_TRANSIENT_TEXT = re.compile(
    r"429|50[0234]|unavailable|resource[_ ]exhausted|deadline|timed?[_ ]?out|"
    r"overloaded|rate.?limit|too many requests|connection reset|temporarily",
    re.IGNORECASE,
)
_AUTH_TEXT = re.compile(
    r"401|403|api[_ ]?key|permission[_ ]denied|unauthenticated|unauthorized|"
    r"invalid[_ ]authentication|credential",
    re.IGNORECASE,
)


def from_status(status: int, detail: str = "") -> ModelError:
    """Build the right error for an HTTP status code.

    This is the path every adapter should take when it has a real status code;
    :func:`classify` is only for exceptions that never carried one.
    """
    message = f"HTTP {status}: {detail}".rstrip(": ")
    if status == 429:
        return RateLimitError(message)
    if status in _RETRYABLE_STATUS:
        return TransientError(message)
    if status in (401, 403):
        return AuthError(message)
    if 400 <= status < 500:
        return BadRequestError(message)
    return TransientError(message)


def classify(exc: BaseException) -> ModelError:
    """Best-effort classification of an SDK exception with no status code.

    Used by adapters that sit on a vendor SDK which raises its own exception
    types. Prefer a real status code via :func:`from_status` when one exists --
    this reads the message, and messages change.
    """
    if isinstance(exc, ModelError):
        return exc
    if isinstance(exc, TimeoutError):
        return TransientError(f"timed out: {exc}")

    # Several SDKs expose the status on the exception even when they do not
    # subclass anything we know; use it if it is there.
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and 100 <= status < 600:
        return from_status(status, str(exc))

    text = str(exc)
    if _AUTH_TEXT.search(text):
        return AuthError(text)
    if _TRANSIENT_TEXT.search(text):
        return TransientError(text)
    return ModelError(text)


def backoff_seconds(attempt: int, error: BaseException | None = None) -> float:
    """Seconds to wait before attempt ``attempt`` (0-based), with jitter.

    Rate limits get the full exponential curve because the server is telling us
    to slow down; everything else gets a gentler linear ramp, because the usual
    cause is one bad connection rather than sustained pressure. Jitter matters:
    a sweep runs many workers, and without it they retry in lockstep and
    reproduce the burst that caused the limit.
    """
    import random  # noqa: PLC0415 - only needed on the failure path

    if isinstance(error, RateLimitError):
        return min(32.0, 2.0**attempt) * (0.7 + 0.6 * random.random())  # noqa: S311
    return 0.8 * (attempt + 1) * (0.7 + 0.6 * random.random())  # noqa: S311
