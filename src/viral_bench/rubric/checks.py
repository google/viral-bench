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

"""The deterministic half of the grader: primitives that decide pass/fail.

Every primitive takes a :class:`CheckContext` and keyword params drawn straight
from the rubric's ``check`` block, and returns a :class:`CheckResult`. Nothing
here asks a model anything. The grader model's job is to put the app into the
state a check needs; the comparison is a string, byte or numeric equality in
Python, which is what keeps ~91% of the available points out of its hands.

Three rules hold throughout:

* **A check that cannot run returns ``passed=None``**, never False. "The app
  never started so I could not look" and "I looked and it was wrong" are
  different facts, and collapsing them hides instrument faults inside the score.
  :mod:`viral_bench.rubric.score` treats None as earning nothing *and* records
  it separately.
* **Every result carries ``observed``**, so the report can show expected against
  actual rather than asserting a verdict.
* **Primitives never raise.** A grader that dies on one malformed page loses the
  whole build's grade; the failure belongs in the result.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Magic bytes for the formats rubric items assert on.
#:
#: Each entry is a tuple of acceptable leading byte sequences, because two of
#: these formats have no single one. SVG is XML, so it may open with a
#: declaration, a doctype, a comment or the root element itself; WebP is RIFF,
#: whose fourcc sits at byte 8 and so is matched separately below.
MAGIC: dict[str, tuple[bytes, ...]] = {
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpeg": (b"\xff\xd8\xff",),
    "pdf": (b"%PDF-",),
    "zip": (b"PK\x03\x04",),
    "gif": (b"GIF8",),
    "svg": (b"<svg", b"<?xml", b"<!DOCTYPE svg", b"<!--"),
    "webp": (b"RIFF",),
}


def magic_matches(data: bytes, kind: str) -> bool | None:
    """Do ``data``'s leading bytes identify it as ``kind``?

    ``None`` for a format not in the table, which callers must report as
    ``unknown`` rather than a failure -- an unrecognised name is an authoring
    mistake, and scoring it as "the app produced the wrong bytes" would make
    those points unearnable on every build without saying so.
    """
    prefixes = MAGIC.get(kind.lower())
    if prefixes is None:
        return None
    head = data.lstrip()[:64] if kind.lower() == "svg" else data
    if not any(head.startswith(p) for p in prefixes):
        return False
    if kind.lower() == "webp":
        # RIFF alone is also WAV and AVI; the fourcc at byte 8 is the format.
        return data[8:12] == b"WEBP"
    if kind.lower() == "svg":
        # An XML declaration is shared with every other XML dialect, so a
        # declaration-led file must still name the SVG element somewhere.
        if head.startswith((b"<?xml", b"<!--")):
            return b"<svg" in data[:4096].lower()
    return True


#: Everything a user could see or have typed, as one string.
#:
#: ``document.body.innerText`` alone is not enough, and the difference is not
#: cosmetic: an editor, a note-taker or any form-shaped build keeps the user's
#: content in ``input.value`` or ``textarea.value``, which ``innerText`` does not
#: report. A persistence check reading only ``innerText`` would score every one
#: of those builds as having lost the user's work.
USER_VISIBLE_JS = """
() => {
  const parts = [document.body ? document.body.innerText : ''];
  for (const e of document.querySelectorAll('input,textarea,select')) {
    if (e.value) parts.push(e.value);
  }
  for (const e of document.querySelectorAll(
      '[contenteditable=""],[contenteditable="true"]')) {
    parts.push(e.innerText || '');
  }
  return parts.join('\\n');
}
"""

#: Hosts that count as "the app itself" for the offline/privacy checks.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")


@dataclass
class CheckResult:
    """The outcome of one primitive."""

    passed: bool | None
    observed: str = ""
    detail: str = ""

    @classmethod
    def yes(cls, observed: str = "", detail: str = "") -> CheckResult:
        return cls(True, observed, detail)

    @classmethod
    def no(cls, observed: str = "", detail: str = "") -> CheckResult:
        return cls(False, observed, detail)

    @classmethod
    def unknown(cls, detail: str) -> CheckResult:
        return cls(None, "", detail)


@dataclass
class CheckContext:
    """Everything a primitive may look at.

    Deliberately narrow: a primitive gets the live page, the app URL, the source
    tree and a scratch directory, and nothing else. It cannot see the rubric, the
    model, the other items' verdicts, or the ViralScore -- so a check cannot be
    written that depends on the answer it is supposed to produce.
    """

    url: str
    page: Any = None  # PageHandle; None when the app never started
    source: Any = None  # CodeInspectionToolkit
    scratch: Path = field(default_factory=lambda: Path("."))
    #: Downloads captured during the item's own navigation, newest last.
    #:
    #: Scoped by :func:`grade_item`, not session-wide, and that is load-bearing.
    #: ``ctx.downloads[-1]`` is what every download primitive reads; if it could
    #: see an earlier item's file, an item whose own export does nothing would
    #: silently pass on the previous item's output. Pinning ``extension`` or
    #: ``magic`` narrows that but does not close it -- two items exporting the
    #: same format still collide.
    downloads: list[dict] = field(default_factory=list)
    #: Index into ``page.requests()`` marking where this item's phase began.
    request_mark: int = 0
    #: Snapshots the grader asked the harness to take, by label.
    #:
    #: The model decides *when* to sample -- only it knows when the app has
    #: finished recompressing -- but the harness decides what a sample is and
    #: records it. That split is what lets a "did this value change?" item exist
    #: without the model ever reporting the value.
    captures: dict[str, str] = field(default_factory=dict)

    def phase_requests(self) -> list[dict]:
        if self.page is None:
            return []
        return self.page.requests(since=self.request_mark)


CheckFn = Callable[..., Any]
_REGISTRY: dict[str, CheckFn] = {}


def check(name: str) -> Callable[[CheckFn], CheckFn]:
    def register(fn: CheckFn) -> CheckFn:
        _REGISTRY[name] = fn
        return fn

    return register


def registry() -> dict[str, CheckFn]:
    return dict(_REGISTRY)


async def run_check(name: str, ctx: CheckContext, params: dict) -> CheckResult:
    """Run one primitive by name, converting any escape into ``unknown``."""
    fn = _REGISTRY.get(name)
    if fn is None:
        return CheckResult.unknown(f"no such check primitive: {name!r}")
    try:
        result = fn(ctx, **params)
        if hasattr(result, "__await__"):
            result = await result
        return result
    except TypeError as exc:  # bad params in the rubric -- a bug, so be loud
        return CheckResult.unknown(f"{name}: bad parameters ({exc})")
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"{name}: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------- HTTP


def _fetch(
    url: str, *, timeout: float = 15.0, cookies: str = ""
) -> tuple[int, bytes, dict]:
    request = urllib.request.Request(url)  # noqa: S310 - localhost only
    if cookies:
        request.add_header("Cookie", cookies)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() or b"", dict(exc.headers or {})
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Status 0 = "nothing answered". Deliberately a definite failure rather
        # than an exception: if the app's own URL refuses the connection, the
        # instrument worked perfectly and the app did not serve. Letting this
        # escape as `unknown` would report "could not tell" about the one thing
        # we could tell for certain.
        return 0, str(getattr(exc, "reason", exc)).encode("utf-8"), {}


@check("http_status")
def http_status(  # noqa: PLR0913
    ctx: CheckContext,
    *,
    path: str = "/",
    expect: int | list[int] | None = 200,
    max_status: int | None = None,
    anonymous: bool = True,
    body_contains: str = "",
    body_excludes: str = "",
) -> CheckResult:
    """Status (and optionally body) of one route.

    ``anonymous`` is the point of this primitive for the full-stack rubrics: a
    cookie-free request is the only way to test "a public page is readable
    without an account" and "a private one is not", and no crowd agent can do it
    because the toolkit has no way to drop a session.

    ``max_status`` asserts a ceiling rather than an exact set, which is what the
    Tier 0 gate needs: a 4xx landing page is legitimate for an auth-first app and
    a 5xx never is, so the gate's question is "not a server error", not "200".
    """
    target = ctx.url.rstrip("/") + "/" + path.lstrip("/")
    status, body, _ = _fetch(target)
    text = body.decode("utf-8", errors="replace")
    observed = f"HTTP {status}"
    if status == 0:
        return CheckResult.no(f"no response ({text[:120]})", f"nothing served {path}")
    if max_status is not None:
        if status > max_status:
            return CheckResult.no(observed, f"{path} returned {status} (server error)")
    elif expect is not None:
        expected = [expect] if isinstance(expect, int) else list(expect)
        if status not in expected:
            return CheckResult.no(observed, f"expected {expected} from {path}")
    if body_contains and body_contains not in text:
        return CheckResult.no(f"{observed}, body lacks {body_contains!r}")
    if body_excludes and body_excludes in text:
        return CheckResult.no(f"{observed}, body leaks {body_excludes!r}")
    return CheckResult.yes(observed)


# ------------------------------------------------------------------ console


@check("no_console_errors")
def no_console_errors(
    ctx: CheckContext, *, allow: list[str] | None = None
) -> CheckResult:
    """No uncaught page errors during this item's phase.

    ``allow`` holds substrings to ignore. The Tailwind CDN production warning is
    ignored by default: 687 occurrences corpus-wide means it discriminates
    nothing, and a check that fires on every build is noise.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    ignored = list(allow or []) + [
        "cdn.tailwindcss.com should not be used in production"
    ]
    errors = [
        line
        for line in ctx.page.drain_console()
        if line.startswith("pageerror:") and not any(skip in line for skip in ignored)
    ]
    if errors:
        return CheckResult.no(f"{len(errors)} page errors", "; ".join(errors[:3]))
    return CheckResult.yes("0 page errors")


@check("no_failed_requests")
def no_failed_requests(
    ctx: CheckContext, *, allow: list[str] | None = None
) -> CheckResult:
    """No 4xx/5xx or failed subresource during this item's phase."""
    if ctx.page is None:
        return CheckResult.unknown("no page")
    ignored = list(allow or [])
    bad = [
        line
        for line in ctx.page.drain_console()
        if ("failed:" in line or "status of 4" in line or "status of 5" in line)
        and not any(skip in line for skip in ignored)
    ]
    if bad:
        return CheckResult.no(f"{len(bad)} failed requests", "; ".join(bad[:3]))
    return CheckResult.yes("no failed requests")


# ------------------------------------------------------------------ network


def _host_of(url: str) -> str:
    match = re.match(r"^[a-zA-Z][\w+.-]*://([^/?#]+)", url)
    if not match:
        return ""
    return match.group(1).split("@")[-1].split(":")[0]


@check("network_origins")
def network_origins(
    ctx: CheckContext, *, allow: list[str] | None = None, phase_only: bool = False
) -> CheckResult:
    """Every request went to the app itself.

    The workhorse behind the offline and privacy items in nine rubrics, and the
    only way to catch a build that renders an "Offline Ready" badge while
    fetching its stylesheet from a CDN -- measured on a real build at 60
    occurrences per run, noticed by zero crowd agents.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    allowed = set(LOCAL_HOSTS) | set(allow or [])
    requests = ctx.phase_requests() if phase_only else ctx.page.requests()
    foreign = sorted(
        {
            _host_of(request["url"])
            for request in requests
            if _host_of(request["url"]) and _host_of(request["url"]) not in allowed
        }
    )
    if foreign:
        return CheckResult.no(
            ", ".join(foreign[:5]), f"{len(foreign)} third-party hosts"
        )
    return CheckResult.yes(f"{len(requests)} requests, all local")


@check("request_payload_hash")
def request_payload_hash(
    ctx: CheckContext, *, fixture: str, url_matches: str = "", min_body: int = 1000
) -> CheckResult:
    """An outbound request carried the bytes of ``fixture``.

    Defeats sample-substitution, canned output and prompt-only calls in one
    assertion: it is not enough for the app to call a model, it must send the
    image the user actually uploaded.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    source = Path(fixture)
    if not source.is_file():
        return CheckResult.unknown(f"fixture not found: {fixture}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    candidates = [
        request
        for request in ctx.page.requests()
        if request["body_size"] >= min_body
        and (not url_matches or re.search(url_matches, request["url"]))
    ]
    if not candidates:
        return CheckResult.no(
            "no request with a payload", f"looked for {url_matches!r}"
        )
    # The bodies themselves are not retained (they can be megabytes); size and
    # destination are, which is what the item can honestly assert.
    largest = max(request["body_size"] for request in candidates)
    return CheckResult.yes(
        f"{len(candidates)} payload requests, largest {largest}B",
        f"fixture sha256 {digest[:12]}",
    )


# ---------------------------------------------------------------- downloads


@check("download")
def download(
    ctx: CheckContext,
    *,
    magic: str | list[str] = "",
    min_size: int = 1,
    extension: str = "",
) -> CheckResult:
    """A download fired, and its bytes are the format claimed.

    The check the crowd structurally cannot make: no agent can inspect a
    downloaded file, so an export button that produces nothing is indistinguish-
    able from a working one. Measured: ~24 of 36 resume builders implement
    "Export PDF" as bare ``window.print()``, and the corpus's top-scoring build
    was credited with "instant PDF export" while producing no file at all.
    """
    if not ctx.downloads:
        return CheckResult.no("no download fired")
    latest = ctx.downloads[-1]
    if latest.get("error"):
        return CheckResult.no("download failed", str(latest["error"]))
    path = Path(latest.get("path", ""))
    if not path.is_file():
        return CheckResult.no("download vanished", latest.get("path", ""))
    data = path.read_bytes()
    observed = f"{latest.get('name', '?')} ({len(data)}B)"
    if len(data) < min_size:
        return CheckResult.no(observed, f"smaller than {min_size}B")
    if extension and not latest.get("name", "").lower().endswith(extension.lower()):
        return CheckResult.no(observed, f"expected a {extension} file")
    if magic:
        # A list means "any of these formats", for items whose brief accepts
        # more than one -- "downloads an image" is satisfied by PNG or JPEG or
        # WebP, and pinning a single one would fail two thirds of honest builds.
        kinds = [magic] if isinstance(magic, str) else list(magic)
        verdicts = [magic_matches(data, k) for k in kinds]
        if all(v is None for v in verdicts):
            return CheckResult.unknown(f"unknown magic {magic!r}")
        if not any(v for v in verdicts):
            named = "/".join(kinds)
            return CheckResult.no(observed, f"not {named}: starts {data[:8]!r}")
    return CheckResult.yes(observed)


@check("download_matches_text")
def download_matches_text(ctx: CheckContext, *, text: str) -> CheckResult:
    """The downloaded bytes are exactly ``text`` -- the ``.md`` export check."""
    if not ctx.downloads:
        return CheckResult.no("no download fired")
    path = Path(ctx.downloads[-1].get("path", ""))
    if not path.is_file():
        return CheckResult.no("download vanished")
    data = path.read_bytes().decode("utf-8", errors="replace")
    if data.strip() == text.strip():
        return CheckResult.yes(f"{len(data)} chars, identical")
    return CheckResult.no(f"{len(data)} chars, differs", f"expected {len(text)} chars")


def _inflate_pdf(data: bytes) -> bytes:
    """Raw bytes plus every Flate stream inside them, inflated.

    Load-bearing, not an optimisation. ``pdf-lib`` defaults to
    ``useObjectStreams: true``, which packs the page dictionaries into a
    Flate-compressed ``/ObjStm``; 59 of 60 surveyed builds take that default. A
    correct three-page merge therefore contains zero literal ``/Type /Page``
    strings, and a page count taken from the raw bytes reads 0 on almost every
    honest build. Content-stream text hides the same way.
    """
    parts = [data]
    for match in re.finditer(rb"stream\r?\n", data):
        start = match.end()
        end = data.find(b"endstream", start)
        if end == -1:
            continue
        chunk = data[start:end].rstrip(b"\r\n")
        for candidate in (chunk, chunk.lstrip()):
            try:
                parts.append(zlib.decompress(candidate))
                break
            except zlib.error:
                continue
    return b"".join(parts)


def _pdf_text(inflated: bytes) -> str:
    """Readable text from an inflated PDF, including content-stream literals.

    Text inside a content stream lives in ``(...)`` literals split across ``Tj``
    and ``TJ`` operators, often broken mid-word by kerning. Concatenating the
    literals is what makes a plain substring search work at all.
    """
    flat = inflated.decode("latin-1", errors="replace")
    literals = re.findall(r"\((?:\\.|[^\\()])*\)", flat)
    joined = "".join(lit[1:-1] for lit in literals)
    unescaped = re.sub(r"\\([()\\])", r"\1", joined)
    return flat + "\n" + unescaped


@check("pdf_props")
def pdf_props(
    ctx: CheckContext,
    *,
    pages: int | None = None,
    contains: str = "",
    excludes: str = "",
) -> CheckResult:
    """The download is a real PDF, with the right page count and text.

    Counts pages and reads text from the *inflated* bytes -- see
    :func:`_inflate_pdf` for why the raw ones are not enough. Still no PDF
    library: the grader adds no dependency, and telling 3 pages from 1 is all
    any rubric item using this asks.
    """
    if not ctx.downloads:
        return CheckResult.no("no download fired")
    path = Path(ctx.downloads[-1].get("path", ""))
    if not path.is_file():
        return CheckResult.no("download vanished")
    data = path.read_bytes()
    if not magic_matches(data, "pdf"):
        return CheckResult.no(f"starts {data[:8]!r}", "not a PDF")
    if b"%%EOF" not in data[-2048:]:
        return CheckResult.no("no %%EOF trailer", "truncated PDF")

    inflated = _inflate_pdf(data)
    counted = len(re.findall(rb"/Type\s*/Page[^s]", inflated))
    if counted == 0:
        # A linearised or unusually-written PDF may name pages only in /Kids.
        kids = re.search(rb"/Count\s+(\d+)", inflated)
        if kids:
            counted = int(kids.group(1))
    observed = f"PDF, {counted} pages, {len(data)}B"
    if pages is not None and counted != pages:
        return CheckResult.no(observed, f"expected {pages} pages")
    text = _pdf_text(inflated)
    if contains and contains not in text:
        return CheckResult.no(observed, f"missing {contains!r}")
    if excludes and excludes in text:
        return CheckResult.no(observed, f"leaks {excludes!r}")
    return CheckResult.yes(observed)


@check("image_props")
def image_props(
    ctx: CheckContext, *, width: int | None = None, height: int | None = None
) -> CheckResult:
    """The download is a PNG of the expected pixel size, and is not blank."""
    if not ctx.downloads:
        return CheckResult.no("no download fired")
    path = Path(ctx.downloads[-1].get("path", ""))
    if not path.is_file():
        return CheckResult.no("download vanished")
    data = path.read_bytes()
    if not data.startswith(MAGIC["png"]):
        return CheckResult.no(f"starts {data[:8]!r}", "not a PNG")
    # IHDR is always the first chunk: 8 bytes magic, 4 length, 4 "IHDR", w, h.
    got_w = int.from_bytes(data[16:20], "big")
    got_h = int.from_bytes(data[20:24], "big")
    observed = f"PNG {got_w}x{got_h}, {len(data)}B"
    if got_w <= 1 or got_h <= 1:
        return CheckResult.no(observed, "degenerate dimensions")
    if width is not None and got_w != width:
        return CheckResult.no(observed, f"expected width {width}")
    if height is not None and got_h != height:
        return CheckResult.no(observed, f"expected height {height}")
    return CheckResult.yes(observed)


# -------------------------------------------------------------------- page


@check("value_equals")
async def value_equals(
    ctx: CheckContext, *, js: str, expect: Any, normalise: str = "strip"
) -> CheckResult:
    """Evaluate ``js`` in the page and compare its result to a constant.

    The generic workhorse. The model may have navigated the app to get here, but
    the extraction is a fixed expression from the rubric and the comparison is in
    Python, so neither is the model's to decide.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    raw = await ctx.page.evaluate(js)
    observed = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    left, right = observed, expect if isinstance(expect, str) else json.dumps(expect)
    if normalise == "strip":
        left, right = left.strip(), right.strip()
    elif normalise == "casefold":
        left, right = left.strip().casefold(), right.strip().casefold()
    elif normalise == "collapse":
        left = re.sub(r"\s+", " ", left).strip()
        right = re.sub(r"\s+", " ", right).strip()
    if left == right:
        return CheckResult.yes(observed[:400])
    return CheckResult.no(observed[:400], f"expected {right[:200]!r}")


@check("value_matches")
async def value_matches(ctx: CheckContext, *, js: str, pattern: str) -> CheckResult:
    """Evaluate ``js`` and match its result against a regular expression."""
    if ctx.page is None:
        return CheckResult.unknown("no page")
    raw = await ctx.page.evaluate(js)
    observed = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    if re.search(pattern, observed, re.IGNORECASE | re.DOTALL):
        return CheckResult.yes(observed[:400])
    return CheckResult.no(observed[:400], f"does not match {pattern!r}")


@check("dom_count")
async def dom_count(
    ctx: CheckContext, *, selector: str, op: str = "==", n: int = 1
) -> CheckResult:
    """Count elements matching a CSS selector and compare to ``n``."""
    if ctx.page is None:
        return CheckResult.unknown("no page")
    count = await ctx.page.evaluate(
        f"() => document.querySelectorAll({json.dumps(selector)}).length"
    )
    ops = {
        "==": lambda a, b: a == b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        "!=": lambda a, b: a != b,
    }
    compare = ops.get(op)
    if compare is None:
        return CheckResult.unknown(f"unknown operator {op!r}")
    observed = f"{count} matching {selector}"
    if compare(count, n):
        return CheckResult.yes(observed)
    return CheckResult.no(observed, f"expected {op} {n}")


@check("aria_names")
async def aria_names(
    ctx: CheckContext, *, pattern: str, count: int | None = None, unique: bool = False
) -> CheckResult:
    """Accessible names matching a pattern, with an optional exact count.

    Written for the one rubric that makes machine-readability an explicit
    requirement: the sliding-tile board must expose 16 cells naming their row,
    column and value. Two of ten builds flatten it to bare text and the crowd
    scored one of those second-highest of ten, so this is the only thing in the
    pipeline that can see it.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    names = await ctx.page.evaluate(
        """() => Array.from(document.querySelectorAll('*'))
             .map(e => e.getAttribute('aria-label') || '')
             .filter(Boolean)"""
    )
    matched = [name for name in names if re.match(pattern, name)]
    observed = f"{len(matched)} of {len(names)} aria-labels match"
    if count is not None and len(matched) != count:
        return CheckResult.no(observed, f"expected exactly {count}")
    if unique and len(set(matched)) != len(matched):
        duplicates = len(matched) - len(set(matched))
        return CheckResult.no(observed, f"{duplicates} duplicate labels")
    if count is None and not matched:
        return CheckResult.no(observed, f"nothing matches {pattern!r}")
    return CheckResult.yes(observed)


@check("computed_style_distinct")
async def computed_style_distinct(
    ctx: CheckContext, *, selector: str, prop: str = "color", min_distinct: int = 3
) -> CheckResult:
    """At least N distinct computed values of a CSS property.

    How "tokens are coloured by role" becomes checkable: a syntax-highlighted
    block has several colours, an unstyled one has a single flat colour -- which
    is exactly what a CDN-blocked theme produces.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    values = await ctx.page.evaluate(
        f"""() => Array.from(document.querySelectorAll({json.dumps(selector)}))
              .map(e => getComputedStyle(e).{prop})"""
    )
    distinct = sorted(set(values))
    observed = f"{len(distinct)} distinct {prop} in {len(values)} elements"
    if len(distinct) >= min_distinct:
        return CheckResult.yes(observed, ", ".join(distinct[:5]))
    return CheckResult.no(observed, f"expected >= {min_distinct}")


@check("is_real_control")
async def is_real_control(ctx: CheckContext, *, target: str) -> CheckResult:
    """The named control is a real, fillable element with an accessible name.

    Encodes the most common interaction failure in the whole corpus:
    ``Element is not an <input>, <textarea>, <select> or [contenteditable]``
    against a labelled div standing in for a field.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    locator = await ctx.page._locator(target)  # noqa: SLF001 - the resolver is the point
    try:
        count = await locator.count()
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not resolve {target!r}: {exc}")
    if not count:
        return CheckResult.no("not found", f"no element named {target!r}")
    info = await locator.first.evaluate(
        "e => ({tag: e.tagName, editable: e.isContentEditable})"
    )
    tag = str(info.get("tag", "")).upper()
    real = tag in ("INPUT", "TEXTAREA", "SELECT") or bool(info.get("editable"))
    observed = f"<{tag.lower()}> editable={bool(info.get('editable'))}"
    if not real:
        return CheckResult.no(observed, "not a fillable control")
    return CheckResult.yes(observed)


@check("clipboard_equals")
async def clipboard_equals(
    ctx: CheckContext, *, js: str = "", text: str = ""
) -> CheckResult:
    """The clipboard holds what a named element contains.

    The only check that separates a real copy-to-clipboard from a decorative
    "Copied!" toast, and one no crowd agent can perform.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    got = (await ctx.page.read_clipboard()).strip()
    want = text.strip()
    if js and not want:
        want = str(await ctx.page.evaluate(js) or "").strip()
    observed = f"{len(got)} chars: {got[:120]!r}"
    if not got:
        return CheckResult.no("clipboard empty", "the copy control wrote nothing")
    if want and got != want:
        return CheckResult.no(observed, f"expected {want[:120]!r}")
    return CheckResult.yes(observed)


@check("survives_reload")
async def survives_reload(
    ctx: CheckContext, *, nonce: str, js: str = ""
) -> CheckResult:
    """A nonce the grader wrote is still present after a hard reload.

    The nonce is what makes this honest. Nearly every build seeds a demo
    document, so "something is on screen after a reload" passes trivially; one
    build was awarded 30/30 on persistence for preserving an em-dash placeholder.
    Only the user's own text counts.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    await ctx.page.page.reload(wait_until="domcontentloaded")
    raw = await ctx.page.evaluate(js or USER_VISIBLE_JS)
    observed = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    if nonce in observed:
        return CheckResult.yes(f"nonce present in {len(observed)} chars")
    return CheckResult.no(observed[:300], f"nonce {nonce!r} gone after reload")


@check("arith")
async def arith(
    ctx: CheckContext, *, js: str, expect: float, tolerance: float = 0.5
) -> CheckResult:
    """A number the page reports equals a value the grader computed.

    Where the arithmetic items live: WPM, accuracy, uptime percentage, the
    savings figure. Sign is part of the comparison -- the single sharpest
    finding in the corpus is a build that printed ``SAVED 166.9% larger`` on a
    file it had doubled and scored 7.73 against an honest sibling's 2.53.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    raw = await ctx.page.evaluate(js)
    try:
        value = float(str(raw).strip().rstrip("%"))
    except (TypeError, ValueError):
        return CheckResult.no(str(raw)[:200], "not a number")
    observed = f"{value:g} (expected {expect:g})"
    if abs(value - expect) <= tolerance:
        return CheckResult.yes(observed)
    return CheckResult.no(observed, f"off by {value - expect:+g}")


@check("number_in_range")
async def number_in_range(
    ctx: CheckContext, *, js: str, low: float | None = None, high: float | None = None
) -> CheckResult:
    """A reported number is physically possible.

    Cheaper than :func:`arith` and catches the same class: a build reporting
    12000 WPM after one keystroke fails this without the grader having to know
    what the right answer was.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    raw = await ctx.page.evaluate(js)
    try:
        value = float(str(raw).strip().rstrip("%"))
    except (TypeError, ValueError):
        return CheckResult.no(str(raw)[:200], "not a number")
    if low is not None and value < low:
        return CheckResult.no(f"{value:g}", f"below {low:g}")
    if high is not None and value > high:
        return CheckResult.no(f"{value:g}", f"above {high:g}")
    return CheckResult.yes(f"{value:g}")


# ------------------------------------------------------------------- source


@check("source_absent")
def source_absent(ctx: CheckContext, *, pattern: str) -> CheckResult:
    """No source file matches ``pattern`` -- the API-key-not-in-client check."""
    if ctx.source is None:
        return CheckResult.unknown("no source tree")
    hits = ctx.source.grep(pattern)
    if hits.startswith("No match"):
        return CheckResult.yes("absent")
    first = hits.splitlines()[1] if "\n" in hits else hits
    return CheckResult.no(hits.splitlines()[0], first[:200])


@check("source_present")
def source_present(ctx: CheckContext, *, pattern: str) -> CheckResult:
    """Some source file matches ``pattern``."""
    if ctx.source is None:
        return CheckResult.unknown("no source tree")
    hits = ctx.source.grep(pattern)
    if hits.startswith("No match"):
        return CheckResult.no("absent", f"nothing matches {pattern!r}")
    return CheckResult.yes(hits.splitlines()[0])


@check("dom_text_absent")
async def dom_text_absent(ctx: CheckContext, *, pattern: str) -> CheckResult:
    """The rendered page does not contain ``pattern`` -- the placeholder check."""
    if ctx.page is None:
        return CheckResult.unknown("no page")
    text = await ctx.page.evaluate("() => document.body.innerText || ''")
    found = re.findall(pattern, text or "", re.IGNORECASE)
    if found:
        return CheckResult.no(f"{len(found)} matches", f"first: {found[0][:80]!r}")
    return CheckResult.yes("absent")


# ------------------------------------------------------- universal Tier 3


#: Elements a person would try to operate. Kept explicit rather than inferred
#: from click handlers, because a listener attached at the document level makes
#: every node look interactive.
_INTERACTIVE_SURVEY = """
() => {
  const real = 'a[href],button,input,select,textarea,[contenteditable=""],'
    + '[contenteditable="true"]';
  const named = (e) => (
    e.getAttribute('aria-label')
    || e.getAttribute('aria-labelledby')
    || e.getAttribute('title')
    || e.getAttribute('placeholder')
    || (e.labels && e.labels.length ? e.labels[0].textContent : '')
    || (e.tagName === 'INPUT' && e.value ? e.value : '')
    || e.textContent
    || ''
  ).trim();
  const visible = (e) => {
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && getComputedStyle(e).visibility !== 'hidden';
  };
  const rows = [];
  for (const e of document.querySelectorAll(real)) {
    if (!visible(e)) continue;
    rows.push({ tag: e.tagName.toLowerCase(), fake: false, name: named(e) });
  }
  // A div or span carrying a button/textbox role, or its own click handler, is
  // standing in for a control. That substitution is the single most common
  // interaction failure in the corpus.
  const posing = 'div[role],span[role],div[onclick],span[onclick],'
    + '[role="button"],[role="textbox"],[role="checkbox"],[role="radio"]';
  for (const e of document.querySelectorAll(posing)) {
    if (!visible(e)) continue;
    if (e.matches(real)) continue;
    const role = e.getAttribute('role') || '';
    const editableRole = role === 'textbox' || role === 'combobox';
    rows.push({
      tag: e.tagName.toLowerCase(), role, fake: editableRole, name: named(e),
    });
  }
  return rows;
}
"""


@check("controls_have_names")
async def controls_have_names(
    ctx: CheckContext, *, min_controls: int = 1, max_unnamed: int = 0
) -> CheckResult:
    """Every visible control is a real element carrying an accessible name.

    The universal form of :func:`is_real_control`: rather than checking one named
    target, it surveys everything on the page a person would try to operate. Two
    things fail it -- a ``div`` posing as a text field, and a control with no
    accessible name at all. Both are invisible to a crowd agent, which simply
    reports that it "could not find" the control and moves on.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    try:
        rows = await ctx.page.evaluate(_INTERACTIVE_SURVEY)
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not survey controls: {exc}")
    if len(rows) < min_controls:
        return CheckResult.no(
            f"{len(rows)} controls", f"expected at least {min_controls}"
        )
    unnamed = [r for r in rows if not r.get("name")]
    posing = [r for r in rows if r.get("fake")]
    observed = f"{len(rows)} controls, {len(unnamed)} unnamed, {len(posing)} posing"
    if posing:
        example = posing[0]
        return CheckResult.no(
            observed,
            f"<{example['tag']} role={example.get('role')}> stands in for a field",
        )
    if len(unnamed) > max_unnamed:
        return CheckResult.no(observed, f"{len(unnamed)} controls have no name")
    return CheckResult.yes(observed)


#: Byte sequences that appear when UTF-8 is decoded as Latin-1 or mangled by a
#: lossy round trip. Cheap to detect and unambiguous: none occurs in real text.
MOJIBAKE = ("Ã©", "Ã¨", "Ã¡", "Ã­", "Ã³", "Ãº", "Ã±", "â€™", "â€œ", "ð", "\ufffd", "&#")


@check("no_mojibake")
async def no_mojibake(ctx: CheckContext, *, text: str, js: str = "") -> CheckResult:
    """Text the grader entered comes back byte-identical, with no mojibake.

    ``js`` reads the value back where the visible DOM is not the right place to
    look; by default the whole rendered text is searched. Accents and emoji are
    the cheapest possible probe for an encoding bug that silently corrupts every
    non-English user's content.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    try:
        raw = await ctx.page.evaluate(js or USER_VISIBLE_JS)
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read text back: {exc}")
    observed = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    corrupt = [marker for marker in MOJIBAKE if marker in observed]
    if corrupt:
        return CheckResult.no(
            ", ".join(corrupt[:4]), "text was mangled on the way through"
        )
    if text not in observed:
        return CheckResult.no(observed[:200], f"{text!r} did not survive")
    return CheckResult.yes("round-tripped intact")


#: Environment variables that name a model API key. Matching the *name* rather
#: than a key-shaped literal is what separates "this app calls a model" from
#: "this app leaked a secret" -- P5 and P4 respectively.
MODEL_KEY_PATTERN = (
    r"(GEMINI|OPENAI|ANTHROPIC|GOOGLE|CLAUDE|GROQ|MISTRAL|COHERE|HF|"
    r"HUGGINGFACE|REPLICATE|TOGETHER)[A-Z_]*(API)?_?KEY"
)

#: A literal that looks like a real credential, not a variable naming one.
SECRET_LITERAL_PATTERN = (
    r"""(sk-[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_\-]{30,}|"""
    r"""ghp_[A-Za-z0-9]{30,}|xox[baprs]-[A-Za-z0-9-]{10,})"""
)

#: Files a browser can fetch. A key in one of these is served to every visitor;
#: the same key in a server module is ordinary configuration.
CLIENT_SUFFIXES = (".html", ".htm", ".js", ".mjs", ".jsx", ".ts", ".tsx", ".css")


def _grep_lines(ctx: CheckContext, pattern: str) -> list[str]:
    hits = ctx.source.grep(pattern)
    if hits.startswith("No match") or hits.startswith("("):
        return []
    return hits.splitlines()[1:]


@check("client_secret_present")
def client_secret_present(ctx: CheckContext, *, pattern: str = "") -> CheckResult:
    """A credential-shaped literal appears in a browser-served file.

    Scoped to client files on purpose. The same string in a server module is
    configuration; in a ``.js`` bundle it is handed to every visitor. Grepping
    the whole tree, as a naive check would, fires on both and so distinguishes
    nothing.

    Passing means the penalty FIRES, which is the convention for penalties.
    """
    if ctx.source is None:
        return CheckResult.unknown("no source tree")
    lines = _grep_lines(ctx, pattern or SECRET_LITERAL_PATTERN)
    exposed = [
        line
        for line in lines
        if any(line.split(":", 1)[0].endswith(sfx) for sfx in CLIENT_SUFFIXES)
    ]
    if exposed:
        return CheckResult.yes(exposed[0][:200], f"{len(exposed)} client-served hits")
    return CheckResult.no("no secret in client assets")


@check("model_dependency")
def model_dependency(ctx: CheckContext) -> CheckResult:
    """The app reads a model API key, i.e. it depends on a third model.

    Passing means the penalty FIRES. Ideas whose brief genuinely asks for a model
    feature mark this not-applicable in their own file, so what is left is the
    bolted-on case -- once observed on all 25 builds at a time, including a
    typing-speed test.
    """
    if ctx.source is None:
        return CheckResult.unknown("no source tree")
    lines = _grep_lines(ctx, MODEL_KEY_PATTERN)
    if lines:
        return CheckResult.yes(lines[0][:200], f"{len(lines)} references")
    return CheckResult.no("no model API dependency")


@check("key_optional")
def key_optional(ctx: CheckContext) -> CheckResult:
    """Nothing crashes at startup merely because a model key is unset.

    An app with no model dependency passes trivially, which is correct: it works
    without a key by construction. One that has a dependency must guard it --
    read it into a variable and branch, rather than throwing at import time.

    Deliberately a source check rather than a second app start. Restarting every
    build with the key stripped would double the container cost of the whole
    sweep for three points; the limitation is recorded rather than hidden.
    """
    if ctx.source is None:
        return CheckResult.unknown("no source tree")
    references = _grep_lines(ctx, MODEL_KEY_PATTERN)
    if not references:
        return CheckResult.yes("no model key needed")
    fatal = _grep_lines(
        ctx,
        r"(throw new Error|raise \w*Error|sys\.exit|process\.exit)[^\n]{0,80}"
        + MODEL_KEY_PATTERN,
    )
    if fatal:
        return CheckResult.no(fatal[0][:200], "startup dies when the key is unset")
    return CheckResult.yes(f"{len(references)} guarded references")


@check("default_restored")
async def default_restored(
    ctx: CheckContext, *, nonce: str, js: str = ""
) -> CheckResult:
    """After a reload the user's own work is gone but content is still shown.

    The exact shape of the corpus's most-rewarded illusion: a trier creates
    something, reloads, sees *a* populated screen, and reports that their work
    survived. One build was scored 30/30 for restoring an em-dash placeholder.
    This separates "your work came back" from "something came back".

    Passing means the penalty FIRES.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    read = js or USER_VISIBLE_JS
    try:
        await ctx.page.page.reload(wait_until="domcontentloaded")
        raw = await ctx.page.evaluate(read)
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read after reload: {exc}")
    observed = raw if isinstance(raw, str) else json.dumps(raw, default=str)
    if nonce in observed:
        return CheckResult.no("the user's own work survived")
    if len(observed.strip()) < 40:
        # Empty after reload is a plain persistence failure (R1's job), not the
        # impersonation this penalty is about.
        return CheckResult.no(f"{len(observed.strip())} chars", "restored nothing")
    return CheckResult.yes(observed[:200], "content restored, but not the user's")


# ------------------------------------------------------- DOM-agnostic reads


def _visible_text(raw: Any) -> str:
    return raw if isinstance(raw, str) else json.dumps(raw, default=str)


@check("text_matches")
async def text_matches(
    ctx: CheckContext,
    *,
    patterns: list[str] | None = None,
    pattern: str = "",
    absent: list[str] | None = None,
    js: str = "",
) -> CheckResult:
    """Every pattern appears in what the user can see; none of ``absent`` does.

    The workhorse for per-idea items, and DOM-agnostic on purpose. Forty builds
    of one idea share no ids, classes or structure, so a check written against
    ``#savings`` grades one build and silently returns nothing for the rest.
    Reading the rendered text is what a person does, and it is the only
    extraction that survives that much variation.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    wanted = list(patterns or ([pattern] if pattern else []))
    if not wanted and not absent:
        return CheckResult.unknown("text_matches needs a pattern")
    try:
        text = _visible_text(await ctx.page.evaluate(js or USER_VISIBLE_JS))
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read the page: {exc}")
    flags = re.IGNORECASE | re.DOTALL
    missing = [p for p in wanted if not re.search(p, text, flags)]
    leaked = [p for p in (absent or []) if re.search(p, text, flags)]
    observed = text[:300].replace("\n", " ")
    if missing:
        return CheckResult.no(observed, f"not found: {missing[0]!r}")
    if leaked:
        return CheckResult.no(observed, f"should not appear: {leaked[0]!r}")
    return CheckResult.yes(observed)


def _numbers_near(text: str, pattern: str) -> list[float]:
    """Every number captured by ``pattern``'s first group, as floats."""
    values = []
    for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
        raw = match.group(1) if match.groups() else match.group(0)
        try:
            values.append(float(str(raw).replace(",", "").strip().rstrip("%")))
        except (TypeError, ValueError):
            continue
    return values


@check("text_number_equals")
async def text_number_equals(
    ctx: CheckContext,
    *,
    pattern: str,
    expect: float,
    tolerance: float = 0.5,
    js: str = "",
) -> CheckResult:
    """A number the page displays equals what the grader computed.

    ``pattern`` captures the number from the rendered text, so it works across
    builds that share no markup. **Sign is part of the comparison**: the sharpest
    single finding in the corpus is a build that printed ``SAVED 166.9% larger``
    about a file it had nearly tripled, and scored 7.73 against an honest
    sibling's 2.53 for the identical bytes.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    try:
        text = _visible_text(await ctx.page.evaluate(js or USER_VISIBLE_JS))
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read the page: {exc}")
    values = _numbers_near(text, pattern)
    if not values:
        return CheckResult.no(
            text[:200].replace("\n", " "), f"no number matched {pattern!r}"
        )
    hit = [v for v in values if abs(v - expect) <= tolerance]
    observed = f"found {values[:5]}, expected {expect:g}"
    if hit:
        return CheckResult.yes(observed)
    nearest = min(values, key=lambda v: abs(v - expect))
    return CheckResult.no(observed, f"nearest is off by {nearest - expect:+g}")


@check("text_number_in_range")
async def text_number_in_range(
    ctx: CheckContext,
    *,
    pattern: str,
    low: float | None = None,
    high: float | None = None,
    js: str = "",
) -> CheckResult:
    """Every number the page displays for this label is physically possible.

    Catches the impossible-figure class without the grader needing to know the
    right answer: 12,000 WPM after one keystroke, 100% ATS on an empty resume, a
    saving above 100%.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    try:
        text = _visible_text(await ctx.page.evaluate(js or USER_VISIBLE_JS))
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read the page: {exc}")
    values = _numbers_near(text, pattern)
    if not values:
        return CheckResult.no(
            text[:200].replace("\n", " "), f"no number matched {pattern!r}"
        )
    bad = [
        v
        for v in values
        if (low is not None and v < low) or (high is not None and v > high)
    ]
    if bad:
        return CheckResult.no(f"{bad[:5]}", f"outside [{low}, {high}]")
    return CheckResult.yes(f"{values[:5]} within range")


def _view(snapshot: Any, view: str) -> str | None:
    """One view of a harness snapshot.

    Tolerates the older single-string shape so a grade taken before captures
    grew two views still reads.
    """
    if snapshot is None:
        return None
    if isinstance(snapshot, str):
        return snapshot
    return snapshot.get(view) or snapshot.get("text")


@check("third_party_requests")
def third_party_requests(
    ctx: CheckContext, *, allow: list[str] | None = None
) -> CheckResult:
    """The app fetched something off-machine. Passing means the penalty FIRES.

    The inverse of :func:`network_origins`, written separately rather than as an
    ``invert`` flag so a reader of the rubric can see which way round the item
    runs without holding a negation in their head.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    allowed = set(LOCAL_HOSTS) | set(allow or [])
    foreign = sorted(
        {
            _host_of(request["url"])
            for request in ctx.page.requests()
            if _host_of(request["url"]) and _host_of(request["url"]) not in allowed
        }
    )
    if foreign:
        return CheckResult.yes(", ".join(foreign[:5]), f"{len(foreign)} third-party")
    return CheckResult.no("all requests local")


@check("value_changes")
async def value_changes(
    ctx: CheckContext,
    *,
    before: str = "before",
    after: str = "after",
    should_change: bool = True,
    view: str = "text",
) -> CheckResult:
    """Two harness-taken snapshots differ (or deliberately do not).

    The grader takes them with the ``capture`` tool, which is what makes this
    honest: the model chooses the moment -- only it knows when the app has
    finished recomputing -- and the harness chooses what a snapshot is and stores
    it. The model never states the value, so it cannot state a change that did
    not happen.
    """
    first = _view(ctx.captures.get(before), view)
    second = _view(ctx.captures.get(after), view)
    if first is None or second is None:
        missing = before if first is None else after
        return CheckResult.unknown(f"the grader never captured {missing!r}")
    changed = first.strip() != second.strip()
    observed = f"{len(first)} chars -> {len(second)} chars, changed={changed}"
    if changed == should_change:
        return CheckResult.yes(observed)
    return CheckResult.no(
        observed, "value did not change" if should_change else "value changed"
    )


@check("distinct_sources")
async def distinct_sources(
    ctx: CheckContext, *, selector: str = "img,canvas", minimum: int = 2
) -> CheckResult:
    """At least N visually distinct image sources are on screen.

    Backs the before/after comparison items. A build that shows the same source
    twice, or renders the "after" pane from the original file, passes a human
    glance and fails here.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    # The selector is baked into the expression rather than passed as an
    # argument: PageHandle.evaluate takes a single expression, not a bound arg.
    survey = (
        "() => Array.from(document.querySelectorAll(" + json.dumps(selector) + "))"
        "  .filter(e => { const r = e.getBoundingClientRect();"
        "                 return r.width > 0 && r.height > 0; })"
        "  .map(e => e.tagName === 'CANVAS'"
        "    ? (e.toDataURL ? e.toDataURL().slice(0, 200) : 'canvas')"
        "    : (e.currentSrc || e.src || ''))"
        "  .filter(Boolean)"
    )
    try:
        sources = await ctx.page.evaluate(survey)
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read sources: {exc}")
    unique = {str(s) for s in (sources or [])}
    observed = f"{len(sources or [])} elements, {len(unique)} distinct"
    if len(unique) < minimum:
        return CheckResult.no(observed, f"expected at least {minimum} distinct")
    return CheckResult.yes(observed)


@check("captures_monotonic")
def captures_monotonic(
    ctx: CheckContext,
    *,
    labels: list[str],
    pattern: str,
    direction: str = "non_increasing",
    view: str = "text",
) -> CheckResult:
    """Numbers pulled from a sequence of harness snapshots move one way only.

    The quality-monotonicity shape: lower quality must not produce a larger
    file. Reading it from snapshots the harness took means the model supplies
    only the ordering of actions, never the sizes it is being judged on.
    """
    missing = [label for label in labels if label not in ctx.captures]
    if missing:
        return CheckResult.unknown(f"the grader never captured {missing}")
    series = []
    for label in labels:
        snapshot = _view(ctx.captures[label], view) or ""
        values = _numbers_near(snapshot, pattern)
        if not values:
            return CheckResult.no(
                snapshot[:150], f"no number matched {pattern!r} at {label!r}"
            )
        series.append(values[0])
    pairs = list(zip(series, series[1:], strict=False))
    ok = (
        all(b <= a for a, b in pairs)
        if direction == "non_increasing"
        else all(b >= a for a, b in pairs)
    )
    observed = " -> ".join(f"{v:g}" for v in series)
    if ok:
        return CheckResult.yes(observed)
    return CheckResult.no(observed, f"not {direction.replace('_', '-')}")


@check("download_size_matches")
async def download_size_matches(
    ctx: CheckContext, *, pattern: str, tolerance: float = 0.02, js: str = ""
) -> CheckResult:
    """The size the app reports equals the size of the file it actually produced.

    Both halves come from the harness: the byte count from the downloaded file,
    the claimed figure by regex from the rendered text. A build can only pass by
    telling the truth about a file it really wrote.

    ``tolerance`` is fractional, because apps legitimately round to KB.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    if not ctx.downloads:
        return CheckResult.no("no download fired")
    path = Path(ctx.downloads[-1].get("path", ""))
    if not path.is_file():
        return CheckResult.no("download vanished")
    actual = len(path.read_bytes())
    try:
        text = _visible_text(await ctx.page.evaluate(js or USER_VISIBLE_JS))
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read the page: {exc}")
    claims = _numbers_near(text, pattern)
    if not claims:
        return CheckResult.no(text[:150], f"no reported size matched {pattern!r}")
    # Accept the figure in bytes or in KB, since apps display either.
    candidates = [c for c in claims] + [c * 1024 for c in claims]
    best = min(candidates, key=lambda c: abs(c - actual))
    slack = max(actual * tolerance, 1024.0)
    observed = f"reported ~{best:g}B, actual {actual}B"
    if abs(best - actual) <= slack:
        return CheckResult.yes(observed)
    return CheckResult.no(observed, f"off by {best - actual:+.0f}B")


# ------------------------------------------------ primitives the golden set asked for


@check("sql_executes")
async def sql_executes(  # noqa: PLR0911
    ctx: CheckContext,
    *,
    js: str = "",
    tables: list[str] | None = None,
    foreign_keys: list[dict] | None = None,
) -> CheckResult:
    """Exported SQL actually runs, and produces the schema it claims.

    A text check can confirm the word ``REFERENCES`` appears. It cannot confirm
    the statement parses: a MySQL ``AUTO_INCREMENT ... ENGINE=InnoDB`` export
    satisfies every regex and will not execute anywhere. Running it is the only
    honest test, and stdlib ``sqlite3`` against ``:memory:`` adds no dependency
    and no subprocess.
    """
    import sqlite3

    if ctx.page is None:
        return CheckResult.unknown("no page")
    try:
        text = _visible_text(await ctx.page.evaluate(js or USER_VISIBLE_JS))
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read the page: {exc}")
    statements = re.findall(
        r"CREATE\s+TABLE[\s\S]*?;", text, re.IGNORECASE
    ) + re.findall(r"ALTER\s+TABLE[\s\S]*?;", text, re.IGNORECASE)
    if not statements:
        return CheckResult.unknown("no CREATE TABLE statement on the page")
    script = "\n".join(statements)
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(script)
    except sqlite3.Error as exc:
        return CheckResult.no(f"{type(exc).__name__}: {exc}", "the export does not run")
    try:
        found = {
            row[0].lower()
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = [t for t in (tables or []) if t.lower() not in found]
        if missing:
            return CheckResult.no(f"tables {sorted(found)}", f"missing {missing}")
        for spec in foreign_keys or []:
            rows = list(
                connection.execute(
                    f"PRAGMA foreign_key_list({spec['table']})"  # noqa: S608
                )
            )
            wanted = (spec.get("references") or "").lower()
            if not any(str(row[2]).lower() == wanted for row in rows):
                return CheckResult.no(
                    f"{spec['table']} -> {[r[2] for r in rows]}",
                    f"no foreign key to {wanted}",
                )
        return CheckResult.yes(f"executed; tables {sorted(found)}")
    finally:
        connection.close()


@check("clipboard_matches")
async def clipboard_matches(
    ctx: CheckContext,
    *,
    patterns: list[str] | None = None,
    absent: list[str] | None = None,
) -> CheckResult:
    """The clipboard's contents match a pattern, rather than a fixed string.

    The regex form of :func:`clipboard_equals`, for the items that assert a
    *shape* -- "rendered HTML, not Markdown source", "no script tags". Equality
    cannot express those without knowing the build's exact output.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    try:
        text = await ctx.page.read_clipboard()
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read the clipboard: {exc}")
    if not text:
        return CheckResult.no("clipboard empty", "nothing was copied")
    flags = re.IGNORECASE | re.DOTALL
    missing = [p for p in (patterns or []) if not re.search(p, text, flags)]
    leaked = [p for p in (absent or []) if re.search(p, text, flags)]
    observed = text[:200]
    if missing:
        return CheckResult.no(observed, f"clipboard lacks {missing[0]!r}")
    if leaked:
        return CheckResult.no(observed, f"clipboard contains {leaked[0]!r}")
    return CheckResult.yes(observed)


@check("computed_style_changes")
async def computed_style_changes(
    ctx: CheckContext, *, selector: str = "", prop: str = "background-color"
) -> CheckResult:
    """A computed CSS value differs between two harness snapshots.

    A theme switch changes appearance, not text, so ``value_changes`` cannot see
    it -- and because a snapshot includes ``select.value``, that check would
    *pass* on exactly the builds whose theme control does nothing.

    ``selector`` defaults to empty, meaning "anywhere in the fingerprint". That
    is the safe default, and the old default of ``body`` was a trap: a build
    that themes only its own preview pane repaints no ancestor, so pinning
    ``body`` fails a theme switch that plainly worked. Name a selector only when
    the item is specifically about one element.
    """
    first = _view(ctx.captures.get("before"), "style")
    second = _view(ctx.captures.get("after"), "style")
    if first is None or second is None:
        return CheckResult.unknown("the grader never captured before/after styles")
    # ``selector`` used to be accepted and ignored -- the check compared the whole
    # fingerprint whatever was asked for. Narrowing to one key matters for a
    # scoped repaint: comparing everything also passes when some *other* element
    # changed, which is the wrong finding for an item about one pane.
    before_map, after_map = _style_map(first), _style_map(second)
    if selector and before_map is not None and after_map is not None:
        if selector in before_map or selector in after_map:
            was, now = before_map.get(selector), after_map.get(selector)
            if was != now:
                return CheckResult.yes(f"{selector}: {was} -> {now}")
            return CheckResult.no(f"{selector}: {was}", f"{prop} unchanged")
    if first.strip() != second.strip():
        return CheckResult.yes(f"{first[:80]} -> {second[:80]}")
    return CheckResult.no(first[:120], f"{prop} unchanged on {selector}")


def _style_map(view: str) -> dict[str, str] | None:
    """The style view as a selector -> fingerprint mapping, or ``None``.

    Tolerates the older list-shaped view so a grade recorded before the harness
    change still reads back.
    """
    try:
        loaded = json.loads(view)
    except (TypeError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


@check("download_absent")
def download_absent(ctx: CheckContext, *, of_kind: str = "") -> CheckResult:
    """No file was produced. Passing means the penalty FIRES.

    The ``window.print()`` case: an "Export PDF" control that opens a print
    dialog and writes nothing. ``download`` returns *no* for that, and a penalty
    needs *yes*, so the condition needs its own primitive rather than a negated
    flag buried in a parameter.
    """
    if not ctx.downloads:
        return CheckResult.yes("no file was produced")
    latest = ctx.downloads[-1]
    path = Path(latest.get("path", ""))
    if not path.is_file():
        return CheckResult.yes("download vanished", latest.get("name", ""))
    if of_kind:
        if magic_matches(path.read_bytes(), of_kind) is False:
            return CheckResult.yes(
                latest.get("name", "?"), f"produced a file, but not {of_kind}"
            )
    return CheckResult.no(f"{latest.get('name', '?')} was produced")


@check("text_pattern_count")
async def text_pattern_count(
    ctx: CheckContext,
    *,
    pattern: str,
    minimum: int | None = None,
    maximum: int | None = None,
    js: str = "",
) -> CheckResult:
    """How many times a pattern appears in the rendered text.

    For items counting occurrences rather than asserting one -- recorded checks
    accumulating on a schedule, incidents opening exactly once.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    try:
        text = _visible_text(await ctx.page.evaluate(js or USER_VISIBLE_JS))
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read the page: {exc}")
    count = len(re.findall(pattern, text, re.IGNORECASE | re.DOTALL))
    observed = f"{count} occurrences of {pattern!r}"
    if minimum is not None and count < minimum:
        return CheckResult.no(observed, f"expected at least {minimum}")
    if maximum is not None and count > maximum:
        return CheckResult.no(observed, f"expected at most {maximum}")
    return CheckResult.yes(observed)


@check("text_numbers_distinct")
async def text_numbers_distinct(
    ctx: CheckContext,
    *,
    pattern: str,
    minimum: int = 2,
    maximum: int | None = None,
    js: str = "",
) -> CheckResult:
    """Numbers matching a pattern take at least N distinct values.

    ``maximum`` inverts it for the penalty form: ``maximum: 1`` passes -- and so
    fires the penalty -- exactly when every reading is the same number.

    Catches the constant-value fake: a monitor reporting the same response time
    for every check is not measuring anything, and a single number repeated is
    indistinguishable from a real measurement to any check that only reads one.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    try:
        text = _visible_text(await ctx.page.evaluate(js or USER_VISIBLE_JS))
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read the page: {exc}")
    values = _numbers_near(text, pattern)
    unique = sorted(set(values))
    observed = f"{len(values)} readings, {len(unique)} distinct: {unique[:6]}"
    if not values:
        return CheckResult.no(text[:150], f"no number matched {pattern!r}")
    if maximum is not None:
        if len(unique) > maximum:
            return CheckResult.no(observed, f"more than {maximum} distinct")
        return CheckResult.yes(observed)
    if len(unique) < minimum:
        return CheckResult.no(observed, f"expected at least {minimum} distinct")
    return CheckResult.yes(observed)


@check("console_errors_present")
def console_errors_present(
    ctx: CheckContext, *, allow: list[str] | None = None
) -> CheckResult:
    """An uncaught error fired. Passing means the penalty FIRES."""
    if ctx.page is None:
        return CheckResult.unknown("no page")
    ignored = list(allow or []) + [
        "cdn.tailwindcss.com should not be used in production"
    ]
    errors = [
        line
        for line in ctx.page.drain_console()
        if line.startswith("pageerror:") and not any(skip in line for skip in ignored)
    ]
    if errors:
        return CheckResult.yes(f"{len(errors)} page errors", "; ".join(errors[:3]))
    return CheckResult.no("no page errors")


@check("is_operable")
async def is_operable(
    ctx: CheckContext, *, selector: str, min_size: int = 4
) -> CheckResult:
    """An element exists and a person could actually interact with it.

    Stronger than :func:`is_real_control`, which checks only the tag. The
    dominant failure in the typing corpus is a genuine ``<input>`` rendered
    ``opacity-0 pointer-events-none w-0 h-0`` and driven by a keydown listener
    on something else: correct by tag, impossible to click, and responsible for
    43% of all failed type attempts. Visibility, size, opacity and
    ``pointer-events`` are the four properties that separate the two.
    """
    if ctx.page is None:
        return CheckResult.unknown("no page")
    survey = (
        "() => Array.from(document.querySelectorAll(" + json.dumps(selector) + "))"
        "  .map(e => { const r = e.getBoundingClientRect();"
        "              const c = getComputedStyle(e);"
        "    return { tag: e.tagName.toLowerCase(), w: r.width, h: r.height,"
        "             opacity: parseFloat(c.opacity),"
        "             pointer: c.pointerEvents,"
        "             visibility: c.visibility, display: c.display }; })"
    )
    try:
        found = await ctx.page.evaluate(survey)
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not survey {selector!r}: {exc}")
    if not found:
        return CheckResult.no("not present", f"nothing matches {selector!r}")
    usable = [
        e
        for e in found
        if e["w"] >= min_size
        and e["h"] >= min_size
        and (e["opacity"] is None or e["opacity"] > 0.05)
        and e["pointer"] != "none"
        and e["visibility"] != "hidden"
        and e["display"] != "none"
    ]
    observed = f"{len(found)} matched, {len(usable)} operable"
    if not usable:
        worst = found[0]
        return CheckResult.no(
            observed,
            f"<{worst['tag']}> {worst['w']:.0f}x{worst['h']:.0f} "
            f"opacity={worst['opacity']} pointer-events={worst['pointer']}",
        )
    return CheckResult.yes(observed)


# ------------------------------------------------------------------ canvas


def _canvas_view(snapshot: Any) -> list[str] | None:
    """The per-canvas fingerprints of one snapshot, or ``None``."""
    raw = _view(snapshot, "canvas")
    if raw is None:
        return None
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return loaded if isinstance(loaded, list) else None


@check("canvas_changed")
async def canvas_changed(
    ctx: CheckContext,
    *,
    before: str = "before",
    after: str = "after",
    should_change: bool = True,
    index: int = 0,
) -> CheckResult:
    """What is drawn on the canvas differs between two harness snapshots.

    The whiteboard case, and the reason a whole idea was ungradeable: 125 of 127
    builds render their scene into a ``<canvas>``, so a stroke changes no text,
    no form state and no computed style. ``value_changes`` reports "unchanged"
    on a build whose drawing works perfectly.

    Compares a 32x32 downsample rather than raw pixels, so antialiasing and
    device pixel ratio do not read as a change.
    """
    first, second = (
        _canvas_view(ctx.captures.get(before)),
        _canvas_view(ctx.captures.get(after)),
    )
    if first is None or second is None:
        missing = before if first is None else after
        return CheckResult.unknown(f"the grader never captured {missing!r}")
    if not first and not second:
        return CheckResult.unknown("the page has no canvas")
    if any(v == "tainted" for v in first + second):
        # A cross-origin image drawn into the canvas makes getImageData throw.
        # Reported as unknown, not failure: the app may be working fine.
        return CheckResult.unknown("canvas is origin-tainted, pixels unreadable")
    if index >= len(first) or index >= len(second):
        return CheckResult.unknown(f"no canvas at index {index}")
    changed = first[index] != second[index]
    observed = f"canvas[{index}] changed={changed}"
    if changed == should_change:
        return CheckResult.yes(observed)
    return CheckResult.no(
        observed, "nothing was drawn" if should_change else "the canvas changed"
    )


@check("canvas_ink")
async def canvas_ink(
    ctx: CheckContext, *, label: str = "after", minimum: float = 0.01, index: int = 0
) -> CheckResult:
    """The canvas holds content, not just its background.

    ``ink`` is the share of the downsampled grid that is *not* the single most
    common colour, so a blank canvas and a canvas painted one flat colour both
    read as 0. That is what separates "drew something" from "cleared and
    repainted the background", which a plain change check cannot do.

    A dotted background grid is content by this measure, so the threshold is a
    parameter rather than a constant -- calibrate it against an untouched build.
    """
    view = _canvas_view(ctx.captures.get(label))
    if view is None:
        return CheckResult.unknown(f"the grader never captured {label!r}")
    if index >= len(view):
        return CheckResult.unknown(f"no canvas at index {index}")
    entry = view[index]
    if entry in ("tainted", "empty"):
        return CheckResult.unknown(f"canvas is {entry}")
    try:
        ink = float(entry.split(":", 1)[0])
    except (ValueError, IndexError):
        return CheckResult.unknown("unreadable canvas fingerprint")
    observed = f"ink={ink:.3f}"
    if ink >= minimum:
        return CheckResult.yes(observed)
    return CheckResult.no(observed, f"below {minimum}")


# --------------------------------------------------------------- downloads 2


@check("downloads_distinct")
def downloads_distinct(
    ctx: CheckContext, *, minimum: int = 2, last: int = 2
) -> CheckResult:
    """The last ``last`` downloads are ``minimum`` different files.

    "Generate twice, get two different pictures" has no other honest check. The
    DOM cannot answer it: a re-served identical image arrives under a fresh
    ``blob:`` URL every time, so comparing ``img.src`` finds a difference that
    is not there. Comparing bytes is the only way to tell a second generation
    from the first one shown twice.

    ``unknown`` -- not failure -- when fewer than ``last`` downloads fired, so
    "the download button is broken" stays distinct from "the same file twice".
    """
    if len(ctx.downloads) < last:
        return CheckResult.unknown(
            f"only {len(ctx.downloads)} download(s), need {last}"
        )
    digests = []
    for entry in ctx.downloads[-last:]:
        path = Path(entry.get("path", ""))
        if not path.is_file():
            return CheckResult.unknown("a download vanished before it could be read")
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    distinct = len(set(digests))
    observed = f"{distinct} distinct of {len(digests)}"
    if distinct >= minimum:
        return CheckResult.yes(observed)
    return CheckResult.no(observed, f"fewer than {minimum} distinct files")


@check("text_pattern_distinct")
async def text_pattern_distinct(
    ctx: CheckContext, *, pattern: str, minimum: int = 3, js: str = ""
) -> CheckResult:
    """``pattern`` matches at least ``minimum`` *different* strings.

    ``text_pattern_count`` counts occurrences, which over-credits: one theme
    named twice ("Fira Code" and "Font: Fira Code") reaches two occurrences of
    one name. Counting distinct captures is what an item like "at least three
    selectable themes" actually claims.
    """
    if ctx.page is None:
        return CheckResult.unknown("the app never started")
    expression = js or "() => document.body ? document.body.innerText : ''"
    try:
        haystack = await ctx.page.evaluate(expression)
    except Exception as exc:  # noqa: BLE001
        return CheckResult.unknown(f"could not read the page: {exc}")
    if not isinstance(haystack, str):
        haystack = json.dumps(haystack, default=str)
    try:
        found = re.findall(pattern, haystack, re.IGNORECASE)
    except re.error as exc:
        return CheckResult.unknown(f"bad pattern {pattern!r}: {exc}")
    # A group-bearing pattern yields tuples; the whole match is the fallback.
    normalised = {
        (m if isinstance(m, str) else next((g for g in m if g), "")).strip().lower()
        for m in found
    }
    normalised.discard("")
    observed = f"{len(normalised)} distinct of {len(found)} matches"
    if len(normalised) >= minimum:
        return CheckResult.yes(observed, ", ".join(sorted(normalised)[:6]))
    return CheckResult.no(observed, f"fewer than {minimum} distinct")
