#!/usr/bin/env python3
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

"""Measure the build cohort once, so check authors do not each re-measure it.

Why this exists. Authoring a ``check:`` block is a bet about what the corpus
looks like -- "is the scene in a canvas or in SVG?", "what do builds
name their export file?", "does anything set an accessible name?". Get it wrong
and the check grades the rendering technology rather than the feature.

The first authoring pass let each subagent answer those questions for itself, by
grepping 127-141 build directories from inside its own agent loop. That was the
single most expensive thing in the pass -- one worker accumulated 317 KB of shell
output across 95 calls, crossed the provider's context threshold, and spent 21
hours producing nothing -- and it made the answers unreviewable, because each
number existed only as a sentence inside one worker's report.

Measuring once fixes all three: it is fast, the numbers are identical across
ideas, and they land in a file that can be read, diffed and re-derived. Re-run it
after any harness change that alters what a primitive can see. The source caps in
particular used to truncate every file at 20 KB, which silently changed the answer
to every "does this pattern appear anywhere" question.

    python scripts/rubric_corpus_facts.py                  # refresh the JSON
    python scripts/rubric_corpus_facts.py --sheet team_wiki  # one idea, readable

RE-RUN IT PER COHORT. The committed ``_corpus_facts.json`` was measured on one
cohort, and a later cohort rebuilds some of the arms, so its answers to "does the
scene live in a canvas?" and "what extension do downloads use?" can differ. A
check authored against a stale sheet is a bet on the wrong corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FLEET = REPO / "builds" / "fleet.json"
OUT = REPO / "ideas" / "rubrics" / "_corpus_facts.json"

#: Client-served text worth reading. Anything else is an asset or a build
#: artefact and only inflates the scan.
CLIENT_SUFFIXES = {".html", ".htm", ".js", ".jsx", ".ts", ".tsx", ".css", ".svelte"}
SERVER_SUFFIXES = {".py", ".go", ".rb", ".php"}

#: Per-file read cap. Deliberately generous: the crowd's 20 KB cap is what made
#: an absent-pattern check false-fire on 18 of 128 builds, because one of them
#: registered its handler at byte 30k of a 61k bundle.
MAX_FILE_BYTES = 2_000_000


@dataclass
class Probe:
    """One yes/no question asked of every build's client source."""

    name: str
    pattern: str
    where: str = "client"  # client | server | any
    note: str = ""

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, re.IGNORECASE)


#: The questions the first authoring pass kept asking. Each one decided at least
#: one real check, and the note records what it decides.
PROBES: list[Probe] = [
    Probe("canvas_2d", r"getContext\(\s*['\"]2d", note="scene is pixels, not DOM"),
    Probe("canvas_el", r"<canvas\b|createElement\(\s*['\"]canvas"),
    Probe("svg_geometry", r"<svg\b[^>]*>[\s\S]{0,400}?<(path|rect|circle|line)\b"),
    Probe("excalidraw", r"excalidraw"),
    Probe("fabric_konva", r"\bfabric\b|\bkonva\b"),
    Probe("pointer_events", r"pointerdown|pointermove|mousedown", note="drag target"),
    Probe("aria_label", r"aria-label\s*=", note="controls_have_names / aria_names"),
    Probe("title_attr", r"\btitle\s*=\s*['\"]"),
    Probe("local_storage", r"localStorage", note="survives_reload"),
    Probe("indexed_db", r"indexedDB"),
    Probe("download_anchor", r"\.download\s*=|createObjectURL", note="download items"),
    Probe("print_only", r"window\.print\(", note="the dead-export shape"),
    Probe("clipboard", r"navigator\.clipboard"),
    Probe("file_input", r"type\s*=\s*['\"]file"),
    Probe("fetch_or_xhr", r"\bfetch\s*\(|XMLHttpRequest"),
    Probe("websocket", r"new WebSocket"),
    Probe("service_worker", r"serviceWorker"),
    Probe("contenteditable", r"contenteditable"),
    Probe("dialog_element", r"<dialog\b|showModal\("),
    Probe("css_variables", r"--[a-z-]+\s*:", note="themeable via :root"),
    Probe("dark_class_toggle", r"classList\.(toggle|add)\(\s*['\"](dark|theme)"),
    Probe("auth_gate", r"session|cookie|authorization", where="server"),
    Probe("api_key_env", r"GEMINI_API_KEY|GOOGLE_API_KEY|process\.env", where="any"),
    Probe("genai_client", r"google-genai|@google/genai|generativelanguage"),
]

#: Extensions builds give their downloads, which is what a ``download``
#: check must pin. ``ctx.downloads[-1]`` is scoped per item now, but an item with
#: two exports still needs the right one.
DOWNLOAD_NAME = re.compile(r"""\.download\s*=\s*[`'"]([^`'"]{1,80})""")
#: Files a build serves that a "no key in client assets" claim must cover.
SECRET_HINT = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|sk-[0-9A-Za-z]{20,}")


@dataclass
class IdeaFacts:
    idea_id: str
    builds: int = 0
    with_client_source: int = 0
    hits: Counter = field(default_factory=Counter)
    download_ext: Counter = field(default_factory=Counter)
    largest_client_file: int = 0
    over_20kb: int = 0
    leaked_key: int = 0

    def as_dict(self) -> dict:
        n = self.with_client_source or 1
        return {
            "builds": self.builds,
            "with_client_source": self.with_client_source,
            "probes": {
                k: {"n": v, "pct": round(100 * v / n)}
                for k, v in sorted(self.hits.items())
            },
            "download_extensions": dict(self.download_ext.most_common(8)),
            "largest_client_file_bytes": self.largest_client_file,
            "builds_with_a_file_over_20kb": self.over_20kb,
            "builds_with_a_literal_key": self.leaked_key,
        }


def _read(path: Path) -> str:
    try:
        return path.read_bytes()[:MAX_FILE_BYTES].decode("utf-8", errors="replace")
    except OSError:
        return ""


def scan_build(app_dir: Path, facts: IdeaFacts) -> None:
    client, server = [], []
    for path in app_dir.rglob("*"):
        if not path.is_file() or "node_modules" in path.parts:
            continue
        if path.suffix.lower() in CLIENT_SUFFIXES:
            client.append(path)
        elif path.suffix.lower() in SERVER_SUFFIXES:
            server.append(path)
    if not client:
        return
    facts.with_client_source += 1

    client_text, server_text = [], []
    any_big = False
    for path in client:
        text = _read(path)
        size = len(text)
        facts.largest_client_file = max(facts.largest_client_file, size)
        any_big = any_big or size > 20_000
        client_text.append(text)
    if any_big:
        facts.over_20kb += 1
    for path in server:
        server_text.append(_read(path))

    joined_client = "\n".join(client_text)
    joined_server = "\n".join(server_text)
    joined_any = joined_client + "\n" + joined_server

    for probe in PROBES:
        haystack = {
            "client": joined_client,
            "server": joined_server,
            "any": joined_any,
        }[probe.where]
        if haystack and probe.compiled().search(haystack):
            facts.hits[probe.name] += 1

    seen_ext = set()
    for name in DOWNLOAD_NAME.findall(joined_client):
        suffix = Path(name.split("?")[0]).suffix.lower()
        if suffix and suffix not in seen_ext:
            seen_ext.add(suffix)
            facts.download_ext[suffix] += 1
    if SECRET_HINT.search(joined_client):
        facts.leaked_key += 1


def collect(cohort: str = "r4", generation: str = "") -> dict:
    """Survey every build in scope. See viral_bench.rubric.corpus for selection."""
    sys.path.insert(0, str(REPO / "src"))
    from viral_bench.rubric.corpus import select_builds

    builds = select_builds(REPO / "builds", cohort=cohort, generation=generation)
    scope = generation or cohort
    if not builds:
        raise SystemExit(f"no builds in scope for {scope!r} -- is the cohort tagged?")

    by_idea: dict[str, IdeaFacts] = {}
    missing = 0
    for build in builds:
        facts = by_idea.setdefault(build.idea_id, IdeaFacts(build.idea_id))
        facts.builds += 1
        app_dir = REPO / "builds" / "work" / build.build_id / "app"
        if not app_dir.is_dir():
            missing += 1
            continue
        scan_build(app_dir, facts)

    return {
        "generation": scope,
        "builds_indexed": len(builds),
        "builds_without_an_app_dir": missing,
        "probe_notes": {p.name: p.note for p in PROBES if p.note},
        "ideas": {name: f.as_dict() for name, f in sorted(by_idea.items())},
    }


def sheet(data: dict, idea_id: str) -> str:
    """A compact, readable facts sheet for one idea."""
    idea = data["ideas"].get(idea_id)
    if idea is None:
        raise SystemExit(f"no facts for {idea_id!r}")
    n = idea["with_client_source"] or 1
    lines = [
        f"# Corpus facts: {idea_id} ({data['generation']})",
        "",
        f"{idea['builds']} builds indexed, {n} with client source.",
        f"Largest client file {idea['largest_client_file_bytes']:,} bytes; "
        f"{idea['builds_with_a_file_over_20kb']} builds ship a file over 20 KB.",
        "",
        "| signal | builds | share |",
        "|---|---|---|",
    ]
    for name, hit in sorted(idea["probes"].items(), key=lambda kv: -kv[1]["n"]):
        if hit["n"] == 0:
            continue
        note = data["probe_notes"].get(name, "")
        label = f"{name}: {note}" if note else name
        lines.append(f"| {label} | {hit['n']}/{n} | {hit['pct']}% |")
    if idea["download_extensions"]:
        exts = ", ".join(f"`{k}` ({v})" for k, v in idea["download_extensions"].items())
        lines += ["", f"**Download extensions used:** {exts}"]
    if idea["builds_with_a_literal_key"]:
        lines += [
            "",
            f"**Literal API key in client source:** "
            f"{idea['builds_with_a_literal_key']} builds",
        ]
    zero = [k for k, v in idea["probes"].items() if v["n"] == 0]
    if zero:
        lines += [
            "",
            f"**Absent everywhere** (a check on these discriminates "
            f"nothing): {', '.join(sorted(zero))}",
        ]
    return "\n".join(lines)


def catalogue() -> str:
    """The primitive table, generated from the live registry.

    Hand-maintained, it drifts -- and a drifted catalogue is worse than none. An
    author binds a parameter that no longer exists, the call raises TypeError,
    the item records ``unknown``, and those points are unearnable on every build
    with nothing appearing to be wrong. That was two of the first pass's bugs.
    """
    import inspect

    sys.path.insert(0, str(REPO / "src"))
    from viral_bench.rubric.checks import registry

    reg = registry()
    rows = ["| primitive | params | what it decides |", "|---|---|---|"]
    for name in sorted(reg):
        signature = inspect.signature(reg[name])
        params = [
            p.name
            if p.default is inspect.Parameter.empty
            else f"{p.name}={p.default!r}"
            for p in list(signature.parameters.values())[1:]
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        ]
        doc = (inspect.getdoc(reg[name]) or "").split("\n")[0]
        rows.append(f"| `{name}` | `{', '.join(params)}` | {doc} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="r4")
    parser.add_argument(
        "--generation", default="", help="legacy r3 fleet-key selection"
    )
    parser.add_argument("--sheet", metavar="IDEA", help="print one idea's sheet")
    parser.add_argument(
        "--catalogue",
        action="store_true",
        help="print the primitive catalogue for ideas/rubrics/AUTHORING.md",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    if args.catalogue:
        print(catalogue())
        return 0

    if args.sheet and args.out.is_file():
        print(sheet(json.loads(args.out.read_text()), args.sheet))
        return 0

    data = collect(args.cohort, args.generation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    ideas = data["ideas"]
    print(
        f"{args.out.relative_to(REPO)}: {len(ideas)} ideas, "
        f"{data['builds_indexed']} builds, "
        f"{data['builds_without_an_app_dir']} without an app dir"
    )
    if args.sheet:
        print()
        print(sheet(data, args.sheet))
    return 0


if __name__ == "__main__":
    sys.exit(main())
