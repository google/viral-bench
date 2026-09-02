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

"""Drive the viewer in a real browser and report what rendered.

Checking a UI by asserting on its JSON API proves the server works, not that the
page does -- a broken template, a JS exception or a tab that renders `[object
Object]` all pass an API test. This loads the page in headed-capable Chrome, runs
assertions against the live DOM, and writes a screenshot, so a claim that the
viewer "looks right" has evidence behind it.

    viz/tools/shoot.py founder <build_id> [--tab Thinking] [--out name.png]
    viz/tools/shoot.py crowd <run_id> [--tab "Trial replay"]
    viz/tools/shoot.py url http://localhost:8000/ --out home.png

Runs under the benchmark's own venv, which is where Playwright lives:

    .venv/bin/python viz/tools/shoot.py ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

SHOTS = Path(__file__).resolve().parent.parent / "cache" / "shots-ui"

#: What the page must have painted before a screenshot means anything. Waiting on
#: these rather than a fixed sleep is what keeps the check honest on a slow build.
READY = {
    "founder": "() => document.querySelectorAll('.ev').length > 0",
    "crowd": "() => document.querySelectorAll('.agent-row').length > 0",
    "rubric": "() => document.querySelectorAll('#items tbody tr').length > 0",
    "url": "() => true",
}


def probe(page) -> dict:
    """Everything worth asserting on, read out of the live DOM in one pass."""
    return page.evaluate("""() => {
      const txt = (s) =>
        [...document.querySelectorAll(s)].map(n => n.textContent.trim());
      const body = document.body.innerText;
      return {
        title: document.title,
        tabs: txt('#tabs .tab'),
        midTabs: txt('#midTabs .tab'),
        chips: txt('#head .chip').slice(0, 24),
        lanes: txt('.lane-name'),
        feedRows: document.querySelectorAll('.ev').length,
        feedKinds: [...new Set([...document.querySelectorAll('.ev')]
          .map(n => (n.className.match(/kind-(\\w+)/) || [])[1]).filter(Boolean))],
        agents: document.querySelectorAll('.agent-row').length,
        itemRows: document.querySelectorAll('#items table.grid tbody tr').length,
        verdicts: txt('#items .ok-text, #items .err-text, #items .faint').slice(0, 8),
        statLabels: txt('#head .stat .k'),
        statValues: txt('#head .stat .v'),
        steps: document.querySelectorAll('.step').length,
        images: [...document.querySelectorAll('img')].map(i => i.naturalWidth > 0),
        // The two failure modes a screenshot alone would not catch.
        objectObject:
          (body.match(/\\[object (Object|HTMLDivElement)\\]/g) || []).length,
        undefinedText: (body.match(/\\bundefined\\b/g) || []).length,
        bodyChars: body.length,
      };
    }""")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view", choices=("founder", "crowd", "rubric", "url"))
    parser.add_argument("target", help="build id, run id, or full URL")
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument(
        "--tab", action="append", default=[], help="tab to click (repeatable)"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--full", action="store_true", help="full-page screenshot")
    parser.add_argument("--click", default=None, help="CSS selector to click first")
    args = parser.parse_args(argv)

    if args.view == "url":
        url = args.target
    else:
        key = {"founder": "build", "crowd": "run", "rubric": "grade"}[args.view]
        url = f"{args.base}/{args.view}?{key}={args.target}"

    SHOTS.mkdir(parents=True, exist_ok=True)
    out = SHOTS / (args.out or f"{args.view}-{args.target[:40]}.png")

    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1680, "height": 1150})
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on(
            "console",
            lambda m: (
                errors.append(f"console.{m.type}: {m.text}")
                if m.type == "error"
                else None
            ),
        )
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        try:
            page.wait_for_function(READY[args.view], timeout=45000)
        except Exception:
            errors.append("readiness check timed out")
        page.wait_for_timeout(1200)

        if args.click:
            page.click(args.click, timeout=10000)
            page.wait_for_timeout(800)

        for tab in args.tab:
            # Tabs are plain divs, so match on text rather than a role selector.
            page.evaluate(
                """(name) => {
                  const t = [...document.querySelectorAll('.tab')]
                    .find(n => n.textContent.trim().startsWith(name));
                  if (t) t.click();
                }""",
                tab,
            )
            page.wait_for_timeout(1000)

        result = probe(page)
        page.screenshot(path=str(out), full_page=args.full)
        browser.close()

    result["url"] = url
    result["screenshot"] = str(out)
    result["errors"] = errors[:12]
    print(json.dumps(result, indent=1))
    return 1 if errors or result["objectObject"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
