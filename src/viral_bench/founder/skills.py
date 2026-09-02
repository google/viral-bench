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

"""Role skills for the founder team: reusable SKILL.md playbooks.

Each founder specialist can load a small set of on-demand **skills** -- opencode's
native mechanism for reusable, discoverable instructions (see
https://opencode.ai/docs/skills/). A skill is a ``SKILL.md`` file with YAML
frontmatter (``name`` + ``description``) and a body of concrete guidance. opencode
lists available skills in its ``skill`` tool and the agent loads the full body
on demand.

We ship the skills defined here into each build's private
``<workspace_root>/.opencode/skills/`` directory (so they are discovered when
opencode runs with ``--dir app`` but never ship inside the app), and gate each
skill to a single role via that role's ``permission.skill`` map (see
:mod:`viral_bench.founder.opencode_agents`). This is a real capability boost: the
Designer that can load the ``virality-playbook`` is measurably better at its job
than a generic build agent without it.

The content is intentionally self-contained and runtime-agnostic so it stays
useful whatever stack a team picks.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "SKILLS",
    "all_skill_names",
    "skill_markdown",
    "write_skills",
]

_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# Skill bodies (markdown, without frontmatter). Keyed by skill name; the name
# must match the directory that will contain the SKILL.md (opencode requirement).
_SKILL_BODIES: dict[str, tuple[str, str]] = {
    "runtime-architecture": (
        "Choose a stack and architecture that runs cleanly in the target runtime "
        "and de-risk it before teammates depend on it.",
        """\
## What I do
Help the Architect pick a **simple, reliable** stack and shape the app so the
rest of the team can move fast without hitting runtime walls.

## Runtime-fit rules
- Pick the stack the IDEA needs, then keep it as simple as that allows. A
  client-app with no backend should not grow one; a full-stack-app needs a real
  server and a database and should not be faked with browser storage. Fewer
  moving parts start faster and fail in fewer ways, but "simple" never means
  shipping less than the idea asks for.
- A build step is fine (Node and npm are available) as long as the manifest
  states the exact commands and the app still starts within 90 seconds.
- Bind servers to `0.0.0.0`, never localhost: the app is reached from outside its
  container. Framework dev servers default to loopback -- pass `--host 0.0.0.0`.
- Persist anything that must outlive a restart to `$VIRALBENCH_DATA_DIR`
  (`/data`). The app directory is a throwaway clone and is re-made on restart.
- If the app needs ANY server-side route (e.g. an API-key proxy), write a
  CONCURRENT server: build on `http.server.ThreadingHTTPServer`, never the
  single-threaded `HTTPServer`; read the request body defensively; and set
  timeouts on outbound calls. A single-threaded server serialises the browser's
  parallel requests and locks up the instant one request stalls -- the app then
  looks dead even though the process is up.
- Size those timeouts for what is being called. A few seconds suits a small REST
  API but is far too short for an LLM, which often thinks for tens of seconds
  before its first byte: give model calls 60s+. Cutting the model off silently
  hands every request to the fallback path, so the app looks healthy while
  serving canned output.
- Confirm every runtime dependency is available in the target runtime (Python
  3.12 via `uv`, Node/npm). If you must add a dependency, verify it installs and
  imports before committing the team to it.
- The app must start and pass a network-free health check WITHOUT any API key;
  gate any LLM feature behind a graceful fallback.

## Decide the viral "wow" first
- Name the single shareable moment in one sentence before choosing tech. The
  architecture exists to make that moment effortless (e.g. instant shareable
  result URLs => keep all state in the URL/localStorage).

## Leave clean seams
- Scaffold entry point(s), a clear module boundary per responsibility, and a
  stub `viralbench.json` shape so Implementer / Designer / QA each know where
  their work plugs in.
- Write decisions down where the team collaborates (shared doc or design file)
  with the "why", not just the "what".

## When to use me
Use at the start and whenever a technical direction or dependency decision could
block a teammate.""",
    ),
    "core-loop-implementation": (
        "Turn the design into a working core loop in small, verifiable increments "
        "that always keep the app runnable.",
        """\
## What I do
Help the Implementer build the core features fast and correctly, meeting the
success criteria without breaking the build.

## Working style
- Implement the **smallest end-to-end slice** that a user can actually do, then
  grow it. Keep `main`/the entry point runnable after every increment.
- Use code-intelligence (LSP) and grep to understand existing code before adding
  to it; extend teammates' work, don't rewrite it.
- Match the Architect's chosen stack and seams. If a decision is missing or
  wrong, flag it to the team rather than inventing a divergent one.

## Correctness
- Handle the obvious edge cases for the core loop (empty input, error paths,
  restart). Prefer pure functions you can reason about.
- Never hard-code secrets; read keys from environment variables and degrade
  gracefully when they are absent.
- Leave the code in a state QA can run with a single command.

## When to use me
Use whenever you are writing or extending the app's core functionality.""",
    ),
    "virality-playbook": (
        "Add a concrete, proven shareable hook so the app spreads organically -- "
        "not just correctness, but a reason to share.",
        """\
## What I do
Help the Designer install a **specific, concrete** viral mechanic, because the
app is judged on virality, not only on working.

## Shareable hooks that work
- **Result cards**: a great-looking summary of the user's result with a
  one-click "Copy" / "Share" and a subtle app watermark.
- **Shareable links**: encode the interesting state in the URL so a shared link
  reproduces exactly what the sender saw (great for SPAs).
- **Numbers people brag about**: scores, streaks, "faster than X%", personal
  bests -- something worth screenshotting.
- **Before/after or reveal moments**: a satisfying transition that begs to be
  recorded.
- **Challenge a friend**: a link that drops the recipient straight into the same
  challenge.

## Rules
- Pick ONE hook and make it excellent and effortless (<= 1 click). Many weak
  hooks < one strong one.
- The hook must work offline / without credentials we do not have (no real
  social-platform posting -- copyable link/text instead).
- Make the shared artifact look good on its own (open graph title/description for
  web).

## When to use me
Use when adding or sharpening the reason a user would spread the app.""",
    ),
    "ux-polish": (
        "Make the app feel delightful and trustworthy: responsive layout, "
        "feedback, empty/error states, and micro-interactions.",
        """\
## What I do
Help the Designer raise the app from "works" to "feels great", which is what
makes people adopt and share it.

## Polish checklist
- **First impression**: a clear, attractive landing state; the user knows what
  to do within seconds.
- **Feedback**: every action gives immediate visual feedback (hover, active,
  loading, success, error).
- **States**: design the empty state, the loading state, and the error state --
  not just the happy path.
- **Motion**: small, fast transitions (150-250ms) that guide attention; never
  janky or blocking.
- **Responsive**: usable on a phone and a laptop; test narrow widths.
- **Accessibility**: real contrast, focus rings, keyboard support, alt text.

## Rules
- Keep it cohesive: one type scale, one spacing scale, a small color palette.
- Delight without breaking: verify the core loop still works after each change.

## When to use me
Use when improving the look, feel, and perceived quality of the app.""",
    ),
    "release-checklist": (
        "Verify the app against every success criterion and the manifest contract, "
        "and only ship when it genuinely runs from a clean checkout.",
        """\
## What I do
Help the QA & Finisher prove the app is real: it meets the criteria and launches
exactly as the manifest claims.

## Verify the contract
- `viralbench.json` exists at the app root and is valid JSON with the right
  `app_type`.
- `run.command` actually starts the app from a clean checkout in the runtime.
  Run it and confirm `run.port` and `run.url` actually work.
- `test.smoke`, if present, is fast, network-free, and exits 0 when healthy. Run
  it and confirm.

## Verify the product
- Load the `live-app-testing` skill and ACTUALLY run and drive the app end to end
  (in a real browser, driving the UI). Reading
  the code or a passing smoke test is not enough.
- Walk every success criterion explicitly and confirm each one against the
  running app; when a browser is available, click through it and watch the DOM
  and console.
- Try to break it: empty input, rapid actions, reload mid-flow, missing API key
  (it must still start and pass the health check).
- Remove dead code, debug logging, and scratch files; confirm no secrets are
  committed.

## Ship signal
- Only when everything above passes AND you have exercised the running app this
  turn, end your turn with the ship signal your instructions specify. The signal
  is only honoured if you actually ran the app (a `browser_*` interaction for a
  app in a real browser) -- reading code is not
  enough. If anything fails, fix it or hand a precise list to the team instead of
  shipping.

## When to use me
Use on every finishing/verification turn.""",
    ),
    "live-app-testing": (
        "Run the built app in its real runtime and exercise it end-to-end like a "
        "user (in a real browser, driving the actual UI) "
        "to reproduce and fix bugs before shipping.",
        """\
## What I do
Help the QA & Finisher **actually run and use the app**, not just read it. Most
shipped bugs (a merge that never renders, a form that silently drops its input,
data that vanishes on restart) are only visible at runtime -- so reading the code or a
passing file-existence smoke test is NOT verification.

## First: start the app from a clean state
- Read `viralbench.json` for `app_type`, `run.command`, `run.cwd`, `run.port`,
  `run.url`, and `test.smoke`.
- Start it exactly as the manifest says, from a clean checkout, with NO API key
  set (it must still start and be usable via its graceful fallback).
- Wait until it is genuinely ready (port open / process healthy) before testing;
  stop anything you started at the end of your turn.

## Then exercise it like a real user
- Open `run.url` in the browser tool (`browser_*`). A `curl` that returns 200 is
  NOT a test -- it does not execute the page's JavaScript.
- Drive the CORE loop the way a user would: click, type, use the keyboard,
  drag/swipe. Do the main thing the app is for, several times in a row.
- After each interaction check BOTH the rendered DOM (did the expected thing
  actually appear / update / animate?) and the browser console for errors.
- Screenshot anything that looks wrong so the bug is concrete. Test unhappy
  paths: reload mid-flow, rapid repeated actions, empty/edge input, small viewport.

### Extra checks for a full-stack-app
- Sign up, sign out, and sign back in. Confirm the account and its data are still
  there -- if they are not, you are probably writing state somewhere other than
  `$VIRALBENCH_DATA_DIR` (`/data`), which is discarded on restart.
- RESTART the app and check the data is STILL there. This is the single most
  common way a full-stack build fails review.
- Use it as TWO different users (a second browser context, or sign out and make
  another account). Confirm what should be shared IS visible to the second user,
  and that private data is NOT. Reviewers arrive several at a time against one
  running instance, so they will find this.
- Check the app still starts from an EMPTY database: delete `/data`'s contents,
  run the setup step, and start again.

## Reproduce -> fix -> re-verify
- When you find a bug, reproduce it with the smallest sequence of steps, then FIX
  it (you can edit the code), then run the SAME steps again and confirm it is
  gone. Re-run the smoke test after fixes.

## Evidence before you ship
- Only signal ready when you have actually exercised the RUNNING app this turn and
  watched the core loop work with your own tools. State concretely what you did --
  the interactions and the DOM/console/output you observed. If you did not run it,
  it is not verified.

## When to use me
Use on every QA/verification turn, before deciding whether to ship.""",
    ),
}


def all_skill_names() -> list[str]:
    """Return every shipped skill name."""
    return list(_SKILL_BODIES)


def skill_markdown(name: str) -> str:
    """Return the full ``SKILL.md`` text (frontmatter + body) for ``name``.

    Raises:
        KeyError: If ``name`` is not a known skill.
    """
    if name not in _SKILL_BODIES:
        raise KeyError(f"unknown skill {name!r}. Available: {all_skill_names()}")
    description, body = _SKILL_BODIES[name]
    # Frontmatter fields recognised by opencode: name + description (required).
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"


#: Public mapping of skill name -> rendered SKILL.md content.
SKILLS: dict[str, str] = {name: skill_markdown(name) for name in _SKILL_BODIES}


def write_skills(dest_dir: str | Path, names: list[str] | None = None) -> list[Path]:
    """Write the named skills as ``<dest_dir>/<name>/SKILL.md`` files.

    Args:
        dest_dir: The ``.opencode/skills`` directory to populate.
        names: Skill names to write; defaults to all shipped skills. Names must be
            valid opencode skill names (lowercase, hyphen-separated).

    Returns:
        The list of written ``SKILL.md`` paths.
    """
    dest = Path(dest_dir)
    written: list[Path] = []
    for name in names if names is not None else all_skill_names():
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid skill name {name!r}")
        skill_dir = dest / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        path = skill_dir / "SKILL.md"
        path.write_text(skill_markdown(name), encoding="utf-8")
        written.append(path)
    return written
