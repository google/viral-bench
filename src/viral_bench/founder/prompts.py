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

"""Prompt construction for the founder agents.

Three shapes of prompt, matching the three founder configurations:

* **Solo** (the N=1 baseline) receives the original two prompts:
  :func:`design_prompt` (write ``DESIGN.md``) then :func:`build_prompt`
  (implement the app + ``README.md`` + the ``viralbench.json`` manifest). These
  are intentionally untouched so the solo path is byte-for-byte the pre-team
  pipeline.
* **Team** (the four-specialist round-table) uses :func:`team_turn_prompt` for
  every turn: a role-specialised, round-aware message telling the agent to catch
  up on its teammates' work and advance its own responsibility this round. How
  the team *collaborates* (shared local files vs Google Workspace) is supplied by
  the collaboration toolset as a ``collaboration_brief`` string, so this module
  stays agnostic to the medium.
* **Dynamic** (:func:`dynamic_founder_prompt`) hands the model the idea, the
  deliverable contract, and an honest description of opencode's delegation
  machinery -- then gets out of the way. It prescribes no roles, no rounds and no
  division of labour, because in that mode the *orchestration* is the thing under
  test, not just the code. :func:`dynamic_continue_prompt` is the deliberately
  content-free nudge used when a turn ends with the deliverables unfinished.

All three share the same run/test contract. The user chose to let founders pick
their own tech stack, so we do not *forbid* stacks; we *inform* the agent of the
runtime the app runs and is tested in (:data:`RUNTIME_NOTES`) and require a
manifest so whatever it builds launches uniformly.

**Do not edit the shared blocks below** (:data:`RUNTIME_NOTES`,
:data:`_SCOPE_GUIDANCE`, :func:`_deliverables_block`, :func:`design_prompt`,
:func:`build_prompt`) without meaning to: :func:`brief_fingerprint` hashes them,
and every build already on disk is keyed by that hash. Adding a *new* prompt
function, as the dynamic mode does, leaves the fingerprint untouched by
construction -- which is why the dynamic brief is assembled from these blocks
rather than by amending them.
"""

from __future__ import annotations

import hashlib

from viral_bench.founder.manifest import MANIFEST_FILENAME, example_manifest_json
from viral_bench.founder.roles import Role
from viral_bench.ideas import Idea

#: Control token the QA & Finisher emits (only) when the app is ready to ship.
#: The round-table structure watches QA's transcript for this to stop early.
SHIP_SIGNAL = "FOUNDER_SHIP_IT"

#: Control token the dynamic orchestrator emits when it considers the app
#: finished. Deliberately a DIFFERENT token from :data:`SHIP_SIGNAL`: the two
#: modes have different contracts (QA's signal is gated on runtime evidence,
#: this one is the model's own declaration), and sharing a token would make a
#: transcript from one mode readable as the other.
DYNAMIC_DONE_SIGNAL = "FOUNDER_BUILD_COMPLETE"

# What the app's run/test runtime provides. The agent designs/writes code on the
# host, but the *app* is later run and tested (by a human and by the crowd) in a
# Linux container carrying this toolchain -- so target what is listed here. Kept
# factual so the model can make runnable choices even though it may pick any
# stack.
RUNTIME_NOTES = """\
App run/test runtime (your app is launched and tested in a Linux container with
this toolchain, so target what is available):
- OS: Debian-based Linux. A POSIX shell, git, curl, wget, and ca-certificates.
- Python 3.12 is available via `uv` (preferred: `uv run python`, `uv add`).
- Node.js and npm are available for JS/TS apps.
- The app runs INSIDE a container; do NOT rely on Docker/Podman being available
  to the app itself (no nested containers). There is no separate database server
  available for the same reason -- use SQLite if you need a database.
- Resources are capped at 1 GB RAM and 1.0 CPU, with no GPU. Anything that needs
  model weights on the machine (local diffusion, a local LLM, local speech-to-
  text) cannot run here; call a hosted model instead.
- WRITABLE STATE: the directory named by `VIRALBENCH_DATA_DIR` (`/data`) is the
  ONLY place that survives a restart. The app directory does not: it is a
  throwaway clone, re-made whenever the app is restarted. Put your database,
  uploads, and anything else you must not lose in `/data`.
- Network access IS available at run time, but prefer minimal external runtime
  dependencies so the app is reliable and reproducible across runs.
- An LLM is available at run time, but calling it is STRICTLY OPTIONAL and is the
  exception rather than the default. Add a generative-AI feature ONLY when THIS
  idea's core features or success criteria actually require one -- that is, when
  the app cannot do its stated job without a model. Most ideas here need no model
  at all. Do NOT bolt on an "AI assistant", a "smart" helper, or a one-click
  generator to seem impressive: on an idea that did not ask for it that is scope
  creep, it dilutes the core product, and reviewers read it as unfocused rather
  than delightful. A tool that does its one job well beats the same tool with a
  chatbot stapled to the side. If you are unsure, leave it out and spend the
  effort on the core loop instead.
- IF -- and only if -- the idea genuinely requires a model, call an
  OpenAI-compatible chat-completions endpoint. The runtime provides three
  environment variables and you must use all three exactly as given, whichever
  model is writing this code:
    VIRALBENCH_APP_LLM_BASE_URL   the endpoint (already ends in /v1 or equivalent)
    VIRALBENCH_APP_LLM_API_KEY    the key
    VIRALBENCH_APP_LLM_MODEL      the model id to send
  Never hard-code a key, a model id or a vendor's URL. The key is a SERVER-SIDE
  secret: if your app is a single-page/browser app, add a tiny same-origin backend
  endpoint that reads these from the environment and proxies the call. NEVER
  expose the key to browser JavaScript or read it from window/localStorage -- a
  static page cannot see the runtime env, so a purely client-side key wiring will
  silently never work. The app MUST still start (and pass a network-free health
  check) WITHOUT a key, so degrade that feature gracefully when
  VIRALBENCH_APP_LLM_API_KEY is absent.
- Do NOT require any OTHER paid service or account to run the app.
- Keep the app easy to launch with a single command described in the manifest.
"""

# Rules that hold for EVERY app, since every app in the bench is now a web app.
_WEB_COMMON = """\
- Your app is opened in a real browser by reviewers who read the RENDERED DOM and
  the accessibility tree, and who can also see screenshots. Label your controls.
  A control with no accessible name, and state drawn only into a <canvas>, are
  invisible to them -- and what a reviewer cannot perceive, they score as absent.
- BIND TO 0.0.0.0, NEVER localhost/127.0.0.1. The app runs in a container and its
  port is published to the host; a server bound to loopback inside the container
  is unreachable from outside and the app looks dead. Framework dev servers get
  this wrong by default -- pass `--host 0.0.0.0` (Vite, Next) explicitly.
- Serve requests CONCURRENTLY. A browser opens several connections at once and
  several reviewers use the app at the same time. If you write your own Python
  server, build it on `http.server.ThreadingHTTPServer`, NEVER the single-threaded
  `HTTPServer`: one slow or half-open connection freezes the accept loop and the
  whole app looks dead. Read the request body defensively (tolerate a missing or
  short `Content-Length` rather than blocking forever in `rfile.read`), and set a
  timeout on every outbound network call so a dead socket cannot wedge a worker.
- SIZE THAT TIMEOUT FOR THE CALL. A few seconds is right for a small REST API; it
  is far too short for an LLM, which routinely thinks for tens of seconds before
  the first byte. Give Gemini calls a generous deadline (60s+) and stream or show
  progress rather than cutting the model off. A too-short deadline is the worst
  kind of bug here: combined with the graceful fallback above, the app keeps
  answering and looks completely healthy while quietly serving canned output
  instead of anything the model produced.
- The manifest's `run` MUST include a `port` and a `url` (e.g.
  http://localhost:8000/) so a tester can open it in a browser.
- Use any stack you like (Node and npm are available, as are Python and `uv`),
  but the manifest must describe the exact commands and the app must start within
  90 seconds. A no-build-step app starts faster and has fewer ways to fail."""

# Per-scope guidance nudging the agent toward something we can actually test.
_SCOPE_GUIDANCE: dict[str, str] = {
    "client-app": f"""\
This is a CLIENT-APP: a web app that runs in the browser.
{_WEB_COMMON}
- No backend is required. Keep state in the browser (localStorage/IndexedDB) and
  serve the files with something simple.
- Most client-apps need NO server-side route at all. The one reason to add one is
  proxying the Gemini API so the key is never exposed to browser JavaScript, and
  that only applies to an idea that genuinely requires a model (see the runtime
  notes). If you do add one, keep it to that, and respect the concurrency rules
  above.""",
    "full-stack-app": f"""\
This is a FULL-STACK-APP: a web app with a real backend and a database.
{_WEB_COMMON}
- PERSIST STATE TO DISK, in the directory given by the `VIRALBENCH_DATA_DIR`
  environment variable (it is `/data` in this runtime). SQLite there is the
  expected choice and needs no extra service. Anything you write elsewhere --
  including inside the app directory -- is DISCARDED when the app restarts, and
  reviewers will find their accounts and data gone mid-session.
- Create the schema in a `setup` step (e.g. a migration script). `setup` runs
  before the app starts and shares the same `/data`, so the server finds the
  tables already there. Make it idempotent: it may run more than once.
- Support MULTIPLE USERS properly: real sign-up and sign-in, and correct
  isolation between accounts. Several different reviewers use the SAME running
  instance at the same time, so what one of them creates must be visible to the
  others exactly as your product intends -- and private data must NOT be.
- Hash passwords, and keep sessions in a cookie. Never store a password in plain
  text; reviewers read your source.
- Use SQLite in WAL mode (`PRAGMA journal_mode=WAL`) so concurrent readers do not
  block on a writer.""",
}


def render_idea(idea: Idea) -> str:
    """Render an :class:`Idea` spec as a readable brief for the agent."""
    features = "\n".join(f"  - {f}" for f in idea.core_features)
    return f"""\
Title: {idea.title}
One-line pitch: {idea.pitch}
Problem: {idea.problem}
Target user: {idea.target_user}
Core features:
{features}
Success criteria: {idea.success_criteria}
Scope (app type): {idea.allowed_scope}
Difficulty: {idea.difficulty}"""


# --------------------------------------------------------------------------- #
# Solo founder prompts (the N=1 baseline). Do not change -- these define the
# reproducible single-agent path.
# --------------------------------------------------------------------------- #


def design_prompt(idea: Idea) -> str:
    """Prompt for the Design phase: write ``DESIGN.md`` into the app root."""
    return f"""\
You are a startup founder-engineer. You have been given a product idea to design
and (next) build as a real, shippable app. Right now, focus ONLY on design.

=== PRODUCT IDEA ===
{render_idea(idea)}

=== YOUR TASK (design phase) ===
Write a concise design document to a file named `DESIGN.md` in the current
directory. It should cover:
1. The core user experience and the single "wow" that makes people want to
   share it (this is judged on virality, not just correctness).
2. The concrete feature list you will implement to meet the success criteria.
3. The tech stack and architecture you will use, and WHY it will run cleanly in
   the run/test runtime described below.
4. How the app will be tested/tried (what a user or agent will do to experience
   it).

Keep it focused and buildable in one session. Do not write application code yet
-- only `DESIGN.md`.

{RUNTIME_NOTES}"""


def _deliverables_block(idea: Idea) -> str:
    """The required deliverables + manifest contract shared by every build that
    must leave a *complete*, shippable app (the solo founder and the team's QA &
    Finisher)."""
    example = example_manifest_json(idea.allowed_scope)
    return f"""\
=== REQUIRED DELIVERABLES ===
1. Working application code that fulfills the success criteria above.
2. A `README.md` with a short description and exact run/test instructions.
3. A run/test manifest named `{MANIFEST_FILENAME}` at the app root. It tells the
   benchmark how to launch and test your app. It MUST be valid JSON matching this
   shape (here is a concrete example for this app type -- adapt the values to
   YOUR app, do not copy blindly):

```json
{example}
```

Manifest rules:
- `app_type` MUST be "{idea.allowed_scope}".
- `run.command` MUST actually start the app from the app root in this runtime.
- For a single-page-app, include `run.port` and `run.url`.
- `test.smoke` MUST be a fast command that exits 0 ONLY when the app is genuinely
  healthy. Your app is STARTED BEFORE smoke runs, and the command executes where
  the app is reachable, so probe the app itself -- e.g.
  `curl -fsS http://localhost:<port>/healthz`. Prefer that to a file-existence
  check like `test -f index.html`, which passes even when the app is broken.
  It must not depend on the PUBLIC internet or a third-party service; talking to
  your OWN server is expected and is the point.
- `test.manual` MUST demonstrate the app's REAL headline feature (the live,
  AI/network-backed path when it has one), because that is what testers will
  judge. If you also ship an offline/mock/demo mode, list it only as a clearly
  labelled fallback AFTER the real steps -- never as the primary way to try the
  app, or the feature that matters will never actually get exercised.

=== QUALITY BAR ===
- The app MUST run in the runtime with the exact commands in the manifest.
- No secrets in code; read any keys from environment variables.
- Prefer a delightful, polished result -- the app will be shown to (simulated)
  users who decide whether to adopt and share it.

When you are done, double-check that `{MANIFEST_FILENAME}` exists and is valid
JSON, and that `run.command` works from a clean checkout."""


def build_prompt(idea: Idea) -> str:
    """Prompt for the Build phase: implement the app + README + manifest."""
    scope_guidance = _SCOPE_GUIDANCE.get(idea.allowed_scope, "")
    return f"""\
Now BUILD the app you designed in `DESIGN.md`, in the current directory.

=== PRODUCT IDEA (recap) ===
{render_idea(idea)}

{scope_guidance}

{_deliverables_block(idea)}

{RUNTIME_NOTES}"""


# --------------------------------------------------------------------------- #
# Team round-table prompts. Every specialist turn (across every round) uses
# team_turn_prompt. The collaboration medium is injected via collaboration_brief
# (from the toolset), so this module never hard-codes files vs Workspace.
# --------------------------------------------------------------------------- #


def _focus_bullets(role: Role) -> str:
    return "\n".join(f"- {r.focus}" for r in role.responsibilities)


def _roster(teammates: list[tuple[str, str]]) -> str:
    if not teammates:
        return "  (you are working solo this run)"
    return "\n".join(f"  - {title}: owns {resp}" for title, resp in teammates)


def team_turn_prompt(
    idea: Idea,
    role: Role,
    *,
    round_index: int,
    max_rounds: int,
    min_rounds: int = 1,
    agent_index: int,
    n_agents: int,
    teammates: list[tuple[str, str]],
    is_first_turn: bool,
    collaboration_brief: str,
) -> str:
    """Build one specialist's turn prompt for round ``round_index``.

    Args:
        idea: The product idea.
        role: This agent's role (persona + responsibilities).
        round_index: 1-based current round.
        max_rounds: Hard cap on rounds (for the agent's situational awareness).
        min_rounds: Minimum rounds the team must run before QA may ship. In rounds
            below this floor, QA is told NOT to ship and to keep improving.
        agent_index: 1-based position of this agent on the team.
        n_agents: Team size.
        teammates: ``(title, responsibility_label)`` for the other roles.
        is_first_turn: Whether this is this agent's first turn of the whole build
            (it starts a fresh session; otherwise it resumes its own context).
        collaboration_brief: Medium-specific collaboration instructions from the
            toolset (shared local files, or Google Workspace).
    """
    scope_guidance = _SCOPE_GUIDANCE.get(idea.allowed_scope, "")

    if is_first_turn:
        continuity = (
            "This is your FIRST turn. Introduce your plan for your responsibility "
            "to the team through the collaboration channel below, then start work."
        )
    else:
        continuity = (
            "You are resuming your OWN context from earlier rounds -- you remember "
            "your prior work and decisions. Do NOT restart from scratch."
        )

    if role.owns_qa:
        if round_index >= min_rounds:
            ship_block = f"""\
=== SHIP DECISION ===
When -- and ONLY when -- the app genuinely meets EVERY success criterion and runs
from a clean checkout with the exact manifest command, end your message with a
single final line containing exactly:

{SHIP_SIGNAL}

This token is only honoured if you ACTUALLY exercised the running app this turn --
open it in the browser and click/type through the core loop for a web app, or run
the real command for a CLI/bot. Reading the code or a passing file-existence smoke
test does not count, and the ship signal is ignored without that evidence.

If it is not ready, do NOT write that token. Instead fix what you can now and use
the collaboration channel to give your teammates a precise, prioritised list of
what remains for the next round."""
        else:
            ship_block = f"""\
=== DO NOT SHIP YET ===
This founding run must iterate through at least round {min_rounds} before it can
ship, and this is only round {round_index}. Do NOT write any ship token yet --
even if the app looks done, there is always more polish, virality, and hardening
to add. Instead: do your full QA pass, FIX what you can right now, and hand the
team a precise, prioritised punch-list (via the collaboration channel) of what to
raise the bar on next round -- more delight, a stronger viral hook, edge cases,
performance. Push the team to make it genuinely better, not just working."""
        task = f"""\
=== YOUR TASK THIS ROUND (round {round_index} of at most {max_rounds}) ===
{continuity}
First, catch up: run the app, review what your teammates changed since your last
turn, and check the collaboration channel. Then do YOUR job:
{_focus_bullets(role)}

You own the finish line. Every time you act, leave the app in a runnable state.

{_deliverables_block(idea)}

{ship_block}"""
    else:
        task = f"""\
=== YOUR TASK THIS ROUND (round {round_index} of at most {max_rounds}) ===
{continuity}
First, catch up: review what your teammates changed since your last turn (their
code, and the collaboration channel below). Then advance YOUR responsibility:
{_focus_bullets(role)}

Grow the app forward and keep it runnable for whoever acts next. You are not the
finisher -- coordinate through the collaboration channel rather than trying to
complete everyone else's work."""

    return f"""\
You are {role.mission}.

You are the {role.title} (agent {agent_index} of {n_agents}) on a founding team
building ONE app together over multiple rounds. Your teammates each keep their own
memory and work in parallel; you share the app's working directory. Your
teammates:
{_roster(teammates)}

{task}

=== PRODUCT IDEA ===
{render_idea(idea)}

{scope_guidance}

=== HOW YOUR TEAM COLLABORATES ===
{collaboration_brief}

{RUNTIME_NOTES}"""


# --------------------------------------------------------------------------- #
# Dynamic orchestrator prompts. ONE agent gets the idea, the deliverable
# contract, and a factual description of what opencode's delegation machinery can
# do -- then decides for itself whether to use any of it. Nothing here names a
# role, a phase, a round or an order of work. that omission is the whole point of
# the mode, so resist the urge to "helpfully" suggest a team shape here.
# --------------------------------------------------------------------------- #


def _delegation_brief(agents_dir: str) -> str:
    """Describe the subagent machinery available to the dynamic orchestrator.

    Every claim here was verified live against opencode 1.17.14 before being
    written down, because a brief that promises a capability the harness does not
    actually have is worse than no brief: the model spends its turns fighting the
    tool instead of building the app, and we would read that as the model being
    bad at orchestration. In particular the "next turn" rule for self-defined
    agents is real -- opencode loads agent definitions when a turn starts, and an
    agent written mid-turn fails that same turn with "Unknown agent type".
    """
    return f"""\
- `task` spawns a subagent. It runs autonomously with its own tools and its own
  context, and returns ONE final message to you. Its context starts empty, so its
  prompt must carry everything it needs and say exactly what to report back.
- Issue SEVERAL `task` calls in a single message and those subagents run
  CONCURRENTLY, which is how you get real parallelism rather than a relay.
- A finished task returns a `task_id`. Pass that `task_id` back to `task` to
  continue that SAME subagent -- it keeps everything it did before -- instead of
  starting a fresh one. That is how a subagent becomes a persistent teammate
  rather than a one-shot.
- Built-in subagent types: `general` (full tools; reads and writes code),
  `explore` (fast, read-only codebase search), `scout` (read-only research into
  external docs and dependencies).
- You can DEFINE YOUR OWN subagent types. Write `{agents_dir}/<name>.md`:

      ---
      description: one line on what this agent is for and when to use it
      mode: subagent
      temperature: 0.2
      ---

      The system prompt that shapes this agent.

  It then becomes a valid `subagent_type`. Note the mechanics: opencode reads
  agent definitions when a turn STARTS, so an agent you write is usable on your
  NEXT turn, not the one you wrote it in -- and you get a next turn by ending
  this one WITHOUT the completion signal (see FINISHING). Designing a team and
  then taking another turn to actually run it is a legitimate way to build this
  product, not a failure to finish. That directory sits outside the app and
  never ships with it.
- You and every subagent share ONE working directory -- the app itself. There is
  no other channel between you: whatever a subagent needs to know must be in the
  prompt you give it or in a file it can read, and whatever it did comes back to
  you only in its final message or in the files it changed."""


def dynamic_founder_prompt(
    idea: Idea, *, app_dir: str, agents_dir: str, max_turns: int
) -> str:
    """Prompt for the dynamic orchestrator's first turn.

    Section order is deliberate and was set by observation, not taste. In the
    first version the delegation section sat in the middle, with the deliverables
    and the (long, concrete, app-focused) runtime notes after it -- and two live
    builds ran to completion without the founder mentioning subagents once, on a
    model that a direct probe confirmed HAD the ``task`` tool. The last thing a
    model reads is what it acts on, so the question of how to run the build now
    comes last, immediately before it starts.

    For the same reason the founder is asked to state its plan for running the
    build in its first message. That is the one piece of process here, and it is
    the measurement instrument rather than a workflow: it forces the
    orchestration decision to be MADE and recorded. "I will do this alone"
    remains a complete and legitimate answer -- what is not acceptable is the
    decision never being taken.

    The two directories are spelled out for a measured reason. This mode is the
    only one that names a path OUTSIDE the app dir (where self-defined subagents
    go), and in a live build that was enough to re-anchor a model's idea of the
    project root: it wrote `viralbench.json` one level up, in the workspace root,
    on all three of its turns, and the build was recorded `manifest_missing`. A
    mode-specific prompt detail turning into an apparent inability to ship a
    manifest is precisely the failure this bench must not record: a harness
    fault scored as a model fault. So both paths are stated explicitly.

    Args:
        idea: The product idea.
        app_dir: Absolute path of the app directory -- the tree that ships.
        agents_dir: Absolute path of the directory where self-defined subagents
            go (outside the app, so they never ship).
        max_turns: Hard cap on orchestrator turns, told to the model for
            situational awareness -- the same courtesy the team gets about rounds.
    """
    scope_guidance = _SCOPE_GUIDANCE.get(idea.allowed_scope, "")
    return f"""\
You are the founder, and the lead agent on this build. You have one job: turn this
idea into a real, working app that people would actually want to use and pass on
to a friend. It is judged on whether it works and on whether it is good enough to
spread.

=== WHERE YOU ARE WORKING ===
`{app_dir}` is the app. Everything that ships lives in there, `{MANIFEST_FILENAME}`
goes at its root, and it is your working directory. Nothing outside it ships.

=== PRODUCT IDEA ===
{render_idea(idea)}

{scope_guidance}

=== WHAT YOU MUST LEAVE BEHIND ===
{_deliverables_block(idea)}

{RUNTIME_NOTES}

=== HOW YOU RUN THIS BUILD ===
This is the part that is yours. You are not a lone coder here: you are the lead
agent, and you have a team available -- one you bring into existence yourself.
Nobody has picked it for you. There are no assigned roles, no phases, no rounds
and no required order of work. Who works on this, how the work is split, and in
what order are your calls, and how well you make them is measured here alongside
the app you ship.

{_delegation_brief(agents_dir)}

Design whatever team this product actually needs, and drive it: one generalist, a
set of specialists you write for the job, a cast you add to as you learn what the
work really is. Nothing about that shape is fixed and you are not committed to
your first plan -- change it as the build teaches you something.

Start your first message by stating, in a line or two, how you intend to run this
build and why. Then run it that way.

=== FINISHING ===
You have at most {max_turns} turns. A turn ends when you stop working and reply,
so keep going within a turn for as long as there is useful work to do. The cap is
on turns, not on work: there is no limit on how many subagents you spawn or resume
inside one turn, nor on how many rounds you go with them.

When the app is genuinely finished -- it runs from a clean checkout with the exact
command in the manifest, and it meets every success criterion -- end your final
message with a line containing exactly:

{DYNAMIC_DONE_SIGNAL}

Do not write that line before it is true. Ending a turn WITHOUT it is how you
take another one -- a deliberate choice to keep working, and the only way to use
a subagent you defined during this turn. What gains you nothing is claiming to be
done before you are: the deliverables are checked, and you will simply be told to
continue."""


def dynamic_continue_prompt(*, turn_index: int, max_turns: int, gaps: list[str]) -> str:
    """The nudge sent when a dynamic turn ended with the build unfinished.

    Deliberately content-free about *how* to proceed: it states which deliverable
    is missing (a fact about the contract, identical in every founder mode) and
    nothing else. Any advice here -- "try delegating", "test it in a browser" --
    would be the harness doing the orchestration the mode exists to measure.

    Args:
        turn_index: 1-based index of the turn about to run.
        max_turns: Hard cap on orchestrator turns.
        gaps: Short factual statements of what is not done yet.
    """
    gap_lines = "\n".join(f"- {gap}" for gap in gaps) or "- The build is not finished."
    last = (
        "\nThis is your LAST turn: whatever exists when it ends is what ships.\n"
        if turn_index >= max_turns
        else ""
    )
    return f"""\
Continue -- this is turn {turn_index} of at most {max_turns}.

Where things stand:
{gap_lines}
{last}
Carry on however you think best. When the app genuinely runs from a clean checkout
with the exact command in the manifest and meets every success criterion, end your
message with a line containing exactly:

{DYNAMIC_DONE_SIGNAL}"""


def brief_fingerprint(idea: Idea) -> str:
    """A short hash of everything that defines this idea's build brief.

    Two builds are only comparable if the model was asked for the same thing. The
    fleet index used to assume that, checking only the founder *shape* (agents,
    collab, rounds) -- so when the corpus and the prompts changed in the web-dev
    pivot, 12 builds from six days earlier still counted as current results. They
    had been told to "strongly prefer plain static files" and "no backend
    required", instructions the bench now gives the opposite of. Comparing them
    against freshly-built arms would have measured the prompt rewrite and
    attributed it to the founder structure.

    The fingerprint covers the DESIGN and BUILD prompts in full, which between
    them embed the idea spec, the scope guidance, the runtime notes and the
    example manifest -- i.e. every input the model sees. Any change to any of
    them retires the affected builds instead of silently reusing them.

    Every founder mode is fingerprinted by these two prompts even though only the
    solo mode is *given* them, because the team and dynamic briefs are assembled
    from the same blocks: change a block and all three modes move together, which
    is exactly the coupling the check wants. The corollary is that adding a mode
    must not touch those blocks -- 400+ builds on disk are keyed by this hash, so
    an "improvement" to the shared wording silently retires the whole corpus.
    """
    payload = "\x00".join((design_prompt(idea), build_prompt(idea)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
