<!--
 Copyright 2026 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# ViralBench trajectory viewers

Three browser UIs over the benchmark. Two replay a half of it over time: the founder
pipeline building an app, and the crowd pipeline deciding whether anyone would use
it. Both are clickable, scrubbable, and read straight from `builds/`. The third shows
the rubric grade for that same app, the second scoring track, item by item, with the
evidence behind every verdict and the ViralScore beside it.

```bash
.venv/bin/python viz/serve.py
```

```
http://127.0.0.1:8770/founder     http://127.0.0.1:8770/crowd
http://127.0.0.1:8770/rubric
```

That is the whole install. No dependencies, no build step, no `npm`, only
standard-library Python and vanilla JavaScript.

---

## Launching

| what you want | command |
|---|---|
| the default instance | `.venv/bin/python viz/serve.py` |
| ...opening a view in a browser | `.venv/bin/python viz/serve.py --open founder` |
| a different port | `.venv/bin/python viz/serve.py --port 9000` |
| point at another builds tree | `.venv/bin/python viz/serve.py --builds-root /path/to/builds` |
| log every request while debugging | `.venv/bin/python viz/serve.py -v` |

By default it reads this repo's `builds/`, **read-only**, and every build shows up
the moment it lands. Override the location with `--builds-root` or
`VIRAL_BENCH_BUILDS_DIR`.

Nothing is ever written to the builds tree. Caches, rescued screenshots and
throwaway app copies all go to `viz/cache/` (gitignored), and
`core.paths.assert_writable` refuses any write outside it.

---

## 1. Founder viewer: `/founder`

Replays one build: every turn, tool call, file edit and browser action on a
timeline you can scrub.

### Loading a build

Three ways, all equivalent:

- **Paste any build id** into the box at the top and press Enter.
- **`Browse all builds…`** (or press `/`) opens a searchable table over every build,
  with filters for mode, status, and "has crowd runs". Click a row to open it.
- **Deep link**: `http://localhost:8770/founder?build=<build_id>`.

### What the screen shows

**Header.** Idea, model, a mode badge, and the run's headline numbers: turns, tool
calls, events, cost, tokens, wall clock. Chips carry status, rounds, `shipped
early`, `QA verified`, tool-error count, and the reasoning-token summary.

**Swimlanes.** One row per actor, with each turn drawn on the real time axis and a
dot per tool call, so you can see at a glance who did the work and where a lane sat
idle. **Click a segment to jump the clock there**, or click anywhere on a track to
scrub. The lane label shows the opencode session id.

**Transport.** `▶ Play` replays the build at 1 minute of build time per second
(1×–16×). `◀ step` / `step ▶` move event by event. Spacebar toggles play, arrow
keys step, and the whole screen is a projection of the clock: events appear in the
feed as their moment arrives.

**Event feed** (left) is the run as it happened. Filter by lane, by tool family
(browser / shell / edit / read / task / skill / …), by errors only, or turn on
`model steps` to see per-step cost and tokens. Click any row to inspect it.

**Detail tabs** (right):

| tab | what's in it |
|---|---|
| **Event** | The selected call in full: arguments, output, duration, status. `edit`/`write` render a coloured diff. Browser calls split into the Playwright code that ran, the page title/URL, console errors, and a collapsible accessibility snapshot. Screenshots render inline. Anything truncated has a **Load the full record from disk** button. |
| **Overview** | The whole `build.json`, plus the app manifest and the manual test steps the build declared. |
| **Thinking** | The model's chain of thought (see below). |
| **Prompts** | Every prompt the agent was given. |
| **Sessions** | Every opencode session the build opened, with depth, cost and tokens. |
| **Turns** | One row per turn: session id, return code, duration, tool count, cost, tokens, transcript size. Click to jump the clock. |
| **Files** | Which files the agents edited, and how often. |
| **Tools** | Tool-call histogram with failure counts. |
| **Orchestration** | *Dynamic builds only.* |
| **Screenshots** | Every PNG the agents saved under `app/.playwright-mcp/`. |
| **Run the app** | Start the built app and use it (see below). |
| **Files on disk** | The exact paths of every raw JSON, plus this build's crowd runs. |

### All three run structures

The lanes mean different things per mode, so the layout changes with the structure:

- **Single agent** (`solo`). One actor, two turns (`design`, `build`) in one session.
- **4-agent team, local** (`team` + `collab local`). Four specialist lanes, rounds
  running left to right. **One session id per lane across every round** is the
  visible proof that a specialist resumes its own memory rather than being
  re-rolled, and the header says so. Three rounds of four agents draws 12 turns.
- **Dynamic orchestrator** (`dynamic`). One orchestrator lane plus a **spawn gantt**
  underneath, drawn on the subagents' real start/end times. The bars overlap, which
  is the point: that overlap is the only place the orchestrator's parallelism is
  visible. The **Orchestration** tab shows spawn counts, peak concurrency, the
  delegation prompts, and the agent definitions the founder *wrote for itself* at
  `.opencode/agents/*.md`.

Retired `specialist` relay builds still open, labelled "Legacy".

### Thinking, prompts, patches and subagents

Founder builds record the model's **chain of thought**, plus three other things that
never reached the stdout transcript. All four come from opencode's own session
store, which the harness dumps per turn to `transcript/sessions/<session_id>.jsonl`:

| what | where it shows |
|---|---|
| **Chain of thought** | Purple `thinking` rows in the feed, and the **Thinking** tab |
| **Prompts**, the text the agent was given, never echoed to stdout | **Prompts** tab, `📥` rows in the feed |
| **File patches**, never printed at all | `patch` rows, with the files each one touched |
| **Subagent sessions**, which opencode's printer drops below the root | Their own swimlanes, and the **Sessions** tab |

The **Thinking** tab shows every thought in order, clickable to jump to that moment
in the run, plus a per-turn breakdown of where the model thought hardest.

Two things it is careful about, because both are traps:

- **Reasoning is measured in characters, not tokens.** Vertex Anthropic reports
  `tokens_reasoning: 0` while returning thousands of characters of thought, so a
  token rollup makes a Claude arm read as "did not think". The token count is still
  shown, never as the headline.
- **A redacted thought is not an absent one.** Vertex returns Claude's thinking
  encrypted unless the request asks for a summary: signature present, text empty.
  Those render as `redacted` and are counted separately, because "it thought and the
  text is unreadable" and "it did not think" are different findings.

The build record carries two counts, and the viewer keeps them apart because they
answer different questions. `reasoning_chars_root` is measured off the stdout
transcript, which opencode filters to each turn's root session. `reasoning_chars_all`
is measured off the session dumps, which include the subagents. On a build that
delegated, the gap between them **is the team's share of the thinking**, and the
Thinking tab reports it as that ("the orchestrator thought X, its subagents thought
Y more, Z% of the total") rather than as an inconsistency.

A real disagreement is only possible between `_all` and what the viewer reads, since
those are the same file read by different code. That case gets a warning and both
numbers, never an average.

Builds recorded before the counts were split carry the bare `reasoning_chars`, and
the viewer falls back to it rather than reporting a confident zero.

Builds recorded before the capture existed still open, marked **partial trace**:
no thinking, no prompts, no subagents, no patches, and token/cost figures that are
unknown rather than zero.

### Where the trace comes from

The viewer reads, in precedence order:

1. an exported bundle, `builds/trajectories/<build_id>/{trajectory.json,events.jsonl}`,
   if `viral-bench trajectory` has been run,
2. `transcript/sessions/*.jsonl`, the per-turn session dump, which is the normal path
   and is written automatically on every build,
3. `transcript/<phase>.json`, the stdout NDJSON, for builds with neither.

`viz/core/trace.py` normalizes all three to the schema `viral_bench.founder.trajectory`
defines (`SCHEMA_VERSION = 1`), so no export step is needed to open a build, and the
viewer's own **`Zip ↓`** writes a byte-conformant `trajectory.json` + `events.jsonl`
pair. It deliberately does not import the benchmark: the viewer has to open a build
recorded by any revision without being pinned to the one checked out. That port is
verified against the real thing, since reading the same build both ways produces
identical events, field for field.

## 2. Crowd viewer: `/crowd`

Replays one crowd simulation: the social graph forming, the launch post spreading,
and each agent's hands-on trial of the app.

### Loading a run

- **Paste a crowd run id**, or paste a **build id** and it lists that build's runs.
- **`Browse all runs…`** (or `/`) lists every run with its engagement, would-use rate,
  delight and trial count.
- **Deep link**: `/crowd?run=<run_id>` or `/crowd?build=<build_id>`.
- From the founder viewer, the **Crowd runs** button jumps straight across.

### What the screen shows

**Header.** It leads with the founder side: which **pipeline** built the app being
scored (single agent / 4-agent local / dynamic orchestrator), which **founder
model**, how many turns it took, whether it spawned subagents, and how much it
thought. Click either of the first two chips to jump straight to that build's
founder trajectory.

The crowd's own settings follow after a divider, and the crowd model is labelled
`crowd: …` on purpose. A run's `config.model_id` is the *crowd's* model, never the
founder's, and reading one as the other is the easy mistake. A dynamic build that
declined to delegate is called out explicitly (`⑂ no subagents, it chose to work
alone`), since that is a finding rather than a missing value.

The two headline scores on the right are labelled **`would use (trial)`** and
**`delight (trial)`** rather than plain "would use" / "delight", because every
agent is scored *twice* and the two passes are different questions:

| pass | who it covers | when |
|---|---|---|
| **trial** | only the agents who drove the app in a browser | right after their hands-on session |
| **interview** | every agent, including ones who never opened it and only saw a post | at the end of the run |

They usually land close, which is what makes quoting the wrong one easy, and they do
not always. On one broken build the trial delight was **1.7** against an interview
delight of **0.8**, and on another run the trial would-use rate was **40%** against
an interview rate of **13%**. Hovering either stat shows both numbers side by side
with their agent counts, and clicking opens the Verdicts tab, where each section says
in a line who it polled.

**Social graph** (top left) is the follow network, laid out once so positions stay
stable. `@founder` is agent 0 at the centre. Nodes are coloured by tier (founder /
trier / latecomer / reactor) and **grow as the crowd engages with them**. As
playback advances, green and pink arcs fire between agents for each repost and
quote at that step. Hover to name a node, click to select the agent.

**Timeline.** The simulation clock, one pill per step:
`t0 sign-ups & follow graph → t1 launch post → t2 Round 1 → … → interview`.
Press `▶ Play` to watch the cascade, or click any step to jump. Spacebar and arrow
keys work here too.

The number on each pill is how many things the crowd **did** at that step: posts,
reposts, quotes, comments, likes, follows. Feed refreshes and explicit do-nothings
are deliberately excluded, since every agent logs one of those every round whether or
not it engaged, so counting them makes a dead round look busy. On one real run the raw
totals read 95 → 68 → 56 across three rounds, which looks like a gentle decline. The
acted counts are 64 → 11 → 2, which is what happened. Hover a pill for the full
per-action breakdown and the passive count.

**The crowd** (left) lists every agent with tier, follower count, posts so far and
delight score. Click to select.

**Middle pane:**

| tab | what's in it |
|---|---|
| **Feed** | Every post up to the current step, newest first, with repost/quote lineage and comment threads. Click a post to open it on the right. |
| **Action log** | Every action in order: who, when, what. |
| **Verdicts** | The two passes side by side. Hands-on trier scores (would-use, would-share, delight, and the craft facets) plus the delight histogram, then the exit-interview aggregates, each under a line saying who it polled. The per-agent table below merges them, trial first, and every merged cell says on hover which pass it came from, so an interview figure standing in for a missing trial is visible rather than silent. |
| **Autorating** | The LLM autorater's dimensions and reasons. Its evidence chips are clickable and jump to the agent or post cited. |
| **Run health** | Health failures, build-validity gate, turn accounting, interview coverage, crowd integrity, reach and cascade stats. |

**Right pane:**

| tab | what's in it |
|---|---|
| **@agent** | Persona, tier, influence and bio, the trial verdict with all craft facets, the exit interview in their own words, **their recorded reasoning, verbatim**, the persona prompt they were given, and every post they made. |
| **Trial replay** | The main event (see below). |
| **What they saw** | The rendered feed this agent was shown at each step. This is the only time-resolved exposure record, so it answers "was this agent ever even given the chance to see the launch". |
| **Post** | For a selected post: who was shown it, who liked it, who amplified it, each name clickable. |
| **Run the app** | Start the app the crowd tested and use it yourself. |
| **Files on disk** | Every raw artifact path for this run. |

### Trial replay: watching an agent use the app

`Trial replay` opens on the agent with the deepest trial. It lists every step the
agent took (`open`, `look`, `click`, `type`, `press`, `select`, `reload`,
`screenshot`, `finish`, plus `run_command` / `send_message` for older CLI and bot
runs) with its target.

Press **`▶ Replay the trial`** and it plays at the pace the agent worked at,
scrolling itself as it goes. For each step you get:

- the **real screenshot**, when the agent took one, and the most recent one dimmed,
  as visual context, when it did not,
- **what the app showed back**: page title, URL, visible text, the controls in
  reach, the accessibility tree, and any console errors,
- the arguments, whether it succeeded, and how long it took.

Beside it, **Run the app** starts that same build on a free port and shows it in the
pane, so you can repeat the agent's steps by hand and judge its verdict yourself.

Screenshots live in `/tmp/viralbench-shots`, and `/tmp` does not survive a reboot.
The viewer serves them from wherever they still are and copies each one it renders
into `viz/cache/shots`. **`Save shots`** in the toolbar rescues a whole run at once.

---

## 3. Rubric viewer: `/rubric`

The second scoring track. The crowd decides whether anyone would *want* the app.
The rubric decides whether it does what its brief asked for. Same build, two
independent numbers, and the interesting cases are where they disagree.

A grade is not a recording, so this page has **no transport bar**. The other two
viewers replay something that happened over time, whereas a grade is a verdict
document, and the only temporal thing in it, the grader's own tool log, lives in the
Transcript tab.

**Loading one.** Paste a grade id, or a build id to list that build's grades, or
deep-link with `?grade=<id>` / `?build=<id>`. `Browse all grades…` lists every
grade with both scores side by side, and has two filters worth knowing:
**disagrees with ViralScore** (|Δ| ≥ 20, the whole point of the track) and
**gate failures only**.

**The header** carries the score, and beside it the arithmetic that produced it:
`points → base → penalties → final`, plus the ViralScore and the difference. Chips
name the arm and model that *built* the app, the idea, the grader model, the pass
count, and the gate result. When the harness had to overrule the model on more
than 10% of items, a red chip says so. A high override rate means the grader was
not navigating reliably and the transcript should not be trusted even where code
had the last word.

When the gate failed, a band explains it. That matters because every other number
on the page is then zero, and the graded items are shown **unresolved rather than
failed**: the build was never observed, so one manifest typo is one defect, not
twenty-seven.

**The left pane** lists every item, grouped by tier, filterable by
All / Failed / Passed / Unresolved / Disagreed / Penalties. Each row shows points
earned over available, how the item was decided, and the verdict. `assert`,
`probe` and `source` are decided by **code**, while `agent` is decided by the model.
About 91% of available points never pass through a model at all, which is the
short answer to "you used a model to grade a model". Not-applicable items are listed
at the bottom with the reason they do not apply.

| tab | what's in it |
|---|---|
| **Item** | The selected item: expected against observed, the reason, per-pass verdicts, and the evidence. Says plainly when the harness overruled the model. |
| **Score math** | Per-tier bars, the not-applicable list, every penalty and whether the −25 cap bound, and the arithmetic written out. The answer to "why is this 62 and not 70". |
| **vs ViralScore** | Both scores, the difference, and which items the rubric took points off for. Empty when the build has no crowd runs. |
| **Transcript** | The grader's recorded tool calls, the harness's own log, written before the model saw each result. Lazily fetched. |
| **Rubric** | The rubric exactly as graded, including items that passed. Recorded in the grade, so it stays accurate after the rubric file is edited. |
| **Files on disk** | The `paths` block and any evidence screenshots. |

**Three verdicts, not two.** `PASS`, `FAIL` and `UNRESOLVED` are distinct
everywhere. Unresolved and failed both earn zero, but they mean opposite things:
one is a defect in the app, the other is a fault in the instrument. Collapsing
them is how a flaky grader starts looking like a bad app.

Grades live in `builds/rubric/<build_id>__rubric-<timestamp>/`, holding
`grade.json`, `transcript.jsonl` and `shots/`. The directory convention is the
same prefix-glob one the crowd runs use, so a build can be graded more than once
(new rubric version, different grader model, a re-run) without clobbering.

---

## 4. Running a built app

Both viewers have a **Run the app** tab. It:

1. copies the build's `app/` into a throwaway dir under `viz/cache/runs`,
2. runs the manifest's `setup` steps (failures are reported, not fatal, because most
   builds ship a prebuilt `dist/`),
3. starts the app on a private free port, exporting `PORT` and a durable
   `VIRALBENCH_DATA_DIR`,
4. fronts it with a proxy on a second port and shows it in the pane.

The proxy exists for two reasons, both ported from the benchmark's `serve-build`:

- **It forbids browser caching.** Most builds serve `/app.js` and `/style.css` from
  a hand-rolled `http.server` with no `ETag` or `Cache-Control`, which puts Chrome
  into heuristic caching. Test build A then build B and the browser silently reuses
  A's JavaScript against B's markup: right-ish layout, no styling, dead buttons.
  That is indistinguishable from a broken build, and expensive to misdiagnose.
- **It moves the port.** Most manifests declare port 8000, so builds cannot run side
  by side as written. Each one gets its own pair of ports, so open as many as you
  like.

`■ Stop` kills the process group and deletes the throwaway copy. Everything still
running is stopped when you Ctrl-C the server.

A build with no valid `viralbench.json` is *undeliverable* and says so instead of
failing obscurely. If a start fails you get the app's log in the pane.

> Where `npm` is missing from `PATH` and only `node` is present, a `npm run build`
> setup step reports `exit 127`. Apps that ship a prebuilt `dist/`, which is most
> of them, start anyway.

---

## 5. Getting the JSON out

The raw trajectories are already in the build folder, and the **Files on disk** tab
shows the exact paths:

```
builds/work/<build_id>/build.json           the build record
builds/work/<build_id>/transcript/*.json    one NDJSON transcript per turn
builds/crowd/<run_id>/run_summary.json      the crowd run
builds/crowd/<run_id>/actions.jsonl         every social action
builds/crowd/<run_id>/simulation.db         users, posts, follows, likes, comments
builds/crowd/<run_id>/traces/agent_N.json   one agent's hands-on trial
builds/crowd/<run_id>/trajectories.json     per-agent reasoning
```

Two toolbar buttons on both viewers:

- **`JSON ↓`** gives one bundle with everything: the record, the merged timeline, the
  per-turn transcripts parsed into event arrays (crowd: the run plus every agent's
  trial). Base64 screenshot payloads are dropped, since they can be 93% of a
  transcript's bytes and the images are exported separately.
- **`Zip ↓`** gives the raw artifacts as they sit on disk, plus `trajectory.json` and
  the screenshots.

Or hit the API directly:

```bash
curl localhost:8770/api/founder/<build_id>/download > trajectory.json
curl localhost:8770/api/crowd/<run_id>/download?format=zip > run.zip
```

---

## 6. Reproducing the three-arm comparison

Build one brief on every arm, crowd-score each build, then host them side by side.

```bash
uv run viral-bench found <idea> --agents 1        # single agent
uv run viral-bench found <idea> --agents 4        # 4-agent team, local
uv run viral-bench found <idea> --agents dynamic  # dynamic orchestrator
uv run viral-bench crowd-run <build_id>           # once per build
python3 viz/serve_apps.py --build <solo_id>=8001 --build <team_id>=8002 --build <dyn_id>=8003
```

Run each build under `scripts/netns_run.sh` with its own `XDG_DATA_HOME`. Both are
load-bearing: every arm serves the app it is writing on the port its own manifest
declares (almost always 8000) and agents tidy up with `pkill -f`, so without a
private network and PID namespace one build's cleanup kills another's server and one
build's QA reviews another's app. The private data home keeps each build's opencode
session store separate, which is what `dump_session_trace` walks each turn.

`serve_apps.py` pins each app to a fixed port behind the same no-cache proxy the
viewer uses. It does not assume an app honours the port it is given: some read
`$PORT`, some take it on the command line, and some hard-code it in their own source
and ignore both. It starts the app, waits to see which port opened, and refuses to
proxy to a port another app already holds, because "serving the previous build's app"
is indistinguishable from "this build works".

## 7. Checking the UI renders

```bash
.venv/bin/python viz/tools/shoot.py founder <build_id> --tab Thinking
.venv/bin/python viz/tools/shoot.py crowd <run_id> --tab "Trial replay"
```

Loads the page in real Chrome, asserts against the live DOM, and writes a screenshot
to `viz/cache/shots-ui/`. It reports JS exceptions, console errors, and the two
failures a screenshot alone would miss: `[object Object]` leaking into the page, and
stray `undefined`. Runs under the benchmark's venv, which is where Playwright lives.

## Keyboard

| key | does |
|---|---|
| `space` | play / pause |
| `←` `→` | step back / forward |
| `/` | open the browser dialog |
| `esc` | close it |

---

## How it is put together

```
viz/
  serve.py          entry point: routes, JSON API, static files
  core/
    paths.py        build discovery, the index, the read-only guard
    founder.py      NDJSON transcripts -> a timeline (all three structures)
    crowd.py        run_summary + actions.jsonl + sqlite + traces -> graph & timeline
    rubric.py       grade.json -> items, evidence index, paged transcript
    apphost.py      run a built app behind a no-cache proxy
    export.py       download bundles
    cache.py        rescue crowd screenshots out of /tmp
  core/trace.py     session dumps -> the trajectory schema (thinking, prompts, patches)
  serve_apps.py     hold several built apps up on fixed ports
  tools/shoot.py    render the UI in real Chrome and assert on the DOM
  static/           index / founder / crowd / rubric, plain HTML, CSS, ES modules
  tests/            118 tests: parsers, trace semantics, damaged artifacts, routes
  cache/            gitignored: parsed caches, rescued shots, app run dirs
```

Design decisions worth knowing if you extend it:

- **The readers never import `viral_bench`.** They are driven by what is on disk, so
  a build recorded in July opens with the same code as one recorded today, no matter
  which branch is checked out. It also means the viewer cannot be broken by a
  refactor in the benchmark.
- **Damaged artifacts are normal.** A killed turn leaves a truncated final line, a
  failed turn leaves a 0-byte file, and `actions.jsonl` can end mid-object. Every
  reader parses per-line and skips what it cannot read, because a failed build is
  exactly the kind you want to look at.
- **Timelines are compact, bodies are lazy.** The timeline carries previews and
  records the `(phase, line)` each event came from, and the full record is re-read
  from disk when you click. The largest transcript seen so far is 5.26 MB, 93% of
  it base64 screenshots.
- **Two clocks in the crowd viewer.** The social clock is a discrete integer
  (`round N` is `created_at N+1`). The trial clock is real wall-clock seconds inside
  one agent's session. They play independently because one happens inside a tick of
  the other.

### What shows up, and when

The viewer is pointed at a builds tree an active benchmark keeps writing into, so
"will tomorrow's run appear?" is a property worth stating exactly. Nothing has to
be registered, imported or pinned. Discovery is a filesystem scan, and there is no
list of known ids anywhere.

| you do | what happens | latency |
|---|---|---|
| paste a **build id** or **crowd run id** | resolved straight to a path and stat-ed, never via the index | **immediate**, no restart |
| **browse / search** the pickers | served from a TTL cache over a directory scan | ≤60s builds, ≤5min crowd runs |
| hit **`?refresh=1`** (the pickers' reload) | rescans now | immediate |
| re-run a build in place | mtime+size signature per build invalidates its parsed cache | immediate |

Two consequences worth knowing:

- **A run that finished one second ago opens by id.** It will not be in the
  *browsable list* until the TTL rolls over or you refresh. Both are covered by
  tests, so a future change that routes lookup through the index would fail loudly
  rather than making fresh builds look lost.
- **New pipeline arms are listed, not dropped.** `classify_mode` falls through to a
  descriptive label for a `structure` it has never seen, so adding a fourth arm to
  the benchmark does not make its builds invisible here until the viewer is taught
  about it. Same for the crowd: `CROWD_SETS` covers `crowd/ ablation/ calibration/
  smoke/`, and a new sibling directory is the one case that needs a one-line edit.

### Coverage

Every artifact in a corpus is parsed through these readers rather than sampled, and
the readers are expected to open the whole tree without crashing on any of it. Two
known limits, both about absent data rather than the reader:

- **Some builds and crowd runs are empty by origin.** They open and report their
  status correctly, but the timeline is blank because the artifact is 0 bytes:
  `harness_timeout` builds whose transcript was never written, and failed runs whose
  `actions.jsonl` is empty. There is nothing to draw.
- **A directory in `builds/work` that never wrote `build.json` is invisible** to the
  viewer entirely, even though it holds transcripts. The record is written at the end
  of a build, so anything killed before that has no id the index can find. Fixing it
  would mean synthesizing a record from the directory, and the deliberate choice is to
  leave those out rather than invent one.

### Why the test modules are named `test_viz_*`

Pytest imports test modules by bare name with their own directory on `sys.path`, and
this repo's `tests/crowd/` contains both a `conftest.py` and a `test_trace.py`.
Collected together with the obvious layout, `from conftest import ...` silently
resolves to the *crowd* conftest and `test_trace` collides outright, giving three
collection errors whose winner depends on import order. Hence: helpers live in
`vizfixtures.py` rather than `conftest.py`, and the modules carry a `test_viz_`
prefix. Renaming either back reintroduces the collision.

### Tests

```bash
.venv/bin/python -m pytest viz/tests -q
```

118 tests. They also run as part of a bare `pytest`, since `testpaths` lists
`viz/tests` alongside `tests`, and `addopts` measures coverage of `viz` as well as
`viral_bench`.

```bash
.venv/bin/python -m ruff check viz/ && .venv/bin/python -m ruff format --check viz/
```
