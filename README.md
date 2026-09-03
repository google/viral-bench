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

# ViralBench

ViralBench measures whether a model can act as a **founder**: take a one-paragraph
product brief, design and build a working web app end to end, and then find out whether
anyone would want it.

That second half is the unusual part. Most coding benchmarks grade the artifact against a
test suite the author wrote. ViralBench grades it by *use*. The finished app is started in
a sandbox and handed to a crowd of simulated, persona-driven users who open it in a real
browser, click around, form opinions, post about it, and decide whether to pass it on. Out
comes a **ViralScore** (0-100), plus a second, independent **RubricScore** asking the
ordinary question: does it do what the brief said? ViralBench ships no model and no
default provider. You bring a key.

[Results](#results) · [Prerequisites](#prerequisites) · [Quickstart](#quickstart) · [Bring your own model](#bring-your-own-model) ·
[Ideas](#the-idea-bench) · [Founder](#the-founder-stage) · [Sandboxing](#sandboxing) · [Crowd](#the-crowd) ·
[ViralScore](#viralscore) · [RubricScore](#rubricscore) · [Trajectories](#trajectories-and-the-viewers)

```
idea (ideas/*.yaml)
   │
   ├─ founder ──► an app + a viralbench.json manifest, shipped to builds/store
   │              the model under test, driving the opencode agent CLI
   ├─ crowd ────► simulation.db, per-agent trials, run_summary.json
   │              persona agents on a mock social platform, in a real browser
   ├─ score ────► ViralScore 0-100   would anyone adopt and share it?
   └─ rubric ───► RubricScore 0-100  does it do what the brief asked?
```

Only the founder model is meant to vary, because it is the thing under test. The crowd, the
rubric grader and the optional autorater are the measuring instrument and are configured
separately, so swapping the model under test cannot move the yardstick it is measured
against. `tests/test_models.py` asserts that separation.

## Results

Eleven models, 25 briefs, three founder configurations, one frozen instrument. 817 of 824
builds healthy over 2446 crowd runs. ViralScore runs 0 to 100.

Measured seed-to-seed noise is **3.67 points**.

### Baseline: one founder agent

Every model below built the same 25 briefs under the **single-agent** configuration
alone: one agent, one design turn, one build turn. The only thing that differs between
rows is the model that wrote the code. This is the baseline the other two configurations
are measured against, not an average over them; the three are set side by side in the
next section. Each ViralScore is that model's mean over its briefs, and every build was
judged by the same crowd at the same three seeds.

| # | Model | Built | ViralScore | Craft | Adoption |
|---|---|---|---|---|---|
| 1 | Claude Opus 5 | 25/25 | 68.9 | 0.82 | 0.79 |
| 2 | Gemini 3.8 Flash | 25/25 | 67.4 | 0.82 | 0.78 |
| 3 | Gemini 3.6 Flash | 25/25 | 63.6 | 0.78 | 0.69 |
| 4 | Claude Opus 4.7 | 25/25 | 61.9 | 0.79 | 0.69 |
| 5 | Claude Opus 4.8 | 25/25 | 61.6 | 0.80 | 0.70 |
| 6 | Gemini 3.7 Flash | 24/25 | 60.6 | 0.79 | 0.68 |
| 7 | Claude Sonnet 5 | 25/25 | 56.8 | 0.75 | 0.64 |
| 8 | Gemini 3.5 Flash | 23/25 | 46.4 | 0.67 | 0.45 |
| 9 | Gemini 3.5 Flash Lite | 22/25 | 32.9 | 0.52 | 0.32 |
| 10 | Gemini 3.1 Pro Preview | 25/25 | 30.8 | 0.57 | 0.22 |
| 11 | Gemini 2.5 Flash | 15/25 | 8.0 | 0.42 | 0.03 |

Delivery separates the bottom: Gemini 2.5 Flash produced a deliverable app for 15 of 25
briefs. Craft spans 0.42 to 0.82 while adoption spans 0.03 to 0.79, so what divides these
models is not whether the code looks competent but whether the crowd would use it.

### Model against pipeline

Run all eleven under each configuration and the arm means land at 50.8 for the single
agent, 53.4 for the 4-agent team and 52.1 for dynamic.

**Which model gets the brief matters about 20 times more than how the founding team is
organised.** The one effect that survives is conditional: a fixed four-role process is
worth about a point to the strongest models and about five to the weakest, substituting
for capability rather than multiplying it.

### Is the instrument measuring anything?

A deliberately broken control app scores **3.5** against a median working build of
**61.8**. The control builds, runs, passes its own smoke check and is reached by every
agent: it is a functional but useless app rather than a dead one, so the crowd rates it
purely on merit and no gate is involved.

### What this does not tell you

- 25 briefs, all single-page or small full-stack web apps. Nothing here speaks to systems
  work, mobile, or anything long-lived.
- The crowd is a simulation. It predicts what a simulated population does with an app,
  which is a proxy for adoption and not a measurement of it.
- Resource constraints meant only Gemini and Anthropic models were measured, both through
  Vertex AI. No OpenAI, Llama, Qwen, Mistral or local model appears above. That is a limit
  of this sweep, not of the harness: every one of them runs, and the next section is how.
- One cohort, one frozen instrument version. These numbers are cohort r4, produced under
  score profile `v7_equal`, `SCORE_VERSION` 1.8, crowd architecture v14. The released
  founder prompt has since changed in one respect, described in [RESULTS.md](RESULTS.md),
  so a rerun today will not reproduce them cell for cell.

The full model-by-pipeline table, the RubricScore cross-check and the methodology are in
[RESULTS.md](RESULTS.md).

## Prerequisites

A full run costs real API money. The founder can run for an hour on a hard idea, and a
30-agent crowd run is thirty agents each taking several tool-calling turns. Start with
one idea and a cheap model.

| What | Why |
|---|---|
| Python 3.12+ and [uv](https://docs.astral.sh/uv/) | the benchmark itself |
| An API key for any supported provider | there is no bundled model |
| Node 20+ and [opencode](https://opencode.ai) | the founder drives the opencode CLI, a Node program |
| podman or docker | generated apps are untrusted model-written code and run in a container, never directly on your machine |
| Chrome or Chromium | the crowd drives a real browser. Without one, trials degrade to a static HTTP fetch that cannot run JavaScript |
| A second Python 3.11 venv (`.venv-crowd`) | the simulation engine, [OASIS](https://github.com/camel-ai/oasis), needs Python <3.12 and pins heavy ML dependencies (torch, transformers, sentence-transformers) |

`viral-bench doctor` checks all of these and prints the fix for whatever is missing,
before you spend anything. `scripts/setup_crowd_env.sh` builds the isolated crowd env
and prefetches the TWHIN-BERT recommender (~1 GB, one time). Pass `--recsys twitter` to
skip that download.

## Quickstart

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # install uv, one time
git clone https://github.com/google/viral-bench.git
cd viral-bench
uv sync

# Pick a provider, paste a key, choose a model per stage (writes .env and
# config/local.yaml, both gitignored), then check node, opencode, a container
# runtime, a browser, the crowd env, and that every model can be called.
uv run viral-bench init
uv run viral-bench doctor          # --offline to skip the live model calls

# Build an app. <idea> is an idea_id from ideas/ or a path to an idea .yaml.
uv run viral-bench found sliding_tile_game --model openai/gpt-5-mini
uv run viral-bench builds                      # list builds and their status
uv run viral-bench serve-build <build_id>      # run it yourself on :8000
uv run viral-bench try-build <build_id>        # drive it as a crowd agent would

scripts/setup_crowd_env.sh                     # one time. --check to only report
uv run viral-bench crowd-run <build_id>
uv run viral-bench score <build_id>            # ViralScore: would anyone want it?
uv run viral-bench grade <build_id> -m <model> # RubricScore: does it do the job?
.venv/bin/python viz/serve.py --open founder   # http://127.0.0.1:8770
```

Two cheap ways to shake out the wiring first: `crowd-run --no-llm` (scripted actions, no
model calls) and `score --profile v1_deterministic` (no autorater).

## Bring your own model

You bring a key, name it once, and every stage that needs a model uses it.

```bash
uv run viral-bench models                             # providers + which keys you have
uv run viral-bench models --check openai/gpt-5-mini   # a live call, not a lookup

uv run viral-bench found <idea> --model anthropic/claude-sonnet-4-5
uv run viral-bench found <idea> --model openrouter/meta-llama/llama-3.3-70b-instruct
uv run viral-bench found <idea> --model ollama/qwen3-coder
```

`--check` places a live call, because several providers answer a metadata request for a
model the account was never entitled to, so a listing can report a model as available
right up until the first build fails. Models are always `<provider>/<model>`. A bare id
is rejected, since with no default provider there is nothing to guess.

The providers are one registry, `src/viral_bench/providers/spec.py`: OpenAI, Anthropic,
the Google Gemini API, OpenRouter, Groq, Together, Fireworks, DeepSeek, xAI, Mistral,
Ollama, Google Vertex AI (Gemini and Claude), and a generic `custom` entry. Most speak
the OpenAI wire format, so adding one is a single registry entry naming a base URL and a
key variable, with no new code.

**A local or private endpoint works too.** Point `CUSTOM_BASE_URL` at anything
OpenAI-compatible (vLLM, LM Studio, a gateway), or use `ollama`. The founder stage runs
through opencode, and ViralBench synthesises an OpenAI-compatible provider block for
anything opencode has no built-in for, so a provider it has never heard of still works.
Any base URL can be overridden with `<PROVIDER>_BASE_URL`.

### A model per stage

Five stages take a model. `viral-bench init` sets all of them, and `config/local.yaml` is
where the answers land, gitignored so an upgrade never overwrites your choices.

| Stage | What it is | Needs | Notes |
|---|---|---|---|
| `founder` | builds the app. The model under test. | tool calling | the only one meant to vary |
| `crowd` | the simulated users | tool calling, image input | hold fixed across a comparison |
| `grader` | rubric inspection of the shipped app | tool calling, image input | hold fixed |
| `autorater` | optional qualitative scoring | JSON mode | leave blank to disable |
| `app` | what the built apps call, if a brief needs a model | OpenAI-compatible endpoint | give it its own key |

Only the founder is the subject. The other four are the instrument, which is why changing
one of them invalidates a comparison. The `app` stage deserves its own key rather than a
shared one: built apps are untrusted model-written code, and the container can read
whatever it is given.

A model missing a capability is refused when the run starts, rather than scoring a design
dimension it never observed three hours in.

### Where things live

| What | Where |
|---|---|
| The provider registry, every entry | `src/viral_bench/providers/spec.py`, `PROVIDERS` |
| Credential lookup and `<PROVIDER>_BASE_URL` overrides | `src/viral_bench/providers/credentials.py` |
| Key variable names, one per provider | `.env.example` |
| Which model each stage uses | `src/viral_bench/config.py`, `stage_model()` |
| How the founder's provider reaches the opencode CLI | `src/viral_bench/providers/opencode.py` |
| The setup wizard | `src/viral_bench/setup.py` |

Credentials come from the process environment first, then the repo `.env`, never from
`config/`, which is tracked. Config picks *which model*, never *which key*.

### Adding a provider that is not listed

One entry in `PROVIDERS`. If it speaks the OpenAI wire format that is the whole change:

```python
ProviderSpec(
    id="my-gateway",
    name="Internal gateway",
    transport="openai_compat",
    key_env="MY_GATEWAY_API_KEY",
    base_url="https://gateway.example.com/v1",
    capabilities=Capability.TOOLS | Capability.JSON_MODE,
)
```

Then `--model my-gateway/whatever-it-serves`. A provider with a different wire format
needs an adapter beside `openai_compat.py`, `anthropic.py` and `google_genai.py`.

To run with no cloud and no key at all:

```bash
ollama serve
uv run viral-bench found quick_notes_app --model ollama/qwen3-coder
```

## The idea bench

`ideas/` is the frozen test set of 25 briefs, one YAML file each, so every model gets an
identical spec: pitch, problem, target user, must-have features, success criteria,
difficulty and scope. Every idea is a web app, and `allowed_scope` is either `client-app`
(runs in the browser, state in localStorage) or `full-stack-app` (a real backend with
multi-user behaviour, where what one visitor writes another can see). The set leans
client-side, because the apps that spread tend to need no account. Ideas must
fit the sandbox: 1 GB RAM, 1 CPU, no GPU, no local model weights, no second credential.
See [`ideas/README.md`](ideas/README.md) for the schema and how to add one.

## The founder stage

The founder pipeline drives [opencode](https://opencode.ai) non-interactively to design,
build and ship an app, then validates that it produced a `viralbench.json` manifest
saying how to run and smoke-test it. `--agents` selects one of exactly three
configurations.

**`--agents 1` (solo)** is the baseline: one design turn, then one build turn, in a
single session.

**`--agents 4` (team)** is a round-table of four specialists (Architect, Implementer, UX
& Virality Designer, QA & Finisher) over multiple rounds, each a distinct opencode agent
with its own system prompt, temperature, tool permissions and gated skills. It does not
flatten to one agent. **Each keeps its own context**: an agent's first turn opens a fresh
session and every later round *resumes that same session*, so the Architect always reasons
with the Architect's memory even though three others ran in between. **They iterate**:
each round every specialist acts once, reacting to what the others changed, so QA's
findings in round *k* get addressed in *k+1*. **It terminates sensibly**: `--rounds` caps
the work, and QA may ship early, but only after `--min-rounds`, and only if QA was seen
driving the running app that turn.

**`--agents dynamic`** is one founder agent that picks its own team. It gets the idea,
the same deliverable contract as every other mode, and opencode's delegation machinery:
the `task` tool, resumable subagents, and a directory it can write its own agent
definitions into. No roles, no rounds, no division of labour.

That mode exists because the other two encode *one human's* answer to "how should a
founding team be organised?", a fixed relay decided in advance, identical for every
idea and every model, which bottlenecks a strong model into a shape it did not choose.
Dynamic mode removes the answer and keeps the question, so a model is measured on
orchestrating an agentic process as well as on writing code. Whether it delegates at all
is itself a result, and `build.json` records spawn counts and types, subagents *resumed*
rather than re-created, peak concurrency, and the agents it defined for itself.

The harness insists only on finishing: a turn ending with deliverables missing gets a
deliberately content-free nudge naming the gap, and the same session resumes, up to
`--turns`. That is a completion backstop, not a process. Without it a model that stops
after planning would be recorded as unable to build an app, a harness artifact reported
as a capability difference.

```bash
uv run viral-bench found <idea> --agents 4 --rounds 3 --min-rounds 2
uv run viral-bench found <idea> --agents dynamic --turns 4
uv run viral-bench found <idea> --no-browser-tools   # deny the Designer/QA a browser
```

Builds land in a gitignored `builds/`: each app in its own workspace under
`builds/work/<build_id>/`, and shipped (unless `--no-ship`) into a single git store
`builds/store` as an orphan branch `build/<build_id>`, so one repo holds many branches
rather than one repo per build.

## Sandboxing

Generated apps are untrusted, model-written code. They run in a rootless container
(podman or docker, running a plain OCI image built from
[`docker/Containerfile`](docker/Containerfile)) capped at 1 GB RAM / 1 CPU / 512 pids,
with only the app directory mounted. The founder *build* itself runs on the host into a
fresh workspace. Containment applies to running the finished app, which is what the
crowd, the rubric grader and `serve-build --container` all do.

An app that itself calls a model never sees the key the benchmark runs on. It gets its
own, provider-neutral (`VIRALBENCH_APP_LLM_BASE_URL`, `_API_KEY`, `_MODEL`) and resolved
from a separately configured `app` stage. Give that stage its own credential, as a
blast-radius control. One caveat: every built app must start and pass its health check
*without* a key, so a missing or exhausted app key does not look like a failure. It
answers from its fallback and reads as healthy while the crowd rates canned output.

`serve-build` also fronts the app with a no-cache proxy on a private port. Nearly every
build declares port 8000 and serves its code from the same few URLs, so without that,
testing build A then build B hands you A's JavaScript against B's markup. That is
indistinguishable from a broken build, and expensive to misdiagnose.

## The crowd

The crowd stage puts persona-driven agents on a mock social platform, shows them the
app's launch, lets them try it, and records what they do.

- **Triers** get the full interaction toolkit and use the app first-hand in a real
  browser (open, look, click, type, select, upload, reload, screenshot), then publish
  their own take. By default every agent is a trier. **Latecomers** hold the app tools
  but only use them if the feed convinces them, so their conversion rate is virality
  *earned* rather than computed.
- Agents act over rounds with the full social action set (post, comment, like, dislike,
  repost, quote, follow), and an interest-based recommender (TWHIN-BERT) ranks each
  feed, so reach is earned rather than handed to everyone alike.
- Every run is self-contained, with a fresh database, a fresh agent graph and a fresh
  app instance, so scoring one app never leaks into another.

```bash
uv run viral-bench crowd-run <build_id>                         # 30 agents, 3 rounds
uv run viral-bench crowd-run <build_id> --agents 8 --rounds 2   # smaller and cheaper
uv run viral-bench crowd-run <build_id> --recsys twitter --seed 1
```

Artifacts land in `builds/crowd/<build_id>__crowd-<ts>/`: `simulation.db`,
`traces/agent_<id>.json` (each trier's step-by-step evidence and verdict),
`actions.jsonl`, `trajectories.json` (per-agent reasoning) and `run_summary.json`.

**The crowd is a measuring instrument, so it is deliberately fixed.** Hold its model
constant across a comparison, because results from two crowd models are not comparable.
Choose a cheap, fast one and resist upgrading it. A stronger judge sounds obviously
better and measured worse: it was harsher on everything, including apps that deserved
praise, so the top of the scale compressed while a deliberately broken control stayed
put and separation collapsed. A throttled turn is also recorded as an agent choosing to
do nothing, a silent measurement error rather than a visible failure, so lower
`--semaphore` if you see rate limiting.

Crowd size defaults to 30, calibrated rather than picked: 50 measured worse on every
axis at twice the cost. Personas are frozen in `data/crowd/personas.csv`, and asking for
more than the pool holds warns loudly and marks the run clamped, because a scored run
must never be silently under-sized.

## ViralScore

`viral-bench score` turns a crowd run's artifacts into one comparable 0-100 number.

```bash
uv run viral-bench score <build_id>              # the latest crowd run
uv run viral-bench score <crowd_run_dir>         # one specific run
uv run viral-bench score <build_id> --all --autorate --evidence
```

The default profile weighs six parts equally: five deterministic components, plus the
autorater counted as one. That leaves ~83% of the score decided by counts rather than
by a model's opinion.

| Component | What it is |
|---|---|
| `adoption` | fraction of the crowd who say they would use it |
| `advocacy` | √influence-weighted fraction who would put their name behind sharing it |
| `craft` | facet mean (functionality, usability, design, simplicity) from hands-on trials |
| `advocacy_spread` | reposts and quotes **of a peer, by an agent who would itself share it** |
| `persistence` | whether work survived a reload, per agent who checked |
| autorater | `substance`, `severity`, `word_of_mouth`, 1/18 each |

Design decisions, each driven by a measurement rather than by taste:

- **Per-capita participation, never raw counts.** Counts scale with crowd size and are
  heavy-tailed, whereas distinct-actor fractions of the *exposed* audience are bounded
  in [0,1] and comparable between an 8-agent dev run and a 30-agent scored run.
- **Likes are excluded, and cascade is weighted zero.** Likes had no between-app variance
  (every crowd likes everything) while carrying real run-to-run noise, and measured
  strictly, engagement landing on a repost or quote barely occurs at all. Both are still
  reported, and neither moves the number.
- **Reposting the founder's own launch post is not spread.** It was the most
  discriminating single signal measured, for an unflattering reason: most reposting
  actors only ever reposted the launch post, needing no contact with another agent, and
  that rate correlated ~+0.95 with the score, counting app quality twice in the slot
  reserved for spread. Hence `advocacy_spread`, whose advocacy filter matters too: raw
  peer interaction runs *higher* on broken apps (thirty agents piling on to confirm the
  same HTTP 500), and requiring the amplifier's own verdict to be "would share" flips
  the sign.
- **A broken app cannot be viral, but a probe does not get the last word.** The validity
  gate scales the score by how much of the crowd got the app working, instead of
  applying a fixed multiplier the moment a container probe calls it dead. One flaky start
  used to cut a run fivefold, and that run had posted the best adoption and craft of its
  seeds.
- **Breadth is scored, resonance is reported,** so one score stays comparable across
  apps aimed at widely different audiences. And **too little evidence is UNSCORABLE, not
  a number**, because a partial run quietly scored on its steadiest surviving component
  looks *more* trustworthy than a healthy one, the worst failure mode a benchmark can
  have.

Scoring is a **pure offline function of stored artifacts**, stamped with a
`score_version` and a profile name so a comparison can never mix definitions. That makes
re-weighting free: edit `config/score.yaml` and re-run `viral-bench score` rather than
re-running a simulation.

**The autorater.** The deterministic components count things. They cannot read.
`--autorate` adds an LLM rater over the crowd's trajectory for three things a formula
cannot judge: **substance** (praise specific and earned, or reflexive), **severity** (how
damaging the strongest criticism is), **word_of_mouth** (did agents persuade
each other, or form parallel opinions independently). It reads a budgeted evidence pack
of metrics, trials, thread and verdicts, each item carrying an id so a rating can
**cite** it, and the pack is **blinded**, because a rater that can infer which model it
judges may rate reputation instead of artifact. It runs at temperature 0 several times
and takes the median, reporting its spread so an unstable dimension stays visible.

## RubricScore

The second, independent track. The crowd decides whether anyone would *want* the app.
The rubric decides whether it does what the brief asked for. Same build, two
numbers, and the interesting cases are where they disagree. No rubric result feeds any
ViralScore component. Each idea has a rubric in `ideas/rubrics/`, in four tiers: a Tier 0
deliverability gate, Tier 1 for the brief's success criteria, Tier 2 for its core
features, a shared Tier 3 for robustness and craft, and spec-gaming penalties capped at
−25.

**The model gathers, the code judges.** An item either carries a `check` block, in which
case a deterministic primitive decides pass/fail and the grader model's only job is
navigating the app into the state the check needs, or the model judges and must cite the
id of a tool call, whereupon the harness reads the value out of *its own* record of that
call rather than out of the model's prose. Across the shipped rubrics that puts ~91% of
available points beyond the model's discretion, which is the short answer to "you used a
model to grade a model".

**Tier 0 involves no model at all**: the manifest must parse strictly, setup must exit 0,
the app must start and bind, the entry page must render something operable. Any failure
scores 0 and the run stops, because paying a model to grade features on an app that will
not start buys a number that means nothing. Undeliverable builds are scored zero, not
skipped: a benchmark that drops the builds that failed hardest reports the average of the
survivors and calls it the average. And there are **three verdicts, not two**: `PASS`,
`FAIL`, `UNRESOLVED`. The last two both earn zero but mean opposite things, a defect in
the app versus a fault in the instrument, and collapsing them is how a flaky grader
starts looking like a bad app.

Grading needs a live app, so unlike the ViralScore it is not free to redo, though every
per-item verdict is persisted so the aggregation stays re-runnable.

```bash
uv run viral-bench grade <build_id> -m openai/gpt-5.6-luna          # one build
uv run python scripts/rubric_sweep.py --cohort r4 -m openai/gpt-5.6-luna   # a cohort
```

A grader model is required either way, with no default, because the id is written into
every `grade.json` and is the only record of which judge produced the number. The sweep
grades a tagged cohort; pass `--builds <id>,<id>` to name builds directly instead, which
is what you want for builds you just made.

## Trajectories and the viewers

A build record says *what* a model shipped. The trajectory says *how*: the prompts, the
chain of thought, every tool call and file patch, each subagent spawned. None of it
survives in the shipped app, so it is exported separately as `trajectory.json` plus
`events.jsonl` (one self-contained time-ordered record per event). Capture is on by
default.

Three browser UIs read `builds/` directly, with no build step and no npm. `/founder`
replays a build turn by turn on a scrubbable timeline. `/crowd` replays a simulation: the
follow graph forming, the launch spreading, each agent's trial step by step with real
screenshots. `/rubric` shows the grade item by item with its evidence. See
[`viz/README.md`](viz/README.md).

```bash
uv run viral-bench trajectory <build_id>          # -> builds/trajectories/<id>/
uv run viral-bench trajectory <build_id> --zip    # one file you can hand over
uv run viral-bench trajectory --all --summary     # what every build captured

.venv/bin/python viz/serve.py                     # http://127.0.0.1:8770
.venv/bin/python viz/serve.py --open crowd --port 9000
```

## Configuration

`config/` is the tuning surface, and each file documents its own knobs:

| File | What it governs |
|---|---|
| `config/founder.yaml` | founder mode defaults, per-turn timeouts, trajectory capture, the git store |
| `config/crowd.yaml` | browser, interaction limits, crowd size, rounds, feed, sampling |
| `config/score.yaml` | ViralScore weight profiles, the validity gate, the autorater, minimums |
| `config/runtime.yaml` | container runtime, resource caps, network posture, app env injection |

`viral-bench init` writes your per-stage model choices to `config/local.yaml`, gitignored
and taking precedence, so an upgrade never overwrites them. Keys go to `.env`.

## Development

```bash
uv sync                       # all deps, including dev tools
uv run pytest                 # the test suite (also covers viz/)
uv run ruff check . && uv run ruff format .
uv run pre-commit install     # run both on commit
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch → pull
request → review workflow, and [`ideas/README.md`](ideas/README.md) to add an idea.

## License

Apache License 2.0. See [LICENSE](LICENSE).

---

This is not an officially supported Google product. This project is not eligible for
the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
