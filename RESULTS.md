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

# ViralBench results

Eleven models across three founder configurations, measured on one frozen instrument. The
[README](README.md) carries the headline tables. This file has the rest, plus everything
needed to judge how much weight the numbers carry.

- [What was run](#what-was-run)
- [Reading a ViralScore](#reading-a-viralscore)
- [Per-pipeline results](#per-pipeline-results)
- [Does the instrument work](#does-the-instrument-work)
- [Cross-checking with the rubric](#cross-checking-with-the-rubric)
- [Limits](#limits)
- [Reproducing this](#reproducing-this)

## What was run

| | |
|---|---|
| Briefs | 25, from `ideas/` |
| Models | 11 |
| Founder configurations | 3 (single agent, 4-agent team, dynamic) |
| Cohort | r4 |
| Builds attempted | 824 |
| Builds healthy | 817 (99.2%) |
| Crowd runs | 2446, at three seeds per build |
| Crowd size | 30 agents per run |
| Score profile | `v7_equal`, `SCORE_VERSION` 1.8 |
| Crowd architecture | v14 |

Coverage by configuration:

| Configuration | Builds healthy | Crowd runs | Seed noise |
|---|---|---|---|
| Single agent | 272/275 | 815 | 3.79 |
| 4-agent team | 272/275 | 813 | 3.68 |
| Dynamic | 273/274 | 818 | 3.55 |

Every model was run on every brief under every configuration, on one cell short of the
full 825, and every build was scored by the same crowd at the same three seeds. The only
thing that differs between cells is the model that wrote the code and the shape of the
founding team.

A fourth configuration was evaluated internally, in which the four agents collaborated
through an external workspace rather than a shared directory. It depended on
Google-internal infrastructure and is not part of this release, so it is excluded here.

## Reading a ViralScore

**The noise floor is 3.67 points.** That is the mean seed-to-seed standard deviation
inside a cell, pooled across the three configurations. Two numbers closer together than
that are not distinguishable, and this document does not present them as ordered.

The score is not a rating of the code. It is computed from what a simulated crowd did
with the running app: whether agents adopted it, whether they recommended it to each
other, and how they rated it after use. A model can write clean code that nobody wants,
and it will score low. That is the intent.

## Per-pipeline results

### Single agent

One agent, one design turn, one build turn, in a single continued session. No specialist
roles, no separate QA gate. This is the baseline every other configuration is measured
against.

| # | Model | Built | ViralScore | Craft | Adoption | Seed SD |
|---|---|---|---|---|---|---|
| 1 | Claude Opus 5 | 25/25 | 68.9 | 0.82 | 0.79 | 2.36 |
| 2 | Gemini 3.8 Flash | 25/25 | 67.4 | 0.82 | 0.78 | 4.26 |
| 3 | Gemini 3.6 Flash | 25/25 | 63.6 | 0.78 | 0.69 | 4.44 |
| 4 | Claude Opus 4.7 | 25/25 | 61.9 | 0.79 | 0.69 | 4.25 |
| 5 | Claude Opus 4.8 | 25/25 | 61.6 | 0.80 | 0.70 | 5.60 |
| 6 | Gemini 3.7 Flash | 24/25 | 60.6 | 0.79 | 0.68 | 3.92 |
| 7 | Claude Sonnet 5 | 25/25 | 56.8 | 0.75 | 0.64 | 3.68 |
| 8 | Gemini 3.5 Flash | 23/25 | 46.4 | 0.67 | 0.45 | 4.26 |
| 9 | Gemini 3.5 Flash Lite | 22/25 | 32.9 | 0.52 | 0.32 | 3.40 |
| 10 | Gemini 3.1 Pro Preview | 25/25 | 30.8 | 0.57 | 0.22 | 4.49 |
| 11 | Gemini 2.5 Flash | 15/25 | 8.0 | 0.42 | 0.03 | 1.04 |

**Built** counts briefs that produced a deliverable app. **Craft** and **Adoption** are
the two heaviest score components.

Delivery is the clearest divider at the bottom of the field. Gemini 2.5 Flash shipped 15
of 25, Flash Lite 22, Gemini 3.5 Flash 23. Everything above them shipped 24 or 25.

### 4-agent team

Architect, Implementer, Designer and QA over a shared working directory, up to three
rounds, with QA able to ship early once the app meets its brief.

| # | Model | Built | ViralScore | vs single agent |
|---|---|---|---|---|
| 1 | Gemini 3.7 Flash | 25/25 | 68.3 | +7.7 |
| 2 | Claude Opus 5 | 25/25 | 67.5 | -1.4 |
| 3 | Claude Opus 4.7 | 25/25 | 65.1 | +3.2 |
| 4 | Gemini 3.8 Flash | 25/25 | 64.9 | -2.5 |
| 5 | Gemini 3.6 Flash | 25/25 | 61.3 | -2.3 |
| 6 | Claude Opus 4.8 | 24/25 | 60.1 | -1.5 |
| 7 | Claude Sonnet 5 | 22/25 | 52.9 | -3.9 |
| 8 | Gemini 3.1 Pro Preview | 24/25 | 50.1 | +19.3 |
| 9 | Gemini 3.5 Flash Lite | 21/25 | 48.4 | +15.5 |
| 10 | Gemini 3.5 Flash | 15/25 | 28.6 | -17.8 |
| 11 | Gemini 2.5 Flash | 22/25 | 19.9 | +11.9 |

The two largest gains and the largest loss all sit in the bottom half, which is the
conditional effect described below rather than a ranking change. This configuration is
also where delivery is most uneven: Gemini 3.5 Flash shipped only 15 of 25 under the
four-role process against 23 alone, while Gemini 2.5 Flash went the other way, 22 against
15.

### Dynamic

One founder agent that designs its own team and spawns subagents through the agent CLI's
own task tool. Nothing prescribes how many or what roles.

| # | Model | Built | ViralScore | vs single agent |
|---|---|---|---|---|
| 1 | Claude Opus 5 | 25/25 | 68.2 | -0.7 |
| 2 | Gemini 3.8 Flash | 25/25 | 65.3 | -2.1 |
| 3 | Claude Opus 4.7 | 25/25 | 62.5 | +0.6 |
| 4 | Gemini 3.7 Flash | 25/25 | 61.9 | +1.3 |
| 5 | Claude Sonnet 5 | 25/25 | 61.8 | +5.0 |
| 6 | Claude Opus 4.8 | 24/25 | 60.8 | -0.8 |
| 7 | Gemini 3.6 Flash | 25/25 | 56.7 | -6.9 |
| 8 | Gemini 3.5 Flash | 25/25 | 47.1 | +0.7 |
| 9 | Gemini 3.1 Pro Preview | 25/25 | 36.8 | +6.0 |
| 10 | Gemini 3.5 Flash Lite | 25/25 | 31.7 | -1.2 |
| 11 | Gemini 2.5 Flash | 25/25 | 20.5 | +12.5 |

Dynamic has the most even delivery of any configuration: ten of the eleven models shipped
all 25 briefs and the eleventh shipped 24. The models the four-role process breaks or
rescues both land at 25 here.

### All three together

| Model | Single agent | 4-agent team | Dynamic | Mean |
|---|---|---|---|---|
| Claude Opus 5 | 68.9 | 67.5 | 68.2 | 68.2 |
| Gemini 3.8 Flash | 67.4 | 64.9 | 65.3 | 65.9 |
| Gemini 3.7 Flash | 60.6 | 68.3 | 61.9 | 63.6 |
| Claude Opus 4.7 | 61.9 | 65.1 | 62.5 | 63.2 |
| Claude Opus 4.8 | 61.6 | 60.1 | 60.8 | 60.8 |
| Gemini 3.6 Flash | 63.6 | 61.3 | 56.7 | 60.5 |
| Claude Sonnet 5 | 56.8 | 52.9 | 61.8 | 57.2 |
| Gemini 3.5 Flash | 46.4 | 28.6 | 47.1 | 40.7 |
| Gemini 3.1 Pro Preview | 30.8 | 50.1 | 36.8 | 39.2 |
| Gemini 3.5 Flash Lite | 32.9 | 48.4 | 31.7 | 37.7 |
| Gemini 2.5 Flash | 8.0 | 19.9 | 20.5 | 16.1 |
| **Arm mean** | **50.8** | **53.4** | **52.1** | |

Three things fall out of this table.

**The model dominates.** The model column spans 52.1 points. The configuration row spans
2.6. Which model receives the brief matters roughly 20 times more than how the founding
team is organised, and this is the finding that has survived every cohort unchanged.

**No configuration is separated from another.** A 2.6-point span sits below the
3.67-point noise floor. There is a visible ordering in the arm means and it should not be
reported as a result.

**The structure effect is conditional on model strength.**

| Group | Single agent | 4-agent team | Gain |
|---|---|---|---|
| Top 5 by mean | 64.1 | 65.2 | +1.1 |
| Bottom 5 by mean | 35.0 | 40.0 | +5.0 |

A fixed four-role process is worth around a point to the strongest models and around five
to the weakest. It substitutes for capability rather than multiplying it. If you can
afford the strong model, scaffolding buys little. If you cannot, it buys a fair amount.

## Does the instrument work

A benchmark that cannot tell a good app from a bad one produces numbers that look like
results. Three checks, all run alongside the fleet under the same crowd.

| Check | Result | What it shows |
|---|---|---|
| Deliberately broken control | 3.5 | The crowd finds the corpse on its own. The formula is not doing the work. |
| Median working build | 61.8, margin +58.3 | The scale is used, not compressed into a narrow band. |
| Working cells above the control | 249 of 272, single agent | Almost no working app scores anywhere near the broken one. |
| Minimal-correct fixture | 37.7 | A plain but correct app lands well above the floor and well below a good one. |
| Seed-to-seed noise | 3.79 / 3.68 / 3.55 by arm | Noise is even across configurations, so no arm is less reliable than another. |

The control is worth describing precisely, because a broken control that fails to start
proves nothing. This one builds, runs, passes its own smoke check and is reached by all 30
agents. It is a functional but useless app. The crowd is rating it purely on merit and no
delivery gate is involved.

## Cross-checking with the rubric

Every app was also scored a second way, by an instrument that never saw the first. A
hidden per-brief rubric grades the shipped app against a deliverability gate and 19 to 21
checked items. Roughly 91% of the points are decided by deterministic code checks rather
than model judgement.

Nothing in the rubric is ever shown to the founder or to the crowd. Showing a model the
checklist makes it optimise for the checklist, and handing the crowd one turns naive users
into QA.

The two agree on the models. Ranking the eleven by each instrument gives a Spearman of
0.864, with 47 of 55 pairwise orderings matching and no model moving more than three
places:

| Model | ViralScore rank | RubricScore | RubricScore rank |
|---|---|---|---|
| Claude Opus 5 | 1 | 52.7 | 3 |
| Gemini 3.8 Flash | 2 | 54.4 | 1 |
| Gemini 3.7 Flash | 3 | 48.0 | 6 |
| Claude Opus 4.7 | 4 | 52.0 | 4 |
| Claude Opus 4.8 | 5 | 53.0 | 2 |
| Gemini 3.6 Flash | 6 | 45.4 | 7 |
| Claude Sonnet 5 | 7 | 51.6 | 5 |
| Gemini 3.5 Flash | 8 | 31.8 | 9 |
| Gemini 3.1 Pro Preview | 9 | 40.8 | 8 |
| Gemini 3.5 Flash Lite | 10 | 25.7 | 10 |
| Gemini 2.5 Flash | 11 | 14.7 | 11 |

**This is the strongest evidence that the crowd is measuring something real.** A
simulation of thirty agents deciding whether they would use an app could plausibly be
measuring its own quirks — persona wording, feed dynamics, how talkative the crowd model
is. If it were, its ranking would not survive contact with a conventional checklist
written independently, never shown to the crowd, and decided ~91% by deterministic code.
It does survive: both instruments put the same models at the top, the same models at the
bottom, and agree on 47 of 55 head-to-head orderings. The crowd is tracking whether the
software is any good, not noise.

What the rubric cannot do is replace it. The two ask different questions — the rubric
asks whether the artifact meets a specification, the crowd asks whether an audience would
adopt and spread it — and the places they part company are where that shows. Gemini 3.7
Flash is sixth on the checklist and third with the crowd; a tile-sliding puzzle can
satisfy every item on its rubric and still be something nobody passes on. Agreement is
highest among the weakest models, where builds fail in ways both instruments can see, and
lowest among the strongest, where what remains is judgement rather than function.

Agreement on individual builds is much lower, 0.496, and that is expected rather than a
problem: a single build is a noisy measurement under both instruments, so the correlation
is attenuated by error on both sides at once. Averaging to the model removes it. That is
the reason a benchmark reports model means rather than per-build scores.

The disagreements are more informative than the agreement. Of the 25 largest per-build
divergences, 15 fall in only two briefs, and each diverges consistently in one direction.
One is a rubric blind spot on a canvas-drawing app, where no build exposes a readable
scene model and several checks fall back to a model that cannot introspect the drawing.
The crowd has no such problem, because using an app does not require introspecting it.

## Limits

- **25 briefs, all web apps**, single-page or small full-stack. Nothing here speaks to
  systems programming, mobile, data engineering or anything long-lived.
- **The crowd is a simulation.** It predicts what a simulated population does with an app.
  That is a proxy for real adoption, not a measurement of it, and it has never been
  validated against real user behaviour.
- **Two vendors.** Resource constraints meant only Gemini and Anthropic models were
  measured, both through Vertex AI. No OpenAI, Llama, Qwen, Mistral or locally-hosted
  model appears here. The harness runs all of them, but this sweep did not.
- **One cohort.** Every number is a single cohort on a single frozen instrument version.
- **A model can be measured only on what it shipped.** A model that fails to produce a
  deliverable app for a brief contributes no score for that cell, so the score of a
  low-delivery model is computed over the subset it managed to build. The `Built` column
  is there so that is visible rather than hidden.

## Reproducing this

These numbers were produced under score profile `v7_equal`, `SCORE_VERSION` 1.8 and crowd
architecture v14, which are the versions this repository ships.

**One thing has changed since they were measured.** The founder prompt used to instruct
generated apps to call a specific vendor's API when a brief needed a model. The released
prompt instructs them to call whichever OpenAI-compatible endpoint the `app` stage is
configured with, so that a generated app has one code path regardless of provider. This
was necessary for the harness to be provider-neutral, and it changes the brief the founder
receives.

The repository detects this on its own. Every build records a fingerprint of the exact
prompts it was given, and the corpus loader refuses to pool builds whose fingerprint does
not match the current one. A rerun today therefore starts a fresh cohort rather than
silently extending this one, which is the behaviour you want. It also means the tables
above will not reproduce cell for cell.

To run your own comparison:

```bash
uv run viral-bench init
uv run viral-bench doctor
uv run viral-bench found <idea> --model <provider>/<model>
uv run viral-bench crowd-run <build_id>
uv run viral-bench score <build_id>
```

For a full sweep rather than a single build, see `scripts/build_fleet.py` and
`scripts/crowd_sweep.py`. Hold the crowd, grader and autorater models fixed across
everything you intend to compare.
