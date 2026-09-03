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

# Grading rubrics — the RubricScore track

A second, **independent** scoring track. Every built app is graded against a
per-app rubric, the way a benchmark is normally scored, so the result can be
compared head to head with the crowd's ViralScore.

> [!IMPORTANT]
> **Nothing in this directory is ever shown to the founder model.** The idea
> loader globs `ideas/*.yaml` **non-recursively**, so this subdirectory is
> excluded by the same mechanism that excludes `templates/`. That exclusion is
> the whole reason the rubrics live here rather than as a field on `Idea` —
> `render_idea()` serialises the entire `Idea` object into the build prompt, so
> a rubric field there would be one careless line away from leaking.
>
> Showing a model the checklist makes it optimise for the checklist. The rubric
> only ever tests what `ideas/<id>.yaml` already promises, so keeping it hidden
> is strict but not unfair.

The crowd never sees these either. Triers are valuable because they are naive
users; handing them a checklist turns them into QA.

## Files

| file | contents |
|---|---|
| `_universal.yaml` | Tier 0 gate, Tier 3 robustness items, universal penalties — applied to every idea |
| `<idea_id>.yaml` | Tier 1 (success criteria), Tier 2 (core features), app-specific penalties |

The leading underscore keeps `_universal.yaml` from being mistaken for an idea.

## Scoring

```
base  = 100 * (points earned / points applicable)
score = max(0, base + penalties)          # penalties are negative, capped at -25
```

Tier 0 is a **gate**: any failure ⇒ score 0, regardless of everything else. Its
item results are still recorded for diagnostics.

| tier | covers | nominal points |
|---|---|---|
| 0 | deliverability | gate |
| 1 | the brief's `success_criteria`, executed end to end | 40 |
| 2 | the brief's `core_features`, each working | 25 |
| 3 | robustness and craft (universal) | 35 |
| penalties | anti-patterns and spec-gaming | 0 to −25 |

Scoring normalises over *applicable* points, so an item marked not-applicable
for an idea (see `universal_overrides`) does not distort that idea's total.

## Item schema

```yaml
- id: S3                    # unique within the file; S=tier1, F=tier2, R=tier3, G=gate, A/P=penalty
  text: "..."               # the falsifiable claim being tested
  points: 10                # integer; negative for penalties
  method: assert            # probe | assert | agent | source
  expect: "..."             # optional: the exact expected value, for assert items
  note: "..."               # optional: evidence or rationale
```

### Methods, in order of preference

| method | who decides pass/fail |
|---|---|
| `probe` | code, no model in the loop (HTTP status, exit code, file bytes) |
| `assert` | **code**, comparing against `expect`; the model only navigates the app to the state and reads off the observed value |
| `agent` | the grader model, which must cite the step and observation proving its verdict |
| `source` | code + model over the built source tree |

`assert` is the backbone. The model gathers, the code judges — that is what
keeps the grader from having discretion over the high-value items.

## Rules the grader must enforce

1. **Every PASS cites evidence.** A step index plus the observation excerpt that
   proves it. No evidence ⇒ recorded FAIL.
2. **Missing evidence is FAIL, never PASS.** "Could not determine" resolves to
   FAIL, reported separately from "checked and failed" so instrument faults stay
   visible.
3. **Blinded.** The grader never learns which model or founder structure
   produced the build, and never sees the crowd's verdicts or the ViralScore.
4. **Three passes**, item passes on ≥2 of 3. Per-item disagreement is recorded as
   the track's own noise floor.

## Authoring rules

- Every item must be derivable from `ideas/<id>.yaml` plus ordinary web-app
  quality standards. If a competent engineer reading only the brief could not
  have written it, it does not belong.
- **Do not import crowd preferences the brief never asked for.** Measured
  example: the crowd docks `image_compressor` heavily for not surviving a
  reload, but its brief never asks for persistence — so `R1` is marked
  not-applicable there rather than punishing a correct build.
- Penalties are the deliberate exception: they are informed by observed
  spec-gaming, and they exist to name ways of satisfying the letter of a brief
  while defeating it.
- Prefer exact expected values. `expect: "Hello World"` beats "output looks
  right".
