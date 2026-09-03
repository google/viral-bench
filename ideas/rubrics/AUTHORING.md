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

# Authoring `check:` blocks for a ViralBench rubric

## What you are doing

Each rubric item is a claim about a built web app. An item whose `method:` is
`assert`, `probe` or `source` must be decided by **code**, not by a model. Your
job is to add a `check:` block that names a primitive and its parameters.

Items with `method: agent` get **no** check block — leave them alone.

## The two rules that matter

**1. DOM-agnostic, always.** Forty different builds implement the same idea with
no shared ids, classes or structure. A check written against `#savings` or
`.result-panel` grades one build and silently returns nothing for the other 39.
Read the *rendered text* (`text_matches`, `text_number_equals`,
`text_number_in_range`) or use a generic structural survey
(`controls_have_names`, `dom_count`, `aria_names`, `distinct_sources`).
Use `value_equals`/`value_matches`/`arith` with bespoke `js` ONLY where the
rubric's own `expect:` names a specific machine-readable contract the brief
requires (e.g. the tile game's aria-label format).

**2. Penalties fire on `passed: true`.** A penalty's check must return *yes* when
the bad thing is present. Use `third_party_requests` (not `network_origins`),
`default_restored`, `model_dependency`, `client_secret_present`, or
`text_matches` with `absent:` (which returns yes when the named pattern is
*missing* — useful for "the app is not showing the user's real data").

## `setup:` — the model's entire job

Where an item has a check, the grader model does not judge it. It only
*navigates*. Write a `setup:` string telling it exactly what state to leave the
app in, in plain imperative prose. Reference fixtures by name. Be specific about
waiting ("let it finish processing"). If the item needs before/after sampling,
instruct it to call `capture(label="...")` at each point.

## Uploadable fixtures

| name | what it is |
|---|---|
| `photo.png` | 320x240 PNG, **exactly 816 bytes**. Re-encodes to JPEG *larger*. |
| `photo_large.png` | 1024x768 photographic PNG, ~2 MB. Genuinely compresses. |
| `screenshot.png` | 908-byte PNG of a simple web UI |
| `document.pdf` | one-page PDF, 637 bytes, contains a short paragraph |
| `data.csv` | 5-row CSV with headers |
| `notes.md` | short Markdown with a heading and a list |
| `config.json` | small nested JSON object |

## YAML shape

```yaml
  - id: S3
    text: "..."            # leave unchanged
    points: 6              # leave unchanged
    method: assert         # leave unchanged
    expect: "..."          # leave unchanged if present
    setup: >-
      Imperative instructions for the grader model.
    note: >-
      Only add a note if you are recording a real limitation or a
      non-obvious reason. Do not restate the item.
    check:
      name: text_number_equals
      pattern: "(-?\\d+(?:\\.\\d+)?)\\s*%"
      expect: -167
      tolerance: 25
```

Regex goes in double quotes with `\\d` (escaped once for YAML). Test your regex
mentally against *rendered text*, not HTML.

## If no primitive fits

Do **not** invent a primitive name. Instead leave the item without a `check:`
and add a line to your report saying exactly what primitive would be needed and
why. A wrong check is worse than an absent one — an absent one falls back to the
model, a wrong one silently scores every build incorrectly.

## Primitive catalogue

Generated from the live registry -- regenerate it rather than editing by hand,
because a catalogue that has drifted is worse than none: an author binds a
parameter that does not exist, the call raises `TypeError`, the item records
`unknown`, and those points become unearnable on every build with nothing
looking broken.

```
python scripts/rubric_corpus_facts.py --catalogue
```

| primitive | params | what it decides |
|---|---|---|
| `aria_names` | `pattern, count=None, unique=False` | Accessible names matching a pattern, with an optional exact count. |
| `arith` | `js, expect, tolerance=0.5` | A number the page reports equals a value the grader computed. |
| `canvas_changed` | `before='before', after='after', should_change=True, index=0` | What is drawn on the canvas differs between two harness snapshots. |
| `canvas_ink` | `label='after', minimum=0.01, index=0` | The canvas holds content, not just its background. |
| `captures_monotonic` | `labels, pattern, direction='non_increasing', view='text'` | Numbers pulled from a sequence of harness snapshots move one way only. |
| `client_secret_present` | `pattern=''` | A credential-shaped literal appears in a browser-served file. |
| `clipboard_equals` | `js='', text=''` | The clipboard holds what a named element contains. |
| `clipboard_matches` | `patterns=None, absent=None` | The clipboard's contents match a pattern, rather than a fixed string. |
| `computed_style_changes` | `selector='body', prop='background-color'` | A computed CSS value differs between two harness snapshots. |
| `computed_style_distinct` | `selector, prop='color', min_distinct=3` | At least N distinct computed values of a CSS property. |
| `console_errors_present` | `allow=None` | An uncaught error fired. Passing means the penalty FIRES. |
| `controls_have_names` | `min_controls=1, max_unnamed=0` | Every visible control is a real element carrying an accessible name. |
| `default_restored` | `nonce, js=''` | After a reload the user's own work is gone but content is still shown. |
| `distinct_sources` | `selector='img,canvas', minimum=2` | At least N visually distinct image sources are on screen. |
| `dom_count` | `selector, op='==', n=1` | Count elements matching a CSS selector and compare to ``n``. |
| `dom_text_absent` | `pattern` | The rendered page does not contain ``pattern`` -- the placeholder check. |
| `download` | `magic='', min_size=1, extension=''` | A download fired, and its bytes are the format claimed. |
| `download_absent` | `of_kind=''` | No file was produced. Passing means the penalty FIRES. |
| `download_matches_text` | `text` | The downloaded bytes are exactly ``text`` -- the ``.md`` export check. |
| `download_size_matches` | `pattern, tolerance=0.02, js=''` | The size the app reports equals the size of the file it actually produced. |
| `downloads_distinct` | `minimum=2, last=2` | The last ``last`` downloads are ``minimum`` different files. |
| `http_status` | `path='/', expect=200, max_status=None, anonymous=True, body_contains='', body_excludes=''` | Status (and optionally body) of one route. |
| `image_props` | `width=None, height=None` | The download is a PNG of the expected pixel size, and is not blank. |
| `is_operable` | `selector, min_size=4` | An element exists and a person could actually interact with it. |
| `is_real_control` | `target` | The named control is a real, fillable element with an accessible name. |
| `key_optional` | `` | Nothing crashes at startup merely because a model key is unset. |
| `model_dependency` | `` | The app reads a model API key, i.e. it depends on a third model. |
| `network_origins` | `allow=None, phase_only=False` | Every request went to the app itself. |
| `no_console_errors` | `allow=None` | No uncaught page errors during this item's phase. |
| `no_failed_requests` | `allow=None` | No 4xx/5xx or failed subresource during this item's phase. |
| `no_mojibake` | `text, js=''` | Text the grader entered comes back byte-identical, with no mojibake. |
| `number_in_range` | `js, low=None, high=None` | A reported number is physically possible. |
| `pdf_props` | `pages=None, contains='', excludes=''` | The download is a real PDF, with the right page count and text. |
| `request_payload_hash` | `fixture, url_matches='', min_body=1000` | An outbound request carried the bytes of ``fixture``. |
| `source_absent` | `pattern` | No source file matches ``pattern`` -- the API-key-not-in-client check. |
| `source_present` | `pattern` | Some source file matches ``pattern``. |
| `sql_executes` | `js='', tables=None, foreign_keys=None` | Exported SQL actually runs, and produces the schema it claims. |
| `survives_reload` | `nonce, js=''` | A nonce the grader wrote is still present after a hard reload. |
| `text_matches` | `patterns=None, pattern='', absent=None, js=''` | Every pattern appears in what the user can see; none of ``absent`` does. |
| `text_number_equals` | `pattern, expect, tolerance=0.5, js=''` | A number the page displays equals what the grader computed. |
| `text_number_in_range` | `pattern, low=None, high=None, js=''` | Every number the page displays for this label is physically possible. |
| `text_numbers_distinct` | `pattern, minimum=2, maximum=None, js=''` | Numbers matching a pattern take at least N distinct values. |
| `text_pattern_count` | `pattern, minimum=None, maximum=None, js=''` | How many times a pattern appears in the rendered text. |
| `text_pattern_distinct` | `pattern, minimum=3, js=''` | ``pattern`` matches at least ``minimum`` *different* strings. |
| `third_party_requests` | `allow=None` | The app fetched something off-machine. Passing means the penalty FIRES. |
| `value_changes` | `before='before', after='after', should_change=True, view='text'` | Two harness-taken snapshots differ (or deliberately do not). |
| `value_equals` | `js, expect, normalise='strip'` | Evaluate ``js`` in the page and compare its result to a constant. |
| `value_matches` | `js, pattern` | Evaluate ``js`` and match its result against a regular expression. |

### The `drag` tool

Not a check -- a grader tool, alongside `click`, `type_text` and `press_key`:

    drag(target, dx=, dy=, from_x=, from_y=, steps=16)

`from_x`/`from_y` are offsets *inside* the target's own box (default: its
centre). Real pointer events are emitted, so an app listening for `pointerdown`
sees a genuine gesture. This is the only way to interact with a `<canvas>`, and
`canvas_changed` / `canvas_ink` are the only way to observe the result -- text,
form state and computed style are all identical either side of a stroke.

## Traps the golden-set pass hit — read these

**`value_changes` and the `view` parameter.** A `capture` snapshot has three
views. `text` (the default) is `body.innerText` only. `full` also includes every
`input.value` and `select.value` — so comparing `full` across an action where the
grader *typed something* reports a change whether or not the app responded. That
would award a "search filters as you type" item to a build whose search filters
nothing. Use the default `text` view unless you specifically want form state.
For a theme switch, which changes no text at all, use `computed_style_changes`
(it reads the `style` view).

**Penalty polarity.** A penalty's check must return **yes** when the bad thing is
present. There are dedicated inverse primitives for this — do not try to negate a
positive one: `third_party_requests` (not `network_origins`),
`console_errors_present` (not `no_console_errors`), `download_absent` (not
`download`), `text_numbers_distinct` with `maximum: 1` (not `minimum: 2`),
`default_restored`, `model_dependency`, `client_secret_present`, and
`text_matches` with only `absent:` (fires when the named thing is missing).

**Do not assume a primitive works — check its signature.** Two real bugs in the
first pass came from passing a parameter a primitive does not accept
(`value_changes` has no `js`). The call becomes a `TypeError`, which is recorded
as `unknown`, which means **those points can never be earned on any build** and
nothing looks broken. Bind every parameter against the catalogue above.

**Prefer an absent check to a wrong one.** A missing check falls back to the
model, which is imperfect but visible. A wrong check silently misgrades all 40
builds of that idea in the same direction, and looks like a finding. If the
honest answer is "no primitive can decide this", leave it out and say so in a
`note:` beginning `NO CHECK —` plus what would be needed.


## What changed on 2026-08-28 — re-read if you authored before this

Four harness bugs were fixed. Each one changed what a correct check looks like,
and measurements taken before the fix are not trustworthy.

1. **`MAGIC` now knows `svg` and `webp`, and `magic:` accepts a list.**
   `magic: svg` used to return `unknown` -- unearnable points, silently. Use
   `magic: [png, jpeg, webp]` for an item whose brief accepts any image.
2. **Source reads are no longer truncated at 20 KB.** The grader now reads up to
   2 MB per file. This matters more than it sounds: 36 of 39 `handdrawn_whiteboard`
   builds ship a client file over 20 KB, and the largest is 1.8 MB, so every
   `source_absent` claim was previously being made about a fifth of a bundle.
   An absent-pattern check that "fires on 18 of 128 builds" was measuring the cap.
3. **`ctx.downloads` is now scoped to the item**, not the whole session. An item
   whose own export does nothing no longer passes on a previous item's file. You
   still want `extension:` or `magic:` when one item produces two files.
4. **The `style` capture view covers the largest visible blocks**, not just
   `body`/`:root`/`main`/`section`, so a build that repaints a scoped element --
   a code card, an editor pane -- is no longer falsely failed. And
   `computed_style_changes` now honours its `selector` parameter, which it
   previously accepted and ignored.

## Measure from the facts sheet, not from the corpus

`ideas/rubrics/_corpus_facts.json` holds a per-idea survey of the r3 slice --
what renders to canvas, what extensions downloads use, who sets accessible
names, who gates behind auth. Read your idea's sheet:

```
python scripts/rubric_corpus_facts.py --sheet <idea_id>
```

**Do not go and grep the corpus yourself.** The first authoring pass did, and it
was the single most expensive thing in it: one worker spent 21 hours, produced
nothing, and left its findings as prose no one could check. If the sheet lacks a
fact you need, say so in your report and fall back to `agent` for that item --
the fact then gets added to the sheet once, for everyone.

Note the sheet covers **r3 only**, which is the population the sweep grades.
Counts taken over all of `builds/work` mix in r2 and earlier eras and will not
match.
