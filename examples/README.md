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

# Example founder builds

This directory holds a **small, hand-picked set of example apps produced by the
ViralBench founder pipeline** (Stage 2), committed so others can see what the
"founder" agent actually generates.

> Note on scope: live builds are generated under the top-level `builds/`
> directory, which is **gitignored** — those throwaway build artifacts are never
> committed. Only curated examples are copied here by hand.

## `founder_builds/sliding_tile_game/` — "TileMerge"

A 2048-style sliding-tile puzzle that runs entirely in the browser (static
files, no build step, offline-capable).

| | |
|---|---|
| Idea spec | [`ideas/sliding_tile_game.yaml`](../ideas/sliding_tile_game.yaml) |
| Founder agent | opencode, single agent (Design → Build) |
| Model | `google/gemini-2.0-flash` |
| Built | 2026-07-07 |
| Command | `uv run viral-bench found sliding_tile_game --model google/gemini-2.0-flash` |

Everything in that folder (`index.html`, `app.js`, `style.css`, plus the agent's
own `DESIGN.md`, `README.md`, and the `viralbench.json` run/test manifest) was
written by the founder agent from the one-line idea spec. The only human change
is the Apache license header prepended to each source file, which this
repository requires on every file. No line of the agent's own output was
altered.

### View it

```bash
cd examples/founder_builds/sliding_tile_game
python3 -m http.server 8000
# then open http://localhost:8000/
```

Play with arrow keys / WASD (or swipe): matching tiles merge, score updates,
best score persists, and there's undo plus custom themes.
