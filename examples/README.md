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

Everything in that folder — `index.html`, `app.js`, `style.css`, plus the
agent's own `DESIGN.md`, `README.md`, and the `viralbench.json` run/test
manifest — was written by the AI founder agent from the one-line idea spec; it
was not hand-edited.

### View it

```bash
cd examples/founder_builds/sliding_tile_game
python3 -m http.server 8000
# then open http://localhost:8000/
```

Play with arrow keys / WASD (or swipe): matching tiles merge, score updates,
best score persists, and there's undo plus custom themes.
