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

# Idea Bench

This directory is the **frozen test set** for ViralBench (Stage 1 of the
[design doc](https://docs.google.com/document/d/172bt3iLzebgJfdOp3FdQL7j9sINjP28In8oSluyaVPg/edit?tab=t.7cqgjw96k1pa)).

Each file describes **one** vibe-coding idea, normalized into a structured spec
so that every model under test receives an identical brief.

- **One idea per file:** `ideas/<slug>.yaml`
- **Validated in CI:** every `*.yaml` file here is checked against the schema by
  `tests/test_ideas.py`. A malformed idea will fail the build.
- **Every `*.yaml` directly in this directory is a live benchmark idea.** The
  loader globs `ideas/*.yaml` non-recursively, so anything dropped here is part
  of the test set. Scaffolding that is *not* an idea (e.g. the template) lives
  in [`templates/`](templates/), which is excluded.

## Schema

| Field | Required | Type | Notes |
|---|---|---|---|
| `idea_id` | yes | string | `snake_case`, unique across all idea files |
| `title` | yes | string | Short product name |
| `pitch` | yes | string | One-line pitch |
| `problem` | yes | string | The problem being solved |
| `target_user` | yes | string | Who it's for |
| `core_features` | yes | list of strings | Must-have features (>= 1) |
| `success_criteria` | yes | string | What "done & good" looks like |
| `allowed_scope` | yes | enum | `client-app` \| `full-stack-app` (see below) |
| `difficulty` | yes | enum | `easy` \| `medium` \| `hard` |
| `ground_truth` | no | object | Only for ideas drawn from real viral apps |

### `allowed_scope`

Every idea is a **web app**. The scope says how much of it lives on the server:

| Value | Means | State lives in |
|---|---|---|
| `client-app` | Runs in the browser. Any server is a static file server, or a thin same-origin proxy that hides an API key. | `localStorage` / IndexedDB |
| `full-stack-app` | A real backend with a database, accounts, and **multi-user** behaviour: what one visitor writes, another can see. | SQLite under `/data` |

Prefer `full-stack-app` only when the multi-user flow is *the point* — one user
acts and another sees it. The crowd shares a single running instance per build,
so that flow is genuinely exercised; an idea that merely *could* have a login is
better off as a `client-app`.

The corpus is deliberately weighted ~18 `client-app` to ~7 `full-stack-app`,
because the apps that actually go viral skew heavily client-side (Excalidraw,
Carbon, it-tools, 2048 all went viral *because* they need no account). Keep both
scopes spanning easy/medium/hard, so difficulty is not confounded with scope.

### Runtime constraints an idea must respect

Apps are built by an agent and run in a container capped at **1 GB RAM / 1.0 CPU**
with no GPU. An idea is only usable if a competent agent could build a working
version under those limits. Concretely, avoid:

- **Local model weights** of any kind (no local diffusion, LLM, or ASR). A hosted
  model via `GEMINI_API_KEY` is fine; the app must still start without a key.
- **Any second credential** — no OAuth provider, managed database, payment
  processor, or email service. The benchmark must run with only a model API key.
- **Anything the crowd cannot perceive.** Agents read the rendered DOM and ARIA
  tree, and can see screenshots; they cannot use a mouse to draw. Prefer ideas
  whose success is visible as text or structure.

### `ground_truth` (optional block)

Include this **only** when the idea is based on a real app that actually went
viral. It is used later to validate the simulation against reality.

| Field | Required | Type | Notes |
|---|---|---|---|
| `source` | yes | string | Where it went viral (e.g. "Twitter/X", "Product Hunt") |
| `metric` | yes | string | The real-world outcome measured (e.g. `github_stars`) |
| `value` | yes | number | The real number at launch/reference time |

**Use real numbers only.** This block is the reference data the simulation is
scored against, so an invented `value` silently corrupts that analysis. Also add
a header comment naming the reference repo and the date you checked it, as the
existing specs do:

```yaml
# Reference app (ground truth): ollama — https://github.com/ollama/ollama
# Verified 175,191 GitHub stars on 2026-06-30.
```

If the idea has no real reference app, omit `ground_truth` entirely.

## Adding a new idea

1. Copy `templates/idea_template.yaml` to `ideas/<your_slug>.yaml`.
2. Fill in every required field; give it a unique `idea_id`.
3. Validate locally:
   ```bash
   uv run pytest tests/test_ideas.py
   ```
4. Open a Pull Request (see [../CONTRIBUTING.md](../CONTRIBUTING.md)).

Aim for **diversity** across `allowed_scope` and `difficulty`, and keep ideas
grounded in real vibe-coding / viral examples where possible.
