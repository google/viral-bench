# Vendored Playwright MCP (browser tooling for the founder team)

This directory provides the [Playwright MCP](https://github.com/microsoft/playwright-mcp)
server so the founder team's **QA & Finisher** and **UX & Virality Designer** can
render and click the app they are building, using only the system `node` at run
time (no `npx`/`npm` once vendored).

## Setup (one-time per machine)
Only `package.json` + `package-lock.json` are committed. Populate `node_modules/`
(gitignored, ~13 MB) with:

```sh
tooling/browser-mcp/bootstrap.sh
```

It uses the system `npm` if present, otherwise fetches a pinned local Node (which
bundles npm) into `.node/`. If you skip this, the browser tooling simply stays
disabled -- `opencode_agents.browser_prereqs_ok()` returns False, so the team
builds without a browser (the Designer/QA reason from markup) and nothing fails.

## How it's wired
- `viral_bench.founder.opencode_agents.DEFAULT_BROWSER_COMMAND` launches it as:
  `node .../@playwright/mcp/cli.js --headless --browser chrome --isolated`.
- It drives the **system Chrome channel** (`google-chrome`), so there is no need
  to match Playwright's bundled-browser revision. The tools appear to the agents
  as `browser_*`.
- `opencode_agents.browser_prereqs_ok()` enables it only when `node`, this
  `cli.js`, and a system Chrome/Chromium are all present; otherwise the harness
  auto-disables the browser (a build never fails just because it is missing).
- On by default for the team; disable per-run with `viral-bench found ... --no-browser-tools`.

## Refreshing / bumping the version
Edit the pinned version in `package.json`, then re-run `bootstrap.sh` (or, with a
system npm, `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install --omit=dev`). Commit
the updated `package.json` + `package-lock.json` only -- `node_modules/` is
gitignored. Browser binaries are never vendored here; the system Chrome is used.
