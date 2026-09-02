#!/usr/bin/env bash
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

# Populate the vendored Playwright MCP (tooling/browser-mcp/node_modules) so the
# founder QA & Designer get a real browser using ONLY the system `node` at run
# time. Safe to re-run (no-op if already present). Uses the system `npm` if
# available; otherwise fetches a local Node (which bundles npm) into ./.node/.
#
# The browser tooling degrades gracefully if this has not been run
# (opencode_agents.browser_prereqs_ok() returns False and the browser is skipped),
# so this is an enablement step, not a hard build dependency.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f node_modules/@playwright/mcp/cli.js ]; then
  echo "browser-mcp: already vendored (node_modules present)."
  exit 0
fi

NPM="$(command -v npm || true)"
if [ -z "$NPM" ]; then
  NODE_VERSION="${NODE_VERSION:-v22.22.2}"
  case "$(uname -m)" in
    x86_64 | amd64) ARCH=x64 ;;
    aarch64 | arm64) ARCH=arm64 ;;
    *) echo "browser-mcp: unsupported arch $(uname -m); install Node/npm manually." >&2; exit 1 ;;
  esac
  NODE_DIR=".node/${NODE_VERSION}"
  if [ ! -x "${NODE_DIR}/bin/npm" ]; then
    echo "browser-mcp: no system npm; fetching Node ${NODE_VERSION} (bundles npm) ..."
    mkdir -p .node
    tarball="node-${NODE_VERSION}-linux-${ARCH}.tar.xz"
    curl -fsSL "https://nodejs.org/dist/${NODE_VERSION}/${tarball}" -o ".node/${tarball}"
    tar -xf ".node/${tarball}" -C .node
    mv ".node/node-${NODE_VERSION}-linux-${ARCH}" "${NODE_DIR}"
    rm -f ".node/${tarball}"
  fi
  export PATH="${PWD}/${NODE_DIR}/bin:${PATH}"
  NPM="${PWD}/${NODE_DIR}/bin/npm"
fi

echo "browser-mcp: installing @playwright/mcp (JS only; system Chrome used at run time) ..."
# Prefer a reproducible install from the lockfile; fall back to install.
PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 "$NPM" ci --omit=dev --no-audit --no-fund 2>/dev/null ||
  PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 "$NPM" install --omit=dev --no-audit --no-fund

if [ -f node_modules/@playwright/mcp/cli.js ]; then
  echo "browser-mcp: ready."
else
  echo "browser-mcp: FAILED to vendor @playwright/mcp." >&2
  exit 1
fi
