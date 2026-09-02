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

#
# setup_crowd_env.sh — prepare a local machine for the crowd stage.
#
# The crowd has two halves:
#   1. app interaction (viral_bench.crowd.interaction) — drive founder-built apps
#      like a human (browser / CLI / bot). Runs in the MAIN env (Python 3.12).
#   2. the social simulation (viral_bench.crowd.sim) — OASIS + CAMEL agents on a
#      mock social platform. OASIS needs Python <3.12 and heavy pinned deps, so it
#      runs in a DEDICATED, isolated Python 3.11 venv (.venv-crowd).
#
# This script sets up both, and is safe to re-run (idempotent). It never runs
# `sudo` itself: steps that need root print the exact command for you to run.
#
# Usage:
#   scripts/setup_crowd_env.sh          # check + install what it can
#   scripts/setup_crowd_env.sh --check  # report status only, install nothing
#
# What it does:
#   - verifies python3 / uv / a system Chrome are present
#   - main env: `uv sync` + confirms Playwright can drive a browser
#   - crowd env: creates .venv-crowd (3.11), installs OASIS, verifies `import oasis`
#   - prefetches the TWHIN-BERT recommender model (large, one-time)
#   - verifies the crowd Gemini model is callable (needs a key)
#   - checks podman (the crowd runs apps in a rootless container by default)

set -euo pipefail

CHECK_ONLY=false
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=true

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CROWD_VENV="$REPO_ROOT/.venv-crowd"
CROWD_PY="$CROWD_VENV/bin/python"

ok()   { printf '  \033[0;32mOK\033[0m   %s\n' "$1"; }
warn() { printf '  \033[0;33mWARN\033[0m %s\n' "$1"; }
info() { printf '  ..   %s\n' "$1"; }

cd "$REPO_ROOT"
echo "== crowd environment setup =="

# 1. Base toolchain.
for tool in python3 uv git; do
  if command -v "$tool" >/dev/null 2>&1; then
    ok "$tool present ($($tool --version 2>&1 | head -1))"
  else
    warn "$tool NOT found on PATH"
  fi
done

# 2. A system browser for the ``chrome`` channel (no bundled-browser download).
browser=""
for b in google-chrome google-chrome-stable chromium chromium-browser chrome microsoft-edge; do
  if command -v "$b" >/dev/null 2>&1; then browser="$b"; break; fi
done
if [[ -n "$browser" ]]; then
  ok "system browser present ($browser)"
else
  warn "no system Chrome/Chromium found (web-app trials need it). Install one, e.g.:"
  warn "  sudo apt install -y chromium"
fi

# 3. Main env dependencies (the app-interaction layer + Playwright).
if $CHECK_ONLY; then
  info "skipping main-env 'uv sync' (--check)"
else
  info "installing main-env dependencies (uv sync) ..."
  uv sync >/dev/null 2>&1 && ok "main-env deps installed (incl. Playwright)" \
    || warn "'uv sync' failed — run it manually to see the error"
fi

# 4. Confirm a browser can actually be driven (main env).
if uv run python -c "from viral_bench.crowd.interaction import browser_available; import sys; sys.exit(0 if browser_available() else 1)" >/dev/null 2>&1; then
  ok "Playwright can drive a browser (single-page-app trials at full fidelity)"
else
  warn "no driveable browser yet — web trials will run in degraded static-HTTP mode"
fi

# 5. podman (the crowd runs untrusted apps in a rootless container by default).
if command -v podman >/dev/null 2>&1; then
  ok "podman present ($(podman --version 2>&1))"
else
  warn "podman NOT installed (needed for the container run mode). sudo apt install -y podman"
fi

# 6. The isolated crowd simulation env (.venv-crowd, Python 3.11 + OASIS).
if [[ -x "$CROWD_PY" ]]; then
  ok "crowd venv present ($("$CROWD_PY" --version 2>&1))"
elif $CHECK_ONLY; then
  warn "crowd venv missing (re-run without --check to create .venv-crowd)"
else
  info "creating crowd venv (.venv-crowd, Python 3.11) ..."
  uv venv --python 3.11 "$CROWD_VENV" >/dev/null 2>&1 \
    && ok "created $CROWD_VENV" \
    || warn "could not create the crowd venv (is Python 3.11 available to uv?)"
fi

if [[ -x "$CROWD_PY" ]]; then
  if "$CROWD_PY" -c "import oasis" >/dev/null 2>&1; then
    ok "OASIS installed in the crowd venv"
  elif $CHECK_ONLY; then
    warn "OASIS not installed in the crowd venv (re-run without --check)"
  else
    info "installing OASIS into the crowd venv (pulls torch; may take a while) ..."
    uv pip install --python "$CROWD_VENV" -r config/crowd-requirements.txt >/dev/null 2>&1 \
      && ok "installed crowd requirements (camel-oasis, playwright, google-genai)" \
      || warn "crowd requirements install failed — run it manually to see the error"
  fi
fi

# 7. Verify the crowd can import the sim + prefetch the TWHIN-BERT recommender.
if [[ -x "$CROWD_PY" ]] && "$CROWD_PY" -c "import oasis" >/dev/null 2>&1; then
  if PYTHONPATH="$REPO_ROOT/src" "$CROWD_PY" -c "import viral_bench.crowd.sim.simulation" >/dev/null 2>&1; then
    ok "crowd simulation modules import in the crowd venv"
  else
    warn "crowd simulation modules failed to import in the crowd venv"
  fi

  if "$CROWD_PY" - <<'PY' >/dev/null 2>&1
from huggingface_hub import try_to_load_from_cache
import sys
hit = try_to_load_from_cache("Twitter/twhin-bert-base", "config.json")
sys.exit(0 if isinstance(hit, str) else 1)
PY
  then
    ok "TWHIN-BERT recommender model already cached"
  elif $CHECK_ONLY; then
    warn "TWHIN-BERT model not cached (re-run without --check to prefetch, ~1GB)"
  else
    info "prefetching TWHIN-BERT (Twitter/twhin-bert-base; large one-time download) ..."
    "$CROWD_PY" -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('Twitter/twhin-bert-base'); AutoModel.from_pretrained('Twitter/twhin-bert-base')" >/dev/null 2>&1 \
      && ok "TWHIN-BERT prefetched" \
      || warn "TWHIN-BERT prefetch failed (network?). The lighter '--recsys twitter' needs no download."
  fi
fi

# 8. Crowd Gemini model reachability (needs a key; GEMINI_API_KEY_CROWD or GEMINI_API_KEY).
if [[ -x "$CROWD_PY" ]] && "$CROWD_PY" -c "import oasis" >/dev/null 2>&1; then
  model="${VIRAL_BENCH_CROWD_MODEL:-gemini-2.0-flash}"
  if PYTHONPATH="$REPO_ROOT/src" "$CROWD_PY" - "$model" <<'PY' >/dev/null 2>&1
import sys
from viral_bench.crowd.sim.model import crowd_model
from camel.agents import ChatAgent
agent = ChatAgent(system_message="Reply with one word.", model=crowd_model(sys.argv[1], temperature=0))
r = agent.step("Say ok.")
sys.exit(0 if r.msgs else 1)
PY
  then
    ok "crowd model '$model' is callable on your key"
  else
    warn "crowd model '$model' not callable (no key, or model unavailable). Set"
    warn "  GEMINI_API_KEY_CROWD (or GEMINI_API_KEY) in .env; try --model gemini-2.0-flash"
  fi
fi

echo "== done =="
echo "Next: uv run viral-bench crowd-run <build_id>            # full crowd simulation"
echo "      uv run viral-bench crowd-run <build_id> --no-llm   # cheap wiring smoke"
