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
# setup_crowd_env.sh: prepare a local machine for the crowd stage.
#
# The crowd has two halves:
#   1. app interaction (viral_bench.crowd.interaction), which drives founder-built apps
#      like a human (browser / CLI / bot). Runs in the MAIN env (Python 3.12).
#   2. the social simulation (viral_bench.crowd.sim), OASIS + CAMEL agents on a
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

case "$(uname -s)" in
  Darwin) HOST_OS=darwin ;;
  Linux)  HOST_OS=linux ;;
  *)      HOST_OS=other ;;
esac

ok()   { printf '  \033[0;32mOK\033[0m   %s\n' "$1"; }
warn() { printf '  \033[0;33mWARN\033[0m %s\n' "$1"; }
info() { printf '  ..   %s\n' "$1"; }

# Package-manager advice that matches the machine. Getting this wrong is not
# cosmetic: `sudo apt install chromium` on a Mac reads as "this tool was not
# meant for me", and the browser is what separates a real trial from a
# static-HTTP one that cannot run JavaScript.
pkg_hint() {
  case "$HOST_OS" in
    darwin) echo "brew install --cask $2" ;;
    *)      echo "sudo apt install -y $1" ;;
  esac
}

cd "$REPO_ROOT"
echo "== crowd environment setup =="
info "host: $(uname -s) $(uname -m)"

# 1. Base toolchain.
for tool in python3 uv git; do
  if command -v "$tool" >/dev/null 2>&1; then
    ok "$tool present ($($tool --version 2>&1 | head -1))"
  else
    warn "$tool NOT found on PATH"
  fi
done

# 2. A system browser for the ``chrome`` channel (no bundled-browser download).
#
# Asks the same code the crowd asks, rather than re-implementing a PATH scan
# here. A PATH scan is the whole story only on Linux: macOS installs Chrome as
# an app bundle and puts nothing on PATH, so this reported "no browser" on every
# stock Mac and the crowd silently fell back to a JavaScript-less fetch.
browser="$(python3 -c "
import sys
sys.path.insert(0, 'src')
from viral_bench.hostplatform import find_chrome
print(find_chrome() or '')
" 2>/dev/null || true)"
if [[ -n "$browser" ]]; then
  ok "system browser present ($browser)"
else
  warn "no system Chrome/Chromium found (web-app trials need it). Install one, e.g.:"
  warn "  $(pkg_hint chromium chromium)"
fi

# 3. Main env dependencies (the app-interaction layer + Playwright).
if $CHECK_ONLY; then
  info "skipping main-env 'uv sync' (--check)"
else
  info "installing main-env dependencies (uv sync) ..."
  uv sync >/dev/null 2>&1 && ok "main-env deps installed (incl. Playwright)" \
    || warn "'uv sync' failed. Run it manually to see the error"
fi

# 4. Confirm a browser can be driven (main env).
if uv run python -c "from viral_bench.crowd.interaction import browser_available; import sys; sys.exit(0 if browser_available() else 1)" >/dev/null 2>&1; then
  ok "Playwright can drive a browser (single-page-app trials at full fidelity)"
else
  warn "no driveable browser yet. Web trials will run in degraded static-HTTP mode"
fi

# 5. podman (the crowd runs untrusted apps in a rootless container by default).
if command -v podman >/dev/null 2>&1; then
  ok "podman present ($(podman --version 2>&1))"
  if [[ "$HOST_OS" == "darwin" ]] \
     && ! podman machine list --format '{{.Running}}' 2>/dev/null | grep -qi true; then
    warn "podman machine is not running. Start one with room for builds:"
    warn "  podman machine init --cpus 4 --memory 8192 && podman machine start"
  fi
elif command -v docker >/dev/null 2>&1; then
  ok "docker present ($(docker --version 2>&1)); usable, though podman is preferred"
else
  warn "no container runtime (needed for the container run mode). Install one:"
  case "$HOST_OS" in
    darwin) warn "  brew install podman && podman machine init --memory 8192 && podman machine start" ;;
    *)      warn "  sudo apt install -y podman" ;;
  esac
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
      || warn "crowd requirements install failed. Run it manually to see the error"
  fi
fi

# A broken crowd env has to be loud here, because everything below is gated on
# the engine importing and would otherwise be skipped in silence -- which is how
# an unpinned mcp 2.x once let this script print "done" and exit 0 over an env
# where every crowd run died on the first import. `import oasis` alone does not
# reach camel, so the deep import is the one worth gating on.
if [[ -x "$CROWD_PY" ]] && ! "$CROWD_PY" -c "from oasis.environment.env import OasisEnv" >/dev/null 2>&1; then
  warn "the crowd engine does not import. The error in full:"
  "$CROWD_PY" -c "from oasis.environment.env import OasisEnv" 2>&1 | tail -3 | sed 's/^/       /'
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

# 8. Crowd model reachability, against whatever `viral-bench init` configured.
# Naming a Gemini model here would contradict the whole point of shipping no
# default provider, and would report a failure to anyone holding a key for one of
# the other thirteen.
if [[ -x "$CROWD_PY" ]] && "$CROWD_PY" -c "import oasis" >/dev/null 2>&1; then
  model="${VIRAL_BENCH_CROWD_MODEL:-$(PYTHONPATH="$REPO_ROOT/src" "$CROWD_PY" -c \
    "from viral_bench.config import stage_model; print(stage_model('crowd'))" 2>/dev/null)}"
  if [[ -z "$model" ]]; then
    warn "no crowd model configured, so it cannot be checked. Run: viral-bench init"
  elif PYTHONPATH="$REPO_ROOT/src" "$CROWD_PY" - "$model" <<'PY' >/dev/null 2>&1
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
    warn "crowd model '$model' not callable (no key, or model unavailable)."
    warn "  Check the provider's key variable is set: viral-bench models"
  fi
fi

echo "== done =="
echo "Next: uv run viral-bench crowd-run <build_id>            # full crowd simulation"
echo "      uv run viral-bench crowd-run <build_id> --no-llm   # cheap wiring smoke"
