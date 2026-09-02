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
# setup_founder_env.sh — prepare a local machine to run the founder harness.
#
# The founder pipeline (viral_bench.founder) drives the open-source `opencode`
# coding agent, backed by whichever model provider you configured, to design +
# build apps in a sandbox. This script codifies the one-time environment setup and
# is safe to re-run (idempotent). It never runs `sudo` itself: steps that need root
# print the exact command for you to run.
#
# Usage:
#   scripts/setup_founder_env.sh          # check + install what it can
#   scripts/setup_founder_env.sh --check  # report status only, install nothing
#
# What it does:
#   - verifies node / python3 / uv are present
#   - installs `opencode` to ~/.opencode/bin (direct release binary; no curl|bash)
#   - checks podman + rootless subuid/subgid ranges (prints fix if missing)
#   - reports whether a model provider is configured (see `viral-bench init`)

set -euo pipefail

CHECK_ONLY=false
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=true

OPENCODE_DIR="$HOME/.opencode/bin"
ok()   { printf '  \033[0;32mOK\033[0m   %s\n' "$1"; }
warn() { printf '  \033[0;33mWARN\033[0m %s\n' "$1"; }
info() { printf '  ..   %s\n' "$1"; }

echo "== founder harness environment setup =="

# 1. Base toolchain.
#
# npm is checked separately below because it is the one that is routinely
# missing: Debian and its derivatives ship `nodejs` without it, so `node` is
# present and `npm` is not. The build prompt promises the founder "Node.js and
# npm are available", so a missing npm makes every JS/TS build fail in a way that
# looks like a model failure rather than a setup gap.
for tool in node python3 uv git; do
  if command -v "$tool" >/dev/null 2>&1; then
    ok "$tool present ($($tool --version 2>&1 | head -1))"
  else
    warn "$tool NOT found on PATH"
  fi
done

if command -v npm >/dev/null 2>&1; then
  ok "npm present ($(npm --version 2>&1 | head -1))"
elif command -v corepack >/dev/null 2>&1; then
  warn "npm NOT found — install it without sudo via:
       corepack enable npm --install-directory \"\$HOME/.local/bin\""
else
  warn "npm NOT found on PATH (and no corepack to install it) — JS/TS builds will fail"
fi

# 2. opencode (the founder harness agent).
if command -v opencode >/dev/null 2>&1 || [[ -x "$OPENCODE_DIR/opencode" ]]; then
  bin="$(command -v opencode || echo "$OPENCODE_DIR/opencode")"
  ok "opencode present ($("$bin" --version 2>&1 | head -1)) at $bin"
elif $CHECK_ONLY; then
  warn "opencode NOT installed (re-run without --check to install)"
else
  info "installing opencode -> $OPENCODE_DIR ..."
  mkdir -p "$OPENCODE_DIR"
  if grep -qwi avx2 /proc/cpuinfo 2>/dev/null; then variant="linux-x64"; else variant="linux-x64-baseline"; fi
  url="https://github.com/anomalyco/opencode/releases/latest/download/opencode-${variant}.tar.gz"
  tmp="$(mktemp -d)"
  if curl -fsSL -o "$tmp/opencode.tar.gz" "$url"; then
    tar -xzf "$tmp/opencode.tar.gz" -C "$tmp"
    mv -f "$tmp/opencode" "$OPENCODE_DIR/opencode"
    chmod 755 "$OPENCODE_DIR/opencode"
    rm -rf "$tmp"
    ok "opencode installed ($("$OPENCODE_DIR/opencode" --version 2>&1 | head -1))"
    case ":$PATH:" in
      *":$OPENCODE_DIR:"*) ;;
      *) warn "add to PATH: export PATH=\"$OPENCODE_DIR:\$PATH\"" ;;
    esac
  else
    rm -rf "$tmp"
    warn "opencode download failed (network/egress?). Install manually from"
    warn "  https://github.com/anomalyco/opencode/releases"
  fi
fi

# 3. podman (rootless OCI sandbox for running untrusted generated apps).
if command -v podman >/dev/null 2>&1; then
  ok "podman present ($(podman --version 2>&1))"
  if grep -q "^$(id -un):" /etc/subuid 2>/dev/null; then
    ok "rootless subuid/subgid range configured"
  else
    warn "podman rootless needs a subuid/subgid range. Fix with:"
    warn "  sudo usermod --add-subuids 600000-665535 --add-subgids 600000-665535 $(id -un)"
    warn "  podman system migrate"
  fi
else
  warn "podman NOT installed. Install it (on its own, not with npm):"
  warn "  sudo apt install -y podman"
fi

# 4. Model provider credentials.
#
# There is no default provider. The founder runs against whichever one you
# configure, and Vertex AI is a single entry in the registry
# (src/viral_bench/providers/spec.py) alongside OpenAI, Anthropic, the Gemini
# API, OpenRouter, a local Ollama and any OpenAI-compatible endpoint. Most take
# an API key from .env; the Vertex entries use ambient cloud credentials instead,
# and a local Ollama needs none at all. So this reports what is configured rather
# than gating on any one provider being set up.
#
# The "ambient" rows are not probed -- confirming a cloud credential costs a
# network round trip, which belongs in `viral-bench doctor`, not here.
if providers="$(uv run viral-bench models 2>/dev/null)"; then
  ok "provider registry loads; credential state:"
  printf '%s\n' "$providers" | sed -n '/^PROVIDER/,/^$/p' | sed '/^[[:space:]]*$/d; s/^/       /'
  info "nothing configured yet? pick a provider and write .env: uv run viral-bench init"
else
  warn "'viral-bench models' failed — is the main env installed? Run: uv sync"
fi
info "confirm a specific model is really callable, and check the rest of the setup:"
info "  uv run viral-bench models --check <provider>/<model>"
info "  uv run viral-bench doctor"

echo "== done =="
echo "Next: uv run viral-bench found <idea_id>   (see README: Founder pipeline)"
