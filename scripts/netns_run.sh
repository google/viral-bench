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

# Run a command in a private network namespace with outbound internet.
#
# WHY: a founder team build starts the app it is writing so the Designer and QA
# can drive it in a browser. The app's port comes from the model-authored
# manifest, and the example manifest says 8000 -- so N concurrent builds fight
# over one port on the host loopback. That is not merely a crash: build B's QA can
# bind-fail, connect to 8000 anyway, and review build A's app. A harness race
# that hits one model more than the other reads out as a capability difference
#.
#
# The same applies to PROCESSES. Agents clean up after themselves with blunt
# instruments; observed verbatim in the first six builds of this fleet:
#
#     pkill -f "server.py" || true
#     lsof -i :8000 | awk '{print $2}' | xargs kill -9
#     kill -9 4018262 || true          # a raw host pid
#
# With a shared PID namespace every one of those reaches across the whole host,
# so one build's tidy-up kills another build's app server mid-review. Two builds
# died together at the same instant in the pilot. A private PID namespace bounds
# the blast radius to the build that issued the command, which also makes the
# failure attributable instead of anonymous.
#
# HOW: unshare a user+network+PID namespace, then have pasta(1) configure the
# network from outside. Each build gets its own 127.0.0.1 and its own /proc, so
# port 8000 means 50 different things at once and `pkill` means one build.
#
#   --map-current-user   keep the caller's uid. `pasta -- cmd` would map us to
#                        root, and Chrome refuses to run as root without
#                        --no-sandbox, which would silently disable the
#                        Designer/QA browser for the whole fleet.
#   --pid --fork         private PID namespace; --mount-proc gives it a matching
#                        /proc so ps/pkill/lsof see only this build.
#   -t/-u/-T none        no port forwarding in either direction: full loopback
#                        isolation. (-U is left on; pasta needs it for DNS.)
#
# Usage: scripts/netns_run.sh <command> [args...]
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

for tool in unshare pasta; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "netns_run: required tool '$tool' not found" >&2
    exit 127
  }
done

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
readyfile="$tmp/ready"
gofifo="$tmp/go"
mkfifo "$gofifo"

# The command waits on the fifo so it does not start before pasta has configured
# the interface. `unshare` itself joins the new user+net namespace before forking
# for the PID namespace, so $! is the handle pasta attaches to. The inner
# process must NOT be asked for its own pid, which inside a PID namespace is 1.
READYFILE="$readyfile" GOFIFO="$gofifo" unshare \
  --user --map-current-user --net --pid --fork --mount-proc -- \
  bash -c 'echo ready >"$READYFILE"; read -r _ <"$GOFIFO"; exec "$@"' bash "$@" &
child=$!

for _ in $(seq 1 400); do
  [ -s "$readyfile" ] && break
  sleep 0.05
done
if [ ! -s "$readyfile" ]; then
  echo "netns_run: namespace never came up" >&2
  kill "$child" 2>/dev/null || true
  exit 1
fi

if ! pasta --config-net -t none -u none -T none -q "$child"; then
  echo "netns_run: pasta failed to configure the namespace" >&2
  kill "$child" 2>/dev/null || true
  exit 1
fi

echo go >"$gofifo"
wait "$child"
