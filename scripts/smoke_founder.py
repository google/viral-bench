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

"""End-to-end smoke test for the founder pipeline (Stage 2).

Runs the real founder harness (opencode against your configured provider) on one
idea, then reports the build outcome and how to test the app. This makes real,
billable model calls and needs opencode installed and a provider credential
configured, so it is a manual script, not a pytest.

Run:
    uv run python scripts/smoke_founder.py                    # sliding_tile_game
    uv run python scripts/smoke_founder.py quick_notes_app    # a different idea
    uv run python scripts/smoke_founder.py --check            # env preflight only
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from viral_bench.founder.collab import TOOLSETS
from viral_bench.founder.harness import find_opencode_binary
from viral_bench.founder.runner import describe_build


def preflight(model: str) -> bool:
    """Check opencode is installed and ``model`` is callable on Vertex.

    Pings the model the run will use, not a hardcoded Gemini id: the old version
    proved ``gemini-2.0-flash`` worked and then went on to launch a build
    against a Claude model the project had never enabled.
    """
    from viral_bench.founder.harness import HarnessError, preflight_vertex
    from viral_bench.founder.models import UnknownModelError, to_opencode_model
    from viral_bench.founder.vertex import vertex_location, vertex_project

    ok = True
    if find_opencode_binary():
        print(f"opencode: {find_opencode_binary()}")
    else:
        print("opencode: NOT FOUND (install with `npm i -g opencode-ai`)")
        ok = False
    try:
        resolved = to_opencode_model(model)
        preflight_vertex(resolved)
        print(f"Vertex ({vertex_project()}, {vertex_location()}): {resolved} callable")
    except (HarnessError, UnknownModelError) as exc:
        print(f"Vertex: NOT REACHABLE\n  {exc}")
        ok = False
    return ok


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("idea_id", nargs="?", default="sliding_tile_game")
    parser.add_argument(
        "--model",
        default="google-vertex/gemini-2.0-flash",
        help="curated short id, or any '<provider>/<model>' Vertex model garden "
        "model (e.g. google-vertex-anthropic/gemini-2.0-flash)",
    )
    parser.add_argument(
        "--agents",
        choices=("1", "4", "dynamic"),
        default="1",
        help="1 = solo founder, 4 = round-table specialist team, dynamic = one "
        "founder agent that chooses and orchestrates its own subagents",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="max orchestrator turns for --agents dynamic (default: 3)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="max collaboration rounds for the 4-agent team (default: 3)",
    )
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=1,
        help="min rounds before QA may ship, for the 4-agent team (default: 1)",
    )
    parser.add_argument(
        "--collab",
        choices=list(TOOLSETS),
        default="local",
        help="the medium the team collaborates through ('local', the only "
        "built-in, is the shared working directory: DESIGN.md and TEAM_NOTES.md "
        "alongside the code)",
    )
    parser.add_argument(
        "--no-browser-tools",
        dest="browser_tools",
        action="store_false",
        help="disable the team's Designer/QA browser (on by default; auto-disabled "
        "if the host lacks the prerequisites)",
    )
    parser.add_argument(
        "--check", action="store_true", help="only run the environment preflight"
    )
    args = parser.parse_args()

    print("=== preflight ===")
    ready = preflight(args.model)
    if args.check:
        return 0 if ready else 1
    if not ready:
        print("\nPreflight failed; fix the above and retry.", file=sys.stderr)
        return 1

    # Import here so --check works even before the harness deps are ready.
    from viral_bench.founder.build import run_build

    shape = (
        f"turns<={args.turns}"
        if args.agents == "dynamic"
        else f"rounds={args.min_rounds}-{args.rounds}"
    )
    print(
        f"\n=== founding {args.idea_id!r} with {args.model} "
        f"(agents={args.agents}, {shape}, collab={args.collab}) ==="
    )
    record = run_build(
        args.idea_id,
        model=args.model,
        agents=args.agents,
        collab=args.collab,
        rounds=args.rounds,
        min_rounds=args.min_rounds,
        turns=args.turns,
        browser_tools=args.browser_tools,
    )

    print(f"\nBuild:   {record.build_id}")
    print(f"Status:  {record.status}")
    print(f"Harness: {'ok' if record.harness_ok else 'FAILED'}")
    for phase in record.phases:
        print(
            f"  - {phase['phase']}: rc={phase['returncode']} ({phase['duration_s']}s)"
        )
    if record.orchestration:
        orch = record.orchestration
        print(
            f"Subagents: {record.subagents_spawned} "
            f"{orch.get('subagent_types') or {}}; peak "
            f"{orch.get('peak_concurrent_subagents', 0)} concurrent; defined "
            f"{orch.get('self_authored_agents') or []}"
        )
    if record.manifest_error:
        print(f"Manifest error: {record.manifest_error}")
    print()

    if record.status == "ok":
        print(describe_build(record.build_id))
        print(f"\nServe it with: uv run viral-bench serve-build {record.build_id}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
