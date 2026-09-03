#!/usr/bin/env python3
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

"""Report how each model chose to orchestrate its dynamic-mode builds.

The dynamic founder arm (``--agents dynamic``) is the only one where the shape of
the founding team is an OUTCOME rather than a setting, so it is the only one
where "what did the model do?" needs answering separately from "how good is the
app?". This reads the orchestration recorded on every dynamic ``build.json`` and
answers, per model:

* did it delegate at all, and how much,
* did it run subagents concurrently or one at a time,
* did it keep subagents alive (resuming a ``task_id``) or spawn throwaways,
* did it write its own subagent definitions,
* how many orchestrator turns it needed, and whether it declared completion.

Why this is worth a script rather than a glance at the JSON: a model that never
delegates is a legitimate result, but it is indistinguishable at the score level
from a model that delegated well, and both are indistinguishable from a harness
bug that hid the ``task`` tool. Only the spawn record separates the three, and
the answer decides whether a sweep of this arm measured anything at all.

Usage::

    scripts/orchestration_report.py                 # every dynamic build
    scripts/orchestration_report.py --json          # machine-readable
    scripts/orchestration_report.py --idea quick_notes_app
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.score.fleet import load_builds  # noqa: E402


def _short(model: str) -> str:
    return model.rsplit("/", 1)[-1]


def collect(builds_root: Path, idea: str = "") -> list[dict]:
    """Flatten every dynamic build's orchestration into one row each."""
    rows: list[dict] = []
    for build in load_builds(builds_root).values():
        if build.config.get("structure") != "dynamic":
            continue
        if idea and build.idea_id != idea:
            continue
        orch = build.orchestration or {}
        spawns = orch.get("spawns") or []
        rows.append(
            {
                "build_id": build.build_id,
                "idea_id": build.idea_id,
                "model": _short(build.model),
                "status": build.status,
                "seconds": build.build_seconds,
                "turns": orch.get("orchestrator_turns", build.rounds_run),
                "done_signalled": bool(orch.get("done_signalled")),
                "spawned": int(orch.get("subagents_spawned", 0) or 0),
                "types": orch.get("subagent_types") or {},
                "peak": int(orch.get("peak_concurrent_subagents", 0) or 0),
                "resumed": int(orch.get("resumed_subagents", 0) or 0),
                "failed_spawns": int(orch.get("failed_spawns", 0) or 0),
                "authored": orch.get("self_authored_agents") or [],
                # A subagent that spawned its own subagents is invisible in the
                # parent transcript, but a child whose parent is not the main
                # session is not: nested delegation shows up as more than one
                # distinct parent id.
                "parents": len({s.get("parent_session_id") for s in spawns if s}),
            }
        )
    return sorted(rows, key=lambda r: (r["model"], r["idea_id"]))


def summarize(rows: list[dict]) -> str:
    if not rows:
        return "no dynamic builds on disk yet."
    out: list[str] = [f"dynamic-arm orchestration: {len(rows)} builds"]
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    out.append("")
    out.append(
        f"  {'model':<26} {'builds':>6} {'ok':>3} {'deleg':>6} {'spawns':>7} "
        f"{'peak':>5} {'resum':>6} {'turns':>6} {'authored':>9} {'median s':>9}"
    )
    for model, group in sorted(by_model.items()):
        ok = sum(1 for r in group if r["status"] == "ok")
        delegating = sum(1 for r in group if r["spawned"] > 0)
        spawns = statistics.mean(r["spawned"] for r in group)
        peak = max(r["peak"] for r in group)
        resumed = sum(r["resumed"] for r in group)
        turns = statistics.mean(r["turns"] or 0 for r in group)
        authored = sum(len(r["authored"]) for r in group)
        secs = [r["seconds"] for r in group if r["seconds"]]
        out.append(
            f"  {model:<26} {len(group):>6} {ok:>3} "
            f"{delegating:>3}/{len(group):<2} {spawns:>7.1f} {peak:>5} "
            f"{resumed:>6} {turns:>6.1f} {authored:>9} "
            f"{(statistics.median(secs) if secs else 0):>9.0f}"
        )

    out.append("")
    out.append("  per build:")
    for row in rows:
        types = ",".join(f"{k}x{v}" for k, v in sorted(row["types"].items())) or "-"
        flags = []
        if not row["done_signalled"]:
            flags.append("no-done-signal")
        if row["failed_spawns"]:
            flags.append(f"{row['failed_spawns']} failed spawns")
        if row["authored"]:
            flags.append("defined:" + ",".join(row["authored"]))
        if row["parents"] > 1:
            flags.append("nested delegation")
        out.append(
            f"    {row['idea_id']:<20} {row['model']:<24} {row['status']:<10} "
            f"turns={row['turns']} spawned={row['spawned']:<3} peak={row['peak']} "
            f"[{types}]" + (f"  {'; '.join(flags)}" if flags else "")
        )

    # The one finding that would invalidate a sweep of this arm.
    silent = [m for m, g in by_model.items() if all(r["spawned"] == 0 for r in g)]
    out.append("")
    if silent:
        out.append(
            f"  NOTE: {', '.join(sorted(silent))} never delegated in any build. "
            "That is a legitimate result -- but confirm from a transcript that "
            "the model DECIDED to work alone, rather than the task tool being "
            "unavailable, before reporting it as one."
        )
    else:
        out.append("  every model delegated at least once.")
    kinds = Counter(k for r in rows for k in r["types"])
    out.append(f"  subagent types used: {dict(kinds) or 'none'}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", default="", help="only this idea_id")
    parser.add_argument("--json", action="store_true", help="dump rows as JSON")
    args = parser.parse_args(argv)

    rows = collect(REPO / "builds", args.idea)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
