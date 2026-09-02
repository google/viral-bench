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

"""Name a set of builds, so a result can say which corpus it covers.

WHY. A sweep has no name of its own. "The builds from that run" is
reconstructible only as "fleet replicate N, these arm names, brief fingerprint
current" -- three facts that have to be restated, and re-verified, every time
anyone asks what a number covered. It also cannot survive the obvious next step:
a follow-up sweep that KEEPS some arms and REBUILDS others spans two build eras,
so no replicate number describes it.

So a cohort is a label, applied to build records and carried into the crowd runs
made over them:

* ``build.json`` gains ``"cohort": "<name>"``.
* ``run_summary.json`` gains the same, copied automatically at run time -- the
  same contract ``crowd_arch_version`` and ``crowd_transport`` already have.
  Record what produced a number, at the time, because it cannot be recovered
  afterwards.
* ``builds/cohorts/<name>.json`` lists the members, so one file answers "which
  builds?" without anyone re-deriving it.

A cohort is NOT a selection axis. Sweeps still select with
``--fleet-structure`` / ``--fleet-replicate``, which already work. Adding a
parallel way to choose builds would mean two answers to the same question. This
only ever *names* what those flags already select.

Usage::

    # tag the arms being kept, now
    scripts/cohort.py tag r4 --structures solo,dynamic --replicate 3

    # after team is rebuilt, tag it into the same cohort
    scripts/cohort.py tag r4 --structures team --replicate 3

    scripts/cohort.py status r4          # membership, per arm, plus any gaps
    scripts/cohort.py ls                 # every cohort on disk
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.score.fleet import (  # noqa: E402
    fleet_replicates,
    load_builds,
    structure_name,
)

BUILDS = REPO / "builds"
COHORTS = BUILDS / "cohorts"

#: Arms a full sweep is expected to carry, in reporting order.
ARMS = ("solo", "team", "dynamic")


def manifest_path(name: str) -> Path:
    return COHORTS / f"{name}.json"


def load_manifest(name: str) -> dict:
    try:
        return json.loads(manifest_path(name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def select(replicate: int, structures: set[str]) -> list:
    """Fleet-indexed builds at one replicate, narrowed to the named arms."""
    builds = load_builds(BUILDS)
    out = []
    for build_id, rep in sorted(fleet_replicates(BUILDS).items()):
        if rep != replicate or build_id not in builds:
            continue
        build = builds[build_id]
        if structures and structure_name(build.config) not in structures:
            continue
        out.append(build)
    return out


def stamp(build_id: str, name: str) -> bool:
    """Write the cohort into one ``build.json``. True if it changed.

    Rewrites the file in place rather than through ``BuildRecord``, so a record
    carrying a field this checkout does not know about is preserved instead of
    being silently dropped on the round trip.
    """
    path = BUILDS / "work" / build_id / "build.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("cohort") == name:
        return False
    data["cohort"] = name
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    return True


def write_manifest(name: str, members: dict[str, str], criteria: list[dict]) -> None:
    """Persist the member list. Merges, because a cohort is filled in stages.

    A cohort may tag solo and dynamic in one pass and team whenever it finishes
    rebuilding, so a manifest that overwrote would lose whichever half was
    written first.
    """
    existing = load_manifest(name)
    merged = {**(existing.get("members") or {}), **members}
    # Drop members the fleet index no longer points at. Merging is what lets a
    # cohort be filled in stages, but it would also preserve a build that has
    # since been REPLACED: rebuilding screenshot_to_code/opus-5 gave that cell a
    # new build id, and without this the cohort would claim both, so the arm
    # would read 251 of 250 and the superseded build would be swept and scored.
    indexed = set(fleet_replicates(BUILDS))
    stale = [b for b in merged if b not in indexed]
    for build_id in stale:
        del merged[build_id]
    if stale:
        print(f"  dropped {len(stale)} superseded member(s): {', '.join(stale[:4])}")
    seen = {json.dumps(c, sort_keys=True) for c in existing.get("criteria") or []}
    all_criteria = list(existing.get("criteria") or [])
    for entry in criteria:
        if json.dumps(entry, sort_keys=True) not in seen:
            all_criteria.append(entry)
    payload = {
        "cohort": name,
        "created_at": existing.get("created_at") or datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "note": (
            "A LABEL for a set of builds, not a replicate number and not a way to "
            "select them. Sweeps still select with --fleet-structure and "
            "--fleet-replicate; this records which builds those flags covered."
        ),
        "criteria": all_criteria,
        "count": len(merged),
        "by_arm": _by_arm(merged),
        "members": dict(sorted(merged.items())),
    }
    COHORTS.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path(name).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(manifest_path(name))


def _by_arm(members: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for arm in members.values():
        counts[arm] = counts.get(arm, 0) + 1
    return dict(sorted(counts.items()))


def cmd_tag(args: argparse.Namespace) -> int:
    structures = {s.strip() for s in args.structures.split(",") if s.strip()}
    selected = select(args.replicate, structures)
    if not selected:
        print(
            f"nothing matched replicate {args.replicate} "
            f"structures={sorted(structures) or 'any'}",
            file=sys.stderr,
        )
        return 1

    members = {b.build_id: structure_name(b.config) for b in selected}
    changed = 0
    if not args.dry_run:
        changed = sum(stamp(bid, args.name) for bid in members)
        write_manifest(
            args.name,
            members,
            [
                {
                    "replicate": args.replicate,
                    "structures": sorted(structures) or list(ARMS),
                    "tagged_at": datetime.now(UTC).isoformat(),
                }
            ],
        )
    verb = "would tag" if args.dry_run else "tagged"
    print(f"{verb} {len(members)} builds as cohort {args.name!r}")
    for arm, n in _by_arm(members).items():
        print(f"  {arm:9} {n}")
    if not args.dry_run:
        print(f"  ({changed} build.json files updated, rest already correct)")
        # Not `relative_to(REPO)`: builds can live outside the checkout via
        # VIRAL_BENCH_BUILDS_DIR, and relative_to raises rather than falling back.
        path = manifest_path(args.name)
        try:
            shown = path.relative_to(REPO)
        except ValueError:
            shown = path
        print(f"  manifest: {shown}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    data = load_manifest(args.name)
    if not data:
        print(f"no cohort {args.name!r} on disk", file=sys.stderr)
        return 1
    members = data.get("members") or {}
    print(
        f"cohort {args.name}: {len(members)} builds (updated {data.get('updated_at')})"
    )
    by_arm = _by_arm(members)
    for arm in ARMS:
        n = by_arm.get(arm, 0)
        flag = (
            "" if n == args.expect_per_arm else f"  <-- expected {args.expect_per_arm}"
        )
        print(f"  {arm:9} {n}{flag}")
    extra = {a: n for a, n in by_arm.items() if a not in ARMS}
    if extra:
        print(f"  unexpected arms: {extra}")

    # A manifest can drift from the build records: a build could be re-tagged, or
    # rebuilt under a new id. Check both directions rather than trusting the file.
    missing_stamp, gone = [], []
    for build_id in members:
        path = BUILDS / "work" / build_id / "build.json"
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            gone.append(build_id)
            continue
        if rec.get("cohort") != args.name:
            missing_stamp.append(build_id)
    if gone:
        print(f"  {len(gone)} member(s) have no readable build.json: {gone[:5]}")
    if missing_stamp:
        print(f"  {len(missing_stamp)} member(s) not stamped: {missing_stamp[:5]}")
    if not gone and not missing_stamp:
        print("  every member is stamped and readable")
    return 0


def cmd_ls(_args: argparse.Namespace) -> int:
    if not COHORTS.is_dir():
        print("no cohorts yet")
        return 0
    for path in sorted(COHORTS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"{data.get('cohort', path.stem):12} {data.get('count', 0):5} builds  "
            f"{data.get('by_arm')}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    tag = sub.add_parser("tag", help="stamp builds into a cohort")
    tag.add_argument("name")
    tag.add_argument("--replicate", type=int, default=3)
    tag.add_argument("--structures", default="", help="solo,team,dynamic")
    tag.add_argument("--dry-run", action="store_true")
    tag.set_defaults(func=cmd_tag)

    status = sub.add_parser("status", help="membership and drift")
    status.add_argument("name")
    status.add_argument("--expect-per-arm", type=int, default=250)
    status.set_defaults(func=cmd_status)

    sub.add_parser("ls", help="list cohorts").set_defaults(func=cmd_ls)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
