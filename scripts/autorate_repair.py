#!/usr/bin/env python
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

"""Re-run the autorater over stored crowd runs that have no ``autorating.json``.

WHY THIS IS NEEDED. The autorater's three dimensions carry 15% of the active
scoring profile (``substance``/``severity``/``word_of_mouth``, 0.05 each). A run
with no ``autorating.json`` is still scored, but that 15% is renormalised away --
so it is scored under a *different* profile than a run that has one.

This happens for real: when the crowd model's key hits sustained HTTP 429
``RESOURCE_EXHAUSTED``, the sweep's own in-line autorate pass gives up after its
retries and a slice of the runs comes out unrated. And -- this is the part that
matters -- that slice has been observed to land unevenly BY ARM rather than at
random. That is a systematic difference along the exact axis the sweep exists to
compare, so it cannot be left as noise.

The repair is safe and cheap because rating is a pure function of artifacts
already on disk: three LLM calls against the stored evidence pack, no simulation
re-run. It is also idempotent -- a run that already has a rating is skipped -- so
this can be run repeatedly as quota recovers.

Usage::

    scripts/autorate_repair.py --replicate 3            # repair, low concurrency
    scripts/autorate_repair.py --replicate 3 --status   # count only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))


def arm_of(key: str) -> str:
    """Arm name from a fleet index key (the default `team` arm omits it)."""
    parts = key.split("::")
    return parts[2] if len(parts) > 3 else "team"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicate", type=int, default=3)
    ap.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="rating calls in flight. Deliberately low: this exists because the "
        "key was rate-limited, and hammering it again reproduces the failure.",
    )
    ap.add_argument("--status", action="store_true", help="report, rate nothing")
    args = ap.parse_args()

    fleet = json.loads((REPO / "builds" / "fleet.json").read_text())
    suffix = f"::r{args.replicate}"
    build_arm = {
        v["build_id"]: arm_of(k)
        for k, v in fleet["entries"].items()
        if k.endswith(suffix) and v.get("build_id")
    }

    missing: list[Path] = []
    by_arm_missing: Counter[str] = Counter()
    by_arm_have: Counter[str] = Counter()
    for summary in (REPO / "builds" / "crowd").glob("*/run_summary.json"):
        run_dir = summary.parent
        build_id = run_dir.name.split("__crowd-")[0]
        arm = build_arm.get(build_id)
        if arm is None:
            continue
        if (run_dir / "autorating.json").is_file():
            by_arm_have[arm] += 1
        else:
            by_arm_missing[arm] += 1
            missing.append(run_dir)

    arms = sorted(set(by_arm_have) | set(by_arm_missing))
    print(f"replicate {args.replicate}: autorating coverage by arm")
    for arm in arms:
        have, miss = by_arm_have[arm], by_arm_missing[arm]
        total = have + miss
        pct = 100.0 * have / total if total else 0.0
        print(f"  {arm:8s} rated {have:4d}/{total:4d} ({pct:5.1f}%)  missing {miss}")
    print(f"  TOTAL missing: {len(missing)}")

    if args.status or not missing:
        return 0

    from crowd_sweep import autorate_missing  # noqa: PLC0415 - heavy import

    ok, total = autorate_missing(missing, concurrency=args.concurrency)
    print(f"repaired {ok}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
