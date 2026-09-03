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

"""Serve several built apps at once, each on a fixed port, and hold them up.

The viewer can launch an app on a free port from its own UI, which is right for
poking at one build. This is for the other case: three arms' builds of the same
brief, side by side on ports that are tunnelled somewhere, staying up so someone
can compare them by hand.

    viz/serve_apps.py --arm solo=8001 --arm team=8002 --arm dynamic=8003
    viz/serve_apps.py --build <build_id>=8005
    viz/serve_apps.py --from-armruns 8001      # every arm that built, in order

``--arm`` resolves a name against the build ids recorded under ``cache/armruns/``,
so the ports stay stable across rebuilds. Each app is copied out of its build tree
first and fronted by the same no-cache proxy the viewer uses, because otherwise
three builds of one brief -- all serving ``/app.js`` from port 8000 -- would show
each other's JavaScript.

Ctrl-C stops every app and deletes the throwaway copies.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.apphost import AppLauncher  # noqa: E402
from core.paths import default_builds_root, founder_paths, read_json  # noqa: E402

ARMRUNS = Path(__file__).resolve().parent / "cache" / "armruns"
#: The order arms are assigned ports in by ``--from-armruns``.
ARM_ORDER = ("solo", "team", "dynamic", "solo_claude", "dynamic_claude")


def arm_build_id(arm: str) -> str:
    path = ARMRUNS / f"{arm}.build_id"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=PORT",
        help="serve the build this arm produced on PORT",
    )
    parser.add_argument(
        "--build",
        action="append",
        default=[],
        metavar="BUILD_ID=PORT",
        help="serve an explicit build id on PORT",
    )
    parser.add_argument(
        "--from-armruns",
        type=int,
        default=None,
        metavar="FIRST_PORT",
        help="serve every arm that produced a build, numbering up from FIRST_PORT",
    )
    parser.add_argument("--builds-root", type=Path, default=None)
    args = parser.parse_args(argv)

    builds_root = (args.builds_root or default_builds_root()).resolve()

    wanted: list[tuple[str, str, int]] = []  # (label, build_id, port)
    for spec in args.arm:
        name, _, port = spec.partition("=")
        build_id = arm_build_id(name)
        if not build_id:
            print(f"no build recorded for arm {name!r}", file=sys.stderr)
            continue
        wanted.append((name, build_id, int(port)))
    for spec in args.build:
        build_id, _, port = spec.partition("=")
        wanted.append((build_id[:24], build_id, int(port)))
    if args.from_armruns is not None:
        port = args.from_armruns
        for name in ARM_ORDER:
            build_id = arm_build_id(name)
            if build_id:
                wanted.append((name, build_id, port))
                port += 1

    if not wanted:
        parser.error("nothing to serve")

    launcher = AppLauncher(builds_root)
    print(f"builds  {builds_root}\n")

    started: list[tuple[str, str, int]] = []
    for label, build_id, port in wanted:
        manifest = read_json(
            founder_paths(builds_root, build_id).app_dir / "viralbench.json"
        )
        title = (manifest or {}).get("title") or build_id
        print(f"  {label:14} :{port}  {title}  ({build_id})")
        result = launcher.launch(build_id, port=port)
        if result.get("ok"):
            print(f"  {'':14} -> {result['url']}  [{result['command']}]")
            started.append((label, result["url"], port))
        else:
            print(f"  {'':14} -> FAILED: {result.get('error')}")
            for line in (result.get("setup_log") or [])[-6:]:
                print(f"  {'':16} {line}")
        print()

    if not started:
        print("nothing came up", file=sys.stderr)
        return 1

    print("running:")
    for label, url, _port in started:
        print(f"  {label:14} {url}")
    print("\nCtrl-C to stop them all.")

    done = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: done.set())
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    try:
        done.wait()
    finally:
        print("\nstopping...")
        launcher.stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
