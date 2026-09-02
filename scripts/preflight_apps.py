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

"""Can every shipped build be started the way the crowd starts it?

WHY THIS EXISTS. On an early sweep, dozens of builds never produced a single
scored run, and nothing on disk said why. A build whose app cannot start writes no
``run_summary.json``, so ``existing_runs()`` cannot tell "tried and died" from "never
tried". The cell was re-offered on every pass, held a slot for the full wall clock, and
leaked one ~188k-file app clone per restart attempt (15-34 of them per build). The
reason survived only inside the disposable clone, which is deleted.

The fix for that is elsewhere. What this script provides is the EVIDENCE: it exercises
the exact path the crowd exercises -- ``git clone --branch <shipped_ref>`` of the
shipped orphan branch, dependency provisioning, container start, real HTTP probe --
and records one durable verdict per build in ``builds/app_preflight.json``.

Run it BEFORE and AFTER a harness change to prove the change did something. Run it
before a sweep to warm the per-build dependency cache, so no crowd slot ever pays an
install and no crowd run is the first thing to discover an app is unstartable.

The verdict carries a REASON CLASS, not merely a boolean, because "the harness
packaging dropped the dependencies" and "the app segfaults" are the same symptom
and opposite attributions -- the first is a harness bug that must be excluded
from a model's score, the second is a real model failure that belongs at the
broken-app floor.

Usage::

    scripts/preflight_apps.py --replicate 3                  # a whole replicate
    scripts/preflight_apps.py --replicate 3 --no-provision   # baseline, pre-fix
    scripts/preflight_apps.py --builds a,b,c --verbose       # a named set
    scripts/preflight_apps.py --status                       # read the last report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.founder.manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    ManifestError,
    load_manifest,
)
from viral_bench.founder.runner import (  # noqa: E402
    RunnerError,
    open_session,
    reclaim_trash,
)
from viral_bench.founder.runtime import (  # noqa: E402
    HARNESS_CLASSES,
    classify_start_failure,
)
from viral_bench.score.fleet import (  # noqa: E402
    fleet_replicates,
    load_builds,
    structure_name,
)

BUILDS = REPO / "builds"
DEFAULT_REPORT = BUILDS / "app_preflight.json"
#: Where this invocation writes. Overridable so a before/after pair can be kept side
#: by side -- the whole value of this script is the delta between two passes, and a
#: single fixed path makes the "before" unreadable the moment the "after" runs.
REPORT = DEFAULT_REPORT


@dataclass
class Verdict:
    """One build's answer to "can the crowd start this?"."""

    build_id: str
    idea_id: str = ""
    model: str = ""
    structure: str = ""
    ok: bool = False
    reason_class: str = ""
    reason: str = ""
    #: What provisioning installed, if anything (empty when --no-provision).
    provisioned: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    #: How readiness was established, e.g. "HTTP 200".
    ready_detail: str = ""
    checked_at: str = ""

    @property
    def harness_fault(self) -> bool:
        return not self.ok and self.reason_class in HARNESS_CLASSES


def _provision(build_id: str, timeout: float, image: str | None) -> list[str]:
    """Warm the build's dependency cache, tolerating an older checkout.

    Imported lazily and guarded so this script still runs -- and still produces a
    baseline -- on a tree where provisioning does not exist yet. That is the whole
    point of a before/after measurement: the "before" side must not require the fix.
    """
    try:
        from viral_bench.founder.provision import provision_build
    except ImportError:
        return []
    return list(provision_build(build_id, timeout=timeout, image=image))


def _repair(
    build_id: str, error_text: str, timeout: float, image: str | None
) -> list[str]:
    """Install what the app's own error says is missing. Tolerates an old tree."""
    try:
        from viral_bench.founder.provision import repair_build
    except ImportError:
        return []
    from viral_bench.founder.runner import materialize_build, retire_dir

    run_dir = materialize_build(build_id)
    try:
        return list(
            repair_build(
                build_id,
                run_dir / "app",
                error_text,
                timeout=timeout,
                image=image,
            )
        )
    finally:
        retire_dir(run_dir)


def _attempt(
    build_id: str,
    verdict: Verdict,
    *,
    container: bool,
    start_wait: float,
    setup_timeout: float,
    image: str | None,
) -> None:
    """One materialize-start-probe cycle, writing its outcome into ``verdict``."""
    session = None
    verdict.ok = False
    verdict.reason = ""
    verdict.reason_class = ""
    try:
        session = open_session(build_id, container=container, image=image)
        session.setup(timeout=setup_timeout)
        app = session.start(wait_timeout=start_wait)
        verdict.ready_detail = app.ready_detail or ""
        # A web app that exposes no URL is not startable however healthy its process
        # is: the crowd drives apps through a browser, and there is nothing to open.
        verdict.ok = bool(app.url)
        if not verdict.ok:
            verdict.reason = f"started but exposed no URL ({verdict.ready_detail})"
            verdict.reason_class = "no_url"
    except (RunnerError, OSError, ValueError) as exc:
        verdict.reason = str(exc)[-4000:]
        verdict.reason_class = classify_start_failure(verdict.reason)
    except Exception as exc:  # noqa: BLE001 - one bad build must not stop the pass
        verdict.reason = f"{type(exc).__name__}: {exc}"[-4000:]
        verdict.reason_class = "harness_error"
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - teardown must not mask the verdict
                pass


def check_build(
    build_id: str,
    *,
    provision: bool,
    container: bool,
    start_wait: float,
    setup_timeout: float,
    image: str | None = None,
    repair_rounds: int = 2,
) -> Verdict:
    """Materialize, provision, start and probe one build. Never raises.

    On a dependency failure it does not merely record the verdict: it reads what
    the app said was missing, installs that into the build's cache, and tries
    again. A static plan is a guess and will never be complete -- an app reaches
    jinja2 through ``Jinja2Templates`` and python-multipart through ``Form(...)``
    without naming either -- whereas the app's own error names them exactly. Two
    rounds is enough to converge in practice, and each one is paid once per build
    rather than once per run.
    """
    verdict = Verdict(build_id=build_id, checked_at=datetime.now(UTC).isoformat())
    started = time.time()
    try:
        app_dir = BUILDS / "work" / build_id / "app"
        try:
            load_manifest(app_dir / MANIFEST_FILENAME)
        except (ManifestError, OSError) as exc:
            verdict.reason = f"no usable {MANIFEST_FILENAME}: {exc}"
            verdict.reason_class = "no_manifest"
            return verdict

        if provision:
            verdict.provisioned = _provision(build_id, setup_timeout, image)

        _attempt(
            build_id,
            verdict,
            container=container,
            start_wait=start_wait,
            setup_timeout=setup_timeout,
            image=image,
        )
        rounds = repair_rounds if provision else 0
        for _ in range(rounds):
            if verdict.ok:
                break
            installed = _repair(build_id, verdict.reason, setup_timeout, image)
            if not installed:
                # The error names nothing installable, so the failure is the
                # app's own and another attempt would return the same verdict.
                break
            verdict.provisioned.extend(installed)
            _attempt(
                build_id,
                verdict,
                container=container,
                start_wait=start_wait,
                setup_timeout=setup_timeout,
                image=image,
            )
    except Exception as exc:  # noqa: BLE001 - one bad build must not stop the pass
        verdict.reason = f"{type(exc).__name__}: {exc}"[-4000:]
        verdict.reason_class = "harness_error"
    finally:
        verdict.elapsed_s = round(time.time() - started, 1)
    return verdict


def select_builds(replicate: int, structures: set[str], ideas: set[str]) -> list:
    """Fleet builds at one replicate, optionally narrowed by arm or idea."""
    builds = load_builds(BUILDS)
    replicates = fleet_replicates(BUILDS)
    out = []
    for build_id, rep in sorted(replicates.items()):
        if rep != replicate or build_id not in builds:
            continue
        build = builds[build_id]
        arm = structure_name(build.config)
        if structures and arm not in structures:
            continue
        if ideas and build.idea_id not in ideas:
            continue
        out.append(build)
    return out


def load_report() -> dict[str, dict]:
    try:
        data = json.loads(REPORT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = data.get("builds")
    return entries if isinstance(entries, dict) else {}


_WRITE_LOCK = threading.Lock()


def save_report(entries: dict[str, dict], meta: dict) -> None:
    """Persist atomically, under a lock: the pass writes as it goes.

    Written incrementally rather than once at the end because this pass takes hours
    over 1,000 builds and has to survive being interrupted -- an OOM took the whole
    cgroup down seven times during the last sweep, and a report that only exists after
    the final build is a report that never exists.
    """
    with _WRITE_LOCK:
        payload = {**meta, "builds": entries}
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        tmp = REPORT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(REPORT)


def current_fleet_only(entries: dict[str, dict]) -> tuple[dict[str, dict], int]:
    """Entries the fleet index still points at, plus how many were superseded.

    The report is an accumulating log -- written incrementally so an interrupted
    pass is not lost, and nothing ever removes an entry. The RATE is a claim about
    the corpus as it stands, and the two diverge the moment a cell is rebuilt:
    replacing screenshot_to_code/opus-5 left its predecessor in the file and the
    pass reported 929 of 1001, a denominator that does not exist over a numerator
    counting a build no sweep will ever score.

    An entry with no fleet cell at all -- a control, an ad-hoc probe -- is KEPT.
    It was never part of the count, and dropping it would hide a result somebody
    asked for deliberately. Only a build whose idea the fleet still holds, but
    which the index no longer names, has been superseded.
    """
    indexed = set(fleet_replicates(BUILDS))
    if not indexed:
        return entries, 0
    ideas = {b.rsplit("__", 2)[0] for b in indexed}
    kept, dropped = {}, 0
    for build_id, row in entries.items():
        if build_id not in indexed and build_id.rsplit("__", 2)[0] in ideas:
            dropped += 1
            continue
        kept[build_id] = row
    return kept, dropped


def summarize(entries: dict[str, dict]) -> str:
    """A short report: start rate overall, per arm, and per reason class."""
    if not entries:
        return "no preflight results yet"
    entries, superseded = current_fleet_only(entries)
    rows = list(entries.values())
    ok = sum(1 for r in rows if r.get("ok"))
    note = f"   [{superseded} superseded ignored]" if superseded else ""
    lines = [
        f"preflight: {ok}/{len(rows)} builds start "
        f"({100.0 * ok / len(rows):.1f}%){note}",
        "",
        f"{'arm':10} {'ok':>5} {'fail':>5} {'rate':>7}",
    ]
    arms: dict[str, list[dict]] = {}
    for row in rows:
        arms.setdefault(row.get("structure", "?"), []).append(row)
    for arm in sorted(arms):
        group = arms[arm]
        good = sum(1 for r in group if r.get("ok"))
        lines.append(
            f"{arm:10} {good:5} {len(group) - good:5} {100.0 * good / len(group):6.1f}%"
        )
    classes: dict[str, int] = {}
    for row in rows:
        if not row.get("ok"):
            classes[row.get("reason_class") or "unknown"] = (
                classes.get(row.get("reason_class") or "unknown", 0) + 1
            )
    if classes:
        lines += ["", "failures by class (harness-attributable marked *):"]
        for name, count in sorted(classes.items(), key=lambda kv: -kv[1]):
            mark = "*" if name in HARNESS_CLASSES else " "
            lines.append(f"  {mark} {name:22} {count}")
        blame = sum(c for n, c in classes.items() if n in HARNESS_CLASSES)
        lines.append(f"  -> {blame} of {sum(classes.values())} failures are ours")
    return "\n".join(lines)


def compare(before_path: Path, after_path: Path) -> str:
    """Two passes side by side: what the harness change was worth, in builds.

    A pass on its own says how many apps start, and only a pair says whether that is
    the fix working or the corpus being easy. Reported per reason class as well
    as in total, because "63 harness-attributable failures became 4" is the claim
    that can be checked, while a single percentage is the claim that cannot.
    """

    def _load(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("builds", {})
        except (OSError, json.JSONDecodeError):
            return {}

    before, _ = current_fleet_only(_load(before_path))
    after, _ = current_fleet_only(_load(after_path))
    shared = sorted(set(before) & set(after))
    if not shared:
        return f"no builds in common between {before_path} and {after_path}"

    b_ok = sum(1 for k in shared if before[k].get("ok"))
    a_ok = sum(1 for k in shared if after[k].get("ok"))
    fixed = [k for k in shared if not before[k].get("ok") and after[k].get("ok")]
    broke = [k for k in shared if before[k].get("ok") and not after[k].get("ok")]

    # Named, not silently dropped. A build rebuilt since the baseline was taken
    # has no "before" to compare against, and quietly shrinking the denominator
    # to hide that is how a corpus of 1,000 starts reporting itself as 999.
    no_baseline = sorted(set(after) - set(before))
    a_ok_all = sum(1 for k in after if after[k].get("ok"))

    lines = [
        f"current fleet: {a_ok_all}/{len(after)} start "
        f"({100 * a_ok_all / len(after):.1f}%)",
        "",
        f"comparable (present in both): {len(shared)}",
        f"  before : {b_ok}/{len(shared)} start ({100 * b_ok / len(shared):.1f}%)",
        f"  after  : {a_ok}/{len(shared)} start ({100 * a_ok / len(shared):.1f}%)",
        f"  fixed  : {len(fixed)}",
        f"  broken : {len(broke)}" + (" <-- REGRESSION" if broke else ""),
    ]
    if no_baseline:
        lines.append(
            f"  no baseline: {len(no_baseline)} rebuilt since it was taken -- "
            + ", ".join(b[:44] for b in no_baseline[:3])
        )
    lines += [
        "",
        f"{'reason class':24} {'before':>7} {'after':>7} {'delta':>7}",
    ]
    classes = sorted(
        {r.get("reason_class") or "unknown" for r in before.values() if not r.get("ok")}
        | {
            r.get("reason_class") or "unknown"
            for r in after.values()
            if not r.get("ok")
        }
    )
    for name in classes:
        nb = sum(
            1
            for k in shared
            if not before[k].get("ok") and (before[k].get("reason_class") or "") == name
        )
        na = sum(
            1
            for k in shared
            if not after[k].get("ok") and (after[k].get("reason_class") or "") == name
        )
        mark = "*" if name in HARNESS_CLASSES else " "
        lines.append(f"{mark}{name:23} {nb:7} {na:7} {na - nb:+7}")
    if broke:
        lines += ["", "REGRESSIONS (started before, not after):"]
        lines += [f"  {k}: {after[k].get('reason_class')}" for k in broke[:20]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate", type=int, default=3)
    parser.add_argument("--structures", default="", help="solo,team,dynamic")
    parser.add_argument("--ideas", default="")
    parser.add_argument("--builds", default="", help="explicit build ids, comma-sep")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="parallel builds (default: %(default)s). NOT bound by the crowd's "
        "ceiling of 10, and the difference is worth stating because confusing "
        "the two could crash the box: that ceiling exists because one crowd run "
        "peaks at ~180 Chrome processes, and preflight starts no browser at all "
        "-- one memory-capped container per slot. 16 measured at load 15 on 64 "
        "cores with 87 GB free and memory PSI 0.03. Still check "
        "/proc/pressure/memory before raising it, never load average, which "
        "reads backwards on this box.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--no-provision",
        action="store_true",
        help="skip dependency provisioning -- the pre-fix baseline",
    )
    parser.add_argument(
        "--recheck",
        action="store_true",
        help="re-check builds already in the report (default: skip them)",
    )
    parser.add_argument("--start-wait", type=float, default=90.0)
    parser.add_argument("--setup-timeout", type=float, default=600.0)
    parser.add_argument(
        "--host", action="store_true", help="run on the host, not in a container"
    )
    parser.add_argument(
        "--image",
        default="",
        help="container image to run apps in; empty means the runtime default. "
        "Pin it when measuring a before/after, so the only thing that moved is "
        "the thing under test.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_REPORT),
        help="where to write the report (default: %(default)s)",
    )
    parser.add_argument(
        "--status", action="store_true", help="print the report, run nothing"
    )
    parser.add_argument(
        "--compare",
        default="",
        help="print a before/after diff against this earlier report and exit. "
        "A single pass says how many apps start; only a pair says whether that "
        "is the fix working.",
    )
    args = parser.parse_args(argv)

    if args.compare:
        print(compare(Path(args.compare), Path(args.out)))
        return 0

    global REPORT
    REPORT = Path(args.out)
    # A baseline must not silently inherit a cache an earlier pass populated.
    # Skipping provisioning is NOT enough: `open_session` mounts whatever cache
    # exists for the build, so a "before" pass reports the fixed behaviour and
    # the measurement quietly destroys itself. Observed exactly that -- known-bad
    # builds returning HTTP 200 in the pass meant to reproduce their failure.
    if args.no_provision:
        os.environ["VIRALBENCH_DEPCACHE"] = "0"
    entries = load_report()
    if args.status:
        print(summarize(entries))
        return 0

    if args.builds:
        wanted_ids = [b.strip() for b in args.builds.split(",") if b.strip()]
        all_builds = load_builds(BUILDS)
        targets = [all_builds[b] for b in wanted_ids if b in all_builds]
        missing = [b for b in wanted_ids if b not in all_builds]
        if missing:
            print(f"unknown build ids: {', '.join(missing)}", file=sys.stderr)
    else:
        targets = select_builds(
            args.replicate,
            {s.strip() for s in args.structures.split(",") if s.strip()},
            {i.strip() for i in args.ideas.split(",") if i.strip()},
        )

    if not args.recheck:
        targets = [b for b in targets if b.build_id not in entries]
    if args.limit:
        targets = targets[: args.limit]

    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "provisioned": not args.no_provision,
        "replicate": args.replicate,
        "container": not args.host,
        "image": args.image or "default",
    }
    print(
        f"preflight: {len(targets)} builds to check at concurrency "
        f"{args.concurrency} (provision={not args.no_provision})",
        flush=True,
    )
    if not targets:
        print(summarize(entries))
        return 0

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                check_build,
                b.build_id,
                provision=not args.no_provision,
                container=not args.host,
                start_wait=args.start_wait,
                setup_timeout=args.setup_timeout,
                image=args.image or None,
            ): b
            for b in targets
        }
        for future in as_completed(futures):
            build = futures[future]
            verdict = future.result()
            verdict.idea_id = build.idea_id
            verdict.model = build.model
            verdict.structure = structure_name(build.config)
            entries[verdict.build_id] = asdict(verdict)
            done += 1
            failed += 0 if verdict.ok else 1
            if verdict.ok:
                note = verdict.ready_detail
            else:
                note = f"{verdict.reason_class}: {verdict.reason.splitlines()[0][:90]}"
            print(
                f"[{done}/{len(targets)}] {'ok  ' if verdict.ok else 'FAIL'} "
                f"{verdict.build_id[:52]:<52} {verdict.elapsed_s:6.1f}s {note}",
                flush=True,
            )
            if done % 10 == 0:
                save_report(entries, meta)
            # Each check materializes a clone, and retiring is O(1), but the bytes still
            # have to go back. Same time-budgeted reclaim the crowd sweep uses.
            if done % 25 == 0:
                reclaim_trash(budget_s=15.0)

    save_report(entries, meta)
    print()
    print(summarize(entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
