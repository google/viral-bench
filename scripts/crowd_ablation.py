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

"""Run one deliberate instrument variation over a fixed subset, and compare it.

A sweep tells you what the instrument says. An ablation tells you *which part of
the instrument is saying it* -- and that is the only honest way to settle a
question like "should every agent try the app?", because the alternative is
changing five things at once and attributing the result to whichever one you
happened to believe in.

Three properties make this a comparison rather than a vibe:

* **Same builds, same seeds.** A variant runs over exactly the subset the
  baseline ran over, so the difference cannot be a difference of fleet.
* **Its own architecture tag.** Every run records ``<version>+<variant>``, so an
  ablation can never pool into the sweep it exists to inform. This is enforced
  by the corpus loader's existing architecture filter, not by a convention.
* **Judged on discrimination, not on the gap.** The statistic reported is how
  far apart the instrument puts *different builds* relative to how far apart it
  puts *the same build on different seeds*. A variant that widens the model gap
  by making every number noisier has not improved anything, and this is the
  number that says so.

Usage::

    scripts/crowd_ablation.py --variant triers8 --triers 8 --builds 8 --seeds 2
    scripts/crowd_ablation.py --report            # compare everything on disk
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.crowd.sim_defaults import CROWD_ARCH_VERSION  # noqa: E402
from viral_bench.score.fleet import (  # noqa: E402
    CURRENT_FLEET,
    FleetCorpus,
    fleet_replicates,
    load_builds,
    load_corpus,
    paired_gap,
    self_separation,
    within_cell_sd,
)

BUILDS = REPO / "builds"
ABLATION_DIR = BUILDS / "ablation"
LOG_DIR = BUILDS / "ablation_logs"

#: The negative control travels with every arm. A variant that stops
#: distinguishing a corpse from a product has failed regardless of what it does
#: to the model gap.
CONTROL_BUILD = "quick_notes_app__20260728-000000__brokn0"


def fleet_builds() -> list:
    """The builds under test, ordered by idea so a subset is model-balanced."""
    builds = load_builds(BUILDS)
    reps = fleet_replicates(BUILDS)
    for bid, rep in reps.items():
        if bid in builds:
            builds[bid] = replace(builds[bid], replicate=rep)
    mine = [b for bid, b in builds.items() if bid in reps and CURRENT_FLEET.wants(b)]
    return sorted(mine, key=lambda b: (b.idea_id, b.model))


def subset(n_ideas: int) -> list:
    """``n_ideas`` ideas, both models each -- a paired subset, never a slice.

    Slicing the build list would take whole ideas of one model and none of the
    other whenever the count is odd, and a paired statistic over unpaired cells
    is not a paired statistic.
    """
    builds = fleet_builds()
    ideas = sorted({b.idea_id for b in builds})[:n_ideas]
    return [b for b in builds if b.idea_id in ideas]


def run_one(build_id: str, seed: int, opts: argparse.Namespace) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = ABLATION_DIR / f"{build_id}__{opts.variant}-{stamp}-s{seed}"
    argv = [
        "uv",
        "run",
        "viral-bench",
        "crowd-run",
        build_id,
        "--agents",
        str(opts.agents),
        "--triers",
        str(opts.triers),
        "--latecomers",
        str(opts.latecomers),
        "--rounds",
        str(opts.rounds),
        "--seed",
        str(seed),
        "--variant",
        opts.variant,
        "--out",
        str(out),
    ]
    if opts.model:
        argv += ["--model", opts.model]
    if opts.temperature is not None:
        argv += ["--temperature", str(opts.temperature)]
    if opts.recsys:
        argv += ["--recsys", opts.recsys]
    if opts.thinking_level:
        argv += ["--thinking-level", opts.thinking_level]
    if opts.semaphore:
        argv += ["--semaphore", str(opts.semaphore)]
    if opts.trial_max_steps:
        argv += ["--trial-max-steps", str(opts.trial_max_steps)]
    if opts.min_interactions is not None:
        argv += ["--min-interactions", str(opts.min_interactions)]
    if opts.disable_tools:
        argv += ["--disable-tools", opts.disable_tools]
    if opts.no_env_notice:
        argv.append("--no-env-notice")
    if opts.follow_peers is not None:
        argv += ["--follow-peers", str(opts.follow_peers)]
    if opts.feed_max_posts:
        argv += ["--feed-max-posts", str(opts.feed_max_posts)]
    env = dict(os.environ)
    env["PATH"] = f"{Path.home() / '.local' / 'bin'}:{env.get('PATH', '')}"
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=opts.timeout,
        )
        rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    (LOG_DIR / f"{opts.variant}__{build_id}__s{seed}.log").write_text(
        f"$ {' '.join(argv)}\nrc={rc}\n{stdout}\n--- stderr ---\n"
        + "\n".join((stderr or "").splitlines()[-30:]),
        encoding="utf-8",
    )
    summary = {}
    if (out / "run_summary.json").is_file():
        try:
            summary = json.loads((out / "run_summary.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    return {
        "build_id": build_id,
        "seed": seed,
        "ok": bool(summary.get("ok")),
        "elapsed_s": round(time.time() - started, 1),
        "out": str(out),
    }


def discrimination(arch: str, ideas: set[str] | None = None) -> dict:
    """How well one instrument variation separates builds, on this fleet.

    ``ideas`` restricts every arm to the same subset. Without it a 40-run
    ablation over 10 briefs would be compared against a 150-run sweep over 25,
    and the difference in spread would be mostly the difference in which apps
    each arm happened to see.
    """
    corpus = load_corpus(REPO, spec=CURRENT_FLEET, arch_version=arch)
    if ideas:
        keep = {b.build_id for b in corpus.fleet_builds() if b.idea_id in ideas}
        corpus = FleetCorpus(
            builds=corpus.builds,
            runs=[
                r
                for r in corpus.runs
                if r.build_id in keep or r.build_id not in corpus.fleet_ids
            ],
            fleet_ids=corpus.fleet_ids & keep,
            profile=corpus.profile,
            arch_version=corpus.arch_version,
        )
    runs = corpus.fleet_runs()
    cells = corpus.cells()
    means = [statistics.fmean(r.score for r in v) for v in cells.values() if v]
    noise = within_cell_sd(corpus)
    gap = paired_gap(corpus, CURRENT_FLEET.model_a, CURRENT_FLEET.model_b)
    null_a = self_separation(corpus, CURRENT_FLEET.model_a)
    control = [r.score for r in corpus.control_runs()]
    return {
        "arch": arch,
        "runs": len(runs),
        "cells": len(cells),
        "between_sd": round(statistics.pstdev(means), 2) if len(means) > 1 else None,
        "within_sd": noise,
        "ratio": (
            round(statistics.pstdev(means) / noise, 2)
            if noise and len(means) > 1
            else None
        ),
        "gap": gap.gap,
        "gap_ci": (gap.ci_low, gap.ci_high),
        "n_ideas": gap.n_ideas,
        "null": null_a.gap,
        "control_mean": round(statistics.fmean(control), 1) if control else None,
        "control_margin": (
            round(statistics.median(means) - statistics.fmean(control), 1)
            if control and means
            else None
        ),
    }


def archs_on_disk() -> list[str]:
    """Every architecture tag present under ``builds/``, base first."""
    seen: set[str] = set()
    for summary_path in BUILDS.glob("*/*/run_summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        arch = str(data.get("crowd_arch_version", ""))
        if arch.startswith(CROWD_ARCH_VERSION):
            seen.add(arch)
    return sorted(seen, key=lambda a: (a != CROWD_ARCH_VERSION, a))


def arm_ideas() -> dict[str, set[str]]:
    """Ideas each architecture tag has measured."""
    out: dict[str, set[str]] = {}
    for arch in archs_on_disk():
        corpus = load_corpus(REPO, spec=CURRENT_FLEET, arch_version=arch)
        found = {r.idea_id for r in corpus.fleet_runs()}
        if found:
            out[arch] = found
    return out


def shared_ideas(n_ideas: int) -> set[str]:
    """The briefs every arm was ASKED to cover -- the deterministic subset.

    Derived from the same rule the runner uses, not from what happens to be on
    disk. Intersecting whatever each arm has measured so far sounds equivalent
    and is not: a mid-flight arm shrinks the set for everybody, which silently
    re-scores every other arm on a smaller, different sample and makes the table
    look like the architecture moved when only the sample did. Observed live --
    one running arm took the shared set from 20 briefs to 12 and swung the
    BASELINE's reported gap from -17.8 to -3.9.
    """
    return {b.idea_id for b in subset(n_ideas)}


def report(n_ideas: int = 10) -> str:
    arms = arm_ideas()
    ideas = shared_ideas(n_ideas)
    lines = [
        f"ABLATION COMPARISON  ({len(ideas)} briefs measured by every COMPLETE arm)",
        "  between_sd = spread across builds (signal); within_sd = spread across",
        "  seeds of one build (noise); ratio is what decides whether an",
        "  instrument can rank apps at all. A variant that widens the model gap",
        "  while raising within_sd has bought nothing.",
        "",
        f"  {'arch':<18}{'runs':>5}{'cells':>6}{'betw':>7}{'with':>7}{'ratio':>7}"
        f"{'gap':>8}{'null':>7}{'ctrl':>7}{'margin':>8}",
    ]
    for arch in archs_on_disk():
        partial = len(arms.get(arch, set()) & ideas) < len(ideas)
        d = discrimination(arch, ideas)
        if not d["runs"]:
            continue

        def fmt(key: str, spec: str = ".2f", d: dict = d) -> str:
            value = d[key]
            return "-" if value is None else format(value, spec)

        lines.append(
            f"  {arch:<18}{d['runs']:>5}{d['cells']:>6}{fmt('between_sd'):>7}"
            f"{fmt('within_sd'):>7}{fmt('ratio'):>7}{fmt('gap', '+.1f'):>8}"
            f"{fmt('null', '+.1f'):>7}{fmt('control_mean', '.1f'):>7}"
            f"{fmt('control_margin', '.1f'):>8}"
            + ("  (still running)" if partial else "")
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="", help="name for this arm")
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--triers", type=int, default=-1)
    parser.add_argument("--latecomers", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--recsys", default="")
    parser.add_argument("--thinking-level", default="")
    parser.add_argument("--semaphore", type=int, default=0)
    parser.add_argument("--trial-max-steps", type=int, default=0)
    parser.add_argument("--min-interactions", type=int, default=None)
    parser.add_argument("--disable-tools", default="")
    parser.add_argument("--no-env-notice", action="store_true")
    parser.add_argument("--follow-peers", type=int, default=None)
    parser.add_argument("--feed-max-posts", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--model", default="", help="crowd model override")
    # Must match what the queue runs with, or the report silently compares arms
    # on a NARROWER set than they measured -- still matched, but throwing away
    # data and moving every number.
    parser.add_argument("--builds", type=int, default=10, help="how many IDEAS")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--report", action="store_true", help="compare, run nothing")
    args = parser.parse_args(argv)

    if args.report or not args.variant:
        print(report(args.builds))
        return 0

    builds = subset(args.builds)
    ids = [b.build_id for b in builds]
    if not args.no_control and (BUILDS / "work" / CONTROL_BUILD).is_dir():
        ids.append(CONTROL_BUILD)
    cells = [(bid, seed) for seed in range(args.seeds) for bid in ids]
    print(
        f"ablation {args.variant!r}: arch {CROWD_ARCH_VERSION}+{args.variant}, "
        f"{len(ids)} builds x {args.seeds} seeds = {len(cells)} runs "
        f"(agents={args.agents} triers={args.triers} "
        f"latecomers={args.latecomers} rounds={args.rounds})"
    )
    done = ok = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(run_one, b, s, args): (b, s) for b, s in cells}
        for future in as_completed(futures):
            res = future.result()
            done += 1
            ok += 1 if res["ok"] else 0
            print(
                f"[{done}/{len(cells)}] {'ok  ' if res['ok'] else 'FAIL'} "
                f"{res['build_id'][:46]:<46} s{res['seed']} {res['elapsed_s']}s",
                flush=True,
            )
    print(f"\n{ok}/{done} runs ok\n")
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
