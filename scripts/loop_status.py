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

"""Report where the ViralBench iteration loop stands, and whether it may stop.

Deterministic, read-only, no LLM calls: every number is derived from ``builds/``
by re-scoring stored artifacts (free), plus the repo's own lint/format/test
commands. Run it as the last command of a turn so the state of the instrument is
in the transcript verbatim.

The floor it enforces is docs/loop.md section 6. The thresholds are set here, on
purpose, so "done" is a property of the disk rather than of anyone's mood:

* **G1 fleet** -- all 25 ideas *attempted* by both models under one identical
  recorded founder config, and all 25 ideas *measured* (a scorable crowd run) for
  both models.

  There is no longer an "ok builds" threshold, because there is no longer a
  reason for one. It existed so nobody could declare a result without having
  measured anything, back when a build with a missing or malformed
  ``viralbench.json`` was skipped by the crowd entirely. Every build is simulated
  now -- an undeliverable one is presented to the crowd, found to be unlaunchable,
  and scored at the floor by the same machinery that handles apps which build but
  do not start. So coverage is what the gate checks, and it checks all 25 rather
  than 18. **Build success rate is a headline result, not a gate**: a model
  failing builds is the finding, and gating on it would let a worse model block
  the loop forever.
* **G2 crowd** -- >= 3 seeds in every paired cell (2 seeds cannot separate
  cell-to-cell noise from a difference; 3 is the smallest that can), and >= 90%
  of fleet runs scorable. Unscorable runs are *counted*, never skipped.
* **G3 control** -- can the instrument tell a broken app from a working one? Three
  clauses, because the obvious phrasing ("far below *every* real build") is wrong:
  some real builds are genuinely worse than a deliberately broken page that at
  least renders a loading message. I got this wrong twice before writing it this
  way, both times by testing the slogan instead of the thing.

  1. ``control_max < 20/100`` -- an absolute floor.
  2. ``median(working cells) - control_mean >= 25`` -- the *bulk* of real builds
     sit far above it. This is the clause a compressed scoring profile cannot
     satisfy: squeeze everything toward the middle and the median comes down with
     the control.
  3. at most 10% of working cells may score at or below the control's mean --
     the "below every real build" clause, relaxed by exactly the amount that real
     corpses exist, and no more.

  Plus: no build that *does not run* may outscore the median working build.
  "Does not run" is builds/runs False; an app that runs but fails its own smoke
  check is a manifest-contract violation by an app agents used happily, and
  belongs in the working set.
* **G4 verdict** -- a ``## VERDICT`` section in the iteration log.
* **G5 CI** -- ruff check, ruff format --check, pytest.

The headline is a **paired per-idea gap in points**, not Cohen's d; see
``viral_bench.score.fleet`` for why pooling across ideas measures idea
difficulty and rewards profiles that compress the scale.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.score.fleet import (  # noqa: E402
    CURRENT_FLEET,
    FleetCorpus,
    FleetSpec,
    control_separation,
    load_corpus,
    paired_gap,
    self_separation,
    within_cell_sd,
)
from viral_bench.score.viralscore import ScoreWeights  # noqa: E402

#: The fleet under test comes from one place (viral_bench.score.fleet), so the
#: status report, the sweep and the analysis can never disagree about which
#: builds are in the experiment.
MODEL_A = CURRENT_FLEET.model_a
MODEL_B = CURRENT_FLEET.model_b

N_IDEAS = 25
MIN_ATTEMPTED = 25
#: Ideas that must have a scorable crowd run for BOTH models. Every build is
#: simulated, so this is 25 -- full coverage, not a survivor subset.
MIN_MEASURED_IDEAS = 25
MIN_SEEDS = 3
MIN_SCORABLE_FRACTION = 0.90
#: The bulk of working builds must sit this far above the broken control.
MIN_CONTROL_MEDIAN_MARGIN = 25.0
#: Fraction of working cells allowed to score at or below the control.
MAX_BELOW_CONTROL_SHARE = 0.10
MAX_CONTROL_SCORE = 20.0
MIN_VERDICT_CHARS = 400
ITERATION_CAP = 30

LOG_PATH = REPO / "docs" / "crowd_agent_iteration_2.md"
_ITERATION_RE = re.compile(r"^##\s+(?:Sweep|Iteration)\s+\d+", re.MULTILINE)
_VERDICT_RE = re.compile(r"^##\s+VERDICT\s*$", re.MULTILINE)


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str


def _fmt(value, spec: str = ".1f", missing: str = "n/a") -> str:
    return missing if value is None else format(value, spec)


def iteration_count() -> int:
    try:
        return len(_ITERATION_RE.findall(LOG_PATH.read_text(encoding="utf-8")))
    except OSError:
        return 0


def verdict_section() -> str:
    try:
        text = LOG_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = _VERDICT_RE.search(text)
    return text[match.end() :].strip() if match else ""


def run_ci() -> list[Gate]:
    checks = [
        ("ruff check", ["uv", "run", "ruff", "check", "."]),
        ("ruff format", ["uv", "run", "ruff", "format", "--check", "."]),
        ("pytest", ["uv", "run", "pytest", "-q"]),
    ]
    gates: list[Gate] = []
    env = {"PATH": f"{Path.home() / '.local' / 'bin'}:/usr/bin:/bin"}
    import os

    full_env = dict(os.environ)
    full_env["PATH"] = env["PATH"] + ":" + full_env.get("PATH", "")
    for name, argv in checks:
        try:
            proc = subprocess.run(
                argv, cwd=REPO, capture_output=True, text=True, env=full_env
            )
        except OSError as exc:
            gates.append(Gate(name, False, f"could not run: {exc}"))
            continue
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()
        gates.append(
            Gate(
                name,
                proc.returncode == 0,
                tail[-1][:110] if tail else f"rc={proc.returncode}",
            )
        )
    return gates


def fleet_section(corpus: FleetCorpus) -> tuple[list[str], Gate, dict]:
    builds = corpus.fleet_builds()
    lines = ["FLEET"]
    if not builds:
        lines.append("  no builds/fleet.json entries yet -- the fleet is not built")
        return lines, Gate("G1 fleet", False, "fleet not built"), {}

    configs = {json.dumps(b.config, sort_keys=True) for b in builds}
    identical = len(configs) == 1
    lines.append(f"  {len(builds)} fleet builds; config identical: {identical}")
    if identical:
        lines.append(f"  config: {next(iter(configs))}")

    stats: dict = {}
    for model in (MODEL_A, MODEL_B):
        mine = [b for b in builds if b.model == model]
        ok = [b for b in mine if b.ok]
        failures: dict[str, int] = {}
        for b in mine:
            if not b.ok:
                failures[b.status] = failures.get(b.status, 0) + 1
        stats[model] = {
            "attempted": len(mine),
            "ok": len(ok),
            "ideas_attempted": len({b.idea_id for b in mine}),
            "n_replicates": max(1, round(len(mine) / N_IDEAS)) if mine else 0,
            "ok_ideas": sorted({b.idea_id for b in ok}),
            "failures": failures,
            "rounds": statistics.fmean([b.rounds_run or 0 for b in ok]) if ok else None,
            "turns": statistics.fmean([b.turns_spent or 0 for b in ok]) if ok else None,
            "ship_at_1": (
                sum(1 for b in ok if b.shipped_early) / len(ok) if ok else None
            ),
            "qa_verified": (
                sum(1 for b in ok if b.qa_verified) / len(ok) if ok else None
            ),
            "minutes": (
                statistics.fmean([(b.build_seconds or 0) / 60 for b in ok])
                if ok
                else None
            ),
        }
        fail_txt = " ".join(f"{k}={v}" for k, v in sorted(failures.items())) or "-"
        # Denominator is BUILDS attempted, not ideas. With two replicates of the
        # fleet there are 50 builds over 25 ideas, and dividing one by the other
        # printed the nonsense "ok 50/25".
        lines.append(
            f"  {model:<18} ok {len(ok):>2}/{len(mine):<2} builds "
            f"({stats[model]['n_replicates']} replicate(s) x {N_IDEAS} ideas)  "
            f"failures: {fail_txt}"
        )

    # Pairing is now by MEASUREMENT, not by build success: every build is
    # simulated, so an idea is paired once both models have a scorable run for
    # it -- including runs where the crowd found nothing to launch.
    measured = corpus.cells_pooling_structures()
    paired = sorted(
        {idea for (idea, model) in measured if model == MODEL_A}
        & {idea for (idea, model) in measured if model == MODEL_B}
    )
    built_both = set(stats[MODEL_A]["ok_ideas"]) & set(stats[MODEL_B]["ok_ideas"])
    lines.append(
        f"  ideas MEASURED for both models: {len(paired)}/{N_IDEAS}   "
        f"(of which built ok by both: {len(built_both)})"
    )
    lines.append("")
    lines.append("FOUNDER STAGE (free signal: capability visible before the crowd)")
    lines.append("  model               rounds turns ship@1 qa_ver min/build")
    for model in (MODEL_A, MODEL_B):
        s = stats[model]
        lines.append(
            f"  {model:<18} {_fmt(s['rounds'], '.2f'):>6} "
            f"{_fmt(s['turns'], '.1f'):>5} {_fmt(s['ship_at_1'], '.0%'):>6} "
            f"{_fmt(s['qa_verified'], '.0%'):>6} {_fmt(s['minutes'], '.1f'):>9}"
        )

    # Build outcome is a headline result, not a footnote. If one model ships
    # working software on 24 of 25 briefs and the other on 17, that is a bigger
    # and more actionable statement than any score gap, and it is measured before
    # the crowd says a word.
    ok_a, ok_b = stats[MODEL_A]["ok"], stats[MODEL_B]["ok"]
    n_a, n_b = stats[MODEL_A]["attempted"], stats[MODEL_B]["attempted"]
    rate_a = 1 - ok_a / n_a if n_a else 0.0
    rate_b = 1 - ok_b / n_b if n_b else 0.0
    lines.append(
        f"  BUILD OUTCOME  A {ok_a}/{n_a} vs B {ok_b}/{n_b} builds shipped "
        f"-> failure rate {rate_a:.0%} vs {rate_b:.0%}"
    )

    # Which cells scored at the floor because nothing was shipped to launch, as
    # opposed to scoring low because users tried them and were unimpressed. Both
    # are real outcomes; conflating them hides which problem a model has.
    undeliverable = sorted(
        {
            f"{r.idea_id}[{'A' if r.model == MODEL_A else 'B'}]"
            for r in corpus.fleet_runs()
            if r.undeliverable
        }
    )
    if undeliverable:
        lines.append(
            f"  UNDELIVERABLE (no runnable viralbench.json, scored at the floor): "
            f"{len(undeliverable)} cells"
        )
        for i in range(0, len(undeliverable), 4):
            lines.append("    " + "  ".join(undeliverable[i : i + 4]))

    reasons = []
    if not identical:
        reasons.append(f"{len(configs)} distinct founder configs in the fleet")
    for model in (MODEL_A, MODEL_B):
        attempted = stats[model]["ideas_attempted"]
        if attempted < MIN_ATTEMPTED:
            reasons.append(f"{model} attempted {attempted}<{MIN_ATTEMPTED} ideas")
    if len(paired) < MIN_MEASURED_IDEAS:
        reasons.append(
            f"only {len(paired)}/{MIN_MEASURED_IDEAS} ideas measured for both models"
        )
    gate = Gate(
        "G1 fleet",
        not reasons,
        "; ".join(reasons)
        or f"{N_IDEAS}/{N_IDEAS} attempted and {len(paired)}/{N_IDEAS} measured "
        f"for both models, one config",
    )
    return lines, gate, {"paired": paired, "stats": stats}


def crowd_section(corpus: FleetCorpus, paired: list[str]) -> tuple[list[str], Gate]:
    all_runs = corpus.fleet_runs(scorable_only=False)
    scorable = [r for r in all_runs if r.scorable]
    frac = len(scorable) / len(all_runs) if all_runs else 0.0
    cells = corpus.cells_pooling_structures()
    lines = ["", "CROWD"]
    sizes = sorted({r.n_agents for r in scorable})
    lines.append(
        f"  {len(all_runs)} runs over {len({r.build_id for r in all_runs})} builds; "
        f"crowd sizes {sizes or '-'}"
    )
    lines.append(
        f"  scorable {len(scorable)}/{len(all_runs)} ({frac:.1%}); "
        f"cells with runs: {len(cells)}"
    )
    stale = corpus.stale_runs()
    if stale:
        lines.append(
            f"  excluded: {len(stale)} runs from an older crowd architecture "
            f"(corpus is v{corpus.arch_version} only)"
        )
    thin = [
        f"{idea}[{'A' if model == MODEL_A else 'B'}]"
        for idea in paired
        for model in (MODEL_A, MODEL_B)
        if len({r.seed for r in cells.get((idea, model), [])}) < MIN_SEEDS
    ]
    full_pairs = sum(
        1
        for idea in paired
        if all(
            len({r.seed for r in cells.get((idea, m), [])}) >= MIN_SEEDS
            for m in (MODEL_A, MODEL_B)
        )
    )
    lines.append(
        f"  paired ideas with >= {MIN_SEEDS} seeds on BOTH models: "
        f"{full_pairs}/{len(paired)}"
    )
    if thin:
        lines.append(f"  thin cells: {', '.join(thin[:12])}"[:200])
    unscorable = [r for r in all_runs if not r.scorable]
    if unscorable:
        why: dict[str, int] = {}
        for run in unscorable:
            key = (run.blockers or ["unknown"])[0][:60]
            why[key] = why.get(key, 0) + 1
        for key, count in sorted(why.items(), key=lambda kv: -kv[1])[:3]:
            lines.append(f"  unscorable x{count}: {key}")

    reasons = []
    if full_pairs < MIN_MEASURED_IDEAS:
        reasons.append(f"{full_pairs} paired cells with {MIN_SEEDS}+ seeds")
    if frac < MIN_SCORABLE_FRACTION:
        reasons.append(f"scorable {frac:.0%}<{MIN_SCORABLE_FRACTION:.0%}")
    return lines, Gate(
        "G2 crowd",
        not reasons,
        "; ".join(reasons)
        or f"{full_pairs} paired cells x{MIN_SEEDS} seeds, {frac:.0%} scorable",
    )


def signal_section(
    corpus: FleetCorpus, spec: FleetSpec = CURRENT_FLEET
) -> tuple[list[str], dict]:
    gap = paired_gap(corpus, MODEL_A, MODEL_B)
    null_a = self_separation(corpus, MODEL_A)
    null_b = self_separation(corpus, MODEL_B)
    noise = within_cell_sd(corpus)
    cells = corpus.cells_pooling_structures()

    def mean_for(model: str) -> float | None:
        vals = [r.score for r in corpus.fleet_runs() if r.model == model]
        return round(statistics.fmean(vals), 2) if vals else None

    lines = ["", "SIGNAL  (paired per-idea gap in ViralScore points)"]
    lines.append(
        f"  A {MODEL_A} mean {_fmt(mean_for(MODEL_A), '.1f')}   "
        f"B {MODEL_B} mean {_fmt(mean_for(MODEL_B), '.1f')}"
    )
    lines.append(
        f"  gap A-B = {_fmt(gap.gap, '+.2f')} points over {gap.n_ideas} ideas   "
        f"95% CI [{_fmt(gap.ci_low, '+.2f')}, {_fmt(gap.ci_high, '+.2f')}]   "
        f"A wins {gap.wins_a}/{gap.n_ideas}"
    )
    lines.append(f"  noise floor (pooled within-cell SD): {_fmt(noise, '.2f')}")
    for label, null in (("A", null_a), ("B", null_b)):
        lines.append(
            f"  null {label} vs itself (even/odd seeds): "
            f"{_fmt(null.gap, '+.2f')} over {null.n_ideas} ideas   "
            f"CI [{_fmt(null.ci_low, '+.2f')}, {_fmt(null.ci_high, '+.2f')}]"
        )
    # The same gap under every other scoring profile. docs/loop.md is explicit
    # that a difference which only exists under one weighting is a weighting,
    # not a difference -- so the report shows all of them rather than making
    # anyone go and check. Re-scoring is free.
    others = []
    for name in ("v2_hybrid", "v1_deterministic", "qualitative_heavy"):
        try:
            weights = ScoreWeights.from_profile(name)
            alt = load_corpus(REPO, weights, spec=spec)
        except Exception:  # noqa: BLE001 - a broken profile must not break the report
            continue
        alt_gap = paired_gap(alt, MODEL_A, MODEL_B)
        if alt_gap.gap is not None:
            others.append(
                f"{name} {alt_gap.gap:+.1f} (A wins {alt_gap.wins_a}/{alt_gap.n_ideas})"
            )
    if others:
        lines.append(f"  same gap under other profiles: {'; '.join(others)}")
    if gap.per_idea:
        lines.append("  per-idea A-B (sorted):")
        items = sorted(gap.per_idea.items(), key=lambda kv: -kv[1])
        for i in range(0, len(items), 3):
            chunk = items[i : i + 3]
            lines.append(
                "    "
                + "  ".join(f"{name[:22]:<22}{delta:+6.1f}" for name, delta in chunk)
            )
    return lines, {
        "gap": gap,
        "null_a": null_a,
        "null_b": null_b,
        "noise": noise,
        "n_cells": len(cells),
    }


def control_section(corpus: FleetCorpus) -> tuple[list[str], Gate]:
    sep = control_separation(corpus)
    lines = ["", "CONTROL (deliberately broken app)"]
    lines.append(
        f"  {sep.n_control_runs} runs; mean {_fmt(sep.control_mean, '.1f')}, "
        f"max {_fmt(sep.control_max, '.1f')}"
    )
    median_margin = (
        None
        if sep.working_median is None or sep.control_mean is None
        else sep.working_median - sep.control_mean
    )
    lines.append(
        f"  working cells: {sep.n_working_cells}, median "
        f"{_fmt(sep.working_median, '.1f')} -> median margin "
        f"{_fmt(median_margin, '+.1f')} points"
    )
    lines.append(
        f"  weakest working cell: {sep.weakest_real_cell or '-'} "
        f"{_fmt(sep.weakest_real_mean, '.1f')};  working cells at or below the "
        f"control: {len(sep.below_control)} "
        f"({', '.join(sep.below_control[:3]) or 'none'})"
    )
    # The POSITIVE control: a deliberately correct full-stack app, verified
    # 12/12 on the server-side contract. It is not a ceiling -- it is a plain
    # memo board and the crowd rightly finds it dull -- but if the crowd ever
    # stops reporting persistence and multi-user visibility on THIS build, the
    # harness has broken rather than the fleet.
    positive = [r.score for r in corpus.control_runs(kind="positive")]
    if positive:
        lines.append(
            f"  positive control (known-good full-stack): {len(positive)} runs, "
            f"mean {_fmt(statistics.fmean(positive), '.1f')}"
        )
    if sep.dead_cells:
        worst = sorted(sep.dead_cells.items(), key=lambda kv: -kv[1])[:3]
        lines.append(
            f"  real builds that do not run: {len(sep.dead_cells)}"
            f" (top: {', '.join(f'{k} {v:.1f}' for k, v in worst)})"
        )
    reasons = []
    if sep.n_control_runs < MIN_SEEDS:
        reasons.append(f"{sep.n_control_runs} control runs < {MIN_SEEDS}")
    if sep.control_max is None or sep.control_max >= MAX_CONTROL_SCORE:
        reasons.append(f"control max {_fmt(sep.control_max, '.1f')} not < 20")
    if median_margin is None or median_margin < MIN_CONTROL_MEDIAN_MARGIN:
        reasons.append(
            f"median working build is only {_fmt(median_margin, '.1f')} above "
            f"the control (need {MIN_CONTROL_MEDIAN_MARGIN:.0f})"
        )
    if sep.n_working_cells:
        share = len(sep.below_control) / sep.n_working_cells
        if share > MAX_BELOW_CONTROL_SHARE:
            reasons.append(
                f"{len(sep.below_control)} of {sep.n_working_cells} working cells "
                f"({share:.0%}) score at or below the control"
            )
    # An app that does not run must not score like a typical working one. Bound
    # against the MEDIAN rather than the minimum, for the same reason clause 2
    # does: the weakest working build can itself be worse than a corpse.
    if (
        sep.dead_max is not None
        and sep.working_median is not None
        and sep.dead_max > sep.working_median
    ):
        reasons.append(
            f"an app that does not run scores {sep.dead_max:.1f}, above the "
            f"median working build at {sep.working_median:.1f}"
        )
    return lines, Gate(
        "G3 control",
        not reasons,
        "; ".join(reasons)
        or f"control {_fmt(sep.control_mean, '.1f')}, median working build "
        f"{_fmt(sep.working_median, '.1f')}, {len(sep.below_control)} exceptions",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ViralBench loop status")
    parser.add_argument("--skip-ci", action="store_true", help="skip lint/format/test")
    parser.add_argument("--json", action="store_true", help="also dump raw JSON")
    # Which founder arm the report is about. Defaults to CURRENT_FLEET so the
    # habitual `loop_status.py` is unchanged; naming an arm here is how a new
    # one (e.g. dynamic) gets reported without editing the scoring constant,
    # which is code every other reader depends on meaning one specific thing.
    parser.add_argument(
        "--fleet-structure",
        default=CURRENT_FLEET.structure,
        help="founder arm to report on (default: %(default)s)",
    )
    parser.add_argument(
        "--fleet-replicate",
        type=int,
        default=CURRENT_FLEET.replicate,
        help="fleet replicate to report on (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    spec = replace(
        CURRENT_FLEET,
        structure=args.fleet_structure,
        replicate=args.fleet_replicate,
    )
    corpus = load_corpus(REPO, spec=spec)
    iteration = iteration_count()
    out: list[str] = [
        f"ViralBench loop status  {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}  "
        f"iteration {iteration}/{ITERATION_CAP}  crowd arch v{corpus.arch_version}"
        f"  score profile: {corpus.profile}"
    ]

    fleet_lines, g1, fleet_info = fleet_section(corpus)
    out += fleet_lines
    crowd_lines, g2 = crowd_section(corpus, fleet_info.get("paired", []))
    out += crowd_lines
    signal_lines, signal = signal_section(corpus, spec)
    out += signal_lines
    control_lines, g3 = control_section(corpus)
    out += control_lines

    verdict = verdict_section()
    g4 = Gate(
        "G4 verdict",
        len(verdict) >= MIN_VERDICT_CHARS,
        f"## VERDICT section is {len(verdict)} chars (need >= {MIN_VERDICT_CHARS})",
    )
    gates = [g1, g2, g3, g4]
    if args.skip_ci:
        out += ["", "CI  skipped (--skip-ci)"]
    else:
        ci_gates = run_ci()
        gates += [
            Gate(
                "G5 ci",
                all(g.passed for g in ci_gates),
                "; ".join(f"{g.name}={'ok' if g.passed else 'FAIL'}" for g in ci_gates),
            )
        ]

    out += ["", "GATES"]
    for gate in gates:
        out.append(
            f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name}: {gate.detail}"
        )

    if all(g.passed for g in gates) and not args.skip_ci:
        exit_line = "LOOP_EXIT: DONE"
    elif iteration >= ITERATION_CAP:
        exit_line = "LOOP_EXIT: STOP"
    else:
        exit_line = "LOOP_EXIT: CONTINUE"

    text = "\n".join(out)
    if len(text) > 12000:
        text = text[:11900] + "\n  ... (truncated)"
    print(text)
    if args.json:
        gap = signal["gap"]
        print(
            "\nJSON "
            + json.dumps(
                {
                    "iteration": iteration,
                    "gap": gap.gap,
                    "ci": [gap.ci_low, gap.ci_high],
                    "n_ideas": gap.n_ideas,
                    "noise": signal["noise"],
                    "gates": {g.name: g.passed for g in gates},
                },
                separators=(",", ":"),
            )
        )
    print()
    print(exit_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
