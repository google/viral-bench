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

"""Compare RubricScore with ViralScore over a build cohort.

The question the whole track exists to answer: what does a crowd of naive users
see that a rubric does not, and vice versa. Both instruments score the same
builds, so the comparison is a join rather than an argument.

Read the noise floors first. A cohort holds several crowd seeds per build and the
rubric grades each item three times, so both instruments have a measurable
pass-to-pass spread. A gap between them that is smaller than either spread is
not a finding, and this report prints the spreads beside every correlation so
that cannot be quietly forgotten.

Because a cohort spans every founder arm as well as every model, the same join
answers two questions rather than one: **model vs model** and **founder
architecture vs founder architecture**, under both scoring regimes.

Usage::

    scripts/rubric_vs_viralscore.py --cohort r4 --arch 14
    scripts/rubric_vs_viralscore.py --json out.json --markdown report.md

PASS ``--arch``. A cohort may keep solo and dynamic from the previous
generation, so those build ids carry crowd runs under both the old architecture
and the new one, and
averaging across them mixes two instruments into one number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BUILDS = REPO / "builds"
CROWD = BUILDS / "crowd"
RUBRIC = BUILDS / "rubric"


# ---------------------------------------------------------------- statistics


def _rank(values: list[float]) -> list[float]:
    """Average ranks, so ties do not bias the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None
    mean_a, mean_b = sum(a) / n, sum(b) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    den_a = sum((x - mean_a) ** 2 for x in a) ** 0.5
    den_b = sum((y - mean_b) ** 2 for y in b) ** 0.5
    if den_a == 0 or den_b == 0:
        return None
    return num / (den_a * den_b)


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    return _pearson(_rank([p[0] for p in pairs]), _rank([p[1] for p in pairs]))


# ------------------------------------------------------------------- loading


@dataclass
class BuildRow:
    build_id: str
    idea_id: str
    model: str
    arm: str
    status: str
    viral: float | None = None
    viral_runs: list[float] = field(default_factory=list)
    rubric: float | None = None
    rubric_gate: bool | None = None
    rubric_items: int = 0
    rubric_disagreeing: int = 0
    rubric_override_rate: float | None = None
    rubric_code_disagreement: float | None = None
    rubric_agent_disagreement: float | None = None

    @property
    def viral_spread(self) -> float | None:
        """Pass-to-pass SD of the crowd, this build's own noise floor."""
        if len(self.viral_runs) < 2:
            return None
        return statistics.stdev(self.viral_runs)


def run_arch(run_dir: Path) -> str:
    """The crowd architecture a run was produced under, or "" if unreadable."""
    try:
        summary = json.loads((run_dir / "run_summary.json").read_text())
    except (OSError, ValueError):
        return ""
    return str(summary.get("crowd_arch_version") or "")


def load_rows(
    cohort: str = "r4", generation: str = "", arch: str = ""
) -> list[BuildRow]:
    """Both instruments' scores for every build in scope, joined by build id.

    ``arch`` restricts the crowd side to one architecture, and matters more than
    it looks. A cohort may keep the solo and dynamic builds from the previous
    generation, so those ids carry crowd runs under BOTH the old architecture and the
    new one. Averaging across them would silently mix two different instruments
    into one number and attribute the difference to the builds.
    """
    from viral_bench.rubric.corpus import select_builds

    rows: dict[str, BuildRow] = {}
    for build in select_builds(BUILDS, cohort=cohort, generation=generation):
        rows[build.build_id] = BuildRow(
            build_id=build.build_id,
            idea_id=build.idea_id,
            model=build.model,
            arm=build.arm,
            status=build.status,
        )

    # ViralScore: every seed of every build, averaged. Several seeds per build
    # is what makes the crowd's own noise floor measurable at all.
    if CROWD.is_dir():
        for run_dir in CROWD.iterdir():
            if not run_dir.is_dir() or "__crowd-" not in run_dir.name:
                continue
            row = rows.get(run_dir.name.split("__crowd-")[0])
            if row is None:
                continue
            if arch and run_arch(run_dir) != arch:
                continue
            score_path = run_dir / "score.json"
            if not score_path.is_file():
                continue
            try:
                document = json.loads(score_path.read_text())
            except (OSError, ValueError):
                continue
            # `score` is None exactly when the run was unscorable -- the
            # persisted document carries no separate `scorable` flag, so testing
            # for one silently discards every run.
            if document.get("score") is not None:
                row.viral_runs.append(float(document["score"]))

    for row in rows.values():
        if row.viral_runs:
            row.viral = statistics.fmean(row.viral_runs)

    # RubricScore: the newest grade per build.
    if RUBRIC.is_dir():
        newest: dict[str, tuple[str, dict]] = {}
        for run_dir in RUBRIC.glob("*__rubric-*"):
            grade = run_dir / "grade.json"
            if not grade.is_file():
                continue
            try:
                document = json.loads(grade.read_text())
            except (OSError, ValueError):
                continue
            build_id = document.get("build_id", "")
            if build_id not in rows:
                continue
            if build_id not in newest or run_dir.name > newest[build_id][0]:
                newest[build_id] = (run_dir.name, document)
        for build_id, (_, document) in newest.items():
            row = rows[build_id]
            # `score`, not `final`, and `tiers[].items`, not a flat `items`. Reading
            # the wrong keys yields None everywhere and an empty comparison that
            # looks like "no data" rather than "wrong reader".
            row.rubric = document.get("score")
            row.rubric_gate = (document.get("gate") or {}).get("passed")
            reliability = document.get("reliability") or {}
            row.rubric_items = reliability.get("items_total") or 0
            row.rubric_disagreeing = reliability.get("items_disagreeing") or 0
            row.rubric_override_rate = reliability.get("override_rate")
            row.rubric_code_disagreement = reliability.get("code_disagreement")
            row.rubric_agent_disagreement = reliability.get("agent_disagreement")

    return list(rows.values())


# -------------------------------------------------------------------- report


def _group(rows: list[BuildRow], key) -> dict[str, list[BuildRow]]:
    out: dict[str, list[BuildRow]] = defaultdict(list)
    for row in rows:
        out[key(row)].append(row)
    return dict(out)


def _paired(rows: list[BuildRow]) -> list[tuple[float, float]]:
    return [
        (r.viral, r.rubric)
        for r in rows
        if r.viral is not None and r.rubric is not None
    ]


def build_report(rows: list[BuildRow], generation: str, arch_label: str = "") -> dict:
    both = [r for r in rows if r.viral is not None and r.rubric is not None]
    pairs = _paired(rows)

    spreads = [r.viral_spread for r in rows if r.viral_spread is not None]
    report = {
        "generation": generation,
        "builds": len(rows),
        "with_viralscore": sum(1 for r in rows if r.viral is not None),
        "with_rubricscore": sum(1 for r in rows if r.rubric is not None),
        "compared": len(both),
        "crowd_arch": arch_label,
        "noise": {
            "crowd_pass_to_pass_sd_median": (
                round(statistics.median(spreads), 2) if spreads else None
            ),
            "crowd_builds_with_multiple_seeds": len(spreads),
        },
        "overall_spearman": (
            round(spearman(pairs), 3) if spearman(pairs) is not None else None
        ),
        "by_idea": {},
        "by_arm": {},
        "by_model": {},
        "divergences": [],
    }

    for label, grouped in (
        ("by_idea", _group(rows, lambda r: r.idea_id)),
        ("by_arm", _group(rows, lambda r: r.arm or "?")),
        ("by_model", _group(rows, lambda r: r.model)),
    ):
        for name, members in sorted(grouped.items()):
            member_pairs = _paired(members)
            rho = spearman(member_pairs)
            viral = [r.viral for r in members if r.viral is not None]
            rubric = [r.rubric for r in members if r.rubric is not None]
            report[label][name] = {
                "n": len(members),
                "compared": len(member_pairs),
                "spearman": round(rho, 3) if rho is not None else None,
                "viral_mean": round(statistics.fmean(viral), 2) if viral else None,
                "rubric_mean": round(statistics.fmean(rubric), 2) if rubric else None,
            }

    # The interesting builds: where the two instruments most disagree on RANK.
    if len(pairs) >= 3:
        viral_ranks = _rank([r.viral for r in both])
        rubric_ranks = _rank([r.rubric for r in both])
        n = len(both)
        gaps = sorted(
            (
                {
                    "build_id": row.build_id,
                    "idea_id": row.idea_id,
                    "model": row.model,
                    "arm": row.arm,
                    "viral": round(row.viral, 2),
                    "rubric": round(row.rubric, 2),
                    "viral_pct": round(100 * vr / n, 1),
                    "rubric_pct": round(100 * rr / n, 1),
                    "rank_gap_pct": round(100 * (vr - rr) / n, 1),
                }
                for row, vr, rr in zip(both, viral_ranks, rubric_ranks, strict=True)
            ),
            key=lambda d: -abs(d["rank_gap_pct"]),
        )
        report["divergences"] = gaps[:25]

    # L4 discrimination, for each instrument on its own terms: a benchmark whose
    # between-build spread is not several times its own pass-to-pass spread is
    # reporting noise. Computed per instrument so they can be compared directly.
    viral_means = [r.viral for r in rows if r.viral is not None]
    rubric_means = [r.rubric for r in rows if r.rubric is not None]
    if len(viral_means) > 2:
        report["noise"]["viral_between_build_sd"] = round(
            statistics.stdev(viral_means), 2
        )
        floor = report["noise"]["crowd_pass_to_pass_sd_median"]
        if floor:
            report["noise"]["viral_discrimination"] = round(
                statistics.stdev(viral_means) / floor, 1
            )
    if len(rubric_means) > 2:
        report["noise"]["rubric_between_build_sd"] = round(
            statistics.stdev(rubric_means), 2
        )

    # The rubric's own noise floor, from the three passes per item. L3's number:
    # a gap between the instruments smaller than this is not a finding either.
    graded = [r for r in rows if r.rubric is not None and r.rubric_items]
    if graded:

        def _mean(values):
            values = [v for v in values if v is not None]
            return round(statistics.fmean(values), 4) if values else None

        report["noise"]["rubric_code_disagreement"] = _mean(
            r.rubric_code_disagreement for r in graded
        )
        report["noise"]["rubric_agent_disagreement"] = _mean(
            r.rubric_agent_disagreement for r in graded
        )
        report["noise"]["rubric_harness_override_rate"] = _mean(
            r.rubric_override_rate for r in graded
        )
    return report


def render(report: dict) -> str:
    lines = [
        f"# RubricScore vs ViralScore, {report['generation']}",
        "",
        f"{report['builds']} builds; {report['with_viralscore']} have a ViralScore, "
        f"{report['with_rubricscore']} a RubricScore, **{report['compared']} both**."
        + (
            f" Crowd runs limited to arch {report['crowd_arch']}."
            if report.get("crowd_arch") not in ("", "all")
            else ""
        ),
        "",
    ]
    noise = report["noise"]
    if noise["crowd_pass_to_pass_sd_median"] is not None:
        lines += [
            f"**Crowd noise floor:** median pass-to-pass SD "
            f"{noise['crowd_pass_to_pass_sd_median']} across "
            f"{noise['crowd_builds_with_multiple_seeds']} multi-seed builds. "
            "A gap smaller than this is not a finding.",
            "",
        ]
    if noise.get("viral_discrimination") is not None:
        lines += [
            f"**Discrimination:** ViralScore between-build SD "
            f"{noise['viral_between_build_sd']} = "
            f"{noise['viral_discrimination']}x its own noise floor"
            + (
                f"; RubricScore between-build SD {noise['rubric_between_build_sd']}."
                if noise.get("rubric_between_build_sd") is not None
                else "."
            ),
            "",
        ]
    if noise.get("rubric_code_disagreement") is not None:
        lines += [
            f"**Rubric noise floor:** code-judged pass-to-pass disagreement "
            f"{noise['rubric_code_disagreement']:.1%}, agent-judged "
            f"{noise['rubric_agent_disagreement']:.1%}, harness override rate "
            f"{noise['rubric_harness_override_rate']:.1%}.",
            "",
        ]
    lines += [f"**Overall Spearman:** {report['overall_spearman']}", ""]

    for label, title in (
        ("by_arm", "By founder architecture"),
        ("by_model", "By model"),
        ("by_idea", "By idea"),
    ):
        lines += [
            f"## {title}",
            "",
            "| group | n | ρ | ViralScore | RubricScore |",
            "|---|---|---|---|---|",
        ]
        for name, stats in report[label].items():
            lines.append(
                f"| {name} | {stats['compared']}/{stats['n']} | "
                f"{stats['spearman']} | {stats['viral_mean']} | "
                f"{stats['rubric_mean']} |"
            )
        lines.append("")

    if report["divergences"]:
        lines += [
            "## Where the instruments most disagree",
            "",
            "Rank percentile under each instrument; positive gap means the crowd "
            "liked it more than the rubric did.",
            "",
            "| build | idea | model | arm | viral %ile | rubric %ile | gap |",
            "|---|---|---|---|---|---|---|",
        ]
        for d in report["divergences"]:
            lines.append(
                f"| `{d['build_id'][:40]}` | {d['idea_id']} | {d['model']} | "
                f"{d['arm']} | {d['viral_pct']} | {d['rubric_pct']} | "
                f"{d['rank_gap_pct']:+} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="r4")
    parser.add_argument(
        "--arch",
        default="",
        help="only count crowd runs from this crowd_arch_version (e.g. 14)",
    )
    parser.add_argument(
        "--generation", default="", help="legacy r3 fleet-key selection"
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    opts = parser.parse_args(argv)

    rows = load_rows(opts.cohort, opts.generation, opts.arch)
    report = build_report(rows, opts.generation or opts.cohort, opts.arch or "all")
    text = render(report)

    if opts.json:
        opts.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if opts.markdown:
        opts.markdown.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
