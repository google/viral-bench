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

"""Read a crowd sweep end to end: trials, feed, verdicts, and what separates.

``loop_status.py`` answers "may the loop stop?". This answers the question that
drives the loop: **what did the crowd DO, and which of it discriminates
between two apps?** Everything comes from artifacts already on disk -- the run
summaries, each trier's interaction trace, and the OASIS database -- so it is
free to re-run and never touches a model.

Sections, each independently selectable with ``--section``:

* ``health``   -- did the runs complete, and what did they lose on the way.
* ``trials``   -- what a hands-on trial consists of: depth, which verbs
  succeed, how many agents ever operate the app rather than reading it.
* ``social``   -- the feed: actions per round, where comments attach, whether
  agents name each other, whether exposure is earned or uniform.
* ``verdicts`` -- the crowd's own judgements, and how much they vary.
* ``separate`` -- per-signal separation A vs B and working vs the broken
  control. A signal that cannot tell a corpse from an app cannot rank two apps.
* ``samples``  -- verbatim text. Numbers hide the failure modes that reading one
  post makes obvious.

Usage::

    scripts/crowd_forensics.py                       # every current-arch run
    scripts/crowd_forensics.py --section trials,social
    scripts/crowd_forensics.py --build <build_id>    # one build's runs
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.crowd.sim_defaults import CROWD_ARCH_VERSION  # noqa: E402
from viral_bench.score.fleet import (  # noqa: E402
    CURRENT_FLEET,
    load_builds,
    load_corpus,
)

SECTIONS = ("health", "trials", "social", "verdicts", "separate", "samples")

#: An @handle in agent text. The feed labels authors "@username (Name)", so a
#: mention is the one direct measure of agents addressing each other.
_MENTION = re.compile(r"@([A-Za-z0-9_]{2,})")


@dataclass
class RunView:
    """One crowd run, with every artifact it wrote already parsed."""

    path: Path
    summary: dict
    build_id: str
    model: str
    idea_id: str
    seed: int
    is_control: bool
    traces: list[dict] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.idea_id}[{self.model}]s{self.seed}"

    def db(self) -> sqlite3.Connection | None:
        path = self.path / "simulation.db"
        if not path.is_file():
            return None
        try:
            return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        except sqlite3.Error:
            return None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_runs(
    builds_root: Path,
    *,
    arch: str,
    build_filter: str = "",
    with_traces: bool = True,
) -> list[RunView]:
    """Every crowd run of ``arch`` on disk, with its traces loaded."""
    builds = load_builds(builds_root)
    out: list[RunView] = []
    for summary_path in sorted(builds_root.glob("crowd/*/run_summary.json")):
        summary = _read_json(summary_path)
        if not summary or str(summary.get("crowd_arch_version", "0")) != arch:
            continue
        build_id = summary.get("build_id", "")
        build = builds.get(build_id)
        if build is None:
            continue
        if build_filter and build_filter not in build_id:
            continue
        run = RunView(
            path=summary_path.parent,
            summary=summary,
            build_id=build_id,
            model=build.model,
            idea_id=build.idea_id,
            seed=int((summary.get("config") or {}).get("seed", 0) or 0),
            is_control=build.is_control,
        )
        if with_traces:
            for trace_path in sorted((run.path / "traces").glob("agent_*.json")):
                payload = _read_json(trace_path)
                if payload:
                    run.traces.append(payload)
        out.append(run)
    return out


def _pct(num: float, den: float) -> str:
    return f"{num / den:.0%}" if den else "n/a"


def _dist(values: list[float], spec: str = ".1f") -> str:
    if not values:
        return "no data"
    values = sorted(values)
    return (
        f"n={len(values)} mean={statistics.fmean(values):{spec}} "
        f"median={statistics.median(values):{spec}} "
        f"p10={values[int(0.1 * (len(values) - 1))]:{spec}} "
        f"p90={values[int(0.9 * (len(values) - 1))]:{spec}} "
        f"min={values[0]:{spec}} max={values[-1]:{spec}}"
    )


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #


def health_section(runs: list[RunView]) -> list[str]:
    lines = ["HEALTH"]
    ok = [r for r in runs if r.summary.get("ok")]
    lines.append(f"  {len(ok)}/{len(runs)} runs ok")
    failures: Counter[str] = Counter()
    for run in runs:
        for reason in (run.summary.get("health") or {}).get("failures", []):
            failures[re.sub(r"\d+", "N", reason)[:80]] += 1
    for reason, count in failures.most_common(8):
        lines.append(f"    x{count:<4} {reason}")
    lines.append(
        f"  duration_s: {_dist([r.summary.get('duration_s', 0) for r in runs])}"
    )

    turns = sum((r.summary.get("turn_stats") or {}).get("turns", 0) for r in runs)
    skipped = sum((r.summary.get("turn_stats") or {}).get("skipped", 0) for r in runs)
    limited = sum(
        (r.summary.get("turn_stats") or {}).get("skipped_rate_limited", 0) for r in runs
    )
    exhausted = sum(
        (r.summary.get("turn_stats") or {}).get("budget_exhausted", 0) for r in runs
    )
    lines.append(
        f"  model turns {turns}: skipped {skipped} ({_pct(skipped, turns)}, "
        f"{limited} rate-limited), budget-exhausted {exhausted}"
    )
    losses = [
        (r.summary.get("verdicts") or {}).get("interviews", {}).get("loss_rate", 0.0)
        for r in runs
    ]
    lines.append(f"  interview loss rate: {_dist(losses, '.3f')}")
    repairs = sum(
        (r.summary.get("interview_stats") or {}).get("repair_passes", 0) for r in runs
    )
    lines.append(f"  interview repair passes fired: {repairs}")
    return lines


# --------------------------------------------------------------------------- #
# trials
# --------------------------------------------------------------------------- #

#: Trace verbs that count as operating the app rather than reading it.
_USE_VERBS = ("click", "type", "press", "upload")


def trials_section(runs: list[RunView]) -> list[str]:
    lines = ["", "TRIALS (hands-on evidence)"]
    steps_per_trial: list[float] = []
    verb_total: Counter[str] = Counter()
    verb_failed: Counter[str] = Counter()
    reachable = interacted = finished = total = 0
    fail_msgs: Counter[str] = Counter()
    by_app_type: dict[str, list[int]] = defaultdict(list)

    for run in runs:
        app_type = run.summary.get("app_type", "?")
        for payload in run.traces:
            trace = payload.get("trace") or {}
            steps = trace.get("steps") or []
            total += 1
            steps_per_trial.append(len(steps))
            by_app_type[app_type].append(len(steps))
            if trace.get("app_reachable"):
                reachable += 1
            if trace.get("verdict"):
                finished += 1
            used = False
            for step in steps:
                action = step.get("action", "?")
                verb_total[action] += 1
                ok = step.get("ok", True)
                summary = (step.get("summary") or "").lower()
                bad = (not ok) or summary.startswith("could not")
                if bad:
                    verb_failed[action] += 1
                    fail_msgs[_norm_failure(step.get("summary") or "")] += 1
                elif action in _USE_VERBS:
                    used = True
            if used:
                interacted += 1

    lines.append(
        f"  {total} trials: reachable {_pct(reachable, total)}, "
        f"finished {_pct(finished, total)}, "
        f"actually operated the app {_pct(interacted, total)}"
    )
    lines.append(f"  steps/trial: {_dist(steps_per_trial)}")
    for app_type, vals in sorted(by_app_type.items()):
        lines.append(f"    {app_type:<16} {_dist(vals)}")
    lines.append("  verb            calls   failed")
    for verb, count in verb_total.most_common():
        lines.append(
            f"    {verb:<14}{count:>6}  {verb_failed[verb]:>5} "
            f"({_pct(verb_failed[verb], count)})"
        )
    if fail_msgs:
        lines.append("  top failure messages:")
        for msg, count in fail_msgs.most_common(8):
            lines.append(f"    x{count:<4} {msg[:96]}")
    return lines


def _norm_failure(text: str) -> str:
    """Collapse a failure message to its shape so they can be counted."""
    text = " ".join(text.split())
    text = re.sub(r"\d+", "N", text)
    return text[:120]


# --------------------------------------------------------------------------- #
# social
# --------------------------------------------------------------------------- #


def social_section(runs: list[RunView]) -> list[str]:
    lines = ["", "SOCIAL (the feed)"]
    action_by_round: dict[int, Counter[str]] = defaultdict(Counter)
    comments_on_launch = comments_total = 0
    mentions = texts = 0
    named_someone = 0
    uniform_feeds = feeds_seen = 0
    orig_posts: list[int] = []
    authors_per_run: list[int] = []

    for run in runs:
        exposure = (run.summary.get("engagement") or {}).get("exposure") or {}
        if exposure:
            feeds_seen += 1
            uniform_feeds += 1 if exposure.get("uniform") else 0
        con = run.db()
        if con is None:
            continue
        try:
            for user_id, created_at, action in con.execute(
                "SELECT user_id, created_at, action FROM trace"
            ):
                del user_id
                action_by_round[int(created_at or 0)][action] += 1
            launch = run.summary.get("launch_post_id") or 1
            for (post_id,) in con.execute("SELECT post_id FROM comment"):
                comments_total += 1
                comments_on_launch += 1 if post_id == launch else 0
            authors = set()
            for user_id, content in con.execute(
                "SELECT user_id, content FROM post UNION ALL "
                "SELECT user_id, content FROM comment"
            ):
                authors.add(user_id)
                texts += 1
                found = _MENTION.findall(content or "")
                mentions += len(found)
                named_someone += 1 if found else 0
            authors_per_run.append(len(authors))
            orig_posts.append(
                con.execute(
                    "SELECT COUNT(*) FROM post WHERE original_post_id IS NULL"
                ).fetchone()[0]
            )
        except sqlite3.Error:
            pass
        finally:
            con.close()

    lines.append(
        f"  exposure: {uniform_feeds}/{feeds_seen} runs gave every agent a "
        f"byte-identical feed"
    )
    on_launch = _pct(comments_on_launch, comments_total)
    lines.append(
        f"  comments: {comments_total}, of which {on_launch} hang off the launch "
        f"post (a flat feed has no conversation)"
    )
    lines.append(
        f"  texts: {texts}, {_pct(named_someone, texts)} name another agent "
        f"({mentions} @mentions)"
    )
    lines.append(f"  original posts per run: {_dist(orig_posts)}")
    lines.append(f"  distinct authors per run: {_dist(authors_per_run)}")
    rounds = sorted(action_by_round)
    verbs = sorted({v for c in action_by_round.values() for v in c})
    header = "  round " + "".join(f"{v[:11]:>12}" for v in verbs)
    lines.append(header)
    for rnd in rounds:
        row = f"  {rnd:<6}" + "".join(f"{action_by_round[rnd][v]:>12}" for v in verbs)
        lines.append(row)
    return lines


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #


def verdicts_section(runs: list[RunView]) -> list[str]:
    lines = ["", "VERDICTS"]
    by_tier: dict[str, list[int]] = defaultdict(list)
    adopt_by_tier: dict[str, list[bool]] = defaultdict(list)
    facet_vectors: Counter[tuple] = Counter()
    craft_vals: list[float] = []
    delights: list[int] = []
    checks: dict[str, list] = {"work_survived": [], "saw_other_users": []}
    unanimous = scored = conv_yes = conv_total = 0

    for run in runs:
        conv = (run.summary.get("verdicts") or {}).get("conversion") or {}
        conv_total += int(conv.get("n_latecomers") or 0)
        conv_yes += int(conv.get("n_converted") or 0)
        interviews = (run.summary.get("verdicts") or {}).get("interviews") or {}
        rows = interviews.get("per_agent") or []
        uses = [r.get("would_use") for r in rows if r.get("would_use") is not None]
        if uses:
            scored += 1
            unanimous += 1 if len(set(uses)) == 1 else 0
        for row in rows:
            tier = row.get("tier", "?")
            if row.get("delight") is not None:
                by_tier[tier].append(row["delight"])
                delights.append(row["delight"])
            if row.get("would_use") is not None:
                adopt_by_tier[tier].append(bool(row["would_use"]))
        for row in ((run.summary.get("verdicts") or {}).get("triers") or {}).get(
            "per_agent"
        ) or []:
            vec = tuple(
                row.get(f)
                for f in ("functionality", "usability", "design", "simplicity")
            )
            if any(v is not None for v in vec):
                facet_vectors[vec] += 1
            if row.get("craft") is not None:
                craft_vals.append(row["craft"])
            for name in checks:
                checks[name].append(row.get(name))

    lines.append(f"  interview delight: {_dist([float(d) for d in delights])}")
    for tier in sorted(by_tier):
        lines.append(
            f"    {tier:<9} delight {statistics.fmean(by_tier[tier]):.2f} "
            f"adoption {_pct(sum(adopt_by_tier[tier]), len(adopt_by_tier[tier]))} "
            f"(n={len(by_tier[tier])})"
        )
    lines.append(f"  trial craft: {_dist(craft_vals, '.2f')}")
    for name in ("work_survived", "saw_other_users"):
        checked = sum(1 for v in checks[name] if v is not None)
        yes = sum(1 for v in checks[name] if v)
        lines.append(
            f"  {name}: checked by {checked} of {len(checks[name])} trials, "
            f"yes {_pct(yes, checked)}"
        )
    total_vecs = sum(facet_vectors.values())
    if total_vecs:
        top, count = facet_vectors.most_common(1)[0]
        lines.append(
            f"  facet vectors: {len(facet_vectors)} distinct over {total_vecs} "
            f"verdicts; most common {top} is {_pct(count, total_vecs)}"
        )
    lines.append(f"  runs with a unanimous would_use: {unanimous}/{scored}")
    if conv_total:
        lines.append(
            f"  latecomers the feed converted: {conv_yes}/{conv_total} "
            f"({_pct(conv_yes, conv_total)}) -- virality earned, not computed"
        )
    return lines


# --------------------------------------------------------------------------- #
# separation
# --------------------------------------------------------------------------- #


def _auc(pos: list[float], neg: list[float]) -> float | None:
    """P(a random positive outranks a random negative). 0.5 is no signal."""
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _signals(run: RunView) -> dict[str, float]:
    """The per-run quantities worth checking for discriminating power."""
    summary = run.summary
    interviews = (summary.get("verdicts") or {}).get("interviews") or {}
    triers = (summary.get("verdicts") or {}).get("triers") or {}
    engagement = summary.get("engagement") or {}
    reach = engagement.get("reach") or {}
    exposed = float(reach.get("exposed_agents") or 0) or 1.0
    out = {
        "adoption": interviews.get("would_use_rate"),
        "advocacy": interviews.get("would_share_rate"),
        "delight": interviews.get("delight_mean"),
        "delight_sd": interviews.get("delight_stdev"),
        "craft": triers.get("craft_mean"),
        "functionality": triers.get("functionality_mean"),
        "design": triers.get("design_mean"),
        "trial_delight": triers.get("delight_mean"),
        "like_rate": reach.get("actors_liked", 0) / exposed,
        "repost_rate": reach.get("actors_reposted", 0) / exposed,
        "comment_rate": reach.get("actors_commented", 0) / exposed,
        "negative_rate": reach.get("actors_negative", 0) / exposed,
        "secondary_share": (engagement.get("cascade") or {}).get("secondary_share"),
        "work_survived": triers.get("work_survived_rate"),
        "saw_other_users": triers.get("saw_other_users_rate"),
        "distinct_feeds": (engagement.get("exposure") or {}).get("distinct_feeds"),
        "conversion": ((summary.get("verdicts") or {}).get("conversion") or {}).get(
            "conversion_rate"
        ),
    }
    return {k: float(v) for k, v in out.items() if isinstance(v, int | float)}


def separate_section(runs: list[RunView], model_a: str, model_b: str) -> list[str]:
    lines = ["", "SEPARATION (per signal)"]
    a = [_signals(r) for r in runs if r.model == model_a and not r.is_control]
    b = [_signals(r) for r in runs if r.model == model_b and not r.is_control]
    control = [_signals(r) for r in runs if r.is_control]
    real = a + b
    keys = sorted({k for s in (a + b + control) for k in s})
    lines.append(
        f"  A={model_a} n={len(a)}   B={model_b} n={len(b)}   control n={len(control)}"
    )
    lines.append(
        f"  {'signal':<16}{'A mean':>9}{'B mean':>9}{'A-B':>9}{'AUC A>B':>9}"
        f"{'ctrl':>9}{'AUC real>ctrl':>15}"
    )
    for key in keys:
        va = [s[key] for s in a if key in s]
        vb = [s[key] for s in b if key in s]
        vc = [s[key] for s in control if key in s]
        vr = [s[key] for s in real if key in s]
        if not va or not vb:
            continue
        auc_ab = _auc(va, vb)
        auc_rc = _auc(vr, vc)
        lines.append(
            f"  {key:<16}{statistics.fmean(va):>9.3f}{statistics.fmean(vb):>9.3f}"
            f"{statistics.fmean(va) - statistics.fmean(vb):>9.3f}"
            f"{(f'{auc_ab:.2f}' if auc_ab else '-'):>9}"
            f"{(f'{statistics.fmean(vc):.3f}' if vc else '-'):>9}"
            f"{(f'{auc_rc:.2f}' if auc_rc else '-'):>15}"
        )
    return lines


# --------------------------------------------------------------------------- #
# samples
# --------------------------------------------------------------------------- #


def samples_section(runs: list[RunView], n: int) -> list[str]:
    lines = ["", "SAMPLES (verbatim)"]
    for run in runs[:n]:
        lines.append(f"  --- {run.label} {run.build_id[:40]}")
        con = run.db()
        if con is None:
            continue
        try:
            for (content,) in con.execute(
                "SELECT content FROM post WHERE post_id > 1 LIMIT 3"
            ):
                lines.append(f"    POST: {' '.join((content or '').split())[:200]}")
            for (content,) in con.execute("SELECT content FROM comment LIMIT 4"):
                lines.append(f"    CMT : {' '.join((content or '').split())[:200]}")
        except sqlite3.Error:
            pass
        finally:
            con.close()
        rows = ((run.summary.get("verdicts") or {}).get("interviews") or {}).get(
            "per_agent"
        ) or []
        for row in rows[:3]:
            lines.append(
                f"    WHY : [{row.get('tier', '?')} use={row.get('would_use')} "
                f"d={row.get('delight')}] {str(row.get('why', ''))[:180]}"
            )
        for payload in run.traces[:1]:
            trace = (payload.get("trace") or {}).get("steps") or []
            path = " -> ".join(s.get("action", "?") for s in trace)
            lines.append(f"    TRIAL {payload.get('persona', '?')}: {path[:220]}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default=CROWD_ARCH_VERSION)
    parser.add_argument("--section", default=",".join(SECTIONS))
    parser.add_argument("--build", default="", help="substring filter on build id")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--scores",
        action="store_true",
        help="also re-score the fleet and print per-cell ViralScores",
    )
    args = parser.parse_args(argv)

    wanted = [s.strip() for s in args.section.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        print(f"unknown section(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    runs = load_runs(
        REPO / "builds",
        arch=args.arch,
        build_filter=args.build,
        with_traces="trials" in wanted or "samples" in wanted,
    )
    out = [f"crowd forensics: {len(runs)} runs at arch v{args.arch}"]
    if not runs:
        print("\n".join(out))
        return 0
    if "health" in wanted:
        out += health_section(runs)
    if "trials" in wanted:
        out += trials_section(runs)
    if "social" in wanted:
        out += social_section(runs)
    if "verdicts" in wanted:
        out += verdicts_section(runs)
    if "separate" in wanted:
        out += separate_section(runs, CURRENT_FLEET.model_a, CURRENT_FLEET.model_b)
    if "samples" in wanted:
        out += samples_section(runs, args.samples)
    if args.scores:
        corpus = load_corpus(REPO, spec=CURRENT_FLEET)
        cells: dict[tuple, list[float]] = defaultdict(list)
        for run in corpus.fleet_runs():
            cells[(run.idea_id, run.model)].append(run.score)
        out.append("")
        out.append("SCORES (per cell mean)")
        for (idea, model), scores in sorted(cells.items()):
            out.append(
                f"  {idea:<28}{model:<26}{statistics.fmean(scores):>7.1f} "
                f"(n={len(scores)})"
            )
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
