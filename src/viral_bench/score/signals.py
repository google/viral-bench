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

"""Extract the raw signals a ViralScore is computed from.

Scoring is a **pure, offline function of the artifacts a crowd run already
wrote** -- ``run_summary.json`` plus the OASIS database. Nothing here re-runs a
simulation, which is what makes calibration affordable: when the weights change,
every historical run can simply be re-scored.

The extraction is deliberately forgiving. Runs recorded before a signal existed
(facet ratings, audience fit, the validity gate) still load, with the missing
pieces reported as ``None`` and surfaced as confidence warnings rather than
silently scored as zero -- a missing measurement is not the same as a bad one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from viral_bench.score.spread import extract_spread

#: Filename written by the crowd simulation.
RUN_SUMMARY = "run_summary.json"


@dataclass(frozen=True)
class RunSignals:
    """Everything the score reads from one crowd run, already normalised."""

    build_id: str
    crowd_dir: str

    # -- judgement: every agent, independently (the stable backbone) ---------
    n_interviews: int = 0
    adoption_rate: float | None = None  # unweighted fraction: would_use
    advocacy_rate: float | None = None  # influence-weighted fraction: would_share
    advocacy_unweighted: float | None = None
    delight_mean: float | None = None
    delight_stdev: float | None = None

    # -- hands-on craft: triers only, non-degraded reachable trials only ----
    n_valid_trials: int = 0
    n_trials: int = 0
    #: Trials whose agent never reached the running app. Excluded from craft and
    #: surfaced so "craft is unmeasured" can never look like "craft is fine".
    n_unreachable_trials: int = 0
    craft_mean: float | None = None  # 0..10, multi-item facet mean
    facets: dict = field(default_factory=dict)  # per-facet 0..10 means
    #: Fraction of the agents who CHECKED whose work survived a page reload.
    #: Behavioural, verifiable, and the one quality signal no rating captures:
    #: an agent made something, reloaded, and looked. ``None`` when nobody
    #: checked, in which case the component is re-weighted out rather than
    #: scored as a failure -- the crowd's incuriosity is not the app's defect.
    persistence_rate: float | None = None
    n_persistence_checked: int = 0

    # -- behaviour: what the crowd actually DID, per exposed agent ----------
    exposed_agents: int = 0
    repost_participation: float | None = None
    comment_participation: float | None = None
    like_participation: float | None = None
    negative_participation: float | None = None
    #: Reposts and quotes OF ANOTHER AGENT, by agents who themselves said they
    #: would share it. The honest version of ``repost_participation``, which
    #: counts anyone who reposted anything -- and 68% of those actors only ever
    #: reposted the founder's launch post, an act needing no contact with a peer.
    #: ``None`` when the run has no readable ``simulation.db``, so a corpus that
    #: predates this re-weights the component out instead of scoring it zero.
    #: See :mod:`viral_bench.score.spread` for the measurements behind it.
    advocate_amplification: float | None = None

    # -- cascade shape (measured; see ScoreWeights for why it is unweighted) -
    secondary_share: float | None = None
    late_action_share: float | None = None

    # -- audience fit: breadth vs resonance ---------------------------------
    audience_fit_rate: float | None = None
    resonance_adoption: float | None = None
    resonance_advocacy: float | None = None

    # -- gates and run integrity --------------------------------------------
    does_what_it_claims: bool | None = None
    builds: bool | None = None
    runs: bool | None = None
    validity_detail: str = ""
    requested_agents: int = 0
    actual_agents: int = 0
    clamped: bool = False
    rounds_ok: bool = True
    run_ok: bool = True
    interview_mode: str = ""


def _rate(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(bool(v) for v in vals) / len(vals), 4) if vals else None


def _influence_weighted_rate(
    rows: list[dict], key: str, influence: dict
) -> float | None:
    """Fraction saying yes, weighted by ``sqrt(influence)``.

    Advocacy is the viral coefficient: a share from a hub reaches far more people
    than one from a lurker, so it should not count the same. The square root
    damps that -- a single influence-10 account should tilt the score, not decide
    it, which keeps the measure from hinging on one agent's coin flip.
    """
    num = den = 0.0
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        weight = math.sqrt(max(1, influence.get(row.get("agent_id"), 1)))
        den += weight
        if value:
            num += weight
    return round(num / den, 4) if den else None


def load_run_summary(path: str | Path) -> dict:
    """Load ``run_summary.json`` from a crowd run directory or a direct path."""
    p = Path(path)
    if p.is_dir():
        p = p / RUN_SUMMARY
    if not p.is_file():
        raise FileNotFoundError(f"no {RUN_SUMMARY} at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def extract_signals(crowd_dir: str | Path) -> RunSignals:
    """Read one crowd run's artifacts into :class:`RunSignals`."""
    crowd_dir = Path(crowd_dir)
    summary = load_run_summary(crowd_dir)

    verdicts = summary.get("verdicts") or {}
    interviews = verdicts.get("interviews") or {}
    triers = verdicts.get("triers") or {}
    engagement = summary.get("engagement") or {}
    reach = engagement.get("reach") or {}
    cascade = engagement.get("cascade") or {}
    integrity = summary.get("crowd_integrity") or {}
    validity = summary.get("validity") or {}

    influence = {c["agent_id"]: c.get("influence", 1) for c in summary.get("crowd", [])}
    interview_rows = interviews.get("per_agent") or []

    # Hands-on craft: only a trial that finished AND is KNOWN to have reached a
    # working app is evidence about quality. An unfinished trial has no verdict;
    # a degraded one saw the app through a narrow window; an unreachable one
    # never saw it at all -- it rated the source tree, or nothing.
    #
    # The test is ``app_reachable is True``, not ``is not False``. "We never
    # established that this agent got the app to work" is missing evidence, and
    # missing evidence must not read as good evidence: 6 of 24 bot trials in one
    # sweep finished with a full craft verdict having never sent the bot a single
    # message. 36 of 314 stored trials were in the same category historically,
    # every one of them filed a confident four-facet verdict, and all 36 sat in a
    # single build -- which is precisely how a harness race gets reported as a
    # difference between founder models.
    trial_rows = triers.get("per_agent") or []
    valid_trials = [
        r
        for r in trial_rows
        if r.get("finished")
        and not r.get("degraded")
        and r.get("app_reachable") is True
    ]
    n_unreachable_trials = sum(
        1 for r in trial_rows if r.get("app_reachable") is not True
    )
    craft_vals = [r["craft"] for r in valid_trials if r.get("craft") is not None] or [
        r["delight"] for r in valid_trials if r.get("delight") is not None
    ]
    craft_mean = round(sum(craft_vals) / len(craft_vals), 3) if craft_vals else None

    facets = {}
    for facet in ("functionality", "usability", "design", "simplicity"):
        vals = [r[facet] for r in valid_trials if r.get(facet) is not None]
        if vals:
            facets[facet] = round(sum(vals) / len(vals), 3)

    exposed = int(reach.get("exposed_agents") or 0)

    def per_capita(key: str) -> float | None:
        value = reach.get(key)
        if value is None or not exposed:
            return None
        return round(min(1.0, value / exposed), 4)

    in_aud = interviews.get("in_audience") or {}

    # Peer-directed amplification lives in the OASIS database, not the summary:
    # the summary's `reach` block was aggregated before anyone thought to ask
    # who the repost was OF. Reading it here keeps scoring a pure function of
    # stored artifacts -- no re-simulation -- at the cost of one sqlite open.
    spread = extract_spread(crowd_dir)
    advocate = None
    if spread.ok:
        parts = [
            v
            for v in (
                spread.advocate_repost_participation,
                spread.advocate_quote_participation,
            )
            if v is not None
        ]
        advocate = min(1.0, sum(parts)) if parts else None

    return RunSignals(
        build_id=summary.get("build_id", ""),
        crowd_dir=str(crowd_dir),
        n_interviews=int(interviews.get("n") or 0),
        adoption_rate=interviews.get("would_use_rate"),
        advocacy_rate=_influence_weighted_rate(
            interview_rows, "would_share", influence
        ),
        advocacy_unweighted=interviews.get("would_share_rate"),
        delight_mean=interviews.get("delight_mean"),
        delight_stdev=interviews.get("delight_stdev"),
        n_valid_trials=len(valid_trials),
        n_trials=len(trial_rows),
        n_unreachable_trials=n_unreachable_trials,
        craft_mean=craft_mean,
        facets=facets,
        persistence_rate=triers.get("work_survived_rate"),
        n_persistence_checked=int(triers.get("work_survived_checked") or 0),
        exposed_agents=exposed,
        repost_participation=per_capita("actors_reposted"),
        comment_participation=per_capita("actors_commented"),
        like_participation=per_capita("actors_liked"),
        negative_participation=per_capita("actors_negative"),
        advocate_amplification=advocate,
        secondary_share=cascade.get("secondary_share"),
        late_action_share=cascade.get("late_action_share"),
        audience_fit_rate=interviews.get("audience_fit_rate"),
        resonance_adoption=in_aud.get("would_use_rate"),
        resonance_advocacy=in_aud.get("would_share_rate"),
        does_what_it_claims=validity.get("does_what_it_claims"),
        builds=validity.get("builds"),
        runs=validity.get("runs"),
        validity_detail=validity.get("detail", ""),
        requested_agents=int(integrity.get("requested_n_agents") or 0),
        actual_agents=int(
            integrity.get("actual_n_agents") or len(summary.get("crowd", []))
        ),
        clamped=bool(integrity.get("clamped")),
        rounds_ok=all(r.get("ok", True) for r in summary.get("rounds", [])),
        run_ok=bool(summary.get("ok")),
        interview_mode=(summary.get("interview_stats") or {}).get("mode", ""),
    )


def find_crowd_runs(build_id: str, builds_root: str | Path = "builds") -> list[Path]:
    """All crowd run directories for ``build_id``, oldest first."""
    root = Path(builds_root) / "crowd"
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.glob(f"{build_id}__crowd-*") if (d / RUN_SUMMARY).is_file()
    )
