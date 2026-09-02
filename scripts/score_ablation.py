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

"""Re-score the frozen corpus under many ViralScore definitions, and compare.

The crowd half of an ablation costs an hour a variable. The scoring half costs
nothing: a score is a pure function of artifacts already on disk, so every
weighting and every formula change can be measured against the *same* 150 runs
in seconds. That asymmetry is the whole reason to keep the two stages separate,
and it means there is no excuse for shipping a weight nobody has priced.

Each variant changes ONE thing. They are judged on four numbers, in this order:

1. **null** -- what the instrument reports for a model against ITSELF (seeds
   split by parity). A definition that separates a model from itself is
   measuring noise, and nothing else about it matters.
2. **ratio** -- between-build spread over within-build noise. Whether the score
   can rank apps at all.
3. **control margin** -- median working build minus the broken control. Whether
   it can still see a corpse.
4. **gap** -- the model difference, in points. Listed last on purpose: a
   variant that widens the gap while worsening 1-3 has found a scale, not a
   difference.

Usage::

    scripts/score_ablation.py                 # every variant
    scripts/score_ablation.py --only drop_reception,equal_weights
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from viral_bench.score.fleet import (  # noqa: E402
    CURRENT_FLEET,
    FleetCorpus,
    ScoredRun,
    control_separation,
    load_builds,
    paired_gap,
    self_separation,
    structure_name,
    within_cell_sd,
)
from viral_bench.score.signals import RunSignals, extract_signals  # noqa: E402
from viral_bench.score.spread import (  # noqa: E402
    SpreadSignals,
    extract_spread,
    spread_score,
)
from viral_bench.score.viralscore import ScoreWeights, score_run  # noqa: E402

AUTORATER = {"substance": 0.05, "severity": 0.05, "word_of_mouth": 0.05}

#: A variant is (weights, component-override). The override receives the run's
#: signals plus its raw summary and may add or replace components, which is how
#: a formula change is expressed without editing the scorer.
Override = Callable[[RunSignals, dict, dict], None] | None


def _W_STD() -> ScoreWeights:
    """The shipped v4_earned weights, so a formula variant changes only formula."""
    return _w(
        adoption=0.18,
        advocacy=0.18,
        craft=0.22,
        amplification=0.15,
        persistence=0.12,
        autorater=dict(AUTORATER),
    )


def _w(**kw) -> ScoreWeights:
    base = dict(
        adoption=0.0,
        advocacy=0.0,
        craft=0.0,
        amplification=0.0,
        reception=0.0,
        persistence=0.0,
        cascade=0.0,
        autorater={},
    )
    base.update(kw)
    return ScoreWeights(**base)


def _craft_without_delight(sig: RunSignals, summary: dict, comp: dict) -> None:
    """craft = mean of the four facets, with the overall item taken out.

    ``TrialVerdict.craft`` averages five items and one of them is ``delight``,
    the overall judgement. That makes craft partly a restatement of the same
    number the interview asks for, so the score counts one opinion twice.
    """
    rows = ((summary.get("verdicts") or {}).get("triers") or {}).get("per_agent") or []
    vals = [
        statistics.fmean(v)
        for v in (
            [
                r[f]
                for f in ("functionality", "usability", "design", "simplicity")
                if r.get(f) is not None
            ]
            for r in rows
            if r.get("app_reachable") is True and r.get("finished")
        )
        if v
    ]
    comp["craft"] = statistics.fmean(vals) / 10.0 if vals else None


def _advocacy_unweighted(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Advocacy without the sqrt(influence) weighting."""
    comp["advocacy"] = sig.advocacy_unweighted


def _resonance(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Adoption among the people the app is FOR, not the whole crowd."""
    comp["adoption"] = sig.resonance_adoption
    comp["advocacy"] = sig.resonance_advocacy


def _survived(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Did the crowd's work persist -- behaviour, not opinion."""
    triers = (summary.get("verdicts") or {}).get("triers") or {}
    comp["reception"] = triers.get("work_survived_rate")


def _delight_component(sig: RunSignals, summary: dict, comp: dict) -> None:
    comp["reception"] = (
        sig.delight_mean / 10.0 if sig.delight_mean is not None else None
    )


def _persistence_slot(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Put "did the app keep the user's work" in the (unused) cascade slot.

    ``cascade`` has carried weight 0.0 since it was measured to be degenerate,
    so it is a free named slot for pricing a component the score does not have.
    Persistence is the one behavioural quality signal nothing else captures: an
    agent made something, reloaded, and looked.
    """
    triers = (summary.get("verdicts") or {}).get("triers") or {}
    comp["cascade"] = triers.get("work_survived_rate")


def _consensus(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Reward agreement: a crowd that splits has not been convinced."""
    sd = sig.delight_stdev
    comp["reception"] = None if sd is None else max(0.0, 1.0 - sd / 5.0)


def _trier_rows(summary: dict) -> list[dict]:
    """Hands-on rows that are real evidence (finished, reached the app)."""
    rows = ((summary.get("verdicts") or {}).get("triers") or {}).get("per_agent") or []
    return [r for r in rows if r.get("app_reachable") is True and r.get("finished")]


def _facet_only(name: str):
    """craft = one facet alone. Which facet is craft made of?"""

    def override(sig: RunSignals, summary: dict, comp: dict) -> None:
        vals = [r[name] for r in _trier_rows(summary) if r.get(name) is not None]
        comp["craft"] = statistics.fmean(vals) / 10.0 if vals else None

    return override


def _craft_worst_facet(sig: RunSignals, summary: dict, comp: dict) -> None:
    """craft = each trier's WORST facet, averaged.

    A product is often judged by its weakest part, and a mean over four facets
    lets a beautiful broken thing score like a plain working one.
    """
    vals = []
    for r in _trier_rows(summary):
        facets = [
            r[f]
            for f in ("functionality", "usability", "design", "simplicity")
            if r.get(f) is not None
        ]
        if facets:
            vals.append(min(facets))
    comp["craft"] = statistics.fmean(vals) / 10.0 if vals else None


def _craft_median(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Median trier instead of mean: robust to one enthusiast or one hater."""
    vals = [r["craft"] for r in _trier_rows(summary) if r.get("craft") is not None]
    comp["craft"] = statistics.median(vals) / 10.0 if vals else None


def _adoption_strict(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Adoption counts only agents who would BOTH use and share it."""
    rows = ((summary.get("verdicts") or {}).get("interviews") or {}).get(
        "per_agent"
    ) or []
    vals = [
        bool(r.get("would_use")) and bool(r.get("would_share"))
        for r in rows
        if r.get("would_use") is not None
    ]
    comp["adoption"] = sum(vals) / len(vals) if vals else None


def _delight_p25(sig: RunSignals, summary: dict, comp: dict) -> None:
    """The unimpressed quartile. A launch is killed by its detractors."""
    rows = ((summary.get("verdicts") or {}).get("interviews") or {}).get(
        "per_agent"
    ) or []
    vals = sorted(r["delight"] for r in rows if r.get("delight") is not None)
    if not vals:
        comp["craft"] = None
        return
    comp["craft"] = vals[max(0, int(0.25 * (len(vals) - 1)))] / 10.0


def _amplification_log(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Diminishing returns on reposts, the way real reach saturates."""
    value = sig.repost_participation
    comp["amplification"] = (
        None if value is None else math.log1p(9.0 * value) / math.log(10.0)
    )


def _advocacy_linear(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Influence weighted linearly rather than by its square root."""
    rows = ((summary.get("verdicts") or {}).get("interviews") or {}).get(
        "per_agent"
    ) or []
    influence = {c["agent_id"]: c.get("influence", 1) for c in summary.get("crowd", [])}
    num = den = 0.0
    for r in rows:
        if r.get("would_share") is None:
            continue
        w = max(1, influence.get(r.get("agent_id"), 1))
        den += w
        num += w if r["would_share"] else 0.0
    comp["advocacy"] = (num / den) if den else None


def _spread_for(summary: dict) -> SpreadSignals | None:
    """The run's propagation signals, or None when the DB is unreadable."""
    crowd_dir = summary.get("_crowd_dir")
    if not crowd_dir:
        return None
    sig = extract_spread(crowd_dir)
    return sig if sig.ok else None


def _amplification_peer_only(sig: RunSignals, summary: dict, comp: dict) -> None:
    """amplification = reposts OF ANOTHER AGENT, not of the announcement.

    ``actors_reposted`` counts anyone who reposted anything, and 67.8% of those
    actors only ever reposted the founder's launch post. That is the crowd
    forwarding an advert, which every agent can do without reading a word any
    other agent wrote. This keeps only the part that requires a peer.
    """
    spread = _spread_for(summary)
    comp["amplification"] = spread.peer_repost_participation if spread else None


def _amplification_peer_broad(sig: RunSignals, summary: dict, comp: dict) -> None:
    """As above, but a quote counts too -- repost with a stated reason.

    Quoting has the highest peer share of any amplification verb (33.5%), so
    dropping it throws away the most peer-directed evidence there is.
    """
    spread = _spread_for(summary)
    if spread is None:
        comp["amplification"] = None
        return
    parts = [
        v
        for v in (
            spread.peer_repost_participation,
            spread.peer_quote_participation,
        )
        if v is not None
    ]
    comp["amplification"] = min(1.0, sum(parts)) if parts else None


def _amplification_advocate(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Peer amplification restricted to agents who said they would share it.

    The principled version of the two above. Raw peer amplification runs 0.442
    on broken apps against 0.121 on working ones -- thirty agents piling on to
    confirm the same 500 -- so it pays a corpse for the noise it generates.
    Requiring the amplifier's own verdict to be ``would_share=yes`` flips the
    sign (r -0.483 -> +0.280) and is what a repost is supposed to mean.
    """
    spread = _spread_for(summary)
    if spread is None:
        comp["amplification"] = None
        return
    parts = [
        v
        for v in (
            spread.advocate_repost_participation,
            spread.advocate_quote_participation,
        )
        if v is not None
    ]
    comp["amplification"] = min(1.0, sum(parts)) if parts else None


def _advocacy_spread_with_comments(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Fold advocate peer COMMENTING into the shipped advocacy_spread term.

    The shipped term counts reposts and quotes only, and is zero in 55% of runs.
    Some of that floor is a narrow verb rather than a real absence: in 28% of the
    zero-repost runs an endorsing agent did engage a peer, by commenting on their
    post. Adding it drops the zero rate 55% -> 40%. The other 72% are true zeros
    -- nobody who endorsed the app engaged a peer at all -- which is a fact about
    the crowd, not something a wider definition can fix.
    """
    spread = _spread_for(summary)
    if spread is None:
        comp["advocacy_spread"] = None
        return
    parts = [
        v
        for v in (
            spread.advocate_repost_participation,
            spread.advocate_quote_participation,
            spread.advocate_comment_participation,
        )
        if v is not None
    ]
    comp["advocacy_spread"] = min(1.0, sum(parts)) if parts else None


def _cascade_strict(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Put TRUE second-generation spread in the (unused) cascade slot.

    ``signals.secondary_share`` buckets every agent's own post as "primary"
    alongside the founder's launch post, so 46.7% of its denominator is
    peer-authored first-generation content and the term drifts upward without
    any cascade occurring. This is the strict version: engagement landing on a
    repost or a quote, as a share of all post engagement.
    """
    spread = _spread_for(summary)
    comp["cascade"] = spread.derived_engagement_share if spread else None


def _spread_axis(sig: RunSignals, summary: dict, comp: dict) -> None:
    """Put the whole propagation index in the cascade slot, to price it.

    ``cascade`` has carried weight 0.0 since it was measured degenerate, which
    makes it the free named slot for costing a component the score does not yet
    have -- the same trick ``_persistence_slot`` uses.
    """
    spread = _spread_for(summary)
    value = spread_score(spread) if spread else None
    comp["cascade"] = None if value is None else value / 100.0


VARIANTS: dict[str, tuple[ScoreWeights, Override]] = {
    # The live default, loaded from config rather than restated here, so this row
    # moves when the shipped definition moves. It must reproduce
    # `amplification_advocate` below: that one reaches simulation.db through an
    # override, this one through signals.py, and if they ever disagree the
    # wiring is losing the signal somewhere between them.
    "shipped_v5_advocacy": (ScoreWeights.from_profile("v5_advocacy"), None),
    "prev_v4_earned": (_W_STD(), None),
    "prev_v3_valence": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.09,
            reception=0.18,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    # -- one component removed at a time (weight spread over the rest) -------
    "drop_reception": (
        _w(
            adoption=0.20,
            advocacy=0.20,
            craft=0.25,
            amplification=0.20,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    "drop_amplification": (
        _w(
            adoption=0.22,
            advocacy=0.22,
            craft=0.26,
            reception=0.15,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    "drop_craft": (
        _w(
            adoption=0.25,
            advocacy=0.25,
            amplification=0.15,
            reception=0.20,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    "drop_adoption": (
        _w(
            advocacy=0.30,
            craft=0.30,
            amplification=0.10,
            reception=0.15,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    "drop_advocacy": (
        _w(
            adoption=0.30,
            craft=0.30,
            amplification=0.10,
            reception=0.15,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    "drop_autorater": (
        _w(
            adoption=0.22, advocacy=0.22, craft=0.26, amplification=0.11, reception=0.19
        ),
        None,
    ),
    # -- one component at a time, alone --------------------------------------
    "only_craft": (_w(craft=1.0), None),
    "only_adoption": (_w(adoption=1.0), None),
    "only_amplification": (_w(amplification=1.0), None),
    "only_autorater": (
        _w(autorater={k: 1 / 3 for k in AUTORATER}),
        None,
    ),
    # -- reweightings ---------------------------------------------------------
    "equal_five": (
        _w(
            adoption=0.17,
            advocacy=0.17,
            craft=0.17,
            amplification=0.17,
            reception=0.17,
            autorater={k: 0.05 for k in AUTORATER},
        ),
        None,
    ),
    "behaviour_heavy": (
        _w(
            adoption=0.12,
            advocacy=0.12,
            craft=0.16,
            amplification=0.30,
            reception=0.15,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    "judgement_heavy": (
        _w(
            adoption=0.28,
            advocacy=0.28,
            craft=0.24,
            amplification=0.05,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    "autorater_heavy": (
        _w(
            adoption=0.15,
            advocacy=0.15,
            craft=0.15,
            amplification=0.10,
            autorater={k: 0.15 for k in AUTORATER},
        ),
        None,
    ),
    # -- formula changes ------------------------------------------------------
    "craft_no_delight": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.09,
            reception=0.18,
            autorater=dict(AUTORATER),
        ),
        _craft_without_delight,
    ),
    "advocacy_unweighted": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.09,
            reception=0.18,
            autorater=dict(AUTORATER),
        ),
        _advocacy_unweighted,
    ),
    "in_audience_only": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.09,
            reception=0.18,
            autorater=dict(AUTORATER),
        ),
        _resonance,
    ),
    "persistence_for_reception": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.09,
            reception=0.18,
            autorater=dict(AUTORATER),
        ),
        _survived,
    ),
    "delight_for_reception": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.09,
            reception=0.18,
            autorater=dict(AUTORATER),
        ),
        _delight_component,
    ),
    "consensus_for_reception": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.09,
            reception=0.18,
            autorater=dict(AUTORATER),
        ),
        _consensus,
    ),
    # -- what IS craft made of? ----------------------------------------------
    "craft_functionality_only": (_W_STD(), _facet_only("functionality")),
    "craft_usability_only": (_W_STD(), _facet_only("usability")),
    "craft_design_only": (_W_STD(), _facet_only("design")),
    "craft_simplicity_only": (_W_STD(), _facet_only("simplicity")),
    "craft_worst_facet": (_W_STD(), _craft_worst_facet),
    "craft_median_trier": (_W_STD(), _craft_median),
    "craft_delight_p25": (_W_STD(), _delight_p25),
    # -- stricter / softer readings of the crowd ------------------------------
    "adoption_use_and_share": (_W_STD(), _adoption_strict),
    "advocacy_linear_influence": (_W_STD(), _advocacy_linear),
    "amplification_log": (_W_STD(), _amplification_log),
    # -- weight on the behavioural quality term -------------------------------
    "persistence_heavy": (
        _w(
            adoption=0.15,
            advocacy=0.15,
            craft=0.20,
            amplification=0.10,
            persistence=0.25,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    "persistence_none": (
        _w(
            adoption=0.20,
            advocacy=0.20,
            craft=0.25,
            amplification=0.20,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    # -- which half of the evidence matters -----------------------------------
    "interview_only": (
        _w(adoption=0.42, advocacy=0.43, autorater=dict(AUTORATER)),
        None,
    ),
    "trial_only": (
        _w(craft=0.55, persistence=0.30, autorater=dict(AUTORATER)),
        None,
    ),
    # -- the gate -------------------------------------------------------------
    "gate_hard": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.09,
            reception=0.18,
            autorater=dict(AUTORATER),
            gate_broken_app=0.0,
            gate_failed_self_check=0.3,
        ),
        None,
    ),
    "gate_soft_05": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.09,
            reception=0.18,
            autorater=dict(AUTORATER),
            gate_broken_app=0.5,
            gate_failed_self_check=0.8,
        ),
        None,
    ),
    # -- the validity gate as a POLICY, not merely a multiplier -------------
    #
    # The gate is the only part of the score that is not the crowd's opinion.
    # `evidence_graded` scales the floor up towards 1.0 by the fraction of the
    # crowd that got the app working, so a container probe can no longer
    # overrule thirty agents who used it. See scripts/gate_sweep.py for the
    # 64-combination sweep that picked the floor.
    "gate_witnessed": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.15,
            persistence=0.12,
            autorater=dict(AUTORATER),
            gate_broken_app=0.1,
            gate_failed_self_check=0.55,
            gate_policy="evidence_graded",
        ),
        None,
    ),
    "gate_witnessed_floor02": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.15,
            persistence=0.12,
            autorater=dict(AUTORATER),
            gate_broken_app=0.2,
            gate_failed_self_check=0.6,
            gate_policy="evidence_graded",
        ),
        None,
    ),
    # -- the best single-change candidates, combined -------------------------
    "judgement_heavy_delight": (
        _w(
            adoption=0.25,
            advocacy=0.25,
            craft=0.25,
            amplification=0.10,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    "amp30": (
        _w(
            adoption=0.15,
            advocacy=0.15,
            craft=0.20,
            amplification=0.30,
            reception=0.05,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    "amp05": (
        _w(
            adoption=0.20,
            advocacy=0.20,
            craft=0.25,
            amplification=0.05,
            reception=0.15,
            autorater=dict(AUTORATER),
        ),
        None,
    ),
    "no_reception_add_persistence": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.15,
            cascade=0.12,
            autorater=dict(AUTORATER),
        ),
        _persistence_slot,
    ),
    "no_reception_persistence_light": (
        _w(
            adoption=0.19,
            advocacy=0.19,
            craft=0.24,
            amplification=0.17,
            cascade=0.06,
            autorater=dict(AUTORATER),
        ),
        _persistence_slot,
    ),
    "gate_off": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            amplification=0.09,
            reception=0.18,
            autorater=dict(AUTORATER),
            gate_broken_app=1.0,
            gate_failed_self_check=1.0,
        ),
        None,
    ),
    # -- propagation: is spread separable from quality? -----------------------
    # These do not try to win the ladder. They ask whether the social terms are
    # measuring the crowd or measuring the app, which the ladder cannot see: a
    # variant can score identically to the shipped one and still be reading a
    # completely different thing.
    "amplification_peer_only": (_W_STD(), _amplification_peer_only),
    "amplification_peer_broad": (_W_STD(), _amplification_peer_broad),
    "amplification_advocate": (_W_STD(), _amplification_advocate),
    "v5_plus_advocate_comments": (
        ScoreWeights.from_profile("v5_advocacy"),
        _advocacy_spread_with_comments,
    ),
    # The advocate term is sparse -- zero in 56% of runs -- so at 0.15 it acts
    # partly as a flat penalty on runs where no endorser amplified anyone. Price
    # the weight rather than inheriting the one tuned for a term with a mean of
    # 0.609 and no zeroes.
    "amplification_advocate_08": (
        _w(
            adoption=0.19,
            advocacy=0.19,
            craft=0.24,
            amplification=0.08,
            persistence=0.15,
            autorater=dict(AUTORATER),
        ),
        _amplification_advocate,
    ),
    "amplification_advocate_22": (
        _w(
            adoption=0.16,
            advocacy=0.16,
            craft=0.21,
            amplification=0.22,
            persistence=0.10,
            autorater=dict(AUTORATER),
        ),
        _amplification_advocate,
    ),
    "cascade_strict_05": (
        _w(
            adoption=0.17,
            advocacy=0.17,
            craft=0.21,
            amplification=0.14,
            persistence=0.11,
            cascade=0.05,
            autorater=dict(AUTORATER),
        ),
        _cascade_strict,
    ),
    "spread_axis_05": (
        _w(
            adoption=0.17,
            advocacy=0.17,
            craft=0.21,
            amplification=0.14,
            persistence=0.11,
            cascade=0.05,
            autorater=dict(AUTORATER),
        ),
        _spread_axis,
    ),
    "spread_axis_15": (
        _w(
            adoption=0.15,
            advocacy=0.15,
            craft=0.20,
            amplification=0.13,
            persistence=0.07,
            cascade=0.15,
            autorater=dict(AUTORATER),
        ),
        _spread_axis,
    ),
    # Replace amplification outright, rather than adding a term next to it.
    "spread_replaces_amplification": (
        _w(
            adoption=0.18,
            advocacy=0.18,
            craft=0.22,
            persistence=0.12,
            cascade=0.15,
            autorater=dict(AUTORATER),
        ),
        _spread_axis,
    ),
}


def score_corpus_with(
    weights: ScoreWeights, override: Override, arch: str
) -> FleetCorpus:
    """Re-score every run of ``arch`` under one definition."""
    from viral_bench.score.fleet import _load_autorating, fleet_replicates

    builds_root = REPO / "builds"
    builds = load_builds(builds_root)
    reps = fleet_replicates(builds_root)
    from dataclasses import replace as dc_replace

    for bid, rep in reps.items():
        if bid in builds:
            builds[bid] = dc_replace(builds[bid], replicate=rep)
    runs: list[ScoredRun] = []
    for summary_path in sorted(builds_root.glob("*/*/run_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(summary.get("crowd_arch_version", "0")) != arch:
            continue
        build = builds.get(summary.get("build_id", ""))
        if build is None:
            continue
        crowd_dir = summary_path.parent
        try:
            sig = extract_signals(crowd_dir)
        except (FileNotFoundError, ValueError, KeyError):
            continue
        result = score_run(sig, weights, _load_autorating(crowd_dir))
        components = dict(result.components)
        if override is not None:
            # An override that needs the raw social graph -- anything peer- vs
            # founder-directed -- has to reach simulation.db, and the summary is
            # the only channel it has. `config.out_dir` records the path the run
            # was WRITTEN to, which is wrong the moment a corpus is copied, so
            # pass the directory that was opened.
            summary["_crowd_dir"] = str(crowd_dir)
            override(sig, summary, components)
            # Recombine by hand so an override changes the number.
            w = weights.as_dict()
            pairs = [
                (components[k], w[k])
                for k in w
                if components.get(k) is not None and w[k] > 0
            ]
            total = sum(x for _, x in pairs)
            raw = sum(v * x for v, x in pairs) / total if total else None
            score = round(100.0 * result.gate * raw, 1) if raw is not None else None
        else:
            score = result.score
        runs.append(
            ScoredRun(
                crowd_dir=str(crowd_dir),
                build_id=build.build_id,
                idea_id=build.idea_id,
                model=build.model,
                seed=int((summary.get("config") or {}).get("seed", 0) or 0),
                n_agents=int((summary.get("config") or {}).get("n_agents", 0) or 0),
                arch_version=arch,
                gate=float(result.gate),
                dead=sig.builds is False or sig.runs is False,
                undeliverable=bool(summary.get("undeliverable")),
                replicate=build.replicate,
                structure=structure_name(build.config),
                score=score,
                blockers=list(result.confidence),
                components=components,
            )
        )
    fleet_ids = {b for b in reps if b in builds and CURRENT_FLEET.wants(builds[b])}
    return FleetCorpus(builds=builds, runs=runs, fleet_ids=fleet_ids, arch_version=arch)


def restrict_seeds(corpus: FleetCorpus, seeds: set[int]) -> FleetCorpus:
    """A view of the corpus holding only the named seeds.

    Choosing a scoring profile on the same runs you then report it on is how a
    weighting gets overfitted to one corpus. Seeds are independent draws of the
    crowd, so fitting on {0,1} and reporting on {2} is a real holdout that costs
    nothing.
    """
    return FleetCorpus(
        builds=corpus.builds,
        runs=[r for r in corpus.runs if r.seed in seeds],
        fleet_ids=corpus.fleet_ids,
        profile=corpus.profile,
        arch_version=corpus.arch_version,
    )


def stats_for(corpus: FleetCorpus) -> dict:
    A, B = CURRENT_FLEET.model_a, CURRENT_FLEET.model_b
    gap = paired_gap(corpus, A, B)
    na, nb = self_separation(corpus, A), self_separation(corpus, B)
    sep = control_separation(corpus)
    noise = within_cell_sd(corpus)
    means = [statistics.fmean(r.score for r in v) for v in corpus.cells().values() if v]
    worst_null = max(abs(na.gap or 0.0), abs(nb.gap or 0.0))
    return {
        "gap": gap.gap,
        "ci": (gap.ci_low, gap.ci_high),
        "wins_a": gap.wins_a,
        "null": worst_null,
        "noise": noise,
        "ratio": (
            round(statistics.pstdev(means) / noise, 2)
            if noise and len(means) > 1
            else None
        ),
        "control": sep.control_mean,
        "margin": (
            round(sep.working_median - sep.control_mean, 1)
            if sep.working_median is not None and sep.control_mean is not None
            else None
        ),
        "below": len(sep.below_control),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    parser.add_argument("--arch", default="")
    parser.add_argument(
        "--seeds", default="", help="comma-separated seeds to score (holdout)"
    )
    args = parser.parse_args(argv)
    from viral_bench.crowd.sim_defaults import CROWD_ARCH_VERSION

    arch = args.arch or CROWD_ARCH_VERSION

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(VARIANTS)
    seeds = {int(s) for s in args.seeds.split(",") if s.strip()}
    label = f" seeds={sorted(seeds)}" if seeds else ""
    print(f"SCORE ABLATIONS on arch v{arch}{label} (same runs, different definitions)")
    print(
        f"  {'variant':<28}{'null':>7}{'ratio':>7}{'margin':>8}{'ctrl':>7}"
        f"{'noise':>7}{'gap':>8}{'ci_lo':>8}{'ci_hi':>8}{'winA':>6}"
    )
    for name in names:
        weights, override = VARIANTS[name]
        weights.validate()
        corpus = score_corpus_with(weights, override, arch)
        if seeds:
            corpus = restrict_seeds(corpus, seeds)
        s = stats_for(corpus)

        def f(key, spec=".2f", s=s):
            return "-" if s[key] is None else format(s[key], spec)

        print(
            f"  {name:<28}{f('null'):>7}{f('ratio'):>7}{f('margin', '.1f'):>8}"
            f"{f('control', '.1f'):>7}{f('noise'):>7}{f('gap', '+.1f'):>8}"
            f"{s['ci'][0]:>8.1f}{s['ci'][1]:>8.1f}{s['wins_a']:>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
