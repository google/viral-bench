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

"""Adversarial stress tests for the scorer.

The scorer is the last stage in the pipeline and the only one whose output gets
published, so a crash or a silently-wrong number here is the most expensive kind
of defect: it lands in a comparison table with no indication anything went wrong.

These tests do not check that the score is *right* for a realistic run -- that is
what the calibration work covers. They check that it cannot be made to lie or
explode by inputs the pipeline can genuinely produce: a crowd where every agent
agreed, a run where nobody reached the app, a build that failed the validity
gate, rates at exactly 0 and exactly 1, and every optional signal missing at
once. Each of those has occurred in the stored corpus.
"""

from __future__ import annotations

import itertools
import math

import pytest

from viral_bench.score.signals import RunSignals
from viral_bench.score.viralscore import (
    ScoreWeights,
    compute_components,
    unscorable_reasons,
    validity_gate,
)


def _sig(**kw) -> RunSignals:
    base = {"build_id": "b", "crowd_dir": "/tmp/x"}
    base.update(kw)
    return RunSignals(**base)


def test_everything_missing_does_not_crash_and_scores_nothing() -> None:
    """A run that collected no signal must be unscorable, not zero.

    Zero is a claim about the app ("nobody liked it"); unmeasured is a claim
    about the run. Collapsing the second into the first would let a crashed
    crowd run masquerade as a terrible app.
    """
    sig = _sig()
    components = compute_components(sig)
    assert all(v is None for v in components.values()), components
    reasons = unscorable_reasons(sig, components)
    assert reasons, "a run with no signal at all must report why it cannot score"


def test_unanimous_crowd_is_scorable() -> None:
    """Unanimity is real and common -- would_use was unanimous in 26/45 runs.

    A zero-variance input must not produce NaN via a stdev-based path.
    """
    sig = _sig(
        n_interviews=30,
        adoption_rate=1.0,
        advocacy_rate=1.0,
        advocacy_unweighted=1.0,
        delight_mean=10.0,
        delight_stdev=0.0,
        n_valid_trials=8,
        n_trials=8,
        craft_mean=10.0,
        exposed_agents=30,
        repost_participation=1.0,
        comment_participation=1.0,
        like_participation=1.0,
        negative_participation=0.0,
    )
    components = compute_components(sig)
    for name, value in components.items():
        if value is None:
            continue
        assert not math.isnan(value), f"{name} is NaN on a unanimous crowd"
        assert 0.0 <= value <= 1.0, f"{name}={value} outside [0,1]"


def test_rates_at_the_boundaries_stay_in_range() -> None:
    """Every component must stay in [0,1] for every combination of 0.0 and 1.0.

    Components are combined with weights that assume a bounded range; one term
    escaping it silently re-weights every other term.
    """
    fields = [
        "adoption_rate",
        "advocacy_rate",
        "advocacy_unweighted",
        "repost_participation",
        "comment_participation",
        "like_participation",
        "negative_participation",
        "secondary_share",
        "late_action_share",
        "audience_fit_rate",
        "resonance_adoption",
        "resonance_advocacy",
    ]
    for combo in itertools.product((0.0, 1.0), repeat=4):
        kw = dict(zip(fields[:4], combo, strict=False))
        for rest in fields[4:]:
            kw[rest] = combo[0]
        sig = _sig(
            n_interviews=10,
            n_valid_trials=4,
            n_trials=4,
            exposed_agents=10,
            delight_mean=5.0,
            craft_mean=5.0,
            **kw,
        )
        for name, value in compute_components(sig).items():
            if value is None:
                continue
            assert 0.0 <= value <= 1.0, f"{name}={value} for {kw}"
            assert not math.isnan(value), f"{name} NaN for {kw}"


def test_no_reachable_trial_leaves_craft_unmeasured_not_zero() -> None:
    """If nobody reached the app, craft is unknown -- and must say so.

    Scoring craft 0 here would punish an app for a harness or startup failure,
    which is the exact fabricated-capability failure the bench must not have.
    Eight of 37 builds in the last sweep were unreachable in every trial, so this
    is a routine input, not a corner case.
    """
    sig = _sig(
        n_interviews=30,
        adoption_rate=0.4,
        delight_mean=4.0,
        n_trials=8,
        n_valid_trials=0,
        n_unreachable_trials=8,
        craft_mean=None,
        exposed_agents=30,
    )
    components = compute_components(sig)
    assert components.get("craft") is None
    # Craft unmeasured must not be reported as craft zero.
    assert components.get("craft") is None


def test_validity_gate_multiplier_is_bounded_and_monotone() -> None:
    """A broken app must never score above a working one via the gate."""
    weights = ScoreWeights()
    working = _sig(builds=True, runs=True, does_what_it_claims=True)
    broken = _sig(builds=True, runs=False, does_what_it_claims=False)
    unknown = _sig()

    g_working = validity_gate(working, weights)
    g_broken = validity_gate(broken, weights)
    g_unknown = validity_gate(unknown, weights)

    for name, g in (
        ("working", g_working),
        ("broken", g_broken),
        ("unknown", g_unknown),
    ):
        assert 0.0 <= g <= 1.0, f"{name} gate={g} outside [0,1]"
    assert g_broken < g_working, "a broken app must be gated below a working one"


def test_negative_and_out_of_range_inputs_are_not_silently_accepted() -> None:
    """Corrupt signals should fail loudly or clamp -- never propagate.

    A rate above 1 or below 0 means an upstream extraction bug; letting it flow
    into a weighted sum produces a plausible-looking score from broken data.
    """
    sig = _sig(
        n_interviews=10,
        adoption_rate=1.5,
        advocacy_rate=-0.2,
        delight_mean=99.0,
        craft_mean=-3.0,
        n_valid_trials=4,
        n_trials=4,
        exposed_agents=10,
    )
    components = compute_components(sig)
    offenders = {
        name: v
        for name, v in components.items()
        if v is not None and not (0.0 <= v <= 1.0)
    }
    assert not offenders, (
        f"corrupt inputs produced out-of-range components: {offenders}; "
        "the scorer should clamp or reject rather than propagate"
    )


@pytest.mark.parametrize("n", [0, 1, 2])
def test_tiny_crowds_do_not_crash(n: int) -> None:
    """Pilot runs use tiny crowds; they must degrade, not explode."""
    sig = _sig(
        n_interviews=n,
        adoption_rate=1.0 if n else None,
        delight_mean=5.0 if n else None,
        exposed_agents=n,
        n_trials=n,
        n_valid_trials=n,
        craft_mean=5.0 if n else None,
    )
    components = compute_components(sig)
    unscorable_reasons(sig, components)  # must not raise
    for name, v in components.items():
        assert v is None or 0.0 <= v <= 1.0, f"{name}={v}"


def test_autorater_dimensions_are_clamped_like_every_other_component() -> None:
    """Autorater scores must not bypass the range check.

    They are merged into `components` AFTER compute_components, so they used to
    skip its clamp entirely -- and `AutoRating.from_dict` read stored scores with
    a bare float(), trusting that whatever wrote the JSON had enforced the 0-10
    rubric. normalized() divides by 10, so a stored 50 contributes 5.0 to a term
    the weighted sum requires to be in [0,1].
    """
    from viral_bench.score.autorater import AutoRating
    from viral_bench.score.viralscore import score_run

    rating = AutoRating.from_dict(
        {
            "build_id": "b",
            "model": "m",
            "repeats": 1,
            "dimensions": {
                "substance": {"score": 50},  # far outside the rubric
                "severity": {"score": -7},
                "word_of_mouth": {"score": 8},
            },
        }
    )
    # Clamped at the boundary on the way in.
    assert rating.dimensions["substance"].score == 10.0
    assert rating.dimensions["severity"].score == 0.0
    assert rating.dimensions["word_of_mouth"].score == 8.0

    sig = _sig(
        n_interviews=30,
        adoption_rate=0.5,
        advocacy_rate=0.4,
        delight_mean=6.0,
        craft_mean=7.0,
        n_valid_trials=8,
        n_trials=8,
        exposed_agents=30,
        builds=True,
        runs=True,
        does_what_it_claims=True,
    )
    result = score_run(sig, autorating=rating)
    for name, value in result.components.items():
        if value is None:
            continue
        assert 0.0 <= value <= 1.0, f"{name}={value} escaped [0,1]"
    assert result.score is None or 0.0 <= result.score <= 100.0
