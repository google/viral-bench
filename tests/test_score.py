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

"""Tests for the ViralScore (signals -> components -> composite)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from viral_bench.score.report import render_score, write_score
from viral_bench.score.signals import RunSignals, extract_signals
from viral_bench.score.viralscore import (
    BROKEN_APP_MULTIPLIER,
    CALIBRATED_CROWD_SIZE,
    FAILED_SELF_CHECK_MULTIPLIER,
    GATE_EVIDENCE_GRADED,
    GATE_HARD,
    SCORE_VERSION,
    ScoreWeights,
    _confidence_warnings,
    compute_components,
    score_run,
    validity_gate,
    witness_rate,
)


def _signals(**over) -> RunSignals:
    """A healthy, fully-measured run. Override individual fields per test."""
    base = dict(
        build_id="demo__1",
        crowd_dir="/tmp/demo",
        n_interviews=50,
        adoption_rate=0.6,
        advocacy_rate=0.5,
        advocacy_unweighted=0.5,
        delight_mean=7.0,
        delight_stdev=1.5,
        n_valid_trials=10,
        n_trials=10,
        craft_mean=8.0,
        facets={"functionality": 8.0, "usability": 7.5},
        persistence_rate=0.9,
        n_persistence_checked=10,
        exposed_agents=40,
        repost_participation=0.4,
        # A strict subset of repost_participation: peer reposts by agents who
        # would themselves share it. The fixture's contract is "fully measured",
        # so it must carry every weighted component -- otherwise "a clean run
        # has no warnings" quietly stops testing anything.
        advocate_amplification=0.2,
        comment_participation=0.3,
        like_participation=0.9,
        negative_participation=0.0,
        secondary_share=0.0,
        late_action_share=0.5,
        audience_fit_rate=0.7,
        resonance_adoption=0.8,
        resonance_advocacy=0.7,
        does_what_it_claims=True,
        builds=True,
        runs=True,
        requested_agents=40,
        actual_agents=40,
    )
    base.update(over)
    return RunSignals(**base)


# -- weights ---------------------------------------------------------------- #


def test_default_weights_sum_to_one_and_split_65_35() -> None:
    w = ScoreWeights()
    w.validate()
    judgement = w.adoption + w.advocacy + w.craft
    behaviour = w.amplification + w.reception + w.cascade
    assert judgement == pytest.approx(0.65)
    assert behaviour == pytest.approx(0.35)


def test_invalid_weights_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        ScoreWeights(adoption=0.9, advocacy=0.9).validate()


# -- components ------------------------------------------------------------- #


def test_components_normalised_to_unit_interval() -> None:
    c = compute_components(_signals())
    for name, value in c.items():
        assert value is None or 0.0 <= value <= 1.0, name
    assert c["craft"] == pytest.approx(0.8)  # 8.0/10


def test_likes_are_excluded_from_amplification() -> None:
    # Likes measured zero between-app variance, so they must not move the score.
    low = compute_components(_signals(like_participation=0.0))["amplification"]
    high = compute_components(_signals(like_participation=1.0))["amplification"]
    assert low == high


def test_amplification_counts_reposts_and_ignores_comment_volume() -> None:
    """Commenting is not amplifying, and measured, it points the wrong way.

    Over the calibration corpus comment participation runs 0.868 (broken) /
    0.839 (mid) / 0.750 (good): people comment to complain. While it sat inside
    amplification at weight 0.35 of 0.27, a worse app earned a higher score on
    that term.
    """
    reposty = compute_components(
        _signals(repost_participation=0.8, comment_participation=0.0)
    )["amplification"]
    chatty = compute_components(
        _signals(repost_participation=0.0, comment_participation=0.8)
    )["amplification"]
    assert reposty == pytest.approx(0.8)
    assert chatty == pytest.approx(0.0)

    # Valence lives in its own component now, not smuggled into amplification.
    loud = compute_components(
        _signals(repost_participation=0.5, negative_participation=0.9)
    )
    quiet = compute_components(
        _signals(repost_participation=0.5, negative_participation=0.0)
    )
    assert loud["amplification"] == quiet["amplification"] == pytest.approx(0.5)
    assert loud["reception"] < quiet["reception"]


def test_reception_is_net_valence_and_keeps_resolution_among_bad_apps() -> None:
    """No floor clip: a corpse and a merely-disliked app must not both read 0.

    Clipping at zero is what made the old amplification term look like a perfect
    discriminator -- every bad app pinned to 0.000 -- while destroying all
    resolution across the bottom half of the range.
    """
    loved = compute_components(
        _signals(like_participation=0.9, negative_participation=0.0)
    )["reception"]
    mixed = compute_components(
        _signals(like_participation=0.4, negative_participation=0.4)
    )["reception"]
    hated = compute_components(
        _signals(like_participation=0.1, negative_participation=0.9)
    )["reception"]
    corpse = compute_components(
        _signals(like_participation=0.0, negative_participation=1.0)
    )["reception"]
    assert loved > mixed > hated > corpse
    assert mixed == pytest.approx(0.5)
    assert corpse == pytest.approx(0.0)


def test_reception_is_unmeasured_not_zero_when_the_signals_are_missing() -> None:
    sig = _signals(like_participation=None, negative_participation=None)
    assert compute_components(sig)["reception"] is None


def test_amplification_is_per_capita_so_crowd_size_does_not_inflate_it() -> None:
    # Same participation fraction at 8 and 50 agents -> same component.
    small = compute_components(_signals(exposed_agents=8))["amplification"]
    large = compute_components(_signals(exposed_agents=50))["amplification"]
    assert small == large


# -- composite + gate ------------------------------------------------------- #


def test_score_is_0_100_and_stamped_with_version() -> None:
    r = score_run(_signals())
    assert 0 <= r.score <= 100
    assert r.score_version == SCORE_VERSION
    assert r.scorable


def test_broken_app_is_capped_by_the_validity_gate() -> None:
    working = score_run(_signals(does_what_it_claims=True))
    broken = score_run(_signals(does_what_it_claims=False, runs=False, builds=True))
    assert broken.gate == BROKEN_APP_MULTIPLIER
    assert broken.score == pytest.approx(working.score * BROKEN_APP_MULTIPLIER, abs=0.2)
    assert broken.score < working.score


def test_better_app_scores_higher() -> None:
    weak = score_run(
        _signals(
            adoption_rate=0.1,
            advocacy_rate=0.05,
            craft_mean=3.0,
            repost_participation=0.05,
            comment_participation=0.05,
        )
    )
    strong = score_run(
        _signals(
            adoption_rate=0.9,
            advocacy_rate=0.8,
            craft_mean=9.0,
            repost_participation=0.7,
            comment_participation=0.6,
        )
    )
    assert strong.score > weak.score + 30  # a wide, usable dynamic range


def test_unmeasured_components_are_reweighted_not_zeroed() -> None:
    # A missing measurement must not be scored as a bad one.
    full = score_run(_signals())
    no_craft = score_run(_signals(craft_mean=None, n_valid_trials=0))
    assert no_craft.scorable
    assert no_craft.components["craft"] is None
    # craft was above the others here, so dropping it lowers the mean slightly,
    # but nowhere near what treating it as 0.0 would do.
    zeroed = 100 * (full.score / 100 - ScoreWeights().craft * 0.8)
    assert no_craft.score > zeroed + 5
    assert any("craft is unmeasured" in w for w in no_craft.confidence)


def test_cascade_measured_but_unweighted_in_v1() -> None:
    # Secondary engagement was identically zero in every run recorded, so it is
    # reported but must not move the number until it becomes non-degenerate.
    assert ScoreWeights().cascade == 0.0
    a = score_run(_signals(secondary_share=0.0))
    b = score_run(_signals(secondary_share=0.9))
    assert a.score == b.score
    assert b.components["cascade"] == 0.9  # still reported


# -- confidence ------------------------------------------------------------- #


def test_small_crowd_and_clamp_raise_confidence_warnings() -> None:
    r = score_run(
        _signals(n_interviews=8, clamped=True, requested_agents=30, actual_agents=8)
    )
    joined = " ".join(r.confidence)
    assert "only 8 interviews" in joined
    assert "clamped" in joined


def test_calibrated_crowd_size_is_the_warning_threshold() -> None:
    # 30, re-measured on the fixed instrument: pooled within-cell noise 3.5
    # points at n=30 against 9.6 at n=50, with a WIDER good-minus-broken margin
    # (59.7 vs 45.9) at half the wall clock. The old 50 came from a reliability
    # table whose n=50 bucket contained no runs of the strong build at all.
    assert CALIBRATED_CROWD_SIZE == 30
    assert score_run(_signals(n_interviews=CALIBRATED_CROWD_SIZE)).confidence == []
    assert score_run(_signals(n_interviews=CALIBRATED_CROWD_SIZE - 1)).confidence


def test_missing_validity_gate_is_flagged() -> None:
    r = score_run(_signals(does_what_it_claims=None))
    assert any("never verified" in w for w in r.confidence)
    assert r.gate == 1.0  # unverified is not the same as broken


def test_clean_run_has_no_warnings() -> None:
    assert score_run(_signals()).confidence == []


def test_confidence_interval_narrows_as_the_crowd_grows() -> None:
    small = score_run(_signals(n_interviews=8))
    large = score_run(_signals(n_interviews=50))
    assert (small.ci_high - small.ci_low) > (large.ci_high - large.ci_low)


# -- artifacts -------------------------------------------------------------- #


def test_score_json_round_trips(tmp_path) -> None:
    r = score_run(_signals())
    path = write_score(r, tmp_path)
    data = json.loads(path.read_text())
    assert data["score"] == r.score
    assert data["score_version"] == SCORE_VERSION
    assert set(data["weights"]) == set(r.weights)


def test_render_is_readable_and_explains_the_score() -> None:
    text = render_score(score_run(_signals()))
    assert "ViralScore:" in text
    for component in ("adoption", "advocacy", "craft", "amplification"):
        assert component in text
    assert "audience fit" in text


def test_extract_signals_from_a_run_summary(tmp_path) -> None:
    summary = {
        "build_id": "demo__1",
        "ok": True,
        "crowd": [{"agent_id": 1, "influence": 9}, {"agent_id": 2, "influence": 1}],
        "rounds": [{"round": 1, "ok": True}],
        "validity": {"does_what_it_claims": True, "detail": "smoke=ok"},
        "crowd_integrity": {
            "requested_n_agents": 2,
            "actual_n_agents": 2,
            "clamped": False,
        },
        "engagement": {
            "reach": {
                "exposed_agents": 2,
                "actors_reposted": 1,
                "actors_commented": 2,
                "actors_liked": 2,
                "actors_negative": 0,
            },
            "cascade": {"secondary_share": 0.0, "late_action_share": 0.4},
        },
        "verdicts": {
            "interviews": {
                "n": 2,
                "would_use_rate": 0.5,
                "would_share_rate": 0.5,
                "delight_mean": 6.0,
                "per_agent": [
                    {"agent_id": 1, "would_share": True},
                    {"agent_id": 2, "would_share": False},
                ],
            },
            "triers": {
                "n": 1,
                "per_agent": [
                    {
                        "agent_id": 1,
                        "finished": True,
                        "degraded": False,
                        "app_reachable": True,
                        "craft": 7.0,
                    }
                ],
            },
        },
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    sig = extract_signals(tmp_path)
    assert sig.n_interviews == 2
    assert sig.craft_mean == 7.0
    assert sig.does_what_it_claims is True
    assert sig.repost_participation == 0.5
    # influence weighting: agent 1 (influence 9 -> weight 3) said yes,
    # agent 2 (influence 1 -> weight 1) said no => 3/4.
    assert sig.advocacy_rate == pytest.approx(0.75)
    assert sig.advocacy_unweighted == 0.5


def test_degraded_trials_are_not_counted_as_hands_on_evidence(tmp_path) -> None:
    summary = {
        "build_id": "demo__1",
        "ok": True,
        "crowd": [],
        "verdicts": {
            "triers": {
                "n": 2,
                "per_agent": [
                    {
                        "agent_id": 1,
                        "finished": True,
                        "degraded": True,
                        "app_reachable": True,
                        "craft": 9.0,
                    },
                    {
                        "agent_id": 2,
                        "finished": True,
                        "degraded": False,
                        "app_reachable": True,
                        "craft": 4.0,
                    },
                ],
            }
        },
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    sig = extract_signals(tmp_path)
    assert sig.n_valid_trials == 1
    assert sig.craft_mean == 4.0  # the degraded 9.0 is excluded


def test_unreachable_trials_are_not_counted_as_hands_on_evidence(tmp_path) -> None:
    """A trial that never reached the app rated the source tree, not the app.

    36 of 314 stored trials were in exactly this state -- every ``open`` failed,
    ``degraded`` was never set, and a full craft verdict was emitted anyway. All
    36 sat in one build, so the harness race was being read as a difference
    between founder models.
    """
    summary = {
        "build_id": "demo__1",
        "ok": True,
        "crowd": [],
        "verdicts": {
            "triers": {
                "n": 2,
                "per_agent": [
                    # Never reached the app, but claims a confident rating.
                    {
                        "agent_id": 1,
                        "finished": True,
                        "degraded": False,
                        "app_reachable": False,
                        "craft": 9.0,
                    },
                    {
                        "agent_id": 2,
                        "finished": True,
                        "degraded": False,
                        "app_reachable": True,
                        "craft": 4.0,
                    },
                ],
            }
        },
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    sig = extract_signals(tmp_path)
    assert sig.n_valid_trials == 1
    assert sig.n_unreachable_trials == 1
    assert sig.craft_mean == 4.0  # the fabricated 9.0 is excluded


def test_craft_is_unmeasured_when_no_trial_reached_the_app(tmp_path) -> None:
    """Whole-run case: craft must read as unmeasured, never as a number."""
    summary = {
        "build_id": "demo__1",
        "ok": True,
        "crowd": [],
        "verdicts": {
            "triers": {
                "n": 2,
                "per_agent": [
                    {
                        "agent_id": i,
                        "finished": True,
                        "degraded": False,
                        "app_reachable": False,
                        "craft": 7.5,
                    }
                    for i in (1, 2)
                ],
            }
        },
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    sig = extract_signals(tmp_path)
    assert sig.n_valid_trials == 0
    assert sig.n_unreachable_trials == 2
    assert sig.craft_mean is None


def test_a_trial_that_never_proved_it_reached_the_app_is_not_evidence(
    tmp_path,
) -> None:
    """Unknown reachability is missing evidence, and must not read as good.

    The test is ``app_reachable is True``, not ``is not False``. 6 of 24 bot
    trials in one sweep filed a full craft verdict having never sent the bot a
    single message, and historic artifacts predate the field entirely.
    """
    summary = {
        "build_id": "demo__1",
        "ok": True,
        "crowd": [],
        "verdicts": {
            "triers": {
                "n": 1,
                "per_agent": [
                    {"agent_id": 1, "finished": True, "degraded": False, "craft": 6.0}
                ],
            }
        },
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    sig = extract_signals(tmp_path)
    assert sig.n_valid_trials == 0
    assert sig.n_unreachable_trials == 1
    assert sig.craft_mean is None


def test_gate_is_graded_between_a_dead_app_and_a_bad_self_check() -> None:
    # "app is dead" and "app works but its self-written smoke check is wrong"
    # are different failures, and calibration hit the second on a real build.
    dead = score_run(_signals(does_what_it_claims=False, builds=True, runs=False))
    bad_check = score_run(_signals(does_what_it_claims=False, builds=True, runs=True))
    working = score_run(_signals())
    assert dead.gate == BROKEN_APP_MULTIPLIER
    assert bad_check.gate == FAILED_SELF_CHECK_MULTIPLIER
    assert dead.score < bad_check.score < working.score
    assert any("smoke check" in w for w in bad_check.confidence)
    assert any("failed to build or start" in w for w in dead.confidence)


def test_run_without_interviews_is_unscorable_not_silently_scored() -> None:
    # The failure that contaminated the first calibration: an upstream recsys bug
    # destroyed every interview in runs that generated a lot of discussion, and
    # those runs were still scored off craft + amplification alone -- looking MORE
    # stable than healthy runs because craft is the steadiest component.
    r = score_run(_signals(n_interviews=0, adoption_rate=None, advocacy_rate=None))
    assert r.score is None
    assert not r.scorable
    assert any("no interviews recorded" in w for w in r.confidence)


def test_incomplete_simulation_is_unscorable() -> None:
    r = score_run(_signals(run_ok=False))
    assert not r.scorable
    assert any("did not complete" in w for w in r.confidence)


def test_a_healthy_run_is_still_scorable() -> None:
    # The guard must not be so eager that normal runs stop producing a number.
    assert score_run(_signals()).scorable


# -- model discrimination ---------------------------------------------------- #


def test_cohens_d_scales_the_gap_by_the_noise() -> None:
    from viral_bench.score.discriminate import ModelScores, separation

    a, b = ModelScores("A"), ModelScores("B")
    for i, s in enumerate([70, 72, 68, 71]):
        a.add(f"idea{i}", s)
    for i, s in enumerate([50, 52, 48, 51]):
        b.add(f"idea{i}", s)
    sep = separation(a, b)
    assert sep.gap == 20.0
    assert sep.cohens_d > 5  # a 20-point gap against ~2-point noise is enormous
    assert sep.verdict == "large"


def test_a_gap_smaller_than_the_noise_is_negligible() -> None:
    from viral_bench.score.discriminate import ModelScores, separation

    a, b = ModelScores("A"), ModelScores("B")
    for i, s in enumerate([60, 40, 70, 30]):
        a.add(f"idea{i}", s)
    for i, s in enumerate([58, 42, 68, 32]):
        b.add(f"idea{i}", s)
    sep = separation(a, b)
    assert sep.verdict == "negligible"  # 2-point gap, ~17-point spread


def test_paired_comparison_removes_idea_difficulty() -> None:
    # Idea difficulty is the biggest nuisance factor: a hard idea drags every
    # model down. Differencing within an idea should expose a consistent edge
    # that the unpaired spread would otherwise bury.
    from viral_bench.score.discriminate import ModelScores, separation

    a, b = ModelScores("A"), ModelScores("B")
    a.add("easy", 90)
    b.add("easy", 85)
    a.add("hard", 30)
    b.add("hard", 25)
    sep = separation(a, b)
    assert sep.n_ideas_shared == 2
    assert sep.paired_gap == 5.0  # consistent +5 edge on every shared idea
    assert abs(sep.cohens_d) < 0.5  # unpaired, the 60-point idea spread hides it


# -- searching the scoring configuration -------------------------------------- #


def _write_run(tmp_path, name: str, *, would_use: float, craft: float) -> str:
    """A minimal scorable crowd run on disk, tunable in strength."""
    d = tmp_path / name
    d.mkdir()
    n = 10
    yes = int(round(would_use * n))
    (d / "run_summary.json").write_text(
        json.dumps(
            {
                "build_id": name.rsplit("__", 1)[0],
                "ok": True,
                "rounds": [{"round": 1, "ok": True}],
                "crowd": [{"agent_id": i, "influence": 5} for i in range(n)],
                "validity": {"does_what_it_claims": True},
                "engagement": {
                    "reach": {
                        "exposed_agents": n,
                        "actors_reposted": yes,
                        "actors_commented": yes,
                        "actors_liked": n,
                        "actors_negative": 0,
                    },
                    "cascade": {"secondary_share": 0.0, "late_action_share": 0.4},
                },
                "verdicts": {
                    "interviews": {
                        "n": n,
                        "would_use_rate": would_use,
                        "would_share_rate": would_use,
                        "delight_mean": craft,
                        "per_agent": [
                            {"agent_id": i, "would_share": i < yes} for i in range(n)
                        ],
                    },
                    "triers": {
                        "n": n,
                        "per_agent": [
                            {
                                "agent_id": i,
                                "finished": True,
                                "degraded": False,
                                "app_reachable": True,
                                "craft": craft,
                            }
                            for i in range(n)
                        ],
                    },
                },
            }
        )
    )
    return str(d)


def test_a_rating_on_disk_changes_the_score_it_is_weighted_into(tmp_path) -> None:
    # The bug this guards: discriminate.py used to score every run without ever
    # loading autorating.json, so the autorater weights were renormalised away
    # and no change to the rater could move the separation number.
    from viral_bench.score.autorater import AutoRating, DimensionRating
    from viral_bench.score.discriminate import score_runs_by_model

    run = _write_run(tmp_path, "ideaA__b1__crowd-1", would_use=0.5, craft=5.0)
    runs = {"A": [run]}
    weights = ScoreWeights.from_profile("qualitative_heavy")

    unrated = score_runs_by_model(runs, weights)["A"].mean
    rating = AutoRating(
        build_id="ideaA__b1",
        model="stub",
        repeats=1,
        dimensions={
            d: DimensionRating(score=10.0)
            for d in ("substance", "severity", "word_of_mouth")
        },
    )
    rated = score_runs_by_model(runs, weights, {run: rating})["A"].mean
    assert rated > unrated


def test_component_dashboard_includes_the_autorater_dimensions(tmp_path) -> None:
    # Without this the dashboard can only ever say the deterministic components
    # separate the models, because they are the only ones it looks at.
    from viral_bench.score.autorater import AutoRating, DimensionRating
    from viral_bench.score.discriminate import component_separation

    strong = _write_run(tmp_path, "ideaA__b1__crowd-1", would_use=0.9, craft=9.0)
    weak = _write_run(tmp_path, "ideaA__b2__crowd-1", would_use=0.2, craft=3.0)

    def _rating(score: float) -> AutoRating:
        return AutoRating(
            build_id="x",
            model="stub",
            repeats=1,
            dimensions={"substance": DimensionRating(score=score)},
        )

    comps = component_separation(
        {"A": [strong], "B": [weak]},
        {strong: _rating(9.0), weak: _rating(2.0)},
    )
    assert "substance" in comps
    assert comps["substance"]["per_model_mean"] == {"A": 0.9, "B": 0.2}


def test_sweep_ranks_the_profile_that_separates_most_first(tmp_path) -> None:
    from viral_bench.score.discriminate import sweep_profiles

    # Two models differing ONLY in craft (adoption is held identical), each with
    # real run-to-run spread. A craft-weighted profile must surface above one
    # that ignores craft -- that ordering is the whole point of the sweep: it
    # says which scoring configuration to keep.
    runs = {
        "A": [
            _write_run(tmp_path, f"idea{i}__a__crowd-1", would_use=0.5, craft=c)
            for i, c in enumerate([9.0, 8.5, 9.5])
        ],
        "B": [
            _write_run(tmp_path, f"idea{i}__b__crowd-1", would_use=0.5, craft=c)
            for i, c in enumerate([3.0, 2.5, 3.5])
        ],
    }
    craft_only = ScoreWeights(
        adoption=0.0,
        advocacy=0.0,
        craft=1.0,
        amplification=0.0,
        reception=0.0,
        cascade=0.0,
    )
    no_craft = ScoreWeights(
        adoption=0.5,
        advocacy=0.0,
        craft=0.0,
        amplification=0.5,
        reception=0.0,
        cascade=0.0,
    )
    rows = sweep_profiles(runs, {"craft_only": craft_only, "no_craft": no_craft})
    assert [r["profile"] for r in rows] == ["craft_only", "no_craft"]
    assert rows[0]["max_abs_cohens_d"] > 1.0
    assert not rows[1]["max_abs_cohens_d"]  # identical scores => no separation


def test_a_malformed_profile_is_reported_not_fatal(tmp_path) -> None:
    # Profiles are hand-edited between iterations. One typo must not cost the
    # whole sweep.
    from viral_bench.score.discriminate import sweep_profiles

    runs = {
        "A": [
            _write_run(tmp_path, f"idea{i}__a__crowd-1", would_use=0.9, craft=c)
            for i, c in enumerate([9.0, 8.5, 9.5])
        ],
        "B": [
            _write_run(tmp_path, f"idea{i}__b__crowd-1", would_use=0.2, craft=c)
            for i, c in enumerate([3.0, 2.5, 3.5])
        ],
    }
    broken = ScoreWeights(adoption=0.9, advocacy=0.9, craft=0.9)  # sums to 2.7
    rows = sweep_profiles(runs, {"broken": broken, "good": ScoreWeights()})
    by_name = {r["profile"]: r for r in rows}
    assert "must sum to 1.0" in by_name["broken"]["error"]
    assert by_name["good"]["max_abs_cohens_d"] is not None
    assert rows[0]["profile"] == "good"  # the unusable one sinks, stays visible


def test_a_crashed_validity_gate_is_unverified_not_broken(tmp_path) -> None:
    """The harness failing is not evidence about the app.

    The gate's error path used to return builds/runs/does_what_it_claims=False,
    which fires the 0.2x broken-app multiplier and prints "app failed to build
    or start": a verdict about the founder model, caused by a harness crash.
    """
    summary = {
        "build_id": "demo__1",
        "ok": True,
        "crowd": [],
        "validity": {
            "builds": None,
            "runs": None,
            "does_what_it_claims": None,
            "gate_errored": True,
            "detail": "validity gate errored: RuntimeError: podman exploded",
        },
        "verdicts": {
            "interviews": {"n": 3, "would_use_rate": 0.6, "per_agent": []},
            "triers": {"n": 0, "per_agent": []},
        },
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    sig = extract_signals(tmp_path)

    assert sig.does_what_it_claims is None
    assert validity_gate(sig) == 1.0  # no discount for a harness crash

    warnings = _confidence_warnings(sig, compute_components(sig))
    assert any("never verified" in w for w in warnings)
    assert not any("failed to build" in w for w in warnings)


def test_the_gate_is_reachable_from_the_profile() -> None:
    """Editing gates in score.yaml must change the score.

    The multipliers were module constants, so the `gates:` block in every
    profile was decoration: changing broken_app from 0.2 to 0.1 altered nothing
    and warned nobody. The gate is the strongest single lever in the score --
    0.6 is a 40% haircut, ~22 points on a 55-point build, larger than any weight
    change -- so leaving it outside config also left it outside the sweep.
    """
    sig = _signals(does_what_it_claims=False, builds=False, runs=False)
    lenient = ScoreWeights(gate_broken_app=0.9)
    harsh = ScoreWeights(gate_broken_app=0.1)

    assert validity_gate(sig, lenient) == 0.9
    assert validity_gate(sig, harsh) == 0.1
    assert score_run(sig, lenient).score > score_run(sig, harsh).score


def test_gates_are_not_counted_as_component_weights() -> None:
    """Gates are multipliers, so they must stay out of the sum-to-1.0 check."""
    w = ScoreWeights(gate_broken_app=0.1, gate_failed_self_check=0.5)
    w.validate()  # must not raise
    assert "gate_broken_app" not in w.as_dict()
    assert w.gates() == {
        "broken_app": 0.1,
        "failed_self_check": 0.5,
        "policy": GATE_HARD,
    }


def test_a_failed_self_check_is_not_treated_as_a_dead_app() -> None:
    sig = _signals(does_what_it_claims=False, builds=True, runs=True)
    w = ScoreWeights(gate_broken_app=0.2, gate_failed_self_check=0.6)
    assert validity_gate(sig, w) == 0.6


# -- undeliverable is a verdict, a harness fault is not ---------------------


def test_undeliverable_build_takes_the_broken_app_multiplier() -> None:
    """No manifest means the app WAS CHECKED and cannot run, not that the check
    never happened.

    The distinction is the whole point. `None` means the harness could not verify
    the app and carries no penalty, while `False` means the app does not run.
    A build with no `viralbench.json` is the second thing: there is no command
    to try, and that omission is the founder's.
    """
    from viral_bench.crowd.sim.verdicts import undeliverable_validity

    verdict = undeliverable_validity("some_build")
    assert verdict["builds"] is False
    assert verdict["runs"] is False
    assert verdict["does_what_it_claims"] is False
    assert "no valid viralbench.json" in verdict["detail"]

    sig = _signals(
        builds=False,
        runs=False,
        does_what_it_claims=False,
        n_valid_trials=0,
        craft_mean=None,
    )
    result = score_run(sig)
    assert result.gate == BROKEN_APP_MULTIPLIER
    assert result.score is not None and result.score < 20.0


def test_an_undeliverable_build_scores_below_a_working_one() -> None:
    """The floor has to come out of the same machinery, not an imputed constant."""
    working = score_run(
        _signals(adoption_rate=0.6, advocacy_rate=0.5, craft_mean=7.0)
    ).score
    undeliverable = score_run(
        _signals(
            builds=False,
            runs=False,
            does_what_it_claims=False,
            adoption_rate=0.02,
            advocacy_rate=0.0,
            craft_mean=None,
            n_valid_trials=0,
        )
    ).score
    assert undeliverable < working
    assert undeliverable < 20.0


def test_persistence_is_scored_when_measured_and_dropped_when_not() -> None:
    """Nobody checking is missing evidence, not a failure to persist.

    Scoring an unchecked run as 0 would charge the crowd's incuriosity to the
    app, and scoring it as 1 would reward an app for not being examined. Both are
    wrong, so the component is re-weighted out and the run says so.
    """
    from viral_bench.score.viralscore import ScoreWeights

    weights = ScoreWeights(
        adoption=0.2,
        advocacy=0.2,
        craft=0.2,
        amplification=0.2,
        reception=0.0,
        persistence=0.2,
    )
    kept = score_run(_signals(persistence_rate=1.0), weights)
    lost = score_run(_signals(persistence_rate=0.0), weights)
    assert kept.score > lost.score
    assert kept.components["persistence"] == 1.0

    unchecked = score_run(
        _signals(persistence_rate=None, n_persistence_checked=0), weights
    )
    assert unchecked.components["persistence"] is None
    assert any("persistence" in w for w in unchecked.confidence)
    # Re-weighted out, not scored as zero: it must beat the app that lost data.
    assert unchecked.score > lost.score


# -- redundancy: are the components N measurements, or one measured N times? -- #


def test_redundancy_flags_a_component_that_impersonates_the_whole_score(
    tmp_path,
) -> None:
    """The finding this panel exists to make routine.

    Runs whose strength varies together across every component -- which is what
    the real corpus looks like -- must show keep-one near 1.0. If the panel
    cannot see that, it cannot warn that a weight sweep is tuning nothing.
    """
    from viral_bench.score.discriminate import redundancy_panel

    dirs = [
        _write_run(tmp_path, f"idea__b{i}__crowd-1", would_use=v, craft=10 * v)
        for i, v in enumerate([0.1, 0.3, 0.5, 0.7, 0.9])
    ]
    panel = redundancy_panel(dirs, ScoreWeights.from_profile("v1_deterministic"))

    assert panel.n_runs == 5
    worst = panel.worst_keep_one
    assert worst is not None and worst[1] > 0.9
    assert panel.correlations["adoption"]["craft"] > 0.9
    # Dropping any one term leaves the ranking intact: that is the warning.
    assert all(v > 0.9 for v in panel.drop_one.values() if v is not None)


def test_redundancy_sees_independence_when_it_is_there(tmp_path) -> None:
    """Craft deliberately anti-ordered against adoption must not read as 1.0."""
    from viral_bench.score.discriminate import redundancy_panel

    adoption = [0.1, 0.3, 0.5, 0.7, 0.9]
    dirs = [
        _write_run(tmp_path, f"idea__b{i}__crowd-1", would_use=a, craft=10 * (1.0 - a))
        for i, a in enumerate(adoption)
    ]
    panel = redundancy_panel(dirs, ScoreWeights.from_profile("v1_deterministic"))
    assert panel.correlations["adoption"]["craft"] < -0.9


def test_redundancy_excludes_gated_runs_by_default(tmp_path) -> None:
    """The gate multiplies every component at once, so it manufactures agreement.

    A panel that silently included gated runs would report the gate's own
    variance as evidence that the components agree.
    """
    import json as _json

    from viral_bench.score.discriminate import redundancy_panel

    dirs = [
        _write_run(tmp_path, f"idea__b{i}__crowd-1", would_use=v, craft=10 * v)
        for i, v in enumerate([0.2, 0.4, 0.6, 0.8])
    ]
    broken = _write_run(tmp_path, "idea__dead__crowd-1", would_use=0.5, craft=5.0)
    path = Path(broken) / "run_summary.json"
    summary = _json.loads(path.read_text())
    summary["validity"] = {"does_what_it_claims": False, "builds": False, "runs": False}
    # An app that never starts has no successful hands-on trials. The fixture has
    # to say so: under the evidence_graded gate (score_version 1.7) the discount
    # is scaled by how many agents got the app working, so a "dead" run whose
    # every trial succeeded is not a gated run at all -- and this test needs one.
    for row in summary["verdicts"]["triers"]["per_agent"]:
        row["app_reachable"] = False
    path.write_text(_json.dumps(summary))

    panel = redundancy_panel(dirs + [broken])
    assert panel.n_runs == 4
    assert panel.n_excluded_gated == 1

    kept = redundancy_panel(dirs + [broken], clean_only=False)
    assert kept.n_runs == 5
    assert kept.n_excluded_gated == 0


def test_redundancy_renders_without_a_measurable_corpus() -> None:
    """An empty sweep must report "too few runs", not divide by zero."""
    from viral_bench.score.discriminate import redundancy_panel

    panel = redundancy_panel([])
    assert panel.n_runs == 0
    assert "too few runs" in panel.render()


# -- evidence_graded validity gate (score_version 1.7) ----------------------- #


def _graded(**over) -> ScoreWeights:
    return ScoreWeights(gate_policy=GATE_EVIDENCE_GRADED, **over)


def test_hard_is_still_the_default_policy() -> None:
    """Every profile written before 1.7 must score exactly as it always did."""
    assert ScoreWeights().gate_policy == GATE_HARD
    sig = _signals(does_what_it_claims=False, builds=False, runs=False)
    assert validity_gate(sig) == BROKEN_APP_MULTIPLIER


def test_witness_rate_is_trials_over_exposed_and_stays_in_unit() -> None:
    assert witness_rate(_signals(n_valid_trials=10, exposed_agents=40)) == 0.25
    assert witness_rate(_signals(n_valid_trials=0, exposed_agents=40)) == 0.0
    assert witness_rate(_signals(n_valid_trials=40, exposed_agents=40)) == 1.0
    # More trials than exposed agents is nonsense but must not exceed 1.0.
    assert witness_rate(_signals(n_valid_trials=99, exposed_agents=40)) == 1.0
    # No crowd at all must not divide by zero.
    assert witness_rate(_signals(exposed_agents=0)) == 0.0


def test_graded_gate_spares_an_app_the_whole_crowd_got_working() -> None:
    """A container probe must not overrule thirty agents who used the app.

    This is the collaborative_table regression: seed 2 recorded the highest
    adoption and craft of its three seeds and the hard gate still cut it 5x.
    """
    sig = _signals(
        does_what_it_claims=False,
        builds=False,
        runs=False,
        n_valid_trials=30,
        exposed_agents=30,
    )
    assert validity_gate(sig, _graded(gate_broken_app=0.0)) == 1.0
    assert (
        score_run(sig, _graded(gate_broken_app=0.0)).score
        == score_run(sig, _graded(gate_broken_app=0.0)).score
    )


def test_graded_gate_applies_the_full_floor_when_nobody_got_in() -> None:
    sig = _signals(
        does_what_it_claims=False,
        builds=False,
        runs=False,
        n_valid_trials=0,
        exposed_agents=30,
    )
    assert validity_gate(sig, _graded(gate_broken_app=0.0)) == 0.0
    assert validity_gate(sig, _graded(gate_broken_app=0.2)) == pytest.approx(0.2)


def test_graded_gate_interpolates_with_the_witness_rate() -> None:
    half = _signals(
        does_what_it_claims=False,
        builds=False,
        runs=False,
        n_valid_trials=15,
        exposed_agents=30,
    )
    assert validity_gate(half, _graded(gate_broken_app=0.0)) == pytest.approx(0.5)
    # floor + (1 - floor) * rate
    assert validity_gate(half, _graded(gate_broken_app=0.2)) == pytest.approx(0.6)


def test_graded_gate_is_monotone_in_evidence() -> None:
    w = _graded(gate_broken_app=0.0)
    gates = [
        validity_gate(
            _signals(
                does_what_it_claims=False,
                builds=False,
                runs=False,
                n_valid_trials=n,
                exposed_agents=30,
            ),
            w,
        )
        for n in (0, 5, 10, 20, 30)
    ]
    assert gates == sorted(gates)
    assert gates[0] == 0.0 and gates[-1] == 1.0


def test_graded_gate_never_touches_a_verified_or_unverified_app() -> None:
    """The gate only ever fires on a run the probe actively failed."""
    w = _graded(gate_broken_app=0.0)
    # Unverified: no evidence either way, and no penalty. Zero witnesses must
    # NOT drag it to zero, which would punish the harness for failing to check.
    unverified = _signals(does_what_it_claims=None, n_valid_trials=0, exposed_agents=30)
    assert validity_gate(unverified, w) == 1.0
    assert validity_gate(_signals(), w) == 1.0


def test_graded_gate_keeps_the_two_failure_tiers_apart() -> None:
    """A bad self-check still starts from a softer floor than a dead app."""
    w = _graded(gate_broken_app=0.0, gate_failed_self_check=0.5)
    common = dict(does_what_it_claims=False, n_valid_trials=0, exposed_agents=30)
    dead = _signals(builds=False, runs=False, **common)
    bad_check = _signals(builds=True, runs=True, **common)
    assert validity_gate(dead, w) == 0.0
    assert validity_gate(bad_check, w) == 0.5


def test_an_unknown_gate_policy_is_rejected_loudly() -> None:
    """Silently scoring under a different gate than the one in config is the
    exact failure this module keeps having to fix."""
    with pytest.raises(ValueError, match="gate policy"):
        ScoreWeights(gate_policy="lenient").validate()


def test_gate_policy_is_not_counted_as_a_component_weight() -> None:
    w = ScoreWeights(gate_policy=GATE_EVIDENCE_GRADED)
    w.validate()
    assert "gate_policy" not in w.as_dict()
    assert w.gates()["policy"] == GATE_EVIDENCE_GRADED


def test_the_shipped_broken_app_floor_is_not_zero() -> None:
    """A floor of exactly 0.0 wins the gate sweep by cheating, so guard it.

    Zero sends every dead run to a hard 0.0 and flattens whole cells to zero
    variance. That shrinks the pooled within-cell SD and the seed-parity null
    without the instrument resolving anything better -- scored on the cells it
    has NOT flattened its discrimination is 3.66, worse than the hard gate it
    replaces. The sweep reports that corrected figure now, but the cheapest
    guard is refusing to let the floor go back to zero unnoticed.
    """
    w = ScoreWeights.from_profile()
    if w.gate_policy != GATE_EVIDENCE_GRADED:
        pytest.skip("active profile does not use the graded gate")
    assert w.gate_broken_app > 0.0
    sig = _signals(
        does_what_it_claims=False,
        builds=False,
        runs=False,
        n_valid_trials=0,
        exposed_agents=30,
    )
    assert validity_gate(sig, w) > 0.0


# -- v7_equal: one weight per component (score_version 1.8) ------------------ #


def test_active_profile_weights_every_component_equally() -> None:
    """The shipped score is six equal parts, with the autorater as one part.

    The point of v7_equal is that no weight needs defending, so the guard is
    that the weights are flat rather than merely close: five
    deterministic components at 1/6 and an autorater whose dimensions sum to
    the same 1/6. A future profile that quietly re-tunes one of them should
    fail here rather than in a meeting.
    """
    w = ScoreWeights.from_profile()
    if w.profile != "v7_equal":
        pytest.skip("active profile is not v7_equal")

    sixth = 1.0 / 6.0
    live = {k: v for k, v in w.deterministic_weights().items() if v}
    assert set(live) == {
        "adoption",
        "advocacy",
        "craft",
        "advocacy_spread",
        "persistence",
    }
    for name, weight in live.items():
        assert weight == pytest.approx(sixth, abs=1e-9), name

    # The rater is ONE part, however many dimensions it is measured over.
    assert sum(w.autorater.values()) == pytest.approx(sixth, abs=1e-9)
    assert len(set(w.autorater.values())) == 1, "rater dimensions must be flat too"


def test_active_profile_still_sums_to_one_and_caps_the_rater() -> None:
    """Flat weights must not quietly break the two invariants that predate them."""
    w = ScoreWeights.from_profile()
    w.validate()
    assert sum(w.as_dict().values()) == pytest.approx(1.0, abs=1e-9)
    # Counts dominate, the rater stays a minority -- now a consequence of the
    # arithmetic (5/6 vs 1/6) rather than a separately chosen 0.85/0.15 split.
    assert sum(w.autorater.values()) < sum(w.deterministic_weights().values())


def test_equal_weights_inherit_the_witnessed_gate() -> None:
    """v7 changes weighting only: the 1.7 gate must come through untouched."""
    w = ScoreWeights.from_profile()
    if w.profile != "v7_equal":
        pytest.skip("active profile is not v7_equal")
    assert w.gate_policy == GATE_EVIDENCE_GRADED
    assert w.gate_broken_app == 0.1
    assert w.gate_failed_self_check == 0.55


def test_older_profiles_keep_their_tuned_weights() -> None:
    """Going flat must not rewrite history: v6 still scores as v6.

    Stored runs carry the profile they were scored under, so an older number
    stays reproducible only while the older profile keeps its own weights.
    """
    v6 = ScoreWeights.from_profile("v6_witnessed")
    assert v6.craft == 0.22
    assert v6.persistence == 0.12
    assert v6.advocacy_spread == 0.15
    v6.validate()
