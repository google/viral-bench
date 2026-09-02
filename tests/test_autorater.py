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

"""Tests for the evidence pack and the agentic autorater (no network)."""

from __future__ import annotations

import json
import sqlite3

import pytest

from viral_bench.providers import Reply
from viral_bench.score.autorater import (
    DIMENSIONS,
    AutoRating,
    DimensionRating,
    _parse_response,
    rate_pack,
)
from viral_bench.score.evidence import build_evidence_pack, write_evidence_pack
from viral_bench.score.signals import extract_signals
from viral_bench.score.viralscore import ScoreWeights, score_run


@pytest.fixture
def crowd_run(tmp_path):
    """A minimal but complete crowd run directory."""
    summary = {
        "build_id": "demo__1",
        "app_type": "client-app",
        "ok": True,
        "rounds": [{"round": 1, "ok": True}],
        "crowd": [
            {"agent_id": 1, "username": "maya", "influence": 7},
            {"agent_id": 2, "username": "raj", "influence": 3},
        ],
        "validity": {"does_what_it_claims": True, "builds": True, "runs": True},
        "crowd_integrity": {"clamped": False},
        "engagement": {
            "posts": 2,
            "comments": 1,
            "reposts": 1,
            "likes": 2,
            "dislikes": 0,
            "reports": 0,
            "reach": {
                "exposed_agents": 2,
                "actors_reposted": 1,
                "actors_commented": 1,
                "actors_liked": 2,
                "actors_negative": 0,
            },
            "cascade": {"secondary_share": 0.0, "late_action_share": 0.5},
        },
        "verdicts": {
            "interviews": {
                "n": 2,
                "would_use_rate": 0.5,
                "would_share_rate": 0.5,
                "delight_mean": 6.0,
                "per_agent": [
                    {
                        "agent_id": 1,
                        "username": "maya",
                        "tier": "trier",
                        "would_use": True,
                        "would_share": True,
                        "delight": 8,
                        "for_me": True,
                        "why": "the share link is why I'd switch",
                    },
                    {
                        "agent_id": 2,
                        "username": "raj",
                        "tier": "reactor",
                        "would_use": False,
                        "would_share": False,
                        "delight": 4,
                        "for_me": False,
                        "why": "not aimed at me",
                    },
                ],
            },
            "triers": {
                "n": 1,
                "craft_mean": 7.5,
                "per_agent": [
                    {
                        "agent_id": 1,
                        "username": "maya",
                        "finished": True,
                        "degraded": False,
                        "craft": 7.5,
                        "delight": 8,
                        "functionality": 8,
                        "usability": 7,
                    }
                ],
            },
        },
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    (tmp_path / "trajectories.json").write_text(
        json.dumps(
            [
                {
                    "agent_id": 1,
                    "username": "maya",
                    "tier": "trier",
                    "reasoning": ["I opened it and the share link worked instantly."],
                }
            ]
        )
    )
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "agent_1.json").write_text(
        json.dumps(
            {
                "agent_id": 1,
                "trace": {
                    "steps": [{"action": "open"}, {"action": "type"}],
                    "verdict": {"notes": "fast and genuinely useful"},
                },
            }
        )
    )
    con = sqlite3.connect(tmp_path / "simulation.db")
    con.execute(
        "CREATE TABLE post (post_id INT, user_id INT, content TEXT, "
        "original_post_id INT)"
    )
    con.execute("CREATE TABLE comment (comment_id INT, user_id INT, content TEXT)")
    con.execute("INSERT INTO post VALUES (1, 0, 'Launching QuickMemo today', NULL)")
    con.execute("INSERT INTO post VALUES (2, 1, 'worth a look', 1)")
    con.execute("INSERT INTO comment VALUES (1, 2, 'the share link is clever')")
    con.commit()
    con.close()
    return tmp_path


# -- evidence pack ----------------------------------------------------------- #


def test_pack_joins_every_evidence_source(crowd_run) -> None:
    pack = build_evidence_pack(crowd_run)
    assert pack.build_id == "demo__1"
    assert pack.metrics["crowd_size"] == 2
    assert pack.metrics["agents_exposed"] == 2
    assert len(pack.trials) == 1
    assert pack.trials[0]["actions"] == ["open", "type"]  # from traces/
    assert "genuinely useful" in pack.trials[0]["notes"]
    assert pack.trials[0]["reasoning"]  # from trajectories.json
    assert {d["kind"] for d in pack.discussion} == {"post", "repost", "comment"}
    assert len(pack.verdicts) == 2


def test_pack_does_not_anchor_the_rater_on_the_score_it_should_be_independent_of(
    crowd_run,
) -> None:
    """The rater must not be handed the deterministic components as context.

    The metrics block renders first, so anything in it primes the rater before
    it reads any evidence. It used to lead with adoption_rate, craft_mean and
    app_builds_and_runs -- exactly the components the rater's dimensions are
    supposed to be independent of -- which turns its 0.15 weight into a
    paraphrase of the formula instead of the judgement the formula cannot make.
    """
    pack = build_evidence_pack(crowd_run)
    leaked = {
        "adoption_rate",
        "share_rate",
        "craft_mean",
        "delight_mean",
        "audience_fit_rate",
        "repost_participation",
        "comment_participation",
        "secondary_engagement_share",
        "app_builds_and_runs",
    } & set(pack.metrics)
    assert not leaked, f"evidence pack anchors the rater on score components: {leaked}"


def test_pack_is_blinded_to_the_founder_model(crowd_run) -> None:
    # A rater that knows which model built the app can rate its reputation.
    text = build_evidence_pack(crowd_run).to_prompt().lower()
    for leak in ("gemini", "claude", "gpt", "founder model", "google-vertex"):
        assert leak not in text, f"evidence pack leaks {leak!r}"


def test_pack_orders_verdicts_by_influence_deterministically(crowd_run) -> None:
    a = build_evidence_pack(crowd_run).verdicts
    b = build_evidence_pack(crowd_run).verdicts
    assert a == b  # no sampling: same run always yields the same pack
    assert a[0]["username"] == "maya"  # influence 7 before influence 3


def test_pack_prompt_carries_metrics_trials_and_reasons(crowd_run) -> None:
    text = build_evidence_pack(crowd_run).to_prompt()
    assert "HARD METRICS" in text and "FIRST-HAND TRIALS" in text
    assert "PUBLIC DISCUSSION" in text and "CROWD VERDICTS" in text
    assert "why: the share link is why I'd switch" in text


def test_write_evidence_pack_round_trips(crowd_run) -> None:
    path = write_evidence_pack(crowd_run)
    data = json.loads(open(path).read())
    assert data["build_id"] == "demo__1"
    assert data["integrity"]["n_interviews"] == 2


# -- response parsing -------------------------------------------------------- #


def test_parse_handles_bare_fenced_and_noisy_json() -> None:
    body = '{"substance": {"score": 7}}'
    assert _parse_response(body)["substance"]["score"] == 7
    assert _parse_response(f"```json\n{body}\n```")["substance"]["score"] == 7
    assert _parse_response(f"Sure!\n{body}\nHope that helps")["substance"]["score"] == 7
    assert _parse_response("not json at all") == {}
    assert _parse_response("") == {}


# -- rating aggregation ------------------------------------------------------ #


class _StubClient:
    """A provider client with canned replies that records the prompts it was given.

    Speaks the same contract as a real one -- ``generate(messages) -> Reply``, the
    canonical shape from :mod:`viral_bench.providers` -- so the rater is exercised
    over the interface it calls rather than a simpler stand-in.
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, messages, *, tools=None, system=""):
        self.prompts.append(messages[-1]["content"])
        if not self.replies:
            # Explicit, so a test that under-supplies replies says so rather than
            # surfacing as an opaque IndexError swallowed by the retry loop.
            raise RuntimeError("_StubClient ran out of canned replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Reply(text=reply)


def _reply(sub, sev, wom, evidence="agent 1"):
    return json.dumps(
        {
            "substance": {"score": sub, "evidence": [evidence], "reason": "r"},
            "severity": {"score": sev, "evidence": [evidence], "reason": "r"},
            "word_of_mouth": {"score": wom, "evidence": [evidence], "reason": "r"},
        }
    )


def test_rating_uses_the_median_of_repeats(crowd_run) -> None:
    # A single LLM sample is not a measurement, and the median resists one outlier.
    client = _StubClient([_reply(8, 7, 5), _reply(8, 6, 4), _reply(1, 7, 4)])
    r = rate_pack(build_evidence_pack(crowd_run), client=client, repeats=3)
    assert r.dimensions["substance"].score == 8.0  # not dragged down by the 1
    assert r.dimensions["substance"].spread == 7.0  # but the spread is visible
    assert r.normalized()["substance"] == 0.8


def test_the_rater_asks_for_json_and_a_ceiling_that_bounds_nothing(
    monkeypatch,
) -> None:
    # Two properties of the client the rater builds for itself.
    #
    # JSON mode, because the rater parses its own replies: prose is an
    # unparseable sample, i.e. a silently lost measurement.
    #
    # And a ceiling far above any reply the rubric can produce. The cap went
    # 2048 -> 8192 -> none, and the 2048 demonstrably cost a sample (the only stored
    # rating records "unparseable"), and on recent Gemini thinking tokens share the
    # budget, so a tight cap throttles the reasoning before it reaches the JSON.
    # The provider layer always sends some ceiling -- Anthropic requires one --
    # so what is locked here is that it is not a small one, and that it is pinned
    # by the rater rather than inherited from the layer's default.
    from viral_bench.score.autorater import _default_client

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = _default_client("openai/gpt-test", 0.0)

    assert client.json_mode is True
    assert client.temperature == 0.0
    assert client.max_tokens >= 8192


def test_the_rater_refuses_a_provider_that_cannot_produce_json() -> None:
    # Refused where the client is built, not sample by sample: a rater whose
    # every draw is unparseable produces an unrated run for a reason no error
    # anywhere states.
    from viral_bench.providers import UnsupportedCapabilityError
    from viral_bench.score.autorater import _default_client

    with pytest.raises(UnsupportedCapabilityError, match="autorater"):
        _default_client("ollama/llama-test", 0.0)


def test_an_unconfigured_rater_is_disabled_not_defaulted(
    monkeypatch, crowd_run
) -> None:
    """No model configured means the rater is OFF, not that it picks one.

    ViralBench ships no default model. Falling back to one would bill an account
    the user never named and rate every run with a model that appears in no
    configuration -- and raising instead would take a whole sweep down over a
    component that is optional by design. So it returns an unrated result, which
    is the same thing every consumer already handles for a rating that failed.
    """
    from viral_bench import config as config_module

    monkeypatch.setattr(config_module, "autorater_config", lambda: {"repeats": 3})

    r = rate_pack(build_evidence_pack(crowd_run))
    assert not r.ok
    assert r.normalized() == {}
    assert r.model == ""
    assert any("no autorater model" in e for e in r.errors)


def test_a_failed_sample_is_retried_rather_than_dropped(crowd_run) -> None:
    # A truncated or throttled draw is transient, so retry it. Dropping one
    # silently shrinks the median's support -- which is how the only stored
    # rating in the corpus came to stand on 2 samples while reporting 3.
    client = _StubClient(["garbage", _reply(4, 4, 4), _reply(4, 4, 4)])
    r = rate_pack(build_evidence_pack(crowd_run), client=client, repeats=2)
    assert r.dimensions["substance"].samples == [4.0, 4.0]


def test_rating_survives_a_failed_or_unparseable_sample(crowd_run) -> None:
    # Sample 1 fails every attempt it is given and is written off, but the rating
    # still stands on the two that worked, and records that it is thinner.
    client = _StubClient(
        [_reply(6, 6, 6), "garbage", RuntimeError("boom"), "garbage", _reply(6, 6, 6)]
    )
    r = rate_pack(build_evidence_pack(crowd_run), client=client, repeats=3)
    assert r.ok
    assert r.dimensions["substance"].score == 6.0
    assert r.dimensions["substance"].samples == [6.0, 6.0]
    assert any("dropped after retries" in e for e in r.errors)


def test_rating_with_no_usable_samples_is_not_ok(crowd_run) -> None:
    # 2 samples x 3 attempts each: every draw fails, so there is nothing to
    # aggregate and the rating must refuse to produce a number.
    client = _StubClient(["garbage"] * 6)
    r = rate_pack(build_evidence_pack(crowd_run), client=client, repeats=2)
    assert not r.ok
    assert r.normalized() == {}


def test_scores_are_clamped_to_the_rubric_range(crowd_run) -> None:
    client = _StubClient([_reply(99, -5, "x")])
    r = rate_pack(build_evidence_pack(crowd_run), client=client, repeats=1)
    assert r.dimensions["substance"].score == 10.0
    assert r.dimensions["severity"].score == 0.0
    assert "word_of_mouth" not in r.dimensions  # non-numeric is dropped, not coerced


def test_rater_prompt_contains_the_rubric_and_the_transcript(crowd_run) -> None:
    client = _StubClient([_reply(5, 5, 5)])
    rate_pack(build_evidence_pack(crowd_run), client=client, repeats=1)
    prompt = client.prompts[0]
    for dim in DIMENSIONS:
        assert dim in prompt
    assert "HARD METRICS" in prompt


# -- integration with the composite score ------------------------------------ #


def test_autorater_dimensions_are_folded_into_the_score(crowd_run) -> None:
    sig = extract_signals(crowd_run)
    weights = ScoreWeights.from_profile("v2_hybrid")

    without = score_run(sig, weights)
    rating = AutoRating(
        build_id="demo__1",
        model="stub",
        repeats=1,
        dimensions={d: DimensionRating(score=10.0) for d in DIMENSIONS},
    )
    with_max = score_run(sig, weights, rating)
    assert with_max.score > without.score
    assert with_max.components["substance"] == 1.0


def test_missing_autorating_is_reweighted_not_zeroed(crowd_run) -> None:
    # A run that was never rated must not be penalised as though it scored zero.
    sig = extract_signals(crowd_run)
    weights = ScoreWeights.from_profile("v2_hybrid")
    r = score_run(sig, weights)
    assert r.scorable
    assert any("re-weighted out" in w for w in r.confidence)


def test_rating_round_trips_through_disk(crowd_run) -> None:
    # Rating is an expensive LLM call paid for once at score time, and every later
    # re-score reads it back from autorating.json instead of paying again.
    client = _StubClient([_reply(8, 6, 4)])
    original = rate_pack(build_evidence_pack(crowd_run), client=client, repeats=1)
    path = crowd_run / "autorating.json"
    path.write_text(json.dumps(original.as_dict()), encoding="utf-8")

    restored = AutoRating.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert restored.ok
    assert restored.normalized() == original.normalized()
    assert restored.build_id == original.build_id
    assert restored.dimensions["substance"].evidence == ["agent 1"]


@pytest.mark.parametrize(
    "payload",
    [
        {},  # empty file
        {"dimensions": {}},  # rated, but nothing usable came back
        {"dimensions": {"substance": {"score": None}}},  # truncated mid-write
        {"dimensions": {"substance": {"score": "n/a"}}},  # non-numeric
        {"dimensions": "not-a-dict"},  # corrupt
        [],  # not even an object
    ],
)
def test_a_malformed_rating_loads_as_unrated_rather_than_raising(payload) -> None:
    # A sweep re-scores hundreds of runs. One truncated file must cost that run
    # its autorater dimensions, not take the whole comparison down.
    r = AutoRating.from_dict(payload)
    assert not r.ok
    assert r.normalized() == {}


def test_profiles_load_and_sum_to_one() -> None:
    from viral_bench import config

    for name in config.score_profile_names():
        ScoreWeights.from_profile(name).validate()


def test_v1_profile_reproduces_the_deterministic_formula() -> None:
    w = ScoreWeights.from_profile("v1_deterministic")
    assert w.autorater == {}
    assert w.craft == 0.25 and w.amplification == 0.35


def test_rubric_scores_the_app_not_the_eloquence_of_the_criticism() -> None:
    """The rubric must not reward an articulate takedown of a broken app.

    Measured before this wording: rating 8 real runs spanning ViralScore 0.5 to
    77.5, `substance` separated low from high by only +0.5 and the deliberately
    broken control scored 9/10 -- higher than the worst real app -- because the
    rubric asked for "the QUALITY of reasons given, not how positive people
    were", and a crowd dissecting a broken app gives excellent reasons.
    `word_of_mouth` was worse: -1.2, actively INVERTED, because a broken app
    generates a long critical thread and the rubric measured conversation volume.

    After the rewrite: substance +5.0, severity +6.2, word_of_mouth +6.0, and the
    control scores 0/0/0. This test pins the wording that produced that.
    """
    from viral_bench.score.autorater import _RUBRIC

    # substance must be anchored to observed behaviour, with an explicit cap.
    assert "OBSERVED" in _RUBRIC
    assert "cannot exceed 2" in _RUBRIC.lower()
    assert "takedown of a broken app is LOW substance" in _RUBRIC

    # word_of_mouth must measure spread of advocacy, not volume of talk.
    assert "spread of ADVOCACY, not the volume of conversation" in _RUBRIC
    assert "warning each other off" in _RUBRIC

    # severity must stay anchored to the worst substantiated problem.
    assert "WORST substantiated problem" in _RUBRIC


def test_every_dimension_states_its_direction() -> None:
    """A dimension whose direction is ambiguous is the inversion risk.

    `word_of_mouth` was inverted for exactly this reason: nothing in its rubric
    said which end was good for the APP, only which end was more conversation.
    """
    from viral_bench.score.autorater import _RUBRIC, DIMENSIONS

    for dim in DIMENSIONS:
        assert dim in _RUBRIC, f"{dim} missing from the rubric"
    # Each of the three carries an explicit orientation cue.
    assert "DIRECTION" in _RUBRIC
    assert "DECISIVE RULE" in _RUBRIC
