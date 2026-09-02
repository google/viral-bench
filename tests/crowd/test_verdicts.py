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

"""Tests for crowd verdict parsing + aggregation (oasis/camel-free)."""

from __future__ import annotations

import json
import sqlite3

from viral_bench.crowd.sim.verdicts import (
    assess_run_health,
    conversion_stats,
    distribution,
    interview_verdicts,
    normalize_reply,
    parse_bool_field,
    parse_score,
    trier_verdicts,
)

# -- field parsing ----------------------------------------------------------- #


def test_parse_bool_field_variants() -> None:
    text = "would_use: yes\nwould_share: no\n"
    assert parse_bool_field(text, "would_use") is True
    assert parse_bool_field(text, "would_share") is False
    assert parse_bool_field("WOULD_USE = Y", "would_use") is True
    assert parse_bool_field("nothing here", "would_use") is None


def test_parse_score_clamped_and_optional() -> None:
    assert parse_score("score: 7") == 7
    assert parse_score("delight: 10") == 10
    assert parse_score("score: 42") == 10  # clamped
    assert parse_score("no score line") is None


# -- distribution ------------------------------------------------------------ #


def test_distribution_computes_rates_and_spread() -> None:
    rows = [
        {"would_use": True, "would_share": True, "delight": 9},
        {"would_use": True, "would_share": False, "delight": 5},
        {"would_use": False, "would_share": False, "delight": 2},
        {"would_use": False, "would_share": False, "delight": 4},
    ]
    d = distribution(rows)
    assert d["n"] == 4
    assert d["would_use_rate"] == 0.5
    assert d["would_share_rate"] == 0.25
    assert d["delight_min"] == 2 and d["delight_max"] == 9
    assert d["delight_mean"] == 5.0
    assert d["delight_stdev"] > 0  # there IS spread (the whole point)
    # histogram is indexed 0..10
    assert len(d["delight_histogram"]) == 11
    assert d["delight_histogram"][9] == 1 and d["delight_histogram"][2] == 1


def test_distribution_ignores_missing_fields() -> None:
    rows = [
        {"would_use": True, "would_share": None, "delight": None},
        {"would_use": None, "would_share": False, "delight": 6},
    ]
    d = distribution(rows)
    assert d["n"] == 2
    assert d["would_use_rate"] == 1.0  # only 1 non-None
    assert d["would_share_rate"] == 0.0
    assert d["delight_mean"] == 6.0  # only 1 non-None


def test_distribution_empty() -> None:
    d = distribution([])
    assert d["n"] == 0
    assert "delight_mean" not in d  # nothing to average


def test_distribution_reports_facet_means() -> None:
    rows = [
        {"delight": 8, "functionality": 9, "usability": 7, "design": 6, "craft": 7.5},
        {"delight": 4, "functionality": 5, "usability": 3, "design": 2, "craft": 3.5},
    ]
    d = distribution(rows)
    assert d["functionality_mean"] == 7.0
    assert d["design_mean"] == 4.0
    assert d["craft_mean"] == 5.5  # the multi-item signal the score reads


def test_distribution_splits_breadth_from_in_audience_resonance() -> None:
    # A dev tool: loved by the two it targets, correctly rejected by the others.
    rows = [
        {"would_use": True, "would_share": True, "delight": 8, "for_me": True},
        {"would_use": True, "would_share": True, "delight": 8, "for_me": True},
        {"would_use": False, "would_share": False, "delight": 2, "for_me": False},
        {"would_use": False, "would_share": False, "delight": 1, "for_me": False},
    ]
    d = distribution(rows)
    assert d["would_use_rate"] == 0.5  # breadth across the whole crowd
    assert d["audience_fit_rate"] == 0.5
    assert d["in_audience"]["n"] == 2
    assert d["in_audience"]["would_use_rate"] == 1.0  # resonance within audience
    assert d["in_audience"]["delight_mean"] == 8.0


# -- trier + interview aggregation ------------------------------------------ #


class _FakeVerdict:
    def __init__(self, use, share, delight, craft=None, facets=()):
        self.would_use, self.would_share, self.delight = use, share, delight
        self.craft = craft
        names = ("functionality", "usability", "design", "simplicity")
        for name, value in zip(names, facets or (None,) * 4, strict=False):
            setattr(self, name, value)


class _FakeToolkit:
    def __init__(self, verdict, degraded=False):
        self.trace = type("T", (), {"verdict": verdict, "degraded": degraded})()


class _FakePersona:
    def __init__(self, username):
        self.username = username


class _FakeCrowd:
    def __init__(self):
        self.trier_ids = [1, 2]
        self.crowd_ids = [1, 2]
        self.persona_by_id = {1: _FakePersona("maya"), 2: _FakePersona("raj")}
        self.tier_by_id = {1: "trier", 2: "trier"}
        self.trier_toolkits = {
            1: _FakeToolkit(_FakeVerdict(True, True, 8)),
            2: _FakeToolkit(None),  # never finished
        }


def test_trier_verdicts_marks_unfinished() -> None:
    d = trier_verdicts(_FakeCrowd())
    assert d["n"] == 2
    by_agent = {r["agent_id"]: r for r in d["per_agent"]}
    assert by_agent[1]["finished"] is True and by_agent[1]["delight"] == 8
    assert by_agent[2]["finished"] is False and by_agent[2]["delight"] is None
    # only the finished trier contributes to the delight stats
    assert d["delight_mean"] == 8.0


def test_interview_verdicts_parses_db(tmp_path) -> None:
    db = tmp_path / "sim.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE trace (user_id INTEGER, action TEXT, info TEXT)")
    r1 = json.dumps({"response": "would_use: yes\nwould_share: no\nscore: 7"})
    r2 = json.dumps({"response": "would_use: no\nwould_share: no\nscore: 3"})
    other = json.dumps({"content": "hi"})  # ignored (not an interview row)
    con.execute("INSERT INTO trace VALUES (?,?,?)", (1, "interview", r1))
    con.execute("INSERT INTO trace VALUES (?,?,?)", (2, "interview", r2))
    con.execute("INSERT INTO trace VALUES (?,?,?)", (0, "create_post", other))
    con.commit()
    con.close()

    crowd = _FakeCrowd()
    d = interview_verdicts(str(db), crowd)
    assert d["n"] == 2
    assert d["would_use_rate"] == 0.5
    assert d["would_share_rate"] == 0.0
    assert d["delight_min"] == 3 and d["delight_max"] == 7
    # attributed to the right personas
    assert {r["username"] for r in d["per_agent"]} == {"maya", "raj"}


def test_interview_verdicts_dedupes_repeat_interviews(tmp_path) -> None:
    # A batch interview failure is retried per-agent, which can re-ask someone.
    # Counting both answers would double-weight that persona in the score.
    db = tmp_path / "sim.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE trace (user_id INTEGER, action TEXT, info TEXT)")
    first = json.dumps({"response": "would_use: yes\nwould_share: yes\nscore: 8"})
    again = json.dumps({"response": "would_use: no\nwould_share: no\nscore: 2"})
    con.execute("INSERT INTO trace VALUES (?,?,?)", (1, "interview", first))
    con.execute("INSERT INTO trace VALUES (?,?,?)", (1, "interview", again))
    con.commit()
    con.close()

    d = interview_verdicts(str(db), _FakeCrowd())
    assert d["n"] == 1  # one row per agent
    row = d["per_agent"][0]
    assert row["delight"] == 8 and row["would_use"] is True  # first answer wins


def test_interview_verdicts_skips_unparseable_non_answers(tmp_path) -> None:
    # A skipped turn records "(no response)". Counting it would inflate n with a
    # row carrying no signal, making a degraded run look better-evidenced.
    db = tmp_path / "sim.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE trace (user_id INTEGER, action TEXT, info TEXT)")
    real = json.dumps({"response": "would_use: yes\nwould_share: no\nscore: 6"})
    empty = json.dumps({"response": "(no response)"})
    con.execute("INSERT INTO trace VALUES (?,?,?)", (1, "interview", real))
    con.execute("INSERT INTO trace VALUES (?,?,?)", (2, "interview", empty))
    con.commit()
    con.close()

    d = interview_verdicts(str(db), _FakeCrowd())
    assert d["n"] == 1  # only the agent that answered
    assert d["per_agent"][0]["agent_id"] == 1


# -- robust parsing + drop accounting ---------------------------------------
#
# Two verdicts in the stored corpus were lost to markdown decoration, both of
# them a "yes", and 43 duplicate replies were discarded with no counter. A
# parser that silently eats replies is indistinguishable from a crowd that never
# answered, and n is what the score's confidence checks read.


def test_markdown_bold_field_names_still_parse() -> None:
    """The exact string that was lost: ``**would_use**: yes``."""
    reply = "**would_use**: yes\n**would_share**: no\n**score**: 8"
    assert parse_bool_field(reply, "would_use") is True
    assert parse_bool_field(reply, "would_share") is False
    assert parse_score(reply) == 8


def test_bulleted_field_names_still_parse() -> None:
    reply = "- would_use: yes\n- would_share: yes\n- delight: 7"
    assert parse_bool_field(reply, "would_use") is True
    assert parse_score(reply) == 7


def test_numbered_and_backticked_fields_parse() -> None:
    reply = "1. `would_use`: no\n2. `score`: 3"
    assert parse_bool_field(reply, "would_use") is False
    assert parse_score(reply) == 3


def test_json_replies_parse() -> None:
    reply = 'Sure:\n{"would_use": true, "would_share": false, "delight": 9}'
    assert parse_bool_field(reply, "would_use") is True
    assert parse_bool_field(reply, "would_share") is False
    assert parse_score(reply) == 9


def test_plain_format_still_parses() -> None:
    reply = "would_use: yes\nwould_share: no\nscore: 5"
    assert parse_bool_field(reply, "would_use") is True
    assert parse_score(reply) == 5


def test_underscores_in_field_names_survive_normalization() -> None:
    """Normalization must not eat the underscore in ``would_use``."""
    assert "would_use" in normalize_reply("**would_use**: yes")


def test_distribution_reports_dropped_replies() -> None:
    rows = [{"agent_id": 1, "would_use": True, "would_share": True, "delight": 6}]
    dist = distribution(rows, {"duplicate": 3, "unparseable": 1})
    assert dist["n"] == 1
    assert dist["n_dropped"] == 4
    assert dist["dropped"]["duplicate"] == 3
    assert dist["loss_rate"] == 0.8  # 4 of 5 considered


def test_distribution_without_drops_is_unchanged() -> None:
    dist = distribution([{"agent_id": 1, "would_use": True}])
    assert "n_dropped" not in dist


# -- run health --------------------------------------------------------------
#
# Six stored runs report ok:true having completed 2 of 4 rounds and collected
# zero interviews. They were then scored off craft alone -- the most stable
# component -- so losing the primary signal made the number look MORE confident.


def _healthy(**over):
    kw = dict(
        error=None,
        rounds_run=4,
        rounds_expected=4,
        round_log=[{"round": i, "ok": True} for i in range(1, 5)],
        verdicts={"interviews": {"n": 50, "loss_rate": 0.0}},
        interview_expected=True,
    )
    kw.update(over)
    return assess_run_health(**kw)


def test_a_complete_run_is_healthy() -> None:
    assert _healthy() == {"ok": True, "failures": []}


def test_zero_interviews_fails_the_run() -> None:
    h = _healthy(verdicts={"interviews": {"n": 0}})
    assert h["ok"] is False
    assert any("primary scoring signal" in f for f in h["failures"])


def test_lost_rounds_fail_the_run() -> None:
    h = _healthy(
        rounds_run=2,
        round_log=[
            {"round": 1, "ok": True},
            {"round": 2, "ok": True},
            {"round": 3, "ok": False},
            {"round": 4, "ok": False},
        ],
    )
    assert h["ok"] is False
    assert any("2 of 4 rounds" in f for f in h["failures"])
    assert any("[3, 4]" in f for f in h["failures"])


def test_high_verdict_loss_fails_the_run() -> None:
    h = _healthy(verdicts={"interviews": {"n": 18, "expected": 30}})
    assert h["ok"] is False
    assert any(
        "40% of the crowd produced no usable verdict" in f for f in h["failures"]
    )


def test_small_verdict_loss_is_tolerated() -> None:
    assert _healthy(verdicts={"interviews": {"n": 49, "expected": 50}})["ok"]


def test_a_repair_pass_that_WORKED_does_not_fail_the_run() -> None:
    """Retries are not lost data, and conflating them threw healthy runs away.

    Loss was dropped/(rows+dropped) -- failed *attempts* over attempts -- so a
    run whose repair pass recovered every agent was recorded as having lost 12%
    of its crowd. All 14 runs of the v11 sweep heard from 30 of 30 agents and
    one was failed anyway. The gate fires hardest where the model dropped turns,
    which tracks load and the apps that generate the most output.
    """
    h = _healthy(
        verdicts={
            "interviews": {"n": 30, "expected": 30, "retry_rate": 0.118, "n_dropped": 4}
        }
    )
    assert h["ok"] is True, h["failures"]


def test_loss_falls_back_to_the_old_field_for_runs_without_coverage() -> None:
    # Summaries written before coverage existed must still be judged.
    h = _healthy(verdicts={"interviews": {"n": 30, "loss_rate": 0.4}})
    assert h["ok"] is False


def test_an_exception_still_fails_the_run() -> None:
    h = _healthy(error="KeyError: 'post_id'")
    assert h["ok"] is False
    assert any("run raised" in f for f in h["failures"])


def test_no_llm_smoke_does_not_require_interviews() -> None:
    """A scripted wiring smoke has no crowd verdicts by design."""
    h = _healthy(verdicts={}, interview_expected=False)
    assert h["ok"] is True


# -- a throttled run is not an unengaging app --------------------------------


def test_health_fails_when_too_many_model_turns_were_lost() -> None:
    """Throttling must not be reported as crowd indifference.

    OASIS records a skipped turn as an agent choosing to do nothing, so a run
    made while the API is rate-limiting reads as an app nobody engaged with. If
    two founder models are ever measured under different load, that becomes a
    difference between the models.
    """
    from viral_bench.crowd.sim.verdicts import assess_run_health

    base = dict(
        error=None,
        rounds_run=4,
        rounds_expected=4,
        round_log=[],
        verdicts={"interviews": {"n": 30, "loss_rate": 0.0}},
        interview_expected=True,
    )
    healthy = assess_run_health(
        **base, turn_stats={"turns": 100, "skipped": 5, "skipped_rate_limited": 5}
    )
    assert healthy["ok"]

    throttled = assess_run_health(
        **base, turn_stats={"turns": 100, "skipped": 40, "skipped_rate_limited": 38}
    )
    assert not throttled["ok"]
    assert "40 of 100 model turns were lost" in throttled["failures"][0]
    assert "38 to rate limits" in throttled["failures"][0]


def test_turns_cut_off_at_the_iteration_ceiling_fail_the_run() -> None:
    """Budget exhaustion is harness silence, exactly like a skipped turn.

    CAMEL enforces ``max_iteration`` by breaking out of its tool-call loop with
    no exception and no log line, so an agent cut off mid-turn is recorded as
    one that had nothing to say. The budget arithmetic is meant to make this
    impossible, and this is what notices when it has not.
    """
    from viral_bench.crowd.sim.verdicts import assess_run_health

    base = dict(
        error=None,
        rounds_run=4,
        rounds_expected=4,
        round_log=[],
        verdicts={"interviews": {"n": 30, "loss_rate": 0.0}},
        interview_expected=True,
    )
    # One truncated turn in 100 is noise, not a broken guarantee.
    healthy = assess_run_health(
        **base, turn_stats={"turns": 100, "budget_exhausted": 1}
    )
    assert healthy["ok"]

    starved = assess_run_health(
        **base, turn_stats={"turns": 100, "budget_exhausted": 20}
    )
    assert not starved["ok"]
    assert "hit the max_iteration ceiling" in starved["failures"][0]
    assert "social_headroom" in starved["failures"][0]


def test_health_without_turn_stats_is_unchanged() -> None:
    from viral_bench.crowd.sim.verdicts import assess_run_health

    result = assess_run_health(
        error=None,
        rounds_run=4,
        rounds_expected=4,
        round_log=[],
        verdicts={"interviews": {"n": 30, "loss_rate": 0.0}},
        interview_expected=True,
    )
    assert result["ok"]


def test_persistence_is_measured_over_agents_who_checked_not_all_agents():
    """Silence is not a "no".

    A trial that never reloaded says nothing about whether the app saved
    anything. Counting its silence as a failure would report the crowd's
    incuriosity as a defect in the app, so the denominator is agents who
    looked, and how many that was travels with the number.
    """
    rows = [
        {"agent_id": 1, "work_survived": True, "saw_other_users": None},
        {"agent_id": 2, "work_survived": False, "saw_other_users": True},
        {"agent_id": 3, "work_survived": None, "saw_other_users": None},
        {"agent_id": 4, "saw_other_users": False},
    ]
    dist = distribution(rows)
    assert dist["work_survived_checked"] == 2
    assert dist["work_survived_rate"] == 0.5
    assert dist["saw_other_users_checked"] == 2
    assert dist["saw_other_users_rate"] == 0.5


def test_nobody_checking_reports_no_rate_at_all():
    dist = distribution([{"agent_id": 1}, {"agent_id": 2}])
    assert dist["work_survived_checked"] == 0
    assert "work_survived_rate" not in dist


class _FakeStep:
    def __init__(self, action):
        self.action = action


class _FakeTrace:
    def __init__(self, actions, verdict=None, reachable=None):
        self.steps = [_FakeStep(a) for a in actions]
        self.verdict = verdict
        self.app_reachable = reachable

    def had_effect(self):
        return any(s.action == "click" for s in self.steps)


class _LateCrowd:
    """22 triers is overkill for a test, and two of each tier is the whole shape."""

    def __init__(self, late_opened: bool):
        self.trier_ids = [1]
        self.latecomer_ids = [2]
        self.reactor_ids = [3]
        self.crowd_ids = [1, 2, 3]
        self.hands_on_ids = [1, 2]
        self.persona_by_id = {
            1: _FakePersona("maya"),
            2: _FakePersona("raj"),
            3: _FakePersona("li"),
        }
        self.tier_by_id = {1: "trier", 2: "latecomer", 3: "reactor"}
        late = _FakeToolkit(_FakeVerdict(True, True, 7) if late_opened else None)
        late.trace = _FakeTrace(
            ["open", "click", "finish"] if late_opened else [],
            verdict=_FakeVerdict(True, True, 7) if late_opened else None,
            reachable=True if late_opened else None,
        )
        first = _FakeToolkit(_FakeVerdict(False, False, 4))
        first.trace = _FakeTrace(
            ["open", "click", "finish"],
            verdict=_FakeVerdict(False, False, 4),
            reachable=True,
        )
        self.trier_toolkits = {1: first, 2: late}


def test_a_latecomer_the_feed_never_convinced_is_not_a_bad_review():
    """Its silence is a conversion failure, not a craft rating.

    Counting a not-yet-user's absent verdict as hands-on evidence would mark an
    app down twice for the same thing: once for failing to interest that person
    and again for a trial it never ran.
    """
    crowd = _LateCrowd(late_opened=False)
    conv = conversion_stats(crowd, {2: {"decided": False, "because": "no reason"}})
    assert conv["n_latecomers"] == 1
    assert conv["n_answered"] == 1
    assert conv["n_decided_yes"] == 0
    assert conv["conversion_rate"] == 0.0
    assert conv["n_converted"] == 0
    # Only the real trier's verdict is hands-on evidence.
    assert trier_verdicts(crowd)["n"] == 1


def test_a_latecomer_the_feed_DID_convince_counts_as_hands_on():
    crowd = _LateCrowd(late_opened=True)
    conv = conversion_stats(crowd, {2: {"decided": True, "convinced_by": "@maya"}})
    assert conv["n_decided_yes"] == 1 and conv["conversion_rate"] == 1.0
    assert conv["n_converted"] == 1 and conv["opened_rate"] == 1.0
    assert conv["per_agent"][0]["convinced_by"] == "@maya"
    assert trier_verdicts(crowd)["n"] == 2


def test_a_decision_nobody_gave_is_not_counted_as_a_no():
    """An unparseable answer is missing data, not a refusal to try.

    Scoring it as a "no" would let a model failure read as an app that nobody
    wanted, which is the same class of error as recording a skipped turn as
    crowd indifference.
    """
    conv = conversion_stats(_LateCrowd(late_opened=False), {})
    assert conv["n_answered"] == 0
    assert conv["conversion_rate"] is None


def test_conversion_is_absent_rather_than_zero_without_the_tier():
    class _NoLate(_LateCrowd):
        def __init__(self):
            super().__init__(late_opened=False)
            self.latecomer_ids = []

    conv = conversion_stats(_NoLate())
    assert conv["n_latecomers"] == 0
    assert conv["conversion_rate"] is None
