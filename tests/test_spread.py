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

"""Tests for the propagation axis (peer spread, measured apart from quality).

The module exists because the score's social terms were measured to be reading
something other than what they are named, so these tests pin the distinctions
that finding rests on: founder vs peer, self vs other, and talk vs advocacy.
"""

from __future__ import annotations

import json
import sqlite3

from viral_bench.score.spread import (
    SPREAD_WEIGHTS,
    SpreadSignals,
    extract_spread,
    spread_score,
)

_SCHEMA = """
CREATE TABLE post (post_id INTEGER PRIMARY KEY, user_id INTEGER,
    original_post_id INTEGER, content TEXT, quote_content TEXT,
    created_at INTEGER, num_likes INTEGER DEFAULT 0,
    num_dislikes INTEGER DEFAULT 0, num_shares INTEGER DEFAULT 0,
    num_reports INTEGER DEFAULT 0);
CREATE TABLE comment (comment_id INTEGER PRIMARY KEY, post_id INTEGER,
    user_id INTEGER, content TEXT, created_at INTEGER,
    num_likes INTEGER DEFAULT 0, num_dislikes INTEGER DEFAULT 0);
CREATE TABLE rec (user_id INTEGER, post_id INTEGER);
CREATE TABLE user (user_id INTEGER PRIMARY KEY, agent_id INTEGER,
    user_name TEXT, name TEXT, bio TEXT);
"""


def _run(
    tmp_path,
    *,
    posts: list[tuple],
    comments: list[tuple] = (),
    exposed: int = 4,
    advocates: set[int] = frozenset({1, 2, 3, 4}),
    agents: int = 4,
):
    """Build a minimal crowd run on disk.

    ``posts`` rows are ``(post_id, user_id, original_post_id, quote_content,
    num_likes)``; ``comments`` rows are ``(comment_id, post_id, user_id)``.
    """
    con = sqlite3.connect(tmp_path / "simulation.db")
    con.executescript(_SCHEMA)
    con.execute("INSERT INTO user VALUES (0, 0, 'founder', 'The Founder', NULL)")
    for a in range(1, agents + 1):
        con.execute(
            "INSERT INTO user VALUES (?, ?, ?, ?, NULL)", (a, a, f"agent_{a}", f"A{a}")
        )
    for pid, uid, orig, quote, likes in posts:
        con.execute(
            "INSERT INTO post (post_id, user_id, original_post_id, content, "
            "quote_content, created_at, num_likes) VALUES (?,?,?,'x',?,1,?)",
            (pid, uid, orig, quote, likes),
        )
    for cid, pid, uid in comments:
        con.execute(
            "INSERT INTO comment (comment_id, post_id, user_id, content, created_at) "
            "VALUES (?,?,?,'x',1)",
            (cid, pid, uid),
        )
    for u in range(exposed):
        con.execute("INSERT INTO rec VALUES (?, 1)", (u,))
    con.commit()
    con.close()

    summary = {
        "verdicts": {
            "interviews": {
                "per_agent": [
                    {"agent_id": a, "would_share": a in advocates}
                    for a in range(1, agents + 1)
                ]
            }
        }
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return tmp_path


def test_missing_database_is_unmeasured_not_zero(tmp_path) -> None:
    sig = extract_spread(tmp_path / "nope")
    assert sig.ok is False
    assert spread_score(sig) is None


def test_reposting_the_founder_is_not_peer_spread(tmp_path) -> None:
    """The 68% case: everyone forwards the announcement and nobody reads a peer.

    This is what ``amplification`` mostly counts today, and it requires no
    contact with another agent at all.
    """
    run = _run(
        tmp_path,
        posts=[
            (1, 0, None, None, 0),  # founder launch
            (2, 1, 1, None, 0),  # agent 1 reposts the founder
            (3, 2, 1, None, 0),  # agent 2 reposts the founder
        ],
    )
    sig = extract_spread(run)
    assert sig.founder_repost_participation == 0.5
    assert sig.peer_repost_participation == 0.0
    assert sig.advocate_repost_participation == 0.0
    assert spread_score(sig) == 0.0


def test_reposting_your_own_post_is_not_spread(tmp_path) -> None:
    run = _run(
        tmp_path,
        posts=[
            (1, 0, None, None, 0),
            (2, 1, None, None, 0),  # agent 1's own take
            (3, 1, 2, None, 0),  # agent 1 reposts itself
        ],
    )
    assert extract_spread(run).peer_repost_participation == 0.0


def test_peer_repost_and_quote_are_counted_separately(tmp_path) -> None:
    run = _run(
        tmp_path,
        posts=[
            (1, 0, None, None, 0),
            (2, 1, None, None, 0),  # agent 1's own take
            (3, 2, 2, None, 0),  # agent 2 plain-reposts a peer
            (4, 3, 2, "worth it", 0),  # agent 3 QUOTES a peer
        ],
    )
    sig = extract_spread(run)
    assert sig.peer_repost_participation == 0.25
    assert sig.peer_quote_participation == 0.25


def test_only_agents_who_would_share_count_as_advocacy(tmp_path) -> None:
    """The finding that shapes the module: raw peer volume is a complaint signal.

    Two agents amplify a peer; only one of them would put their name behind the
    app. Raw participation sees two, advocacy sees one.
    """
    run = _run(
        tmp_path,
        posts=[
            (1, 0, None, None, 0),
            (2, 1, None, None, 0),
            (3, 2, 2, None, 0),  # agent 2 amplifies -- and endorses
            (4, 3, 2, None, 0),  # agent 3 amplifies -- but would NOT share
        ],
        advocates={2},
    )
    sig = extract_spread(run)
    assert sig.peer_repost_participation == 0.5
    assert sig.advocate_repost_participation == 0.25


def test_commenting_on_the_founder_is_not_peer_conversation(tmp_path) -> None:
    """76.3% of stored comments hang off the launch post; none of them is spread."""
    run = _run(
        tmp_path,
        posts=[(1, 0, None, None, 0), (2, 1, None, None, 0)],
        comments=[(1, 1, 2), (2, 1, 3), (3, 2, 4)],
    )
    sig = extract_spread(run)
    # Only agent 4, commenting on agent 1's post, is talking to a peer.
    assert sig.peer_comment_participation == 0.25


def test_commenting_on_your_own_post_is_not_peer_conversation(tmp_path) -> None:
    run = _run(
        tmp_path,
        posts=[(1, 0, None, None, 0), (2, 1, None, None, 0)],
        comments=[(1, 2, 1)],
    )
    assert extract_spread(run).peer_comment_participation == 0.0


def test_derived_engagement_share_is_the_strict_cascade(tmp_path) -> None:
    """Engagement on a repost, not on an original. Zero in 91% of real runs."""
    run = _run(
        tmp_path,
        posts=[
            (1, 0, None, None, 6),  # launch post earns 6
            (2, 1, None, None, 2),  # an agent's own take earns 2
            (3, 2, 2, None, 2),  # a repost earns 2
        ],
    )
    sig = extract_spread(run)
    assert sig.derived_engagement_share == 0.2
    # The agent's own post is first generation, not "primary alongside the
    # founder" -- the mis-bucketing that made signals.secondary_share drift.
    assert sig.n_agent_posts == 1


def test_rates_are_per_exposed_agent_so_crowd_size_does_not_inflate_them(
    tmp_path,
) -> None:
    run = _run(
        tmp_path,
        posts=[(1, 0, None, None, 0), (2, 1, None, None, 0), (3, 2, 2, None, 0)],
        exposed=2,
    )
    assert extract_spread(run).peer_repost_participation == 0.5


def test_spread_score_weights_sum_to_one_and_uses_advocacy_fields() -> None:
    assert abs(sum(SPREAD_WEIGHTS.values()) - 1.0) < 1e-9
    assert all(name.startswith("advocate_") for name in SPREAD_WEIGHTS)


def test_spread_score_renormalises_over_measured_signals() -> None:
    """A missing signal is dropped, never scored as zero."""
    full = SpreadSignals(
        crowd_dir="x",
        exposed_agents=10,
        advocate_repost_participation=0.5,
        advocate_quote_participation=0.5,
        advocate_comment_participation=0.5,
    )
    partial = SpreadSignals(
        crowd_dir="x",
        exposed_agents=10,
        advocate_repost_participation=0.5,
        advocate_quote_participation=None,
        advocate_comment_participation=0.5,
    )
    assert spread_score(full) == 50.0
    assert spread_score(partial) == 50.0


def test_spread_score_is_none_without_an_exposed_audience() -> None:
    sig = SpreadSignals(
        crowd_dir="x", exposed_agents=0, advocate_repost_participation=1.0
    )
    assert spread_score(sig) is None


# -- the v5 wiring: does the profile actually reach the database? ------------ #


def test_advocate_amplification_reaches_the_score_from_the_database(tmp_path) -> None:
    """The bug this guards: the signal lives in simulation.db, not the summary.

    ``reach`` was aggregated before anyone asked who a repost was OF, so a v5
    profile that silently read the old summary field would score every run on
    founder reposts while claiming to measure peer advocacy.
    """
    from viral_bench.score.signals import extract_signals

    run = _run(
        tmp_path,
        posts=[
            (1, 0, None, None, 0),
            (2, 1, None, None, 0),  # agent 1's own take
            (3, 2, 2, None, 0),  # agent 2 amplifies a peer -- and endorses
        ],
        advocates={2},
    )
    (run / "run_summary.json").write_text(
        json.dumps(
            {
                "build_id": "demo__1",
                "ok": True,
                "validity": {"does_what_it_claims": True},
                "engagement": {"reach": {"exposed_agents": 4, "actors_reposted": 4}},
                "verdicts": {
                    "interviews": {
                        "n": 4,
                        "would_use_rate": 0.5,
                        "per_agent": [
                            {"agent_id": a, "would_share": a == 2} for a in (1, 2, 3, 4)
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    sig = extract_signals(run)
    # The old field sees everyone; the new one sees the single real advocate.
    assert sig.repost_participation == 1.0
    assert sig.advocate_amplification == 0.25


def test_advocate_amplification_is_unmeasured_without_a_database(tmp_path) -> None:
    """A corpus predating this must re-weight the term out, never score it 0."""
    from viral_bench.score.signals import extract_signals
    from viral_bench.score.viralscore import ScoreWeights, score_run

    (tmp_path / "run_summary.json").write_text(
        json.dumps(
            {
                "build_id": "demo__1",
                "ok": True,
                "validity": {"does_what_it_claims": True},
                "engagement": {"reach": {"exposed_agents": 4, "actors_reposted": 2}},
                "verdicts": {
                    "interviews": {"n": 4, "would_use_rate": 1.0, "per_agent": []}
                },
            }
        ),
        encoding="utf-8",
    )
    sig = extract_signals(tmp_path)
    assert sig.advocate_amplification is None

    result = score_run(sig, ScoreWeights.from_profile("v5_advocacy"))
    assert result.components["advocacy_spread"] is None
    # Re-weighted out: an app nobody could measure spread for still scores well.
    assert result.score is not None and result.score > 50
    assert any("advocacy_spread" in w for w in result.confidence)
