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

"""Propagation signals: did the app move *through* the crowd, or only reach it?

The ViralScore measures how good an app is. Every one of its components turns
out to measure that same thing -- on the stored corpus, restricted to runs whose
app verifiably works, any single component alone reproduces the full ranking at
Spearman 0.84-0.97 (:func:`viral_bench.score.discriminate.redundancy_panel`).
That is a defensible quality index, but it is not a virality metric, and the
terms that sound like virality are not behaving like one:

* ``amplification`` counts distinct agents who reposted **anything**. Of 5,574
  reposting actors sampled across 300 runs, **67.8% reposted only the founder's
  own launch post**. Measured here, ``founder_repost_participation`` correlates
  **+0.95** with the ViralScore -- forwarding the announcement is a proxy for
  "the app is good", not evidence the crowd spread it among itself.
* ``cascade`` is 0-weighted as degenerate. ``signals.secondary_share`` now reads
  0.142 mean, but its "primary" bucket lumps the founder's launch post with all
  thirty agent posts, so 46.7% of that denominator is peer-authored
  first-generation content. Measured strictly here as
  ``derived_engagement_share``, real second-generation spread is **0.005 mean
  and exactly zero in 91% of runs**. It is still dead.

**The finding that shapes this module: raw peer interaction is a complaint
signal, not a virality signal.** Over 738 default-architecture runs:

| definition (per exposed agent) | r vs ViralScore | working apps | broken apps |
|---|---:|---:|---:|
| raw peer amplification | **-0.483** | 0.121 | **0.442** |
| raw peer commenting | -0.113 | 0.173 | 0.224 |
| advocate-only peer amplification | +0.280 | 0.044 | 0.012 |
| advocate-only peer commenting | +0.385 | 0.088 | 0.012 |

A broken app produces *more* peer-to-peer traffic than a working one, because
thirty agents pile on to confirm the same 500. Scoring raw peer volume would
rank a corpse above a product -- which is exactly what the autorater rubric
already warns about ("a broken app that generates a long thread of criticism is
0-3, not 8"). So every scored rate here is restricted to agents whose **own**
stated verdict was ``would_share=yes``: not "how much did the crowd talk" but
"how many people who would put their name behind this passed it on".

Two further exclusions, both deliberate:

* **The founder's own content is not peer content.** Every peer rate resolves
  the target post's author and requires it to be an agent.
* **Self-engagement is not spread.** An agent reposting itself is excluded
  (186 such events exist in the corpus).

Rates are per exposed agent, matching :mod:`viral_bench.score.signals`: raw
counts scale with crowd size and are heavy-tailed, distinct-actor fractions are
bounded in [0,1] and comparable across crowd sizes.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

#: The founder is always user 0 in an OASIS run (see ``crowd/sim/platform.py``).
FOUNDER_ID = 0


@dataclass(frozen=True)
class SpreadSignals:
    """Propagation measured separately from quality, for one crowd run."""

    crowd_dir: str
    exposed_agents: int = 0

    # -- scored: peer engagement BY AGENTS WHO ENDORSE THE APP ---------------
    #: Distinct would-share agents who reposted another agent's post, per
    #: exposed agent. The advocacy spreading, rather than the volume of talk.
    advocate_repost_participation: float | None = None
    #: The same for quotes -- a repost with a stated reason. Quoting has the
    #: highest peer share of any amplification verb (33.5%).
    advocate_quote_participation: float | None = None
    #: Distinct would-share agents who commented on another agent's post.
    #: Commenting on the launch post is talking to the founder, while commenting
    #: on a peer's post is the only unthreaded conversation the platform can
    #: record.
    advocate_comment_participation: float | None = None

    # -- diagnostic: the same rates without the advocacy filter --------------
    #: Kept because they are what proves the filter is necessary. These run
    #: *higher* on broken apps. See the module docstring.
    peer_repost_participation: float | None = None
    peer_quote_participation: float | None = None
    peer_comment_participation: float | None = None
    #: What ``amplification`` is mostly made of today (r=+0.95 with the score).
    founder_repost_participation: float | None = None
    #: Strict multi-generation cascade: engagement landing on a repost or quote
    #: as a share of all post engagement. Zero in 91% of runs.
    derived_engagement_share: float | None = None

    # -- scale context, not scored -------------------------------------------
    n_agent_posts: int = 0
    n_advocates: int = 0

    @property
    def ok(self) -> bool:
        """Whether the run had an exposed audience to measure against."""
        return self.exposed_agents > 0


def _rate(count: int, exposed: int) -> float | None:
    if not exposed:
        return None
    return round(min(1.0, count / exposed), 4)


def _advocate_user_ids(crowd_dir: Path, con: sqlite3.Connection) -> set[int]:
    """Platform user ids of agents whose own verdict was ``would_share=yes``.

    Resolved through the ``user`` table rather than assuming ``user_id ==
    agent_id``. That identity holds in all 222 runs checked, but it is an OASIS
    implementation detail and a silent off-by-one here would not fail loudly --
    it would quietly attribute one agent's advocacy to another.
    """
    summary_path = crowd_dir / "run_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    rows = ((summary.get("verdicts") or {}).get("interviews") or {}).get(
        "per_agent"
    ) or []
    advocates = {r.get("agent_id") for r in rows if r.get("would_share")}
    advocates.discard(None)
    if not advocates:
        return set()
    try:
        return {
            int(uid)
            for uid, aid in con.execute("SELECT user_id, agent_id FROM user")
            if aid in advocates
        }
    except sqlite3.Error:
        return set()


def extract_spread(crowd_dir: str | Path) -> SpreadSignals:
    """Read one crowd run's ``simulation.db`` into :class:`SpreadSignals`.

    Returns an empty (``ok`` False) result rather than raising when the database
    is missing or unreadable. Older runs predate some of this, and a sweep over
    a mixed corpus must not die on one of them -- an unmeasured run is dropped
    by the caller, which is not the same as scoring it zero.
    """
    crowd_dir = Path(crowd_dir)
    db = crowd_dir / "simulation.db"
    if not db.is_file():
        return SpreadSignals(crowd_dir=str(crowd_dir))

    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return SpreadSignals(crowd_dir=str(crowd_dir))

    def scalar(sql: str, *args) -> int:
        row = con.execute(sql, args).fetchone()
        return int(row[0] or 0) if row else 0

    def actors(sql: str) -> set[int]:
        return {int(r[0]) for r in con.execute(sql) if r[0] is not None}

    try:
        exposed = scalar("SELECT COUNT(DISTINCT user_id) FROM rec")
        advocates = _advocate_user_ids(crowd_dir, con)

        # Derived content (repost or quote) whose ORIGINAL was written by
        # another agent. `quote_content IS NOT NULL` is what separates a quote
        # from a plain repost in the OASIS schema.
        peer_derived = (
            "FROM post p JOIN post o ON p.original_post_id = o.post_id "
            f"WHERE o.user_id != {FOUNDER_ID} AND p.user_id != o.user_id"
        )
        reposters = actors(
            f"SELECT DISTINCT p.user_id {peer_derived} AND p.quote_content IS NULL"
        )
        quoters = actors(
            f"SELECT DISTINCT p.user_id {peer_derived} AND p.quote_content IS NOT NULL"
        )
        commenters = actors(
            "SELECT DISTINCT c.user_id FROM comment c "
            "JOIN post p ON c.post_id = p.post_id "
            f"WHERE p.user_id != {FOUNDER_ID} AND c.user_id != p.user_id"
        )
        founder_reposters = scalar(
            "SELECT COUNT(DISTINCT p.user_id) FROM post p "
            "JOIN post o ON p.original_post_id = o.post_id "
            f"WHERE o.user_id = {FOUNDER_ID}"
        )

        derived_engagement = scalar(
            "SELECT COALESCE(SUM(num_likes + num_shares + num_dislikes), 0) "
            "FROM post WHERE original_post_id IS NOT NULL"
        )
        all_engagement = scalar(
            "SELECT COALESCE(SUM(num_likes + num_shares + num_dislikes), 0) FROM post"
        )
        n_agent_posts = scalar(
            "SELECT COUNT(*) FROM post "
            f"WHERE original_post_id IS NULL AND user_id != {FOUNDER_ID}"
        )
    except sqlite3.Error:
        con.close()
        return SpreadSignals(crowd_dir=str(crowd_dir))
    con.close()

    return SpreadSignals(
        crowd_dir=str(crowd_dir),
        exposed_agents=exposed,
        advocate_repost_participation=_rate(len(reposters & advocates), exposed),
        advocate_quote_participation=_rate(len(quoters & advocates), exposed),
        advocate_comment_participation=_rate(len(commenters & advocates), exposed),
        peer_repost_participation=_rate(len(reposters), exposed),
        peer_quote_participation=_rate(len(quoters), exposed),
        peer_comment_participation=_rate(len(commenters), exposed),
        founder_repost_participation=_rate(founder_reposters, exposed),
        derived_engagement_share=(
            round(derived_engagement / all_engagement, 4) if all_engagement else None
        ),
        n_agent_posts=n_agent_posts,
        n_advocates=len(advocates),
    )


#: Weights for :func:`spread_score`. Deliberately uncalibrated: the first
#: question is whether a propagation axis is *separable* from the quality axis
#: and *points the right way*, and both are properties of the signals rather
#: than of the weighting. Tuning these before that is settled would be fitting a
#: number nobody has shown means anything. Quotes are folded in with reposts
#: because they are the same act with a reason attached, and alone they are zero
#: in 77% of runs.
SPREAD_WEIGHTS: dict[str, float] = {
    "advocate_repost_participation": 0.40,
    "advocate_quote_participation": 0.20,
    "advocate_comment_participation": 0.40,
}


def spread_score(sig: SpreadSignals) -> float | None:
    """A 0-100 propagation index, or ``None`` when nothing was measurable.

    Reported *beside* the ViralScore, never inside it. Folding it in would
    re-create the problem it exists to expose: one number cannot say both "the
    app is good" and "the crowd passed it on", and averaging them hides which
    one moved. Measured on 605 clean runs it correlates +0.40 with the
    ViralScore -- related, as it should be, but far from the 0.84-0.97 that every
    existing component manages.
    """
    if not sig.ok:
        return None
    pairs = [
        (getattr(sig, name), weight)
        for name, weight in SPREAD_WEIGHTS.items()
        if getattr(sig, name) is not None
    ]
    total = sum(weight for _, weight in pairs)
    if not pairs or total <= 0:
        return None
    return round(100.0 * sum(v * w for v, w in pairs) / total, 1)
