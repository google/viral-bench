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

"""The mock social platform: a Twitter-like feed with an interest-based recsys.

Virality is about amplification through a feed, so we use OASIS's Twitter-like
platform with the **TWHIN-BERT** recommender (``recsys_type="twhin-bert"``): posts
are surfaced to agents by interest similarity, which is what lets a genuinely
appealing app cascade through the crowd rather than being seen by everyone
uniformly. A lighter ``"twitter"`` history-based recsys is available for cheap
wiring smokes (no model download).

Anti-gaming: ``allow_self_rating=False`` so the founder can't like its own launch.
Feed reach is widened (``refresh_rec_post_count`` / ``following_post_count``) so
that, at the small crowd sizes we start with, the launch and the early-adopters'
takes reliably reach agents' feeds.

This module also patches upstream OASIS recsys bugs that otherwise make every
large crowd run unusable -- see :func:`_patch_oasis_recsys`.
"""

from __future__ import annotations

import logging
from ast import literal_eval
from dataclasses import dataclass

import oasis
from oasis.social_platform.platform import Platform

from viral_bench import config as _config

_LOG = logging.getLogger("viral_bench.crowd.platform")


def _trace_post_ids(trace_table, user_id=None, action=None) -> list:
    """Post ids from trace rows, read from the ``info`` JSON where they live.

    The ``trace`` table's columns are ``(user_id, created_at, action, info)`` --
    there is no ``post_id`` column, though upstream indexes one in two places.
    """
    out = []
    for trace in trace_table:
        if user_id is not None and trace.get("user_id") != user_id:
            continue
        if action is not None and trace.get("action") != action:
            continue
        info = trace.get("info")
        if isinstance(info, str):
            try:
                info = literal_eval(info)
            except (ValueError, SyntaxError):
                continue
        if isinstance(info, dict) and "post_id" in info:
            out.append(info["post_id"])
    return out


def _patch_oasis_recsys() -> None:
    """Fix the upstream ``twitter`` recsys, which cannot run at benchmark scale.

    ``rec_sys_personalized_with_trace`` has three defects that only fire once a
    run has MORE posts than ``max_rec_post_len`` -- the cheap branch above that
    threshold hides them entirely:

    1. ``get_trace_contents`` reads ``trace['post_id']``, a column that does not
       exist (the id is inside the ``info`` JSON, as OASIS's own sibling helper
       ``get_like_post_id`` correctly parses). Raises ``KeyError: 'post_id'``.
    2. The like/dislike similarity block calls ``model.encode(...)``
       unconditionally, even though the *same function* guards ``if model is not
       None`` a few lines earlier. ``recsys.model`` is ``None`` for this recsys,
       so the moment any agent has liked something this raises
       ``AttributeError: 'NoneType' object has no attribute 'encode'``.
    3. The ``swap_rate`` block repeats defect 1 with another ``trace['post_id']``.

    Why this matters more than a normal upstream bug: the threshold is crossed
    only when a run generates enough discussion, so it destroys the end-of-run
    interview -- the benchmark's primary signal -- on precisely the apps that
    provoke the most conversation. Measured across 41 stored runs, all 9 that
    crossed the threshold lost 100% of their interviews and all 32 below it kept
    them. That is a systematic bias against engaging apps, i.e. against exactly
    the differences this benchmark exists to detect.

    We reimplement the function faithfully (same scoring, same ordering, same
    swap behaviour) with the three defects fixed. Idempotent.
    """
    import random

    import numpy as np
    from oasis.social_platform import recsys
    from oasis.social_platform.typing import ActionType

    if getattr(recsys.rec_sys_personalized_with_trace, "_viral_bench_patched", False):
        return

    def get_trace_contents(user_id, action, post_table, trace_table):
        ids = _trace_post_ids(trace_table, user_id=user_id, action=action)
        return [p["content"] for p in post_table if p["post_id"] in ids]

    def _cos(model, a: str, b: str) -> float:
        ea, eb = model.encode(a), model.encode(b)
        denom = np.linalg.norm(ea) * np.linalg.norm(eb)
        return float(np.dot(ea, eb) / denom) if denom else 0.0

    def _mean_similarity(model, content: str, others: list) -> float:
        # Guard the model: without an embedding model there is no text
        # similarity to compute, so like/dislike history contributes nothing
        # rather than crashing (upstream's own base-similarity branch already
        # degrades this way).
        if not others or model is None:
            return 0.0
        return sum(_cos(model, content, o) for o in others) / len(others)

    def rec_sys_personalized_with_trace(
        user_table,
        post_table,
        trace_table,
        rec_matrix,
        max_rec_post_len,
        swap_rate: float = 0.1,
    ):
        model = recsys.model
        new_rec_matrix = []
        post_ids = [p["post_id"] for p in post_table]
        if len(post_ids) <= max_rec_post_len:
            return [post_ids] * (len(rec_matrix) - 1)

        for idx in range(1, len(rec_matrix)):
            user_id = user_table[idx - 1]["user_id"]
            user_bio = user_table[idx - 1]["bio"]
            available = [
                (p["post_id"], p["content"])
                for p in post_table
                if p["user_id"] != user_id
            ]
            likes = get_trace_contents(
                user_id, ActionType.LIKE_POST.value, post_table, trace_table
            )
            dislikes = get_trace_contents(
                user_id, ActionType.UNLIKE_POST.value, post_table, trace_table
            )

            post_scores = [
                (
                    pid,
                    _cos(model, user_bio, content)
                    if model is not None
                    else random.random(),
                )
                for pid, content in available
            ]

            new_post_scores = []
            for pid, base in post_scores:
                content = post_table[post_ids.index(pid)]["content"]
                adjusted = recsys.normalize_similarity_adjustments(
                    post_scores,
                    base,
                    _mean_similarity(model, content, likes),
                    _mean_similarity(model, content, dislikes),
                )
                new_post_scores.append((pid, adjusted))

            new_post_scores.sort(key=lambda x: x[1], reverse=True)
            rec_post_ids = [pid for pid, _ in new_post_scores[:max_rec_post_len]]

            if swap_rate > 0:
                seen = set(_trace_post_ids(trace_table))
                swap_free = [
                    pid
                    for pid in post_ids
                    if pid not in rec_post_ids and pid not in seen
                ]
                rec_post_ids = recsys.swap_random_posts(
                    rec_post_ids, swap_free, swap_rate
                )
            new_rec_matrix.append(rec_post_ids)
        return new_rec_matrix

    rec_sys_personalized_with_trace._viral_bench_patched = True
    get_trace_contents._viral_bench_patched = True
    recsys.get_trace_contents = get_trace_contents
    recsys.rec_sys_personalized_with_trace = rec_sys_personalized_with_trace
    # platform.py imported the originals by name at import time.
    try:
        from oasis.social_platform import platform as _oasis_platform

        _oasis_platform.rec_sys_personalized_with_trace = (
            rec_sys_personalized_with_trace
        )
    except Exception:  # noqa: BLE001 - best effort; module layout may change
        pass
    _LOG.debug("patched oasis twitter recsys (3 upstream defects at scale)")


def _patch_oasis_feed_identity() -> None:
    """Put NAMES in the feed. Agents cannot talk about each other without them.

    OASIS renders a feed as a JSON dump of post dicts whose only author field is
    a numeric ``user_id``, and its repost/quote strings read literally "User 13
    reposted a post from User 4". Measured over 45 stored runs: **zero** of 1,346
    agent texts named another agent, and all 7 @mentions read "@User 13".

    That is not cosmetic. Word-of-mouth is the mechanism this benchmark exists to
    measure, and the reactors -- three quarters of the crowd -- judge entirely
    from this feed. Their verdicts separate a good app from a broken one far
    worse than the hands-on triers' do (advocacy: 0.225 vs 0.923 good-minus-
    broken over 27 calibration runs), and an anonymous feed is a large part of
    why: you cannot be persuaded by someone whose taste you cannot identify.

    The patch adds an ``author`` field to every post and comment and rewrites
    "User N" inside repost/quote strings, resolved from the ``user`` table. It
    changes no OASIS semantics -- same rows, same ordering, one extra key.
    Idempotent.
    """
    import re

    from oasis.social_platform.platform_utils import PlatformUtils

    original = PlatformUtils._add_comments_to_posts
    if getattr(original, "_viral_bench_patched", False):
        return

    _user_re = re.compile(r"\bUser (\d+)\b")

    def _handles(self) -> dict[int, str]:
        try:
            self.db_cursor.execute("SELECT user_id, user_name, name FROM user")
            rows = self.db_cursor.fetchall()
        except Exception:  # noqa: BLE001 - a missing table must not kill a feed
            return {}
        return {
            uid: (f"@{uname} ({name})" if name else f"@{uname}")
            for uid, uname, name in rows
            if uname
        }

    def _add_comments_to_posts(self, posts_results):
        posts = original(self, posts_results)
        handles = _handles(self)
        if not handles:
            return posts

        def label(uid) -> str:
            return handles.get(uid, f"User {uid}")

        for post in posts:
            post["author"] = label(post.get("user_id"))
            content = post.get("content")
            if isinstance(content, str):
                post["content"] = _user_re.sub(
                    lambda m: label(int(m.group(1))), content
                )
            for comment in post.get("comments", []) or []:
                comment["author"] = label(comment.get("user_id"))
        return posts

    _add_comments_to_posts._viral_bench_patched = True
    PlatformUtils._add_comments_to_posts = _add_comments_to_posts
    _LOG.debug("patched oasis feed to carry author handles")


_patch_oasis_recsys()
_patch_oasis_feed_identity()

# Recsys string values (RecsysType enum values). TWHIN-BERT is interest-based;
# "twitter" is a lighter history-based recsys with no model download.
RECSYS_TWHIN = "twhin-bert"
RECSYS_TWITTER = "twitter"


@dataclass
class PlatformConfig:
    """Tunables for the mock platform + recommender."""

    recsys_type: str = RECSYS_TWHIN
    allow_self_rating: bool = False  # founder can't rate its own content
    #: Posts surfaced per refresh.
    refresh_rec_post_count: int = _config.crowd_feed("refresh_posts", 20)
    #: Max posts in a user's rec buffer. In OASIS this doubles as the threshold
    #: that switches the recommender between two algorithms: at or below it
    #: every agent is handed every post, above it TWHIN-BERT ranks a
    #: personalised top-N by similarity between the reader's bio and the post.
    #:
    #: The history here is a lesson in second-order effects. It was 10, which a
    #: 20-agent run crossed partway through, so one run mixed two recommendation
    #: regimes. The fix was to raise it above any post count a run could reach
    #: -- 40 -- which made runs comparable and made **every agent's feed
    #: byte-identical in 47 of 47 runs**. Comparable, and measuring nothing: the
    #: one mechanism by which a better app can earn distribution was switched
    #: off, so `amplification` and `cascade` had no way to reflect quality.
    #:
    #: 20 is chosen so the ranked branch runs for the whole of a scored run
    #: rather than partway through it. Every agent now publishes its own take,
    #: so a 30-agent run produces 35-45 posts from the first social round
    #: onward, comfortably above the threshold and below any plausible collapse
    #: back under it. Feeds are then 20 posts each, ranked per reader -- big
    #: enough to see the discussion, small enough that being seen is a thing an
    #: app's advocates have to win.
    max_rec_post_len: int = _config.crowd_feed("max_rec_posts", 20)
    #: Posts pulled from accounts this agent follows. Raised with the feed: the
    #: follow graph now has interest neighbourhoods, and this is the channel
    #: through which a neighbourhood's enthusiasm actually reaches its members.
    following_post_count: int = _config.crowd_feed("following_posts", 15)
    use_openai_embedding: bool = False  # TWHIN uses local TWHIN-BERT, not OpenAI
    show_score: bool = False  # keep likes/dislikes separate (Twitter style)
    semaphore: int = 4  # cap concurrent LLM requests (compute control)


def build_platform(db_path: str, config: PlatformConfig | None = None) -> Platform:
    """Construct the OASIS :class:`Platform` for one simulation run."""
    config = config or PlatformConfig()
    return Platform(
        db_path=str(db_path),
        recsys_type=config.recsys_type,
        allow_self_rating=config.allow_self_rating,
        refresh_rec_post_count=config.refresh_rec_post_count,
        max_rec_post_len=config.max_rec_post_len,
        following_post_count=config.following_post_count,
        use_openai_embedding=config.use_openai_embedding,
        show_score=config.show_score,
    )


def make_env(agent_graph, db_path: str, config: PlatformConfig | None = None):
    """Build the OASIS environment for ``agent_graph`` on a fresh platform.

    ``env.reset()`` (called by the caller) wires the platform's channel to every
    agent in the graph and signs them up, so agents are built without a channel.
    """
    config = config or PlatformConfig()
    platform = build_platform(db_path, config)
    return oasis.make(
        agent_graph=agent_graph,
        platform=platform,
        database_path=str(db_path),
        semaphore=config.semaphore,
    )
