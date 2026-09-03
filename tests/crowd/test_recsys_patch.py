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

"""Regression tests for the patched OASIS twitter recsys.

These reproduce the exact conditions that silently destroyed the benchmark's
primary signal: a run with MORE posts than ``max_rec_post_len`` (which switches
OASIS from a trivial "show everything" branch to its personalised one) where the
agents have already liked things and no embedding model is loaded.

Measured across 41 stored runs before the fix: all 9 runs that crossed the
threshold lost 100% of their end-of-run interviews, and all 32 that stayed below
it kept theirs. The previous fix was never validated above the threshold, which
is precisely why this test pins the crossing case.

Skipped unless the isolated crowd env is importable (``oasis`` lives there).
"""

from __future__ import annotations

import pytest

oasis_recsys = pytest.importorskip(
    "oasis.social_platform.recsys", reason="needs the crowd env (.venv-crowd)"
)

from viral_bench.crowd.sim import platform as vb_platform  # noqa: E402,F401


def _tables(n_posts: int, *, likes: bool):
    """A crowd mid-run: ``n_posts`` posts, two users, optional like history."""
    users = [
        {"user_id": 1, "bio": "indie hacker who loves maker tools"},
        {"user_id": 2, "bio": "privacy engineer, self-hosting"},
    ]
    posts = [
        {"post_id": i, "user_id": (i % 3), "content": f"post number {i}"}
        for i in range(1, n_posts + 1)
    ]
    trace = [{"user_id": 1, "created_at": 0, "action": "refresh", "info": "{}"}]
    if likes:
        # post_id lives in the info JSON -- there is no post_id COLUMN.
        trace.append(
            {
                "user_id": 1,
                "created_at": 1,
                "action": "like_post",
                "info": '{"post_id": 2, "like_id": 1}',
            }
        )
    return users, posts, trace


def test_trace_post_ids_reads_the_info_json_not_a_column() -> None:
    _, _, trace = _tables(3, likes=True)
    assert vb_platform._trace_post_ids(trace) == [2]
    assert vb_platform._trace_post_ids(trace, user_id=1, action="like_post") == [2]
    assert vb_platform._trace_post_ids(trace, user_id=99) == []


def test_below_threshold_shows_every_post() -> None:
    # The cheap branch: this is the path small crowds always took, which is why
    # the defects below went unnoticed for so long.
    users, posts, trace = _tables(5, likes=True)
    rec = oasis_recsys.rec_sys_personalized_with_trace(
        users, posts, trace, [[], [], []], max_rec_post_len=10
    )
    assert rec == [[p["post_id"] for p in posts]] * 2


def test_above_threshold_with_likes_and_no_model_does_not_crash() -> None:
    # THE regression. Before the fix this raised AttributeError from an
    # unguarded model.encode(), or KeyError: 'post_id' from the swap block.
    assert oasis_recsys.model is None, "this test is about the no-model path"
    users, posts, trace = _tables(14, likes=True)
    rec = oasis_recsys.rec_sys_personalized_with_trace(
        users, posts, trace, [[], [], []], max_rec_post_len=10
    )
    assert len(rec) == 2  # one recommendation list per user
    for row in rec:
        assert 0 < len(row) <= 10
        assert all(isinstance(pid, int) for pid in row)


def test_above_threshold_without_likes_also_survives_the_swap_block() -> None:
    # Isolates defect 3: the swap block's own trace['post_id'] lookup.
    users, posts, trace = _tables(14, likes=False)
    rec = oasis_recsys.rec_sys_personalized_with_trace(
        users, posts, trace, [[], [], []], max_rec_post_len=10
    )
    assert len(rec) == 2
    assert all(row for row in rec)


def test_recommendations_exclude_your_own_posts() -> None:
    # Behaviour preserved from upstream: you are not recommended your own posts.
    users, posts, trace = _tables(14, likes=True)
    rec = oasis_recsys.rec_sys_personalized_with_trace(
        users, posts, trace, [[], [], []], max_rec_post_len=10
    )
    own = {p["post_id"] for p in posts if p["user_id"] == 1}
    assert not (set(rec[0]) & own)


def test_patch_is_idempotent() -> None:
    vb_platform._patch_oasis_recsys()
    vb_platform._patch_oasis_recsys()
    assert getattr(
        oasis_recsys.rec_sys_personalized_with_trace, "_viral_bench_patched", False
    )
