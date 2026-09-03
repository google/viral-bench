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

"""Tests for the curated persona set + crowd selection (oasis-free)."""

from __future__ import annotations

from viral_bench.crowd.sim.personas import Persona, load_personas, select_crowd


def test_load_personas_populated() -> None:
    personas = load_personas()
    assert len(personas) >= 10
    p = personas[0]
    assert isinstance(p, Persona)
    assert p.name and p.username and p.persona
    assert 1 <= p.influence <= 10
    assert p.interest_list  # comma-split interests


def test_load_personas_has_diverse_tiers_and_influence() -> None:
    personas = load_personas()
    assert any(p.early_adopter for p in personas)  # some natural triers
    assert any(not p.early_adopter for p in personas)  # some reactors
    assert any(p.influence >= 8 for p in personas)  # at least one influencer


def test_select_crowd_is_deterministic() -> None:
    personas = load_personas()
    a = select_crowd(personas, n_agents=8, n_triers=5, seed=42)
    b = select_crowd(personas, n_agents=8, n_triers=5, seed=42)
    assert [p.username for p in a.all] == [p.username for p in b.all]


def test_select_crowd_respects_counts() -> None:
    personas = load_personas()
    sel = select_crowd(personas, n_agents=8, n_triers=5, seed=0)
    assert len(sel.all) == 8
    assert len(sel.triers) == 5
    assert len(sel.reactors) == 3
    # triers + reactors partition the crowd with no overlap
    assert not (set(sel.triers) & set(sel.reactors))


def test_select_crowd_prefers_early_adopters_as_triers() -> None:
    personas = load_personas()
    sel = select_crowd(personas, n_agents=10, n_triers=4, seed=0)
    # With enough early adopters available, triers should skew early-adopter.
    early_triers = sum(1 for p in sel.triers if p.early_adopter)
    assert early_triers >= 2


def test_select_crowd_clamps_to_available() -> None:
    personas = load_personas()
    sel = select_crowd(personas, n_agents=999, n_triers=999, seed=0)
    assert len(sel.all) == len(personas)
    assert len(sel.triers) == len(personas)  # n_triers clamped to n_agents
    assert len(sel.reactors) == 0


def test_tier_of() -> None:
    personas = load_personas()
    sel = select_crowd(personas, n_agents=6, n_triers=3, seed=1)
    for p in sel.triers:
        assert sel.tier_of(p) == "trier"
    for p in sel.reactors:
        assert sel.tier_of(p) == "reactor"


def test_a_negative_trier_count_makes_the_whole_crowd_hands_on():
    """ "Everyone can click a link" has to be expressible, and be exact.

    Measured over 44 stored runs, first-hand verdicts separated builds at
    between/within 7.17 against 3.66 for feed-only verdicts, so who is allowed
    to touch the app is the single biggest lever on the instrument. The knob
    must therefore mean exactly what it says: -1 is every agent, not "most".
    """
    personas = load_personas()
    sel = select_crowd(personas, n_agents=12, n_triers=-1, seed=3)
    assert len(sel.triers) == 12
    assert sel.reactors == []
    assert {p.username for p in sel.all} == {p.username for p in sel.triers}


def test_latecomers_are_a_representative_draw_not_the_least_interested():
    """Sampling the least early-adopter-ish produced a metric that was always 0.

    Those people are also the furthest from any given app's audience, so 8 of 8
    declined a well-liked app with variations on "nobody said how it helps
    someone like me". A conversion rate that is zero for every app measures the
    sampling rule, not the app.
    """
    personas = load_personas()
    crowd_rate = sum(1 for p in personas if p.early_adopter) / len(personas)
    shares = []
    for seed in range(8):
        sel = select_crowd(
            personas, n_agents=30, n_triers=-1, n_latecomers=10, seed=seed
        )
        assert len(sel.latecomers) == 10
        assert len(sel.triers) == 20
        assert not set(sel.latecomers) & set(sel.triers)
        shares.append(sum(1 for p in sel.latecomers if p.early_adopter) / 10)
    # Averaged over seeds the tier looks like the pool it is drawn from, rather
    # than being systematically composed of people who never try anything.
    assert abs(sum(shares) / len(shares) - crowd_rate) < 0.2
