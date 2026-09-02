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

"""Load the curated crowd personas and split them into triers and reactors.

The crowd is the benchmark's *measuring instrument*, so it must be fixed and
reproducible across every app and model it judges. It is therefore a curated,
version-controlled persona set (``data/crowd/personas.csv``) rather than
generating personas per run. The schema is deliberately rich (interests,
influence weight, skepticism, early-adopter flag) so the same file can later be
grown -- or swapped for a frozen generated population -- without changing the
harness.

:func:`select_crowd` deterministically picks ``n_agents`` personas and labels a
small number of them **triers** (they will first-hand run the app) and the rest
**reactors** (they react to the launch post, the discussion, and the code). This
keeps the expensive first-hand trial cost roughly constant as the crowd scales.
"""

from __future__ import annotations

import csv
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LOG = logging.getLogger("viral_bench.crowd.personas")

__all__ = [
    "Persona",
    "CrowdSelection",
    "personas_path",
    "load_personas",
    "select_crowd",
]


@dataclass(frozen=True)
class Persona:
    """One curated crowd member."""

    name: str
    username: str
    archetype: str
    interests: str
    influence: int  # 1..10, higher = more followers / amplification weight
    skepticism: str  # low | medium | high
    early_adopter: bool  # a natural first-hand "trier"
    persona: str  # rich free-text description for the system prompt

    @property
    def interest_list(self) -> list[str]:
        return [i.strip() for i in self.interests.split(",") if i.strip()]


@dataclass(frozen=True)
class CrowdSelection:
    """A chosen crowd, partitioned by tier (stable order).

    ``requested_n_agents`` / ``pool_size`` are carried so a scored run can record
    (and refuse) a crowd that was silently smaller than asked for -- see
    :attr:`clamped`.

    Three tiers, because they measure three different things:

    * **trier** -- uses the app immediately and forms its own opinion. The
      independent measurement, and the bulk of the crowd.
    * **latecomer** -- has the app within reach but has NOT tried it, and will
      only do so if the feed talks it into it. This is the one tier that can
      measure virality as something *earned*: the fraction of them who go and
      try it is a conversion rate the simulation produces rather than a formula
      assigns. It exists because measuring opinion change across 44 runs found
      ``to_yes`` was **0.00 in every single one** -- once everybody has tried the
      app, discussion can sink it but can never lift it, so the entire upside
      half of word-of-mouth was unobservable.
    * **reactor** -- never touches the app and judges from the feed alone. Kept
      because it is the honest control for the trier tier, not because it is
      good: split by tier over 44 stored runs it separates builds at roughly
      half the trier tier's rate.
    """

    triers: list[Persona]
    reactors: list[Persona]
    latecomers: list[Persona] = field(default_factory=list)
    requested_n_agents: int = 0
    pool_size: int = 0

    @property
    def all(self) -> list[Persona]:
        """Every selected persona, triers first (order = crowd agent order)."""
        return [*self.triers, *self.latecomers, *self.reactors]

    @property
    def clamped(self) -> bool:
        """True if the persona pool was too small to honour ``n_agents``.

        A clamped crowd is smaller than the run asked for, which changes both the
        score's noise floor and its comparability -- the ViralScore stage treats
        it as a low-confidence run rather than trusting the number.
        """
        return bool(self.requested_n_agents) and len(self.all) < self.requested_n_agents

    def tier_of(self, persona: Persona) -> str:
        if persona in self.triers:
            return "trier"
        if persona in self.latecomers:
            return "latecomer"
        return "reactor"


def personas_path(path: str | Path | None = None) -> Path:
    """Resolve the persona CSV path (override with ``VIRAL_BENCH_PERSONAS_FILE``)."""
    if path is not None:
        return Path(path)
    override = os.environ.get("VIRAL_BENCH_PERSONAS_FILE")
    if override:
        return Path(override)
    return _REPO_ROOT / "data" / "crowd" / "personas.csv"


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ("yes", "true", "1", "y")


def load_personas(path: str | Path | None = None) -> list[Persona]:
    """Load all personas from the curated CSV, in file order."""
    csv_path = personas_path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"persona file not found: {csv_path}")
    personas: list[Persona] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            personas.append(
                Persona(
                    name=row["name"].strip(),
                    username=row["username"].strip(),
                    archetype=row["archetype"].strip(),
                    interests=row["interests"].strip(),
                    influence=int(row.get("influence", "5") or 5),
                    skepticism=row.get("skepticism", "medium").strip() or "medium",
                    early_adopter=_as_bool(row.get("early_adopter", "no")),
                    persona=row["persona"].strip(),
                )
            )
    if not personas:
        raise ValueError(f"no personas found in {csv_path}")
    return personas


def select_crowd(
    personas: list[Persona],
    *,
    n_agents: int,
    n_triers: int,
    n_latecomers: int = 0,
    seed: int = 0,
) -> CrowdSelection:
    """Pick ``n_agents`` personas and label ``n_triers`` of them as triers.

    Selection and tiering are deterministic given ``seed`` (reproducibility).
    Triers are preferentially the early-adopters, then the most influential, so
    the first-hand testers are the people who realistically try new things first.

    Args:
        personas: The full curated set.
        n_agents: How many to include in the crowd (clamped to what's available).
        n_triers: How many of the crowd should first-hand run the app (clamped
            to ``n_agents``). **Negative means everyone.** In the real world a
            link is something you can click, so the default is that the whole
            crowd can form its own opinion. Measured over 44 stored runs the
            hands-on tier separated builds nearly twice as well as the
            feed-only tier (between/within 7.17 vs 3.66 on adoption, 7.86 vs
            4.74 on delight) *and* did it with less run-to-run noise despite
            being a quarter of the sample. Twenty-two agents reading one feed
            are not twenty-two measurements, but one measurement echoed.
        seed: RNG seed for reproducible selection.

    Returns:
        A :class:`CrowdSelection` with triers first.
    """
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    pool = list(personas)
    rng = random.Random(seed)
    n = min(n_agents, len(pool))
    chosen = rng.sample(pool, n) if n < len(pool) else list(pool)

    if n < n_agents:
        _LOG.warning(
            "crowd clamped to %d agents: the persona pool (%s) only has %d "
            "personas but %d were requested. A scored run MUST NOT be silently "
            "under-sized -- grow the persona file or lower --agents.",
            n,
            personas_path(),
            len(pool),
            n_agents,
        )

    # Latecomers come out of the trier budget, not out of thin air: the crowd
    # size is the crowd size, and a variant that quietly adds eight agents is
    # not comparable with the arm it is being compared against.
    n_latecomers = max(0, min(n_latecomers, n))
    n_triers = n if n_triers < 0 else max(0, min(n_triers, n))
    n_triers = min(n_triers, n - n_latecomers)

    # Latecomers are drawn at RANDOM from the crowd, deterministically by seed,
    # and drawn FIRST.
    #
    # The obvious alternative -- take the least early-adopter-ish people, since
    # they are the ones who would plausibly wait to be convinced -- is a trap,
    # and it produced a dead metric on the first attempt: those are also the
    # people furthest from any given app's audience, so 8 of 8 declined a
    # well-liked app with variations on "nobody said how it helps someone like
    # me". A conversion rate that is 0 for every app measures the sampling rule,
    # not the app. A random draw makes the tier a representative sample of the
    # same crowd, so its conversion rate is an unbiased estimate of the fraction
    # of ordinary onlookers this launch would convert.
    late_set = set(rng.sample(chosen, n_latecomers)) if n_latecomers else set()
    eligible = [p for p in chosen if p not in late_set]
    trier_set = set(_rank_triers(eligible, n_triers))
    return CrowdSelection(
        triers=[p for p in chosen if p in trier_set],
        latecomers=[p for p in chosen if p in late_set],
        reactors=[p for p in chosen if p not in trier_set and p not in late_set],
        requested_n_agents=n_agents,
        pool_size=len(pool),
    )


def _rank_triers(chosen: list[Persona], n_triers: int) -> list[Persona]:
    """Pick the hands-on triers, stratified across skepticism.

    Triers are the only agents who *use* the app, so they produce the
    benchmark's richest evidence. Ranking purely by ``early_adopter`` (the old
    behaviour) handed that evidence exclusively to enthusiasts -- measured on
    three real runs, every trier scored 7-8 while the wider crowd spread 1-8, so
    the most evidence-grounded signal was also the least discriminating.

    Selection therefore round-robins across skepticism buckets (high first), and only
    prefer early adopters *within* a bucket. Tough critics still get to try the
    app first-hand, while early adopters remain the natural first movers.
    Deterministic: no RNG, stable tiebreak on username.
    """
    buckets: dict[str, list[Persona]] = {"high": [], "medium": [], "low": []}
    for p in chosen:
        buckets.setdefault(p.skepticism, buckets["medium"]).append(p)
    for bucket in buckets.values():
        bucket.sort(key=lambda p: (not p.early_adopter, -p.influence, p.username))

    picked: list[Persona] = []
    order = ("high", "medium", "low")
    while len(picked) < n_triers:
        progressed = False
        for name in order:
            if len(picked) >= n_triers:
                break
            if buckets[name]:
                picked.append(buckets[name].pop(0))
                progressed = True
        if not progressed:  # every bucket drained
            break
    return picked
