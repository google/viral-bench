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

"""Load and validate the rubric files in ``ideas/rubrics/``.

Two kinds of file, both YAML:

* ``_universal.yaml`` -- the Tier 0 gate, the Tier 3 robustness items and the
  universal penalties, applied to every idea.
* ``<idea_id>.yaml`` -- Tier 1 (the brief's ``success_criteria``), Tier 2 (its
  ``core_features``) and app-specific penalties, plus any
  ``universal_overrides`` marking a universal item not-applicable.

Validation is strict and runs in the test suite over all 25 files, because a
rubric that silently sums to 39 points would make one idea quietly incomparable
with the other 24 and nothing downstream would notice.

**These files are never shown to the founder model.** The idea loader globs
``ideas/*.yaml`` non-recursively, so this subdirectory is excluded by the same
mechanism that excludes ``templates/`` -- see ``ideas/rubrics/README.md``. The
guarantee is tested in ``tests/rubric/test_schema.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

#: Matches the convention in :mod:`viral_bench.ideas` -- the package lives two
#: levels below the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Point totals every idea rubric must hit exactly. Tier 3 lives in the
#: universal file and is shared, so it is checked once rather than per idea.
TIER1_POINTS = 40
TIER2_POINTS = 25
TIER3_POINTS = 35

#: How much a build can lose to anti-patterns in total. A build should be able
#: to lose a quarter of its score to spec-gaming without a single penalty stack
#: driving an otherwise-working app to zero.
PENALTY_CAP = -25

#: How an item is decided, ordered most-deterministic first. ``checks.py``
#: implements the first two, the grader model resolves ``agent``, and ``source``
#: is a read over the built source tree.
METHODS = ("probe", "assert", "agent", "source")

#: Tier labels, used by the viewer and the report so the wording is defined once.
TIER_LABELS = {
    0: "Deliverability gate",
    1: "Success criteria",
    2: "Core features",
    3: "Robustness and craft",
}


class RubricError(ValueError):
    """A rubric file is malformed. Always fatal -- never graded around."""


@dataclass(frozen=True)
class Check:
    """A deterministic check: which primitive decides this item, and with what.

    ``name`` selects a function in :mod:`viral_bench.rubric.checks`, and
    ``params`` is passed to it as keyword arguments. Keeping the params as data
    rather than as code is what lets the rubric stay declarative and auditable
    -- a reader can see exactly what was asserted without reading Python.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any, *, where: str) -> Check | None:
        if raw is None:
            return None
        if isinstance(raw, str):
            return cls(name=raw, params={})
        if not isinstance(raw, dict) or "name" not in raw:
            raise RubricError(
                f"{where}: 'check' must be a string or a mapping with 'name'"
            )
        params = {k: v for k, v in raw.items() if k != "name"}
        return cls(name=str(raw["name"]), params=params)


@dataclass(frozen=True)
class RubricItem:
    """One scored line of a rubric.

    ``points`` is positive for tier items and negative for penalties. ``expect``
    is human-readable prose describing the expected value, while the
    machine-readable form lives in ``check.params``. Both are kept: the prose is
    what the grader model reads and what the viewer renders, the params are what
    the code runs.
    """

    id: str
    text: str
    points: int
    method: str
    tier: int
    expect: str = ""
    note: str = ""
    #: Navigation instructions for the grader model. Where an item carries a
    #: ``check``, this is the model's *entire* job -- put the app into the state
    #: the primitive expects, then stop. Keeping it separate from ``text`` is
    #: what lets the same item read as a claim to a human and as a script to the
    #: model.
    setup: str = ""
    check: Check | None = None
    #: Penalties only: a floor on the total this id may contribute when it can
    #: fire more than once (e.g. "-5 each, max -10").
    max_total: int | None = None

    @property
    def is_penalty(self) -> bool:
        return self.points < 0

    @property
    def code_judged(self) -> bool:
        """Is pass/fail decided by the harness rather than by the model?"""
        return self.check is not None or self.method == "probe"


@dataclass(frozen=True)
class Universal:
    """The sections shared by every idea."""

    gate: tuple[RubricItem, ...]
    tier3: tuple[RubricItem, ...]
    penalties: tuple[RubricItem, ...]

    def tier3_by_id(self) -> dict[str, RubricItem]:
        return {item.id: item for item in self.tier3}

    def overridable(self) -> dict[str, RubricItem]:
        """Universal items an idea may mark not-applicable.

        Penalties are included, not only Tier 3. P5 penalises depending on a
        model API the brief never asked for -- which is exactly right for a
        typing test that bolted on a chatbot, and exactly wrong for the ideas
        whose brief *does* ask for a model feature. Without a way to say so,
        that penalty would fire on every AI idea by design.
        """
        return {item.id: item for item in self.tier3 + self.penalties}


@dataclass(frozen=True)
class Rubric:
    """One idea's complete rubric, universal sections already merged in."""

    idea_id: str
    rubric_version: str
    tier1: tuple[RubricItem, ...]
    tier2: tuple[RubricItem, ...]
    tier3: tuple[RubricItem, ...]
    gate: tuple[RubricItem, ...]
    penalties: tuple[RubricItem, ...]
    #: universal item id -> why it does not apply to this idea. Rendered in the
    #: report so nobody wonders why a tier does not sum to its nominal total.
    not_applicable: dict[str, str] = field(default_factory=dict)

    @property
    def scored_items(self) -> tuple[RubricItem, ...]:
        return self.tier1 + self.tier2 + self.tier3

    @property
    def points_applicable(self) -> int:
        return sum(item.points for item in self.scored_items)

    def items_by_id(self) -> dict[str, RubricItem]:
        every = self.gate + self.scored_items + self.penalties
        return {item.id: item for item in every}


def rubrics_dir(root: Path | None = None) -> Path:
    return (root or _REPO_ROOT) / "ideas" / "rubrics"


def _require_list(raw: dict, key: str, *, where: str) -> list:
    value = raw.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise RubricError(f"{where}: '{key}' must be a list")
    return value


def _parse_item(
    raw: Any, *, tier: int, where: str, needs_points: bool = True
) -> RubricItem:
    if not isinstance(raw, dict):
        raise RubricError(f"{where}: each item must be a mapping")
    item_id = str(raw.get("id") or "").strip()
    if not item_id:
        raise RubricError(f"{where}: every item needs a non-empty 'id'")
    text = str(raw.get("text") or "").strip()
    if not text:
        raise RubricError(f"{where}:{item_id}: every item needs non-empty 'text'")
    method = str(raw.get("method") or "").strip()
    if method not in METHODS:
        raise RubricError(
            f"{where}:{item_id}: 'method' must be one of "
            f"{', '.join(METHODS)}, got {method!r}"
        )
    points = raw.get("points")
    if points is None:
        if needs_points:
            raise RubricError(f"{where}:{item_id}: missing 'points'")
        points = 0
    if not isinstance(points, int) or isinstance(points, bool):
        raise RubricError(f"{where}:{item_id}: 'points' must be an integer")
    max_total = raw.get("max_total")
    if max_total is not None and (not isinstance(max_total, int) or max_total >= 0):
        raise RubricError(f"{where}:{item_id}: 'max_total' must be a negative integer")
    return RubricItem(
        id=item_id,
        text=text,
        points=points,
        method=method,
        tier=tier,
        expect=str(raw.get("expect") or ""),
        note=str(raw.get("note") or ""),
        setup=str(raw.get("setup") or ""),
        check=Check.parse(raw.get("check"), where=f"{where}:{item_id}"),
        max_total=max_total,
    )


@lru_cache(maxsize=1)
def load_universal(root: str | None = None) -> Universal:
    path = rubrics_dir(Path(root) if root else None) / "_universal.yaml"
    if not path.is_file():
        raise RubricError(f"no universal rubric at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    where = path.name

    gate = tuple(
        _parse_item(item, tier=0, where=where, needs_points=False)
        for item in _require_list(raw, "gate", where=where)
    )
    tier3 = tuple(
        _parse_item(item, tier=3, where=where)
        for item in _require_list(raw, "tier3", where=where)
    )
    penalties = tuple(
        _parse_item(item, tier=-1, where=where)
        for item in _require_list(raw, "penalties", where=where)
    )

    if not gate:
        raise RubricError(f"{where}: the gate cannot be empty")
    total3 = sum(item.points for item in tier3)
    if total3 != TIER3_POINTS:
        raise RubricError(f"{where}: tier3 must sum to {TIER3_POINTS}, got {total3}")
    for item in penalties:
        if item.points >= 0:
            raise RubricError(
                f"{where}:{item.id}: a penalty must carry negative points"
            )
    _reject_duplicates(gate + tier3 + penalties, where=where)
    return Universal(gate=gate, tier3=tier3, penalties=penalties)


def _reject_duplicates(items: tuple[RubricItem, ...], *, where: str) -> None:
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise RubricError(f"{where}: duplicate item id {item.id!r}")
        seen.add(item.id)


def load_rubric(idea_id: str, root: Path | None = None) -> Rubric:
    """Load one idea's rubric with the universal sections merged in.

    Raises :class:`RubricError` on anything malformed. There is deliberately no
    lenient mode: a rubric is the definition of the measurement, and a
    half-applied definition is worse than a missing one.
    """
    path = rubrics_dir(root) / f"{idea_id}.yaml"
    if not path.is_file():
        raise RubricError(f"no rubric for idea {idea_id!r} at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    where = path.name

    declared = str(raw.get("idea_id") or "")
    if declared != idea_id:
        raise RubricError(
            f"{where}: declares idea_id {declared!r} but is named {idea_id!r}"
        )

    tier1 = tuple(
        _parse_item(item, tier=1, where=where)
        for item in _require_list(raw, "tier1", where=where)
    )
    tier2 = tuple(
        _parse_item(item, tier=2, where=where)
        for item in _require_list(raw, "tier2", where=where)
    )
    penalties = tuple(
        _parse_item(item, tier=-1, where=where)
        for item in _require_list(raw, "penalties", where=where)
    )

    total1 = sum(item.points for item in tier1)
    total2 = sum(item.points for item in tier2)
    if total1 != TIER1_POINTS:
        raise RubricError(f"{where}: tier1 must sum to {TIER1_POINTS}, got {total1}")
    if total2 != TIER2_POINTS:
        raise RubricError(f"{where}: tier2 must sum to {TIER2_POINTS}, got {total2}")
    for item in penalties:
        if item.points >= 0:
            raise RubricError(
                f"{where}:{item.id}: a penalty must carry negative points"
            )

    universal = load_universal(str(root) if root else None)
    overrides = raw.get("universal_overrides") or {}
    if not isinstance(overrides, dict):
        raise RubricError(f"{where}: 'universal_overrides' must be a mapping")

    known = universal.overridable()
    not_applicable: dict[str, str] = {}
    for item_id, spec in overrides.items():
        if item_id not in known:
            raise RubricError(
                f"{where}: universal_overrides names unknown item {item_id!r}"
            )
        if not isinstance(spec, dict):
            raise RubricError(f"{where}:{item_id}: an override must be a mapping")
        if spec.get("applicable", True) is False:
            reason = str(spec.get("reason") or "").strip()
            if not reason:
                raise RubricError(
                    f"{where}:{item_id}: marking an item not-applicable "
                    "requires a 'reason'"
                )
            not_applicable[item_id] = reason

    tier3 = tuple(item for item in universal.tier3 if item.id not in not_applicable)
    # A not-applicable penalty is dropped rather than merely never fired, so it
    # cannot show up in the report as a penalty that happened not to trigger.
    universal_penalties = tuple(
        item for item in universal.penalties if item.id not in not_applicable
    )
    merged = tier1 + tier2 + tier3 + universal.gate + penalties + universal_penalties
    _reject_duplicates(merged, where=where)

    return Rubric(
        idea_id=idea_id,
        rubric_version=str(raw.get("rubric_version") or "1"),
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
        gate=universal.gate,
        penalties=penalties + universal_penalties,
        not_applicable=not_applicable,
    )


def available_rubrics(root: Path | None = None) -> list[str]:
    """Every idea id that has a rubric file, sorted. Excludes ``_universal``."""
    directory = rubrics_dir(root)
    if not directory.is_dir():
        return []
    return sorted(
        path.stem for path in directory.glob("*.yaml") if not path.name.startswith("_")
    )
