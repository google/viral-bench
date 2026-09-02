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

"""Stage 1 — Idea Bench: load and validate vibe-coding idea specs.

An *idea spec* is a single YAML file under the top-level ``ideas/`` directory
describing one vibe-coding project. Every model under test receives the same
spec as its brief, so the schema must be consistent and validated.

Typical usage::

    from viral_bench.ideas import load_ideas
    ideas = load_ideas()           # loads every ideas/*.yaml
    for idea in ideas:
        print(idea.idea_id, idea.title)

The :func:`load_ideas` / :func:`load_idea_file` functions raise
:class:`IdeaValidationError` with a clear message if a spec is malformed, which
is what the test suite and CI rely on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Allowed enum values, mirroring ideas/README.md and the design doc.
#
# ViralBench is a WEB benchmark: every idea is a web app, and the scope says how
# much of it lives on the server.
#
# * ``client-app``     -- runs in the browser. State is local (localStorage,
#                         IndexedDB); any server is just a static file server or
#                         a thin same-origin proxy hiding an API key.
# * ``full-stack-app`` -- has a real backend and a database. Accounts, persisted
#                         records, and MULTI-USER behaviour: one visitor's write
#                         is visible to another. That last property is why this
#                         scope exists -- the crowd shares one running instance
#                         per build, so it is the only scope that exercises it.
#
# The earlier ``single-page-app | cli | bot`` triple is gone: a benchmark whose
# corpus was one third terminal programs could not be described as measuring web
# development, and its craft rubric only ever worked for the web third.
ALLOWED_SCOPES = frozenset({"client-app", "full-stack-app"})
ALLOWED_DIFFICULTIES = frozenset({"easy", "medium", "hard"})

REQUIRED_FIELDS = (
    "idea_id",
    "title",
    "pitch",
    "problem",
    "target_user",
    "core_features",
    "success_criteria",
    "allowed_scope",
    "difficulty",
)

GROUND_TRUTH_REQUIRED_FIELDS = ("source", "metric", "value")


class IdeaValidationError(ValueError):
    """Raised when an idea spec file does not match the required schema."""


@dataclass(frozen=True)
class GroundTruth:
    """Real-world outcome for an idea drawn from an app that went viral."""

    source: str
    metric: str
    value: float


@dataclass(frozen=True)
class Idea:
    """A single, validated vibe-coding idea spec (Stage 1 — Idea Bench)."""

    idea_id: str
    title: str
    pitch: str
    problem: str
    target_user: str
    core_features: tuple[str, ...]
    success_criteria: str
    allowed_scope: str
    difficulty: str
    ground_truth: GroundTruth | None = None


def ideas_dir() -> Path:
    """Return the absolute path to the top-level ``ideas/`` directory."""
    # ideas.py lives at src/viral_bench/ideas.py; ideas/ is at the repo root.
    return Path(__file__).resolve().parents[2] / "ideas"


def _validate_ground_truth(raw: object, *, source: str) -> GroundTruth:
    if not isinstance(raw, dict):
        raise IdeaValidationError(f"{source}: 'ground_truth' must be a mapping")
    missing = [f for f in GROUND_TRUTH_REQUIRED_FIELDS if f not in raw]
    if missing:
        raise IdeaValidationError(
            f"{source}: 'ground_truth' missing field(s): {', '.join(missing)}"
        )
    if not isinstance(raw["value"], (int, float)) or isinstance(raw["value"], bool):
        raise IdeaValidationError(f"{source}: 'ground_truth.value' must be a number")
    return GroundTruth(
        source=str(raw["source"]),
        metric=str(raw["metric"]),
        value=float(raw["value"]),
    )


def parse_idea(raw: object, *, source: str) -> Idea:
    """Validate a raw mapping and return an :class:`Idea`.

    Args:
        raw: The object parsed from a YAML file (expected to be a mapping).
        source: A label (usually the filename) used in error messages.

    Raises:
        IdeaValidationError: If any field is missing or has the wrong type/value.
    """
    if not isinstance(raw, dict):
        raise IdeaValidationError(f"{source}: top-level YAML must be a mapping")

    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise IdeaValidationError(
            f"{source}: missing required field(s): {', '.join(missing)}"
        )

    # String fields must be non-empty strings.
    for field in (
        "idea_id",
        "title",
        "pitch",
        "problem",
        "target_user",
        "success_criteria",
    ):
        if not isinstance(raw[field], str) or not raw[field].strip():
            raise IdeaValidationError(f"{source}: '{field}' must be a non-empty string")

    # core_features must be a non-empty list of non-empty strings.
    features = raw["core_features"]
    if not isinstance(features, list) or not features:
        raise IdeaValidationError(f"{source}: 'core_features' must be a non-empty list")
    if not all(isinstance(f, str) and f.strip() for f in features):
        raise IdeaValidationError(
            f"{source}: every entry in 'core_features' must be a non-empty string"
        )

    # Enum fields.
    if raw["allowed_scope"] not in ALLOWED_SCOPES:
        raise IdeaValidationError(
            f"{source}: 'allowed_scope' must be one of "
            f"{sorted(ALLOWED_SCOPES)}, got {raw['allowed_scope']!r}"
        )
    if raw["difficulty"] not in ALLOWED_DIFFICULTIES:
        raise IdeaValidationError(
            f"{source}: 'difficulty' must be one of "
            f"{sorted(ALLOWED_DIFFICULTIES)}, got {raw['difficulty']!r}"
        )

    ground_truth = None
    if raw.get("ground_truth") is not None:
        ground_truth = _validate_ground_truth(raw["ground_truth"], source=source)

    return Idea(
        idea_id=raw["idea_id"],
        title=raw["title"],
        pitch=raw["pitch"],
        problem=raw["problem"],
        target_user=raw["target_user"],
        core_features=tuple(features),
        success_criteria=raw["success_criteria"],
        allowed_scope=raw["allowed_scope"],
        difficulty=raw["difficulty"],
        ground_truth=ground_truth,
    )


def load_idea_file(path: Path) -> Idea:
    """Load and validate a single idea spec file."""
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return parse_idea(raw, source=path.name)


def load_ideas(directory: Path | None = None) -> list[Idea]:
    """Load and validate every ``*.yaml`` file in the ideas directory.

    Args:
        directory: Directory to scan. Defaults to the repo's ``ideas/``.

    Returns:
        Ideas sorted by ``idea_id``.

    Raises:
        IdeaValidationError: If any file is malformed or if two ideas share an
            ``idea_id``.
    """
    directory = directory or ideas_dir()
    ideas: list[Idea] = []
    seen: dict[str, str] = {}

    for path in sorted(directory.glob("*.yaml")):
        idea = load_idea_file(path)
        if idea.idea_id in seen:
            raise IdeaValidationError(
                f"Duplicate idea_id {idea.idea_id!r} in {path.name} "
                f"(also in {seen[idea.idea_id]})"
            )
        seen[idea.idea_id] = path.name
        ideas.append(idea)

    return sorted(ideas, key=lambda i: i.idea_id)
