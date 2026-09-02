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

"""Persist and render a RubricScore.

Writes ``builds/rubric/<build_id>__rubric-<ts>/grade.json``, following the
``<build_id>__<kind>-<timestamp>`` convention every other derived artifact under
``builds/`` uses. A directory rather than a bare file because the grade travels
with its evidence: the grader transcript and any screenshots it captured.

The document is deliberately **self-describing**. The viewer's readers never
import ``viral_bench`` (a stated design property, ``viz/README.md:452-456``), so
item text, tier labels, point values and every number the header shows are all
written into the file rather than reconstructed at render time. That also makes a
stored grade reproducible after the rubric is edited: the file records the rubric
it was graded against, not the rubric as it stands today.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from viral_bench.founder.workspace import builds_root
from viral_bench.rubric import RUBRIC_SCORE_VERSION
from viral_bench.rubric.schema import TIER_LABELS, Rubric
from viral_bench.rubric.score import RubricResult, tier_breakdown

#: The document kind, so a reader can tell a grade from a crowd summary without
#: inspecting its shape.
GRADE_KIND = "viral_bench.rubric_grade"
GRADE_DOC_VERSION = 1

GRADE_FILENAME = "grade.json"
TRANSCRIPT_FILENAME = "transcript.jsonl"
SHOTS_DIRNAME = "shots"

#: Crowd-shaped sets that may hold a ``score.json`` for the same build. Mirrors
#: ``viz.core.paths.CROWD_SETS``; duplicated rather than imported because
#: ``viral_bench`` must not depend on the viewer.
CROWD_SETS = ("crowd", "ablation", "calibration", "smoke")

#: Files whose bytes define "the app source" for invalidation purposes. Anything
#: generated at run time is excluded, or a grade would invalidate itself.
_SOURCE_SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".playwright-mcp",
    "dist",
    "build",
    ".next",
    ".cache",
}


def rubric_root(root: Path | None = None) -> Path:
    return (root or builds_root()) / "rubric"


def new_run_id(build_id: str, when: datetime | None = None) -> str:
    """``<build_id>__rubric-<YYYYmmdd-HHMMSS>`` -- discovered by prefix glob."""
    stamp = (when or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    return f"{build_id}__rubric-{stamp}"


def rubric_run_dir(run_id: str, root: Path | None = None) -> Path:
    return rubric_root(root) / run_id


def rubric_runs_for_build(build_id: str, root: Path | None = None) -> list[Path]:
    """Every grade recorded against *build_id*, newest first."""
    base = rubric_root(root)
    if not base.is_dir():
        return []
    found = [
        entry
        for entry in base.glob(f"{build_id}__rubric-*")
        if (entry / GRADE_FILENAME).is_file()
    ]
    return sorted(found, key=lambda path: path.name, reverse=True)


def _read_json(path: Path, default=None):
    """Read JSON, returning *default* on any failure.

    Grades are written by long jobs that get killed; a truncated or absent file
    is normal and must never take a sweep down with it.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def source_hash(app_dir: Path) -> str:
    """A content hash over the built app's source tree.

    Nothing in the repo has one today, so nothing can currently invalidate a
    stored score when the source underneath it changes. Paths are included
    alongside bytes so a rename is a different hash, and the walk is sorted so
    the result does not depend on directory order.
    """
    if not app_dir.is_dir():
        return ""
    digest = hashlib.sha256()
    for path in sorted(app_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _SOURCE_SKIP_DIRS for part in path.relative_to(app_dir).parts):
            continue
        digest.update(str(path.relative_to(app_dir)).encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()[:16]


def founder_block(build_id: str, root: Path | None = None) -> dict:
    """What was graded: which pipeline built the app, with which model.

    The same shape ``viz.core.crowd._founder_build`` produces, written at grade
    time rather than resolved at render time -- a grade outlives its build, and a
    score with no subject is not interpretable. Returns ``on_disk: False`` rather
    than raising when the build has been pruned.
    """
    empty = {"on_disk": False, "build_id": build_id}
    if not build_id:
        return empty
    record = _read_json((root or builds_root()) / "work" / build_id / "build.json")
    if not isinstance(record, dict):
        return empty

    structure = record.get("structure") or ""
    collab = record.get("collab") or ""
    arm = {
        ("solo", "local"): "solo",
        ("team", "local"): "team",
        ("dynamic", "local"): "dynamic",
    }.get((structure, collab), structure or "unknown")
    label = {
        "solo": "single agent",
        "team": "4-agent team",
        "dynamic": "dynamic orchestrator",
    }.get(arm, arm)

    model = str(record.get("model") or "")
    return {
        "on_disk": True,
        "build_id": build_id,
        "idea_id": record.get("idea_id") or "",
        "arm": arm,
        "arm_label": label,
        "structure": structure,
        "collab": collab,
        "n_agents": record.get("n_agents"),
        "model": model,
        "model_short": model.rsplit("/", 1)[-1].split("@", 1)[0],
        "status": record.get("status") or "",
        "turns_spent": record.get("turns_spent"),
        "shipped_early": record.get("shipped_early"),
        "qa_verified": record.get("qa_verified"),
        "app_title": (record.get("manifest") or {}).get("title") or "",
        "created_at": record.get("created_at") or "",
        "brief_fingerprint": record.get("brief_fingerprint") or "",
    }


def comparison_block(build_id: str, root: Path | None = None) -> dict | None:
    """The ViralScore(s) already recorded for this build, or ``None``.

    Filled at grade time from stored ``score.json`` files so the two tracks sit
    side by side in the viewer without the viewer importing ``score/``. The whole
    point of the rubric track is this comparison, so it is captured with the
    grade rather than joined later.
    """
    base = root or builds_root()
    scores: list[float] = []
    for group in CROWD_SETS:
        directory = base / group
        if not directory.is_dir():
            continue
        for entry in directory.glob(f"{build_id}__*"):
            record = _read_json(entry / "score.json")
            if not isinstance(record, dict):
                continue
            value = record.get("score")
            if isinstance(value, int | float) and record.get("scorable", True):
                scores.append(float(value))
    if not scores:
        return None
    return {
        "viral_score_mean": round(sum(scores) / len(scores), 1),
        "viral_score_min": round(min(scores), 1),
        "viral_score_max": round(max(scores), 1),
        "crowd_runs": len(scores),
    }


def _gate_rows(rubric: Rubric, result: RubricResult) -> dict:
    rows = []
    for item in rubric.gate:
        verdict = result.verdicts.get(item.id)
        rows.append(
            {
                "id": item.id,
                "text": item.text,
                "method": item.method,
                "passed": bool(verdict and verdict.passed),
                "detail": verdict.reason if verdict else "not evaluated",
                "observed": verdict.observed if verdict else "",
                "evidence": list(verdict.evidence) if verdict else [],
                "passes": list(verdict.passes) if verdict else [],
            }
        )
    return {
        "passed": not result.gate_failures,
        "label": TIER_LABELS[0],
        "failures": list(result.gate_failures),
        "items": rows,
    }


def _penalty_rows(rubric: Rubric, result: RubricResult) -> list[dict]:
    rows = []
    for item in rubric.penalties:
        verdict = result.verdicts.get(item.id)
        fired = bool(verdict and verdict.passed)
        rows.append(
            {
                "id": item.id,
                "text": item.text,
                "points": item.points,
                "max_total": item.max_total,
                "method": item.method,
                "fired": fired,
                "reason": verdict.reason if verdict else "",
                "observed": verdict.observed if verdict else "",
                "evidence": list(verdict.evidence) if verdict else [],
                "passes": list(verdict.passes) if verdict else [],
                "harness_override": bool(verdict and verdict.harness_override),
            }
        )
    return rows


def build_grade_document(
    rubric: Rubric,
    result: RubricResult,
    *,
    run_id: str,
    grader_model: str,
    graded_at: str = "",
    source_digest: str = "",
    brief_fingerprint: str = "",
    founder: dict | None = None,
    comparison: dict | None = None,
    paths: dict | None = None,
) -> dict:
    """Assemble the whole verdict document. Pure -- writes nothing."""
    return {
        "kind": GRADE_KIND,
        "version": GRADE_DOC_VERSION,
        "rubric_version": rubric.rubric_version,
        "score_version": RUBRIC_SCORE_VERSION,
        "run_id": run_id,
        "build_id": result.build_id,
        "idea_id": result.idea_id,
        "graded_at": graded_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "grader_model": grader_model,
        "passes": result.passes,
        "brief_fingerprint": brief_fingerprint,
        "source_hash": source_digest,
        "ok": result.ok,
        "error": result.error or None,
        "founder": founder if founder is not None else founder_block(result.build_id),
        "score": result.score,
        # Pre-computed so the header never recomputes and never disagrees with
        # the scorer. Reconstruction in the viewer would be a second
        # implementation of the arithmetic, and two are one too many.
        "math": {
            "points_earned": result.points_earned,
            "points_applicable": result.points_applicable,
            "base": result.base,
            "penalty_total": result.penalty_total,
            "penalty_capped": result.penalty_capped,
            "floor_applied": result.floor_applied,
            "gate_zeroed": result.gate_zeroed,
        },
        "gate": _gate_rows(rubric, result),
        "tiers": tier_breakdown(rubric, result),
        "penalties": _penalty_rows(rubric, result),
        "not_applicable": [
            {"id": item_id, "reason": reason}
            for item_id, reason in sorted(result.not_applicable.items())
        ],
        "unresolved": list(result.unresolved),
        "reliability": result.reliability(),
        "comparison": comparison,
        "paths": paths or {},
    }


def write_grade(document: dict, run_dir: Path) -> Path:
    """Persist ``grade.json`` into *run_dir*; return its path.

    Write-once by convention: the sweep skips a build that already has a grade
    for the current rubric and source hash, so re-running is cheap and safe.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / GRADE_FILENAME
    document.setdefault("paths", {})
    document["paths"] = {
        "run_dir": str(run_dir),
        "grade": str(target),
        "transcript": str(run_dir / TRANSCRIPT_FILENAME),
        "shots": str(run_dir / SHOTS_DIRNAME),
    } | {k: v for k, v in document["paths"].items() if k not in {"run_dir", "grade"}}
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return target


def read_grade(run_dir: Path) -> dict | None:
    """Read a stored grade, or ``None`` if it is missing or truncated."""
    document = _read_json(Path(run_dir) / GRADE_FILENAME)
    return document if isinstance(document, dict) else None


def _mark(passed: bool, *, unresolved: bool = False) -> str:
    if unresolved:
        return "?"
    return "PASS" if passed else "FAIL"


def render_grade(document: dict) -> str:
    """A readable breakdown of one graded build.

    Shows the arithmetic rather than just the number: which tier lost the points,
    which items disagreed across passes, and how often the harness had to overrule
    the model. A grade nobody can audit is a grade nobody should trust.
    """
    math = document.get("math") or {}
    lines: list[str] = []

    head = f"RubricScore: {document.get('score')} / 100"
    if math.get("gate_zeroed"):
        head += "   [GATE FAILED -- scored 0]"
    lines.append(head)
    founder = document.get("founder") or {}
    lines.append(
        f"build: {document.get('build_id')}"
        f"  idea: {document.get('idea_id')}"
        f"  arm: {founder.get('arm_label') or '?'}"
        f"  built by: {founder.get('model_short') or '?'}"
    )
    lines.append(
        f"grader: {document.get('grader_model')}"
        f"  passes: {document.get('passes')}"
        f"  rubric v{document.get('rubric_version')}"
        f"  score v{document.get('score_version')}"
    )
    if document.get("error"):
        lines.append(f"!! ERROR: {document['error']}")
    lines.append("")

    gate = document.get("gate") or {}
    gate_mark = "PASS" if gate.get("passed") else "FAIL"
    lines.append(f"{gate.get('label', 'gate')}: {gate_mark}")
    for row in gate.get("items") or []:
        if not row.get("passed"):
            lines.append(f"  FAIL {row['id']}  {row['text']}")
            if row.get("detail"):
                lines.append(f"       {row['detail']}")
    lines.append("")

    for tier in document.get("tiers") or []:
        lines.append(
            f"Tier {tier['tier']} -- {tier['label']}: {tier['earned']}/{tier['points']}"
        )
        for row in tier.get("items") or []:
            flags = ""
            if row.get("disagreement"):
                flags += " ~"
            if row.get("harness_override"):
                flags += " !override"
            mark = _mark(
                row.get("passed", False), unresolved=row.get("unresolved", False)
            )
            lines.append(
                f"  {mark:<4} {row['id']:<5} {row['points']:>3}p  "
                f"{row['text'][:72]}{flags}"
            )
            if not row.get("passed") and row.get("reason"):
                lines.append(f"       {row['reason'][:100]}")
        lines.append("")

    fired = [row for row in document.get("penalties") or [] if row.get("fired")]
    if fired:
        lines.append(f"Penalties: {math.get('penalty_total')}")
        for row in fired:
            lines.append(f"  {row['points']:>4}p  {row['id']}  {row['text'][:70]}")
        if math.get("penalty_capped"):
            lines.append("  (capped)")
        lines.append("")

    na = document.get("not_applicable") or []
    if na:
        lines.append("Not applicable:")
        lines += [f"  {row['id']}: {row['reason']}" for row in na]
        lines.append("")

    if math:
        penalty = math.get("penalty_total") or 0
        lines.append(
            f"arithmetic: {math.get('points_earned')}"
            f"/{math.get('points_applicable')} pts"
            f" = base {math.get('base')}"
            f"  {penalty:+d} penalties"
            f"  => {document.get('score')}"
        )

    rel = document.get("reliability") or {}
    if rel:
        lines.append(
            f"reliability: {rel.get('items_disagreeing')}/{rel.get('items_total')}"
            f" items disagreed across passes"
            f"  (code {rel.get('code_disagreement')},"
            f" agent {rel.get('agent_disagreement')})"
            f"  override rate {rel.get('override_rate')}"
            f"  unresolved {rel.get('unresolved')}"
        )

    comparison = document.get("comparison")
    if comparison:
        lines.append(
            f"ViralScore for the same build: {comparison.get('viral_score_mean')}"
            f" (mean of {comparison.get('crowd_runs')} crowd runs)"
        )
    return "\n".join(lines)
