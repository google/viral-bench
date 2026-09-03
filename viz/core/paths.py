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

"""Where the artifacts are, and the rule that they are only ever read.

The viewer lives in the benchmark repo but reads its ``builds/`` tree as data,
never importing any of the benchmark's code. Two reasons that matters:

**Nothing may be written there.** The builds tree belongs to an active development
checkout, and a sweep may be writing into it right now. Everything this tool
produces -- parsed caches, rescued screenshots, throwaway app run dirs -- goes
under ``viz/cache/``, and :func:`assert_writable` is the one gate that enforces it.

**The formats outlive the code.** A build recorded in July parses with the same
reader as one recorded today, because the reader is driven by what is on disk and
not by whichever branch happens to be checked out. Legacy records (``structure:
"specialist"``, or no ``structure`` at all) are read, not rejected.

A build id maps straight to a directory: ``builds/work/<build_id>/``. Every derived
artifact prefixes that id and appends a suffix, so a build's crowd runs are
``builds/crowd/<build_id>__crowd-*`` and its data dir is ``builds/data/<build_id>``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

VIZ_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = VIZ_ROOT.parent
CACHE_ROOT = VIZ_ROOT / "cache"

#: Directories under ``builds/`` that hold crowd simulation runs. ``ablation``
#: uses an identical file layout but tags runs with ``<version>+<variant>`` so they
#: never pool with the main corpus -- worth viewing, worth keeping distinguishable.
CROWD_SETS = ("crowd", "ablation", "calibration", "smoke")

#: Directories under ``builds/`` that hold RubricScore grades. A grade is a
#: property of the *build*, not of a crowd run, but it follows the same
#: ``<build_id>__<kind>-<timestamp>`` convention so one build can be graded more
#: than once -- new rubric version, different grader, a re-run -- without
#: clobbering, and so discovery is the same cheap prefix glob.
RUBRIC_SETS = ("rubric",)

#: What makes a directory a grade, the way ``run_summary.json`` does for a run.
GRADE_FILE = "grade.json"


class OutsideCache(Exception):
    """Raised when something tries to write outside ``viz/cache/``."""


def assert_writable(path: Path) -> Path:
    """Return *path*, or refuse if it is not inside ``viz/cache/``.

    Every write in this package goes through here. The viewer reads a live
    development checkout, and a stray write into it -- a cache file, a rescued
    screenshot -- would show up as an unexplained dirty file in someone's ``git
    status`` at best, and clobber a real artifact at worst.
    """
    resolved = Path(path).resolve()
    root = CACHE_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise OutsideCache(f"refusing to write outside {root}: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def cache_dir(*parts: str) -> Path:
    """Return (and create) a directory under ``viz/cache/``."""
    path = CACHE_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_builds_root() -> Path:
    """The repo's own ``builds/``, unless ``VIRAL_BENCH_BUILDS_DIR`` overrides it.

    The override exists so the viewer can be pointed at a builds tree copied off
    another machine, or at an archived one, without moving anything.
    """
    env = os.environ.get("VIRAL_BENCH_BUILDS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "builds").resolve()


@dataclass(frozen=True)
class FounderPaths:
    """The on-disk locations that make up one founder build."""

    build_id: str
    root: Path
    build_json: Path
    transcript_dir: Path
    app_dir: Path
    agents_dir: Path
    screenshots_dir: Path

    def exists(self) -> bool:
        return self.build_json.is_file()


def founder_paths(builds_root: Path, build_id: str) -> FounderPaths:
    root = builds_root / "work" / build_id
    return FounderPaths(
        build_id=build_id,
        root=root,
        build_json=root / "build.json",
        transcript_dir=root / "transcript",
        app_dir=root / "app",
        # Dynamic mode writes the agents the founder invented for itself here,
        # deliberately outside app/ so they do not ship with the product.
        agents_dir=root / ".opencode" / "agents",
        screenshots_dir=root / "app" / ".playwright-mcp",
    )


def crowd_run_dir(builds_root: Path, run_id: str) -> Path | None:
    """Resolve a crowd run id to its directory, searching every crowd-shaped set."""
    for group in CROWD_SETS:
        candidate = builds_root / group / run_id
        if (candidate / "run_summary.json").is_file():
            return candidate
    return None


def crowd_runs_for_build(builds_root: Path, build_id: str) -> list[Path]:
    """Every crowd run recorded against *build_id*, newest first.

    Run dirs are named ``<build_id>__crowd-<ts>[-s<seed>]`` (or ``__<arm>-<ts>-sN``
    for ablations), so the prefix match is exact and cheap -- no need to open 1,976
    summaries to find the handful that belong to one build.
    """
    found: list[Path] = []
    for group in CROWD_SETS:
        base = builds_root / group
        if not base.is_dir():
            continue
        for entry in base.glob(f"{build_id}__*"):
            if (entry / "run_summary.json").is_file():
                found.append(entry)
    return sorted(found, key=lambda p: p.name, reverse=True)


def rubric_run_dir(builds_root: Path, run_id: str) -> Path | None:
    """Resolve a rubric grade id to its directory."""
    for group in RUBRIC_SETS:
        candidate = builds_root / group / run_id
        if (candidate / GRADE_FILE).is_file():
            return candidate
    return None


def rubric_grades_for_build(builds_root: Path, build_id: str) -> list[Path]:
    """Every grade recorded against *build_id*, newest first.

    Same prefix-glob reasoning as :func:`crowd_runs_for_build`: the directory is
    named ``<build_id>__rubric-<ts>``, so finding a build's grades never means
    opening every grade on disk.
    """
    found: list[Path] = []
    for group in RUBRIC_SETS:
        base = builds_root / group
        if not base.is_dir():
            continue
        for entry in base.glob(f"{build_id}__*"):
            if (entry / GRADE_FILE).is_file():
                found.append(entry)
    return sorted(found, key=lambda p: p.name, reverse=True)


def read_json(path: Path, default=None):
    """Read a JSON file, returning *default* on any failure.

    Artifacts are written by long-running jobs that get killed, so a truncated or
    absent file is normal and must never take the viewer down with it.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


# --------------------------------------------------------------------------
# Build index
# --------------------------------------------------------------------------


#: How a build.json's raw fields map to the arm names people say out
#: loud. The tuple is ``(structure, collab)``, and ``n_agents`` is not part of the key
#: because dynamic records ``n_agents: 1`` even though the whole point is that the
#: agent count is an outcome rather than a setting.
_MODE_LABELS = {
    ("solo", "local"): ("solo", "Single agent"),
    ("team", "local"): ("team", "4-agent team (local)"),
    ("dynamic", "local"): ("dynamic", "Dynamic orchestrator"),
}


def classify_mode(record: dict) -> tuple[str, str]:
    """Return ``(mode_key, human_label)`` for a build record.

    Falls back to a descriptive label for the retired ``specialist`` relay mode and
    for the single pre-structure record that predates the field entirely, so the
    picker can show them instead of hiding them behind a crash.
    """
    structure = record.get("structure")
    collab = record.get("collab")
    known = _MODE_LABELS.get((structure, collab))
    if known:
        return known
    if structure == "specialist":
        n = record.get("n_agents") or "?"
        return "legacy", f"Legacy relay ({n}-agent)"
    if structure:
        return "other", str(structure)
    return "legacy", "Legacy (pre-structure)"


@dataclass
class BuildSummary:
    """One row in the build picker."""

    build_id: str
    idea_id: str
    model: str
    mode: str
    mode_label: str
    structure: str | None
    n_agents: int | None
    collab: str | None
    status: str
    created_at: str
    rounds_run: int | None
    turns_spent: int | None
    shipped_early: bool | None
    qa_verified: bool | None
    duration_s: float
    phase_count: int
    transcript_bytes: int
    has_app: bool
    app_type: str | None
    app_title: str | None
    subagents_spawned: int
    crowd_runs: int = 0
    tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "build_id": self.build_id,
            "idea_id": self.idea_id,
            "model": self.model,
            "mode": self.mode,
            "mode_label": self.mode_label,
            "structure": self.structure,
            "n_agents": self.n_agents,
            "collab": self.collab,
            "status": self.status,
            "created_at": self.created_at,
            "rounds_run": self.rounds_run,
            "turns_spent": self.turns_spent,
            "shipped_early": self.shipped_early,
            "qa_verified": self.qa_verified,
            "duration_s": round(self.duration_s, 1),
            "phase_count": self.phase_count,
            "transcript_bytes": self.transcript_bytes,
            "has_app": self.has_app,
            "app_type": self.app_type,
            "app_title": self.app_title,
            "subagents_spawned": self.subagents_spawned,
            "crowd_runs": self.crowd_runs,
            "tags": self.tags,
        }


def _short_model(model: str) -> str:
    """``google-vertex-anthropic/gemini-2.0-flash`` -> ``gemini-2.0-flash``."""
    if not model:
        return ""
    tail = model.rsplit("/", 1)[-1]
    return tail.split("@", 1)[0]


def summarize_build(
    paths: FounderPaths, record: dict, crowd_runs: int = 0
) -> BuildSummary:
    phases = record.get("phases") or []
    manifest = record.get("manifest") or {}
    transcript_bytes = 0
    if paths.transcript_dir.is_dir():
        transcript_bytes = sum(
            f.stat().st_size for f in paths.transcript_dir.glob("*.json") if f.is_file()
        )
    mode, label = classify_mode(record)

    tags: list[str] = []
    if record.get("shipped_early"):
        tags.append("shipped")
    if record.get("qa_verified"):
        tags.append("qa-verified")
    if record.get("orchestration"):
        tags.append("orchestrated")
    if (record.get("model") or "").lower().startswith("control"):
        tags.append("control")

    return BuildSummary(
        build_id=record.get("build_id") or paths.build_id,
        idea_id=record.get("idea_id") or "",
        model=_short_model(record.get("model") or ""),
        mode=mode,
        mode_label=label,
        structure=record.get("structure"),
        n_agents=record.get("n_agents"),
        collab=record.get("collab"),
        status=record.get("status") or "unknown",
        created_at=record.get("created_at") or "",
        rounds_run=record.get("rounds_run"),
        turns_spent=record.get("turns_spent"),
        shipped_early=record.get("shipped_early"),
        qa_verified=record.get("qa_verified"),
        duration_s=sum(float(p.get("duration_s") or 0) for p in phases),
        phase_count=len(phases),
        transcript_bytes=transcript_bytes,
        has_app=paths.app_dir.is_dir(),
        app_type=manifest.get("app_type"),
        app_title=manifest.get("title"),
        subagents_spawned=int(record.get("subagents_spawned") or 0),
        crowd_runs=crowd_runs,
        tags=tags,
    )


def _crowd_counts(builds_root: Path) -> dict[str, int]:
    """Count crowd runs per build id in one pass over the directory names.

    ``builds/crowd`` alone holds ~2,000 entries and ``builds/ablation`` another
    ~1,200. Globbing per build would be 500 x 3,000 stats, whereas splitting the
    names once
    is a single listdir per set.
    """
    counts: dict[str, int] = {}
    for group in CROWD_SETS:
        base = builds_root / group
        if not base.is_dir():
            continue
        try:
            names = os.listdir(base)
        except OSError:
            continue
        for name in names:
            build_id, sep, _ = name.partition("__crowd-")
            if not sep:
                # Ablation arms use ``<build_id>__<arm>-<ts>-sN``. The build id
                # itself contains "__", so split on the last one and keep the head.
                head, sep2, _tail = name.rpartition("__")
                if not sep2:
                    continue
                build_id = head
            counts[build_id] = counts.get(build_id, 0) + 1
    return counts


def index_builds(builds_root: Path) -> list[BuildSummary]:
    """Scan ``builds/work`` and return every build, newest first.

    Only ``build.json`` is opened -- roughly 550 small files, a few hundred
    milliseconds cold and cached by the OS thereafter. Transcripts are parsed
    lazily, when a build is opened.
    """
    work = builds_root / "work"
    if not work.is_dir():
        return []
    counts = _crowd_counts(builds_root)
    out: list[BuildSummary] = []
    for entry in sorted(work.iterdir()):
        if not entry.is_dir():
            continue
        record = read_json(entry / "build.json")
        if not isinstance(record, dict):
            continue
        paths = founder_paths(builds_root, entry.name)
        out.append(summarize_build(paths, record, counts.get(entry.name, 0)))
    out.sort(key=lambda b: (b.created_at or "", b.build_id), reverse=True)
    return out


def index_crowd_runs(builds_root: Path, limit: int | None = None) -> list[dict]:
    """Scan every crowd-shaped set and return one summary row per run.

    ~3,200 ``run_summary.json`` files, each a few KB. Slower than the founder
    index, so the server caches the result and refreshes it on demand.
    """
    rows: list[dict] = []
    for group in CROWD_SETS:
        base = builds_root / group
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            summary = read_json(entry / "run_summary.json")
            if not isinstance(summary, dict):
                continue
            rows.append(_crowd_row(entry, group, summary))
            if limit and len(rows) >= limit:
                break
    rows.sort(key=lambda r: r["run_id"], reverse=True)
    return rows


def _crowd_row(entry: Path, group: str, summary: dict) -> dict:
    config = summary.get("config") or {}
    engagement = summary.get("engagement") or {}
    triers = (summary.get("verdicts") or {}).get("triers") or {}
    trace_dir = entry / "traces"
    return {
        "run_id": entry.name,
        "group": group,
        "build_id": summary.get("build_id") or "",
        "ok": bool(summary.get("ok")),
        "app_type": summary.get("app_type") or "",
        "arch_version": str(summary.get("crowd_arch_version") or ""),
        "n_agents": config.get("n_agents"),
        "rounds": summary.get("rounds_run"),
        "model": config.get("model_id") or "",
        "recsys": config.get("recsys_type") or "",
        "seed": config.get("seed"),
        "duration_s": round(float(summary.get("duration_s") or 0), 1),
        "posts": engagement.get("posts") or 0,
        "likes": engagement.get("likes") or 0,
        "comments": engagement.get("comments") or 0,
        "reposts": engagement.get("reposts") or 0,
        "follows": engagement.get("follows") or 0,
        "would_use_rate": triers.get("would_use_rate"),
        "delight_mean": triers.get("delight_mean"),
        "n_traces": len(list(trace_dir.glob("agent_*.json")))
        if trace_dir.is_dir()
        else 0,
        "undeliverable": bool(summary.get("undeliverable")),
    }


def index_rubric_grades(builds_root: Path, limit: int | None = None) -> list[dict]:
    """Scan every rubric set and return one summary row per grade.

    Same shape and caching contract as :func:`index_crowd_runs`. A ``grade.json``
    is larger than a run summary but there are far fewer of them, so one pass is
    affordable and the server caches it anyway.
    """
    rows: list[dict] = []
    for group in RUBRIC_SETS:
        base = builds_root / group
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            grade = read_json(entry / GRADE_FILE)
            if not isinstance(grade, dict):
                continue
            rows.append(_rubric_row(entry, group, grade))
            if limit and len(rows) >= limit:
                break
    rows.sort(key=lambda r: r["run_id"], reverse=True)
    return rows


def _rubric_row(entry: Path, group: str, grade: dict) -> dict:
    """Flatten one grade into a picker row.

    Carries the ViralScore alongside the RubricScore where the grade recorded
    one, so the picker can be sorted by the disagreement between the two tracks
    -- which is the entire reason the rubric track exists, and a listing that
    forced you to open each grade to find the interesting ones would bury it.
    """
    math = grade.get("math") or {}
    founder = grade.get("founder") or {}
    comparison = grade.get("comparison") or {}
    score = grade.get("score")
    viral = comparison.get("viral_score_mean")
    delta = None
    if isinstance(score, int | float) and isinstance(viral, int | float):
        delta = round(float(score) - float(viral), 1)
    return {
        "run_id": entry.name,
        "group": group,
        "build_id": grade.get("build_id") or "",
        "idea_id": grade.get("idea_id") or "",
        "ok": bool(grade.get("ok", True)),
        "score": score,
        "viral_score": viral,
        "delta": delta,
        "crowd_runs": comparison.get("crowd_runs") or 0,
        "gate_passed": bool((grade.get("gate") or {}).get("passed")),
        "gate_zeroed": bool(math.get("gate_zeroed")),
        "points_earned": math.get("points_earned"),
        "points_applicable": math.get("points_applicable"),
        "penalty_total": math.get("penalty_total"),
        "grader_model": grade.get("grader_model") or "",
        "passes": grade.get("passes"),
        "rubric_version": str(grade.get("rubric_version") or ""),
        "graded_at": grade.get("graded_at") or "",
        "arm": founder.get("arm") or "",
        "arm_label": founder.get("arm_label") or "",
        "model": founder.get("model") or "",
        "model_short": founder.get("model_short") or "",
        "app_title": founder.get("app_title") or "",
        "override_rate": (grade.get("reliability") or {}).get("override_rate"),
    }
