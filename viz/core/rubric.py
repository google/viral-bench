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

"""Read a RubricScore grade off disk.

Thin by design, and that is the point. ``grade.json`` is written
*self-describing*: item text, tier labels, point values, the score arithmetic and
the founder block are all already in the file. So unlike :mod:`core.crowd`, which
fuses five sources and resolves the build it scored, this reader mostly hands the
document over and adds the parts that are the viewer's job -- paging
the transcript, indexing evidence, and pulling the per-item view together.

That self-description is what lets this module obey the rule the rest of
``viz/core`` obeys: **it does not import ``viral_bench``**. A grade written in
August opens with this code no matter what the grading harness looks like later,
and a refactor in the benchmark cannot break the viewer.
"""

from __future__ import annotations

import json
from pathlib import Path

from .paths import GRADE_FILE, read_json

#: Transcript rows returned in one page. The grader's log for a whole build runs
#: to a few hundred calls with large results, so the UI pages it the way the crowd
#: viewer pages feeds.
TRANSCRIPT_PAGE = 200

#: Longest tool result kept in a transcript row. Full bodies stay on disk.
RESULT_CHARS = 4000


def load_grade(run_dir: Path) -> dict | None:
    """The whole grade document, or ``None`` if missing or truncated.

    Returns the stored document near-verbatim -- adding a derived view
    here would be a second implementation of arithmetic the scorer already did,
    and two implementations of one number is one too many.
    """
    grade = read_json(Path(run_dir) / GRADE_FILE)
    if not isinstance(grade, dict):
        return None
    grade = dict(grade)
    grade["run_id"] = grade.get("run_id") or Path(run_dir).name
    grade["evidence_index"] = _evidence_index(Path(run_dir))
    grade["has_transcript"] = (Path(run_dir) / "transcript.jsonl").is_file()
    grade["shots"] = _shots(Path(run_dir))
    return grade


def _shots(run_dir: Path) -> list[str]:
    """Basenames of captured evidence screenshots, sorted."""
    directory = run_dir / "shots"
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_file())


def _evidence_index(run_dir: Path) -> dict:
    """``item_id -> [tool_call_id, ...]`` read from the transcript.

    Built here rather than trusted from the grade because the transcript is the
    harness's own record of what it ran, written before the model ever saw the
    result. That is the same reasoning the anti-fabrication rule rests on, and it
    means the viewer shows evidence that demonstrably exists rather than evidence
    a verdict merely claimed.
    """
    index: dict[str, list[str]] = {}
    for row in _iter_transcript(run_dir):
        item_id = str(row.get("item_id") or "")
        call_id = str(row.get("id") or "")
        if item_id and call_id:
            index.setdefault(item_id, []).append(call_id)
    return index


def _iter_transcript(run_dir: Path):
    """Yield parsed transcript rows, skipping anything unreadable.

    A killed sweep leaves a truncated final line. Per-line parsing means one bad
    row costs one row, not the whole grade -- the convention every reader in
    ``viz/core`` follows.
    """
    path = Path(run_dir) / "transcript.jsonl"
    if not path.is_file():
        return
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                yield row


def load_transcript(
    run_dir: Path,
    *,
    offset: int = 0,
    limit: int = TRANSCRIPT_PAGE,
    item_id: str = "",
) -> dict:
    """One page of the grader's tool log, optionally filtered to a single item."""
    rows = []
    for row in _iter_transcript(Path(run_dir)):
        if item_id and str(row.get("item_id") or "") != item_id:
            continue
        result = str(row.get("result") or "")
        rows.append(
            {
                "id": row.get("id") or "",
                "name": row.get("name") or "",
                "args": row.get("args") if isinstance(row.get("args"), dict) else {},
                "result": result[:RESULT_CHARS],
                "truncated": len(result) > RESULT_CHARS,
                "ok": bool(row.get("ok", True)),
                "item_id": row.get("item_id") or "",
            }
        )
    total = len(rows)
    offset = max(0, offset)
    window = rows[offset : offset + max(1, limit)]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "item_id": item_id,
        "calls": window,
    }


def item_rows(grade: dict) -> list[dict]:
    """Every scored item across every tier, flattened, in rubric order.

    The left pane renders one list, so keeping the flattening here means the page
    cannot disagree with the tier totals the scorer wrote.
    """
    rows: list[dict] = []
    for tier in grade.get("tiers") or []:
        for item in tier.get("items") or []:
            rows.append({**item, "tier": tier.get("tier"), "kind": "item"})
    for penalty in grade.get("penalties") or []:
        rows.append({**penalty, "tier": -1, "kind": "penalty"})
    return rows


def find_item(grade: dict, item_id: str) -> dict | None:
    """One item by id, searched across tiers, penalties and the gate."""
    for row in item_rows(grade):
        if row.get("id") == item_id:
            return row
    for row in (grade.get("gate") or {}).get("items") or []:
        if row.get("id") == item_id:
            return {**row, "tier": 0, "kind": "gate"}
    return None
