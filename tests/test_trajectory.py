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

"""Tests for assembling a founder build's trajectory into a portable bundle."""

from __future__ import annotations

import json
import zipfile

import pytest

from viral_bench.founder.trajectory import (
    SCHEMA_VERSION,
    TrajectoryError,
    build_trajectory,
    export_trajectory,
    reasoning_from_dumps,
)


def _write(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _jsonl(path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _build(root, build_id="b1", *, sessions=True, phases=None):
    """Lay out one build workspace the way the harness leaves it."""
    work = root / build_id
    _write(
        work / "build.json",
        {
            "build_id": build_id,
            "idea_id": "tile_game",
            "model": "google-vertex/gemini-test",
            "created_at": "2026-08-22T00:00:00+00:00",
            "status": "ok",
            "structure": "dynamic",
            "subagents_spawned": 1,
            "trajectory": {"reasoning_parts_root": 1, "reasoning_chars_root": 3},
            "phases": phases
            if phases is not None
            else [
                {
                    "phase": "t1_founder",
                    "role": "founder",
                    "turn": "dynamic",
                    "session_id": "root",
                    "ok": True,
                    "reasoning_parts_root": 1,
                    "reasoning_chars_root": 3,
                    "transcript": str(work / "transcript" / "t1_founder.json"),
                }
            ],
        },
    )
    if sessions:
        _jsonl(
            work / "transcript" / "sessions" / "root.jsonl",
            [
                {
                    "record": "session",
                    "session_id": "root",
                    "parent_session_id": "",
                    "title": "founder",
                    "agent": "build",
                    "model": "google-vertex/gemini-test",
                    "cost": 0.5,
                    "tokens": {"input": 10, "output": 20, "reasoning": 30},
                },
                {
                    "record": "session",
                    "session_id": "kid",
                    "parent_session_id": "root",
                    "title": "the subagent",
                    "agent": "general",
                    "model": "google-vertex/gemini-test",
                    "cost": 0.25,
                    "tokens": {"input": 1, "output": 2, "reasoning": 3},
                },
                {
                    "record": "part",
                    "part_id": "p1",
                    "message_id": "m1",
                    "session_id": "root",
                    "role": "user",
                    "time_created": 1,
                    "part": {"type": "text", "text": "THE PROMPT"},
                },
                {
                    "record": "part",
                    "part_id": "p2",
                    "message_id": "m2",
                    "session_id": "root",
                    "role": "assistant",
                    "time_created": 2,
                    "part": {"type": "reasoning", "text": "hmm"},
                },
                {
                    "record": "part",
                    "part_id": "p3",
                    "message_id": "m3",
                    "session_id": "kid",
                    "role": "assistant",
                    "time_created": 3,
                    "part": {"type": "text", "text": "SUBAGENT OUTPUT"},
                },
                {
                    "record": "part",
                    "part_id": "p4",
                    "message_id": "m4",
                    "session_id": "root",
                    "role": "assistant",
                    "time_created": 4,
                    "part": {
                        "type": "tool",
                        "tool": "bash",
                        "state": {
                            "status": "completed",
                            "input": {"command": "ls"},
                            "output": "app/",
                            "time": {"start": 1, "end": 2},
                        },
                    },
                },
            ],
        )
    return work


def test_bundle_carries_prompts_reasoning_and_subagents(tmp_path) -> None:
    """The point of the whole exercise: a bundle has what stdout never did."""
    _build(tmp_path)
    trajectory = build_trajectory("b1", root=tmp_path)

    assert trajectory.manifest["schema_version"] == SCHEMA_VERSION
    assert trajectory.manifest["source"] == "session_store"

    by_type = {event["type"]: event for event in trajectory.events}
    assert by_type["reasoning"]["text"] == "hmm"
    texts = [e.get("text") for e in trajectory.events if e["type"] == "text"]
    assert "THE PROMPT" in texts
    assert "SUBAGENT OUTPUT" in texts

    # A user prompt stays attributable, so an SFT consumer can pair input to output.
    prompt = next(e for e in trajectory.events if e.get("text") == "THE PROMPT")
    assert prompt["role"] == "user"

    # The subagent is one level down, and the founder is the root.
    subagent = next(e for e in trajectory.events if e.get("text") == "SUBAGENT OUTPUT")
    assert subagent["depth"] == 1
    assert trajectory.manifest["totals"]["max_depth"] == 1


def test_events_are_ordered_and_sequenced(tmp_path) -> None:
    _build(tmp_path)
    events = build_trajectory("b1", root=tmp_path).events
    assert [e["seq"] for e in events] == list(range(len(events)))
    timestamps = [e["ts"] for e in events if e["ts"] is not None]
    assert timestamps == sorted(timestamps)


def test_tool_calls_are_flattened(tmp_path) -> None:
    _build(tmp_path)
    tool = next(
        e for e in build_trajectory("b1", root=tmp_path).events if e["type"] == "tool"
    )
    assert tool["tool"]["name"] == "bash"
    assert tool["tool"]["input"] == {"command": "ls"}
    assert tool["tool"]["output"] == "app/"
    assert tool["tool"]["status"] == "completed"


def test_totals_roll_up_tokens_and_cost(tmp_path) -> None:
    _build(tmp_path)
    totals = build_trajectory("b1", root=tmp_path).manifest["totals"]
    assert totals["sessions"] == 2
    assert totals["tokens"]["reasoning"] == 33  # root 30 + subagent 3
    assert totals["cost"] == pytest.approx(0.75)
    assert totals["reasoning_chars"] == 3


def test_redacted_thinking_is_flagged_not_silently_blank(tmp_path) -> None:
    """Claude returns an encrypted thinking block when the request did not ask
    for a summary. A reader must be able to tell that apart from a model that
    never thought at all."""
    work = _build(tmp_path)
    _jsonl(
        work / "transcript" / "sessions" / "root.jsonl",
        [
            {
                "record": "part",
                "part_id": "p9",
                "session_id": "root",
                "message_id": "m9",
                "role": "assistant",
                "time_created": 1,
                "part": {
                    "type": "reasoning",
                    "text": "",
                    "metadata": {"anthropic": {"signature": "Eo8CCn..."}},
                },
            }
        ],
    )
    trajectory = build_trajectory("b1", root=tmp_path)
    reasoning = next(e for e in trajectory.events if e["type"] == "reasoning")
    assert reasoning["text"] == ""
    assert reasoning["redacted"] is True
    assert trajectory.manifest["totals"]["reasoning_redacted"] == 1


def test_falls_back_to_the_stdout_transcript(tmp_path) -> None:
    """A build made before the session dump existed still exports -- and says so,
    so a partial trace is never mistaken for a full one."""
    work = _build(tmp_path, sessions=False)
    (work / "transcript").mkdir(parents=True, exist_ok=True)
    (work / "transcript" / "t1_founder.json").write_text(
        "\n".join(
            json.dumps(e)
            for e in (
                {
                    "type": "text",
                    "timestamp": 1,
                    "sessionID": "root",
                    "part": {"type": "text", "text": "hello"},
                },
                {
                    "type": "reasoning",
                    "timestamp": 2,
                    "sessionID": "root",
                    "part": {"type": "reasoning", "text": "thinking"},
                },
            )
        ),
        encoding="utf-8",
    )
    trajectory = build_trajectory("b1", root=tmp_path)
    assert trajectory.manifest["source"] == "transcript"
    assert [e["type"] for e in trajectory.events] == ["text", "reasoning"]
    assert trajectory.manifest["totals"]["sessions"] == 0


def test_duplicate_parts_across_per_turn_dumps_are_collapsed(tmp_path) -> None:
    """Every turn re-dumps its whole subtree, so the same part legitimately
    appears in several files."""
    work = _build(tmp_path)
    original = (work / "transcript" / "sessions" / "root.jsonl").read_text("utf-8")
    (work / "transcript" / "sessions" / "root2.jsonl").write_text(
        original, encoding="utf-8"
    )
    events = build_trajectory("b1", root=tmp_path).events
    assert sum(1 for e in events if e.get("text") == "THE PROMPT") == 1


def test_export_writes_a_directory_and_a_zip(tmp_path) -> None:
    _build(tmp_path)

    out = export_trajectory("b1", tmp_path / "out", root=tmp_path)
    manifest = json.loads((out / "trajectory.json").read_text("utf-8"))
    lines = (out / "events.jsonl").read_text("utf-8").splitlines()
    assert manifest["build_id"] == "b1"
    assert len(lines) == manifest["totals"]["events"]
    assert all(json.loads(line)["seq"] == i for i, line in enumerate(lines))

    archive_path = export_trajectory(
        "b1", tmp_path / "bundle", root=tmp_path, as_zip=True
    )
    assert archive_path.suffix == ".zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert sorted(archive.namelist()) == [
            "b1/events.jsonl",
            "b1/trajectory.json",
        ]


def test_missing_build_raises(tmp_path) -> None:
    with pytest.raises(TrajectoryError, match="no build record"):
        build_trajectory("nope", root=tmp_path)


def test_build_with_nothing_captured_yields_an_empty_stream(tmp_path) -> None:
    """More useful than an exception: the manifest says what is missing."""
    _build(tmp_path, sessions=False, phases=[])
    trajectory = build_trajectory("b1", root=tmp_path)
    assert trajectory.events == []
    assert trajectory.manifest["totals"]["events"] == 0
    assert trajectory.manifest["source"] == "transcript"


# -- the subagent blind spot in the transcript-derived count ----------------- #


def test_manifest_turns_are_labelled_root_only(tmp_path) -> None:
    """Per-turn numbers come from stdout and cannot see a subagent. Summing them
    to less than totals.reasoning_chars is correct -- the suffix is what stops
    that reading as a bug in the bundle."""
    _build(tmp_path)
    turns = build_trajectory("b1", root=tmp_path).manifest["turns"]
    assert turns[0]["reasoning_chars_root"] == 3
    assert "reasoning_chars" not in turns[0]


def _dump(path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _part(part_id, session, text, *, kind="reasoning", ts=1, meta=None):
    payload = {"type": kind, "text": text}
    if meta is not None:
        payload["metadata"] = meta
    return {
        "record": "part",
        "part_id": part_id,
        "message_id": "m" + part_id,
        "session_id": session,
        "role": "assistant",
        "time_created": ts,
        "part": payload,
    }


def test_reasoning_from_dumps_counts_subagent_thinking(tmp_path) -> None:
    """The bug in one assertion: the root session holds a fraction of the
    thinking on a delegating build, and the transcript can only see that."""
    _dump(
        tmp_path / "sessions" / "root.jsonl",
        [
            {"record": "session", "session_id": "root", "parent_session_id": ""},
            {"record": "session", "session_id": "kid", "parent_session_id": "root"},
            _part("p1", "root", "orchestrator thought", ts=1),
            _part("p2", "kid", "subagent thought, much longer than the parent", ts=2),
        ],
    )
    totals = reasoning_from_dumps(tmp_path / "sessions")
    assert totals["parts"] == 2
    assert totals["chars"] == len("orchestrator thought") + len(
        "subagent thought, much longer than the parent"
    )


def test_reasoning_from_dumps_dedupes_across_per_turn_dumps(tmp_path) -> None:
    """Every turn re-dumps its whole subtree, so a naive sum double-counts --
    the solo build turn's dump already contains the design turn's reasoning."""
    early = [
        {"record": "session", "session_id": "root", "parent_session_id": ""},
        _part("p1", "root", "design thinking", ts=1),
    ]
    late = early + [_part("p2", "root", "build thinking", ts=2)]
    _dump(tmp_path / "sessions" / "turn1.jsonl", early)
    _dump(tmp_path / "sessions" / "turn2.jsonl", late)

    totals = reasoning_from_dumps(tmp_path / "sessions")
    assert totals["parts"] == 2  # not 3
    assert totals["chars"] == len("design thinking") + len("build thinking")
    assert totals["events"] == 2  # parts only, deduped across both dumps
    assert totals["records"] == 3  # + the one distinct session line


def test_reasoning_from_dumps_separates_redacted_from_absent(tmp_path) -> None:
    """An encrypted Claude thinking block is not the same as no thinking."""
    _dump(
        tmp_path / "sessions" / "root.jsonl",
        [
            {"record": "session", "session_id": "root", "parent_session_id": ""},
            _part("p1", "root", "", meta={"anthropic": {"signature": "Eo8C"}}),
        ],
    )
    totals = reasoning_from_dumps(tmp_path / "sessions")
    assert totals["parts"] == 0
    assert totals["chars"] == 0
    assert totals["redacted"] == 1


def test_reasoning_from_dumps_is_zero_without_a_dump(tmp_path) -> None:
    """Zero here must never be read as 'the model did not think' -- the caller
    keeps the transcript-derived count precisely to tell those apart."""
    assert reasoning_from_dumps(tmp_path / "nope") == {
        "parts": 0,
        "chars": 0,
        "redacted": 0,
        "records": 0,
        "events": 0,
    }
    (tmp_path / "empty").mkdir()
    assert reasoning_from_dumps(tmp_path / "empty")["records"] == 0
