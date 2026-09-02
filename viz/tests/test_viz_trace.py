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

"""Tests for reading what a build thought.

These pin the semantics this reader shares with
``viral_bench.founder.trajectory``: the reasoning tri-state, prompts identified by
role, subagent depth, cross-file dedup, global ordering, and the transcript
fallback. Getting any of them subtly wrong produces a plausible-looking trace,
which is the failure mode worth testing against.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import crowd as crowd_mod  # noqa: E402
from core import founder as founder_mod  # noqa: E402
from core import trace as trace_mod  # noqa: E402
from core.paths import founder_paths, read_json  # noqa: E402
from vizfixtures import part_row, session_row, write_dump  # noqa: E402


def _trace(root: Path, build_id: str = "idea__traced") -> dict:
    record = read_json(founder_paths(root, build_id).build_json)
    return trace_mod.load_trace(root, build_id, record)


def test_session_store_is_preferred_over_stdout(traced):
    assert _trace(traced)["manifest"]["source"] == "session_store"


def test_prompt_is_a_user_role_text_part(traced):
    """There is no prompt event type; role is the only discriminator."""
    events = _trace(traced)["events"]
    prompts = [e for e in events if e["type"] == "text" and e["role"] == "user"]
    assert [p["text"] for p in prompts] == ["THE PROMPT"]


def test_reasoning_is_tri_state(traced):
    """Readable, redacted and genuinely-absent are three different findings."""
    events = [e for e in _trace(traced)["events"] if e["type"] == "reasoning"]
    readable = [e for e in events if e.get("text")]
    redacted = [e for e in events if not e.get("text") and e.get("redacted")]
    empty = [e for e in events if not e.get("text") and not e.get("redacted")]
    assert [e["text"] for e in readable] == ["thinking", "sub"]
    assert len(redacted) == 1, "encrypted thinking must not read as absent"
    assert len(empty) == 1, "a turn with no thought must not read as redacted"


def test_subagent_work_is_captured_at_depth_one(traced):
    """The delegated work opencode's stdout printer drops entirely."""
    bundle = _trace(traced)
    deep = [e for e in bundle["events"] if e["depth"] == 1]
    assert {e["session_id"] for e in deep} == {"ses_kid"}
    assert "SUBAGENT OUTPUT" in [e.get("text") for e in deep]
    assert bundle["manifest"]["totals"]["max_depth"] == 1


def test_patches_survive(traced):
    patches = [e for e in _trace(traced)["events"] if e["type"] == "patch"]
    assert patches[0]["patch"] == {"hash": "abc123", "files": ["/x/app/index.html"]}


def test_tool_is_flattened_to_the_schema_shape(traced):
    tool = next(e for e in _trace(traced)["events"] if e["type"] == "tool")["tool"]
    assert set(tool) == {
        "name",
        "status",
        "input",
        "output",
        "error",
        "metadata",
        "start_ms",
        "end_ms",
    }
    assert tool["name"] == "bash"
    assert tool["input"] == {"command": "ls"}


def test_events_are_globally_ordered_across_sessions(traced):
    """One timeline, so a subagent's work sits where it actually happened."""
    events = _trace(traced)["events"]
    assert [e["seq"] for e in events] == list(range(len(events)))
    stamps = [e["ts"] for e in events if e["ts"] is not None]
    assert stamps == sorted(stamps)


def test_duplicate_parts_across_per_turn_dumps_collapse(traced):
    """Every turn re-dumps the whole subtree, so parts repeat across files."""
    build_dir = traced / "work" / "idea__traced"
    source = build_dir / "transcript" / "sessions" / "ses_root.jsonl"
    shutil.copy(source, source.with_name("ses_root_turn2.jsonl"))
    events = _trace(traced)["events"]
    assert sum(1 for e in events if e.get("text") == "THE PROMPT") == 1


def test_session_model_blob_is_normalised(traced):
    """The store writes a JSON blob in the session's model column, not a name."""
    root = next(
        s for s in _trace(traced)["manifest"]["sessions"] if not s["parent_session_id"]
    )
    assert root["model"] == "google-vertex/gemini-test"


def test_totals_sum_tokens_and_cost_across_sessions(traced):
    totals = _trace(traced)["manifest"]["totals"]
    assert totals["sessions"] == 2
    assert totals["tokens"]["reasoning"] == 60  # 30 per session
    assert totals["cost"] == 1.0
    assert totals["reasoning_chars"] == len("thinking") + len("sub")
    assert totals["reasoning_redacted"] == 1


def test_orphan_session_is_flagged_not_silently_rooted(traced):
    """A part whose session has no row cannot be placed; say so rather than guess."""
    build_dir = traced / "work" / "idea__traced"
    write_dump(
        build_dir,
        "orphan",
        [part_row("p99", "ses_ghost", {"type": "text", "text": "nowhere"}, ts=1090)],
    )
    ghost = next(e for e in _trace(traced)["events"] if e["session_id"] == "ses_ghost")
    assert ghost["unknown_session"] is True


def test_falls_back_to_stdout_when_no_dump_exists(builds):
    """An older build still opens, and labels itself as the partial trace it is."""
    record = read_json(founder_paths(builds, "idea__team").build_json)
    bundle = trace_mod.load_trace(builds, "idea__team", record)
    assert bundle["manifest"]["source"] == "transcript"
    assert bundle["manifest"]["sessions"] == []
    assert bundle["events"], "the root agent's own output survives"
    assert all(e["depth"] == 0 for e in bundle["events"])
    assert not any(e["role"] == "user" for e in bundle["events"])


def test_capture_summary_separates_the_two_harness_counts(traced):
    """Root-vs-stream is the subagents' share; all-vs-stream is a real conflict.

    Comparing the dump-derived stream against the transcript-derived root count
    would flag every delegating build as broken, which is the architecture, not
    a bug.
    """
    bundle = _trace(traced)
    summary = trace_mod.capture_summary(bundle["manifest"], bundle["events"])
    assert summary["has_thinking"] is True
    assert summary["stream_chars"] == 11  # "thinking" + "sub"
    assert summary["harness_root"] == 17  # stdout only, per the fixture's record
    assert summary["harness_all"] == 0  # this fixture predates the split
    assert summary["disagrees"] is False, "no same-source pair to disagree"
    assert summary["prompts"] == 1


def test_a_same_source_mismatch_is_flagged(traced):
    """`_all` and the stream read the same dumps, so a gap means a reader is wrong."""
    bundle = _trace(traced)
    bundle["manifest"]["capture"]["reasoning_chars_all"] = 999
    summary = trace_mod.capture_summary(bundle["manifest"], bundle["events"])
    assert summary["disagrees"] is True


def test_the_subagents_share_is_reported_not_treated_as_a_gap(traced):
    """On a delegating build, stream minus root IS the team's thinking."""
    bundle = _trace(traced)
    bundle["manifest"]["capture"]["reasoning_chars_root"] = 8  # "thinking" only
    bundle["manifest"]["capture"]["reasoning_chars_all"] = 11
    summary = trace_mod.capture_summary(bundle["manifest"], bundle["events"])
    assert summary["team_chars"] == 3, "the subagent's 'sub'"
    assert summary["disagrees"] is False


def test_legacy_and_renamed_capture_keys_both_read(builds):
    """Builds exist from before the rename, during it, and after."""
    legacy = {"capture": {"reasoning_chars": 42}, "totals": {}}
    renamed = {"capture": {"reasoning_chars_root": 42}, "totals": {}}
    for manifest in (legacy, renamed):
        assert trace_mod.capture_summary(manifest, [])["harness_root"] == 42


def test_an_exported_bundle_wins_over_the_raw_dump(traced):
    """A bundle is the authoritative shape when someone has exported one."""
    import json

    out = trace_mod.bundle_dir_for(traced, "idea__traced")
    out.mkdir(parents=True)
    (out / "trajectory.json").write_text(
        json.dumps(
            {"schema_version": 1, "source": "session_store", "totals": {"events": 1}}
        )
    )
    (out / "events.jsonl").write_text(
        json.dumps({"seq": 0, "type": "text", "text": "FROM THE BUNDLE"}) + "\n"
    )
    bundle = _trace(traced)
    assert [e["text"] for e in bundle["events"]] == ["FROM THE BUNDLE"]
    assert bundle["manifest"]["bundle_dir"] == str(out)


# --------------------------------------------------------------------------
# How the viewer's timeline uses it
# --------------------------------------------------------------------------


def test_ui_maps_kinds_and_lanes_subagents(traced):
    trajectory = founder_mod.load_trajectory(traced, "idea__traced")
    kinds = {e["kind"] for e in trajectory["events"]}
    assert {"prompt", "reasoning", "tool", "patch"} <= kinds
    assert "step-start" not in kinds, "bookkeeping parts are not feed rows"

    sub = [lane for lane in trajectory["lanes"] if lane["kind"] == "subagent"]
    assert len(sub) == 1 and sub[0]["session_id"] == "ses_kid"
    assert all(
        e["lane"] == sub[0]["key"] for e in trajectory["events"] if e["depth"] == 1
    )


def test_ui_totals_lead_with_characters_not_tokens(traced):
    """Vertex Anthropic reports 0 reasoning tokens while returning real thought."""
    totals = founder_mod.load_trajectory(traced, "idea__traced")["totals"]
    assert totals["reasoning_chars"] == 11
    assert totals["reasoning_redacted"] == 1
    assert totals["prompts"] == 1
    assert totals["patches"] == 1
    assert totals["subagent_sessions"] == 1
    assert totals["root_sessions"] == 1


def test_solo_turns_are_split_out_of_one_shared_session(builds):
    """Solo runs design and build in ONE session; the dump cannot separate them."""
    root = builds / "work" / "idea__solo"
    write_dump(
        root,
        "ses_a",
        [
            session_row("ses_a", "", title="solo", ts=900),
            # 1000 falls in the design window, 2000 in the build window.
            part_row(
                "q1", "ses_a", {"type": "reasoning", "text": "designing"}, ts=1000
            ),
            part_row("q2", "ses_a", {"type": "reasoning", "text": "building"}, ts=2000),
        ],
    )
    events = founder_mod.load_trajectory(builds, "idea__solo")["events"]
    by_phase = {e["text"]: e["phase"] for e in events if e["kind"] == "reasoning"}
    assert by_phase == {"designing": "design", "building": "build"}


def test_a_turns_prompt_is_credited_to_the_turn_it_starts(builds):
    """A prompt lands in the gap between turns, before the turn it belongs to.

    The previous turn's window closes on its last output, so a naive
    nearest-window rule credits the build prompt to the design turn.
    """
    root = builds / "work" / "idea__solo"
    write_dump(
        root,
        "ses_a",
        [
            session_row("ses_a", "", title="solo", ts=900),
            # design runs 1000..1010, build runs 2000..2010 (see the fixture).
            part_row(
                "g1",
                "ses_a",
                {"type": "text", "text": "DESIGN PROMPT"},
                role="user",
                ts=990,
            ),
            part_row(
                "g2",
                "ses_a",
                {"type": "text", "text": "BUILD PROMPT"},
                role="user",
                ts=1500,
            ),
        ],
    )
    events = founder_mod.load_trajectory(builds, "idea__solo")["events"]
    by_text = {e["text"]: e["phase"] for e in events if e["kind"] == "prompt"}
    assert by_text == {"DESIGN PROMPT": "design", "BUILD PROMPT": "build"}


def test_team_roots_are_not_counted_as_subagents(builds):
    """Four specialists are four ROOT sessions, not one root and three children."""
    root = builds / "work" / "idea__team"
    write_dump(
        root,
        "specialists",
        [session_row(f"ses_{i}", "", title=f"agent {i}", ts=900 + i) for i in range(4)]
        + [
            part_row(
                f"t{i}", f"ses_{i}", {"type": "reasoning", "text": "x"}, ts=1000 + i
            )
            for i in range(4)
        ],
    )
    totals = founder_mod.load_trajectory(builds, "idea__team")["totals"]
    assert totals["root_sessions"] == 4
    assert totals["subagent_sessions"] == 0


# --------------------------------------------------------------------------
# A crowd run has to name the pipeline it scored
# --------------------------------------------------------------------------


def test_a_crowd_run_names_the_founder_arm_and_model(builds):
    """A score is meaningless without its subject.

    The model in a crowd run's own config is the CROWD's, not the founder's, so
    the run has to carry the founder side explicitly or the two get conflated.
    """
    from vizfixtures import make_crowd_run

    run_dir = make_crowd_run(builds, "idea__team__crowd-20260101-000000-s0")
    founder = crowd_mod.load_run(run_dir)["founder"]
    assert founder["on_disk"] is True
    assert founder["arm"] == "team"
    assert founder["arm_label"] == "4-agent team (local)"
    assert founder["model_short"] == "gemini-test"


def test_a_crowd_run_whose_build_was_pruned_says_so(builds):
    """Crowd runs outlive their builds; that must not read as a broken run."""
    from vizfixtures import make_crowd_run

    run_dir = make_crowd_run(builds, "idea__vanished__crowd-20260101-000000-s0")
    founder = crowd_mod.load_run(run_dir)["founder"]
    assert founder["on_disk"] is False
    assert founder["build_id"] == "idea__vanished"


def test_founder_reasoning_prefers_the_subagent_inclusive_count(builds):
    """`_all` includes the subagents; the bare key predates the split."""
    import json

    from vizfixtures import make_crowd_run

    record_path = builds / "work" / "idea__dynamic" / "build.json"
    record = json.loads(record_path.read_text())
    record["trajectory"] = {"reasoning_chars_root": 100, "reasoning_chars_all": 900}
    record_path.write_text(json.dumps(record))

    run_dir = make_crowd_run(builds, "idea__dynamic__crowd-20260101-000000-s0")
    assert crowd_mod.load_run(run_dir)["founder"]["reasoning_chars"] == 900
