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

"""Tests for the trajectory readers.

The cases that matter are the damaged ones: a killed turn leaves a truncated final
line, a failed turn leaves a 0-byte file, and retired run structures still have to
open. Fixtures live in conftest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import crowd as crowd_mod  # noqa: E402
from core import export as export_mod  # noqa: E402
from core import founder as founder_mod  # noqa: E402
from core import paths as paths_mod  # noqa: E402
from vizfixtures import make_crowd_run  # noqa: E402

# --------------------------------------------------------------------------
# founder
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build_id,mode,lanes,phases",
    [
        ("idea__solo", "solo", 1, 2),
        ("idea__team", "team", 4, 8),
        ("idea__dynamic", "dynamic", 1, 1),
        ("idea__legacy", "legacy", 2, 2),
    ],
)
def test_every_structure_loads(builds, build_id, mode, lanes, phases):
    """All three live structures, plus the retired relay mode, must open."""
    trajectory = founder_mod.load_trajectory(builds, build_id)
    assert trajectory is not None
    assert trajectory["mode"] == mode
    assert len(trajectory["lanes"]) == lanes
    assert len(trajectory["phases"]) == phases


def test_team_lanes_keep_one_session_each(builds):
    """A specialist resumes its own session every round; the lane must show that."""
    trajectory = founder_mod.load_trajectory(builds, "idea__team")
    for lane in trajectory["lanes"]:
        assert len(lane["phases"]) == 2, "one phase per round"
        assert len(lane["session_ids"]) == 1, "the same session across both rounds"


def test_timeline_is_sorted_and_indexed(builds):
    trajectory = founder_mod.load_trajectory(builds, "idea__team")
    stamps = [e["t"] for e in trajectory["events"]]
    assert stamps == sorted(stamps)
    assert [e["i"] for e in trajectory["events"]] == list(range(len(stamps)))


def test_damaged_transcripts_do_not_break_the_reader(builds):
    """A 0-byte file and a truncated final line are normal, not exceptional."""
    trajectory = founder_mod.load_trajectory(builds, "idea__legacy")
    assert trajectory["phases"][0]["transcript_bytes"] == 0
    # The intact line survives; the truncated one is skipped rather than fatal.
    assert trajectory["totals"]["tools"] == 1


def test_missing_build_returns_none(builds):
    assert founder_mod.load_trajectory(builds, "does__not__exist") is None


def test_cost_and_tokens_roll_up(builds):
    trajectory = founder_mod.load_trajectory(builds, "idea__solo")
    assert trajectory["totals"]["cost"] == pytest.approx(0.51)
    assert trajectory["totals"]["steps"] == 2


def test_a_build_with_no_dump_reports_no_thinking(builds):
    """An untraced build must say so, not show token counts as if they were thought."""
    trajectory = founder_mod.load_trajectory(builds, "idea__team")
    assert trajectory["trace"]["source"] == "transcript"
    assert trajectory["totals"]["reasoning_chars"] == 0
    assert not any(e["kind"] == "reasoning" for e in trajectory["events"])
    # The token counter still reads non-zero; it must never be the headline.
    assert trajectory["totals"]["tokens_reasoning"] == 56  # 8 turns x 7


def test_the_thinking_note_states_what_the_states_mean(builds):
    """The note is shown wherever thinking is; it has to carry the two traps."""
    note = founder_mod.load_trajectory(builds, "idea__team")["thinking_note"]
    assert "redacted" in note, "encrypted thinking must be distinguished from absent"
    assert "tokens_reasoning" in note, "the Claude zero-token trap must be stated"


def test_edit_diff_stats_are_extracted(builds):
    trajectory = founder_mod.load_trajectory(builds, "idea__solo")
    write = next(e for e in trajectory["events"] if e.get("tool") == "write")
    assert write["diff"] == {
        "file": "main.py",
        "additions": 1,
        "deletions": 0,
        "has_patch": True,
    }


def test_browser_result_splits_into_sections(builds):
    trajectory = founder_mod.load_trajectory(builds, "idea__dynamic")
    click = next(e for e in trajectory["events"] if e.get("group") == "browser")
    assert click["browser"]["code"] == "await page.click();"
    assert click["browser"]["url"] == "http://localhost:8000/"
    assert click["browser"]["title"] == "Demo"
    assert "button [ref=e5]" in click["browser"]["snapshot"]


def test_dynamic_spawns_carry_overlapping_intervals(builds):
    trajectory = founder_mod.load_trajectory(builds, "idea__dynamic")
    spawns = trajectory["spawns"]
    assert len(spawns) == 2
    # The overlap is the finding: two subagents ran at once.
    assert spawns[0]["start_ms"] < spawns[1]["start_ms"] < spawns[0]["end_ms"]


def test_event_body_is_fetchable_by_part_id(builds):
    """The detail pane's click-through, on a build with no session dump."""
    trajectory = founder_mod.load_trajectory(builds, "idea__solo")
    write = next(e for e in trajectory["events"] if e.get("tool") == "write")
    body = founder_mod.load_event_body(builds, "idea__solo", part_id=write["part_id"])
    assert body["part"]["state"]["input"]["content"] == "hi"


def test_index_classifies_every_mode(builds):
    rows = {b.build_id: b for b in paths_mod.index_builds(builds)}
    assert rows["idea__solo"].mode == "solo"
    assert rows["idea__team"].mode_label == "4-agent team (local)"
    assert rows["idea__legacy"].mode == "legacy"


# --------------------------------------------------------------------------
# crowd
# --------------------------------------------------------------------------


def test_crowd_run_loads(tmp_path):
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-20260101-000000-s0")
    run = crowd_mod.load_run(run_dir)
    assert run["app_type"] == "client-app"
    assert len(run["agents"]) == 3  # founder + 2
    assert len(run["posts"]) == 2
    assert run["posts"][1]["kind"] == "repost"


def test_timeline_maps_steps_to_rounds(tmp_path):
    """0 = signup, 1 = launch, round N = N+1, last = interview."""
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-1")
    labels = {s["t"]: s["label"] for s in crowd_mod.load_run(run_dir)["timeline"]}
    assert labels[0] == "Sign-ups & follow graph"
    assert labels[1] == "Launch post"
    assert labels[2] == "Round 1"
    assert labels[3] == "Round 2"


def test_double_encoded_info_is_decoded(tmp_path):
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-2")
    run = crowd_mod.load_run(run_dir)
    posted = next(a for a in run["actions"] if a["action"] == "create_post")
    assert posted["text"] == "launched"
    assert posted["post_id"] == 1


def test_corrupt_action_line_is_skipped(tmp_path):
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-3")
    run = crowd_mod.load_run(run_dir)
    assert len(run["actions"]) == 6, "the corrupt seventh line is dropped, not fatal"


def test_feeds_are_time_resolved_but_bodies_are_lazy(tmp_path):
    """Feed bodies are the biggest thing in a run; the index keeps ids only."""
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-4")
    run = crowd_mod.load_run(run_dir)
    assert run["feeds"]["1"]["2"] == [1]
    posts = crowd_mod.load_feed(run_dir, 1, 2)
    assert posts[0]["content"] == "launched"
    assert posts[0]["author"] == "@founder"


def test_agent_roster_fuses_every_source(tmp_path):
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-5")
    maya = next(a for a in crowd_mod.load_run(run_dir)["agents"] if a["id"] == 1)
    assert maya["username"] == "maya"
    assert maya["tier"] == "trier"
    assert maya["trial"]["delight"] == 8
    assert maya["interview"]["why"] == "it is good"
    # Crowd reasoning is real, unlike the founder's.
    assert maya["reasoning"] == ["I tried the app and it worked."]


def test_the_two_verdict_passes_stay_separate(tmp_path):
    """An agent is scored twice and the passes disagree, so the payload must
    keep them apart. The header reads its headline delight off `triers` and
    names the source in the label; if these ever collapsed into one number the
    UI would be quoting the wrong population without saying so."""
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-9")
    verdicts = crowd_mod.load_run(run_dir)["verdicts"]
    # Hands-on triers: only the agents who actually drove the app.
    assert verdicts["triers"]["n"] == 1
    assert verdicts["triers"]["delight_mean"] == 8.0
    assert verdicts["triers"]["would_use_rate"] == 1.0
    # Exit interviews: everyone, including agents who only saw a post.
    assert verdicts["interviews"]["n"] == 2
    assert verdicts["interviews"]["delight_mean"] == 6.0
    assert verdicts["interviews"]["would_use_rate"] == 0.5


def test_a_non_trier_carries_an_interview_but_no_trial(tmp_path):
    """The per-agent table falls back from trial to interview, so it has to be
    able to tell which one it got."""
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-10")
    agents = {a["id"]: a for a in crowd_mod.load_run(run_dir)["agents"]}
    assert agents[1]["trial"] is not None and agents[1]["interview"] is not None
    assert agents[2]["trial"] is None
    assert agents[2]["interview"]["delight"] == 4


def test_founder_is_agent_zero(tmp_path):
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-6")
    founder = next(a for a in crowd_mod.load_run(run_dir)["agents"] if a["id"] == 0)
    assert founder["tier"] == "founder"


def test_trial_exposes_steps_and_screenshots(tmp_path):
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-7")
    trial = crowd_mod.load_trial(run_dir, 1)
    assert [s["action"] for s in trial["steps"]] == ["open", "click", "finish"]
    click = trial["steps"][1]
    assert click["screenshot"] == "shot.png"
    assert click["screenshot_exists"] is True
    assert trial["steps"][0]["screenshot"] is None


def test_missing_trial_returns_none(tmp_path):
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-8")
    assert crowd_mod.load_trial(run_dir, 99) is None


def test_run_dir_resolution(tmp_path):
    root = tmp_path / "builds"
    make_crowd_run(root, "bid__crowd-20260101-000000-s0")
    assert paths_mod.crowd_run_dir(root, "bid__crowd-20260101-000000-s0") is not None
    assert paths_mod.crowd_run_dir(root, "nope") is None
    assert [p.name for p in paths_mod.crowd_runs_for_build(root, "bid")] == [
        "bid__crowd-20260101-000000-s0"
    ]


# --------------------------------------------------------------------------
# export and safety
# --------------------------------------------------------------------------


def test_founder_bundle_carries_raw_transcripts(builds):
    bundle = export_mod.founder_bundle(builds, "idea__team")
    assert bundle["kind"] == "viral_bench.founder_trajectory"
    assert len(bundle["transcripts"]) == 8
    assert bundle["mode"] == "team"


def test_founder_zip_contains_the_build_folder(builds):
    import io
    import zipfile

    blob = export_mod.founder_zip(builds, "idea__solo")
    names = zipfile.ZipFile(io.BytesIO(blob)).namelist()
    assert "idea__solo/build.json" in names
    assert "idea__solo/transcript/design.json" in names
    assert "idea__solo/trajectory.json" in names


def test_crowd_bundle_includes_trials(tmp_path):
    run_dir = make_crowd_run(tmp_path / "builds", "idea__x__crowd-9")
    bundle = export_mod.crowd_bundle(run_dir)
    assert "1" in bundle["trials"]
    assert len(bundle["trials"]["1"]["steps"]) == 3


def test_writes_outside_the_cache_are_refused(tmp_path):
    """The builds tree belongs to a live checkout and must never be written to."""
    with pytest.raises(paths_mod.OutsideCache):
        paths_mod.assert_writable(tmp_path / "somewhere" / "else.png")
    ok = paths_mod.assert_writable(paths_mod.CACHE_ROOT / "shots" / "x.png")
    assert paths_mod.CACHE_ROOT.resolve() in ok.parents


def test_screenshot_lookup_rejects_path_traversal():
    from core import cache as cache_mod

    assert cache_mod.resolve_shot("../../etc/passwd") is None
    assert cache_mod.resolve_shot("") is None
