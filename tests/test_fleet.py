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

"""Tests for the fleet corpus and the loop's stopping gates.

These cover the two things that decide whether the benchmark may declare a
result: how runs on disk are turned into a comparable corpus, and the paired
statistics that corpus is judged by. The statistics are exercised on
hand-constructed corpora with known answers, because "the gap is +12 points" is
only trustworthy if the arithmetic behind it is pinned down.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from viral_bench.crowd.sim_defaults import CROWD_ARCH_VERSION
from viral_bench.score.fleet import (
    FleetBuild,
    FleetCorpus,
    FleetSpec,
    ScoredRun,
    control_separation,
    fleet_build_ids,
    load_builds,
    load_corpus,
    paired_gap,
    self_separation,
    within_cell_sd,
)

REPO = Path(__file__).resolve().parents[1]

A = "model-a"
B = "model-b"


def _build(idea: str, model: str, build_id: str, **kw) -> FleetBuild:
    return FleetBuild(
        build_id=build_id,
        idea_id=idea,
        model=model,
        status=kw.pop("status", "ok"),
        config={"structure": "team", "n_agents": 4},
        **kw,
    )


def _corpus(spec: dict[tuple[str, str], list[float]], controls=()) -> FleetCorpus:
    """Build a corpus from {(idea, model): [scores]} plus control scores."""
    builds: dict[str, FleetBuild] = {}
    runs: list[ScoredRun] = []
    for (idea, model), scores in spec.items():
        build_id = f"{idea}__{model}"
        builds[build_id] = _build(idea, model, build_id)
        for seed, score in enumerate(scores):
            runs.append(
                ScoredRun(
                    crowd_dir=f"/tmp/{build_id}__{seed}",
                    build_id=build_id,
                    idea_id=idea,
                    model=model,
                    seed=seed,
                    n_agents=50,
                    arch_version=CROWD_ARCH_VERSION,
                    score=score,
                )
            )
    for i, score in enumerate(controls):
        build_id = "control__broken"
        builds[build_id] = FleetBuild(
            build_id=build_id,
            idea_id="quick_notes_app",
            model="broken",
            status="ok",
            is_control=True,
        )
        runs.append(
            ScoredRun(
                crowd_dir=f"/tmp/control_{i}",
                build_id=build_id,
                idea_id="quick_notes_app",
                model="broken",
                seed=i,
                n_agents=50,
                arch_version=CROWD_ARCH_VERSION,
                score=score,
            )
        )
    return FleetCorpus(
        builds=builds, runs=runs, fleet_ids=set(builds) - {"control__broken"}
    )


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #


def test_paired_gap_is_the_mean_of_per_idea_differences():
    corpus = _corpus(
        {
            ("i1", A): [60.0, 62.0],
            ("i1", B): [50.0, 52.0],
            ("i2", A): [30.0],
            ("i2", B): [26.0],
        }
    )
    gap = paired_gap(corpus, A, B)
    assert gap.per_idea == {"i1": 10.0, "i2": 4.0}
    assert gap.gap == pytest.approx(7.0)
    assert gap.n_ideas == 2
    assert gap.wins_a == 2 and gap.wins_b == 0


def test_paired_gap_ignores_ideas_only_one_model_built():
    corpus = _corpus({("i1", A): [60.0], ("i1", B): [50.0], ("i2", A): [99.0]})
    gap = paired_gap(corpus, A, B)
    assert gap.n_ideas == 1
    assert gap.gap == pytest.approx(10.0)


def test_paired_gap_is_immune_to_idea_difficulty():
    """Adding a constant per idea must not move the paired gap.

    This is the property Cohen's d over pooled runs does not have: shifting one
    idea by +40 points inflates the pooled SD and shrinks d, while the actual
    model difference is unchanged.
    """
    flat = _corpus(
        {("i1", A): [60.0], ("i1", B): [50.0], ("i2", A): [60.0], ("i2", B): [50.0]}
    )
    spread = _corpus(
        {("i1", A): [60.0], ("i1", B): [50.0], ("i2", A): [20.0], ("i2", B): [10.0]}
    )
    assert paired_gap(flat, A, B).gap == paired_gap(spread, A, B).gap == 10.0


def test_paired_gap_ci_needs_at_least_three_ideas():
    corpus = _corpus({("i1", A): [60.0], ("i1", B): [50.0]})
    gap = paired_gap(corpus, A, B)
    assert gap.ci_low is None and gap.ci_high is None
    assert not gap.excludes_zero


def test_paired_gap_ci_brackets_a_real_gap():
    spec: dict[tuple[str, str], list[float]] = {}
    for i in range(8):
        spec[(f"i{i}", A)] = [50.0 + i]
        spec[(f"i{i}", B)] = [40.0 + i]
    gap = paired_gap(_corpus(spec), A, B)
    assert gap.gap == pytest.approx(10.0)
    assert gap.ci_low == gap.ci_high == pytest.approx(10.0)
    assert gap.excludes_zero


def test_self_separation_splits_seeds_and_finds_no_gap_when_stable():
    spec = {(f"i{i}", A): [50.0, 50.0, 50.0] for i in range(4)}
    null = self_separation(_corpus(spec), A)
    assert null.n_ideas == 4
    assert null.gap == pytest.approx(0.0)


def test_self_separation_reports_noise_when_seeds_disagree():
    # Seed 0 and 2 (the "even" half) sit 10 points above seed 1.
    spec = {(f"i{i}", A): [60.0, 50.0, 60.0] for i in range(4)}
    null = self_separation(_corpus(spec), A)
    assert null.gap == pytest.approx(10.0)


def test_self_separation_skips_single_seed_cells():
    corpus = _corpus({("i1", A): [50.0], ("i2", A): [50.0, 52.0]})
    assert self_separation(corpus, A).n_ideas == 1


def test_within_cell_sd_is_the_pooled_seed_noise():
    corpus = _corpus({("i1", A): [48.0, 52.0], ("i2", A): [48.0, 52.0]})
    # variance of [48, 52] is 8.0; pooled sd = sqrt(8)
    assert within_cell_sd(corpus) == pytest.approx(8.0**0.5, abs=0.01)


def test_within_cell_sd_is_none_without_repeats():
    assert within_cell_sd(_corpus({("i1", A): [50.0]})) is None


# --------------------------------------------------------------------------- #
# Control separation
# --------------------------------------------------------------------------- #


def test_control_separation_measures_the_gap_to_the_weakest_real_cell():
    corpus = _corpus(
        {("i1", A): [60.0], ("i1", B): [30.0], ("i2", A): [55.0], ("i2", B): [40.0]},
        controls=[3.0, 5.0, 4.0],
    )
    sep = control_separation(corpus)
    assert sep.n_control_runs == 3
    assert sep.control_max == 5.0
    assert sep.weakest_real_cell == "i1[model-b]"
    assert sep.margin == pytest.approx(25.0)


def test_control_separation_without_controls_is_reported_not_crashed():
    sep = control_separation(_corpus({("i1", A): [60.0]}))
    assert sep.n_control_runs == 0
    assert sep.margin is None


def test_control_runs_are_never_part_of_a_model_comparison():
    corpus = _corpus({("quick_notes_app", A): [60.0]}, controls=[3.0])
    assert all(r.model != "broken" for r in corpus.fleet_runs())
    assert len(corpus.control_runs()) == 1


# --------------------------------------------------------------------------- #
# Loading from disk
# --------------------------------------------------------------------------- #


def _write_build(root: Path, build_id: str, **record) -> None:
    d = root / "work" / build_id
    d.mkdir(parents=True, exist_ok=True)
    payload = {"build_id": build_id, "idea_id": "i1", "model": "m", "status": "ok"}
    payload.update(record)
    (d / "build.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_run(root: Path, build_id: str, seed: int, *, would_use: bool) -> Path:
    d = root / "crowd" / f"{build_id}__crowd-{seed}"
    d.mkdir(parents=True, exist_ok=True)
    per_agent = [
        {"agent_id": i, "would_use": would_use, "would_share": would_use, "delight": 8}
        for i in range(5)
    ]
    (d / "run_summary.json").write_text(
        json.dumps(
            {
                "build_id": build_id,
                "crowd_arch_version": CROWD_ARCH_VERSION,
                "ok": True,
                "config": {"seed": seed, "n_agents": 5},
                "rounds": [{"round": 1, "ok": True}],
                "crowd": [{"agent_id": i, "influence": 1} for i in range(5)],
                "engagement": {"reach": {"exposed_agents": 5, "actors_commented": 2}},
                "verdicts": {
                    "interviews": {
                        "n": 5,
                        "would_use_rate": 1.0 if would_use else 0.0,
                        "per_agent": per_agent,
                    },
                    "triers": {"per_agent": []},
                },
            }
        ),
        encoding="utf-8",
    )
    return d


def test_load_builds_reads_records_and_flags_the_control(tmp_path):
    root = tmp_path / "builds"
    _write_build(root, "b1", model="google-vertex/gemini-test", rounds_run=2)
    _write_build(root, "ctl", model="control/broken")
    builds = load_builds(root)
    assert builds["b1"].model == "gemini-test"  # provider prefix stripped
    assert builds["b1"].rounds_run == 2
    assert builds["b1"].is_control is False
    assert builds["ctl"].is_control is True


def test_load_builds_survives_a_corrupt_record(tmp_path):
    root = tmp_path / "builds"
    _write_build(root, "b1")
    bad = root / "work" / "b2"
    bad.mkdir(parents=True)
    (bad / "build.json").write_text("{not json", encoding="utf-8")
    assert set(load_builds(root)) == {"b1"}


def test_build_seconds_sums_the_phase_durations(tmp_path):
    root = tmp_path / "builds"
    _write_build(
        root,
        "b1",
        phases=[{"duration_s": 100.0}, {"duration_s": 40.5}, {"duration_s": None}],
    )
    assert load_builds(root)["b1"].build_seconds == pytest.approx(140.5)


def test_fleet_build_ids_reads_the_index(tmp_path):
    root = tmp_path / "builds"
    root.mkdir(parents=True)
    (root / "fleet.json").write_text(
        json.dumps({"entries": {"k": {"build_id": "b1"}, "j": {"build_id": ""}}}),
        encoding="utf-8",
    )
    assert fleet_build_ids(root) == {"b1"}


def test_fleet_build_ids_is_empty_without_an_index(tmp_path):
    assert fleet_build_ids(tmp_path / "builds") == set()


def test_load_corpus_scores_runs_and_attributes_them_to_builds(tmp_path):
    root = tmp_path / "builds"
    _write_build(root, "b1", model="gemini-test", idea_id="i1")
    _write_run(root, "b1", 0, would_use=True)
    _write_run(root, "b1", 1, would_use=False)
    root.mkdir(parents=True, exist_ok=True)
    (root / "fleet.json").write_text(
        json.dumps({"entries": {"i1::a": {"build_id": "b1"}}}), encoding="utf-8"
    )
    corpus = load_corpus(tmp_path)
    assert len(corpus.runs) == 2
    assert {r.seed for r in corpus.runs} == {0, 1}
    assert all(r.idea_id == "i1" and r.model == "gemini-test" for r in corpus.runs)
    # The enthusiastic crowd must score strictly above the indifferent one.
    by_seed = {r.seed: r.score for r in corpus.runs}
    assert by_seed[0] > by_seed[1]
    assert len(corpus.fleet_runs()) == 2


def test_a_fleet_spec_keeps_out_every_other_arm_on_disk(tmp_path):
    """builds/fleet.json is append-only, so "the fleet" must be a named subset.

    The index accumulates every experiment the repo has ever run -- two model
    pairs, three founder structures, two replicates. Loading "the fleet" without
    saying which one silently pools a 4-agent team build of an idea with a solo
    build of the same idea and calls the difference seed noise.
    """
    root = tmp_path / "builds"
    # The arm under test: solo, replicate 2, both models under comparison.
    _write_build(root, "want_a", idea_id="i1", model="gemini-test", n_agents=1)
    _write_build(
        root,
        "want_b",
        idea_id="i1",
        model="google-vertex-anthropic/claude-test@default",
        n_agents=1,
    )
    # Same idea, same models, but a different founder structure / replicate /
    # model -- each of which is a different experiment.
    _write_build(root, "team", idea_id="i1", model="gemini-test", n_agents=4)
    _write_build(root, "older_rep", idea_id="i1", model="gemini-test", n_agents=1)
    _write_build(root, "other_model", idea_id="i1", model="gemini-old-test", n_agents=1)
    for bid in ("want_a", "want_b", "team", "older_rep", "other_model"):
        _write_run(root, bid, 0, would_use=True)
    (root / "fleet.json").write_text(
        json.dumps(
            {
                "entries": {
                    "want_a": {"build_id": "want_a", "replicate": 2},
                    "want_b": {"build_id": "want_b", "replicate": 2},
                    "team": {"build_id": "team", "replicate": 2},
                    "older_rep": {"build_id": "older_rep", "replicate": 1},
                    "other_model": {"build_id": "other_model", "replicate": 2},
                }
            }
        ),
        encoding="utf-8",
    )
    spec = FleetSpec(
        model_a="gemini-test",
        model_b="claude-test@default",
        structure="solo",
        replicate=2,
    )
    corpus = load_corpus(tmp_path, spec=spec)
    assert corpus.fleet_ids == {"want_a", "want_b"}
    # collab defaults to absent in the fixture, so a 4-agent build is not "team"
    # -- what matters is that it is not "solo" and so cannot enter the fleet.
    assert {r.build_id for r in corpus.fleet_runs()} == {"want_a", "want_b"}
    # Runs themselves are NOT filtered: the control is not a fleet build and has
    # to survive, or control_separation loses its only measurement.
    assert len(corpus.runs) == 5


def test_load_corpus_ignores_runs_with_no_matching_build(tmp_path):
    root = tmp_path / "builds"
    _write_build(root, "b1")
    _write_run(root, "orphan", 0, would_use=True)
    assert load_corpus(tmp_path).runs == []


def test_stored_autorating_is_folded_in_without_calling_a_model(tmp_path):
    root = tmp_path / "builds"
    _write_build(root, "b1", model="gemini-test", idea_id="i1")
    run_dir = _write_run(root, "b1", 0, would_use=True)
    (run_dir / "autorating.json").write_text(
        json.dumps(
            {
                "build_id": "b1",
                "model": "rater",
                "repeats": 3,
                "dimensions": {"substance": {"score": 9.0}, "severity": {"score": 8.0}},
            }
        ),
        encoding="utf-8",
    )
    run = load_corpus(tmp_path).runs[0]
    assert run.components["substance"] == pytest.approx(0.9)
    assert run.components["severity"] == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# The stopping gates
# --------------------------------------------------------------------------- #


def _loop_status():
    """Load scripts/loop_status.py with a two-model fleet configured.

    The shipped ``MODEL_A``/``MODEL_B`` are empty, because a released benchmark
    has no business defaulting to somebody's model pair. These gates are all
    about comparing two models, so the fixture names two fake ones -- and they
    must be DISTINCT: with both empty the A and B cells collide and the gate
    silently compares a model against itself.
    """
    spec = importlib.util.spec_from_file_location(
        "loop_status", REPO / "scripts" / "loop_status.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["loop_status"] = module
    spec.loader.exec_module(module)
    module.MODEL_A = "openai/gpt-test-a"
    module.MODEL_B = "anthropic/claude-test-b"
    return module


def test_loop_status_thresholds_are_a_real_floor():
    ls = _loop_status()
    assert ls.MIN_ATTEMPTED == 25  # every idea, both models, no partial fleet
    assert ls.MIN_MEASURED_IDEAS == 25  # every idea measured for both models
    assert ls.MIN_SEEDS >= 3
    assert ls.MIN_SCORABLE_FRACTION >= 0.9
    assert ls.MIN_CONTROL_MEDIAN_MARGIN >= 25.0
    assert ls.MAX_BELOW_CONTROL_SHARE <= 0.10
    assert ls.MAX_CONTROL_SCORE <= 20.0
    assert ls.ITERATION_CAP == 30


def test_fleet_gate_fails_when_ideas_were_never_attempted():
    """A fleet that only ever tried 20 ideas is not the fleet."""
    ls = _loop_status()
    builds = {}
    for i in range(20):
        for model in (ls.MODEL_A, ls.MODEL_B):
            bid = f"i{i}__{model}"
            builds[bid] = FleetBuild(
                build_id=bid,
                idea_id=f"i{i}",
                model=model,
                status="ok",
                config={"n_agents": 4},
            )
    _, gate, _ = ls.fleet_section(
        FleetCorpus(builds=builds, runs=[], fleet_ids=set(builds))
    )
    assert not gate.passed
    assert "attempted 20<25" in gate.detail


def _full_corpus(ls, *, statuses=None):
    """25 ideas x 2 models, every cell measured by one scorable run."""
    spec = {}
    for i in range(25):
        for model in (ls.MODEL_A, ls.MODEL_B):
            spec[(f"i{i}", model)] = [50.0]
    corpus = _corpus(spec)
    for bid, build in list(corpus.builds.items()):
        status = (statuses or {}).get(bid, "ok")
        corpus.builds[bid] = FleetBuild(
            build_id=bid,
            idea_id=build.idea_id,
            model=build.model,
            status=status,
            config={"n_agents": 4},
        )
    return corpus


def test_fleet_gate_fails_on_mixed_founder_configs():
    ls = _loop_status()
    corpus = _full_corpus(ls)
    builds = corpus.builds
    _, gate, _ = ls.fleet_section(corpus)
    assert gate.passed
    # One build under a different structure invalidates the whole fleet.
    odd = next(iter(builds))
    builds[odd] = FleetBuild(
        build_id=odd,
        idea_id="i0",
        model=ls.MODEL_A,
        status="ok",
        config={"n_agents": 1},
    )
    _, gate, _ = ls.fleet_section(corpus)
    assert not gate.passed
    assert "distinct founder configs" in gate.detail


def test_fleet_gate_ignores_build_failures_once_every_cell_is_measured():
    """A model failing builds is the RESULT, not a blocked loop.

    Every build is simulated now -- one with no runnable manifest is presented
    to the crowd, found unlaunchable and scored at the floor -- so coverage is
    what the gate checks. Gating on build success let a worse model block the
    loop forever, which is what a 20-of-25 threshold did.
    """
    ls = _loop_status()
    statuses = {f"i{i}__{ls.MODEL_B}": "manifest_missing" for i in range(10)}
    corpus = _full_corpus(ls, statuses=statuses)
    lines, gate, info = ls.fleet_section(corpus)
    assert gate.passed
    assert len(info["paired"]) == 25  # measured for both, not built by both
    text = "\n".join(lines)
    assert "BUILD OUTCOME  A 25/25 vs B 15/25 builds shipped" in text
    assert "failure rate 0% vs 40%" in text
    assert "manifest_missing=10" in text


def test_fleet_gate_fails_when_a_cell_was_never_measured():
    """Coverage is the gate: an idea with no crowd run for one model blocks."""
    ls = _loop_status()
    corpus = _full_corpus(ls)
    corpus.runs = [
        r for r in corpus.runs if not (r.idea_id == "i7" and r.model == ls.MODEL_B)
    ]
    _, gate, info = ls.fleet_section(corpus)
    assert not gate.passed
    assert "24/25 ideas measured" in gate.detail
    assert len(info["paired"]) == 24


def test_crowd_gate_requires_three_seeds_on_both_models():
    ls = _loop_status()
    spec: dict[tuple[str, str], list[float]] = {}
    for i in range(25):
        spec[(f"i{i}", ls.MODEL_A)] = [50.0, 51.0, 52.0]
        spec[(f"i{i}", ls.MODEL_B)] = [40.0, 41.0, 42.0]
    corpus = _corpus(spec)
    paired = [f"i{i}" for i in range(25)]
    _, gate = ls.crowd_section(corpus, paired)
    assert gate.passed
    # Drop one model's third seed everywhere: no longer enough paired cells.
    thin = _corpus({k: (v[:2] if k[1] == ls.MODEL_B else v) for k, v in spec.items()})
    _, gate = ls.crowd_section(thin, paired)
    assert not gate.passed


def test_control_gate_fails_when_the_corpse_scores_like_a_real_app():
    ls = _loop_status()
    corpus = _corpus({("i1", ls.MODEL_A): [30.0]}, controls=[25.0, 26.0, 27.0])
    _, gate = ls.control_section(corpus)
    assert not gate.passed


def test_control_gate_passes_with_a_wide_margin():
    ls = _loop_status()
    corpus = _corpus({("i1", ls.MODEL_A): [40.0]}, controls=[3.0, 4.0, 5.0])
    _, gate = ls.control_section(corpus)
    assert gate.passed


def test_control_gate_tolerates_a_few_real_builds_worse_than_the_corpse():
    """Some real apps genuinely are worse than a page that at least renders.

    One build could not be reached by a single trier and scores 7.4, below the
    control's 11.2. That is the score being right. What must not happen is the
    *bulk* of real builds sitting near the control.
    """
    ls = _loop_status()
    spec = {(f"i{i}", ls.MODEL_A): [50.0 + i] for i in range(19)}
    spec[("bad", ls.MODEL_A)] = [4.0]
    corpus = _corpus(spec, controls=[7.0, 8.0, 9.0])
    _, gate = ls.control_section(corpus)
    assert gate.passed  # 1 of 20 below the control is inside the 10% budget

    for i in range(3):
        spec[(f"worse{i}", ls.MODEL_A)] = [4.0]
    _, gate = ls.control_section(_corpus(spec, controls=[7.0, 8.0, 9.0]))
    assert not gate.passed
    assert "score at or below the control" in gate.detail


def test_control_gate_fails_a_profile_that_compresses_the_scale():
    """Squeezing every score toward the middle must not look like separation.

    A profile can shrink Cohen's d's denominator by compressing scores; the
    median-margin clause is what stops the same trick working here.
    """
    ls = _loop_status()
    spec = {(f"i{i}", ls.MODEL_A): [30.0 + i * 0.1] for i in range(10)}
    corpus = _corpus(spec, controls=[14.0, 15.0, 16.0])
    _, gate = ls.control_section(corpus)
    assert not gate.passed
    assert "median working build" in gate.detail


def test_verdict_gate_reads_the_iteration_log(tmp_path, monkeypatch):
    ls = _loop_status()
    log = tmp_path / "log.md"
    log.write_text("## Iteration 1\nstuff\n", encoding="utf-8")
    monkeypatch.setattr(ls, "LOG_PATH", log)
    assert ls.iteration_count() == 1
    assert ls.verdict_section() == ""
    log.write_text(
        "## Iteration 1\nstuff\n## Iteration 2\nmore\n## VERDICT\n" + "x" * 500,
        encoding="utf-8",
    )
    assert ls.iteration_count() == 2
    assert len(ls.verdict_section()) >= 500


def test_loop_status_runs_end_to_end_and_ends_with_the_exit_line(capsys):
    ls = _loop_status()
    assert ls.main(["--skip-ci"]) == 0
    out = capsys.readouterr().out
    assert len(out) < 13000
    assert out.strip().splitlines()[-1].startswith("LOOP_EXIT: ")
    # --skip-ci can never declare DONE: the CI gate was not evaluated.
    assert "LOOP_EXIT: DONE" not in out


def test_runs_from_an_older_crowd_architecture_are_excluded(tmp_path):
    """A crowd change cannot be applied retroactively, so old runs must not pool."""
    root = tmp_path / "builds"
    _write_build(root, "b1", model="gemini-test", idea_id="i1")
    new_run = _write_run(root, "b1", 0, would_use=True)
    old_run = _write_run(root, "b1", 1, would_use=True)
    payload = json.loads((old_run / "run_summary.json").read_text(encoding="utf-8"))
    payload["crowd_arch_version"] = "0"
    (old_run / "run_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    (root / "fleet.json").write_text(
        json.dumps({"entries": {"i1::a": {"build_id": "b1"}}}), encoding="utf-8"
    )
    corpus = load_corpus(tmp_path)
    assert len(corpus.runs) == 2
    assert [r.crowd_dir for r in corpus.fleet_runs()] == [str(new_run)]
    assert [r.crowd_dir for r in corpus.stale_runs()] == [str(old_run)]


def test_control_separation_ignores_real_builds_that_are_themselves_broken():
    """A real build that fails its validity gate is a corpse too, not a floor.

    One model shipped an app that does not run; it correctly scores next to the
    broken control. Comparing the control against the *weakest* real cell would
    read that as the instrument failing, when it is the instrument being right.
    """
    corpus = _corpus(
        {("i1", A): [60.0], ("i2", A): [55.0], ("i2", B): [40.0]},
        controls=[3.0, 4.0, 5.0],
    )
    # i1[B] is a real build whose app does not run: gate 0.2, score 7.
    build_id = "i1__model-b"
    corpus.builds[build_id] = _build("i1", B, build_id)
    corpus.fleet_ids.add(build_id)
    corpus.runs.append(
        ScoredRun(
            crowd_dir="/tmp/i1_b",
            build_id=build_id,
            idea_id="i1",
            model=B,
            seed=0,
            n_agents=30,
            arch_version=CROWD_ARCH_VERSION,
            gate=0.2,
            dead=True,
            score=7.0,
        )
    )
    sep = control_separation(corpus)
    assert sep.weakest_real_cell == "i2[model-b]"
    assert sep.margin == pytest.approx(35.0)
    assert sep.dead_cells == {"i1[model-b]": 7.0}
    assert sep.dead_max == 7.0


def test_control_gate_fails_when_a_dead_build_outscores_a_working_one():
    ls = _loop_status()
    corpus = _corpus({("i1", ls.MODEL_A): [20.0]}, controls=[3.0, 4.0, 5.0])
    build_id = "i2__dead"
    corpus.builds[build_id] = _build("i2", ls.MODEL_B, build_id)
    corpus.fleet_ids.add(build_id)
    corpus.runs.append(
        ScoredRun(
            crowd_dir="/tmp/dead",
            build_id=build_id,
            idea_id="i2",
            model=ls.MODEL_B,
            seed=0,
            n_agents=30,
            arch_version=CROWD_ARCH_VERSION,
            gate=0.2,
            dead=True,
            score=30.0,
        )
    )
    _, gate = ls.control_section(corpus)
    assert not gate.passed
    assert "does not run scores 30.0" in gate.detail
    assert "median working build" in gate.detail


def test_a_failed_self_check_is_not_a_corpse():
    """0.2x and 0.6x exist because they are different failures; keep them apart.

    An app that runs but whose author wrote the wrong smoke command is an app the
    crowd used successfully. Lumping it in with apps that do not start flagged a
    39.4-scoring, well-liked build as a corpse outscoring working ones.
    """
    ls = _loop_status()
    corpus = _corpus({("i1", ls.MODEL_A): [20.0]}, controls=[3.0, 4.0, 5.0])
    build_id = "i2__selfcheck"
    corpus.builds[build_id] = _build("i2", ls.MODEL_B, build_id)
    corpus.fleet_ids.add(build_id)
    corpus.runs.append(
        ScoredRun(
            crowd_dir="/tmp/sc",
            build_id=build_id,
            idea_id="i2",
            model=ls.MODEL_B,
            seed=0,
            n_agents=30,
            arch_version=CROWD_ARCH_VERSION,
            gate=0.6,
            dead=False,
            score=39.4,
        )
    )
    sep = control_separation(corpus)
    assert sep.dead_cells == {}
    # Named from the fleet's own arm A rather than a literal, so this reads the
    # same whichever pair of models the current fleet happens to compare.
    assert sep.weakest_real_cell == f"i1[{ls.MODEL_A}]"
    _, gate = ls.control_section(corpus)
    assert gate.passed


def test_profile_comparison_line_is_present_in_the_signal_section(monkeypatch):
    """A gap that exists only under one weighting is a weighting, not a gap.

    The report has to show the alternatives rather than leaving it to whoever
    reads it to go and check.
    """
    ls = _loop_status()
    spec = {}
    for i in range(4):
        spec[(f"i{i}", ls.MODEL_A)] = [60.0]
        spec[(f"i{i}", ls.MODEL_B)] = [40.0]
    corpus = _corpus(spec)
    calls = []

    def fake_load(_repo, weights=None, **kw):
        calls.append(getattr(weights, "profile", None))
        return corpus

    monkeypatch.setattr(ls, "load_corpus", fake_load)
    lines, _ = ls.signal_section(corpus)
    text = "\n".join(lines)
    assert "same gap under other profiles" in text
    assert "v2_hybrid" in text and "v1_deterministic" in text
    assert len(calls) >= 3


# --------------------------------------------------------------------------- #
# Fleet replicates (build-to-build variance)
# --------------------------------------------------------------------------- #


def _build_fleet():
    spec = importlib.util.spec_from_file_location(
        "build_fleet", REPO / "scripts" / "build_fleet.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_fleet"] = module
    spec.loader.exec_module(module)
    return module


def test_replicate_one_keeps_its_original_key():
    """The existing 50-build fleet must never be rebuilt by adding replicates."""
    bf = _build_fleet()
    assert bf.Cell("quick_notes_app", "m").key == "quick_notes_app::m"
    assert bf.Cell("quick_notes_app", "m", 1).key == "quick_notes_app::m"
    assert bf.Cell("quick_notes_app", "m", 2).key == "quick_notes_app::m::r2"


def test_short_model_matches_a_cell_however_the_model_was_named():
    """Cell and record must compare equal across every prefix this fleet has seen.

    A mismatch here does not fail loudly -- it makes ``cell_ok`` return False for
    a cell that is already built, so a resume silently pays to rebuild it. Three
    cases have to survive: the legacy ``google/`` prefix (11 existing builds
    predate the move to ``google-vertex``), the current ``google-vertex/``, and a
    partner model whose id itself carries an ``@`` version suffix.
    """
    bf = _build_fleet()
    for cell_name, record_name in [
        ("gemini-flash-test", "google/gemini-flash-test"),
        ("gemini-test", "google-vertex/gemini-test"),
        ("claude-test@default", "google-vertex-anthropic/claude-test@default"),
        # Both sides fully qualified, and both sides bare, must also agree.
        (
            "google-vertex-anthropic/claude-test@default",
            "google-vertex-anthropic/claude-test@default",
        ),
        ("gemini-test", "gemini-test"),
    ]:
        assert bf._short_model(cell_name) == bf._short_model(record_name), cell_name
    # ...but genuinely different models must still not collide.
    assert bf._short_model("google-vertex/gemini-test") != bf._short_model(
        "google-vertex-anthropic/claude-test@default"
    )


def test_a_second_replicate_is_pending_even_when_the_first_is_built():
    """Replicate 2 re-builds the same brief; it is a second draw, not a cache hit."""
    bf = _build_fleet()
    fleet = {"entries": {}}
    models = [bf.MODEL_A]
    first = bf.pending_cells(fleet, models, replicate=1)
    second = bf.pending_cells(fleet, models, replicate=2)
    assert len(first) == len(second) == 25
    assert {c.replicate for c in first} == {1}
    assert {c.replicate for c in second} == {2}
    assert not {c.key for c in first} & {c.key for c in second}


def test_status_counts_the_queue_the_same_retry_would_actually_build():
    """--status must answer for the --retry it was given, or it answers nothing.

    ``report`` used to call ``pending_cells`` with no retry set, so previewing
    the r4 team rebuild printed 236 while the build it previewed would run 250 --
    the 14 ``manifest_*`` cells that only a named retry reaches. An operator
    cannot confirm a queue before committing hours to it if the preview counts a
    different queue, and a parallel-arm driver derives its stop condition from
    this very line, so it would call an arm COMPLETE with those cells unbuilt.
    """
    bf = _build_fleet()
    ideas = sorted(i.idea_id for i in bf.load_ideas())
    models = [bf.MODEL_A]
    # One cell already failed with a status only an explicit --retry reaches.
    stale = bf.Cell(ideas[0], bf.MODEL_A, 3, "team")
    fleet = {"entries": {stale.key: {"status": "manifest_missing"}}}

    plain = bf.report(fleet, models, 3, ("team",))
    named = bf.report(fleet, models, 3, ("team",), frozenset({"manifest_missing"}))

    assert f"pending (to build): {len(ideas) - 1}" in plain
    assert f"pending (to build, incl. --retry manifest_missing): {len(ideas)}" in named
    # And the number quoted is the number that would be built.
    assert len(
        bf.pending_cells(
            fleet,
            models,
            frozenset({"manifest_missing"}),
            replicate=3,
            structures=("team",),
        )
    ) == len(ideas)


def test_a_whole_arm_rebuild_gives_each_stale_cell_one_fresh_draw_not_two():
    """--retry-before is what keeps a restarted rebuild from becoming best-of-N.

    Rebuilding an arm end to end may legitimately re-attempt a
    model-attributable status like ``manifest_missing``, because every cell gets
    exactly one new draw. Restart the driver, though, and an unbounded retry
    re-offers the cells whose fresh draw already landed and re-failed -- so the
    models that fail this way collect extra draws and the models that succeed
    collect none. Bounding the retry to results recorded before the rebuild
    began makes the redraw once-per-cell however many times the driver restarts.
    """
    bf = _build_fleet()
    ideas = sorted(i.idea_id for i in bf.load_ideas())
    models = [bf.MODEL_A]
    started = "2026-08-29T05:42:36+00:00"
    stale = bf.Cell(ideas[0], bf.MODEL_A, 3, "team")  # never redrawn
    redrawn = bf.Cell(ideas[1], bf.MODEL_A, 3, "team")  # drew, and failed again
    fleet = {
        "entries": {
            stale.key: {
                "status": "manifest_missing",
                "finished_at": "2026-08-26T16:52:27+00:00",
            },
            redrawn.key: {
                "status": "manifest_missing",
                "finished_at": "2026-08-29T07:06:29+00:00",
            },
        }
    }
    kw = dict(replicate=3, structures=("team",))
    retry = frozenset({"manifest_missing"})

    unbounded = {c.key for c in bf.pending_cells(fleet, models, retry, **kw)}
    bounded = {
        c.key
        for c in bf.pending_cells(fleet, models, retry, retry_before=started, **kw)
    }

    assert stale.key in unbounded and redrawn.key in unbounded
    assert stale.key in bounded, "a cell that never drew still owes one draw"
    assert redrawn.key not in bounded, "its one fresh draw already landed"
    # Re-running the bounded plan is idempotent, which is the property that
    # makes a restart safe rather than merely unlikely to hurt.
    assert bounded == {
        c.key
        for c in bf.pending_cells(fleet, models, retry, retry_before=started, **kw)
    }


def test_our_own_failures_are_retried_regardless_of_the_redraw_cutoff():
    """``--retry-before`` must not strand a harness failure.

    The always-ours statuses hold no evidence about any model, so re-running one
    hands nobody an extra draw. If the cutoff bounded them too, a harness crash
    during the rebuild would be frozen in as that cell's result.
    """
    bf = _build_fleet()
    ideas = sorted(i.idea_id for i in bf.load_ideas())
    cell = bf.Cell(ideas[0], bf.MODEL_A, 3, "team")
    fleet = {
        "entries": {
            cell.key: {
                "status": "harness_timeout",
                "finished_at": "2026-08-29T07:06:29+00:00",  # after the cutoff
            }
        }
    }
    pending = bf.pending_cells(
        fleet,
        [bf.MODEL_A],
        frozenset(),
        replicate=3,
        structures=("team",),
        retry_before="2026-08-29T05:42:36+00:00",
    )
    assert cell.key in {c.key for c in pending}


def test_two_arms_writing_disjoint_cells_do_not_revert_each_other(
    tmp_path, monkeypatch
):
    """Concurrent build_fleet processes must both survive in the index.

    THE BUG. ``persist`` re-read the index under an exclusive lock and then
    layered the writer's whole STARTUP SNAPSHOT of the index over it. The lock
    made the write atomic; it did nothing about the payload being stale, so
    every write faithfully restored the peer arm's keys to their startup values.
    Caught by polling two keys every 10s while two arms ran together: one
    persist put ``form_builder[.../team]`` back by days while restoring
    ``group_scheduling_poll[.../solo]`` to today, and the next swapped which one
    was current. Neither process could see it -- each only ever read back its own
    writes.

    So this test asserts the property the lock never gave: a writer must not
    revert a key it does not own, even when its own snapshot disagrees.
    """
    import threading

    bf = _build_fleet()
    fleet_path = tmp_path / "fleet.json"
    monkeypatch.setattr(bf, "FLEET_PATH", fleet_path)
    monkeypatch.setattr(bf, "FLEET_LOCK_PATH", tmp_path / ".fleet.lock")

    # Both arms start from the same stale snapshot, as two real processes do.
    stale = {
        "entries": {
            "idea::m::r3": {"status": "ok", "finished_at": "2026-08-22T00:00:00"},
            "idea::m::team::r3": {"status": "ok", "finished_at": "2026-08-23T00:00:00"},
        }
    }
    fleet_path.write_text(json.dumps(stale), encoding="utf-8")

    def arm(key: str, stamp: str) -> None:
        for _ in range(40):
            bf.merge_fleet_entries({key: {"status": "ok", "finished_at": stamp}})

    threads = [
        threading.Thread(target=arm, args=("idea::m::r3", "2026-08-29T18:00:00")),
        threading.Thread(target=arm, args=("idea::m::team::r3", "2026-08-29T18:00:01")),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    final = json.loads(fleet_path.read_text())["entries"]
    # Both of today's results are present AT THE SAME TIME. Under the old code
    # whichever arm wrote last reverted the other back to its startup value.
    assert final["idea::m::r3"]["finished_at"] == "2026-08-29T18:00:00"
    assert final["idea::m::team::r3"]["finished_at"] == "2026-08-29T18:00:01"


def test_merging_never_writes_a_key_the_caller_did_not_pass():
    """The signature is the fix: there is nothing to write that is not owned.

    Guards against a future refactor handing the merge a whole fleet dict again,
    which is precisely how the lost update got in.
    """
    bf = _build_fleet()
    import inspect

    params = inspect.signature(bf.merge_fleet_entries).parameters
    assert "entries" in params, "takes the entries to write, not a fleet snapshot"
    assert "fleet" not in params


def test_launches_are_spaced_so_a_shared_sqlite_lock_cannot_decide_a_build():
    """Ten opencode processes starting at once contend on their shared state DB.

    One loses with "database is locked" four seconds in and is recorded as a
    harness failure on whichever cell happened to be unlucky -- a coin flip
    deciding a model's build outcome.
    """
    import time as _time

    bf = _build_fleet()
    bf._START_GAP_S = 0.05
    bf._last_start = 0.0
    started = _time.monotonic()
    for _ in range(4):
        bf._stagger_start()
    assert _time.monotonic() - started >= 0.15


def test_build_counts_are_per_build_not_per_idea_across_replicates():
    """Two replicates is 50 builds over 25 ideas, not "ok 50/25".

    The gate counts distinct IDEAS attempted; the report counts BUILDS. Dividing
    one by the other printed a denominator that could not be right.
    """
    ls = _loop_status()
    corpus = _full_corpus(ls)
    # Add a second replicate of every cell: same ideas, new build ids.
    for (idea, model, _structure), runs in list(corpus.cells().items()):
        bid = f"{idea}__{model}__r2"
        corpus.builds[bid] = FleetBuild(
            build_id=bid,
            idea_id=idea,
            model=model,
            status="ok",
            config={"n_agents": 4},
        )
        corpus.fleet_ids.add(bid)
        corpus.runs.append(
            ScoredRun(
                crowd_dir=f"/tmp/{bid}",
                build_id=bid,
                idea_id=idea,
                model=model,
                seed=0,
                n_agents=30,
                arch_version=CROWD_ARCH_VERSION,
                score=runs[0].score,
            )
        )
    lines, gate, info = ls.fleet_section(corpus)
    text = "\n".join(lines)
    assert "ok 50/50 builds (2 replicate(s) x 25 ideas)" in text
    assert "50/25" not in text
    assert gate.passed  # 25 distinct ideas attempted, not 50
    assert len(info["paired"]) == 25


# --------------------------------------------------------------------------- #
# Replicate statistics
# --------------------------------------------------------------------------- #


def _replicated(spec: dict[tuple[str, str, int], list[float]]) -> FleetCorpus:
    """Corpus from {(idea, model, replicate): [scores per seed]}."""
    builds: dict[str, FleetBuild] = {}
    runs: list[ScoredRun] = []
    for (idea, model, rep), scores in spec.items():
        bid = f"{idea}__{model}__r{rep}"
        builds[bid] = FleetBuild(
            build_id=bid, idea_id=idea, model=model, status="ok", replicate=rep
        )
        for seed, score in enumerate(scores):
            runs.append(
                ScoredRun(
                    crowd_dir=f"/tmp/{bid}_{seed}",
                    build_id=bid,
                    idea_id=idea,
                    model=model,
                    seed=seed,
                    n_agents=30,
                    arch_version=CROWD_ARCH_VERSION,
                    replicate=rep,
                    score=score,
                )
            )
    return FleetCorpus(builds=builds, runs=runs, fleet_ids=set(builds))


def test_restrict_gives_one_replicate_and_reuses_every_statistic():
    corpus = _replicated(
        {
            ("i1", A, 1): [60.0],
            ("i1", B, 1): [40.0],
            ("i1", A, 2): [50.0],
            ("i1", B, 2): [45.0],
        }
    )
    assert corpus.replicates() == [1, 2]
    assert paired_gap(corpus.restrict(replicate=1), A, B).gap == pytest.approx(20.0)
    assert paired_gap(corpus.restrict(replicate=2), A, B).gap == pytest.approx(5.0)
    # Pooled averages the runs, not the per-replicate gaps.
    assert paired_gap(corpus, A, B).gap == pytest.approx(12.5)


def test_variance_components_separates_crowd_noise_from_build_noise():
    """Build noise is only visible once a brief has been built twice."""
    from viral_bench.score.fleet import variance_components

    # Seeds agree perfectly within a build; the two builds differ by 20 points.
    corpus = _replicated(
        {
            **{(f"i{i}", A, 1): [40.0, 40.0, 40.0] for i in range(6)},
            **{(f"i{i}", A, 2): [60.0, 60.0, 60.0] for i in range(6)},
        }
    )
    v = variance_components(corpus)
    assert v.crowd_sd == pytest.approx(0.0)
    # variance of [40, 60] is 200 -> sd sqrt(200) ~ 14.14
    assert v.build_sd == pytest.approx(200**0.5, abs=0.05)
    assert v.n_cells == 6
    assert v.median_abs_diff == pytest.approx(20.0)
    assert v.ratio is None  # crowd_sd is zero: the ratio is undefined, not huge


def test_variance_components_finds_no_build_noise_when_builds_agree():
    from viral_bench.score.fleet import variance_components

    corpus = _replicated(
        {
            **{(f"i{i}", A, 1): [45.0, 55.0] for i in range(6)},
            **{(f"i{i}", A, 2): [45.0, 55.0] for i in range(6)},
        }
    )
    v = variance_components(corpus)
    assert v.crowd_sd == pytest.approx(50**0.5, abs=0.05)
    assert v.build_sd == pytest.approx(0.0, abs=0.01)


def test_variance_components_needs_a_second_replicate():
    from viral_bench.score.fleet import variance_components

    single = _replicated({("i1", A, 1): [40.0, 42.0], ("i2", A, 1): [50.0, 52.0]})
    v = variance_components(single)
    assert v.crowd_sd is not None
    assert v.build_sd is None  # cannot be estimated, and is not guessed


def test_winner_flips_reports_briefs_that_changed_sides():
    from viral_bench.score.fleet import winner_flips

    corpus = _replicated(
        {
            ("steady", A, 1): [60.0],
            ("steady", B, 1): [40.0],
            ("steady", A, 2): [58.0],
            ("steady", B, 2): [41.0],
            ("flipper", A, 1): [40.0],
            ("flipper", B, 1): [50.0],
            ("flipper", A, 2): [70.0],
            ("flipper", B, 2): [45.0],
        }
    )
    flips = winner_flips(corpus, A, B)
    assert [f.idea_id for f in flips] == ["flipper"]
    assert flips[0].per_replicate == {1: -10.0, 2: 25.0}
    assert flips[0].swing == pytest.approx(35.0)


def test_winner_flips_is_empty_without_replicates():
    from viral_bench.score.fleet import winner_flips

    corpus = _replicated({("i1", A, 1): [60.0], ("i1", B, 1): [40.0]})
    assert winner_flips(corpus, A, B) == []


def test_wilson_interval_is_sane_at_zero_failures():
    """The normal approximation gives a zero-width interval at 0/50, which lies."""
    from viral_bench.score.fleet import wilson_interval

    lo, hi = wilson_interval(0, 50)
    assert lo == 0.0
    assert 0.0 < hi < 0.10  # "we saw none" is not "it never happens"
    lo, hi = wilson_interval(8, 50)
    assert lo < 8 / 50 < hi
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_delivery_stats_counts_manifest_failures_per_replicate():
    from viral_bench.score.fleet import delivery_stats

    corpus = _replicated({("i1", B, 1): [10.0], ("i1", B, 2): [20.0]})
    corpus.builds["i1__model-b__r1"] = FleetBuild(
        build_id="i1__model-b__r1",
        idea_id="i1",
        model=B,
        status="manifest_invalid",
        replicate=1,
    )
    d = delivery_stats(corpus, B)
    assert d.attempted == 2
    assert d.manifest_failures == 1
    assert d.per_replicate == {1: (1, 1), 2: (0, 1)}
    assert d.failure_rate == pytest.approx(0.5)


def test_harness_failures_are_not_counted_as_the_models_fault():
    """A timeout is ours; a malformed manifest is the model's. Only one counts."""
    from viral_bench.score.fleet import delivery_stats

    corpus = _replicated({("i1", B, 1): [10.0]})
    corpus.builds["i1__model-b__r1"] = FleetBuild(
        build_id="i1__model-b__r1",
        idea_id="i1",
        model=B,
        status="harness_failed",
        replicate=1,
    )
    d = delivery_stats(corpus, B)
    assert d.manifest_failures == 0
    assert d.shipped == 0  # still not a shipped build


def _synthetic_repo(root: pathlib.Path) -> pathlib.Path:
    """A self-contained repo with two models x two replicates of two ideas.

    The CLI test used to run against the DEVELOPER'S REAL builds/ directory, so
    whether it passed depended on which sweeps happened to be on disk. Bumping
    CROWD_ARCH_VERSION to 9 broke it, correctly: the live corpus no longer held
    any second-model runs at the current arch, so there was nothing to compare
    and the CLI exited 1. That is the CLI behaving properly and a test asserting
    on someone's working directory.
    """
    builds = root / "builds"
    entries = {}
    for idea in ("i1", "i2"):
        for model in (A, B):
            for rep in (1, 2):
                bid = f"{idea}__{model}__r{rep}"
                _write_build(builds, bid, idea_id=idea, model=model)
                entries[f"{idea}::{model}::r{rep}"] = {
                    "build_id": bid,
                    "replicate": rep,
                }
                for seed in (0, 1):
                    _write_run(builds, bid, seed, would_use=(model == A))
    (builds / "fleet.json").write_text(
        json.dumps({"entries": entries}), encoding="utf-8"
    )
    return root


def test_replicate_analysis_cli_runs_end_to_end(capsys, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "replicate_analysis", REPO / "scripts" / "replicate_analysis.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["replicate_analysis"] = module
    spec.loader.exec_module(module)
    repo = str(_synthetic_repo(tmp_path))

    # A model name that is not in the corpus must be reported, not crash.
    rc = module.main(
        ["--all-eras", "--repo", repo, "--model-a", "no-such-model", "--model-b", B]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "no builds in this corpus for: no-such-model" in out
    assert "models present:" in out

    # And the real pair produces the full report.
    # --all-eras: this synthetic corpus predates brief fingerprinting.
    rc = module.main(["--all-eras", "--repo", repo, "--model-a", A, "--model-b", B])
    out = capsys.readouterr().out
    assert rc == 0
    assert "MODEL GAP" in out


def test_a_positive_control_never_enters_the_broken_control_floor(tmp_path):
    """A gate that gets EASIER when you add a good app to the corpus is not a gate.

    Both controls carry a ``control/`` model prefix, so before this the working
    full-stack control would have been pooled into ``control_runs()`` and its
    high score would have lifted the very floor G3 measures against.
    """
    root = tmp_path / "builds"
    _write_build(root, "bad", model="control/broken", idea_id="quick_notes_app")
    _write_build(root, "good", model="control/fullstack-ok", idea_id="quick_notes_app")
    _write_run(root, "bad", 0, would_use=False)
    _write_run(root, "good", 0, would_use=True)
    (root / "fleet.json").write_text(json.dumps({"entries": {}}), encoding="utf-8")

    corpus = load_corpus(tmp_path)
    assert {r.build_id for r in corpus.control_runs()} == {"bad"}
    assert {r.build_id for r in corpus.control_runs(kind="positive")} == {"good"}
    assert corpus.builds["good"].is_control is True


# --------------------------------------------------------------------------- #
# The dynamic founder arm
# --------------------------------------------------------------------------- #


def test_structure_name_recognises_the_dynamic_arm():
    """It must NOT derive from (n_agents, collab) like the other two.

    A dynamic build runs one orchestrator process, so deriving would name it
    "solo" and pool it with the solo baseline -- two different experiments in
    one cell, which is exactly what this function exists to prevent.
    """
    from viral_bench.score.fleet import structure_name

    dynamic = {"structure": "dynamic", "n_agents": 1, "collab": "local"}
    assert structure_name(dynamic) == "dynamic"
    assert (
        structure_name({"structure": "solo", "n_agents": 1, "collab": "local"})
        == "solo"
    )
    assert (
        structure_name({"structure": "team", "n_agents": 4, "collab": "local"})
        == "team"
    )


def test_dynamic_cells_get_their_own_fleet_key():
    """A dynamic build of an idea must never be counted as its solo/team build."""
    bf = _build_fleet()
    assert "dynamic" in bf.STRUCTURES
    key = bf.Cell("quick_notes_app", "m", 1, "dynamic").key
    assert key == "quick_notes_app::m::dynamic"
    assert key != bf.Cell("quick_notes_app", "m", 1, "team").key


def test_dynamic_config_match_ignores_outcomes():
    """Team size and turn count are the MODEL's choices in this arm.

    Verifying them the way the fixed arms are verified would reject every build
    whose founder finished early or delegated at all -- i.e. reject the arm for
    working as designed. What is fixed, and so what is checked, is the structure,
    the toolset and the cap it was given.
    """
    bf = _build_fleet()
    record = {
        "structure": "dynamic",
        "collab": "local",
        "max_turns": bf.STRUCTURES["dynamic"]["turns"],
        "n_agents": 1,
        "rounds_run": 1,
        "subagents_spawned": 7,
    }
    assert bf.config_matches(record, "dynamic") is True
    # A build that ran under a different turn cap is a different experiment.
    assert bf.config_matches({**record, "max_turns": 99}, "dynamic") is False
    # A solo build must not satisfy the dynamic arm just by being one agent.
    assert bf.config_matches({**record, "structure": "solo"}, "dynamic") is False


def test_dynamic_cell_passes_turns_not_rounds():
    """--rounds/--min-rounds are team knobs; the CLI rejects them here."""
    bf = _build_fleet()
    cfg = bf.STRUCTURES["dynamic"]
    assert cfg["agents"] == "dynamic"
    assert "rounds" not in cfg and "min_rounds" not in cfg


def test_dynamic_arm_has_its_own_wall_clock():
    """The 90-minute fleet default kills a fanned-out Opus 5 build mid-work.

    Measured: one dynamic orchestrator turn ran ~95 minutes while driving a
    five-subagent team the model designed itself. Capping that at the default
    records harness_timeout on the model that orchestrated hardest.
    """
    bf = _build_fleet()
    assert bf.STRUCTURES["dynamic"]["timeout_s"] > 90 * 60
    # Solo is one model turn twice over -- it keeps the shared default.
    assert "timeout_s" not in bf.STRUCTURES["solo"]
    # The team arm has since earned its own cap too, so asserting it has none
    # (as this test originally did) went stale the moment that landed. The
    # invariant that actually matters is the ORDERING: a dynamic turn can hold a
    # whole fan-out AND the brief now invites a second turn, so its cap has to
    # stay above the fixed team arm, not merely above the 90-minute default.
    assert bf.STRUCTURES["dynamic"]["timeout_s"] > bf.STRUCTURES["team"]["timeout_s"]


# --------------------------------------------------------------------------- #
# Coverage, at the grain that can see a hole
# --------------------------------------------------------------------------- #


def test_coverage_is_counted_in_builds_not_models():
    """A model with runs in every arm can still be missing whole builds.

    THIS IS THE BUG. The seed-0 pass reported "complete, all 10 models x 4
    pipelines" and was correct at that grain while 26 of 1,000 builds had never
    scored at all: a model's mean is taken over the runs that exist, so a missing
    build silently leaves the denominator rather than showing up as a gap. Only a
    build-level count can see it.
    """
    corpus = _corpus({("i1", A): [50.0], ("i2", A): [], ("i3", A): [60.0, 61.0]})
    cov = corpus.build_coverage()

    assert cov.builds == 3
    assert cov.covered == 2
    assert not cov.complete
    assert cov.uncovered_ids == ["i2__model-a"], "a gap must be nameable"
    assert cov.at_least(2) == 1
    assert cov.seed_histogram == {0: 1, 1: 1, 2: 1}


def test_coverage_is_complete_only_when_every_build_scored():
    corpus = _corpus({("i1", A): [50.0], ("i2", A): [55.0]})
    cov = corpus.build_coverage()
    assert cov.complete and cov.fraction == 1.0 and cov.uncovered_ids == []


def test_an_unstartable_app_is_excluded_from_scoring_and_reported():
    """Our packaging bug must not be scored as the model's failure.

    A build whose app the harness could not start produced no evidence about the
    model at all -- one such app, served by hand, returned HTTP 200 in 55 ms and
    rendered 1,216 DOM nodes. It leaves the mean and is reported as a coverage
    caveat, the same treatment ``harness_failed`` gets on the build side.
    """
    corpus = _corpus({("i1", A): [50.0], ("i2", A): [12.0]})
    broken = next(r for r in corpus.runs if r.build_id == "i2__model-a")
    corpus.runs = [r for r in corpus.runs if r is not broken] + [
        replace(broken, app_start_failed=True)
    ]

    assert [r.build_id for r in corpus.fleet_runs()] == ["i1__model-a"]
    assert [r.build_id for r in corpus.app_start_failures()] == ["i2__model-a"]

    cov = corpus.build_coverage()
    assert cov.covered == 1 and cov.builds == 2
    assert cov.app_start_failed == 1
    assert "app_start_failed" in cov.summary()


# --------------------------------------------------------------------------- #
# A refusal is excluded; a missing manifest is floored
# --------------------------------------------------------------------------- #


def test_a_missing_manifest_is_floored_not_excluded():
    """The model ran its turns and shipped no launch contract. That IS a result.

    16 of the solo arm's cells are exactly this, concentrated in the weakest
    models -- every one spent its full turn budget, the slowest took 440s against
    a 5,400s cap so no wall clock bit, and several wrote zero files. Excluding
    them would delete the clearest capability signal the benchmark has.
    """
    build = _build("i1", A, "b1", status="manifest_missing")
    assert not build.unscorable


def test_a_provider_refusal_is_excluded_not_floored():
    """A safety filter blocked the output, so there is no evidence either way.

    Flooring it would confound provider POLICY with model CAPABILITY, and
    asymmetrically: whichever provider filters hardest would lose the most score.
    """
    assert _build("i1", A, "b1", status="provider_refusal").unscorable


def test_refused_builds_leave_the_denominator_and_are_named():
    corpus = _corpus({("i1", A): [50.0], ("i2", A): [40.0]})
    corpus.builds["i2__model-a"] = replace(
        corpus.builds["i2__model-a"], status="provider_refusal"
    )

    # Its runs are excluded even though they exist and scored.
    assert [r.build_id for r in corpus.fleet_runs()] == ["i1__model-a"]
    assert [b.build_id for b in corpus.refusals()] == ["i2__model-a"]

    cov = corpus.build_coverage()
    # Out of the denominator: one build remains, and the arm can read complete.
    assert (cov.builds, cov.covered, cov.refused) == (1, 1, 1)
    assert cov.complete, "an exclusion is not a gap the sweep can ever close"
    assert "excluded (provider refusal)" in cov.summary()


def test_the_index_verdict_overrides_the_build_record(tmp_path):
    """A refusal is only recognisable once its error text is classified.

    build.json carries what the harness assigned while running; fleet.json carries
    the cell verdict, which --reclassify-infra can revise afterwards. Where they
    disagree the index is the later and better-informed of the two -- and without
    this overlay the scoring layer keeps reading the stale build.json and the
    exclusion silently never happens.
    """
    from viral_bench.score.fleet import apply_fleet_status

    work = tmp_path / "work" / "b1"
    work.mkdir(parents=True)
    (work / "build.json").write_text(
        json.dumps(
            {
                "build_id": "b1",
                "idea_id": "i1",
                "model": "m",
                "status": "harness_failed",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "fleet.json").write_text(
        json.dumps(
            {"entries": {"i1::m": {"build_id": "b1", "status": "provider_refusal"}}}
        ),
        encoding="utf-8",
    )

    builds = apply_fleet_status(load_builds(tmp_path), tmp_path)
    assert builds["b1"].status == "provider_refusal"
    assert builds["b1"].unscorable


def test_the_overlay_only_touches_unscorable_statuses(tmp_path):
    """It must not become a general way to rewrite an ordinary outcome."""
    from viral_bench.score.fleet import apply_fleet_status

    work = tmp_path / "work" / "b1"
    work.mkdir(parents=True)
    (work / "build.json").write_text(
        json.dumps({"build_id": "b1", "idea_id": "i1", "model": "m", "status": "ok"}),
        encoding="utf-8",
    )
    (tmp_path / "fleet.json").write_text(
        json.dumps(
            {"entries": {"i1::m": {"build_id": "b1", "status": "manifest_missing"}}}
        ),
        encoding="utf-8",
    )
    builds = apply_fleet_status(load_builds(tmp_path), tmp_path)
    assert builds["b1"].status == "ok", "only UNSCORABLE_STATUSES may be overlaid"


def test_an_app_that_cannot_start_through_its_own_fault_is_floored_not_dropped():
    """The crowd did everything right; nobody could use the thing. That is a result.

    Only a HARNESS fault is excluded -- there the number would describe our
    packaging rather than the model. An app whose own source has a syntax error,
    or whose manifest names a file never committed, stays in and takes the floor.
    """
    corpus = _corpus({("i1", A): [50.0], ("i2", A): [0.1], ("i3", A): [0.1]})
    corpus.runs = [
        replace(r, app_start_failed=True, app_start_fault="app")
        if r.build_id == "i2__model-a"
        else replace(r, app_start_failed=True, app_start_fault="harness")
        if r.build_id == "i3__model-a"
        else r
        for r in corpus.runs
    ]

    scored = {r.build_id for r in corpus.fleet_runs()}
    assert "i2__model-a" in scored, "the app's own failure is a floored result"
    assert "i3__model-a" not in scored, "our packaging failure is excluded"

    assert len(corpus.app_start_failures()) == 2, "both are reported"
    assert len(corpus.app_start_failures(fault="harness")) == 1

    # Only the EXCLUDED one is a coverage hole. The floored build has a score,
    # so it is covered like any other.
    cov = corpus.build_coverage()
    assert cov.app_start_failed == 1
    assert "i2__model-a" not in cov.uncovered_ids
    assert "i3__model-a" in cov.uncovered_ids


def test_a_run_recorded_before_attribution_existed_keeps_the_safe_treatment():
    """Absent fault means excluded, not promoted into the denominator.

    Runs written before app_start_fault existed carry no attribution. Defaulting
    them to "app" would quietly floor builds nobody has judged.
    """
    from viral_bench.score.fleet import ScoredRun

    old = ScoredRun(
        crowd_dir="/t",
        build_id="b",
        idea_id="i",
        model="m",
        seed=0,
        n_agents=30,
        score=0.1,
        app_start_failed=True,
    )
    assert old.app_start_fault == ""
    assert not old.scorable
