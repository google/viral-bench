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

"""Tests for the sweep driver's non-simulation logic.

The parts worth pinning down are the ones that decide what gets measured and
what silently does not: which runs count as already done, and whether the
autorater ran. Not one stored crowd run had an ``autorating.json``, so 15% of
the active scoring profile was being renormalised away in every number the
benchmark had ever reported, a gap no test existed to catch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from viral_bench.crowd.sim_defaults import CROWD_ARCH_VERSION

REPO = Path(__file__).resolve().parents[1]


def Cellish(cells) -> set[tuple[str, int]]:
    """Cells as plain (build_id, seed) pairs, for order-free comparison."""
    return {(c.build_id, c.seed) for c in cells}


def _sweep(tmp_path: Path):
    """Load crowd_sweep with its paths pointed at a scratch tree."""
    spec = importlib.util.spec_from_file_location(
        "crowd_sweep", REPO / "scripts" / "crowd_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["crowd_sweep"] = module
    spec.loader.exec_module(module)
    module.BUILDS = tmp_path / "builds"
    module.CROWD_DIR = module.BUILDS / "crowd"
    module.LOG_DIR = module.BUILDS / "sweep_logs"
    module.TIMEOUT_LEDGER = module.CROWD_DIR / ".timeouts.json"
    return module


def _run_dir(
    root: Path, name: str, *, ok: bool = True, arch: str | None = None
) -> Path:
    d = root / "crowd" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_summary.json").write_text(
        json.dumps(
            {
                "build_id": name.split("__crowd")[0],
                "crowd_arch_version": arch or CROWD_ARCH_VERSION,
                "ok": ok,
                "config": {"seed": 0},
            }
        ),
        encoding="utf-8",
    )
    return d


def test_only_healthy_current_arch_runs_of_wanted_builds_are_rated(tmp_path):
    sweep = _sweep(tmp_path)
    root = tmp_path / "builds"
    _run_dir(root, "b1__crowd-1")
    _run_dir(root, "b1__crowd-2", ok=False)
    _run_dir(root, "b2__crowd-1", arch="0")
    _run_dir(root, "other__crowd-1")
    found = {d.name for d in sweep.healthy_run_dirs({"b1", "b2"})}
    assert found == {"b1__crowd-1"}


def test_autorate_skips_runs_that_already_have_a_rating(tmp_path, monkeypatch):
    sweep = _sweep(tmp_path)
    root = tmp_path / "builds"
    already = _run_dir(root, "b1__crowd-1")
    (already / "autorating.json").write_text("{}", encoding="utf-8")
    fresh = _run_dir(root, "b1__crowd-2")

    import viral_bench.score.autorater as autorater
    import viral_bench.score.evidence as evidence

    rated: list[Path] = []

    class _Rating:
        ok = True

        def as_dict(self):
            return {"dimensions": {"substance": {"score": 7.0}}}

    monkeypatch.setattr(evidence, "build_evidence_pack", lambda d: rated.append(d) or d)
    monkeypatch.setattr(autorater, "rate_pack", lambda pack: _Rating())

    ok, done = sweep.autorate_missing([already, fresh], concurrency=2)
    assert (ok, done) == (1, 1)
    assert rated == [fresh]
    assert json.loads((fresh / "autorating.json").read_text())["dimensions"]


def test_a_rating_that_fails_is_reported_not_written(tmp_path, monkeypatch):
    sweep = _sweep(tmp_path)
    root = tmp_path / "builds"
    run = _run_dir(root, "b1__crowd-1")

    import viral_bench.score.autorater as autorater
    import viral_bench.score.evidence as evidence

    monkeypatch.setattr(evidence, "build_evidence_pack", lambda d: d)

    def _boom(pack):
        raise RuntimeError("no api key")

    monkeypatch.setattr(autorater, "rate_pack", _boom)
    ok, done = sweep.autorate_missing([run], concurrency=1)
    assert (ok, done) == (0, 1)
    assert not (run / "autorating.json").exists()


def test_a_timed_out_cell_is_recorded_and_counted(tmp_path):
    sweep = _sweep(tmp_path)
    cell = sweep.Cell("b1", 2)
    assert sweep.timeouts_seen() == {}
    sweep.record_timeout(cell)
    sweep.record_timeout(cell)
    assert sweep.timeouts_seen()[("b1", 2)] == 2


def test_the_ledger_is_scoped_to_one_architecture(tmp_path):
    """A cell that timed out under the old crowd deserves a fresh attempt.

    Bumping CROWD_ARCH_VERSION retires the measurements, and it has to retire
    the give-ups with them -- otherwise a build written off by a crowd that no
    longer exists is never measured again.
    """
    sweep = _sweep(tmp_path)
    sweep.record_timeout(sweep.Cell("b1", 0))
    ledger = json.loads(sweep.TIMEOUT_LEDGER.read_text())
    assert list(ledger) == [CROWD_ARCH_VERSION]

    ledger["previous-arch"] = {"b2::0": 9}
    sweep.TIMEOUT_LEDGER.write_text(json.dumps(ledger), encoding="utf-8")
    seen = sweep.timeouts_seen()
    assert ("b1", 0) in seen
    assert ("b2", 0) not in seen


def test_a_cell_at_the_attempt_limit_leaves_the_queue(tmp_path):
    sweep = _sweep(tmp_path)
    todo, skipped = sweep.plan_cells(
        ["hangs", "fine"],
        seeds=2,
        have={},
        timed_out={("hangs", 0): 1},
        max_timeout_attempts=1,
    )
    assert Cellish(skipped) == {("hangs", 0)}
    # Only the cell that timed out is dropped: seed 1 of the same
    # build is a different measurement and stays queued, because a build that
    # exceeds the cap at one seed routinely completes at another.
    assert Cellish(todo) == {("hangs", 1), ("fine", 0), ("fine", 1)}


def test_the_attempt_limit_can_be_disabled(tmp_path):
    sweep = _sweep(tmp_path)
    todo, skipped = sweep.plan_cells(
        ["hangs"],
        seeds=1,
        have={},
        timed_out={("hangs", 0): 99},
        max_timeout_attempts=0,
    )
    assert skipped == []
    assert Cellish(todo) == {("hangs", 0)}


def test_a_cell_already_measured_is_never_queued_or_skipped(tmp_path):
    """Coverage beats the ledger: a healthy run means the cell is done."""
    sweep = _sweep(tmp_path)
    todo, skipped = sweep.plan_cells(
        ["b1"],
        seeds=1,
        have={("b1", 0): [tmp_path]},
        timed_out={("b1", 0): 5},
        max_timeout_attempts=1,
    )
    assert (todo, skipped) == ([], [])


def test_only_a_wall_clock_kill_reaches_the_ledger(tmp_path, monkeypatch):
    """rc=1 and friends are the transient failures the retry pass recovers.

    Recording them here would make one bad minute permanent, which is the
    opposite of what this ledger is for.
    """
    import subprocess

    sweep = _sweep(tmp_path)
    sweep.LOG_DIR.mkdir(parents=True, exist_ok=True)

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "transient boom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Proc())
    opts = argparse.Namespace(agents=30, triers=-1, rounds=3, timeout=1.0)
    res = sweep.run_cell(sweep.Cell("b1", 0), opts)

    assert res["rc"] == 1
    assert sweep.timeouts_seen() == {}


def test_a_wall_clock_kill_does_reach_the_ledger(tmp_path, monkeypatch):
    import subprocess

    sweep = _sweep(tmp_path)
    sweep.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="viral-bench", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", _timeout)
    opts = argparse.Namespace(agents=30, triers=-1, rounds=3, timeout=1.0)
    res = sweep.run_cell(sweep.Cell("b1", 0), opts)

    assert res["rc"] == 124
    assert sweep.timeouts_seen()[("b1", 0)] == 1
