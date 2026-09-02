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

"""Tests for the 3.12-side crowd launcher (subprocess mocked; oasis-free)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from viral_bench.crowd import launch


def test_crowd_python_path() -> None:
    assert launch.crowd_python().as_posix().endswith(".venv-crowd/bin/python")


def test_run_crowd_sim_errors_without_env(monkeypatch) -> None:
    monkeypatch.setattr(launch, "crowd_env_ready", lambda: False)
    with pytest.raises(launch.CrowdLaunchError, match="crowd env"):
        launch.run_crowd_sim("some_build")


def test_default_out_dir_shape(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(launch, "builds_root", lambda: tmp_path)
    out = launch._default_out_dir("idea__ts__abc")
    assert out.parent == tmp_path / "crowd"
    assert out.name.startswith("idea__ts__abc__crowd-")


def test_run_crowd_sim_builds_argv_and_reads_summary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(launch, "crowd_env_ready", lambda: True)
    monkeypatch.setattr(launch, "load_build_record", lambda bid: object())

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env", {})
        # Emulate the runner writing its summary artifact.
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_summary.json").write_text(
            json.dumps({"ok": True, "engagement": {"likes": 3}}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)

    result = launch.run_crowd_sim(
        "build-1",
        agents=12,
        triers=6,
        rounds=3,
        model="gemini-x",
        recsys="twitter",
        seed=7,
        container=False,
        no_llm=True,
        interview=False,
        out_dir=tmp_path / "out",
    )

    argv = captured["argv"]
    assert argv[1:3] == ["-m", "viral_bench.crowd.sim.runner"]
    # flags threaded through
    for flag, val in [
        ("--build-id", "build-1"),
        ("--agents", "12"),
        ("--triers", "6"),
        ("--rounds", "3"),
        ("--model", "gemini-x"),
        ("--recsys", "twitter"),
        ("--seed", "7"),
    ]:
        assert flag in argv and argv[argv.index(flag) + 1] == val
    assert "--host" in argv  # container=False
    assert "--no-llm" in argv
    assert "--no-interview" in argv
    # PYTHONPATH points at repo src so the runner imports viral_bench
    assert captured["env"]["PYTHONPATH"].split(":")[0].endswith("/src")

    assert result.ok is True
    assert result.summary["engagement"]["likes"] == 3
    assert result.db_path.endswith("simulation.db")


def test_run_crowd_sim_reports_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(launch, "crowd_env_ready", lambda: True)
    monkeypatch.setattr(launch, "load_build_record", lambda bid: object())

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, "", "boom failed")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    result = launch.run_crowd_sim("build-1", out_dir=tmp_path / "out", stream=False)
    assert result.ok is False
    assert result.returncode == 1
    assert "boom failed" in result.stderr_tail
