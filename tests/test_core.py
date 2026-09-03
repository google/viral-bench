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

"""Tests for viral_bench.core.

These double as examples for the intern: each test is small and shows how to
exercise one behavior.
"""

from __future__ import annotations

from viral_bench.core import Task, exact_match, run_benchmark


def test_exact_match_is_case_insensitive() -> None:
    assert exact_match("Paris", "paris")
    assert exact_match("  4 ", "4")
    assert not exact_match("London", "Paris")


def test_run_benchmark_all_pass() -> None:
    tasks = [
        Task(task_id="t1", prompt="2+2", expected="4"),
        Task(task_id="t2", prompt="3+3", expected="6"),
    ]
    answers = {"2+2": "4", "3+3": "6"}

    result = run_benchmark(lambda p: answers[p], tasks)

    assert result.total == 2
    assert result.passed == 2
    assert result.score == 1.0
    assert result.failures == []


def test_run_benchmark_records_failures() -> None:
    tasks = [
        Task(task_id="good", prompt="2+2", expected="4"),
        Task(task_id="bad", prompt="3+3", expected="6"),
    ]
    # Model always answers "4", so only the first task passes.
    result = run_benchmark(lambda p: "4", tasks)

    assert result.passed == 1
    assert result.failures == ["bad"]
    assert result.score == 0.5


def test_empty_benchmark_scores_zero() -> None:
    result = run_benchmark(lambda p: "anything", [])
    assert result.total == 0
    assert result.score == 0.0


def test_a_refusal_and_an_infra_failure_are_told_apart() -> None:
    """Three categories, because two of them give the wrong answer for a refusal.

    Observed live: one typing_speed_test/dynamic cell died in 108s to
    "Output blocked by content filtering policy" -- on a TYPING SPEED TEST, from a
    model that built the other 24 ideas fine.

    It is not harness_infra ("a harness fault, so retry it"): retrying hands that
    model extra draws no other cell gets. It is not manifest_missing either,
    which is a MODEL result correctly scored at the floor. So it is its own
    terminal, unscored status.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_fleet as bf

    blocked = 'error.message="Output blocked by content filtering policy"'
    assert bf._is_refusal("", blocked)
    assert not bf._is_infra_failure("", blocked), "a refusal must not be retryable"

    locked = "Failed to execute statement"
    assert bf._is_infra_failure("", locked)
    assert not bf._is_refusal("", locked)

    # A model that did not write a manifest at all is neither.
    no_manifest = "viralbench.json not found at app root"
    assert not bf._is_infra_failure("", no_manifest)
    assert not bf._is_refusal("", no_manifest)


def test_a_refusal_is_never_auto_retried() -> None:
    """Terminal by design: the user's call, and the control-safe one.

    Retrying only the cells one model had refused would give it attempts the rest
    of the grid never got -- the same reason manifest_missing is off the list.
    """
    import inspect
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_fleet as bf

    src = inspect.getsource(bf.pending_cells)
    always = src[src.index("always_retry = ") : src.index("always_retry = ") + 200]
    assert "provider_refusal" not in always
    assert "harness_timeout" in always, "our wall clock still is retryable"


def test_reclassify_only_ever_relaxes_in_our_own_direction() -> None:
    """It may reclassify OUR failure, never launder a model's.

    A verdict is written at build time, so adding a signature later does not reach
    the cells it was written for. Reclassifying fixes that, and must only ever
    move harness_failed / no_record / harness_infra into harness_infra or
    provider_refusal -- never touch manifest_missing, manifest_invalid or ok.
    """
    import inspect
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_fleet as bf

    src = inspect.getsource(bf.reclassify_infra)
    assert '"harness_failed",' in src and '"no_record",' in src
    assert '"provider_refusal"' in src
    for protected in ("manifest_missing", "manifest_invalid"):
        assert protected not in src, f"{protected} must be untouchable"
