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

"""Core data types and the benchmark runner.

This module is intentionally small and dependency-free so it is easy to read
and extend. The mental model is:

    Task           -> a single test case (prompt + expected answer)
    ModelFn        -> any callable that takes a prompt and returns a response
    BenchmarkResult-> the aggregated score over a list of tasks

A new contributor can add benchmarks by creating ``Task`` objects (see
``viral_bench.tasks``) and scoring functions, without touching the runner.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

# A "model" is anything that maps a prompt string to a response string.
# In real usage this would call an LLM API; in tests we pass a fake function.
ModelFn = Callable[[str], str]


@dataclass(frozen=True)
class Task:
    """A single benchmark test case.

    Attributes:
        task_id: A unique, human-readable identifier.
        prompt: The input given to the model.
        expected: The reference answer used for scoring.
    """

    task_id: str
    prompt: str
    expected: str


@dataclass
class BenchmarkResult:
    """Aggregated outcome of running a model over a set of tasks."""

    total: int = 0
    passed: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Fraction of tasks passed, in the range [0.0, 1.0]."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total


def exact_match(response: str, expected: str) -> bool:
    """Default scorer: case-insensitive, whitespace-trimmed equality."""
    return response.strip().lower() == expected.strip().lower()


def run_benchmark(
    model: ModelFn,
    tasks: Iterable[Task],
    scorer: Callable[[str, str], bool] = exact_match,
) -> BenchmarkResult:
    """Run ``model`` against ``tasks`` and return an aggregated result.

    Args:
        model: A callable mapping a prompt to a response.
        tasks: The benchmark tasks to evaluate.
        scorer: Predicate deciding whether a response counts as correct.

    Returns:
        A populated :class:`BenchmarkResult`.
    """
    result = BenchmarkResult()
    for task in tasks:
        result.total += 1
        response = model(task.prompt)
        if scorer(response, task.expected):
            result.passed += 1
        else:
            result.failures.append(task.task_id)
    return result
