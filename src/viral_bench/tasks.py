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

"""Example benchmark tasks.

This is a placeholder dataset so the project runs end-to-end out of the box.
Replace or extend ``SAMPLE_TASKS`` with real benchmark data.
"""

from __future__ import annotations

from viral_bench.core import Task

SAMPLE_TASKS: list[Task] = [
    Task(task_id="arith-1", prompt="What is 2 + 2?", expected="4"),
    Task(
        task_id="capital-1", prompt="What is the capital of France?", expected="Paris"
    ),
    Task(
        task_id="color-1",
        prompt="What color is the sky on a clear day?",
        expected="blue",
    ),
]
