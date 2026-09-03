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

"""Import and construct the crowd-sim objects INSIDE the crowd venv.

Everything in ``viral_bench.crowd.sim`` imports ``oasis``, which only exists in
the isolated Python 3.11 ``.venv-crowd``. So the main test suite cannot import
those modules at all, and for their whole life they have had no test that even
proves they load. That is not a theoretical gap: a field added to the
``CrowdAgents`` dataclass in the wrong position raised
``TypeError: non-default argument 'trier_toolkits' follows default argument`` at
import time, and nothing found it until a live run failed twelve minutes in.
Before that it would have wasted a whole ablation.

This shells out to the crowd interpreter and does the cheapest possible thing
that would have caught it: import the modules, build a real crowd selection at
every tier, and construct the agent graph's own dataclass. No LLM, no browser,
no OASIS environment -- about two seconds.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CROWD_PY = REPO / ".venv-crowd" / "bin" / "python"

pytestmark = pytest.mark.skipif(
    not CROWD_PY.is_file(), reason="needs the crowd env (.venv-crowd)"
)

#: Runs in the CROWD interpreter, not this one.
_PROGRAM = """
import sys

from viral_bench.crowd.sim.agents import CrowdAgents, build_crowd_agents  # noqa: F401
from viral_bench.crowd.sim.personas import load_personas, select_crowd
from viral_bench.crowd.sim.prompts import build_profile
from viral_bench.crowd.sim.simulation import SimulationConfig, arch_version
from viral_bench.crowd.sim.verdicts import conversion_stats

# 1. Every tier is expressible and adds up to the crowd size.
personas = load_personas()
sel = select_crowd(personas, n_agents=12, n_triers=-1, n_latecomers=4, seed=0)
assert len(sel.triers) == 8, len(sel.triers)
assert len(sel.latecomers) == 4, len(sel.latecomers)
assert len(sel.all) == 12
assert {sel.tier_of(p) for p in sel.all} == {"trier", "latecomer"}

# 2. The prompt for every tier renders (a stray brace in a mission template is
#    a KeyError at format() time, i.e. mid-run).
for tier in ("trier", "latecomer", "reactor"):
    for app_type in ("client-app", "full-stack-app"):
        text = build_profile(sel.all[0], tier, app_type)["mission"]
        assert text and "{" not in text, (tier, app_type)

# 3. The dataclass constructs. This is the one that broke.
crowd = CrowdAgents(
    agent_graph=None,
    founder_id=0,
    trier_ids=[1],
    reactor_ids=[3],
    trier_toolkits={},
    persona_by_id={1: sel.all[0], 2: sel.all[1], 3: sel.all[2]},
    tier_by_id={1: "trier", 2: "latecomer", 3: "reactor"},
    latecomer_ids=[2],
)
assert crowd.crowd_ids == [1, 2, 3]
assert crowd.hands_on_ids == [1, 2]
assert conversion_stats(crowd)["n_latecomers"] == 1

# 4. SimulationConfig accepts what the runner passes it, and tags its variant.
cfg = SimulationConfig(
    build_id="b", out_dir="/tmp/x", n_agents=30, n_triers=-1, n_latecomers=8
)
assert cfg.n_latecomers == 8
assert arch_version("triers8").endswith("+triers8")

print("crowd-env smoke ok")
"""


def test_the_crowd_modules_import_and_construct_in_their_own_venv():
    proc = subprocess.run(
        [str(CROWD_PY), "-c", _PROGRAM],
        cwd=REPO,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stdout.write(proc.stderr)
    assert proc.returncode == 0, proc.stderr.strip().splitlines()[-3:]
    assert "crowd-env smoke ok" in proc.stdout
