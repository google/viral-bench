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

"""The OASIS-backed crowd simulation (runs in the isolated Python 3.11 env).

Everything in this subpackage imports :mod:`oasis` (camel-oasis), which requires
Python <3.12 and pulls a heavy, exactly-pinned dependency set (torch, pandas,
camel-ai 0.2.78). It therefore runs ONLY inside the dedicated ``.venv-crowd``
environment, never in the main 3.12 process. The main process talks to it through
:mod:`viral_bench.crowd.launch`, which spawns :mod:`viral_bench.crowd.sim.runner`
as a subprocess and reads back the artifacts it writes.

The modules:

* :mod:`~viral_bench.crowd.sim.personas` -- load the curated, frozen persona set.
* :mod:`~viral_bench.crowd.sim.model` -- the CAMEL Gemini backend the agents use.
* :mod:`~viral_bench.crowd.sim.prompts` -- tier-specific system prompts + the
  founder launch-post text.
* :mod:`~viral_bench.crowd.sim.agents` -- build the OASIS ``AgentGraph``: a few
  first-hand "trier" agents (full app-interaction toolkit) and many "reactor"
  agents (cheap read-only code inspection), plus a seeded follow graph.
* :mod:`~viral_bench.crowd.sim.platform` -- the Twitter-like platform + recsys.
* :mod:`~viral_bench.crowd.sim.simulation` -- the seed -> rounds -> interview loop
  and the artifacts it emits.
* :mod:`~viral_bench.crowd.sim.runner` -- the subprocess entry point.

This layer stands up the *social environment* and records rich interactions to an
OASIS SQLite database; turning those into a ViralScore is a separate, later stage.
"""

from __future__ import annotations
