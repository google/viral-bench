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

"""Stage 2 -- Founder Harness: design + build + ship an app for one idea.

Public API:

    from viral_bench.founder import run_build, open_session
    record = run_build("sliding_tile_game")   # design -> build -> ship (host)
    with open_session(record.build_id) as ...  # materialize + run for testing

The founder agent is driven by opencode (backed by Gemini) behind the
:class:`~viral_bench.founder.harness.FounderHarness` interface. A build always
runs on the host in a fresh :mod:`~viral_bench.founder.workspace` directory and
emits a :mod:`~viral_bench.founder.manifest` describing how to run/test the app.
Running the built app -- on the host or in a container -- is handled by
:mod:`~viral_bench.founder.runtime` via :mod:`~viral_bench.founder.runner`.
"""

from viral_bench.founder.apphost import AppHost
from viral_bench.founder.build import (
    BuildRecord,
    list_builds,
    load_build_record,
    run_build,
)
from viral_bench.founder.collab import (
    CollaborationToolset,
    LocalToolset,
    build_toolset,
)
from viral_bench.founder.harness import (
    FounderHarness,
    OpenCodeHarness,
    OpenCodeRunner,
)
from viral_bench.founder.manifest import Manifest, load_manifest
from viral_bench.founder.roles import TEAM_SIZE, Role, roles_for
from viral_bench.founder.runner import AppSession, describe_build, open_session
from viral_bench.founder.runtime import (
    ContainerRuntime,
    LocalRuntime,
    RunningApp,
    reap_orphans,
)
from viral_bench.founder.structures import (
    DEFAULT_ROUNDS,
    CollaborationStructure,
    RoundTableTeam,
    SoloPipeline,
    build_structure,
)
from viral_bench.founder.verify import (
    TryResult,
    VerifyResult,
    probe_endpoint,
    try_app,
    verify_code,
)
from viral_bench.founder.workspace import BuildWorkspace

__all__ = [
    # Stage 2 -- build + ship (host)
    "run_build",
    "list_builds",
    "load_build_record",
    "BuildRecord",
    "OpenCodeHarness",
    "OpenCodeRunner",
    "FounderHarness",
    # Founder collaboration structures (solo baseline + round-table team)
    "CollaborationStructure",
    "SoloPipeline",
    "RoundTableTeam",
    "build_structure",
    "DEFAULT_ROUNDS",
    "Role",
    "roles_for",
    "TEAM_SIZE",
    # Collaboration toolsets
    "CollaborationToolset",
    "LocalToolset",
    "build_toolset",
    "Manifest",
    "load_manifest",
    "BuildWorkspace",
    # Run / test an app (host or container)
    "open_session",
    "AppSession",
    "describe_build",
    "LocalRuntime",
    "ContainerRuntime",
    "RunningApp",
    "reap_orphans",
    # Stage 3 -- crowd-facing checks
    "AppHost",
    "verify_code",
    "try_app",
    "probe_endpoint",
    "VerifyResult",
    "TryResult",
]
