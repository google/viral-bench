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

"""Tests for collaboration toolsets.

One toolset ships today (``local``); these cover it plus the contract every
toolset has to satisfy -- what ships from the working directory, and how a
toolset is constructed by name.
"""

from __future__ import annotations

import pytest

from viral_bench.founder.collab import (
    DESIGN_FILE,
    SCRATCH_FILES,
    TOOLSETS,
    CollabError,
    LocalToolset,
    build_toolset,
    files_to_strip,
)
from viral_bench.founder.roles import roles_for

# -- files_to_strip ---------------------------------------------------------- #


def test_local_keeps_design_strips_scratch() -> None:
    strip = files_to_strip("local")
    assert DESIGN_FILE not in strip
    for name in SCRATCH_FILES:
        assert name in strip


def test_a_toolset_hosting_the_design_elsewhere_strips_it_too() -> None:
    # A toolset that keeps the design outside the working directory must strip
    # DESIGN.md too, or a stale on-disk copy ships with the app.
    strip = files_to_strip("hosted-elsewhere")
    assert DESIGN_FILE in strip
    for name in SCRATCH_FILES:
        assert name in strip


# -- local toolset ----------------------------------------------------------- #


def test_local_toolset_is_files_only() -> None:
    tools = LocalToolset()
    tools.prepare("b1", roles_for(4))
    assert tools.turn_env(1) == {}
    brief = tools.collaboration_brief(1)
    assert DESIGN_FILE in brief
    assert "shared working directory" in brief
    assert tools.metadata() == {"collab": "local"}
    tools.cleanup()  # must not raise


# -- build_toolset ----------------------------------------------------------- #


def test_only_the_reproducible_toolset_ships() -> None:
    """``--collab`` offers exactly what the registry holds, and it holds one.

    The externally-hosted collaboration arm was removed: it needed an account,
    a network surface and provisioned seats, so a scored run could not be
    reproduced by anyone who did not have all three -- and an arm nobody else
    can re-run is not a benchmark result. Advertising a toolset that does not
    ship would put the choice back on the CLI, which builds its choices from
    this tuple.
    """
    assert TOOLSETS == ("local",)


def test_build_toolset_local() -> None:
    assert isinstance(build_toolset("local", n_agents=1), LocalToolset)
    assert isinstance(build_toolset("local", n_agents=4), LocalToolset)


def test_build_toolset_unknown_raises() -> None:
    # An unknown name is the only way to fail: the one toolset that ships has no
    # requirement of its own, so every team size is accepted.
    with pytest.raises(CollabError, match="unknown collab") as err:
        build_toolset("carrier-pigeon", n_agents=4)
    assert "local" in str(err.value)  # the message names what is available
