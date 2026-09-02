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

"""Tests for the viral-bench CLI argument parsing.

We only test parser behavior that carries real logic -- an unknown ``--model`` is
rejected while an unlisted model garden model is not, only the two supported
founder configurations (1 or 4) are accepted, and ``--collab`` is constrained.
Plain argparse plumbing (defaults, store_true flags, that a value round-trips) is
not worth a test.
"""

from __future__ import annotations

import pytest

from viral_bench.cli import _cmd_found, build_parser


def test_found_rejects_a_model_with_no_provider_prefix(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Rejected by the command, not by argparse ``choices``.

    ``choices`` would also reject any model released after this code was
    written, which is precisely the case ``--model`` has to keep open. What IS
    rejected is a bare id: with no default provider there is nothing sensible to
    guess, and guessing would bill an account the user never chose.
    """
    args = build_parser().parse_args(["found", "x", "--model", "gpt-test"])
    assert _cmd_found(args) == 2
    assert "provider prefix" in capsys.readouterr().err


def test_found_with_no_model_at_all_says_how_to_choose_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The state of a fresh checkout: no model configured for any stage.

    Passed explicitly rather than relying on the parser default, which picks up
    a developer's own config/local.yaml.
    """
    args = build_parser().parse_args(["found", "x", "--model", ""])
    assert _cmd_found(args) == 2
    assert "viral-bench init" in capsys.readouterr().err


def test_found_accepts_a_model_id_it_has_never_heard_of() -> None:
    """Parsing must not be where a model released this morning gets blocked."""
    novel = "google-vertex-anthropic/claude-test"
    args = build_parser().parse_args(["found", "x", "--model", novel])
    assert args.model == novel


@pytest.mark.parametrize("bad", ["0", "2", "3", "5", "-1"])
def test_found_rejects_unsupported_agent_counts(bad: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["found", "x", "--agents", bad])


def test_found_rejects_unknown_collab() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["found", "x", "--collab", "slack"])


def test_every_queued_ablation_knob_is_reachable_from_the_cli():
    """A knob the queue passes and the CLI rejects wastes a whole arm.

    `--temperature` was in the ablation queue and not on `crowd-run`, so the
    temp0 arm burned 42 runs in five minutes producing nothing but
    "unrecognized arguments". It failed loudly, which is why it cost minutes
    rather than a day -- but nothing stopped it being queued in the first place.
    """
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ablation_queue", repo / "scripts" / "ablation_queue.py"
    )
    queue = importlib.util.module_from_spec(spec)
    sys.modules["ablation_queue"] = queue
    spec.loader.exec_module(queue)

    help_text = subprocess.run(
        [sys.executable, "-m", "viral_bench.cli", "crowd-run", "--help"],
        capture_output=True,
        text=True,
        cwd=repo,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo / "src")},
    ).stdout
    for name, argv in queue.QUEUE.items():
        for token in argv:
            if token.startswith("--"):
                assert token in help_text, (
                    f"{name} passes {token}, crowd-run has no such flag"
                )


def test_found_accepts_dynamic_agents() -> None:
    args = build_parser().parse_args(["found", "x", "--agents", "dynamic"])
    assert args.agents == "dynamic"


def test_found_turns_is_parsed() -> None:
    args = build_parser().parse_args(
        ["found", "x", "--agents", "dynamic", "--turns", "5"]
    )
    assert args.turns == 5


def test_found_rejects_an_unregistered_collab_toolset() -> None:
    """``--collab`` is constrained to the registry, so a typo never starts a build."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["found", "x", "--collab", "carrier-pigeon"])
