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

"""Launch the crowd simulation from the main (Python 3.12) process.

The crowd simulation itself lives in :mod:`viral_bench.crowd.sim`, which imports
:mod:`oasis` and therefore only runs in the isolated Python 3.11 ``.venv-crowd``.
This module is the bridge: it spawns
``.venv-crowd/bin/python -m viral_bench.crowd.sim.runner`` as a subprocess (with
``PYTHONPATH=src`` so the runner imports the repo source without a pip install),
waits for it, and loads the artifacts it produced.

It imports no crowd-sim / oasis code, so it is safe to import in the 3.12 harness.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from viral_bench.crowd.sim_defaults import (
    DEFAULT_AGENTS,
    DEFAULT_LATECOMERS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_RECSYS,
    DEFAULT_ROUNDS,
    DEFAULT_SEED,
    DEFAULT_SEMAPHORE,
    DEFAULT_START_WAIT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRIAL_MAX_STEPS,
    DEFAULT_TRIERS,
)
from viral_bench.founder.build import load_build_record
from viral_bench.founder.workspace import builds_root

_REPO_ROOT = Path(__file__).resolve().parents[3]
CROWD_VENV = _REPO_ROOT / ".venv-crowd"

#: Wall clock for one crowd run before the subprocess is killed.
#:
#: This has to clear the DEFAULT workload, or the shipped command fails for
#: everyone who runs it as documented. Thirty agents over three rounds against a
#: frontier model was still in round two at the old 3600s ceiling, and a run
#: killed at the wall writes no run_summary.json, so the hour bought nothing and
#: there was no flag to raise it. Lower it with --timeout for a cheap smoke run.
#:
#: 3600 -> 10800 -> 21600, each step measured rather than guessed. 10800 was set
#: from runs of 4025s and 7343s, which looked like ample headroom. It was not: a
#: default 30-agent run against grok-4.6 then finished in 10442s, clearing the
#: ceiling by 358 seconds -- 3.3%. The margin has to be generous because the two
#: errors are not symmetric. Overshooting costs a slow run some extra patience,
#: while undershooting throws away everything: no summary is written, so several
#: hours of wall clock and the entire API spend produce nothing that can be
#: scored, and the user cannot tell a too-tight clock from a broken harness.
#: A crowd's cost scales with agents x rounds x how talkative the model is, and
#: none of those are bounded by this module, so the default now sits at roughly
#: 2x the slowest run actually observed.
DEFAULT_TIMEOUT_S = 21600.0


class CrowdLaunchError(RuntimeError):
    """Raised when the crowd simulation cannot be launched."""


@dataclass
class CrowdRunResult:
    """Outcome of a launched crowd run (as seen from the main process)."""

    build_id: str
    out_dir: str
    ok: bool
    returncode: int
    summary: dict | None = None
    stderr_tail: str = ""

    @property
    def db_path(self) -> str:
        """Path to the OASIS SQLite DB (the scoring stage's primary input)."""
        return str(Path(self.out_dir) / "simulation.db")


def crowd_python() -> Path:
    """Path to the crowd env's Python interpreter."""
    return CROWD_VENV / "bin" / "python"


def crowd_env_ready() -> bool:
    """True if the isolated crowd env exists (created by setup_crowd_env.sh)."""
    return crowd_python().is_file()


def _default_out_dir(build_id: str) -> Path:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return builds_root() / "crowd" / f"{build_id}__crowd-{ts}"


def check_out_dir_name(build_id: str, out_dir: str | Path) -> None:
    """Refuse an ``--out`` whose name hides the run from its own build.

    The directory name is not cosmetic, it is the only join key between a build
    and its runs. Nothing indexes runs: every lookup is a prefix glob for
    ``<build_id>__*`` -- ``crowd_runs_for_build`` in the viewer, ``score
    <build_id>`` picking the latest run, and ``comparison_block`` collecting the
    ViralScores to set beside a grade.

    A name that does not carry the prefix therefore produces no error and no
    empty result, just a quieter wrong answer everywhere at once: the run is
    absent from the build's run list, ``score <build_id>`` silently scores some
    *other* run, and the grader records ``comparison: null`` -- which is
    persisted, so the rubric viewer reports "this build has no crowd runs" until
    the build is graded again, at full price. That was found the expensive way.

    Both in-repo callers already comply (``_default_out_dir`` and
    ``crowd_sweep.py``), so this only ever fires on a hand-written ``--out``,
    and it fires before the run starts rather than after it has been paid for.
    """
    name = Path(out_dir).name
    if name.startswith(f"{build_id}__"):
        return
    raise CrowdLaunchError(
        f"--out directory {name!r} must be named '{build_id}__<something>'.\n"
        f"  A crowd run is joined to its build by that filename prefix and by "
        f"nothing else, so a run stored under any other name is invisible to "
        f"`viral-bench score {build_id}`, to the viewers, and to the rubric "
        f"grader's ViralScore comparison -- silently, and permanently once a "
        f"grade has been written.\n"
        f"  Try: --out builds/crowd/{build_id}__crowd-<timestamp>\n"
        f"  Or omit --out entirely and one is named for you."
    )


def run_crowd_sim(
    build_id: str,
    *,
    agents: int = DEFAULT_AGENTS,
    triers: int = DEFAULT_TRIERS,
    latecomers: int = DEFAULT_LATECOMERS,
    rounds: int = DEFAULT_ROUNDS,
    model: str = DEFAULT_MODEL,
    recsys: str = DEFAULT_RECSYS,
    seed: int = DEFAULT_SEED,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    semaphore: int = DEFAULT_SEMAPHORE,
    start_wait: float = DEFAULT_START_WAIT,
    trial_max_steps: int = DEFAULT_TRIAL_MAX_STEPS,
    container: bool = True,
    no_llm: bool = False,
    interview: bool = True,
    validity_gate: bool = True,
    personas: str | None = None,
    variant: str = "",
    thinking_level: str = "",
    min_interactions: int | None = None,
    disable_tools: str = "",
    env_notice: bool = True,
    follow_peers: int | None = None,
    feed_max_posts: int = 0,
    out_dir: str | Path | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    stream: bool = True,
) -> CrowdRunResult:
    """Run one crowd simulation for ``build_id`` in the isolated crowd env.

    Args mirror :class:`~viral_bench.crowd.sim.simulation.SimulationConfig`.
    Validates the build exists (fast, in-process) before spawning the subprocess.

    Raises:
        CrowdLaunchError: If the crowd env is missing or the build is unknown.
    """
    if not crowd_env_ready():
        raise CrowdLaunchError(
            f"crowd env not found at {CROWD_VENV}. Create it with "
            "scripts/setup_crowd_env.sh (installs OASIS in an isolated Python "
            "3.11 venv)."
        )
    load_build_record(build_id)  # fail fast if the build id is unknown

    if out_dir:
        check_out_dir_name(build_id, out_dir)
    out = Path(out_dir) if out_dir else _default_out_dir(build_id)
    out.mkdir(parents=True, exist_ok=True)

    argv = [
        str(crowd_python()),
        "-m",
        "viral_bench.crowd.sim.runner",
        "--build-id",
        build_id,
        "--out",
        str(out),
        "--agents",
        str(agents),
        "--triers",
        str(triers),
        "--latecomers",
        str(latecomers),
        "--rounds",
        str(rounds),
        "--model",
        model,
        "--recsys",
        recsys,
        "--seed",
        str(seed),
        "--temperature",
        str(temperature),
        # These three existed on the runner but were never forwarded, so the
        # config values could not reach an actual run.
        "--semaphore",
        str(semaphore),
        "--start-wait",
        str(start_wait),
        "--trial-max-steps",
        str(trial_max_steps),
    ]
    # Omitted entirely when None, so "no cap" stays the default all the way
    # down instead of being spelled as some large magic integer.
    if max_tokens is not None:
        argv += ["--max-tokens", str(max_tokens)]
    if not container:
        argv.append("--host")
    if no_llm:
        argv.append("--no-llm")
    if not interview:
        argv.append("--no-interview")
    if not validity_gate:
        argv.append("--no-validity-gate")
    if personas:
        argv += ["--personas", personas]
    if variant:
        argv += ["--variant", variant]
    if thinking_level:
        argv += ["--thinking-level", thinking_level]
    if min_interactions is not None:
        argv += ["--min-interactions", str(min_interactions)]
    if disable_tools:
        argv += ["--disable-tools", disable_tools]
    if not env_notice:
        argv.append("--no-env-notice")
    if follow_peers is not None:
        argv += ["--follow-peers", str(follow_peers)]
    if feed_max_posts:
        argv += ["--feed-max-posts", str(feed_max_posts)]

    env = dict(os.environ)
    # Prepend repo src so the runner imports viral_bench without a pip install.
    src = str(_REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    try:
        proc = subprocess.run(
            argv,
            env=env,
            cwd=str(_REPO_ROOT),
            text=True,
            capture_output=not stream,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        # A run killed at the wall writes no run_summary.json, so the whole run
        # is lost. Say that plainly and name the flag, rather than surfacing a
        # bare TimeoutExpired traceback from deep in subprocess.
        raise CrowdLaunchError(
            f"the crowd run passed its {timeout:.0f}s wall clock and was killed, "
            f"so no run_summary.json was written and nothing can be scored.\n"
            f"  Partial simulation state is in {out}\n"
            f"  Raise the limit with --timeout <seconds>, or make the run smaller "
            f"with --agents / --rounds."
        ) from exc
    stderr_tail = ""
    if not stream and proc.stderr:
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])

    summary = None
    summary_path = out / "run_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = None

    ok = proc.returncode == 0 and bool(summary) and summary.get("ok", False)
    return CrowdRunResult(
        build_id=build_id,
        out_dir=str(out),
        ok=ok,
        returncode=proc.returncode,
        summary=summary,
        stderr_tail=stderr_tail,
    )
