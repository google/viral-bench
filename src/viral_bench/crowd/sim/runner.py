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

"""Subprocess entry point for the crowd simulation (runs in ``.venv-crowd``).

The main (Python 3.12) harness cannot import :mod:`oasis`, so it launches this
module in the isolated 3.11 crowd environment:

    PYTHONPATH=src .venv-crowd/bin/python -m viral_bench.crowd.sim.runner \
        --build-id <id> --out <dir> [--agents 8 --triers 5 --rounds 4 ...]

It runs one :func:`~viral_bench.crowd.sim.simulation.run_simulation`, writes the
artifacts into ``--out``, also writes ``result.json`` there, prints a one-line
status, and exits non-zero on failure. See :mod:`viral_bench.crowd.launch` for the
launching side.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from viral_bench.crowd.sim.simulation import SimulationConfig, run_simulation
from viral_bench.crowd.sim_defaults import (
    DEFAULT_AGENTS,
    DEFAULT_ENV_NOTICE,
    DEFAULT_FOLLOW_PEERS,
    DEFAULT_INTERVIEW,
    DEFAULT_LATECOMERS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_INTERACTIONS,
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


def _build_config(argv: list[str] | None = None) -> SimulationConfig:
    p = argparse.ArgumentParser(prog="viral_bench.crowd.sim.runner")
    p.add_argument("--build-id", required=True)
    p.add_argument("--out", required=True, help="artifact output directory")
    p.add_argument("--agents", type=int, default=DEFAULT_AGENTS)
    p.add_argument("--triers", type=int, default=DEFAULT_TRIERS)
    p.add_argument(
        "--latecomers",
        type=int,
        default=DEFAULT_LATECOMERS,
        help="agents who hold the app tools but only use them if the feed "
        "convinces them. Their conversion rate is virality EARNED rather than "
        "computed: measured over 44 runs where everyone tried the app first, "
        "the fraction of agents talked INTO it was 0.00 in every single one.",
    )
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="the crowd's model as '<provider>/<model>', e.g. 'openai/gpt-5-mini' "
        "or 'google-vertex/gemini-2.0-flash'. Defaults to "
        "simulation.model.id in config/crowd.yaml; there is no built-in default, "
        "and a bare model id with no provider prefix is rejected",
    )
    p.add_argument("--recsys", default=DEFAULT_RECSYS)
    p.add_argument("--semaphore", type=int, default=DEFAULT_SEMAPHORE)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="output token cap per LLM call; omit for no cap (the default)",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--trial-max-steps", type=int, default=DEFAULT_TRIAL_MAX_STEPS)
    p.add_argument(
        "--start-wait",
        type=float,
        default=DEFAULT_START_WAIT,
        help="seconds to wait for a web app to answer HTTP before giving up",
    )
    p.add_argument(
        "--host", action="store_true", help="run the app on the host (not a container)"
    )
    p.add_argument(
        "--no-llm", action="store_true", help="scripted wiring smoke, no LLM spend"
    )
    p.add_argument("--no-interview", action="store_true")
    p.add_argument(
        "--no-validity-gate",
        action="store_true",
        help="skip the verify_code gate (faster, but leaves the run unverified)",
    )
    p.add_argument("--personas", default=None, help="override persona CSV path")
    p.add_argument(
        "--thinking-level", default="", help="low | high | '' (model default)"
    )
    p.add_argument("--min-interactions", type=int, default=DEFAULT_MIN_INTERACTIONS)
    p.add_argument(
        "--disable-tools",
        default="",
        help="comma-separated tool names to withhold (e.g. reload_page,screenshot)",
    )
    p.add_argument("--no-env-notice", action="store_true")
    p.add_argument("--follow-peers", type=int, default=DEFAULT_FOLLOW_PEERS)
    p.add_argument("--feed-max-posts", type=int, default=0)
    p.add_argument(
        "--variant",
        default="",
        help="name a deliberate instrument variation (an ablation). The run "
        "records its architecture as <version>+<variant>, which keeps it out "
        "of the main corpus -- an ablation is a different instrument and must "
        "not pool with the sweep it exists to inform.",
    )
    args = p.parse_args(argv)
    return SimulationConfig(
        build_id=args.build_id,
        out_dir=args.out,
        n_agents=args.agents,
        n_triers=args.triers,
        n_latecomers=args.latecomers,
        rounds=args.rounds,
        model_id=args.model,
        recsys_type=args.recsys,
        semaphore=args.semaphore,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        container=not args.host,
        # The YAML default is honoured as well as the flag, so
        # `simulation.interview: false` in crowd.yaml actually turns the
        # interview off instead of being a knob that does nothing.
        interview=DEFAULT_INTERVIEW and not args.no_interview,
        no_llm=args.no_llm,
        personas_file=args.personas,
        validity_gate=not args.no_validity_gate,
        trial_max_steps=args.trial_max_steps,
        start_wait=args.start_wait,
        variant=args.variant,
        thinking_level=args.thinking_level or None,
        min_interactions=args.min_interactions,
        disabled_tools=tuple(
            x.strip() for x in args.disable_tools.split(",") if x.strip()
        ),
        env_notice=DEFAULT_ENV_NOTICE and not args.no_env_notice,
        follow_peers=args.follow_peers,
        feed_max_posts=args.feed_max_posts,
    )


def setup_logging(out_dir: str) -> Path:
    """Give the harness somewhere to record what went wrong.

    Every ``viral_bench.*`` logger was handler-less, so a 147 MB log directory
    contained exactly zero WARNING lines from our own code -- swallowed rounds,
    skipped turns and unreachable apps all left no trace. That is why the bugs
    in docs/crowd_bugs.md survived as long as they did.

    OASIS is separately noisy: it echoes every prompt and observation at INFO to
    a hard-coded ``./log``, which is ~7000:1 noise to signal. Cap it at WARNING
    so the volume goes to the run that produced it, not to the repo root.
    """
    log_path = Path(out_dir) / "crowd_sim.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    root = logging.getLogger("viral_bench")
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(logging.WARNING)
    stderr.setFormatter(logging.Formatter("[crowd-sim] %(levelname)s %(message)s"))
    root.addHandler(stderr)

    for noisy in ("oasis", "social.agent", "social.twitter", "camel"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


def main(argv: list[str] | None = None) -> int:
    config = _build_config(argv)
    log_path = setup_logging(config.out_dir)
    log = logging.getLogger("viral_bench.crowd.runner")
    # Always record the configuration that produced this run. A log that is
    # empty on a healthy run is indistinguishable from a log that is empty
    # because nothing was wired up -- which is the state this repo was in.
    log.info("crowd run start: %s", config)
    result = asyncio.run(run_simulation(config))
    log.info(
        "crowd run end: ok=%s rounds=%s/%s duration=%ss failures=%s (log: %s)",
        result.ok,
        result.rounds_run,
        config.rounds,
        result.duration_s,
        (result.health or {}).get("failures"),
        log_path,
    )
    (Path(config.out_dir) / "result.json").write_text(
        result.to_json(), encoding="utf-8"
    )
    # A run can fail without raising -- lost rounds, or a crowd that never
    # answered -- so report the health failures, not just the exception channel.
    reasons = (result.health or {}).get("failures") or (
        [result.error] if result.error else ["unknown"]
    )
    status = "OK" if result.ok else "FAILED: " + "; ".join(reasons)
    eng = result.engagement or {}
    print(
        f"[crowd-sim] {result.build_id} ({result.app_type}) {status} | "
        f"agents={result.n_agents} triers={result.n_triers} "
        f"rounds={result.rounds_run} | posts={eng.get('posts', 0)} "
        f"likes={eng.get('likes', 0)} reposts={eng.get('reposts', 0)} "
        f"comments={eng.get('comments', 0)} follows={eng.get('follows', 0)}",
        file=sys.stderr,
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
