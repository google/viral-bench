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

"""Command-line entry point for viral_bench.

Subcommands:
    viral-bench bench                    # run the sample benchmark (default)
    viral-bench found <idea> [--model M] [--agents 1|4] [--rounds N] [--min-rounds N]
                            [--collab local] [--no-browser-tools]
    viral-bench models [--check]         # list selectable founder models
    viral-bench builds                   # list founder builds
    viral-bench describe-build <id>      # show how to test a build
    viral-bench trajectory <id> [--zip]  # export what a build thought and did
    viral-bench serve-build <id> [--port N]   # start a build's app for testing
    viral-bench try-build <id>           # drive a build like a crowd agent would
    viral-bench crowd-run <id>           # run the crowd social simulation on a build

``<idea>`` is an idea_id from ideas/ or a path to an idea .yaml file.
Running ``viral-bench`` with no subcommand runs ``bench`` (unchanged behavior).
"""

from __future__ import annotations

import argparse
import sys

from viral_bench import config as _config
from viral_bench.core import run_benchmark
from viral_bench.crowd import sim_defaults as _CROWD_DEFAULTS
from viral_bench.crowd.launch import DEFAULT_TIMEOUT_S as _CROWD_DEFAULT_TIMEOUT_S
from viral_bench.founder.collab import TOOLSETS
from viral_bench.founder.serve import DEFAULT_PORT as _SERVE_DEFAULT_PORT
from viral_bench.rubric.run import DEFAULT_PASSES as _RUBRIC_DEFAULT_PASSES
from viral_bench.tasks import SAMPLE_TASKS


def dummy_model(prompt: str) -> str:
    """A trivial stand-in 'model' so the CLI works without API keys.

    Replace this with a real LLM call (e.g. an API client) when integrating
    an actual model.
    """
    canned = {
        "What is 2 + 2?": "4",
        "What is the capital of France?": "Paris",
        "What color is the sky on a clear day?": "blue",
    }
    return canned.get(prompt, "I don't know")


def _cmd_bench(_args: argparse.Namespace) -> int:
    result = run_benchmark(dummy_model, SAMPLE_TASKS)
    print(f"Ran {result.total} tasks")
    print(f"Passed: {result.passed}")
    print(f"Score:  {result.score:.0%}")
    if result.failures:
        print("Failures:", ", ".join(result.failures))
    return 0


def _cmd_found(args: argparse.Namespace) -> int:
    from viral_bench.founder.build import BuildError, run_build
    from viral_bench.founder.collab import CollabError, build_toolset
    from viral_bench.founder.harness import HarnessError, require_model
    from viral_bench.founder.models import REQUIRED, UnknownModelError, resolve_model
    from viral_bench.founder.roles import TEAM_SIZE
    from viral_bench.founder.structures import (
        DYNAMIC,
        StructureError,
        build_structure,
        normalize_agents,
    )
    from viral_bench.providers import UnsupportedCapabilityError, check_support

    # Validated here rather than by argparse `choices`: the set of model ids a
    # provider serves changes constantly, so pinning it in code would make a
    # newly released model unreachable without an edit.
    try:
        spec = resolve_model(require_model(args.model))
        check_support(spec, REQUIRED, stage="founder")
    except (HarnessError, UnknownModelError, UnsupportedCapabilityError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    model = spec.qualified
    try:
        agents = normalize_agents(args.agents)
        team = build_structure(
            agents,
            max_rounds=args.rounds,
            min_rounds=args.min_rounds,
            max_turns=args.turns,
        )
        # Validate the collaboration toolset early, before any build work, so a
        # misconfigured toolset fails fast instead of after the long build.
        build_toolset(args.collab, n_agents=agents)
    except (StructureError, CollabError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Founding idea {args.idea!r} with model {model} ...")
    print(f"Team:    {team.describe()}")
    print(f"Collab:  {args.collab}")
    if agents != TEAM_SIZE and args.collab != "local":
        print(f"Note:    --collab is ignored for --agents {args.agents}.")
    try:
        record = run_build(
            args.idea,
            model=model,
            agents=agents,
            collab=args.collab,
            rounds=args.rounds,
            min_rounds=args.min_rounds,
            turns=args.turns,
            browser_tools=args.browser_tools,
            ship=not args.no_ship,
        )
    except (BuildError, StructureError, CollabError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"\nBuild:   {record.build_id}")
    print(f"Status:  {record.status}")
    if record.structure == DYNAMIC:
        orch = record.orchestration or {}
        done = "declared complete" if orch.get("done_signalled") else "hit the turn cap"
        print(f"Turns:   {record.rounds_run} of {record.max_turns} ({done})")
        by_type = orch.get("subagent_types") or {}
        shape = ", ".join(f"{k}x{v}" for k, v in sorted(by_type.items())) or "none"
        print(
            f"Spawned: {record.subagents_spawned} subagents [{shape}]; peak "
            f"{orch.get('peak_concurrent_subagents', 0)} at once, "
            f"{orch.get('resumed_subagents', 0)} resumed"
        )
        authored = orch.get("self_authored_agents") or []
        if authored:
            print(f"Defined: {', '.join(authored)}")
    elif record.n_agents > 1:
        early = " (QA shipped early)" if record.shipped_early else ""
        print(
            f"Rounds:  {record.rounds_run} (min {record.min_rounds}, max "
            f"{record.max_rounds}){early}; {record.turns_spent} turns"
        )
    if record.collab != "local" and record.collab_meta:
        # A non-local toolset records whatever identifies its collaboration
        # surface, so print it generically instead of assuming a shape.
        detail = ", ".join(
            f"{k}={v}" for k, v in sorted(record.collab_meta.items()) if k != "collab"
        )
        print(f"Collab:  {record.collab}" + (f" ({detail})" if detail else ""))
    print(f"Harness: {'ok' if record.harness_ok else 'FAILED'}")
    for phase in record.phases:
        print(
            f"  - {phase['phase']:<24} rc={phase['returncode']} "
            f"({phase['duration_s']}s)"
        )
        if not phase.get("ok") and phase.get("stderr_tail"):
            for line in str(phase["stderr_tail"]).splitlines():
                print(f"      | {line}")
    if record.error:
        print(f"\nReason:  {record.error}")
    if record.shipped_ref:
        print(f"Shipped: {record.shipped_ref} (in {record.store_path})")
    print(f"App dir: {record.app_dir}")
    if record.status == "ok":
        print(f"\nTry it:  viral-bench serve-build {record.build_id}")
        return 0
    return 1


def _cmd_models(args: argparse.Namespace) -> int:
    """List providers and, with --check, prove one model is callable."""
    from viral_bench.founder.models import describe_models

    print(describe_models())
    if not args.check:
        return 0

    from viral_bench.founder.models import REQUIRED, UnknownModelError, resolve_model
    from viral_bench.providers import (
        MissingCredentialError,
        ModelError,
        UnsupportedCapabilityError,
        check_support,
        make_client,
    )

    # A live call, not a catalogue lookup. Several providers answer a metadata
    # request for a model the account was never entitled to, so a listing can
    # report a model as available right up until the first real build fails.
    print(f"\nChecking {args.check} ...")
    try:
        spec = resolve_model(args.check)
        check_support(spec, REQUIRED, stage="founder")
        make_client(spec).ping()
    except (
        UnknownModelError,
        MissingCredentialError,
        UnsupportedCapabilityError,
        ModelError,
    ) as exc:
        print(f"  NOT USABLE: {exc}", file=sys.stderr)
        return 1
    print(f"  {spec.qualified} is reachable and callable.")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    """Interactively write a .env and a model choice for each stage."""
    from viral_bench.setup import run_init

    return run_init(non_interactive=args.yes)


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Check every prerequisite a real run needs, and say which are missing."""
    from viral_bench.setup import run_doctor

    return run_doctor(probe_models=not args.offline)


def _cmd_builds(_args: argparse.Namespace) -> int:
    from viral_bench.founder.build import list_builds

    records = list_builds()
    if not records:
        print("No builds yet.")
        return 0
    for r in records:
        ship = r.shipped_ref or "-"
        print(f"{r.build_id}  [{r.status}]  {r.model}  ship={ship}")
    return 0


def _cmd_describe_build(args: argparse.Namespace) -> int:
    from viral_bench.founder.runner import describe_build

    print(describe_build(args.build_id))
    return 0


def _cmd_trajectory(args: argparse.Namespace) -> int:
    """Export what a founder build thought, ran and delegated, as a bundle."""
    from pathlib import Path

    from viral_bench.founder.build import list_builds
    from viral_bench.founder.trajectory import (
        TrajectoryError,
        build_trajectory,
        export_trajectory,
    )

    if args.all:
        build_ids = [record.build_id for record in list_builds()]
    elif args.build_id:
        build_ids = [args.build_id]
    else:
        print("pass a build_id, or --all for every build", file=sys.stderr)
        return 2
    if not build_ids:
        print("no builds found", file=sys.stderr)
        return 1

    out_root = Path(args.out) if args.out else Path("builds/trajectories")
    failures = 0
    for build_id in build_ids:
        try:
            if args.summary:
                manifest = build_trajectory(build_id).manifest
                totals = manifest["totals"]
                print(
                    f"{build_id}  source={manifest['source']} "
                    f"events={totals['events']} sessions={totals['sessions']} "
                    f"depth={totals['max_depth']} "
                    f"reasoning={totals['reasoning_chars']}c "
                    f"redacted={totals['reasoning_redacted']}"
                )
            else:
                print(export_trajectory(build_id, out_root / build_id, as_zip=args.zip))
        except TrajectoryError as exc:
            print(f"{build_id}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures == len(build_ids) else 0


def _cmd_serve_build(args: argparse.Namespace) -> int:
    """Serve one build on a port you pick, with browser caching disabled.

    The app is moved to a free internal port and fronted by a no-cache proxy on
    ``--port``. That combination is what lets you test build after build on the
    same port without the browser silently serving you the previous one, and lets
    you run many builds side by side even though nearly all of them declare 8000.
    """
    import time
    from urllib.parse import urlsplit

    from viral_bench.founder.runner import describe_build, open_session
    from viral_bench.founder.serve import NoCacheProxy, free_port, rebind

    print(describe_build(args.build_id))
    print("-" * 60)

    where = "container" if args.container else "host"
    session = open_session(args.build_id, container=args.container)
    print(f"Runtime:   {session.runtime.describe()} ({where})")

    proxy: NoCacheProxy | None = None
    internal = free_port()
    if not args.no_proxy and session.manifest.run.port is not None:
        session.manifest = rebind(session.manifest, internal)
        # Belt and braces: rebind rewrites the port in the command, and PORT
        # covers the apps that read it from the environment instead.
        session.runtime.add_env({"PORT": str(internal)})

    try:
        if not args.no_setup:
            session.setup()

        if args.smoke:
            smoke = session.smoke()
            if smoke is not None:
                status = (
                    "ok" if smoke.returncode == 0 else f"FAILED ({smoke.returncode})"
                )
                print(f"Smoke test: {status}")

        app = session.start()
        if args.no_proxy or session.manifest.run.port is None:
            if app.url:
                print(f"\nApp running at: {app.url}   (caching NOT disabled)")
            else:
                print("\nApp started; no URL declared in manifest.")
        else:
            # Where the app is reachable FROM THE HOST, which is not necessarily
            # the port it was told to bind. A container binds `internal` inside
            # its own network namespace and the runtime publishes that on an
            # arbitrary host port, so aiming the proxy at `internal` reached
            # nothing and every request came back 502. On the host runtime the
            # two are the same number, which is why only --container was broken.
            # `app.url` is the runtime's own statement of where it answers.
            upstream = urlsplit(app.url).port if app.url else None
            proxy = NoCacheProxy(
                port=args.port, upstream_port=upstream or internal, host=args.host
            ).start()
            print(f"\nApp running at: {proxy.url}")
            print(
                f"  no-cache proxy {args.host}:{args.port} -> "
                f"app on :{upstream or internal}"
            )
        print(f"Logs: {app.log_path}")
        print("Press Ctrl+C to stop.")
        try:
            while app.is_running():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping ...")
    finally:
        if proxy is not None:
            proxy.stop()
        session.close()
    return 0


def _cmd_try_build(args: argparse.Namespace) -> int:
    """Drive a build the way a crowd agent would: use it, and print the trace."""
    import asyncio
    import json
    from pathlib import Path

    from viral_bench.crowd.interaction import browser_available, try_app
    from viral_bench.founder.runner import describe_build

    print(describe_build(args.build_id))
    print("-" * 60)
    if not browser_available():
        print(
            "NOTE: no browser detected -- web apps run in DEGRADED static-HTTP "
            "mode.\n      For full fidelity: uv sync && uv run playwright install "
            "chrome\n" + "-" * 60
        )

    script = None
    if args.script:
        try:
            script = json.loads(Path(args.script).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"ERROR: could not read --script {args.script!r}: {exc}")
            return 2
        if not isinstance(script, list):
            print('ERROR: --script must be a JSON list of {"action": ...} objects.')
            return 2

    use_browser = False if args.no_browser else None
    try:
        if args.interactive:
            trace = asyncio.run(
                _interactive_trial(
                    args.build_id,
                    container=args.container,
                    max_steps=args.max_steps,
                    use_browser=use_browser,
                )
            )
        else:
            trace = asyncio.run(
                try_app(
                    args.build_id,
                    container=args.container,
                    script=script,
                    max_steps=args.max_steps,
                    use_browser=use_browser,
                )
            )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"ERROR: trial failed: {exc}")
        return 1

    print("-" * 60)
    print(trace.to_json() if args.json else trace.render())
    return 0


# One command set: every app in the bench is a web app.
_INTERACTIVE_HELP = (
    "Commands: look | click <target> | type <target> = <text> | "
    "press <key> | upload <fixture> | shot [note] | finish <notes> | quit"
)


async def _interactive_trial(
    build_id: str, *, container: bool, max_steps: int, use_browser: bool | None
):
    """A tiny REPL to drive an app by hand (great for poking at a build)."""
    import asyncio

    from viral_bench.crowd.interaction import AppInteractionToolkit

    toolkit = AppInteractionToolkit(
        build_id, container=container, max_steps=max_steps, use_browser=use_browser
    )
    help_text = _INTERACTIVE_HELP
    print(f"Interactive trial ({toolkit.app_type}). {help_text}")
    try:
        print(await _first_look(toolkit))
        while True:
            try:
                line = (await asyncio.to_thread(input, "app> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit", "q"):
                break
            if line.lower() in ("help", "?"):
                print(help_text)
                continue
            print(await _dispatch_interactive(toolkit, line))
    finally:
        await toolkit.close()
    return toolkit.trace


async def _first_look(toolkit) -> str:
    return await toolkit.open_app()


async def _dispatch_interactive(toolkit, line: str) -> str:
    parts = line.split(maxsplit=1)
    cmd = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    if cmd == "look":
        return await toolkit.look()
    if cmd == "click":
        return await toolkit.click(rest)
    if cmd == "press":
        return await toolkit.press_key(rest)
    if cmd == "shot":
        return await toolkit.screenshot(rest)
    if cmd == "type":
        if "=" not in rest:
            return "usage: type <target> = <text>"
        target, text = rest.split("=", 1)
        return await toolkit.type_text(target.strip(), text.strip())
    if cmd == "upload":
        return await toolkit.upload_file(rest.strip() or "photo.png")
    if cmd == "finish":
        return await toolkit.finish_trial(True, False, 5, rest)
    return f"unknown command {cmd!r}; type 'help'"


def _render_autorating(rating) -> str:
    """Show the autorater's dimensions, its own spread, and its citations."""
    lines = [f"autorater ({rating.model}, {rating.repeats} repeats):"]
    for dim, d in rating.dimensions.items():
        lines.append(f"  {dim:<15} {d.score:>4}/10  spread={d.spread}  {d.reason}")
        if d.evidence:
            lines.append(f"      cites: {', '.join(d.evidence[:6])}")
    for err in rating.errors:
        lines.append(f"  ! {err}")
    return "\n".join(lines)


def _cmd_score(args: argparse.Namespace) -> int:
    """Score a crowd run (or the latest run for a build) into a ViralScore."""
    import json
    from pathlib import Path

    from viral_bench.score import (
        ScoreWeights,
        extract_signals,
        find_crowd_runs,
        render_score,
        score_run,
        write_score,
    )
    from viral_bench.score.evidence import build_evidence_pack, write_evidence_pack

    target = Path(args.target)
    if not target.is_dir():
        runs = find_crowd_runs(args.target)
        if not runs:
            print(
                f"ERROR: no crowd runs found for {args.target!r}. Run "
                "`viral-bench crowd-run <build_id>` first, or pass a crowd "
                "run directory.",
                file=sys.stderr,
            )
            return 2
        # Score every run for a build when asked, otherwise only the newest.
        runs = runs if args.all else runs[-1:]
    else:
        runs = [target]

    weights = ScoreWeights.from_profile(args.profile)
    exit_code = 0
    for run_dir in runs:
        autorating = None
        if args.autorate:
            from viral_bench.score.autorater import rate_pack

            try:
                autorating = rate_pack(build_evidence_pack(run_dir))
            except Exception as exc:  # noqa: BLE001 - rate what can be rated
                print(f"WARNING: autorater unavailable ({exc})", file=sys.stderr)
        if args.evidence:
            print(f"wrote {write_evidence_pack(run_dir)}")
        try:
            result = score_run(extract_signals(run_dir), weights, autorating)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            exit_code = 2
            continue
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            print(render_score(result))
            print()
        if autorating is not None and not args.json:
            print(_render_autorating(autorating))
        if not args.no_write:
            if autorating is not None:
                (Path(run_dir) / "autorating.json").write_text(
                    json.dumps(autorating.as_dict(), indent=2), encoding="utf-8"
                )
            path = write_score(result, run_dir)
            if not args.json:
                print(f"wrote {path}")
        if not result.scorable:
            exit_code = 1
    return exit_code


def _cmd_grade(args: argparse.Namespace) -> int:
    """Grade one build against its idea's rubric."""
    import asyncio
    import json

    from viral_bench.providers.client import UnsupportedCapabilityError
    from viral_bench.providers.spec import UnknownProviderError
    from viral_bench.rubric.report import render_grade
    from viral_bench.rubric.run import GradeAborted, grade_build

    try:
        _, document = asyncio.run(
            grade_build(
                args.build_id,
                grader_model=args.grader_model,
                passes=args.passes,
                container=not args.host,
                write=not args.no_write,
            )
        )
    except (GradeAborted, UnsupportedCapabilityError, UnknownProviderError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(document, indent=2))
    else:
        print(render_grade(document))
    return 0


def _cmd_crowd_run(args: argparse.Namespace) -> int:
    """Run the OASIS crowd social simulation on a build (isolated 3.11 env)."""
    from viral_bench.crowd.launch import (
        CrowdLaunchError,
        crowd_env_ready,
        run_crowd_sim,
    )

    if not crowd_env_ready():
        print(
            "ERROR: the crowd environment is not set up. Create it with:\n"
            "  scripts/setup_crowd_env.sh\n"
            "(installs OASIS in an isolated Python 3.11 venv at .venv-crowd)"
        )
        return 2

    try:
        result = run_crowd_sim(
            args.build_id,
            agents=args.agents,
            triers=args.triers,
            rounds=args.rounds,
            model=args.model,
            recsys=args.recsys,
            seed=args.seed,
            container=not args.host,
            no_llm=args.no_llm,
            interview=not args.no_interview,
            validity_gate=not args.no_validity_gate,
            personas=args.personas,
            latecomers=args.latecomers,
            temperature=(
                _CROWD_DEFAULTS.DEFAULT_TEMPERATURE
                if args.temperature is None
                else args.temperature
            ),
            variant=args.variant,
            out_dir=args.out,
            timeout=args.timeout,
        )
    except CrowdLaunchError as exc:
        print(f"ERROR: {exc}")
        return 2

    print("-" * 60)
    if result.ok:
        eng = (result.summary or {}).get("engagement", {})
        print(f"Crowd run OK. Artifacts in: {result.out_dir}")
        print(f"  DB:      {result.db_path}")
        print(f"  Summary: {result.out_dir}/run_summary.json")
        print(
            f"  Engagement: posts={eng.get('posts', 0)} likes={eng.get('likes', 0)} "
            f"reposts={eng.get('reposts', 0)} comments={eng.get('comments', 0)} "
            f"follows={eng.get('follows', 0)} dislikes={eng.get('dislikes', 0)}"
        )
        launch = eng.get("launch_post")
        if launch:
            print(f"  Launch post: {launch}")
    else:
        print(f"Crowd run FAILED (rc={result.returncode}).")
        if result.stderr_tail:
            print(result.stderr_tail)
    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="viral-bench", description=__doc__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("bench", help="run the sample benchmark").set_defaults(
        func=_cmd_bench
    )

    from viral_bench.founder.harness import DEFAULT_MODEL
    from viral_bench.founder.roles import TEAM_SIZE
    from viral_bench.founder.structures import (
        DEFAULT_DYNAMIC_TURNS,
        DEFAULT_ROUNDS,
        DYNAMIC,
    )

    p_found = sub.add_parser("found", help="run the founder pipeline on an idea")
    p_found.add_argument(
        "idea", help="idea_id from ideas/, or a path to an idea .yaml file"
    )
    # Deliberately NOT `choices=`: the set of model ids a provider serves
    # changes constantly, and pinning it here would make a newly released model
    # unreachable without a code edit. _cmd_found validates instead.
    p_found.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help="founder model to build with, as '<provider>/<model>' "
        "(e.g. openai/gpt-5-mini). "
        + (
            f"Default: {DEFAULT_MODEL}."
            if DEFAULT_MODEL
            else "No default -- run `viral-bench init`, or pass one here. "
            "`viral-bench models` lists the providers."
        ),
    )
    # Defaults come from config/founder.yaml `collaboration:`, which used to be
    # documentation only -- nothing loaded it, so it could (and did) disagree
    # with the real defaults here.
    # `--agents` is a string, not an int, because one of the three founder
    # configurations is precisely the one whose agent count the MODEL decides.
    # Spelling that as a number ("0"? "-1"?) would be a riddle. `dynamic` says it.
    p_found.add_argument(
        "--agents",
        type=str,
        choices=("1", str(TEAM_SIZE), DYNAMIC),
        default=str(_config.founder_collaboration("agents", 1)),
        metavar="N",
        help=f"founder configuration: 1 = solo founder (design -> build), "
        f"{TEAM_SIZE} = the round-table specialist team that collaborates over "
        f"multiple rounds, {DYNAMIC} = one founder agent that chooses and "
        "orchestrates its own subagents, with no roles or rounds imposed "
        "(default: %(default)s).",
    )
    p_found.add_argument(
        "--turns",
        type=int,
        default=_config.founder_collaboration("turns", DEFAULT_DYNAMIC_TURNS),
        metavar="N",
        help=f"max orchestrator turns for --agents {DYNAMIC} (default: "
        "%(default)s). The founder stops as soon as it declares the build "
        "complete and the manifest is valid; a turn that ends unfinished is "
        f"resumed. Ignored for --agents 1/{TEAM_SIZE}.",
    )
    p_found.add_argument(
        "--rounds",
        type=int,
        default=_config.founder_collaboration("rounds", DEFAULT_ROUNDS),
        metavar="N",
        help=f"max collaboration rounds for the {TEAM_SIZE}-agent team; QA can "
        "still ship earlier (default: %(default)s). Ignored for the solo founder.",
    )
    p_found.add_argument(
        "--min-rounds",
        type=int,
        # Fallback matches config/founder.yaml rather than the library floor of
        # 1: a code default that silently disagrees with the config block is the
        # exact bug that block's own header records having shipped before.
        default=_config.founder_collaboration("min_rounds", 2),
        metavar="N",
        help=f"minimum rounds the {TEAM_SIZE}-agent team must run before QA may "
        "ship (default: %(default)s); must be <= --rounds. Each round is "
        f"{TEAM_SIZE} turns. Ignored for the solo founder.",
    )
    p_found.add_argument(
        "--collab",
        choices=list(TOOLSETS),
        default=_config.founder_collaboration("toolset", "local"),
        help="the medium the team collaborates through. 'local' (the default and "
        "only built-in) means the shared working directory: the design lives in "
        "DESIGN.md and short notes in TEAM_NOTES.md alongside the code. See "
        "viral_bench.founder.collab to add another.",
    )
    p_found.add_argument(
        "--no-browser-tools",
        dest="browser_tools",
        action="store_false",
        default=bool(_config.founder_collaboration("browser_tools", True)),
        help="disable the team's Designer/QA browser (on by default so they can "
        "render and click the running app; auto-disabled anyway if the host lacks "
        "the browser prerequisites).",
    )
    p_found.add_argument(
        "--no-ship", action="store_true", help="skip shipping to the git store"
    )
    p_found.set_defaults(func=_cmd_found)

    p_models = sub.add_parser("models", help="list providers and credentials")
    p_models.add_argument(
        "--check",
        metavar="MODEL",
        help="verify one '<provider>/<model>' is actually reachable and "
        "callable (may bill a single token)",
    )
    p_models.set_defaults(func=_cmd_models)

    p_init = sub.add_parser("init", help="choose a provider and write .env")
    p_init.add_argument(
        "--yes",
        action="store_true",
        help="do not prompt; report what is already configured and exit",
    )
    p_init.set_defaults(func=_cmd_init)

    p_doctor = sub.add_parser("doctor", help="check everything a real run needs")
    p_doctor.add_argument(
        "--offline",
        action="store_true",
        help="skip the live model calls and only check local tooling",
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    sub.add_parser("builds", help="list founder builds").set_defaults(func=_cmd_builds)

    p_desc = sub.add_parser("describe-build", help="show how to test a build")
    p_desc.add_argument("build_id")
    p_desc.set_defaults(func=_cmd_describe_build)

    p_traj = sub.add_parser(
        "trajectory",
        help="export a build's full trajectory (thinking, tools, subagents)",
        description=(
            "Bundle one founder build's trajectory: the prompts it was given, "
            "its chain of thought, every tool call and file patch, and each "
            "subagent it spawned. Writes trajectory.json (manifest) plus "
            "events.jsonl (the ordered stream) for a viewer or an SFT set."
        ),
    )
    p_traj.add_argument("build_id", nargs="?", help="build id (omit with --all)")
    p_traj.add_argument(
        "--all", action="store_true", help="export every build under builds/work"
    )
    p_traj.add_argument(
        "--out",
        metavar="DIR",
        help="output directory (default: builds/trajectories/<build_id>)",
    )
    p_traj.add_argument(
        "--zip", action="store_true", help="write a single .zip instead of a directory"
    )
    p_traj.add_argument(
        "--summary",
        action="store_true",
        help="print what was captured per build instead of writing anything",
    )
    p_traj.set_defaults(func=_cmd_trajectory)

    p_serve = sub.add_parser("serve-build", help="start a build's app for testing")
    p_serve.add_argument("build_id")
    p_serve.add_argument(
        "--port",
        type=int,
        default=_SERVE_DEFAULT_PORT,
        help=(
            f"port to serve the app on (default {_SERVE_DEFAULT_PORT}); pick a "
            "different one per build to run several at once"
        ),
    )
    p_serve.add_argument(
        "--host",
        default="0.0.0.0",
        help="address to bind (default 0.0.0.0; use 127.0.0.1 to keep it local)",
    )
    p_serve.add_argument(
        "--container",
        action="store_true",
        help="run the app in a rootless container instead of on the host",
    )
    p_serve.add_argument(
        "--no-setup", action="store_true", help="skip manifest setup steps"
    )
    p_serve.add_argument(
        "--smoke", action="store_true", help="run the smoke test before starting"
    )
    p_serve.add_argument(
        "--no-proxy",
        action="store_true",
        help=(
            "serve straight off the manifest's own port, without the no-cache "
            "proxy (the browser WILL cache across builds; debugging only)"
        ),
    )
    p_serve.set_defaults(func=_cmd_serve_build)

    p_try = sub.add_parser(
        "try-build",
        help="drive a build like a crowd agent would (browser/CLI/bot) + print trace",
    )
    p_try.add_argument("build_id")
    p_try.add_argument(
        "--container",
        action="store_true",
        help="run the app in a rootless container instead of on the host",
    )
    p_try.add_argument(
        "--interactive",
        action="store_true",
        help="drive the app yourself via a small REPL (else run a scripted trial)",
    )
    p_try.add_argument(
        "--script",
        metavar="FILE",
        help='JSON list of actions to run, e.g. [{"action":"click","target":"Roll"}]',
    )
    p_try.add_argument(
        "--no-browser",
        action="store_true",
        help="force the static-HTTP fallback even if a browser is available",
    )
    p_try.add_argument(
        "--max-steps", type=int, default=40, help="cap on interaction steps"
    )
    p_try.add_argument(
        "--json", action="store_true", help="print the trace as JSON instead of text"
    )
    p_try.set_defaults(func=_cmd_try_build)

    p_crowd = sub.add_parser(
        "crowd-run",
        help="run the OASIS crowd social simulation on a build (isolated env)",
    )
    p_crowd.add_argument("build_id")
    p_crowd.add_argument("--agents", type=int, default=_CROWD_DEFAULTS.DEFAULT_AGENTS)
    p_crowd.add_argument("--triers", type=int, default=_CROWD_DEFAULTS.DEFAULT_TRIERS)
    p_crowd.add_argument(
        "--latecomers",
        type=int,
        default=_CROWD_DEFAULTS.DEFAULT_LATECOMERS,
        help="agents who can try the app but only will if the feed convinces "
        "them; their conversion rate is virality earned rather than computed",
    )
    p_crowd.add_argument("--rounds", type=int, default=_CROWD_DEFAULTS.DEFAULT_ROUNDS)
    p_crowd.add_argument("--model", default=_CROWD_DEFAULTS.DEFAULT_MODEL)
    p_crowd.add_argument(
        "--recsys",
        default=_CROWD_DEFAULTS.DEFAULT_RECSYS,
        help='recommender: "twhin-bert" (interest-based) or "twitter" (light)',
    )
    p_crowd.add_argument("--seed", type=int, default=0)
    p_crowd.add_argument(
        "--host", action="store_true", help="run the app on the host (not a container)"
    )
    p_crowd.add_argument(
        "--no-llm",
        action="store_true",
        help="scripted wiring smoke: ManualActions only, no LLM spend",
    )
    p_crowd.add_argument("--no-interview", action="store_true")
    p_crowd.add_argument(
        "--no-validity-gate",
        action="store_true",
        help="skip the verify_code gate (faster, but leaves the run unverified)",
    )
    p_crowd.add_argument(
        "--out",
        default=None,
        help="artifact output directory. Its NAME must start with "
        "'<build_id>__', because that prefix is the only thing joining a run "
        "to its build: scoring by build id, the viewers, and the rubric's "
        "ViralScore comparison all find runs by globbing for it. Omit this and "
        "a conforming name is generated",
    )
    p_crowd.add_argument(
        "--timeout",
        type=float,
        default=_CROWD_DEFAULT_TIMEOUT_S,
        help="wall clock in seconds for the whole run (default: "
        f"{_CROWD_DEFAULT_TIMEOUT_S:.0f}). A run killed here writes no summary "
        "and cannot be scored, so lower it only for a deliberately small run",
    )
    p_crowd.add_argument("--personas", default=None, help="override persona CSV path")
    p_crowd.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="crowd sampling temperature; lower is more repeatable, higher is "
        "livelier. A real lever on the reliability/richness tradeoff.",
    )
    p_crowd.add_argument(
        "--trial-max-steps",
        type=int,
        default=None,
        help="hard cap on interaction steps in one trier's trial. The trier's "
        "total tool budget is derived from this plus social headroom, so a "
        "thorough trial can never silence the agent.",
    )
    p_crowd.add_argument("--thinking-level", default="", help="low | high")
    p_crowd.add_argument(
        "--semaphore",
        type=int,
        default=None,
        help="max concurrent LLM requests inside one run; lower it when the "
        "crowd model is rate-limited, because a throttled turn is recorded as "
        "an agent choosing to do nothing",
    )
    p_crowd.add_argument("--min-interactions", type=int, default=None)
    p_crowd.add_argument("--disable-tools", default="")
    p_crowd.add_argument("--no-env-notice", action="store_true")
    p_crowd.add_argument("--follow-peers", type=int, default=None)
    p_crowd.add_argument("--feed-max-posts", type=int, default=0)
    p_crowd.add_argument(
        "--variant",
        default="",
        help="tag this run as a named instrument variation (ablation), so it "
        "records arch <version>+<variant> and never pools with the main sweep",
    )
    p_crowd.set_defaults(func=_cmd_crowd_run)

    p_grade = sub.add_parser(
        "grade",
        help="grade a build against its rubric (RubricScore)",
        description=(
            "Grade one build against its idea's rubric and print the result item "
            "by item. This is the single-build front door to the same grader "
            "scripts/rubric_sweep.py runs over a whole cohort."
        ),
    )
    p_grade.add_argument("build_id")
    p_grade.add_argument(
        "-m",
        "--grader-model",
        required=True,
        help=(
            "the model that grades, as 'provider/model'. Required: there is no "
            "default grader, and this id is recorded in the grade as the only "
            "account of which judge produced the number."
        ),
    )
    p_grade.add_argument(
        "--passes",
        type=int,
        default=_RUBRIC_DEFAULT_PASSES,
        help="independent grading passes; the majority verdict wins",
    )
    p_grade.add_argument(
        "--host",
        action="store_true",
        help="run the app on the host (not a container)",
    )
    p_grade.add_argument("--json", action="store_true", help="emit the grade as JSON")
    p_grade.add_argument(
        "--no-write", action="store_true", help="do not persist the grade"
    )
    p_grade.set_defaults(func=_cmd_grade)

    p_score = sub.add_parser(
        "score",
        help="compute the ViralScore for a crowd run",
        description=(
            "Turn a crowd run's artifacts into a 0-100 ViralScore. Pure offline "
            "function of run_summary.json + simulation.db, so re-scoring an "
            "existing run is free."
        ),
    )
    p_score.add_argument(
        "target", help="a build_id (scores its latest crowd run) or a crowd run dir"
    )
    p_score.add_argument(
        "--all", action="store_true", help="score every crowd run for the build"
    )
    p_score.add_argument("--json", action="store_true", help="emit JSON only")
    p_score.add_argument(
        "--autorate",
        action="store_true",
        help="run the agentic autorater over the crowd's trajectory (LLM calls)",
    )
    p_score.add_argument(
        "--evidence", action="store_true", help="also write evidence.json"
    )
    p_score.add_argument(
        "--profile",
        default=None,
        help="weight profile from config/score.yaml (default: active_profile)",
    )
    p_score.add_argument(
        "--no-write", action="store_true", help="do not write score.json"
    )
    p_score.set_defaults(func=_cmd_score)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # Preserve original behavior: `viral-bench` runs the sample benchmark.
        raise SystemExit(_cmd_bench(args))
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main(sys.argv[1:])
