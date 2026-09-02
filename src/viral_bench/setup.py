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

"""First-run setup: ``viral-bench init`` and ``viral-bench doctor``.

ViralBench needs more than an API key to run a real benchmark -- it needs a
container runtime to sandbox generated apps, a browser for the crowd to click
with, Node for the founder's agent CLI, and a second Python environment for the
simulation engine. Discovering those one failure at a time, forty minutes into a
build, is a miserable way to start.

So there are two commands. ``init`` asks which provider you want and writes it
down; ``doctor`` checks everything a run will need and tells you what is missing
and how to fix it, before you spend anything.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from viral_bench.founder.appenv import env_file
from viral_bench.providers import PROVIDERS, ProviderSpec, has_credential

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Where `init` records the model chosen for each stage. Kept separate from the
#: tracked config/*.yaml so an upgrade never overwrites your choices, and a
#: stray `git add -A` never publishes them.
LOCAL_CONFIG = _REPO_ROOT / "config" / "local.yaml"

#: The stages that need a model, and what each one is for.
STAGES: dict[str, str] = {
    "founder": "builds the app -- this is the model under test",
    "crowd": "simulated users who try the app (hold fixed across a comparison)",
    "grader": "inspects the shipped app against the rubric",
    "autorater": "optional qualitative scoring; leave blank to disable",
    "app": "what the BUILT apps call, if an idea needs a model at all",
}


#: Stages a run can legitimately do without. The autorater is an optional
#: scoring component, and an app only needs a model if its idea calls for one.
_OPTIONAL_STAGES = frozenset({"autorater", "app"})


@dataclass
class Check:
    """One doctor check and how to fix it."""

    name: str
    ok: bool
    detail: str = ""
    fix: str = ""
    required: bool = True


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return answer or default


def _providers_with_keys() -> list[ProviderSpec]:
    return [p for p in PROVIDERS.values() if has_credential(p)]


def _write_env(values: dict[str, str]) -> Path:
    """Merge ``values`` into the repo .env, preserving anything already there."""
    path = env_file()
    existing: dict[str, str] = {}
    order: list[str] = []
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            existing[key.strip()] = val.strip()
            order.append(key.strip())
    for key, val in values.items():
        if key not in existing:
            order.append(key)
        existing[key] = val

    body = [
        "# ViralBench credentials. Written by `viral-bench init`.",
        "# This file is gitignored -- never commit real keys.",
        "",
    ]
    body.extend(f"{key}={existing[key]}" for key in dict.fromkeys(order))
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def _write_local_config(models: dict[str, str]) -> Path:
    lines = [
        "# Model chosen for each stage, written by `viral-bench init`.",
        "# Overrides config/*.yaml. A stage left blank has no model.",
        "models:",
    ]
    for stage, value in models.items():
        lines.append(f"  {stage}: {value!r}" if value else f"  {stage}: ''")
    LOCAL_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_CONFIG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return LOCAL_CONFIG


def run_init(*, non_interactive: bool = False) -> int:
    """Walk through choosing a provider and a model for each stage."""
    print("ViralBench setup\n" + "=" * 60)
    print(
        "ViralBench ships no model and no default provider. Bring a key for\n"
        "whichever provider you want to benchmark and it will be used for every\n"
        "stage that needs one.\n"
    )

    found = _providers_with_keys()
    keyed = [p for p in found if p.key_env]
    if keyed:
        print("Credentials already in your environment or .env:")
        for provider in keyed:
            print(f"  - {provider.id:<24} ({provider.key_env})")
    else:
        print("No provider credentials found yet.")
    print()

    if non_interactive:
        if not keyed:
            print(
                "Nothing to do without a credential. Set one (e.g. "
                "OPENAI_API_KEY) and re-run.",
                file=sys.stderr,
            )
            return 1
        chosen = keyed[0]
        print(f"--yes: using {chosen.id}. Set a model per stage in {LOCAL_CONFIG}.")
        return 0

    print("Providers ViralBench can use:")
    listed = list(PROVIDERS.values())
    for index, provider in enumerate(listed, start=1):
        mark = "*" if has_credential(provider) and provider.key_env else " "
        print(f" {mark}{index:>3}. {provider.id:<24} {provider.name}")
    print("      (* = a credential is already set)\n")

    picked = _ask("Which provider? (name or number)")
    if picked.isdigit() and 1 <= int(picked) <= len(listed):
        provider = listed[int(picked) - 1]
    else:
        provider = PROVIDERS.get(picked)
    if provider is None:
        print(f"Unknown provider {picked!r}.", file=sys.stderr)
        return 2

    env_values: dict[str, str] = {}
    if provider.key_env and not has_credential(provider):
        print(f"\n{provider.name} needs {provider.key_env}.")
        print(f"  Get one: {provider.signup_hint}")
        key = _ask(f"Paste your {provider.key_env} (blank to skip)")
        if key:
            env_values[provider.key_env] = key
    if not provider.base_url and not provider.ambient_auth:
        from viral_bench.providers.credentials import base_url_env

        url = _ask(f"Base URL for {provider.name} ({base_url_env(provider)})")
        if url:
            env_values[base_url_env(provider)] = url

    print(
        f"\nNow pick a model id from {provider.name} for each stage.\n"
        f"Enter the bare id -- the '{provider.id}/' prefix is added for you.\n"
    )
    models: dict[str, str] = {}
    default_model = ""
    for stage, purpose in STAGES.items():
        print(f"  {stage}: {purpose}")
        answer = _ask(f"  model for {stage}", default_model)
        if answer:
            models[stage] = answer if "/" in answer else f"{provider.id}/{answer}"
            default_model = default_model or answer
        else:
            models[stage] = ""
        print()

    written = []
    if env_values:
        written.append(_write_env(env_values))
    written.append(_write_local_config(models))

    print("Wrote:")
    for path in written:
        print(f"  {path}")
    print("\nNext: `viral-bench doctor` to check everything else a run needs.")
    return 0


def _which(binary: str) -> str:
    return shutil.which(binary) or ""


def _version(cmd: list[str]) -> str:
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd, capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (
        (out.stdout or out.stderr).strip().splitlines()[0][:60]
        if out.stdout or out.stderr
        else ""
    )


def _tool_checks() -> list[Check]:
    checks: list[Check] = []

    checks.append(
        Check(
            "python",
            sys.version_info >= (3, 12),
            f"{sys.version.split()[0]}",
            "ViralBench needs Python 3.12+",
        )
    )

    node = _which("node")
    checks.append(
        Check(
            "node",
            bool(node),
            _version(["node", "--version"]) if node else "not found",
            "install Node 20+ (https://nodejs.org) -- the founder's agent CLI needs it",
        )
    )

    from viral_bench.founder.harness import find_opencode_binary

    opencode = find_opencode_binary()
    checks.append(
        Check(
            "opencode",
            bool(opencode),
            opencode or "not found",
            "npm i -g opencode-ai   (or run scripts/setup_founder_env.sh)",
        )
    )

    runtime = _which("podman") or _which("docker")
    checks.append(
        Check(
            "container runtime",
            bool(runtime),
            runtime or "not found",
            "install podman or docker -- generated apps are run sandboxed, "
            "never directly on your machine",
        )
    )

    crowd_venv = _REPO_ROOT / ".venv-crowd"
    checks.append(
        Check(
            "crowd environment",
            crowd_venv.is_dir(),
            str(crowd_venv) if crowd_venv.is_dir() else "not created",
            "scripts/setup_crowd_env.sh   (a separate Python 3.11 env for the "
            "simulation engine)",
            required=False,
        )
    )

    chrome = (
        _which("google-chrome")
        or _which("chromium")
        or _which("chromium-browser")
        or _which("google-chrome-stable")
    )
    checks.append(
        Check(
            "chrome",
            bool(chrome),
            chrome or "not found",
            "the crowd drives a real browser; install Chrome/Chromium",
            required=False,
        )
    )
    return checks


def _model_checks(*, probe: bool) -> list[Check]:
    from viral_bench import config as _config
    from viral_bench.providers import (
        MissingCredentialError,
        ModelError,
        UnknownProviderError,
        make_client,
        resolve,
    )

    checks: list[Check] = []
    configured = _config.stage_models()
    for stage in STAGES:
        model = configured.get(stage, "")
        if not model:
            checks.append(
                Check(
                    f"model: {stage}",
                    stage in _OPTIONAL_STAGES,
                    "not set",
                    "viral-bench init",
                    required=stage not in _OPTIONAL_STAGES,
                )
            )
            continue
        try:
            spec = resolve(model)
        except UnknownProviderError as exc:
            checks.append(
                Check(f"model: {stage}", False, str(exc)[:90], "viral-bench init")
            )
            continue
        if not probe:
            checks.append(
                Check(f"model: {stage}", True, f"{spec.qualified} (not probed)")
            )
            continue
        try:
            make_client(spec).ping()
        except (MissingCredentialError, ModelError) as exc:
            checks.append(
                Check(
                    f"model: {stage}",
                    False,
                    str(exc).splitlines()[0][:90],
                    f"check the credential for {spec.provider.id}",
                )
            )
            continue
        checks.append(Check(f"model: {stage}", True, f"{spec.qualified} reachable"))
    return checks


def run_doctor(*, probe_models: bool = True) -> int:
    """Report on everything a real run needs. Returns non-zero if any fail."""
    print("ViralBench doctor\n" + "=" * 60)
    checks = _tool_checks() + _model_checks(probe=probe_models)

    width = max(len(c.name) for c in checks)
    failed_required = 0
    for check in checks:
        if check.ok:
            mark = "ok  "
        elif check.required:
            mark = "FAIL"
            failed_required += 1
        else:
            mark = "warn"
        print(f"[{mark}] {check.name:<{width}}  {check.detail}")
        if not check.ok and check.fix:
            print(f"       {'':<{width}}  -> {check.fix}")

    print()
    if failed_required:
        print(f"{failed_required} required check(s) failed. See the fixes above.")
        return 1
    print("Everything required is in place.")
    if not probe_models:
        print("(--offline: models were not actually called.)")
    return 0


def env_summary() -> dict[str, str]:
    """Which provider credentials are visible, for diagnostics. Never values."""
    return {
        provider.key_env: ("set" if os.environ.get(provider.key_env) else "unset")
        for provider in PROVIDERS.values()
        if provider.key_env
    }
