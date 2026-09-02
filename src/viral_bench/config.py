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

"""Load the documented runtime knobs from ``config/*.yaml``.

The YAML files under ``config/`` are the human-facing description of the
defaults. This module makes them the *source of truth* for the handful of knobs
the code reads at import time -- the founder model + Vertex target, and the
crowd model + sizes -- so changing them no longer requires editing code
(removing the old "values here mirror the code and wiring a loader is a
follow-up" drift).

It is deliberately defensive: every getter takes the code default and falls back
to it if PyYAML or the file is missing/malformed, so importing this never breaks a
run (including the crowd's isolated ``.venv-crowd``). Environment overrides (e.g.
``VERTEX_PROJECT``) are applied by the callers and take precedence over YAML.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

# config.py lives at src/viral_bench/config.py -> parents[2] is the repo root.
_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@functools.cache
def _load(name: str) -> dict:
    """Parse one config YAML into a dict, returning {} on any problem."""
    try:
        import yaml

        with (_CONFIG_DIR / name).open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - config is best-effort, so fall back to code
        return {}


def _get(name: str, path: tuple[str, ...], default: Any) -> Any:
    """Read a nested key from a config file, else ``default``."""
    cur: Any = _load(name)
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# -- which model each stage uses ---------------------------------------------- #

#: The four stages that need a model. ``config/local.yaml`` (written by
#: ``viral-bench init``, gitignored) is consulted first so a user's own choices
#: survive an upgrade and never end up in a commit. The tracked ``config/*.yaml``
#: are the fallback, and ship every stage empty.
_STAGE_FALLBACK: dict[str, tuple[str, tuple[str, ...]]] = {
    "founder": ("founder.yaml", ("model", "id")),
    "crowd": ("crowd.yaml", ("simulation", "model", "id")),
    "grader": ("founder.yaml", ("grader", "model")),
    "autorater": ("score.yaml", ("autorater", "model")),
    "app": ("runtime.yaml", ("app", "model")),
}


def stage_model(stage: str, default: str = "") -> str:
    """The ``provider/model`` configured for one stage, or ``default``.

    Empty is a legitimate answer and the shipped state: ViralBench presumes no
    provider, so every stage starts unset and ``viral-bench init`` fills them in.
    Callers must handle "" rather than substituting a model of their own.
    """
    local = _get("local.yaml", ("models", stage), None)
    if local:
        return str(local)
    where = _STAGE_FALLBACK.get(stage)
    if where is None:
        return default
    raw = _get(where[0], where[1], None)
    return str(raw) if raw else default


def stage_models() -> dict[str, str]:
    """Every stage's configured model, for ``viral-bench doctor``."""
    return {stage: stage_model(stage) for stage in _STAGE_FALLBACK}


# -- founder pipeline -------------------------------------------------------- #


def founder_model_id(default: str) -> str:
    """Founder model, as a ``provider/model`` string.

    Returned verbatim because the prefix is what selects the provider, and
    :mod:`viral_bench.founder.models` validates whatever comes back.
    """
    return stage_model("founder", default)


def vertex_project(default: str) -> str:
    """Vertex project from ``founder.yaml`` ``vertex.project``."""
    return str(_get("founder.yaml", ("vertex", "project"), default))


def vertex_location(default: str) -> str:
    """Vertex location from ``founder.yaml`` ``vertex.location``."""
    return str(_get("founder.yaml", ("vertex", "location"), default))


def founder_collaboration(key: str, default: Any) -> Any:
    """One value from ``founder.yaml`` ``collaboration``.

    This block documented the founder's shape (agents / rounds / min_rounds /
    toolset / browser_tools) for a long time without being loaded by anything,
    so the CLI defaults were the real source of truth and the file could
    disagree with them silently -- ``browser_tools: false`` here against a code
    default of ``True`` was exactly that.
    """
    return _get("founder.yaml", ("collaboration", key), default)


def founder_timeout(key: str, default: float | None) -> float | None:
    """One value from ``founder.yaml`` ``timeouts_seconds``.

    ``null`` (or a non-positive number) means no wall-clock limit at all. These
    are a deadlock backstop, not a compute budget: a turn killed part-way is
    recorded as a build failure, so a limit low enough to bite turns *slowness*
    into a fabricated capability difference between models.
    """
    value = _get("founder.yaml", ("timeouts_seconds", key), default)
    if value is None:
        return None
    parsed = _as_float(value, 0.0)
    return parsed if parsed > 0 else None


def founder_trajectory(key: str, default: bool) -> bool:
    """One boolean from ``founder.yaml`` ``trajectory``.

    Controls how much of a founder build is recorded for later replay: the
    model's own reasoning, and the full opencode session store (which is the
    only place a subagent's work and the prompts it was given are written down).
    """
    value = _get("founder.yaml", ("trajectory", key), default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(default) if value is None else bool(value)


# -- crowd simulation -------------------------------------------------------- #


def crowd_model_id(default: str = "") -> str:
    """Crowd model, as a ``provider/model`` string (empty when unset)."""
    return stage_model("crowd", default)


def crowd_agents(default: int) -> int:
    return _as_int(_get("crowd.yaml", ("simulation", "agents"), default), default)


def crowd_triers(default: int) -> int:
    return _as_int(_get("crowd.yaml", ("simulation", "triers"), default), default)


def crowd_latecomers(default: int) -> int:
    return _as_int(_get("crowd.yaml", ("simulation", "latecomers"), default), default)


def crowd_rounds(default: int) -> int:
    return _as_int(_get("crowd.yaml", ("simulation", "rounds"), default), default)


def crowd_seed(default: int) -> int:
    return _as_int(_get("crowd.yaml", ("simulation", "seed"), default), default)


def crowd_interview(default: bool) -> bool:
    value = _get("crowd.yaml", ("simulation", "interview"), default)
    return value if isinstance(value, bool) else default


def crowd_temperature(default: float) -> float:
    return _as_float(
        _get("crowd.yaml", ("simulation", "model", "temperature"), default), default
    )


def crowd_max_tokens(default: int | None = None) -> int | None:
    """Output-token cap for crowd calls. ``None`` means send no cap at all.

    Returns ``None`` when the key is absent, ``null``, or non-positive, so the
    "no artificial ceiling" case is expressible in YAML rather than being a
    magic large integer.
    """
    value = _get("crowd.yaml", ("simulation", "model", "max_tokens"), default)
    if value is None:
        return None
    parsed = _as_int(value, 0)
    return parsed if parsed > 0 else None


def crowd_semaphore(default: int) -> int:
    return _as_int(
        _get("crowd.yaml", ("simulation", "model", "semaphore"), default), default
    )


def crowd_trial_max_steps(default: int) -> int:
    return _as_int(_get("crowd.yaml", ("interaction", "max_steps"), default), default)


def crowd_social_headroom(default: int) -> int:
    return _as_int(
        _get("crowd.yaml", ("simulation", "social_headroom"), default), default
    )


def crowd_max_iteration_reactor(default: int) -> int:
    return _as_int(
        _get("crowd.yaml", ("simulation", "max_iteration_reactor"), default), default
    )


def crowd_start_wait(default: float) -> float:
    return _as_float(
        _get("crowd.yaml", ("interaction", "start_wait_s"), default), default
    )


def crowd_min_interactions(default: int) -> int:
    return _as_int(
        _get("crowd.yaml", ("simulation", "min_interactions"), default), default
    )


def crowd_env_notice(default: bool) -> bool:
    value = _get("crowd.yaml", ("interaction", "env_notice"), default)
    return value if isinstance(value, bool) else default


def crowd_follow_peers(default: int) -> int:
    return _as_int(_get("crowd.yaml", ("simulation", "follow_peers"), default), default)


def crowd_feed(key: str, default: int) -> int:
    """One size from ``crowd.yaml`` ``simulation.feed:``.

    The feed is where reach is either earned or handed out, so its sizes belong
    in config rather than buried as dataclass defaults no experiment can reach.
    """
    return _as_int(_get("crowd.yaml", ("simulation", "feed", key), default), default)


def crowd_browser(key: str, default: Any) -> Any:
    """One value from ``crowd.yaml`` ``browser:``."""
    return _get("crowd.yaml", ("browser", key), default)


# -- scoring ----------------------------------------------------------------- #


def score_config() -> dict:
    """The whole of ``config/score.yaml`` (empty dict if absent/malformed)."""
    return _load("score.yaml")


def score_profile(name: str | None = None) -> dict:
    """One named weight profile, or the configured ``active_profile``.

    Returns ``{}`` when the profile is unknown, so callers fall back to the
    code defaults rather than scoring with a half-applied configuration.
    """
    cfg = score_config()
    name = name or cfg.get("active_profile")
    profiles = cfg.get("profiles") or {}
    profile = profiles.get(name)
    return dict(profile, name=name) if isinstance(profile, dict) else {}


def score_profile_names() -> list[str]:
    return sorted((score_config().get("profiles") or {}).keys())


def autorater_config() -> dict:
    """Autorater settings (model, temperature, repeats, blinding).

    ``model`` is overlaid from :func:`stage_model` so ``viral-bench init`` can
    set it without editing the tracked ``score.yaml``.
    """
    cfg = score_config().get("autorater")
    cfg = dict(cfg) if isinstance(cfg, dict) else {}
    chosen = stage_model("autorater")
    if chosen:
        cfg["model"] = chosen
    return cfg


def score_minimum(key: str, default: int) -> int:
    """One value from ``score.yaml`` ``minimums``.

    Formerly documented in the YAML as "NOT YET WIRED". It is wired now, so the
    file no longer advertises knobs that do nothing.
    """
    return _as_int(_get("score.yaml", ("minimums", key), default), default)
