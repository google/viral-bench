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

"""Reading a value from the process environment, then the repo ``.env``.

Deliberately the lowest layer in the package: it imports nothing from
``viral_bench``. Credential resolution needs it, and so does the founder
subpackage, so anything it depended on would become a cycle.

Precedence is process environment first, then the repo ``.env``, so a value can
be added to ``.env`` with no exports and still be overridden for one command.
An empty value in either source counts as absent -- ``FOO=`` in a ``.env`` is
how you comment a variable out, not how you set it to the empty string.
"""

from __future__ import annotations

import os
from pathlib import Path

# env.py lives at src/viral_bench/env.py -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def env_file() -> Path:
    """Path to the repo ``.env`` (overridable via ``VIRAL_BENCH_ENV_FILE``)."""
    override = os.environ.get("VIRAL_BENCH_ENV_FILE")
    return Path(override) if override else _REPO_ROOT / ".env"


def parse_env_file(path: Path) -> dict[str, str]:
    """Minimal ``KEY=VALUE`` .env parser (skips blanks/comments, strips quotes).

    Hand-rolled rather than python-dotenv because the crowd runs in a separate,
    deliberately minimal virtualenv, and a credential lookup failing on a
    missing dependency there would be a poor trade for the few features of a
    full parser that this file does not use.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key:
            values[key] = val.strip().strip('"').strip("'")
    return values


def read_key(name: str, *, env_file_path: Path | None = None) -> str | None:
    """Return a var's value from the process env, else the repo ``.env``."""
    val = os.environ.get(name)
    if val:
        return val
    return parse_env_file(env_file_path or env_file()).get(name) or None
