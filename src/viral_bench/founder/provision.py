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

"""Give a shipped build its dependencies back, once, and cheaply thereafter.

THE PROBLEM. ``materialize_build`` prefers a clean ``git clone`` of the shipped
branch over copying the founder's work tree, and that is correct -- it is what the
world would get. But it means the app arrives with only what the founder committed,
while the founder built and tested against a host that had far more: a ``.venv`` its
own ``.gitignore`` excludes, a ``node_modules`` likewise, or simply a ``pip install``
it ran in its shell and never wrote down. The container then starts an app whose
imports cannot resolve, no agent can open it, and the run dies at the wall clock
having produced nothing.

Three separate mechanisms produced that, and only one of them was ever diagnosed:

1. **Gitignored dependency directories.** The known one. The clone has no
   ``node_modules``/``.venv`` and the manifest declares no step to recreate them.
2. **Undeclared dependencies.** The app imports ``fastapi`` or ``PIL`` and nothing
   anywhere says so -- no ``requirements.txt``, no setup step. Measured on the r3
   corpus, this was the single largest class.
3. **`pip install` in a setup step being silently discarded.** ``ContainerRuntime``
   runs setup in ``podman run --rm``, where only ``/work`` and ``/data`` are bind
   mounts. ``npm install`` survives because it writes ``./node_modules`` under
   ``/work``; ``pip install`` writes to the image's site-packages and vanishes with
   the container. So a founder that *correctly declared its install* was punished
   exactly as hard as one that declared nothing.

THE FIX, in one sentence: give each build a dependency root that lives on the host,
is bind-mounted at a FIXED absolute path into every container, and is populated once.

    builds/depcache/<build_id>/<inputs-hash>/venv          -> /deps/venv
    builds/depcache/<build_id>/<inputs-hash>/node/<slug>   -> /work/<dir>/node_modules

``/deps/venv/bin`` goes first on ``PATH``, so ``python``, ``python3`` and ``pip`` all
resolve into the cache. That single property fixes mechanism 3 for free: a manifest's
own ``pip install`` now lands in a bind mount and is still there when the app starts.

The path is absolute and constant on purpose. A virtualenv bakes its own prefix into
``pyvenv.cfg`` and into the shebang of every console script it installs, so it is
relocatable only if the path never moves. ``/deps/venv`` is identical in the
provisioning container and in every run container afterwards.

Rejected alternatives, and why:

* **Copy ``node_modules`` beside the clone.** Restores the ~188k-file-per-run copy
  that commit ``3d36e2b`` removed; that cost took sweep throughput from ~90/h to ~2/h.
* **Bind-mount the host's ``node_modules`` read-only.** O(1), but native modules are
  ABI-bound to the host that built them -- observed directly as ``better_sqlite3``
  built for ``NODE_MODULE_VERSION 127`` against a container wanting 115. Installing
  *inside* the container is what makes the cache ABI-correct.
* **Synthesize an install step on every run.** Correct but pays the install on all
  ~3,000 runs of a sweep instead of once per build.

At run time the cache is mounted ``:O`` (podman overlay) rather than read-only, so a
framework that writes inside its own dependency tree -- ``node_modules/.vite`` is the
common one -- still works, and the write is discarded with the container. Read-only
would turn a working app into a crash.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from viral_bench.founder.manifest import MANIFEST_FILENAME, Manifest, load_manifest
from viral_bench.founder.workspace import builds_root

__all__ = [
    "CONTAINER_VENV",
    "ProvisionPlan",
    "cache_mounts",
    "depcache_enabled",
    "is_install_command",
    "neutralize_pruning",
    "container_env",
    "depcache_root",
    "plan_for",
    "provision_build",
]

#: Bumped whenever provisioning itself changes what it would install, so an
#: existing cache built by an older version is not silently reused.
#:
#: BUMP IT WHEN YOU EDIT THE MAPPING TABLES, not only when you edit the planning
#: code. Adding the framework extras below without bumping left a warm cache
#: holding bare `fastapi`, so the very build the extras were added for kept
#: failing and looked like the fix had not worked.
PROVISION_VERSION = "7"

#: Written only after a cache has been fully populated, and the single source of
#: truth for "is this cache usable?". The cache DIRECTORY's existence is not: it
#: is created before the populating container starts, and the files a naive
#: check looks for (``pyvenv.cfg``, a non-empty ``node_modules``) both appear
#: early in that container's life.
STAMP_NAME = ".provisioned.json"

#: Where the dependency root is mounted inside the container. Absolute and fixed --
#: see the module docstring for why it may never become per-run.
CONTAINER_DEPS = "/deps"
CONTAINER_VENV = f"{CONTAINER_DEPS}/venv"

#: Memory for a provisioning container. Larger than the app's runtime cap because
#: it runs bundlers: measured, markdown_slides died in `npm run build` with
#: "JavaScript heap out of memory" under 1g.
PROVISION_MEMORY = "4g"

#: Files whose contents decide what a build needs. Hashed into the cache key, so a
#: build whose lockfile changes gets a fresh cache rather than a stale one.
_DEP_FILES = (
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
)

#: Import name -> distribution name, for the cases where they differ. Only the ones
#: this corpus actually produces; an unknown import is installed under its own name,
#: which is right far more often than it is wrong and costs one failed resolve when
#: it is not.
_IMPORT_TO_DIST = {
    # Frameworks are installed WITH their batteries-included extra, because the
    # app reaches their optional dependencies through the framework's own API
    # rather than by importing them, so an import scan cannot see the need. Bare
    # `fastapi` produced three separate failures of exactly this shape:
    # `Jinja2Templates` needs jinja2, `Form(...)` needs python-multipart, and
    # neither name appears anywhere in the app's source.
    "fastapi": "fastapi[standard]",
    "starlette": "starlette[full]",
    "uvicorn": "uvicorn[standard]",
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "jwt": "pyjwt",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "sklearn": "scikit-learn",
    "serial": "pyserial",
    "OpenSSL": "pyopenssl",
    "magic": "python-magic",
    "fitz": "pymupdf",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "attr": "attrs",
    "google": "google-genai",
    "multipart": "python-multipart",
    "jose": "python-jose",
    "socketio": "python-socketio",
    "engineio": "python-engineio",
    "psycopg2": "psycopg2-binary",
    "markdown_it": "markdown-it-py",
    "PyPDF2": "pypdf2",
    "skimage": "scikit-image",
}

#: Distributions already in the runtime image. Installing them again is a waste and
#: can downgrade what the image pinned.
_IMAGE_PROVIDED = {"pip", "setuptools", "wheel"}

_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+([A-Za-z_][\w.]*)|from\s+([A-Za-z_][\w.]*)\s+import)",
    re.MULTILINE,
)

#: Marks a setup command that installs things, so provisioning can tell an install
#: step apart from a migration step. A manifest whose only setup is
#: ``python3 migrate.py`` declares no dependencies at all, which is precisely the
#: case that used to be indistinguishable from a build with nothing to install.
_INSTALL_HINTS = (
    "npm i",
    "npm ci",
    "npm install",
    "yarn",
    "pnpm",
    "bun install",
    "uv sync",
    "uv pip",
    "uv add",
    "pip install",
    "pip3 install",
    "poetry install",
    "python -m venv",
    "python3 -m venv",
    "uv venv",
)

#: Server runners named on a COMMAND LINE rather than in an import.
#:
#: An import scan cannot see these and it is the commonest remaining hole: an app
#: whose manifest runs ``uv run uvicorn app.main:app`` imports ``fastapi`` in its
#: source and ``uvicorn`` nowhere at all, so provisioning installed the framework
#: and left out the thing that serves it. Measured directly -- the app then failed
#: with ``Failed to spawn: uvicorn`` after a successful install.
_COMMAND_TOOLS = {
    "uvicorn": "uvicorn[standard]",
    "gunicorn": "gunicorn",
    "hypercorn": "hypercorn",
    "daphne": "daphne",
    "waitress-serve": "waitress",
    "streamlit": "streamlit",
    "fastapi": "fastapi[standard]",
    "flask": "flask",
    "celery": "celery",
    "alembic": "alembic",
}

#: ``python -m <module>`` names a dependency just as an import does.
_DASH_M_RE = re.compile(r"python[0-9.]*\s+-m\s+([A-Za-z_][\w.]*)")


#: Builds this PROCESS has already settled, so a sweep does not re-clone a build
#: once per cell just to rediscover that its cache is warm.
_PROVISIONED: set[str] = set()
_MEMO_LOCK = threading.Lock()


def depcache_root() -> Path:
    """Root for per-build dependency caches (gitignored, rebuildable)."""
    return builds_root() / "depcache"


@dataclass
class ProvisionPlan:
    """What one build needs installed, and where the result is cached."""

    build_id: str
    cache_dir: Path
    #: Directories (relative to the app root) that hold a ``package.json``.
    node_dirs: list[str] = field(default_factory=list)
    #: Shell commands to run inside the container, in order.
    steps: list[str] = field(default_factory=list)
    #: True when a Python environment is wanted at all.
    wants_venv: bool = False

    @property
    def empty(self) -> bool:
        return not self.steps


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _slug(rel: str) -> str:
    """A filesystem-safe name for a subdirectory path ('.' -> 'root')."""
    cleaned = rel.strip("./")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", cleaned) or "root"


def _package_dirs(app_dir: Path) -> list[str]:
    """Relative dirs holding a ``package.json``, app root first, depth <= 2.

    Depth 2 covers the ``frontend/`` + ``backend/`` split the corpus actually
    produces without walking into ``node_modules`` itself, which would find
    thousands of nested manifests and mount a cache over each one.
    """
    found: list[str] = []
    if (app_dir / "package.json").is_file():
        found.append(".")
    skip = {"node_modules", ".venv", ".git", "venv", "__pycache__", "dist", "build"}
    for child in sorted(app_dir.iterdir()) if app_dir.is_dir() else []:
        if not child.is_dir() or child.name in skip or child.name.startswith("."):
            continue
        if (child / "package.json").is_file():
            found.append(child.name)
    return found


#: Directories that never hold the app's own source, pruned from every walk.
_SKIP_DIRS = frozenset(
    {"node_modules", ".venv", "venv", ".git", "__pycache__", "tests", ".pytest_cache"}
)


def _iter_dirs(app_dir: Path):
    try:
        return [p for p in app_dir.iterdir() if p.is_dir()]
    except OSError:
        return []


def _python_imports(app_dir: Path) -> set[str]:
    """Third-party top-level imports the app's own Python files make.

    The fallback for the largest failure class: an app that imports ``fastapi``
    while nothing on disk says it depends on anything. Deliberately a text scan
    rather than an AST parse -- some shipped files do not parse under the container's
    Python, and a dependency list is still worth having for those.
    """
    # os.walk with in-place pruning, NOT rglob. 236 builds in this corpus commit
    # their node_modules, and rglob walks all ~188,600 files before the filter
    # gets to reject them -- twice, since the local-module set was a second pass.
    # Measured: that alone put ~90s on every such build's provisioning.
    found: set[str] = set()
    py_files: list[Path] = []
    local = {p.name for p in _iter_dirs(app_dir)}
    for dirpath, dirnames, filenames in os.walk(app_dir):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                py_files.append(Path(dirpath) / name)
    local |= {p.stem for p in py_files}
    for path in py_files:
        # A test file's imports are not the app's runtime dependencies. Without
        # this the crowd's container installs pytest for every build that shipped
        # a test suite -- paid once per build, but still pure waste, and it makes
        # the provisioning log read as though the app needs a test runner to boot.
        if path.name.startswith("test_") or path.stem.endswith("_test"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _IMPORT_RE.finditer(text):
            name = (match.group(1) or match.group(2) or "").split(".")[0]
            if not name or name in local:
                continue
            if name in sys.stdlib_module_names or name.startswith("_"):
                continue
            found.add(name)
    return found


def _command_dists(manifest: Manifest | None) -> set[str]:
    """Distributions named by the manifest's commands rather than by an import.

    Covers the two shapes an import scan is blind to: a server binary invoked
    directly (``uv run uvicorn app.main:app``) and ``python -m <module>``.
    """
    if manifest is None:
        return set()
    commands = [manifest.run.command, *manifest.setup]
    if manifest.test.smoke:
        commands.append(manifest.test.smoke)
    blob = " ".join(commands)
    found: set[str] = set()
    for token, dist in _COMMAND_TOOLS.items():
        # Word-bounded so `flask` does not match `flask_login` in a path, and so
        # a mention inside a longer identifier does not install a whole framework.
        if re.search(rf"(?<![\w.-]){re.escape(token)}(?![\w.-])", blob):
            found.add(dist)
    for match in _DASH_M_RE.finditer(blob):
        module = match.group(1).split(".")[0]
        if module in sys.stdlib_module_names or module in _IMAGE_PROVIDED:
            continue
        found.add(_IMPORT_TO_DIST.get(module, module))
    return found


def neutralize_pruning(command: str) -> str:
    """Stop a setup step from deleting what provisioning just installed.

    `uv sync` makes the environment EXACTLY the project's declared dependencies,
    removing anything else -- so an app whose pyproject.toml is thinner than its
    requirements.txt (or than its actual imports) has its dependencies installed
    and then taken away again, in that order, and fails at start with the very
    module we provisioned. Observed on collaborative_table__...8f4e81: fastapi
    installed from requirements.txt, then pruned by `uv sync`, then
    ModuleNotFoundError.

    `--inexact` keeps extraneous packages. It cannot change what the app itself
    asked for -- everything the project declares is still installed at the
    declared version -- so this narrows the command's blast radius without
    altering its meaning for the app.
    """
    stripped = command.strip()
    if re.match(r"^(uv\s+sync)(\s|$)", stripped) and "--inexact" not in stripped:
        return stripped.replace("uv sync", "uv sync --inexact", 1)
    return command


def is_install_command(command: str) -> bool:
    """True if this setup command exists to install dependencies.

    The distinction decides whether a failing setup step is fatal. An install
    command is redundant once provisioning has run, so its failure says nothing
    about the app; a migration is real work, and its failure does.
    """
    return any(hint in str(command) for hint in _INSTALL_HINTS)


def _declares_install(manifest: Manifest | None) -> bool:
    if manifest is None:
        return False
    return any(is_install_command(cmd) for cmd in manifest.setup)


def _load_manifest_quietly(app_dir: Path) -> Manifest | None:
    try:
        return load_manifest(app_dir / MANIFEST_FILENAME)
    except Exception:  # noqa: BLE001 - an unusable manifest is just "no manifest"
        return None


def plan_for(build_id: str, app_dir: Path) -> ProvisionPlan:
    """Decide what to install for ``app_dir``, and where to cache it.

    ``app_dir`` must be the MATERIALIZED clone, not the founder's work tree: the
    whole problem is that the two differ, and a plan drawn from the work tree would
    conclude the dependencies are already present.
    """
    manifest = _load_manifest_quietly(app_dir)
    node_dirs = _package_dirs(app_dir)
    imports = _python_imports(app_dir)

    steps: list[str] = []
    wants_venv = False

    for rel in node_dirs:
        # Only when the clone lacks them. A build that committed node_modules --
        # 236 of the r3 corpus did -- already ships what it needs, and mounting a
        # cache over that directory would HIDE the very files that make it work.
        target = app_dir / rel / "node_modules"
        if target.is_dir() and any(target.iterdir()):
            continue
        cd = "" if rel == "." else f"cd {rel} && "
        lock = (app_dir / rel / "package-lock.json").is_file()
        # Three attempts, strictest first, because each failure mode is real here.
        #
        # `npm ci` is exact but refuses a lockfile out of sync with package.json,
        # which a hand-edited app hits often. Plain `npm install` then resolves --
        # unless npm 7+ rejects a peer-dependency conflict, which it does by
        # default and which was five of the thirteen remaining failures: a
        # transitive dep pinning react ^15||^16||^17 against the app's react 19.
        # The founder's tree resolved (its lockfile predates the conflict, or its
        # npm was older), so refusing here is our toolchain judging a tree the
        # author never had to argue with. --legacy-peer-deps restores npm 6
        # behaviour: install it and let the app tell us whether it works, which
        # is the question actually being asked.
        attempts = ["npm ci --no-audit --no-fund"] if lock else []
        attempts += [
            "npm install --no-audit --no-fund",
            "npm install --no-audit --no-fund --legacy-peer-deps",
        ]
        steps.append(cd + " || ".join(attempts))

    # Derived from the pruned scan above rather than a fresh rglob: a build that
    # commits node_modules would otherwise pay a second ~188,600-file walk here
    # to answer a question the first walk already answered.
    py_files = bool(imports) or any(
        (app_dir / name).is_file()
        for name in ("main.py", "server.py", "app.py", "setup.py", "wsgi.py", "run.py")
    )
    req_candidates = (
        "requirements.txt",
        "backend/requirements.txt",
        "app/requirements.txt",
    )
    requirements = [rel for rel in req_candidates if (app_dir / rel).is_file()]
    has_pyproject = (app_dir / "pyproject.toml").is_file()

    if py_files or requirements or has_pyproject:
        wants_venv = True
        # --system-site-packages so the image's own libraries stay visible: the
        # cache adds to the runtime rather than replacing it.
        steps.append(
            f"test -x {CONTAINER_VENV}/bin/python || "
            f"python3 -m venv --system-site-packages {CONTAINER_VENV}"
        )
        for rel in requirements:
            # Falls back to installing line by line, because ONE bad line must
            # not cost every other dependency in the file. Measured:
            # collaborative_table's requirements.txt lists `aoisqlite`, a typo for
            # `aiosqlite` that exists nowhere on PyPI -- pip rejects the whole
            # file, and the app then died on a missing `fastapi` that was listed
            # right there and installs fine on its own. The typo is the app's
            # fault; losing fastapi over it was ours.
            steps.append(
                f"pip install --no-input -q -r {rel} || "
                f"grep -vE '^[[:space:]]*(#|$)' {rel} | "
                f"xargs -r -n1 pip install --no-input -q"
            )
        if has_pyproject and not requirements:
            # An app packaged as a project: install it, which pulls its declared
            # dependencies. `|| true` because a malformed pyproject must not block
            # the import fallback below from rescuing the run.
            steps.append(
                "pip install --no-input -q -e . || pip install --no-input -q . || true"
            )
        from_imports = {
            _IMPORT_TO_DIST.get(name, name)
            for name in imports
            if name not in _IMAGE_PROVIDED
        }
        # Installed one at a time, because the list is INFERRED: one bad guess
        # must not take the whole install down with it, which a single
        # `pip install a b c` would.
        if from_imports and not requirements and not has_pyproject:
            # Nothing declared what this app needs, so its own imports are the
            # only statement of intent available.
            for dist in sorted(from_imports):
                steps.append(f"pip install --no-input -q {dist} || true")
        # Command-named tools are added even when a requirements.txt exists: a
        # file that lists `fastapi` and omits `uvicorn` still cannot be served,
        # and re-installing something already satisfied costs a no-op.
        # Compared on the base name so `uvicorn` from an import and
        # `uvicorn[standard]` from the run command are not both installed.
        base = {d.split("[", 1)[0] for d in from_imports}
        for dist in sorted(_command_dists(manifest)):
            if dist.split("[", 1)[0] not in base:
                steps.append(f"pip install --no-input -q {dist} || true")

    # The manifest's own setup steps run last, and now persist. They are included
    # here rather than left to ContainerRuntime.setup so that a `pip install` in a
    # setup step is paid ONCE per build instead of on every one of a sweep's runs.
    if manifest is not None:
        steps.extend(neutralize_pruning(cmd) for cmd in manifest.setup)

    key = _cache_key(app_dir, node_dirs, imports, manifest)
    return ProvisionPlan(
        build_id=build_id,
        cache_dir=depcache_root() / build_id / key,
        node_dirs=node_dirs,
        steps=steps,
        wants_venv=wants_venv,
    )


def _cache_key(
    app_dir: Path,
    node_dirs: list[str],
    imports: set[str],
    manifest: Manifest | None,
) -> str:
    """A digest of everything that decides what would be installed."""
    parts = [f"v{PROVISION_VERSION}"]
    for rel in ("", *node_dirs):
        base = app_dir / rel if rel else app_dir
        for name in _DEP_FILES:
            path = base / name
            if path.is_file():
                try:
                    parts.append(f"{rel}/{name}={_sha(path.read_bytes())}")
                except OSError:
                    continue
    parts.append("imports=" + ",".join(sorted(imports)))
    if manifest is not None:
        parts.append("setup=" + "|".join(manifest.setup))
    return _sha("\n".join(parts).encode())[:16]


#: Set to "0" to run apps exactly as they did before this cache existed.
#:
#: Needed for one specific job: measuring what the cache is worth. A "before"
#: pass that merely skips PROVISIONING still MOUNTS any cache an earlier run left
#: behind, so the baseline quietly reports the fixed behaviour and the
#: measurement destroys itself -- observed directly, with known-bad builds coming
#: up HTTP 200 in a pass that was supposed to reproduce their failure. Also the
#: escape hatch if the cache is ever suspected of changing a result.
DEPCACHE_ENV = "VIRALBENCH_DEPCACHE"


def depcache_enabled() -> bool:
    """False when the dependency cache is switched off for this process."""
    value = os.environ.get(DEPCACHE_ENV, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def cache_mounts(build_id: str, app_dir: Path) -> list[tuple[Path, str, bool]]:
    """``(host_dir, container_path, writable)`` for one build's dependency cache.

    Returns only mounts that exist and hold something, so a build that was never
    provisioned -- or needs nothing -- runs exactly as it did before. ``writable``
    is False at run time, which the runtime renders as podman's ``:O`` overlay:
    the app may write inside its dependency tree and the write is discarded.
    """
    if not depcache_enabled():
        return []
    plan = plan_for(build_id, app_dir)
    if not (plan.cache_dir / STAMP_NAME).is_file():
        return []
    return _mounts_for(plan, app_dir, writable=False)


def _mounts_for(
    plan: ProvisionPlan, app_dir: Path, *, writable: bool
) -> list[tuple[Path, str, bool]]:
    mounts: list[tuple[Path, str, bool]] = []
    venv = plan.cache_dir / "venv"
    # `pyvenv.cfg`, NOT `bin/python`. A virtualenv's interpreter is a SYMLINK to
    # the base interpreter -- here /usr/local/bin/python3, which exists inside the
    # container and nowhere on the host. `Path.exists()` follows symlinks, so
    # probing bin/python from the host reports False for a perfectly good venv and
    # the mount is silently dropped: a fully provisioned build then starts with no
    # dependencies and fails exactly as it did before, which is the most confusing
    # possible outcome. `pyvenv.cfg` is a real file and is always present.
    if writable or (venv / "pyvenv.cfg").is_file():
        mounts.append((venv, CONTAINER_VENV, writable))
    # Some apps hardcode their interpreter as `.venv/bin/uvicorn` rather than
    # relying on PATH, and `uv venv` in their setup step builds that venv inside a
    # clone that is thrown away before the app runs. Mounting the cache there too
    # costs nothing and makes those run commands work as written -- observed on
    # collaborative_table__...34fcc3, whose run command is `.venv/bin/uvicorn`.
    # Skipped when the clone ships its own, for the same reason as node_modules.
    local_venv = app_dir / ".venv"
    if (writable or (venv / "pyvenv.cfg").is_file()) and not local_venv.is_dir():
        mounts.append((venv, "/work/.venv", writable))
    for rel in plan.node_dirs:
        target = app_dir / rel / "node_modules"
        if target.is_dir() and any(target.iterdir()):
            continue  # the clone ships its own; never mount over it
        host = plan.cache_dir / "node" / _slug(rel)
        if not writable and not (host.is_dir() and any(host.iterdir())):
            continue
        where = "/work/node_modules" if rel == "." else f"/work/{rel}/node_modules"
        mounts.append((host, where, writable))
    return mounts


def container_env() -> dict[str, str]:
    """Environment every container needs to see the dependency cache.

    ``PATH`` first is what does the work: it makes ``python``, ``python3`` and
    ``pip`` resolve to the cached virtualenv, so an app started as
    ``python3 server.py`` finds its libraries and a setup step's ``pip install``
    writes somewhere that survives the container.
    """
    return {
        "PATH": f"{CONTAINER_VENV}/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/bin",
        "VIRTUAL_ENV": CONTAINER_VENV,
        # So `uv sync` / `uv pip install` target the same environment instead of
        # creating a per-run `.venv` inside the throwaway clone.
        "UV_PROJECT_ENVIRONMENT": CONTAINER_VENV,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_ROOT_USER_ACTION": "ignore",
        "npm_config_fund": "false",
        "npm_config_audit": "false",
    }


def is_provisioned(build_id: str, app_dir: Path) -> bool:
    """True if this build's cache is already populated for its current inputs."""
    plan = plan_for(build_id, app_dir)
    return (plan.cache_dir / STAMP_NAME).is_file()


@contextlib.contextmanager
def _population_lock(cache_dir: Path):
    """Exclusive inter-process lock for one build's cache directory.

    ``_MEMO_LOCK`` is per-process, so it does nothing once two sweeps share a
    corpus: the crowd and the rubric provision the SAME builds, and without this
    two of them run ``npm install`` and ``pip install`` into one directory at the
    same time.

    The lock file sits BESIDE the cache directory rather than inside it. A lock
    inside would be one more file whose presence a future existence-check could
    mistake for a populated cache -- which is exactly the bug class this area
    already had.
    """
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.parent / f".{cache_dir.name}.lock"
    with open(lock_path, "w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def provision_build(
    build_id: str,
    *,
    timeout: float = 900.0,
    force: bool = False,
    runtime: str = "podman",
    image: str | None = None,
) -> list[str]:
    """Populate one build's dependency cache. Returns the steps that ran.

    Runs INSIDE the container, against a fresh clone, with the cache mounted
    read-write at exactly the paths it will occupy at run time -- so native modules
    match the ABI they will be loaded under, and a virtualenv's baked-in prefix is
    already correct.

    Idempotent, twice over. A build whose cache exists for its current inputs
    returns ``[]`` without starting a container, and a build already handled by
    THIS process returns without even materializing a clone -- which matters
    because a sweep calls this once per (build, seed) cell, so the naive version
    pays three git clones per build for two answers it already has.

    Safe to call before every run, and cheap enough that it should be.
    """
    from viral_bench.founder.runner import materialize_build, retire_dir
    from viral_bench.founder.runtime import DEFAULT_IMAGE

    if not depcache_enabled():
        return []
    with _MEMO_LOCK:
        if build_id in _PROVISIONED and not force:
            return []

    run_dir = materialize_build(build_id)
    app_dir = run_dir / "app"
    try:
        plan = plan_for(build_id, app_dir)
        stamp = plan.cache_dir / STAMP_NAME
        if stamp.is_file() and not force:
            with _MEMO_LOCK:
                _PROVISIONED.add(build_id)
            return []

        # Populate under an inter-process lock. The stamp gate in cache_mounts
        # stops a half-built cache being CONSUMED; this stops two writers
        # building it into each other in the first place.
        with _population_lock(plan.cache_dir):
            # Re-check under the lock: whoever held it before us may have just
            # finished the very cache we queued up to build.
            if stamp.is_file() and not force:
                with _MEMO_LOCK:
                    _PROVISIONED.add(build_id)
                return []
            if plan.empty:
                plan.cache_dir.mkdir(parents=True, exist_ok=True)
                stamp.write_text(
                    json.dumps({"steps": [], "ok": True}), encoding="utf-8"
                )
                return []

            mounts = _mounts_for(plan, app_dir, writable=True)
            for host, _where, _w in mounts:
                host.mkdir(parents=True, exist_ok=True)

            # --memory: provisioning runs bundlers, and a bundler is not the app. See
            # ContainerRuntime.setup_memory for why the app's 1g cap must not apply.
            argv = [
                runtime,
                "run",
                "--rm",
                "--memory",
                PROVISION_MEMORY,
                "-v",
                f"{app_dir}:/work",
            ]
            for host, where, _w in mounts:
                argv += ["-v", f"{host}:{where}"]
            for key, value in container_env().items():
                argv += ["-e", f"{key}={value}"]
            # Installs pull from the network by definition, so the default (outbound
            # on) network is required here whatever a run is later configured with.
            argv += [
                "-w",
                "/work",
                image or DEFAULT_IMAGE,
                "sh",
                "-c",
                # `;` and not `&&`. Provisioning is best-effort dependency SUPPLY, and
                # the authoritative test of whether it worked is the app starting
                # afterwards. Chaining on success meant the first unsatisfiable
                # package discarded every install queued behind it -- two builds died
                # on a missing framework a later step would have provided, because an
                # earlier step hit a typo or a broken pyproject. Order is preserved,
                # which is what matters (install before migrate); a migration that
                # runs too early fails harmlessly and ContainerRuntime.setup runs it
                # again at session start.
                " ; ".join(f"({step})" for step in plan.steps),
            ]

            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, check=False
            )
            plan.cache_dir.mkdir(parents=True, exist_ok=True)
            stamp.write_text(
                json.dumps(
                    {
                        "build_id": build_id,
                        "steps": plan.steps,
                        "returncode": proc.returncode,
                        "ok": proc.returncode == 0,
                        "tail": (proc.stderr or proc.stdout or "")[-4000:],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            # A partial install is still worth keeping: `pip install fastapi` may have
            # succeeded while a later inferred guess failed, and the app needs only the
            # former. The stamp records the returncode so a caller can tell.
            with _MEMO_LOCK:
                _PROVISIONED.add(build_id)
            return list(plan.steps)
    finally:
        retire_dir(run_dir)


#: Ways an app says a Python dependency is missing. The first group is the plain
#: import failure; the second is a library telling you, in prose, which extra of
#: itself you forgot -- FastAPI does this for python-multipart.
_MISSING_MODULE_RES = (
    re.compile(r"No module named ['\"]([A-Za-z_][\w.]*)['\"]"),
    re.compile(r"ModuleNotFoundError: No module named ['\"]([A-Za-z_][\w.]*)['\"]"),
    re.compile(r"requires ['\"]([A-Za-z0-9_.-]+)['\"] to be installed"),
    # "jinja2 must be installed to use Jinja2Templates" -- starlette naming an
    # extra of FastAPI's that the app reaches through the framework's API and so
    # never imports by name. A requirements.txt listing plain `fastapi` cannot
    # express it either, which is why the extras mapping does not catch this one.
    re.compile(r"\b([A-Za-z0-9_.-]+) must be installed"),
    # Quotes and backticks are stripped: pydantic says
    #   email-validator is not installed, run `pip install 'pydantic[email]'`
    # and an unquoted-only pattern misses it, which left two builds failing on an
    # extra the library had already named for us.
    re.compile(r"pip install ['\"`]?([A-Za-z0-9_.\[\]-]+)['\"`]?"),
    re.compile(r"Failed to spawn: `([A-Za-z0-9_.-]+)`"),
)


def missing_dependencies(error_text: str) -> list[str]:
    """Distributions an app's own error message says it is missing.

    The static plan is a good guess and will never be a complete one: an app can
    reach a dependency through a framework's API without naming it anywhere
    (``Jinja2Templates`` needs jinja2, ``Form(...)`` needs python-multipart), and
    no scan of the source can see that. What CAN see it is the app itself, which
    says so precisely when it fails to start.

    So this closes the loop: run it, read the complaint, install what it named,
    run it again. That turns provisioning from a fixed list of guesses into
    something that converges, and it is why the mapping table above only has to
    be good rather than exhaustive.
    """
    found: list[str] = []
    for pattern in _MISSING_MODULE_RES:
        for match in pattern.finditer(error_text or ""):
            name = match.group(1).split(".")[0]
            if not name or name in sys.stdlib_module_names or name in _IMAGE_PROVIDED:
                continue
            dist = _IMPORT_TO_DIST.get(name, name)
            if dist not in found:
                found.append(dist)
    return found


def repair_build(
    build_id: str,
    app_dir: Path,
    error_text: str,
    *,
    timeout: float = 600.0,
    runtime: str = "podman",
    image: str | None = None,
) -> list[str]:
    """Install what an app's start error says it is missing. Returns what it tried.

    Writes into the build's EXISTING cache, so the repair is paid once and every
    later run of that build inherits it. Returns ``[]`` when the error names
    nothing installable, which is the signal that the failure is the app's own.
    """
    from viral_bench.founder.runtime import DEFAULT_IMAGE

    wanted = missing_dependencies(error_text)
    if not wanted:
        return []
    plan = plan_for(build_id, app_dir)
    mounts = _mounts_for(plan, app_dir, writable=True)
    for host, _where, _w in mounts:
        host.mkdir(parents=True, exist_ok=True)

    argv = [runtime, "run", "--rm", "-v", f"{app_dir}:/work"]
    for host, where, _w in mounts:
        argv += ["-v", f"{host}:{where}"]
    for key, value in container_env().items():
        argv += ["-e", f"{key}={value}"]
    # `|| true` per package: the list is derived from an error message, so a name
    # that is not a real distribution must not abort the ones that are.
    script = " ; ".join(
        f"pip install --no-input -q '{dist}' || true" for dist in wanted
    )
    argv += ["-w", "/work", image or DEFAULT_IMAGE, "sh", "-c", script]
    subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return wanted


def clear_cache(build_id: str) -> None:
    """Drop a build's dependency cache entirely (next run reprovisions)."""
    target = depcache_root() / build_id
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)


def cache_size_bytes() -> int:
    """Total bytes under the dependency cache, for reporting."""
    root = depcache_root()
    if not root.is_dir():
        return 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                continue
    return total
