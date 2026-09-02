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

"""Per-build workspaces: fresh host directories for founder builds.

A build never runs inside a container. The founder harness (opencode) runs on
the *host* and writes the app into a fresh directory tree; this module owns that
directory. Each build gets one clean ``builds/work/<build_id>/`` so no state
leaks between runs (the design doc's Per-Run State Isolation requirement).

This is *directory* isolation only -- a convention about where cooperative code
writes, not a security boundary. Running and testing the *built* app safely
(where real, kernel-enforced containment matters) is a separate concern handled
by :mod:`viral_bench.founder.runtime`.

Directory layout (all under a gitignored ``builds/`` at the repo root)::

    builds/
      work/<build_id>/
        app/           <- the app the founder writes (this is what ships)
        transcript/    <- opencode JSON transcripts
        build.json     <- build record
      store/           <- single git repo; one orphan branch per build
      runs/            <- ephemeral clones used to run/test built apps
      data/<build_id>/ <- an app's own mutable state (its database), which
                          OUTLIVES the ephemeral run dirs above
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def builds_root() -> Path:
    """Return the root directory for all build artifacts (gitignored).

    Overridable via ``VIRAL_BENCH_BUILDS_DIR`` (used by tests).
    """
    override = os.environ.get("VIRAL_BENCH_BUILDS_DIR")
    return Path(override) if override else _REPO_ROOT / "builds"


def build_data_dir(build_id: str) -> Path:
    """Return (creating it) the durable data directory for one build's app.

    This is where a server-side app keeps the state it owns -- its SQLite file,
    uploads, anything it must still have after a restart. It is mounted into the
    container at ``/data`` and exported to host runs as ``VIRALBENCH_DATA_DIR``.

    It deliberately lives OUTSIDE ``builds/runs/<run_id>/``. A run dir is a
    throwaway clone that :meth:`AppSession.close` deletes, and the crowd's shared
    :class:`~viral_bench.founder.apphost.AppHost` re-materializes a fresh one
    whenever an instance dies -- so a database kept under the app dir is wiped by
    any restart, silently and mid-run. Keeping state here is what makes "agent A
    writes, agent B reads" survive that.

    The lifetime is one crowd run: :func:`reset_build_data` is called when the
    run's ``AppHost`` is created, so state accumulates across the agents *within*
    a run (that is the multi-user signal) and never leaks *between* runs (that
    would make runs non-comparable).
    """
    path = builds_root() / "data" / build_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_build_data(build_id: str) -> Path:
    """Delete and recreate a build's data directory, returning the empty dir."""
    path = builds_root() / "data" / build_id
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


class BuildWorkspace:
    """A fresh host directory tree for one founder build (no container)."""

    def __init__(self, build_id: str, root: Path | None = None) -> None:
        self.build_id = build_id
        self.root = root or (builds_root() / "work" / build_id)

    @property
    def app_dir(self) -> Path:
        """Directory the founder writes the app into (and that ships)."""
        return self.root / "app"

    @property
    def transcript_dir(self) -> Path:
        """Directory the harness writes opencode transcripts into."""
        return self.root / "transcript"

    def create(self) -> BuildWorkspace:
        """Create the (fresh) directory tree. Fails if it already exists."""
        if self.root.exists():
            raise FileExistsError(f"workspace already exists: {self.root}")
        self.app_dir.mkdir(parents=True)
        self.transcript_dir.mkdir(parents=True)
        self.init_git_boundary()
        return self

    def init_git_boundary(self) -> bool:
        """Give the workspace its own git repo. Returns False if git is absent.

        The build runs on the host (see the module docstring) with the workspace
        nested inside the ViralBench checkout, so a bare ``git`` run from the app
        dir walks up the tree and resolves to *our* repo. Agents do run git --
        they are handed an unrestricted shell, and a built app may commit on
        purpose -- so without a nearer repo their commits, ``add -f`` and
        ``reset`` all land in ViralBench's own history.

        Owning a repo here makes the workspace the nearest worktree, so that
        activity is captured by a directory we throw away.

        Placed at the workspace root, not in ``app_dir``, on two counts. It must
        sit above ``.opencode/skills`` (written by ``OpenCodeRunner.prepare``),
        which opencode locates by walking up from ``--dir`` to the worktree
        boundary -- an ``app_dir/.git`` would cut that walk short and hide the
        role skills. And only ``app_dir`` ships, so a ``.git`` up here cannot
        leak into a built app.

        This is containment against accident, not a sandbox: an agent that cds
        out of the workspace is still outside it. Real isolation would mean
        running the build itself in a container, which
        :mod:`viral_bench.founder.runtime` deliberately does not do.

        Best-effort -- git missing or failing must never fail a build.
        """
        if shutil.which("git") is None:
            return False
        commands = [
            ["git", "init", "-q", "-b", "main", str(self.root)],
            # A throwaway identity of its own: agent commits must not be
            # attributed to the developer. A built app has previously run
            # `git config user.name` with no --global against the enclosing
            # checkout, silently re-authoring every later commit there.
            ["git", "-C", str(self.root), "config", "user.name", "ViralBench Build"],
            [
                "git",
                "-C",
                str(self.root),
                "config",
                "user.email",
                "build@viralbench.invalid",
            ],
        ]
        for argv in commands:
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                return False
            if proc.returncode != 0:
                return False
        return True

    def describe(self) -> str:
        """Human-readable one-liner about this workspace."""
        return f"host build dir {self.app_dir}"
