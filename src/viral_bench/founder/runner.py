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

"""Run a founder-built app so a human or agent can test it.

Testing means different things per app type (the user's requirement):

* ``single-page-app`` -- start the app's server (per its manifest) and hand back
  a URL to open and interact with in a browser.
* ``cli`` -- surface the run/smoke commands to execute in the terminal.
* ``bot`` -- launch its local REPL/simulator entrypoint for offline chatting.

Everything is driven by the app's ``viralbench.json`` manifest and executed
through an :class:`~viral_bench.founder.runtime.AppRuntime`, so the *same* app is
tested identically whether it runs on the host or in a container -- by you now,
or by the OASIS crowd's ``verify_code`` / ``try_app`` tools later.

Isolation of runs: an app is never tested in place in its ``builds/work`` tree.
Each test session first *materializes* a fresh throwaway copy under
``builds/runs/<run_id>/`` (cloned from the shipped git branch when available).
Two testers -- or fifty crowd agents -- therefore never mutate each other's
copy, which is what makes at-scale, parallel crowd testing safe.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from viral_bench.founder.appenv import resolve_app_env
from viral_bench.founder.build import BuildRecord, load_build_record
from viral_bench.founder.manifest import (
    MANIFEST_FILENAME,
    Manifest,
    ManifestError,
    load_manifest,
)
from viral_bench.founder.runtime import (
    AppRuntime,
    LocalRuntime,
    RunningApp,
)
from viral_bench.founder.runtime import (
    AppRuntimeError as RunnerError,  # re-exported: one error type for this layer
)
from viral_bench.founder.workspace import build_data_dir, builds_root

__all__ = [
    "AppSession",
    "RunnerError",
    "describe_build",
    "materialize_build",
    "open_session",
    "runs_root",
    "sweep_run_dirs",
]


def runs_root() -> Path:
    """Root for ephemeral run/test clones (gitignored, throwaway)."""
    return builds_root() / "runs"


def trash_root() -> Path:
    """Holding pen for retired clones, awaiting out-of-band deletion.

    Beside ``builds/runs`` rather than inside it, so anything enumerating the
    runs root does not pick retired clones back up.
    """
    return builds_root() / ".trash"


def retire_dir(path: Path) -> None:
    """Take a directory out of service in O(1), instead of deleting it inline.

    Deleting an app clone inline was the most expensive thing in the whole
    sweep. One app tree is ~188,600 files (``node_modules``), and ``verify_code``
    calls :meth:`AppSession.close` from a ``finally`` on EVERY app open -- so the
    crowd paid a full recursive unlink each time it restarted a dead app
    instance, not once at the end.

    Measured: a py-spy profile of a hung run put 25 of 145 samples inside
    ``shutil.rmtree``, six times the next entry, and ``wchan`` on hung runners
    showed ``filename_unlinkat`` / ``do_get_write_access`` /
    ``ext4_read_bh_lock`` -- ext4 unlink and metadata paths, not the network.
    Crowd throughput decayed from 102 runs/hour to about 2 as ``builds/runs``
    accumulated 1,674 leaked clones, and it was self-reinforcing: the stall
    reaper killed runs mid-rmtree, so that tree leaked too and made the next
    unlink slower.

    ``os.replace`` is O(1) within a filesystem, so the caller returns at once and
    the bytes are reclaimed later by :func:`reclaim_trash`.
    """
    try:
        root = trash_root()
        root.mkdir(parents=True, exist_ok=True)
        os.replace(path, root / f"{path.name}-{uuid.uuid4().hex[:8]}")
    except OSError:
        # Cross-device, a vanished dir, or a permission problem: fall back to the
        # old behaviour rather than leave the clone in the runs root.
        shutil.rmtree(path, ignore_errors=True)


def reclaim_trash(*, budget_s: float = 30.0) -> int:
    """Delete retired clones, spending at most ``budget_s`` doing it.

    Time-bounded on purpose. Reclaiming millions of small files takes far longer
    than the runs take to create them, so an unbounded sweep here would stall the
    caller for minutes. Bounded, it drains steadily in the background and
    ``builds/.trash`` oscillates instead of growing without limit.

    Deletes per entry and never removes the trash directory itself, because
    :func:`retire_dir` renames into it concurrently -- removing the root races
    with those renames and can fail with "Directory not empty" partway through.
    """
    root = trash_root()
    if not root.is_dir():
        return 0
    deadline = time.time() + budget_s
    removed = 0
    for path in root.iterdir():
        if time.time() > deadline:
            break
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


def sweep_run_dirs(*, keep_newest: int = 0, older_than_hours: float = 6.0) -> int:
    """Delete orphaned run clones, returning how many were removed.

    ``AppSession.close`` removes its own run dir, but only on the happy path: a
    crash, a kill, or a start that raised before the session was cached leaks the
    clone. These accumulate silently and are individually large. Measured on this
    checkout before any sweeper existed: 1,412 leaked directories, 36 GB, with
    368 belonging to a single build -- and that is with apps that are plain static
    files. A framework build carrying ``node_modules`` is ~440 MB apiece, which
    turns the same leak rate into hundreds of gigabytes.

    Only directories last modified more than ``older_than_hours`` ago are removed,
    so a concurrent run's live clone is never pulled out from under it.

    THIS EXISTED FOR MONTHS WITH NO CALLERS, and the leak it describes duly
    happened: 1,674 clones accumulated, at which point the inline unlink in
    :meth:`AppSession.close` had taken crowd throughput from 102 runs/hour to
    about 2. A sweeper nothing invokes is a comment. It is now called from
    ``scripts/crowd_sweep.py`` between cells -- if you add another long-running
    driver, call it there too.

    Retires by rename rather than deleting inline, for the reason in
    :func:`retire_dir`. :func:`reclaim_trash` does the deleting itself, under a
    time budget.
    """
    root = runs_root()
    if not root.is_dir():
        return 0
    cutoff = time.time() - (older_than_hours * 3600.0)
    candidates = [p for p in root.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for path in candidates[keep_newest:]:
        try:
            if path.stat().st_mtime > cutoff:
                continue
            retire_dir(path)
            removed += 1
        except OSError:
            continue
    return removed


def _load(build_id: str) -> tuple[BuildRecord, Manifest, Path]:
    """Load a build and its manifest, or raise ``RunnerError``.

    Absent and malformed are the same failure to every caller here -- there is no
    usable contract saying how to run this app -- so both arrive as one error
    type rather than as a hand-rolled ``is_file()`` check beside a ManifestError
    that callers were not catching.
    """
    record = load_build_record(build_id)
    app_dir = Path(record.app_dir)
    try:
        return record, load_manifest(app_dir / MANIFEST_FILENAME), app_dir
    except ManifestError as exc:
        raise RunnerError(
            f"build {build_id!r} has no usable {MANIFEST_FILENAME}: {exc}"
        ) from exc


def _run(argv: list[str]) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RunnerError(
            f"command failed ({' '.join(argv[:3])} ...): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )


def materialize_build(build_id: str) -> Path:
    """Create a fresh, isolated copy of a build's app for one test session.

    Prefers a clean ``git clone`` of the shipped orphan branch, and falls back to
    copying the work tree for builds that were never shipped. Returns the run
    directory, whose ``app/`` subdir holds the materialized app.
    """
    record = load_build_record(build_id)
    run_id = f"{build_id}__run-{uuid.uuid4().hex[:8]}"
    run_dir = runs_root() / run_id
    app_dst = run_dir / "app"
    run_dir.mkdir(parents=True, exist_ok=False)

    if record.shipped_ref and record.store_path:
        _run(
            [
                "git",
                "clone",
                "--quiet",
                "--branch",
                record.shipped_ref,
                "--single-branch",
                record.store_path,
                str(app_dst),
            ]
        )
    else:
        shutil.copytree(record.app_dir, app_dst)
    return run_dir


@dataclass
class AppSession:
    """One isolated run/test session for a build (host or container runtime)."""

    build_id: str
    record: BuildRecord
    manifest: Manifest
    run_dir: Path
    runtime: AppRuntime
    app: RunningApp | None = None
    data_dir: Path | None = None

    @property
    def app_dir(self) -> Path:
        return self.run_dir / "app"

    def setup(self, *, timeout: float | None = None):
        return self.runtime.setup(self.manifest, timeout=timeout)

    def smoke(self, *, timeout: float | None = None):
        # Pass the running app (if any) so a smoke command can probe the app's own
        # port -- in container mode that only works from inside its netns.
        return self.runtime.smoke(self.manifest, timeout=timeout, app=self.app)

    def start(self, *, wait_timeout: float = 90.0) -> RunningApp:
        self.app = self.runtime.start(self.manifest, wait_timeout=wait_timeout)
        return self.app

    def stop(self) -> None:
        if self.app is not None:
            self.app.stop()

    def close(self) -> None:
        """Stop the app and retire the throwaway run directory.

        Deliberately leaves ``data_dir`` alone. The run dir is a disposable clone
        of the code, but the data dir is the app's own state, and the crowd
        restarts a dead instance mid-run -- deleting it here would silently reset
        a database under the agents' feet. Its lifetime is the crowd run, and
        :func:`~viral_bench.founder.workspace.reset_build_data` owns clearing it.

        RETIRED BY RENAME, NOT BY DELETE, because deleting it inline was the most
        expensive thing in the sweep. One app tree is ~188,600 files
        (``node_modules``), and ``verify_code`` calls this from a ``finally`` on
        EVERY app open -- so the crowd paid a full recursive unlink each time it
        restarted a dead app instance, not once at the end.

        Measured: a py-spy profile of a hung run put 25 of 145 samples inside
        ``shutil.rmtree`` here, six times the next entry, and ``wchan`` on hung
        runners showed ``filename_unlinkat`` / ``do_get_write_access`` /
        ``ext4_read_bh_lock`` -- ext4 unlink and metadata paths, not the network.
        Throughput decayed from 102 runs/hour to ~2 as ``builds/runs`` accumulated
        1,674 leaked clones, and it was self-reinforcing: the stall reaper killed
        runs mid-rmtree, so that tree leaked too and made the next one slower.

        ``os.replace`` is O(1) within a filesystem, so the caller returns at once
        and the bytes are reclaimed out of band. The trash sits beside
        ``builds/runs`` rather than inside it, so anything globbing the runs root
        does not pick retired clones back up.
        """
        self.stop()
        retire_dir(self.run_dir)


def open_session(
    build_id: str,
    *,
    container: bool = False,
    image: str | None = None,
    network: str | None = None,
    env_map: dict[str, str] | None = None,
) -> AppSession:
    """Materialize a build and return a session bound to the chosen runtime.

    Args:
        build_id: The build to run.
        container: If true, use the container runtime, otherwise run on the host.
        image: Override the container base image (container runtime only).
        network: Override the container network mode (container runtime only).
        env_map: ``{container_var: host_var}`` mapping resolved from the env / the
            repo ``.env`` and injected into the container, so an LLM-powered app
            reads a key like ``GEMINI_API_KEY`` (valued from the host's
            ``GEMINI_API_KEY_FOUNDER``). Defaults to
            :data:`~viral_bench.founder.appenv.DEFAULT_ENV_MAP`. Container runtime
            only -- host runs already inherit the full environment.

    The session's code lives in a fresh throwaway clone, but its *state* lives in
    the build's durable data directory (mounted at ``/data``, also exported as
    ``VIRALBENCH_DATA_DIR``), so an app's database survives the restarts the
    crowd's shared host performs.
    """
    record, manifest, _app_dir = _load(build_id)
    run_dir = materialize_build(build_id)
    app_dir = run_dir / "app"
    data_dir = build_data_dir(build_id)

    runtime: AppRuntime
    if container:
        # Imported lazily so the host path never needs a container runtime.
        from viral_bench.founder.provision import cache_mounts, is_provisioned
        from viral_bench.founder.runtime import ContainerRuntime

        # The dependency cache is computed from the CLONE, not the work tree: the
        # difference between the two is the whole reason it exists, and a decision
        # taken from the work tree would conclude everything is already present.
        runtime = ContainerRuntime(
            app_dir,
            name=run_dir.name,
            image=image,
            network=network,
            env=resolve_app_env(env_map),
            data_dir=data_dir,
            dep_mounts=cache_mounts(build_id, app_dir),
            deps_provisioned=is_provisioned(build_id, app_dir),
        )
    else:
        runtime = LocalRuntime(app_dir, data_dir=data_dir)

    return AppSession(
        build_id=build_id,
        record=record,
        manifest=manifest,
        run_dir=run_dir,
        runtime=runtime,
        data_dir=data_dir,
    )


def describe_build(build_id: str) -> str:
    """Return human-readable instructions for testing a build."""
    record, manifest, app_dir = _load(build_id)
    lines = [
        f"Build:     {record.build_id}",
        f"Idea:      {record.idea_id}",
        f"App type:  {manifest.app_type}",
        f"Title:     {manifest.title}",
        f"Summary:   {manifest.summary}",
        f"App dir:   {app_dir}",
        f"Shipped:   {record.shipped_ref or '(not shipped)'}",
        "",
        f"Setup:     {list(manifest.setup) or '(none)'}",
        f"Run:       {manifest.run.command}  (cwd={manifest.run.cwd})",
    ]
    if manifest.run.url:
        lines.append(f"Open:      {manifest.run.url}")
    if manifest.test.smoke:
        lines.append(f"Smoke:     {manifest.test.smoke}")
    if manifest.test.manual:
        lines.append("Manual test steps:")
        lines += [f"  {i}. {s}" for i, s in enumerate(manifest.test.manual, 1)]
    return "\n".join(lines)
