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

"""How a *built* app is run and tested -- on the host or in a container.

This is the one place containment lives. A founder build always happens on the
host (see :mod:`viral_bench.founder.workspace`), but *running* the resulting,
untrusted, agent-generated app is where kernel-enforced isolation matters. Both
backends implement the same :class:`AppRuntime` interface, driven entirely by
the app's ``viralbench.json`` manifest, so a human tester and the OASIS crowd
exercise an app identically whether it runs on the host or in a container:

* :class:`LocalRuntime` -- runs the app as a host subprocess. Fast, zero infra,
  and fine while *you* are testing and only one app runs at a time.
* :class:`ContainerRuntime` -- runs the app inside a rootless Podman (or Docker)
  container: only the app dir is mounted, resources are capped, and each app
  gets its own network namespace (so many apps can bind the same internal port
  without colliding). This is what makes unattended, at-scale crowd testing
  safe. See :meth:`create_runtime`.

Parity note: because both backends execute the *same* manifest commands, the
observable behavior (URL to open, CLI to run, smoke result) is the same. The one
unavoidable asymmetry is environment: a host process must inherit ``PATH`` etc.
to find its interpreter, whereas a container starts from a clean env and
receives only an explicit allowlist -- which is exactly what keeps host
credentials out of untrusted code.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from viral_bench.founder.manifest import Manifest


def record_app_start_failure(app_dir: Path, reason: str, logs: str) -> None:
    """Append an app-start failure somewhere it will still be there tomorrow.

    Failing to start the app is the most consequential thing that can happen to a
    crowd run: no agent can open the app, the run produces nothing, and it dies
    at a wall clock or under the stall reaper. Yet in one measured pass, of 26
    builds suppressed as unscorable, **not one** carried the reason in its
    ``crowd_sim.log`` -- Python logging is buffered and the reaper's SIGKILL
    discards the buffer. The reason survived only in the clone's
    ``container.log``, inside a run directory that is retired to ``builds/.trash``
    and deleted.

    Recovering it took archaeology across surviving trash, and it should not
    have: **19 of the 23** recoverable cases were one bug. ``materialize_build``
    prefers ``git clone`` of the shipped branch, the app's own ``.gitignore``
    excludes ``node_modules`` and virtualenvs, and the manifest declares no
    ``setup`` step to reinstall them -- so the container starts an app whose
    dependencies are absent (``vite: not found``, ``No module named flask``).
    That is a harness packaging bug scored as a model failure, and it stayed
    invisible for days purely because the evidence was not durable.

    Written outside the disposable clone, keyed by build, appended and closed
    immediately so a SIGKILL cannot lose it.
    """
    try:
        # app_dir is builds/runs/<build_id>__run-<uuid>/app
        clone = app_dir.parent
        build_id = clone.name.split("__run-")[0] or "unknown"
        out = clone.parent.parent / "app_start_failures"
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with (out / f"{build_id}.log").open("a", encoding="utf-8") as fh:
            fh.write(f"=== {stamp} clone={clone.name}\n{reason}\n{logs[-2000:]}\n")
    except OSError:
        # A diagnostic must never be the reason a run fails.
        pass


# --------------------------------------------------------------------------- #
# Who is at fault when an app will not start
#
# This decides how a run is SCORED, so it belongs in the library and not in the
# script that first needed it. An app the harness shipped without its
# dependencies must be excluded -- scoring a model down for the harness's
# packaging manufactures a capability difference. An app that does not start because its
# own source has a syntax error is a result, and is floored like any other broken
# app: the crowd did everything right, nobody could use the thing, and that is
# exactly what the benchmark exists to notice.
# --------------------------------------------------------------------------- #

#: Substrings that classify a start failure by who is at fault. Order matters: the
#: first match wins, so the specific harness-attributable shapes are listed before
#: the generic ones.
#:
#: The classes exist because the scoring layer must treat them differently. A missing
#: dependency is a HARNESS packaging bug -- the app is fine, it was shipped without
#: its libraries -- and scoring a model down for it manufactures a capability
#: difference. A segfault after full provisioning is the app.
START_FAILURE_CLASSES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "missing_python_dep",
        (
            "modulenotfounderror",
            "no module named",
            "importerror",
            "failed to spawn",
        ),
    ),
    (
        "missing_node_dep",
        (
            "cannot find module",
            "err_module_not_found",
            "vite: not found",
            "sh: 1: ",
            "command not found",
        ),
    ),
    (
        # Container toolchain older than the host the founder built on. A harness
        # fault: the model could not have known which node the runtime would use.
        "toolchain_skew",
        ("err_unknown_builtin_module", "node_module_version", "unsupported engine"),
    ),
    ("port_conflict", ("already in use", "exposes no host port")),
    (
        # A setup step naming a file the founder never committed is a DELIVERY
        # defect, not a packaging one: a human following the README would hit the
        # identical error. Listed before `setup_failed` so it wins, and kept out
        # of HARNESS_CLASSES so it is not counted against the harness.
        "manifest_broken",
        (
            "can't open file",
            "no such file or directory, open",
            "could not open requirements file",
            "no such file or directory: '/work",
        ),
    ),
    (
        # An install step that fails IS the packaging problem, caught one
        # stage earlier than an import error. `uv pip install` with no active
        # environment and `npm install` under the wrong node major both land
        # here, and both are the harness's.
        "setup_failed",
        ("setup step failed",),
    ),
    ("app_crash", ("segmentation fault", "core dumped", "killed", "oom")),
    ("no_manifest", ("has no usable", "manifest")),
)

#: Which classes are the harness's fault rather than the model's. Everything else is
#: attributed to the app.
HARNESS_CLASSES = frozenset(
    {
        "missing_python_dep",
        "missing_node_dep",
        "toolchain_skew",
        "port_conflict",
        "setup_failed",
    }
)


def classify_start_failure(reason: str) -> str:
    """Name who is at fault for a start failure, from its text."""
    low = (reason or "").lower()
    for name, needles in START_FAILURE_CLASSES:
        if any(n in low for n in needles):
            return name
    return "unknown"


class RuntimeError_(RuntimeError):
    """Raised when an app cannot be run in a given runtime."""


# Kept as a distinct, importable name (``RuntimeError`` is a builtin).
AppRuntimeError = RuntimeError_


@dataclass
class RunningApp:
    """A started app plus how to reach and control it (backend-agnostic)."""

    runtime: AppRuntime
    app_type: str
    command: str
    workdir: Path
    log_path: Path
    url: str | None = None
    port: int | None = None
    # How readiness was established (e.g. "HTTP 200"), or why it was not. Kept so
    # a caller can report the reason an app was never reachable rather than
    # silently handing the crowd a URL that resets.
    ready_detail: str | None = None
    # Exactly one backend handle is set.
    process: subprocess.Popen | None = None
    container: str | None = None

    def is_running(self) -> bool:
        return self.runtime.is_running(self)

    def stop(self) -> None:
        self.runtime.stop(self)

    def logs(self) -> str:
        return self.runtime.logs(self)


@runtime_checkable
class AppRuntime(Protocol):
    """Runs and tests one built app: host or container behind one interface."""

    def setup(
        self, manifest: Manifest, *, timeout: float | None = None
    ) -> list[subprocess.CompletedProcess[str]]:
        """Run the manifest's ``setup`` (dependency install) commands."""
        ...

    def smoke(
        self,
        manifest: Manifest,
        *,
        timeout: float | None = None,
        app: RunningApp | None = None,
    ) -> subprocess.CompletedProcess[str] | None:
        """Run the manifest's ``test.smoke`` command, if any.

        When ``app`` is a running instance, the command runs where that app is
        reachable, so a smoke test may legitimately probe the app's own port.
        """
        ...

    def exec(
        self,
        command: str,
        *,
        cwd: str = ".",
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a one-shot command (used by the crowd's ``try_app`` for CLIs)."""
        ...

    def start(self, manifest: Manifest, *, wait_timeout: float = 90.0) -> RunningApp:
        """Start the app's ``run`` command and return a handle (with URL if web)."""
        ...

    def is_running(self, app: RunningApp) -> bool: ...

    def stop(self, app: RunningApp) -> None: ...

    def logs(self, app: RunningApp) -> str: ...

    def describe(self) -> str: ...


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.25)
    return False


def _port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    """True if anything is already listening on ``host:port``.

    A plain TCP connect on purpose: the question is "is this port taken", and a
    non-HTTP listener squatting on the app's port is as much of a conflict
    as an HTTP one.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _wait_for_http(host: str, port: int, timeout: float = 15.0) -> tuple[bool, str]:
    """Wait until an HTTP server on ``host:port`` answers a request.

    A TCP connect is NOT sufficient readiness evidence. Under rootless podman the
    host-side port forwarder accepts the connection before anything inside the
    container is listening, so :func:`_wait_for_port` returns True against a dead
    app and the caller hands out a URL that resets on first use. That is exactly
    how crowd triers ended up reviewing ``net::ERR_CONNECTION_RESET`` pages as if
    they were the product.

    Any HTTP status proves a server is serving -- a 404 or a 500 is a live app
    with an unhappy route, which is the app's business and not a readiness
    failure. Only connection-level errors are retried.

    Returns ``(ready, detail)`` so the caller can report *why* an app was never
    reachable instead of silently continuing.
    """
    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}/"
    last = "no attempt"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            # Never let one attempt outlive the overall budget: a socket that
            # accepts but never replies would otherwise hold the caller for the full
            # per-request timeout after the deadline has already passed.
            with urllib.request.urlopen(url, timeout=min(2.0, remaining)) as resp:
                return True, f"HTTP {resp.status}"
        except urllib.error.HTTPError as exc:
            return True, f"HTTP {exc.code}"
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
            time.sleep(0.25)
    return False, f"no HTTP response within {timeout:.0f}s: {last}"


class LocalRuntime:
    """Run an app as a host subprocess (no container)."""

    def __init__(
        self,
        app_dir: Path,
        *,
        env: dict[str, str] | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.app_dir = Path(app_dir)
        # Extra env merged over the inherited host env (host processes must keep
        # PATH/HOME to find their interpreter).
        self.extra_env = env or {}
        # A host run has no mount namespace, so the durable data directory is
        # passed by path instead. Apps read VIRALBENCH_DATA_DIR either way, which
        # keeps one contract across both runtimes.
        self.data_dir = Path(data_dir) if data_dir is not None else None
        if self.data_dir is not None:
            self.extra_env = {
                **self.extra_env,
                "VIRALBENCH_DATA_DIR": str(self.data_dir),
            }

    def _workdir(self, manifest: Manifest) -> Path:
        return (self.app_dir / manifest.run.cwd).resolve()

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.extra_env)
        return env

    def setup(
        self, manifest: Manifest, *, timeout: float | None = None
    ) -> list[subprocess.CompletedProcess[str]]:
        workdir = self._workdir(manifest)
        results = []
        for command in manifest.setup:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=workdir,
                env=self._env(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            results.append(proc)
            if proc.returncode != 0:
                raise AppRuntimeError(
                    f"setup step failed: {command}\n"
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )
        return results

    def smoke(
        self,
        manifest: Manifest,
        *,
        timeout: float | None = None,
        app: RunningApp | None = None,
    ) -> subprocess.CompletedProcess[str] | None:
        # ``app`` needs no special handling on the host: a host-run app listens on
        # the host's own loopback, so a smoke command already reaches it.
        if not manifest.test.smoke:
            return None
        return subprocess.run(
            manifest.test.smoke,
            shell=True,
            cwd=self._workdir(manifest),
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def exec(
        self,
        command: str,
        *,
        cwd: str = ".",
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        run_env = self._env()
        if env:
            run_env.update(env)
        return subprocess.run(
            command,
            shell=True,
            cwd=(self.app_dir / cwd).resolve(),
            env=run_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def start(self, manifest: Manifest, *, wait_timeout: float = 90.0) -> RunningApp:
        workdir = self._workdir(manifest)
        log_path = self.app_dir.parent / "run.log"
        # Refuse to start on top of somebody else's server. The readiness probe
        # below asks "does anything answer on this port", NOT "is the thing
        # answering the process that was started here" -- and it cannot ask the second
        # question, because the manifest command is a shell string that may exec,
        # fork or background whatever it likes. So if the port is already taken,
        # every later check would pass against the wrong app: the probe returns
        # HTTP 200, app.url is handed out, and a crowd trier (or a QA agent)
        # reviews an application nobody asked for. 74 of 126 shipped manifests
        # declare port 8000, and agents leak app servers, so this is not
        # hypothetical -- see reap_workspace_processes in harness.py.
        #
        # Fail loudly instead. Starting anyway would add a second doomed
        # process and make the confusion worse.
        port = manifest.run.port
        if port is not None and _port_in_use("127.0.0.1", port):
            raise AppRuntimeError(
                f"port {port} is already in use before the app was "
                f"started, so anything answering on it is NOT this app. Refusing "
                f"to start rather than review the wrong application. Free the "
                f"port (a leaked app server from an earlier build is the usual "
                f"cause) and retry, or run builds under scripts/netns_run.sh so "
                f"each one gets its own loopback."
            )
        # start_new_session so the whole process group can be killed on stop. The
        # child keeps its own dup of the log fd, so this one is closed right away
        # to avoid leaking a handle in the parent.
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                manifest.run.command,
                shell=True,
                cwd=workdir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=self._env(),
                start_new_session=True,
            )
        app = RunningApp(
            runtime=self,
            app_type=manifest.app_type,
            command=manifest.run.command,
            workdir=workdir,
            log_path=log_path,
            port=manifest.run.port,
            process=process,
        )
        if manifest.run.port is not None:
            ready, detail = _wait_for_http(
                "127.0.0.1", manifest.run.port, timeout=wait_timeout
            )
            # An app whose launcher has exited but whose port answers is almost
            # always a self-backgrounding command (`python3 server.py &`), which
            # is legitimate -- so record it rather than raising. It is worth
            # recording because it is also the shape a port hijack would take if
            # one slipped past the pre-start check above.
            if ready and process.poll() is not None:
                detail = f"{detail}; launcher process exited"
            app.ready_detail = detail
            if ready:
                app.url = manifest.run.url or f"http://localhost:{manifest.run.port}/"
            elif process.poll() is not None:
                raise AppRuntimeError(
                    f"app exited immediately; see log: {log_path}\n"
                    + log_path.read_text(encoding="utf-8")[-2000:]
                )
            else:
                # Still running but never answered. Leave app.url unset: a caller
                # that hands out a URL here is handing out an error page, and the
                # crowd will review it as if it were the product.
                raise AppRuntimeError(
                    f"app never became reachable ({detail}); see log: {log_path}\n"
                    + log_path.read_text(encoding="utf-8")[-2000:]
                )
        return app

    def is_running(self, app: RunningApp) -> bool:
        return app.process is not None and app.process.poll() is None

    def stop(self, app: RunningApp) -> None:
        if app.process is None or not self.is_running(app):
            return
        try:
            os.killpg(os.getpgid(app.process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            app.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(app.process.pid), signal.SIGKILL)

    def logs(self, app: RunningApp) -> str:
        if app.log_path.is_file():
            return app.log_path.read_text(encoding="utf-8")
        return ""

    def describe(self) -> str:
        return f"host process in {self.app_dir}"


#: A run command that daemonizes itself, which a container cannot survive.
_SELF_BACKGROUNDING = re.compile(r"^\s*(?:setsid\s+)?(.*?)\s*&\s*$", re.DOTALL)


def _foreground(command: str) -> str:
    """Strip a trailing ``&`` (and ``setsid``) so the app runs as PID 1.

    A container lives exactly as long as its main process. A run command that
    backgrounds itself therefore starts the server, returns immediately, and the
    container exits with the app still notionally "starting" -- podman then tears
    the cgroup down and the port never appears. Measured on image_compressor,
    whose manifest says ``setsid python server.py > /dev/null 2>&1 < /dev/null &``:
    the container exposed no host port and produced no logs at all, because
    everything had been redirected to /dev/null before it died.

    Backgrounding is a habit from running things in your own shell, where it is
    correct. It is never what the manifest means here. Stripping it preserves the
    intent -- run this server -- and is what LocalRuntime already tolerates, where
    it notes a self-backgrounding command as legitimate rather than failing.

    Strips exactly the ONE trailing ``&``, and nothing else. Commands containing
    ``&&`` are left untouched, because there the ``&`` is a conjunction rather
    than a job-control operator. A redirection like ``2>&1`` is deliberately not
    counted, and is why this cannot reject any command containing ``&``.
    """
    text = command.strip()
    if "&&" in text or not text.endswith("&"):
        return command
    match = _SELF_BACKGROUNDING.match(text)
    if not match or not match.group(1).strip():
        return command
    return match.group(1).strip()


# Label stamped on every container created here, so orphans are always findable.
VIRALBENCH_LABEL = "viralbench=1"

# Default image: built from docker/Containerfile (python + uv + node). Override
# via config or the ``image=`` argument. Not auto-built, see docker/Containerfile.
DEFAULT_IMAGE = "localhost/viralbench-runtime:latest"

# Where a build's durable data directory is mounted inside the container. Apps
# are told to keep all mutable state here (and are handed the same path in
# ``VIRALBENCH_DATA_DIR`` so a host run can honour it too).
CONTAINER_DATA_DIR = "/data"


class ContainerRuntime:
    """Run an app inside a rootless Podman (or Docker) container.

    ``app_dir`` is bind-mounted at ``/work`` and, when given, ``data_dir`` at
    ``/data``. Resources are capped, and each container has its own network
    namespace, so many apps can bind the same *internal* port without colliding
    (the host port is assigned dynamically and read back). This is the safe path
    for unattended, at-scale crowd testing.

    One-shot commands (``setup`` / ``smoke`` / ``exec``) run in ephemeral
    ``--rm`` containers that share the same mounts, so installed dependencies
    persist into ``app_dir`` and are visible to the long-lived ``start``
    container. The ``/data`` mount is what makes that true for *state* as well as
    code: without it a migration run in ``setup`` writes into a container layer
    that ``--rm`` throws away, and the app starts against an empty database.
    Everything created here is labeled ``viralbench=1`` for cleanup.
    """

    def __init__(
        self,
        app_dir: Path,
        *,
        name: str | None = None,
        image: str | None = None,
        runtime: str = "podman",
        network: str | None = None,
        memory: str = "1g",
        setup_memory: str = "4g",
        cpus: str = "1.0",
        pids_limit: int = 512,
        env_allowlist: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
        data_dir: Path | None = None,
        dep_mounts: list[tuple[Path, str, bool]] | None = None,
        deps_provisioned: bool = False,
    ) -> None:
        self.app_dir = Path(app_dir)
        # Per-build dependency cache: (host_dir, container_path, writable). Empty
        # for a build that ships everything it needs. See founder/provision.py --
        # the short version is that a git clone of the shipped branch legitimately
        # lacks node_modules/.venv, and a `pip install` in a setup step is thrown
        # away by `--rm` because it writes to the image layer rather than a mount.
        self.dep_mounts = list(dep_mounts or [])
        # Whether provisioning has already run for this build. NOT the same as
        # having mounts: a build with nothing to install produces no mounts and is
        # still fully provisioned, and its redundant `npm install` (against a
        # package.json that was never committed) deserves the same leniency as
        # everyone else's. Keying the rule on dep_mounts got that case wrong.
        self.deps_provisioned = deps_provisioned or bool(self.dep_mounts)
        # Durable state, mounted at /data. Unlike app_dir -- a throwaway clone
        # that AppSession.close deletes -- this outlives restarts, so a
        # server-side app's database does too.
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.name = name or f"viralbench-{os.urandom(4).hex()}"
        self.image = image or DEFAULT_IMAGE
        self.runtime = runtime
        # ``None`` => podman's default network (has outbound access = "on"),
        # which is what ViralBench uses by default. Set to "none" to isolate.
        self.network = network
        self.memory = memory
        # A BUILD is not the app, and must not be capped like one.
        #
        # 1g is a deliberate, realistic ceiling for a running app: it is untrusted,
        # and a real deployment is bounded. A bundler is neither. Measured:
        # markdown_slides died in `npm run build` with "FATAL ERROR: Ineffective
        # mark-compacts near heap limit - JavaScript heap out of memory", and
        # collaborative_table with a bare "Killed" -- both apps that build fine
        # with normal headroom. Capping the compile at the app's runtime budget
        # turns a harness packaging choice into a model that cannot ship, which is the
        # fabricated capability difference: a harness fault scored as a model fault.
        #
        # Applied only to one-shot setup/build containers, which run the harness's
        # own toolchain and exit. The long-lived app container keeps `memory`.
        self.setup_memory = setup_memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.env_allowlist = env_allowlist
        self.env = env or {}

    @staticmethod
    def available(runtime: str = "podman") -> bool:
        """True if the container runtime binary is on PATH."""
        return shutil.which(runtime) is not None

    def image_exists(self) -> bool:
        """True if the configured image is present locally (no pull attempted)."""
        proc = subprocess.run(
            [self.runtime, "image", "exists", self.image],
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0

    # -- argv construction ---------------------------------------------------

    def _container_workdir(self, cwd: str) -> str:
        return "/work" if cwd in ("", ".") else f"/work/{cwd.lstrip('/')}"

    def _common_args(self, *, memory: str | None = None) -> list[str]:
        args = [
            "--label",
            VIRALBENCH_LABEL,
            "--label",
            f"run={self.name}",
            "--memory",
            memory or self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.pids_limit),
            "-v",
            f"{self.app_dir}:/work:Z",
        ]
        if self.data_dir is not None:
            # Same mount for setup, smoke and start, so a migration and the
            # server it prepares see one and the same database.
            args += ["-v", f"{self.data_dir}:{CONTAINER_DATA_DIR}:Z"]
            args += ["-e", f"VIRALBENCH_DATA_DIR={CONTAINER_DATA_DIR}"]
        if self.dep_mounts:
            # The cache's environment is runtime plumbing, kept OUT of ``self.env``
            # so that field stays exactly what the caller asked the app to see --
            # its API keys and nothing else. Emitted first, so a caller-supplied
            # value always wins.
            from viral_bench.founder.provision import container_env

            for key, value in container_env().items():
                args += ["-e", f"{key}={value}"]
        for host, where, writable in self.dep_mounts:
            # `:O` is podman's overlay mount: the container may write inside its
            # own dependency tree and the write is discarded on exit. Read-only
            # would be cheaper to reason about and would break real apps -- vite
            # writes `node_modules/.vite` on startup, so `ro` turns a working app
            # into a crash. Writable is used only while provisioning, where the
            # point is to populate the cache.
            args += ["-v", f"{host}:{where}" + ("" if writable else ":O")]
        if self.network not in (None, "", "default"):
            args.append(f"--network={self.network}")
        for key in self.env_allowlist:
            if key in os.environ:
                args += ["-e", f"{key}={os.environ[key]}"]
        for key, value in self.env.items():
            args += ["-e", f"{key}={value}"]
        return args

    def _run_oneshot(
        self, command: str, *, cwd: str, timeout: float | None
    ) -> subprocess.CompletedProcess[str]:
        if not self.available(self.runtime):
            raise AppRuntimeError(f"{self.runtime} is not installed or not on PATH")
        argv = (
            [self.runtime, "run", "--rm"]
            + self._common_args(memory=self.setup_memory)
            + [
                "-e",
                # Node sizes its old-space from the cgroup it can see, and gets it
                # wrong often enough to OOM at exactly the wrong moment. Say it.
                f"NODE_OPTIONS=--max-old-space-size={self._node_heap_mb()}",
                "-w",
                self._container_workdir(cwd),
                self.image,
                "sh",
                "-c",
                command,
            ]
        )
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)

    def _node_heap_mb(self) -> int:
        """Node heap ceiling for a build step: ~75% of the container's memory."""
        raw = str(self.setup_memory).strip().lower()
        mb = 4096
        try:
            if raw.endswith("g"):
                mb = int(float(raw[:-1]) * 1024)
            elif raw.endswith("m"):
                mb = int(float(raw[:-1]))
        except ValueError:
            pass
        return max(512, int(mb * 0.75))

    # -- AppRuntime interface ------------------------------------------------

    def setup(
        self, manifest: Manifest, *, timeout: float | None = None
    ) -> list[subprocess.CompletedProcess[str]]:
        """Run the manifest's setup steps.

        A failing step aborts -- EXCEPT an install step when this build already
        has a provisioned dependency cache, which is then only recorded.

        The exception is narrow and load-bearing. Provisioning has already
        installed this app's dependencies, from its lockfile or requirements or
        (failing both) its own imports, so a second install command is redundant
        by the time it runs. When such a command is *also* broken -- `uv sync`
        with no pyproject.toml in the clone, `pip install -r requirements.txt`
        naming a file that was never committed -- aborting here throws away a
        working app over a command nobody needed. Measured: six of the thirteen
        failures remaining after provisioning landed were exactly this.

        A NON-install step still aborts, because a migration that fails is real
        work that did not happen, and an app whose schema is missing should be
        seen to be broken rather than quietly started.
        """
        from viral_bench.founder.provision import (
            is_install_command,
            neutralize_pruning,
        )

        results = []
        for raw in manifest.setup:
            # Same transform provisioning applied. Without it the second run of
            # `uv sync` prunes the cache all over again, at session start, and
            # the app fails on a module that is demonstrably installed.
            command = neutralize_pruning(raw) if self.deps_provisioned else raw
            proc = self._run_oneshot(command, cwd=manifest.run.cwd, timeout=timeout)
            results.append(proc)
            if proc.returncode == 0:
                continue
            if self.deps_provisioned and is_install_command(command):
                record_app_start_failure(
                    self.app_dir,
                    f"redundant setup install failed (dependencies already "
                    f"provisioned, continuing): {command}",
                    proc.stderr or proc.stdout or "",
                )
                continue
            # BOTH streams, not `stderr or stdout`. npm writes its notices to
            # stderr and the actual failure to stdout, so preferring stderr
            # yielded "New major version of npm available!" as the entire
            # explanation for a build that could not be diagnosed at all.
            detail = "\n".join(
                part
                for part in (proc.stdout.strip()[-2000:], proc.stderr.strip()[-2000:])
                if part
            )
            raise AppRuntimeError(f"setup step failed: {command}\n{detail}")
        return results

    def smoke(
        self,
        manifest: Manifest,
        *,
        timeout: float | None = None,
        app: RunningApp | None = None,
    ) -> subprocess.CompletedProcess[str] | None:
        """Run the manifest's smoke command, inside the live app when there is one.

        ``app`` matters more than it looks. A one-shot runs in its OWN container
        with its OWN network namespace, so a smoke command like
        ``curl localhost:8000/health`` can never reach the server -- it is talking
        to an empty netns. Handed a running app, this ``exec``s in that container
        instead, where the app's port is local in fact. Without this, "check the
        app answers" is unexpressible as a smoke test in container mode.
        """
        if not manifest.test.smoke:
            return None
        if app is not None and app.container and self.is_running(app):
            return self._exec_in_running(
                app, manifest.test.smoke, cwd=manifest.run.cwd, timeout=timeout
            )
        return self._run_oneshot(
            manifest.test.smoke, cwd=manifest.run.cwd, timeout=timeout
        )

    def _exec_in_running(
        self,
        app: RunningApp,
        command: str,
        *,
        cwd: str = ".",
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command inside the app's already-running container."""
        argv = [
            self.runtime,
            "exec",
            "-w",
            self._container_workdir(cwd),
            str(app.container),
            "sh",
            "-c",
            command,
        ]
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)

    def exec(
        self,
        command: str,
        *,
        cwd: str = ".",
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        # Merge any per-call env on top of the runtime's for this one command.
        if env:
            saved = dict(self.env)
            self.env = {**self.env, **env}
            try:
                return self._run_oneshot(command, cwd=cwd, timeout=timeout)
            finally:
                self.env = saved
        return self._run_oneshot(command, cwd=cwd, timeout=timeout)

    def start(self, manifest: Manifest, *, wait_timeout: float = 90.0) -> RunningApp:
        if not self.available(self.runtime):
            raise AppRuntimeError(f"{self.runtime} is not installed or not on PATH")
        port = manifest.run.port
        argv = [self.runtime, "run", "-d", "--name", self.name] + self._common_args()
        if port is not None:
            # Publish the container's internal port to a random loopback host
            # port (parallel-safe: each container has its own netns).
            argv += ["-p", f"127.0.0.1::{port}"]
        argv += [
            "-w",
            self._container_workdir(manifest.run.cwd),
            self.image,
            "sh",
            "-c",
            _foreground(manifest.run.command),
        ]
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise AppRuntimeError(f"could not start container: {detail}")

        app = RunningApp(
            runtime=self,
            app_type=manifest.app_type,
            command=manifest.run.command,
            workdir=self.app_dir,
            log_path=self.app_dir.parent / "container.log",
            port=port,
            container=self.name,
        )
        if port is not None:
            host_port = self._host_port(port)
            if host_port is None:
                _logs = self.logs(app)
                record_app_start_failure(
                    self.app_dir, f"no host port for {port}", _logs
                )
                raise AppRuntimeError(
                    f"container exposes no host port for {port}; logs:\n{_logs[-2000:]}"
                )
            ready, detail = _wait_for_http("127.0.0.1", host_port, timeout=wait_timeout)
            app.ready_detail = detail
            if ready:
                app.url = f"http://localhost:{host_port}/"
            elif not self.is_running(app):
                _logs = self.logs(app)
                record_app_start_failure(
                    self.app_dir, "container exited immediately", _logs
                )
                raise AppRuntimeError(
                    f"app container exited immediately; logs:\n{_logs[-2000:]}"
                )
            else:
                # The forwarder accepts long before the server serves, so a TCP
                # probe here would pass against a dead app and the crowd would
                # review a connection-reset page. Refuse to hand out the URL.
                _logs = self.logs(app)
                record_app_start_failure(
                    self.app_dir, f"never became reachable ({detail})", _logs
                )
                raise AppRuntimeError(
                    f"app container never became reachable ({detail}); "
                    f"logs:\n{_logs[-2000:]}"
                )
        return app

    def _host_port(self, container_port: int) -> int | None:
        proc = subprocess.run(
            [self.runtime, "port", self.name, f"{container_port}/tcp"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        # Output looks like "127.0.0.1:43765", so take the last colon-separated part.
        return int(proc.stdout.strip().splitlines()[0].rsplit(":", 1)[-1])

    def is_running(self, app: RunningApp) -> bool:
        if app.container is None:
            return False
        proc = subprocess.run(
            [self.runtime, "inspect", "-f", "{{.State.Running}}", app.container],
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0 and proc.stdout.strip() == "true"

    def stop(self, app: RunningApp) -> None:
        if app.container is None:
            return
        # Persist logs before removal so a caller can still read them.
        self.logs(app)
        subprocess.run(
            [self.runtime, "rm", "-f", "-t", "10", app.container],
            capture_output=True,
            text=True,
        )

    def logs(self, app: RunningApp) -> str:
        if app.container is None:
            return ""
        proc = subprocess.run(
            [self.runtime, "logs", app.container], capture_output=True, text=True
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        try:
            app.log_path.write_text(text, encoding="utf-8")
        except OSError:
            pass
        return text

    def describe(self) -> str:
        return f"{self.runtime} container '{self.name}' ({self.image})"


def reap_orphans(
    runtime: str = "podman", *, running_grace_seconds: float = 7200.0
) -> int:
    """Force-remove leftover ViralBench containers. Returns the count.

    A safety net against leaks: containers are normally torn down by
    :meth:`ContainerRuntime.stop`, but a crash can leave orphans. No-op if the
    runtime is unavailable.

    Crucially this does NOT remove every container carrying the ViralBench label.
    It used to, and that is unsafe whenever more than one run is in flight -- the
    sweep driver runs four simulations at once, so the first to finish would
    force-remove the other three's *live* app containers and their agents would
    suddenly be reviewing a dead port. A container is treated as an orphan only
    if it has already stopped, or if it is still running but older than
    ``running_grace_seconds`` (long past any real trial, so leaked).
    """
    if shutil.which(runtime) is None:
        return 0
    listed = subprocess.run(
        [
            runtime,
            "ps",
            "-a",
            "--filter",
            f"label={VIRALBENCH_LABEL}",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        return 0
    try:
        entries = json.loads(listed.stdout or "[]")
    except json.JSONDecodeError:
        return 0

    now = time.time()
    ids: list[str] = []
    for entry in entries:
        cid = entry.get("Id") or entry.get("ID")
        if not cid:
            continue
        state = str(entry.get("State", "")).lower()
        if state != "running":
            ids.append(cid)  # already dead: definitely an orphan
            continue
        created = entry.get("Created")
        age = None
        if isinstance(created, (int, float)):
            age = now - float(created)
        if age is not None and age > running_grace_seconds:
            ids.append(cid)  # running far longer than any trial: leaked

    if not ids:
        return 0
    subprocess.run([runtime, "rm", "-f", *ids], capture_output=True, text=True)
    return len(ids)
