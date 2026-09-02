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

"""Shared running-app instances for the crowd (the ``try_app`` scaling lever).

When the OASIS crowd "tries" an app, we do *not* spin up a fresh container per
agent trial -- that would not scale and is unrealistic. Instead, like a real
deployed app, one instance runs and many agents hit it. :class:`AppHost` owns
those shared instances: it starts an app on first request, caches the handle
keyed by ``build_id``, hands the same running app to every subsequent caller,
transparently restarts one that has died, and tears everything down (including
a label-based orphan sweep) on :meth:`close`.

By default instances run in containers (the crowd path is container-forced);
pass ``container=False`` for host-mode local testing.
"""

from __future__ import annotations

import threading

from viral_bench.founder.runner import AppSession, open_session, sweep_run_dirs
from viral_bench.founder.runtime import AppRuntimeError, RunningApp, reap_orphans

#: How many times one build may fail to start before the host stops trying.
#:
#: An app that cannot start does not start on the fifth attempt either. Measured on
#: the r3 sweep, where there was no cap: a single unstartable build was retried
#: 15-34 times in one run -- once per agent trial -- because every failure raised
#: out of ``get`` and the next agent simply called it again. Each attempt clones the
#: whole app tree (~188,600 files for a node app), so the run leaked twenty-odd
#: trees, produced nothing, and was finally culled at the 3600s wall clock having
#: written no ``run_summary.json`` at all. The cell then looked untried and was
#: re-offered on the next pass, forever.
#:
#: Three is enough to ride out a genuinely transient start (a port still in
#: TIME_WAIT, a slow first import) and small enough that a dead build costs seconds.
MAX_START_ATTEMPTS = 3


class AppStartFailed(AppRuntimeError):
    """This build's app could not be started, and retrying will not help.

    Distinct from ``AppRuntimeError`` so a caller can tell "this attempt failed"
    from "this build is settled as unstartable" and record the second as an
    outcome instead of retrying into the wall clock.
    """


class AppHost:
    """A pool of shared, long-lived running app instances, keyed by build id."""

    def __init__(
        self,
        *,
        container: bool = True,
        image: str | None = None,
        network: str | None = None,
        env_map: dict[str, str] | None = None,
        runtime_name: str = "podman",
    ) -> None:
        self.container = container
        self.image = image
        self.network = network
        self.env_map = env_map
        self.runtime_name = runtime_name
        self._sessions: dict[str, AppSession] = {}
        self._failures: dict[str, int] = {}
        #: build_id -> why it was given up on, for the run summary.
        self.start_failures: dict[str, str] = {}
        self._lock = threading.Lock()

    def unstartable(self, build_id: str) -> str | None:
        """Why this build was given up on, or ``None`` if it was not."""
        return self.start_failures.get(build_id)

    def get(self, build_id: str, *, wait_timeout: float = 90.0) -> RunningApp:
        """Return the shared running app for ``build_id``, starting it if needed.

        Concurrency-safe: many crowd agents may call this at once; the app is
        started exactly once and shared. A dead instance is replaced.

        Raises :class:`AppStartFailed` once the build has failed to start
        :data:`MAX_START_ATTEMPTS` times, and thereafter without trying again.
        """
        with self._lock:
            settled = self.start_failures.get(build_id)
            if settled is not None:
                raise AppStartFailed(settled)

            session = self._sessions.get(build_id)
            if (
                session is not None
                and session.app is not None
                and session.app.is_running()
            ):
                return session.app

            # Stale/dead session: clean it up before recreating. This is also the
            # only thing that retires the previous clone, so it must happen even
            # when the start below goes on to fail.
            if session is not None:
                session.close()
                self._sessions.pop(build_id, None)

            session = None
            try:
                session = open_session(
                    build_id,
                    container=self.container,
                    image=self.image,
                    network=self.network,
                    env_map=self.env_map,
                )
                session.setup()
                app = session.start(wait_timeout=wait_timeout)
            except Exception as exc:
                # Retire this attempt's clone here rather than leaving it for the
                # sweeper: a failing build is exactly the one that produces the
                # most of them.
                if session is not None:
                    try:
                        session.close()
                    except Exception:  # noqa: BLE001
                        pass
                count = self._failures.get(build_id, 0) + 1
                self._failures[build_id] = count
                if count >= MAX_START_ATTEMPTS:
                    reason = f"{type(exc).__name__}: {exc}"
                    self.start_failures[build_id] = reason
                    raise AppStartFailed(reason) from exc
                raise
            self._sessions[build_id] = session
            return app

    def session(self, build_id: str) -> AppSession | None:
        """Return the live session for ``build_id`` (or ``None``)."""
        with self._lock:
            return self._sessions.get(build_id)

    def running_builds(self) -> list[str]:
        """Build ids that currently have a live shared instance."""
        with self._lock:
            return [
                build_id
                for build_id, s in self._sessions.items()
                if s.app is not None and s.app.is_running()
            ]

    def stop(self, build_id: str) -> None:
        """Stop and clean up the shared instance for one build (if any)."""
        with self._lock:
            session = self._sessions.pop(build_id, None)
        if session is not None:
            session.close()

    def close(self) -> None:
        """Stop every instance, delete run dirs, and sweep container orphans."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                session.close()
            except Exception:
                # Best-effort teardown: one bad app must not block the rest.
                pass
        if self.container:
            reap_orphans(self.runtime_name)
        # Sessions that died before we could close them leak their clone. Sweep
        # the stale ones here (the only place that reliably runs at the end of a
        # crowd run) or they accumulate unboundedly -- and a clone carrying
        # node_modules is ~440 MB, not the few MB a static app leaves behind.
        try:
            sweep_run_dirs()
        except Exception:  # noqa: BLE001 - disk hygiene must never fail a run
            pass

    def __enter__(self) -> AppHost:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
