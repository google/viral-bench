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

"""Crowd-facing checks: does an app build/run, and what happens when you use it.

These are the two OASIS extensions the design doc calls out, implemented as
plain, framework-independent functions so they can be tested now and wrapped as
``SocialAgent(tools=[...])`` later:

* :func:`verify_code` -- the validity gate. Clone the app into a fresh, isolated
  copy, install deps, run its smoke test, and confirm it starts, all in
  an ephemeral container that is torn down afterwards. Returns
  ``{builds, runs, does_what_it_claims}`` so the crowd can filter or down-weight
  broken apps before amplifying them.
* :func:`try_app` -- the delight probe. Have an agent *use* a *running*
  app the way a human would -- click/type through a single-page app in a real
  browser, run a CLI with real inputs, or hold a multi-turn bot conversation --
  and return what it observed, which informs its LIKE / REPOST / DO_NOTHING
  choice. This delegates to :mod:`viral_bench.crowd.interaction` (the one
  canonical "use the app" path). The older blind HTTP GET -- which never ran a
  single-page app's JavaScript and so could not see the rendered UI -- is gone.

Both default to running in a container (the crowd path is container-forced). Pass
``container=False`` for host-mode local testing.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

from viral_bench.founder.apphost import AppHost
from viral_bench.founder.runner import open_session
from viral_bench.founder.runtime import AppRuntimeError

_MAX_OBSERVATION = 4096


def _http_probe(
    url: str,
    *,
    timeout: float = 5.0,
    deadline_s: float = 30.0,
    delay: float = 0.3,
) -> tuple[bool, int | None, str]:
    """Fetch ``url``, retrying transient connection errors until a deadline.

    Returns ``(responded, status, body)``. A rootless port-forwarder accepts the
    TCP connection before the in-container server is serving, so the
    first request can be reset, and connection-level failures are retried. An HTTP error
    status (4xx/5xx) still counts as "responded" -- the server is up. Callers
    that need to know whether the app WORKS must inspect the returned status:
    ``verify_code`` treats a 5xx on the entry URL as not running.

    The budget is WALL-CLOCK, not a fixed number of attempts. It used to be 12
    tries 0.3s apart, which sounds generous and is not: a connection refused by a
    not-yet-ready forwarder fails instantly, so the whole budget was ~3.6 seconds.
    Under a concurrent sweep that is short enough to lose the race, and losing it
    is expensive -- this probe alone decides ``runs``, which gates the score.

    Measured on the stored corpus: 9 runs recorded ``runs=False`` for an app the
    crowd then used successfully, and the clearest case reported
    ``smoke=ok, url=...`` with 8 of 8 trials reaching the app -- i.e. the smoke
    command, which executes *inside* the container against the app's own port,
    proved the server was serving while the host-side probe had already given up.
    That single false negative moved the build's score from 55.2 to 13.2, a
    42-point swing between two seeds of the same app.
    """
    last = "no attempt"
    started = time.monotonic()
    wait = delay
    while True:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                body = resp.read(_MAX_OBSERVATION).decode("utf-8", "replace")
                return True, resp.status, body
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read(_MAX_OBSERVATION).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                pass
            return True, exc.code, body
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            last = str(exc)
            if time.monotonic() - started >= deadline_s:
                return False, None, f"error after {deadline_s:.0f}s: {last}"
            time.sleep(wait)
            # Back off gently: a forwarder that is not ready in 300ms is often
            # not ready in 600ms either, and hammering it adds load to the
            # contention that caused the delay in the first place.
            wait = min(wait * 1.5, 2.0)


@dataclass
class VerifyResult:
    """Outcome of the validity gate for one build."""

    build_id: str
    builds: bool  # dependencies installed cleanly (or nothing to install)
    runs: bool  # the app started / responded
    does_what_it_claims: bool  # heuristic gate: runs AND smoke passed
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TryResult:
    """What a crowd agent observed when it used a running app."""

    build_id: str
    app_type: str
    ok: bool
    observation: str
    url: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def verify_code(
    build_id: str,
    *,
    container: bool = True,
    image: str | None = None,
    network: str | None = None,
    env_map: dict[str, str] | None = None,
    setup_timeout: float = 600.0,
    smoke_timeout: float = 120.0,
    start_wait: float = 90.0,
) -> VerifyResult:
    """Clone, install, smoke-test and start an app in a disposable sandbox.

    Everything runs in a fresh materialized copy and (by default) an ephemeral
    container that is removed on return -- so verifying a hostile app cannot
    affect the host or any other build. ``env_map`` maps container var names to
    host var names (resolved from the env / repo ``.env``, defaulting to
    :data:`~viral_bench.founder.appenv.DEFAULT_ENV_MAP`), so an app that needs a
    key like ``GEMINI_API_KEY`` at run time can be validated end-to-end.
    """
    session = open_session(
        build_id,
        container=container,
        image=image,
        network=network,
        env_map=env_map,
    )
    manifest = session.manifest
    builds = runs = smoke_ok = False
    detail: list[str] = []
    try:
        # 1. Install dependencies -> "builds".
        try:
            session.setup(timeout=setup_timeout)
            builds = True
        except AppRuntimeError as exc:
            return VerifyResult(build_id, False, False, False, f"setup failed: {exc}")

        # 2. Start the app -> "runs" (type-specific).
        #
        # This runs BEFORE the smoke test, which is the opposite of the original
        # order, and the inversion was not cosmetic. Smoke used to run first, so
        # any smoke command that talked to the app's own port was unpassable by
        # construction -- the app did not exist yet. Measured over the stored
        # corpus, network-dependent smoke commands passed 0 times out of 45 while
        # local ones passed 372/450, and the apps were not the problem. Starting
        # first makes "check the app answers" a legal health check.
        app = None
        try:
            if manifest.run.port is not None:
                app = session.start(wait_timeout=start_wait)
        except AppRuntimeError as exc:
            detail.append(f"start failed: {exc}")

        # 3. Smoke test -> health signal, now able to reach a running server.
        smoke = session.smoke(timeout=smoke_timeout)
        smoke_ok = smoke is None or smoke.returncode == 0
        detail.append(f"smoke={'ok' if smoke_ok else 'fail'}")

        if manifest.run.port is not None:
            # Fetch it (with warmup retries) so "runs" means the app serves,
            # not merely that the TCP port is open.
            responded, status, _body = (
                _http_probe(app.url)
                if app is not None and app.url
                else (False, None, "")
            )
            # A 5xx on the app's OWN declared entry URL means it does not work.
            #
            # This used to count as "responded -- the server is up", which is the
            # right answer to "did the process start" and the wrong answer to
            # "does the app work". The crowd found the difference: one build's
            # landing page returned 500 while its /healthz returned 200, so the
            # gate passed it runs=True AND does_what_it_claims=True, and the
            # smoke test passed too because the model had pointed it at /healthz.
            # Six apps in one 25-build sweep served errors the gate did not see.
            #
            # 4xx is deliberately still "responded": a 401/403 landing page is a
            # legitimate design for an app that puts auth first, and the crowd
            # can sign up. A 5xx is never a design.
            server_error = status is not None and 500 <= status < 600
            runs = (
                responded and not server_error and app is not None and app.is_running()
            )
            if app is not None:
                detail.append(f"url={app.url}")
            if server_error:
                detail.append(
                    f"entry URL returned HTTP {status}: the app starts but does "
                    f"not serve"
                )
        else:
            # Every app in the bench is a web app, so a manifest with no port
            # cannot be served and cannot be reviewed, whatever else it does.
            runs = False
            detail.append("no run.port: a web app must declare the port it serves on")

        claims = builds and runs and smoke_ok
        return VerifyResult(build_id, builds, runs, claims, "; ".join(detail))
    finally:
        session.close()


def try_app(
    build_id: str,
    host: AppHost,
    *,
    script: list[dict] | None = None,
    max_steps: int = 24,
) -> TryResult:
    """Have an agent *use* a running app and report what it saw.

    This is the sync convenience wrapper over the async
    :func:`viral_bench.crowd.interaction.try_app`: it drives the app like a human
    per app type (real browser for a single-page app, terminal for a CLI, the
    chat loop for a bot). A web app's shared instance comes from ``host`` (one app,
    many agents). The returned :class:`TryResult` carries the full interaction
    transcript as its observation.

    The crowd loop, which is already async, should call
    :func:`viral_bench.crowd.interaction.try_app` directly rather than this
    wrapper. ``script`` optionally pins the exact actions to run (see that
    function). Otherwise a type-appropriate default interaction is used.
    """
    # Imported here (not at module top) so the founder package does not hard-depend
    # on the crowd interaction stack -- and its optional browser -- merely to import
    # verify. The crowd layer itself degrades gracefully when no browser is present.
    from viral_bench.crowd.interaction import try_app as _crowd_try_app

    trace = _run_async(
        _crowd_try_app(
            build_id,
            app_host=host,
            container=host.container,
            script=script,
            max_steps=max_steps,
        )
    )
    substantive = [s for s in trace.steps if s.action != "finish"]
    ok = any(s.ok for s in substantive)
    return TryResult(
        build_id=build_id,
        app_type=trace.app_type,
        ok=ok,
        observation=trace.render(),
        url=trace.target_url,
    )


def probe_endpoint(
    build_id: str,
    host: AppHost,
    *,
    path: str = "/",
    timeout: float = 10.0,
) -> TryResult:
    """Fetch one specific HTTP endpoint of a running app (a targeted infra probe).

    Unlike :func:`try_app`, this is a low-level HTTP GET, not a human-like trial:
    it does not run the page's JavaScript. It is the right tool when you need to
    validate a specific route/response -- e.g. confirming an API-key-proxy
    endpoint echoes a nonce (see ``scripts/smoke_key_injection.py``) -- rather
    than judging the app as a user would. For a non-web app it falls back to the
    manifest's smoke command via the runtime.
    """
    app = host.get(build_id)
    if app.url:
        target = app.url.rstrip("/") + "/" + path.lstrip("/")
        responded, status, body = _http_probe(target, timeout=timeout)
        ok = responded and status is not None and 200 <= status < 400
        return TryResult(build_id, app.app_type, ok, body, app.url)

    session = host.session(build_id)
    assert session is not None  # host.get(...) created it above
    probe = session.manifest.test.smoke or "true"
    proc = session.runtime.exec(probe, cwd=session.manifest.run.cwd, timeout=timeout)
    observation = (proc.stdout or proc.stderr or "")[:_MAX_OBSERVATION]
    return TryResult(build_id, app.app_type, proc.returncode == 0, observation)


def _run_async(coro):
    """Run an async coroutine to completion from sync code, safely.

    Uses a dedicated thread with its own event loop so this works whether or not
    the caller already has a running loop (e.g. if invoked from within async
    code), sidestepping ``asyncio.run``'s "loop already running" error.
    """
    import asyncio
    import threading

    box: dict[str, object] = {}

    def runner() -> None:
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller thread
            box["error"] = exc

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["result"]
