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

"""Open a trial for a build, and guarantee it is cleaned up.

``open_trial(build_id, ...)`` is the one entry point the toolkit (and the CLI,
and tests) use to get a ready-to-drive
:class:`~viral_bench.crowd.interaction.clients.AppClient`.

Every app in the bench is a web app, so there is no per-type dispatch left. A
*running* instance is obtained -- a shared one from an
:class:`~viral_bench.founder.apphost.AppHost` when provided, so many agents hit
one app like a real deployment, otherwise a private one -- and a real browser
page is attached. If no browser is available it degrades to a static-HTTP client
rather than failing the run.

The returned client owns whatever it created (browser context, app server,
materialized copy) and releases it on ``await client.close()``. A shared
``AppHost`` instance is deliberately left running for other agents.
"""

from __future__ import annotations

from pathlib import Path

from viral_bench.crowd.interaction.browser import (
    BrowserConfig,
    BrowserEngine,
    browser_available,
)
from viral_bench.crowd.interaction.clients import (
    AppClient,
    StaticWebAppClient,
    UndeliverableAppClient,
    WebAppClient,
)
from viral_bench.crowd.interaction.trace import InteractionTrace
from viral_bench.founder.apphost import AppHost
from viral_bench.founder.build import BuildError, load_build_record
from viral_bench.founder.manifest import (
    ALLOWED_APP_TYPES,
    MANIFEST_FILENAME,
    Manifest,
    ManifestError,
    load_manifest,
)
from viral_bench.founder.runner import open_session


class TrialError(RuntimeError):
    """Raised when a trial cannot be opened for a build."""


def peek_manifest(build_id: str) -> Manifest:
    """Load a build's manifest without materializing or starting it.

    Raises ``ManifestError`` when the file is absent or malformed --
    :func:`manifest_or_none` is the caller-friendly form.
    """
    record = load_build_record(build_id)
    return load_manifest(Path(record.app_dir) / MANIFEST_FILENAME)


def manifest_or_none(build_id: str) -> Manifest | None:
    """The build's manifest, or ``None`` if it shipped without a usable one.

    ``None`` means *undeliverable*: the founder produced no machine-readable
    contract saying how to start the app, so nothing -- not the runtime, not the
    validity gate, not a trier -- can launch it. That is a fact about the build,
    not an error in the harness, and the two must not be confused: a harness
    fault leaves the app unverified (no score penalty), while an undeliverable
    build is a delivery failure the crowd should see and score.
    """
    try:
        return peek_manifest(build_id)
    except (ManifestError, BuildError, OSError):
        return None


async def open_trial(
    build_id: str,
    *,
    app_host: AppHost | None = None,
    container: bool = True,
    browser_engine: BrowserEngine | None = None,
    browser_config: BrowserConfig | None = None,
    use_browser: bool | None = None,
    env_map: dict[str, str] | None = None,
    start_wait: float = 90.0,
    trace: InteractionTrace | None = None,
    undeliverable_app_type: str | None = None,
    env_notice: bool = True,
) -> AppClient:
    """Open a hands-on trial for ``build_id`` and return an app client.

    Args:
        build_id: The build to try.
        app_host: A shared running-instance pool. When given, a web app is served
            once and shared across agents. When ``None`` a private instance is
            started for this trial and torn down on close.
        container: Run the app in a container (the crowd default) vs on the host.
            Ignored for the web path when ``app_host`` is given (the host's own
            setting wins).
        browser_engine: A shared browser to reuse (crowd scale). When ``None`` a
            private engine is created for a web trial and closed on close.
        browser_config: Browser settings when a private engine is created.
        use_browser: ``None`` = use a real browser iff one is available (else
            degrade to static). ``True`` = require a real browser (raise if it
            cannot launch). ``False`` = force the static-HTTP client.
        env_map: ``{container_var: host_var}`` env mapping for the app (defaults
            to the founder's ``DEFAULT_ENV_MAP``). Container runs only.
        start_wait: How long to wait for a web app's port to come up.
        trace: An existing trace to append to (the toolkit shares one across
            tool calls). A fresh trace is created when omitted.
        undeliverable_app_type: App type to assume when the build has no usable
            manifest (from the idea's declared scope). Only used in that case.

    Returns:
        A ready :class:`~viral_bench.crowd.interaction.clients.AppClient`.
    """
    manifest = manifest_or_none(build_id)
    # An undeliverable build still gets a trial -- the agent discovers
    # there is nothing to launch, which is exactly what a real user would find.
    # ``app_type`` then comes from the idea's declared scope rather than from the
    # manifest the founder failed to write.
    app_type = (
        manifest.app_type
        if manifest is not None
        else (undeliverable_app_type or "client-app")
    )
    if trace is None:
        trace = InteractionTrace(build_id=build_id, app_type=app_type)
    if manifest is None:
        return UndeliverableAppClient(build_id, app_type, trace)

    # Every app in the bench is a web app now, so there is nothing to dispatch on:
    # a client-app and a full-stack-app are both opened in a browser and driven
    # the same way. The scope survives only as a label for analysis (and as the
    # thing that decides whether an app needs a database), not as a code path.
    if app_type not in ALLOWED_APP_TYPES:
        raise TrialError(f"unsupported app_type {app_type!r} for build {build_id!r}")

    return await _open_web_trial(
        build_id,
        manifest,
        trace,
        app_host=app_host,
        container=container,
        browser_engine=browser_engine,
        browser_config=browser_config,
        use_browser=use_browser,
        env_map=env_map,
        start_wait=start_wait,
        env_notice=env_notice,
    )


async def _open_web_trial(
    build_id: str,
    manifest: Manifest,
    trace: InteractionTrace,
    *,
    app_host: AppHost | None,
    container: bool,
    browser_engine: BrowserEngine | None,
    browser_config: BrowserConfig | None,
    use_browser: bool | None,
    env_map: dict[str, str] | None,
    start_wait: float,
    env_notice: bool = True,
) -> AppClient:
    # 1. Get a running instance + its URL (shared via AppHost, or private).
    owned_session = None
    if app_host is not None:
        app = app_host.get(build_id, wait_timeout=start_wait)
        url = app.url
    else:
        owned_session = open_session(build_id, container=container, env_map=env_map)
        owned_session.setup()
        app = owned_session.start(wait_timeout=start_wait)
        url = app.url

    if not url:
        if owned_session is not None:
            owned_session.close()
        raise TrialError(
            f"web app {build_id!r} did not expose a URL (no run.port?); cannot drive it"
        )
    trace.target_url = url

    # 2. Decide real browser vs static fallback.
    want_browser = browser_available() if use_browser is None else use_browser
    if not want_browser:
        return StaticWebAppClient(url, trace, session=owned_session)

    engine = browser_engine
    owns_engine = False
    try:
        if engine is None:
            engine = BrowserEngine(browser_config)
            owns_engine = True
            await engine.start()
        page = await engine.open_page()
    except Exception as exc:  # noqa: BLE001
        # Real browser requested-or-auto but could not launch. If the caller
        # forced it, surface the error, otherwise degrade gracefully.
        if owns_engine and engine is not None:
            await engine.close()
        if use_browser is True:
            if owned_session is not None:
                owned_session.close()
            raise TrialError(
                f"could not start a browser for {build_id!r}: {exc}"
            ) from exc
        return StaticWebAppClient(url, trace, session=owned_session)

    return WebAppClient(
        url,
        page,
        trace,
        engine=engine,
        owns_engine=owns_engine,
        session=owned_session,
        env_notice=env_notice,
    )
