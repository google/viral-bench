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

"""Google Vertex AI wiring, for the ``google-vertex*`` providers only.

The founder harness drives opencode against **Vertex AI model garden** rather
than the Gemini Developer API, which is what lets the founder model be any
serverless model garden model -- Gemini via opencode's ``google-vertex``
provider, Claude via ``google-vertex-anthropic`` -- with only a flag change (see
:mod:`viral_bench.founder.models`). The crowd and the founder-built apps
deliberately stay on the Gemini Developer API and do NOT use this module.

Authentication is Application Default Credentials (ADC) -- no API key. Two things
make this safe to run alongside Cloud Code (which itself reaches Vertex via the
ambient ``GOOGLE_CLOUD_PROJECT`` + ADC):

* We NEVER mutate this process's environment, the gcloud default project, or the
  ADC file. Instead we scope our project + quota project to *our own* google-genai
  client (via explicit credentials) and to the *opencode child process* env only.
* This machine's ADC has no quota project set, and Vertex needs one, so every
  client/subprocess we create sets the quota project explicitly.

Override the project/location by exporting ``VERTEX_PROJECT`` / ``VERTEX_LOCATION``
(or adding them to the repo ``.env``).
"""

from __future__ import annotations

from viral_bench import config as _config
from viral_bench.env import read_key

#: No default project: Vertex bills a project you own, and guessing one would
#: either fail confusingly or spend someone else's budget. Set VERTEX_PROJECT
#: (env or .env) or `vertex.project` in config/founder.yaml. Only needed if you
#: actually name a `google-vertex/...` model -- every other provider ignores it.
DEFAULT_VERTEX_PROJECT = ""

#: Vertex location. ``global`` maximises availability at no extra cost; use a
#: region (e.g. ``us-central1``) only for data-residency needs.
DEFAULT_VERTEX_LOCATION = "global"


def vertex_project() -> str:
    """Return the Vertex project.

    Precedence: ``VERTEX_PROJECT`` (env/.env) > config/founder.yaml > the default.
    """
    return read_key("VERTEX_PROJECT") or _config.vertex_project(DEFAULT_VERTEX_PROJECT)


def vertex_location() -> str:
    """Return the Vertex location (env/.env > config/founder.yaml > default)."""
    return read_key("VERTEX_LOCATION") or _config.vertex_location(
        DEFAULT_VERTEX_LOCATION
    )


def vertex_subprocess_env() -> dict[str, str]:
    """Env overrides that point a CHILD process (opencode) at our Vertex project.

    These are merged into the opencode subprocess env only (see
    :meth:`viral_bench.founder.harness.OpenCodeRunner._build_env`); they override
    the inherited ``GOOGLE_CLOUD_PROJECT`` (which Cloud Code sets to *its* project)
    for the child alone, so Cloud Code's own process is never affected.

    Both Vertex providers are covered, because they do not read the same vars:

    * ``google-vertex`` (Gemini) reads ``GOOGLE_CLOUD_PROJECT`` / ``VERTEX_LOCATION``.
    * ``google-vertex-anthropic`` (Claude) reads ``GOOGLE_VERTEX_PROJECT`` and
      resolves location as ``GOOGLE_VERTEX_LOCATION -> GOOGLE_CLOUD_LOCATION ->
      VERTEX_LOCATION -> "us-central1"``. That last fallback is the dangerous
      one: no Claude 5 model is served in ``us-central1``, so an unset location
      would fail as a confusing not-found rather than an obvious misconfiguration.
      We therefore set every name explicitly rather than trusting the chain.

    ``GOOGLE_CLOUD_QUOTA_PROJECT`` covers the no-quota-project ADC, and
    ``GOOGLE_API_USE_CLIENT_CERTIFICATE=false`` avoids a context-aware mTLS path
    that can otherwise fail on the first call.
    """
    project = vertex_project()
    location = vertex_location()
    return {
        "GOOGLE_CLOUD_PROJECT": project,
        "GOOGLE_CLOUD_QUOTA_PROJECT": project,
        "GOOGLE_VERTEX_PROJECT": project,
        "VERTEX_LOCATION": location,
        "GOOGLE_CLOUD_LOCATION": location,
        "GOOGLE_VERTEX_LOCATION": location,
        "GOOGLE_API_USE_CLIENT_CERTIFICATE": "false",
    }


def vertex_genai_client():
    """Build a google-genai ``Client`` for Vertex, scoped to our project.

    Uses ADC but sets the quota project on the credentials object (not globally),
    so it neither warns nor borrows Cloud Code's ambient project. Used for the
    founder model preflight and ``viral-bench models --check``.

    Raises:
        RuntimeError: if google-genai / google-auth are unavailable or ADC is not
            configured (with an actionable hint).
    """
    import os
    import warnings

    try:
        import google.auth
        import httpx as _httpx
        from google import genai
        from google.genai import types as genai_types
    except ImportError as exc:  # pragma: no cover - deps are declared
        raise RuntimeError(
            "google-genai + google-auth are required for the Vertex path "
            "(they ship with the project; run `uv sync`)."
        ) from exc

    # Force plain TLS (skip the context-aware device client certificate) for THIS
    # process. On corp machines the mTLS path can fail the first Vertex call with a
    # missing-pyOpenSSL error; plain TLS + the OAuth bearer token is accepted for
    # Vertex here. This is the viral-bench process, not Cloud Code, and it changes
    # only our TLS transport -- never the project, billing, or ADC.
    os.environ.setdefault("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")

    project = vertex_project()
    try:
        # ADC here has no quota project; we set it on the creds below, so silence
        # the (correct-but-noisy) warning emitted during default().
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
    except Exception as exc:  # noqa: BLE001 - surface ADC setup problems clearly
        raise RuntimeError(
            "no Application Default Credentials found for Vertex. Run "
            "`gcloud auth application-default login` (or set "
            "GOOGLE_APPLICATION_CREDENTIALS)."
        ) from exc

    # Scope the quota project to THIS client only; ADC here has none set.
    try:
        creds = creds.with_quota_project(project)
    except Exception:  # noqa: BLE001 - some credential types lack this; harmless
        pass

    # A REQUEST TIMEOUT, because without one a dead connection hangs the caller
    # forever. google-genai defaults to no timeout, so a request whose socket the
    # far end has torn down never returns and nothing upstream can tell the
    # difference between "thinking" and "gone".
    #
    # That is not hypothetical: it is what stopped this sweep. Crowd runs reached
    # the final interview phase, issued their per-agent calls, and froze -- 13
    # connections to 13 distinct Google frontends all in CLOSE-WAIT (the remote
    # closed, the client never did), zero ESTABLISHED, trace rows pinned to the
    # same count for 31 minutes, one thread spinning at ~100% while 243 of 244
    # waited on a futex. Nine of ten runs per batch died that way, each holding a
    # slot until the 3600s wall clock, and the whole sweep completed ZERO cells
    # in an hour.
    #
    # With a timeout the request raises instead, and the existing machinery does
    # the right thing: `_is_transient` treats a timeout as retryable, and if the
    # retries are exhausted `_skip_turn` records one skipped agent turn and the
    # run CONTINUES to a scored result. A run that finishes having skipped a turn
    # is worth immeasurably more than one that never finishes at all.
    #
    # 180s is chosen to be far above any legitimate call -- a whole successful run
    # of 30 agents over 3 rounds has a median duration of 286s -- and far below
    # both the 3600s wall clock and the 900s stall reaper, so a stuck call fails,
    # retries, and recovers inside the budget instead of being culled.
    timeout_ms = int(os.environ.get("VERTEX_REQUEST_TIMEOUT_MS", "180000"))

    # AND NO CONNECTION REUSE, which is the other half of the same failure.
    #
    # Vertex itself is healthy under this workload -- three sequential probes
    # during the worst of the stalling returned "OK" in 0.59-0.77s. The damage is
    # client-side and only appears under the crowd's concurrency (30 agents per
    # run x 10 runs). The signature is a keepalive pool holding connections the
    # far end has already closed: every socket in CLOSE-WAIT, zero ESTABLISHED,
    # one thread spinning at ~100% while 243 of 244 park on a futex, and no
    # further progress ever -- not one reaped run has resumed.
    #
    # `max_keepalive_connections=0` makes every request open a fresh connection,
    # so a dead socket can never be handed back out of the pool. The cost is one
    # TLS handshake per call, tens of milliseconds against a call that takes
    # 0.6s+, which is nothing next to losing the run.
    #
    # Both client kwargs are set: the crowd uses the sync path for agent turns
    # and the async path (`aio.models.generate_content`) for the interview phase,
    # and it is the interview phase that was freezing.
    limits = {"limits": _httpx.Limits(max_keepalive_connections=0)}
    return genai.Client(
        vertexai=True,
        project=project,
        location=vertex_location(),
        credentials=creds,
        http_options=genai_types.HttpOptions(
            timeout=timeout_ms,
            client_args=limits,
            async_client_args=limits,
        ),
    )


# -- non-Gemini (partner) model access --------------------------------------- #
#
# google-genai speaks only the Gemini API surface, so it cannot reach a partner
# model like Claude. Those live behind Vertex's `rawPredict` passthrough, which
# is a plain REST call -- so we make it with google-auth + urllib rather than
# adding an `anthropic[vertex]` dependency for one health check. That also keeps
# the crowd's isolated .venv-crowd unaffected.

#: Vertex multi-region endpoints, which use a different host shape to both the
#: global endpoint and ordinary regions.
_MULTI_REGIONS = ("us", "eu")


def vertex_api_host(location: str | None = None) -> str:
    """Return the Vertex API host for a location.

    Three shapes, and getting this wrong is a 404 rather than a clear error:
    ``global`` has no prefix at all (``global-aiplatform...`` does not exist),
    multi-region endpoints are ``aiplatform.<loc>.rep``, and everything else is
    the familiar ``<loc>-aiplatform``.
    """
    loc = location or vertex_location()
    if loc == "global":
        return "aiplatform.googleapis.com"
    if loc in _MULTI_REGIONS:
        return f"aiplatform.{loc}.rep.googleapis.com"
    return f"{loc}-aiplatform.googleapis.com"


def vertex_access_token() -> str:
    """Return a fresh OAuth access token from ADC for Vertex REST calls.

    Raises:
        RuntimeError: if google-auth is unavailable or ADC is not configured.
    """
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:  # pragma: no cover - deps are declared
        raise RuntimeError(
            "google-auth is required for the Vertex partner-model path "
            "(it ships with the project; run `uv sync`)."
        ) from exc

    import warnings

    try:
        with warnings.catch_warnings():
            # ADC here has no quota project; we pass the project on the URL path,
            # so the (correct-but-noisy) warning is not actionable.
            warnings.simplefilter("ignore")
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        creds.refresh(google.auth.transport.requests.Request())
    except Exception as exc:  # noqa: BLE001 - surface ADC setup problems clearly
        raise RuntimeError(
            "no Application Default Credentials found for Vertex. Run "
            "`gcloud auth application-default login` (or set "
            "GOOGLE_APPLICATION_CREDENTIALS)."
        ) from exc
    token = getattr(creds, "token", None)
    if not token:
        raise RuntimeError("ADC returned no access token for Vertex.")
    return str(token)


def anthropic_ping(model_id: str, *, timeout_s: float = 30.0) -> None:
    """Prove a Claude model is actually callable on our Vertex project.

    Sends a one-token ``rawPredict``. That costs a fraction of a cent, and it is
    deliberately not the free ``count-tokens`` endpoint: count-tokens answers 200
    on a project that has NOT enabled the model (measured -- it returned
    ``{"input_tokens":8}`` from a project whose inference calls 404), so it
    proves reachability but not entitlement. Entitlement is exactly the failure
    this preflight exists to catch, since a partner model needs a one-time Model
    Garden enable per project.

    Raises:
        RuntimeError: with the API's own message on any non-200 response.
    """
    import json
    import urllib.error
    import urllib.request

    project = vertex_project()
    location = vertex_location()
    url = (
        f"https://{vertex_api_host(location)}/v1/projects/{project}"
        f"/locations/{location}/publishers/anthropic/models/{model_id}:rawPredict"
    )
    payload = json.dumps(
        {
            "anthropic_version": "vertex-2023-10-16",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - fixed https Vertex endpoint
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {vertex_access_token()}",
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-user-project": project,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            body = json.loads(exc.read().decode("utf-8", "replace"))
            detail = str(body.get("error", {}).get("message", "")).strip()
        except Exception:  # noqa: BLE001 - fall back to the status line
            detail = ""
        raise RuntimeError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        host = vertex_api_host(location)
        raise RuntimeError(f"could not reach {host}: {exc.reason}") from exc
