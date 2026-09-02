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

"""End-to-end smoke test for GEMINI_API_KEY injection into founded apps.

Proves the full chain the design promises: a founder-built app, running inside the
rootless Podman sandbox, can read the injected ``GEMINI_API_KEY`` (sourced from
the host's ``GEMINI_API_KEY_FOUNDER`` in ``.env``) and return a *live*
LLM-generated response to a "crowd" tester -- exactly the runtime path the OASIS
crowd's app trials use.

What it does:
  1. Reads GEMINI_API_KEY_FOUNDER from the environment / .env and preflights it
     against the Gemini REST API, auto-picking a text model the key can call.
  2. Ships a tiny single-page "founded app" whose ``/generate`` endpoint calls
     Gemini using the in-container GEMINI_API_KEY (stdlib only; no SDK install).
   3. Runs it in Podman via the real crowd path: verify_code (validity gate) then
      AppHost + probe_endpoint (a targeted HTTP probe) hitting ``/generate``. A
      random nonce in the prompt proves the response is live, not canned.
  4. Runs the same app WITHOUT the key to show it degrades gracefully (HTTP 503)
     while ``/`` still serves -- proving the key is what unlocks the LLM feature.

This makes real (cheap) Gemini calls and needs Podman + the runtime image, so it
is a manual script, not a pytest.

Run:
    uv run python scripts/smoke_key_injection.py
    uv run python scripts/smoke_key_injection.py --check          # key preflight only
    uv run python scripts/smoke_key_injection.py --model gemini-2.0-flash
    uv run python scripts/smoke_key_injection.py --keep           # keep temp build dir
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlencode

from dotenv import load_dotenv

_GLM = "https://generativelanguage.googleapis.com/v1beta"

# Preferred text models, newest/cheapest-friendly first. Intersected with what
# the founder key can actually call; falls back to any flash text model.
_MODEL_PRIORITY = (
    "gemini-2.0-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash",
)
_SKIP = ("tts", "image", "embedding", "vision", "aqa", "learnlm", "live", "audio")


def _list_generate_models(key: str) -> list[str]:
    """Return short ids of models the key can call with generateContent."""
    req = urllib.request.Request(
        f"{_GLM}/models", headers={"x-goog-api-key": key}, method="GET"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    out = []
    for m in data.get("models", []):
        methods = m.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            out.append(m.get("name", "").split("/")[-1])
    return [x for x in out if x]


def _rest_generate(key: str, model: str, prompt: str, *, timeout: float = 30.0) -> str:
    """Call Gemini generateContent over REST and return the text (raises on error)."""
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(
        f"{_GLM}/models/{model}:generateContent",
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _pick_model(key: str, override: str | None) -> str:
    """Choose a text model the founder key can call (verified with a tiny call)."""
    if override:
        _rest_generate(key, override, "ping")  # raises if unusable
        return override
    available = set(_list_generate_models(key))
    ordered = [m for m in _MODEL_PRIORITY if m in available]
    ordered += sorted(
        m
        for m in available
        if "flash" in m and not any(s in m for s in _SKIP) and m not in ordered
    )
    ordered += sorted(
        m for m in available if not any(s in m for s in _SKIP) and m not in ordered
    )
    last_err = "no candidate models"
    for model in ordered:
        try:
            _rest_generate(key, model, "ping")
            return model
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            last_err = f"{model}: {exc}"
    raise RuntimeError(f"no usable Gemini text model for this key ({last_err})")


# The founded app. A single-page app served by stdlib http.server:
#   GET /          -> static HTML (works with no key -> passes the validity gate)
#   GET /generate  -> calls Gemini with GEMINI_API_KEY; HTTP 503 if the key is absent
# __MODEL__ is replaced with the chosen model before shipping.
_APP_PY = r'''"""Key-injection smoke app (stdlib only)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

MODEL = "__MODEL__"
_URL = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"

INDEX = b"""<!doctype html><meta charset=utf-8><title>KeyInjectSmoke</title>
<h1>Gemini key-injection smoke app</h1>
<p>Try <a href="/generate?prompt=hello">/generate?prompt=...</a></p>"""


def generate(prompt):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return 503, {"error": "GEMINI_API_KEY not set; LLM feature unavailable"}
    payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
    req = urllib.request.Request(
        _URL % MODEL,
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": "gemini HTTP %s" % exc.code,
                          "detail": exc.read().decode("utf-8", "replace")[:400]}
    except Exception as exc:  # noqa: BLE001
        return 502, {"error": "request failed: %s" % exc}
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return 502, {"error": "unexpected response", "raw": data}
    return 200, {"model": MODEL, "text": text}


class H(BaseHTTPRequestHandler):
    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, INDEX, "text/html; charset=utf-8")
        elif u.path == "/generate":
            prompt = (parse_qs(u.query).get("prompt") or ["Say hi."])[0]
            self._send(*generate(prompt))
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
'''

_MANIFEST = {
    "app_type": "single-page-app",
    "title": "KeyInjectSmoke",
    "summary": "Calls Gemini with the injected key and returns generated text.",
    "setup": [],
    "run": {
        "command": "python3 app.py",
        "port": 8000,
        "url": "http://localhost:8000/",
    },
    "test": {
        "manual": ["Open /", "GET /generate?prompt=... for an LLM response"],
        # network-free health check (syntax-compile the app; no key needed)
        "smoke": 'python3 -c \'compile(open("app.py").read(), "app.py", "exec")\'',
    },
}


class _SmokeHarness:
    """Ships the fixture app (app.py + manifest + README) for one build."""

    def __init__(self, app_py: str) -> None:
        self.app_py = app_py

    def run(self, idea, workspace):  # noqa: ANN001 - matches FounderHarness protocol
        from viral_bench.founder.harness import HarnessResult, PhaseResult

        (workspace.app_dir / "app.py").write_text(self.app_py, encoding="utf-8")
        (workspace.app_dir / "viralbench.json").write_text(
            json.dumps(_MANIFEST, indent=2), encoding="utf-8"
        )
        (workspace.app_dir / "README.md").write_text(
            "# KeyInjectSmoke\nGET /generate?prompt=... for a Gemini response.\n",
            encoding="utf-8",
        )
        t = workspace.app_dir / "t"
        return HarnessResult(
            model="smoke-fixture",
            phases=[
                PhaseResult("design", 0, t, 0.0),
                PhaseResult("build", 0, t, 0.0),
            ],
        )


def _observation_text(observation: str) -> str:
    """Pull the 'text' field out of a /generate JSON body (else return raw)."""
    try:
        return json.loads(observation).get("text", observation)
    except (json.JSONDecodeError, AttributeError):
        return observation


def main() -> int:
    load_dotenv()  # read .env at repo root so GEMINI_API_KEY_FOUNDER is available
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="force a specific Gemini model")
    parser.add_argument(
        "--check", action="store_true", help="only preflight the founder key"
    )
    parser.add_argument("--keep", action="store_true", help="keep the temp build dir")
    args = parser.parse_args()

    from viral_bench.founder.appenv import DEFAULT_ENV_MAP, read_key

    print("=== 1. founder key preflight (host) ===")
    founder_key = read_key("GEMINI_API_KEY_FOUNDER")
    if not founder_key:
        print("ERROR: GEMINI_API_KEY_FOUNDER not set (checked env and .env).")
        return 1
    print(f"GEMINI_API_KEY_FOUNDER: found (len {len(founder_key)})")
    try:
        model = _pick_model(founder_key, args.model)
    except (urllib.error.URLError, RuntimeError) as exc:
        print(f"ERROR: founder key cannot call the Gemini API: {exc}")
        return 1
    print(f"Usable Gemini text model for the founder key: {model}")
    print(f"env_map (container_var: host_var): {DEFAULT_ENV_MAP}")
    if args.check:
        print("\nPreflight OK.")
        return 0

    from viral_bench.founder.apphost import AppHost
    from viral_bench.founder.build import run_build
    from viral_bench.founder.runtime import ContainerRuntime
    from viral_bench.founder.verify import probe_endpoint, verify_code

    if not ContainerRuntime.available("podman"):
        print("ERROR: podman is not installed / not on PATH.")
        return 1
    if not ContainerRuntime(Path("."), runtime="podman").image_exists():
        print(
            "ERROR: runtime image missing. Build it with:\n"
            "  podman build -t viralbench-runtime:latest -f docker/Containerfile docker"
        )
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="vb-smoke-keyinject-"))
    os.environ["VIRAL_BENCH_BUILDS_DIR"] = str(tmp)
    rc = 1
    try:
        print("\n=== 2. ship the fixture 'founded app' ===")
        app_py = _APP_PY.replace("__MODEL__", model)
        record = run_build("sliding_tile_game", harness=_SmokeHarness(app_py))
        if record.status != "ok":
            print(f"ERROR: fixture build failed: {record.status} {record.error}")
            return 1
        print(f"build_id: {record.build_id}  (builds dir: {tmp})")

        print("\n=== 3. validity gate in Podman (verify_code, container) ===")
        vr = verify_code(record.build_id, container=True)
        print(
            f"builds={vr.builds} runs={vr.runs} "
            f"does_what_it_claims={vr.does_what_it_claims}  [{vr.detail}]"
        )
        if not (vr.builds and vr.runs):
            print("ERROR: app did not build/run in the container.")
            return 1

        print("\n=== 4. crowd trial WITH key -> live Gemini response (try_app) ===")
        nonce = uuid.uuid4().hex[:12]
        prompt = (
            "Reply in one short friendly sentence for a sandboxed-app smoke test, "
            f"and include this exact code once: {nonce}"
        )
        path = "generate?" + urlencode({"prompt": prompt})
        with AppHost(container=True) as host:  # default env_map -> founder key
            tried = probe_endpoint(record.build_id, host, path=path, timeout=60.0)
        text = _observation_text(tried.observation)
        print(f"probe_endpoint ok={tried.ok}  url={tried.url}")
        print(f"prompt nonce: {nonce}")
        print(f"LLM response: {text!r}")
        live_ok = tried.ok and nonce in tried.observation
        print(f"live LLM call succeeded (nonce echoed): {live_ok}")

        print("\n=== 5. crowd trial WITHOUT key -> graceful 503 (probe_endpoint) ===")
        with AppHost(container=True, env_map={}) as host:  # no key injected
            gen = probe_endpoint(
                record.build_id, host, path="generate?prompt=hi", timeout=20.0
            )
            root = probe_endpoint(record.build_id, host, path="/", timeout=20.0)
        print(f"/generate ok={gen.ok} (want False)  body={gen.observation!r}")
        print(f"/ ok={root.ok} (want True; app still serves without a key)")
        no_key_ok = (not gen.ok) and root.ok

        print("\n=== RESULT ===")
        if live_ok and no_key_ok:
            print(
                "PASS: injected key produced a live LLM response in the sandbox, "
                "and the app degrades gracefully without it."
            )
            rc = 0
        else:
            print("FAIL: see above.")
            rc = 1
    finally:
        if args.keep:
            print(f"\n(temp build dir kept: {tmp})")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
