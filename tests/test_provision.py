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

"""Tests for the per-build dependency cache.

The bug this exists to prevent is not subtle once stated: ``materialize_build``
hands the crowd a ``git clone`` of the shipped branch, which correctly contains
only what the founder committed, and the founder committed neither the
``node_modules`` its own .gitignore excludes nor the ``pip install`` it ran in
its shell. Measured on the r3 sweep, apps that could not start were the entire
unscorable set -- and because a failed run writes no ``run_summary.json``, they
were indistinguishable from cells nobody had tried.
"""

from __future__ import annotations

import json

import pytest

from viral_bench.founder import provision

# Aliased: pytest tries to collect anything named Test*.
from viral_bench.founder.manifest import Manifest, RunSpec
from viral_bench.founder.manifest import TestSpec as ManifestTestSpec


def _app(tmp_path, **files):
    app = tmp_path / "app"
    app.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        target = app / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return app


def _manifest(command="python3 server.py", setup=(), smoke=None) -> Manifest:
    return Manifest(
        app_type="client-app",
        title="t",
        summary="s",
        run=RunSpec(command=command, port=8000),
        setup=tuple(setup),
        test=ManifestTestSpec(smoke=smoke),
    )


def test_undeclared_imports_are_the_dependency_list(tmp_path, monkeypatch) -> None:
    """An app that declares nothing still names its dependencies -- by importing them.

    This is the LARGEST failure class, not an edge case: the founder installed
    Pillow on the host, imported it, and shipped neither a requirements.txt nor a
    setup step, because from where it stood nothing needed installing.
    """
    monkeypatch.setattr(provision, "depcache_root", lambda: tmp_path / "cache")
    app = _app(
        tmp_path,
        **{
            "server.py": "import os\nfrom PIL import Image\nimport flask\n",
            "viralbench.json": json.dumps(
                {
                    "app_type": "client-app",
                    "title": "t",
                    "summary": "s",
                    "run": {"command": "python3 server.py", "port": 8000},
                }
            ),
        },
    )
    plan = provision.plan_for("b1", app)
    joined = "\n".join(plan.steps)
    assert "pillow" in joined, "PIL must map to its distribution name"
    assert "flask" in joined
    assert " os" not in joined, "stdlib must never be installed"


def test_a_runner_named_only_on_the_command_line_is_installed(tmp_path, monkeypatch):
    """`uv run uvicorn app:x` imports fastapi and never imports uvicorn.

    Found by the fix failing: provisioning installed the framework, the app still
    died with ``Failed to spawn: uvicorn``, because an import scan cannot see a
    binary that only ever appears in the run command.
    """
    monkeypatch.setattr(provision, "depcache_root", lambda: tmp_path / "cache")
    app = _app(
        tmp_path,
        **{
            "main.py": "from fastapi import FastAPI\napp = FastAPI()\n",
            "viralbench.json": json.dumps(
                {
                    "app_type": "client-app",
                    "title": "t",
                    "summary": "s",
                    "run": {
                        "command": "uv run uvicorn main:app --port 8000",
                        "port": 8000,
                    },
                }
            ),
        },
    )
    steps = "\n".join(provision.plan_for("b1", app).steps)
    assert "fastapi" in steps
    assert "uvicorn" in steps


def test_test_only_imports_are_not_runtime_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(provision, "depcache_root", lambda: tmp_path / "cache")
    app = _app(
        tmp_path,
        **{
            "server.py": "import flask\n",
            "test_server.py": "import pytest\nimport responses\n",
            "viralbench.json": json.dumps(
                {
                    "app_type": "client-app",
                    "title": "t",
                    "summary": "s",
                    "run": {"command": "python3 server.py", "port": 8000},
                }
            ),
        },
    )
    steps = "\n".join(provision.plan_for("b1", app).steps)
    assert "flask" in steps
    assert "pytest" not in steps
    assert "responses" not in steps


def test_a_clone_that_ships_node_modules_is_never_mounted_over(tmp_path, monkeypatch):
    """236 r3 builds committed node_modules. Mounting a cache there hides them."""
    monkeypatch.setattr(provision, "depcache_root", lambda: tmp_path / "cache")
    app = _app(
        tmp_path,
        **{
            "package.json": '{"name":"t","dependencies":{"express":"^4"}}',
            "node_modules/express/index.js": "module.exports = 1;",
            "viralbench.json": json.dumps(
                {
                    "app_type": "client-app",
                    "title": "t",
                    "summary": "s",
                    "run": {"command": "node server.js", "port": 8000},
                }
            ),
        },
    )
    plan = provision.plan_for("b1", app)
    assert not any("npm" in s for s in plan.steps)
    assert provision.cache_mounts("b1", app) == []


def test_a_clone_without_node_modules_gets_an_install(tmp_path, monkeypatch):
    monkeypatch.setattr(provision, "depcache_root", lambda: tmp_path / "cache")
    app = _app(
        tmp_path,
        **{
            "package.json": '{"name":"t","dependencies":{"express":"^4"}}',
            "package-lock.json": "{}",
            "viralbench.json": json.dumps(
                {
                    "app_type": "client-app",
                    "title": "t",
                    "summary": "s",
                    "run": {"command": "node server.js", "port": 8000},
                }
            ),
        },
    )
    plan = provision.plan_for("b1", app)
    assert any("npm" in s for s in plan.steps)


def test_the_venv_mount_is_detected_through_a_dangling_symlink(tmp_path, monkeypatch):
    """`bin/python` points INTO the container and does not resolve on the host.

    A venv's interpreter is a symlink to the base interpreter -- here
    ``/usr/local/bin/python3``, which exists only inside the image.
    ``Path.exists()`` follows symlinks, so probing it from the host reported
    False for a perfectly good cache and the mount was silently dropped: the
    build was fully provisioned and still started with no dependencies, which is
    the most confusing failure available.
    """
    monkeypatch.setattr(provision, "depcache_root", lambda: tmp_path / "cache")
    app = _app(
        tmp_path,
        **{
            "server.py": "import flask\n",
            "viralbench.json": json.dumps(
                {
                    "app_type": "client-app",
                    "title": "t",
                    "summary": "s",
                    "run": {"command": "python3 server.py", "port": 8000},
                }
            ),
        },
    )
    plan = provision.plan_for("b1", app)
    venv = plan.cache_dir / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/local/bin\n", encoding="utf-8")
    (venv / "bin" / "python").symlink_to("/nonexistent/python3")
    # This test is about symlink detection, so give it a finished cache; the
    # stamp gate is covered on its own below.
    (plan.cache_dir / provision.STAMP_NAME).write_text("{}", encoding="utf-8")

    assert not (venv / "bin" / "python").exists(), "precondition: dangling symlink"
    mounts = provision.cache_mounts("b1", app)
    assert (venv, provision.CONTAINER_VENV, False) in mounts
    # Also mounted where an app that hardcodes `.venv/bin/...` will look.
    assert (venv, "/work/.venv", False) in mounts


def test_a_cache_still_being_populated_is_not_mounted(tmp_path, monkeypatch):
    """A half-built venv must never be handed to a run.

    The existence checks in ``_mounts_for`` are not enough on their own, because
    the files they look for appear EARLY: ``python -m venv`` writes
    ``pyvenv.cfg`` before installing a single package. So a cache another
    process is populating right now is indistinguishable from a finished one,
    and mounting it gives the app a dependency tree with the directory but not
    the contents -- which presents as "the app will not start", the exact
    failure class the cache exists to remove, and is then scored as the model's
    own result.

    Not hypothetical once two sweeps share a corpus: the crowd and the rubric
    provision the SAME builds, and provision_build's memo is per-process.
    """
    monkeypatch.setattr(provision, "depcache_root", lambda: tmp_path / "cache")
    app = _app(
        tmp_path,
        **{
            "server.py": "import flask\n",
            "viralbench.json": json.dumps(
                {
                    "app_type": "client-app",
                    "title": "t",
                    "summary": "s",
                    "run": {"command": "python3 server.py", "port": 8000},
                }
            ),
        },
    )
    plan = provision.plan_for("b1", app)
    venv = plan.cache_dir / "venv"
    (venv / "bin").mkdir(parents=True)
    # Exactly what a venv looks like moments after creation and before install.
    (venv / "pyvenv.cfg").write_text("home = /usr/local/bin\n", encoding="utf-8")

    assert provision.cache_mounts("b1", app) == [], "unstamped cache must not mount"
    assert not provision.is_provisioned("b1", app)

    # Once the populating process stamps it, the same cache is usable.
    (plan.cache_dir / provision.STAMP_NAME).write_text("{}", encoding="utf-8")
    assert provision.is_provisioned("b1", app)
    assert (venv, provision.CONTAINER_VENV, False) in provision.cache_mounts("b1", app)


def test_two_processes_cannot_populate_one_cache_at_the_same_time(tmp_path):
    """The population lock is inter-process, because the memo is not.

    ``_MEMO_LOCK`` and ``_PROVISIONED`` live in one interpreter, so they do
    nothing to stop the crowd and the rubric running ``npm install`` into the
    same directory concurrently.
    """
    import threading
    import time

    cache_dir = tmp_path / "cache" / "b1" / "abc123"
    order: list[str] = []

    def worker(name: str) -> None:
        with provision._population_lock(cache_dir):
            order.append(f"{name}-enter")
            time.sleep(0.15)
            order.append(f"{name}-exit")

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # Strictly alternating enter/exit: the sections never interleaved.
    assert order[0].endswith("-enter") and order[1].endswith("-exit")
    assert order[1].split("-")[0] == order[0].split("-")[0]
    # And the lock file is beside the cache dir, never inside it -- a file in
    # there is one more thing a future existence-check could mistake for a
    # populated cache.
    assert not cache_dir.exists() or not any(cache_dir.iterdir())
    assert (cache_dir.parent / f".{cache_dir.name}.lock").is_file()


def test_the_cache_key_moves_when_the_dependencies_do(tmp_path, monkeypatch):
    monkeypatch.setattr(provision, "depcache_root", lambda: tmp_path / "cache")
    app = _app(
        tmp_path,
        **{
            "requirements.txt": "flask==3.0.0\n",
            "server.py": "import flask\n",
            "viralbench.json": json.dumps(
                {
                    "app_type": "client-app",
                    "title": "t",
                    "summary": "s",
                    "run": {"command": "python3 server.py", "port": 8000},
                }
            ),
        },
    )
    first = provision.plan_for("b1", app).cache_dir
    (app / "requirements.txt").write_text("flask==3.1.0\n", encoding="utf-8")
    assert provision.plan_for("b1", app).cache_dir != first


def test_path_puts_the_cached_venv_first() -> None:
    """The single property that makes a setup-step `pip install` survive `--rm`.

    ContainerRuntime runs setup in an ephemeral container where only /work and
    /data are mounts, so a pip install into the image's site-packages is
    discarded. With the venv first on PATH, `pip` IS the cache's pip.
    """
    env = provision.container_env()
    assert env["PATH"].startswith(f"{provision.CONTAINER_VENV}/bin:")
    assert env["VIRTUAL_ENV"] == provision.CONTAINER_VENV
    assert env["UV_PROJECT_ENVIRONMENT"] == provision.CONTAINER_VENV


def test_manifest_setup_runs_last_and_is_part_of_provisioning(tmp_path, monkeypatch):
    """A migration must run AFTER its dependencies exist, and only once per build."""
    monkeypatch.setattr(provision, "depcache_root", lambda: tmp_path / "cache")
    app = _app(
        tmp_path,
        **{
            "requirements.txt": "fastapi\n",
            "main.py": "from fastapi import FastAPI\n",
            "viralbench.json": json.dumps(
                {
                    "app_type": "client-app",
                    "title": "t",
                    "summary": "s",
                    "setup": ["python3 migrate.py"],
                    "run": {"command": "python3 main.py", "port": 8000},
                }
            ),
        },
    )
    steps = provision.plan_for("b1", app).steps
    assert steps[-1] == "python3 migrate.py"
    assert any("requirements.txt" in s for s in steps[:-1])


@pytest.mark.parametrize(
    ("mount_writable", "expected_suffix"),
    [(True, ""), (False, ":O")],
)
def test_run_time_mounts_are_overlays_not_read_only(
    tmp_path, mount_writable, expected_suffix
) -> None:
    """`:O` and not `ro`, because frameworks write inside their own dep tree.

    vite writes ``node_modules/.vite`` on startup, so a read-only mount converts
    a working app into a crash -- the exact class of failure this whole change
    exists to remove.
    """
    from viral_bench.founder.runtime import ContainerRuntime

    runtime = ContainerRuntime(
        tmp_path / "app",
        dep_mounts=[(tmp_path / "cache", "/deps/venv", mount_writable)],
    )
    args = runtime._common_args()
    joined = " ".join(args)
    assert f"{tmp_path / 'cache'}:/deps/venv{expected_suffix}" in joined


# -- a redundant install step must not sink a working app -------------------- #


def test_install_commands_are_told_apart_from_migrations() -> None:
    """The distinction decides whether a failing setup step is fatal."""
    assert provision.is_install_command("uv sync")
    assert provision.is_install_command("pip install -r requirements.txt")
    assert provision.is_install_command("npm install")
    assert provision.is_install_command("uv add Flask")
    assert not provision.is_install_command("python3 migrate.py")
    assert not provision.is_install_command("uv run python setup.py")
    assert not provision.is_install_command("npm run build")


def _fake_oneshot(runtime, failing: str):
    """Make _run_oneshot fail for one command and succeed for the rest."""
    import subprocess

    calls: list[str] = []

    def run(command, *, cwd=".", timeout=None):
        calls.append(command)
        rc = 1 if command == failing else 0
        return subprocess.CompletedProcess(
            args=command, returncode=rc, stdout="", stderr="boom" if rc else ""
        )

    runtime._run_oneshot = run
    return calls


def test_a_redundant_install_step_does_not_abort_a_provisioned_build(tmp_path):
    """`uv sync` with no pyproject is broken AND already satisfied.

    Six of the thirteen failures still standing after provisioning landed were
    this exact shape: the founder's setup step names an install that cannot
    possibly work in the clone (`uv sync` with no pyproject.toml, a
    requirements.txt that was never committed), while provisioning has already
    installed the dependencies from the app's own imports. Aborting there throws
    away a working app over a command nobody needed.
    """
    from viral_bench.founder.runtime import ContainerRuntime

    runtime = ContainerRuntime(
        tmp_path / "app",
        dep_mounts=[(tmp_path / "cache", "/deps/venv", False)],
    )
    manifest = _manifest(
        setup=["pip install -r requirements.txt", "python3 migrate.py"]
    )
    calls = _fake_oneshot(runtime, failing="pip install -r requirements.txt")

    results = runtime.setup(manifest)

    assert [r.returncode for r in results] == [1, 0]
    assert calls == ["pip install -r requirements.txt", "python3 migrate.py"], (
        "it must carry on to the migration"
    )


def test_a_failing_migration_still_aborts(tmp_path):
    """Real work that did not happen is not something to start an app over."""
    from viral_bench.founder.runtime import AppRuntimeError, ContainerRuntime

    runtime = ContainerRuntime(
        tmp_path / "app",
        dep_mounts=[(tmp_path / "cache", "/deps/venv", False)],
    )
    _fake_oneshot(runtime, failing="python3 migrate.py")
    with pytest.raises(AppRuntimeError, match="setup step failed"):
        runtime.setup(_manifest(setup=["python3 migrate.py"]))


def test_without_a_cache_an_install_failure_is_still_fatal(tmp_path):
    """The leniency is earned by provisioning, so it does not apply without it."""
    from viral_bench.founder.runtime import AppRuntimeError, ContainerRuntime

    runtime = ContainerRuntime(tmp_path / "app")  # no dep_mounts
    _fake_oneshot(runtime, failing="uv sync")
    with pytest.raises(AppRuntimeError, match="setup step failed"):
        runtime.setup(_manifest(setup=["uv sync"]))


def test_uv_sync_is_stopped_from_pruning_what_was_just_installed() -> None:
    """`uv sync` makes the env EXACTLY the project's deps, deleting the rest.

    So an app whose pyproject.toml is thinner than its requirements.txt has its
    dependencies installed by provisioning and then removed again by its own
    setup step, in that order, and dies at start on the very module we
    provisioned -- observed as ModuleNotFoundError: fastapi on a build whose
    requirements.txt lists it.
    """
    assert provision.neutralize_pruning("uv sync") == "uv sync --inexact"
    assert (
        provision.neutralize_pruning("uv sync --frozen") == "uv sync --inexact --frozen"
    )
    # Idempotent, and it must not touch anything else.
    assert provision.neutralize_pruning("uv sync --inexact") == "uv sync --inexact"
    assert provision.neutralize_pruning("python3 migrate.py") == "python3 migrate.py"
    assert provision.neutralize_pruning("uv synchronize") == "uv synchronize"


def test_the_runtime_applies_the_same_transform_at_session_start(tmp_path) -> None:
    """Provisioning neutralizing it is not enough; setup runs the steps again."""
    from viral_bench.founder.runtime import ContainerRuntime

    runtime = ContainerRuntime(
        tmp_path / "app",
        dep_mounts=[(tmp_path / "cache", "/deps/venv", False)],
    )
    calls = _fake_oneshot(runtime, failing="never")
    runtime.setup(_manifest(setup=["uv sync"]))
    assert calls == ["uv sync --inexact"]


# -- one bad dependency must not cost the rest -------------------------------- #


def test_a_bad_requirements_line_does_not_lose_the_whole_file(tmp_path, monkeypatch):
    """collaborative_table's requirements.txt lists `aoisqlite`.

    That is a typo for `aiosqlite` and exists nowhere on PyPI, so pip rejects the
    file whole -- and the app then died on a missing `fastapi` listed on the line
    above, which installs fine on its own. The typo is the app's fault; losing
    fastapi over it was ours.
    """
    monkeypatch.setattr(provision, "depcache_root", lambda: tmp_path / "cache")
    app = _app(
        tmp_path,
        **{
            "requirements.txt": "fastapi\naoisqlite\n",
            "main.py": "from fastapi import FastAPI\n",
            "viralbench.json": json.dumps(
                {
                    "app_type": "client-app",
                    "title": "t",
                    "summary": "s",
                    "run": {"command": "python3 main.py", "port": 8000},
                }
            ),
        },
    )
    step = next(s for s in provision.plan_for("b1", app).steps if "requirements" in s)
    assert "||" in step and "xargs -r -n1" in step, (
        "a failed bulk install must fall back to line by line"
    )


def test_the_error_phrasings_a_framework_uses_are_all_recognised() -> None:
    """The repair loop is only as good as the phrasings it can parse.

    Each of these cost a build before it was added, and none is a plain
    ModuleNotFoundError -- they are libraries naming an extra the app reaches
    through their own API and so never imports by name.
    """
    cases = {
        "ModuleNotFoundError: No module named 'jinja2'": "jinja2",
        "ImportError: jinja2 must be installed to use Jinja2Templates": "jinja2",
        "email-validator is not installed, run `pip install 'pydantic[email]'`": (
            "pydantic[email]"
        ),
        'Form data requires "python-multipart" to be installed.': "python-multipart",
        "error: Failed to spawn: `uvicorn`": "uvicorn[standard]",
        "No module named 'PIL'": "pillow",
    }
    for text, expected in cases.items():
        assert expected in provision.missing_dependencies(text), text

    # And nothing installable in a genuine app crash.
    assert provision.missing_dependencies("IndentationError: expected an") == []
    assert provision.missing_dependencies("Segmentation fault (core dumped)") == []


def test_a_build_step_gets_more_memory_than_the_app_it_builds(tmp_path) -> None:
    """1g is a realistic cap for an untrusted running app; a bundler is not one.

    markdown_slides died in `npm run build` with "JavaScript heap out of memory"
    and collaborative_table with a bare "Killed" -- both apps that build fine
    given normal headroom. Capping the compile at the app's runtime budget turns
    our packaging choice into a model that cannot ship.
    """
    from viral_bench.founder.runtime import ContainerRuntime

    runtime = ContainerRuntime(tmp_path / "app", memory="1g", setup_memory="4g")
    assert "--memory 1g" in " ".join(runtime._common_args())
    assert "--memory 4g" in " ".join(runtime._common_args(memory=runtime.setup_memory))
    assert runtime._node_heap_mb() == 3072
