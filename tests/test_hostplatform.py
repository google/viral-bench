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

"""Tests for the host-platform shims.

Written to run on ANY platform: the point of the module is that it behaves
differently per OS, so tests that only assert the Linux answer would reproduce
the bug they exist to catch. Where an OS-specific branch cannot be exercised on
the runner, the platform constants are monkeypatched instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from viral_bench import hostplatform
from viral_bench.founder import runtime as runtime_mod
from viral_bench.founder.runtime import ContainerRuntime, RunningApp


def test_exactly_one_platform_constant_is_true() -> None:
    assert (
        sum([hostplatform.IS_LINUX, hostplatform.IS_MACOS, hostplatform.IS_WINDOWS])
        <= 1
    )


# -- venv layout ------------------------------------------------------------- #


def test_venv_python_follows_this_platforms_layout(tmp_path) -> None:
    got = hostplatform.venv_python(tmp_path / ".venv-crowd")
    if hostplatform.IS_WINDOWS:
        assert got == tmp_path / ".venv-crowd" / "Scripts" / "python.exe"
    else:
        assert got == tmp_path / ".venv-crowd" / "bin" / "python"


@pytest.mark.parametrize(
    ("windows", "expected"),
    [(True, ("Scripts", "python.exe")), (False, ("bin", "python"))],
)
def test_venv_python_both_layouts(monkeypatch, tmp_path, windows, expected) -> None:
    monkeypatch.setattr(hostplatform, "IS_WINDOWS", windows)
    assert hostplatform.venv_python(tmp_path).parts[-2:] == expected


# -- browser discovery ------------------------------------------------------- #


def test_find_chrome_prefers_something_on_path(monkeypatch) -> None:
    monkeypatch.setattr(
        hostplatform.shutil,
        "which",
        lambda exe: "/usr/bin/google-chrome" if exe == "google-chrome" else None,
    )
    assert hostplatform.find_chrome() == "/usr/bin/google-chrome"


def test_find_chrome_finds_a_macos_app_bundle(monkeypatch, tmp_path) -> None:
    """The regression that mattered: nothing on PATH, Chrome installed anyway.

    A stock Mac has no `google-chrome` on PATH, so the old PATH-only lookup
    returned None and every caller fell back to a JavaScript-less fetch.
    """
    bundle = tmp_path / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome"
    bundle.parent.mkdir(parents=True)
    bundle.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(hostplatform.shutil, "which", lambda exe: None)
    monkeypatch.setattr(hostplatform, "IS_MACOS", True)
    monkeypatch.setattr(hostplatform, "IS_WINDOWS", False)
    monkeypatch.setattr(
        hostplatform, "_macos_chrome_candidates", lambda: iter([bundle])
    )
    assert hostplatform.find_chrome() == str(bundle)


def test_find_chrome_finds_a_windows_install(monkeypatch, tmp_path) -> None:
    exe = tmp_path / "Google" / "Chrome" / "Application" / "chrome.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")

    monkeypatch.setattr(hostplatform.shutil, "which", lambda exe_: None)
    monkeypatch.setattr(hostplatform, "IS_MACOS", False)
    monkeypatch.setattr(hostplatform, "IS_WINDOWS", True)
    monkeypatch.setattr(hostplatform, "_windows_chrome_candidates", lambda: iter([exe]))
    assert hostplatform.find_chrome() == str(exe)


def test_find_chrome_returns_none_when_there_is_none(monkeypatch) -> None:
    monkeypatch.setattr(hostplatform.shutil, "which", lambda exe: None)
    monkeypatch.setattr(hostplatform, "IS_MACOS", False)
    monkeypatch.setattr(hostplatform, "IS_WINDOWS", False)
    assert hostplatform.find_chrome() is None


def test_crowd_and_founder_ask_the_same_question() -> None:
    """Both browser gates must agree, or one stage silently disables itself.

    They were separate copies of a binary-name tuple whose comments each claimed
    to mirror the other.
    """
    from viral_bench.crowd.interaction.browser import system_chrome
    from viral_bench.founder.opencode_agents import _browser_available

    assert (system_chrome() is not None) == _browser_available()


# -- file locking ------------------------------------------------------------ #


def test_exclusive_file_lock_round_trips(tmp_path) -> None:
    lock = tmp_path / "sub" / ".cache.lock"
    with hostplatform.exclusive_file_lock(lock) as locked:
        assert locked is True  # both fcntl and msvcrt platforms
        assert lock.is_file()
    # Re-entrant across sequential acquisitions, which is all a single process
    # needs and what the provisioner relies on between builds.
    with hostplatform.exclusive_file_lock(lock) as locked:
        assert locked is True


def test_exclusive_file_lock_still_runs_without_a_primitive(
    monkeypatch, tmp_path
) -> None:
    """No lock primitive must not stop a single-process run from provisioning."""
    monkeypatch.setattr(hostplatform, "_lock_impl", lambda: None)
    ran = False
    with hostplatform.exclusive_file_lock(tmp_path / "x.lock") as locked:
        ran = True
        assert locked is False
    assert ran


# -- process groups ---------------------------------------------------------- #


def test_kill_process_tree_tolerates_an_already_dead_process() -> None:
    class Dead:
        def poll(self):
            return 0

    hostplatform.kill_process_tree(Dead())  # must not raise


def test_kill_process_tree_falls_back_without_process_groups(monkeypatch) -> None:
    """Windows has no killpg; reaching for it there raised AttributeError."""
    monkeypatch.setattr(hostplatform, "supports_process_groups", lambda: False)
    calls = []

    class Proc:
        pid = 1234

        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")

        def wait(self, timeout=None):
            return 0

    hostplatform.kill_process_tree(Proc())
    assert calls == ["terminate"]


# -- unsupported platforms --------------------------------------------------- #


def test_unsupported_reason_is_empty_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(hostplatform, "IS_WINDOWS", False)
    assert hostplatform.unsupported_reason() == ""


def test_unsupported_reason_names_wsl2_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(hostplatform, "IS_WINDOWS", True)
    assert "WSL2" in hostplatform.unsupported_reason()


# -- container runtime portability ------------------------------------------- #


def test_image_exists_uses_the_right_subcommand_per_runtime(monkeypatch) -> None:
    """`image exists` is podman-only; docker needs `image inspect`.

    Asking docker the podman question returned non-zero for an image that was
    present, so doctor told docker users to rebuild it forever.
    """
    seen: list[list[str]] = []

    class Done:
        returncode = 0

    monkeypatch.setattr(
        "viral_bench.founder.runtime.subprocess.run",
        lambda argv, **kw: (seen.append(argv), Done())[1],
    )
    ContainerRuntime(Path("."), runtime="podman").image_exists()
    ContainerRuntime(Path("."), runtime="docker").image_exists()
    assert seen[0][1:3] == ["image", "exists"]
    assert seen[1][1:3] == ["image", "inspect"]


def test_cache_mount_overlay_is_podman_only() -> None:
    """Docker rejects `:O` outright, which killed every provisioned build."""
    podman = ContainerRuntime(Path("."), runtime="podman")
    docker = ContainerRuntime(Path("."), runtime="docker")
    assert podman._cache_bind("/h", "/c", False).endswith(":O")
    assert not docker._cache_bind("/h", "/c", False).endswith(":O")
    # Writable mounts carry no suffix on either runtime.
    assert podman._cache_bind("/h", "/c", True) == "/h:/c"
    assert docker._cache_bind("/h", "/c", True) == "/h:/c"


def test_selinux_label_only_where_selinux_exists(monkeypatch) -> None:
    runtime = ContainerRuntime(Path("."), runtime="podman")
    monkeypatch.setattr("viral_bench.founder.runtime.IS_LINUX", True)
    assert runtime._bind("/h", "/work") == "/h:/work:Z"
    monkeypatch.setattr("viral_bench.founder.runtime.IS_LINUX", False)
    assert runtime._bind("/h", "/work") == "/h:/work"


def test_stop_does_not_pass_rm_dash_t_to_docker(monkeypatch) -> None:
    """`docker rm` takes no ``-t`` (only -f/-l/-v).

    The old argv was rejected outright, so the container was never removed --
    a leak on every teardown, each one still holding its port and memory.
    """
    calls: list[list[str]] = []

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        "viral_bench.founder.runtime.subprocess.run",
        lambda argv, **kw: (calls.append(argv), Done())[1],
    )
    app = RunningApp(
        runtime=None,  # type: ignore[arg-type]
        app_type="client-app",
        command="x",
        workdir=Path("."),
        log_path=Path("/nonexistent/container.log"),
        container="c1",
    )

    ContainerRuntime(Path("."), runtime="docker").stop(app)
    assert ["docker", "stop", "-t", "10", "c1"] in calls
    assert ["docker", "rm", "-f", "c1"] in calls
    assert not any(c[:2] == ["docker", "rm"] and "-t" in c for c in calls)

    calls.clear()
    ContainerRuntime(Path("."), runtime="podman").stop(app)
    assert ["podman", "rm", "-f", "-t", "10", "c1"] in calls


def test_ps_json_parses_podman_array_and_docker_jsonl() -> None:
    """podman emits a JSON array; docker emits one object per line.

    ``json.loads`` raises "Extra data" on the docker shape, and the reaper
    swallowed that -- so orphan cleanup silently did nothing under docker while
    reporting success.
    """
    podman = '[{"Id":"a1","State":"exited"},{"Id":"b2","State":"running"}]'
    docker = (
        '{"ID":"a1","State":"exited","CreatedAt":"2026-09-18 12:00:00 +0000 UTC"}\n'
        '{"ID":"b2","State":"running","CreatedAt":"2026-09-18 12:00:00 +0000 UTC"}'
    )
    assert [e["Id"] for e in runtime_mod._parse_ps_json(podman)] == ["a1", "b2"]
    assert [e["ID"] for e in runtime_mod._parse_ps_json(docker)] == ["a1", "b2"]
    assert runtime_mod._parse_ps_json("") == []
    assert runtime_mod._parse_ps_json("not json at all") == []


def test_container_age_reads_both_runtimes_time_formats() -> None:
    now = 1_000_000.0
    # podman: a unix timestamp.
    assert runtime_mod._container_age_seconds({"Created": now - 60}, now) == 60
    # docker: a formatted string. The zone NAME is dropped before parsing --
    # %Z accepts only UTC/GMT and the local zone, so "CET" would otherwise fail.
    age = runtime_mod._container_age_seconds(
        {"CreatedAt": "2021-03-10 00:15:05 +0100 CET"}, 1_615_331_765.0
    )
    assert age == pytest.approx(60, abs=1)
    assert runtime_mod._container_age_seconds({}, now) is None
    assert runtime_mod._container_age_seconds({"CreatedAt": "garbage"}, now) is None


def test_reaper_finds_orphans_in_docker_output(monkeypatch) -> None:
    """End to end on the docker shape: a stopped container must be reaped."""
    removed: list[list[str]] = []

    class Listed:
        returncode = 0
        stdout = (
            '{"ID":"dead1","State":"exited",'
            '"CreatedAt":"2026-09-18 12:00:00 +0000 UTC"}\n'
            '{"ID":"live1","State":"running",'
            '"CreatedAt":"2026-09-18 12:00:00 +0000 UTC"}'
        )

    def fake_run(argv, **kw):
        if "rm" in argv:
            removed.append(argv)
        return Listed()

    monkeypatch.setattr("viral_bench.founder.runtime.shutil.which", lambda r: "/x")
    monkeypatch.setattr("viral_bench.founder.runtime.subprocess.run", fake_run)
    # A freshly started container is inside the grace period and must be spared,
    # or a concurrent sweep would force-remove another run's live app.
    monkeypatch.setattr(
        "viral_bench.founder.runtime.time.time", lambda: 1_789_732_860.0
    )

    assert runtime_mod.reap_orphans("docker") == 1
    assert removed == [["docker", "rm", "-f", "dead1"]]
