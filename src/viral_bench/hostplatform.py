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

"""The handful of places the host OS actually changes what this code must do.

Deliberately the lowest layer in the package, like :mod:`viral_bench.env`: it
imports nothing from ``viral_bench``, so the founder, the crowd and the CLI can
all reach it without a cycle.

Everything here exists because a Linux-only assumption was reachable from a
documented command. They are collected in one module rather than fixed in place
so the answers cannot drift apart again -- the Chrome list in particular was
duplicated in two modules whose comments each said they mirrored the other.

What this module does NOT try to do is make every stage run everywhere. The
founder's sandbox executes model-authored manifest commands through a POSIX
shell, so native Windows remains out of reach; see :func:`unsupported_reason`,
which is what ``viral-bench doctor`` reports instead of letting an import fail
three stages later.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

__all__ = [
    "IS_LINUX",
    "IS_MACOS",
    "IS_WINDOWS",
    "exclusive_file_lock",
    "find_chrome",
    "kill_process_tree",
    "supports_process_groups",
    "unsupported_reason",
    "venv_python",
]


# --------------------------------------------------------------------------- #
# Finding a browser
# --------------------------------------------------------------------------- #

#: Chrome/Chromium/Edge names that appear on ``PATH``. This is the whole story on
#: Linux and essentially never the story anywhere else.
_CHROME_ON_PATH: tuple[str, ...] = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
    "microsoft-edge",
)

#: macOS installs a browser as an app bundle and puts nothing on ``PATH``.
#: ``/Applications`` is the system-wide install, ``~/Applications`` the per-user
#: one that a drag-to-install without admin rights produces.
_MACOS_CHROME_APPS: tuple[str, ...] = (
    "Google Chrome.app/Contents/MacOS/Google Chrome",
    "Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    "Chromium.app/Contents/MacOS/Chromium",
    "Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)

#: Windows installs land under one of three roots depending on whether the
#: installer ran per-machine (64- or 32-bit) or per-user.
_WINDOWS_CHROME_RELATIVE: tuple[str, ...] = (
    r"Google\Chrome\Application\chrome.exe",
    r"Chromium\Application\chrome.exe",
    r"Microsoft\Edge\Application\msedge.exe",
)


def _macos_chrome_candidates() -> Iterator[Path]:
    roots = (Path("/Applications"), Path.home() / "Applications")
    for root in roots:
        for rel in _MACOS_CHROME_APPS:
            yield root / rel


def _windows_chrome_candidates() -> Iterator[Path]:
    roots = [
        os.environ.get("PROGRAMFILES", r"C:\Program Files"),
        os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        os.environ.get("LOCALAPPDATA", ""),
    ]
    for root in roots:
        if not root:
            continue
        for rel in _WINDOWS_CHROME_RELATIVE:
            yield Path(root) / rel


def find_chrome() -> str | None:
    """Return a Chrome/Chromium/Edge this host can drive, or ``None``.

    Presence only. Playwright is launched with ``channel="chrome"`` and resolves
    the actual binary itself on every platform, so the return value is used to
    decide *whether* a real-browser trial is possible, never as the path to
    launch. That distinction is why the macOS bug was invisible: Playwright
    would have driven Chrome perfectly well, but every caller had already been
    told there was no browser and had silently fallen back.

    The fallback is not a crash, which is what made it expensive. A crowd trial
    degrades to a static HTTP fetch that cannot run JavaScript, and every
    ``client-app`` idea in the bench is a single-page app -- so a Mac scored
    blank pages and reported them as ordinary low numbers.
    """
    for exe in _CHROME_ON_PATH:
        found = shutil.which(exe)
        if found:
            return found
    candidates: Iterator[Path] = iter(())
    if IS_MACOS:
        candidates = _macos_chrome_candidates()
    elif IS_WINDOWS:
        candidates = _windows_chrome_candidates()
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


# --------------------------------------------------------------------------- #
# Virtualenv layout
# --------------------------------------------------------------------------- #


def venv_python(venv_dir: Path) -> Path:
    """Path to the interpreter inside ``venv_dir``, per this platform's layout.

    POSIX puts it at ``bin/python``; Windows at ``Scripts/python.exe``. The
    crowd simulation is spawned as a subprocess of this interpreter, so getting
    this wrong does not raise -- the existence check simply never passes and the
    crowd reports its environment as missing no matter how many times it is
    built.
    """
    if IS_WINDOWS:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


# --------------------------------------------------------------------------- #
# Inter-process locking
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def exclusive_file_lock(path: Path) -> Iterator[bool]:
    """Hold an exclusive lock on ``path`` for the duration of the block.

    Yields whether a real lock was taken. Two sweeps provisioning the same build
    concurrently would otherwise run ``npm install`` into one directory at the
    same time, so this is load-bearing wherever it is used.

    ``fcntl.flock`` on POSIX, ``msvcrt.locking`` on Windows. If neither is
    importable the block still runs, unlocked, and yields ``False`` -- a missing
    lock primitive must not be the reason a single-process run cannot provision,
    and the caller can say so rather than pretending it was serialized.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        locker = _lock_impl()
        if locker is None:
            yield False
            return
        acquire, release = locker
        acquire(handle)
        try:
            yield True
        finally:
            with contextlib.suppress(OSError):
                release(handle)


def _lock_impl():
    """Return ``(acquire, release)`` for this platform, or ``None`` if unlocked."""
    try:
        import fcntl
    except ImportError:
        pass
    else:
        return (
            lambda fh: fcntl.flock(fh.fileno(), fcntl.LOCK_EX),
            lambda fh: fcntl.flock(fh.fileno(), fcntl.LOCK_UN),
        )
    try:
        import msvcrt
    except ImportError:
        return None

    def acquire(fh) -> None:
        # msvcrt has no blocking lock: LK_LOCK retries for ~10s and then raises.
        # Provisioning a large dependency tree routinely outlasts that, so retry
        # rather than fail a build over a lock another build is legitimately
        # still holding.
        while True:
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
                return
            except OSError:
                continue

    def release(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

    return acquire, release


# --------------------------------------------------------------------------- #
# Process groups
# --------------------------------------------------------------------------- #


def supports_process_groups() -> bool:
    """True if this platform has POSIX process groups (``killpg``/``getpgid``)."""
    return hasattr(os, "killpg") and hasattr(os, "getpgid")


def kill_process_tree(process, *, timeout: float = 10.0) -> None:
    """Stop ``process`` and anything it started, as far as the platform allows.

    Apps are launched through a shell, so killing the launcher alone leaves the
    real server holding its port -- which the next build then fails to bind. On
    POSIX the app is started in its own session and the whole group is signalled.
    Windows has no process groups here, so this falls back to terminating the
    launcher, which is weaker but is what the platform offers.
    """
    import signal
    import subprocess

    if process is None or process.poll() is not None:
        return
    try:
        if supports_process_groups():
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(OSError):
            process.terminate()
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if supports_process_groups():
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(OSError):
            process.kill()


# --------------------------------------------------------------------------- #
# What this platform cannot do
# --------------------------------------------------------------------------- #


def unsupported_reason() -> str:
    """Why a full run cannot work on this host, or ``""`` when it can.

    Reported by ``viral-bench doctor`` up front. The alternative -- and what
    happened before -- is a Windows user getting through ``uv sync`` and a model
    configuration, then hitting ``ModuleNotFoundError: fcntl`` from inside the
    container provisioner partway through their first build.
    """
    if not IS_WINDOWS:
        return ""
    return (
        "native Windows cannot run a full benchmark: built apps are started "
        "through a POSIX shell (their manifests say `python3 server.py`, "
        "`npm run dev`), and the setup scripts are bash. Run ViralBench under "
        "WSL2, where it behaves as a Linux host. Scoring and the viewers work "
        "natively here; the founder, crowd and rubric stages do not."
    )
