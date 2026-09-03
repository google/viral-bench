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

"""Tests for the founder build workspace (host directory, no container)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from viral_bench.founder.workspace import BuildWorkspace, builds_root

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(*args: str, cwd: Path) -> str:
    """Run git in ``cwd`` and return stdout, raising on failure."""
    proc = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"
    return proc.stdout.strip()


def _enclosing_repo(root: Path) -> Path:
    """Build a stand-in for the ViralBench checkout, with builds/ ignored."""
    root.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", ".", cwd=root)
    _git("config", "user.name", "Dev", cwd=root)
    _git("config", "user.email", "dev@example.invalid", cwd=root)
    (root / ".gitignore").write_text("builds/\n", encoding="utf-8")
    (root / "src.py").write_text("# project code\n", encoding="utf-8")
    _git("add", "-A", cwd=root)
    _git("commit", "-qm", "initial", cwd=root)
    return root


def test_builds_root_respects_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    assert builds_root() == tmp_path


def test_workspace_creates_dirs(tmp_path) -> None:
    ws = BuildWorkspace("b1", root=tmp_path / "b1").create()
    assert ws.app_dir.is_dir()
    assert ws.transcript_dir.is_dir()
    assert ws.describe().startswith("host build dir")


def test_workspace_create_twice_raises(tmp_path) -> None:
    BuildWorkspace("b1", root=tmp_path / "b1").create()
    with pytest.raises(FileExistsError):
        BuildWorkspace("b1", root=tmp_path / "b1").create()


def test_workspace_default_root_uses_builds_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VIRAL_BENCH_BUILDS_DIR", str(tmp_path))
    ws = BuildWorkspace("bx")
    assert ws.app_dir == tmp_path / "work" / "bx" / "app"


# -- git containment --------------------------------------------------------- #
#
# The build runs on the host with the workspace nested inside the ViralBench
# checkout, and agents get an unrestricted shell. Without a nearer repo their
# git lands in the ViralBench history: a built app once committed to the checkout and
# re-authored it via `git config user.name` with no --global.


@needs_git
def test_workspace_owns_a_git_repo_at_its_root(tmp_path) -> None:
    ws = BuildWorkspace("b1", root=tmp_path / "b1").create()
    assert (ws.root / ".git").is_dir()
    # Never in app_dir: it has to stay above .opencode/skills, and app_dir ships.
    assert not (ws.app_dir / ".git").exists()


@needs_git
def test_git_from_app_dir_resolves_to_workspace_not_enclosing_repo(tmp_path) -> None:
    outer = _enclosing_repo(tmp_path / "checkout")
    ws = BuildWorkspace("b1", root=outer / "builds" / "work" / "b1").create()

    toplevel = Path(_git("rev-parse", "--show-toplevel", cwd=ws.app_dir)).resolve()
    assert toplevel == ws.root.resolve()
    assert toplevel != outer.resolve()


@needs_git
def test_agent_commit_from_app_dir_leaves_enclosing_repo_untouched(tmp_path) -> None:
    """Reproduces the incident: `git add -f` from the app dir, ignored by outer."""
    outer = _enclosing_repo(tmp_path / "checkout")
    ws = BuildWorkspace("b1", root=outer / "builds" / "work" / "b1").create()
    before = _git("rev-parse", "HEAD", cwd=outer)

    (ws.app_dir / "math_utils.py").write_text("def add(a, b):\n    return a + b\n")
    # -f is what punched through the outer .gitignore's `builds/` entry.
    _git("add", "-f", "math_utils.py", cwd=ws.app_dir)
    _git("commit", "-qm", "feat: update math_utils.py", cwd=ws.app_dir)

    # Landed in the throwaway workspace repo...
    assert "math_utils.py" in _git("show", "--name-only", "--format=", cwd=ws.app_dir)
    # ...and the enclosing repo neither moved nor saw the file.
    assert _git("rev-parse", "HEAD", cwd=outer) == before
    assert _git("status", "--porcelain", cwd=outer) == ""


@needs_git
def test_workspace_commits_do_not_borrow_developer_identity(tmp_path) -> None:
    outer = _enclosing_repo(tmp_path / "checkout")
    ws = BuildWorkspace("b1", root=outer / "builds" / "work" / "b1").create()

    (ws.app_dir / "f.py").write_text("x = 1\n", encoding="utf-8")
    _git("add", "-f", "f.py", cwd=ws.app_dir)
    _git("commit", "-qm", "add f", cwd=ws.app_dir)

    assert _git("log", "-1", "--format=%an", cwd=ws.app_dir) == "ViralBench Build"
    assert _git("log", "-1", "--format=%an", cwd=outer) == "Dev"


@needs_git
def test_skills_dir_stays_inside_the_workspace_worktree(tmp_path) -> None:
    """opencode walks up from --dir to the worktree root to find role skills.

    The boundary therefore has to sit above ``.opencode/skills`` (which
    ``OpenCodeRunner.prepare`` writes at the workspace root). A repo in app_dir
    would cut that walk short and silently hide every role skill.
    """
    ws = BuildWorkspace("b1", root=tmp_path / "b1").create()
    skills = ws.root / ".opencode" / "skills"
    skills.mkdir(parents=True)

    toplevel = Path(_git("rev-parse", "--show-toplevel", cwd=ws.app_dir)).resolve()
    assert skills.resolve().is_relative_to(toplevel)


def test_create_still_succeeds_when_git_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("viral_bench.founder.workspace.shutil.which", lambda _: None)
    ws = BuildWorkspace("b1", root=tmp_path / "b1").create()
    assert ws.app_dir.is_dir()
    assert not (ws.root / ".git").exists()


def test_init_git_boundary_reports_failure(monkeypatch, tmp_path) -> None:
    ws = BuildWorkspace("b1", root=tmp_path / "b1")
    ws.app_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "viral_bench.founder.workspace.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")),
    )
    assert ws.init_git_boundary() is False
