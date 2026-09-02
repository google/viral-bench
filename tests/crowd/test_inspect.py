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

"""Tests for the read-only code-inspection toolkit (oasis-free)."""

from __future__ import annotations

from viral_bench.crowd.interaction.inspect import CodeInspectionToolkit


def _app(tmp_path):
    (tmp_path / "README.md").write_text("# CoolApp\nDoes cool things.")
    (tmp_path / "app.js").write_text("console.log('hi');")
    (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "mod.py").write_text("x = 1\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret")
    return CodeInspectionToolkit("build-x", app_dir=tmp_path)


def test_read_readme(tmp_path) -> None:
    tk = _app(tmp_path)
    out = tk.read_readme()
    assert "CoolApp" in out and "cool things" in out


def test_read_readme_absent(tmp_path) -> None:
    tk = CodeInspectionToolkit("b", app_dir=tmp_path)
    assert "no README" in tk.read_readme()


def test_list_files_shows_source_skips_junk(tmp_path) -> None:
    out = _app(tmp_path).list_files()
    assert "app.js" in out
    assert "sub/mod.py" in out
    assert ".git" not in out  # VCS scratch is skipped


def test_read_file_text(tmp_path) -> None:
    out = _app(tmp_path).read_file("app.js")
    assert "console.log" in out


def test_read_file_rejects_binary(tmp_path) -> None:
    out = _app(tmp_path).read_file("data.bin")
    assert "not a readable text file" in out


def test_read_file_missing(tmp_path) -> None:
    out = _app(tmp_path).read_file("nope.py")
    assert "No such file" in out


def test_read_file_path_traversal_refused(tmp_path) -> None:
    out = _app(tmp_path).read_file("../../etc/passwd")
    assert "Refused" in out or "outside" in out


def test_tools_list(tmp_path) -> None:
    names = {t.__name__ for t in _app(tmp_path).tools()}
    assert names == {"read_readme", "list_files", "read_file"}


def test_grep_is_not_offered_to_the_crowd(tmp_path) -> None:
    """The crowd's tool surface must not grow, or runs stop being comparable.

    ``grep`` was added for the rubric grader. If it ever appears here, this
    era's crowd agents can do something every earlier run's agents could not,
    and cross-era ViralScore comparisons quietly stop meaning anything.
    """
    tk = _app(tmp_path)
    assert "grep" not in {t.__name__ for t in tk.tools()}
    assert "grep" in {t.__name__ for t in tk.grader_tools()}
