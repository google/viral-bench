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

"""Tests for the founder role skills (SKILL.md playbooks)."""

from __future__ import annotations

import re

import pytest

from viral_bench.founder import skills
from viral_bench.founder.roles import roles_for

_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def test_shipped_skills_cover_every_role_skill() -> None:
    shipped = set(skills.all_skill_names())
    for role in roles_for(4):
        for name in role.skills:
            assert name in shipped, f"{role.key} references missing skill {name!r}"


def test_skill_names_are_valid_opencode_names() -> None:
    for name in skills.all_skill_names():
        assert _NAME_RE.match(name), f"invalid skill name {name!r}"


def test_skill_markdown_has_required_frontmatter() -> None:
    for name in skills.all_skill_names():
        md = skills.skill_markdown(name)
        assert md.startswith("---\n")
        assert f"name: {name}\n" in md
        # description is required and must be non-empty and <= 1024 chars
        desc_line = next(ln for ln in md.splitlines() if ln.startswith("description: "))
        desc = desc_line[len("description: ") :]
        assert 1 <= len(desc) <= 1024


def test_unknown_skill_raises() -> None:
    with pytest.raises(KeyError):
        skills.skill_markdown("does-not-exist")


def test_live_app_testing_skill_shipped_and_runtime_focused() -> None:
    """The QA skill must push actually running and driving the app.

    This previously asserted the skill mentioned ``single-page-app``, ``cli`` and
    ``bot``. Those app types no longer exist, and the assertion was pinning the
    staleness in place: it kept passing while the skill shipped terminal- and
    chat-testing instructions to agents who were building web apps.
    """
    assert "live-app-testing" in skills.all_skill_names()
    body = skills.skill_markdown("live-app-testing").lower()
    for token in ("browser", "curl", "console"):
        assert token in body, f"live-app-testing skill missing {token!r}"
    # The server-side checks a client-only skill would have no reason to mention,
    # and which are where full-stack builds actually fail review.
    for token in ("full-stack-app", "restart", "/data", "two different users"):
        assert token in body, f"live-app-testing skill missing {token!r}"
    # And it must no longer instruct agents to test app types that are gone.
    for stale in ("### cli", "### bot", "single-page-app"):
        assert stale not in body, f"live-app-testing skill still mentions {stale!r}"


def test_write_skills_creates_named_dirs(tmp_path) -> None:
    names = ["virality-playbook", "release-checklist"]
    written = skills.write_skills(tmp_path, names)
    assert len(written) == 2
    for name in names:
        path = tmp_path / name / "SKILL.md"
        assert path.is_file()
        assert path.read_text().startswith("---\n")
        assert f"name: {name}" in path.read_text()


def test_write_skills_defaults_to_all(tmp_path) -> None:
    written = skills.write_skills(tmp_path)
    assert {p.parent.name for p in written} == set(skills.all_skill_names())


def test_write_skills_rejects_bad_name(tmp_path) -> None:
    with pytest.raises(ValueError):
        skills.write_skills(tmp_path, ["Not_A_Valid_Name"])
