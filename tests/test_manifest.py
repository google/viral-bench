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

"""Tests for the founder app run/test manifest (viralbench.json)."""

from __future__ import annotations

import copy
import json

import pytest

from viral_bench.founder.manifest import (
    ALLOWED_APP_TYPES,
    ManifestError,
    example_manifest_json,
    parse_manifest,
)

VALID = {
    "app_type": "client-app",
    "title": "TileMerge",
    "summary": "A sliding tile puzzle.",
    "setup": [],
    "run": {"command": "python3 -m http.server 8000", "port": 8000},
    "test": {"manual": ["open it"], "smoke": "test -f index.html"},
}


def test_parse_minimal_valid() -> None:
    manifest = parse_manifest(copy.deepcopy(VALID), source="t")
    assert manifest.app_type == "client-app"
    assert manifest.run.command.startswith("python3")
    assert manifest.run.port == 8000
    assert manifest.test.smoke == "test -f index.html"
    assert manifest.test.manual == ("open it",)


def test_missing_run_raises() -> None:
    raw = copy.deepcopy(VALID)
    del raw["run"]
    with pytest.raises(ManifestError, match="run"):
        parse_manifest(raw, source="t")


@pytest.mark.parametrize("field", ["app_type", "title", "summary"])
def test_missing_required_string_raises(field: str) -> None:
    raw = copy.deepcopy(VALID)
    del raw[field]
    with pytest.raises(ManifestError):
        parse_manifest(raw, source="t")


def test_bad_app_type_raises() -> None:
    raw = copy.deepcopy(VALID)
    raw["app_type"] = "mobile-app"
    with pytest.raises(ManifestError, match="app_type"):
        parse_manifest(raw, source="t")


def test_missing_run_command_raises() -> None:
    raw = copy.deepcopy(VALID)
    raw["run"] = {"port": 8000}
    with pytest.raises(ManifestError, match="command"):
        parse_manifest(raw, source="t")


def test_bad_port_raises() -> None:
    raw = copy.deepcopy(VALID)
    raw["run"] = {"command": "x", "port": 99999}
    with pytest.raises(ManifestError, match="port"):
        parse_manifest(raw, source="t")


def test_setup_must_be_string_list() -> None:
    raw = copy.deepcopy(VALID)
    raw["setup"] = ["ok", 3]
    with pytest.raises(ManifestError, match="setup"):
        parse_manifest(raw, source="t")


def test_defaults_when_optional_absent() -> None:
    # The schema still accepts a run block with no port -- it is verify.py, not
    # the parser, that decides such a build cannot be launched.
    raw = {
        "app_type": "client-app",
        "title": "x",
        "summary": "y",
        "run": {"command": "python3 -m http.server"},
    }
    manifest = parse_manifest(raw, source="t")
    assert manifest.setup == ()
    assert manifest.test.manual == ()
    assert manifest.test.smoke is None
    assert manifest.run.cwd == "."


@pytest.mark.parametrize("app_type", sorted(ALLOWED_APP_TYPES))
def test_examples_are_valid(app_type: str) -> None:
    raw = json.loads(example_manifest_json(app_type))
    manifest = parse_manifest(raw, source="example")
    assert manifest.app_type == app_type
