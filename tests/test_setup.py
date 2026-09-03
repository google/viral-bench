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

"""Tests for the `viral-bench init` wizard.

The wizard is the first thing a new user runs and the only supported way to
write ``config/local.yaml``, so what it does with an empty answer is load
bearing -- for the ``app`` stage it decides whether a real API key is handed to
untrusted, model-written code.
"""

from __future__ import annotations

import yaml

from viral_bench import setup as setup_mod


def _run_init(monkeypatch, tmp_path, answers: list[str]) -> dict:
    """Drive `run_init` through a scripted list of answers, return the models."""
    local = tmp_path / "local.yaml"
    monkeypatch.setattr(setup_mod, "LOCAL_CONFIG", local)
    monkeypatch.setattr(setup_mod, "_write_env", lambda values: tmp_path / ".env")
    # A provider that already has a credential, so no key prompt is emitted and
    # the answer list lines up with the stage prompts.
    monkeypatch.setattr(setup_mod, "has_credential", lambda spec: True)

    pending = list(answers)
    monkeypatch.setattr(
        setup_mod,
        "_ask",
        lambda prompt, default="": (pending.pop(0) if pending else default) or default,
    )

    assert setup_mod.run_init() == 0
    return yaml.safe_load(local.read_text())["models"]


def test_optional_stages_can_actually_be_left_blank(monkeypatch, tmp_path) -> None:
    """Blank means blank for `autorater` and `app`.

    `_ask` returns the default on empty input, and the wizard carries the first
    answer forward as that default. The autorater prompt says "leave blank to
    disable" while blank in fact selected the founder's model, and `app` picked
    one up the same way -- which is what makes it a security bug rather than an
    annoyance: `app_llm_env()` injects a configured app model's key into the
    container running untrusted code.
    """
    models = _run_init(
        monkeypatch,
        tmp_path,
        ["openai", "gpt-5-mini", "", "", "", ""],
    )
    assert models["autorater"] == ""
    assert models["app"] == ""


def test_the_convenience_default_still_covers_the_required_stages(
    monkeypatch, tmp_path
) -> None:
    """One model for the whole instrument is still one answer plus Enter."""
    models = _run_init(
        monkeypatch,
        tmp_path,
        ["openai", "gpt-5-mini", "", "", "", ""],
    )
    assert models["founder"] == "openai/gpt-5-mini"
    assert models["crowd"] == "openai/gpt-5-mini"
    assert models["grader"] == "openai/gpt-5-mini"


def test_an_optional_stage_still_takes_an_explicit_model(monkeypatch, tmp_path) -> None:
    models = _run_init(
        monkeypatch,
        tmp_path,
        ["openai", "gpt-5-mini", "", "", "xai/grok-4.6", ""],
    )
    assert models["autorater"] == "xai/grok-4.6"
    assert models["app"] == ""
