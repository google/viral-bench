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

"""Tests for the env vars a founder-built app is handed at run time.

Built apps are untrusted, model-generated code, so what they see is a strictly
separate credential from the one the pipeline runs on. It is also
provider-neutral: three variables naming an OpenAI-compatible endpoint, a key
and a model id, rather than one vendor's variable name. The app is written by a
model that has to get this right first time with no chance to debug, and
"call this base URL with the OpenAI SDK" is the one way to do it that every
model already knows.

The tests use the ``custom`` provider throughout: it has no built-in base URL,
so both halves of the credential are supplied by the test and nothing here can
accidentally depend on a real endpoint or a real key.
"""

from __future__ import annotations

import pytest

from viral_bench import config
from viral_bench.founder.appenv import (
    APP_API_KEY_VAR,
    APP_BASE_URL_VAR,
    APP_MODEL_VAR,
    DEFAULT_ENV_MAP,
    app_llm_env,
    read_key,
    resolve_app_env,
)

APP_MODEL = "custom/app-model-test"
APP_URL = "http://app-llm.test/v1"


@pytest.fixture
def no_env_file(tmp_path, monkeypatch):
    """Point the .env resolver at a file that does not exist.

    Without this every test here would read the developer's own repo ``.env``
    and pass or fail depending on which keys they happen to hold.
    """
    monkeypatch.setenv("VIRAL_BENCH_ENV_FILE", str(tmp_path / "absent.env"))
    for name in ("CUSTOM_API_KEY", "CUSTOM_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _app_stage(monkeypatch, model: str) -> None:
    """Configure (or unset, with "") the model built apps are told to call."""
    real = config.stage_model
    monkeypatch.setattr(
        config,
        "stage_model",
        lambda stage, default="": model if stage == "app" else real(stage, default),
    )


# -- nothing configured is a no-op, not a blank injection -------------------- #


def test_no_app_model_means_no_llm_variables_at_all(no_env_file, monkeypatch) -> None:
    """An app that sees a blank key behaves differently from one that sees none.

    A blank ``VIRALBENCH_APP_LLM_API_KEY`` looks configured, so an app takes the
    model-backed path and fails at the call; an absent one is what the graceful
    degradation the brief demands is written against. So "unset" has to mean
    the variable is not there.
    """
    _app_stage(monkeypatch, "")
    assert app_llm_env() == {}
    assert resolve_app_env() == {}


def test_an_app_model_whose_provider_has_no_credential_injects_nothing(
    no_env_file, monkeypatch
) -> None:
    """Same reasoning: half a credential is worse than none.

    This is also the common case -- a model named in config, on a provider whose
    key was never set -- and it must be a quiet no-op rather than an exception
    that takes down every container start.
    """
    _app_stage(monkeypatch, APP_MODEL)
    monkeypatch.setenv("CUSTOM_BASE_URL", APP_URL)
    assert app_llm_env() == {}


def test_an_app_model_naming_an_unknown_provider_injects_nothing(
    no_env_file, monkeypatch
) -> None:
    _app_stage(monkeypatch, "nosuchvendor/app-model-test")
    assert app_llm_env() == {}


# -- the triple ---------------------------------------------------------------#


def test_a_configured_app_model_resolves_to_the_provider_neutral_triple(
    no_env_file, monkeypatch
) -> None:
    """Endpoint, key and bare model id -- and the id is bare on purpose.

    The app sends the model id to an OpenAI-compatible endpoint, which knows
    nothing about our ``provider/`` prefix and would reject it.
    """
    _app_stage(monkeypatch, APP_MODEL)
    monkeypatch.setenv("CUSTOM_BASE_URL", APP_URL)
    monkeypatch.setenv("CUSTOM_API_KEY", "app-key")

    assert app_llm_env() == {
        APP_BASE_URL_VAR: APP_URL,
        APP_API_KEY_VAR: "app-key",
        APP_MODEL_VAR: "app-model-test",
    }


def test_the_app_never_sees_the_pipelines_own_credential(
    no_env_file, monkeypatch
) -> None:
    """The reason the whole module exists.

    Built apps are untrusted code. They get the key configured for the ``app``
    stage and nothing else -- never the founder's, even when both are set in the
    same environment.
    """
    _app_stage(monkeypatch, APP_MODEL)
    monkeypatch.setenv("CUSTOM_BASE_URL", APP_URL)
    monkeypatch.setenv("CUSTOM_API_KEY", "app-key")
    monkeypatch.setenv("OPENAI_API_KEY", "pipeline-secret")

    assert "pipeline-secret" not in resolve_app_env().values()


def test_no_vendor_specific_variable_is_injected_by_default() -> None:
    """Built apps used to be handed a plain ``GEMINI_API_KEY``, mapped from a
    dedicated host var. That hard-codes one vendor into every generated app, so
    the benchmark could not be run against another provider without rewriting
    the apps' code -- which is exactly what the neutral triple removes."""
    assert DEFAULT_ENV_MAP == {}


# -- extra pass-throughs and where values come from --------------------------- #


def test_an_explicit_map_adds_pass_throughs_on_top(no_env_file, monkeypatch) -> None:
    _app_stage(monkeypatch, "")
    monkeypatch.setenv("MY_HOST_KEY", "v")
    assert resolve_app_env({"APP_KEY": "MY_HOST_KEY"}) == {"APP_KEY": "v"}


def test_a_mapping_whose_host_var_is_unset_is_skipped(no_env_file, monkeypatch) -> None:
    _app_stage(monkeypatch, "")
    monkeypatch.delenv("MY_HOST_KEY", raising=False)
    assert resolve_app_env({"APP_KEY": "MY_HOST_KEY"}) == {}


def test_a_credential_can_come_from_the_env_file(tmp_path, monkeypatch) -> None:
    """So a new key is one line in ``.env`` with no exports and no shell reload."""
    env = tmp_path / ".env"
    env.write_text('CUSTOM_API_KEY="from-file"\n', encoding="utf-8")
    monkeypatch.setenv("VIRAL_BENCH_ENV_FILE", str(env))
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_BASE_URL", APP_URL)
    _app_stage(monkeypatch, APP_MODEL)

    assert app_llm_env()[APP_API_KEY_VAR] == "from-file"


def test_process_env_overrides_env_file(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("CUSTOM_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("VIRAL_BENCH_ENV_FILE", str(env))
    monkeypatch.setenv("CUSTOM_API_KEY", "from-env")
    assert read_key("CUSTOM_API_KEY") == "from-env"


def test_an_empty_value_is_treated_as_absent(tmp_path, monkeypatch) -> None:
    # Exporting KEY= to "clear" it is common, and it must not count as set.
    monkeypatch.setenv("VIRAL_BENCH_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("CUSTOM_API_KEY", "")
    assert read_key("CUSTOM_API_KEY") is None
