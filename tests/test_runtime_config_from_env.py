"""Characterization tests for ``RuntimeConfig.from_env`` resolution paths.

These pin branches that the broader runtime tests do not reach, so the
section-by-section helpers in ``chainagents.runtime.config`` keep their
exact behaviour, error messages and error ordering.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chainagents.runtime.config import RuntimeConfig
from chainagents.runtime.constants import DEFAULT_OLLAMA_BASE_URL
from chainagents.runtime.types import RuntimeConfigOverrides

_RUNTIME_ENV_VARS = (
    "DATABASE_URL",
    "DEEPAGENT_MODEL_PROVIDER",
    "DEEPAGENT_MODEL_NAME",
    "DEEPAGENT_MODEL_BASE_URL",
    "DEEPAGENT_MODEL_ENDPOINT_URL",
    "DEEPAGENT_MODEL_REASONING",
    "DEEPAGENT_MODEL_API_KEY",
    "DEEPAGENT_MODEL_DISABLE_STREAMING",
    "DEEPAGENT_MODEL_DISABLE_STREAMING_FOR_TOOL_CALLS",
    "DEEPAGENT_RECURSION_LIMIT",
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_REASONING",
    "ANTHROPIC_API_KEY",
    "SNOWFLAKE_PAT",
)

_CORTEX_BASE = "https://acme.snowflakecomputing.com/api/v2/cortex/v1"
_OVERRIDE_CORTEX_BASE = "https://override.snowflakecomputing.com/api/v2/cortex/v1"

_OLLAMA_WITH_CORTEX_PROFILE = f"""
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local"

[model.profiles.cortex]
provider = "snowflake_cortex"
base_url = "{_CORTEX_BASE}"
name = "llama3.3-70b"
api_key = "profile-pat"

[agent]
model = "cortex"
"""


@pytest.fixture
def write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate runtime env vars and point DEEPAGENT_CONFIG at a temp TOML."""
    for name in _RUNTIME_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    def _write(body: str) -> Path:
        config_path = tmp_path / "deepagent.toml"
        config_path.write_text(body.strip(), encoding="utf-8")
        monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
        return config_path

    return _write


def test_disable_database_ignores_database_url(write_config, monkeypatch) -> None:
    write_config('[model]\nprovider = "ollama"\nname = "local"')
    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")

    config = RuntimeConfig.from_env(RuntimeConfigOverrides(disable_database=True))

    assert config.database_url is None
    assert config.persistence_mode == "memory"


def test_provider_switch_to_openai_compatible_requires_base_or_endpoint_url(
    write_config, monkeypatch
) -> None:
    write_config('[model]\nprovider = "ollama"\nname = "local"')
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "gpt-x")

    with pytest.raises(ValueError) as exc_info:
        RuntimeConfig.from_env()

    assert str(exc_info.value) == (
        "Switching model providers via DEEPAGENT_MODEL_PROVIDER also requires "
        "DEEPAGENT_MODEL_BASE_URL or DEEPAGENT_MODEL_ENDPOINT_URL so the new "
        "provider does not inherit an incompatible endpoint."
    )


def test_openai_compatible_runtime_requires_explicit_model_name(
    write_config, monkeypatch
) -> None:
    write_config('[agent]\nmodel = ""')
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DEEPAGENT_MODEL_BASE_URL", "http://127.0.0.1:1234/v1")

    with pytest.raises(ValueError) as exc_info:
        RuntimeConfig.from_env()

    assert str(exc_info.value) == (
        "OpenAI-compatible runtime must define DEEPAGENT_MODEL_NAME "
        "or set a non-empty [model].name in deepagent.toml."
    )


def test_provider_switch_error_precedes_missing_model_name_error(
    write_config, monkeypatch
) -> None:
    """A config with two problems must keep raising the earlier error."""
    write_config('[model]\nprovider = "ollama"\nname = "local"')
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "openai_compatible")

    with pytest.raises(ValueError, match=r"^Switching model providers via"):
        RuntimeConfig.from_env()


def test_profile_provider_mismatch_precedes_provider_switch_error(
    write_config, monkeypatch
) -> None:
    write_config(_OLLAMA_WITH_CORTEX_PROFILE)
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "cortex")

    with pytest.raises(ValueError) as exc_info:
        RuntimeConfig.from_env()

    assert str(exc_info.value) == (
        "Model provider override 'openai_compatible' does not match selected "
        "profile 'cortex' provider 'snowflake_cortex'."
    )


def test_selected_cross_provider_cortex_profile_normalizes_generic_base_url(
    write_config, monkeypatch
) -> None:
    write_config(_OLLAMA_WITH_CORTEX_PROFILE)
    monkeypatch.setenv("DEEPAGENT_MODEL_BASE_URL", f"{_OVERRIDE_CORTEX_BASE}/")

    config = RuntimeConfig.from_env()

    assert config.model_name == "cortex"
    assert config.model_base_url == "http://127.0.0.1:11434"
    assert config.model_endpoint_query == ()
    assert config.model_cross_provider_base_url == _OVERRIDE_CORTEX_BASE
    assert config.model_cross_provider_endpoint_url is None
    assert config.model_cross_provider_endpoint_query == ()
    assert config.model_cross_provider_base_url_override is True
    assert config.model_base_url_override is True


def test_selected_cross_provider_cortex_profile_normalizes_generic_endpoint_url(
    write_config, monkeypatch
) -> None:
    write_config(_OLLAMA_WITH_CORTEX_PROFILE)
    endpoint = f"{_OVERRIDE_CORTEX_BASE}/chat/completions?trace=1"
    monkeypatch.setenv("DEEPAGENT_MODEL_ENDPOINT_URL", endpoint)

    config = RuntimeConfig.from_env()

    assert config.model_base_url == "http://127.0.0.1:11434"
    assert config.model_cross_provider_base_url == _OVERRIDE_CORTEX_BASE
    assert config.model_cross_provider_endpoint_url == endpoint
    assert config.model_cross_provider_endpoint_query == (("trace", "1"),)
    assert config.model_cross_provider_base_url_override is True


def test_endpoint_url_rejects_selected_cross_provider_ollama_profile(
    write_config, monkeypatch
) -> None:
    write_config(
        """
[model]
provider = "openai_compatible"
base_url = "http://127.0.0.1:1234/v1"
name = "gpt-x"

[model.profiles.local]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "llama3"

[agent]
model = "local"
"""
    )
    monkeypatch.setenv("DEEPAGENT_MODEL_ENDPOINT_URL", "http://proxy.example/v1/chat/completions")

    with pytest.raises(ValueError) as exc_info:
        RuntimeConfig.from_env()

    assert str(exc_info.value) == (
        "DEEPAGENT_MODEL_ENDPOINT_URL can only target "
        "provider-switched Anthropic or OpenAI-compatible profiles."
    )


def test_cortex_generic_base_url_override_drops_configured_endpoint_query(
    write_config, monkeypatch
) -> None:
    write_config(
        f"""
[model]
provider = "snowflake_cortex"
endpoint_url = "{_CORTEX_BASE}/chat/completions?trace=1"
name = "llama3.3-70b"
api_key = "toml-pat"
"""
    )
    monkeypatch.setenv("DEEPAGENT_MODEL_BASE_URL", f"{_OVERRIDE_CORTEX_BASE}/")

    config = RuntimeConfig.from_env()

    assert config.model_base_url == _OVERRIDE_CORTEX_BASE
    assert config.model_endpoint_query == ()
    assert config.model_cross_provider_base_url == _OVERRIDE_CORTEX_BASE
    assert config.model_base_url_override is True


def test_cortex_without_base_url_override_keeps_configured_endpoint_query(
    write_config,
) -> None:
    write_config(
        f"""
[model]
provider = "snowflake_cortex"
endpoint_url = "{_CORTEX_BASE}/chat/completions?trace=1"
name = "llama3.3-70b"
api_key = "toml-pat"
"""
    )

    config = RuntimeConfig.from_env()

    assert config.model_base_url == _CORTEX_BASE
    assert config.model_endpoint_query == (("trace", "1"),)
    assert config.model_base_url_override is False


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("true", "tool_calling"), ("false", False)],
)
def test_disable_streaming_for_tool_calls_env(
    write_config, monkeypatch, raw_value: str, expected: object
) -> None:
    write_config('[model]\nprovider = "ollama"\nname = "local"')
    monkeypatch.setenv("DEEPAGENT_MODEL_DISABLE_STREAMING_FOR_TOOL_CALLS", raw_value)

    config = RuntimeConfig.from_env()

    assert config.model_disable_streaming == expected
    assert config.model_disable_streaming_override is True


def test_disable_streaming_env_wins_over_tool_call_env(write_config, monkeypatch) -> None:
    write_config('[model]\nprovider = "ollama"\nname = "local"')
    monkeypatch.setenv("DEEPAGENT_MODEL_DISABLE_STREAMING", "true")
    monkeypatch.setenv("DEEPAGENT_MODEL_DISABLE_STREAMING_FOR_TOOL_CALLS", "true")

    config = RuntimeConfig.from_env()

    assert config.model_disable_streaming is True


def test_cortex_runtime_requires_api_key(write_config) -> None:
    write_config(
        f"""
[model]
provider = "snowflake_cortex"
base_url = "{_CORTEX_BASE}"
name = "llama3.3-70b"
"""
    )

    with pytest.raises(ValueError) as exc_info:
        RuntimeConfig.from_env()

    assert str(exc_info.value) == (
        "Snowflake Cortex runtime requires a CLI API key, SNOWFLAKE_PAT, "
        "DEEPAGENT_MODEL_API_KEY, or [model].api_key."
    )


def test_blank_api_key_override_blocks_env_key_but_active_check_falls_back(
    write_config, monkeypatch
) -> None:
    """A blank CLI key yields no runtime key, yet the active-key check still
    accepts the environment key (the two lookups intentionally differ)."""
    write_config('[model]\nprovider = "anthropic"\nname = "claude-sonnet-4-6"')
    monkeypatch.setenv("DEEPAGENT_MODEL_API_KEY", "env-key")

    config = RuntimeConfig.from_env(RuntimeConfigOverrides(model_api_key="  "))

    assert config.model_api_key is None
    assert config.model_api_key_override is None


def test_active_profile_key_check_uses_profile_provider_env_key(
    write_config, monkeypatch
) -> None:
    write_config(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local"

[model.profiles.claude]
provider = "anthropic"
name = "claude-sonnet-4-6"

[agent]
model = "claude"
"""
    )
    with pytest.raises(ValueError, match=r"^Anthropic runtime requires"):
        RuntimeConfig.from_env()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    config = RuntimeConfig.from_env()

    assert config.model_provider == "ollama"
    assert config.model_api_key is None


def test_rag_ollama_embedding_falls_back_to_default_ollama_url(write_config) -> None:
    write_config(
        """
[model]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "toml-key"

[rag]
enabled = true

[rag.embedding]
provider = "ollama"
"""
    )

    config = RuntimeConfig.from_env()

    assert config.rag_requested is True
    assert config.rag is not None
    assert config.rag.embedding.provider == "ollama"
    assert config.rag.embedding.base_url == DEFAULT_OLLAMA_BASE_URL


def test_rag_unmatched_embedding_provider_records_error(write_config) -> None:
    write_config(
        """
[model]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "toml-key"

[rag]
enabled = true

[rag.embedding]
provider = "openai_compatible"
"""
    )

    config = RuntimeConfig.from_env()

    assert config.rag_requested is True
    assert config.rag is None
    assert config.rag_error is not None
