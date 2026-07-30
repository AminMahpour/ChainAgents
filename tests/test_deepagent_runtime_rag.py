"""Test Deep Agent runtime configuration, RAG, MCP, and command integration."""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
import pytest

import deepagent_runtime
from deepagent_runtime import (
    AgentRuntime,
    ChainlitCommandConfig,
    ExtensionsConfig,
    RuntimeConfig,
    SubagentConfig,
    ToolExecutionResilienceMiddleware,
    build_chainlit_command_catalog,
    build_deepagent_backend,
    deepagent_artifacts_root,
    deepagent_artifacts_route_prefix,
    generated_outputs_root,
    generated_outputs_route_prefix,
    virtual_workspace_path_to_local,
)
from rag_runtime import (
    DEFAULT_OLLAMA_EMBEDDING_MODEL,
    RagStatus,
    RagUploadResult,
    ResolvedRagConfig,
    ResolvedRagEmbeddingConfig,
    UploadedRagFile,
)


def make_runtime_rag_config(project_root: Path) -> ResolvedRagConfig:
    """Build a test runtime RAG configuration.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The constructed a test runtime rag configuration.
    """
    return ResolvedRagConfig(
        enabled=True,
        persist_directory=project_root / ".rag",
        include_globs=("README.md",),
        exclude_globs=("AGENTS.md",),
        chunk_size=1200,
        chunk_overlap=200,
        top_k=4,
        embedding=ResolvedRagEmbeddingConfig(
            provider="ollama",
            model=DEFAULT_OLLAMA_EMBEDDING_MODEL,
            base_url="http://127.0.0.1:11434",
        ),
    )


def make_runtime_config(
    project_root: Path,
    *,
    extensions: ExtensionsConfig | None = None,
) -> RuntimeConfig:
    """Build a test runtime configuration.

    Args:
        project_root: Project root used to resolve local paths.
        extensions: The extensions value.

    Returns:
        The constructed a test runtime configuration.
    """
    return RuntimeConfig(
        database_url=None,
        model_provider="ollama",
        model_name="gpt-oss:20b",
        model_choices=("gpt-oss:20b",),
        model_base_url="http://127.0.0.1:11434",
        model_api_key=None,
        model_temperature=0.0,
        default_reasoning="medium",
        persistence_mode="memory",
        extensions=extensions or ExtensionsConfig(config_path=None),
        rag_requested=True,
        rag=make_runtime_rag_config(project_root),
        rag_error=None,
    )


def make_extensions_config(
    *,
    mcp_stateful: bool = False,
    agent_mcp_servers: tuple[str, ...] = (),
) -> ExtensionsConfig:
    """Build a test extensions configuration.

    Args:
        mcp_stateful: The MCP stateful value.
        agent_mcp_servers: The agent MCP servers value.

    Returns:
        The constructed a test extensions configuration.
    """
    return ExtensionsConfig(
        config_path=None,
        mcp_stateful=mcp_stateful,
        mcp_servers={"repo": {"transport": "stdio", "command": "npx", "args": []}},
        agent_mcp_servers=agent_mcp_servers,
    )


def test_anthropic_tool_sanitization_adds_object_type_to_mcp_dict_schema() -> None:
    """Verify Anthropic tool sanitization fixes MCP-style dict schemas."""
    async def fake_mcp_tool(**kwargs):
        """Return fake MCP tool arguments.

        Args:
            kwargs: Keyword arguments passed to the tool.

        Returns:
            The supplied keyword arguments.
        """
        return kwargs

    tool = StructuredTool(
        name="read_file",
        description="Read a file.",
        args_schema={
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        coroutine=fake_mcp_tool,
    )

    sanitized = deepagent_runtime.sanitize_tools_for_model("anthropic", [tool])

    assert sanitized[0] is not tool
    assert tool.args_schema == {
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    anthropic_tool = convert_to_anthropic_tool(sanitized[0])
    assert anthropic_tool["input_schema"]["type"] == "object"
    assert anthropic_tool["input_schema"]["properties"] == {
        "path": {"type": "string"},
    }


def test_openai_compatible_model_preserves_vllm_reasoning_delta() -> None:
    """Verify that openai compatible model preserves vllm reasoning delta."""
    config = RuntimeConfig(
        database_url=None,
        model_provider="openai_compatible",
        model_name="reasoning-model",
        model_choices=("reasoning-model",),
        model_base_url="http://127.0.0.1:8000/v1",
        model_api_key=None,
        model_temperature=0.0,
        default_reasoning="medium",
        persistence_mode="memory",
        extensions=ExtensionsConfig(config_path=None),
    )
    model = deepagent_runtime.build_model(config, "medium")

    chunk = {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "thinking through the answer",
                },
                "finish_reason": None,
                "index": 0,
            }
        ]
    }

    generation_chunk = model._convert_chunk_to_generation_chunk(
        chunk,
        AIMessageChunk,
        {},
    )

    assert generation_chunk is not None
    assert generation_chunk.message.additional_kwargs["reasoning_content"] == (
        "thinking through the answer"
    )


def write_skill(
    root: Path,
    directory: str,
    *,
    name: str,
    description: str,
) -> None:
    """Write a temporary skill file for command catalog tests.

    Args:
        root: The root value.
        directory: The directory value.
        name: The name value.
        description: The description value.
    """
    skill_path = root / directory / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(
        (
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            "---\n\n"
            f"# {name}\n"
        ),
        encoding="utf-8",
    )


def test_virtual_workspace_path_to_local_maps_project_files(tmp_path: Path) -> None:
    """Verify that virtual workspace path to local maps project files.

    Args:
        tmp_path: Path to the tmp.
    """
    assert virtual_workspace_path_to_local(
        "/workspace/skills/reviewer/SKILL.md",
        tmp_path,
    ) == str(tmp_path / "skills/reviewer/SKILL.md")


def test_virtual_workspace_path_to_local_leaves_unknown_paths_unchanged(
    tmp_path: Path,
) -> None:
    """Verify that virtual workspace path to local leaves unknown paths unchanged.

    Args:
        tmp_path: Path to the tmp.
    """
    assert virtual_workspace_path_to_local(
        "/memories/skills/reviewer/SKILL.md",
        tmp_path,
    ) == "/memories/skills/reviewer/SKILL.md"
    assert virtual_workspace_path_to_local(
        "/workspace/../outside/SKILL.md",
        tmp_path,
    ) == "/workspace/../outside/SKILL.md"


def test_runtime_config_reports_rag_error_for_openai_auto_embeddings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that runtime config reports RAG error for openai auto embeddings.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "openai_compatible"
base_url = "http://127.0.0.1:1234/v1"
name = "chat-model"

[rag]
enabled = true

[rag.embedding]
provider = "auto"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.rag_requested is True
    assert config.rag is None
    assert config.rag_error is not None
    assert "rag.embedding.model" in config.rag_error


def test_runtime_config_uses_selected_profile_url_for_matching_rag_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify matching RAG providers inherit the selected profile endpoint."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-chat"

[model.profiles.lmstudio]
provider = "openai_compatible"
base_url = "https://lmstudio.example/v1"
name = "tool-model"
api_key = "profile-key"

[rag]
enabled = true

[rag.embedding]
provider = "openai_compatible"
model = "text-embedding-3-small"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "lmstudio")

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.rag is not None
    assert config.rag.embedding.provider == "openai_compatible"
    assert config.rag.embedding.base_url == "https://lmstudio.example/v1"


def test_runtime_config_keeps_explicit_rag_provider_on_default_model_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify explicit RAG providers do not inherit selected chat profile URLs."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-chat"

[model.profiles.claude]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "profile-key"

[agent]
model = "claude"

[rag]
enabled = true

[rag.embedding]
provider = "ollama"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.rag is not None
    assert config.rag.embedding.provider == "ollama"
    assert config.rag.embedding.base_url == "http://127.0.0.1:11434"


def test_load_extensions_config_reads_mcp_stateful_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config reads MCP stateful flag.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[mcp]
stateful = true

[mcp.servers.repo]
transport = "stdio"
command = "npx"
args = ["server"]

[agent]
mcp_servers = ["repo"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.load_extensions_config()

    assert config.mcp_stateful is True


def test_runtime_config_reads_model_choices_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that runtime config reads model choices from TOML.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "gpt-oss:20b"
models = ["gpt-oss:20b", "gemma4:27b"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.model_name == "gpt-oss:20b"
    assert config.model_choices == ("gpt-oss:20b", "gemma4:27b")


def test_runtime_config_reads_named_model_profiles_and_agent_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify named model profiles can be selected as the main agent model."""
    from langchain_anthropic import ChatAnthropic

    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "default-local"
models = ["default-local", "backup-local"]

[model.profiles.fast]
name = "fast-local"
temperature = 0.2
reasoning_effort = "low"

[model.profiles.claude]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "toml-key"
temperature = 0.4
thinking = "disabled"

[agent]
model = "claude"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert config.model_name == "claude"
    assert config.model_choices == ("claude", "default-local", "backup-local", "fast")
    assert set(config.model_profiles) == {"fast", "claude"}
    assert config.extensions.agent_model == "claude"
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-4-6"
    assert model.anthropic_api_url == deepagent_runtime.DEFAULT_ANTHROPIC_BASE_URL
    assert model.anthropic_api_key.get_secret_value() == "toml-key"
    assert model.temperature == 0.4
    assert model.effort == "medium"
    assert model.thinking is None


def test_profile_agent_model_does_not_change_raw_model_default_reasoning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify raw model choices keep the base model reasoning default."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "default-local"
reasoning_effort = "medium"

[model.profiles.fast]
name = "fast-local"
reasoning_effort = "low"

[agent]
model = "fast"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    selected_profile = deepagent_runtime.resolve_runtime_model_profile(config)
    raw_model = deepagent_runtime.resolve_runtime_model_profile(
        config,
        "default-local",
    )

    assert config.default_reasoning == "medium"
    assert selected_profile.reasoning_effort == "low"
    assert raw_model.reasoning_effort == "medium"


def test_runtime_config_validates_credentials_against_selected_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify unused top-level providers do not require credentials."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "anthropic"
name = "claude-sonnet-4-6"

[model.profiles.local]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-model"

[agent]
model = "local"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENT_MODEL_API_KEY", raising=False)

    config = deepagent_runtime.RuntimeConfig.from_env()
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert config.model_name == "local"
    assert active_model.provider == "ollama"
    assert active_model.name == "local-model"


def test_model_profile_api_key_prefers_env_over_profile_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify environment credentials override selected profile keys."""
    from langchain_anthropic import ChatAnthropic

    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "default-local"

[model.profiles.claude]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "stale-profile-key"

[agent]
model = "claude"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert isinstance(model, ChatAnthropic)
    assert model.anthropic_api_key.get_secret_value() == "env-anthropic-key"


def test_model_profile_inherits_runtime_base_url_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify inherited profile endpoints are rebased on runtime overrides."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://toml-ollama.example:11434"
name = "default-local"

[model.profiles.fast]
name = "fast-local"

[agent]
model = "fast"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-ollama.example:11434")

    config = deepagent_runtime.RuntimeConfig.from_env()
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert config.model_base_url == "http://env-ollama.example:11434"
    assert active_model.base_url == "http://env-ollama.example:11434"
    assert active_model.name == "fast-local"


def test_model_profile_explicit_base_url_uses_runtime_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify runtime endpoint overrides replace same-provider profile URLs."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://toml-ollama.example:11434"
name = "default-local"

[model.profiles.fast]
base_url = "http://profile-ollama.example:11434"
name = "fast-local"

[agent]
model = "fast"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-ollama.example:11434")

    config = deepagent_runtime.RuntimeConfig.from_env()
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert active_model.name == "fast-local"
    assert active_model.base_url == "http://env-ollama.example:11434"


def test_provider_switched_profile_base_url_uses_generic_runtime_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify generic endpoint overrides replace provider-switched profile URLs."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "default-local"

[model.profiles.lmstudio]
provider = "openai_compatible"
base_url = "https://profile-openai.example/v1"
name = "tool-model"
api_key = "profile-key"

[agent]
model = "lmstudio"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_BASE_URL", "https://env-openai.example/v1")

    config = deepagent_runtime.RuntimeConfig.from_env()
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert active_model.provider == "openai_compatible"
    assert active_model.name == "tool-model"
    assert active_model.base_url == "https://env-openai.example/v1"


def test_provider_switched_profile_endpoint_url_uses_generic_runtime_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify endpoint URL overrides can rebase provider-switched profiles."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "default-local"

[model.profiles.lmstudio]
provider = "openai_compatible"
base_url = "https://profile-openai.example/v1"
name = "tool-model"
api_key = "profile-key"

[agent]
model = "lmstudio"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv(
        "DEEPAGENT_MODEL_ENDPOINT_URL",
        (
            "https://env-openai.example/openai/deployments/tool/chat/completions"
            "?api-version=2026-01-01"
        ),
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert active_model.provider == "openai_compatible"
    assert active_model.name == "tool-model"
    assert active_model.base_url == "https://env-openai.example/openai/deployments/tool"
    assert active_model.endpoint_query == (("api-version", "2026-01-01"),)


def test_rebased_profile_preserves_runtime_overrides_for_child_profiles() -> None:
    """Verify inherited rebased profiles carry runtime override provenance."""
    base_model = deepagent_runtime.ModelDefaults(
        provider="ollama",
        base_url="http://env-ollama.example:11434",
        name="default-local",
        temperature=0.7,
        runtime_override_fields=frozenset({"base_url", "temperature"}),
    )
    main_profile = deepagent_runtime.ModelDefaults(
        provider="ollama",
        base_url="http://profile-main.example:11434",
        name="fast-local",
        temperature=0.1,
        explicit_fields=frozenset({"base_url", "name", "temperature"}),
    )
    child_profile = deepagent_runtime.ModelDefaults(
        provider="ollama",
        base_url="http://profile-child.example:11434",
        name="child-local",
        temperature=0.2,
        explicit_fields=frozenset({"base_url", "name", "temperature"}),
    )

    rebased_main = deepagent_runtime.rebase_model_profile_defaults(
        main_profile,
        base_model,
    )
    rebased_child = deepagent_runtime.rebase_model_profile_defaults(
        child_profile,
        rebased_main,
    )

    assert rebased_main.runtime_override_fields == frozenset(
        {"base_url", "temperature"}
    )
    assert rebased_child.base_url == "http://env-ollama.example:11434"
    assert rebased_child.temperature == 0.7


def test_model_profile_temperature_uses_runtime_override(
    tmp_path: Path,
) -> None:
    """Verify runtime temperature overrides replace profile defaults."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "default-local"

[model.profiles.fast]
name = "fast-local"
temperature = 0.1

[agent]
model = "fast"
""".strip(),
        encoding="utf-8",
    )

    config = deepagent_runtime.RuntimeConfig.from_env(
        deepagent_runtime.RuntimeConfigOverrides(
            config_path=config_path,
            model_temperature=0.7,
        )
    )
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert active_model.name == "fast-local"
    assert active_model.temperature == 0.7


def test_model_profile_without_name_inherits_default_name_before_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify inherited model lists do not make profile names explicit."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "prod-local"
models = ["debug-local"]

[model.profiles.fast]
temperature = 0.1

[agent]
model = "fast"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert active_model.name == "prod-local"
    assert active_model.models == ("debug-local",)
    assert active_model.temperature == 0.1


def test_model_profile_name_override_inherits_default_model_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify a profile reference override does not become the inherited model name."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "prod-local"
models = ["debug-local"]

[model.profiles.fast]
temperature = 0.1
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "fast")

    config = deepagent_runtime.RuntimeConfig.from_env()
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert config.model_name == "fast"
    assert active_model.name == "prod-local"
    assert active_model.temperature == 0.1


def test_provider_override_bypasses_agent_model_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify provider overrides do not keep selecting [agent].model profiles."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-default"

[model.profiles.claude]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "profile-key"

[agent]
model = "claude"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DEEPAGENT_MODEL_BASE_URL", "https://openai.example/v1")
    monkeypatch.setenv("DEEPAGENT_MODEL_API_KEY", "openai-key")

    config = deepagent_runtime.RuntimeConfig.from_env()
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert config.model_provider == "openai_compatible"
    assert config.model_name == "local-default"
    assert active_model.provider == "openai_compatible"
    assert active_model.base_url == "https://openai.example/v1"


def test_provider_override_allows_selected_profile_endpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify selected profile endpoints can satisfy provider-switch preflight."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-default"

[model.profiles.lmstudio]
provider = "openai_compatible"
base_url = "https://lmstudio.example/v1"
name = "tool-model"
api_key = "profile-key"

[model.profiles.claude-reviewer]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "anthropic-key"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "lmstudio")

    config = deepagent_runtime.RuntimeConfig.from_env()
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert config.model_provider == "openai_compatible"
    assert config.model_name == "lmstudio"
    assert config.model_choices == ("lmstudio",)
    assert active_model.provider == "openai_compatible"
    assert active_model.base_url == "https://lmstudio.example/v1"


def test_provider_override_rejects_selected_profile_provider_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify provider overrides cannot select a mismatched named profile."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-default"

[model.profiles.claude-reviewer]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "profile-key"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "claude-reviewer")
    monkeypatch.setenv("DEEPAGENT_MODEL_BASE_URL", "https://openai.example/v1")
    monkeypatch.setenv("DEEPAGENT_MODEL_API_KEY", "openai-key")

    with pytest.raises(ValueError, match="provider.*claude-reviewer"):
        deepagent_runtime.RuntimeConfig.from_env()


def test_same_provider_override_preserves_agent_model_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify no-op provider overrides still select [agent].model profiles."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-default"

[model.profiles.fast]
name = "fast-local"
temperature = 0.2

[agent]
model = "fast"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "ollama")

    config = deepagent_runtime.RuntimeConfig.from_env()
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert config.model_provider == "ollama"
    assert config.model_name == "fast"
    assert active_model.name == "fast-local"
    assert active_model.temperature == 0.2


def test_build_model_resolves_named_profile_over_raw_model_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify profile names can switch provider settings at model-build time."""
    from langchain_openai import ChatOpenAI

    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "default-local"

[model.profiles.openai-tools]
provider = "openai_compatible"
base_url = "https://openai-compatible.example/v1"
name = "tool-model"
api_key = "profile-key"
temperature = 0.1
disable_streaming = "tool_calling"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "high", model_name="openai-tools")

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "tool-model"
    assert str(model.openai_api_base).rstrip("/") == "https://openai-compatible.example/v1"
    assert model.openai_api_key.get_secret_value() == "profile-key"
    assert model.temperature == 0.1
    assert model.disable_streaming == "tool_calling"
    assert model.stream_usage is None


def test_build_model_enables_stream_usage_when_profile_opts_in(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Ignoring the profile opt-in must disable streaming usage metadata."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "default-local"

[model.profiles.usage-capable]
provider = "openai_compatible"
base_url = "https://usage-capable.example/v1"
name = "usage-model"
stream_usage = true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    profile = config.model_profiles["usage-capable"]
    model = deepagent_runtime.build_model(
        config,
        "medium",
        model_name="usage-capable",
    )

    assert profile.stream_usage is True
    assert model.stream_usage is True


def test_runtime_config_reads_langfuse_enabled_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that runtime config reads Langfuse tracing from TOML.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "gpt-oss:20b"

[langfuse]
enabled = true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.langfuse.enabled is True


def test_runtime_config_rejects_non_boolean_langfuse_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify invalid Langfuse tracing config fails clearly.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "gpt-oss:20b"

[langfuse]
enabled = "yes"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="langfuse.enabled"):
        deepagent_runtime.RuntimeConfig.from_env()


def test_shutdown_langfuse_client_skips_disabled_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify disabled Langfuse tracing does not import or shutdown Langfuse.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    monkeypatch.setitem(sys.modules, "langfuse", None)

    assert (
        deepagent_runtime.shutdown_langfuse_client(make_runtime_config(tmp_path)) is False
    )


def test_shutdown_langfuse_client_flushes_enabled_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify enabled Langfuse tracing shuts down the shared client.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    shutdown_calls: list[str] = []

    class FakeLangfuseClient:
        """Represent the Langfuse client singleton."""

        def shutdown(self) -> None:
            """Record shutdown requests."""
            shutdown_calls.append("shutdown")

    fake_langfuse = ModuleType("langfuse")
    fake_langfuse.get_client = lambda: FakeLangfuseClient()
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)
    config = dataclasses.replace(
        make_runtime_config(tmp_path),
        langfuse=deepagent_runtime.LangfuseConfig(enabled=True),
    )

    assert deepagent_runtime.shutdown_langfuse_client(config) is True
    assert shutdown_calls == ["shutdown"]


def test_build_langgraph_run_config_attaches_langfuse_callback_handler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Langfuse tracing adds a LangChain callback to agent runs.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    created_handlers: list[object] = []

    class FakeLangfuseCallbackHandler:
        """Represent the Langfuse callback handler."""

        def __init__(self) -> None:
            """Initialize the fake handler."""
            created_handlers.append(self)

    monkeypatch.setattr(
        deepagent_runtime,
        "_import_langfuse_callback_handler",
        lambda: FakeLangfuseCallbackHandler,
    )
    config = dataclasses.replace(
        make_runtime_config(tmp_path),
        langfuse=deepagent_runtime.LangfuseConfig(enabled=True),
    )

    run_config = deepagent_runtime.build_langgraph_run_config(
        config,
        thread_id="thread-1",
    )

    assert run_config["configurable"] == {"thread_id": "thread-1"}
    assert run_config["recursion_limit"] == config.recursion_limit
    assert isinstance(
        run_config["callbacks"][0],
        deepagent_runtime.TokenUsageFileCallbackHandler,
    )
    assert run_config["callbacks"][1:] == created_handlers
    assert run_config["metadata"]["langfuse_session_id"] == "thread-1"
    assert "chainagents" in run_config["tags"]


def test_build_langgraph_run_config_keeps_token_callback_when_langfuse_is_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify token logging remains active without optional Langfuse tracing.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    monkeypatch.setattr(
        deepagent_runtime,
        "_import_langfuse_callback_handler",
        lambda: pytest.fail("disabled Langfuse should not import the callback"),
        raising=False,
    )

    run_config = deepagent_runtime.build_langgraph_run_config(
        make_runtime_config(tmp_path),
        thread_id="thread-1",
    )

    assert run_config["configurable"] == {"thread_id": "thread-1"}
    assert run_config["recursion_limit"] == deepagent_runtime.DEFAULT_RECURSION_LIMIT
    assert len(run_config["callbacks"]) == 1
    assert isinstance(
        run_config["callbacks"][0],
        deepagent_runtime.TokenUsageFileCallbackHandler,
    )
    assert "metadata" not in run_config
    assert "tags" not in run_config


def test_build_langgraph_run_config_token_callback_uses_project_root(
    tmp_path: Path,
) -> None:
    """Ignoring the supplied root must write token usage outside the test project."""
    run_config = deepagent_runtime.build_langgraph_run_config(
        make_runtime_config(tmp_path),
        thread_id="thread-1",
        project_root=tmp_path,
    )
    root_id = uuid4()

    run_config["callbacks"][0].on_chain_end({}, run_id=root_id)

    log_path = tmp_path / ".files" / "token-usage.jsonl"
    assert log_path.exists()
    assert str(root_id) in log_path.read_text()


def test_runtime_config_disables_streaming_for_tool_calls_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify TOML can disable streaming only for tool-calling requests.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "gpt-oss:20b"
disable_streaming_for_tool_calls = true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert config.model_disable_streaming == "tool_calling"
    assert model.disable_streaming == "tool_calling"


def test_runtime_config_disable_streaming_env_overrides_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify environment can override the model disable_streaming option.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "gpt-oss:20b"
disable_streaming = false
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_DISABLE_STREAMING", "tool_calling")

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.model_disable_streaming == "tool_calling"


def test_model_profile_disable_streaming_uses_runtime_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify runtime disable_streaming overrides replace profile defaults."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "default-local"
disable_streaming = false

[model.profiles.fast]
name = "fast-local"
disable_streaming = false

[agent]
model = "fast"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_DISABLE_STREAMING", "tool_calling")

    config = deepagent_runtime.RuntimeConfig.from_env()
    active_model = deepagent_runtime.resolve_runtime_model_profile(config)

    assert config.model_disable_streaming == "tool_calling"
    assert active_model.disable_streaming == "tool_calling"


def test_runtime_config_reads_openai_endpoint_url_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that runtime config reads openai endpoint URL from TOML.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "openai_compatible"
endpoint_url = "https://api.example.test/openai/deployments/local/chat/completions?api-version=2026-01-01"
name = "local-model"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert config.model_base_url == "https://api.example.test/openai/deployments/local"
    assert config.model_endpoint_query == (("api-version", "2026-01-01"),)
    assert model.default_query == {"api-version": "2026-01-01"}


def test_normalize_model_provider_accepts_claude_alias() -> None:
    """Verify that normalize model provider accepts claude alias."""
    assert deepagent_runtime.normalize_model_provider("claude") == "anthropic"


def test_runtime_config_reads_anthropic_model_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that runtime config reads Anthropic model settings from TOML.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    from langchain_anthropic import ChatAnthropic

    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "anthropic"
base_url = "https://claude-proxy.example"
name = "claude-sonnet-4-6"
models = ["claude-sonnet-4-6", "claude-opus-4-8"]
api_key = "toml-key"
temperature = 0.2
reasoning_effort = "low"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert config.model_provider == "anthropic"
    assert config.model_name == "claude-sonnet-4-6"
    assert config.model_choices == ("claude-sonnet-4-6", "claude-opus-4-8")
    assert config.model_base_url == "https://claude-proxy.example"
    assert config.model_api_key == "toml-key"
    assert config.model_temperature == 0.2
    assert config.default_reasoning == "low"
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-sonnet-4-6"
    assert model.anthropic_api_url == "https://claude-proxy.example"
    assert model.anthropic_api_key.get_secret_value() == "toml-key"
    assert model.temperature == 0.2
    assert model.effort == "medium"
    assert model.thinking == {"type": "adaptive"}


def test_anthropic_adaptive_thinking_is_not_enabled_for_unsupported_models(
    tmp_path: Path,
) -> None:
    """Verify unsupported Anthropic models do not receive adaptive thinking."""
    from langchain_anthropic import ChatAnthropic

    config = make_runtime_config(tmp_path)
    config = RuntimeConfig(
        database_url=config.database_url,
        model_provider="anthropic",
        model_name="claude-haiku-4-5-20251001",
        model_choices=("claude-haiku-4-5-20251001",),
        model_base_url="https://api.anthropic.com",
        model_api_key="test-key",
        model_temperature=config.model_temperature,
        default_reasoning=config.default_reasoning,
        persistence_mode=config.persistence_mode,
        extensions=config.extensions,
        rag_requested=config.rag_requested,
        rag=config.rag,
        rag_error=config.rag_error,
    )

    model = deepagent_runtime.build_model(config, "medium")

    assert isinstance(model, ChatAnthropic)
    assert model.effort == "medium"
    assert model.thinking is None


def test_runtime_config_disables_anthropic_thinking_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify TOML can disable Anthropic thinking."""
    from langchain_anthropic import ChatAnthropic

    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "toml-key"
thinking = "disabled"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert config.model_thinking == "disabled"
    assert isinstance(model, ChatAnthropic)
    assert model.thinking is None


def test_runtime_config_forces_anthropic_adaptive_thinking_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify TOML can force Anthropic adaptive thinking."""
    from langchain_anthropic import ChatAnthropic

    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "anthropic"
name = "claude-haiku-4-5-20251001"
api_key = "toml-key"
thinking = "adaptive"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert config.model_thinking == "adaptive"
    assert isinstance(model, ChatAnthropic)
    assert model.thinking == {"type": "adaptive"}


def test_runtime_config_rejects_invalid_model_thinking(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify invalid model thinking config is rejected."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "toml-key"
thinking = "manual"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="model.thinking"):
        deepagent_runtime.RuntimeConfig.from_env()


def test_runtime_config_reads_anthropic_endpoint_url_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that runtime config reads Anthropic endpoint URL from TOML.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "anthropic"
endpoint_url = "https://claude-proxy.example/anthropic/v1/messages"
name = "claude-sonnet-4-6"
api_key = "toml-key"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert config.model_base_url == "https://claude-proxy.example/anthropic"
    assert model.anthropic_api_url == "https://claude-proxy.example/anthropic"


def test_runtime_config_prefers_anthropic_api_key_for_anthropic_provider(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Anthropic uses its provider-specific key before generic keys.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "toml-key"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_API_KEY", "generic-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert config.model_api_key == "anthropic-key"
    assert model.anthropic_api_key.get_secret_value() == "anthropic-key"


def test_runtime_config_switches_to_anthropic_without_base_url_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Anthropic provider switch does not inherit an Ollama base URL.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-model"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "claude")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "claude-opus-4-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.delenv("DEEPAGENT_MODEL_BASE_URL", raising=False)

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.model_provider == "anthropic"
    assert config.model_name == "claude-opus-4-8"
    assert config.model_base_url == deepagent_runtime.DEFAULT_ANTHROPIC_BASE_URL


def test_runtime_config_switches_to_anthropic_with_endpoint_url_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Anthropic provider switch accepts an endpoint URL override.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-model"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "claude-opus-4-8")
    monkeypatch.setenv(
        "DEEPAGENT_MODEL_ENDPOINT_URL",
        "https://claude-proxy.example/proxy/v1/messages",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.delenv("DEEPAGENT_MODEL_BASE_URL", raising=False)

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.model_provider == "anthropic"
    assert config.model_base_url == "https://claude-proxy.example/proxy"


def test_runtime_config_rejects_non_anthropic_toml_key_for_anthropic_switch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Anthropic provider switches do not reuse another provider's key.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "openai_compatible"
base_url = "https://api.openai.example/v1"
name = "openai-model"
api_key = "openai-key"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "claude-opus-4-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPAGENT_MODEL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        deepagent_runtime.RuntimeConfig.from_env()


def test_runtime_config_rejects_stale_base_url_for_anthropic_switch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Anthropic provider switches do not inherit generic stale base URLs.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-model"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "claude-opus-4-8")
    monkeypatch.setenv("DEEPAGENT_MODEL_BASE_URL", "https://api.openai.example/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")

    with pytest.raises(ValueError, match="DEEPAGENT_MODEL_ENDPOINT_URL"):
        deepagent_runtime.RuntimeConfig.from_env()


def test_runtime_config_allows_cli_base_url_for_anthropic_switch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify explicit CLI base URL overrides work for Anthropic switches.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-model"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env(
        deepagent_runtime.RuntimeConfigOverrides(
            model_provider="anthropic",
            model_base_url="https://corp-proxy.example",
            model_name="claude-opus-4-8",
            model_api_key="cli-key",
        )
    )

    assert config.model_provider == "anthropic"
    assert config.model_base_url == "https://corp-proxy.example"
    assert config.model_api_key == "cli-key"


def test_runtime_config_preserves_anthropic_endpoint_query_params(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Anthropic endpoint URL query params remain available to proxies.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "anthropic"
endpoint_url = "https://claude-proxy.example/anthropic/v1/messages?route=claude&token=abc"
name = "claude-sonnet-4-6"
api_key = "toml-key"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert config.model_base_url == "https://claude-proxy.example/anthropic"
    assert config.model_endpoint_query == (("route", "claude"), ("token", "abc"))
    assert model.anthropic_api_url == "https://claude-proxy.example/anthropic"
    assert model._client.default_query == {"route": "claude", "token": "abc"}
    assert model._async_client.default_query == {"route": "claude", "token": "abc"}


def test_runtime_config_requires_anthropic_model_name_when_switching_providers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify Anthropic provider switch requires a Claude model name.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-model"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    monkeypatch.delenv("DEEPAGENT_MODEL_NAME", raising=False)

    with pytest.raises(ValueError, match="DEEPAGENT_MODEL_NAME"):
        deepagent_runtime.RuntimeConfig.from_env()


def test_runtime_config_rejects_ollama_provider_switch_with_only_endpoint_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that runtime config rejects ollama provider switch with only endpoint URL.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "openai_compatible"
base_url = "https://api.example.test/openai/v1"
name = "remote-model"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "ollama")
    monkeypatch.setenv(
        "DEEPAGENT_MODEL_ENDPOINT_URL",
        "http://127.0.0.1:11434/v1/chat/completions",
    )
    monkeypatch.delenv("DEEPAGENT_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="DEEPAGENT_MODEL_BASE_URL"):
        deepagent_runtime.RuntimeConfig.from_env()


def test_runtime_config_reads_recursion_limit_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that runtime config reads recursion limit from TOML.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
recursion_limit = 64
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.delenv("DEEPAGENT_RECURSION_LIMIT", raising=False)

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.recursion_limit == 64
    assert config.extensions.recursion_limit == 64


def test_runtime_config_reads_agent_state_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that runtime config reads agent state from TOML."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
state = "stateless"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.agent_state == "stateless"
    assert config.extensions.agent_state == "stateless"


def test_runtime_config_defaults_agent_scoped_memory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify agent-scoped memory defaults preserve legacy namespace behavior."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.extensions.agent_memory_namespace == "filesystem"
    assert config.extensions.agent_memory_files == ("/memories/AGENTS.md",)


def test_runtime_config_reads_agent_scoped_memory_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify agent memory namespace and startup files are configurable."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
memory_namespace = "repo-agent"
memory_files = ["/memories/AGENTS.md", "/memories/preferences.md"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.extensions.agent_memory_namespace == "repo-agent"
    assert config.extensions.agent_memory_files == (
        "/memories/AGENTS.md",
        "/memories/preferences.md",
    )


def test_runtime_config_allows_empty_agent_memory_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify memory_files = [] disables startup memory loading."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
memory_files = []
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.extensions.agent_memory_files == ()


def test_runtime_config_defaults_reflection_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify reflection is disabled unless explicitly configured."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.extensions.agent_reflection.enabled is False
    assert config.extensions.agent_reflection.memory_file == "/memories/AGENTS.md"
    assert config.extensions.agent_reflection.max_lesson_chars == 700
    assert config.extensions.agent_reflection.tool_failure_mode == "unrecovered"


def test_runtime_config_reads_reflection_from_toml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify reflection config is parsed from [agent.reflection]."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
state = "stateful"

[agent.reflection]
enabled = true
memory_file = "/memories/AGENTS.md"
max_lesson_chars = 512
tool_failure_mode = "unrecovered"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.extensions.agent_reflection.enabled is True
    assert config.extensions.agent_reflection.memory_file == "/memories/AGENTS.md"
    assert config.extensions.agent_reflection.max_lesson_chars == 512
    assert config.extensions.agent_reflection.tool_failure_mode == "unrecovered"


@pytest.mark.parametrize(
    ("agent_config", "message"),
    (
        ('reflection = "yes"', "agent.reflection"),
        (
            '[agent.reflection]\nenabled = "yes"',
            "agent.reflection.enabled",
        ),
        (
            '[agent.reflection]\nenabled = true\nmemory_file = "memories/AGENTS.md"',
            "agent.reflection.memory_file",
        ),
        (
            '[agent.reflection]\nenabled = true\nmemory_file = "/workspace/AGENTS.md"',
            "agent.reflection.memory_file",
        ),
        (
            '[agent.reflection]\nenabled = true\nmax_lesson_chars = 0',
            "agent.reflection.max_lesson_chars",
        ),
        (
            '[agent.reflection]\nenabled = true\ntool_failure_mode = "all"',
            "agent.reflection.tool_failure_mode",
        ),
    ),
)
def test_runtime_config_rejects_invalid_reflection_config(
    tmp_path: Path,
    monkeypatch,
    agent_config: str,
    message: str,
) -> None:
    """Verify invalid reflection config fails clearly."""
    config_path = tmp_path / "deepagent.toml"
    if agent_config.startswith("[agent.reflection]"):
        toml = f"[agent]\nstate = \"stateful\"\n\n{agent_config}"
    else:
        toml = f"[agent]\n{agent_config}"
    config_path.write_text(toml, encoding="utf-8")
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match=message):
        deepagent_runtime.RuntimeConfig.from_env()


def test_runtime_config_rejects_reflection_when_stateless(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify reflection writes require stateful agent memory."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
state = "stateless"

[agent.reflection]
enabled = true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="agent.reflection.enabled"):
        deepagent_runtime.RuntimeConfig.from_env()


@pytest.mark.parametrize(
    ("agent_config", "message"),
    (
        ('memory_namespace = ""', "agent.memory_namespace"),
        ('memory_namespace = "agent-*"', "agent.memory_namespace"),
        ('memory_namespace = "agent namespace"', "agent.memory_namespace"),
        ('memory_namespace = "agent/namespace"', "agent.memory_namespace"),
        ('memory_namespace = "agent?namespace"', "agent.memory_namespace"),
        ('memory_namespace = "agent[namespace]"', "agent.memory_namespace"),
        ('memory_files = "/memories/AGENTS.md"', "agent.memory_files"),
        ('memory_files = ["memories/AGENTS.md"]', "agent.memory_files"),
        ('memory_files = ["/workspace/AGENTS.md"]', "agent.memory_files"),
    ),
)
def test_runtime_config_rejects_invalid_agent_memory_config(
    tmp_path: Path,
    monkeypatch,
    agent_config: str,
    message: str,
) -> None:
    """Verify invalid agent memory config fails clearly."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        f"""
[agent]
{agent_config}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match=message):
        deepagent_runtime.RuntimeConfig.from_env()


def test_runtime_config_rejects_invalid_agent_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that invalid agent state config is rejected."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
state = "sometimes"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="agent.state"):
        deepagent_runtime.RuntimeConfig.from_env()


def test_runtime_config_env_overrides_recursion_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that runtime config environment overrides recursion limit.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
recursion_limit = 64
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_RECURSION_LIMIT", "88")

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.recursion_limit == 88


def test_build_deepagent_backend_stores_large_tool_results_inside_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that build deepagent backend stores large tool results inside project.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    monkeypatch.setattr(deepagent_runtime, "PROJECT_ROOT", tmp_path)

    backend = deepagent_runtime.build_deepagent_backend()
    artifacts_root = deepagent_artifacts_root()
    offloaded_path = f"{deepagent_artifacts_route_prefix()}large_tool_results/tool-call-1"

    assert artifacts_root == tmp_path / ".files" / "deepagent"
    assert offloaded_path.startswith(f"{tmp_path.as_posix()}/.files/deepagent/")

    write_result = backend.write(offloaded_path, "tool output")

    assert write_result.error is None
    assert write_result.path == offloaded_path
    assert backend.artifacts_root == artifacts_root.as_posix()
    assert (artifacts_root / "large_tool_results" / "tool-call-1").read_text(
        encoding="utf-8"
    ) == "tool output"

    read_result = backend.read(offloaded_path)

    assert read_result.error is None
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "tool output"


def test_build_deepagent_backend_routes_generated_outputs_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify downloadable generated outputs use a separate filesystem route."""
    monkeypatch.setattr(deepagent_runtime, "PROJECT_ROOT", tmp_path)

    backend = deepagent_runtime.build_deepagent_backend()
    outputs_root = generated_outputs_root()
    output_path = f"{generated_outputs_route_prefix()}reports/result.txt"

    assert outputs_root == tmp_path / ".files" / "outputs"
    assert output_path.startswith(f"{tmp_path.as_posix()}/.files/outputs/")
    assert backend.artifacts_root == (tmp_path / ".files" / "deepagent").as_posix()

    write_result = backend.write(output_path, "downloadable output")

    assert write_result.error is None
    assert write_result.path == output_path
    assert (outputs_root / "reports" / "result.txt").read_text(
        encoding="utf-8"
    ) == "downloadable output"

    read_result = backend.read(output_path)

    assert read_result.error is None
    assert read_result.file_data is not None
    assert read_result.file_data["content"] == "downloadable output"


def test_build_deepagent_backend_uses_explicit_agent_memory_namespace(
    tmp_path: Path,
) -> None:
    """Verify /memories/ uses an explicit agent-scoped store namespace."""
    backend = deepagent_runtime.build_deepagent_backend(
        project_root=tmp_path,
        memory_namespace="repo-agent",
    )

    memory_backend = backend.routes["/memories/"]

    assert memory_backend._namespace(None) == ("repo-agent",)  # noqa: SLF001


def test_tool_execution_resilience_middleware_returns_error_tool_message() -> None:
    """Verify that tool execution resilience middleware returns error tool message."""
    middleware = ToolExecutionResilienceMiddleware()
    request = ToolCallRequest(
        tool_call={
            "id": "call-1",
            "name": "repo_read_file",
            "args": {"path": "README.md"},
            "type": "tool_call",
        },
        tool=SimpleNamespace(name="repo_read_file"),
        state={},
        runtime=SimpleNamespace(),
    )

    async def failing_handler(_request: ToolCallRequest):
        """Raise a test exception from a middleware-wrapped tool.

        Args:
            _request: The request value.

        Raises:
            ValueError: If the supplied value is invalid.
        """
        raise ValueError("bad path")

    result = asyncio.run(middleware.awrap_tool_call(request, failing_handler))

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "call-1"
    assert result.name == "repo_read_file"
    assert "ValueError: bad path" in str(result.content)
    assert "without aborting the run" in str(result.content)


def test_tool_execution_middleware_maps_workspace_path_tool_args(tmp_path: Path) -> None:
    """Verify that tool execution middleware maps workspace path tool args.

    Args:
        tmp_path: Path to the tmp.
    """
    middleware = ToolExecutionResilienceMiddleware(project_root=tmp_path)
    request = ToolCallRequest(
        tool_call={
            "id": "call-1",
            "name": "read_file",
            "args": {"path": "/workspace/skills/reviewer/SKILL.md"},
            "type": "tool_call",
        },
        tool=SimpleNamespace(name="read_file"),
        state={},
        runtime=SimpleNamespace(),
    )

    def handler(updated_request: ToolCallRequest) -> ToolMessage:
        """Capture tool-call arguments for middleware tests.

        Args:
            updated_request: The updated request value.

        Returns:
            The handler result.
        """
        assert updated_request.tool_call["args"] == {
            "path": str(tmp_path / "skills/reviewer/SKILL.md")
        }
        return ToolMessage(
            content="ok",
            name="read_file",
            tool_call_id="call-1",
            status="success",
        )

    result = middleware.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "success"


def test_agent_runtime_initialize_runs_rag_startup_check(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that agent runtime initialize runs RAG startup check.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    created_services: list[object] = []

    class DummyRAG:
        """Represent dummy r a g."""

        def __init__(self, config, *, project_root: Path) -> None:
            """Initialize the dummy r a g instance.

            Args:
                config: Configuration object used by the operation.
                project_root: Project root used to resolve local paths.
            """
            self.config = config
            self.project_root = project_root
            self.ensure_ready_calls = 0
            created_services.append(self)

        def ensure_ready(self) -> RagStatus:
            """Ensure ready.

            Returns:
                The ready object or status.
            """
            self.ensure_ready_calls += 1
            return RagStatus.ready_status(
                file_count=2,
                chunk_count=3,
                persist_directory=self.config.persist_directory,
            )

        def snapshot(self) -> RagStatus:
            """Return a snapshot of.

            Returns:
                A snapshot of.
            """
            return RagStatus.ready_status(
                file_count=2,
                chunk_count=3,
                persist_directory=self.config.persist_directory,
            )

    monkeypatch.setattr(deepagent_runtime, "WorkspaceDocsRAG", DummyRAG)

    runtime = AgentRuntime(make_runtime_config(tmp_path))
    asyncio.run(runtime._initialize())

    assert len(created_services) == 1
    assert created_services[0].ensure_ready_calls == 1
    assert runtime.rag_status.ready is True


def test_agent_runtime_initialize_skips_postgres_state_handles_when_stateless(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify stateless runtime does not open LangGraph Postgres state handles."""
    config = make_runtime_config(tmp_path)
    config = dataclasses.replace(
        config,
        database_url="postgresql://example.invalid/chainagents",
        persistence_mode="postgres",
        agent_state="stateless",
        extensions=dataclasses.replace(config.extensions, agent_state="stateless"),
        rag_requested=False,
        rag=None,
    )

    def fail_from_conn_string(database_url: str):
        raise AssertionError(f"Unexpected Postgres state init for {database_url}")

    monkeypatch.setattr(
        deepagent_runtime.AsyncPostgresStore,
        "from_conn_string",
        fail_from_conn_string,
    )
    monkeypatch.setattr(
        deepagent_runtime.AsyncPostgresSaver,
        "from_conn_string",
        fail_from_conn_string,
    )

    runtime = AgentRuntime(config)
    asyncio.run(runtime._initialize())

    assert runtime._store is None
    assert runtime._checkpointer is None


def test_persistence_enabled_requires_stateful_agent_state(tmp_path: Path) -> None:
    """Verify stateless mode is not reported as agent persistence enabled."""
    config = make_runtime_config(tmp_path)
    config = dataclasses.replace(
        config,
        database_url="postgresql://example.invalid/chainagents",
        persistence_mode="postgres",
        agent_state="stateless",
        extensions=dataclasses.replace(config.extensions, agent_state="stateless"),
    )
    runtime = AgentRuntime(config)

    assert runtime.persistence_enabled is False


def test_get_agent_includes_rag_tool_when_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that get agent includes RAG tool when ready.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
        """Capture Deep Agent factory arguments for tests.

        Args:
            tools: The tools value.
            kwargs: The kwargs value.

        Returns:
            The fake create deep agent result.
        """
        captured["tools"] = tools or []
        captured["kwargs"] = kwargs
        return object()

    class ReadyRAG:
        """Represent ready r a g."""

        def snapshot(self) -> RagStatus:
            """Return a snapshot of.

            Returns:
                A snapshot of.
            """
            return RagStatus.ready_status(
                file_count=1,
                chunk_count=1,
                persist_directory=tmp_path / ".rag",
            )

        def search(
            self,
            *,
            query: str,
            top_k: int | None = None,
            thread_id: str | None = None,
        ):
            """Search the ready r a g.

            Args:
                query: Search query text.
                top_k: Maximum number of search results to return.
                thread_id: Conversation thread identifier.

            Returns:
                Search results matching the query.
            """
            return {"query": query, "results": []}

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    runtime = AgentRuntime(make_runtime_config(tmp_path))
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()
    runtime._rag_service = ReadyRAG()

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    tool_names = [tool.name for tool in captured["tools"]]
    assert "search_workspace_knowledge" in tool_names
    middleware = captured["kwargs"]["middleware"]
    assert any(isinstance(item, ToolExecutionResilienceMiddleware) for item in middleware)


def test_get_agent_passes_deepagents_backend_instance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that get agent passes a concrete DeepAgents backend instance.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
        """Capture Deep Agent factory arguments for tests.

        Args:
            tools: The tools value.
            kwargs: The kwargs value.

        Returns:
            The fake create deep agent result.
        """
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    runtime = AgentRuntime(make_runtime_config(tmp_path), project_root=tmp_path)
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    backend = captured["kwargs"]["backend"]
    assert isinstance(backend, deepagent_runtime.CompositeBackend)
    assert not callable(backend)
    assert backend.routes["/workspace/"].cwd == tmp_path


@pytest.mark.parametrize(
    ("delete_tool_enabled", "execute_tool_enabled", "expected_high_risk_tools"),
    [
        (False, False, set()),
        (True, False, {"delete"}),
        (False, True, {"execute"}),
        (True, True, {"delete", "execute"}),
    ],
)
def test_deepagents_middleware_restores_todos_and_controls_high_risk_tools(
    tmp_path: Path,
    delete_tool_enabled: bool,
    execute_tool_enabled: bool,
    expected_high_risk_tools: set[str],
) -> None:
    """Verify DeepAgents 0.7 planning state and high-risk tool policy."""
    extensions = ExtensionsConfig(
        config_path=None,
        delete_tool_enabled=delete_tool_enabled,
        execute_tool_enabled=execute_tool_enabled,
    )
    config = make_runtime_config(tmp_path, extensions=extensions)
    backend = build_deepagent_backend(project_root=tmp_path)

    middleware = deepagent_runtime.build_agent_middleware(
        config=config,
        backend=backend,
        project_root=tmp_path,
    )

    assert any(isinstance(item, TodoListMiddleware) for item in middleware)
    filesystem_middleware = [
        item for item in middleware if isinstance(item, FilesystemMiddleware)
    ]
    assert len(filesystem_middleware) == 1
    assert {tool.name for tool in filesystem_middleware[0].tools} == {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        *expected_high_risk_tools,
    }

    graph = deepagent_runtime.create_deep_agent_with_configured_summarization(
        config,
        model=FakeListChatModel(responses=["ok"]),
        middleware=middleware,
        backend=backend,
    )
    tool_names = set(graph.nodes["tools"].bound.tools_by_name)
    assert "write_todos" in tool_names
    assert ("delete" in tool_names) is delete_tool_enabled
    assert ("execute" in tool_names) is execute_tool_enabled
    assert "todos" in graph.channels


def test_get_agent_passes_agent_memory_files_when_stateful(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify stateful agents load configured long-term memory files."""
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
        """Capture Deep Agent factory arguments for tests."""
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    extensions = ExtensionsConfig(
        config_path=None,
        agent_memory_namespace="repo-agent",
        agent_memory_files=("/memories/AGENTS.md", "/memories/preferences.md"),
    )
    runtime = AgentRuntime(
        make_runtime_config(tmp_path, extensions=extensions),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    assert captured["kwargs"]["memory"] == [
        "/memories/AGENTS.md",
        "/memories/preferences.md",
    ]


def test_get_agent_omits_memory_when_stateful_memory_files_empty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify stateful agents can keep /memories/ without startup memory files."""
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
        """Capture Deep Agent factory arguments for tests."""
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    extensions = ExtensionsConfig(
        config_path=None,
        agent_memory_namespace="repo-agent",
        agent_memory_files=(),
    )
    runtime = AgentRuntime(
        make_runtime_config(tmp_path, extensions=extensions),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    assert "memory" not in captured["kwargs"]
    assert "/memories/" in captured["kwargs"]["backend"].routes


def test_get_agent_omits_store_and_checkpointer_when_stateless(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify stateless agents do not receive LangGraph state handles."""
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
        """Capture Deep Agent factory arguments for tests."""
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    config = make_runtime_config(tmp_path)
    config = dataclasses.replace(
        config,
        agent_state="stateless",
        extensions=dataclasses.replace(config.extensions, agent_state="stateless"),
    )
    runtime = AgentRuntime(config, project_root=tmp_path)
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    assert "store" not in captured["kwargs"]
    assert "checkpointer" not in captured["kwargs"]


def test_get_agent_disables_memories_backend_and_prompt_when_stateless(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify stateless agents do not expose unusable memory routes."""
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
        """Capture Deep Agent factory arguments for tests."""
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    config = make_runtime_config(tmp_path)
    config = dataclasses.replace(
        config,
        agent_state="stateless",
        extensions=dataclasses.replace(config.extensions, agent_state="stateless"),
    )
    runtime = AgentRuntime(config, project_root=tmp_path)
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    backend = captured["kwargs"]["backend"]
    system_prompt = captured["kwargs"]["system_prompt"]
    assert "/memories/" not in backend.routes
    assert "/memories/" not in system_prompt
    assert "Agent memory is disabled for this runtime." in system_prompt
    assert "memory" not in captured["kwargs"]


def test_get_agent_leaves_summarization_middleware_to_deepagents_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that get agent does not duplicate DeepAgents summarization middleware.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
        """Capture Deep Agent factory arguments for tests.

        Args:
            tools: The tools value.
            kwargs: The kwargs value.

        Returns:
            The fake create deep agent result.
        """
        captured["tools"] = tools or []
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    runtime = AgentRuntime(
        make_runtime_config(
            tmp_path,
            extensions=ExtensionsConfig(
                config_path=None,
                summarization_middleware_enabled=True,
                summarization_trigger_tokens=5000,
                summarization_keep_tokens=2000,
                subagents=(
                    SubagentConfig(
                        name="repo-researcher",
                        description="Researches the repo",
                        system_prompt="Do research",
                        model="gpt-oss:120b",
                    ),
                ),
            ),
        )
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    middleware = captured["kwargs"]["middleware"]
    assert any(isinstance(item, ToolExecutionResilienceMiddleware) for item in middleware)
    assert any(isinstance(item, TodoListMiddleware) for item in middleware)
    assert not any(
        isinstance(item, deepagent_runtime.SummarizationStatusMiddleware)
        for item in middleware
    )
    subagent_specs = captured["kwargs"]["subagents"]
    subagent_middleware = subagent_specs[0]["middleware"]
    assert any(
        isinstance(item, ToolExecutionResilienceMiddleware)
        for item in subagent_middleware
    )
    assert any(isinstance(item, TodoListMiddleware) for item in subagent_middleware)
    assert not any(
        isinstance(item, deepagent_runtime.SummarizationStatusMiddleware)
        for item in subagent_middleware
    )


def test_get_agent_with_summarization_subagent_does_not_duplicate_middleware(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify DeepAgents managed summarization does not collide with subagents.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    monkeypatch.setattr(
        deepagent_runtime,
        "build_model",
        lambda *_args, **_kwargs: FakeListChatModel(responses=["ok"]),
    )

    runtime = AgentRuntime(
        make_runtime_config(
            tmp_path,
            extensions=ExtensionsConfig(
                config_path=None,
                summarization_middleware_enabled=True,
                summarization_trigger_tokens=5000,
                summarization_keep_tokens=2000,
                subagents=(
                    SubagentConfig(
                        name="repo-researcher",
                        description="Researches the repo",
                        system_prompt="Do research",
                    ),
                ),
            ),
        )
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    agent = asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    assert agent is not None


def test_get_agent_applies_configured_deepagents_summarization_thresholds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify configured summarization token thresholds reach DeepAgents.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    created_summarizers: list[dict[str, object]] = []

    class CapturingSummarizationMiddleware(deepagent_runtime.AgentMiddleware):
        """Capture summarization construction arguments."""

        @property
        def name(self) -> str:
            """Return the public summarization middleware name."""
            return "SummarizationMiddleware"

        def __init__(self, model, *, backend, trigger=None, keep=None, **kwargs) -> None:
            """Initialize the fake summarization middleware instance."""
            created_summarizers.append(
                {
                    "model": model,
                    "backend": backend,
                    "trigger": trigger,
                    "keep": keep,
                    "kwargs": kwargs,
                }
            )

    monkeypatch.setattr(
        "deepagents.middleware.summarization.SummarizationMiddleware",
        CapturingSummarizationMiddleware,
    )
    monkeypatch.setattr(
        deepagent_runtime,
        "build_model",
        lambda *_args, **_kwargs: FakeListChatModel(responses=["ok"]),
    )

    runtime = AgentRuntime(
        make_runtime_config(
            tmp_path,
            extensions=ExtensionsConfig(
                config_path=None,
                summarization_trigger_tokens=5000,
                summarization_keep_tokens=2000,
                subagents=(
                    SubagentConfig(
                        name="repo-researcher",
                        description="Researches the repo",
                        system_prompt="Do research",
                    ),
                ),
            ),
        )
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    agent = asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    assert agent is not None
    assert created_summarizers
    assert {item["trigger"] for item in created_summarizers} == {("tokens", 5000)}
    assert {item["keep"] for item in created_summarizers} == {("tokens", 2000)}


def test_get_agent_builds_compiled_subagents_for_nested_sync_subagents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify nested sync subagents are compiled into child DeepAgents graphs."""
    created_graphs: list[SimpleNamespace] = []
    mcp_tool_calls: list[tuple[str, ...]] = []

    def fake_create_deep_agent(**kwargs):
        """Capture every DeepAgents graph creation call."""
        graph = SimpleNamespace(kwargs=kwargs)
        created_graphs.append(graph)
        return graph

    def fake_build_model(config, reasoning_level, *, model_name=None):
        """Return a visible model marker for graph-construction assertions."""
        return f"model:{model_name or config.model_name}:{reasoning_level}"

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(deepagent_runtime, "build_model", fake_build_model)

    runtime = AgentRuntime(
        make_runtime_config(
            tmp_path,
            extensions=ExtensionsConfig(
                config_path=None,
                mcp_servers={
                    "manager-mcp": {"transport": "stdio", "command": "npx", "args": []},
                    "private-mcp": {"transport": "stdio", "command": "npx", "args": []},
                    "reviewer-mcp": {"transport": "stdio", "command": "npx", "args": []},
                },
                subagents=(
                    SubagentConfig(
                        name="manager",
                        description="Coordinates specialist agents.",
                        system_prompt="Manage the work.",
                        skills=("/workspace/manager-skills/",),
                        mcp_servers=("manager-mcp",),
                        model="manager-model",
                        nested_subagent_names=("reviewer",),
                        subagents=(
                            SubagentConfig(
                                name="private-reviewer",
                                description="Reviews manager output.",
                                system_prompt="Review privately.",
                                skills=("/workspace/private-skills/",),
                                mcp_servers=("private-mcp",),
                            ),
                        ),
                    ),
                    SubagentConfig(
                        name="reviewer",
                        description="Reviews output.",
                        system_prompt="Review publicly.",
                        mcp_servers=("reviewer-mcp",),
                    ),
                ),
            ),
        ),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    async def fake_get_mcp_tools(
        server_names,
        *,
        thread_id=None,
        mcp_session_id=None,
    ):
        """Return visible fake MCP tools for each requested server tuple."""
        mcp_tool_calls.append(tuple(server_names))
        if not server_names:
            return []
        return [SimpleNamespace(name=f"tools:{','.join(server_names)}")]

    runtime._get_mcp_tools = fake_get_mcp_tools  # type: ignore[assignment]

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    assert len(created_graphs) == 2
    manager_kwargs = created_graphs[0].kwargs
    main_kwargs = created_graphs[1].kwargs

    top_level_specs = main_kwargs["subagents"]
    assert [spec["name"] for spec in top_level_specs] == ["manager", "reviewer"]
    assert top_level_specs[0]["description"] == "Coordinates specialist agents."
    assert top_level_specs[0]["runnable"] is created_graphs[0]
    assert "runnable" not in top_level_specs[1]

    assert manager_kwargs["model"] == "model:manager-model:medium"
    assert manager_kwargs["system_prompt"] == "Manage the work."
    assert manager_kwargs["skills"] == ["/workspace/manager-skills/"]
    assert manager_kwargs["tools"][0].name == "tools:manager-mcp"

    nested_specs = manager_kwargs["subagents"]
    assert [spec["name"] for spec in nested_specs] == ["private-reviewer", "reviewer"]
    assert nested_specs[0]["skills"] == ["/workspace/private-skills/"]
    assert nested_specs[0]["tools"][0].name == "tools:private-mcp"
    assert nested_specs[1]["tools"][0].name == "tools:reviewer-mcp"
    assert ("manager-mcp",) in mcp_tool_calls
    assert ("private-mcp",) in mcp_tool_calls
    assert ("reviewer-mcp",) in mcp_tool_calls


def test_get_agent_uses_subagent_model_profile_for_model_and_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify subagent model profiles control model construction and tool schemas."""
    from langchain_anthropic import ChatAnthropic

    async def fake_mcp_tool(**kwargs):
        """Return fake MCP tool arguments."""
        return kwargs

    mcp_tool = StructuredTool(
        name="read_file",
        description="Read a file.",
        args_schema={
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        coroutine=fake_mcp_tool,
    )
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs):
        """Capture the main DeepAgents graph creation call."""
        captured["kwargs"] = kwargs
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    runtime = AgentRuntime(
        RuntimeConfig(
            database_url=None,
            model_provider="openai_compatible",
            model_name="default-openai",
            model_choices=("default-openai", "claude-reviewer"),
            model_base_url="https://openai-compatible.example/v1",
            model_api_key="openai-key",
            model_temperature=0.0,
            default_reasoning="medium",
            persistence_mode="memory",
            extensions=ExtensionsConfig(
                config_path=None,
                mcp_servers={
                    "repo": {"transport": "stdio", "command": "npx", "args": []},
                },
                subagents=(
                    SubagentConfig(
                        name="reviewer",
                        description="Reviews output.",
                        system_prompt="Review the work.",
                        mcp_servers=("repo",),
                        model="claude-reviewer",
                    ),
                ),
            ),
            model_profiles={
                "claude-reviewer": deepagent_runtime.ModelDefaults(
                    provider="anthropic",
                    base_url=deepagent_runtime.DEFAULT_ANTHROPIC_BASE_URL,
                    name="claude-sonnet-4-6",
                    api_key="anthropic-key",
                    temperature=0.2,
                    reasoning_effort="high",
                    thinking="disabled",
                )
            },
        ),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    async def fake_get_mcp_tools(
        server_names,
        *,
        thread_id=None,
        mcp_session_id=None,
    ):
        """Return the fake MCP tool for the configured subagent server."""
        if tuple(server_names) == ("repo",):
            return [mcp_tool]
        return []

    runtime._get_mcp_tools = fake_get_mcp_tools  # type: ignore[assignment]

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    subagent_spec = captured["kwargs"]["subagents"][0]
    assert isinstance(subagent_spec["model"], ChatAnthropic)
    assert subagent_spec["model"].model == "claude-sonnet-4-6"
    assert subagent_spec["model"].anthropic_api_key.get_secret_value() == "anthropic-key"
    assert subagent_spec["model"].effort == "high"
    assert len(subagent_spec["tools"]) == 1
    anthropic_tool = convert_to_anthropic_tool(subagent_spec["tools"][0])
    assert anthropic_tool["input_schema"]["type"] == "object"


def test_get_agent_resanitizes_inherited_tools_for_profile_subagent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify inherited tools are sanitized for an explicit subagent profile."""

    async def fake_mcp_tool(**kwargs):
        """Return fake MCP tool arguments."""
        return kwargs

    inherited_tool = StructuredTool(
        name="read_file",
        description="Read a file.",
        args_schema={
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        coroutine=fake_mcp_tool,
    )
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs):
        """Capture the main DeepAgents graph creation call."""
        captured["kwargs"] = kwargs
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    runtime = AgentRuntime(
        RuntimeConfig(
            database_url=None,
            model_provider="ollama",
            model_name="default-local",
            model_choices=("default-local", "claude-reviewer"),
            model_base_url="http://127.0.0.1:11434",
            model_api_key=None,
            model_temperature=0.0,
            default_reasoning="medium",
            persistence_mode="memory",
            extensions=ExtensionsConfig(
                config_path=None,
                subagents=(
                    SubagentConfig(
                        name="reviewer",
                        description="Reviews output.",
                        system_prompt="Review the work.",
                        model="claude-reviewer",
                    ),
                ),
            ),
            model_profiles={
                "claude-reviewer": deepagent_runtime.ModelDefaults(
                    provider="anthropic",
                    base_url=deepagent_runtime.DEFAULT_ANTHROPIC_BASE_URL,
                    name="claude-sonnet-4-6",
                    api_key="anthropic-key",
                    thinking="disabled",
                )
            },
        ),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    async def fake_build_main_tools(*, thread_id=None, mcp_session_id=None):
        """Return an inherited tool needing Anthropic root-object normalization."""
        return [inherited_tool]

    runtime._build_main_tools = fake_build_main_tools  # type: ignore[assignment]

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    subagent_spec = captured["kwargs"]["subagents"][0]
    inherited_tools = subagent_spec["tools"]
    assert inherited_tools[0] is not inherited_tool
    anthropic_tool = convert_to_anthropic_tool(inherited_tools[0])
    assert anthropic_tool["input_schema"]["type"] == "object"
    assert anthropic_tool["input_schema"]["properties"] == {
        "path": {"type": "string"},
    }


def test_get_agent_preserves_raw_inherited_tools_for_profile_subagent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify provider-switched subagents can use raw main tools."""

    async def fake_mcp_tool(**kwargs):
        """Return fake MCP tool arguments."""
        return kwargs

    inherited_tool = StructuredTool(
        name="read_file",
        description="Read a file.",
        args_schema={
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        coroutine=fake_mcp_tool,
    )
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs):
        """Capture the main DeepAgents graph creation call."""
        captured["kwargs"] = kwargs
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(
        deepagent_runtime,
        "tool_supports_openai_compatible_schema",
        lambda tool: False,
    )

    runtime = AgentRuntime(
        RuntimeConfig(
            database_url=None,
            model_provider="openai_compatible",
            model_name="default-openai",
            model_choices=("default-openai", "claude-reviewer"),
            model_base_url="https://openai-compatible.example/v1",
            model_api_key="openai-key",
            model_temperature=0.0,
            default_reasoning="medium",
            persistence_mode="memory",
            extensions=ExtensionsConfig(
                config_path=None,
                subagents=(
                    SubagentConfig(
                        name="reviewer",
                        description="Reviews output.",
                        system_prompt="Review the work.",
                        model="claude-reviewer",
                    ),
                ),
            ),
            model_profiles={
                "claude-reviewer": deepagent_runtime.ModelDefaults(
                    provider="anthropic",
                    base_url=deepagent_runtime.DEFAULT_ANTHROPIC_BASE_URL,
                    name="claude-sonnet-4-6",
                    api_key="anthropic-key",
                    thinking="disabled",
                )
            },
        ),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    async def fake_build_main_tools(*, thread_id=None, mcp_session_id=None):
        """Return a raw main-agent tool filtered out by the OpenAI main model."""
        return [inherited_tool]

    runtime._build_main_tools = fake_build_main_tools  # type: ignore[assignment]

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    assert captured["kwargs"]["tools"] is None
    subagent_spec = captured["kwargs"]["subagents"][0]
    inherited_tools = subagent_spec["tools"]
    assert inherited_tools[0] is not inherited_tool
    anthropic_tool = convert_to_anthropic_tool(inherited_tools[0])
    assert anthropic_tool["input_schema"]["type"] == "object"


def test_get_agent_does_not_inherit_tools_when_configured_tools_are_filtered(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify filtered own tools do not fall back to inherited main-agent tools."""

    async def fake_mcp_tool(**kwargs):
        """Return fake MCP tool arguments."""
        return kwargs

    inherited_tool = StructuredTool(
        name="read_file",
        description="Read a file.",
        args_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        coroutine=fake_mcp_tool,
    )
    filtered_tool = SimpleNamespace(
        name="bad_schema",
        args_schema={"type": "string"},
    )
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs):
        """Capture the main DeepAgents graph creation call."""
        captured["kwargs"] = kwargs
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    runtime = AgentRuntime(
        RuntimeConfig(
            database_url=None,
            model_provider="ollama",
            model_name="default-local",
            model_choices=("default-local", "openai-reviewer"),
            model_base_url="http://127.0.0.1:11434",
            model_api_key=None,
            model_temperature=0.0,
            default_reasoning="medium",
            persistence_mode="memory",
            extensions=ExtensionsConfig(
                config_path=None,
                mcp_servers={
                    "bad": {"transport": "stdio", "command": "npx", "args": []},
                },
                subagents=(
                    SubagentConfig(
                        name="reviewer",
                        description="Reviews output.",
                        system_prompt="Review the work.",
                        mcp_servers=("bad",),
                        model="openai-reviewer",
                    ),
                ),
            ),
            model_profiles={
                "openai-reviewer": deepagent_runtime.ModelDefaults(
                    provider="openai_compatible",
                    base_url="https://openai.example/v1",
                    name="tool-model",
                    api_key="openai-key",
                )
            },
        ),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    async def fake_build_main_tools(*, thread_id=None, mcp_session_id=None):
        """Return an inherited tool valid for OpenAI-compatible schemas."""
        return [inherited_tool]

    async def fake_get_mcp_tools(
        server_names,
        *,
        thread_id=None,
        mcp_session_id=None,
    ):
        """Return a configured subagent tool that will be filtered out."""
        if tuple(server_names) == ("bad",):
            return [filtered_tool]
        return []

    runtime._build_main_tools = fake_build_main_tools  # type: ignore[assignment]
    runtime._get_mcp_tools = fake_get_mcp_tools  # type: ignore[assignment]

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    subagent_spec = captured["kwargs"]["subagents"][0]
    assert "tools" not in subagent_spec


def test_get_agent_inherits_selected_profile_for_model_less_compiled_subagent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify model-less compiled subagents inherit the selected main profile."""
    created_graphs: list[SimpleNamespace] = []

    def fake_create_deep_agent(**kwargs):
        """Capture every DeepAgents graph creation call."""
        graph = SimpleNamespace(kwargs=kwargs)
        created_graphs.append(graph)
        return graph

    def fake_build_model(config, reasoning_level, *, model_name=None, model_profile=None):
        """Capture the effective model and reasoning level."""
        selected_name = model_profile.name if model_profile is not None else model_name
        return f"model:{selected_name}:{reasoning_level}"

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(deepagent_runtime, "build_model", fake_build_model)

    runtime = AgentRuntime(
        RuntimeConfig(
            database_url=None,
            model_provider="ollama",
            model_name="default-local",
            model_choices=("default-local", "claude-reviewer"),
            model_base_url="http://127.0.0.1:11434",
            model_api_key=None,
            model_temperature=0.0,
            default_reasoning="medium",
            persistence_mode="memory",
            extensions=ExtensionsConfig(
                config_path=None,
                subagents=(
                    SubagentConfig(
                        name="manager",
                        description="Coordinates review.",
                        system_prompt="Manage the work.",
                        subagents=(
                            SubagentConfig(
                                name="reviewer",
                                description="Reviews output.",
                                system_prompt="Review the work.",
                            ),
                        ),
                    ),
                ),
            ),
            model_profiles={
                "claude-reviewer": deepagent_runtime.ModelDefaults(
                    provider="anthropic",
                    base_url=deepagent_runtime.DEFAULT_ANTHROPIC_BASE_URL,
                    name="claude-sonnet-4-6",
                    api_key="anthropic-key",
                    thinking="disabled",
                )
            },
        ),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(
        runtime.get_agent(
            "medium",
            model_name="claude-reviewer",
            thread_id="thread-1",
        )
    )

    assert len(created_graphs) == 2
    manager_kwargs = created_graphs[0].kwargs
    assert manager_kwargs["model"] == "model:claude-sonnet-4-6:medium"


def test_get_agent_uses_selected_profile_reasoning_effort(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify per-request model profile picks apply profile reasoning defaults."""
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs):
        """Capture the main DeepAgents graph creation call."""
        captured["kwargs"] = kwargs
        return SimpleNamespace(kwargs=kwargs)

    def fake_build_model(config, reasoning_level, *, model_name=None, model_profile=None):
        """Capture the effective model and reasoning level."""
        selected_name = model_profile.name if model_profile is not None else model_name
        return f"model:{selected_name}:{reasoning_level}"

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(deepagent_runtime, "build_model", fake_build_model)

    runtime = AgentRuntime(
        RuntimeConfig(
            database_url=None,
            model_provider="ollama",
            model_name="default-local",
            model_choices=("default-local", "fast-local"),
            model_base_url="http://127.0.0.1:11434",
            model_api_key=None,
            model_temperature=0.0,
            default_reasoning="medium",
            persistence_mode="memory",
            extensions=ExtensionsConfig(config_path=None),
            model_profiles={
                "fast-local": deepagent_runtime.ModelDefaults(
                    provider="ollama",
                    base_url="http://127.0.0.1:11434",
                    name="fast-model",
                    reasoning_effort="low",
                    explicit_fields=frozenset({"name", "reasoning_effort"}),
                )
            },
        ),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("medium", model_name="fast-local", thread_id="thread-1"))

    assert captured["kwargs"]["model"] == "model:fast-model:low"


def test_get_agent_explicit_reasoning_overrides_selected_profile_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify explicit runtime reasoning overrides profile reasoning defaults."""
    captured: dict[str, Any] = {}
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "default-local"

[model.profiles.fast]
name = "fast-model"
reasoning_effort = "low"

[agent]
model = "fast"
""".strip(),
        encoding="utf-8",
    )

    def fake_create_deep_agent(**kwargs):
        """Capture the main DeepAgents graph creation call."""
        captured["kwargs"] = kwargs
        return SimpleNamespace(kwargs=kwargs)

    def fake_build_model(config, reasoning_level, *, model_name=None, model_profile=None):
        """Capture the effective model and reasoning level."""
        selected_name = model_profile.name if model_profile is not None else model_name
        return f"model:{selected_name}:{reasoning_level}"

    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    monkeypatch.setenv("DEEPAGENT_MODEL_REASONING", "high")
    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(deepagent_runtime, "build_model", fake_build_model)

    config = deepagent_runtime.RuntimeConfig.from_env()
    runtime = AgentRuntime(config, project_root=tmp_path)
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent(config.default_reasoning, thread_id="thread-1"))

    assert config.default_reasoning == "high"
    assert captured["kwargs"]["model"] == "model:fast-model:high"


def test_get_agent_explicit_default_reasoning_overrides_profile_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify explicit per-run reasoning equal to default still overrides profiles."""
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs):
        """Capture the main DeepAgents graph creation call."""
        captured["kwargs"] = kwargs
        return SimpleNamespace(kwargs=kwargs)

    def fake_build_model(config, reasoning_level, *, model_name=None, model_profile=None):
        """Capture the effective model and reasoning level."""
        selected_name = model_profile.name if model_profile is not None else model_name
        return f"model:{selected_name}:{reasoning_level}"

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(deepagent_runtime, "build_model", fake_build_model)

    runtime = AgentRuntime(
        RuntimeConfig(
            database_url=None,
            model_provider="ollama",
            model_name="default-local",
            model_choices=("default-local", "reviewer"),
            model_base_url="http://127.0.0.1:11434",
            model_api_key=None,
            model_temperature=0.0,
            default_reasoning="low",
            persistence_mode="memory",
            extensions=ExtensionsConfig(config_path=None),
            model_profiles={
                "reviewer": deepagent_runtime.ModelDefaults(
                    provider="ollama",
                    base_url="http://127.0.0.1:11434",
                    name="review-model",
                    reasoning_effort="high",
                    explicit_fields=frozenset({"name", "reasoning_effort"}),
                )
            },
        ),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(
        runtime.get_agent(
            "low",
            model_name="reviewer",
            reasoning_level_is_explicit=True,
            thread_id="thread-1",
        )
    )

    assert captured["kwargs"]["model"] == "model:review-model:low"


def test_get_agent_cache_distinguishes_reasoning_explicitness_for_subagents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify agent caching preserves subagent reasoning explicitness."""
    created_graphs: list[SimpleNamespace] = []

    def fake_create_deep_agent(**kwargs):
        """Capture each DeepAgents graph creation call."""
        graph = SimpleNamespace(kwargs=kwargs)
        created_graphs.append(graph)
        return graph

    def fake_build_model(config, reasoning_level, *, model_name=None, model_profile=None):
        """Capture the effective model and reasoning level."""
        selected_name = model_profile.name if model_profile is not None else model_name
        return f"model:{selected_name}:{reasoning_level}"

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(deepagent_runtime, "build_model", fake_build_model)

    runtime = AgentRuntime(
        RuntimeConfig(
            database_url=None,
            model_provider="ollama",
            model_name="main",
            model_choices=("main", "reviewer"),
            model_base_url="http://127.0.0.1:11434",
            model_api_key=None,
            model_temperature=0.0,
            default_reasoning="high",
            persistence_mode="memory",
            extensions=ExtensionsConfig(
                config_path=None,
                subagents=(
                    SubagentConfig(
                        name="reviewer",
                        description="Reviews output.",
                        system_prompt="Review the work.",
                        model="reviewer",
                    ),
                ),
            ),
            model_profiles={
                "main": deepagent_runtime.ModelDefaults(
                    provider="ollama",
                    base_url="http://127.0.0.1:11434",
                    name="main-model",
                    reasoning_effort="high",
                    explicit_fields=frozenset({"name", "reasoning_effort"}),
                ),
                "reviewer": deepagent_runtime.ModelDefaults(
                    provider="ollama",
                    base_url="http://127.0.0.1:11434",
                    name="review-model",
                    reasoning_effort="low",
                    explicit_fields=frozenset({"name", "reasoning_effort"}),
                ),
            },
        ),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("high", model_name="main", thread_id="thread-1"))
    asyncio.run(
        runtime.get_agent(
            "high",
            model_name="main",
            reasoning_level_is_explicit=True,
            thread_id="thread-1",
        )
    )

    assert len(created_graphs) == 2
    assert created_graphs[0].kwargs["subagents"][0]["model"] == "model:review-model:low"
    assert created_graphs[1].kwargs["subagents"][0]["model"] == "model:review-model:high"


def test_create_configured_graph_uses_agent_profile_reasoning_effort(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify static graph export applies the selected main profile reasoning."""
    captured: dict[str, Any] = {}
    config = RuntimeConfig(
        database_url=None,
        model_provider="ollama",
        model_name="fast",
        model_choices=("fast",),
        model_base_url="http://127.0.0.1:11434",
        model_api_key=None,
        model_temperature=0.0,
        default_reasoning="medium",
        persistence_mode="memory",
        extensions=ExtensionsConfig(config_path=None),
        model_profiles={
            "fast": deepagent_runtime.ModelDefaults(
                provider="ollama",
                base_url="http://127.0.0.1:11434",
                name="fast-model",
                reasoning_effort="high",
                explicit_fields=frozenset({"name", "reasoning_effort"}),
            )
        },
    )

    def fake_create_deep_agent(**kwargs):
        """Capture the configured static graph."""
        captured["kwargs"] = kwargs
        return SimpleNamespace(kwargs=kwargs)

    def fake_build_model(config, reasoning_level, *, model_name=None, model_profile=None):
        """Capture the effective model and reasoning level."""
        selected_name = model_profile.name if model_profile is not None else model_name
        return f"model:{selected_name}:{reasoning_level}"

    monkeypatch.setattr(
        deepagent_runtime.RuntimeConfig,
        "from_env",
        staticmethod(lambda: config),
    )
    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(deepagent_runtime, "build_model", fake_build_model)
    monkeypatch.setattr(
        deepagent_runtime,
        "build_deepagent_backend",
        lambda **kwargs: SimpleNamespace(),
    )

    deepagent_runtime.create_configured_graph(include_async_subagents=False)

    assert captured["kwargs"]["model"] == "model:fast-model:high"


def test_get_agent_preserves_inherited_tools_for_compiled_nested_subagents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify compiled nested parents keep inherited main-agent tools."""
    created_graphs: list[SimpleNamespace] = []

    def fake_create_deep_agent(**kwargs):
        """Capture every DeepAgents graph creation call."""
        graph = SimpleNamespace(kwargs=kwargs)
        created_graphs.append(graph)
        return graph

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(
        deepagent_runtime,
        "build_model",
        lambda config, reasoning_level, *, model_name=None: (
            f"model:{model_name or config.model_name}:{reasoning_level}"
        ),
    )

    runtime = AgentRuntime(
        make_runtime_config(
            tmp_path,
            extensions=ExtensionsConfig(
                config_path=None,
                subagents=(
                    SubagentConfig(
                        name="manager",
                        description="Coordinates specialist agents.",
                        system_prompt="Manage the work.",
                        subagents=(
                            SubagentConfig(
                                name="reviewer",
                                description="Reviews manager output.",
                                system_prompt="Review privately.",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        project_root=tmp_path,
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    async def fake_build_main_tools(*, thread_id=None, mcp_session_id=None):
        """Return a visible inherited main-agent tool."""
        return [SimpleNamespace(name="main-tool")]

    runtime._build_main_tools = fake_build_main_tools  # type: ignore[assignment]

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    assert len(created_graphs) == 2
    manager_kwargs = created_graphs[0].kwargs
    assert manager_kwargs["tools"][0].name == "main-tool"
    assert "tools" not in manager_kwargs["subagents"][0]


def test_build_graph_subagent_specs_preserves_static_inherited_tools_for_compiled_subagents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify static compiled parents keep inherited graph tools."""
    created_graphs: list[SimpleNamespace] = []

    def fake_create_deep_agent(**kwargs):
        """Capture every DeepAgents graph creation call."""
        graph = SimpleNamespace(kwargs=kwargs)
        created_graphs.append(graph)
        return graph

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(
        deepagent_runtime,
        "build_model",
        lambda config, reasoning_level, *, model_name=None: (
            f"model:{model_name or config.model_name}:{reasoning_level}"
        ),
    )

    config = make_runtime_config(
        tmp_path,
        extensions=ExtensionsConfig(
            config_path=None,
            subagents=(
                SubagentConfig(
                    name="manager",
                    description="Coordinates specialist agents.",
                    system_prompt="Manage the work.",
                    subagents=(
                        SubagentConfig(
                            name="reviewer",
                            description="Reviews manager output.",
                            system_prompt="Review privately.",
                        ),
                    ),
                ),
            ),
        ),
    )

    specs = deepagent_runtime.build_graph_subagent_specs(
        config,
        include_async_subagents=False,
        backend=object(),
        project_root=tmp_path,
        inherited_tools=[SimpleNamespace(name="static-tool")],
    )

    assert specs[0]["runnable"] is created_graphs[0]
    manager_kwargs = created_graphs[0].kwargs
    assert manager_kwargs["tools"][0].name == "static-tool"
    assert "tools" not in manager_kwargs["subagents"][0]


def test_summarization_status_middleware_emits_stream_events() -> None:
    """Verify that summarization status middleware emits stream events."""
    events: list[dict[str, str]] = []

    class FakeRuntime:
        """Represent fake runtime."""

        def stream_writer(self, event: dict[str, str]) -> None:
            """Capture custom stream events emitted by middleware tests.

            Args:
                event: LangGraph stream event to process.
            """
            events.append(event)

    class FakeSummarizationMiddleware:
        """Represent fake summarization middleware."""

        def token_counter(self, messages) -> int:
            """Return a fixed token count for summarization tests.

            Args:
                messages: The messages value.

            Returns:
                A fixed token count for summarization tests.
            """
            return 12

        def _should_summarize(self, messages, total_tokens: int) -> bool:
            """Return the configured summarization decision for tests.

            Args:
                messages: The messages value.
                total_tokens: The total tokens value.

            Returns:
                The configured summarization decision for tests.
            """
            return total_tokens >= 10

        def _determine_cutoff_index(self, messages) -> int:
            """Return the configured summarization cutoff index for tests.

            Args:
                messages: The messages value.

            Returns:
                The configured summarization cutoff index for tests.
            """
            return 1

        def before_model(self, state, runtime):
            """Run middleware logic before a model invocation.

            Args:
                state: Runtime state to inspect or update.
                runtime: Agent runtime used by the operation.

            Returns:
                The before model result.
            """
            return {"messages": ["summary"]}

    middleware = deepagent_runtime.SummarizationStatusMiddleware(
        FakeSummarizationMiddleware(),
        source="main-agent",
    )

    result = middleware.before_model(
        {"messages": ["one", "two"]},
        FakeRuntime(),
    )

    assert result == {"messages": ["summary"]}
    assert events == [
        {
            "kind": "summarization_status",
            "status": "started",
            "source": "main-agent",
            "message": "Conversation summarization triggered.",
        },
        {
            "kind": "summarization_status",
            "status": "completed",
            "source": "main-agent",
            "message": "Conversation summarization completed.",
        },
    ]


def test_get_agent_omits_rag_tool_when_service_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that get agent omits RAG tool when service is missing.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
        """Capture Deep Agent factory arguments for tests.

        Args:
            tools: The tools value.
            kwargs: The kwargs value.

        Returns:
            The fake create deep agent result.
        """
        captured["tools"] = tools or []
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    runtime = AgentRuntime(make_runtime_config(tmp_path))
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    tool_names = [tool.name for tool in captured["tools"]]
    assert "search_workspace_knowledge" not in tool_names
    middleware = captured["kwargs"]["middleware"]
    assert any(isinstance(item, ToolExecutionResilienceMiddleware) for item in middleware)


def test_get_agent_includes_render_chainlit_ui_tool_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the main agent receives the built-in Chainlit UI render tool."""
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
        """Capture Deep Agent factory arguments for tests."""
        captured["tools"] = tools or []
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    runtime = AgentRuntime(make_runtime_config(tmp_path))
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    tool_names = [tool.name for tool in captured["tools"]]
    assert "render_chainlit_ui" in tool_names


def test_system_prompt_directs_active_chainlit_ui_interaction() -> None:
    """Verify the built-in prompt pushes active Chainlit generated UI use."""
    prompt = deepagent_runtime.SYSTEM_PROMPT

    assert "Actively use `render_chainlit_ui`" in prompt
    assert "next-step action buttons" in prompt
    assert "For simple one-sentence answers" in prompt
    assert "Do not describe generated panels as above or below the answer" in prompt


def test_get_agent_omits_render_chainlit_ui_tool_when_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the render UI tool is omitted when generative UI is disabled."""
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
        """Capture Deep Agent factory arguments for tests."""
        captured["tools"] = tools or []
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(deepagent_runtime, "create_deep_agent", fake_create_deep_agent)

    runtime = AgentRuntime(
        make_runtime_config(
            tmp_path,
            extensions=ExtensionsConfig(
                config_path=None,
                chainlit_generative_ui_enabled=False,
            ),
        )
    )
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    tool_names = [tool.name for tool in captured["tools"]]
    assert "render_chainlit_ui" not in tool_names


def test_render_chainlit_ui_tool_pushes_generated_panel(monkeypatch) -> None:
    """Verify the render UI tool emits a LangGraph GeneratedPanel UI message."""
    pushed: list[dict[str, Any]] = []

    def fake_push_ui_message(name: str, props: dict[str, Any], **kwargs: Any):
        """Capture UI messages without requiring a runnable context."""
        pushed.append({"name": name, "props": props, **kwargs})
        return {"type": "ui", "id": kwargs.get("id") or "panel-1", "name": name}

    monkeypatch.setattr(deepagent_runtime, "push_ui_message", fake_push_ui_message)

    tool = deepagent_runtime.create_render_chainlit_ui_tool()
    result = tool.invoke(
        {
            "id": "panel-1",
            "title": "Build result",
            "summary": "All checks completed.",
            "facts": {"Tests": "passing"},
            "items": ["Config parsed", "Bridge rendered"],
            "table": {
                "columns": ["Check", "Status"],
                "rows": [["pytest", "pass"]],
            },
            "actions": [
                {"label": "Run full suite", "prompt": "Run the full test suite."}
            ],
        }
    )

    assert pushed == [
        {
            "name": "GeneratedPanel",
            "props": {
                "title": "Build result",
                "summary": "All checks completed.",
                "facts": {"Tests": "passing"},
                "items": ["Config parsed", "Bridge rendered"],
                "table": {
                    "columns": ["Check", "Status"],
                    "rows": [["pytest", "pass"]],
                },
                "actions": [
                    {"label": "Run full suite", "prompt": "Run the full test suite."}
                ],
            },
            "id": "panel-1",
            "metadata": {"source": "main-agent"},
            "state_key": None,
        }
    ]
    assert result == {
        "rendered": True,
        "component": "GeneratedPanel",
        "id": "panel-1",
    }


def test_render_chainlit_ui_tool_promotes_action_items_without_duplicate_list(
    monkeypatch,
) -> None:
    """Verify action-shaped items become buttons without duplicate list rows."""
    pushed: list[dict[str, Any]] = []

    def fake_push_ui_message(name: str, props: dict[str, Any], **kwargs: Any):
        """Capture UI messages without requiring a runnable context."""
        pushed.append({"name": name, "props": props, **kwargs})
        return {"type": "ui", "id": kwargs.get("id") or "panel-1", "name": name}

    monkeypatch.setattr(deepagent_runtime, "push_ui_message", fake_push_ui_message)

    tool = deepagent_runtime.create_render_chainlit_ui_tool()
    tool.invoke(
        {
            "id": "mock-checklist-panel",
            "title": "Task Checklist",
            "summary": "Example of a short checklist panel.",
            "actions": [
                {
                    "label": "Configure model provider",
                    "prompt": "Show me how to change the model provider.",
                },
            ],
            "items": [
                {
                    "label": "Configure model provider",
                    "prompt": "Show me how to change the model provider.",
                },
                {
                    "label": "Set up MCP servers",
                    "prompt": "List the configured MCP servers.",
                },
            ],
        }
    )

    assert "items" not in pushed[0]["props"]
    assert pushed[0]["props"]["actions"] == [
        {
            "label": "Configure model provider",
            "prompt": "Show me how to change the model provider.",
        },
        {
            "label": "Set up MCP servers",
            "prompt": "List the configured MCP servers.",
        },
    ]


def test_stateful_mcp_reuses_session_per_chainlit_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that stateful MCP reuses session per chainlit session.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    created_sessions: list[tuple[str, object]] = []
    closed_sessions: list[object] = []
    load_calls: list[tuple[object, str]] = []

    class FakeMCPClient:
        """Represent fake m c p client.

        Attributes:
            callbacks: The callbacks value.
            tool_interceptors: The tool interceptors value.
        """

        callbacks = object()
        tool_interceptors: list[object] = []

        @asynccontextmanager
        async def session(self, server_name: str, *, auto_initialize: bool = True):
            """Return an async context manager for fake MCP sessions.

            Args:
                server_name: The server name value.
                auto_initialize: The auto initialize value.

            Yields:
                Values produced by session.
            """
            session = object()
            created_sessions.append((server_name, session))
            try:
                yield session
            finally:
                closed_sessions.append(session)

        async def get_tools(self, *, server_name: str | None = None):
            """Return fake MCP tools from the fake client.

            Args:
                server_name: The server name value.

            Raises:
                AssertionError: If the underlying operation fails.
            """
            raise AssertionError("stateful MCP mode should not call get_tools()")

    async def fake_load_mcp_tools(
        session,
        *,
        server_name: str | None = None,
        **kwargs,
    ):
        """Return fake MCP tools for runtime tests.

        Args:
            session: The session value.
            server_name: The server name value.
            kwargs: The kwargs value.

        Returns:
            Fake MCP tools for runtime tests.
        """
        assert kwargs["tool_name_prefix"] is True
        load_calls.append((session, str(server_name)))
        return [SimpleNamespace(name=f"{server_name}_tool", session=session)]

    monkeypatch.setattr(deepagent_runtime, "load_mcp_tools", fake_load_mcp_tools)

    runtime = AgentRuntime(
        make_runtime_config(
            tmp_path,
            extensions=make_extensions_config(
                mcp_stateful=True,
                agent_mcp_servers=("repo",),
            ),
        )
    )
    runtime._mcp_client = FakeMCPClient()

    async def exercise_runtime():
        """Exercise runtime agent loading for a Chainlit session.

        Returns:
            The exercise runtime result.
        """
        session_1_tools_first = await runtime._get_mcp_tools(
            ("repo",),
            thread_id="thread-1",
            mcp_session_id="session-1",
        )
        session_1_tools_second = await runtime._get_mcp_tools(
            ("repo",),
            thread_id="thread-2",
            mcp_session_id="session-1",
        )
        session_2_tools = await runtime._get_mcp_tools(
            ("repo",),
            thread_id="thread-1",
            mcp_session_id="session-2",
        )
        runtime._agents[("medium", "thread-1", None, "session-1")] = object()
        runtime._agents[("medium", "thread-1", None, "session-2")] = object()
        await runtime.close_mcp_session("session-1")
        assert len(closed_sessions) == 1
        assert ("session-1", "repo") not in runtime._mcp_sessions
        assert ("session-1", ("repo",)) not in runtime._mcp_tools_cache
        assert ("medium", "thread-1", None, "session-1") not in runtime._agents
        assert ("medium", "thread-1", None, "session-2") in runtime._agents
        await runtime.close_mcp_session("session-2")
        return session_1_tools_first, session_1_tools_second, session_2_tools

    session_1_tools_first, session_1_tools_second, session_2_tools = asyncio.run(
        exercise_runtime()
    )

    assert len(created_sessions) == 2
    assert len(load_calls) == 2
    assert len(closed_sessions) == 2
    assert session_1_tools_first[0].session is session_1_tools_second[0].session
    assert session_1_tools_first[0].session is not session_2_tools[0].session
    assert closed_sessions[0] is session_1_tools_first[0].session
    assert closed_sessions[1] is session_2_tools[0].session


def test_rebuild_rag_index_clears_cached_agents(
    tmp_path: Path,
) -> None:
    """Verify that rebuild RAG index clears cached agents.

    Args:
        tmp_path: Path to the tmp.
    """
    class RebuildableRAG:
        """Represent rebuildable r a g."""

        def rebuild(self) -> RagStatus:
            """Rebuild the rebuildable r a g.

            Returns:
                The rebuilt object or status.
            """
            return RagStatus.ready_status(
                file_count=3,
                chunk_count=4,
                persist_directory=tmp_path / ".rag",
            )

        def snapshot(self) -> RagStatus:
            """Return a snapshot of.

            Returns:
                A snapshot of.
            """
            return self.rebuild()

    runtime = AgentRuntime(make_runtime_config(tmp_path))
    runtime._rag_service = RebuildableRAG()
    runtime._agents[("medium", "thread-1", None)] = object()

    status = asyncio.run(runtime.rebuild_rag_index())

    assert status.ready is True
    assert runtime._agents == {}


def test_ingest_rag_uploads_delegates_to_rag_service(tmp_path: Path) -> None:
    """Verify that ingest RAG uploads delegates to RAG service.

    Args:
        tmp_path: Path to the tmp.
    """
    class UploadableRAG:
        """Represent uploadable r a g."""

        def ingest_uploaded_files(self, *, thread_id: str, uploads: list[UploadedRagFile]) -> RagUploadResult:
            """Ingest uploaded files.

            Args:
                thread_id: Conversation thread identifier.
                uploads: Uploaded files supplied by the user.

            Returns:
                The ingest uploaded files result.
            """
            return RagUploadResult(
                thread_id=thread_id,
                added_files=tuple(upload.name for upload in uploads),
                indexed_files=len(uploads),
                chunk_count=len(uploads),
            )

    upload_path = tmp_path / "notes.md"
    upload_path.write_text("hello", encoding="utf-8")

    runtime = AgentRuntime(make_runtime_config(tmp_path))
    runtime._rag_service = UploadableRAG()

    result = asyncio.run(
        runtime.ingest_rag_uploads(
            thread_id="thread-9",
            uploads=[UploadedRagFile(path=upload_path, name="notes.md")],
        )
    )

    assert result.success is True
    assert result.added_files == ("notes.md",)


def test_load_extensions_config_parses_chainlit_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config parses chainlit commands.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[mcp.servers.repo]
transport = "stdio"
command = "npx"
args = ["server"]

[[subagents]]
name = "repo-researcher"
description = "Researches the repo"
system_prompt = "Do research"

[chainlit]
model_mode_enabled = false
reasoning_mode_enabled = false
reasoning_steps_enabled = false
tool_steps_enabled = false
startup_status_enabled = false
chronological_ui_enabled = false
commands = [
  { name = "ask-researcher", description = "Delegate to subagent", target = "subagent", value = "repo-researcher" },
  { name = "run-tool", description = "Call MCP tool", target = "mcp_tool", value = "repo_read_file", mcp_server = "repo" },
  { name = "rewrite", description = "Prompt rewrite", target = "prompt", value = "Rewrite prompt", template = "Rewrite: {input}" }
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    extensions = deepagent_runtime.load_extensions_config()

    assert len(extensions.chainlit_commands) == 3
    assert extensions.chainlit_commands[0].name == "ask-researcher"
    assert extensions.chainlit_commands[1].target == "mcp_tool"
    assert extensions.chainlit_commands[2].template == "Rewrite: {input}"
    assert extensions.chainlit_model_mode_enabled is False
    assert extensions.chainlit_reasoning_mode_enabled is False
    assert extensions.chainlit_reasoning_steps_enabled is False
    assert extensions.chainlit_tool_steps_enabled is False
    assert extensions.chainlit_startup_status_enabled is False
    assert extensions.chainlit_chronological_ui_enabled is False
    assert extensions.chainlit_generative_ui_enabled is True


def test_load_extensions_config_parses_chainlit_generative_ui_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config parses generative UI toggle."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[chainlit]
generative_ui_enabled = false
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    extensions = deepagent_runtime.load_extensions_config()

    assert extensions.chainlit_generative_ui_enabled is False


def test_load_extensions_config_parses_chainlit_starters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config parses chainlit starters.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[chainlit]
starters = [
  { label = "Explain repo", message = "Explain this repository.", icon = "book-open" },
  { label = "Review diff", message = "Review the current changes.", command = "review" }
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    extensions = deepagent_runtime.load_extensions_config()

    assert len(extensions.chainlit_starters) == 2
    assert extensions.chainlit_starters[0].label == "Explain repo"
    assert extensions.chainlit_starters[0].message == "Explain this repository."
    assert extensions.chainlit_starters[0].icon == "book-open"
    assert extensions.chainlit_starters[0].command is None
    assert extensions.chainlit_starters[1].label == "Review diff"
    assert extensions.chainlit_starters[1].command == "review"
    assert extensions.chainlit_starters[1].icon is None


def test_load_extensions_config_parses_agent_custom_instruction_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify main-agent custom instructions can be loaded from a file."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    instruction_path = prompts_dir / "main-agent.md"
    instruction_path.write_text(
        "\nPrefer precise answers from local code.\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
custom_instruction_file = "prompts/main-agent.md"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    extensions = deepagent_runtime.load_extensions_config()

    assert extensions.custom_instruction == "Prefer precise answers from local code."


def test_load_extensions_config_rejects_ambiguous_agent_custom_instruction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify inline and file-based main-agent instructions are mutually exclusive."""
    instruction_path = tmp_path / "main-agent.md"
    instruction_path.write_text("Prefer local code.", encoding="utf-8")
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
custom_instruction = "Prefer direct answers."
custom_instruction_file = "main-agent.md"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="custom_instruction_file"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_invalid_chainlit_starters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config rejects invalid chainlit starters.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[chainlit]
starters = [
  { label = "Missing message" }
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="must include a non-empty 'message'"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_parses_summarization_middleware_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config parses summarization middleware flag.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
summarization_middleware_enabled = true
summarization_trigger_tokens = 6000
summarization_keep_tokens = 2400
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    extensions = deepagent_runtime.load_extensions_config()

    assert extensions.summarization_middleware_enabled is True
    assert extensions.summarization_trigger_tokens == 6000
    assert extensions.summarization_keep_tokens == 2400


def test_load_extensions_config_defaults_high_risk_tools_to_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify delete and execute remain unavailable unless the user opts in."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text("[agent]\n", encoding="utf-8")
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    extensions = deepagent_runtime.load_extensions_config()

    assert extensions.delete_tool_enabled is False
    assert extensions.execute_tool_enabled is False


def test_load_extensions_config_parses_high_risk_tool_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify users can opt in to the DeepAgents delete and execute tools."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        "[agent]\ndelete_tool_enabled = true\nexecute_tool_enabled = true\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    extensions = deepagent_runtime.load_extensions_config()

    assert extensions.delete_tool_enabled is True
    assert extensions.execute_tool_enabled is True


def test_load_extensions_config_rejects_non_boolean_delete_tool_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the delete-tool opt-in rejects ambiguous values."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        '[agent]\ndelete_tool_enabled = "yes"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="delete_tool_enabled"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_non_boolean_execute_tool_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the execute-tool opt-in rejects ambiguous values."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        '[agent]\nexecute_tool_enabled = "yes"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="execute_tool_enabled"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_non_boolean_summarization_middleware_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config rejects non boolean summarization middleware flag.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
summarization_middleware_enabled = "yes"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="summarization_middleware_enabled"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_invalid_summarization_token_thresholds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config rejects invalid summarization token thresholds.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
summarization_trigger_tokens = 0
summarization_keep_tokens = "many"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="summarization_trigger_tokens"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_non_boolean_startup_status_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config rejects non boolean startup status flag.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[chainlit]
startup_status_enabled = "no"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="startup_status_enabled"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_non_boolean_reasoning_mode_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config rejects non boolean reasoning mode flag.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[chainlit]
reasoning_mode_enabled = "no"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="reasoning_mode_enabled"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_non_boolean_reasoning_steps_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config rejects non boolean reasoning steps flag.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[chainlit]
reasoning_steps_enabled = "no"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="reasoning_steps_enabled"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_non_boolean_tool_steps_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config rejects non boolean tool steps flag.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[chainlit]
tool_steps_enabled = "no"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="tool_steps_enabled"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_non_boolean_chronological_ui_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config rejects non boolean chronological UI flag.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[chainlit]
chronological_ui_enabled = "sometimes"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="chronological_ui_enabled"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_non_boolean_generative_ui_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config rejects non boolean generative UI flag."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[chainlit]
generative_ui_enabled = "sometimes"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="generative_ui_enabled"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_non_boolean_model_mode_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config rejects non boolean model mode flag.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[chainlit]
model_mode_enabled = "no"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="model_mode_enabled"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_parses_inline_nested_subagents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify inline nested sync subagents are parsed under their parent."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[mcp.servers.repo]
transport = "stdio"
command = "npx"
args = ["server"]

[[subagents]]
name = "manager"
description = "Coordinates specialist agents."
system_prompt = "Manage the work."

[[subagents.subagents]]
name = "private-reviewer"
description = "Reviews manager output."
system_prompt = "Review the work."
skills = ["/workspace/private-reviewer"]
mcp_servers = ["repo"]
model = "gpt-oss:120b"

[[subagents]]
name = "public-reviewer"
description = "Reviews top-level output."
system_prompt = "Review top-level work."
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    extensions = deepagent_runtime.load_extensions_config()

    manager = extensions.subagents[0]
    assert manager.name == "manager"
    assert manager.nested_subagent_names == ()
    assert len(manager.subagents) == 1
    private_reviewer = manager.subagents[0]
    assert private_reviewer.name == "private-reviewer"
    assert private_reviewer.skills == ("/workspace/private-reviewer/",)
    assert private_reviewer.mcp_servers == ("repo",)
    assert private_reviewer.model == "gpt-oss:120b"
    assert extensions.subagents[1].name == "public-reviewer"


def test_load_extensions_config_parses_nested_subagent_references(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify top-level sync subagents can be reused as nested children."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[[subagents]]
name = "manager"
description = "Coordinates specialist agents."
system_prompt = "Manage the work."
nested_subagents = ["reviewer"]

[[subagents]]
name = "reviewer"
description = "Reviews output."
system_prompt = "Review the work."
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    extensions = deepagent_runtime.load_extensions_config()

    assert extensions.subagents[0].name == "manager"
    assert extensions.subagents[0].nested_subagent_names == ("reviewer",)
    assert extensions.subagents[0].subagents == ()


def test_load_extensions_config_rejects_unknown_nested_subagent_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify missing nested subagent references fail at config load."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[[subagents]]
name = "manager"
description = "Coordinates specialist agents."
system_prompt = "Manage the work."
nested_subagents = ["missing-reviewer"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="unknown nested subagent 'missing-reviewer'"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_nested_subagent_reference_cycles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify nested subagent references cannot create delegation cycles."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[[subagents]]
name = "manager"
description = "Coordinates specialist agents."
system_prompt = "Manage the work."
nested_subagents = ["reviewer"]

[[subagents]]
name = "reviewer"
description = "Reviews output."
system_prompt = "Review the work."
nested_subagents = ["manager"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="nested subagent cycle"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_duplicate_nested_child_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify a parent cannot expose two direct children with the same name."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[[subagents]]
name = "manager"
description = "Coordinates specialist agents."
system_prompt = "Manage the work."
nested_subagents = ["reviewer"]

[[subagents.subagents]]
name = "reviewer"
description = "Private reviewer."
system_prompt = "Review privately."

[[subagents]]
name = "reviewer"
description = "Public reviewer."
system_prompt = "Review publicly."
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="duplicate nested child subagent 'reviewer'"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_nested_async_subagents(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify nested subagents are limited to synchronous subagent configs."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[[subagents]]
name = "manager"
description = "Coordinates specialist agents."
system_prompt = "Manage the work."

[[subagents.subagents]]
name = "remote-reviewer"
description = "Remote reviewer."
graph_id = "reviewer"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="nested async subagents are not supported"):
        deepagent_runtime.load_extensions_config()


def test_load_extensions_config_rejects_unknown_chainlit_subagent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that load extensions config rejects unknown chainlit subagent.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[chainlit]
commands = [
  { name = "ask-researcher", description = "Delegate to subagent", target = "subagent", value = "missing-subagent" }
]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="unknown subagent"):
        deepagent_runtime.load_extensions_config()


def test_build_chainlit_command_catalog_includes_main_and_subagent_skills(
    tmp_path: Path,
) -> None:
    """Verify that build chainlit command catalog includes main and subagent skills.

    Args:
        tmp_path: Path to the tmp.
    """
    write_skill(
        tmp_path,
        "skills/reviewer",
        name="reviewer",
        description="Review code for bugs.",
    )
    write_skill(
        tmp_path,
        "subskills/repo-guide",
        name="repo-guide",
        description="Walk the repository.",
    )
    extensions = ExtensionsConfig(
        config_path=None,
        skills=("/workspace/skills/",),
        subagents=(
            SubagentConfig(
                name="repo-researcher",
                description="Researches the repo",
                system_prompt="Do research",
                skills=("/workspace/subskills/",),
            ),
        ),
    )

    commands, notes = build_chainlit_command_catalog(
        extensions,
        backend=build_deepagent_backend(project_root=tmp_path),
        project_root=tmp_path,
    )

    assert [command.name for command in commands] == ["reviewer", "repo-guide"]
    assert commands[0].target == "skill"
    assert commands[0].value == str(tmp_path / "skills/reviewer/SKILL.md")
    assert commands[1].value == str(tmp_path / "subskills/repo-guide/SKILL.md")
    assert notes == ()


def test_compose_agent_system_prompt_includes_agents_md(
    tmp_path: Path,
) -> None:
    """Verify that compose agent system prompt includes agents md.

    Args:
        tmp_path: Path to the tmp.
    """
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text(
        "# Agent Notes\n\nPrefer focused changes.",
        encoding="utf-8",
    )

    prompt = deepagent_runtime.compose_agent_system_prompt(
        "Base prompt.",
        "Use direct answers.",
        project_root=tmp_path,
    )

    assert "Base prompt." in prompt
    assert "Repository instructions from AGENTS.md:" in prompt
    assert "Prefer focused changes." in prompt
    assert prompt.index("Prefer focused changes.") < prompt.index("Use direct answers.")


def test_compose_agent_system_prompt_ignores_missing_agents_md(
    tmp_path: Path,
) -> None:
    """Verify that compose agent system prompt ignores missing agents md.

    Args:
        tmp_path: Path to the tmp.
    """
    prompt = deepagent_runtime.compose_agent_system_prompt(
        "Base prompt.",
        None,
        project_root=tmp_path,
    )

    assert prompt == "Base prompt."


def test_build_chainlit_command_catalog_uses_backend_workspace_root_when_project_root_missing(
    tmp_path: Path,
) -> None:
    """Verify that build chainlit command catalog uses backend workspace root when project root missing.

    Args:
        tmp_path: Path to the tmp.
    """
    write_skill(
        tmp_path,
        "skills/reviewer",
        name="reviewer",
        description="Review code for bugs.",
    )
    extensions = ExtensionsConfig(
        config_path=None,
        skills=("/workspace/skills/",),
    )

    commands, notes = build_chainlit_command_catalog(
        extensions,
        backend=build_deepagent_backend(project_root=tmp_path),
    )

    assert len(commands) == 1
    assert commands[0].value == str(tmp_path / "skills/reviewer/SKILL.md")
    assert notes == ()


def test_build_chainlit_command_catalog_prefers_explicit_command_over_skill(
    tmp_path: Path,
) -> None:
    """Verify that build chainlit command catalog prefers explicit command over skill.

    Args:
        tmp_path: Path to the tmp.
    """
    write_skill(
        tmp_path,
        "skills/reviewer",
        name="reviewer",
        description="Review code for bugs.",
    )
    extensions = ExtensionsConfig(
        config_path=None,
        skills=("/workspace/skills/",),
        chainlit_commands=(
            ChainlitCommandConfig(
                name="reviewer",
                description="Explicit reviewer command",
                target="prompt",
                value="Review this",
            ),
        ),
    )

    commands, notes = build_chainlit_command_catalog(
        extensions,
        backend=build_deepagent_backend(project_root=tmp_path),
        project_root=tmp_path,
    )

    assert len(commands) == 1
    assert commands[0].target == "prompt"
    assert len(notes) == 1
    assert "hidden by explicit Chainlit command" in notes[0]


def test_build_chainlit_command_catalog_prefers_main_agent_skill_over_subagent_skill(
    tmp_path: Path,
) -> None:
    """Verify that build chainlit command catalog prefers main agent skill over subagent skill.

    Args:
        tmp_path: Path to the tmp.
    """
    write_skill(
        tmp_path,
        "skills/reviewer",
        name="reviewer",
        description="Main reviewer",
    )
    write_skill(
        tmp_path,
        "subskills/reviewer",
        name="reviewer",
        description="Subagent reviewer",
    )
    extensions = ExtensionsConfig(
        config_path=None,
        skills=("/workspace/skills/",),
        subagents=(
            SubagentConfig(
                name="repo-researcher",
                description="Researches the repo",
                system_prompt="Do research",
                skills=("/workspace/subskills/",),
            ),
        ),
    )

    commands, notes = build_chainlit_command_catalog(
        extensions,
        backend=build_deepagent_backend(project_root=tmp_path),
        project_root=tmp_path,
    )

    assert len(commands) == 1
    assert commands[0].target == "skill"
    assert commands[0].description == "Main reviewer"
    assert commands[0].value == str(tmp_path / "skills/reviewer/SKILL.md")
    assert len(notes) == 1
    assert "main agent skill" in notes[0]


def test_build_chainlit_command_catalog_uses_later_skill_source_in_same_bucket(
    tmp_path: Path,
) -> None:
    """Verify that build chainlit command catalog uses later skill source in same bucket.

    Args:
        tmp_path: Path to the tmp.
    """
    write_skill(
        tmp_path,
        "skills-a/reviewer",
        name="reviewer",
        description="Earlier reviewer",
    )
    write_skill(
        tmp_path,
        "skills-b/reviewer",
        name="reviewer",
        description="Later reviewer",
    )
    extensions = ExtensionsConfig(
        config_path=None,
        skills=("/workspace/skills-a/", "/workspace/skills-b/"),
    )

    commands, notes = build_chainlit_command_catalog(
        extensions,
        backend=build_deepagent_backend(project_root=tmp_path),
        project_root=tmp_path,
    )

    assert len(commands) == 1
    assert commands[0].description == "Later reviewer"
    assert commands[0].value == str(tmp_path / "skills-b/reviewer/SKILL.md")
    assert notes == ()


def test_invoke_mcp_tool_command_calls_configured_tool(tmp_path: Path) -> None:
    """Verify that invoke MCP tool command calls configured tool.

    Args:
        tmp_path: Path to the tmp.
    """
    class FakeTool:
        """Represent fake tool.

        Attributes:
            name: The name value.
        """

        name = "repo_read_file"

        async def ainvoke(self, payload):
            """Return a fake asynchronous tool invocation result.

            Args:
                payload: The payload value.

            Returns:
                A fake asynchronous tool invocation result.
            """
            return {"ok": True, "payload": payload}

    runtime = AgentRuntime(
        make_runtime_config(
            tmp_path,
            extensions=ExtensionsConfig(
                config_path=None,
                mcp_servers={"repo": {"transport": "stdio", "command": "npx", "args": []}},
            ),
        )
    )

    async def fake_get_mcp_tools(server_names, *, thread_id=None, mcp_session_id=None):
        """Return fake MCP tools for command invocation tests.

        Args:
            server_names: The server names value.
            thread_id: Conversation thread identifier.
            mcp_session_id: MCP session identifier.

        Returns:
            Fake MCP tools for command invocation tests.
        """
        assert server_names == ("repo",)
        assert thread_id == "thread-1"
        assert mcp_session_id is None
        return [FakeTool()]

    runtime._get_mcp_tools = fake_get_mcp_tools  # type: ignore[assignment]

    result = asyncio.run(
        runtime.invoke_mcp_tool_command(
            tool_name="repo_read_file",
            raw_args='{"path":"README.md"}',
            thread_id="thread-1",
            server_name="repo",
        )
    )

    assert result == {"ok": True, "payload": {"path": "README.md"}}
