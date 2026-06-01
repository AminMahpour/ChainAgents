"""Test Deep Agent runtime configuration, RAG, MCP, and command integration."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk, ToolMessage
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
    assert extensions.chainlit_startup_status_enabled is False
    assert extensions.chainlit_chronological_ui_enabled is False


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
