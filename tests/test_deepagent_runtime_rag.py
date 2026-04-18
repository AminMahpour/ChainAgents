from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
import pytest

import deepagent_runtime
from deepagent_runtime import (
    AgentRuntime,
    ExtensionsConfig,
    RuntimeConfig,
    ToolExecutionResilienceMiddleware,
    deepagent_artifacts_root,
    deepagent_artifacts_route_prefix,
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
    return RuntimeConfig(
        database_url=None,
        model_provider="ollama",
        model_name="gpt-oss:20b",
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
    return ExtensionsConfig(
        config_path=None,
        mcp_stateful=mcp_stateful,
        mcp_servers={"repo": {"transport": "stdio", "command": "npx", "args": []}},
        agent_mcp_servers=agent_mcp_servers,
    )


def test_runtime_config_reports_rag_error_for_openai_auto_embeddings(
    tmp_path: Path,
    monkeypatch,
) -> None:
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


def test_build_deepagent_backend_stores_large_tool_results_inside_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        raise ValueError("bad path")

    result = asyncio.run(middleware.awrap_tool_call(request, failing_handler))

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "call-1"
    assert result.name == "repo_read_file"
    assert "ValueError: bad path" in str(result.content)
    assert "without aborting the run" in str(result.content)


def test_agent_runtime_initialize_runs_rag_startup_check(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_services: list[object] = []

    class DummyRAG:
        def __init__(self, config, *, project_root: Path) -> None:
            self.config = config
            self.project_root = project_root
            self.ensure_ready_calls = 0
            created_services.append(self)

        def ensure_ready(self) -> RagStatus:
            self.ensure_ready_calls += 1
            return RagStatus.ready_status(
                file_count=2,
                chunk_count=3,
                persist_directory=self.config.persist_directory,
            )

        def snapshot(self) -> RagStatus:
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
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
        captured["tools"] = tools or []
        captured["kwargs"] = kwargs
        return object()

    class ReadyRAG:
        def snapshot(self) -> RagStatus:
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


def test_get_agent_omits_rag_tool_when_service_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_create_deep_agent(*, tools=None, **kwargs):
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


def test_stateful_mcp_reuses_session_per_thread(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_sessions: list[tuple[str, object]] = []
    load_calls: list[tuple[object, str]] = []

    class FakeMCPClient:
        callbacks = object()
        tool_interceptors: list[object] = []

        @asynccontextmanager
        async def session(self, server_name: str, *, auto_initialize: bool = True):
            session = object()
            created_sessions.append((server_name, session))
            yield session

        async def get_tools(self, *, server_name: str | None = None):
            raise AssertionError("stateful MCP mode should not call get_tools()")

    async def fake_load_mcp_tools(
        session,
        *,
        server_name: str | None = None,
        **kwargs,
    ):
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
        thread_1_tools_first = await runtime._get_mcp_tools(
            ("repo",),
            thread_id="thread-1",
        )
        thread_1_tools_second = await runtime._get_mcp_tools(
            ("repo",),
            thread_id="thread-1",
        )
        thread_2_tools = await runtime._get_mcp_tools(
            ("repo",),
            thread_id="thread-2",
        )
        await runtime._exit_stack.aclose()
        return thread_1_tools_first, thread_1_tools_second, thread_2_tools

    thread_1_tools_first, thread_1_tools_second, thread_2_tools = asyncio.run(
        exercise_runtime()
    )

    assert len(created_sessions) == 2
    assert len(load_calls) == 2
    assert thread_1_tools_first[0].session is thread_1_tools_second[0].session
    assert thread_1_tools_first[0].session is not thread_2_tools[0].session


def test_rebuild_rag_index_clears_cached_agents(
    tmp_path: Path,
) -> None:
    class RebuildableRAG:
        def rebuild(self) -> RagStatus:
            return RagStatus.ready_status(
                file_count=3,
                chunk_count=4,
                persist_directory=tmp_path / ".rag",
            )

        def snapshot(self) -> RagStatus:
            return self.rebuild()

    runtime = AgentRuntime(make_runtime_config(tmp_path))
    runtime._rag_service = RebuildableRAG()
    runtime._agents[("medium", "thread-1", None)] = object()

    status = asyncio.run(runtime.rebuild_rag_index())

    assert status.ready is True
    assert runtime._agents == {}


def test_ingest_rag_uploads_delegates_to_rag_service(tmp_path: Path) -> None:
    class UploadableRAG:
        def ingest_uploaded_files(self, *, thread_id: str, uploads: list[UploadedRagFile]) -> RagUploadResult:
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


def test_load_extensions_config_rejects_unknown_chainlit_subagent(
    tmp_path: Path,
    monkeypatch,
) -> None:
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


def test_invoke_mcp_tool_command_calls_configured_tool(tmp_path: Path) -> None:
    class FakeTool:
        name = "repo_read_file"

        async def ainvoke(self, payload):
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

    async def fake_get_mcp_tools(server_names, *, thread_id=None):
        assert server_names == ("repo",)
        assert thread_id == "thread-1"
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
