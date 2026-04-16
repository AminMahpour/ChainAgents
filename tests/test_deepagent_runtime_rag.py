from __future__ import annotations

import asyncio
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

import deepagent_runtime
from deepagent_runtime import (
    AgentRuntime,
    ExtensionsConfig,
    RuntimeConfig,
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


def make_runtime_config(project_root: Path) -> RuntimeConfig:
    return RuntimeConfig(
        database_url=None,
        model_provider="ollama",
        model_name="gpt-oss:20b",
        model_base_url="http://127.0.0.1:11434",
        model_api_key=None,
        model_temperature=0.0,
        default_reasoning="medium",
        persistence_mode="memory",
        extensions=ExtensionsConfig(config_path=None),
        rag_requested=True,
        rag=make_runtime_rag_config(project_root),
        rag_error=None,
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
