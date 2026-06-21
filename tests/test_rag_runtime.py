"""Test RAG configuration, indexing, searching, and upload handling."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.utils.function_calling import convert_to_openai_tool

import rag_runtime
from rag_runtime import JsonVectorStore
from rag_runtime import (
    DEFAULT_OLLAMA_EMBEDDING_MODEL,
    DEFAULT_RAG_EXCLUDE_GLOBS,
    DEFAULT_RAG_INCLUDE_GLOBS,
    RagConfig,
    RagEmbeddingConfig,
    ResolvedRagConfig,
    ResolvedRagEmbeddingConfig,
    UploadedRagFile,
    WorkspaceDocsRAG,
    create_search_workspace_knowledge_tool,
    parse_rag_config,
    resolve_rag_config,
)


class DummyEmbeddings:
    """Represent deterministic embeddings for local vector-store tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts.

        Args:
            texts: Text values to embed.

        Returns:
            Deterministic vectors for the supplied texts.
        """
        return [self._vector_for(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Args:
            text: Query text to embed.

        Returns:
            A deterministic vector for the supplied query.
        """
        return self._vector_for(text)

    def _vector_for(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("release")),
            float(lowered.count("rag")),
            float(len(lowered.split())),
        ]


class DummySplitter:
    """Represent dummy splitter."""

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize the dummy splitter instance.

        Args:
            chunk_size: The chunk size value.
            chunk_overlap: The chunk overlap value.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents with the test text splitter.

        Args:
            documents: The documents value.

        Returns:
            The split documents result.
        """
        return list(documents)


def make_resolved_rag_config(project_root: Path) -> ResolvedRagConfig:
    """Build a resolved RAG configuration for tests.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The constructed a resolved rag configuration for tests.
    """
    return ResolvedRagConfig(
        enabled=True,
        persist_directory=project_root / ".rag",
        include_globs=DEFAULT_RAG_INCLUDE_GLOBS,
        exclude_globs=DEFAULT_RAG_EXCLUDE_GLOBS,
        chunk_size=1200,
        chunk_overlap=200,
        top_k=4,
        embedding=ResolvedRagEmbeddingConfig(
            provider="ollama",
            model=DEFAULT_OLLAMA_EMBEDDING_MODEL,
            base_url="http://127.0.0.1:11434",
        ),
    )


def test_json_vector_store_persists_and_reloads_searchable_documents(tmp_path: Path) -> None:
    """Verify the local JSON vector store persists and reloads searchable docs."""
    store_directory = tmp_path / "json-store"
    documents = [
        Document(
            page_content="release notes document",
            metadata={"path": "README.md"},
        ),
        Document(
            page_content="rag upload document",
            metadata={"path": "chainlit.md"},
        ),
    ]

    JsonVectorStore.from_documents(
        documents=documents,
        embedding=DummyEmbeddings(),
        persist_directory=store_directory,
    )
    reloaded = JsonVectorStore.load(
        embedding_function=DummyEmbeddings(),
        persist_directory=store_directory,
    )

    results = reloaded.similarity_search_with_relevance_scores(
        "release notes",
        k=1,
    )

    assert (store_directory / "index.json").exists()
    assert results[0][0].metadata["path"] == "README.md"
    assert results[0][1] > 0


def test_parse_rag_config_defaults(tmp_path: Path) -> None:
    """Verify that parse RAG config defaults.

    Args:
        tmp_path: Path to the tmp.
    """
    config = parse_rag_config({}, tmp_path / "deepagent.toml")

    assert config.enabled is False
    assert config.include_globs == DEFAULT_RAG_INCLUDE_GLOBS
    assert config.exclude_globs == DEFAULT_RAG_EXCLUDE_GLOBS
    assert config.persist_directory == (tmp_path / ".rag").resolve()


def test_resolve_rag_config_auto_ollama_defaults(tmp_path: Path) -> None:
    """Verify that resolve RAG config auto ollama defaults.

    Args:
        tmp_path: Path to the tmp.
    """
    config = RagConfig(
        enabled=True,
        persist_directory=tmp_path / ".rag",
        embedding=RagEmbeddingConfig(provider="auto"),
    )

    resolved = resolve_rag_config(
        config,
        model_provider="ollama",
        model_base_url="http://127.0.0.1:11434",
    )

    assert resolved is not None
    assert resolved.embedding.provider == "ollama"
    assert resolved.embedding.model == DEFAULT_OLLAMA_EMBEDDING_MODEL
    assert resolved.embedding.base_url == "http://127.0.0.1:11434"


def test_resolve_rag_config_requires_model_for_openai_compatible(tmp_path: Path) -> None:
    """Verify that resolve RAG config requires model for openai compatible.

    Args:
        tmp_path: Path to the tmp.
    """
    config = RagConfig(
        enabled=True,
        persist_directory=tmp_path / ".rag",
        embedding=RagEmbeddingConfig(provider="auto"),
    )

    with pytest.raises(ValueError, match="rag.embedding.model"):
        resolve_rag_config(
            config,
            model_provider="openai_compatible",
            model_base_url="http://127.0.0.1:1234/v1",
        )


def test_resolve_rag_config_rejects_auto_for_anthropic(tmp_path: Path) -> None:
    """Verify that auto RAG embeddings reject unsupported Anthropic inference.

    Args:
        tmp_path: Path to the tmp.
    """
    config = RagConfig(
        enabled=True,
        persist_directory=tmp_path / ".rag",
        embedding=RagEmbeddingConfig(provider="auto"),
    )

    with pytest.raises(ValueError, match="rag.embedding.provider"):
        resolve_rag_config(
            config,
            model_provider="anthropic",
            model_base_url="https://api.anthropic.com",
        )


def test_discover_source_paths_only_indexes_docs(tmp_path: Path) -> None:
    """Verify that discover source paths only indexes docs.

    Args:
        tmp_path: Path to the tmp.
    """
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    (tmp_path / "chainlit.md").write_text("chainlit", encoding="utf-8")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "repo-researcher.md").write_text("prompt", encoding="utf-8")
    (tmp_path / "skills" / "reviewer").mkdir(parents=True)
    (tmp_path / "skills" / "reviewer" / "SKILL.md").write_text(
        "skill",
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text("local notes", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('nope')", encoding="utf-8")

    service = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)

    discovered = [path.relative_to(tmp_path).as_posix() for path in service.discover_source_paths()]

    assert discovered == [
        "README.md",
        "chainlit.md",
        "prompts/repo-researcher.md",
        "skills/reviewer/SKILL.md",
    ]


def test_manifest_staleness_detects_doc_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that manifest staleness detects doc changes.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    service = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)
    status = service.rebuild()

    assert status.ready is True
    assert service._manifest_is_current_locked() is True

    (tmp_path / "README.md").write_text("readme updated", encoding="utf-8")

    assert service._manifest_is_current_locked() is False


def test_search_workspace_knowledge_tool_has_object_schema() -> None:
    """Verify that search workspace knowledge tool has object schema."""
    class FakeRAG:
        """Represent fake r a g."""

        def search(
            self,
            *,
            query: str,
            top_k: int | None = None,
            thread_id: str | None = None,
        ) -> dict[str, object]:
            """Search the fake r a g.

            Args:
                query: Search query text.
                top_k: Maximum number of search results to return.
                thread_id: Conversation thread identifier.

            Returns:
                Search results matching the query.
            """
            return {"query": query, "results": [{"path": "README.md", "excerpt": "", "score": 1.0}]}

    tool = create_search_workspace_knowledge_tool(FakeRAG())
    schema = convert_to_openai_tool(tool)
    parameters = schema["function"]["parameters"]

    assert parameters["type"] == "object"
    assert "query" in parameters["properties"]
    assert "top_k" in parameters["properties"]


def test_ingest_uploaded_files_adds_thread_scoped_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that ingest uploaded files adds thread scoped results.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    upload_source = tmp_path / "notes.md"
    upload_source.write_text("uploaded content about release notes", encoding="utf-8")

    service = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)
    upload_result = service.ingest_uploaded_files(
        thread_id="thread-1",
        uploads=[UploadedRagFile(path=upload_source, name="notes.md")],
    )

    assert upload_result.success is True
    assert upload_result.added_files == ("notes.md",)

    search_result = service.search(
        query="release notes",
        thread_id="thread-1",
    )

    assert search_result["results"]
    assert search_result["results"][0]["path"] == "uploaded/notes.md"


def test_ingest_uploaded_files_rejects_unsupported_extensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that ingest uploaded files rejects unsupported extensions.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    upload_source = tmp_path / "binary.exe"
    upload_source.write_text("not really binary", encoding="utf-8")

    service = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)
    upload_result = service.ingest_uploaded_files(
        thread_id="thread-2",
        uploads=[UploadedRagFile(path=upload_source, name="binary.exe")],
    )

    assert upload_result.success is False
    assert upload_result.rejected_files == ("binary.exe",)
