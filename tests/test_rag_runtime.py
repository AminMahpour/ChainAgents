"""Test RAG configuration, indexing, searching, and upload handling."""

from __future__ import annotations

import hashlib
from dataclasses import replace
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


def test_parsed_rag_config_rejects_symlinked_persist_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify parsed relative storage retains ancestors for symlink checks."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    project_root = tmp_path / "project"
    project_root.mkdir()
    outside_persist = tmp_path / "outside-persist"
    outside_persist.mkdir()
    (project_root / "cache-link").symlink_to(
        outside_persist,
        target_is_directory=True,
    )
    upload_source = project_root / "notes.md"
    upload_source.write_text("release notes", encoding="utf-8")
    parsed = parse_rag_config(
        {
            "rag": {
                "enabled": True,
                "persist_directory": "cache-link/.rag",
            }
        },
        project_root / "deepagent.toml",
    )
    resolved = resolve_rag_config(
        parsed,
        model_provider="ollama",
        model_base_url="http://127.0.0.1:11434",
    )
    assert resolved is not None

    with pytest.raises(ValueError, match="symlink"):
        service = WorkspaceDocsRAG(resolved, project_root=project_root)
        service.ingest_uploaded_files(
            thread_id="thread-1",
            uploads=[UploadedRagFile(path=upload_source, name="notes.md")],
        )

    assert list(outside_persist.iterdir()) == []


def test_ingest_rechecks_symlinked_persist_ancestor_added_after_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify upload operations recheck configured storage ancestors."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    project_root = tmp_path / "project"
    project_root.mkdir()
    persist_parent = project_root / "cache-link"
    config = replace(
        make_resolved_rag_config(project_root),
        persist_directory=persist_parent / ".rag",
    )
    service = WorkspaceDocsRAG(config, project_root=project_root)
    outside_persist = tmp_path / "outside-persist"
    outside_persist.mkdir()
    persist_parent.symlink_to(outside_persist, target_is_directory=True)
    upload_source = project_root / "notes.md"
    upload_source.write_text("release notes", encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        service.ingest_uploaded_files(
            thread_id="thread-1",
            uploads=[UploadedRagFile(path=upload_source, name="notes.md")],
        )

    assert list(outside_persist.iterdir()) == []


def test_trusted_project_root_symlink_can_be_canonicalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a trusted project/config root may use a platform-style alias."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    canonical_project_root = tmp_path / "canonical-project"
    canonical_project_root.mkdir()
    project_alias = tmp_path / "project-alias"
    project_alias.symlink_to(canonical_project_root, target_is_directory=True)
    upload_source = canonical_project_root / "notes.md"
    upload_source.write_text("release notes", encoding="utf-8")
    parsed = parse_rag_config(
        {"rag": {"enabled": True}},
        project_alias / "deepagent.toml",
    )
    resolved = resolve_rag_config(
        parsed,
        model_provider="ollama",
        model_base_url="http://127.0.0.1:11434",
    )
    assert resolved is not None

    service = WorkspaceDocsRAG(resolved, project_root=project_alias)
    result = service.ingest_uploaded_files(
        thread_id="thread-1",
        uploads=[UploadedRagFile(path=upload_source, name="notes.md")],
    )

    assert result.success is True
    assert service.persist_directory == canonical_project_root / ".rag"


def test_explicit_external_persist_directory_remains_supported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a clean explicit path outside the project remains usable."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    project_root = tmp_path / "project"
    project_root.mkdir()
    external_persist = tmp_path / "external" / "rag"
    config = replace(
        make_resolved_rag_config(project_root),
        persist_directory=external_persist,
    )
    upload_source = project_root / "notes.md"
    upload_source.write_text("release notes", encoding="utf-8")

    service = WorkspaceDocsRAG(config, project_root=project_root)
    result = service.ingest_uploaded_files(
        thread_id="thread-1",
        uploads=[UploadedRagFile(path=upload_source, name="notes.md")],
    )

    assert result.success is True
    assert service.persist_directory == external_persist


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


def test_thread_traversal_id_cannot_replace_workspace_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a traversal-like thread ID cannot address the workspace index."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    (tmp_path / "README.md").write_text("workspace release guide", encoding="utf-8")
    upload_source = tmp_path / "notes.md"
    upload_source.write_text("uploaded rag notes", encoding="utf-8")
    service = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)
    assert service.rebuild().ready is True
    workspace_index_before = service.collection_directory.joinpath("index.json").read_bytes()
    workspace_manifest_before = service.manifest_path.read_bytes()

    result = service.ingest_uploaded_files(
        thread_id="..",
        uploads=[UploadedRagFile(path=upload_source, name="notes.md")],
    )

    assert result.success is True
    assert service._thread_upload_root("..").resolve().is_relative_to(
        service.uploads_root.resolve()
    )
    assert (
        service.collection_directory.joinpath("index.json").read_bytes()
        == workspace_index_before
    )
    assert service.manifest_path.read_bytes() == workspace_manifest_before


def test_lossy_equivalent_thread_ids_keep_uploads_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify punctuation in a thread ID cannot collide with a literal underscore."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    first_source = tmp_path / "colon.md"
    first_source.write_text("release notes for colon", encoding="utf-8")
    second_source = tmp_path / "underscore.md"
    second_source.write_text("release notes for underscore", encoding="utf-8")
    service = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)

    service.ingest_uploaded_files(
        thread_id="team:1",
        uploads=[UploadedRagFile(path=first_source, name="colon.md")],
    )
    service.ingest_uploaded_files(
        thread_id="team_1",
        uploads=[UploadedRagFile(path=second_source, name="underscore.md")],
    )

    first_root = service._thread_upload_root("team:1")
    second_root = service._thread_upload_root("team_1")
    assert first_root != second_root
    assert first_root.name == hashlib.sha256(b"team:1").hexdigest()
    assert {
        result["path"]
        for result in service.search(query="release notes", top_k=10, thread_id="team:1")[
            "results"
        ]
        if str(result["path"]).startswith("uploaded/")
    } == {"uploaded/colon.md"}
    assert {
        result["path"]
        for result in service.search(query="release notes", top_k=10, thread_id="team_1")[
            "results"
        ]
        if str(result["path"]).startswith("uploaded/")
    } == {"uploaded/underscore.md"}


@pytest.mark.parametrize(
    "operation", ["search", "ingest", "clone-source", "clone-target", "restart"]
)
def test_digest_shaped_legacy_uploads_are_never_adopted_or_modified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Keep all old uploads intact even when their name equals a new thread digest."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())
    config = make_resolved_rag_config(tmp_path)
    legacy_root = config.persist_directory / "uploads"
    legacy_files = legacy_root / hashlib.sha256(b"new-thread").hexdigest() / "files"
    legacy_files.mkdir(parents=True)
    (legacy_files / "old-private.md").write_text("legacy confidential release rag data")
    (legacy_files.parent / "empty-directory").mkdir()

    def legacy_snapshot():
        return {
            str(path.relative_to(legacy_root)): (
                path.stat().st_ino,
                path.stat().st_mtime_ns,
                path.read_bytes() if path.is_file() else None,
            )
            for path in [legacy_root, *legacy_root.rglob("*")]
        }

    before = legacy_snapshot()
    service = WorkspaceDocsRAG(config, project_root=tmp_path)
    source = tmp_path / "fresh.md"
    source.write_text("fresh release rag content")
    expected = set()
    if operation in {"ingest", "clone-target", "restart"}:
        source_thread = "clean-source" if operation == "clone-target" else "new-thread"
        assert service.ingest_uploaded_files(
            thread_id=source_thread,
            uploads=[UploadedRagFile(path=source, name="fresh.md")],
        ).success
        expected = {"uploaded/fresh.md"}
    if operation == "clone-source":
        result = service.clone_thread_uploads(
            source_thread_id="new-thread", target_thread_id="branch"
        )
        assert result.added_files == ()
        assert not service._thread_upload_files_directory("branch").exists()
    elif operation == "clone-target":
        result = service.clone_thread_uploads(
            source_thread_id="clean-source", target_thread_id="new-thread"
        )
        assert result.success
        assert result.added_files == ("fresh.md",)
    elif operation == "restart":
        service = WorkspaceDocsRAG(config, project_root=tmp_path)

    uploaded_paths = {
        result["path"]
        for result in service.search(
            query="release rag", thread_id="new-thread", top_k=10
        )["results"]
        if result["path"].startswith("uploaded/")
    }
    assert uploaded_paths == expected
    assert legacy_snapshot() == before
    assert service.uploads_root.parent == legacy_root.parent
    assert service.uploads_root != legacy_root
    assert not service.uploads_root.is_relative_to(legacy_root)


def test_clone_thread_uploads_isolated_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify branch threads clone source uploads without escaping or duplicating."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    upload_source = tmp_path / "notes.md"
    upload_source.write_text("uploaded content about release notes", encoding="utf-8")
    service = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)
    service.ingest_uploaded_files(
        thread_id="source-thread",
        uploads=[UploadedRagFile(path=upload_source, name="notes.md")],
    )

    first = service.clone_thread_uploads(
        source_thread_id="source-thread",
        target_thread_id="../../branch-thread",
    )
    second = service.clone_thread_uploads(
        source_thread_id="source-thread",
        target_thread_id="../../branch-thread",
    )

    assert first.added_files == ("notes.md",)
    assert second.added_files == ()
    assert second.conflict is True
    assert service.search(
        query="release notes",
        thread_id="../../branch-thread",
    )["results"][0]["path"] == "uploaded/notes.md"
    target_files = service._thread_upload_files_directory("../../branch-thread")
    assert target_files.resolve().is_relative_to(service.uploads_root.resolve())
    assert [path.name for path in target_files.iterdir()] == ["notes.md"]


def test_clone_thread_uploads_remains_isolated_after_service_recreation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify cloned uploads reload independently after a service restart."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    shared_source = tmp_path / "shared.md"
    shared_source.write_text("release notes shared by branch", encoding="utf-8")
    target_source = tmp_path / "target.md"
    target_source.write_text("release notes for target only", encoding="utf-8")
    service = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)
    service.ingest_uploaded_files(
        thread_id="source-thread",
        uploads=[UploadedRagFile(path=shared_source, name="shared.md")],
    )
    assert service.clone_thread_uploads(
        source_thread_id="source-thread",
        target_thread_id="target-thread",
    ).success
    service.ingest_uploaded_files(
        thread_id="target-thread",
        uploads=[UploadedRagFile(path=target_source, name="target.md")],
    )

    reloaded = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)
    source_paths = {
        result["path"]
        for result in reloaded.search(
            query="release notes",
            top_k=10,
            thread_id="source-thread",
        )["results"]
        if str(result["path"]).startswith("uploaded/")
    }
    target_paths = {
        result["path"]
        for result in reloaded.search(
            query="release notes",
            top_k=10,
            thread_id="target-thread",
        )["results"]
        if str(result["path"]).startswith("uploaded/")
    }

    assert source_paths == {"uploaded/shared.md"}
    assert target_paths == {"uploaded/shared.md", "uploaded/target.md"}


def test_ingest_rejects_symlinked_thread_scope_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ingestion rejects a thread storage symlink before copying data."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    upload_source = tmp_path / "notes.md"
    upload_source.write_text("release notes", encoding="utf-8")
    outside_scope = tmp_path / "outside-scope"
    outside_scope.mkdir()
    service = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)
    service.uploads_root.mkdir(parents=True)
    service._thread_upload_root("thread-1").symlink_to(
        outside_scope,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symlink"):
        service.ingest_uploaded_files(
            thread_id="thread-1",
            uploads=[UploadedRagFile(path=upload_source, name="notes.md")],
        )

    assert list(outside_scope.iterdir()) == []


def test_ingest_rejects_symlinked_upload_parent_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ingestion rejects a symlinked persistence parent."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    upload_source = tmp_path / "notes.md"
    upload_source.write_text("release notes", encoding="utf-8")
    outside_persist = tmp_path / "outside-persist"
    outside_persist.mkdir()
    service = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)
    service.persist_directory.symlink_to(outside_persist, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        service.ingest_uploaded_files(
            thread_id="thread-1",
            uploads=[UploadedRagFile(path=upload_source, name="notes.md")],
        )

    assert list(outside_persist.iterdir()) == []


def test_ingest_rejects_symlinked_collection_before_copy_or_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ingestion validates the whole scope before copying or rebuilding."""
    monkeypatch.setattr(rag_runtime, "RecursiveCharacterTextSplitter", DummySplitter)
    monkeypatch.setattr(rag_runtime, "OllamaEmbeddings", lambda **_: DummyEmbeddings())

    upload_source = tmp_path / "notes.md"
    upload_source.write_text("release notes", encoding="utf-8")
    outside_collection = tmp_path / "outside-collection"
    outside_collection.mkdir()
    marker = outside_collection / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    service = WorkspaceDocsRAG(make_resolved_rag_config(tmp_path), project_root=tmp_path)
    thread_root = service._thread_upload_root("thread-1")
    thread_root.mkdir(parents=True)
    service._thread_upload_collection_directory("thread-1").symlink_to(
        outside_collection,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="symlink"):
        service.ingest_uploaded_files(
            thread_id="thread-1",
            uploads=[UploadedRagFile(path=upload_source, name="notes.md")],
        )

    assert service._thread_upload_files_directory("thread-1").exists() is False
    assert marker.read_text(encoding="utf-8") == "keep"


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
