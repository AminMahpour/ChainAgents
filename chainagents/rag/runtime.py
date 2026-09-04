"""Configure and operate workspace retrieval augmented generation for ChainAgents."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Sequence

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


RagEmbeddingProvider = Literal["auto", "ollama", "openai_compatible"]
ResolvedRagEmbeddingProvider = Literal["ollama", "openai_compatible"]

DEFAULT_RAG_PERSIST_DIRECTORY = ".rag"
DEFAULT_RAG_INCLUDE_GLOBS = (
    "README.md",
    "chainlit.md",
    "prompts/**/*.md",
    "skills/**/*.md",
)
DEFAULT_RAG_EXCLUDE_GLOBS = (
    "AGENTS.md",
    "AGENT.md",
)
DEFAULT_RAG_CHUNK_SIZE = 1200
DEFAULT_RAG_CHUNK_OVERLAP = 200
DEFAULT_RAG_TOP_K = 4
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
RAG_MANIFEST_VERSION = 1
RAG_COLLECTION_NAME = "workspace_docs"
RAG_UPLOADS_DIRECTORY_NAME = "uploads"
RAG_UPLOAD_FILES_DIRECTORY_NAME = "files"
RAG_UPLOAD_COLLECTION_DIRECTORY_NAME = "vectorstore"
RAG_UPLOAD_MANIFEST_FILENAME = "manifest.json"
VECTOR_STORE_DIRECTORY_NAME = "vectorstore"
VECTOR_STORE_INDEX_FILENAME = "index.json"
ALLOWED_RAG_UPLOAD_EXTENSIONS = (
    ".csv",
    ".json",
    ".log",
    ".md",
    ".py",
    ".rst",
    ".text",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
)

RAG_SYSTEM_PROMPT_SUFFIX = """

Knowledge retrieval:
- When the user asks about project documentation, instructions, prompts, or skill docs, use `search_workspace_knowledge` first.
- Cite retrieved sources using the relative file paths returned by the tool.
- If the tool returns no useful matches, say so briefly and fall back to browsing `/workspace/` directly.
""".rstrip()


def compose_rag_system_prompt(base_prompt: str, *, rag_enabled: bool) -> str:
    """Compose RAG system prompt.

    Args:
        base_prompt: The base prompt value.
        rag_enabled: The RAG enabled value.

    Returns:
        The composed value.
    """
    if not rag_enabled:
        return base_prompt
    return f"{base_prompt.rstrip()}\n{RAG_SYSTEM_PROMPT_SUFFIX}"


class JsonVectorStore:
    """Persist and search embedded documents with a JSON index."""

    def __init__(
        self,
        *,
        embedding_function: Any,
        persist_directory: str | Path,
        entries: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the JSON vector store.

        Args:
            embedding_function: Embeddings provider used for queries.
            persist_directory: Directory that stores the JSON index.
            entries: Existing index entries.
        """
        self.embedding_function = embedding_function
        self.persist_directory = Path(persist_directory)
        self.index_path = self.persist_directory / VECTOR_STORE_INDEX_FILENAME
        self._entries = entries or []

    @classmethod
    def from_documents(
        cls,
        *,
        documents: list[Document],
        embedding: Any,
        persist_directory: str | Path,
    ) -> "JsonVectorStore":
        """Build and persist a vector store from documents.

        Args:
            documents: Documents to index.
            embedding: Embeddings provider used to embed the documents.
            persist_directory: Directory that stores the JSON index.

        Returns:
            The persisted vector store.
        """
        texts = [document.page_content for document in documents]
        vectors = embedding.embed_documents(texts) if texts else []
        if len(vectors) != len(documents):
            raise ValueError("Embedding provider returned an unexpected document count.")

        entries = [
            {
                "embedding": cls._coerce_vector(vector),
                "metadata": dict(document.metadata),
                "page_content": document.page_content,
            }
            for document, vector in zip(documents, vectors, strict=True)
        ]
        store = cls(
            embedding_function=embedding,
            persist_directory=persist_directory,
            entries=entries,
        )
        store.persist()
        return store

    @classmethod
    def load(
        cls,
        *,
        embedding_function: Any,
        persist_directory: str | Path,
    ) -> "JsonVectorStore":
        """Load a vector store from a JSON index.

        Args:
            embedding_function: Embeddings provider used for queries.
            persist_directory: Directory containing the JSON index.

        Returns:
            The loaded vector store.
        """
        persist_path = Path(persist_directory)
        index_path = persist_path / VECTOR_STORE_INDEX_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(f"Vector store index does not exist: {index_path}")
        with index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_entries = payload.get("documents", []) if isinstance(payload, dict) else []
        if not isinstance(raw_entries, list):
            raise ValueError("Vector store index is invalid.")

        entries = [
            {
                "embedding": cls._coerce_vector(entry.get("embedding", [])),
                "metadata": dict(entry.get("metadata", {})),
                "page_content": str(entry.get("page_content", "")),
            }
            for entry in raw_entries
            if isinstance(entry, dict)
        ]
        return cls(
            embedding_function=embedding_function,
            persist_directory=persist_path,
            entries=entries,
        )

    @staticmethod
    def index_exists(persist_directory: str | Path) -> bool:
        """Return whether a persisted JSON index exists."""
        return (Path(persist_directory) / VECTOR_STORE_INDEX_FILENAME).exists()

    def persist(self) -> None:
        """Persist the vector store index."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": self._entries,
            "version": RAG_MANIFEST_VERSION,
        }
        with self.index_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        *,
        k: int,
    ) -> list[tuple[Document, float]]:
        """Return the top matching documents scored by cosine similarity."""
        if not self._entries or k <= 0:
            return []

        query_vector = self._coerce_vector(self.embedding_function.embed_query(query))
        scored_entries = [
            (
                Document(
                    page_content=str(entry["page_content"]),
                    metadata=dict(entry["metadata"]),
                ),
                self._cosine_similarity(query_vector, entry["embedding"]),
            )
            for entry in self._entries
        ]
        return sorted(scored_entries, key=lambda item: item[1], reverse=True)[:k]

    @staticmethod
    def _coerce_vector(vector: Sequence[Any]) -> list[float]:
        return [float(value) for value in vector]

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if len(left) != len(right) or not left:
            return 0.0

        dot_product = sum(
            left_value * right_value
            for left_value, right_value in zip(left, right)
        )
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot_product / (left_norm * right_norm)


def normalize_rag_embedding_provider(
    value: Any | None,
    *,
    default: RagEmbeddingProvider = "auto",
) -> RagEmbeddingProvider:
    """Normalize RAG embedding provider.

    Args:
        value: Value to normalize, convert, or serialize.
        default: Fallback value used when no explicit value is available.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    candidate = str(value or default).strip().lower().replace("-", "_")
    if not candidate:
        return default
    if candidate not in {"auto", "ollama", "openai_compatible"}:
        raise ValueError(
            "RAG embedding provider must be 'auto', 'ollama', or 'openai_compatible'."
        )
    return candidate  # type: ignore[return-value]


def normalize_bool(value: Any | None, *, field_name: str, default: bool) -> bool:
    """Normalize bool.

    Args:
        value: Value to normalize, convert, or serialize.
        field_name: The field name value.
        default: Fallback value used when no explicit value is available.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"'{field_name}' must be a boolean.")


def normalize_int(
    value: Any | None,
    *,
    field_name: str,
    default: int,
    minimum: int,
) -> int:
    """Normalize int.

    Args:
        value: Value to normalize, convert, or serialize.
        field_name: The field name value.
        default: Fallback value used when no explicit value is available.
        minimum: The minimum value.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return default
    try:
        candidate = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be an integer.") from exc
    if candidate < minimum:
        raise ValueError(f"'{field_name}' must be at least {minimum}.")
    return candidate


def normalize_glob_list(
    value: Any | None,
    *,
    field_name: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    """Normalize glob list.

    Args:
        value: Value to normalize, convert, or serialize.
        field_name: The field name value.
        default: Fallback value used when no explicit value is available.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError(f"'{field_name}' must be an array of glob strings.")
    globs = tuple(str(item).strip() for item in value if str(item).strip())
    if not globs:
        raise ValueError(f"'{field_name}' must include at least one glob.")
    return globs


def normalize_optional_string(value: Any | None) -> str | None:
    """Normalize optional string.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized value.
    """
    candidate = str(value or "").strip()
    return candidate or None


@dataclass(frozen=True)
class RagEmbeddingConfig:
    """Store raw embedding configuration loaded from deepagent.toml.

    Attributes:
        provider: The provider value.
        model: Model name or model object used by the runtime.
        base_url: URL for the base.
        api_key: The API key value.
    """

    provider: RagEmbeddingProvider = "auto"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class ResolvedRagEmbeddingConfig:
    """Store validated embedding settings used to build the vector store.

    Attributes:
        provider: The provider value.
        model: Model name or model object used by the runtime.
        base_url: URL for the base.
        api_key: The API key value.
    """

    provider: ResolvedRagEmbeddingProvider
    model: str
    base_url: str
    api_key: str | None = None


@dataclass(frozen=True)
class RagConfig:
    """Store raw RAG configuration loaded from deepagent.toml.

    Attributes:
        enabled: The enabled value.
        persist_directory: The persist directory value.
        include_globs: Whether to include globs.
        exclude_globs: The exclude globs value.
        chunk_size: The chunk size value.
        chunk_overlap: The chunk overlap value.
        top_k: Maximum number of search results to return.
        embedding: The embedding value.
    """

    enabled: bool = False
    persist_directory: Path = Path(DEFAULT_RAG_PERSIST_DIRECTORY)
    include_globs: tuple[str, ...] = DEFAULT_RAG_INCLUDE_GLOBS
    exclude_globs: tuple[str, ...] = DEFAULT_RAG_EXCLUDE_GLOBS
    chunk_size: int = DEFAULT_RAG_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_RAG_CHUNK_OVERLAP
    top_k: int = DEFAULT_RAG_TOP_K
    embedding: RagEmbeddingConfig = RagEmbeddingConfig()


@dataclass(frozen=True)
class ResolvedRagConfig:
    """Store validated RAG settings and derived storage paths.

    Attributes:
        enabled: The enabled value.
        persist_directory: The persist directory value.
        include_globs: Whether to include globs.
        exclude_globs: The exclude globs value.
        chunk_size: The chunk size value.
        chunk_overlap: The chunk overlap value.
        top_k: Maximum number of search results to return.
        embedding: The embedding value.
        collection_name: The collection name value.
    """

    enabled: bool
    persist_directory: Path
    include_globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding: ResolvedRagEmbeddingConfig
    collection_name: str = RAG_COLLECTION_NAME


@dataclass(frozen=True)
class RagStatus:
    """Describe the current availability and freshness of workspace RAG.

    Attributes:
        enabled: The enabled value.
        ready: The ready value.
        file_count: The file count value.
        chunk_count: The chunk count value.
        reason: The reason value.
        persist_directory: The persist directory value.
    """

    enabled: bool
    ready: bool
    file_count: int = 0
    chunk_count: int = 0
    reason: str | None = None
    persist_directory: Path | None = None

    @classmethod
    def disabled(cls) -> "RagStatus":
        """Create a disabled RAG status.

        Returns:
            A disabled RAG status.
        """
        return cls(enabled=False, ready=False)

    @classmethod
    def unavailable(
        cls,
        *,
        reason: str,
        persist_directory: Path | None = None,
    ) -> "RagStatus":
        """Create an unavailable RAG status with a reason.

        Args:
            reason: The reason value.
            persist_directory: The persist directory value.

        Returns:
            An unavailable RAG status containing the supplied reason.
        """
        return cls(
            enabled=True,
            ready=False,
            reason=reason,
            persist_directory=persist_directory,
        )

    @classmethod
    def ready_status(
        cls,
        *,
        file_count: int,
        chunk_count: int,
        persist_directory: Path,
    ) -> "RagStatus":
        """Create a ready RAG status.

        Args:
            file_count: The file count value.
            chunk_count: The chunk count value.
            persist_directory: The persist directory value.

        Returns:
            A ready RAG status containing file and chunk counts.
        """
        return cls(
            enabled=True,
            ready=True,
            file_count=file_count,
            chunk_count=chunk_count,
            persist_directory=persist_directory,
        )


@dataclass(frozen=True)
class UploadedRagFile:
    """Describe a file accepted for thread-scoped RAG ingestion.

    Attributes:
        path: Filesystem path to read or write.
        name: The name value.
    """

    path: Path
    name: str


@dataclass(frozen=True)
class RagUploadResult:
    """Describe the outcome of ingesting uploaded RAG files.

    Attributes:
        thread_id: Conversation thread identifier.
        added_files: The added files value.
        indexed_files: The indexed files value.
        chunk_count: The chunk count value.
        rejected_files: The rejected files value.
        reason: The reason value.
        conflict: Whether an operation rejected a non-fresh target thread.
    """

    thread_id: str
    added_files: tuple[str, ...] = ()
    indexed_files: int = 0
    chunk_count: int = 0
    rejected_files: tuple[str, ...] = ()
    reason: str | None = None
    conflict: bool = False

    @property
    def success(self) -> bool:
        """Return whether uploaded RAG ingestion succeeded.

        Returns:
            True when files were added and no rejection reason is set; otherwise, False.
        """
        return bool(self.added_files) and self.reason is None


def parse_rag_config(raw_config: dict[str, Any], config_path: Path) -> RagConfig:
    """Parse RAG config.

    Args:
        raw_config: Raw config to process.
        config_path: Path to the config.

    Returns:
        The parsed rag config.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    raw_rag = raw_config.get("rag", {})
    if raw_rag and not isinstance(raw_rag, dict):
        raise ValueError("The top-level 'rag' config must be a table/object.")

    base_dir = config_path.parent
    raw_embedding = raw_rag.get("embedding", {})
    if raw_embedding and not isinstance(raw_embedding, dict):
        raise ValueError("The '[rag.embedding]' config must be a table/object.")

    persist_directory_raw = str(
        raw_rag.get("persist_directory", DEFAULT_RAG_PERSIST_DIRECTORY)
    ).strip()
    persist_directory = Path(persist_directory_raw or DEFAULT_RAG_PERSIST_DIRECTORY)
    if not persist_directory.is_absolute():
        persist_directory = (base_dir / persist_directory).resolve()

    chunk_size = normalize_int(
        raw_rag.get("chunk_size"),
        field_name="rag.chunk_size",
        default=DEFAULT_RAG_CHUNK_SIZE,
        minimum=1,
    )
    chunk_overlap = normalize_int(
        raw_rag.get("chunk_overlap"),
        field_name="rag.chunk_overlap",
        default=DEFAULT_RAG_CHUNK_OVERLAP,
        minimum=0,
    )
    if chunk_overlap >= chunk_size:
        raise ValueError("'rag.chunk_overlap' must be smaller than 'rag.chunk_size'.")

    return RagConfig(
        enabled=normalize_bool(raw_rag.get("enabled"), field_name="rag.enabled", default=False),
        persist_directory=persist_directory,
        include_globs=normalize_glob_list(
            raw_rag.get("include_globs"),
            field_name="rag.include_globs",
            default=DEFAULT_RAG_INCLUDE_GLOBS,
        ),
        exclude_globs=normalize_glob_list(
            raw_rag.get("exclude_globs"),
            field_name="rag.exclude_globs",
            default=DEFAULT_RAG_EXCLUDE_GLOBS,
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=normalize_int(
            raw_rag.get("top_k"),
            field_name="rag.top_k",
            default=DEFAULT_RAG_TOP_K,
            minimum=1,
        ),
        embedding=RagEmbeddingConfig(
            provider=normalize_rag_embedding_provider(raw_embedding.get("provider")),
            model=normalize_optional_string(raw_embedding.get("model")),
            base_url=normalize_optional_string(raw_embedding.get("base_url")),
            api_key=normalize_optional_string(raw_embedding.get("api_key")),
        ),
    )


def resolve_rag_config(
    config: RagConfig,
    *,
    model_provider: str,
    model_base_url: str,
) -> ResolvedRagConfig | None:
    """Resolve RAG config.

    Args:
        config: Configuration object used by the operation.
        model_provider: The model provider value.
        model_base_url: URL for the model base.

    Returns:
        The resolved rag config.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if not config.enabled:
        return None

    provider = config.embedding.provider
    resolved_provider: ResolvedRagEmbeddingProvider
    if provider == "auto":
        if model_provider not in {"ollama", "openai_compatible"}:
            raise ValueError(
                "RAG embedding provider 'auto' cannot infer embeddings for "
                f"model provider '{model_provider}'. Set 'rag.embedding.provider' "
                "to 'ollama' or 'openai_compatible' with an embedding model."
            )
        resolved_provider = model_provider
    else:
        resolved_provider = provider

    base_url = (config.embedding.base_url or model_base_url).strip()
    if not base_url:
        raise ValueError("RAG embedding base URL cannot be empty.")

    model = config.embedding.model
    if resolved_provider == "ollama":
        model = (model or DEFAULT_OLLAMA_EMBEDDING_MODEL).strip()
    else:
        model = (model or "").strip()
        if not model:
            raise ValueError(
                "RAG with an OpenAI-compatible embedding provider requires "
                "'rag.embedding.model' to be set."
            )

    return ResolvedRagConfig(
        enabled=True,
        persist_directory=config.persist_directory,
        include_globs=config.include_globs,
        exclude_globs=config.exclude_globs,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        top_k=config.top_k,
        embedding=ResolvedRagEmbeddingConfig(
            provider=resolved_provider,
            model=model,
            base_url=base_url,
            api_key=config.embedding.api_key,
        ),
    )


class SearchWorkspaceKnowledgeInput(BaseModel):
    """Define the schema for workspace knowledge search requests.

    Attributes:
        query: Search query text.
        top_k: Maximum number of search results to return.
    """

    query: str = Field(..., min_length=1, description="Search query for workspace docs.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional number of top matches to return.",
    )


def create_search_workspace_knowledge_tool(
    rag: "WorkspaceDocsRAG",
    *,
    thread_id: str | None = None,
) -> BaseTool:
    """Create the LangChain tool used to search workspace knowledge.

    Args:
        rag: The RAG value.
        thread_id: Conversation thread identifier.

    Returns:
        A LangChain tool bound to the provided RAG service.
    """
    @tool(
        "search_workspace_knowledge",
        args_schema=SearchWorkspaceKnowledgeInput,
        return_direct=False,
    )
    def search_workspace_knowledge(query: str, top_k: int | None = None) -> dict[str, Any]:
        """Search the persisted workspace documentation index for relevant excerpts.

        Args:
            query: Search query text.
            top_k: Maximum number of search results to return.

        Returns:
            Search results matching the query.
        """

        return rag.search(query=query, top_k=top_k, thread_id=thread_id)

    return search_workspace_knowledge


class WorkspaceDocsRAG:
    """Manage workspace and uploaded-document RAG indexes."""

    def __init__(
        self,
        config: ResolvedRagConfig,
        *,
        project_root: Path,
    ) -> None:
        """Initialize the workspace docs r a g instance.

        Args:
            config: Configuration object used by the operation.
            project_root: Project root used to resolve local paths.
        """
        self.config = config
        self.project_root = project_root.resolve()
        self._configured_persist_directory = config.persist_directory.absolute()
        self.persist_directory = self._configured_persist_directory.resolve()
        self.collection_directory = self.persist_directory / VECTOR_STORE_DIRECTORY_NAME
        self.manifest_path = self.persist_directory / "manifest.json"
        self.uploads_root = self.persist_directory / RAG_UPLOADS_DIRECTORY_NAME
        self._lock = threading.RLock()
        self._vectorstore: JsonVectorStore | None = None
        self._uploaded_vectorstores: dict[str, JsonVectorStore] = {}
        self._status = RagStatus.unavailable(
            reason="Knowledge index has not been initialized yet.",
            persist_directory=self.persist_directory,
        )

    def snapshot(self) -> RagStatus:
        """Return a snapshot of.

        Returns:
            A snapshot of.
        """
        with self._lock:
            return self._status

    def ensure_ready(self) -> RagStatus:
        """Ensure ready.

        Returns:
            The ready object or status.
        """
        with self._lock:
            try:
                if self._manifest_is_current_locked():
                    self._load_vectorstore_locked()
                    self._status = self._status_from_manifest_locked()
                    return self._status
                return self._rebuild_locked()
            except Exception as exc:
                self._vectorstore = None
                self._status = RagStatus.unavailable(
                    reason=str(exc),
                    persist_directory=self.persist_directory,
                )
                return self._status

    def rebuild(self) -> RagStatus:
        """Rebuild the workspace docs r a g.

        Returns:
            The rebuilt object or status.
        """
        with self._lock:
            try:
                return self._rebuild_locked()
            except Exception as exc:
                self._vectorstore = None
                self._status = RagStatus.unavailable(
                    reason=str(exc),
                    persist_directory=self.persist_directory,
                )
                return self._status

    def ingest_uploaded_files(
        self,
        *,
        thread_id: str,
        uploads: list[UploadedRagFile],
    ) -> RagUploadResult:
        """Ingest uploaded files.

        Args:
            thread_id: Conversation thread identifier.
            uploads: Uploaded files supplied by the user.

        Returns:
            The ingest uploaded files result.

        Raises:
            ValueError: If the supplied value is invalid.
        """
        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ValueError("A non-empty thread ID is required for uploaded RAG files.")

        with self._lock:
            added_files, rejected_files = self._store_uploaded_files_locked(
                normalized_thread_id,
                uploads,
            )
            if not added_files:
                return RagUploadResult(
                    thread_id=normalized_thread_id,
                    rejected_files=tuple(rejected_files),
                    reason="No supported text files were uploaded.",
                )

            indexed_files, chunk_count = self._rebuild_thread_uploads_locked(
                normalized_thread_id
            )
            return RagUploadResult(
                thread_id=normalized_thread_id,
                added_files=tuple(added_files),
                indexed_files=indexed_files,
                chunk_count=chunk_count,
                rejected_files=tuple(rejected_files),
            )

    def clone_thread_uploads(
        self,
        *,
        source_thread_id: str,
        target_thread_id: str,
    ) -> RagUploadResult:
        """Clone one thread's stored RAG files into a fresh branch thread."""
        source_id = source_thread_id.strip()
        target_id = target_thread_id.strip()
        if not source_id or not target_id:
            raise ValueError("Source and target thread IDs must be non-empty.")
        if source_id == target_id:
            return RagUploadResult(
                thread_id=target_id,
                reason="Source and target threads are identical.",
            )

        with self._lock:
            self._validate_thread_upload_scope(source_id)
            self._validate_thread_upload_scope(target_id)
            source_directory = self._thread_upload_files_directory(source_id)
            target_directory = self._thread_upload_files_directory(target_id)
            if target_directory.is_dir() and any(target_directory.iterdir()):
                return RagUploadResult(
                    thread_id=target_id,
                    reason="Target thread already has uploaded files.",
                    conflict=True,
                )
            if not source_directory.is_dir():
                return RagUploadResult(
                    thread_id=target_id,
                    reason="Source thread has no uploaded files.",
                )

            source_files = self._thread_upload_source_paths(source_id)
            if not source_files:
                return RagUploadResult(
                    thread_id=target_id,
                    reason="Source thread has no supported uploaded files.",
                )

            target_directory.mkdir(parents=True, exist_ok=True)
            for source_path in source_files:
                destination = target_directory / source_path.name
                self._assert_safe_upload_storage_path(destination)
                shutil.copy2(source_path, destination)
            indexed_files, chunk_count = self._rebuild_thread_uploads_locked(target_id)
            return RagUploadResult(
                thread_id=target_id,
                added_files=tuple(path.name for path in source_files),
                indexed_files=indexed_files,
                chunk_count=chunk_count,
            )

    def search(
        self,
        *,
        query: str,
        top_k: int | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Search the workspace docs r a g.

        Args:
            query: Search query text.
            top_k: Maximum number of search results to return.
            thread_id: Conversation thread identifier.

        Returns:
            Search results matching the query.

        Raises:
            RuntimeError: If the runtime is not in a usable state.
        """
        normalized_query = query.strip()
        if not normalized_query:
            return {"query": "", "results": []}

        with self._lock:
            limit = top_k or self.config.top_k
            matches: list[tuple[Document, float]] = []
            has_any_store = False

            status = self.ensure_ready()
            if status.ready:
                has_any_store = True
                matches.extend(
                    self._search_store_locked(
                        self._load_vectorstore_locked(),
                        normalized_query,
                        limit=limit,
                    )
                )

            normalized_thread_id = (thread_id or "").strip()
            if normalized_thread_id:
                upload_matches = self._search_thread_uploads_locked(
                    normalized_thread_id,
                    normalized_query,
                    limit=limit,
                )
                if upload_matches:
                    has_any_store = True
                    matches.extend(upload_matches)

            if not has_any_store:
                raise RuntimeError(status.reason or "Knowledge index is unavailable.")

        results = [
            {
                "path": str(doc.metadata.get("path", "")).strip(),
                "excerpt": self._excerpt(doc.page_content),
                "score": float(score),
            }
            for doc, score in sorted(matches, key=lambda item: item[1], reverse=True)[:limit]
        ]
        return {
            "query": normalized_query,
            "results": results,
        }

    def _rebuild_locked(self) -> RagStatus:
        """Rebuild locked.

        Returns:
            The rebuilt object or status.
        """
        source_paths = self.discover_source_paths()
        documents = self._load_documents(
            source_paths,
            display_path_for=self._relative_path,
            loader=self._read_workspace_document,
        )
        chunks = self._split_documents(documents)

        shutil.rmtree(self.collection_directory, ignore_errors=True)
        self.manifest_path.unlink(missing_ok=True)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        embeddings = self._build_embeddings()
        self._vectorstore = JsonVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.collection_directory,
        )

        manifest = self._build_manifest(
            source_paths,
            file_count=len(documents),
            chunk_count=len(chunks),
            display_path_for=self._relative_path,
        )
        self._write_manifest_locked(manifest)
        self._status = RagStatus.ready_status(
            file_count=len(documents),
            chunk_count=len(chunks),
            persist_directory=self.persist_directory,
        )
        return self._status

    def discover_source_paths(self) -> list[Path]:
        """Discover source paths.

        Returns:
            The discovered paths or configuration values.
        """
        discovered: dict[str, Path] = {}
        for pattern in self.config.include_globs:
            normalized_pattern = pattern.lstrip("/")
            for candidate in self.project_root.glob(normalized_pattern):
                if not candidate.is_file():
                    continue
                relative = self._relative_path(candidate)
                if self._is_excluded(relative):
                    continue
                discovered[relative] = candidate.resolve()
        return [discovered[path] for path in sorted(discovered)]

    def _load_documents(
        self,
        source_paths: list[Path],
        *,
        display_path_for,
        loader,
    ) -> list[Document]:
        """Load documents.

        Args:
            source_paths: Paths to the source.
            display_path_for: The display path for value.
            loader: The loader value.

        Returns:
            The loaded value.
        """
        documents: list[Document] = []
        for path in source_paths:
            documents.append(
                Document(
                    page_content=loader(path),
                    metadata={
                        "path": display_path_for(path),
                    },
                )
            )
        return documents

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into configured text chunks.

        Args:
            documents: The documents value.

        Returns:
            Documents split into chunks for indexing.
        """
        if not documents:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return splitter.split_documents(documents)

    def _manifest_is_current_locked(self) -> bool:
        """Return whether the workspace manifest matches current source files.

        Returns:
            True when the manifest is current; otherwise, False.
        """
        manifest = self._read_manifest_locked()
        if manifest is None:
            return False
        if not JsonVectorStore.index_exists(self.collection_directory):
            return False
        current_signature = self._signature_for_paths(
            self.discover_source_paths(),
            display_path_for=self._relative_path,
        )
        return manifest.get("signature") == current_signature

    def _status_from_manifest_locked(self) -> RagStatus:
        """Extract status from manifest locked.

        Returns:
            The extracted status from manifest locked.
        """
        manifest = self._read_manifest_locked() or {}
        return RagStatus.ready_status(
            file_count=int(manifest.get("file_count", 0)),
            chunk_count=int(manifest.get("chunk_count", 0)),
            persist_directory=self.persist_directory,
        )

    def _load_vectorstore_locked(self) -> JsonVectorStore:
        """Load vectorstore locked.

        Returns:
            The loaded value.
        """
        if self._vectorstore is not None:
            return self._vectorstore
        self._vectorstore = JsonVectorStore.load(
            embedding_function=self._build_embeddings(),
            persist_directory=self.collection_directory,
        )
        return self._vectorstore

    def _build_embeddings(self) -> Any:
        """Build embeddings.

        Returns:
            The constructed embeddings.
        """
        if self.config.embedding.provider == "ollama":
            return OllamaEmbeddings(
                model=self.config.embedding.model,
                base_url=self.config.embedding.base_url,
            )

        return OpenAIEmbeddings(
            model=self.config.embedding.model,
            deployment=self.config.embedding.model,
            base_url=self.config.embedding.base_url,
            api_key=self.config.embedding.api_key or "deepagent",
            tiktoken_enabled=False,
        )

    def _build_manifest(
        self,
        source_paths: list[Path],
        *,
        file_count: int,
        chunk_count: int,
        display_path_for,
    ) -> dict[str, Any]:
        """Build manifest.

        Args:
            source_paths: Paths to the source.
            file_count: The file count value.
            chunk_count: The chunk count value.
            display_path_for: The display path for value.

        Returns:
            The constructed manifest.
        """
        return {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": chunk_count,
            "file_count": file_count,
            "signature": self._signature_for_paths(
                source_paths,
                display_path_for=display_path_for,
            ),
            "version": RAG_MANIFEST_VERSION,
        }

    def _signature_for_paths(
        self,
        source_paths: list[Path],
        *,
        display_path_for,
    ) -> dict[str, Any]:
        """Build a content signature for indexed paths.

        Args:
            source_paths: Paths to the source.
            display_path_for: The display path for value.

        Returns:
            A manifest signature for the supplied paths.
        """
        return {
            "chunk_overlap": self.config.chunk_overlap,
            "chunk_size": self.config.chunk_size,
            "embedding": {
                "base_url": self.config.embedding.base_url,
                "model": self.config.embedding.model,
                "provider": self.config.embedding.provider,
            },
            "exclude_globs": list(self.config.exclude_globs),
            "files": [
                {
                    "mtime_ns": path.stat().st_mtime_ns,
                    "path": display_path_for(path),
                    "size": path.stat().st_size,
                }
                for path in source_paths
            ],
            "include_globs": list(self.config.include_globs),
            "version": RAG_MANIFEST_VERSION,
        }

    def _read_manifest_locked(self) -> dict[str, Any] | None:
        """Read manifest locked.

        Returns:
            The read manifest locked result.
        """
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            raw_manifest = json.load(handle)
        if not isinstance(raw_manifest, dict):
            return None
        return raw_manifest

    def _write_manifest_locked(self, manifest: dict[str, Any]) -> None:
        """Write manifest locked.

        Args:
            manifest: The manifest value.
        """
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    def _read_workspace_document(self, path: Path) -> str:
        """Read workspace document.

        Args:
            path: Filesystem path to read or write.

        Returns:
            The read workspace document result.
        """
        return path.read_text(encoding="utf-8")

    def _read_uploaded_document(self, path: Path) -> str:
        """Read uploaded document.

        Args:
            path: Filesystem path to read or write.

        Returns:
            The read uploaded document result.

        Raises:
            ValueError: If the supplied value is invalid.
        """
        raw = path.read_bytes()
        if b"\x00" in raw:
            raise ValueError(f"Uploaded file '{path.name}' is not a supported text document.")

        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode uploaded file '{path.name}' as text.")

    def _store_uploaded_files_locked(
        self,
        thread_id: str,
        uploads: list[UploadedRagFile],
    ) -> tuple[list[str], list[str]]:
        """Store uploaded files locked.

        Args:
            thread_id: Conversation thread identifier.
            uploads: Uploaded files supplied by the user.

        Returns:
            The stored value.
        """
        self._validate_thread_upload_scope(thread_id)
        files_directory = self._thread_upload_files_directory(thread_id)
        files_directory.mkdir(parents=True, exist_ok=True)

        added_files: list[str] = []
        rejected_files: list[str] = []
        for upload in uploads:
            upload_name = Path(upload.name).name
            if not self._supports_uploaded_file(upload_name):
                rejected_files.append(upload_name)
                continue

            destination = self._unique_upload_destination(files_directory, upload_name)
            self._assert_safe_upload_storage_path(destination)
            shutil.copy2(upload.path, destination)
            added_files.append(destination.name)

        return added_files, rejected_files

    def _rebuild_thread_uploads_locked(self, thread_id: str) -> tuple[int, int]:
        """Rebuild thread uploads locked.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            The rebuilt object or status.
        """
        self._validate_thread_upload_scope(thread_id)
        files_directory = self._thread_upload_files_directory(thread_id)
        source_paths = self._thread_upload_source_paths(thread_id)
        collection_directory = self._thread_upload_collection_directory(thread_id)
        manifest_path = self._thread_upload_manifest_path(thread_id)

        shutil.rmtree(collection_directory, ignore_errors=True)
        manifest_path.unlink(missing_ok=True)
        self._uploaded_vectorstores.pop(thread_id, None)

        if not source_paths:
            return 0, 0

        documents = self._load_documents(
            source_paths,
            display_path_for=self._uploaded_display_path,
            loader=self._read_uploaded_document,
        )
        chunks = self._split_documents(documents)

        embeddings = self._build_embeddings()
        store = JsonVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=collection_directory,
        )

        self._uploaded_vectorstores[thread_id] = store
        manifest = self._build_manifest(
            source_paths,
            file_count=len(documents),
            chunk_count=len(chunks),
            display_path_for=self._uploaded_display_path,
        )
        self._write_json(manifest_path, manifest)
        return len(documents), len(chunks)

    def _search_thread_uploads_locked(
        self,
        thread_id: str,
        query: str,
        *,
        limit: int,
    ) -> list[tuple[Document, float]]:
        """Search thread uploads locked.

        Args:
            thread_id: Conversation thread identifier.
            query: Search query text.
            limit: The limit value.

        Returns:
            Search results matching the query.
        """
        if not self._thread_has_uploaded_files(thread_id):
            return []

        if self._thread_upload_manifest_is_current_locked(thread_id):
            store = self._load_thread_upload_store_locked(thread_id)
        else:
            _, chunk_count = self._rebuild_thread_uploads_locked(thread_id)
            if chunk_count == 0:
                return []
            store = self._load_thread_upload_store_locked(thread_id)

        return self._search_store_locked(store, query, limit=limit)

    def _search_store_locked(
        self,
        store: JsonVectorStore,
        query: str,
        *,
        limit: int,
    ) -> list[tuple[Document, float]]:
        """Search store locked.

        Args:
            store: The store value.
            query: Search query text.
            limit: The limit value.

        Returns:
            Search results matching the query.
        """
        return store.similarity_search_with_relevance_scores(query, k=limit)

    def _thread_has_uploaded_files(self, thread_id: str) -> bool:
        """Return whether the thread has uploaded RAG files.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            True when the thread has uploaded files; otherwise, False.
        """
        self._validate_thread_upload_scope(thread_id)
        return bool(self._thread_upload_source_paths(thread_id))

    def _thread_upload_manifest_is_current_locked(self, thread_id: str) -> bool:
        """Return whether the thread upload manifest matches uploaded files.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            True when the upload manifest is current; otherwise, False.
        """
        self._validate_thread_upload_scope(thread_id)
        manifest = self._read_json(self._thread_upload_manifest_path(thread_id))
        if manifest is None:
            return False

        collection_directory = self._thread_upload_collection_directory(thread_id)
        if not JsonVectorStore.index_exists(collection_directory):
            return False

        files_directory = self._thread_upload_files_directory(thread_id)
        source_paths = self._thread_upload_source_paths(thread_id)
        current_signature = self._signature_for_paths(
            source_paths,
            display_path_for=self._uploaded_display_path,
        )
        return manifest.get("signature") == current_signature

    def _load_thread_upload_store_locked(self, thread_id: str) -> JsonVectorStore:
        """Load thread upload store locked.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            The loaded value.
        """
        self._validate_thread_upload_scope(thread_id)
        store = self._uploaded_vectorstores.get(thread_id)
        if store is not None:
            return store

        store = JsonVectorStore.load(
            embedding_function=self._build_embeddings(),
            persist_directory=self._thread_upload_collection_directory(thread_id),
        )
        self._uploaded_vectorstores[thread_id] = store
        return store

    def _thread_upload_root(self, thread_id: str) -> Path:
        """Return the thread upload root.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            The thread upload root.
        """
        normalized_thread_id = self._normalize_thread_id(thread_id)
        directory_name = hashlib.sha256(normalized_thread_id.encode("utf-8")).hexdigest()
        return self._assert_safe_upload_storage_path(self.uploads_root / directory_name)

    def _thread_upload_files_directory(self, thread_id: str) -> Path:
        """Return the thread upload files directory.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            The thread upload files directory.
        """
        return self._assert_safe_upload_storage_path(
            self._thread_upload_root(thread_id) / RAG_UPLOAD_FILES_DIRECTORY_NAME
        )

    def _thread_upload_collection_directory(self, thread_id: str) -> Path:
        """Return the thread upload collection directory.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            The thread upload collection directory.
        """
        return self._assert_safe_upload_storage_path(
            self._thread_upload_root(thread_id) / RAG_UPLOAD_COLLECTION_DIRECTORY_NAME
        )

    def _thread_upload_manifest_path(self, thread_id: str) -> Path:
        """Return the thread upload manifest path.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            The thread upload manifest path.
        """
        return self._assert_safe_upload_storage_path(
            self._thread_upload_root(thread_id) / RAG_UPLOAD_MANIFEST_FILENAME
        )

    def _normalize_thread_id(self, thread_id: str) -> str:
        """Normalize thread ID.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            The normalized value.

        Raises:
            ValueError: If the supplied value is empty.
        """
        normalized = thread_id.strip()
        if not normalized:
            raise ValueError("A non-empty thread ID is required for uploaded RAG files.")
        return normalized

    def _validate_thread_upload_scope(self, thread_id: str) -> None:
        """Reject unsafe links anywhere in a thread's fixed storage scope."""
        thread_root = self._thread_upload_root(thread_id)
        files_directory = self._thread_upload_files_directory(thread_id)
        collection_directory = self._thread_upload_collection_directory(thread_id)
        fixed_paths = (
            thread_root,
            files_directory,
            collection_directory,
            collection_directory / VECTOR_STORE_INDEX_FILENAME,
            self._thread_upload_manifest_path(thread_id),
        )
        for path in fixed_paths:
            self._assert_safe_upload_storage_path(path)

        if files_directory.is_dir():
            for path in files_directory.iterdir():
                self._assert_safe_upload_storage_path(path)

    def _thread_upload_source_paths(self, thread_id: str) -> list[Path]:
        """Return supported regular files after validating the upload scope."""
        files_directory = self._thread_upload_files_directory(thread_id)
        if not files_directory.exists():
            return []

        source_paths: list[Path] = []
        for path in sorted(files_directory.iterdir(), key=lambda item: item.name):
            self._assert_safe_upload_storage_path(path)
            if path.is_file() and self._supports_uploaded_file(path.name):
                source_paths.append(path)
        return source_paths

    def _assert_safe_upload_storage_path(self, path: Path) -> Path:
        """Return a contained upload path after rejecting existing symlinks."""
        if self._configured_persist_directory.is_symlink():
            raise ValueError(
                "RAG upload storage cannot contain symlinks: "
                f"{self._configured_persist_directory}"
            )

        try:
            relative_path = path.relative_to(self.uploads_root)
        except ValueError as exc:
            raise ValueError("RAG upload storage must remain inside its upload root.") from exc

        current = self.uploads_root
        for part in relative_path.parts:
            if current.is_symlink():
                raise ValueError(f"RAG upload storage cannot contain symlinks: {current}")
            current = current / part
        if current.is_symlink():
            raise ValueError(f"RAG upload storage cannot contain symlinks: {current}")

        resolved_root = self.uploads_root.resolve(strict=False)
        resolved_path = path.resolve(strict=False)
        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError("RAG upload storage must remain inside its upload root.")
        return path

    def _uploaded_display_path(self, path: Path) -> str:
        """Return the uploaded display path.

        Args:
            path: Filesystem path to read or write.

        Returns:
            The uploaded display path.
        """
        return f"uploaded/{path.name}"

    def _supports_uploaded_file(self, name: str) -> bool:
        """Return whether an uploaded file name has a supported extension.

        Args:
            name: The name value.

        Returns:
            True when the file extension can be indexed; otherwise, False.
        """
        return Path(name).suffix.lower() in ALLOWED_RAG_UPLOAD_EXTENSIONS

    def _unique_upload_destination(self, directory: Path, upload_name: str) -> Path:
        """Return an unused destination path for an uploaded file.

        Args:
            directory: The directory value.
            upload_name: The upload name value.

        Returns:
            A non-conflicting destination path in the upload directory.
        """
        original = Path(upload_name)
        stem = original.stem or "upload"
        suffix = original.suffix
        candidate = directory / f"{stem}{suffix}"
        counter = 2
        while candidate.exists():
            candidate = directory / f"{stem}-{counter}{suffix}"
            counter += 1
        return candidate

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        """Read JSON.

        Args:
            path: Filesystem path to read or write.

        Returns:
            The read JSON result.
        """
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            raw_value = json.load(handle)
        if not isinstance(raw_value, dict):
            return None
        return raw_value

    def _write_json(self, path: Path, value: dict[str, Any]) -> None:
        """Write JSON.

        Args:
            path: Filesystem path to read or write.
            value: Value to normalize, convert, or serialize.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)

    def _is_excluded(self, relative_path: str) -> bool:
        """Return whether is excluded.

        Args:
            relative_path: Path to the relative.

        Returns:
            Whether is excluded.
        """
        candidate = PurePosixPath(relative_path)
        for pattern in self.config.exclude_globs:
            if candidate.match(pattern.lstrip("/")):
                return True
        return False

    def _relative_path(self, path: Path) -> str:
        """Return the relative path.

        Args:
            path: Filesystem path to read or write.

        Returns:
            The relative path.
        """
        return path.resolve().relative_to(self.project_root).as_posix()

    @staticmethod
    def _excerpt(text: str, *, limit: int = 280) -> str:
        """Return a compact excerpt of document text.

        Args:
            text: Text content to process.
            limit: The limit value.

        Returns:
            Collapsed text limited to the requested length.
        """
        collapsed = " ".join(text.split())
        if len(collapsed) <= limit:
            return collapsed
        return f"{collapsed[: limit - 3].rstrip()}..."
