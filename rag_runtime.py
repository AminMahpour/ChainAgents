from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from langchain_chroma import Chroma
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

RAG_SYSTEM_PROMPT_SUFFIX = """

Knowledge retrieval:
- When the user asks about project documentation, instructions, prompts, or skill docs, use `search_workspace_knowledge` first.
- Cite retrieved sources using the relative file paths returned by the tool.
- If the tool returns no useful matches, say so briefly and fall back to browsing `/workspace/` directly.
""".rstrip()


def compose_rag_system_prompt(base_prompt: str, *, rag_enabled: bool) -> str:
    if not rag_enabled:
        return base_prompt
    return f"{base_prompt.rstrip()}\n{RAG_SYSTEM_PROMPT_SUFFIX}"


def normalize_rag_embedding_provider(
    value: Any | None,
    *,
    default: RagEmbeddingProvider = "auto",
) -> RagEmbeddingProvider:
    candidate = str(value or default).strip().lower().replace("-", "_")
    if not candidate:
        return default
    if candidate not in {"auto", "ollama", "openai_compatible"}:
        raise ValueError(
            "RAG embedding provider must be 'auto', 'ollama', or 'openai_compatible'."
        )
    return candidate  # type: ignore[return-value]


def normalize_bool(value: Any | None, *, field_name: str, default: bool) -> bool:
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
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError(f"'{field_name}' must be an array of glob strings.")
    globs = tuple(str(item).strip() for item in value if str(item).strip())
    if not globs:
        raise ValueError(f"'{field_name}' must include at least one glob.")
    return globs


def normalize_optional_string(value: Any | None) -> str | None:
    candidate = str(value or "").strip()
    return candidate or None


@dataclass(frozen=True)
class RagEmbeddingConfig:
    provider: RagEmbeddingProvider = "auto"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


@dataclass(frozen=True)
class ResolvedRagEmbeddingConfig:
    provider: ResolvedRagEmbeddingProvider
    model: str
    base_url: str
    api_key: str | None = None


@dataclass(frozen=True)
class RagConfig:
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
    enabled: bool
    ready: bool
    file_count: int = 0
    chunk_count: int = 0
    reason: str | None = None
    persist_directory: Path | None = None

    @classmethod
    def disabled(cls) -> "RagStatus":
        return cls(enabled=False, ready=False)

    @classmethod
    def unavailable(
        cls,
        *,
        reason: str,
        persist_directory: Path | None = None,
    ) -> "RagStatus":
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
        return cls(
            enabled=True,
            ready=True,
            file_count=file_count,
            chunk_count=chunk_count,
            persist_directory=persist_directory,
        )


def parse_rag_config(raw_config: dict[str, Any], config_path: Path) -> RagConfig:
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
    model_provider: ResolvedRagEmbeddingProvider,
    model_base_url: str,
) -> ResolvedRagConfig | None:
    if not config.enabled:
        return None

    provider = config.embedding.provider
    resolved_provider: ResolvedRagEmbeddingProvider
    if provider == "auto":
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
    query: str = Field(..., min_length=1, description="Search query for workspace docs.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional number of top matches to return.",
    )


def create_search_workspace_knowledge_tool(rag: "WorkspaceDocsRAG") -> BaseTool:
    @tool(
        "search_workspace_knowledge",
        args_schema=SearchWorkspaceKnowledgeInput,
        return_direct=False,
    )
    def search_workspace_knowledge(query: str, top_k: int | None = None) -> dict[str, Any]:
        """Search the persisted workspace documentation index for relevant excerpts."""

        return rag.search(query=query, top_k=top_k)

    return search_workspace_knowledge


class WorkspaceDocsRAG:
    def __init__(
        self,
        config: ResolvedRagConfig,
        *,
        project_root: Path,
    ) -> None:
        self.config = config
        self.project_root = project_root.resolve()
        self.persist_directory = config.persist_directory.resolve()
        self.collection_directory = self.persist_directory / "chroma"
        self.manifest_path = self.persist_directory / "manifest.json"
        self._lock = threading.RLock()
        self._vectorstore: Chroma | None = None
        self._status = RagStatus.unavailable(
            reason="Knowledge index has not been initialized yet.",
            persist_directory=self.persist_directory,
        )

    def snapshot(self) -> RagStatus:
        with self._lock:
            return self._status

    def ensure_ready(self) -> RagStatus:
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

    def search(self, *, query: str, top_k: int | None = None) -> dict[str, Any]:
        normalized_query = query.strip()
        if not normalized_query:
            return {"query": "", "results": []}

        with self._lock:
            status = self.ensure_ready()
            if not status.ready:
                raise RuntimeError(status.reason or "Knowledge index is unavailable.")

            store = self._load_vectorstore_locked()
            matches = store.similarity_search_with_relevance_scores(
                normalized_query,
                k=top_k or self.config.top_k,
            )

        results = [
            {
                "path": str(doc.metadata.get("path", "")).strip(),
                "excerpt": self._excerpt(doc.page_content),
                "score": float(score),
            }
            for doc, score in matches
        ]
        return {
            "query": normalized_query,
            "results": results,
        }

    def _rebuild_locked(self) -> RagStatus:
        source_paths = self.discover_source_paths()
        documents = self._load_source_documents(source_paths)
        chunks = self._split_documents(documents)

        shutil.rmtree(self.persist_directory, ignore_errors=True)
        self.collection_directory.mkdir(parents=True, exist_ok=True)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        embeddings = self._build_embeddings()
        if chunks:
            self._vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=self.config.collection_name,
                persist_directory=str(self.collection_directory),
            )
        else:
            self._vectorstore = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.collection_directory),
            )

        manifest = self._build_manifest(
            source_paths,
            file_count=len(documents),
            chunk_count=len(chunks),
        )
        self._write_manifest_locked(manifest)
        self._status = RagStatus.ready_status(
            file_count=len(documents),
            chunk_count=len(chunks),
            persist_directory=self.persist_directory,
        )
        return self._status

    def discover_source_paths(self) -> list[Path]:
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

    def _load_source_documents(self, source_paths: list[Path]) -> list[Document]:
        documents: list[Document] = []
        for path in source_paths:
            documents.append(
                Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={
                        "path": self._relative_path(path),
                    },
                )
            )
        return documents

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return splitter.split_documents(documents)

    def _manifest_is_current_locked(self) -> bool:
        manifest = self._read_manifest_locked()
        if manifest is None:
            return False
        if not self.collection_directory.exists():
            return False
        current_signature = self._signature_for_paths(self.discover_source_paths())
        return manifest.get("signature") == current_signature

    def _status_from_manifest_locked(self) -> RagStatus:
        manifest = self._read_manifest_locked() or {}
        return RagStatus.ready_status(
            file_count=int(manifest.get("file_count", 0)),
            chunk_count=int(manifest.get("chunk_count", 0)),
            persist_directory=self.persist_directory,
        )

    def _load_vectorstore_locked(self) -> Chroma:
        if self._vectorstore is not None:
            return self._vectorstore
        self._vectorstore = Chroma(
            collection_name=self.config.collection_name,
            embedding_function=self._build_embeddings(),
            persist_directory=str(self.collection_directory),
        )
        return self._vectorstore

    def _build_embeddings(self) -> Any:
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
    ) -> dict[str, Any]:
        return {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": chunk_count,
            "file_count": file_count,
            "signature": self._signature_for_paths(source_paths),
            "version": RAG_MANIFEST_VERSION,
        }

    def _signature_for_paths(self, source_paths: list[Path]) -> dict[str, Any]:
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
                    "path": self._relative_path(path),
                    "size": path.stat().st_size,
                }
                for path in source_paths
            ],
            "include_globs": list(self.config.include_globs),
            "version": RAG_MANIFEST_VERSION,
        }

    def _read_manifest_locked(self) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            raw_manifest = json.load(handle)
        if not isinstance(raw_manifest, dict):
            return None
        return raw_manifest

    def _write_manifest_locked(self, manifest: dict[str, Any]) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    def _is_excluded(self, relative_path: str) -> bool:
        candidate = PurePosixPath(relative_path)
        for pattern in self.config.exclude_globs:
            if candidate.match(pattern.lstrip("/")):
                return True
        return False

    def _relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.project_root).as_posix()

    @staticmethod
    def _excerpt(text: str, *, limit: int = 280) -> str:
        collapsed = " ".join(text.split())
        if len(collapsed) <= limit:
            return collapsed
        return f"{collapsed[: limit - 3].rstrip()}..."
