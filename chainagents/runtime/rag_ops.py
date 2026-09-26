"""RAG index status, rebuild, and thread-scoped upload operations."""

from __future__ import annotations

import asyncio
from typing import Any

from chainagents.rag.runtime import RagStatus, RagUploadResult, UploadedRagFile


def rag_enabled(runtime: Any) -> bool:
    """Return whether the RAG service is available.

    Args:
        runtime: Agent runtime used by the operation.

    Returns:
        Whether the RAG service is available.
    """
    return runtime.config.rag_requested


def rag_status(runtime: Any) -> RagStatus:
    """Return the current RAG service status.

    Args:
        runtime: Agent runtime used by the operation.

    Returns:
        The current RAG service status.
    """
    if runtime._rag_service is not None:
        return runtime._rag_service.snapshot()
    if runtime.config.rag_requested:
        return RagStatus.unavailable(
            reason=runtime.config.rag_error or "Knowledge index is unavailable.",
            persist_directory=(
                runtime.config.rag.persist_directory
                if runtime.config.rag is not None
                else None
            ),
        )
    return RagStatus.disabled()


async def rebuild_rag_index(runtime: Any) -> RagStatus:
    """Rebuild the configured RAG index.

    Args:
        runtime: Agent runtime used by the operation.

    Returns:
        The rebuilt object or status.
    """
    if runtime._rag_service is None:
        if runtime.config.rag_requested:
            return RagStatus.unavailable(
                reason=runtime.config.rag_error or "Knowledge index is unavailable.",
                persist_directory=(
                    runtime.config.rag.persist_directory
                    if runtime.config.rag is not None
                    else None
                ),
            )
        return RagStatus.disabled()

    status = await asyncio.to_thread(runtime._rag_service.rebuild)
    await runtime._clear_agent_cache()
    return status


async def ingest_rag_uploads(
    runtime: Any,
    *,
    thread_id: str,
    uploads: list[UploadedRagFile],
) -> RagUploadResult:
    """Ingest RAG uploads.

    Args:
        runtime: Agent runtime used by the operation.
        thread_id: Conversation thread identifier.
        uploads: Uploaded files supplied by the user.

    Returns:
        The ingest RAG uploads result.
    """
    if runtime._rag_service is None:
        return RagUploadResult(
            thread_id=thread_id,
            reason=runtime.config.rag_error or "Knowledge index is unavailable.",
        )

    return await asyncio.to_thread(
        runtime._rag_service.ingest_uploaded_files,
        thread_id=thread_id,
        uploads=uploads,
    )


async def clone_rag_uploads(
    runtime: Any,
    *,
    source_thread_id: str,
    target_thread_id: str,
) -> RagUploadResult:
    """Clone thread-scoped RAG uploads for a fresh conversation branch.

    Args:
        runtime: Agent runtime used by the operation.
        source_thread_id: Thread identifier to copy uploads from.
        target_thread_id: Thread identifier to copy uploads into.

    Returns:
        The clone RAG uploads result.
    """
    if runtime._rag_service is None:
        return RagUploadResult(
            thread_id=target_thread_id,
            reason=runtime.config.rag_error or "Knowledge index is unavailable.",
        )
    return await asyncio.to_thread(
        runtime._rag_service.clone_thread_uploads,
        source_thread_id=source_thread_id,
        target_thread_id=target_thread_id,
    )
