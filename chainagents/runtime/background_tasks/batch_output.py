"""Formatting and persistent-file storage for subagent batch results."""

from __future__ import annotations

import asyncio
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from deepagents.backends import BackendProtocol

from chainagents.runtime.background_tasks.models import BackgroundTaskSnapshot
from chainagents.runtime.background_tasks.queues import await_preserving_cancellation


def _format_batch_markdown_section(
    snapshot: BackgroundTaskSnapshot,
    *,
    title: str,
    title_level: int,
    blank_after_title: bool = False,
) -> str:
    """Render one self-describing batch result Markdown section."""
    body_heading = "Error" if snapshot.error else "Report"
    body = snapshot.error or snapshot.result or "_No report returned._"
    title_prefix = "#" * title_level
    body_prefix = "#" * (title_level + 1)
    title_lines = [f"{title_prefix} {title}"]
    if blank_after_title:
        title_lines.append("")
    return "\n".join(
        [
            *title_lines,
            f"- Task ID: `{snapshot.task_id}`",
            f"- Status: `{snapshot.status}`",
            "",
            f"{body_prefix} Request",
            snapshot.description,
            "",
            f"{body_prefix} {body_heading}",
            body,
        ]
    )


def _format_batch_markdown(
    snapshots: Sequence[BackgroundTaskSnapshot],
) -> str:
    """Render terminal batch snapshots as readable, pageable Markdown."""
    sections = [
        _format_batch_markdown_section(
            snapshot,
            title=f"{index}. {snapshot.agent_name}",
            title_level=2,
        )
        for index, snapshot in enumerate(snapshots, start=1)
    ]
    return "# Subagent batch results\n\n" + "\n\n---\n\n".join(sections)


def _format_batch_json(
    snapshots: Sequence[BackgroundTaskSnapshot],
) -> dict[str, object]:
    """Restore the original structured batch result payload."""
    return {
        "results": [snapshot.to_payload() for snapshot in snapshots],
    }


@dataclass(frozen=True)
class BatchResultOutputStore:
    """Backend route and public path mapping for persistent batch reports."""

    backend: BackendProtocol
    backend_prefix: str
    public_prefix: str = "/workspace/.files/outputs/"

    def backend_path(self, relative_path: str) -> str:
        """Return a backend-routed path for an output-relative path."""
        return f"{self.backend_prefix.rstrip('/')}/{relative_path.lstrip('/')}"

    def public_path(self, relative_path: str) -> str:
        """Return the virtual workspace path exposed to the user."""
        return f"{self.public_prefix.rstrip('/')}/{relative_path.lstrip('/')}"


def create_batch_result_output_store(
    backend: BackendProtocol,
    *,
    backend_prefix: str,
) -> BatchResultOutputStore:
    """Create persistent batch-result storage over an existing agent backend."""
    return BatchResultOutputStore(
        backend=backend,
        backend_prefix=backend_prefix,
    )


def _safe_batch_path_component(value: str, *, fallback: str) -> str:
    """Normalize an untrusted identifier into one bounded path component."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("._-")[:80]
    normalized = normalized.strip("._-")
    return normalized or fallback


def _batch_file_markdown(snapshot: BackgroundTaskSnapshot) -> str:
    """Render one standalone batch report file."""
    return _format_batch_markdown_section(
        snapshot,
        title=snapshot.agent_name,
        title_level=1,
        blank_after_title=True,
    )


def _is_missing_delete_error(error: str) -> bool:
    normalized = error.casefold()
    return any(
        marker in normalized
        for marker in ("not found", "does not exist", "no such file")
    )


async def _write_batch_markdown_files(
    snapshots: Sequence[BackgroundTaskSnapshot],
    *,
    tool_call_id: str,
    store: BatchResultOutputStore,
) -> dict[str, object]:
    """Write one persistent Markdown file per snapshot with atomic visibility."""
    call_component = _safe_batch_path_component(tool_call_id, fallback="batch")
    directory = f"subagent-batches/{call_component}-{uuid.uuid4().hex}"
    directory_backend_path = store.backend_path(directory)
    width = max(2, len(str(len(snapshots))))
    attempted_paths: list[str] = []
    manifest: list[dict[str, object]] = []

    async def write_one(backend_path: str, content: str) -> None:
        attempted_paths.append(backend_path)
        result = await store.backend.awrite(backend_path, content)
        if result.error:
            raise RuntimeError(
                f"Failed to write batch result '{backend_path}': {result.error}"
            )

    async def rollback() -> list[BaseException]:
        errors: list[BaseException] = []
        for backend_path in [*reversed(attempted_paths), directory_backend_path]:
            try:
                result = await store.backend.adelete(backend_path)
            except BaseException as exc:
                errors.append(exc)
                continue
            if result.error and not _is_missing_delete_error(result.error):
                errors.append(
                    RuntimeError(
                        f"Failed to clean up batch result '{backend_path}': "
                        f"{result.error}"
                    )
                )
        return errors

    try:
        for index, snapshot in enumerate(snapshots, start=1):
            agent_component = _safe_batch_path_component(
                snapshot.agent_name,
                fallback="agent",
            )
            task_component = _safe_batch_path_component(
                snapshot.task_id,
                fallback="task",
            )
            relative_path = (
                f"{directory}/{index:0{width}d}-{agent_component}-"
                f"{task_component}.md"
            )
            backend_path = store.backend_path(relative_path)
            write_task = asyncio.create_task(
                write_one(backend_path, _batch_file_markdown(snapshot)),
                name=f"chainagents-write-batch-result-{index}",
            )
            await await_preserving_cancellation(write_task)
            manifest.append(
                {
                    "task_id": snapshot.task_id,
                    "agent_name": snapshot.agent_name,
                    "status": snapshot.status,
                    "path": store.public_path(relative_path),
                }
            )
    except BaseException as primary:
        cleanup_task = asyncio.create_task(
            rollback(),
            name="chainagents-rollback-batch-results",
        )
        try:
            cleanup_errors = await await_preserving_cancellation(cleanup_task)
        except BaseException as cleanup_interruption:
            cleanup_errors = []
            if (
                isinstance(cleanup_interruption, asyncio.CancelledError)
                and cleanup_task.done()
                and not cleanup_task.cancelled()
            ):
                cleanup_errors.extend(cleanup_task.result())
            cleanup_errors.append(cleanup_interruption)
        if cleanup_errors:
            grouped = [primary, *cleanup_errors]
            if any(not isinstance(error, Exception) for error in grouped):
                raise BaseExceptionGroup(
                    "Batch result write and cleanup failed.",
                    grouped,
                ) from primary
            raise ExceptionGroup(
                "Batch result write and cleanup failed.",
                cast(list[Exception], grouped),
            ) from primary
        raise

    return {"files": manifest}
