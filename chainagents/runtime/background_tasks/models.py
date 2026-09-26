"""Data models shared across the background task subsystem."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from chainagents.events.stream import AgentStreamEvent
from chainagents.runtime.artifacts import ArtifactSessionHandle
from chainagents.runtime.background_tasks.context import BackgroundSessionGeneration

BackgroundTaskStatus = Literal[
    "pending",
    "running",
    "success",
    "error",
    "cancelled",
]
TERMINAL_BACKGROUND_TASK_STATUSES = frozenset({"success", "error", "cancelled"})


@dataclass(frozen=True)
class BackgroundTaskSnapshot:
    """Immutable public view of one local background task."""

    task_id: str
    session_id: str
    agent_name: str
    description: str
    agent_path: tuple[str, ...]
    parent_task_id: str | None
    status: BackgroundTaskStatus
    result: str | None
    error: str | None
    created_at: float
    completed_at: float | None

    def to_payload(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "description": self.description,
            "agent_path": list(self.agent_path),
            "parent_task_id": self.parent_task_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True)
class BackgroundTaskActivity:
    """One live event or terminal snapshot from a local background task."""

    task_id: str
    session_id: str
    agent_name: str
    description: str
    event: AgentStreamEvent | None = None
    snapshot: BackgroundTaskSnapshot | None = None


class BackgroundSubagentBatchRequest(BaseModel):
    """One independently executable child request in a subagent batch."""

    description: str = Field(
        description=(
            "A complete task description for one subagent. Include all context "
            "the isolated child needs."
        )
    )
    subagent_type: str = Field(
        description="The configured direct child subagent that should run the task."
    )


BackgroundRunner = Callable[[str], Awaitable[str]]
BackgroundCleanup = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class _BackgroundTaskSubmission:
    agent_name: str
    description: str
    agent_path: tuple[str, ...]
    runner: BackgroundRunner
    cleanup: BackgroundCleanup | None = None


@dataclass
class _BackgroundTaskRecord:
    task_id: str
    session_id: str
    agent_name: str
    description: str
    agent_path: tuple[str, ...]
    owner_path: tuple[str, ...]
    session_generation: BackgroundSessionGeneration | None
    artifact_handle: ArtifactSessionHandle | None
    parent_task_id: str | None
    status: BackgroundTaskStatus
    created_at: float
    result: str | None = None
    error: str | None = None
    completed_at: float | None = None
    execution: asyncio.Task[None] | None = None
    completion: asyncio.Event | None = None
    cleanup: BackgroundCleanup | None = None
    cleanup_task: asyncio.Task[None] | None = None
    cancel_task: asyncio.Task[BackgroundTaskSnapshot] | None = None
    cancelling: bool = False
    # Number of callers blocked on this record; pins it against eviction.
    waiters: int = 0

    def snapshot(self) -> BackgroundTaskSnapshot:
        return BackgroundTaskSnapshot(
            task_id=self.task_id,
            session_id=self.session_id,
            agent_name=self.agent_name,
            description=self.description,
            agent_path=self.agent_path,
            parent_task_id=self.parent_task_id,
            status=self.status,
            result=self.result,
            error=self.error,
            created_at=self.created_at,
            completed_at=self.completed_at,
        )
