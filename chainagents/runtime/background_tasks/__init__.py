"""Process-local background execution for configured synchronous subagents."""

from __future__ import annotations

from chainagents.runtime.background_tasks.batch_output import (
    BatchResultOutputStore,
    _write_batch_markdown_files,
    create_batch_result_output_store,
)
from chainagents.runtime.background_tasks.context import (
    BackgroundSessionGeneration,
    current_background_invocation_path,
    current_background_session_generation,
    current_background_session_id,
    current_background_task_id,
)
from chainagents.runtime.background_tasks.manager import BackgroundTaskManager
from chainagents.runtime.background_tasks.models import (
    TERMINAL_BACKGROUND_TASK_STATUSES,
    BackgroundSubagentBatchRequest,
    BackgroundTaskActivity,
    BackgroundTaskSnapshot,
    BackgroundTaskStatus,
    _BackgroundTaskSubmission,
)
from chainagents.runtime.background_tasks.queues import (
    SUBSCRIBER_QUEUE_MAXSIZE,
    await_preserving_cancellation,
)
from chainagents.runtime.background_tasks.scoping import (
    scope_background_session_invocation,
    scope_background_task_invocation,
)
from chainagents.runtime.background_tasks.tools import (
    BACKGROUND_TASK_TOOL_NAMES,
    create_background_task_tools,
)

__all__ = [
    "BACKGROUND_TASK_TOOL_NAMES",
    "TERMINAL_BACKGROUND_TASK_STATUSES",
    "SUBSCRIBER_QUEUE_MAXSIZE",
    "BackgroundSessionGeneration",
    "BackgroundSubagentBatchRequest",
    "BackgroundTaskActivity",
    "BackgroundTaskManager",
    "BackgroundTaskSnapshot",
    "BackgroundTaskStatus",
    "BatchResultOutputStore",
    "create_background_task_tools",
    "create_batch_result_output_store",
    "current_background_invocation_path",
    "current_background_session_generation",
    "current_background_session_id",
    "current_background_task_id",
    "scope_background_session_invocation",
    "scope_background_task_invocation",
    "await_preserving_cancellation",
    "_BackgroundTaskSubmission",
    "_write_batch_markdown_files",
]
