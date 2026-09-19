"""Process-local background execution for configured synchronous subagents."""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.runtime import Runtime

from chainagents.runtime.types import BackgroundSubagentConfig


BackgroundTaskStatus = Literal[
    "pending",
    "running",
    "success",
    "error",
    "cancelled",
]
TERMINAL_BACKGROUND_TASK_STATUSES = frozenset({"success", "error", "cancelled"})
BACKGROUND_TASK_TOOL_NAMES = frozenset(
    {
        "spawn_background_task",
        "list_background_tasks",
        "get_background_task",
        "cancel_background_task",
    }
)
_LANGGRAPH_CHECKPOINTER_KEY = "__pregel_checkpointer"
_LANGGRAPH_RUNTIME_KEY = "__pregel_runtime"
_T = TypeVar("_T")

_CURRENT_BACKGROUND_TASK_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "chainagents_background_task_id",
    default=None,
)
_CURRENT_BACKGROUND_SESSION_ID: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar(
        "chainagents_background_session_id",
        default=None,
    )
)
_CURRENT_BACKGROUND_INVOCATION_PATH: contextvars.ContextVar[tuple[str, ...]] = (
    contextvars.ContextVar(
        "chainagents_background_invocation_path",
        default=(),
    )
)


def current_background_task_id() -> str | None:
    """Return the task ID of the currently executing background subagent."""
    return _CURRENT_BACKGROUND_TASK_ID.get()


def current_background_invocation_path() -> tuple[str, ...]:
    """Return the ownership path of the current configured-agent invocation."""
    return _CURRENT_BACKGROUND_INVOCATION_PATH.get()


async def await_preserving_cancellation(task: asyncio.Task[_T]) -> _T:
    """Delay caller cancellation until a lifecycle task has finished."""
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc
    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result


class _BackgroundInvocationScopedRunnable(Runnable[Any, Any]):
    """Give each configured subagent invocation a distinct task-tree scope."""

    def __init__(self, runnable: object) -> None:
        self.runnable = runnable

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        token = _CURRENT_BACKGROUND_INVOCATION_PATH.set(
            (*current_background_invocation_path(), uuid.uuid4().hex)
        )
        try:
            return self.runnable.invoke(input, config, **kwargs)  # type: ignore[attr-defined]
        finally:
            _CURRENT_BACKGROUND_INVOCATION_PATH.reset(token)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        token = _CURRENT_BACKGROUND_INVOCATION_PATH.set(
            (*current_background_invocation_path(), uuid.uuid4().hex)
        )
        try:
            return await self.runnable.ainvoke(  # type: ignore[attr-defined]
                input,
                config,
                **kwargs,
            )
        finally:
            _CURRENT_BACKGROUND_INVOCATION_PATH.reset(token)


def scope_background_task_invocation(runnable: object) -> Runnable[Any, Any]:
    """Wrap a configured subagent with invocation-local task ownership."""
    return _BackgroundInvocationScopedRunnable(runnable)


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


@dataclass
class _BackgroundTaskRecord:
    task_id: str
    session_id: str
    agent_name: str
    description: str
    agent_path: tuple[str, ...]
    owner_path: tuple[str, ...]
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


BackgroundRunner = Callable[[str], Awaitable[str]]
BackgroundCleanup = Callable[[str], Awaitable[None]]


def _session_id_from_runtime(runtime: ToolRuntime) -> str:
    configurable = runtime.config.get("configurable", {})
    session_id = str(
        _CURRENT_BACKGROUND_SESSION_ID.get()
        or configurable.get("thread_id")
        or ""
    ).strip()
    if not session_id:
        raise ValueError(
            "Local background tasks require an explicit conversation identity."
        )
    return session_id


def _result_text(result: object) -> str:
    if not isinstance(result, dict):
        raise ValueError("Background subagent returned an unsupported result.")
    structured = result.get("structured_response")
    if structured is not None:
        if hasattr(structured, "model_dump_json"):
            return str(structured.model_dump_json())
        if dataclasses.is_dataclass(structured) and not isinstance(structured, type):
            return json.dumps(dataclasses.asdict(structured))
        return json.dumps(structured)
    messages = result.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Background subagent result does not contain messages.")
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.text:
            text = message.text.rstrip()
            if text:
                return text
    return ""


def create_background_task_tools(
    *,
    manager: "BackgroundTaskManager",
    subagents: dict[str, object],
    agent_path: tuple[str, ...],
    recursion_limit: int,
    existing_tools: Iterable[object] = (),
) -> list[object]:
    """Create task tools scoped to the direct children of one agent."""
    collisions = sorted(
        {
            name
            for tool in existing_tools
            if (
                name := str(
                    getattr(tool, "name", None)
                    or getattr(tool, "__name__", "")
                ).strip()
            )
            in BACKGROUND_TASK_TOOL_NAMES
        }
    )
    if collisions:
        names = ", ".join(collisions)
        raise ValueError(
            "Configured tools use reserved background task tool names: "
            f"{names}. Rename or prefix the configured tools."
        )
    allowed_names = tuple(sorted(subagents))
    allowed_text = ", ".join(allowed_names) or "none"

    @tool("spawn_background_task")
    async def spawn_background_task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> dict[str, object]:
        """Launch an allowed local subagent and return its task ID immediately."""
        session_id = _session_id_from_runtime(runtime)
        child = subagents.get(subagent_type)
        if child is None:
            raise ValueError(
                f"Allowed subagents for background work: {allowed_text}."
            )
        parent_configurable = runtime.config.get("configurable", {})
        shared_checkpointer = parent_configurable.get(_LANGGRAPH_CHECKPOINTER_KEY)
        shared_store = runtime.store
        delete_checkpoint_thread = getattr(
            shared_checkpointer,
            "adelete_thread",
            None,
        )

        async def run_child(task_id: str) -> str:
            state = {"messages": [HumanMessage(content=description)]}
            configurable: dict[str, object] = {
                "thread_id": f"{session_id}:background:{task_id}",
                "checkpoint_ns": "",
                "ls_agent_type": "background_subagent",
            }
            if shared_checkpointer is not None:
                configurable[_LANGGRAPH_CHECKPOINTER_KEY] = shared_checkpointer
            if shared_store is not None:
                configurable[_LANGGRAPH_RUNTIME_KEY] = Runtime(store=shared_store)
            config = {
                "configurable": configurable,
                "recursion_limit": recursion_limit,
            }
            result = await child.ainvoke(state, config)  # type: ignore[attr-defined]
            return _result_text(result)

        async def cleanup_child(task_id: str) -> None:
            await delete_checkpoint_thread(
                f"{session_id}:background:{task_id}"
            )

        snapshot = await manager.spawn(
            session_id=session_id,
            agent_name=subagent_type,
            description=description,
            agent_path=(*agent_path, subagent_type),
            owner_path=current_background_invocation_path(),
            parent_task_id=current_background_task_id(),
            runner=run_child,
            cleanup=cleanup_child if callable(delete_checkpoint_thread) else None,
        )
        return snapshot.to_payload()

    @tool("list_background_tasks")
    async def list_background_tasks(runtime: ToolRuntime) -> list[dict[str, object]]:
        """List local background tasks visible to the current agent."""
        session_id = _session_id_from_runtime(runtime)
        snapshots = await manager.list(
            session_id,
            scope_path=agent_path,
            owner_path=current_background_invocation_path(),
            ancestor_task_id=current_background_task_id(),
        )
        return [snapshot.to_payload() for snapshot in snapshots]

    @tool("get_background_task")
    async def get_background_task(
        task_id: str,
        wait_seconds: float = 0,
        runtime: ToolRuntime = None,  # type: ignore[assignment]
    ) -> dict[str, object]:
        """Get a visible background task, optionally waiting up to 60 seconds."""
        session_id = _session_id_from_runtime(runtime)
        snapshot = await manager.get(
            session_id,
            task_id,
            scope_path=agent_path,
            owner_path=current_background_invocation_path(),
            ancestor_task_id=current_background_task_id(),
            wait_seconds=wait_seconds,
        )
        return snapshot.to_payload()

    @tool("cancel_background_task")
    async def cancel_background_task(
        task_id: str,
        runtime: ToolRuntime,
    ) -> dict[str, object]:
        """Cancel a visible background task and all of its descendants."""
        session_id = _session_id_from_runtime(runtime)
        snapshot = await manager.cancel(
            session_id,
            task_id,
            scope_path=agent_path,
            owner_path=current_background_invocation_path(),
            ancestor_task_id=current_background_task_id(),
        )
        return snapshot.to_payload()

    return [
        spawn_background_task,
        list_background_tasks,
        get_background_task,
        cancel_background_task,
    ]


class BackgroundTaskManager:
    """Own process-local background subagent jobs for all conversations."""

    def __init__(self, config: BackgroundSubagentConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._records: dict[str, _BackgroundTaskRecord] = {}
        self._session_task_ids: dict[str, list[str]] = {}
        self._subscribers: dict[
            str, set[asyncio.Queue[BackgroundTaskSnapshot]]
        ] = {}
        self._closing_sessions: set[str] = set()
        self._session_close_locks: dict[str, asyncio.Lock] = {}
        self._session_close_users: dict[str, int] = {}
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    async def spawn(
        self,
        *,
        session_id: str,
        agent_name: str,
        description: str,
        agent_path: tuple[str, ...],
        owner_path: tuple[str, ...] = (),
        runner: BackgroundRunner,
        parent_task_id: str | None = None,
        cleanup: BackgroundCleanup | None = None,
    ) -> BackgroundTaskSnapshot:
        """Start a job immediately and return before the runner finishes."""
        normalized_session = session_id.strip()
        if not normalized_session:
            raise ValueError("Background tasks require a non-empty session ID.")
        if not self.config.enabled:
            raise RuntimeError("Local background subagents are disabled.")

        async with self._lock:
            if self._closed:
                raise RuntimeError("The background task manager is closed.")
            if normalized_session in self._closing_sessions:
                raise RuntimeError("The background task session is closing.")
            session_ids = self._session_task_ids.setdefault(normalized_session, [])
            running_session = sum(
                self._records[task_id].status not in TERMINAL_BACKGROUND_TASK_STATUSES
                for task_id in session_ids
            )
            if running_session >= self.config.max_running_per_session:
                raise RuntimeError(
                    "Background running limit reached for this session "
                    f"({self.config.max_running_per_session})."
                )
            running_total = sum(
                record.status not in TERMINAL_BACKGROUND_TASK_STATUSES
                for record in self._records.values()
            )
            if running_total >= self.config.max_running_total:
                raise RuntimeError(
                    "Background running limit reached for this process "
                    f"({self.config.max_running_total})."
                )
            if len(session_ids) >= self.config.max_tasks_per_session:
                raise RuntimeError(
                    "Background retained task limit reached for this session "
                    f"({self.config.max_tasks_per_session})."
                )
            if parent_task_id is not None:
                parent = self._records.get(parent_task_id)
                if parent is None or parent.session_id != normalized_session:
                    raise ValueError("Background parent task does not exist in this session.")

            task_id = f"bg-{uuid.uuid4().hex[:12]}"
            record = _BackgroundTaskRecord(
                task_id=task_id,
                session_id=normalized_session,
                agent_name=agent_name,
                description=description,
                agent_path=agent_path,
                owner_path=owner_path,
                parent_task_id=parent_task_id,
                status="pending",
                created_at=time.time(),
                completion=asyncio.Event(),
                cleanup=cleanup,
            )
            self._records[task_id] = record
            session_ids.append(task_id)
            context = contextvars.Context()
            record.execution = asyncio.create_task(
                self._execute(record, runner),
                name=f"chainagents-{task_id}",
                context=context,
            )
            return record.snapshot()

    async def _execute(
        self,
        record: _BackgroundTaskRecord,
        runner: BackgroundRunner,
    ) -> None:
        task_token = _CURRENT_BACKGROUND_TASK_ID.set(record.task_id)
        session_token = _CURRENT_BACKGROUND_SESSION_ID.set(record.session_id)
        owner_token = _CURRENT_BACKGROUND_INVOCATION_PATH.set(record.owner_path)
        try:
            async with self._lock:
                if record.status == "pending":
                    record.status = "running"
                elif record.status in TERMINAL_BACKGROUND_TASK_STATUSES:
                    return
            result = await runner(record.task_id)
        except asyncio.CancelledError:
            cleanup_error = await self._cleanup_record(record)
            await self._cancel_descendants(record.session_id, record.task_id)
            await self._finish(record, status="cancelled", error=cleanup_error)
            raise
        except Exception as exc:  # noqa: BLE001
            detail = " ".join(str(exc).split()).strip()
            error = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
            cleanup_error = await self._cleanup_record(record)
            if cleanup_error:
                error = f"{error}; {cleanup_error}"
            await self._cancel_descendants(record.session_id, record.task_id)
            await self._finish(record, status="error", error=error)
        else:
            cleanup_error = await self._cleanup_record(record)
            if cleanup_error:
                await self._cancel_descendants(record.session_id, record.task_id)
                await self._finish(record, status="error", error=cleanup_error)
            else:
                await self._finish(record, status="success", result=str(result))
        finally:
            _CURRENT_BACKGROUND_INVOCATION_PATH.reset(owner_token)
            _CURRENT_BACKGROUND_SESSION_ID.reset(session_token)
            _CURRENT_BACKGROUND_TASK_ID.reset(task_token)

    async def _cleanup_record(self, record: _BackgroundTaskRecord) -> str | None:
        async with self._lock:
            cleanup = record.cleanup
            cleanup_task = record.cleanup_task
            if cleanup is not None and cleanup_task is None:
                cleanup_task = asyncio.create_task(
                    cleanup(record.task_id),
                    name=f"chainagents-cleanup-{record.task_id}",
                )
                record.cleanup_task = cleanup_task
        if cleanup is None:
            return None
        assert cleanup_task is not None
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            if cleanup_task.cancelled():
                async with self._lock:
                    if record.cleanup_task is cleanup_task:
                        record.cleanup_task = None
            raise
        except Exception as exc:  # noqa: BLE001
            async with self._lock:
                if record.cleanup_task is cleanup_task:
                    record.cleanup_task = None
            detail = " ".join(str(exc).split()).strip()
            summary = (
                f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
            )
            return f"Background checkpoint cleanup failed: {summary}"
        async with self._lock:
            if record.cleanup_task is cleanup_task:
                record.cleanup_task = None
                record.cleanup = None
        return None

    async def _finish(
        self,
        record: _BackgroundTaskRecord,
        *,
        status: BackgroundTaskStatus,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            if record.status in TERMINAL_BACKGROUND_TASK_STATUSES:
                return
            record.status = status
            record.result = result
            record.error = error
            record.completed_at = time.time()
            if record.completion is not None:
                record.completion.set()
            snapshot = record.snapshot()
            subscribers = tuple(self._subscribers.get(record.session_id, ()))
        for queue in subscribers:
            queue.put_nowait(snapshot)

    def _is_visible(
        self,
        record: _BackgroundTaskRecord,
        scope_path: tuple[str, ...],
        owner_path: tuple[str, ...],
        ancestor_task_id: str | None,
    ) -> bool:
        if scope_path and record.agent_path[: len(scope_path)] != scope_path:
            return False
        if owner_path and record.owner_path[: len(owner_path)] != owner_path:
            return False
        if ancestor_task_id is None:
            return True
        parent_task_id = record.parent_task_id
        visited: set[str] = set()
        while parent_task_id is not None and parent_task_id not in visited:
            if parent_task_id == ancestor_task_id:
                return True
            visited.add(parent_task_id)
            parent = self._records.get(parent_task_id)
            parent_task_id = parent.parent_task_id if parent is not None else None
        return False

    async def list(
        self,
        session_id: str,
        *,
        scope_path: tuple[str, ...] = (),
        owner_path: tuple[str, ...] = (),
        ancestor_task_id: str | None = None,
    ) -> list[BackgroundTaskSnapshot]:
        """List retained tasks visible to an agent scope."""
        async with self._lock:
            return [
                record.snapshot()
                for task_id in self._session_task_ids.get(session_id, ())
                for record in [self._records[task_id]]
                if self._is_visible(
                    record,
                    scope_path,
                    owner_path,
                    ancestor_task_id,
                )
            ]

    async def get(
        self,
        session_id: str,
        task_id: str,
        *,
        scope_path: tuple[str, ...] = (),
        owner_path: tuple[str, ...] = (),
        ancestor_task_id: str | None = None,
        wait_seconds: float = 0,
    ) -> BackgroundTaskSnapshot:
        """Return a visible task, optionally waiting for terminal state."""
        if wait_seconds < 0 or wait_seconds > 60:
            raise ValueError("wait_seconds must be between 0 and 60.")
        async with self._lock:
            record = self._visible_record(
                session_id,
                task_id,
                scope_path,
                owner_path,
                ancestor_task_id,
            )
            completion = record.completion
            snapshot = record.snapshot()
        if (
            wait_seconds
            and snapshot.status not in TERMINAL_BACKGROUND_TASK_STATUSES
            and completion is not None
        ):
            try:
                await asyncio.wait_for(completion.wait(), timeout=wait_seconds)
            except TimeoutError:
                pass
            async with self._lock:
                record = self._visible_record(
                    session_id,
                    task_id,
                    scope_path,
                    owner_path,
                    ancestor_task_id,
                )
                snapshot = record.snapshot()
        return snapshot

    def _visible_record(
        self,
        session_id: str,
        task_id: str,
        scope_path: tuple[str, ...],
        owner_path: tuple[str, ...],
        ancestor_task_id: str | None,
    ) -> _BackgroundTaskRecord:
        record = self._records.get(task_id)
        if (
            record is None
            or record.session_id != session_id
            or not self._is_visible(
                record,
                scope_path,
                owner_path,
                ancestor_task_id,
            )
        ):
            raise KeyError(f"Background task '{task_id}' is not visible in this session.")
        return record

    async def cancel(
        self,
        session_id: str,
        task_id: str,
        *,
        scope_path: tuple[str, ...] = (),
        owner_path: tuple[str, ...] = (),
        ancestor_task_id: str | None = None,
    ) -> BackgroundTaskSnapshot:
        """Cancel a visible task and all of its descendants."""
        async with self._lock:
            record = self._visible_record(
                session_id,
                task_id,
                scope_path,
                owner_path,
                ancestor_task_id,
            )
            if record.cancel_task is None:
                record.cancel_task = asyncio.create_task(
                    self._cancel_record(record),
                    name=f"chainagents-cancel-{record.task_id}",
                )
            cancel_task = record.cancel_task
        return await await_preserving_cancellation(cancel_task)

    async def _cancel_record(
        self,
        record: _BackgroundTaskRecord,
    ) -> BackgroundTaskSnapshot:
        """Cancel and finalize a record independently of its caller."""
        async with self._lock:
            execution = record.execution
        if execution is not None and not execution.done():
            execution.cancel()
        await self._cancel_descendants(record.session_id, record.task_id)
        if execution is not None:
            await asyncio.gather(execution, return_exceptions=True)
        if record.status not in TERMINAL_BACKGROUND_TASK_STATUSES:
            cleanup_error = await self._cleanup_record(record)
            await self._finish(record, status="cancelled", error=cleanup_error)
        async with self._lock:
            return record.snapshot()

    async def _cancel_descendants(self, session_id: str, parent_task_id: str) -> None:
        async with self._lock:
            descendants: list[_BackgroundTaskRecord] = []
            descendant_ids: set[str] = set()
            parent_ids = {parent_task_id}
            while parent_ids:
                children = [
                    record
                    for record in self._records.values()
                    if record.session_id == session_id
                    and record.parent_task_id in parent_ids
                    and record.task_id not in descendant_ids
                ]
                descendants.extend(children)
                descendant_ids.update(record.task_id for record in children)
                parent_ids = {record.task_id for record in children}
            executions = [
                record.execution
                for record in descendants
                if record.execution is not None and not record.execution.done()
            ]
        for execution in executions:
            execution.cancel()
        if executions:
            await asyncio.gather(*executions, return_exceptions=True)
        for record in descendants:
            if record.status not in TERMINAL_BACKGROUND_TASK_STATUSES:
                cleanup_error = await self._cleanup_record(record)
                await self._finish(record, status="cancelled", error=cleanup_error)

    async def wait_session(self, session_id: str) -> list[BackgroundTaskSnapshot]:
        """Wait until every task currently or subsequently running in a session ends."""
        while True:
            async with self._lock:
                executions = [
                    record.execution
                    for task_id in self._session_task_ids.get(session_id, ())
                    for record in [self._records[task_id]]
                    if record.execution is not None and not record.execution.done()
                ]
            if not executions:
                return await self.list(session_id)
            await asyncio.gather(*executions, return_exceptions=True)

    async def _close_session_tasks(self, session_id: str) -> None:
        """Cancel, await, and forget task records for a serialized session."""
        async with self._lock:
            records = [
                self._records[task_id]
                for task_id in self._session_task_ids.get(session_id, ())
                if task_id in self._records
            ]
            executions = [
                record.execution
                for record in records
                if record.execution is not None and not record.execution.done()
            ]
        for execution in executions:
            execution.cancel()
        if executions:
            await asyncio.gather(*executions, return_exceptions=True)
        for record in records:
            if record.status not in TERMINAL_BACKGROUND_TASK_STATUSES:
                cleanup_error = await self._cleanup_record(record)
                await self._finish(record, status="cancelled", error=cleanup_error)
            else:
                await self._cleanup_record(record)
        async with self._lock:
            for record in records:
                self._records.pop(record.task_id, None)
            self._session_task_ids.pop(session_id, None)

    @asynccontextmanager
    async def closing_session(self, session_id: str) -> AsyncIterator[None]:
        """Close tasks and reject new work until dependent resources are released."""
        normalized_session = session_id.strip()
        if not normalized_session:
            raise ValueError("Background tasks require a non-empty session ID.")
        async with self._lock:
            close_lock = self._session_close_locks.setdefault(
                normalized_session,
                asyncio.Lock(),
            )
            self._session_close_users[normalized_session] = (
                self._session_close_users.get(normalized_session, 0) + 1
            )
            self._closing_sessions.add(normalized_session)
        try:
            async with close_lock:
                await self._close_session_tasks(normalized_session)
                yield
        finally:
            async with self._lock:
                users = self._session_close_users[normalized_session] - 1
                if users:
                    self._session_close_users[normalized_session] = users
                else:
                    self._session_close_users.pop(normalized_session, None)
                    self._session_close_locks.pop(normalized_session, None)
                    self._closing_sessions.discard(normalized_session)

    async def close_session(self, session_id: str) -> None:
        """Cancel, await, and forget all work owned by one conversation."""
        close_task = asyncio.create_task(
            self._close_session(session_id),
            name=f"chainagents-close-background-session-{session_id}",
        )
        await await_preserving_cancellation(close_task)

    async def _close_session(self, session_id: str) -> None:
        """Run a complete session close in its own cancellation scope."""
        async with self.closing_session(session_id):
            pass

    def subscribe(self, session_id: str) -> asyncio.Queue[BackgroundTaskSnapshot]:
        """Subscribe to terminal task snapshots for one conversation."""
        queue: asyncio.Queue[BackgroundTaskSnapshot] = asyncio.Queue()
        self._subscribers.setdefault(session_id, set()).add(queue)
        return queue

    def unsubscribe(
        self,
        session_id: str,
        queue: asyncio.Queue[BackgroundTaskSnapshot],
    ) -> None:
        """Remove a previously registered completion subscriber."""
        subscribers = self._subscribers.get(session_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(session_id, None)

    async def close(self) -> None:
        """Cancel all work and prevent future spawns."""
        async with self._lock:
            if self._close_task is None:
                self._closed = True
                session_ids = list(self._session_task_ids)
                self._close_task = asyncio.create_task(
                    self._close_all_sessions(session_ids),
                    name="chainagents-close-background-tasks",
                )
            close_task = self._close_task
        await await_preserving_cancellation(close_task)

    async def _close_all_sessions(self, session_ids: list[str]) -> None:
        """Close the manager's sessions and release completion subscribers."""
        await asyncio.gather(
            *(self.close_session(session_id) for session_id in session_ids),
            return_exceptions=False,
        )
        self._subscribers.clear()
