"""Process-local background execution for configured synchronous subagents."""

from __future__ import annotations

import asyncio
import builtins
import contextvars
import dataclasses
import json
import time
import uuid
import weakref
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast, overload

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from chainagents.events.stream import (
    AgentStreamEvent,
    AgentStreamEventAdapter,
    langgraph_part_from_event_chunk,
)
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
        "run_subagent_batch",
        "list_background_tasks",
        "get_background_task",
        "cancel_background_task",
    }
)
_LANGGRAPH_CHECKPOINTER_KEY = "__pregel_checkpointer"
_LANGGRAPH_RUNTIME_KEY = "__pregel_runtime"
_T = TypeVar("_T")


class BackgroundSessionGeneration:
    """Weakly tracked capability for one open background-task session."""

    __slots__ = ("active", "session_id", "__weakref__")

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.active = True

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
_CURRENT_BACKGROUND_SESSION_GENERATION: contextvars.ContextVar[
    BackgroundSessionGeneration | None
] = contextvars.ContextVar(
    "chainagents_background_session_generation",
    default=None,
)


def current_background_task_id() -> str | None:
    """Return the task ID of the currently executing background subagent."""
    return _CURRENT_BACKGROUND_TASK_ID.get()


def current_background_invocation_path() -> tuple[str, ...]:
    """Return the ownership path of the current configured-agent invocation."""
    return _CURRENT_BACKGROUND_INVOCATION_PATH.get()


def current_background_session_generation() -> BackgroundSessionGeneration | None:
    """Return the lifecycle capability attached to the current graph run."""
    return _CURRENT_BACKGROUND_SESSION_GENERATION.get()


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

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        token = _CURRENT_BACKGROUND_INVOCATION_PATH.set(
            (*current_background_invocation_path(), uuid.uuid4().hex)
        )
        try:
            async for chunk in self.runnable.astream(  # type: ignore[attr-defined]
                input,
                config,
                **kwargs,
            ):
                yield chunk
        finally:
            _CURRENT_BACKGROUND_INVOCATION_PATH.reset(token)


def scope_background_task_invocation(runnable: object) -> Runnable[Any, Any]:
    """Wrap a configured subagent with invocation-local task ownership."""
    return _BackgroundInvocationScopedRunnable(runnable)


class _BackgroundSessionScopedRunnable(Runnable[Any, Any]):
    """Attach a session lifecycle capability to one exported graph run."""

    def __init__(
        self,
        runnable: object,
        manager: "BackgroundTaskManager",
    ) -> None:
        self.runnable = runnable
        self.manager = manager

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runnable, name)

    @property
    def InputType(self) -> Any:  # noqa: N802
        return self.runnable.InputType  # type: ignore[attr-defined]

    @property
    def OutputType(self) -> Any:  # noqa: N802
        return self.runnable.OutputType  # type: ignore[attr-defined]

    @property
    def config_specs(self) -> list[Any]:
        return list(self.runnable.config_specs)  # type: ignore[attr-defined]

    def get_input_schema(self, config: RunnableConfig | None = None) -> Any:
        return self.runnable.get_input_schema(config)  # type: ignore[attr-defined]

    def get_output_schema(self, config: RunnableConfig | None = None) -> Any:
        return self.runnable.get_output_schema(config)  # type: ignore[attr-defined]

    def get_graph(self, config: RunnableConfig | None = None) -> Any:
        return self.runnable.get_graph(config)  # type: ignore[attr-defined]

    def _set_generation(
        self,
        config: RunnableConfig | None,
    ) -> contextvars.Token[BackgroundSessionGeneration | None] | None:
        configurable = (config or {}).get("configurable", {})
        session_id = str(configurable.get("thread_id") or "").strip()
        if not session_id:
            return None
        return _CURRENT_BACKGROUND_SESSION_GENERATION.set(
            self.manager.session_generation(session_id)
        )

    @staticmethod
    def _reset_generation(
        token: contextvars.Token[BackgroundSessionGeneration | None] | None,
    ) -> None:
        if token is not None:
            _CURRENT_BACKGROUND_SESSION_GENERATION.reset(token)

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        token = self._set_generation(config)
        try:
            return self.runnable.invoke(input, config, **kwargs)  # type: ignore[attr-defined]
        finally:
            self._reset_generation(token)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        token = self._set_generation(config)
        try:
            return await self.runnable.ainvoke(  # type: ignore[attr-defined]
                input,
                config,
                **kwargs,
            )
        finally:
            self._reset_generation(token)

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        token = self._set_generation(config)
        try:
            yield from self.runnable.stream(  # type: ignore[attr-defined]
                input,
                config,
                **kwargs,
            )
        finally:
            self._reset_generation(token)

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        token = self._set_generation(config)
        try:
            async for chunk in self.runnable.astream(  # type: ignore[attr-defined]
                input,
                config,
                **kwargs,
            ):
                yield chunk
        finally:
            self._reset_generation(token)

    @overload
    def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]: ...

    @overload
    def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v3"],
        **kwargs: Any,
    ) -> Awaitable[Any]: ...

    def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2", "v3"] = "v2",
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any] | Awaitable[Any]:
        configurable = (config or {}).get("configurable", {})
        session_id = str(configurable.get("thread_id") or "").strip()
        generation = (
            self.manager.session_generation(session_id)
            if session_id
            else None
        )
        if version == "v3":
            result = self.runnable.astream_events(  # type: ignore[attr-defined]
                input,
                config,
                version=version,
                **kwargs,
            )

            async def await_events() -> Any:
                token = _CURRENT_BACKGROUND_SESSION_GENERATION.set(generation)
                try:
                    stream = await cast(Awaitable[Any], result)
                    graph_iterator = getattr(stream, "_graph_aiter", None)
                    if graph_iterator is not None:
                        stream._graph_aiter = (  # noqa: SLF001
                            _BackgroundSessionScopedAsyncIterator(
                                graph_iterator,
                                generation,
                            )
                        )
                    return stream
                finally:
                    _CURRENT_BACKGROUND_SESSION_GENERATION.reset(token)

            return await_events()

        result = self.runnable.astream_events(  # type: ignore[attr-defined]
            input,
            config,
            version=version,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_names=exclude_names,
            exclude_types=exclude_types,
            exclude_tags=exclude_tags,
            **kwargs,
        )

        async def iterate_events() -> AsyncIterator[Any]:
            token = _CURRENT_BACKGROUND_SESSION_GENERATION.set(generation)
            try:
                async for event in cast(AsyncIterator[Any], result):
                    yield event
            finally:
                _CURRENT_BACKGROUND_SESSION_GENERATION.reset(token)

        return iterate_events()


class _BackgroundSessionScopedAsyncIterator:
    """Activate a session capability for each lazy v3 graph pull."""

    def __init__(
        self,
        iterator: AsyncIterator[Any],
        generation: BackgroundSessionGeneration | None,
    ) -> None:
        self.iterator = iterator
        self.generation = generation

    def __aiter__(self) -> "_BackgroundSessionScopedAsyncIterator":
        return self

    async def __anext__(self) -> Any:
        token = _CURRENT_BACKGROUND_SESSION_GENERATION.set(self.generation)
        try:
            return await self.iterator.__anext__()
        finally:
            _CURRENT_BACKGROUND_SESSION_GENERATION.reset(token)

    async def aclose(self) -> None:
        close = getattr(self.iterator, "aclose", None)
        if close is None:
            return
        token = _CURRENT_BACKGROUND_SESSION_GENERATION.set(self.generation)
        try:
            await close()
        finally:
            _CURRENT_BACKGROUND_SESSION_GENERATION.reset(token)


def scope_background_session_invocation(
    runnable: object,
    manager: "BackgroundTaskManager",
) -> Runnable[Any, Any]:
    """Wrap an exported graph with invocation-scoped session invalidation."""
    return _BackgroundSessionScopedRunnable(runnable, manager)


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
    session_generation: BackgroundSessionGeneration | None = None,
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

    def expected_generation() -> BackgroundSessionGeneration | None:
        return session_generation or current_background_session_generation()

    def build_submission(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
        session_id: str,
    ) -> _BackgroundTaskSubmission:
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
            if manager.config.stream_activity:
                adapter = AgentStreamEventAdapter(prompt=description)
                final_values: object = None
                async for chunk in child.astream(  # type: ignore[attr-defined]
                    state,
                    config,
                    stream_mode=["values", "messages", "updates"],
                    subgraphs=True,
                ):
                    part = langgraph_part_from_event_chunk(chunk)
                    if part is None:
                        continue
                    namespace = tuple(part.get("ns", ()))
                    if part.get("type") == "values" and not namespace:
                        final_values = part.get("data")
                    for event in adapter.events_from_part(part):
                        source = (
                            subagent_type
                            if not namespace
                            else f"{subagent_type} / {event.source}"
                        )
                        await manager.publish_activity(
                            task_id,
                            dataclasses.replace(event, source=source),
                        )
                return _result_text(final_values)
            result = await child.ainvoke(state, config)  # type: ignore[attr-defined]
            return _result_text(result)

        cleanup: BackgroundCleanup | None = None
        if callable(delete_checkpoint_thread):
            async def cleanup_child(task_id: str) -> None:
                await delete_checkpoint_thread(
                    f"{session_id}:background:{task_id}"
                )

            cleanup = cleanup_child

        return _BackgroundTaskSubmission(
            agent_name=subagent_type,
            description=description,
            agent_path=(*agent_path, subagent_type),
            runner=run_child,
            cleanup=cleanup,
        )

    @tool("spawn_background_task")
    async def spawn_background_task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> dict[str, object]:
        """Launch an allowed local subagent and return its task ID immediately."""
        session_id = _session_id_from_runtime(runtime)
        submission = build_submission(
            description,
            subagent_type,
            runtime,
            session_id,
        )

        snapshot = await manager.spawn(
            session_id=session_id,
            agent_name=submission.agent_name,
            description=submission.description,
            agent_path=submission.agent_path,
            owner_path=current_background_invocation_path(),
            parent_task_id=current_background_task_id(),
            expected_session_generation=expected_generation(),
            runner=submission.runner,
            cleanup=submission.cleanup,
        )
        return snapshot.to_payload()

    @tool("run_subagent_batch")
    async def run_subagent_batch(
        tasks: list[BackgroundSubagentBatchRequest],
        runtime: ToolRuntime,
    ) -> dict[str, object]:
        """Run independent child tasks concurrently and return all reports in order."""
        if not tasks:
            raise ValueError("A subagent batch requires at least one task.")
        session_id = _session_id_from_runtime(runtime)
        normalized = [
            task
            if isinstance(task, BackgroundSubagentBatchRequest)
            else BackgroundSubagentBatchRequest.model_validate(task)
            for task in tasks
        ]
        for task in normalized:
            if not task.description.strip():
                raise ValueError("Each subagent batch description must be non-empty.")
        submissions = [
            build_submission(
                task.description,
                task.subagent_type,
                runtime,
                session_id,
            )
            for task in normalized
        ]
        owner_path = current_background_invocation_path()
        parent_task_id = current_background_task_id()
        generation = expected_generation()
        snapshots = await manager.spawn_batch(
            session_id=session_id,
            submissions=submissions,
            owner_path=owner_path,
            parent_task_id=parent_task_id,
            expected_session_generation=generation,
        )
        task_ids = [snapshot.task_id for snapshot in snapshots]
        try:
            completed = await manager.wait_batch(
                session_id,
                task_ids,
                scope_path=agent_path,
                owner_path=owner_path,
                ancestor_task_id=parent_task_id,
                expected_session_generation=generation,
            )
        except asyncio.CancelledError:
            async def cancel_batch() -> None:
                await asyncio.gather(
                    *(
                        manager.cancel(
                            session_id,
                            task_id,
                            scope_path=agent_path,
                            owner_path=owner_path,
                            ancestor_task_id=parent_task_id,
                            only_if_unfinished=True,
                        )
                        for task_id in task_ids
                    ),
                    return_exceptions=True,
                )

            cancellation = asyncio.create_task(
                cancel_batch(),
                name=f"chainagents-cancel-batch-{runtime.tool_call_id}",
            )
            await await_preserving_cancellation(cancellation)
            raise
        return {"results": [snapshot.to_payload() for snapshot in completed]}

    @tool("list_background_tasks")
    async def list_background_tasks(runtime: ToolRuntime) -> list[dict[str, object]]:
        """List local background tasks visible to the current agent."""
        session_id = _session_id_from_runtime(runtime)
        snapshots = await manager.list(
            session_id,
            scope_path=agent_path,
            owner_path=current_background_invocation_path(),
            ancestor_task_id=current_background_task_id(),
            expected_session_generation=expected_generation(),
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
            expected_session_generation=expected_generation(),
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
            expected_session_generation=expected_generation(),
        )
        return snapshot.to_payload()

    return [
        spawn_background_task,
        run_subagent_batch,
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
        self._activity_subscribers: dict[
            str, set[asyncio.Queue[BackgroundTaskActivity]]
        ] = {}
        self._closing_sessions: set[str] = set()
        self._session_close_locks: dict[str, asyncio.Lock] = {}
        self._session_close_users: dict[str, int] = {}
        self._session_generations: weakref.WeakValueDictionary[
            str, BackgroundSessionGeneration
        ] = weakref.WeakValueDictionary()
        self._close_task: asyncio.Task[None] | None = None
        self._closed = False

    def session_generation(self, session_id: str) -> BackgroundSessionGeneration:
        """Return the current lifecycle generation for one session."""
        normalized_session = session_id.strip()
        if not normalized_session:
            raise ValueError("Background tasks require a non-empty session ID.")
        if self._closed:
            raise RuntimeError("The background task manager is closed.")
        if normalized_session in self._closing_sessions:
            raise RuntimeError("The background task session is closing.")
        generation = self._session_generations.get(normalized_session)
        if generation is None:
            generation = BackgroundSessionGeneration(normalized_session)
            self._session_generations[normalized_session] = generation
        return generation

    def _validate_session_generation(
        self,
        session_id: str,
        expected: BackgroundSessionGeneration | None,
    ) -> None:
        if expected is None:
            return
        if (
            not expected.active
            or expected.session_id != session_id
            or self._session_generations.get(session_id) is not expected
        ):
            raise RuntimeError(
                "The background task session was closed; start a new foreground run."
            )

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
        expected_session_generation: BackgroundSessionGeneration | None = None,
        cleanup: BackgroundCleanup | None = None,
    ) -> BackgroundTaskSnapshot:
        """Start a job immediately and return before the runner finishes."""
        snapshots = await self.spawn_batch(
            session_id=session_id,
            submissions=[
                _BackgroundTaskSubmission(
                    agent_name=agent_name,
                    description=description,
                    agent_path=agent_path,
                    runner=runner,
                    cleanup=cleanup,
                )
            ],
            owner_path=owner_path,
            parent_task_id=parent_task_id,
            expected_session_generation=expected_session_generation,
        )
        return snapshots[0]

    async def spawn_batch(
        self,
        *,
        session_id: str,
        submissions: Sequence[_BackgroundTaskSubmission],
        owner_path: tuple[str, ...] = (),
        parent_task_id: str | None = None,
        expected_session_generation: BackgroundSessionGeneration | None = None,
    ) -> builtins.list[BackgroundTaskSnapshot]:
        """Atomically admit and start an ordered batch of background jobs."""
        normalized_session = session_id.strip()
        if not normalized_session:
            raise ValueError("Background tasks require a non-empty session ID.")
        if not self.config.enabled:
            raise RuntimeError("Local background subagents are disabled.")
        if not submissions:
            raise ValueError("Background task batches require at least one submission.")

        async with self._lock:
            if self._closed:
                raise RuntimeError("The background task manager is closed.")
            if normalized_session in self._closing_sessions:
                raise RuntimeError("The background task session is closing.")
            self._validate_session_generation(
                normalized_session,
                expected_session_generation,
            )
            session_ids = self._session_task_ids.get(normalized_session, [])
            running_session = sum(
                self._records[task_id].status not in TERMINAL_BACKGROUND_TASK_STATUSES
                for task_id in session_ids
            )
            batch_size = len(submissions)
            if running_session + batch_size > self.config.max_running_per_session:
                raise RuntimeError(
                    "Background running limit reached for this session "
                    f"({self.config.max_running_per_session})."
                )
            running_total = sum(
                record.status not in TERMINAL_BACKGROUND_TASK_STATUSES
                for record in self._records.values()
            )
            if running_total + batch_size > self.config.max_running_total:
                raise RuntimeError(
                    "Background running limit reached for this process "
                    f"({self.config.max_running_total})."
                )
            if len(session_ids) + batch_size > self.config.max_tasks_per_session:
                raise RuntimeError(
                    "Background retained task limit reached for this session "
                    f"({self.config.max_tasks_per_session})."
                )
            if parent_task_id is not None:
                parent = self._records.get(parent_task_id)
                if parent is None or parent.session_id != normalized_session:
                    raise ValueError("Background parent task does not exist in this session.")
                if parent.cancelling:
                    raise RuntimeError("Background parent task is cancelling.")

            retained_ids = self._session_task_ids.setdefault(normalized_session, [])
            records: list[_BackgroundTaskRecord] = []
            for submission in submissions:
                task_id = f"bg-{uuid.uuid4().hex[:12]}"
                record = _BackgroundTaskRecord(
                    task_id=task_id,
                    session_id=normalized_session,
                    agent_name=submission.agent_name,
                    description=submission.description,
                    agent_path=submission.agent_path,
                    owner_path=owner_path,
                    session_generation=expected_session_generation,
                    parent_task_id=parent_task_id,
                    status="pending",
                    created_at=time.time(),
                    completion=asyncio.Event(),
                    cleanup=submission.cleanup,
                )
                self._records[task_id] = record
                retained_ids.append(task_id)
                records.append(record)
            for record, submission in zip(records, submissions, strict=True):
                record.execution = asyncio.create_task(
                    self._execute(record, submission.runner),
                    name=f"chainagents-{record.task_id}",
                    context=contextvars.Context(),
                )
            return [record.snapshot() for record in records]

    async def _execute(
        self,
        record: _BackgroundTaskRecord,
        runner: BackgroundRunner,
    ) -> None:
        task_token = _CURRENT_BACKGROUND_TASK_ID.set(record.task_id)
        session_token = _CURRENT_BACKGROUND_SESSION_ID.set(record.session_id)
        owner_token = _CURRENT_BACKGROUND_INVOCATION_PATH.set(record.owner_path)
        generation_token = _CURRENT_BACKGROUND_SESSION_GENERATION.set(
            record.session_generation
        )
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
            if record.cancelling:
                await self._cancel_descendants(record.session_id, record.task_id)
                await self._finish(
                    record,
                    status="cancelled",
                    error=cleanup_error,
                )
            elif cleanup_error:
                await self._cancel_descendants(record.session_id, record.task_id)
                await self._finish(record, status="error", error=cleanup_error)
            else:
                await self._finish(record, status="success", result=str(result))
        finally:
            _CURRENT_BACKGROUND_SESSION_GENERATION.reset(generation_token)
            _CURRENT_BACKGROUND_INVOCATION_PATH.reset(owner_token)
            _CURRENT_BACKGROUND_SESSION_ID.reset(session_token)
            _CURRENT_BACKGROUND_TASK_ID.reset(task_token)

    async def _cleanup_record(self, record: _BackgroundTaskRecord) -> str | None:
        async with self._lock:
            cleanup = record.cleanup
            cleanup_task = record.cleanup_task
            if cleanup is not None and cleanup_task is None:
                async def run_cleanup() -> None:
                    await cleanup(record.task_id)

                cleanup_task = asyncio.create_task(
                    run_cleanup(),
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
            activity_subscribers = tuple(
                self._activity_subscribers.get(record.session_id, ())
            )
        for completion_queue in subscribers:
            completion_queue.put_nowait(snapshot)
        terminal_activity = BackgroundTaskActivity(
            task_id=record.task_id,
            session_id=record.session_id,
            agent_name=record.agent_name,
            description=record.description,
            snapshot=snapshot,
        )
        for activity_queue in activity_subscribers:
            activity_queue.put_nowait(terminal_activity)

    async def publish_activity(
        self,
        task_id: str,
        event: AgentStreamEvent,
    ) -> None:
        """Publish one live event to activity subscribers for its task session."""
        async with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return
            subscribers = tuple(
                self._activity_subscribers.get(record.session_id, ())
            )
            activity = BackgroundTaskActivity(
                task_id=record.task_id,
                session_id=record.session_id,
                agent_name=record.agent_name,
                description=record.description,
                event=event,
            )
        for queue in subscribers:
            queue.put_nowait(activity)

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
        expected_session_generation: BackgroundSessionGeneration | None = None,
    ) -> builtins.list[BackgroundTaskSnapshot]:
        """List retained tasks visible to an agent scope."""
        async with self._lock:
            self._validate_session_generation(
                session_id,
                expected_session_generation,
            )
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
        expected_session_generation: BackgroundSessionGeneration | None = None,
    ) -> BackgroundTaskSnapshot:
        """Return a visible task, optionally waiting for terminal state."""
        if wait_seconds < 0 or wait_seconds > 60:
            raise ValueError("wait_seconds must be between 0 and 60.")
        async with self._lock:
            self._validate_session_generation(
                session_id,
                expected_session_generation,
            )
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
                self._validate_session_generation(
                    session_id,
                    expected_session_generation,
                )
                record = self._visible_record(
                    session_id,
                    task_id,
                    scope_path,
                    owner_path,
                    ancestor_task_id,
                )
                snapshot = record.snapshot()
        return snapshot

    async def wait_batch(
        self,
        session_id: str,
        task_ids: Sequence[str],
        *,
        scope_path: tuple[str, ...] = (),
        owner_path: tuple[str, ...] = (),
        ancestor_task_id: str | None = None,
        expected_session_generation: BackgroundSessionGeneration | None = None,
    ) -> builtins.list[BackgroundTaskSnapshot]:
        """Wait for an exact visible batch and return snapshots in input order."""
        async with self._lock:
            self._validate_session_generation(
                session_id,
                expected_session_generation,
            )
            records = [
                self._visible_record(
                    session_id,
                    task_id,
                    scope_path,
                    owner_path,
                    ancestor_task_id,
                )
                for task_id in task_ids
            ]
            completions = [
                record.completion
                for record in records
                if record.status not in TERMINAL_BACKGROUND_TASK_STATUSES
                and record.completion is not None
            ]
        if completions:
            await asyncio.shield(
                asyncio.gather(*(completion.wait() for completion in completions))
            )
        async with self._lock:
            return [record.snapshot() for record in records]

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
        expected_session_generation: BackgroundSessionGeneration | None = None,
        only_if_unfinished: bool = False,
    ) -> BackgroundTaskSnapshot:
        """Cancel a visible task and all of its descendants."""
        async with self._lock:
            self._validate_session_generation(
                session_id,
                expected_session_generation,
            )
            record = self._visible_record(
                session_id,
                task_id,
                scope_path,
                owner_path,
                ancestor_task_id,
            )
            if (
                only_if_unfinished
                and record.status in TERMINAL_BACKGROUND_TASK_STATUSES
            ):
                return record.snapshot()
            record.cancelling = True
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
            for record in descendants:
                record.cancelling = True
        for execution in executions:
            execution.cancel()
        if executions:
            await asyncio.gather(*executions, return_exceptions=True)
        for record in descendants:
            if record.status not in TERMINAL_BACKGROUND_TASK_STATUSES:
                cleanup_error = await self._cleanup_record(record)
                await self._finish(record, status="cancelled", error=cleanup_error)

    async def wait_session(
        self,
        session_id: str,
    ) -> builtins.list[BackgroundTaskSnapshot]:
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
            for record in records:
                if record.status not in TERMINAL_BACKGROUND_TASK_STATUSES:
                    record.cancelling = True
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
            generation = self._session_generations.pop(normalized_session, None)
            if generation is not None:
                generation.active = False
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

    def subscribe_activity(
        self,
        session_id: str,
    ) -> asyncio.Queue[BackgroundTaskActivity]:
        """Subscribe to ordered live and terminal task activity."""
        queue: asyncio.Queue[BackgroundTaskActivity] = asyncio.Queue()
        self._activity_subscribers.setdefault(session_id, set()).add(queue)
        return queue

    def unsubscribe_activity(
        self,
        session_id: str,
        queue: asyncio.Queue[BackgroundTaskActivity],
    ) -> None:
        """Remove a previously registered task activity subscriber."""
        subscribers = self._activity_subscribers.get(session_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._activity_subscribers.pop(session_id, None)

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

    async def _close_all_sessions(
        self,
        session_ids: builtins.list[str],
    ) -> None:
        """Close the manager's sessions and release completion subscribers."""
        await asyncio.gather(
            *(self.close_session(session_id) for session_id in session_ids),
            return_exceptions=False,
        )
        self._subscribers.clear()
        self._activity_subscribers.clear()
