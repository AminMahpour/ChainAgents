"""Agent-facing tools for spawning and observing local background subagents."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import uuid
from collections.abc import Iterable
from contextlib import nullcontext
from typing import TYPE_CHECKING

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from chainagents.events.stream import (
    AgentStreamEventAdapter,
    langgraph_part_from_event_chunk,
)
from chainagents.runtime.background_tasks.batch_output import (
    BatchResultOutputStore,
    _format_batch_json,
    _format_batch_markdown,
    _write_batch_markdown_files,
)
from chainagents.runtime.background_tasks.context import (
    _CURRENT_BACKGROUND_SESSION_ID,
    BackgroundSessionGeneration,
    current_background_invocation_path,
    current_background_session_generation,
    current_background_task_id,
)
from chainagents.runtime.background_tasks.models import (
    BackgroundCleanup,
    BackgroundSubagentBatchRequest,
    _BackgroundTaskSubmission,
)
from chainagents.runtime.background_tasks.queues import await_preserving_cancellation

if TYPE_CHECKING:
    from chainagents.runtime.background_tasks.manager import BackgroundTaskManager
    from chainagents.runtime.tracing import LangSmithTracing


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


def _session_id_from_runtime(runtime: ToolRuntime) -> str:
    configurable = runtime.config.get("configurable", {})
    session_id = str(
        _CURRENT_BACKGROUND_SESSION_ID.get() or configurable.get("thread_id") or ""
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
    batch_output_store: BatchResultOutputStore | None = None,
    existing_tools: Iterable[object] = (),
    langsmith_tracing: "LangSmithTracing | None" = None,
) -> list[object]:
    """Create task tools scoped to the direct children of one agent."""
    collisions = sorted(
        {
            name
            for tool in existing_tools
            if (
                name := str(
                    getattr(tool, "name", None) or getattr(tool, "__name__", "")
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
            raise ValueError(f"Allowed subagents for background work: {allowed_text}.")
        parent_configurable = runtime.config.get("configurable", {})
        parent_task_id = current_background_task_id()
        parent_trace = (
            langsmith_tracing.capture_parent(runtime.config)
            if langsmith_tracing is not None
            else None
        )
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
            scope = nullcontext()
            if langsmith_tracing is not None:
                mode = langsmith_tracing.config.background_trace_mode
                if mode == "separate":
                    link = "separate"
                elif parent_trace is None:
                    link = "parent_unavailable"
                else:
                    link = "linked"
                metadata: dict[str, object] = {
                    "session_id": session_id,
                    "background_task_id": task_id,
                    "background_agent": subagent_type,
                    "background_agent_path": [*agent_path, subagent_type],
                    "background_trace_mode": mode,
                    "background_trace_link": link,
                }
                if parent_task_id is not None:
                    metadata["background_parent_task_id"] = parent_task_id
                if parent_trace is not None:
                    metadata["originating_parent_run_id"] = parent_trace.run_id
                    metadata["originating_parent_trace_id"] = parent_trace.trace_id
                config["metadata"] = metadata
                config["run_name"] = f"background/{subagent_type}/{task_id}"
                config["run_id"] = uuid.uuid4()
                config["tags"] = ["chainagents", "background-subagent"]
                scope = langsmith_tracing.background_scope(parent_trace, mode=mode)
            with scope:
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
                await delete_checkpoint_thread(f"{session_id}:background:{task_id}")

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
    ) -> str | dict[str, object]:
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
        if manager.config.batch_result_format == "json":
            return _format_batch_json(completed)
        if manager.config.batch_result_format == "markdown":
            return _format_batch_markdown(completed)
        if batch_output_store is None:
            raise RuntimeError(
                "Markdown-file batch results require a generated-output store."
            )
        return await _write_batch_markdown_files(
            completed,
            tool_call_id=runtime.tool_call_id,
            store=batch_output_store,
        )

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
