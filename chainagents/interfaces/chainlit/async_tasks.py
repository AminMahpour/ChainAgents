"""Monitor asynchronous LangGraph subagent tasks and notify Chainlit users."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import chainlit as cl
from chainlit.utils import utc_now
from langgraph_sdk import get_client

from chainagents.events.stream import AgentStreamEvent
from chainagents.runtime import AsyncSubagentConfig
from chainagents.runtime.background_tasks import (
    BackgroundTaskActivity,
    BackgroundTaskManager,
    BackgroundTaskSnapshot,
)


DEFAULT_POLL_SECONDS = 5.0
ACTIVITY_FLUSH_SECONDS = 0.05
DEFAULT_AGENT_PROTOCOL_URL = "http://127.0.0.1:2024"
TERMINAL_STATUSES = {"success", "error", "cancelled", "interrupted", "timeout"}
logger = logging.getLogger("chainagents.interfaces.chainlit.async_tasks")


def format_local_task_result(snapshot: BackgroundTaskSnapshot) -> str:
    """Format one terminal process-local task snapshot for Chainlit."""
    content = (
        f"Local background subagent `{snapshot.agent_name}` finished with status "
        f"`{snapshot.status}`.\n\nTask ID: `{snapshot.task_id}`"
    )
    if snapshot.error:
        return f"{content}\n\nError: {snapshot.error}"
    return content


@dataclass
class _LocalToolActivityState:
    """Identity and Chainlit step for one streamed background tool call."""

    call_id: str
    source: str
    name: str
    step: cl.Step


@dataclass
class _LocalTaskActivityState:
    """Chainlit steps owned by one process-local background task."""

    parent: cl.Step
    reasoning_steps: dict[str, cl.Step] = field(default_factory=dict)
    tool_steps: dict[str, _LocalToolActivityState] = field(default_factory=dict)
    suppressed_tool_call_ids: set[str] = field(default_factory=set)
    suppressed_tool_keys: set[tuple[str, str]] = field(default_factory=set)


class LocalBackgroundTaskNotifier:
    """Deliver local task activity and one terminal Chainlit message."""

    def __init__(
        self,
        *,
        manager: BackgroundTaskManager,
        session_id: str,
        reasoning_steps_enabled: bool = True,
        tool_steps_enabled: bool = True,
    ) -> None:
        self.manager = manager
        self.session_id = session_id
        self.reasoning_steps_enabled = reasoning_steps_enabled
        self.tool_steps_enabled = tool_steps_enabled
        self.queue: asyncio.Queue[BackgroundTaskSnapshot] | None = None
        self.activity_queue: asyncio.Queue[BackgroundTaskActivity] | None = None
        self.activity_states: dict[str, _LocalTaskActivityState] = {}
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Subscribe and start consuming completion events."""
        if self.task is not None and not self.task.done():
            return
        if self.manager.config.stream_activity:
            self.activity_queue = self.manager.subscribe_activity(self.session_id)
            self.task = asyncio.create_task(self._run_activity())
        else:
            self.queue = self.manager.subscribe(self.session_id)
            self.task = asyncio.create_task(self._run())

    def configure(
        self, *, reasoning_steps_enabled: bool, tool_steps_enabled: bool
    ) -> None:
        """Apply the current chat's visibility switches to future activity."""
        self.reasoning_steps_enabled = reasoning_steps_enabled
        self.tool_steps_enabled = tool_steps_enabled

    async def _run(self) -> None:
        if self.queue is None:
            return
        while True:
            snapshot = await self.queue.get()
            try:
                await cl.Message(
                    content=format_local_task_result(snapshot),
                    author="Background subagent",
                ).send()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to send local background task notice for %s.",
                    snapshot.task_id,
                )

    async def _run_activity(self) -> None:
        if self.activity_queue is None:
            return
        while True:
            first = await self.activity_queue.get()
            batch = [first]
            if first.snapshot is None:
                await asyncio.sleep(ACTIVITY_FLUSH_SECONDS)
            while True:
                try:
                    batch.append(self.activity_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            await self._handle_activity_batch(batch)

    async def _handle_activity_batch(
        self,
        batch: list[BackgroundTaskActivity],
    ) -> None:
        pending: BackgroundTaskActivity | None = None
        pending_text = ""

        async def flush_reasoning() -> None:
            nonlocal pending, pending_text
            if pending is None or pending.event is None:
                return
            try:
                await self._handle_live_event(
                    pending,
                    AgentStreamEvent(
                        kind="reasoning_delta",
                        source=pending.event.source,
                        text=pending_text,
                    ),
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to render local background task activity for %s.",
                    pending.task_id,
                )
            pending = None
            pending_text = ""

        for activity in batch:
            event = activity.event
            if event is not None and event.kind == "reasoning_delta":
                if (
                    pending is not None
                    and pending.task_id == activity.task_id
                    and pending.event is not None
                    and pending.event.source == event.source
                ):
                    pending_text += event.text
                    continue
                await flush_reasoning()
                pending = activity
                pending_text = event.text
                continue

            await flush_reasoning()
            try:
                if event is not None:
                    await self._handle_live_event(activity, event)
                elif activity.snapshot is not None:
                    await self._handle_terminal(activity)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to render local background task activity for %s.",
                    activity.task_id,
                )
        await flush_reasoning()

    async def _activity_state(
        self,
        activity: BackgroundTaskActivity,
    ) -> _LocalTaskActivityState:
        state = self.activity_states.get(activity.task_id)
        if state is not None:
            return state
        parent = cl.Step(
            name=f"{activity.agent_name} (background)",
            type="run",
            default_open=True,
        )
        parent.input = activity.description
        parent.start = utc_now()
        state = _LocalTaskActivityState(parent=parent)
        self.activity_states[activity.task_id] = state
        await parent.send()
        return state

    async def _handle_live_event(
        self,
        activity: BackgroundTaskActivity,
        event: AgentStreamEvent,
    ) -> None:
        if event.kind == "reasoning_delta" and event.text:
            state = await self._activity_state(activity)
            if not self.reasoning_steps_enabled:
                return
            step = state.reasoning_steps.get(event.source)
            if step is None:
                step = cl.Step(
                    name=f"{event.source} reasoning",
                    type="llm",
                    parent_id=state.parent.id,
                    default_open=True,
                )
                step.start = utc_now()
                state.reasoning_steps[event.source] = step
                await step.send()
            await step.stream_token(event.text)
            return

        if event.kind == "tool_call":
            state = await self._activity_state(activity)
            if event.previous_tool_call_id and event.tool_call_id:
                previous = state.tool_steps.pop(event.previous_tool_call_id, None)
                if previous is not None:
                    previous.call_id = event.tool_call_id
                    state.tool_steps[event.tool_call_id] = previous
            if not self.tool_steps_enabled:
                if event.tool_call_id in state.tool_steps:
                    return
                if event.tool_call_id:
                    state.suppressed_tool_call_ids.add(event.tool_call_id)
                if event.tool_name:
                    state.suppressed_tool_keys.add((event.source, event.tool_name))
                return
            state.suppressed_tool_call_ids.discard(event.tool_call_id)
            if event.tool_name:
                state.suppressed_tool_keys.discard((event.source, event.tool_name))
            call_id = event.tool_call_id or event.source
            tool_state = state.tool_steps.get(call_id)
            if tool_state is None:
                step = cl.Step(
                    name=f"{event.source} · {event.tool_name or 'tool'}",
                    type="tool",
                    parent_id=state.parent.id,
                    default_open=True,
                    show_input="json",
                    language="json",
                )
                step.start = utc_now()
                step.output = "Running..."
                tool_state = _LocalToolActivityState(
                    call_id=call_id,
                    source=event.source,
                    name=event.tool_name or "tool",
                    step=step,
                )
                state.tool_steps[call_id] = tool_state
                await step.send()
            step = tool_state.step
            if event.tool_name:
                tool_state.name = event.tool_name
                step.name = f"{event.source} · {event.tool_name}"
            if event.tool_args:
                step.input = event.tool_args
            await step.update()
            return

        if event.kind == "tool_result":
            if not self.tool_steps_enabled:
                existing_state = self.activity_states.get(activity.task_id)
                if existing_state is None:
                    return
                if event.tool_call_id not in existing_state.tool_steps:
                    if (
                        event.tool_call_id in existing_state.suppressed_tool_call_ids
                        or event.previous_tool_call_id
                        in existing_state.suppressed_tool_call_ids
                        or (event.source, event.tool_name)
                        in existing_state.suppressed_tool_keys
                    ):
                        return
                    visible = self._resolve_tool_state(existing_state, event)
                    if (
                        visible is None
                        or not event.tool_name
                        or visible.name != event.tool_name
                    ):
                        return
                elif event.tool_call_id in existing_state.suppressed_tool_call_ids:
                    return
            state = await self._activity_state(activity)
            call_id = event.tool_call_id or event.source
            tool_state = self._resolve_tool_state(state, event)
            if tool_state is None:
                step = cl.Step(
                    name=f"{event.source} · {event.tool_name or 'tool'}",
                    type="tool",
                    parent_id=state.parent.id,
                    default_open=True,
                    show_input="json",
                    language="json",
                )
                step.start = utc_now()
                tool_state = _LocalToolActivityState(
                    call_id=call_id,
                    source=event.source,
                    name=event.tool_name or "tool",
                    step=step,
                )
                state.tool_steps[call_id] = tool_state
                await step.send()
            elif call_id and tool_state.call_id != call_id:
                for existing_id, existing in tuple(state.tool_steps.items()):
                    if existing is tool_state:
                        state.tool_steps.pop(existing_id, None)
                tool_state.call_id = call_id
                state.tool_steps[call_id] = tool_state
            step = tool_state.step
            if event.tool_name:
                tool_state.name = event.tool_name
                step.name = f"{event.source} · {event.tool_name}"
            step.output = event.tool_result
            step.end = utc_now()
            await step.update()

    @staticmethod
    def _resolve_tool_state(
        state: _LocalTaskActivityState,
        event: AgentStreamEvent,
    ) -> _LocalToolActivityState | None:
        if event.tool_call_id and event.tool_call_id in state.tool_steps:
            return state.tool_steps[event.tool_call_id]
        candidates = list({id(item): item for item in state.tool_steps.values()}.values())
        source_name = [
            item
            for item in candidates
            if item.source == event.source
            and bool(event.tool_name)
            and item.name == event.tool_name
        ]
        if len(source_name) == 1:
            return source_name[0]
        source = [item for item in candidates if item.source == event.source]
        if len(source) == 1:
            return source[0]
        return None

    async def _handle_terminal(self, activity: BackgroundTaskActivity) -> None:
        snapshot = activity.snapshot
        if snapshot is None:
            return
        try:
            state = await self._activity_state(activity)
            await self._close_activity_state(
                activity.task_id,
                state,
                parent_output=f"Finished with status: {snapshot.status}",
            )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to close local background task steps for %s.",
                activity.task_id,
            )
        self.activity_states.pop(activity.task_id, None)
        await cl.Message(
            content=format_local_task_result(snapshot),
            author="Background subagent",
        ).send()

    async def _close_activity_state(
        self,
        task_id: str,
        state: _LocalTaskActivityState,
        *,
        parent_output: str,
    ) -> None:
        for step in state.reasoning_steps.values():
            if getattr(step, "end", None) is None:
                step.end = utc_now()
            try:
                await step.update()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close a reasoning step for %s.", task_id)
        tool_states = {
            id(item): item for item in state.tool_steps.values()
        }.values()
        for tool_state in tool_states:
            step = tool_state.step
            if getattr(step, "end", None) is None:
                step.end = utc_now()
            try:
                await step.update()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close a tool step for %s.", task_id)
        state.parent.output = parent_output
        state.parent.end = utc_now()
        try:
            await state.parent.update()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to close the background task parent step for %s.",
                task_id,
            )

    def _unsubscribe(self) -> None:
        if self.queue is not None:
            self.manager.unsubscribe(self.session_id, self.queue)
            self.queue = None
        if self.activity_queue is not None:
            self.manager.unsubscribe_activity(self.session_id, self.activity_queue)
            self.activity_queue = None

    async def aclose(self) -> None:
        """Stop notifications and best-effort close all rendered activity steps."""
        self._unsubscribe()
        consumer = self.task
        self.task = None
        if consumer is not None:
            consumer.cancel()
            try:
                await consumer
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                logger.exception("Local background task notifier stopped unexpectedly.")
        for task_id, state in tuple(self.activity_states.items()):
            await self._close_activity_state(
                task_id,
                state,
                parent_output="Stopped",
            )
        self.activity_states.clear()

    def cancel(self) -> None:
        """Stop notifications without changing the underlying jobs."""
        self._unsubscribe()
        if self.task is not None:
            self.task.cancel()
            self.task = None
        self.activity_states.clear()


def async_subagent_url_override() -> str | None:
    """Return the configured Agent Protocol URL for async subagents.

    Returns:
        The configured Agent Protocol URL, or the local default.
    """
    for key in (
        "CHAINLIT_ASYNC_SUBAGENT_URL",
        "LANGGRAPH_SERVER_URL",
        "LANGGRAPH_API_URL",
    ):
        value = os.getenv(key, "").strip()
        if value:
            return value.rstrip("/")
    return DEFAULT_AGENT_PROTOCOL_URL


def async_task_poll_seconds() -> float:
    """Return the positive polling interval for async task monitoring.

    Returns:
        The positive polling interval in seconds.
    """
    try:
        seconds = float(os.getenv("CHAINLIT_ASYNC_TASK_POLL_SECONDS", "").strip())
    except ValueError:
        return DEFAULT_POLL_SECONDS
    return seconds if seconds > 0 else DEFAULT_POLL_SECONDS


def resolved_headers(spec: AsyncSubagentConfig) -> dict[str, str]:
    """Return request headers with the default LangSmith auth scheme applied.

    Args:
        spec: Configuration specification to convert or inspect.

    Returns:
        Headers with a default authentication scheme included.
    """
    headers = dict(spec.headers or {})
    if "x-auth-scheme" not in headers:
        headers["x-auth-scheme"] = "langsmith"
    return headers


def task_values(snapshot: Any) -> dict[str, Any]:
    """Extract async task records from a LangGraph state snapshot.

    Args:
        snapshot: The snapshot value.

    Returns:
        A mapping of async task IDs to task records.
    """
    values = getattr(snapshot, "values", {}) or {}
    async_tasks = values.get("async_tasks") if isinstance(values, dict) else None
    return async_tasks if isinstance(async_tasks, dict) else {}


def result_from_thread_values(thread_values: dict[str, Any]) -> str:
    """Return the final message content from completed thread values.

    Args:
        thread_values: The thread values value.

    Returns:
        The final message content, or a fallback completion message.
    """
    messages = thread_values.get("messages", [])
    if not messages:
        return "(completed with no output messages)"
    last = messages[-1]
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(getattr(last, "content", last))


def format_task_result(result: dict[str, Any]) -> str:
    """Format task result.

    Args:
        result: Result payload to format or inspect.

    Returns:
        A user-facing summary of the async task result.
    """
    status = result.get("status", "unknown")
    agent_name = result.get("agent_name", "async subagent")
    task_id = result.get("task_id", "")
    prefix = f"Async subagent `{agent_name}` finished with status `{status}`."
    if task_id:
        prefix += f"\n\nTask ID: `{task_id}`"

    if status == "success":
        return f"{prefix}\n\n{result.get('result', '')}".strip()
    if result.get("error"):
        return f"{prefix}\n\nError: {result['error']}"
    return prefix


class AsyncTaskNotifier:
    """Poll async subagent runs and send one Chainlit completion notice per task."""

    def __init__(
        self,
        *,
        agent: Any,
        async_subagents: tuple[AsyncSubagentConfig, ...],
        url_override: str | None,
    ) -> None:
        """Initialize the async task notifier instance.

        Args:
            agent: Agent or runtime object used for the operation.
            async_subagents: Async subagent configurations available for monitoring.
            url_override: Agent Protocol URL override, if one is configured.
        """
        self.agent = agent
        self.async_subagents = {subagent.name: subagent for subagent in async_subagents}
        self.url_override = url_override
        self.poll_seconds = async_task_poll_seconds()
        self.watched_task_ids: set[str] = set()
        self.notified_task_ids: set[str] = set()
        self.tasks: set[asyncio.Task[Any]] = set()

    def matches(self, *, agent: Any, url_override: str | None) -> bool:
        """Return whether this notifier matches the active agent and URL.

        Args:
            agent: Agent or runtime object used for the operation.
            url_override: Agent Protocol URL override, if one is configured.

        Returns:
            True when the agent and URL match this notifier; otherwise, False.
        """
        return self.agent is agent and self.url_override == url_override

    async def schedule_from_state(self, *, thread_id: str) -> None:
        """Schedule watchers for non-terminal async tasks in the thread state.

        Args:
            thread_id: Conversation thread identifier.
        """
        if not self.async_subagents:
            return

        snapshot = await self.agent.aget_state(
            {"configurable": {"thread_id": thread_id}},
        )
        for task_id, task in task_values(snapshot).items():
            if not isinstance(task, dict):
                continue
            status = str(task.get("status", "")).lower()
            if status in TERMINAL_STATUSES or task_id in self.watched_task_ids:
                continue
            self.watched_task_ids.add(task_id)
            monitor_task = asyncio.create_task(self._watch_task(dict(task)))
            self.tasks.add(monitor_task)
            monitor_task.add_done_callback(self.tasks.discard)

    def cancel(self) -> None:
        """Cancel the async task notifier."""
        for task in self.tasks:
            task.cancel()
        self.tasks.clear()

    async def _watch_task(self, task: dict[str, Any]) -> None:
        """Poll one async subagent task until it reaches a terminal status.

        Args:
            task: The task value.
        """
        task_id = str(task.get("task_id") or task.get("thread_id") or "").strip()
        agent_name = str(task.get("agent_name") or "").strip()
        run_id = str(task.get("run_id") or "").strip()
        thread_id = str(task.get("thread_id") or task_id).strip()
        if not task_id or not agent_name or not run_id or not thread_id:
            return

        subagent = self.async_subagents.get(agent_name)
        if subagent is None:
            return

        url = subagent.url or self.url_override
        if not url:
            await self._notify_once(
                task_id,
                (
                    f"Async task `{task_id}` is running, but Chainlit realtime "
                    "monitoring needs an Agent Protocol `url`. Omitted `url` uses "
                    "ASGI transport only inside LangGraph Agent Server."
                ),
            )
            return

        client = get_client(url=url, headers=resolved_headers(subagent))
        try:
            failures = 0
            while True:
                try:
                    run = await client.runs.get(thread_id=thread_id, run_id=run_id)
                    failures = 0
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    if failures >= 3:
                        await self._notify_once(
                            task_id,
                            f"Could not monitor async task `{task_id}`: {exc}",
                        )
                        return
                    await asyncio.sleep(self.poll_seconds)
                    continue

                status = str(run.get("status", "")).lower()
                if status in TERMINAL_STATUSES:
                    result: dict[str, Any] = {
                        "task_id": task_id,
                        "agent_name": agent_name,
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "status": status,
                    }
                    if status == "success":
                        thread = await client.threads.get(thread_id=thread_id)
                        values = thread.get("values") or {}
                        result["result"] = result_from_thread_values(values)
                    elif run.get("error"):
                        result["error"] = str(run["error"])
                    await self._notify_once(task_id, format_task_result(result))
                    return

                await asyncio.sleep(self.poll_seconds)
        finally:
            await client.aclose()

    async def _notify_once(self, task_id: str, content: str) -> None:
        """Send one Chainlit notification for a completed async task.

        Args:
            task_id: Task identifier.
            content: Message or document content to process.
        """
        if task_id in self.notified_task_ids:
            return
        self.notified_task_ids.add(task_id)
        await cl.Message(content=content, author="Async subagent").send()
