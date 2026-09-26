"""Bridge LangGraph stream events into Chainlit messages, steps, and task lists."""

from __future__ import annotations

import asyncio
import ast
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.element import Element
from chainlit.utils import utc_now

from chainagents.events.stream import (
    AgentStreamEvent,
    anthropic_thinking_text,  # noqa: F401
    assistant_messages_for_current_prompt,  # noqa: F401
    is_assistant_message,  # noqa: F401
    iter_messages,  # noqa: F401
    message_text,  # noqa: F401
    messages_from_node_data,  # noqa: F401
    namespace_label,  # noqa: F401
    reasoning_text_from_token,  # noqa: F401
    stringify_content,
)
from chainagents.exports.generated_files import GeneratedFileDescriptor
from chainagents.exports.response import attach_response_export_actions
from chainagents.runtime.types import ChainlitResponseActionConfig

DEFAULT_AUTO_COLLAPSE_DELAY_SECONDS = 3.0
CHAINLIT_APP_CONFIG_PATH = Path(__file__).resolve().parents[3] / "chainlit.toml"


def load_auto_collapse_delay_seconds() -> float:
    """Load auto collapse delay seconds.

    Returns:
        The loaded value.
    """
    if not CHAINLIT_APP_CONFIG_PATH.exists():
        return DEFAULT_AUTO_COLLAPSE_DELAY_SECONDS

    try:
        with CHAINLIT_APP_CONFIG_PATH.open("rb") as fh:
            raw_config = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return DEFAULT_AUTO_COLLAPSE_DELAY_SECONDS

    steps_config = raw_config.get("steps", {})
    if not isinstance(steps_config, dict):
        return DEFAULT_AUTO_COLLAPSE_DELAY_SECONDS

    raw_delay = steps_config.get(
        "auto_collapse_delay_seconds",
        DEFAULT_AUTO_COLLAPSE_DELAY_SECONDS,
    )
    try:
        delay = float(raw_delay)
    except (TypeError, ValueError):
        return DEFAULT_AUTO_COLLAPSE_DELAY_SECONDS

    if delay < 0:
        return DEFAULT_AUTO_COLLAPSE_DELAY_SECONDS
    return delay


AUTO_COLLAPSE_DELAY_SECONDS = load_auto_collapse_delay_seconds()
GENERATIVE_UI_COMPONENTS = frozenset({"GeneratedPanel"})


def pretty_data(value: Any) -> str:
    """Format data.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The formatted display value.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=True)
    try:
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    except TypeError:
        return str(value)


def todos_from_write_todos_args(raw_args: str) -> list[dict[str, str]]:
    """Extract todo items from write_todos tool arguments.

    Args:
        raw_args: Raw argument text supplied with the command.

    Returns:
        The extracted todo items from write_todos tool arguments.
    """
    text = raw_args.strip()
    if not text:
        return []

    parsed: Any = None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            break
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue

    if not isinstance(parsed, dict):
        return []

    raw_todos = parsed.get("todos")
    if not isinstance(raw_todos, list):
        return []

    todos: list[dict[str, str]] = []
    for item in raw_todos:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        status = str(item.get("status", "")).strip()
        if not content or not status:
            continue
        todos.append({"content": content, "status": status})
    return todos


def todos_from_tool_message_content(content: Any) -> list[dict[str, str]]:
    """Extract todo items from a write_todos tool result message.

    Args:
        content: Message or document content to process.

    Returns:
        The extracted todo items from a write_todos tool result message.
    """
    text = stringify_content(content).strip()
    prefix = "Updated todo list to "
    if not text.startswith(prefix):
        return []
    raw_todos = text[len(prefix) :].strip()
    parsed: Any = None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(raw_todos)
            break
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue

    if not isinstance(parsed, list):
        return []

    todos: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        status = str(item.get("status", "")).strip()
        if not content or not status:
            continue
        todos.append({"content": content, "status": status})
    return todos


def parse_tool_args(raw_args: str) -> Any:
    """Parse tool args.

    Args:
        raw_args: Raw argument text supplied with the command.

    Returns:
        The parsed tool args.
    """
    text = raw_args.strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
    return None


def shorten_title(text: str, limit: int = 72) -> str:
    """Shorten long titles for compact Chainlit task display.

    Args:
        text: Text content to process.
        limit: The limit value.

    Returns:
        The shorten title result.
    """
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def tool_task_title(source: str, tool_name: str, raw_args: str) -> str:
    """Build the visible task title for a tool invocation.

    Args:
        source: The source value.
        tool_name: Name of the tool to invoke.
        raw_args: Raw argument text supplied with the command.

    Returns:
        The constructed the visible task title for a tool invocation.
    """
    name = tool_name.strip() or "tool"
    parsed = parse_tool_args(raw_args)

    if name == "write_todos":
        return f"{source}: update todo list" if source != "main-agent" else "Update todo list"

    if isinstance(parsed, dict):
        if name == "task":
            subagent = str(parsed.get("subagent_type", "")).strip()
            description = (
                str(parsed.get("description", "")).strip()
                or str(parsed.get("prompt", "")).strip()
                or str(parsed.get("task", "")).strip()
            )
            label = f"Delegate to {subagent}" if subagent else "Delegate task"
            titled = shorten_title(f"{label}: {description}" if description else label)
            return f"{source}: {titled}" if source != "main-agent" else titled

        for key in ("path", "file_path", "pattern", "query", "glob_pattern", "command"):
            value = parsed.get(key)
            if value:
                titled = shorten_title(f"{name}: {value}")
                return f"{source}: {titled}" if source != "main-agent" else titled

    return f"{source}: {name}" if source != "main-agent" else name


@dataclass
class ToolStepState:
    """Track rendered input and output for an active Chainlit tool step.

    Attributes:
        call_id: Call identifier.
        source: The source value.
        step: The step value.
        name: The name value.
        arg_chunks: The arg chunks value.
    """

    call_id: str
    source: str
    step: cl.Step | None
    name: str = "tool"
    arg_chunks: list[str] = field(default_factory=list)

    @property
    def rendered_input(self) -> str:
        """Return the rendered input associated with a tool step.

        Returns:
            The rendered input associated with a tool step.
        """
        return pretty_data("".join(self.arg_chunks).strip())


class RunTaskList:
    """Maintain the Chainlit task list shown for a single agent run."""

    MAIN_REASONING_KEY = "reasoning:main-agent"
    RESPONSE_KEY = "response"

    def __init__(
        self,
        task_list: cl.TaskList,
        *,
        reasoning_steps_enabled: bool = True,
        tool_steps_enabled: bool = True,
    ) -> None:
        """Initialize the run task list instance.

        Args:
            task_list: The task list value.
            reasoning_steps_enabled: Whether to show reasoning task entries.
            tool_steps_enabled: Whether to show tool task entries.
        """
        self.task_list = task_list
        self.reasoning_steps_enabled = reasoning_steps_enabled
        self.tool_steps_enabled = tool_steps_enabled
        self.using_todos = False
        self.tasks_by_key: dict[str, cl.Task] = {}
        self.task_order: list[str] = []
        self.response_for_id: str | None = None

    @classmethod
    async def create(
        cls,
        *,
        reasoning_steps_enabled: bool = True,
        tool_steps_enabled: bool = True,
    ) -> RunTaskList:
        """Create the run task list.

        Returns:
            The created the run task list.
        """
        return cls(
            cl.TaskList(status="Ready"),
            reasoning_steps_enabled=reasoning_steps_enabled,
            tool_steps_enabled=tool_steps_enabled,
        )

    def configure(
        self,
        *,
        reasoning_steps_enabled: bool,
        tool_steps_enabled: bool,
    ) -> None:
        """Update task visibility flags from the current runtime config."""
        self.reasoning_steps_enabled = reasoning_steps_enabled
        self.tool_steps_enabled = tool_steps_enabled

    async def show_ready(self) -> None:
        """Show the task list before the first stream event arrives."""
        self._reset_dynamic_tasks()
        self.task_list.status = "Ready"
        await self.task_list.send()

    async def start(self, response_for_id: str | None = None) -> None:
        """Start the run task list.

        Args:
            response_for_id: Response for identifier.
        """
        self._reset_dynamic_tasks()
        self.response_for_id = response_for_id
        if self.reasoning_steps_enabled:
            self._ensure_task(
                self.MAIN_REASONING_KEY,
                "main-agent reasoning",
                cl.TaskStatus.RUNNING,
            )
            await self._sync()
            return

        self.task_list.tasks = []
        self.task_list.status = "Running..."
        await self.task_list.send()

    async def mark_reasoning(self, source: str, for_id: str | None = None) -> None:
        """Mark reasoning activity on the task list.

        Args:
            source: The source value.
            for_id: For identifier.
        """
        if self.using_todos or not self.reasoning_steps_enabled:
            return
        key = self._reasoning_key(source)
        self._ensure_task(
            key,
            f"{source} reasoning",
            cl.TaskStatus.RUNNING,
            for_id=for_id,
        )
        await self._sync()

    async def mark_tool_started(
        self,
        call_id: str,
        title: str,
        *,
        for_id: str | None = None,
    ) -> None:
        """Mark a tool task as running and attach its input.

        Args:
            call_id: Call identifier.
            title: The title value.
            for_id: For identifier.
        """
        if self.using_todos or not self.tool_steps_enabled:
            return
        self._finish_running_reasoning()
        self._ensure_task(
            self._tool_key(call_id),
            title,
            cl.TaskStatus.RUNNING,
            for_id=for_id,
        )
        await self._sync()

    def migrate_tool_call(self, old_call_id: str, new_call_id: str) -> None:
        """Move an active tool task from a synthetic id to the real call id."""
        if (
            not old_call_id
            or not new_call_id
            or old_call_id == new_call_id
            or self.using_todos
            or not self.tool_steps_enabled
        ):
            return

        old_key = self._tool_key(old_call_id)
        new_key = self._tool_key(new_call_id)
        if new_key in self.tasks_by_key:
            return

        task = self.tasks_by_key.pop(old_key, None)
        if task is None:
            return

        self.tasks_by_key[new_key] = task
        self.task_order = [new_key if key == old_key else key for key in self.task_order]
        self._rebuild_tasks()

    async def mark_tool_finished(
        self,
        call_id: str,
        *,
        title: str | None = None,
        for_id: str | None = None,
        failed: bool = False,
    ) -> None:
        """Mark a tool task as finished and attach its output.

        Args:
            call_id: Call identifier.
            title: The title value.
            for_id: For identifier.
            failed: The failed value.
        """
        if self.using_todos or not self.tool_steps_enabled:
            return
        key = self._tool_key(call_id)
        title = title or "tool"
        self._ensure_task(
            key,
            title,
            cl.TaskStatus.FAILED if failed else cl.TaskStatus.DONE,
            for_id=for_id,
        )
        if failed:
            self.tasks_by_key[key].status = cl.TaskStatus.FAILED
        else:
            self.tasks_by_key[key].status = cl.TaskStatus.DONE
        await self._sync()

    async def mark_response_started(self, for_id: str | None = None) -> None:
        """Mark the final response task as running.

        Args:
            for_id: For identifier.
        """
        if self.using_todos:
            return
        self._finish_running_reasoning()
        response_for_id = for_id or self.response_for_id
        response_task = self.tasks_by_key.get(self.RESPONSE_KEY)
        if response_task is not None and response_task.status == cl.TaskStatus.RUNNING:
            if response_for_id is not None:
                response_task.forId = response_for_id
            return
        self._ensure_task(
            self.RESPONSE_KEY,
            "final response",
            cl.TaskStatus.RUNNING,
            for_id=response_for_id,
        )
        await self._sync()

    async def finish(self) -> None:
        """Finish the run task list."""
        if self.using_todos:
            self.task_list.status = self._status_from_tasks(self.task_list.tasks, finished=True)
            await self.task_list.send()
            return

        self._finish_running_reasoning()
        for key, task in self.tasks_by_key.items():
            if key.startswith("tool:") and task.status == cl.TaskStatus.RUNNING:
                task.status = cl.TaskStatus.DONE

        response_task = self.tasks_by_key.get(self.RESPONSE_KEY)
        if response_task is not None and response_task.status == cl.TaskStatus.RUNNING:
            response_task.status = cl.TaskStatus.DONE

        self.task_list.status = self._status_from_tasks(self.task_list.tasks, finished=True)
        await self.task_list.send()

    async def fail(self) -> None:
        """Fail the run task list."""
        self.task_list.status = "Failed"
        for task in self.task_list.tasks:
            if task.status != cl.TaskStatus.DONE:
                task.status = cl.TaskStatus.FAILED
        await self.task_list.send()

    async def cancel(self) -> None:
        """Mark a stopped run so the task panel does not remain running."""
        self.task_list.status = "Stopped"
        for task in self.task_list.tasks:
            if task.status == cl.TaskStatus.RUNNING:
                task.status = cl.TaskStatus.FAILED
        await self.task_list.send()

    async def update_todos(self, todos: list[dict[str, str]]) -> None:
        """Refresh dynamic todo tasks from streamed todo updates.

        Args:
            todos: The todos value.
        """
        if not todos:
            return

        self.using_todos = True
        self.tasks_by_key.clear()
        self.task_order.clear()
        self.task_list.tasks = [
            cl.Task(
                title=todo["content"],
                status=self._todo_status_to_task_status(todo["status"]),
            )
            for todo in todos
        ]
        self.task_list.status = self._status_from_tasks(self.task_list.tasks, finished=False)
        await self.task_list.send()

    def _reset_dynamic_tasks(self) -> None:
        """Remove dynamic tasks that will be rebuilt from current state."""
        self.using_todos = False
        self.tasks_by_key.clear()
        self.task_order.clear()
        self.task_list.tasks = []
        self.response_for_id = None

    def _ensure_task(
        self,
        key: str,
        title: str,
        status: cl.TaskStatus,
        *,
        for_id: str | None = None,
    ) -> cl.Task:
        """Return an existing task or create it in display order.

        Args:
            key: The key value.
            title: The title value.
            status: The status value.
            for_id: For identifier.

        Returns:
            An existing task or create it in display order.
        """
        task = self.tasks_by_key.get(key)
        if task is None:
            task = cl.Task(title=title, status=status, forId=for_id)
            self.tasks_by_key[key] = task
            self.task_order.append(key)
            self._rebuild_tasks()
            return task

        task.title = title
        task.status = status
        if for_id is not None:
            task.forId = for_id
        return task

    def _finish_running_reasoning(self) -> None:
        """Close any reasoning task that is still marked running."""
        for key, task in self.tasks_by_key.items():
            if key.startswith("reasoning:") and task.status == cl.TaskStatus.RUNNING:
                task.status = cl.TaskStatus.DONE

    def _rebuild_tasks(self) -> None:
        """Rebuild the Chainlit task list from stored task state."""
        self.task_list.tasks = [self.tasks_by_key[key] for key in self.task_order]

    async def _sync(self) -> None:
        """Synchronize task objects with the Chainlit task list element."""
        self._rebuild_tasks()
        self.task_list.status = self._status_from_tasks(self.task_list.tasks, finished=False)
        await self.task_list.send()

    def _reasoning_key(self, source: str) -> str:
        """Build a task key for a reasoning segment.

        Args:
            source: The source value.

        Returns:
            The constructed a task key for a reasoning segment.
        """
        return f"reasoning:{source}"

    def _tool_key(self, call_id: str) -> str:
        """Build a task key for a tool invocation.

        Args:
            call_id: Call identifier.

        Returns:
            The constructed a task key for a tool invocation.
        """
        return f"tool:{call_id}"

    def _status_from_tasks(self, tasks: list[cl.Task], *, finished: bool) -> str:
        """Derive the aggregate task-list status from child tasks.

        Args:
            tasks: The tasks value.
            finished: The finished value.

        Returns:
            The status from tasks result.
        """
        if not tasks:
            return "Done" if finished else "Ready"
        if any(task.status == cl.TaskStatus.FAILED for task in tasks):
            return "Failed"
        if any(task.status == cl.TaskStatus.RUNNING for task in tasks):
            return "Running..."
        if all(task.status == cl.TaskStatus.DONE for task in tasks):
            return "Done"
        return "Returned" if finished else "Pending"

    def _todo_status_to_task_status(self, status: str) -> cl.TaskStatus:
        """Map todo statuses onto Chainlit task statuses.

        Args:
            status: The status value.

        Returns:
            The todo status to task status result.
        """
        normalized = status.strip().lower()
        if normalized == "in_progress":
            return cl.TaskStatus.RUNNING
        if normalized == "completed":
            return cl.TaskStatus.DONE
        return cl.TaskStatus.READY


class ChainlitEventBridge:
    """Translate LangGraph stream events into Chainlit UI updates."""

    def __init__(
        self,
        prompt: str,
        run_task_list: RunTaskList | None = None,
        *,
        chronological_ui_enabled: bool = True,
        reasoning_steps_enabled: bool = True,
        tool_steps_enabled: bool = True,
        generative_ui_enabled: bool = True,
        generated_ui_elements: dict[str, cl.CustomElement] | None = None,
        display_prompt: str | None = None,
        export_label: str = "",
        response_actions: tuple[ChainlitResponseActionConfig, ...] = (),
    ) -> None:
        """Initialize the chainlit event bridge instance.

        Args:
            prompt: The prompt value.
            run_task_list: The run task list value.
            chronological_ui_enabled: The chronological UI enabled value.
            reasoning_steps_enabled: Whether to show reasoning steps.
            tool_steps_enabled: Whether to show tool steps.
            generative_ui_enabled: Whether to render generated UI custom elements.
            generated_ui_elements: Shared generated UI element registry for this session.
        """
        self.prompt = prompt
        self.display_prompt = prompt if display_prompt is None else display_prompt
        self.export_label = export_label
        self.response_actions = response_actions
        self.run_task_list = run_task_list
        self.response_message: cl.Message | None = None
        self.response_buffer = ""
        self.pending_response_stream = ""
        self.response_task_started = False
        self.reasoning_steps: dict[str, cl.Step] = {}
        self.reasoning_buffers: dict[str, str] = {}
        self.tool_steps: dict[str, ToolStepState] = {}
        self.summarization_steps: dict[str, cl.Step] = {}
        self.generated_ui_elements = (
            generated_ui_elements if generated_ui_elements is not None else {}
        )
        self.pending_generated_ui_events: dict[str, AgentStreamEvent] = {}
        self.pending_generated_ui_order: list[str] = []
        self.collapse_scheduled_step_ids: set[str] = set()
        self.pending_collapse_tasks: set[asyncio.Task[Any]] = set()
        self.chronological_ui_enabled = chronological_ui_enabled
        self.reasoning_steps_enabled = reasoning_steps_enabled
        self.tool_steps_enabled = tool_steps_enabled
        self.generative_ui_enabled = generative_ui_enabled

    async def start(self) -> None:
        """Start the chainlit event bridge."""
        if self.run_task_list is not None:
            await self.run_task_list.start()

    async def handle_stream_event(self, event: AgentStreamEvent) -> None:
        """Render one normalized agent stream event."""
        if event.kind == "response_delta":
            await self._stream_response(event.text)
        elif event.kind == "reasoning_delta":
            await self._stream_reasoning(event.source, event.text)
        elif event.kind == "tool_call":
            await self._stream_tool_call_event(event)
        elif event.kind == "tool_result":
            await self._complete_tool_event(event)
        elif event.kind == "summarization_status":
            await self._stream_summarization_status_event(event)
        elif event.kind == "ui_message":
            await self._render_ui_message(event)
        elif event.kind == "ui_remove":
            await self._remove_ui_message(event)

    async def finish(
        self,
        generated_files: Sequence[GeneratedFileDescriptor] = (),
    ) -> None:
        """Finish the chainlit event bridge.

        Args:
            generated_files: Validated generated files attached to the response.
        """
        await self._close_all_open_steps()
        await self._send_final_response_message(generated_files)
        await self._flush_pending_generated_ui_messages()
        if self.run_task_list is not None:
            await self.run_task_list.finish()

    async def cancel(self) -> None:
        """Close the visible steps and task panel of a cancelled turn."""
        await self._close_all_open_steps()
        if self.run_task_list is not None:
            await self.run_task_list.cancel()

    async def fail(
        self,
        exc: Exception,
        details: str,
        *,
        elements: Sequence[Element] = (),
    ) -> None:
        """Fail the chainlit event bridge.

        Args:
            exc: The exc value.
            details: The details value.
            elements: Elements (generated files) attached to the error message.
        """
        await self._close_all_open_steps()
        if self.run_task_list is not None:
            await self.run_task_list.fail()
        async with cl.Step(name="runtime error", type="tool") as step:
            step.input = self.display_prompt
            step.output = details
        await cl.Message(
            content=f"{type(exc).__name__}: {exc}",
            author="System",
            elements=list(elements),
        ).send()

    async def _stream_summarization_status_event(self, event: AgentStreamEvent) -> None:
        """Render a normalized summarization status event."""
        source = event.source or "main-agent"
        status = (event.status or "triggered").strip().lower() or "triggered"
        message = event.text or "Conversation summarization triggered."
        step = self.summarization_steps.get(source)
        if step is None:
            step = cl.Step(
                name=f"{source} summarization",
                type="llm",
                default_open=True,
            )
            step.input = self.display_prompt if source == "main-agent" else ""
            step.start = utc_now()
            await step.send()
            self.summarization_steps[source] = step

        step.output = message
        if status in {"completed", "skipped", "failed"}:
            step.end = utc_now()
            self._schedule_step_auto_collapse(step)
        await step.update()

    async def _render_ui_message(self, event: AgentStreamEvent) -> None:
        """Queue a whitelisted LangGraph UI event for post-response rendering."""
        if not self.generative_ui_enabled:
            return
        if event.ui_name not in GENERATIVE_UI_COMPONENTS or not event.ui_id:
            return

        self._queue_generated_ui_message(event)

    def _queue_generated_ui_message(self, event: AgentStreamEvent) -> None:
        """Store a generated UI event until the final response has been sent."""
        existing_pending = self.pending_generated_ui_events.get(event.ui_id)
        if existing_pending is not None:
            if event.ui_metadata.get("merge") is True:
                event = replace(
                    event,
                    ui_props={
                        **existing_pending.ui_props,
                        **event.ui_props,
                    },
                )
            self.pending_generated_ui_events[event.ui_id] = event
            return

        self.pending_generated_ui_events[event.ui_id] = event
        self.pending_generated_ui_order.append(event.ui_id)

    async def _flush_pending_generated_ui_messages(self) -> None:
        """Render queued generated UI messages after the final response."""
        for ui_id in list(self.pending_generated_ui_order):
            event = self.pending_generated_ui_events.pop(ui_id, None)
            if event is None:
                continue
            await self._send_generated_ui_message(event)
        self.pending_generated_ui_order.clear()

    async def _send_generated_ui_message(self, event: AgentStreamEvent) -> None:
        """Send or update one generated UI CustomElement."""
        existing_element = self.generated_ui_elements.get(event.ui_id)
        if existing_element is not None:
            if event.ui_metadata.get("merge") is True:
                existing_element.props = {
                    **dict(getattr(existing_element, "props", {}) or {}),
                    **event.ui_props,
                }
            else:
                existing_element.props = event.ui_props
            await existing_element.update()
            return

        element = cl.CustomElement(
            name=event.ui_name,
            props=event.ui_props,
            display="inline",
        )
        self.generated_ui_elements[event.ui_id] = element
        await cl.Message(content="", elements=[element]).send()

    async def _remove_ui_message(self, event: AgentStreamEvent) -> None:
        """Remove a tracked Chainlit CustomElement for a LangGraph remove-ui event."""
        if not self.generative_ui_enabled or not event.ui_id:
            return
        self.pending_generated_ui_events.pop(event.ui_id, None)
        if event.ui_id in self.pending_generated_ui_order:
            self.pending_generated_ui_order.remove(event.ui_id)
        element = self.generated_ui_elements.pop(event.ui_id, None)
        if element is not None:
            await element.remove()

    async def _stream_tool_call_event(self, event: AgentStreamEvent) -> None:
        """Render a normalized streamed tool call event."""
        if self.chronological_ui_enabled:
            await self._close_reasoning_step(event.source)

        state = self.tool_steps.get(event.tool_call_id)
        if state is None and event.previous_tool_call_id:
            state = self.tool_steps.pop(event.previous_tool_call_id, None)
            if state is not None:
                state.call_id = event.tool_call_id
                self.tool_steps[event.tool_call_id] = state
                if self.run_task_list is not None:
                    self.run_task_list.migrate_tool_call(
                        event.previous_tool_call_id,
                        event.tool_call_id,
                    )
        if state is None:
            step: cl.Step | None = None
            if self.tool_steps_enabled:
                step = cl.Step(
                    name=f"{event.source} tool",
                    type="tool",
                    default_open=True,
                    show_input="json",
                    language="json",
                )
                step.start = utc_now()
                step.output = "Running..."
                await step.send()
            state = ToolStepState(
                call_id=event.tool_call_id,
                source=event.source,
                step=step,
            )
            self.tool_steps[event.tool_call_id] = state

        if event.tool_name:
            state.name = event.tool_name
            if state.step is not None:
                state.step.name = f"{event.source} · {state.name}"

        if event.tool_args:
            state.arg_chunks = [event.tool_args]
            if state.name == "write_todos" and self.run_task_list is not None:
                todos = todos_from_write_todos_args(event.tool_args)
                if todos:
                    await self.run_task_list.update_todos(todos)

        if self.run_task_list is not None:
            await self.run_task_list.mark_tool_started(
                state.call_id,
                tool_task_title(event.source, state.name, event.tool_args),
                for_id=getattr(state.step, "id", None) if state.step is not None else None,
            )

        if state.step is not None:
            rendered_input = state.rendered_input
            if rendered_input:
                state.step.input = rendered_input
            await state.step.update()

    async def _complete_tool_event(self, event: AgentStreamEvent) -> None:
        """Render a normalized completed tool call event."""
        state = self._resolve_tool_step_from_event(event)
        if state is None:
            step: cl.Step | None = None
            if self.tool_steps_enabled:
                step = cl.Step(
                    name=f"{event.source} · {event.tool_name or 'tool'}",
                    type="tool",
                    default_open=True,
                    show_input="json",
                    language="json",
                )
                step.start = utc_now()
                await step.send()
            state = ToolStepState(
                call_id=event.tool_call_id or event.source,
                source=event.source,
                step=step,
                name=event.tool_name or "tool",
            )

        if event.tool_name:
            state.name = event.tool_name
            if state.step is not None:
                state.step.name = f"{event.source} · {state.name}"
        if state.step is not None:
            if not state.step.input:
                state.step.input = state.rendered_input
            state.step.output = pretty_data(event.tool_result)
            state.step.end = utc_now()
            await state.step.update()
            self._schedule_step_auto_collapse(state.step)

        if self.run_task_list is not None:
            await self.run_task_list.mark_tool_finished(
                state.call_id,
                title=tool_task_title(event.source, state.name, "".join(state.arg_chunks)),
                for_id=getattr(state.step, "id", None) if state.step is not None else None,
                failed=event.status.lower() == "error",
            )
        if state.name == "write_todos" and self.run_task_list is not None:
            todos = todos_from_tool_message_content(event.tool_result)
            if todos:
                await self.run_task_list.update_todos(todos)
        self.tool_steps.pop(state.call_id, None)

    async def _stream_reasoning(self, source: str, text: str) -> None:
        """Stream reasoning text into the active output target.

        Args:
            source: The source value.
            text: Text content to process.
        """
        previous = self.reasoning_buffers.get(source, "")
        delta = text[len(previous) :] if text.startswith(previous) else text
        if not delta:
            return

        if not self.reasoning_steps_enabled:
            self.reasoning_buffers[source] = previous + delta
            return

        step = self.reasoning_steps.get(source)
        if step is None:
            step = cl.Step(
                name=f"{source} reasoning",
                type="llm",
                default_open=True,
            )
            step.input = self.display_prompt if source == "main-agent" else ""
            step.start = utc_now()
            await step.send()
            self.reasoning_steps[source] = step
            if self.run_task_list is not None:
                await self.run_task_list.mark_reasoning(
                    source,
                    for_id=getattr(step, "id", None),
                )

        await step.stream_token(delta)
        self.reasoning_buffers[source] = previous + delta

    async def _stream_response(self, text: str) -> None:
        """Stream final response text into the active output target.

        Args:
            text: Text content to process.
        """
        delta = text[len(self.response_buffer) :] if text.startswith(self.response_buffer) else text
        if not delta:
            return
        await self._close_active_reasoning_steps()
        if (
            not self.response_task_started
            and self.run_task_list is not None
        ):
            await self.run_task_list.mark_response_started()
            self.response_task_started = True
        self.response_buffer += delta
        self.pending_response_stream += delta
        if not self.chronological_ui_enabled:
            if self.response_message is None:
                self.response_message = await cl.Message(content="").send()
            await self._flush_response_stream()

    async def _send_final_response_message(
        self,
        generated_files: Sequence[GeneratedFileDescriptor] = (),
    ) -> None:
        """Send the buffered final response as a Chainlit message."""
        if not self.response_buffer:
            return

        if self.response_message is None:
            self.response_message = await cl.Message(content=self.response_buffer).send()
            self.pending_response_stream = ""
        else:
            await self._flush_response_stream()

        if self.run_task_list is not None:
            await self.run_task_list.mark_response_started(
                for_id=getattr(self.response_message, "id", None)
            )
            self.response_task_started = True

        attach_response_export_actions(
            self.response_message,
            prompt=self.prompt,
            response_text=self.response_buffer,
            generated_files=generated_files,
            response_actions=self.response_actions,
            export_label=self.export_label,
        )
        await self.response_message.update()

    async def _flush_response_stream(self) -> None:
        """Flush buffered response text to the Chainlit message."""
        if not self.pending_response_stream:
            return
        pending = self.pending_response_stream
        if self.response_message is not None:
            await self.response_message.stream_token(pending)
        self.pending_response_stream = ""

    async def _close_reasoning_step(self, source: str) -> None:
        """Close one active Chainlit reasoning step.

        Args:
            source: The source value.
        """
        step = self.reasoning_steps.pop(source, None)
        if step is None:
            return
        if not step.end:
            step.end = utc_now()
        await step.update()
        self._schedule_step_auto_collapse(step)

    async def _close_active_reasoning_steps(self) -> None:
        """Close all active Chainlit reasoning steps."""
        for source in list(self.reasoning_steps):
            await self._close_reasoning_step(source)

    def _resolve_tool_step_from_event(
        self,
        event: AgentStreamEvent,
    ) -> ToolStepState | None:
        """Return the active Chainlit step for a normalized tool event."""
        if event.tool_call_id and event.tool_call_id in self.tool_steps:
            return self.tool_steps[event.tool_call_id]

        source_name_matches = [
            state
            for state in self.tool_steps.values()
            if (
                state.source == event.source
                and bool(event.tool_name)
                and state.name == event.tool_name
            )
        ]
        if source_name_matches:
            return source_name_matches[0]

        source_matches = [
            state for state in self.tool_steps.values() if state.source == event.source
        ]
        if source_matches:
            return source_matches[0]

        name_matches = [
            state
            for state in self.tool_steps.values()
            if bool(event.tool_name) and state.name == event.tool_name
        ]
        if name_matches:
            return name_matches[0]

        if self.tool_steps:
            return next(iter(self.tool_steps.values()))
        return None

    async def _close_all_open_steps(self) -> None:
        """Close all Chainlit steps that remain open at run completion."""
        for state in list(self.tool_steps.values()):
            if state.step is None:
                continue
            if not state.step.output:
                state.step.output = "Finished without a streamed tool result."
            if not state.step.end:
                state.step.end = utc_now()
            await state.step.update()
            self._schedule_step_auto_collapse(state.step)
        self.tool_steps.clear()

        for step in self.reasoning_steps.values():
            if not step.end:
                step.end = utc_now()
            await step.update()
            self._schedule_step_auto_collapse(step)
        self.reasoning_steps.clear()

        for step in self.summarization_steps.values():
            if not step.end:
                step.end = utc_now()
            await step.update()
            self._schedule_step_auto_collapse(step)
        self.summarization_steps.clear()

    def _schedule_step_auto_collapse(self, step: cl.Step) -> None:
        """Schedule delayed auto-collapse for a Chainlit step.

        Args:
            step: The step value.
        """
        if step.id in self.collapse_scheduled_step_ids:
            return

        self.collapse_scheduled_step_ids.add(step.id)

        async def collapse_later() -> None:
            """Collapse a Chainlit step after the configured delay."""
            try:
                await asyncio.sleep(AUTO_COLLAPSE_DELAY_SECONDS)
                step.auto_collapse = True
                await step.update()
            except Exception:
                return

        task = asyncio.create_task(collapse_later())
        self.pending_collapse_tasks.add(task)
        task.add_done_callback(self.pending_collapse_tasks.discard)
