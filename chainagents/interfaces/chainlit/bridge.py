"""Bridge LangGraph stream events into Chainlit messages, steps, and task lists."""

from __future__ import annotations

import asyncio
import ast
import json
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.utils import utc_now

from chainagents.events.stream import AgentStreamEvent, AgentStreamEventAdapter
from chainagents.exports.response import attach_response_export_actions

DEFAULT_AUTO_COLLAPSE_DELAY_SECONDS = 3.0
RESPONSE_STREAM_FLUSH_INTERVAL_SECONDS = 0.05
RESPONSE_STREAM_FLUSH_CHARS = 1024
CHAINLIT_APP_CONFIG_PATH = Path(__file__).resolve().parents[3] / "chainlit.toml"
SUMMARIZATION_STATUS_KIND = "summarization_status"
LANGGRAPH_STREAM_MODES = {
    "values",
    "updates",
    "custom",
    "messages",
    "checkpoints",
    "tasks",
    "debug",
}


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
ANTHROPIC_THINKING_BLOCK_TYPES = {"thinking", "redacted_thinking"}


def stringify_content(value: Any) -> str:
    """Convert content.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The string representation.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [stringify_content(item) for item in value]
        return "".join(part for part in parts if part)
    if isinstance(value, dict):
        if value.get("type") in ANTHROPIC_THINKING_BLOCK_TYPES:
            return ""
        for key in ("text", "reasoning", "content"):
            nested = value.get(key)
            if isinstance(nested, (str, list, dict)):
                return stringify_content(nested)
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    return str(value)


def anthropic_thinking_text(value: Any) -> str:
    """Extract Claude thinking text from LangChain Anthropic content blocks.

    Args:
        value: Message content to inspect.

    Returns:
        Extracted thinking text, if present.
    """
    if isinstance(value, list):
        return "".join(anthropic_thinking_text(item) for item in value)
    if not isinstance(value, dict) or value.get("type") != "thinking":
        return ""
    return stringify_content(value.get("thinking"))


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


def namespace_label(ns: tuple[str, ...], metadata: dict[str, Any]) -> str:
    """Return a display label for a tool namespace.

    Args:
        ns: The ns value.
        metadata: The metadata value.

    Returns:
        A display label for a tool namespace.
    """
    agent_name = metadata.get("lc_agent_name")
    if agent_name:
        return str(agent_name)
    if not ns:
        return "main-agent"

    labels: list[str] = []
    for segment in ns:
        if segment.startswith("tools:"):
            labels.append(f"subagent {segment.split(':', 1)[1]}")
            continue
        labels.append(segment.split(":", 1)[0])
    return " / ".join(labels)


def langgraph_part_from_event_chunk(chunk: Any) -> dict[str, Any] | None:
    """Normalize stream event chunks into LangGraph part metadata.

    Args:
        chunk: Streamed event chunk to normalize.

    Returns:
        The langgraph part from event chunk result.
    """
    if isinstance(chunk, dict):
        mode = chunk.get("type")
        if isinstance(mode, str) and mode in LANGGRAPH_STREAM_MODES and "data" in chunk:
            return chunk
        return None

    if not isinstance(chunk, tuple):
        return None

    if len(chunk) == 3:
        ns, mode, data = chunk
    elif len(chunk) == 2:
        first, data = chunk
        if isinstance(first, tuple):
            ns = first
            mode = "values"
        else:
            ns = ()
            mode = first
    else:
        return None

    if not isinstance(mode, str) or mode not in LANGGRAPH_STREAM_MODES:
        return None

    if isinstance(ns, tuple):
        namespace = ns
    elif ns in (None, ""):
        namespace = ()
    else:
        return None

    return {"type": mode, "ns": namespace, "data": data}


def reasoning_text_from_token(token: Any) -> str:
    """Extract reasoning text from a streamed model token.

    Args:
        token: Streamed model token to inspect.

    Returns:
        The extracted reasoning text from a streamed model token.
    """
    if hasattr(token, "additional_kwargs"):
        text = stringify_content(token.additional_kwargs.get("reasoning_content"))
        if text:
            return text
    if hasattr(token, "reasoning_content"):
        return stringify_content(token.reasoning_content)
    if hasattr(token, "content"):
        return anthropic_thinking_text(token.content)
    return ""


def iter_messages(value: Any) -> list[Any]:
    """Iterate over messages.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        An iterator over the matching values.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        if "messages" in value:
            return iter_messages(value["messages"])
        if "value" in value:
            return iter_messages(value["value"])
        return [value]
    if isinstance(value, (str, bytes)):
        return []

    for attr in ("value", "messages", "data"):
        if hasattr(value, attr):
            nested = getattr(value, attr)
            messages = iter_messages(nested)
            if messages:
                return messages

    try:
        return list(value)
    except TypeError:
        return [value]


def messages_from_node_data(data: Any) -> list[Any]:
    """Extract message objects from LangGraph node update payloads.

    Args:
        data: Payload data to inspect.

    Returns:
        The extracted message objects from langgraph node update payloads.
    """
    if data is None:
        return []
    if isinstance(data, dict):
        return iter_messages(data.get("messages"))
    return iter_messages(data)


def todos_from_node_data(data: Any) -> list[dict[str, str]]:
    """Extract todo items from LangGraph node update payloads.

    Args:
        data: Payload data to inspect.

    Returns:
        The extracted todo items from langgraph node update payloads.
    """
    if data is None:
        return []

    raw_todos: Any = None
    if isinstance(data, dict):
        raw_todos = data.get("todos")
        if raw_todos is None:
            for attr in ("value", "data"):
                nested = data.get(attr)
                todos = todos_from_node_data(nested)
                if todos:
                    return todos
    else:
        for attr in ("todos", "value", "data"):
            if hasattr(data, attr):
                nested = getattr(data, attr)
                if attr == "todos":
                    raw_todos = nested
                    break
                todos = todos_from_node_data(nested)
                if todos:
                    return todos

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


def is_assistant_message(message: Any) -> bool:
    """Return whether assistant message.

    Args:
        message: Chainlit message or LangChain message to process.

    Returns:
        Whether assistant message.
    """
    return getattr(message, "type", None) in {"ai", "AIMessageChunk"}


def message_text(message: Any) -> str:
    """Build the message for text.

    Args:
        message: Chainlit message or LangChain message to process.

    Returns:
        The constructed the message for text.
    """
    return stringify_content(getattr(message, "content", "")).strip()


def assistant_messages_for_current_prompt(messages: list[Any], prompt: str) -> list[Any]:
    """Return assistant messages produced after the current prompt began.

    Args:
        messages: The messages value.
        prompt: The prompt value.

    Returns:
        Assistant messages produced after the current prompt began.
    """
    prompt_text = prompt.strip()
    current_prompt_index = -1
    for index, message in enumerate(messages):
        if getattr(message, "type", None) != "human":
            continue
        if message_text(message) == prompt_text:
            current_prompt_index = index

    if current_prompt_index < 0:
        return []

    return [
        message
        for message in messages[current_prompt_index + 1 :]
        if is_assistant_message(message)
    ]


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
    ) -> None:
        """Initialize the chainlit event bridge instance.

        Args:
            prompt: The prompt value.
            run_task_list: The run task list value.
            chronological_ui_enabled: The chronological UI enabled value.
            reasoning_steps_enabled: Whether to show reasoning steps.
            tool_steps_enabled: Whether to show tool steps.
        """
        self.prompt = prompt
        self.run_task_list = run_task_list
        self.response_message: cl.Message | None = None
        self.response_buffer = ""
        self.pending_response_stream = ""
        self.response_task_started = False
        self.last_response_flush_at = 0.0
        self.response_streamed_from_messages = False
        self.stream_adapter = AgentStreamEventAdapter(prompt=prompt)
        self.reasoning_steps: dict[str, cl.Step] = {}
        self.reasoning_buffers: dict[str, str] = {}
        self.tool_steps: dict[str, ToolStepState] = {}
        self.summarization_steps: dict[str, cl.Step] = {}
        self.collapse_scheduled_step_ids: set[str] = set()
        self.pending_collapse_tasks: set[asyncio.Task[Any]] = set()
        self.chronological_ui_enabled = chronological_ui_enabled
        self.reasoning_steps_enabled = reasoning_steps_enabled
        self.tool_steps_enabled = tool_steps_enabled

    async def start(self) -> None:
        """Start the chainlit event bridge."""
        if self.run_task_list is not None:
            await self.run_task_list.start()

    async def handle_part(self, part: dict[str, Any]) -> None:
        """Handle one normalized LangGraph stream part.

        Args:
            part: The part value.
        """
        if part["type"] == "updates" and self.run_task_list is not None:
            await self._update_todos_from_update_part(part)

        for stream_event in self.stream_adapter.events_from_part(part):
            await self._handle_stream_event(stream_event)

    async def _handle_stream_event(self, event: AgentStreamEvent) -> None:
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

    async def _update_todos_from_update_part(self, part: dict[str, Any]) -> None:
        """Refresh Chainlit task list todos from a LangGraph update part."""
        data_by_node = part.get("data")
        if not isinstance(data_by_node, dict):
            return
        for data in data_by_node.values():
            todos = todos_from_node_data(data)
            if todos:
                await self.run_task_list.update_todos(todos)

    async def handle_event(self, event: dict[str, Any]) -> None:
        """Handle one raw LangGraph stream event.

        Args:
            event: LangGraph stream event to process.
        """
        if event.get("event") != "on_chain_stream":
            return
        if event.get("parent_ids"):
            return

        data = event.get("data")
        if not isinstance(data, dict):
            return

        part = langgraph_part_from_event_chunk(data.get("chunk"))
        if part is None:
            return

        await self.handle_part(part)

    async def finish(self) -> None:
        """Finish the chainlit event bridge."""
        await self._close_all_open_steps()
        await self._send_final_response_message()
        if self.run_task_list is not None:
            await self.run_task_list.finish()

    async def fail(self, exc: Exception, details: str) -> None:
        """Fail the chainlit event bridge.

        Args:
            exc: The exc value.
            details: The details value.
        """
        await self._close_all_open_steps()
        if self.run_task_list is not None:
            await self.run_task_list.fail()
        async with cl.Step(name="runtime error", type="tool") as step:
            step.input = self.prompt
            step.output = details
        await cl.Message(content=f"{type(exc).__name__}: {exc}", author="System").send()

    async def _handle_message_chunk(self, part: dict[str, Any]) -> None:
        """Handle message chunk.

        Args:
            part: The part value.
        """
        token, metadata = part["data"]
        ns = tuple(part.get("ns", ()))
        source = namespace_label(ns, metadata)
        is_main_source = not ns

        reasoning_text = reasoning_text_from_token(token)
        if reasoning_text:
            await self._stream_reasoning(source, reasoning_text)

        tool_call_chunks = getattr(token, "tool_call_chunks", None) or []
        if tool_call_chunks:
            for chunk in tool_call_chunks:
                await self._stream_tool_call(source, chunk)

        token_type = getattr(token, "type", None)
        if token_type == "tool":
            await self._complete_tool_step(source, token)
            return

        content_text = stringify_content(getattr(token, "content", ""))
        if is_main_source and content_text and not tool_call_chunks:
            self.response_streamed_from_messages = True
            await self._stream_response(content_text)

    async def _handle_update_chunk(self, part: dict[str, Any]) -> None:
        """Handle update chunk.

        Args:
            part: The part value.
        """
        ns = tuple(part.get("ns", ()))
        metadata = {"lc_agent_name": None}
        source = namespace_label(ns, metadata)

        for node_name, data in part["data"].items():
            if self.run_task_list is not None:
                todos = todos_from_node_data(data)
                if todos:
                    await self.run_task_list.update_todos(todos)

            if node_name != "tools":
                if not ns and not self.response_streamed_from_messages:
                    assistant_messages = assistant_messages_for_current_prompt(
                        messages_from_node_data(data),
                        self.prompt,
                    )
                    if assistant_messages:
                        content_text = stringify_content(
                            getattr(assistant_messages[-1], "content", "")
                        )
                        if content_text:
                            await self._stream_response(content_text)
                continue

            for message in messages_from_node_data(data):
                if getattr(message, "type", None) == "tool":
                    await self._complete_tool_step(source, message)

    async def _handle_custom_chunk(self, part: dict[str, Any]) -> None:
        """Handle custom chunk.

        Args:
            part: The part value.
        """
        data = part.get("data")
        if not isinstance(data, dict):
            return
        if data.get("kind") != SUMMARIZATION_STATUS_KIND:
            return

        source = str(data.get("source") or "main-agent").strip() or "main-agent"
        status = str(data.get("status") or "triggered").strip().lower() or "triggered"
        message = str(data.get("message") or "Conversation summarization triggered.").strip()
        step = self.summarization_steps.get(source)
        if step is None:
            step = cl.Step(
                name=f"{source} summarization",
                type="llm",
                default_open=True,
            )
            step.input = self.prompt if source == "main-agent" else ""
            step.start = utc_now()
            await step.send()
            self.summarization_steps[source] = step

        step.output = message
        if status in {"completed", "skipped", "failed"}:
            step.end = utc_now()
            self._schedule_step_auto_collapse(step)
        await step.update()

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
            step.input = self.prompt if source == "main-agent" else ""
            step.start = utc_now()
            await step.send()
            self.summarization_steps[source] = step

        step.output = message
        if status in {"completed", "skipped", "failed"}:
            step.end = utc_now()
            self._schedule_step_auto_collapse(step)
        await step.update()

    async def _stream_tool_call_event(self, event: AgentStreamEvent) -> None:
        """Render a normalized streamed tool call event."""
        if self.chronological_ui_enabled:
            await self._close_reasoning_step(event.source)

        state = self.tool_steps.get(event.tool_call_id)
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
            step.input = self.prompt if source == "main-agent" else ""
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

    async def _send_final_response_message(self) -> None:
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
        )
        await self.response_message.update()

    def _should_flush_response_stream(self) -> bool:
        """Return whether buffered response text should be flushed now.

        Returns:
            Whether buffered response text should be flushed now.
        """
        if len(self.pending_response_stream) >= RESPONSE_STREAM_FLUSH_CHARS:
            return True
        if not self.last_response_flush_at:
            return True
        return (
            time.monotonic() - self.last_response_flush_at
            >= RESPONSE_STREAM_FLUSH_INTERVAL_SECONDS
        )

    async def _flush_response_stream(self) -> None:
        """Flush buffered response text to the Chainlit message."""
        if not self.pending_response_stream:
            return
        pending = self.pending_response_stream
        if self.response_message is not None:
            await self.response_message.stream_token(pending)
        self.pending_response_stream = ""
        self.last_response_flush_at = time.monotonic()

    async def _stream_tool_call(self, source: str, chunk: dict[str, Any]) -> None:
        """Render a streamed tool call and its accumulated arguments.

        Args:
            source: The source value.
            chunk: Streamed event chunk to normalize.
        """
        if self.chronological_ui_enabled:
            await self._close_reasoning_step(source)
        call_id = str(chunk.get("id") or f"{source}:{chunk.get('index', '0')}")
        state = self.tool_steps.get(call_id)
        if state is None:
            step: cl.Step | None = None
            if self.tool_steps_enabled:
                step = cl.Step(
                    name=f"{source} tool",
                    type="tool",
                    default_open=True,
                    show_input="json",
                    language="json",
                )
                step.start = utc_now()
                step.output = "Running..."
                await step.send()
            state = ToolStepState(call_id=call_id, source=source, step=step)
            self.tool_steps[call_id] = state

        tool_name = chunk.get("name")
        if tool_name:
            state.name = str(tool_name)
            if state.step is not None:
                state.step.name = f"{source} · {state.name}"

        arg_chunk = chunk.get("args")
        if arg_chunk:
            state.arg_chunks.append(str(arg_chunk))
            if state.name == "write_todos" and self.run_task_list is not None:
                todos = todos_from_write_todos_args("".join(state.arg_chunks))
                if todos:
                    await self.run_task_list.update_todos(todos)

        if self.run_task_list is not None:
            await self.run_task_list.mark_tool_started(
                call_id,
                tool_task_title(source, state.name, "".join(state.arg_chunks)),
                for_id=getattr(state.step, "id", None) if state.step is not None else None,
            )

        if state.step is not None:
            rendered_input = state.rendered_input
            if rendered_input:
                state.step.input = rendered_input
            await state.step.update()

    async def _complete_tool_step(self, source: str, tool_message: Any) -> None:
        """Finish the Chainlit step associated with a completed tool call.

        Args:
            source: The source value.
            tool_message: The tool message value.
        """
        state = self._resolve_tool_step(source, tool_message)
        if state is None:
            step: cl.Step | None = None
            if self.tool_steps_enabled:
                step = cl.Step(
                    name=f"{source} · {getattr(tool_message, 'name', 'tool')}",
                    type="tool",
                    default_open=True,
                    show_input="json",
                    language="json",
                )
                step.start = utc_now()
                await step.send()
            state = ToolStepState(
                call_id=str(getattr(tool_message, "tool_call_id", getattr(tool_message, "id", source))),
                source=source,
                step=step,
                name=str(getattr(tool_message, "name", "tool")),
            )

        if state.step is not None:
            if not state.step.input:
                state.step.input = state.rendered_input
            state.step.output = pretty_data(getattr(tool_message, "content", ""))
            state.step.end = utc_now()
            await state.step.update()
            self._schedule_step_auto_collapse(state.step)
        if self.run_task_list is not None:
            await self.run_task_list.mark_tool_finished(
                state.call_id,
                title=tool_task_title(source, state.name, "".join(state.arg_chunks)),
                for_id=getattr(state.step, "id", None) if state.step is not None else None,
                failed=str(getattr(tool_message, "status", "")).lower() == "error",
            )
        if state.name == "write_todos" and self.run_task_list is not None:
            todos = todos_from_tool_message_content(getattr(tool_message, "content", ""))
            if todos:
                await self.run_task_list.update_todos(todos)
        self.tool_steps.pop(state.call_id, None)

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

    def _resolve_tool_step(self, source: str, tool_message: Any) -> ToolStepState | None:
        """Return the active Chainlit step for a streamed tool call.

        Args:
            source: The source value.
            tool_message: The tool message value.

        Returns:
            The active Chainlit step for a streamed tool call.
        """
        tool_call_id = getattr(tool_message, "tool_call_id", None)
        if tool_call_id and tool_call_id in self.tool_steps:
            return self.tool_steps[tool_call_id]

        tool_name = getattr(tool_message, "name", None)
        source_name_matches = [
            state
            for state in self.tool_steps.values()
            if state.source == source and tool_name is not None and state.name == tool_name
        ]
        if source_name_matches:
            return source_name_matches[0]

        source_matches = [
            state for state in self.tool_steps.values() if state.source == source
        ]
        if source_matches:
            return source_matches[0]

        name_matches = [
            state
            for state in self.tool_steps.values()
            if tool_name is not None and state.name == tool_name
        ]
        if name_matches:
            return name_matches[0]

        if self.tool_steps:
            return next(iter(self.tool_steps.values()))
        return None

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
