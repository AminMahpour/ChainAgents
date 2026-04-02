from __future__ import annotations

import asyncio
import ast
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.utils import utc_now


DEFAULT_AUTO_COLLAPSE_DELAY_SECONDS = 3.0
CHAINLIT_APP_CONFIG_PATH = Path(__file__).resolve().parent / "chainlit.toml"


def load_auto_collapse_delay_seconds() -> float:
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


def stringify_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [stringify_content(item) for item in value]
        return "".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "reasoning", "content"):
            nested = value.get(key)
            if isinstance(nested, (str, list, dict)):
                return stringify_content(nested)
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    return str(value)


def pretty_data(value: Any) -> str:
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


def reasoning_text_from_token(token: Any) -> str:
    if hasattr(token, "additional_kwargs"):
        text = stringify_content(token.additional_kwargs.get("reasoning_content"))
        if text:
            return text
    if hasattr(token, "reasoning_content"):
        return stringify_content(token.reasoning_content)
    return ""


def iter_messages(value: Any) -> list[Any]:
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
    if data is None:
        return []
    if isinstance(data, dict):
        return iter_messages(data.get("messages"))
    return iter_messages(data)


def todos_from_node_data(data: Any) -> list[dict[str, str]]:
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
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def tool_task_title(source: str, tool_name: str, raw_args: str) -> str:
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
    return getattr(message, "type", None) in {"ai", "AIMessageChunk"}


def message_text(message: Any) -> str:
    return stringify_content(getattr(message, "content", "")).strip()


def assistant_messages_for_current_prompt(messages: list[Any], prompt: str) -> list[Any]:
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
    call_id: str
    source: str
    step: cl.Step
    name: str = "tool"
    arg_chunks: list[str] = field(default_factory=list)

    @property
    def rendered_input(self) -> str:
        return pretty_data("".join(self.arg_chunks).strip())


class RunTaskList:
    MAIN_REASONING_KEY = "reasoning:main-agent"
    RESPONSE_KEY = "response"

    def __init__(self, task_list: cl.TaskList) -> None:
        self.task_list = task_list
        self.using_todos = False
        self.tasks_by_key: dict[str, cl.Task] = {}
        self.task_order: list[str] = []
        self.response_for_id: str | None = None

    @classmethod
    async def create(cls) -> RunTaskList:
        return cls(cl.TaskList(status="Ready"))

    async def show_ready(self) -> None:
        self._reset_dynamic_tasks()
        self.task_list.status = "Ready"
        await self.task_list.send()

    async def start(self, response_for_id: str | None = None) -> None:
        self._reset_dynamic_tasks()
        self.response_for_id = response_for_id
        self._ensure_task(
            self.MAIN_REASONING_KEY,
            "main-agent reasoning",
            cl.TaskStatus.RUNNING,
        )
        await self._sync()

    async def mark_reasoning(self, source: str, for_id: str | None = None) -> None:
        if self.using_todos:
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
        if self.using_todos:
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
        if self.using_todos:
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
        if self.using_todos:
            return
        self._finish_running_reasoning()
        response_for_id = for_id or self.response_for_id
        self._ensure_task(
            self.RESPONSE_KEY,
            "final response",
            cl.TaskStatus.RUNNING,
            for_id=response_for_id,
        )
        await self._sync()

    async def finish(self) -> None:
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
        self.task_list.status = "Failed"
        for task in self.task_list.tasks:
            if task.status != cl.TaskStatus.DONE:
                task.status = cl.TaskStatus.FAILED
        await self.task_list.send()

    async def update_todos(self, todos: list[dict[str, str]]) -> None:
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
        for key, task in self.tasks_by_key.items():
            if key.startswith("reasoning:") and task.status == cl.TaskStatus.RUNNING:
                task.status = cl.TaskStatus.DONE

    def _rebuild_tasks(self) -> None:
        self.task_list.tasks = [self.tasks_by_key[key] for key in self.task_order]

    async def _sync(self) -> None:
        self._rebuild_tasks()
        self.task_list.status = self._status_from_tasks(self.task_list.tasks, finished=False)
        await self.task_list.send()

    def _reasoning_key(self, source: str) -> str:
        return f"reasoning:{source}"

    def _tool_key(self, call_id: str) -> str:
        return f"tool:{call_id}"

    def _status_from_tasks(self, tasks: list[cl.Task], *, finished: bool) -> str:
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
        normalized = status.strip().lower()
        if normalized == "in_progress":
            return cl.TaskStatus.RUNNING
        if normalized == "completed":
            return cl.TaskStatus.DONE
        return cl.TaskStatus.READY


class ChainlitEventBridge:
    def __init__(self, prompt: str, run_task_list: RunTaskList | None = None) -> None:
        self.prompt = prompt
        self.run_task_list = run_task_list
        self.response_message: cl.Message | None = None
        self.response_buffer = ""
        self.response_streamed_from_messages = False
        self.reasoning_steps: dict[str, cl.Step] = {}
        self.reasoning_buffers: dict[str, str] = {}
        self.tool_steps: dict[str, ToolStepState] = {}
        self.collapse_scheduled_step_ids: set[str] = set()
        self.pending_collapse_tasks: set[asyncio.Task[Any]] = set()

    async def start(self) -> None:
        self.response_message = await cl.Message(content="").send()
        if self.run_task_list is not None:
            await self.run_task_list.start(response_for_id=self.response_message.id)

    async def handle_part(self, part: dict[str, Any]) -> None:
        kind = part["type"]
        if kind == "messages":
            await self._handle_message_chunk(part)
            return
        if kind == "updates":
            await self._handle_update_chunk(part)

    async def finish(self) -> None:
        await self._close_all_open_steps()
        if self.response_message is not None:
            await self.response_message.update()
        if self.run_task_list is not None:
            await self.run_task_list.finish()

    async def fail(self, exc: Exception, details: str) -> None:
        await self._close_all_open_steps()
        if self.run_task_list is not None:
            await self.run_task_list.fail()
        async with cl.Step(name="runtime error", type="tool") as step:
            step.input = self.prompt
            step.output = details
        await cl.Message(content=f"{type(exc).__name__}: {exc}", author="System").send()

    async def _handle_message_chunk(self, part: dict[str, Any]) -> None:
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

    async def _stream_reasoning(self, source: str, text: str) -> None:
        previous = self.reasoning_buffers.get(source, "")
        delta = text[len(previous) :] if text.startswith(previous) else text
        if not delta:
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
        delta = text[len(self.response_buffer) :] if text.startswith(self.response_buffer) else text
        if not delta:
            return
        if self.run_task_list is not None and self.response_message is not None:
            await self.run_task_list.mark_response_started(
                for_id=getattr(self.response_message, "id", None)
            )
        if self.response_message is not None:
            await self.response_message.stream_token(delta)
        self.response_buffer += delta

    async def _stream_tool_call(self, source: str, chunk: dict[str, Any]) -> None:
        call_id = str(chunk.get("id") or f"{source}:{chunk.get('index', '0')}")
        state = self.tool_steps.get(call_id)
        if state is None:
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
                for_id=getattr(state.step, "id", None),
            )

        rendered_input = state.rendered_input
        if rendered_input:
            state.step.input = rendered_input
        await state.step.update()

    async def _complete_tool_step(self, source: str, tool_message: Any) -> None:
        state = self._resolve_tool_step(source, tool_message)
        if state is None:
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
                for_id=getattr(state.step, "id", None),
                failed=str(getattr(tool_message, "status", "")).lower() == "error",
            )
        if state.name == "write_todos" and self.run_task_list is not None:
            todos = todos_from_tool_message_content(getattr(tool_message, "content", ""))
            if todos:
                await self.run_task_list.update_todos(todos)
        self.tool_steps.pop(state.call_id, None)

    def _resolve_tool_step(self, source: str, tool_message: Any) -> ToolStepState | None:
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

    async def _close_all_open_steps(self) -> None:
        for state in list(self.tool_steps.values()):
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

    def _schedule_step_auto_collapse(self, step: cl.Step) -> None:
        if step.id in self.collapse_scheduled_step_ids:
            return

        self.collapse_scheduled_step_ids.add(step.id)

        async def collapse_later() -> None:
            try:
                await asyncio.sleep(AUTO_COLLAPSE_DELAY_SECONDS)
                step.auto_collapse = True
                await step.update()
            except Exception:
                return

        task = asyncio.create_task(collapse_later())
        self.pending_collapse_tasks.add(task)
        task.add_done_callback(self.pending_collapse_tasks.discard)
