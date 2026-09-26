"""Test the Chainlit front end on the shared turn runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

import main
from chainagents.runtime.reflection import ReflectionConfig


class _Token:
    type = "AIMessageChunk"
    additional_kwargs: ClassVar[dict[str, str]] = {}
    tool_call_chunks: ClassVar[list[dict[str, str]]] = []

    def __init__(self, content: str = "") -> None:
        self.content = content


def _raw(chunk: object) -> dict[str, object]:
    return {"event": "on_chain_stream", "data": {"chunk": chunk}}


def _token(text: str) -> dict[str, object]:
    return _raw(((), "messages", (_Token(text), {})))


def _write_file_events(path: str) -> list[dict[str, object]]:
    call = _Token()
    call.tool_call_chunks = [
        {
            "id": "call-1",
            "name": "write_file",
            "args": json.dumps({"file_path": path, "content": "x"}),
        }
    ]
    result = SimpleNamespace(
        type="tool",
        name="write_file",
        status="success",
        tool_call_id="call-1",
        content="ok",
    )
    return [_raw(((), "messages", (call, {}))), _raw(((), "messages", (result, {})))]


class _Stream:
    def __init__(
        self,
        events: list[dict[str, object]],
        error: Exception | None,
        *,
        block: bool = False,
    ) -> None:
        self.events = list(events)
        self.error = error
        self.block = block
        self.closed = False
        self.started = asyncio.Event()

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> dict[str, object]:
        if self.events:
            return self.events.pop(0)
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _Agent:
    def __init__(self, stream: _Stream) -> None:
        self.stream = stream
        self.prompts: list[Any] = []

    def astream_events(self, payload, **_kwargs):
        self.prompts.append(payload["messages"][-1]["content"])
        return self.stream


class _Runtime:
    def __init__(
        self,
        project_root: Path,
        events: list[dict[str, object]] | None = None,
        error: Exception | None = None,
        *,
        block: bool = False,
        reflection: bool = False,
    ) -> None:
        self.project_root = project_root
        self.agent = _Agent(_Stream(events or [], error, block=block))
        self.agent_requests = 0
        self.config = SimpleNamespace(
            recursion_limit=50,
            model_name="model",
            model_choices=("model",),
            extensions=SimpleNamespace(
                agent_reflection=ReflectionConfig(enabled=reflection),
                chainlit_reasoning_steps_enabled=False,
                chainlit_tool_steps_enabled=False,
                chainlit_reasoning_mode_enabled=False,
                chainlit_model_mode_enabled=False,
                chainlit_chronological_ui_enabled=True,
                chainlit_generative_ui_enabled=False,
                chainlit_response_actions=(),
            ),
        )

    async def get_agent(self, *_args, **_kwargs):
        self.agent_requests += 1
        return self.agent

    def resolve_chainlit_command(self, _name: str):
        return None


class _Session:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


class _Message:
    sent: ClassVar[list[_Message]] = []

    def __init__(self, content: str = "", author: str = "Assistant", **kwargs: Any) -> None:
        self.id = f"message-{len(self.sent)}"
        self.content = content
        self.author = author
        self.elements = list(kwargs.get("elements", []))
        self.actions: list[Any] = []
        self.metadata: dict[str, Any] = {}

    async def send(self) -> _Message:
        self.sent.append(self)
        return self

    async def update(self) -> None:
        return None


class _Step:
    def __init__(self, **kwargs: Any) -> None:
        self.name = kwargs.get("name")
        self.input = ""
        self.output = ""

    async def __aenter__(self) -> _Step:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


@pytest.fixture
def chainlit_turn(monkeypatch):
    _Message.sent = []
    notifier_calls: list[dict[str, Any]] = []
    scheduled: list[str] = []

    class _Notifier:
        async def schedule_from_state(self, *, thread_id: str) -> None:
            scheduled.append(thread_id)

    def get_notifier(**kwargs: Any) -> _Notifier:
        notifier_calls.append(kwargs)
        return _Notifier()

    async def get_run_task_list(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(main.cl, "Message", _Message)
    monkeypatch.setattr(main.cl, "Step", _Step)
    monkeypatch.setattr(main.cl, "user_session", _Session())
    monkeypatch.setattr(main, "get_run_task_list", get_run_task_list)
    monkeypatch.setattr(main, "get_async_task_notifier", get_notifier)
    settings = SimpleNamespace(
        thread_id="thread-1",
        show_reasoning_stream=False,
        show_tool_calls=False,
    )

    async def run(runtime: _Runtime, prompt: str, **kwargs: Any) -> None:
        await main._run_agent_turn(
            runtime=runtime,
            settings=settings,
            agent_prompt=prompt,
            effective_reasoning_level="medium",
            effective_model_name="model",
            reasoning_level_is_explicit=False,
            mcp_session_id=None,
            **kwargs,
        )

    reflections: list[Any] = []

    async def ask_reflection(**kwargs: Any) -> None:
        reflections.append(kwargs["proposal"])

    monkeypatch.setattr(main, "ask_to_save_reflection_lesson", ask_reflection)
    return SimpleNamespace(
        run=run,
        settings=settings,
        notifier_calls=notifier_calls,
        scheduled=scheduled,
        reflections=reflections,
    )


def _output_file(tmp_path: Path) -> Path:
    output = tmp_path / ".files" / "outputs" / "report.csv"
    output.parent.mkdir(parents=True)
    output.write_text("a,b\n")
    return output


@pytest.mark.anyio
async def test_failed_turn_attaches_generated_files_to_error_message(
    chainlit_turn, tmp_path: Path
) -> None:
    output = _output_file(tmp_path)
    runtime = _Runtime(
        tmp_path,
        _write_file_events("/workspace/.files/outputs/report.csv"),
        RuntimeError("agent crashed"),
    )

    await chainlit_turn.run(runtime, "write a report")

    error_message = _Message.sent[-1]
    assert (error_message.author, error_message.content) == (
        "System",
        "RuntimeError: agent crashed",
    )
    assert [element.path for element in error_message.elements] == [
        output.as_posix()
    ]
    assert runtime.agent.stream.closed
    assert chainlit_turn.notifier_calls == []


@pytest.mark.anyio
async def test_completed_turn_sends_response_and_schedules_notifier(
    chainlit_turn, tmp_path: Path
) -> None:
    runtime = _Runtime(tmp_path, [_token("All done.")])

    await chainlit_turn.run(runtime, "hello")

    assert runtime.agent.prompts == ["hello"]
    assert [message.content for message in _Message.sent] == ["All done."]
    assert chainlit_turn.notifier_calls[0]["agent"] is runtime.agent
    assert chainlit_turn.scheduled == ["thread-1"]


@pytest.mark.anyio
async def test_unknown_typed_slash_command_is_an_error(
    chainlit_turn, tmp_path: Path
) -> None:
    runtime = _Runtime(tmp_path, [_token("unused")])

    await chainlit_turn.run(runtime, "/workspace/api.py explain")

    assert runtime.agent_requests == 0
    assert [(message.author, message.content) for message in _Message.sent] == [
        (
            "System",
            "Unknown command `/workspace/api.py`.\n"
            "Use a configured command from startup or send a normal prompt.",
        )
    ]


@pytest.mark.anyio
async def test_response_action_text_is_not_resolved_as_a_command(
    chainlit_turn, tmp_path: Path
) -> None:
    runtime = _Runtime(tmp_path, [_token("Summary.")])

    await chainlit_turn.run(
        runtime,
        "/workspace/api.py summarize",
        resolve_commands=False,
        display_prompt="",
    )

    assert runtime.agent.prompts == ["/workspace/api.py summarize"]
    assert [message.content for message in _Message.sent] == ["Summary."]


@pytest.mark.anyio
async def test_on_message_swallows_a_stopped_turn(monkeypatch) -> None:
    async def cancelled_turn(_message: Any) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(main, "_handle_message", cancelled_turn)

    await main.on_message(SimpleNamespace(content="hi"))


@pytest.mark.anyio
async def test_stopping_a_message_turn_cancels_cleanly(
    chainlit_turn, monkeypatch, tmp_path: Path
) -> None:
    runtime = _Runtime(tmp_path, [_token("partial")], block=True, reflection=True)
    cancelled: list[bool] = []

    class _RecordingBridge(main.ChainlitEventBridge):
        async def cancel(self) -> None:
            cancelled.append(True)
            await super().cancel()

    async def get_runtime() -> _Runtime:
        return runtime

    monkeypatch.setattr(main, "ChainlitEventBridge", _RecordingBridge)
    monkeypatch.setattr(main, "get_runtime_or_notify", get_runtime)
    monkeypatch.setattr(main, "coerce_settings", lambda *_a, **_k: chainlit_turn.settings)
    monkeypatch.setattr(main, "resolve_reasoning_level_for_message", lambda *_a, **_k: "medium")
    monkeypatch.setattr(main, "resolve_model_name_for_message", lambda *_a, **_k: "model")
    monkeypatch.setattr(main, "current_mcp_session_id", lambda: None)
    monkeypatch.setattr(main, "message_uploaded_rag_files", lambda _m: [])
    monkeypatch.setattr(main, "message_uploaded_image_parts", lambda _m: [])
    monkeypatch.setattr(main, "message_uploaded_image_names", lambda _m: ())
    monkeypatch.setattr(main, "unsupported_uploaded_image_names", lambda _m: ())
    monkeypatch.setattr(main, "message_has_reasoning_level_override", lambda *_a, **_k: False)
    monkeypatch.setattr(main, "settings_reasoning_level_is_explicit", lambda *_a, **_k: False)

    # A correction prompt would produce a reflection proposal if the turn finished.
    message = SimpleNamespace(content="That was wrong, fix it", command=None)
    task = asyncio.create_task(main.on_message(message))
    await asyncio.wait_for(runtime.agent.stream.started.wait(), timeout=5)
    task.cancel()
    await task  # the stop is swallowed; nothing is raised out of on_message

    assert cancelled == [True]
    assert runtime.agent.stream.closed
    assert chainlit_turn.reflections == []
    assert chainlit_turn.notifier_calls == []
    assert chainlit_turn.scheduled == []


@pytest.mark.anyio
async def test_completed_turn_attaches_runner_generated_files_to_response(
    chainlit_turn, tmp_path: Path
) -> None:
    output = _output_file(tmp_path)
    runtime = _Runtime(
        tmp_path,
        [
            *_write_file_events("/workspace/.files/outputs/report.csv"),
            _token("Report written."),
        ],
    )

    await chainlit_turn.run(runtime, "write a report")

    response = _Message.sent[-1]
    assert response.content == "Report written."
    assert [element.path for element in response.elements] == [output.as_posix()]
