from __future__ import annotations

from typing import Any

import pytest

import chainlit_bridge
from chainlit_bridge import ChainlitEventBridge, RunTaskList


class _TaskStatus:
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    READY = "ready"


class _Task:
    def __init__(self, title: str, status: str, forId: str | None = None) -> None:
        self.title = title
        self.status = status
        self.forId = forId


class _TaskList:
    def __init__(self) -> None:
        self.status = "Ready"
        self.tasks: list[_Task] = []
        self.send_count = 0

    async def send(self) -> None:
        self.send_count += 1


class _ResponseMessage:
    id = "message-1"

    def __init__(self) -> None:
        self.tokens: list[str] = []
        self.update_count = 0

    async def stream_token(self, token: str) -> None:
        self.tokens.append(token)

    async def update(self) -> None:
        self.update_count += 1


class _Step:
    instances: list["_Step"] = []

    def __init__(
        self,
        name: str,
        type: str,
        default_open: bool = False,
        **_kwargs: Any,
    ) -> None:
        self.name = name
        self.type = type
        self.default_open = default_open
        self.input: Any = None
        self.output: Any = None
        self.start: Any = None
        self.end: Any = None
        self.send_count = 0
        self.update_count = 0
        self.id = f"step-{len(self.instances) + 1}"
        self.instances.append(self)

    async def send(self) -> None:
        self.send_count += 1

    async def update(self) -> None:
        self.update_count += 1


@pytest.fixture(autouse=True)
def _patch_chainlit_tasks(monkeypatch) -> None:
    _Step.instances.clear()
    monkeypatch.setattr(chainlit_bridge.cl, "TaskStatus", _TaskStatus)
    monkeypatch.setattr(chainlit_bridge.cl, "Task", _Task)
    monkeypatch.setattr(chainlit_bridge.cl, "Step", _Step)


@pytest.mark.anyio
async def test_response_task_starts_once_for_rapid_response_tokens() -> None:
    task_list = _TaskList()
    run_task_list = RunTaskList(task_list)  # type: ignore[arg-type]

    await run_task_list.start(response_for_id="message-1")
    await run_task_list.mark_response_started(for_id="message-1")
    await run_task_list.mark_response_started(for_id="message-1")
    await run_task_list.mark_response_started(for_id="message-1")

    assert task_list.send_count == 2
    assert [task.title for task in task_list.tasks] == [
        "main-agent reasoning",
        "final response",
    ]
    assert task_list.tasks[-1].status == _TaskStatus.RUNNING


@pytest.mark.anyio
async def test_response_stream_batches_fast_chunks_until_finish(monkeypatch) -> None:
    response_message = _ResponseMessage()
    bridge = ChainlitEventBridge(prompt="hello")
    bridge.response_message = response_message  # type: ignore[assignment]

    monkeypatch.setattr(chainlit_bridge.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        chainlit_bridge,
        "attach_response_export_actions",
        lambda *args, **kwargs: None,
    )

    await bridge._stream_response("A")
    await bridge._stream_response("B")
    await bridge._stream_response("C")

    assert response_message.tokens == ["A"]
    assert bridge.response_buffer == "ABC"

    await bridge.finish()

    assert response_message.tokens == ["A", "BC"]
    assert response_message.update_count == 1


@pytest.mark.anyio
async def test_astream_events_chain_stream_tuple_chunk_is_normalized() -> None:
    bridge = ChainlitEventBridge(prompt="hello")
    handled_parts: list[dict[str, Any]] = []

    async def handle_part(part: dict[str, Any]) -> None:
        handled_parts.append(part)

    bridge.handle_part = handle_part  # type: ignore[method-assign]

    await bridge.handle_event(
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    ("tools:abc",),
                    "updates",
                    {"tools": {"messages": []}},
                ),
            },
        }
    )

    assert handled_parts == [
        {
            "type": "updates",
            "ns": ("tools:abc",),
            "data": {"tools": {"messages": []}},
        }
    ]


@pytest.mark.anyio
async def test_astream_events_ignores_non_langgraph_stream_chunks() -> None:
    bridge = ChainlitEventBridge(prompt="hello")
    handled_parts: list[dict[str, Any]] = []

    async def handle_part(part: dict[str, Any]) -> None:
        handled_parts.append(part)

    bridge.handle_part = handle_part  # type: ignore[method-assign]

    await bridge.handle_event(
        {
            "event": "on_chat_model_stream",
            "data": {"chunk": "hello"},
        }
    )
    await bridge.handle_event(
        {
            "event": "on_chain_stream",
            "data": {"chunk": {"output": "not a LangGraph stream part"}},
        }
    )
    await bridge.handle_event(
        {
            "event": "on_chain_stream",
            "parent_ids": ["root-run-id"],
            "data": {
                "chunk": (
                    (),
                    "messages",
                    ("duplicate nested token", {}),
                ),
            },
        }
    )

    assert handled_parts == []


@pytest.mark.anyio
async def test_chainlit_bridge_shows_summarization_status() -> None:
    bridge = ChainlitEventBridge(prompt="hello")

    await bridge.handle_event(
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    "custom",
                    {
                        "kind": "summarization_status",
                        "status": "started",
                        "source": "main-agent",
                        "message": "Conversation summarization triggered.",
                    },
                ),
            },
        }
    )

    assert len(_Step.instances) == 1
    step = _Step.instances[0]
    assert step.name == "main-agent summarization"
    assert step.type == "llm"
    assert step.default_open is True
    assert step.output == "Conversation summarization triggered."
    assert step.send_count == 1
    assert step.update_count == 1
