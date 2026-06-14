"""Test Chainlit bridge streaming behavior for LangGraph event chunks."""

from __future__ import annotations

from typing import Any

import pytest

import chainlit_bridge
from chainlit_bridge import ChainlitEventBridge, RunTaskList


class _AnthropicThinkingToken:
    """Provide an internal helper for Anthropic thinking token."""

    type = "AIMessageChunk"
    additional_kwargs: dict[str, str] = {}
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, thinking: str) -> None:
        """Initialize the Anthropic thinking token instance.

        Args:
            thinking: The thinking delta value.
        """
        self.content = [
            {
                "type": "thinking",
                "thinking": thinking,
                "index": 0,
            }
        ]


def test_reasoning_text_from_token_extracts_anthropic_thinking_block() -> None:
    """Verify that Anthropic thinking content blocks are treated as reasoning."""
    token = _AnthropicThinkingToken("checking Claude reasoning")

    assert chainlit_bridge.reasoning_text_from_token(token) == "checking Claude reasoning"


def test_stringify_content_omits_anthropic_thinking_block() -> None:
    """Verify that Anthropic thinking content blocks do not render as answer text."""
    token = _AnthropicThinkingToken("hidden reasoning")

    assert chainlit_bridge.stringify_content(token.content) == ""


class _TaskStatus:
    """Provide an internal helper for task status."""

    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    READY = "ready"


class _Task:
    """Provide an internal helper for task."""

    def __init__(self, title: str, status: str, forId: str | None = None) -> None:
        """Initialize the task instance.

        Args:
            title: The title value.
            status: The status value.
            forId: The for ID value.
        """
        self.title = title
        self.status = status
        self.forId = forId


class _TaskList:
    """Provide an internal helper for task list."""

    def __init__(self) -> None:
        """Initialize the task list instance."""
        self.status = "Ready"
        self.tasks: list[_Task] = []
        self.send_count = 0

    async def send(self) -> None:
        """Record send calls on the test double."""
        self.send_count += 1


class _ResponseMessage:
    """Provide an internal helper for response message.

    Attributes:
        id: The ID value.
    """

    id = "message-1"

    def __init__(self) -> None:
        """Initialize the response message instance."""
        self.tokens: list[str] = []
        self.update_count = 0

    async def stream_token(self, token: str) -> None:
        """Stream token.

        Args:
            token: Streamed model token to inspect.
        """
        self.tokens.append(token)

    async def update(self) -> None:
        """Record update calls on the test double."""
        self.update_count += 1


class _Message:
    """Provide an internal helper for message.

    Attributes:
        instances: The instances value.
    """

    instances: list["_Message"] = []

    def __init__(self, content: str = "", author: str | None = None, **_kwargs: Any) -> None:
        """Initialize the message instance.

        Args:
            content: Message or document content to process.
            author: The author value.
            _kwargs: The kwargs value.
        """
        self.content = content
        self.author = author
        self.id = f"message-{len(self.instances) + 1}"
        self.tokens: list[str] = []
        self.actions: list[Any] = []
        self.send_count = 0
        self.update_count = 0
        self.instances.append(self)

    async def send(self) -> "_Message":
        """Record send calls on the test double.

        Returns:
            The sent message or element.
        """
        self.send_count += 1
        return self

    async def stream_token(self, token: str) -> None:
        """Stream token.

        Args:
            token: Streamed model token to inspect.
        """
        self.tokens.append(token)

    async def update(self) -> None:
        """Record update calls on the test double."""
        self.update_count += 1


class _Step:
    """Provide an internal helper for step.

    Attributes:
        instances: The instances value.
    """

    instances: list["_Step"] = []

    def __init__(
        self,
        name: str,
        type: str,
        default_open: bool = False,
        **_kwargs: Any,
    ) -> None:
        """Initialize the step instance.

        Args:
            name: The name value.
            type: The type value.
            default_open: The default open value.
            _kwargs: The kwargs value.
        """
        self.name = name
        self.type = type
        self.default_open = default_open
        self.input: Any = None
        self.output: Any = None
        self.start: Any = None
        self.end: Any = None
        self.tokens: list[str] = []
        self.send_count = 0
        self.update_count = 0
        self.id = f"step-{len(self.instances) + 1}"
        self.instances.append(self)

    async def send(self) -> None:
        """Record send calls on the test double."""
        self.send_count += 1

    async def stream_token(self, token: str) -> None:
        """Stream token.

        Args:
            token: Streamed model token to inspect.
        """
        self.tokens.append(token)

    async def update(self) -> None:
        """Record update calls on the test double."""
        self.update_count += 1


class _ToolMessage:
    """Provide an internal helper for a completed tool message."""

    type = "tool"

    def __init__(
        self,
        *,
        name: str = "read_file",
        tool_call_id: str = "call-1",
        content: str = "tool result",
        status: str = "",
    ) -> None:
        """Initialize the tool message instance."""
        self.name = name
        self.tool_call_id = tool_call_id
        self.content = content
        self.status = status


@pytest.fixture(autouse=True)
def _patch_chainlit_tasks(monkeypatch) -> None:
    """Patch Chainlit task classes with local test doubles.

    Args:
        monkeypatch: The monkeypatch value.
    """
    _Message.instances.clear()
    _Step.instances.clear()
    monkeypatch.setattr(chainlit_bridge.cl, "TaskStatus", _TaskStatus)
    monkeypatch.setattr(chainlit_bridge.cl, "Task", _Task)
    monkeypatch.setattr(chainlit_bridge.cl, "Step", _Step)
    monkeypatch.setattr(chainlit_bridge.cl, "Message", _Message)


@pytest.mark.anyio
async def test_response_task_starts_once_for_rapid_response_tokens() -> None:
    """Verify that response task starts once for rapid response tokens."""
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
async def test_response_message_is_created_on_finish_after_reasoning_steps(monkeypatch) -> None:
    """Verify that response message is created on finish after reasoning steps.

    Args:
        monkeypatch: The monkeypatch value.
    """
    task_list = _TaskList()
    run_task_list = RunTaskList(task_list)  # type: ignore[arg-type]
    bridge = ChainlitEventBridge(prompt="hello", run_task_list=run_task_list)
    monkeypatch.setattr(
        chainlit_bridge,
        "attach_response_export_actions",
        lambda *args, **kwargs: None,
    )

    await bridge.start()

    assert _Message.instances == []
    assert [task.title for task in task_list.tasks] == ["main-agent reasoning"]
    assert task_list.tasks[0].forId is None

    await bridge._stream_reasoning("main-agent", "thinking")

    assert _Message.instances == []
    assert len(_Step.instances) == 1
    assert _Step.instances[0].name == "main-agent reasoning"

    await bridge._stream_response("Final answer")

    assert _Message.instances == []
    assert bridge.response_buffer == "Final answer"

    await bridge.finish()

    assert len(_Message.instances) == 1
    assert _Message.instances[0].send_count == 1
    assert _Message.instances[0].content == "Final answer"
    assert _Message.instances[0].tokens == []
    assert _Message.instances[0].update_count == 1
    assert _Step.instances[0].end is not None
    assert [task.title for task in task_list.tasks] == [
        "main-agent reasoning",
        "final response",
    ]
    assert task_list.tasks[-1].status == _TaskStatus.DONE
    assert task_list.tasks[-1].forId == _Message.instances[0].id


@pytest.mark.anyio
async def test_reasoning_after_tool_call_starts_a_new_chronological_step() -> None:
    """Verify that reasoning after tool call starts a new chronological step."""
    bridge = ChainlitEventBridge(prompt="hello")

    await bridge._stream_reasoning("main-agent", "first thought")
    await bridge._stream_tool_call(
        "main-agent",
        {"id": "call-1", "name": "read_file", "args": '{"path":"README.md"}'},
    )
    await bridge._stream_reasoning("main-agent", "first thought second thought")

    assert [step.name for step in _Step.instances] == [
        "main-agent reasoning",
        "main-agent · read_file",
        "main-agent reasoning",
    ]
    assert _Step.instances[0].tokens == ["first thought"]
    assert _Step.instances[0].end is not None
    assert _Step.instances[2].tokens == [" second thought"]


@pytest.mark.anyio
async def test_response_stream_buffers_fast_chunks_until_finish(monkeypatch) -> None:
    """Verify that response stream buffers fast chunks until finish.

    Args:
        monkeypatch: The monkeypatch value.
    """
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

    assert bridge.response_buffer == "ABC"
    assert response_message.tokens == []

    await bridge.finish()

    assert response_message.tokens == ["ABC"]
    assert response_message.update_count == 1


@pytest.mark.anyio
async def test_non_chronological_mode_streams_response_immediately() -> None:
    """Verify that non chronological mode streams response immediately."""
    bridge = ChainlitEventBridge(prompt="hello", chronological_ui_enabled=False)

    await bridge._stream_response("A")
    await bridge._stream_response("AB")

    assert len(_Message.instances) == 1
    assert _Message.instances[0].tokens == ["A", "B"]


@pytest.mark.anyio
async def test_non_chronological_mode_keeps_reasoning_step_open_across_tool_call() -> None:
    """Verify that non chronological mode keeps reasoning step open across tool call."""
    bridge = ChainlitEventBridge(prompt="hello", chronological_ui_enabled=False)

    await bridge._stream_reasoning("main-agent", "first thought")
    await bridge._stream_tool_call(
        "main-agent",
        {"id": "call-1", "name": "read_file", "args": '{"path":"README.md"}'},
    )
    await bridge._stream_reasoning("main-agent", "first thought second thought")

    assert [step.name for step in _Step.instances] == [
        "main-agent reasoning",
        "main-agent · read_file",
    ]
    assert _Step.instances[0].tokens == ["first thought", " second thought"]


@pytest.mark.anyio
async def test_bridge_can_hide_reasoning_and_tool_ui_elements(monkeypatch) -> None:
    """Verify that hidden reasoning and tool UI still streams the final response.

    Args:
        monkeypatch: The monkeypatch value.
    """
    task_list = _TaskList()
    run_task_list = RunTaskList(
        task_list,  # type: ignore[arg-type]
        reasoning_steps_enabled=False,
        tool_steps_enabled=False,
    )
    bridge = ChainlitEventBridge(
        prompt="hello",
        run_task_list=run_task_list,
        reasoning_steps_enabled=False,
        tool_steps_enabled=False,
    )
    monkeypatch.setattr(
        chainlit_bridge,
        "attach_response_export_actions",
        lambda *args, **kwargs: None,
    )

    await bridge.start()
    assert task_list.status == "Running..."
    assert task_list.tasks == []

    await bridge._stream_reasoning("main-agent", "first thought")
    await bridge._stream_tool_call(
        "main-agent",
        {"id": "call-1", "name": "read_file", "args": '{"path":"README.md"}'},
    )
    await bridge._complete_tool_step("main-agent", _ToolMessage())
    await bridge._stream_response("Final answer")
    await bridge.finish()

    assert _Step.instances == []
    assert len(_Message.instances) == 1
    assert _Message.instances[0].content == "Final answer"
    assert [task.title for task in task_list.tasks] == ["final response"]


@pytest.mark.anyio
async def test_astream_events_chain_stream_tuple_chunk_is_normalized() -> None:
    """Verify that astream events chain stream tuple chunk is normalized."""
    bridge = ChainlitEventBridge(prompt="hello")
    handled_parts: list[dict[str, Any]] = []

    async def handle_part(part: dict[str, Any]) -> None:
        """Capture normalized stream parts for assertions.

        Args:
            part: The part value.
        """
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
    """Verify that astream events ignores non langgraph stream chunks."""
    bridge = ChainlitEventBridge(prompt="hello")
    handled_parts: list[dict[str, Any]] = []

    async def handle_part(part: dict[str, Any]) -> None:
        """Capture normalized stream parts for assertions.

        Args:
            part: The part value.
        """
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
    """Verify that chainlit bridge shows summarization status."""
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
