"""Test Chainlit bridge streaming behavior for LangGraph event chunks."""

from __future__ import annotations

import json
import weakref
from pathlib import Path
from typing import Any, ClassVar

import pytest

from chainagents.interfaces.chainlit import bridge as chainlit_bridge
from chainagents.interfaces.chainlit.bridge import ChainlitEventBridge, RunTaskList
from chainagents.events.stream import AgentStreamEventAdapter
from chainagents.exports.generated_files import GeneratedFileDescriptor
from chainagents.interfaces.chainlit.renderer import ChainlitTurnRenderer

_FEEDS: weakref.WeakKeyDictionary[ChainlitEventBridge, tuple[AgentStreamEventAdapter, ChainlitTurnRenderer]] = (
    weakref.WeakKeyDictionary()
)


async def _feed(bridge: ChainlitEventBridge, raw_event: dict[str, Any]) -> None:
    """Send one raw LangGraph event down the production path.

    The runner's ``AgentStreamEventAdapter`` normalises the event and the
    ``ChainlitTurnRenderer`` renders it into ``bridge``, one adapter per bridge
    as in a real turn.
    """
    if bridge not in _FEEDS:
        renderer = ChainlitTurnRenderer(lambda _prompt: bridge, prompt=bridge.prompt)
        renderer.bridge = bridge
        _FEEDS[bridge] = (AgentStreamEventAdapter(prompt=bridge.prompt), renderer)
    adapter, renderer = _FEEDS[bridge]
    for event in adapter.events_from_raw_event(raw_event):
        await renderer.on_event(event)


class _AnthropicThinkingToken:
    """Provide an internal helper for Anthropic thinking token."""

    type = "AIMessageChunk"
    additional_kwargs: ClassVar[dict[str, str]] = {}
    tool_call_chunks: ClassVar[list[dict[str, str]]] = []

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

    instances: ClassVar[list[_Message]] = []

    def __init__(self, content: str = "", author: str | None = None, **_kwargs: Any) -> None:
        """Initialize the message instance.

        Args:
            content: Message or document content to process.
            author: The author value.
            _kwargs: The kwargs value.
        """
        self.content = content
        self.author = author
        self.elements = list(_kwargs.get("elements", []) or [])
        self.id = f"message-{len(self.instances) + 1}"
        self.tokens: list[str] = []
        self.actions: list[Any] = []
        self.send_count = 0
        self.update_count = 0
        self.instances.append(self)

    async def send(self) -> _Message:
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

    instances: ClassVar[list[_Step]] = []

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


class _CustomElement:
    """Provide an internal helper for Chainlit custom elements."""

    instances: ClassVar[list[_CustomElement]] = []

    def __init__(self, name: str, props: dict[str, Any], display: str = "inline", **_kwargs: Any) -> None:
        """Initialize the custom element test double."""
        self.name = name
        self.props = props
        self.display = display
        self.update_count = 0
        self.remove_count = 0
        self.instances.append(self)

    async def update(self) -> None:
        """Record update calls on the test double."""
        self.update_count += 1

    async def remove(self) -> None:
        """Record remove calls on the test double."""
        self.remove_count += 1


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


class _ToolCallChunkToken:
    """Provide an internal helper for streamed tool call chunks."""

    type = "AIMessageChunk"
    content = ""
    additional_kwargs: ClassVar[dict[str, str]] = {}

    def __init__(self, chunk: dict[str, Any]) -> None:
        """Initialize the token with one tool-call chunk."""
        self.tool_call_chunks = [chunk]


@pytest.fixture(autouse=True)
def _patch_chainlit_tasks(monkeypatch) -> None:
    """Patch Chainlit task classes with local test doubles.

    Args:
        monkeypatch: The monkeypatch value.
    """
    _Message.instances.clear()
    _Step.instances.clear()
    _CustomElement.instances.clear()
    monkeypatch.setattr(chainlit_bridge.cl, "TaskStatus", _TaskStatus)
    monkeypatch.setattr(chainlit_bridge.cl, "Task", _Task)
    monkeypatch.setattr(chainlit_bridge.cl, "Step", _Step)
    monkeypatch.setattr(chainlit_bridge.cl, "Message", _Message)
    monkeypatch.setattr(chainlit_bridge.cl, "CustomElement", _CustomElement)


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
async def test_hidden_action_turn_hides_prompt_in_steps_and_keeps_response_context(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def capture_actions(_message: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(chainlit_bridge, "attach_response_export_actions", capture_actions)
    bridge = ChainlitEventBridge(
        prompt="A hidden request with private text",
        display_prompt="",
        export_label="Summarize",
        response_actions=("configured action",),
    )

    await bridge._stream_reasoning("main-agent", "thinking")
    await bridge._stream_response("Visible answer")
    await bridge.finish()

    assert _Step.instances[0].input == ""
    assert captured["prompt"] == "A hidden request with private text"
    assert captured["export_label"] == "Summarize"
    assert captured["response_actions"] == ("configured action",)
    assert _Message.instances[0].content == "Visible answer"


@pytest.mark.anyio
async def test_cancelled_turn_closes_steps_and_marks_task_list_stopped() -> None:
    task_list = _TaskList()
    run_task_list = RunTaskList(task_list)  # type: ignore[arg-type]
    bridge = ChainlitEventBridge(prompt="hidden", run_task_list=run_task_list)

    await bridge.start()
    await bridge._stream_reasoning("main-agent", "thinking")
    await bridge.cancel()

    assert _Step.instances[0].end is not None
    assert task_list.status == "Stopped"
    assert all(task.status != _TaskStatus.RUNNING for task in task_list.tasks)


@pytest.mark.anyio
async def test_final_response_attaches_the_runner_generated_files(monkeypatch) -> None:
    """The bridge attaches exactly the generated files the runner resolved."""
    captured: dict[str, Any] = {}

    def capture_export_actions(_message: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    bridge = ChainlitEventBridge(prompt="create a report")
    monkeypatch.setattr(
        chainlit_bridge,
        "attach_response_export_actions",
        capture_export_actions,
    )
    files = [
        GeneratedFileDescriptor(
            name="summary.csv",
            mime_type="text/csv",
            size_bytes=4,
            download_url="/api/generated-files/summary.csv",
            path=Path("/project/.files/outputs/summary.csv"),
        )
    ]

    await bridge._stream_response("Created the report.")
    await bridge.finish(files)

    assert captured["generated_files"] == files


@pytest.mark.anyio
async def test_reasoning_after_tool_call_starts_a_new_chronological_step() -> None:
    """Verify that reasoning after tool call starts a new chronological step."""
    bridge = ChainlitEventBridge(prompt="hello")

    await bridge._stream_reasoning("main-agent", "first thought")
    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (
                        _ToolCallChunkToken(
                            {"id": "call-1", "name": "read_file", "args": '{"path":"README.md"}'}
                        ),
                        {},
                    ),
                )
            },
        }
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
async def test_tool_call_input_accumulates_chunks_by_index_when_id_is_missing() -> None:
    """Verify Chainlit tool steps keep complete args when later chunks omit ids."""
    bridge = ChainlitEventBridge(prompt="hello")

    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (
                        _ToolCallChunkToken(
                            {
                                "id": "call-1",
                                "index": 0,
                                "name": "read_file",
                                "args": "{",
                            }
                        ),
                        {},
                    ),
                )
            },
        }
    )
    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (
                        _ToolCallChunkToken(
                            {
                                "index": 0,
                                "args": '"path":"README.md"}',
                            }
                        ),
                        {},
                    ),
                )
            },
        }
    )

    assert len(_Step.instances) == 1
    assert _Step.instances[0].name == "main-agent · read_file"
    assert _Step.instances[0].input == '{\n  "path": "README.md"\n}'


@pytest.mark.anyio
async def test_tool_call_step_rekeys_when_real_id_replaces_synthetic_id() -> None:
    """Verify Chainlit keeps one tool step when a real id arrives later."""
    task_list = _TaskList()
    run_task_list = RunTaskList(task_list)  # type: ignore[arg-type]
    bridge = ChainlitEventBridge(prompt="hello", run_task_list=run_task_list)

    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (
                        _ToolCallChunkToken(
                            {
                                "index": 0,
                                "name": "read_file",
                                "args": "{",
                            }
                        ),
                        {},
                    ),
                )
            },
        }
    )
    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (
                        _ToolCallChunkToken(
                            {
                                "id": "call-1",
                                "index": 0,
                                "args": '"path":"README.md"}',
                            }
                        ),
                        {},
                    ),
                )
            },
        }
    )
    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (_ToolMessage(tool_call_id="call-1", content="file contents"), {}),
                )
            },
        }
    )

    assert len(_Step.instances) == 1
    assert _Step.instances[0].name == "main-agent · read_file"
    assert _Step.instances[0].input == '{\n  "path": "README.md"\n}'
    assert _Step.instances[0].output == "file contents"
    assert _Step.instances[0].end is not None
    assert [task.title for task in task_list.tasks] == ["read_file: README.md"]
    assert task_list.tasks[0].status == _TaskStatus.DONE
    assert task_list.tasks[0].forId == _Step.instances[0].id


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
    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (
                        _ToolCallChunkToken(
                            {"id": "call-1", "name": "read_file", "args": '{"path":"README.md"}'}
                        ),
                        {},
                    ),
                )
            },
        }
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
    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (
                        _ToolCallChunkToken(
                            {"id": "call-1", "name": "read_file", "args": '{"path":"README.md"}'}
                        ),
                        {},
                    ),
                )
            },
        }
    )
    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (_ToolMessage(), {}),
                )
            },
        }
    )
    await bridge._stream_response("Final answer")
    await bridge.finish()

    assert _Step.instances == []
    assert len(_Message.instances) == 1
    assert _Message.instances[0].content == "Final answer"
    assert [task.title for task in task_list.tasks] == ["final response"]


@pytest.mark.anyio
async def test_chainlit_bridge_shows_summarization_status() -> None:
    """Verify that chainlit bridge shows summarization status."""
    bridge = ChainlitEventBridge(prompt="hello")

    await _feed(
        bridge,
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


@pytest.mark.anyio
async def test_bridge_renders_whitelisted_ui_message_as_custom_element() -> None:
    """Verify whitelisted LangGraph UI events render Chainlit custom elements."""
    bridge = ChainlitEventBridge(prompt="hello")

    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    "custom",
                    {
                        "type": "ui",
                        "id": "panel-1",
                        "name": "GeneratedPanel",
                        "props": {"title": "Build result"},
                        "metadata": {"source": "main-agent"},
                    },
                ),
            },
        }
    )

    assert _CustomElement.instances == []
    assert _Message.instances == []

    await bridge.finish()

    assert len(_CustomElement.instances) == 1
    assert _CustomElement.instances[0].name == "GeneratedPanel"
    assert _CustomElement.instances[0].props == {"title": "Build result"}
    assert _CustomElement.instances[0].display == "inline"
    assert len(_Message.instances) == 1
    assert _Message.instances[0].elements == [_CustomElement.instances[0]]
    assert _Message.instances[0].send_count == 1


@pytest.mark.anyio
async def test_bridge_sends_generated_ui_after_final_response(monkeypatch) -> None:
    """Verify generated UI panels appear after the final answer message."""
    bridge = ChainlitEventBridge(prompt="hello")
    monkeypatch.setattr(
        chainlit_bridge,
        "attach_response_export_actions",
        lambda *args, **kwargs: None,
    )

    await bridge._stream_response("Final answer")
    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    "custom",
                    {
                        "type": "ui",
                        "id": "panel-1",
                        "name": "GeneratedPanel",
                        "props": {"title": "Build result"},
                    },
                ),
            },
        }
    )

    assert _Message.instances == []

    await bridge.finish()

    assert len(_Message.instances) == 2
    assert _Message.instances[0].content == "Final answer"
    assert _Message.instances[0].elements == []
    assert _Message.instances[1].content == ""
    assert _Message.instances[1].elements == [_CustomElement.instances[0]]
    assert _Message.instances[0].send_count == 1
    assert _Message.instances[1].send_count == 1


@pytest.mark.anyio
async def test_bridge_updates_existing_ui_element_by_id() -> None:
    """Verify repeat UI ids update the existing custom element props."""
    bridge = ChainlitEventBridge(prompt="hello")

    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    "custom",
                    {
                        "type": "ui",
                        "id": "panel-1",
                        "name": "GeneratedPanel",
                        "props": {"title": "First"},
                    },
                ),
            },
        }
    )
    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    "custom",
                    {
                        "type": "ui",
                        "id": "panel-1",
                        "name": "GeneratedPanel",
                        "props": {"title": "Updated"},
                    },
                ),
            },
        }
    )

    await bridge.finish()

    assert len(_CustomElement.instances) == 1
    assert _CustomElement.instances[0].props == {"title": "Updated"}
    assert _CustomElement.instances[0].update_count == 0
    assert len(_Message.instances) == 1


@pytest.mark.anyio
async def test_bridge_updates_shared_ui_element_registry_across_instances() -> None:
    """Verify generated UI ids persist when bridge instances share a registry."""
    generated_ui_elements: dict[str, _CustomElement] = {}
    first_bridge = ChainlitEventBridge(
        prompt="hello",
        generated_ui_elements=generated_ui_elements,
    )
    second_bridge = ChainlitEventBridge(
        prompt="next",
        generated_ui_elements=generated_ui_elements,
    )

    await _feed(
        first_bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    "custom",
                    {
                        "type": "ui",
                        "id": "panel-1",
                        "name": "GeneratedPanel",
                        "props": {"title": "First"},
                    },
                ),
            },
        }
    )
    await first_bridge.finish()

    await _feed(
        second_bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    "custom",
                    {
                        "type": "ui",
                        "id": "panel-1",
                        "name": "GeneratedPanel",
                        "props": {"title": "Second"},
                    },
                ),
            },
        }
    )

    await second_bridge.finish()

    assert len(_CustomElement.instances) == 1
    assert _CustomElement.instances[0].props == {"title": "Second"}
    assert _CustomElement.instances[0].update_count == 1
    assert len(_Message.instances) == 1
    assert generated_ui_elements == {"panel-1": _CustomElement.instances[0]}


@pytest.mark.anyio
async def test_bridge_removes_existing_ui_element_by_id() -> None:
    """Verify remove-ui events remove the tracked custom element."""
    bridge = ChainlitEventBridge(prompt="hello")

    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    "custom",
                    {
                        "type": "ui",
                        "id": "panel-1",
                        "name": "GeneratedPanel",
                        "props": {"title": "First"},
                    },
                ),
            },
        }
    )
    await bridge.finish()

    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {"chunk": ("custom", {"type": "remove-ui", "id": "panel-1"})},
        }
    )

    assert _CustomElement.instances[0].remove_count == 1


@pytest.mark.anyio
async def test_bridge_ignores_unknown_or_disabled_ui_components() -> None:
    """Verify unregistered components and disabled generative UI do not render."""
    bridge = ChainlitEventBridge(prompt="hello")
    disabled_bridge = ChainlitEventBridge(prompt="hello", generative_ui_enabled=False)

    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    "custom",
                    {
                        "type": "ui",
                        "id": "panel-1",
                        "name": "UnknownPanel",
                        "props": {"title": "First"},
                    },
                ),
            },
        }
    )
    await _feed(
        disabled_bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    "custom",
                    {
                        "type": "ui",
                        "id": "panel-2",
                        "name": "GeneratedPanel",
                        "props": {"title": "Second"},
                    },
                ),
            },
        }
    )

    assert _CustomElement.instances == []
    assert _Message.instances == []


@pytest.mark.anyio
async def test_subagent_write_todos_reach_the_task_list_through_events() -> None:
    """A namespaced (subagent) write_todos call and result update the task list."""
    task_list = _TaskList()
    run_task_list = RunTaskList(task_list)  # type: ignore[arg-type]
    bridge = ChainlitEventBridge(prompt="plan it", run_task_list=run_task_list)
    namespace = ("task:subagent-1",)
    metadata = {"lc_agent_name": "researcher"}

    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    namespace,
                    "messages",
                    (
                        _ToolCallChunkToken(
                            {
                                "id": "call-todos",
                                "name": "write_todos",
                                "args": json.dumps(
                                    {"todos": [{"content": "Research", "status": "in_progress"}]}
                                ),
                            }
                        ),
                        metadata,
                    ),
                )
            },
        },
    )

    assert [(task.title, task.status) for task in task_list.tasks] == [
        ("Research", _TaskStatus.RUNNING)
    ]

    await _feed(
        bridge,
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    namespace,
                    "updates",
                    {
                        "tools": {
                            "todos": [{"content": "Research", "status": "completed"}],
                            "messages": [
                                _ToolMessage(
                                    name="write_todos",
                                    tool_call_id="call-todos",
                                    content=(
                                        "Updated todo list to "
                                        "[{'content': 'Research', 'status': 'completed'}]"
                                    ),
                                )
                            ],
                        }
                    },
                )
            },
        },
    )

    assert [(task.title, task.status) for task in task_list.tasks] == [
        ("Research", _TaskStatus.DONE)
    ]
