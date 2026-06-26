"""Test normalized LangGraph stream event handling."""

from __future__ import annotations

from types import SimpleNamespace


from agent_stream_events import AgentStreamEvent, AgentStreamEventAdapter


class _Token:
    type = "AIMessageChunk"
    additional_kwargs: dict[str, str] = {}
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, content: str = "") -> None:
        self.content = content


class _ReasoningToken:
    type = "AIMessageChunk"
    content = ""
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, reasoning: str) -> None:
        self.additional_kwargs = {"reasoning_content": reasoning}


class _AnthropicThinkingToken:
    type = "AIMessageChunk"
    additional_kwargs: dict[str, str] = {}
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, content: object) -> None:
        self.content = content


class _ToolCallChunkToken:
    type = "AIMessageChunk"
    content = ""
    additional_kwargs: dict[str, str] = {}

    def __init__(self, chunk: dict[str, str]) -> None:
        self.tool_call_chunks = [chunk]


class _ToolMessage:
    type = "tool"
    name = "read_file"
    status = "success"
    tool_call_id = "call-1"

    def __init__(self, content: str) -> None:
        self.content = content


def _raw_event(
    chunk: object, *, parent_ids: list[str] | None = None
) -> dict[str, object]:
    return {
        "event": "on_chain_stream",
        "parent_ids": parent_ids or [],
        "data": {"chunk": chunk},
    }


def test_adapter_streams_response_deltas_from_main_message_chunks() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    first = adapter.events_from_raw_event(
        _raw_event(((), "messages", (_Token("Hello"), {})))
    )
    second = adapter.events_from_raw_event(
        _raw_event(((), "messages", (_Token("Hello world"), {})))
    )

    assert first == [
        AgentStreamEvent(kind="response_delta", source="main-agent", text="Hello")
    ]
    assert second == [
        AgentStreamEvent(kind="response_delta", source="main-agent", text=" world")
    ]


def test_adapter_streams_reasoning_deltas_by_source() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    first = adapter.events_from_raw_event(
        _raw_event(((), "messages", (_ReasoningToken("thinking"), {})))
    )
    second = adapter.events_from_raw_event(
        _raw_event(((), "messages", (_ReasoningToken("thinking more"), {})))
    )

    assert first == [
        AgentStreamEvent(kind="reasoning_delta", source="main-agent", text="thinking")
    ]
    assert second == [
        AgentStreamEvent(kind="reasoning_delta", source="main-agent", text=" more")
    ]


def test_adapter_streams_anthropic_thinking_blocks_as_reasoning() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    events = adapter.events_from_raw_event(
        _raw_event(
            (
                (),
                "messages",
                (
                    _AnthropicThinkingToken(
                        [
                            {
                                "type": "thinking",
                                "thinking": "checking Claude reasoning",
                            },
                            {"type": "redacted_thinking", "data": "signature"},
                        ]
                    ),
                    {},
                ),
            )
        )
    )

    assert events == [
        AgentStreamEvent(
            kind="reasoning_delta",
            source="main-agent",
            text="checking Claude reasoning",
        )
    ]


def test_adapter_omits_anthropic_thinking_blocks_from_response_text() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    events = adapter.events_from_raw_event(
        _raw_event(
            (
                (),
                "messages",
                (
                    _AnthropicThinkingToken(
                        [
                            {"type": "thinking", "thinking": "private reasoning"},
                            {"type": "text", "text": "Final answer"},
                        ]
                    ),
                    {},
                ),
            )
        )
    )

    assert events == [
        AgentStreamEvent(
            kind="reasoning_delta",
            source="main-agent",
            text="private reasoning",
        ),
        AgentStreamEvent(
            kind="response_delta",
            source="main-agent",
            text="Final answer",
        ),
    ]


def test_adapter_accumulates_tool_call_arguments_and_deduplicates_results() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    start = adapter.events_from_raw_event(
        _raw_event(
            (
                (),
                "messages",
                (
                    _ToolCallChunkToken(
                        {"id": "call-1", "name": "read_file", "args": '{"path":"REA'}
                    ),
                    {},
                ),
            )
        )
    )
    update = adapter.events_from_raw_event(
        _raw_event(
            (
                (),
                "messages",
                (_ToolCallChunkToken({"id": "call-1", "args": 'DME.md"}'}), {}),
            )
        )
    )
    result = adapter.events_from_raw_event(
        _raw_event(((), "messages", (_ToolMessage("content"), {})))
    )
    duplicate_result = adapter.events_from_raw_event(
        _raw_event(((), "messages", (_ToolMessage("content"), {})))
    )

    assert start == [
        AgentStreamEvent(
            kind="tool_call",
            source="main-agent",
            tool_call_id="call-1",
            tool_name="read_file",
            tool_args='{"path":"REA',
            tool_args_delta='{"path":"REA',
            status="start",
        )
    ]
    assert update == [
        AgentStreamEvent(
            kind="tool_call",
            source="main-agent",
            tool_call_id="call-1",
            tool_name="read_file",
            tool_args='{"path":"README.md"}',
            tool_args_delta='DME.md"}',
            status="update",
        )
    ]
    assert result == [
        AgentStreamEvent(
            kind="tool_result",
            source="main-agent",
            tool_call_id="call-1",
            tool_name="read_file",
            tool_result="content",
            status="success",
        )
    ]
    assert duplicate_result == []


def test_adapter_reuses_tool_call_index_after_completed_result() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    first = adapter.events_from_raw_event(
        _raw_event(
            (
                (),
                "messages",
                (
                    _ToolCallChunkToken(
                        {
                            "id": "call-1",
                            "index": 0,
                            "name": "read_file",
                            "args": '{"path":"one.md"}',
                        }
                    ),
                    {},
                ),
            )
        )
    )
    result = adapter.events_from_raw_event(
        _raw_event(((), "messages", (_ToolMessage("first result"), {})))
    )
    second = adapter.events_from_raw_event(
        _raw_event(
            (
                (),
                "messages",
                (
                    _ToolCallChunkToken(
                        {
                            "id": "call-2",
                            "index": 0,
                            "name": "read_file",
                            "args": '{"path":"two.md"}',
                        }
                    ),
                    {},
                ),
            )
        )
    )

    assert first == [
        AgentStreamEvent(
            kind="tool_call",
            source="main-agent",
            tool_call_id="call-1",
            tool_name="read_file",
            tool_args='{"path":"one.md"}',
            tool_args_delta='{"path":"one.md"}',
            status="start",
        )
    ]
    assert result == [
        AgentStreamEvent(
            kind="tool_result",
            source="main-agent",
            tool_call_id="call-1",
            tool_name="read_file",
            tool_result="first result",
            status="success",
        )
    ]
    assert second == [
        AgentStreamEvent(
            kind="tool_call",
            source="main-agent",
            tool_call_id="call-2",
            tool_name="read_file",
            tool_args='{"path":"two.md"}',
            tool_args_delta='{"path":"two.md"}',
            status="start",
        )
    ]


def test_adapter_uses_update_chunks_for_non_streamed_final_response() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")
    human = SimpleNamespace(type="human", content="hello")
    assistant = SimpleNamespace(type="ai", content="Final answer")

    events = adapter.events_from_raw_event(
        _raw_event(("updates", {"agent": {"messages": [human, assistant]}}))
    )

    assert events == [
        AgentStreamEvent(
            kind="response_delta",
            source="main-agent",
            text="Final answer",
        )
    ]


def test_adapter_streams_summarization_status_events() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    events = adapter.events_from_raw_event(
        _raw_event(
            (
                "custom",
                {
                    "kind": "summarization_status",
                    "status": "started",
                    "source": "main-agent",
                    "message": "Conversation summarization triggered.",
                },
            )
        )
    )

    assert events == [
        AgentStreamEvent(
            kind="summarization_status",
            source="main-agent",
            status="started",
            text="Conversation summarization triggered.",
        )
    ]


def test_adapter_ignores_nested_chain_events() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    events = adapter.events_from_raw_event(
        _raw_event(
            ((), "messages", (_Token("Nested"), {})),
            parent_ids=["parent"],
        )
    )

    assert events == []
