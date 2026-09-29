"""Test normalized LangGraph stream event handling."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import ClassVar


from chainagents.events.stream import (
    AgentStreamEvent,
    AgentStreamEventAdapter,
    langgraph_part_from_event_chunk,
)
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessageChunk


class _Token:
    type = "AIMessageChunk"
    additional_kwargs: ClassVar[dict[str, str]] = {}
    tool_call_chunks: ClassVar[list[dict[str, str]]] = []

    def __init__(self, content: str = "") -> None:
        self.content = content


class _ReasoningToken:
    type = "AIMessageChunk"
    content = ""
    tool_call_chunks: ClassVar[list[dict[str, str]]] = []

    def __init__(self, reasoning: str) -> None:
        self.additional_kwargs = {"reasoning_content": reasoning}


class _AnthropicThinkingToken:
    type = "AIMessageChunk"
    additional_kwargs: ClassVar[dict[str, str]] = {}
    tool_call_chunks: ClassVar[list[dict[str, str]]] = []

    def __init__(self, content: object) -> None:
        self.content = content


class _ToolCallChunkToken:
    type = "AIMessageChunk"
    content = ""
    additional_kwargs: ClassVar[dict[str, str]] = {}

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


def test_adapter_hides_internal_token_limit_retry_notice() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")
    notice = HumanMessageChunk(
        content="Shorten or split the tool call; you have one retry.",
        additional_kwargs={"chainagents_token_limit_retry": True},
    )
    assert adapter.events_from_raw_event(
        _raw_event(((), "messages", (notice, {})))
    ) == []
    assert adapter.events_from_raw_event(
        _raw_event(((), "messages", (_Token("Done"), {})))
    ) == [AgentStreamEvent(kind="response_delta", source="main-agent", text="Done")]


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


def test_adapter_deduplicates_named_nested_result_across_stream_modes() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")
    namespace = ("tools:parent-task-id", "tools:child-task-id")

    named = adapter.events_from_raw_event(
        _raw_event(
            (
                namespace,
                "messages",
                (_ToolMessage("content"), {"lc_agent_name": "reviewer"}),
            )
        )
    )
    repeated = adapter.events_from_raw_event(
        _raw_event(
            (namespace, "updates", {"tools": {"messages": [_ToolMessage("content")]}})
        )
    )

    assert named == [
        AgentStreamEvent(
            kind="tool_result",
            source="reviewer",
            tool_call_id="call-1",
            tool_name="read_file",
            tool_result="content",
            status="success",
        )
    ]
    assert repeated == []


def test_adapter_reuses_named_source_for_update_only_result() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")
    namespace = ("tools:parent-task-id", "tools:child-task-id")

    call = adapter.events_from_raw_event(
        _raw_event(
            (
                namespace,
                "messages",
                (
                    _ToolCallChunkToken({"id": "call-1", "name": "read_file"}),
                    {"lc_agent_name": "reviewer"},
                ),
            )
        )
    )
    result = adapter.events_from_raw_event(
        _raw_event(
            (namespace, "updates", {"tools": {"messages": [_ToolMessage("content")]}})
        )
    )

    assert [event.source for event in call] == ["reviewer"]
    assert result == [
        AgentStreamEvent(
            kind="tool_result",
            source="reviewer",
            tool_call_id="call-1",
            tool_name="read_file",
            tool_result="content",
            status="success",
        )
    ]


def test_adapter_keeps_equal_tool_call_ids_in_separate_namespaces() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")
    namespaces = (("tools:first-task",), ("tools:second-task",))

    results = [
        adapter.events_from_raw_event(
            _raw_event(
                (
                    namespace,
                    "messages",
                    (_ToolMessage("content"), {"lc_agent_name": "worker"}),
                )
            )
        )
        for namespace in namespaces
    ]

    assert [event.source for batch in results for event in batch] == [
        "worker",
        "worker",
    ]


def test_adapter_uses_distinct_run_local_labels_without_agent_metadata() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")
    namespaces = (("tools:first-task",), ("tools:second-task",))

    results = [
        adapter.events_from_raw_event(
            _raw_event(
                (
                    namespace,
                    "updates",
                    {"tools": {"messages": [_ToolMessage("content")]}},
                )
            )
        )
        for namespace in namespaces
    ]

    assert [event.source for batch in results for event in batch] == [
        "subagent 1",
        "subagent 2",
    ]


def test_adapter_labels_real_nested_deepagents_tool_once() -> None:
    class ToolAwareFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def delegate(target: str, call_id: str) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"description": "do the work", "subagent_type": target},
                    "id": call_id,
                }
            ],
        )

    def leaf_action(value: str) -> str:
        """Return a leaf result."""
        return value

    leaf = create_deep_agent(
        model=ToolAwareFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "leaf_action",
                            "args": {"value": "ok"},
                            "id": "leaf-call",
                        }
                    ],
                ),
                AIMessage(content="leaf done"),
            ]
        ),
        tools=[leaf_action],
        name="leaf",
    )
    parent = create_deep_agent(
        model=ToolAwareFakeModel(
            responses=[
                delegate("leaf", "parent-call"),
                AIMessage(content="parent done"),
            ]
        ),
        subagents=[{"name": "leaf", "description": "leaf", "runnable": leaf}],
        name="parent",
    )
    main = create_deep_agent(
        model=ToolAwareFakeModel(
            responses=[delegate("parent", "main-call"), AIMessage(content="main done")]
        ),
        subagents=[{"name": "parent", "description": "parent", "runnable": parent}],
        name="main",
    )

    async def collect() -> tuple[list[AgentStreamEvent], list[tuple[str, ...]]]:
        adapter = AgentStreamEventAdapter(prompt="go")
        events: list[AgentStreamEvent] = []
        leaf_namespaces: list[tuple[str, ...]] = []
        async for raw in main.astream_events(
            {"messages": [{"role": "user", "content": "go"}]},
            version="v2",
            stream_mode=["messages", "updates", "custom"],
            subgraphs=True,
        ):
            if raw.get("event") != "on_chain_stream" or raw.get("parent_ids"):
                continue
            part = langgraph_part_from_event_chunk(raw.get("data", {}).get("chunk"))
            if (
                part is not None
                and part["type"] == "updates"
                and "tools" in part["data"]
            ):
                namespace = part["ns"]
                if len(namespace) == 2:
                    leaf_namespaces.append(namespace)
            events.extend(adapter.events_from_raw_event(raw))
        return events, leaf_namespaces

    events, leaf_namespaces = asyncio.run(collect())
    assert leaf_namespaces and all(
        segment.startswith("tools:") for segment in leaf_namespaces[0]
    )
    assert [
        event.source
        for event in events
        if event.kind == "tool_result" and event.tool_name == "leaf_action"
    ] == ["leaf"]


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


def test_adapter_treats_id_bearing_tool_calls_without_index_independently() -> None:
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
                            "name": "read_file",
                            "args": '{"path":"one.md"}',
                        }
                    ),
                    {},
                ),
            )
        )
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


def test_adapter_starts_real_tool_call_id_after_synthetic_id_migration() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    synthetic_start = adapter.events_from_raw_event(
        _raw_event(
            (
                (),
                "messages",
                (
                    _ToolCallChunkToken(
                        {
                            "index": 0,
                            "name": "read_file",
                            "args": '{"path":"REA',
                        }
                    ),
                    {},
                ),
            )
        )
    )
    real_id_start = adapter.events_from_raw_event(
        _raw_event(
            (
                (),
                "messages",
                (
                    _ToolCallChunkToken(
                        {
                            "id": "call-1",
                            "index": 0,
                            "args": 'DME.md"}',
                        }
                    ),
                    {},
                ),
            )
        )
    )

    assert synthetic_start == [
        AgentStreamEvent(
            kind="tool_call",
            source="main-agent",
            tool_call_id="main-agent:0",
            tool_name="read_file",
            tool_args='{"path":"REA',
            tool_args_delta='{"path":"REA',
            status="start",
        )
    ]
    assert real_id_start == [
        AgentStreamEvent(
            kind="tool_call",
            source="main-agent",
            tool_call_id="call-1",
            previous_tool_call_id="main-agent:0",
            tool_name="read_file",
            tool_args='{"path":"README.md"}',
            tool_args_delta='DME.md"}',
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


def test_adapter_streams_langgraph_ui_message_events() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    events = adapter.events_from_raw_event(
        _raw_event(
            (
                "custom",
                {
                    "type": "ui",
                    "id": "panel-1",
                    "name": "GeneratedPanel",
                    "props": {
                        "title": "Build result",
                        "facts": {"Tests": "passing"},
                    },
                    "metadata": {"source": "main-agent"},
                },
            )
        )
    )

    assert events == [
        AgentStreamEvent(
            kind="ui_message",
            source="main-agent",
            ui_id="panel-1",
            ui_name="GeneratedPanel",
            ui_props={
                "title": "Build result",
                "facts": {"Tests": "passing"},
            },
            ui_metadata={"source": "main-agent"},
        )
    ]


def test_adapter_streams_langgraph_ui_remove_events() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    events = adapter.events_from_raw_event(
        _raw_event(("custom", {"type": "remove-ui", "id": "panel-1"}))
    )

    assert events == [
        AgentStreamEvent(
            kind="ui_remove",
            source="main-agent",
            ui_id="panel-1",
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


def test_langgraph_part_from_namespaced_tuple_chunk_is_normalized() -> None:
    from chainagents.events.stream import langgraph_part_from_event_chunk

    part = langgraph_part_from_event_chunk(
        (("tools:abc",), "updates", {"tools": {"messages": []}})
    )

    assert part == {
        "type": "updates",
        "ns": ("tools:abc",),
        "data": {"tools": {"messages": []}},
    }


def test_adapter_ignores_non_langgraph_stream_events() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    assert adapter.events_from_raw_event(
        {"event": "on_chat_model_stream", "data": {"chunk": "hello"}}
    ) == []
    assert adapter.events_from_raw_event(
        {
            "event": "on_chain_stream",
            "data": {"chunk": {"output": "not a LangGraph stream part"}},
        }
    ) == []
