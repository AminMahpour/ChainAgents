"""Test correction reflection proposal behavior."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any

import pytest

import chainagents_api
import chainagents_cli
from chainagents.events.stream import AgentStreamEvent
from chainagents.runtime.reflection import ReflectionCollector, ReflectionConfig


class _Token:
    """Provide a minimal streamed AI token for reflection tests."""

    type = "AIMessageChunk"
    additional_kwargs: dict[str, str] = {}
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, content: str = "") -> None:
        """Initialize the token instance."""
        self.content = content


class _ToolMessage:
    """Provide a minimal streamed tool message for reflection tests."""

    type = "tool"
    name = "read_file"
    tool_call_id = "call-1"

    def __init__(self, content: str, *, status: str = "error") -> None:
        """Initialize the tool message."""
        self.content = content
        self.status = status


def _raw_event(chunk: object) -> dict[str, object]:
    """Build a raw LangGraph stream event."""
    return {"event": "on_chain_stream", "data": {"chunk": chunk}}


class _FakeAgent:
    """Capture agent invocations and return configured stream events."""

    def __init__(self, events: list[dict[str, object]]) -> None:
        """Initialize the fake agent."""
        self.events = events

    def astream_events(self, payload, *, config, version, stream_mode, subgraphs):
        """Return the configured async event stream."""

        async def events():
            for event in self.events:
                yield event

        return events()


class _FakeRuntime:
    """Provide the runtime surface required by CLI and API tests."""

    def __init__(self, agent: _FakeAgent, *, reflection_enabled: bool = True) -> None:
        """Initialize the fake runtime."""
        self.agent = agent
        self.config = SimpleNamespace(
            default_reasoning="medium",
            model_name="fake-model",
            model_provider="ollama",
            model_choices=("fake-model",),
            agent_state="stateful",
            recursion_limit=100,
            persistence_mode="memory",
            extensions=SimpleNamespace(
                agent_reflection=ReflectionConfig(enabled=reflection_enabled)
            ),
        )

    async def get_agent(self, *args, **kwargs):
        """Return the fake agent."""
        return self.agent


def test_reflection_collector_detects_correction_phrase() -> None:
    """Verify correction phrases create compact memory proposals."""
    collector = ReflectionCollector(
        ReflectionConfig(enabled=True),
        prompt="That was wrong; the generated files belong under .files/outputs.",
    )
    proposal = collector.build_proposal()

    assert proposal is not None
    assert proposal.reason == "correction"
    assert proposal.memory_file == "/memories/AGENTS.md"
    assert "That was wrong" in proposal.lesson
    assert len(proposal.lesson) <= 700


def test_reflection_collector_ignores_recovered_tool_failure() -> None:
    """Verify tool failures followed by a final response do not create proposals."""
    collector = ReflectionCollector(ReflectionConfig(enabled=True), prompt="try it")
    collector.record_event(
        AgentStreamEvent(
            kind="tool_result",
            source="main-agent",
            tool_name="read_file",
            tool_result="missing file",
            status="error",
        )
    )
    collector.record_event(
        AgentStreamEvent(kind="response_delta", source="main-agent", text="I recovered.")
    )

    assert collector.build_proposal() is None


def test_reflection_collector_detects_unrecovered_tool_failure() -> None:
    """Verify unrecovered failed tool calls create proposals."""
    collector = ReflectionCollector(ReflectionConfig(enabled=True), prompt="try it")
    collector.record_event(
        AgentStreamEvent(
            kind="tool_result",
            source="main-agent",
            tool_name="read_file",
            tool_result="missing file",
            status="error",
        )
    )

    proposal = collector.build_proposal()

    assert proposal is not None
    assert proposal.reason == "tool_failure"
    assert "read_file" in proposal.lesson


def test_reflection_collector_ignores_disabled_config() -> None:
    """Verify disabled reflection config is a no-op."""
    collector = ReflectionCollector(
        ReflectionConfig(enabled=False),
        prompt="That was wrong.",
    )

    assert collector.build_proposal() is None


def test_api_stream_emits_reflection_proposal_event() -> None:
    """Verify API streaming exposes reflection proposals without writing memory."""
    agent = _FakeAgent(
        [
            _raw_event(
                (
                    (),
                    "messages",
                    (_ToolMessage("file does not exist", status="error"), {}),
                )
            ),
        ]
    )
    runtime = _FakeRuntime(agent)
    app = chainagents_api.create_app(runtime=runtime)

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/agent/stream",
            json={"prompt": "try it", "thread_id": "thread-1"},
        ) as response:
            lines = [json.loads(line) for line in response.iter_lines()]

    assert response.status_code == 200
    assert [line["kind"] for line in lines] == [
        "tool_result",
        "reflection_proposal",
        "done",
    ]
    assert lines[1]["proposal"]["reason"] == "tool_failure"
    assert lines[1]["proposal"]["memory_file"] == "/memories/AGENTS.md"


@pytest.mark.anyio
async def test_cli_json_includes_reflection_proposal() -> None:
    """Verify CLI JSON includes reflection proposals without writing memory."""
    agent = _FakeAgent(
        [
            _raw_event(((), "messages", (_Token("That was wrong."), {}))),
        ]
    )
    runtime = _FakeRuntime(agent)
    args = chainagents_cli.parse_args(["--prompt", "hello", "--json"])
    stdout = io.StringIO()

    code = await chainagents_cli.run_agent_prompt(
        runtime,  # type: ignore[arg-type]
        args,
        prompt="hello",
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    payload = json.loads(stdout.getvalue())
    assert payload["reflection_proposal"]["reason"] == "correction"
    assert payload["reflection_proposal"]["memory_file"] == "/memories/AGENTS.md"
