"""Test the FastAPI access layer for ChainAgents."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

import chainagents_api


class _Token:
    """Provide a minimal streamed AI token for API tests."""

    type = "AIMessageChunk"
    additional_kwargs: dict[str, str] = {}
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, content: str = "") -> None:
        """Initialize the token instance."""
        self.content = content


def _raw_event(chunk: object) -> dict[str, object]:
    """Build a raw LangGraph stream event."""
    return {"event": "on_chain_stream", "data": {"chunk": chunk}}


class _FakeAgent:
    """Capture agent invocations and return configured stream events."""

    def __init__(self, events: list[dict[str, object]]) -> None:
        """Initialize the fake agent."""
        self.events = events
        self.payload: dict[str, Any] | None = None
        self.config: dict[str, Any] | None = None

    def astream_events(self, payload, *, config, version, stream_mode, subgraphs):
        """Return the configured async event stream."""
        self.payload = payload
        self.config = config

        async def events():
            for event in self.events:
                yield event

        return events()


class _FakeRuntime:
    """Provide the runtime surface required by the API module."""

    def __init__(self, agent: _FakeAgent) -> None:
        """Initialize the fake runtime."""
        self.agent = agent
        self.requests: list[dict[str, Any]] = []
        self.config = SimpleNamespace(
            default_reasoning="medium",
            model_name="fake-model",
            model_provider="ollama",
            model_choices=("fake-model", "other-model"),
            agent_state="stateful",
            recursion_limit=100,
            persistence_mode="memory",
        )

    async def get_agent(self, *args, **kwargs):
        """Return the fake agent and capture selection arguments."""
        self.requests.append({"args": args, "kwargs": kwargs})
        return self.agent


def test_health_reports_ok() -> None:
    """Verify the health endpoint reports that the API process is alive."""
    runtime = _FakeRuntime(_FakeAgent([]))
    app = chainagents_api.create_app(runtime=runtime)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_reports_runtime_configuration() -> None:
    """Verify the status endpoint exposes resolved runtime configuration."""
    runtime = _FakeRuntime(_FakeAgent([]))
    app = chainagents_api.create_app(runtime=runtime)

    with TestClient(app) as client:
        response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json() == {
        "model": "fake-model",
        "model_provider": "ollama",
        "model_choices": ["fake-model", "other-model"],
        "default_reasoning": "medium",
        "agent_state": "stateful",
        "recursion_limit": 100,
        "persistence_mode": "memory",
    }


def test_invoke_runs_prompt_through_agent() -> None:
    """Verify the invoke endpoint returns the final streamed response."""
    agent = _FakeAgent(
        [
            _raw_event(((), "messages", (_Token("Hello"), {}))),
            _raw_event(((), "messages", (_Token("Hello world"), {}))),
        ]
    )
    runtime = _FakeRuntime(agent)
    app = chainagents_api.create_app(runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/api/agent/invoke",
            json={
                "prompt": "hello",
                "thread_id": "thread-1",
                "model": "other-model",
                "reasoning": "high",
                "mcp_session_id": "session-1",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "response": "Hello world",
        "thread_id": "thread-1",
        "model": "other-model",
        "reasoning": "high",
    }
    assert runtime.requests == [
        {
            "args": ("high",),
            "kwargs": {
                "model_name": "other-model",
                "thread_id": "thread-1",
                "async_subagent_url_override": None,
                "mcp_session_id": "session-1",
            },
        }
    ]
    assert agent.payload == {"messages": [{"role": "user", "content": "hello"}]}
    assert agent.config == {
        "configurable": {"thread_id": "thread-1"},
        "recursion_limit": 100,
    }


def test_invoke_requires_thread_id() -> None:
    """Verify API callers must provide a thread ID for checkpoint isolation."""
    runtime = _FakeRuntime(_FakeAgent([]))
    app = chainagents_api.create_app(runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/api/agent/invoke",
            json={"prompt": "hello"},
        )

    assert response.status_code == 422
    assert runtime.requests == []


def test_stream_returns_ndjson_agent_events() -> None:
    """Verify the stream endpoint returns normalized agent events as NDJSON."""
    agent = _FakeAgent(
        [
            _raw_event(((), "messages", (_Token("Hello"), {}))),
            _raw_event(((), "messages", (_Token("Hello world"), {}))),
        ]
    )
    runtime = _FakeRuntime(agent)
    app = chainagents_api.create_app(runtime=runtime)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/agent/stream",
            json={"prompt": "hello", "thread_id": "thread-1"},
        ) as response:
            lines = [json.loads(line) for line in response.iter_lines()]

    assert response.status_code == 200
    assert lines == [
        {
            "kind": "response_delta",
            "source": "main-agent",
            "text": "Hello",
            "tool_call_id": "",
            "tool_name": "",
            "tool_args": "",
            "tool_args_delta": "",
            "tool_result": "",
            "status": "",
            "thread_id": "thread-1",
            "model": "fake-model",
            "reasoning": "medium",
        },
        {
            "kind": "response_delta",
            "source": "main-agent",
            "text": " world",
            "tool_call_id": "",
            "tool_name": "",
            "tool_args": "",
            "tool_args_delta": "",
            "tool_result": "",
            "status": "",
            "thread_id": "thread-1",
            "model": "fake-model",
            "reasoning": "medium",
        },
        {
            "kind": "done",
            "thread_id": "thread-1",
            "model": "fake-model",
            "reasoning": "medium",
        },
    ]


def test_stream_requires_thread_id() -> None:
    """Verify streamed API runs also require a caller-provided thread ID."""
    runtime = _FakeRuntime(_FakeAgent([]))
    app = chainagents_api.create_app(runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/api/agent/stream",
            json={"prompt": "hello"},
        )

    assert response.status_code == 422
    assert runtime.requests == []
