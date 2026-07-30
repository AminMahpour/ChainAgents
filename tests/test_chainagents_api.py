"""Test the FastAPI access layer for ChainAgents."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

import chainagents_api
from chainagents.runtime import core as runtime_core
from chainagents.runtime.token_usage import TokenUsageFileCallbackHandler


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


class _DisconnectAgent:
    """Simulate a live LangGraph stream closed by an API consumer."""

    def __init__(self) -> None:
        self.root_run_id = uuid4()

    def astream_events(self, payload, *, config, version, stream_mode, subgraphs):
        async def events():
            for callback in config["callbacks"]:
                callback.on_chain_start(
                    {},
                    payload,
                    run_id=self.root_run_id,
                )
            yield _raw_event(((), "messages", (_Token("partial"), {})))
            await asyncio.sleep(60)

        return events()


class _FakeRuntime:
    """Provide the runtime surface required by the API module."""

    def __init__(
        self,
        agent: _FakeAgent,
        *,
        project_root: Path | None = None,
    ) -> None:
        """Initialize the fake runtime."""
        self.agent = agent
        self.project_root = project_root or Path.cwd()
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


def test_invoke_runs_prompt_through_agent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the invoke endpoint returns the final streamed response."""
    runtime_root = tmp_path / "runtime-root"
    fallback_root = tmp_path / "fallback-root"
    monkeypatch.setattr(runtime_core, "PROJECT_ROOT", fallback_root)
    agent = _FakeAgent(
        [
            _raw_event(((), "messages", (_Token("Hello"), {}))),
            _raw_event(((), "messages", (_Token("Hello world"), {}))),
        ]
    )
    runtime = _FakeRuntime(agent, project_root=runtime_root)
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
                "reasoning_level_is_explicit": True,
                "thread_id": "thread-1",
                "async_subagent_url_override": None,
                "mcp_session_id": "session-1",
            },
        }
    ]
    assert agent.payload == {"messages": [{"role": "user", "content": "hello"}]}
    assert len(agent.config["callbacks"]) == 1
    assert isinstance(
        agent.config["callbacks"][0],
        TokenUsageFileCallbackHandler,
    )
    assert {key: value for key, value in agent.config.items() if key != "callbacks"} == {
        "configurable": {"thread_id": "thread-1"},
        "recursion_limit": 100,
    }
    agent.config["callbacks"][0].on_chain_end({}, run_id=uuid4())
    assert (runtime_root / ".files" / "token-usage.jsonl").exists()
    assert not (fallback_root / ".files" / "token-usage.jsonl").exists()


def test_closing_api_event_stream_records_cancellation(tmp_path: Path) -> None:
    """Dropping a disconnected stream must preserve its partial usage record."""
    agent = _DisconnectAgent()
    runtime = _FakeRuntime(agent, project_root=tmp_path)
    context = chainagents_api.AgentRunContext(
        prompt="hello",
        thread_id="thread-1",
        model_name="fake-model",
        reasoning_level="medium",
        reasoning_level_is_explicit=False,
        async_subagent_url=None,
        mcp_session_id=None,
    )
    events = chainagents_api._iter_agent_events(runtime, context)

    async def consume_then_disconnect():
        first = await anext(events)
        await events.aclose()
        return first

    first = asyncio.run(consume_then_disconnect())

    assert first.text == "partial"
    record = json.loads(
        (tmp_path / ".files" / "token-usage.jsonl").read_text(encoding="utf-8")
    )
    assert record["request_id"] == str(agent.root_run_id)
    assert record["status"] == "cancelled"


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
            "previous_tool_call_id": "",
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
            "previous_tool_call_id": "",
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
