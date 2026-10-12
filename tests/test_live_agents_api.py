"""HTTP API tests for live agents."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from chainagents.interfaces.api import app as chainagents_api


class _Record:
    def __init__(self, ref: str) -> None:
        self.ref = ref

    def to_payload(self) -> dict[str, Any]:
        return {"ref": self.ref, "status": "stopped"}


class _FakeManager:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def list(self, session_id, *, include_stopped=False):
        self.calls.append(("list", session_id, include_stopped))
        return [{"ref": "a1"}]

    def send_from_user(self, session_id, ref, body):
        self.calls.append(("send", session_id, ref, body))
        if ref == "missing":
            raise KeyError(ref)
        if len(body) > 100:
            raise ValueError("body too long")
        return SimpleNamespace(id="msg-1")

    async def teardown(self, session_id, ref, *, reason="requested"):
        self.calls.append(("teardown", session_id, ref, reason))
        if ref == "missing":
            raise KeyError(ref)
        return _Record(ref)


def _client(live_agents: Any) -> TestClient:
    runtime = SimpleNamespace(live_agents=live_agents)
    app = chainagents_api.create_app(runtime=runtime)
    return TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")


def test_list_live_agents() -> None:
    manager = _FakeManager()
    with _client(manager) as client:
        response = client.get("/api/live-agents?thread_id=t1&include_stopped=true")
    assert response.status_code == 200
    assert response.json() == {"agents": [{"ref": "a1"}]}
    assert manager.calls == [("list", "t1", True)]


def test_send_message() -> None:
    manager = _FakeManager()
    with _client(manager) as client:
        response = client.post(
            "/api/live-agents/a1/messages?thread_id=t1", json={"body": "hi"}
        )
    assert response.status_code == 200
    assert response.json() == {"message_id": "msg-1"}
    assert manager.calls == [("send", "t1", "a1", "hi")]


def test_delete_agent() -> None:
    manager = _FakeManager()
    with _client(manager) as client:
        response = client.delete("/api/live-agents/a1?thread_id=t1")
    assert response.status_code == 200
    assert response.json() == {"ref": "a1", "status": "stopped"}
    assert manager.calls == [("teardown", "t1", "a1", "user")]


def test_disabled_returns_404() -> None:
    with _client(None) as client:
        assert client.get("/api/live-agents?thread_id=t1").status_code == 404
        assert (
            client.post(
                "/api/live-agents/a1/messages?thread_id=t1", json={"body": "x"}
            ).status_code
            == 404
        )
        assert client.delete("/api/live-agents/a1?thread_id=t1").status_code == 404


def test_unknown_agent_returns_404() -> None:
    with _client(_FakeManager()) as client:
        assert (
            client.post(
                "/api/live-agents/missing/messages?thread_id=t1", json={"body": "x"}
            ).status_code
            == 404
        )
        assert client.delete("/api/live-agents/missing?thread_id=t1").status_code == 404


def test_blank_body_returns_422() -> None:
    manager = _FakeManager()
    with _client(manager) as client:
        for body in ("", "   \n"):
            response = client.post(
                "/api/live-agents/a1/messages?thread_id=t1", json={"body": body}
            )
            assert response.status_code == 422
    assert manager.calls == []


def test_broker_value_error_returns_422() -> None:
    with _client(_FakeManager()) as client:
        response = client.post(
            "/api/live-agents/a1/messages?thread_id=t1", json={"body": "x" * 200}
        )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_posted_message_reply_visible_in_list() -> None:
    import asyncio

    import httpx
    from test_live_agents_worker import FakeGraph

    from chainagents.runtime.live_agents.config import SpawnRequest
    from chainagents.runtime.live_agents.manager import LiveAgentManager
    from chainagents.runtime.messaging import MessageBroker
    from chainagents.runtime.types import (
        LiveAgentsConfig,
        LiveAgentTemplate,
        MessagingConfig,
    )

    async def factory(_spec, _session_id):
        return FakeGraph(["here is the answer"])

    config = LiveAgentsConfig(
        enabled=True,
        templates=(
            LiveAgentTemplate(name="researcher", description="d", system_prompt="p"),
        ),
    )
    manager = LiveAgentManager(
        config, MessageBroker(MessagingConfig(enabled=True)), factory
    )
    runtime = SimpleNamespace(live_agents=manager)
    app = chainagents_api.create_app(runtime=runtime)
    app.state.runtime = runtime  # Normally set by the lifespan.
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 50000))
    try:
        record, _spec = await manager.spawn("t1", SpawnRequest(template="researcher"))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1"
        ) as client:
            posted = await client.post(
                f"/api/live-agents/{record.name}/messages?thread_id=t1",
                json={"body": "question?"},
            )
            assert posted.status_code == 200

            async def replied() -> dict[str, Any]:
                while True:
                    listed = await client.get("/api/live-agents?thread_id=t1")
                    agent = listed.json()["agents"][0]
                    if agent["last_reply"] is not None:
                        return agent
                    await asyncio.sleep(0.01)

            agent = await asyncio.wait_for(replied(), 2)
        assert agent["last_reply"]["body"] == "here is the answer"
        assert isinstance(agent["last_reply"]["at"], float)
    finally:
        await manager.close()
