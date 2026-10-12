"""HTTP API tests for live agents."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

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
