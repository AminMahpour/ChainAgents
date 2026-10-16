"""HTTP API tests for per-agent context usage."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from chainagents.interfaces.api import app as chainagents_api
from chainagents.runtime.context_usage import (
    ContextUsageRegistry,
    ContextUsageSnapshot,
)


def _snapshot(agent: str, kind: str, at_utc: str) -> ContextUsageSnapshot:
    return ContextUsageSnapshot(
        agent=agent,
        kind=kind,
        model="test-model",
        window=1000,
        used=250,
        categories={
            "system": 40,
            "tools": 60,
            "summary": 0,
            "messages": 100,
            "tool_results": 50,
        },
        trigger=800,
        keep=100,
        exact=True,
        at_utc=at_utc,
        last_summarized_at_utc=None,
        status="active",
    )


def _client(runtime: Any) -> TestClient:
    app = chainagents_api.create_app(runtime=runtime)
    return TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1")


def _registry() -> tuple[ContextUsageRegistry, ContextUsageSnapshot, ContextUsageSnapshot]:
    registry = ContextUsageRegistry()
    sub = _snapshot("researcher", "subagent", "2026-10-15T10:00:00+00:00")
    main = _snapshot("main", "main", "2026-10-15T09:00:00+00:00")
    registry.record("t1", "sub-1", sub)
    registry.record("t1", "main", main)
    return registry, main, sub


def test_context_usage_endpoint_lists_agents() -> None:
    registry, main, sub = _registry()
    with _client(SimpleNamespace(context_usage=registry)) as client:
        response = client.get("/api/context-usage?thread_id=t1")
    assert response.status_code == 200
    assert response.json() == {"agents": [main.to_payload(), sub.to_payload()]}


def test_context_usage_endpoint_unknown_session_is_empty() -> None:
    registry, _, _ = _registry()
    with _client(SimpleNamespace(context_usage=registry)) as client:
        response = client.get("/api/context-usage?thread_id=other")
    assert response.status_code == 200
    assert response.json() == {"agents": []}


def test_context_usage_requires_thread_id() -> None:
    with _client(SimpleNamespace(context_usage=ContextUsageRegistry())) as client:
        assert client.get("/api/context-usage").status_code == 422


def test_context_usage_disabled_returns_404() -> None:
    with _client(SimpleNamespace()) as client:
        assert client.get("/api/context-usage?thread_id=t1").status_code == 404


def test_status_reports_context_usage_feature() -> None:
    from test_chainagents_api import _FakeAgent, _FakeRuntime

    runtime = _FakeRuntime(_FakeAgent([]))
    runtime.context_usage = ContextUsageRegistry()
    with _client(runtime) as client:
        payload = client.get("/api/status").json()
    assert payload["features"]["context_usage"] is True


def test_context_usage_endpoint_requires_auth_in_token_mode(monkeypatch) -> None:
    monkeypatch.setenv("CHAINAGENTS_API_TOKEN", "test-token")
    registry, _, _ = _registry()
    with _client(SimpleNamespace(context_usage=registry)) as client:
        url = "/api/context-usage?thread_id=t1"
        assert client.get(url).status_code == 401
        ok = client.get(url, headers={"Authorization": "Bearer test-token"})
        assert ok.status_code == 200
