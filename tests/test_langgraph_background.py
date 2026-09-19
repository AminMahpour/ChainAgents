"""Exercise local background-task lifecycle for exported Agent Server graphs."""

from fastapi.testclient import TestClient

import chainagents.runtime.graph as runtime_graph
from chainagents.langgraph.http import app


def test_langgraph_http_app_closes_sessions_and_managers() -> None:
    calls: list[tuple[str, str | None]] = []

    class Manager:
        async def close_session(self, session_id: str) -> None:
            calls.append(("session", session_id))

        async def close(self) -> None:
            calls.append(("manager", None))

    manager = Manager()
    runtime_graph._STATIC_BACKGROUND_TASK_MANAGERS.add(manager)

    with TestClient(app) as client:
        response = client.delete("/background-tasks/sessions/thread-1")
        assert response.status_code == 200
        assert response.json() == {"closed": True, "thread_id": "thread-1"}

    assert calls == [("session", "thread-1"), ("manager", None)]
    assert runtime_graph.static_background_task_managers() == ()
