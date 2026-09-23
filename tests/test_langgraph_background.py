"""Exercise local background-task lifecycle for exported Agent Server graphs."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

import chainagents.runtime.graph as runtime_graph
from chainagents.langgraph.http import app, lifespan


def test_langgraph_http_app_closes_sessions_managers_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []

    class Manager:
        @asynccontextmanager
        async def closing_session(self, session_id: str):
            calls.append(("session", session_id))
            yield

        async def close(self) -> None:
            calls.append(("manager", None))

    manager = Manager()

    class Artifacts:
        async def close_session(self, session_id: str) -> None:
            calls.append(("artifacts-session", session_id))

        async def close(self) -> None:
            calls.append(("artifacts", None))

    monkeypatch.setattr(
        runtime_graph,
        "_STATIC_LARGE_TOOL_RESULT_ARTIFACTS",
        Artifacts(),
        raising=False,
    )
    runtime_graph._STATIC_BACKGROUND_TASK_MANAGERS.add(manager)

    with TestClient(app) as client:
        response = client.delete("/background-tasks/sessions/thread-1")
        assert response.status_code == 200
        assert response.json() == {"closed": True, "thread_id": "thread-1"}
        nested_response = client.delete("/background-tasks/sessions/team/research")
        assert nested_response.status_code == 200
        assert nested_response.json() == {
            "closed": True,
            "thread_id": "team/research",
        }

    assert calls == [
        ("session", "thread-1"),
        ("artifacts-session", "thread-1"),
        ("session", "team/research"),
        ("artifacts-session", "team/research"),
        ("manager", None),
        ("artifacts", None),
    ]
    assert runtime_graph.static_background_task_managers() == ()


@pytest.mark.anyio
async def test_langgraph_lifespan_closes_managers_after_application_error() -> None:
    calls: list[str] = []

    class Manager:
        async def close(self) -> None:
            calls.append("manager")

    runtime_graph._STATIC_BACKGROUND_TASK_MANAGERS.add(Manager())

    with pytest.raises(RuntimeError, match="application failed"):
        async with lifespan(app):
            raise RuntimeError("application failed")

    assert calls == ["manager"]
    assert runtime_graph.static_background_task_managers() == ()


@pytest.mark.anyio
async def test_static_shutdown_preserves_manager_and_artifact_cleanup_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Artifact shutdown must run even when exported manager cleanup fails."""
    calls: list[str] = []

    class Manager:
        async def close(self) -> None:
            calls.append("manager")
            raise RuntimeError("manager cleanup failed")

    class Artifacts:
        async def close(self) -> None:
            calls.append("artifacts")
            raise RuntimeError("artifact cleanup failed")

    monkeypatch.setattr(
        runtime_graph,
        "_STATIC_LARGE_TOOL_RESULT_ARTIFACTS",
        Artifacts(),
    )
    runtime_graph._STATIC_BACKGROUND_TASK_MANAGERS.add(Manager())

    with pytest.raises(BaseExceptionGroup) as captured:
        await runtime_graph.close_static_background_tasks()

    assert calls == ["manager", "artifacts"]
    assert [str(error) for error in captured.value.exceptions] == [
        "manager cleanup failed",
        "artifact cleanup failed",
    ]
    assert runtime_graph.static_background_task_managers() == ()


@pytest.mark.anyio
async def test_exported_session_cleanup_finishes_before_cancellation_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request cancellation must not interrupt coordinated session teardown."""
    calls: list[str] = []
    artifact_started = asyncio.Event()
    release_artifact = asyncio.Event()

    class Manager:
        @asynccontextmanager
        async def closing_session(self, session_id: str):
            calls.append(f"manager-enter:{session_id}")
            try:
                yield
            finally:
                calls.append(f"manager-exit:{session_id}")

    class Artifacts:
        async def close_session(self, session_id: str) -> None:
            calls.append(f"artifact-start:{session_id}")
            artifact_started.set()
            await release_artifact.wait()
            calls.append(f"artifact-end:{session_id}")

    manager = Manager()
    monkeypatch.setattr(
        runtime_graph,
        "_STATIC_LARGE_TOOL_RESULT_ARTIFACTS",
        Artifacts(),
    )
    runtime_graph._STATIC_BACKGROUND_TASK_MANAGERS.add(manager)
    cleanup = asyncio.create_task(
        runtime_graph.close_static_background_session("thread")
    )
    await artifact_started.wait()

    cleanup.cancel()
    await asyncio.sleep(0)

    assert not cleanup.done()
    assert calls == ["manager-enter:thread", "artifact-start:thread"]

    release_artifact.set()
    with pytest.raises(asyncio.CancelledError):
        await cleanup

    assert calls == [
        "manager-enter:thread",
        "artifact-start:thread",
        "artifact-end:thread",
        "manager-exit:thread",
    ]
    runtime_graph._STATIC_BACKGROUND_TASK_MANAGERS.discard(manager)
