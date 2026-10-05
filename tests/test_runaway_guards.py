"""Regression tests for the guards that stop and surface runaway agent runs."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphInterrupt, GraphRecursionError

from chainagents.runtime.backends import build_deepagent_backend
from chainagents.runtime.background_tasks.manager import BackgroundTaskManager
from chainagents.runtime.constants import SYSTEM_PROMPT
from chainagents.runtime.middleware import (
    RepeatedToolResultGuardMiddleware,
    ToolExecutionResilienceMiddleware,
    build_agent_middleware,
)
from chainagents.runtime.providers import SnowflakeCortexChatOpenAI
from chainagents.runtime.types import BackgroundSubagentConfig


# --- /workspace path mapping ------------------------------------------------


def _mapped(path: str, project_root: Path) -> str:
    middleware = ToolExecutionResilienceMiddleware(project_root=project_root)
    request = ToolCallRequest(
        tool_call={"id": "c", "name": "ls", "args": {"path": path}, "type": "tool_call"},
        tool=SimpleNamespace(name="ls"),
        state={},
        runtime=SimpleNamespace(),
    )
    middleware._map_workspace_path_args(request)
    return request.tool_call["args"]["path"]


@pytest.mark.parametrize("path", ["/workspace", "/workspace/", "/workspace/skills"])
def test_mapped_workspace_paths_reach_the_real_filesystem(tmp_path: Path, path: str) -> None:
    (tmp_path / "skills" / "reviewer").mkdir(parents=True)
    (tmp_path / "skills" / "reviewer" / "SKILL.md").write_text("skill")
    (tmp_path / "README.md").write_text("readme")
    backend = build_deepagent_backend(project_root=tmp_path, include_memories=False)

    for candidate in (path, _mapped(path, tmp_path)):
        result = backend.ls(candidate)
        assert result.entries, f"{candidate} listed no entries"


def test_mapped_skill_file_is_readable(tmp_path: Path) -> None:
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "SKILL.md").write_text("skill body")
    backend = build_deepagent_backend(project_root=tmp_path, include_memories=False)

    result = backend.read(_mapped("/workspace/skills/SKILL.md", tmp_path))

    assert result.error is None
    assert "skill body" in str(result.file_data)


def test_project_root_route_does_not_shadow_outputs_route(tmp_path: Path) -> None:
    backend = build_deepagent_backend(project_root=tmp_path, include_memories=False)
    root = tmp_path.resolve().as_posix()

    routed, key = backend._get_backend_and_key(f"{root}/.files/outputs/report.md")

    assert routed is backend.routes[f"{root}/.files/outputs/"]
    assert key == "/report.md"


def test_system_prompt_does_not_disclose_the_project_root() -> None:
    from chainagents.runtime.constants import PROJECT_ROOT

    assert str(PROJECT_ROOT) not in SYSTEM_PROMPT
    assert "/workspace/" in SYSTEM_PROMPT


# --- non-recoverable tool exceptions ----------------------------------------


def _tool_request() -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"id": "call-1", "name": "task", "args": {}, "type": "tool_call"},
        tool=SimpleNamespace(name="task"),
        state={},
        runtime=SimpleNamespace(),
    )


@pytest.mark.parametrize(
    "exc", [GraphRecursionError("limit"), GraphInterrupt(()), asyncio.CancelledError()]
)
def test_non_recoverable_exceptions_propagate_from_async_tools(exc: BaseException) -> None:
    async def handler(_request: ToolCallRequest) -> ToolMessage:
        raise exc

    with pytest.raises(type(exc)):
        asyncio.run(ToolExecutionResilienceMiddleware().awrap_tool_call(_tool_request(), handler))


@pytest.mark.parametrize("exc", [GraphRecursionError("limit"), GraphInterrupt(())])
def test_non_recoverable_exceptions_propagate_from_sync_tools(exc: BaseException) -> None:
    def handler(_request: ToolCallRequest) -> ToolMessage:
        raise exc

    with pytest.raises(type(exc)):
        ToolExecutionResilienceMiddleware().wrap_tool_call(_tool_request(), handler)


# --- repeated tool result guard ---------------------------------------------


def _call_pair(index: int, args: dict[str, Any], result: str, name: str = "ls") -> list[Any]:
    call_id = f"call-{index}"
    return [
        AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args}]),
        ToolMessage(content=result, tool_call_id=call_id, name=name),
    ]


def _history(pairs: list[list[Any]]) -> list[Any]:
    messages: list[Any] = [HumanMessage(content="look around")]
    for pair in pairs:
        messages.extend(pair)
    return messages


def _guard(messages: list[Any]) -> dict[str, Any] | None:
    return RepeatedToolResultGuardMiddleware().before_model({"messages": messages}, None)


def test_guard_ends_run_after_ten_identical_results() -> None:
    messages = _history([_call_pair(i, {"path": "/app"}, "No files found") for i in range(10)])

    update = _guard(messages)

    assert update is not None
    assert update["jump_to"] == "end"
    assert "`ls` was called 10 times" in update["messages"][0].content


def test_guard_allows_nine_identical_results() -> None:
    messages = _history([_call_pair(i, {"path": "/app"}, "No files found") for i in range(9)])

    assert _guard(messages) is None


def test_guard_counts_interleaved_repeats() -> None:
    pairs: list[list[Any]] = []
    for i in range(10):
        pairs.append(_call_pair(i, {"path": "/app"}, "No files found"))
        pairs.append(_call_pair(100 + i, {"query": f"q{i}"}, f"hit {i}", name="grep"))

    assert _guard(_history(pairs)) is not None


def test_guard_treats_changing_results_as_progress() -> None:
    messages = _history([_call_pair(i, {"id": "job"}, f"status {i}") for i in range(20)])

    assert _guard(messages) is None


def test_guard_treats_same_result_for_different_arguments_as_progress() -> None:
    messages = _history([_call_pair(i, {"path": f"/d{i}"}, "No files found") for i in range(20)])

    assert _guard(messages) is None


def test_guard_ignores_repeats_from_earlier_turns() -> None:
    earlier = _history([_call_pair(i, {"path": "/app"}, "No files found") for i in range(10)])
    messages = [*earlier, AIMessage(content="Stopped."), HumanMessage(content="try again")]

    assert _guard(messages) is None


def test_agent_middleware_includes_the_repeat_guard(tmp_path: Path) -> None:
    middleware = build_agent_middleware(
        backend=build_deepagent_backend(project_root=tmp_path, include_memories=False),
        project_root=tmp_path,
    )

    assert any(isinstance(item, RepeatedToolResultGuardMiddleware) for item in middleware)


# --- background task visibility ---------------------------------------------


def _manager() -> BackgroundTaskManager:
    return BackgroundTaskManager(
        BackgroundSubagentConfig(
            enabled=True,
            batch_result_format="markdown",
            max_running_per_session=2,
            max_running_total=3,
            max_tasks_per_session=5,
        )
    )


def test_background_tasks_log_start_heartbeat_and_completion(caplog) -> None:
    caplog.set_level(logging.INFO, logger="chainagents.runtime.background_tasks.manager")

    async def exercise() -> None:
        manager = _manager()
        manager.heartbeat_interval = 0.01
        release = asyncio.Event()
        task_ids: list[str] = []

        async def runner(task_id: str) -> str:
            task_ids.append(task_id)
            await manager.publish_activity(task_id, object())  # type: ignore[arg-type]
            await manager.publish_activity(task_id, object())  # type: ignore[arg-type]
            await asyncio.sleep(0.05)
            await release.wait()
            return "done"

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="researcher",
            description="research this",
            agent_path=("researcher",),
            runner=runner,
        )
        await asyncio.sleep(0.1)
        release.set()
        finished = await manager.get("session-a", spawned.task_id, wait_seconds=1)
        assert finished.status == "success"
        await manager.close()

    asyncio.run(exercise())

    messages = [record.getMessage() for record in caplog.records]
    assert any("started: agent=researcher" in message for message in messages)
    assert any(
        "is still running" in message and "events=2" in message for message in messages
    )
    assert any(
        "finished: status=success" in message and "events=2" in message
        for message in messages
    )


# --- Snowflake Cortex prompt caching ----------------------------------------


def _cortex_payload(model: str, messages: list[Any]) -> dict[str, Any]:
    return SnowflakeCortexChatOpenAI(
        model=model,
        base_url="https://acme.snowflakecomputing.com/api/v2/cortex/v1",
        api_key="test-pat",
    )._get_request_payload(messages)


def _turn(assistant_text: str = "Listing the workspace.") -> list[Any]:
    return [
        SystemMessage(content="static system prompt"),
        HumanMessage(content="list files"),
        AIMessage(
            content=assistant_text,
            tool_calls=[{"id": "call-1", "name": "ls", "args": {"path": "/workspace"}}],
        ),
        ToolMessage(content="README.md", tool_call_id="call-1", name="ls"),
    ]


def _marked_roles(messages: list[dict[str, Any]]) -> list[str]:
    return [
        message["role"]
        for message in messages
        if isinstance(message.get("content"), list)
        and any("cache_control" in part for part in message["content"])
    ]


def test_cortex_claude_payload_marks_system_and_newest_assistant_turn() -> None:
    messages = _cortex_payload("claude-sonnet-4-5", _turn())["messages"]

    ephemeral = {"type": "ephemeral"}
    assert messages[0]["content"][-1]["cache_control"] == ephemeral
    assert messages[-1]["role"] == "tool"
    assert "cache_control" not in str(messages[-1])
    assert messages[-2]["content"][-1]["cache_control"] == ephemeral
    assert _marked_roles(messages) == ["system", "assistant"]


def test_cortex_claude_payload_never_marks_tool_results() -> None:
    messages = _cortex_payload("claude-sonnet-4-5", _turn(assistant_text=""))["messages"]

    assert "tool" not in _marked_roles(messages)
    assert _marked_roles(messages) == ["system", "user"]


def test_cortex_non_claude_payload_has_no_cache_control() -> None:
    messages = _cortex_payload("llama3.3-70b", _turn())["messages"]

    assert "cache_control" not in str(messages)
