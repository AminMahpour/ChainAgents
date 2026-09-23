"""Exercise process-local background subagent task ownership and lifecycle."""

from __future__ import annotations

import asyncio
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from deepagents import create_deep_agent
from deepagents.backends.protocol import DeleteResult, WriteResult
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

import chainagents.runtime.background_tasks as background_tasks
from chainagents.runtime.backends import (
    build_deepagent_backend,
    generated_outputs_route_prefix,
)
from chainagents.runtime.background_tasks import (
    BackgroundTaskStatus,
    BackgroundTaskActivity,
    BackgroundTaskManager,
    BackgroundTaskSnapshot,
    BatchResultOutputStore,
    _write_batch_markdown_files,
    create_background_task_tools,
    current_background_invocation_path,
    current_background_session_generation,
    current_background_task_id,
    scope_background_session_invocation,
    scope_background_task_invocation,
)
from chainagents.events.stream import AgentStreamEvent
from chainagents.exports.generated_files import resolve_generated_output
from chainagents.runtime.types import BatchResultFormat, BackgroundSubagentConfig
from chainagents.interfaces.chainlit.async_tasks import LocalBackgroundTaskNotifier


class _ToolCallingFakeModel(FakeMessagesListChatModel):
    """Deterministic fake model that accepts tools without changing responses."""

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return self


def make_manager(
    **overrides: int | bool | BatchResultFormat,
) -> BackgroundTaskManager:
    values: dict[str, int | bool | BatchResultFormat] = {
        "enabled": True,
        "batch_result_format": "markdown",
        "max_running_per_session": 2,
        "max_running_total": 3,
        "max_tasks_per_session": 5,
    }
    values.update(overrides)
    return BackgroundTaskManager(BackgroundSubagentConfig(**values))  # type: ignore[arg-type]


def batch_snapshot(
    *,
    task_id: str,
    agent_name: str = "researcher",
    description: str = "Inspect the code.",
    status: str = "success",
    result: str | None = "Report text",
    error: str | None = None,
) -> BackgroundTaskSnapshot:
    """Build a deterministic terminal snapshot for batch-output tests."""
    return BackgroundTaskSnapshot(
        task_id=task_id,
        session_id="session-a",
        agent_name=agent_name,
        description=description,
        agent_path=(agent_name,),
        parent_task_id=None,
        status=cast("BackgroundTaskStatus", status),
        result=result,
        error=error,
        created_at=1.0,
        completed_at=2.0,
    )


def test_spawn_returns_before_runner_finishes_and_result_can_be_retrieved() -> None:
    async def exercise() -> None:
        manager = make_manager()
        started = asyncio.Event()
        release = asyncio.Event()

        async def runner(task_id: str) -> str:
            started.set()
            await release.wait()
            return f"result:{task_id}"

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="researcher",
            description="research this",
            agent_path=("researcher",),
            runner=runner,
        )

        assert spawned.status in {"pending", "running"}
        await asyncio.wait_for(started.wait(), timeout=1)
        running = await manager.get("session-a", spawned.task_id)
        assert running.status == "running"

        release.set()
        finished = await manager.get("session-a", spawned.task_id, wait_seconds=1)
        assert finished.status == "success"
        assert finished.result == f"result:{spawned.task_id}"
        await manager.close()

    asyncio.run(exercise())


def test_background_activity_channel_orders_live_events_before_completion() -> None:
    """One session queue must receive live activity before its terminal snapshot."""
    async def exercise() -> None:
        manager = make_manager(stream_activity=True)
        queue = manager.subscribe_activity("session-a")

        async def runner(task_id: str) -> str:
            await manager.publish_activity(
                task_id,
                AgentStreamEvent(
                    kind="reasoning_delta",
                    source="researcher",
                    text="Checking the repository",
                ),
            )
            return "finished result"

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="researcher",
            description="research",
            agent_path=("researcher",),
            runner=runner,
        )
        live = await asyncio.wait_for(queue.get(), timeout=1)
        terminal = await asyncio.wait_for(queue.get(), timeout=1)

        assert live == BackgroundTaskActivity(
            task_id=spawned.task_id,
            session_id="session-a",
            agent_name="researcher",
            description="research",
            event=AgentStreamEvent(
                kind="reasoning_delta",
                source="researcher",
                text="Checking the repository",
            ),
        )
        assert terminal.event is None
        assert terminal.snapshot is not None
        assert terminal.snapshot.status == "success"
        assert terminal.snapshot.result == "finished result"
        assert queue.empty()

        manager.unsubscribe_activity("session-a", queue)
        assert "session-a" not in manager._activity_subscribers
        await manager.close()

    asyncio.run(exercise())


def test_session_generation_entries_are_reclaimed() -> None:
    async def exercise() -> None:
        manager = make_manager()
        generation = manager.session_generation("session-a")
        generation_ref = weakref.ref(generation)

        assert tuple(manager._session_generations) == ("session-a",)
        del generation
        gc.collect()
        assert generation_ref() is None
        assert tuple(manager._session_generations) == ()

        await manager.close_session("unknown-session")
        assert tuple(manager._session_generations) == ()
        await manager.close()

    asyncio.run(exercise())


def test_session_generation_rejects_runs_started_during_close() -> None:
    async def exercise() -> None:
        manager = make_manager()

        async with manager.closing_session("session-a"):
            with pytest.raises(RuntimeError, match="session is closing"):
                manager.session_generation("session-a")

        assert manager.session_generation("session-a").active
        await manager.close()

    asyncio.run(exercise())


def test_background_invocation_scope_wraps_each_stream_pull() -> None:
    """Streaming a scoped child must not fall back to one final ainvoke result."""
    async def exercise() -> None:
        captured_paths: list[tuple[str, ...]] = []
        captured_kwargs: list[dict[str, object]] = []

        class Invocation:
            async def ainvoke(self, input, config=None, **kwargs):
                return "fallback"

            async def astream(self, input, config=None, **kwargs):
                captured_kwargs.append(kwargs)
                captured_paths.append(current_background_invocation_path())
                yield "first"
                await asyncio.sleep(0)
                captured_paths.append(current_background_invocation_path())
                yield "second"

        runnable = scope_background_task_invocation(Invocation())
        streamed = [
            chunk
            async for chunk in runnable.astream(
                {},
                {"configurable": {"thread_id": "session-a"}},
                stream_mode=["values", "messages"],
                subgraphs=True,
            )
        ]

        assert streamed == ["first", "second"]
        assert len(captured_paths) == 2
        assert captured_paths[0] == captured_paths[1]
        assert captured_paths[0]
        assert current_background_invocation_path() == ()
        assert captured_kwargs == [
            {
                "stream_mode": ["values", "messages"],
                "subgraphs": True,
            }
        ]

    asyncio.run(exercise())


def test_exported_graph_forwards_v3_events_with_session_generation() -> None:
    async def exercise() -> None:
        manager = make_manager()
        captured: list[object] = []
        captured_kwargs: list[dict[str, object]] = []

        class Invocation:
            def astream_events(self, input, config=None, *, version="v2", **kwargs):
                captured_kwargs.append(kwargs)

                async def v3_result():
                    captured.append(current_background_session_generation())
                    return {"version": version}

                return v3_result()

        graph = scope_background_session_invocation(Invocation(), manager)
        result = graph.astream_events(
            {},
            {"configurable": {"thread_id": "session-a"}},
            version="v3",
        )

        assert await result == {"version": "v3"}
        assert captured == [manager.session_generation("session-a")]
        assert captured_kwargs == [{}]
        await manager.close()

    asyncio.run(exercise())


@pytest.mark.filterwarnings("ignore:The v3 streaming protocol.*")
def test_exported_compiled_graph_v3_pulls_keep_session_generation() -> None:
    async def exercise() -> None:
        manager = make_manager()
        captured: list[object] = []

        async def inspect_generation(state):
            captured.append(current_background_session_generation())
            return state

        builder = StateGraph(dict)
        builder.add_node("inspect", inspect_generation)
        builder.add_edge(START, "inspect")
        builder.add_edge("inspect", END)
        graph = scope_background_session_invocation(
            builder.compile(),
            manager,
        )

        run = await graph.astream_events(
            {},
            {"configurable": {"thread_id": "session-a"}},
            version="v3",
        )
        await run.output()

        assert captured == [manager.session_generation("session-a")]
        await manager.close()

    asyncio.run(exercise())


def test_exported_graph_invocation_is_invalidated_by_overlapping_close() -> None:
    async def exercise() -> None:
        manager = make_manager()
        started = asyncio.Event()
        release = asyncio.Event()
        tools = create_background_task_tools(
            manager=manager,
            subagents={"worker": object()},
            agent_path=(),
            recursion_limit=20,
        )
        spawn_tool = next(
            tool for tool in tools if tool.name == "spawn_background_task"
        )
        tool_runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="exported-spawn",
            store=None,
        )

        class Invocation:
            async def astream(self, input, config=None, **kwargs):
                started.set()
                await release.wait()
                yield await spawn_tool.coroutine(
                    "late work",
                    "worker",
                    tool_runtime,
                )

        graph = scope_background_session_invocation(Invocation(), manager)

        async def consume_stream():
            return [
                chunk
                async for chunk in graph.astream(
                    {},
                    {"configurable": {"thread_id": "session-a"}},
                )
            ]

        run = asyncio.create_task(consume_stream())
        await asyncio.wait_for(started.wait(), timeout=1)
        await manager.close_session("session-a")
        release.set()

        with pytest.raises(RuntimeError, match="session was closed"):
            await run
        await manager.close()

    asyncio.run(exercise())


def test_session_and_nested_agent_scopes_are_isolated() -> None:
    async def exercise() -> None:
        manager = make_manager()

        async def runner(task_id: str) -> str:
            return task_id

        alpha = await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="alpha",
            agent_path=("planner", "worker"),
            runner=runner,
        )
        sibling = await manager.spawn(
            session_id="session-a",
            agent_name="critic",
            description="sibling",
            agent_path=("critic",),
            runner=runner,
        )
        await manager.spawn(
            session_id="session-b",
            agent_name="worker",
            description="other session",
            agent_path=("planner", "worker"),
            runner=runner,
        )

        all_session_a = await manager.list("session-a")
        planner_scope = await manager.list("session-a", scope_path=("planner",))

        assert {task.task_id for task in all_session_a} == {
            alpha.task_id,
            sibling.task_id,
        }
        assert [task.task_id for task in planner_scope] == [alpha.task_id]
        with pytest.raises(KeyError, match="not visible"):
            await manager.get(
                "session-a",
                sibling.task_id,
                scope_path=("planner",),
            )
        await manager.close()

    asyncio.run(exercise())


def test_nested_background_agent_sees_only_its_own_task_instance_descendants() -> None:
    async def exercise() -> None:
        manager = make_manager(max_running_per_session=6, max_running_total=6)

        async def runner(task_id: str) -> str:
            await asyncio.Event().wait()
            return task_id

        first_parent = await manager.spawn(
            session_id="session-a",
            agent_name="manager",
            description="first parent",
            agent_path=("manager",),
            runner=runner,
        )
        second_parent = await manager.spawn(
            session_id="session-a",
            agent_name="manager",
            description="second parent",
            agent_path=("manager",),
            runner=runner,
        )
        first_child = await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="first child",
            agent_path=("manager", "worker"),
            parent_task_id=first_parent.task_id,
            runner=runner,
        )
        await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="second child",
            agent_path=("manager", "worker"),
            parent_task_id=second_parent.task_id,
            runner=runner,
        )

        visible = await manager.list(
            "session-a",
            scope_path=("manager",),
            ancestor_task_id=first_parent.task_id,
        )

        assert [task.task_id for task in visible] == [first_child.task_id]
        await manager.close()

    asyncio.run(exercise())


def test_running_and_retained_task_limits_reject_without_queueing() -> None:
    async def exercise() -> None:
        manager = make_manager(max_running_per_session=1, max_tasks_per_session=1)
        release = asyncio.Event()

        async def runner(task_id: str) -> str:
            await release.wait()
            return task_id

        await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="first",
            agent_path=("worker",),
            runner=runner,
        )
        with pytest.raises(RuntimeError, match="running limit"):
            await manager.spawn(
                session_id="session-a",
                agent_name="worker",
                description="second",
                agent_path=("worker",),
                runner=runner,
            )

        release.set()
        await manager.wait_session("session-a")
        with pytest.raises(RuntimeError, match="retained task limit"):
            await manager.spawn(
                session_id="session-a",
                agent_name="worker",
                description="third",
                agent_path=("worker",),
                runner=runner,
            )
        await manager.close()

    asyncio.run(exercise())


def test_global_limit_wait_timeout_and_repeated_cancel_are_stable() -> None:
    async def exercise() -> None:
        manager = make_manager(max_running_total=1)
        started = asyncio.Event()

        async def runner(task_id: str) -> str:
            started.set()
            await asyncio.Event().wait()
            return task_id

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="first",
            agent_path=("worker",),
            runner=runner,
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        timed_out = await manager.get(
            "session-a", spawned.task_id, wait_seconds=0.01
        )
        assert timed_out.status == "running"
        with pytest.raises(RuntimeError, match="process"):
            await manager.spawn(
                session_id="session-b",
                agent_name="worker",
                description="second",
                agent_path=("worker",),
                runner=runner,
            )

        first_cancel = await manager.cancel("session-a", spawned.task_id)
        second_cancel = await manager.cancel("session-a", spawned.task_id)
        first_get = await manager.get("session-a", spawned.task_id)
        second_get = await manager.get("session-a", spawned.task_id)
        assert first_cancel.status == "cancelled"
        assert second_cancel == first_cancel
        assert first_get == second_get == first_cancel
        await manager.close()

    asyncio.run(exercise())


def test_cancelled_task_cancel_finishes_finalization_before_returning() -> None:
    async def exercise() -> None:
        manager = make_manager(
            max_running_per_session=1,
            max_running_total=1,
        )
        runner_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        async def runner(task_id: str) -> str:
            runner_started.set()
            await asyncio.Event().wait()
            return task_id

        async def cleanup(task_id: str) -> None:
            cleanup_started.set()
            await allow_cleanup.wait()

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="running task",
            agent_path=("worker",),
            runner=runner,
            cleanup=cleanup,
        )
        await asyncio.wait_for(runner_started.wait(), timeout=1)

        cancel_task = asyncio.create_task(
            manager.cancel("session-a", spawned.task_id)
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        cancel_task.cancel()
        await asyncio.sleep(0)

        assert not cancel_task.done()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cancel_task, timeout=1)

        terminal = await manager.get("session-a", spawned.task_id)
        assert terminal.status == "cancelled"
        replacement = await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="replacement task",
            agent_path=("worker",),
            runner=lambda task_id: asyncio.sleep(0, result=task_id),
        )
        assert (
            await manager.get("session-a", replacement.task_id, wait_seconds=1)
        ).status == "success"
        await manager.close()

    asyncio.run(exercise())


def test_cancelling_parent_rejects_late_descendant_spawn() -> None:
    async def exercise() -> None:
        manager = make_manager(max_running_per_session=3, max_running_total=3)
        runner_started = asyncio.Event()
        child_release = asyncio.Event()
        spawn_error: BaseException | None = None

        async def child_runner(task_id: str) -> str:
            await child_release.wait()
            return task_id

        async def parent_runner(task_id: str) -> str:
            nonlocal spawn_error
            runner_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                try:
                    await manager.spawn(
                        session_id="session-a",
                        agent_name="child",
                        description="late child",
                        agent_path=("parent", "child"),
                        parent_task_id=task_id,
                        runner=child_runner,
                    )
                except BaseException as exc:
                    spawn_error = exc
                return "cancel handled"

        parent = await manager.spawn(
            session_id="session-a",
            agent_name="parent",
            description="parent",
            agent_path=("parent",),
            runner=parent_runner,
        )
        await asyncio.wait_for(runner_started.wait(), timeout=1)

        terminal = await manager.cancel("session-a", parent.task_id)

        assert terminal.status == "cancelled"
        assert isinstance(spawn_error, RuntimeError)
        assert "parent task is cancelling" in str(spawn_error)
        assert [task.task_id for task in await manager.list("session-a")] == [
            parent.task_id
        ]
        child_release.set()
        await manager.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_terminal_tasks_release_their_checkpoint_resource(outcome: str) -> None:
    async def exercise() -> None:
        manager = make_manager()
        release = asyncio.Event()
        cleaned: list[str] = []

        async def runner(task_id: str) -> str:
            if outcome == "error":
                raise ValueError("failed")
            if outcome == "cancel":
                await release.wait()
            return task_id

        async def cleanup(task_id: str) -> None:
            cleaned.append(task_id)

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description=outcome,
            agent_path=("worker",),
            runner=runner,
            cleanup=cleanup,
        )
        if outcome == "cancel":
            terminal = await manager.cancel("session-a", spawned.task_id)
        else:
            terminal = await manager.get(
                "session-a", spawned.task_id, wait_seconds=1
            )

        expected_status = {
            "success": "success",
            "error": "error",
            "cancel": "cancelled",
        }[outcome]
        assert terminal.status == expected_status
        assert cleaned == [spawned.task_id]
        await manager.close()

    asyncio.run(exercise())


def test_spawn_racing_with_shutdown_never_leaves_live_work() -> None:
    async def exercise() -> None:
        manager = make_manager()

        async def runner(task_id: str) -> str:
            await asyncio.Event().wait()
            return task_id

        spawn_result, close_result = await asyncio.gather(
            manager.spawn(
                session_id="session-a",
                agent_name="worker",
                description="work",
                agent_path=("worker",),
                runner=runner,
            ),
            manager.close(),
            return_exceptions=True,
        )

        assert close_result is None
        assert not isinstance(spawn_result, BaseException) or isinstance(
            spawn_result, RuntimeError
        )
        assert await manager.list("session-a") == []
        with pytest.raises(RuntimeError, match="closed"):
            await manager.spawn(
                session_id="session-a",
                agent_name="worker",
                description="later",
                agent_path=("worker",),
                runner=runner,
            )

    asyncio.run(exercise())


def test_parent_failure_cancels_descendants_but_parent_success_does_not() -> None:
    async def exercise() -> None:
        manager = make_manager(max_running_per_session=4, max_running_total=4)
        child_release = asyncio.Event()

        async def child_runner(task_id: str) -> str:
            await child_release.wait()
            return task_id

        async def failed_parent_runner(task_id: str) -> str:
            await manager.spawn(
                session_id="session-a",
                agent_name="child",
                description="child",
                agent_path=("parent", "child"),
                parent_task_id=task_id,
                runner=child_runner,
            )
            raise ValueError("parent failed")

        failed_parent = await manager.spawn(
            session_id="session-a",
            agent_name="parent",
            description="parent",
            agent_path=("parent",),
            runner=failed_parent_runner,
        )
        failed = await manager.get("session-a", failed_parent.task_id, wait_seconds=1)
        assert failed.status == "error"
        descendants = [
            task
            for task in await manager.list("session-a")
            if task.parent_task_id == failed_parent.task_id
        ]
        assert len(descendants) == 1
        assert descendants[0].status == "cancelled"

        async def successful_parent_runner(task_id: str) -> str:
            await manager.spawn(
                session_id="session-b",
                agent_name="child",
                description="child",
                agent_path=("parent", "child"),
                parent_task_id=task_id,
                runner=child_runner,
            )
            return "parent done"

        successful_parent = await manager.spawn(
            session_id="session-b",
            agent_name="parent",
            description="parent",
            agent_path=("parent",),
            runner=successful_parent_runner,
        )
        successful = await manager.get(
            "session-b", successful_parent.task_id, wait_seconds=1
        )
        assert successful.status == "success"
        child = next(
            task
            for task in await manager.list("session-b")
            if task.parent_task_id == successful_parent.task_id
        )
        assert child.status == "running"

        child_release.set()
        await manager.wait_session("session-b")
        await manager.close()

    asyncio.run(exercise())


def test_close_session_cancels_tasks_and_publishes_one_terminal_event() -> None:
    async def exercise() -> None:
        manager = make_manager()
        queue = manager.subscribe("session-a")
        cleaned: list[str] = []

        async def runner(task_id: str) -> str:
            await asyncio.Event().wait()
            return task_id

        async def cleanup(task_id: str) -> None:
            cleaned.append(task_id)

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="long task",
            agent_path=("worker",),
            runner=runner,
            cleanup=cleanup,
        )
        await manager.close_session("session-a")

        event = await asyncio.wait_for(queue.get(), timeout=1)
        assert event.task_id == spawned.task_id
        assert event.status == "cancelled"
        assert queue.empty()
        assert cleaned == [spawned.task_id]
        assert await manager.list("session-a") == []
        manager.unsubscribe("session-a", queue)
        await manager.close()

    asyncio.run(exercise())


def test_close_session_awaits_checkpoint_cleanup_when_task_is_cancelled() -> None:
    async def exercise() -> None:
        manager = make_manager()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()
        cleanup_calls = 0
        cleanup_completions = 0

        async def runner(task_id: str) -> str:
            return task_id

        async def cleanup(task_id: str) -> None:
            nonlocal cleanup_calls, cleanup_completions
            cleanup_calls += 1
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_completions += 1

        await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="finishing task",
            agent_path=("worker",),
            runner=runner,
            cleanup=cleanup,
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)

        close_task = asyncio.create_task(manager.close_session("session-a"))
        await asyncio.sleep(0)
        allow_cleanup.set()
        await asyncio.wait_for(close_task, timeout=1)

        assert cleanup_calls == 1
        assert cleanup_completions == 1
        assert await manager.list("session-a") == []
        await manager.close()

    asyncio.run(exercise())


def test_close_session_awaits_cleanup_started_by_concurrent_cancel() -> None:
    async def exercise() -> None:
        manager = make_manager()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()
        cleanup_calls = 0
        cleanup_completions = 0

        async def runner(task_id: str) -> str:
            await asyncio.Event().wait()
            return task_id

        async def cleanup(task_id: str) -> None:
            nonlocal cleanup_calls, cleanup_completions
            cleanup_calls += 1
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_completions += 1

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="running task",
            agent_path=("worker",),
            runner=runner,
            cleanup=cleanup,
        )
        cancel_task = asyncio.create_task(
            manager.cancel("session-a", spawned.task_id)
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)

        close_task = asyncio.create_task(manager.close_session("session-a"))
        await asyncio.sleep(0)
        assert not close_task.done()

        allow_cleanup.set()
        cancelled, _ = await asyncio.wait_for(
            asyncio.gather(cancel_task, close_task),
            timeout=1,
        )

        assert cancelled.status == "cancelled"
        assert cleanup_calls == 1
        assert cleanup_completions == 1
        assert await manager.list("session-a") == []
        await manager.close()

    asyncio.run(exercise())


def test_concurrent_close_session_calls_are_idempotent() -> None:
    async def exercise() -> None:
        manager = make_manager()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()
        cleanup_calls = 0

        async def runner(task_id: str) -> str:
            await asyncio.Event().wait()
            return task_id

        async def cleanup(task_id: str) -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1
            cleanup_started.set()
            await allow_cleanup.wait()

        await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="running task",
            agent_path=("worker",),
            runner=runner,
            cleanup=cleanup,
        )

        first_close = asyncio.create_task(manager.close_session("session-a"))
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        second_close = asyncio.create_task(manager.close_session("session-a"))
        await asyncio.sleep(0)

        allow_cleanup.set()
        await asyncio.wait_for(
            asyncio.gather(first_close, second_close),
            timeout=1,
        )

        assert cleanup_calls == 1
        assert await manager.list("session-a") == []
        await manager.close()

    asyncio.run(exercise())


def test_cancelled_close_session_finishes_cleanup_before_reopening() -> None:
    async def exercise() -> None:
        manager = make_manager()
        runner_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        async def runner(task_id: str) -> str:
            runner_started.set()
            await asyncio.Event().wait()
            return task_id

        async def cleanup(task_id: str) -> None:
            cleanup_started.set()
            await allow_cleanup.wait()

        await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="running task",
            agent_path=("worker",),
            runner=runner,
            cleanup=cleanup,
        )
        await asyncio.wait_for(runner_started.wait(), timeout=1)

        close_task = asyncio.create_task(manager.close_session("session-a"))
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        close_task.cancel()
        await asyncio.sleep(0)

        assert not close_task.done()
        with pytest.raises(RuntimeError, match="session is closing"):
            await manager.spawn(
                session_id="session-a",
                agent_name="worker",
                description="late task",
                agent_path=("worker",),
                runner=runner,
            )

        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(close_task, timeout=1)

        assert await manager.list("session-a") == []
        reopened = await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="new task",
            agent_path=("worker",),
            runner=lambda task_id: asyncio.sleep(0, result=task_id),
        )
        assert (await manager.get("session-a", reopened.task_id, wait_seconds=1)).status == "success"
        await manager.close()

    asyncio.run(exercise())


def test_cancelled_manager_close_finishes_session_cleanup() -> None:
    async def exercise() -> None:
        manager = make_manager()
        runner_started = asyncio.Event()
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        async def runner(task_id: str) -> str:
            runner_started.set()
            await asyncio.Event().wait()
            return task_id

        async def cleanup(task_id: str) -> None:
            cleanup_started.set()
            await allow_cleanup.wait()

        await manager.spawn(
            session_id="session-a",
            agent_name="worker",
            description="running task",
            agent_path=("worker",),
            runner=runner,
            cleanup=cleanup,
        )
        await asyncio.wait_for(runner_started.wait(), timeout=1)

        close_task = asyncio.create_task(manager.close())
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        close_task.cancel()
        await asyncio.sleep(0)

        assert not close_task.done()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(close_task, timeout=1)

        assert await manager.list("session-a") == []
        with pytest.raises(RuntimeError, match="manager is closed"):
            await manager.spawn(
                session_id="session-a",
                agent_name="worker",
                description="late task",
                agent_path=("worker",),
                runner=runner,
            )

    asyncio.run(exercise())


def test_cancel_cascades_through_prestart_descendant_tree() -> None:
    """Cancelling before child coroutines start must still reach grandchildren."""
    async def exercise() -> None:
        manager = make_manager(max_running_per_session=5, max_running_total=5)

        async def runner(task_id: str) -> str:
            await asyncio.Event().wait()
            return task_id

        parent = await manager.spawn(
            session_id="session-a",
            agent_name="parent",
            description="parent",
            agent_path=("parent",),
            runner=runner,
        )
        child = await manager.spawn(
            session_id="session-a",
            agent_name="child",
            description="child",
            agent_path=("parent", "child"),
            parent_task_id=parent.task_id,
            runner=runner,
        )
        grandchild = await manager.spawn(
            session_id="session-a",
            agent_name="grandchild",
            description="grandchild",
            agent_path=("parent", "child", "grandchild"),
            parent_task_id=child.task_id,
            runner=runner,
        )

        await manager.cancel("session-a", parent.task_id)

        statuses = {
            task.task_id: task.status for task in await manager.list("session-a")
        }
        assert statuses == {
            parent.task_id: "cancelled",
            child.task_id: "cancelled",
            grandchild.task_id: "cancelled",
        }
        await manager.close()

    asyncio.run(exercise())


def test_background_tools_spawn_isolated_child_and_retrieve_result() -> None:
    """Removing isolated state/config construction must break this contract."""

    class ChildRunnable:
        def __init__(self) -> None:
            self.calls: list[tuple[dict[str, object], dict[str, object]]] = []

        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            self.calls.append((state, config))
            return {"messages": [AIMessage(content="child result")]}

    async def exercise() -> None:
        manager = make_manager()
        child = ChildRunnable()
        tools = create_background_task_tools(
            manager=manager,
            subagents={"researcher": child},
            agent_path=(),
            recursion_limit=45,
        )
        by_name = {tool.name: tool for tool in tools}
        assert set(by_name) == {
            "spawn_background_task",
            "run_subagent_batch",
            "list_background_tasks",
            "get_background_task",
            "cancel_background_task",
        }
        store = InMemoryStore()
        checkpointer = MemorySaver()
        runtime = ToolRuntime(
            state={"messages": [HumanMessage(content="private parent history")]},
            context=None,
            config={
                "configurable": {
                    "thread_id": "session-a",
                    "__pregel_checkpointer": checkpointer,
                    "checkpoint_id": "parent-checkpoint",
                    "checkpoint_ns": "parent-namespace",
                },
                "callbacks": ["parent-stream-callback"],
            },
            stream_writer=lambda _: None,
            tool_call_id="call-1",
            store=store,
        )

        spawned = await by_name["spawn_background_task"].coroutine(
            "investigate",
            "researcher",
            runtime,
        )
        task_id = spawned["task_id"]
        finished = await by_name["get_background_task"].coroutine(
            task_id,
            1,
            runtime,
        )

        assert finished["status"] == "success"
        assert finished["result"] == "child result"
        assert child.calls[0][0] == {"messages": [HumanMessage(content="investigate")]}
        child_config = child.calls[0][1]
        assert child_config["recursion_limit"] == 45
        configurable = child_config["configurable"]
        assert configurable["thread_id"] == f"session-a:background:{task_id}"
        assert configurable["ls_agent_type"] == "background_subagent"
        assert configurable["__pregel_checkpointer"] is checkpointer
        child_runtime = configurable["__pregel_runtime"]
        assert child_runtime.store is store
        assert child_runtime is not runtime
        assert "checkpoint_id" not in configurable
        assert configurable["checkpoint_ns"] == ""
        assert "callbacks" not in child_config
        await manager.close()

    asyncio.run(exercise())


def test_background_tools_stream_child_activity_with_relabelled_sources() -> None:
    """Opted-in children must publish root and nested activity before completion."""

    class ChildRunnable:
        def __init__(self) -> None:
            self.calls: list[
                tuple[dict[str, object], dict[str, object], dict[str, object]]
            ] = []

        async def ainvoke(self, state, config, **kwargs):
            raise AssertionError("stream activity must not use ainvoke")

        async def astream(self, state, config, **kwargs):
            self.calls.append((state, config, kwargs))
            root_token = SimpleNamespace(
                type="AIMessageChunk",
                additional_kwargs={"reasoning_content": "root thought"},
                tool_call_chunks=[],
                content="",
            )
            nested_token = SimpleNamespace(
                type="AIMessageChunk",
                additional_kwargs={"reasoning_content": "nested thought"},
                tool_call_chunks=[],
                content="",
            )
            yield ((), "messages", (root_token, {}))
            yield (("worker:child-run",), "messages", (nested_token, {}))
            yield (
                (),
                "values",
                {
                    "messages": [
                        HumanMessage(content="investigate"),
                        AIMessage(content="streamed result"),
                    ]
                },
            )

    async def exercise() -> None:
        manager = make_manager(stream_activity=True)
        child = ChildRunnable()
        activity_queue = manager.subscribe_activity("session-a")
        tools = create_background_task_tools(
            manager=manager,
            subagents={"researcher": child},
            agent_path=(),
            recursion_limit=45,
        )
        spawn_tool = next(
            tool for tool in tools if tool.name == "spawn_background_task"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="call-1",
            store=None,
        )

        spawned = await spawn_tool.coroutine("investigate", "researcher", runtime)
        activities = [
            await asyncio.wait_for(activity_queue.get(), timeout=1)
            for _ in range(3)
        ]

        assert [
            activity.event.source
            for activity in activities
            if activity.event is not None
        ] == ["researcher", "researcher / worker"]
        assert [
            activity.event.text
            for activity in activities
            if activity.event is not None
        ] == ["root thought", "nested thought"]
        assert activities[-1].snapshot is not None
        assert activities[-1].snapshot.status == "success"
        assert activities[-1].snapshot.result == "streamed result"
        assert child.calls == [
            (
                {"messages": [HumanMessage(content="investigate")]},
                {
                    "configurable": {
                        "thread_id": f"session-a:background:{spawned['task_id']}",
                        "checkpoint_ns": "",
                        "ls_agent_type": "background_subagent",
                    },
                    "recursion_limit": 45,
                },
                {
                    "stream_mode": ["values", "messages", "updates"],
                    "subgraphs": True,
                },
            )
        ]
        await manager.close()

    asyncio.run(exercise())


def test_run_subagent_batch_executes_repeated_targets_concurrently_in_input_order() -> None:
    """A one-call fanout must overlap child runs and preserve request ordering."""

    class ChildRunnable:
        def __init__(self) -> None:
            self.started: list[str] = []
            self.configs: list[dict[str, object]] = []
            self.both_started = asyncio.Event()
            self.releases = {
                "first": asyncio.Event(),
                "second": asyncio.Event(),
            }

        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            messages = state["messages"]
            assert isinstance(messages, list)
            description = str(messages[0].content)
            self.started.append(description)
            self.configs.append(config)
            if len(self.started) == 2:
                self.both_started.set()
            await self.releases[description].wait()
            return {"messages": [AIMessage(content=f"result:{description}")]}

    async def exercise() -> None:
        manager = make_manager(max_running_per_session=4, max_running_total=4)
        child = ChildRunnable()
        tools = create_background_task_tools(
            manager=manager,
            subagents={"researcher": child},
            agent_path=(),
            recursion_limit=20,
        )
        batch_tool = next(
            tool for tool in tools if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )

        batch_call = asyncio.create_task(
            batch_tool.coroutine(
                [
                    {"description": "first", "subagent_type": "researcher"},
                    {"description": "second", "subagent_type": "researcher"},
                ],
                runtime,
            )
        )
        await asyncio.wait_for(child.both_started.wait(), timeout=1)
        child.releases["second"].set()
        await asyncio.sleep(0)
        assert not batch_call.done()
        child.releases["first"].set()

        result = await asyncio.wait_for(batch_call, timeout=1)

        snapshots = await manager.list("session-a")
        assert result == (
            "# Subagent batch results\n\n"
            "## 1. researcher\n"
            f"- Task ID: `{snapshots[0].task_id}`\n"
            "- Status: `success`\n\n"
            "### Request\nfirst\n\n"
            "### Report\nresult:first\n\n"
            "---\n\n"
            "## 2. researcher\n"
            f"- Task ID: `{snapshots[1].task_id}`\n"
            "- Status: `success`\n\n"
            "### Request\nsecond\n\n"
            "### Report\nresult:second"
        )
        thread_ids = [
            str(config["configurable"]["thread_id"])
            for config in child.configs
        ]
        assert len(set(thread_ids)) == 2
        assert all(thread_id.startswith("session-a:background:bg-") for thread_id in thread_ids)
        await manager.close()

    asyncio.run(exercise())


def test_run_subagent_batch_restores_exact_json_payload_in_input_order() -> None:
    """JSON mode must reproduce the original full snapshot payload."""

    class ChildRunnable:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.releases = {"first": asyncio.Event(), "second": asyncio.Event()}
            self.count = 0

        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            messages = state["messages"]
            assert isinstance(messages, list)
            description = str(messages[0].content)
            self.count += 1
            if self.count == 2:
                self.started.set()
            await self.releases[description].wait()
            return {"messages": [AIMessage(content=f"result:{description}")]}

    async def exercise() -> None:
        manager = make_manager(
            batch_result_format="json",
            max_running_per_session=4,
            max_running_total=4,
        )
        child = ChildRunnable()
        batch_tool = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": child},
                agent_path=(),
                recursion_limit=20,
            )
            if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )
        batch_call = asyncio.create_task(
            batch_tool.coroutine(
                [
                    {"description": "first", "subagent_type": "researcher"},
                    {"description": "second", "subagent_type": "researcher"},
                ],
                runtime,
            )
        )
        await child.started.wait()
        child.releases["second"].set()
        await asyncio.sleep(0)
        assert not batch_call.done()
        child.releases["first"].set()

        result = await batch_call
        snapshots = await manager.list("session-a")

        assert result == {
            "results": [
                {
                    "task_id": snapshots[0].task_id,
                    "session_id": "session-a",
                    "agent_name": "researcher",
                    "description": "first",
                    "agent_path": ["researcher"],
                    "parent_task_id": None,
                    "status": "success",
                    "result": "result:first",
                    "error": None,
                    "created_at": snapshots[0].created_at,
                    "completed_at": snapshots[0].completed_at,
                },
                {
                    "task_id": snapshots[1].task_id,
                    "session_id": "session-a",
                    "agent_name": "researcher",
                    "description": "second",
                    "agent_path": ["researcher"],
                    "parent_task_id": None,
                    "status": "success",
                    "result": "result:second",
                    "error": None,
                    "created_at": snapshots[1].created_at,
                    "completed_at": snapshots[1].completed_at,
                },
            ]
        }
        await manager.close()

    asyncio.run(exercise())


def test_batch_markdown_files_write_standalone_reports_in_input_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """File mode writes one complete, downloadable report per terminal snapshot."""

    async def exercise() -> None:
        backend = build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
        )
        store = BatchResultOutputStore(
            backend=backend,
            backend_prefix=generated_outputs_route_prefix(tmp_path),
        )
        monkeypatch.setattr(
            background_tasks.uuid,
            "uuid4",
            lambda: SimpleNamespace(hex="batchunique"),
        )
        snapshots = [
            batch_snapshot(task_id="task-1"),
            batch_snapshot(
                task_id="task-2",
                agent_name="reviewer",
                description="Review the failure.",
                status="error",
                result=None,
                error="Child failed",
            ),
            batch_snapshot(
                task_id="task-3",
                agent_name="worker",
                description="Finish the change.",
                status="cancelled",
                result=None,
            ),
        ]

        manifest = await _write_batch_markdown_files(
            snapshots,
            tool_call_id="batch-call",
            store=store,
        )

        assert manifest == {
            "files": [
                {
                    "task_id": "task-1",
                    "agent_name": "researcher",
                    "status": "success",
                    "path": (
                        "/workspace/.files/outputs/subagent-batches/"
                        "batch-call-batchunique/01-researcher-task-1.md"
                    ),
                },
                {
                    "task_id": "task-2",
                    "agent_name": "reviewer",
                    "status": "error",
                    "path": (
                        "/workspace/.files/outputs/subagent-batches/"
                        "batch-call-batchunique/02-reviewer-task-2.md"
                    ),
                },
                {
                    "task_id": "task-3",
                    "agent_name": "worker",
                    "status": "cancelled",
                    "path": (
                        "/workspace/.files/outputs/subagent-batches/"
                        "batch-call-batchunique/03-worker-task-3.md"
                    ),
                },
            ]
        }
        output_directory = (
            tmp_path
            / ".files"
            / "outputs"
            / "subagent-batches"
            / "batch-call-batchunique"
        )
        assert (output_directory / "01-researcher-task-1.md").read_text() == (
            "# researcher\n\n"
            "- Task ID: `task-1`\n"
            "- Status: `success`\n\n"
            "## Request\n"
            "Inspect the code.\n\n"
            "## Report\n"
            "Report text"
        )
        assert (output_directory / "02-reviewer-task-2.md").read_text() == (
            "# reviewer\n\n"
            "- Task ID: `task-2`\n"
            "- Status: `error`\n\n"
            "## Request\n"
            "Review the failure.\n\n"
            "## Error\n"
            "Child failed"
        )
        assert (output_directory / "03-worker-task-3.md").read_text() == (
            "# worker\n\n"
            "- Task ID: `task-3`\n"
            "- Status: `cancelled`\n\n"
            "## Request\n"
            "Finish the change.\n\n"
            "## Report\n"
            "_No report returned._"
        )

    asyncio.run(exercise())


def test_batch_markdown_files_use_unique_sanitized_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted names cannot escape output roots and repeated calls do not collide."""

    async def exercise() -> None:
        backend = build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
        )
        store = BatchResultOutputStore(
            backend=backend,
            backend_prefix=generated_outputs_route_prefix(tmp_path),
        )
        identifiers = iter(("firstunique", "secondunique"))
        monkeypatch.setattr(
            background_tasks.uuid,
            "uuid4",
            lambda: SimpleNamespace(hex=next(identifiers)),
        )
        snapshots = [
            batch_snapshot(
                task_id="../task\\id",
                agent_name="../../reviewer / 🔬",
            ),
            batch_snapshot(task_id="!!!", agent_name="..."),
        ]

        first = await _write_batch_markdown_files(
            snapshots,
            tool_call_id="../batch / call",
            store=store,
        )
        second = await _write_batch_markdown_files(
            snapshots,
            tool_call_id="../batch / call",
            store=store,
        )

        first_paths = [item["path"] for item in first["files"]]
        second_paths = [item["path"] for item in second["files"]]
        assert first_paths == [
            (
                "/workspace/.files/outputs/subagent-batches/"
                "batch-call-firstunique/01-reviewer-task-id.md"
            ),
            (
                "/workspace/.files/outputs/subagent-batches/"
                "batch-call-firstunique/02-agent-task.md"
            ),
        ]
        assert second_paths == [
            path.replace("firstunique", "secondunique") for path in first_paths
        ]
        assert set(first_paths).isdisjoint(second_paths)
        for path in [*first_paths, *second_paths]:
            assert isinstance(path, str)
            assert path.startswith(
                "/workspace/.files/outputs/subagent-batches/"
            )
            assert ".." not in path
            assert "\\" not in path
            assert "🔬" not in path
            assert path.endswith(".md")
            assert resolve_generated_output(path, project_root=tmp_path) is not None

    asyncio.run(exercise())


class FailingOutputBackend:
    """Backend double that fails the second write and optionally every delete."""

    def __init__(self, *, fail_delete: bool = False) -> None:
        self.contents: dict[str, str] = {}
        self.write_count = 0
        self.fail_delete = fail_delete

    async def awrite(self, path: str, content: str) -> WriteResult:
        self.write_count += 1
        if self.write_count == 2:
            return WriteResult(error="disk full")
        self.contents[path] = content
        return WriteResult(path=path)

    async def adelete(self, path: str) -> DeleteResult:
        if self.fail_delete:
            return DeleteResult(error="cleanup denied")
        self.contents.pop(path, None)
        return DeleteResult(path=path)


@pytest.mark.parametrize("fail_delete", [False, True])
def test_batch_markdown_files_roll_back_partial_writes(
    fail_delete: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed batch never returns a partial manifest or leaves cleanable files."""

    async def exercise() -> None:
        backend = FailingOutputBackend(fail_delete=fail_delete)
        store = BatchResultOutputStore(
            backend=cast(Any, backend),
            backend_prefix="/backend/outputs/",
        )
        monkeypatch.setattr(
            background_tasks.uuid,
            "uuid4",
            lambda: SimpleNamespace(hex="batchunique"),
        )
        writer = _write_batch_markdown_files(
            [batch_snapshot(task_id="task-1"), batch_snapshot(task_id="task-2")],
            tool_call_id="batch-call",
            store=store,
        )

        if fail_delete:
            with pytest.raises(ExceptionGroup) as raised:
                await writer
            messages = [str(error) for error in raised.value.exceptions]
            assert any("disk full" in message for message in messages)
            assert any("cleanup denied" in message for message in messages)
            assert len(backend.contents) == 1
        else:
            with pytest.raises(RuntimeError, match="disk full") as raised:
                await writer
            assert "02-researcher-task-2.md" in str(raised.value)
            assert backend.contents == {}

    asyncio.run(exercise())


def test_batch_markdown_files_remove_the_failed_write_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend error may arrive after a partial destination was created."""

    class PartialWriteOutputBackend:
        def __init__(self) -> None:
            self.contents: dict[str, str] = {}
            self.deleted_paths: list[str] = []

        async def awrite(self, path: str, content: str) -> WriteResult:
            del content
            self.contents[path] = ""
            return WriteResult(error="disk full")

        async def adelete(self, path: str) -> DeleteResult:
            self.deleted_paths.append(path)
            self.contents.pop(path, None)
            return DeleteResult(path=path)

    async def exercise() -> None:
        backend = PartialWriteOutputBackend()
        store = BatchResultOutputStore(
            backend=cast(Any, backend),
            backend_prefix="/backend/outputs/",
        )
        monkeypatch.setattr(
            background_tasks.uuid,
            "uuid4",
            lambda: SimpleNamespace(hex="batchunique"),
        )
        expected_path = (
            "/backend/outputs/subagent-batches/"
            "batch-call-batchunique/01-researcher-task-1.md"
        )
        expected_directory = (
            "/backend/outputs/subagent-batches/batch-call-batchunique"
        )

        with pytest.raises(RuntimeError, match="disk full"):
            await _write_batch_markdown_files(
                [batch_snapshot(task_id="task-1")],
                tool_call_id="batch-call",
                store=store,
            )

        assert backend.contents == {}
        assert backend.deleted_paths == [expected_path, expected_directory]

    asyncio.run(exercise())


def test_batch_markdown_files_remove_real_filesystem_partial_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filesystem encoding failure must not leave its truncated destination."""

    async def exercise() -> None:
        backend = build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
        )
        store = BatchResultOutputStore(
            backend=backend,
            backend_prefix=generated_outputs_route_prefix(tmp_path),
        )
        monkeypatch.setattr(
            background_tasks.uuid,
            "uuid4",
            lambda: SimpleNamespace(hex="batchunique"),
        )

        with pytest.raises(RuntimeError, match="utf-8.*can't encode"):
            await _write_batch_markdown_files(
                [batch_snapshot(task_id="task-1", result="\ud800")],
                tool_call_id="batch-call",
                store=store,
            )

        output_root = tmp_path / ".files" / "outputs"
        assert not list(output_root.rglob("*.md"))
        assert not (
            output_root
            / "subagent-batches"
            / "batch-call-batchunique"
        ).exists()

    asyncio.run(exercise())


def test_batch_markdown_files_preserve_cleanup_errors_during_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation during rollback cannot replace a returned delete failure."""

    class BlockingCleanupOutputBackend(FailingOutputBackend):
        def __init__(self) -> None:
            super().__init__()
            self.cleanup_started = asyncio.Event()
            self.release_cleanup = asyncio.Event()

        async def adelete(self, path: str) -> DeleteResult:
            del path
            self.cleanup_started.set()
            await self.release_cleanup.wait()
            return DeleteResult(error="cleanup denied")

    def leaf_errors(error: BaseException) -> list[BaseException]:
        if isinstance(error, BaseExceptionGroup):
            return [
                leaf
                for nested in error.exceptions
                for leaf in leaf_errors(nested)
            ]
        return [error]

    async def exercise() -> None:
        backend = BlockingCleanupOutputBackend()
        store = BatchResultOutputStore(
            backend=cast(Any, backend),
            backend_prefix="/backend/outputs/",
        )
        monkeypatch.setattr(
            background_tasks.uuid,
            "uuid4",
            lambda: SimpleNamespace(hex="batchunique"),
        )
        write_task = asyncio.create_task(
            _write_batch_markdown_files(
                [batch_snapshot(task_id="task-1"), batch_snapshot(task_id="task-2")],
                tool_call_id="batch-call",
                store=store,
            )
        )
        await backend.cleanup_started.wait()
        write_task.cancel()
        backend.release_cleanup.set()

        with pytest.raises(BaseExceptionGroup) as raised:
            await write_task
        leaves = leaf_errors(raised.value)
        assert any("disk full" in str(error) for error in leaves)
        assert any("cleanup denied" in str(error) for error in leaves)
        assert any(isinstance(error, asyncio.CancelledError) for error in leaves)

    asyncio.run(exercise())


def test_batch_markdown_files_clean_up_a_write_completed_during_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller cancellation waits for an in-flight write, then removes it."""

    class BlockingOutputBackend:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.contents: dict[str, str] = {}
            self.deleted_paths: list[str] = []

        async def awrite(self, path: str, content: str) -> WriteResult:
            self.started.set()
            await self.release.wait()
            self.contents[path] = content
            return WriteResult(path=path)

        async def adelete(self, path: str) -> DeleteResult:
            self.deleted_paths.append(path)
            self.contents.pop(path, None)
            return DeleteResult(path=path)

    async def exercise() -> None:
        backend = BlockingOutputBackend()
        store = BatchResultOutputStore(
            backend=cast(Any, backend),
            backend_prefix="/backend/outputs/",
        )
        monkeypatch.setattr(
            background_tasks.uuid,
            "uuid4",
            lambda: SimpleNamespace(hex="batchunique"),
        )
        expected_backend_path = (
            "/backend/outputs/subagent-batches/"
            "batch-call-batchunique/01-researcher-task-1.md"
        )
        expected_directory = (
            "/backend/outputs/subagent-batches/batch-call-batchunique"
        )
        write_task = asyncio.create_task(
            _write_batch_markdown_files(
                [batch_snapshot(task_id="task-1")],
                tool_call_id="batch-call",
                store=store,
            )
        )
        await backend.started.wait()
        write_task.cancel()
        backend.release.set()

        with pytest.raises(asyncio.CancelledError):
            await write_task
        assert backend.contents == {}
        assert backend.deleted_paths == [expected_backend_path, expected_directory]

    asyncio.run(exercise())


def test_run_subagent_batch_markdown_files_uses_the_native_output_store(
    tmp_path: Path,
) -> None:
    """The tool selects file mode through a factory dependency, not a tool arg."""

    class ChildRunnable:
        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            del state, config
            return {"messages": [AIMessage(content="Stored report")]}

    async def exercise() -> None:
        manager = make_manager(batch_result_format="markdown_files")
        backend = build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
        )
        store = BatchResultOutputStore(
            backend=backend,
            backend_prefix=generated_outputs_route_prefix(tmp_path),
        )
        tools = create_background_task_tools(
            manager=manager,
            subagents={"researcher": ChildRunnable()},
            agent_path=(),
            recursion_limit=20,
            batch_output_store=store,
        )
        batch_tool = next(
            tool for tool in tools if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )

        result = await batch_tool.coroutine(
            [{"description": "Inspect it", "subagent_type": "researcher"}],
            runtime,
        )

        assert isinstance(result, dict)
        files = result["files"]
        assert isinstance(files, list)
        assert len(files) == 1
        assert files[0]["path"].startswith(
            "/workspace/.files/outputs/subagent-batches/batch-call-"
        )
        assert files[0]["path"].endswith("-researcher-" + files[0]["task_id"] + ".md")
        await manager.close()

    asyncio.run(exercise())


def test_run_subagent_batch_markdown_files_requires_an_output_store() -> None:
    """Miswired file mode fails explicitly after the child reaches terminal state."""

    class ChildRunnable:
        async def ainvoke(self, state, config):
            del state, config
            return {"messages": [AIMessage(content="Stored report")]}

    async def exercise() -> None:
        manager = make_manager(batch_result_format="markdown_files")
        batch_tool = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": ChildRunnable()},
                agent_path=(),
                recursion_limit=20,
            )
            if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )

        with pytest.raises(RuntimeError, match="generated-output store"):
            await batch_tool.coroutine(
                [{"description": "Inspect it", "subagent_type": "researcher"}],
                runtime,
            )
        await manager.close()

    asyncio.run(exercise())


def test_run_subagent_batch_has_an_openai_compatible_nested_schema() -> None:
    """Cortex-compatible tool conversion must retain both batch item fields."""
    manager = make_manager()
    batch_tool = next(
        tool
        for tool in create_background_task_tools(
            manager=manager,
            subagents={"researcher": object()},
            agent_path=(),
            recursion_limit=20,
        )
        if tool.name == "run_subagent_batch"
    )

    schema = convert_to_openai_tool(batch_tool)["function"]["parameters"]

    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"tasks"}
    assert schema["required"] == ["tasks"]
    item_schema = schema["properties"]["tasks"]["items"]
    assert item_schema["type"] == "object"
    assert item_schema["required"] == ["description", "subagent_type"]


def test_run_subagent_batch_rejects_the_whole_batch_when_capacity_is_insufficient() -> None:
    """Batch admission must not launch a prefix that happens to fit."""

    class ChildRunnable:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            self.started.set()
            return {"messages": [AIMessage(content="unexpected")]}

    async def exercise() -> None:
        manager = make_manager(max_running_per_session=2, max_running_total=3)
        blocker_release = asyncio.Event()

        async def blocker(task_id: str) -> str:
            await blocker_release.wait()
            return task_id

        existing = await manager.spawn(
            session_id="session-a",
            agent_name="existing",
            description="existing",
            agent_path=("existing",),
            runner=blocker,
        )
        child = ChildRunnable()
        batch_tool = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": child},
                agent_path=(),
                recursion_limit=20,
            )
            if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )

        with pytest.raises(RuntimeError, match="running limit.*session"):
            await batch_tool.coroutine(
                [
                    {"description": "one", "subagent_type": "researcher"},
                    {"description": "two", "subagent_type": "researcher"},
                ],
                runtime,
            )

        assert not child.started.is_set()
        assert [task.task_id for task in await manager.list("session-a")] == [
            existing.task_id
        ]
        blocker_release.set()
        await manager.close()

    asyncio.run(exercise())


def test_run_subagent_batch_validates_every_request_before_launching() -> None:
    """Empty, blank, and unknown work must leave the session untouched."""

    class ChildRunnable:
        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            return {"messages": [AIMessage(content="unexpected")]}

    async def exercise() -> None:
        manager = make_manager()
        batch_tool = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": ChildRunnable()},
                agent_path=(),
                recursion_limit=20,
            )
            if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )

        with pytest.raises(ValueError, match="at least one"):
            await batch_tool.coroutine([], runtime)
        with pytest.raises(ValueError, match="description"):
            await batch_tool.coroutine(
                [{"description": "   ", "subagent_type": "researcher"}],
                runtime,
            )
        with pytest.raises(ValueError, match="Allowed subagents"):
            await batch_tool.coroutine(
                [{"description": "work", "subagent_type": "unknown"}],
                runtime,
            )

        assert await manager.list("session-a") == []
        await manager.close()

    asyncio.run(exercise())


def test_run_subagent_batch_keeps_sibling_results_when_one_child_fails() -> None:
    """A failed child must become one result instead of failing the whole tool."""

    class ChildRunnable:
        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            messages = state["messages"]
            assert isinstance(messages, list)
            description = str(messages[0].content)
            if description == "fail":
                raise RuntimeError("child failed")
            return {"messages": [AIMessage(content="survived")]}

    async def exercise() -> None:
        manager = make_manager(max_running_per_session=4, max_running_total=4)
        batch_tool = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": ChildRunnable()},
                agent_path=(),
                recursion_limit=20,
            )
            if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )

        result = await batch_tool.coroutine(
            [
                {"description": "fail", "subagent_type": "researcher"},
                {"description": "succeed", "subagent_type": "researcher"},
            ],
            runtime,
        )

        assert isinstance(result, str)
        assert "## 1. researcher" in result
        assert "- Status: `error`" in result
        assert "### Request\nfail\n\n### Error\nRuntimeError: child failed" in result
        assert "## 2. researcher" in result
        assert "- Status: `success`" in result
        assert "### Request\nsucceed\n\n### Report\nsurvived" in result
        await manager.close()

    asyncio.run(exercise())


def test_cancelling_run_subagent_batch_cancels_every_unfinished_child() -> None:
    """Cancelling a waiting batch call must not orphan its child tasks."""

    class ChildRunnable:
        def __init__(self) -> None:
            self.started = 0
            self.both_started = asyncio.Event()
            self.cancelled = 0
            self.both_cancelled = asyncio.Event()

        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            self.started += 1
            if self.started == 2:
                self.both_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                if self.cancelled == 2:
                    self.both_cancelled.set()
                raise

    async def exercise() -> None:
        manager = make_manager(max_running_per_session=4, max_running_total=4)
        child = ChildRunnable()
        batch_tool = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": child},
                agent_path=(),
                recursion_limit=20,
            )
            if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )
        batch_call = asyncio.create_task(
            batch_tool.coroutine(
                [
                    {"description": "one", "subagent_type": "researcher"},
                    {"description": "two", "subagent_type": "researcher"},
                ],
                runtime,
            )
        )
        await asyncio.wait_for(child.both_started.wait(), timeout=1)

        batch_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await batch_call

        await asyncio.wait_for(child.both_cancelled.wait(), timeout=1)
        assert [task.status for task in await manager.list("session-a")] == [
            "cancelled",
            "cancelled",
        ]
        await manager.close()

    asyncio.run(exercise())


def test_cancelling_batch_preserves_descendants_of_completed_children() -> None:
    """Cancelling a batch wait must not cancel work owned by a finished child."""

    class ChildRunnable:
        def __init__(self, manager: BackgroundTaskManager) -> None:
            self.manager = manager
            self.finished_task_id: str | None = None
            self.descendant_task_id: str | None = None
            self.descendant_started = asyncio.Event()
            self.descendant_cancelled = asyncio.Event()
            self.unfinished_started = asyncio.Event()
            self.unfinished_cancelled = asyncio.Event()

        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            messages = state["messages"]
            assert isinstance(messages, list)
            description = str(messages[0].content)
            if description == "finished":
                self.finished_task_id = current_background_task_id()
                assert self.finished_task_id is not None

                async def run_descendant(task_id: str) -> str:
                    self.descendant_started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        self.descendant_cancelled.set()
                        raise

                descendant = await self.manager.spawn(
                    session_id="session-a",
                    agent_name="descendant",
                    description="surviving descendant",
                    agent_path=("researcher", "descendant"),
                    parent_task_id=self.finished_task_id,
                    runner=run_descendant,
                )
                self.descendant_task_id = descendant.task_id
                return {"messages": [AIMessage(content="done")]}

            self.unfinished_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.unfinished_cancelled.set()
                raise

    async def exercise() -> None:
        manager = make_manager(max_running_per_session=5, max_running_total=5)
        child = ChildRunnable(manager)
        batch_tool = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": child},
                agent_path=(),
                recursion_limit=20,
            )
            if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )
        batch_call = asyncio.create_task(
            batch_tool.coroutine(
                [
                    {"description": "finished", "subagent_type": "researcher"},
                    {"description": "unfinished", "subagent_type": "researcher"},
                ],
                runtime,
            )
        )
        await asyncio.wait_for(child.descendant_started.wait(), timeout=1)
        await asyncio.wait_for(child.unfinished_started.wait(), timeout=1)
        assert child.finished_task_id is not None
        finished = await manager.get(
            "session-a",
            child.finished_task_id,
            wait_seconds=1,
        )
        assert finished.status == "success"

        batch_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await batch_call

        await asyncio.wait_for(child.unfinished_cancelled.wait(), timeout=1)
        await asyncio.sleep(0)
        assert not child.descendant_cancelled.is_set()
        assert child.descendant_task_id is not None
        descendant = await manager.get("session-a", child.descendant_task_id)
        assert descendant.status == "running"

        await manager.close()
        assert child.descendant_cancelled.is_set()

    asyncio.run(exercise())


def test_run_subagent_batch_returns_cancelled_results_when_session_closes() -> None:
    """Session teardown must release a waiting batch call without orphaning work."""

    class ChildRunnable:
        def __init__(self) -> None:
            self.started = 0
            self.both_started = asyncio.Event()

        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            self.started += 1
            if self.started == 2:
                self.both_started.set()
            await asyncio.Event().wait()
            return {"messages": [AIMessage(content="unreachable")]}

    async def exercise() -> None:
        manager = make_manager(max_running_per_session=4, max_running_total=4)
        child = ChildRunnable()
        batch_tool = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": child},
                agent_path=(),
                recursion_limit=20,
            )
            if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )
        batch_call = asyncio.create_task(
            batch_tool.coroutine(
                [
                    {"description": "one", "subagent_type": "researcher"},
                    {"description": "two", "subagent_type": "researcher"},
                ],
                runtime,
            )
        )
        await asyncio.wait_for(child.both_started.wait(), timeout=1)

        await manager.close_session("session-a")
        result = await asyncio.wait_for(batch_call, timeout=1)

        assert isinstance(result, str)
        assert result.count("- Status: `cancelled`") == 2
        assert result.count("### Report\n_No report returned._") == 2
        assert await manager.list("session-a") == []
        await manager.close()

    asyncio.run(exercise())


def test_run_subagent_batch_marks_an_empty_success_report() -> None:
    """An empty successful child response must remain explicit in Markdown."""

    class ChildRunnable:
        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            return {"messages": [AIMessage(content="")]}

    async def exercise() -> None:
        manager = make_manager()
        batch_tool = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": ChildRunnable()},
                agent_path=(),
                recursion_limit=20,
            )
            if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )

        result = await batch_tool.coroutine(
            [{"description": "empty", "subagent_type": "researcher"}],
            runtime,
        )

        assert "- Status: `success`" in result
        assert "### Report\n_No report returned._" in result
        await manager.close()

    asyncio.run(exercise())


def test_wait_batch_waits_for_cancellation_cleanup_to_finalize_records() -> None:
    """A cancelled execution is not terminal until its cleanup has finished."""

    async def exercise() -> None:
        manager = make_manager()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()

        async def runner(task_id: str) -> str:
            return task_id

        async def cleanup(task_id: str) -> None:
            cleanup_started.set()
            await release_cleanup.wait()

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="researcher",
            description="work",
            agent_path=("researcher",),
            runner=runner,
            cleanup=cleanup,
        )
        execution = manager._records[spawned.task_id].execution
        assert execution is not None
        waiter = asyncio.create_task(
            manager.wait_batch("session-a", [spawned.task_id])
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        cancellation = asyncio.create_task(
            manager.cancel("session-a", spawned.task_id)
        )

        await asyncio.gather(execution, return_exceptions=True)
        await asyncio.sleep(0)
        assert not cancellation.done()
        assert not waiter.done()

        release_cleanup.set()
        cancelled = await asyncio.wait_for(cancellation, timeout=1)
        completed = await asyncio.wait_for(waiter, timeout=1)

        assert cancelled.status == "cancelled"
        assert [snapshot.status for snapshot in completed] == ["cancelled"]
        assert completed[0].completed_at is not None
        await manager.close()

    asyncio.run(exercise())


def test_wait_batch_waits_for_session_close_cleanup_to_finalize_records() -> None:
    """Session close must publish terminal snapshots before forgetting records."""

    async def exercise() -> None:
        manager = make_manager()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()

        async def runner(task_id: str) -> str:
            return task_id

        async def cleanup(task_id: str) -> None:
            cleanup_started.set()
            await release_cleanup.wait()

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="researcher",
            description="work",
            agent_path=("researcher",),
            runner=runner,
            cleanup=cleanup,
        )
        execution = manager._records[spawned.task_id].execution
        assert execution is not None
        waiter = asyncio.create_task(
            manager.wait_batch("session-a", [spawned.task_id])
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        closing = asyncio.create_task(manager.close_session("session-a"))

        await asyncio.gather(execution, return_exceptions=True)
        await asyncio.sleep(0)
        assert not closing.done()
        assert not waiter.done()

        release_cleanup.set()
        completed = await asyncio.wait_for(waiter, timeout=1)
        await asyncio.wait_for(closing, timeout=1)

        assert [snapshot.status for snapshot in completed] == ["cancelled"]
        assert completed[0].completed_at is not None
        assert await manager.list("session-a") == []
        await manager.close()

    asyncio.run(exercise())


def test_concurrent_subagent_batches_share_atomic_capacity_limits() -> None:
    """Two simultaneous batches cannot both pass a shared capacity boundary."""

    class ChildRunnable:
        def __init__(self) -> None:
            self.started = 0
            self.first_batch_started = asyncio.Event()
            self.release = asyncio.Event()

        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            self.started += 1
            if self.started == 2:
                self.first_batch_started.set()
            await self.release.wait()
            return {"messages": [AIMessage(content="done")]}

    async def exercise() -> None:
        manager = make_manager(max_running_per_session=3, max_running_total=3)
        child = ChildRunnable()
        batch_tool = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": child},
                agent_path=(),
                recursion_limit=20,
            )
            if tool.name == "run_subagent_batch"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="batch-call",
            store=None,
        )
        requests = [
            {"description": "one", "subagent_type": "researcher"},
            {"description": "two", "subagent_type": "researcher"},
        ]
        first = asyncio.create_task(batch_tool.coroutine(requests, runtime))
        await asyncio.wait_for(child.first_batch_started.wait(), timeout=1)

        with pytest.raises(RuntimeError, match="running limit.*session"):
            await batch_tool.coroutine(requests, runtime)

        assert len(await manager.list("session-a")) == 2
        child.release.set()
        await asyncio.wait_for(first, timeout=1)
        await manager.close()

    asyncio.run(exercise())


def test_background_tools_reject_reserved_name_collisions() -> None:
    manager = make_manager()

    with pytest.raises(
        ValueError,
        match="reserved background task tool name.*spawn_background_task",
    ):
        create_background_task_tools(
            manager=manager,
            subagents={},
            agent_path=(),
            recursion_limit=20,
            existing_tools=[SimpleNamespace(name="spawn_background_task")],
        )

    with pytest.raises(
        ValueError,
        match="reserved background task tool name.*run_subagent_batch",
    ):
        create_background_task_tools(
            manager=manager,
            subagents={},
            agent_path=(),
            recursion_limit=20,
            existing_tools=[SimpleNamespace(name="run_subagent_batch")],
        )


def test_nested_background_tool_uses_the_original_conversation_session() -> None:
    class LeafRunnable:
        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            return {"messages": [AIMessage(content="nested result")]}

    async def exercise() -> None:
        manager = make_manager(max_running_per_session=4, max_running_total=4)
        nested_tools = create_background_task_tools(
            manager=manager,
            subagents={"worker": LeafRunnable()},
            agent_path=("manager",),
            recursion_limit=20,
        )
        nested_spawn = next(
            tool for tool in nested_tools if tool.name == "spawn_background_task"
        )

        class ParentRunnable:
            async def ainvoke(
                self,
                state: dict[str, object],
                config: dict[str, object],
            ) -> dict[str, object]:
                runtime = ToolRuntime(
                    state=state,
                    context=None,
                    config=config,
                    stream_writer=lambda _: None,
                    tool_call_id="nested-call",
                    store=None,
                )
                spawned = await nested_spawn.coroutine(
                    "nested work",
                    "worker",
                    runtime,
                )
                return {
                    "messages": [
                        AIMessage(content=f"parent spawned {spawned['task_id']}")
                    ]
                }

        main_tools = create_background_task_tools(
            manager=manager,
            subagents={"manager": ParentRunnable()},
            agent_path=(),
            recursion_limit=20,
        )
        main_spawn = next(
            tool for tool in main_tools if tool.name == "spawn_background_task"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="main-call",
            store=None,
        )

        parent = await main_spawn.coroutine("parent work", "manager", runtime)
        completed = await manager.wait_session("session-a")

        assert len(completed) == 2
        parent_snapshot = next(task for task in completed if task.task_id == parent["task_id"])
        child_snapshot = next(task for task in completed if task.parent_task_id == parent["task_id"])
        assert parent_snapshot.status == "success"
        assert child_snapshot.status == "success"
        assert child_snapshot.session_id == "session-a"
        assert child_snapshot.agent_path == ("manager", "worker")
        assert child_snapshot.result == "nested result"
        await manager.close()

    asyncio.run(exercise())


def test_parallel_foreground_subagent_invocations_have_isolated_task_scopes() -> None:
    class LeafRunnable:
        def __init__(self, release: asyncio.Event) -> None:
            self.release = release

        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            await self.release.wait()
            return {"messages": [AIMessage(content="done")]}

    async def exercise() -> None:
        manager = make_manager(max_running_per_session=4, max_running_total=4)
        release = asyncio.Event()
        tools = create_background_task_tools(
            manager=manager,
            subagents={"worker": LeafRunnable(release)},
            agent_path=("manager",),
            recursion_limit=20,
        )
        by_name = {tool.name: tool for tool in tools}
        spawned_by_label: dict[str, str] = {}
        both_spawned = asyncio.Event()

        class ForegroundInvocation:
            async def ainvoke(
                self,
                state: dict[str, object],
                config: dict[str, object],
            ) -> dict[str, object]:
                label = str(state["label"])
                runtime = ToolRuntime(
                    state=state,
                    context=None,
                    config=config,
                    stream_writer=lambda _: None,
                    tool_call_id=f"spawn-{label}",
                    store=None,
                )
                spawned = await by_name["spawn_background_task"].coroutine(
                    f"work {label}",
                    "worker",
                    runtime,
                )
                spawned_by_label[label] = str(spawned["task_id"])
                if len(spawned_by_label) == 2:
                    both_spawned.set()
                await both_spawned.wait()

                visible = await by_name["list_background_tasks"].coroutine(runtime)
                other_label = "b" if label == "a" else "a"
                try:
                    await by_name["get_background_task"].coroutine(
                        spawned_by_label[other_label],
                        0,
                        runtime,
                    )
                except KeyError:
                    cross_visible = False
                else:
                    cross_visible = True
                return {
                    "visible_ids": [str(task["task_id"]) for task in visible],
                    "cross_visible": cross_visible,
                }

        runnable = scope_background_task_invocation(ForegroundInvocation())
        config = {"configurable": {"thread_id": "session-a"}}
        first, second = await asyncio.gather(
            runnable.ainvoke({"label": "a"}, config),
            runnable.ainvoke({"label": "b"}, config),
        )

        assert first == {
            "visible_ids": [spawned_by_label["a"]],
            "cross_visible": False,
        }
        assert second == {
            "visible_ids": [spawned_by_label["b"]],
            "cross_visible": False,
        }

        release.set()
        await manager.wait_session("session-a")
        await manager.close()

    asyncio.run(exercise())


def test_top_level_config_cannot_override_the_conversation_session() -> None:
    async def exercise() -> None:
        manager = make_manager()

        async def runner(task_id: str) -> str:
            return task_id

        await manager.spawn(
            session_id="session-b",
            agent_name="worker",
            description="private work",
            agent_path=("worker",),
            runner=runner,
        )
        tools = create_background_task_tools(
            manager=manager,
            subagents={},
            agent_path=(),
            recursion_limit=20,
        )
        list_tool = next(tool for tool in tools if tool.name == "list_background_tasks")
        injected_runtime = ToolRuntime(
            state={},
            context=None,
            config={
                "configurable": {
                    "thread_id": "session-a",
                    "__chainagents_background_session_id": "session-b",
                }
            },
            stream_writer=lambda _: None,
            tool_call_id="injected-call",
            store=None,
        )

        assert await list_tool.coroutine(injected_runtime) == []
        await manager.close()

    asyncio.run(exercise())


def test_real_graph_parent_continues_while_background_child_runs() -> None:
    """Exercise the real DeepAgents graph without streaming child messages."""

    async def exercise() -> None:
        manager = make_manager()
        checkpointer = MemorySaver()
        child_started = asyncio.Event()
        child_release = asyncio.Event()

        @tool
        async def wait_for_release() -> str:
            """Wait until the deterministic test releases the child."""
            child_started.set()
            await child_release.wait()
            return "released"

        child = create_deep_agent(
            model=_ToolCallingFakeModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "wait_for_release",
                                "args": {},
                                "id": "child-call",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="private child result"),
                ]
            ),
            tools=[wait_for_release],
            checkpointer=True,
        )
        background_tools = create_background_task_tools(
            manager=manager,
            subagents={"researcher": child},
            agent_path=(),
            recursion_limit=20,
        )
        parent = create_deep_agent(
            model=_ToolCallingFakeModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "spawn_background_task",
                                "args": {
                                    "description": "research",
                                    "subagent_type": "researcher",
                                },
                                "id": "parent-call",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="parent continued immediately"),
                ]
            ),
            tools=background_tools,
            checkpointer=checkpointer,
        )

        result = await parent.ainvoke(
            {"messages": [HumanMessage(content="start both jobs")]},
            {"configurable": {"thread_id": "session-a"}},
        )
        await asyncio.wait_for(child_started.wait(), timeout=1)

        assert result["messages"][-1].text == "parent continued immediately"
        assert all(
            "private child result" not in str(message.content)
            for message in result["messages"]
        )
        tasks = await manager.list("session-a")
        assert len(tasks) == 1
        assert tasks[0].status == "running"
        child_checkpoint_config = {
            "configurable": {
                "thread_id": f"session-a:background:{tasks[0].task_id}"
            }
        }
        for _ in range(20):
            if await checkpointer.aget_tuple(child_checkpoint_config) is not None:
                break
            await asyncio.sleep(0)
        assert await checkpointer.aget_tuple(child_checkpoint_config) is not None

        child_release.set()
        completed = await manager.wait_session("session-a")
        assert completed[0].status == "success"
        assert completed[0].result == "private child result"
        assert await checkpointer.aget_tuple(child_checkpoint_config) is None
        await manager.close()

    asyncio.run(exercise())


def test_real_graph_runs_a_subagent_batch_from_one_supervisor_tool_call() -> None:
    """A single model tool call must fan out into separate concurrent child runs."""

    class ChildRunnable:
        def __init__(self) -> None:
            self.started = 0
            self.both_started = asyncio.Event()

        async def ainvoke(
            self,
            state: dict[str, object],
            config: dict[str, object],
        ) -> dict[str, object]:
            self.started += 1
            if self.started == 2:
                self.both_started.set()
            await self.both_started.wait()
            messages = state["messages"]
            assert isinstance(messages, list)
            description = str(messages[0].content)
            return {"messages": [AIMessage(content=f"report:{description}")]}

    async def exercise() -> None:
        manager = make_manager(max_running_per_session=4, max_running_total=4)
        child = ChildRunnable()
        tools = create_background_task_tools(
            manager=manager,
            subagents={"researcher": child},
            agent_path=(),
            recursion_limit=20,
        )
        parent = create_deep_agent(
            model=_ToolCallingFakeModel(
                responses=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "run_subagent_batch",
                                "args": {
                                    "tasks": [
                                        {
                                            "description": "alpha",
                                            "subagent_type": "researcher",
                                        },
                                        {
                                            "description": "beta",
                                            "subagent_type": "researcher",
                                        },
                                    ]
                                },
                                "id": "batch-call",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="synthesized result"),
                ]
            ),
            tools=tools,
        )

        result = await parent.ainvoke(
            {"messages": [HumanMessage(content="research both topics")]},
            {"configurable": {"thread_id": "session-a"}},
        )

        assert result["messages"][-1].text == "synthesized result"
        completed = await manager.list("session-a")
        assert [task.description for task in completed] == ["alpha", "beta"]
        assert [task.result for task in completed] == [
            "report:alpha",
            "report:beta",
        ]
        assert len({task.task_id for task in completed}) == 2
        await manager.close()

    asyncio.run(exercise())


def test_background_spawn_tool_requires_explicit_session_and_allowed_child() -> None:
    async def exercise() -> None:
        manager = make_manager()
        tools = create_background_task_tools(
            manager=manager,
            subagents={},
            agent_path=("planner",),
            recursion_limit=20,
        )
        spawn_tool = next(tool for tool in tools if tool.name == "spawn_background_task")
        no_session = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {}},
            stream_writer=lambda _: None,
            tool_call_id="call-1",
            store=None,
        )

        with pytest.raises(ValueError, match="conversation identity"):
            await spawn_tool.coroutine("work", "researcher", no_session)

        session = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="call-2",
            store=None,
        )
        with pytest.raises(ValueError, match="Allowed subagents"):
            await spawn_tool.coroutine("work", "researcher", session)
        await manager.close()

    asyncio.run(exercise())


def test_foreground_cancellation_does_not_cancel_spawned_background_work() -> None:
    async def exercise() -> None:
        child_started = asyncio.Event()
        child_release = asyncio.Event()
        foreground_block = asyncio.Event()
        spawned_id: str | None = None
        manager = make_manager()

        class ChildRunnable:
            async def ainvoke(
                self,
                state: dict[str, object],
                config: dict[str, object],
            ) -> dict[str, object]:
                child_started.set()
                await child_release.wait()
                return {"messages": [AIMessage(content="survived cancellation")]}

        spawn_tool = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"worker": ChildRunnable()},
                agent_path=(),
                recursion_limit=20,
            )
            if tool.name == "spawn_background_task"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-a"}},
            stream_writer=lambda _: None,
            tool_call_id="call-1",
            store=None,
        )

        async def foreground() -> None:
            nonlocal spawned_id
            spawned = await spawn_tool.coroutine("work", "worker", runtime)
            spawned_id = str(spawned["task_id"])
            await foreground_block.wait()

        foreground_task = asyncio.create_task(foreground())
        await asyncio.wait_for(child_started.wait(), timeout=1)
        foreground_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await foreground_task

        assert spawned_id is not None
        assert (await manager.get("session-a", spawned_id)).status == "running"
        child_release.set()
        finished = await manager.get("session-a", spawned_id, wait_seconds=1)
        assert finished.status == "success"
        assert finished.result == "survived cancellation"
        await manager.close()

    asyncio.run(exercise())


def test_chainlit_local_notifier_sends_status_without_dumping_result(monkeypatch) -> None:
    """A completion notice should keep successful output available only on demand."""
    async def exercise() -> None:
        manager = make_manager()
        sent: list[tuple[str, str]] = []
        delivered = asyncio.Event()

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                self.content = content
                self.author = author

            async def send(self) -> None:
                sent.append((self.author, self.content))
                delivered.set()

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message",
            Message,
        )
        notifier = LocalBackgroundTaskNotifier(
            manager=manager,
            session_id="session-a",
        )
        notifier.start()

        async def runner(task_id: str) -> str:
            return "finished result"

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="researcher",
            description="research",
            agent_path=("researcher",),
            runner=runner,
        )
        await asyncio.wait_for(delivered.wait(), timeout=1)

        assert sent == [
            (
                "Background subagent",
                "Local background subagent `researcher` finished with status "
                f"`success`.\n\nTask ID: `{spawned.task_id}`",
            )
        ]
        notifier.cancel()
        await manager.close()

    asyncio.run(exercise())


def test_chainlit_local_notifier_streams_nested_activity_before_completion(
    monkeypatch,
) -> None:
    """Enabled activity streaming should render one ordered nested task tree."""
    async def exercise() -> None:
        manager = make_manager(stream_activity=True)
        timeline: list[tuple[str, str, str]] = []
        steps: list[Step] = []
        delivered = asyncio.Event()

        class Step:
            def __init__(
                self,
                *,
                name: str,
                type: str,
                parent_id: str | None = None,
                **kwargs: object,
            ) -> None:
                self.id = f"step-{len(steps) + 1}"
                self.name = name
                self.type = type
                self.parent_id = parent_id
                self.input = ""
                self.output = ""
                self.end = None
                self.tokens: list[str] = []
                steps.append(self)

            async def send(self) -> None:
                timeline.append(("send", self.id, self.name))

            async def update(self) -> None:
                timeline.append(("update", self.id, str(self.output)))

            async def stream_token(self, token: str) -> None:
                self.tokens.append(token)
                timeline.append(("token", self.id, token))

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                self.content = content
                self.author = author

            async def send(self) -> None:
                timeline.append(("message", self.author, self.content))
                delivered.set()

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Step",
            Step,
        )
        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message",
            Message,
        )
        notifier = LocalBackgroundTaskNotifier(
            manager=manager,
            session_id="session-a",
        )
        notifier.start()

        async def runner(task_id: str) -> str:
            for event in (
                AgentStreamEvent(
                    kind="reasoning_delta",
                    source="researcher",
                    text="Inspecting ",
                ),
                AgentStreamEvent(
                    kind="reasoning_delta",
                    source="researcher",
                    text="the repository",
                ),
                AgentStreamEvent(
                    kind="tool_call",
                    source="researcher / worker",
                    tool_call_id="researcher / worker:0",
                    tool_name="read_file",
                    tool_args='{"file_path":"skills/example/SKILL.md"}',
                ),
                AgentStreamEvent(
                    kind="tool_result",
                    source="researcher / worker",
                    tool_call_id="call-real",
                    tool_name="read_file",
                    tool_result="skill contents",
                    status="success",
                ),
            ):
                await manager.publish_activity(task_id, event)
            return "finished result"

        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="researcher",
            description="research",
            agent_path=("researcher",),
            runner=runner,
        )
        await asyncio.wait_for(delivered.wait(), timeout=2)

        parent = next(step for step in steps if step.type == "run")
        reasoning = next(step for step in steps if step.type == "llm")
        tool = next(step for step in steps if step.type == "tool")
        assert parent.name == "researcher (background)"
        assert parent.parent_id is None
        assert reasoning.name == "researcher reasoning"
        assert reasoning.parent_id == parent.id
        assert reasoning.tokens == ["Inspecting the repository"]
        assert tool.name == "researcher / worker · read_file"
        assert tool.parent_id == parent.id
        assert tool.input == '{"file_path":"skills/example/SKILL.md"}'
        assert tool.output == "skill contents"
        assert parent.end is not None
        assert reasoning.end is not None
        assert tool.end is not None
        assert timeline[-1] == (
            "message",
            "Background subagent",
            "Local background subagent `researcher` finished with status "
            f"`success`.\n\nTask ID: `{spawned.task_id}`",
        )
        assert sum(item[0] == "message" for item in timeline) == 1

        notifier.cancel()
        await manager.close()

    asyncio.run(exercise())


def test_chainlit_local_notifier_closes_remaining_steps_after_update_failure(
    monkeypatch,
) -> None:
    """One failed child update must not strand the rest of the activity tree."""
    async def exercise() -> None:
        manager = make_manager(stream_activity=True)
        steps: list[Step] = []
        delivered = asyncio.Event()

        class Step:
            def __init__(
                self,
                *,
                name: str,
                type: str,
                parent_id: str | None = None,
                **kwargs: object,
            ) -> None:
                self.id = f"step-{len(steps) + 1}"
                self.name = name
                self.type = type
                self.parent_id = parent_id
                self.input = ""
                self.output = ""
                self.end = None
                self.update_attempted = False
                steps.append(self)

            async def send(self) -> None:
                return None

            async def update(self) -> None:
                self.update_attempted = True
                if self.type == "llm":
                    raise RuntimeError("reasoning update failed")

            async def stream_token(self, token: str) -> None:
                return None

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                self.content = content
                self.author = author

            async def send(self) -> None:
                delivered.set()

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Step",
            Step,
        )
        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message",
            Message,
        )
        notifier = LocalBackgroundTaskNotifier(
            manager=manager,
            session_id="session-a",
        )
        notifier.start()

        async def runner(task_id: str) -> str:
            await manager.publish_activity(
                task_id,
                AgentStreamEvent(
                    kind="reasoning_delta",
                    source="researcher",
                    text="thinking",
                ),
            )
            await manager.publish_activity(
                task_id,
                AgentStreamEvent(
                    kind="tool_call",
                    source="researcher",
                    tool_call_id="call-1",
                    tool_name="search",
                ),
            )
            return "finished"

        await manager.spawn(
            session_id="session-a",
            agent_name="researcher",
            description="research",
            agent_path=("researcher",),
            runner=runner,
        )
        await asyncio.wait_for(delivered.wait(), timeout=2)

        reasoning = next(step for step in steps if step.type == "llm")
        tool = next(step for step in steps if step.type == "tool")
        parent = next(step for step in steps if step.type == "run")
        assert reasoning.update_attempted is True
        assert tool.update_attempted is True
        assert parent.update_attempted is True
        assert tool.end is not None
        assert parent.end is not None

        notifier.cancel()
        await manager.close()

    asyncio.run(exercise())


def test_chainlit_local_notifier_async_close_ends_open_activity_steps(
    monkeypatch,
) -> None:
    """Session teardown should close rendered steps before discarding state."""
    async def exercise() -> None:
        manager = make_manager(stream_activity=True)
        steps: list[Step] = []
        rendered = asyncio.Event()
        release = asyncio.Event()

        class Step:
            def __init__(
                self,
                *,
                name: str,
                type: str,
                parent_id: str | None = None,
                **kwargs: object,
            ) -> None:
                self.id = f"step-{len(steps) + 1}"
                self.name = name
                self.type = type
                self.parent_id = parent_id
                self.input = ""
                self.output = ""
                self.end = None
                steps.append(self)

            async def send(self) -> None:
                if self.type == "llm":
                    rendered.set()

            async def update(self) -> None:
                return None

            async def stream_token(self, token: str) -> None:
                return None

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Step",
            Step,
        )
        notifier = LocalBackgroundTaskNotifier(
            manager=manager,
            session_id="session-a",
        )
        notifier.start()

        async def runner(task_id: str) -> str:
            await manager.publish_activity(
                task_id,
                AgentStreamEvent(
                    kind="reasoning_delta",
                    source="researcher",
                    text="thinking",
                ),
            )
            await release.wait()
            return "finished"

        await manager.spawn(
            session_id="session-a",
            agent_name="researcher",
            description="research",
            agent_path=("researcher",),
            runner=runner,
        )
        await asyncio.wait_for(rendered.wait(), timeout=2)

        await notifier.aclose()

        assert notifier.task is None
        assert notifier.activity_queue is None
        assert notifier.activity_states == {}
        assert "session-a" not in manager._activity_subscribers
        assert all(step.end is not None for step in steps)
        assert next(step for step in steps if step.type == "run").output == "Stopped"

        release.set()
        await manager.close()

    asyncio.run(exercise())


def test_chainlit_local_notifier_async_close_resumes_interrupted_terminal_close(
    monkeypatch,
) -> None:
    """Cancelling the consumer mid-terminal-close must retain cleanup state."""
    async def exercise() -> None:
        manager = make_manager(stream_activity=True)
        steps: list[Step] = []
        terminal_close_started = asyncio.Event()
        blocked_once = False

        class Step:
            def __init__(
                self,
                *,
                name: str,
                type: str,
                parent_id: str | None = None,
                **kwargs: object,
            ) -> None:
                self.id = f"step-{len(steps) + 1}"
                self.name = name
                self.type = type
                self.parent_id = parent_id
                self.input = ""
                self.output = ""
                self.end = None
                self.update_attempts = 0
                steps.append(self)

            async def send(self) -> None:
                return None

            async def update(self) -> None:
                nonlocal blocked_once
                self.update_attempts += 1
                if self.type == "llm" and not blocked_once:
                    blocked_once = True
                    terminal_close_started.set()
                    await asyncio.Event().wait()

            async def stream_token(self, token: str) -> None:
                return None

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Step",
            Step,
        )
        notifier = LocalBackgroundTaskNotifier(
            manager=manager,
            session_id="session-a",
        )
        notifier.start()

        async def runner(task_id: str) -> str:
            await manager.publish_activity(
                task_id,
                AgentStreamEvent(
                    kind="reasoning_delta",
                    source="researcher",
                    text="thinking",
                ),
            )
            return "finished"

        await manager.spawn(
            session_id="session-a",
            agent_name="researcher",
            description="research",
            agent_path=("researcher",),
            runner=runner,
        )
        await asyncio.wait_for(terminal_close_started.wait(), timeout=2)

        await notifier.aclose()

        reasoning = next(step for step in steps if step.type == "llm")
        parent = next(step for step in steps if step.type == "run")
        assert reasoning.update_attempts == 2
        assert reasoning.end is not None
        assert parent.update_attempts == 1
        assert parent.end is not None
        assert parent.output == "Stopped"
        assert notifier.activity_states == {}
        await manager.close()

    asyncio.run(exercise())


def test_chainlit_local_notifier_continues_after_send_failure(monkeypatch) -> None:
    """One transport failure must not disable later completion notices."""
    async def exercise() -> None:
        manager = make_manager()
        attempts = 0
        delivered = asyncio.Event()

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                self.content = content
                self.author = author

            async def send(self) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("temporary transport failure")
                delivered.set()

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message",
            Message,
        )
        notifier = LocalBackgroundTaskNotifier(
            manager=manager,
            session_id="session-a",
        )
        notifier.start()

        async def runner(task_id: str) -> str:
            return task_id

        first = await manager.spawn(
            session_id="session-a",
            agent_name="first",
            description="first",
            agent_path=("first",),
            runner=runner,
        )
        await manager.get("session-a", first.task_id, wait_seconds=1)
        second = await manager.spawn(
            session_id="session-a",
            agent_name="second",
            description="second",
            agent_path=("second",),
            runner=runner,
        )
        await manager.get("session-a", second.task_id, wait_seconds=1)
        await asyncio.wait_for(delivered.wait(), timeout=1)

        assert attempts == 2
        assert notifier.task is not None and not notifier.task.done()
        notifier.cancel()
        await manager.close()

    asyncio.run(exercise())
