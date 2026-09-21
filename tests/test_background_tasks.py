"""Exercise process-local background subagent task ownership and lifecycle."""

from __future__ import annotations

import asyncio
import gc
import weakref
from types import SimpleNamespace
from typing import Any

import pytest
from deepagents import create_deep_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from chainagents.runtime.background_tasks import (
    BackgroundTaskManager,
    create_background_task_tools,
    current_background_session_generation,
    scope_background_session_invocation,
    scope_background_task_invocation,
)
from chainagents.runtime.types import BackgroundSubagentConfig
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


def make_manager(**overrides: int | bool) -> BackgroundTaskManager:
    values: dict[str, int | bool] = {
        "enabled": True,
        "max_running_per_session": 2,
        "max_running_total": 3,
        "max_tasks_per_session": 5,
    }
    values.update(overrides)
    return BackgroundTaskManager(BackgroundSubagentConfig(**values))


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

        assert [item["description"] for item in result["results"]] == [
            "first",
            "second",
        ]
        assert [item["agent_name"] for item in result["results"]] == [
            "researcher",
            "researcher",
        ]
        assert [item["status"] for item in result["results"]] == [
            "success",
            "success",
        ]
        assert [item["result"] for item in result["results"]] == [
            "result:first",
            "result:second",
        ]
        thread_ids = [
            str(config["configurable"]["thread_id"])
            for config in child.configs
        ]
        assert len(set(thread_ids)) == 2
        assert all(thread_id.startswith("session-a:background:bg-") for thread_id in thread_ids)
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

        assert [item["status"] for item in result["results"]] == [
            "error",
            "success",
        ]
        assert result["results"][0]["error"] == "RuntimeError: child failed"
        assert result["results"][1]["result"] == "survived"
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

        assert [item["status"] for item in result["results"]] == [
            "cancelled",
            "cancelled",
        ]
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


def test_chainlit_local_notifier_sends_one_terminal_result(monkeypatch) -> None:
    """A completed task should notify the chat without starting another agent turn."""
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
                f"`success`.\n\nTask ID: `{spawned.task_id}`\n\nfinished result",
            )
        ]
        notifier.cancel()
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
