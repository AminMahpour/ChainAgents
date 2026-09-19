"""Exercise process-local background subagent task ownership and lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from deepagents import create_deep_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from chainagents.runtime.background_tasks import (
    BackgroundTaskManager,
    create_background_task_tools,
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
