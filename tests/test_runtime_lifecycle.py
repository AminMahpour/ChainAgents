"""Exercise runtime resource ownership without live transports or models."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import anyio
import pytest

import deepagent_runtime as core
import chainagents.runtime.config as runtime_config
import chainagents.runtime.lifecycle as runtime_lifecycle
import chainagents.runtime.middleware as runtime_middleware
from chainagents.runtime.background_tasks import (
    BackgroundTaskManager,
    create_background_task_tools,
)
from chainagents.runtime.types import BackgroundSubagentConfig
from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from test_deepagent_runtime_rag import make_runtime_config, make_extensions_config


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = replace(make_runtime_config(tmp_path), rag=None, rag_requested=False)
    instance = core.AgentRuntime(config, project_root=tmp_path)
    instance._store = InMemoryStore()
    instance._checkpointer = MemorySaver()
    monkeypatch.setattr(instance, '_build_model', lambda *a, **kw: object())
    monkeypatch.setattr(runtime_middleware, 'create_deep_agent_with_configured_summarization', lambda *a, **kw: object())
    return instance


def test_stateful_context_closes_on_its_owner_task(runtime, monkeypatch):
    runtime.config = replace(runtime.config, extensions=make_extensions_config(mcp_stateful=True, agent_mcp_servers=('repo',)))
    events = []

    @asynccontextmanager
    async def session(server):
        async with anyio.create_task_group():
            events.append(('open', asyncio.current_task()))
            try:
                yield object()
            finally:
                events.append(('close', asyncio.current_task()))

    runtime._mcp_client = SimpleNamespace(session=session, callbacks=None, tool_interceptors=[])
    async def load(*a, **kw):
        return []
    monkeypatch.setattr(runtime_lifecycle, 'load_mcp_tools', load)

    async def exercise():
        await asyncio.create_task(runtime.get_agent('medium', thread_id='thread', mcp_session_id='session'))
        await runtime.close_mcp_session('session')
        assert len(events) == 2
        assert events[0][1] is events[1][1]
    asyncio.run(exercise())


@pytest.mark.parametrize('stateful', [False, True])
def test_conversation_close_evicts_graph_with_no_mcp_client(runtime, stateful):
    runtime.config = replace(runtime.config, extensions=replace(runtime.config.extensions, mcp_stateful=stateful))
    async def exercise():
        first = await runtime.get_agent('medium', thread_id='thread', mcp_session_id='session')
        other = await runtime.get_agent('medium', thread_id='other', mcp_session_id='other-session')
        await runtime.close_conversation(thread_id='thread', mcp_session_id='session')
        assert first not in runtime._agents.values()
        assert await runtime.get_agent('medium', thread_id='other', mcp_session_id='other-session') is other
        assert await runtime.get_agent('medium', thread_id='thread', mcp_session_id='session') is not first
        await runtime.close()
    asyncio.run(exercise())


def test_conversation_close_cancels_background_tasks_before_mcp(runtime, monkeypatch):
    """Conversation resources must remain alive until background jobs stop."""
    runtime.background_tasks = BackgroundTaskManager(
        BackgroundSubagentConfig(enabled=True)
    )

    async def exercise():
        queue = runtime.background_tasks.subscribe("thread")

        async def runner(task_id):
            await asyncio.Event().wait()
            return task_id

        spawned = await runtime.background_tasks.spawn(
            session_id="thread",
            agent_name="worker",
            description="work",
            agent_path=("worker",),
            runner=runner,
        )
        events = []

        async def close_mcp_session(session_id):
            events.append((session_id, not queue.empty()))

        monkeypatch.setattr(runtime, "close_mcp_session", close_mcp_session)
        await runtime.close_conversation(thread_id="thread", mcp_session_id="mcp")

        terminal = queue.get_nowait()
        assert terminal.task_id == spawned.task_id
        assert terminal.status == "cancelled"
        assert events == [("mcp", True)]
        assert await runtime.background_tasks.list("thread") == []
        await runtime.background_tasks.close()

    asyncio.run(exercise())


def test_conversation_close_rejects_spawns_until_resource_teardown_finishes(
    runtime,
    monkeypatch,
):
    """A closing conversation cannot launch jobs while MCP resources close."""
    runtime.background_tasks = BackgroundTaskManager(
        BackgroundSubagentConfig(enabled=True)
    )

    async def exercise():
        mcp_close_started = asyncio.Event()
        allow_mcp_close = asyncio.Event()

        async def close_mcp_session(session_id):
            assert session_id == "mcp"
            mcp_close_started.set()
            await allow_mcp_close.wait()

        async def runner(task_id):
            return task_id

        monkeypatch.setattr(runtime, "close_mcp_session", close_mcp_session)
        close_task = asyncio.create_task(
            runtime.close_conversation(thread_id="thread", mcp_session_id="mcp")
        )
        await asyncio.wait_for(mcp_close_started.wait(), timeout=1)

        with pytest.raises(RuntimeError, match="session is closing"):
            await runtime.background_tasks.spawn(
                session_id="thread",
                agent_name="worker",
                description="late work",
                agent_path=("worker",),
                runner=runner,
            )

        allow_mcp_close.set()
        await asyncio.wait_for(close_task, timeout=1)
        await runtime.background_tasks.close()

    asyncio.run(exercise())


def test_cancelled_conversation_close_finishes_resource_teardown(
    runtime,
    monkeypatch,
):
    """Caller cancellation must not reopen a partially closed conversation."""
    runtime.background_tasks = BackgroundTaskManager(
        BackgroundSubagentConfig(enabled=True)
    )

    async def exercise():
        mcp_close_started = asyncio.Event()
        allow_mcp_close = asyncio.Event()

        async def close_mcp_session(session_id):
            assert session_id == "mcp"
            mcp_close_started.set()
            await allow_mcp_close.wait()

        async def runner(task_id):
            return task_id

        monkeypatch.setattr(runtime, "close_mcp_session", close_mcp_session)
        close_task = asyncio.create_task(
            runtime.close_conversation(thread_id="thread", mcp_session_id="mcp")
        )
        await asyncio.wait_for(mcp_close_started.wait(), timeout=1)
        close_task.cancel()
        await asyncio.sleep(0)

        assert not close_task.done()
        with pytest.raises(RuntimeError, match="session is closing"):
            await runtime.background_tasks.spawn(
                session_id="thread",
                agent_name="worker",
                description="late work",
                agent_path=("worker",),
                runner=runner,
            )

        allow_mcp_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(close_task, timeout=1)

        reopened = await runtime.background_tasks.spawn(
            session_id="thread",
            agent_name="worker",
            description="new work",
            agent_path=("worker",),
            runner=runner,
        )
        assert (
            await runtime.background_tasks.get(
                "thread",
                reopened.task_id,
                wait_seconds=1,
            )
        ).status == "success"
        await runtime.background_tasks.close()

    asyncio.run(exercise())


def test_conversation_close_invalidates_existing_background_tools(runtime):
    """A foreground run from before close cannot spawn after teardown."""
    runtime.background_tasks = BackgroundTaskManager(
        BackgroundSubagentConfig(enabled=True)
    )

    async def exercise():
        generation = runtime.background_tasks.session_generation("thread")
        tools = create_background_task_tools(
            manager=runtime.background_tasks,
            subagents={"worker": object()},
            agent_path=(),
            recursion_limit=20,
            session_generation=generation,
        )
        spawn_tool = next(
            tool for tool in tools if tool.name == "spawn_background_task"
        )
        tool_runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "thread"}},
            stream_writer=lambda _: None,
            tool_call_id="stale-spawn",
            store=None,
        )

        await runtime.close_conversation(
            thread_id="thread",
            mcp_session_id="thread",
        )

        with pytest.raises(RuntimeError, match="session was closed"):
            await spawn_tool.coroutine("late work", "worker", tool_runtime)
        await runtime.background_tasks.close()

    asyncio.run(exercise())


def test_runtime_close_cancels_background_tasks(tmp_path):
    """Runtime shutdown must not leave local subagent asyncio tasks alive."""
    config = replace(
        make_runtime_config(tmp_path),
        rag=None,
        rag_requested=False,
        extensions=replace(
            make_runtime_config(tmp_path).extensions,
            background_subagents=BackgroundSubagentConfig(enabled=True),
        ),
    )
    instance = core.AgentRuntime(config, project_root=tmp_path)

    async def exercise():
        queue = instance.background_tasks.subscribe("thread")
        cleaned = []

        teardown_events = []

        async def close_persistence_resource():
            teardown_events.append(("persistence", not queue.empty()))

        instance._exit_stack.push_async_callback(close_persistence_resource)

        async def runner(task_id):
            await asyncio.Event().wait()
            return task_id

        async def cleanup(task_id):
            cleaned.append(task_id)

        await instance.background_tasks.spawn(
            session_id="thread",
            agent_name="worker",
            description="work",
            agent_path=("worker",),
            runner=runner,
            cleanup=cleanup,
        )
        await instance.close()

        assert queue.get_nowait().status == "cancelled"
        assert len(cleaned) == 1
        assert teardown_events == [("persistence", True)]

    asyncio.run(exercise())


@pytest.mark.parametrize('factory', ['get', 'create'])
@pytest.mark.parametrize('cancel', [False, True])
def test_factory_unwinds_resources_after_failed_or_cancelled_startup(runtime, monkeypatch, factory, cancel):
    closed = []
    entered = asyncio.Event()
    @asynccontextmanager
    async def resource():
        try:
            yield object()
        finally:
            closed.append(True)

    async def initialize(self):
        await self._exit_stack.enter_async_context(resource())
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        raise ValueError('startup failed')

    monkeypatch.setattr(runtime_lifecycle.AgentRuntime, '_instance', None)
    monkeypatch.setattr(runtime_lifecycle.AgentRuntime, '_initialize', initialize)
    monkeypatch.setattr(runtime_config.RuntimeConfig, 'from_env', lambda: runtime.config)
    async def exercise():
        task = asyncio.create_task(getattr(core.AgentRuntime, factory)())
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else ValueError):
            await task
        assert closed == [True]
        assert core.AgentRuntime.current() is None
    asyncio.run(exercise())


def test_stateless_mcp_tools_are_shared_across_ended_chats(runtime):
    runtime.config = replace(runtime.config, extensions=make_extensions_config(agent_mcp_servers=('repo',)))
    loads = []
    async def get_tools(**kwargs):
        loads.append(kwargs)
        return []
    runtime._mcp_client = SimpleNamespace(get_tools=get_tools)
    async def exercise():
        for index in range(10):
            thread, session = f'thread-{index}', f'session-{index}'
            await runtime.get_agent('medium', thread_id=thread, mcp_session_id=session)
            await runtime.close_conversation(thread_id=thread, mcp_session_id=session)
            assert not runtime._agents
            assert not runtime._mcp_sessions
        assert len(runtime._mcp_tools_cache) == 1
        assert len(loads) == 1
        await runtime.close()
        assert not runtime._mcp_tools_cache
    asyncio.run(exercise())


@pytest.mark.parametrize('cancel', [False, True])
def test_mcp_tool_loading_unwinds_new_transport_on_error(runtime, monkeypatch, cancel):
    runtime.config = replace(runtime.config, extensions=make_extensions_config(mcp_stateful=True, agent_mcp_servers=('repo',)))
    closed = []
    entered = asyncio.Event()
    @asynccontextmanager
    async def session(server):
        async with anyio.create_task_group():
            try:
                yield object()
            finally:
                closed.append(True)
    runtime._mcp_client = SimpleNamespace(session=session, callbacks=None, tool_interceptors=[])
    async def load(*a, **kw):
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        raise ValueError('tool loading failed')
    monkeypatch.setattr(runtime_lifecycle, 'load_mcp_tools', load)
    async def exercise():
        task = asyncio.create_task(runtime.get_agent('medium', thread_id='thread', mcp_session_id='session'))
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else ValueError):
            await task
        assert closed == [True]
        assert not runtime._mcp_sessions
        assert not runtime._mcp_tools_cache
        assert not runtime._agents
        async def succeeds(*a, **kw):
            return []
        monkeypatch.setattr(runtime_lifecycle, 'load_mcp_tools', succeeds)
        await runtime.get_agent('medium', thread_id='thread', mcp_session_id='session')
        await runtime.close()
        assert closed == [True, True]
    asyncio.run(exercise())


@pytest.mark.parametrize('cancel', [False, True])
def test_mcp_session_startup_unwinds_owner_and_allows_retry(runtime, monkeypatch, cancel):
    runtime.config = replace(runtime.config, extensions=make_extensions_config(mcp_stateful=True, agent_mcp_servers=('repo',)))
    closed = []
    entered = asyncio.Event()
    fail = True
    @asynccontextmanager
    async def session(server):
        async with anyio.create_task_group():
            try:
                entered.set()
                if fail:
                    if cancel:
                        await asyncio.Event().wait()
                    raise ValueError('transport startup failed')
                yield object()
            finally:
                closed.append(True)
    runtime._mcp_client = SimpleNamespace(session=session, callbacks=None, tool_interceptors=[])
    async def load(*a, **kw):
        return []
    monkeypatch.setattr(runtime_lifecycle, 'load_mcp_tools', load)
    async def exercise():
        nonlocal fail
        task = asyncio.create_task(runtime.get_agent('medium', thread_id='thread', mcp_session_id='session'))
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else ExceptionGroup):
            await task
        assert closed == [True]
        assert not runtime._mcp_sessions
        assert not runtime._mcp_session_owners
        fail = False
        await runtime.get_agent('medium', thread_id='thread', mcp_session_id='session')
        await runtime.close()
        assert closed == [True, True]
    asyncio.run(exercise())


def test_conversation_close_retains_other_session_on_same_thread(runtime):
    runtime.config = replace(runtime.config, extensions=replace(runtime.config.extensions, mcp_stateful=True))
    async def exercise():
        first = await runtime.get_agent('medium', thread_id='thread', mcp_session_id='session')
        other = await runtime.get_agent('medium', thread_id='thread', mcp_session_id='other-session')
        await runtime.close_conversation(thread_id='thread', mcp_session_id='session')
        assert first not in runtime._agents.values()
        assert await runtime.get_agent('medium', thread_id='thread', mcp_session_id='other-session') is other
        await runtime.close()
    asyncio.run(exercise())
