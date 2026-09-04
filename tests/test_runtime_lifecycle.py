"""Exercise runtime resource ownership without live transports or models."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import anyio
import pytest

import deepagent_runtime as core
from test_deepagent_runtime_rag import make_runtime_config, make_extensions_config


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = replace(make_runtime_config(tmp_path), rag=None, rag_requested=False)
    instance = core.AgentRuntime(config, project_root=tmp_path)
    instance._store = core.InMemoryStore()
    instance._checkpointer = core.MemorySaver()
    monkeypatch.setattr(instance, '_build_model', lambda *a, **kw: object())
    monkeypatch.setattr(core, 'create_deep_agent_with_configured_summarization', lambda *a, **kw: object())
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
    monkeypatch.setattr(core, 'load_mcp_tools', load)

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

    monkeypatch.setattr(core.AgentRuntime, '_instance', None)
    monkeypatch.setattr(core.AgentRuntime, '_initialize', initialize)
    monkeypatch.setattr(core.RuntimeConfig, 'from_env', lambda: runtime.config)
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
    monkeypatch.setattr(core, 'load_mcp_tools', load)
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
        monkeypatch.setattr(core, 'load_mcp_tools', succeeds)
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
    monkeypatch.setattr(core, 'load_mcp_tools', load)
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
