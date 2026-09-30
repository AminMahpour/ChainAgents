"""Concurrency and teardown contracts for MCP tool discovery."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import chainagents.runtime.mcp_sessions as mcp_sessions


def _pool(*, stateful: bool = False) -> mcp_sessions.MCPSessionPool:
    extensions = SimpleNamespace(mcp_stateful=stateful, mcp_tool_name_prefix=False)
    return mcp_sessions.MCPSessionPool(lambda: extensions)


def test_distinct_servers_load_concurrently_and_keep_requested_order():
    pool = _pool()
    both_started = asyncio.Event()
    release = asyncio.Event()
    started: set[str] = set()
    calls: list[str] = []

    async def get_tools(*, server_name):
        calls.append(server_name)
        started.add(server_name)
        if len(started) == 2:
            both_started.set()
        await release.wait()
        return [SimpleNamespace(name=f"{server_name}_tool")]

    pool.client = SimpleNamespace(get_tools=get_tools)

    async def exercise():
        discovery = asyncio.create_task(pool.tools_with_status(("slow", "fast")))
        try:
            await asyncio.wait_for(both_started.wait(), timeout=1)
        finally:
            release.set()
            tools, failures = await discovery
        assert failures == ()
        assert [tool.name for tool in tools] == ["slow_tool", "fast_tool"]
        cached, failures = await pool.tools_with_status(("fast", "slow"))
        assert failures == ()
        assert [tool.name for tool in cached] == ["fast_tool", "slow_tool"]
        assert calls == ["slow", "fast"]

    asyncio.run(exercise())


def test_overlapping_failed_discovery_is_shared_warns_each_caller_and_retries():
    pool = _pool()
    started = asyncio.Event()
    release = asyncio.Event()
    attempts = 0
    fail = True

    async def get_tools(*, server_name):
        nonlocal attempts
        assert server_name == "broken"
        attempts += 1
        started.set()
        await release.wait()
        if fail:
            raise OSError("offline")
        return [SimpleNamespace(name="recovered")]

    pool.client = SimpleNamespace(get_tools=get_tools)

    async def invoke(warnings):
        token = mcp_sessions._MCP_DISCOVERY_WARNINGS.set(warnings)
        try:
            return await pool.tools_with_status(("broken",))
        finally:
            mcp_sessions._MCP_DISCOVERY_WARNINGS.reset(token)

    async def exercise():
        nonlocal fail
        first_warnings: set[str] = set()
        second_warnings: set[str] = set()
        first = asyncio.create_task(invoke(first_warnings))
        await started.wait()
        second = asyncio.create_task(invoke(second_warnings))
        await asyncio.sleep(0)
        release.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert first_result == second_result == ([], ("broken",))
        assert first_warnings == second_warnings == {"broken"}
        assert attempts == 1
        fail = False
        tools, failures = await pool.tools_with_status(("broken",))
        assert failures == ()
        assert [tool.name for tool in tools] == ["recovered"]
        assert attempts == 2

    asyncio.run(exercise())


def test_cancelling_one_waiter_keeps_shared_discovery_running():
    pool = _pool()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def get_tools(*, server_name):
        nonlocal calls
        assert server_name == "repo"
        calls += 1
        started.set()
        await release.wait()
        return [SimpleNamespace(name="repo_tool")]

    pool.client = SimpleNamespace(get_tools=get_tools)

    async def exercise():
        first = asyncio.create_task(pool.tools_with_status(("repo",)))
        await started.wait()
        second = asyncio.create_task(pool.tools_with_status(("repo",)))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        tools, failures = await second
        assert failures == ()
        assert [tool.name for tool in tools] == ["repo_tool"]
        assert calls == 1

    asyncio.run(exercise())


def test_same_server_in_different_scopes_loads_independently(monkeypatch):
    pool = _pool(stateful=True)
    both_started = asyncio.Event()
    release = asyncio.Event()
    sessions = 0
    loads = 0

    @asynccontextmanager
    async def session(server_name):
        nonlocal sessions
        assert server_name == "repo"
        sessions += 1
        yield SimpleNamespace(name=f"session-{sessions}")

    async def load(session, **kwargs):
        nonlocal loads
        loads += 1
        if loads == 2:
            both_started.set()
        await release.wait()
        return [SimpleNamespace(name=f"{session.name}_tool")]

    pool.client = SimpleNamespace(
        session=session, callbacks=None, tool_interceptors=[]
    )
    monkeypatch.setattr(mcp_sessions, "load_mcp_tools", load)

    async def exercise():
        first = asyncio.create_task(
            pool.tools_with_status(("repo",), mcp_session_id="first")
        )
        second = asyncio.create_task(
            pool.tools_with_status(("repo",), mcp_session_id="second")
        )
        try:
            await asyncio.wait_for(both_started.wait(), timeout=1)
        finally:
            release.set()
            first_result, second_result = await asyncio.gather(first, second)
        assert {
            first_result[0][0].name,
            second_result[0][0].name,
        } == {"session-1_tool", "session-2_tool"}
        assert first_result[1] == second_result[1] == ()
        assert sessions == loads == 2
        await pool.close_owner_entries(await pool.evict_all())

    asyncio.run(exercise())


@pytest.mark.parametrize("evict_all", [False, True])
def test_evict_during_stateful_discovery_cleans_new_owner_and_prevents_cache(
    monkeypatch, evict_all
):
    pool = _pool(stateful=True)
    loading = asyncio.Event()
    closed: list[str] = []
    sessions = 0

    @asynccontextmanager
    async def session(server_name):
        nonlocal sessions
        sessions += 1
        try:
            yield SimpleNamespace(name=server_name)
        finally:
            closed.append(server_name)

    async def load(session, **kwargs):
        if sessions == 1:
            loading.set()
            await asyncio.Event().wait()
        return [SimpleNamespace(name=f"{session.name}_tool")]

    pool.client = SimpleNamespace(
        session=session, callbacks=None, tool_interceptors=[]
    )
    monkeypatch.setattr(mcp_sessions, "load_mcp_tools", load)

    async def exercise():
        discovery = asyncio.create_task(
            pool.tools_with_status(("repo",), mcp_session_id="scope")
        )
        await loading.wait()
        if evict_all:
            owners = await asyncio.wait_for(pool.evict_all(), timeout=1)
        else:
            owners = await asyncio.wait_for(pool.evict_scope("scope"), timeout=1)
        await pool.close_owner_entries(owners)
        with pytest.raises(asyncio.CancelledError):
            await discovery
        assert closed == ["repo"]
        assert not pool._tools_cache
        assert not pool._sessions
        assert not pool._session_owners

        tools, failures = await pool.tools_with_status(
            ("repo",), mcp_session_id="scope"
        )
        assert failures == ()
        assert [tool.name for tool in tools] == ["repo_tool"]
        assert sessions == 2
        owners = await pool.evict_scope("scope")
        await pool.close_owner_entries(owners)
        assert closed == ["repo", "repo"]

    asyncio.run(exercise())


def test_evicted_discovery_cannot_cache_when_loader_suppresses_cancellation(
    monkeypatch,
):
    pool = _pool(stateful=True)
    loading = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    closed: list[str] = []

    @asynccontextmanager
    async def session(server_name):
        try:
            yield SimpleNamespace(name=server_name)
        finally:
            closed.append(server_name)

    async def load(session, **kwargs):
        loading.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return [SimpleNamespace(name=f"{session.name}_tool")]

    pool.client = SimpleNamespace(
        session=session, callbacks=None, tool_interceptors=[]
    )
    monkeypatch.setattr(mcp_sessions, "load_mcp_tools", load)

    async def exercise():
        discovery = asyncio.create_task(
            pool.tools_with_status(("repo",), mcp_session_id="scope")
        )
        await loading.wait()
        eviction = asyncio.create_task(pool.evict_scope("scope"))
        try:
            await asyncio.wait_for(cancelled.wait(), timeout=1)
        finally:
            release.set()
        await pool.close_owner_entries(await eviction)
        with pytest.raises(asyncio.CancelledError):
            await discovery
        assert closed == ["repo"]
        assert not pool._tools_cache
        assert not pool._sessions
        assert not pool._session_owners

    asyncio.run(exercise())


def test_evict_waits_for_session_cleanup_after_discovery_failure(monkeypatch):
    pool = _pool(stateful=True)
    closing = asyncio.Event()
    release_close = asyncio.Event()
    closed = asyncio.Event()

    @asynccontextmanager
    async def session(server_name):
        assert server_name == "repo"
        try:
            yield object()
        finally:
            closing.set()
            await release_close.wait()
            closed.set()

    async def load(*args, **kwargs):
        raise OSError("tool listing failed")

    pool.client = SimpleNamespace(
        session=session, callbacks=None, tool_interceptors=[]
    )
    monkeypatch.setattr(mcp_sessions, "load_mcp_tools", load)

    async def exercise():
        discovery = asyncio.create_task(
            pool.tools_with_status(("repo",), mcp_session_id="scope")
        )
        await closing.wait()
        eviction = asyncio.create_task(pool.evict_scope("scope"))
        try:
            # Eviction may return an owner to close, or wait for the in-flight
            # discovery to finish closing it. It must not silently lose it.
            try:
                early_owners = await asyncio.wait_for(
                    asyncio.shield(eviction), timeout=0.1
                )
            except TimeoutError:
                early_owners = None
            else:
                assert early_owners, "eviction returned while transport was open"
        finally:
            release_close.set()

        owners = await eviction
        await pool.close_owner_entries(owners)
        assert closed.is_set()
        with pytest.raises(asyncio.CancelledError):
            await discovery
        assert not pool._sessions
        assert not pool._session_owners

    asyncio.run(exercise())


def test_eviction_returns_owner_when_inflight_cleanup_needs_retry(monkeypatch):
    pool = _pool(stateful=True)
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    closed = asyncio.Event()
    close_attempts = 0
    original_close = mcp_sessions._MCPSessionOwner.aclose

    @asynccontextmanager
    async def session(server_name):
        try:
            yield object()
        finally:
            closed.set()

    async def load(*args, **kwargs):
        raise OSError("tool listing failed")

    async def flaky_close(owner):
        nonlocal close_attempts
        close_attempts += 1
        if close_attempts == 1:
            close_started.set()
            await release_close.wait()
            raise OSError("temporary close failure")
        await original_close(owner)

    pool.client = SimpleNamespace(
        session=session, callbacks=None, tool_interceptors=[]
    )
    monkeypatch.setattr(mcp_sessions, "load_mcp_tools", load)
    monkeypatch.setattr(mcp_sessions._MCPSessionOwner, "aclose", flaky_close)

    async def exercise():
        discovery = asyncio.create_task(
            pool.tools_with_status(("repo",), mcp_session_id="scope")
        )
        await close_started.wait()
        eviction = asyncio.create_task(pool.evict_scope("scope"))
        await asyncio.sleep(0)
        release_close.set()
        owners = await eviction
        try:
            assert len(owners) == 1
        finally:
            await pool.close_owner_entries(owners or list(pool._session_owners.items()))
        with pytest.raises(asyncio.CancelledError):
            await discovery
        assert close_attempts == 2
        assert closed.is_set()
        assert not pool._session_owners

    asyncio.run(exercise())


@pytest.mark.parametrize("evict_all", [False, True])
def test_new_discovery_waits_until_matching_eviction_finishes(monkeypatch, evict_all):
    pool = _pool(stateful=True)
    first_loading = asyncio.Event()
    first_cancelled = asyncio.Event()
    release_first = asyncio.Event()
    second_loading = asyncio.Event()
    loads = 0

    @asynccontextmanager
    async def session(server_name):
        yield SimpleNamespace(name=server_name)

    async def load(session, **kwargs):
        nonlocal loads
        loads += 1
        if loads == 1:
            first_loading.set()
            try:
                await release_first.wait()
            except asyncio.CancelledError:
                first_cancelled.set()
                await release_first.wait()
            return [SimpleNamespace(name="stale")]
        second_loading.set()
        return [SimpleNamespace(name="fresh")]

    pool.client = SimpleNamespace(
        session=session, callbacks=None, tool_interceptors=[]
    )
    monkeypatch.setattr(mcp_sessions, "load_mcp_tools", load)

    async def exercise():
        first = asyncio.create_task(
            pool.tools_with_status(("repo",), mcp_session_id="scope")
        )
        await first_loading.wait()
        eviction = asyncio.create_task(
            pool.evict_all() if evict_all else pool.evict_scope("scope")
        )
        await first_cancelled.wait()
        second = asyncio.create_task(
            pool.tools_with_status(
                ("repo",), mcp_session_id="other" if evict_all else "scope"
            )
        )
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    asyncio.shield(second_loading.wait()), timeout=0.1
                )
        finally:
            release_first.set()

        owners = await eviction
        await pool.close_owner_entries(owners)
        with pytest.raises(asyncio.CancelledError):
            await first
        tools, failures = await second
        assert failures == ()
        assert [tool.name for tool in tools] == ["fresh"]
        assert loads == 2
        await pool.close_owner_entries(await pool.evict_all())

    asyncio.run(exercise())


@pytest.mark.parametrize("close_all", [False, True])
def test_combined_eviction_blocks_discovery_through_owner_close(monkeypatch, close_all):
    pool = _pool(stateful=True)
    closing = asyncio.Event()
    release_close = asyncio.Event()
    second_opened = asyncio.Event()
    sessions = 0
    closed: list[int] = []

    @asynccontextmanager
    async def session(server_name):
        nonlocal sessions
        sessions += 1
        number = sessions
        if number == 2:
            second_opened.set()
        try:
            yield SimpleNamespace(number=number)
        finally:
            if number == 1:
                closing.set()
                await release_close.wait()
            closed.append(number)

    async def load(session, **kwargs):
        return [SimpleNamespace(name=f"tool_{session.number}")]

    pool.client = SimpleNamespace(
        session=session, callbacks=None, tool_interceptors=[]
    )
    monkeypatch.setattr(mcp_sessions, "load_mcp_tools", load)

    async def exercise():
        first_tools, failures = await pool.tools_with_status(
            ("repo",), mcp_session_id="scope"
        )
        assert failures == ()
        assert [tool.name for tool in first_tools] == ["tool_1"]

        try:
            close = asyncio.create_task(
                pool.evict_all_and_close()
                if close_all
                else pool.evict_scope_and_close("scope")
            )
        except BaseException:
            release_close.set()
            await pool.close_owner_entries(await pool.evict_all())
            raise
        await closing.wait()
        second = asyncio.create_task(
            pool.tools_with_status(
                ("repo",), mcp_session_id="other" if close_all else "scope"
            )
        )
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    asyncio.shield(second_opened.wait()), timeout=0.1
                )
        finally:
            release_close.set()

        await close
        assert closed == [1]
        second_tools, failures = await second
        assert failures == ()
        assert [tool.name for tool in second_tools] == ["tool_2"]
        await pool.evict_all_and_close()
        assert closed == [1, 2]

    asyncio.run(exercise())
