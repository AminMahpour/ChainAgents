"""Regression tests for Chainlit socket disconnects during agent turns."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

import main


async def _set_chat_session(monkeypatch, *, session, active, closed):
    values = {
        main.SESSION_ACTIVE_TURN_KEY: active,
        main.SESSION_SETTINGS_KEY: {"thread_id": "runtime-thread"},
    }

    async def close_conversation(*, thread_id, mcp_session_id):
        closed.append((thread_id, mcp_session_id))

    monkeypatch.setattr(main.cl, "context", SimpleNamespace(session=session))
    monkeypatch.setattr(
        main.cl,
        "user_session",
        SimpleNamespace(get=values.get, set=values.__setitem__),
    )
    runtime = SimpleNamespace(close_conversation=close_conversation)
    session._test_runtime = runtime
    monkeypatch.setattr(main.AgentRuntime, "current", lambda: runtime)
    manager = main.ConversationScopeLeaseManager(idle_seconds=0.01, max_idle=4)
    monkeypatch.setattr(main, "conversation_scopes", manager)
    await manager.lease(
        runtime=runtime,
        owner_id=main._conversation_owner_id(session),
        scope_id="runtime-thread",
    )
    return values


@pytest.mark.anyio
async def test_socket_disconnect_keeps_turn_and_resources_for_reconnect(monkeypatch):
    session = SimpleNamespace(socket_id="socket-1", to_clear=False)
    drafts = {"draft-1": ("runtime-thread", "follow up")}
    setattr(session, main.SESSION_INPUT_DRAFTS_KEY, drafts)
    closed: list[tuple[str | None, str | None]] = []
    finish_turn = asyncio.Event()
    active = asyncio.create_task(finish_turn.wait())
    await _set_chat_session(monkeypatch, session=session, active=active, closed=closed)
    monkeypatch.setattr(main.chainlit_config.project, "session_timeout", 0.01)

    try:
        await main.on_chat_end()
        assert not active.done()
        assert closed == []
        assert getattr(session, main.SESSION_INPUT_DRAFTS_KEY) is drafts

        session.socket_id = "socket-2"
        finish_turn.set()
        await asyncio.wait_for(active, timeout=1)
        await asyncio.wait_for(
            getattr(session, main.SESSION_DISCONNECT_CLEANUP_KEY), timeout=1
        )
        assert active.done() and not active.cancelled()
        assert closed == []
        assert getattr(session, main.SESSION_INPUT_DRAFTS_KEY) is drafts
    finally:
        if not active.done():
            active.cancel()
            with suppress(asyncio.CancelledError):
                await active


@pytest.mark.anyio
async def test_reconnect_replays_output_emitted_while_socket_was_disconnected(
    monkeypatch,
):
    delivered: list[tuple[str, object]] = []
    lost: list[tuple[str, object]] = []

    async def old_emit(event: str, data: object) -> None:
        lost.append((event, data))

    async def new_emit(event: str, data: object) -> None:
        delivered.append((event, data))

    session = SimpleNamespace(socket_id="socket-1", to_clear=False, emit=old_emit)
    closed: list[tuple[str | None, str | None]] = []
    active = asyncio.create_task(asyncio.Event().wait())
    await _set_chat_session(monkeypatch, session=session, active=active, closed=closed)
    monkeypatch.setattr(main.chainlit_config.project, "session_timeout", 0.01)

    def restore(sid, session_id, emit_fn, emit_call_fn, environ, user=None):
        assert sid == "socket-2"
        assert session_id == "session-1"
        session.socket_id = sid
        session.emit = emit_fn
        return True

    async def connection_successful(sid):
        assert sid == "socket-2"
        await session.emit("task_end", {})

    monkeypatch.setattr(main, "_chainlit_restore_existing_session", restore)
    monkeypatch.setattr(main, "_chainlit_connection_successful", connection_successful)
    monkeypatch.setattr(
        main.chainlit_socket.WebsocketSession,
        "get_by_id",
        lambda _session_id: session,
    )
    monkeypatch.setattr(
        main.chainlit_socket.WebsocketSession,
        "get",
        lambda _sid: session,
    )

    try:
        await main.on_chat_end()
        await session.emit("new_message", {"id": "reply", "output": "answer"})
        assert lost == []

        assert main._restore_session_with_output_replay(
            "socket-2", "session-1", new_emit, None, {}
        )
        await session.emit("action", {"forId": "reply", "label": "Download"})
        await main._connection_successful_with_output_replay("socket-2")
        await asyncio.wait_for(
            getattr(session, main.SESSION_DISCONNECT_CLEANUP_KEY), timeout=1
        )

        assert delivered == [
            ("new_message", {"id": "reply", "output": "answer"}),
            ("action", {"forId": "reply", "label": "Download"}),
            ("task_end", {}),
        ]
        assert session.emit is new_emit
        assert not hasattr(session, main.SESSION_DISCONNECTED_OUTPUT_KEY)
        assert closed == []
    finally:
        active.cancel()
        with suppress(asyncio.CancelledError):
            await active


@pytest.mark.anyio
async def test_expired_disconnected_session_stops_turn_and_closes_resources(
    monkeypatch,
):
    session = SimpleNamespace(socket_id="socket-1", to_clear=False)
    closed: list[tuple[str | None, str | None]] = []
    active = asyncio.create_task(asyncio.Event().wait())
    await _set_chat_session(monkeypatch, session=session, active=active, closed=closed)
    monkeypatch.setattr(main.chainlit_config.project, "session_timeout", 0)

    await main.on_chat_end()
    assert not active.done()
    await asyncio.wait_for(
        getattr(session, main.SESSION_DISCONNECT_CLEANUP_KEY), timeout=1
    )

    assert active.cancelled()
    assert closed == []
    await asyncio.sleep(0.03)
    assert closed == [("runtime-thread", "runtime-thread")]


@pytest.mark.anyio
async def test_expired_session_closes_notifiers_created_after_disconnect(monkeypatch):
    class _Notifier:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    class _LocalNotifier:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    session = SimpleNamespace(socket_id="socket-1", to_clear=False)
    closed: list[tuple[str | None, str | None]] = []
    finish_turn = asyncio.Event()
    values = {}
    notifier = _Notifier()
    local_notifier = _LocalNotifier()

    async def turn():
        await finish_turn.wait()
        values[main.SESSION_ASYNC_TASK_NOTIFIER_KEY] = notifier
        values[main.SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY] = local_notifier

    active = asyncio.create_task(turn())
    values = await _set_chat_session(
        monkeypatch, session=session, active=active, closed=closed
    )
    monkeypatch.setattr(main, "AsyncTaskNotifier", _Notifier)
    monkeypatch.setattr(main, "LocalBackgroundTaskNotifier", _LocalNotifier)
    monkeypatch.setattr(main.chainlit_config.project, "session_timeout", 0.1)

    await main.on_chat_end()
    finish_turn.set()
    await asyncio.wait_for(active, timeout=1)
    await asyncio.wait_for(
        getattr(session, main.SESSION_DISCONNECT_CLEANUP_KEY), timeout=1
    )

    assert notifier.cancelled
    assert local_notifier.closed
    await asyncio.sleep(0.03)
    assert closed == [("runtime-thread", "runtime-thread")]


@pytest.mark.anyio
async def test_expired_session_closes_notifiers_if_chainlit_deletes_session_state(
    monkeypatch,
):
    class _Notifier:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    class _LocalNotifier:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    session = SimpleNamespace(socket_id="socket-1", to_clear=False)
    closed: list[tuple[str | None, str | None]] = []
    values = {}
    notifier = _Notifier()
    local_notifier = _LocalNotifier()

    async def turn():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            values.clear()  # Chainlit's timeout can delete user_session here.
            raise

    active = asyncio.create_task(turn())
    values = await _set_chat_session(
        monkeypatch, session=session, active=active, closed=closed
    )
    monkeypatch.setattr(main, "AsyncTaskNotifier", _Notifier)
    monkeypatch.setattr(main, "LocalBackgroundTaskNotifier", _LocalNotifier)
    monkeypatch.setattr(main.chainlit_config.project, "session_timeout", 0)

    await main.on_chat_end()
    values[main.SESSION_ASYNC_TASK_NOTIFIER_KEY] = notifier
    values[main.SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY] = local_notifier
    await asyncio.wait_for(
        getattr(session, main.SESSION_DISCONNECT_CLEANUP_KEY), timeout=1
    )

    assert active.cancelled()
    assert notifier.cancelled
    assert local_notifier.closed
    await asyncio.sleep(0.03)
    assert closed == [("runtime-thread", "runtime-thread")]


@pytest.mark.anyio
async def test_explicit_chat_clear_keeps_busy_turn_until_it_finishes(monkeypatch):
    session = SimpleNamespace(socket_id="socket-1", to_clear=True)
    setattr(session, main.SESSION_INPUT_DRAFTS_KEY, {"draft-1": "pending"})
    closed: list[tuple[str | None, str | None]] = []
    finish_turn = asyncio.Event()
    active = asyncio.create_task(finish_turn.wait())
    await _set_chat_session(monkeypatch, session=session, active=active, closed=closed)

    try:
        await main.on_chat_end()

        assert not active.done()
        assert closed == []
        assert getattr(session, main.SESSION_INPUT_DRAFTS_KEY) == {}

        finish_turn.set()
        await asyncio.wait_for(active, timeout=1)
        await asyncio.sleep(0.03)
        assert closed == [("runtime-thread", "runtime-thread")]
    finally:
        if not active.done():
            active.cancel()
            with suppress(asyncio.CancelledError):
                await active


@pytest.mark.anyio
async def test_explicit_chat_clear_of_idle_thread_releases_scope(monkeypatch):
    session = SimpleNamespace(socket_id="socket-1", to_clear=True)
    closed: list[tuple[str | None, str | None]] = []
    await _set_chat_session(monkeypatch, session=session, active=None, closed=closed)

    await main.on_chat_end()

    await asyncio.sleep(0.03)
    assert closed == [("runtime-thread", "runtime-thread")]


@pytest.mark.anyio
async def test_explicit_clear_does_not_detach_only_a_paused_input_queue(monkeypatch):
    session = SimpleNamespace(socket_id="socket-1", to_clear=True)
    closed: list[tuple[str | None, str | None]] = []
    await _set_chat_session(monkeypatch, session=session, active=None, closed=closed)
    runtime = session._test_runtime

    async def conversation_busy(thread_id, *, include_paused_queue):
        assert thread_id == "runtime-thread"
        # Stop has paused a queued input, with no active turn to preserve.
        return include_paused_queue

    async def wait_conversation_idle(_thread_id, *, include_paused_queue):
        raise AssertionError("A paused-only queue must not start a detached waiter")

    runtime.conversation_busy = conversation_busy
    runtime.wait_conversation_idle = wait_conversation_idle

    await main.on_chat_end()

    await asyncio.sleep(0.03)
    assert closed == [("runtime-thread", "runtime-thread")]
    assert (id(runtime), "runtime-thread") not in main._detached_conversations


@pytest.mark.anyio
async def test_explicit_clear_detaches_background_observer_until_work_finishes(
    monkeypatch,
):
    class _LocalNotifier:
        def __init__(self):
            self.cancelled = False
            self.closed = False
            self.handed_off = False

        def cancel(self):
            self.cancelled = True

        async def aclose_for_handoff(self):
            self.handed_off = True

        async def aclose(self):
            self.closed = True

    session = SimpleNamespace(socket_id="socket-1", to_clear=True)
    closed: list[tuple[str | None, str | None]] = []
    values = await _set_chat_session(
        monkeypatch, session=session, active=None, closed=closed
    )
    background_finished = asyncio.Event()

    async def conversation_busy(thread_id, *, include_paused_queue):
        assert thread_id == "runtime-thread"
        assert not include_paused_queue
        return not background_finished.is_set()

    async def wait_conversation_idle(thread_id, *, include_paused_queue):
        assert thread_id == "runtime-thread"
        assert not include_paused_queue
        await background_finished.wait()

    session._test_runtime.conversation_busy = conversation_busy
    session._test_runtime.wait_conversation_idle = wait_conversation_idle
    notifier = _LocalNotifier()
    values[main.SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY] = notifier
    monkeypatch.setattr(main, "LocalBackgroundTaskNotifier", _LocalNotifier)

    await main.on_chat_end()

    assert notifier.handed_off
    assert not notifier.cancelled
    assert not notifier.closed
    assert closed == []

    background_finished.set()
    await asyncio.sleep(0.03)
    assert closed == [("runtime-thread", "runtime-thread")]


@pytest.mark.anyio
@pytest.mark.parametrize("finish_before_clear", [False, True])
async def test_stateless_clear_replays_background_completion_after_scope_closes(
    monkeypatch, finish_before_clear,
):
    from chainagents.interfaces.chainlit import async_tasks
    from chainagents.runtime.background_tasks import BackgroundTaskManager
    from chainagents.runtime.types import BackgroundSubagentConfig

    manager = BackgroundTaskManager(BackgroundSubagentConfig(enabled=True))
    finish_background = asyncio.Event()
    closed = asyncio.Event()

    async def runner(_task_id):
        await finish_background.wait()
        return "finished"

    spawned = await manager.spawn(
        session_id="runtime-thread",
        agent_name="researcher",
        description="research",
        agent_path=("researcher",),
        runner=runner,
    )

    async def conversation_busy(thread_id, *, include_paused_queue):
        assert not include_paused_queue
        snapshots = await manager.list(thread_id)
        return any(snapshot.status not in {"success", "error", "cancelled"}
                   for snapshot in snapshots)

    async def wait_conversation_idle(thread_id, *, include_paused_queue):
        assert not include_paused_queue
        await manager.wait_session(thread_id)

    async def close_conversation(*, thread_id, mcp_session_id):
        assert thread_id == mcp_session_id == "runtime-thread"
        await manager.close_session(thread_id)
        closed.set()

    runtime = SimpleNamespace(
        background_tasks=manager,
        conversation_busy=conversation_busy,
        wait_conversation_idle=wait_conversation_idle,
        close_conversation=close_conversation,
        config=SimpleNamespace(extensions=SimpleNamespace(mcp_stateful=False)),
    )
    session = SimpleNamespace(id="cleared-tab", socket_id="socket-1", to_clear=True)
    values = {main.SESSION_SETTINGS_KEY: {"thread_id": "runtime-thread"}}
    monkeypatch.setattr(main.cl, "context", SimpleNamespace(session=session))
    monkeypatch.setattr(
        main.cl, "user_session", SimpleNamespace(get=values.get, set=values.__setitem__)
    )
    monkeypatch.setattr(main.AgentRuntime, "current", lambda: runtime)
    scope_manager = main.ConversationScopeLeaseManager(idle_seconds=0.01, max_idle=4)
    monkeypatch.setattr(main, "conversation_scopes", scope_manager)
    await scope_manager.lease(
        runtime=runtime, owner_id=session.id, scope_id="runtime-thread", retain_idle=False
    )
    sent: list[str] = []

    class _Message:
        def __init__(self, *, content, author):
            assert author == "Background subagent"
            self.content = content

        async def send(self):
            sent.append(self.content)

    monkeypatch.setattr(async_tasks.cl, "Message", _Message)

    try:
        if finish_before_clear:
            finish_background.set()
            await manager.wait_session("runtime-thread")
        await main.on_chat_end()
        if not finish_before_clear:
            finish_background.set()
        await asyncio.wait_for(closed.wait(), timeout=1)
        assert await manager.list("runtime-thread") == []

        resumed = async_tasks.LocalBackgroundTaskNotifier(
            manager=manager, session_id="runtime-thread"
        )
        resumed.start()
        await resumed.reconcile_terminal_tasks()
        assert len(sent) == 1
        assert f"Task ID: `{spawned.task_id}`" in sent[0]
        await resumed.aclose()
    finally:
        await manager.close()


@pytest.mark.anyio
async def test_new_detached_work_gets_lease_while_prior_release_finishes(monkeypatch):
    """A prior detached waiter must not consume a newer tab's lease request."""
    closed = asyncio.Event()
    close_while_busy = False
    finish_new_turn = asyncio.Event()
    finish_first_release = asyncio.Event()
    first_release_started = asyncio.Event()

    async def close_conversation(*, thread_id, mcp_session_id):
        nonlocal close_while_busy
        assert thread_id == mcp_session_id == "runtime-thread"
        close_while_busy = not finish_new_turn.is_set()
        closed.set()

    async def conversation_busy(_thread_id, *, include_paused_queue):
        assert not include_paused_queue
        return False

    async def wait_conversation_idle(_thread_id, *, include_paused_queue):
        assert not include_paused_queue
        return None

    runtime = SimpleNamespace(
        close_conversation=close_conversation,
        conversation_busy=conversation_busy,
        wait_conversation_idle=wait_conversation_idle,
        config=SimpleNamespace(extensions=SimpleNamespace(mcp_stateful=False)),
    )
    manager = main.ConversationScopeLeaseManager(idle_seconds=0.01, max_idle=4)
    monkeypatch.setattr(main, "conversation_scopes", manager)
    release = manager.release

    async def delay_first_detached_release(owner_id, **kwargs):
        await release(owner_id, **kwargs)
        if owner_id.startswith("detached:") and not first_release_started.is_set():
            first_release_started.set()
            await finish_first_release.wait()

    monkeypatch.setattr(manager, "release", delay_first_detached_release)
    await manager.lease(
        runtime=runtime, owner_id="tab-a", scope_id="runtime-thread", retain_idle=False
    )
    await main._retain_conversation_until_idle(runtime, "runtime-thread", set())
    await manager.lease(
        runtime=runtime, owner_id="tab-b", scope_id="runtime-thread", retain_idle=False
    )
    await manager.release("tab-a")
    await asyncio.wait_for(first_release_started.wait(), timeout=1)

    new_turn = asyncio.create_task(finish_new_turn.wait())
    await main._retain_conversation_until_idle(runtime, "runtime-thread", {new_turn})
    await manager.release("tab-b")
    closed_before_turn_finished = closed.is_set()

    finish_new_turn.set()
    finish_first_release.set()
    await asyncio.wait_for(new_turn, timeout=1)
    await asyncio.wait_for(closed.wait(), timeout=1)
    assert not closed_before_turn_finished
    assert not close_while_busy
    await manager.aclose()


@pytest.mark.anyio
async def test_cleared_turn_reattached_by_new_tab_keeps_conversation_open(monkeypatch):
    session = SimpleNamespace(socket_id="socket-1", to_clear=True)
    closed: list[tuple[str | None, str | None]] = []
    finish_turn = asyncio.Event()
    active = asyncio.create_task(finish_turn.wait())
    await _set_chat_session(monkeypatch, session=session, active=active, closed=closed)

    try:
        await main.on_chat_end()
        await main.conversation_scopes.lease(
            runtime=session._test_runtime,
            owner_id="new-tab",
            scope_id="runtime-thread",
        )
        finish_turn.set()
        await asyncio.wait_for(active, timeout=1)
        await asyncio.sleep(0.03)

        assert closed == []
        await main.conversation_scopes.release("new-tab")
        await asyncio.sleep(0.03)
        assert closed == [("runtime-thread", "runtime-thread")]
    finally:
        if not active.done():
            active.cancel()
            with suppress(asyncio.CancelledError):
                await active


@pytest.mark.anyio
async def test_cleared_turn_does_not_resurrect_deleted_chainlit_user_session(
    monkeypatch,
):
    from chainlit import user_session as real_user_session
    from chainlit.user_session import user_sessions
    import importlib

    chainlit_user_session_module = importlib.import_module("chainlit.user_session")

    session = SimpleNamespace(
        id="cleared-session-test",
        thread_id="runtime-thread",
        socket_id="socket-1",
        to_clear=True,
        user_env={},
        chat_settings={},
        user=None,
        chat_profile=None,
        client_type="webapp",
    )
    finish_turn = asyncio.Event()

    async def turn():
        await finish_turn.wait()
        main._release_active_turn()
        real_user_session.get("late-turn-state")

    active = asyncio.create_task(turn())
    user_sessions[session.id] = {
        main.SESSION_ACTIVE_TURN_KEY: active,
        main.SESSION_SETTINGS_KEY: {"thread_id": "runtime-thread"},
    }
    monkeypatch.setattr(main.cl, "context", SimpleNamespace(session=session))
    monkeypatch.setattr(
        chainlit_user_session_module, "context", SimpleNamespace(session=session)
    )
    monkeypatch.setattr(main.cl, "user_session", real_user_session)
    closed = []

    async def close_conversation(*, thread_id, mcp_session_id):
        closed.append((thread_id, mcp_session_id))

    runtime = SimpleNamespace(close_conversation=close_conversation)
    monkeypatch.setattr(main.AgentRuntime, "current", lambda: runtime)
    manager = main.ConversationScopeLeaseManager(idle_seconds=0.01, max_idle=4)
    monkeypatch.setattr(main, "conversation_scopes", manager)
    await manager.lease(
        runtime=runtime, owner_id=session.id, scope_id="runtime-thread"
    )

    try:
        await main.on_chat_end()
        user_sessions.pop(session.id)
        finish_turn.set()
        await asyncio.wait_for(active, timeout=1)
        await asyncio.sleep(0.03)

        assert session.id not in user_sessions
        assert closed == [("runtime-thread", "runtime-thread")]
    finally:
        user_sessions.pop(session.id, None)
        if not active.done():
            active.cancel()
            with suppress(asyncio.CancelledError):
                await active


@pytest.mark.anyio
async def test_reconnect_during_timeout_cleanup_keeps_live_tab_lease(monkeypatch):
    """A resumed socket must retain its scope if cleanup was awaiting a turn."""
    session = SimpleNamespace(socket_id="socket-1", to_clear=False)
    closed: list[tuple[str | None, str | None]] = []
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def active_turn():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await allow_cleanup.wait()
            raise

    active = asyncio.create_task(active_turn())
    await _set_chat_session(monkeypatch, session=session, active=active, closed=closed)
    monkeypatch.setattr(main.chainlit_config.project, "session_timeout", 0)

    await main.on_chat_end()
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    session.socket_id = "socket-2"
    allow_cleanup.set()
    await asyncio.wait_for(
        getattr(session, main.SESSION_DISCONNECT_CLEANUP_KEY), timeout=1
    )
    await asyncio.sleep(0.03)

    assert closed == []
    await main.conversation_scopes.aclose()


@pytest.mark.anyio
async def test_reconnect_before_timeout_task_cancel_keeps_turn_running(monkeypatch):
    """The timeout callback rechecks the socket before canceling a live turn."""
    session = SimpleNamespace(socket_id="socket-1", to_clear=False)
    closed: list[tuple[str | None, str | None]] = []
    active = asyncio.create_task(asyncio.Event().wait())
    values = await _set_chat_session(
        monkeypatch, session=session, active=active, closed=closed
    )
    monkeypatch.setattr(main.chainlit_config.project, "session_timeout", 0)

    def get(key):
        if key == main.SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY:
            session.socket_id = "socket-2"
        return values.get(key)

    monkeypatch.setattr(main.cl, "user_session", SimpleNamespace(get=get))
    try:
        await main.on_chat_end()
        await asyncio.wait_for(
            getattr(session, main.SESSION_DISCONNECT_CLEANUP_KEY), timeout=1
        )
        assert not active.done()
        assert closed == []
    finally:
        active.cancel()
        with suppress(asyncio.CancelledError):
            await active
        await main.conversation_scopes.aclose()


@pytest.mark.anyio
async def test_reconnect_during_notifier_close_restores_observers(monkeypatch):
    """A reconnected tab gets fresh observers after its old subscriber closes."""
    local_close_started = asyncio.Event()
    allow_local_close = asyncio.Event()
    local_restored = asyncio.Event()
    async_restored = asyncio.Event()

    class _AsyncNotifier:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    class _LocalNotifier:
        def __init__(self, *, blocking=False):
            self.blocking = blocking
            self.closed = False
            self.session_id = "runtime-thread"

        async def aclose(self):
            if self.blocking:
                local_close_started.set()
                await allow_local_close.wait()
            self.closed = True

    session = SimpleNamespace(socket_id="socket-1", to_clear=False)
    closed: list[tuple[str | None, str | None]] = []
    values = await _set_chat_session(
        monkeypatch, session=session, active=None, closed=closed
    )
    session._test_runtime.config = SimpleNamespace(
        model_name="test",
        model_choices=("test",),
        extensions=SimpleNamespace(
            mcp_stateful=True,
            background_subagents=SimpleNamespace(enabled=True),
            async_subagents=("configured",),
            chainlit_reasoning_steps_enabled=True,
            chainlit_tool_steps_enabled=True,
        ),
    )
    old_async = _AsyncNotifier()
    old_local = _LocalNotifier(blocking=True)
    values[main.SESSION_ASYNC_TASK_NOTIFIER_KEY] = old_async
    values[main.SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY] = old_local
    monkeypatch.setattr(main, "AsyncTaskNotifier", _AsyncNotifier)
    monkeypatch.setattr(main, "LocalBackgroundTaskNotifier", _LocalNotifier)
    monkeypatch.setattr(main.chainlit_config.project, "session_timeout", 0)
    monkeypatch.setattr(
        main,
        "coerce_settings",
        lambda *_args, **_kwargs: main.AppSettings(
            model_name="test", reasoning_level="medium", thread_id="runtime-thread"
        ),
    )

    new_local = _LocalNotifier()
    new_async = _AsyncNotifier()

    async def start_local_background_notifier(**_kwargs):
        values[main.SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY] = new_local
        local_restored.set()

    def start_resume_warmup(**_kwargs):
        values[main.SESSION_ASYNC_TASK_NOTIFIER_KEY] = new_async
        async_restored.set()

    monkeypatch.setattr(main, "start_local_background_notifier", start_local_background_notifier)
    monkeypatch.setattr(main, "_start_resume_warmup", start_resume_warmup)

    await main.on_chat_end()
    cleanup = getattr(session, main.SESSION_DISCONNECT_CLEANUP_KEY)
    await asyncio.wait_for(local_close_started.wait(), timeout=1)
    session.socket_id = "socket-2"
    allow_local_close.set()
    await asyncio.wait_for(cleanup, timeout=1)
    await asyncio.wait_for(local_restored.wait(), timeout=1)
    await asyncio.wait_for(async_restored.wait(), timeout=1)

    assert old_async.cancelled
    assert old_local.closed
    assert values[main.SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY] is new_local
    assert values[main.SESSION_ASYNC_TASK_NOTIFIER_KEY] is new_async
    assert closed == []
    await main.conversation_scopes.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("cleared", [False, True])
async def test_reconnect_after_timeout_cleanup_restores_only_live_tab_lease(
    monkeypatch, cleared
):
    """The reconnect callback handles a socket restored after cleanup returned."""
    session = SimpleNamespace(socket_id="socket-1", to_clear=False)
    closed: list[tuple[str | None, str | None]] = []
    await _set_chat_session(monkeypatch, session=session, active=None, closed=closed)
    session._test_runtime.config = SimpleNamespace(
        model_name="test",
        model_choices=("test",),
        extensions=SimpleNamespace(
            mcp_stateful=True,
            background_subagents=SimpleNamespace(enabled=False),
            async_subagents=(),
            chainlit_reasoning_steps_enabled=True,
            chainlit_tool_steps_enabled=True,
        ),
    )
    monkeypatch.setattr(main.chainlit_config.project, "session_timeout", 0)
    monkeypatch.setattr(
        main,
        "coerce_settings",
        lambda *_args, **_kwargs: main.AppSettings(
            model_name="test", reasoning_level="medium", thread_id="runtime-thread"
        ),
    )

    async def connection_successful(sid):
        assert sid == "socket-2"

    monkeypatch.setattr(main, "_chainlit_connection_successful", connection_successful)
    monkeypatch.setattr(main.chainlit_socket.WebsocketSession, "get", lambda _sid: session)

    await main.on_chat_end()
    await asyncio.wait_for(
        getattr(session, main.SESSION_DISCONNECT_CLEANUP_KEY), timeout=1
    )
    owner_id = main._conversation_owner_id(session)
    assert owner_id not in main.conversation_scopes._owner_scopes

    session.socket_id = "socket-2"
    session.to_clear = cleared
    await main._connection_successful_with_output_replay("socket-2")
    if not cleared:
        await asyncio.wait_for(
            getattr(session, main.SESSION_RECONNECT_RECOVERY_TASK_KEY), timeout=1
        )
    assert (owner_id in main.conversation_scopes._owner_scopes) is not cleared
    assert closed == []
    await main.conversation_scopes.aclose()


@pytest.mark.anyio
async def test_work_added_during_final_busy_check_keeps_detached_lease(monkeypatch):
    """A clear racing the waiter's last busy query must not lose its turn."""
    closed = asyncio.Event()
    close_while_busy = False
    finish_new_turn = asyncio.Event()
    busy_query_started = asyncio.Event()
    answer_busy_query = asyncio.Event()

    async def close_conversation(*, thread_id, mcp_session_id):
        nonlocal close_while_busy
        close_while_busy = not finish_new_turn.is_set()
        closed.set()

    busy_queries = 0

    async def conversation_busy(_thread_id, *, include_paused_queue):
        nonlocal busy_queries
        busy_queries += 1
        # The second query is the final check after retaining missed notices.
        if busy_queries == 2:
            busy_query_started.set()
            await answer_busy_query.wait()
        return False

    async def wait_conversation_idle(_thread_id, *, include_paused_queue):
        return None

    runtime = SimpleNamespace(
        close_conversation=close_conversation,
        conversation_busy=conversation_busy,
        wait_conversation_idle=wait_conversation_idle,
        config=SimpleNamespace(extensions=SimpleNamespace(mcp_stateful=False)),
    )
    manager = main.ConversationScopeLeaseManager(idle_seconds=0.01, max_idle=4)
    monkeypatch.setattr(main, "conversation_scopes", manager)
    await manager.lease(
        runtime=runtime, owner_id="tab-a", scope_id="runtime-thread", retain_idle=False
    )
    await main._retain_conversation_until_idle(runtime, "runtime-thread", set())
    await asyncio.wait_for(busy_query_started.wait(), timeout=1)

    new_turn = asyncio.create_task(finish_new_turn.wait())
    await main._retain_conversation_until_idle(runtime, "runtime-thread", {new_turn})
    await manager.release("tab-a")
    answer_busy_query.set()
    await asyncio.sleep(0.05)
    closed_before_turn_finished = closed.is_set()

    finish_new_turn.set()
    await asyncio.wait_for(new_turn, timeout=1)
    await asyncio.wait_for(closed.wait(), timeout=1)
    assert not closed_before_turn_finished
    assert not close_while_busy
    await manager.aclose()


@pytest.mark.anyio
async def test_cleared_session_files_recreated_by_preserved_turn_are_removed(
    monkeypatch, tmp_path
):
    """A preserved turn's late file element must not leak a deleted session dir."""
    files_dir = tmp_path / "cleared-session-files"
    cleared = SimpleNamespace(id="cleared-files-session", files_dir=files_dir)
    finish_turn = asyncio.Event()

    async def turn():
        await finish_turn.wait()
        # Chainlit's persist_file re-creates the deleted session's directory.
        files_dir.mkdir()
        (files_dir / "report.pdf").write_bytes(b"pdf")

    async def close_conversation(*, thread_id, mcp_session_id):
        return None

    runtime = SimpleNamespace(
        close_conversation=close_conversation,
        config=SimpleNamespace(extensions=SimpleNamespace(mcp_stateful=False)),
    )
    manager = main.ConversationScopeLeaseManager(idle_seconds=0.01, max_idle=4)
    monkeypatch.setattr(main, "conversation_scopes", manager)
    monkeypatch.setattr(
        main.chainlit_socket.WebsocketSession, "get_by_id", lambda _id: None
    )
    active = asyncio.create_task(turn())

    await main._retain_conversation_until_idle(
        runtime, "runtime-thread", {active}, cleared_session=cleared
    )
    entry = main._detached_conversations[(id(runtime), "runtime-thread")]
    finish_turn.set()
    await asyncio.wait_for(entry.task, timeout=1)

    assert not files_dir.exists()
    await manager.aclose()
