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
async def test_explicit_chat_clear_stops_turn_immediately(monkeypatch):
    session = SimpleNamespace(socket_id="socket-1", to_clear=True)
    setattr(session, main.SESSION_INPUT_DRAFTS_KEY, {"draft-1": "pending"})
    closed: list[tuple[str | None, str | None]] = []
    active = asyncio.create_task(asyncio.Event().wait())
    await _set_chat_session(monkeypatch, session=session, active=active, closed=closed)

    await main.on_chat_end()

    assert active.cancelled()
    assert closed == []
    await asyncio.sleep(0.03)
    assert closed == [("runtime-thread", "runtime-thread")]
    assert getattr(session, main.SESSION_INPUT_DRAFTS_KEY) == {}


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
