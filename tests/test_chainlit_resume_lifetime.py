"""Conversation-scoped Chainlit MCP leases and nonblocking resume restoration."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace

import pytest

import main
from chainagents.interfaces.chainlit import settings as chainlit_settings


def _user_session(monkeypatch, values: dict[str, object]) -> None:
    monkeypatch.setattr(
        main.cl,
        "user_session",
        SimpleNamespace(get=values.get, set=values.__setitem__),
    )


def test_mcp_scope_follows_conversation_and_custom_thread(monkeypatch) -> None:
    """A websocket ID must not become the stateful MCP resource key."""
    values: dict[str, object] = {}
    _user_session(monkeypatch, values)
    monkeypatch.setattr(
        main.cl,
        "context",
        SimpleNamespace(session=SimpleNamespace(id="socket-session", thread_id="saved-chat")),
    )

    assert chainlit_settings.store_mcp_session_id() == "saved-chat"
    assert chainlit_settings.current_mcp_session_id() == "saved-chat"

    values[main.SESSION_SETTINGS_KEY] = {"thread_id": "custom-langgraph-thread"}
    assert chainlit_settings.current_mcp_session_id() == "custom-langgraph-thread"


@pytest.mark.anyio
async def test_resume_without_async_subagents_never_builds_agent(monkeypatch) -> None:
    """A saved chat replays local completions without building an async agent."""
    settings = main.AppSettings(model_name="test", reasoning_level="medium", thread_id="saved-chat")
    reconciled: list[str] = []

    class _LocalNotifier:
        async def reconcile_terminal_tasks(self):
            reconciled.append("saved-chat")

    async def unexpected_get_agent(*_args, **_kwargs):
        raise AssertionError("resume should not build an agent without async subagents")

    runtime = SimpleNamespace(
        get_agent=unexpected_get_agent,
        config=SimpleNamespace(
            model_name="test",
            model_choices=("test",),
            extensions=SimpleNamespace(
                async_subagents=(),
                background_subagents=SimpleNamespace(enabled=True),
                chainlit_reasoning_steps_enabled=True,
                chainlit_tool_steps_enabled=True,
                chainlit_model_mode_enabled=False,
                chainlit_reasoning_mode_enabled=False,
                chainlit_response_actions=(),
            ),
        ),
    )
    values: dict[str, object] = {}
    _user_session(monkeypatch, values)
    monkeypatch.setattr(
        main.cl,
        "context",
        SimpleNamespace(session=SimpleNamespace(id="websocket-2", thread_id="fresh-websocket-chat")),
    )

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "get_runtime_or_notify", lambda: asyncio.sleep(0, result=runtime))
    monkeypatch.setattr(main, "publish_native_commands", noop)
    async def start_local_background_notifier(**_kwargs):
        values[main.SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY] = _LocalNotifier()

    monkeypatch.setattr(main, "start_local_background_notifier", start_local_background_notifier)
    monkeypatch.setattr(main, "LocalBackgroundTaskNotifier", _LocalNotifier)
    monkeypatch.setattr(main, "publish_modes", noop)
    monkeypatch.setattr(main, "_restore_saved_response_actions", noop)
    def coerce_saved_settings(raw_settings, **_kwargs):
        assert raw_settings["thread_id"] == "saved-chat"
        return settings

    monkeypatch.setattr(main, "coerce_settings", coerce_saved_settings)
    monkeypatch.setattr(main, "settings_reasoning_level_is_explicit", lambda *_args: False)
    monkeypatch.setattr(main, "get_run_task_list", lambda **_kwargs: asyncio.sleep(0, result=SimpleNamespace(show_ready=noop)))
    monkeypatch.setattr(main, "build_chat_settings", lambda *_args, **_kwargs: SimpleNamespace(send=noop))

    await main.on_chat_resume({"id": "saved-chat", "metadata": {}})

    assert values[main.SESSION_MCP_SESSION_ID_KEY] == "saved-chat"
    assert reconciled == []

    session = main.cl.context.session

    async def connection_successful(_sid):
        reconciled.append("resume_thread")

    monkeypatch.setattr(main, "_chainlit_connection_successful", connection_successful)
    monkeypatch.setattr(
        main.chainlit_socket.WebsocketSession, "get", lambda _sid: session
    )
    await main._connection_successful_with_output_replay("socket-2")
    assert reconciled == ["resume_thread", "saved-chat"]


@pytest.mark.anyio
async def test_resume_warmup_runs_after_callback_and_is_cancelled_on_chat_end(
    monkeypatch,
) -> None:
    """Async notifier discovery must not hold up resume or outlive its tab."""
    settings = main.AppSettings(model_name="test", reasoning_level="medium", thread_id="saved-chat")
    started = asyncio.Event()

    async def get_agent(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    async def close_conversation(*, thread_id, mcp_session_id):
        assert thread_id == mcp_session_id == "saved-chat"

    runtime = SimpleNamespace(
        get_agent=get_agent,
        close_conversation=close_conversation,
        config=SimpleNamespace(
            model_name="test",
            model_choices=("test",),
            extensions=SimpleNamespace(
                async_subagents=("configured",),
                background_subagents=SimpleNamespace(enabled=False),
                chainlit_reasoning_steps_enabled=True,
                chainlit_tool_steps_enabled=True,
                chainlit_model_mode_enabled=False,
                chainlit_reasoning_mode_enabled=False,
                chainlit_response_actions=(),
            ),
        ),
    )
    session = SimpleNamespace(id="websocket-2", thread_id="saved-chat", to_clear=True)
    values: dict[str, object] = {}
    _user_session(monkeypatch, values)
    monkeypatch.setattr(main.cl, "context", SimpleNamespace(session=session))
    manager = main.ConversationScopeLeaseManager(idle_seconds=600, max_idle=4)
    monkeypatch.setattr(main, "conversation_scopes", manager)

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "get_runtime_or_notify", lambda: asyncio.sleep(0, result=runtime))
    monkeypatch.setattr(main, "publish_native_commands", noop)
    monkeypatch.setattr(main, "start_local_background_notifier", noop)
    monkeypatch.setattr(main, "publish_modes", noop)
    monkeypatch.setattr(main, "_restore_saved_response_actions", noop)
    monkeypatch.setattr(main, "coerce_settings", lambda *_args, **_kwargs: settings)
    monkeypatch.setattr(main, "settings_reasoning_level_is_explicit", lambda *_args: False)
    monkeypatch.setattr(main, "get_run_task_list", lambda **_kwargs: asyncio.sleep(0, result=SimpleNamespace(show_ready=noop)))
    monkeypatch.setattr(main, "build_chat_settings", lambda *_args, **_kwargs: SimpleNamespace(send=noop))

    await asyncio.wait_for(main.on_chat_resume({"id": "saved-chat", "metadata": {}}), timeout=0.1)
    await asyncio.wait_for(started.wait(), timeout=1)
    warmup = getattr(session, main.SESSION_RESUME_WARMUP_KEY)
    assert not warmup.done()

    await main.on_chat_end()
    assert warmup.cancelled()
    await manager.aclose()


@pytest.mark.anyio
async def test_cancelled_resume_warmup_does_not_install_stale_notifier(
    monkeypatch,
) -> None:
    """A cancellation-swallowing agent build cannot recreate a closed notifier."""
    started = asyncio.Event()
    notifier_calls: list[object] = []

    async def get_agent(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return object()

    runtime = SimpleNamespace(get_agent=get_agent, config=SimpleNamespace())
    settings = main.AppSettings(model_name="test", reasoning_level="medium", thread_id="chat")
    session = SimpleNamespace(id="tab", thread_id="chat")
    monkeypatch.setattr(main.cl, "context", SimpleNamespace(session=session))
    monkeypatch.setattr(main, "async_subagent_url_override", lambda: None)
    monkeypatch.setattr(main, "settings_reasoning_level_is_explicit", lambda *_args: False)
    monkeypatch.setattr(
        main,
        "get_async_task_notifier",
        lambda **kwargs: notifier_calls.append(kwargs) or None,
    )

    main._start_resume_warmup(runtime=runtime, settings=settings, mcp_session_id="chat")
    await asyncio.wait_for(started.wait(), timeout=1)
    warmup = getattr(session, main.SESSION_RESUME_WARMUP_KEY)
    main._cancel_resume_warmup(session)
    await warmup

    assert notifier_calls == []


@pytest.mark.anyio
async def test_reattach_waits_for_idle_scope_close_before_new_lease() -> None:
    """A scope being evicted cannot close a newly reattached conversation."""
    manager = main.ConversationScopeLeaseManager(idle_seconds=600, max_idle=0)
    closing = asyncio.Event()
    allow_close = asyncio.Event()
    closed: list[str] = []

    async def close_conversation(*, thread_id, mcp_session_id):
        closing.set()
        await allow_close.wait()
        closed.append(thread_id)

    runtime = SimpleNamespace(close_conversation=close_conversation)
    await manager.lease(runtime=runtime, owner_id="old-tab", scope_id="saved-chat")
    await manager.release("old-tab")
    await asyncio.wait_for(closing.wait(), timeout=1)
    reattach = asyncio.create_task(
        manager.lease(runtime=runtime, owner_id="new-tab", scope_id="saved-chat")
    )
    await asyncio.sleep(0)
    assert not reattach.done()

    allow_close.set()
    await asyncio.wait_for(reattach, timeout=1)
    assert closed == ["saved-chat"]
    await manager.aclose()


@pytest.mark.anyio
async def test_idle_scope_reused_by_another_tab_and_evicted_lru() -> None:
    """Only owner-free scopes are closed, and the oldest idle scope goes first."""
    manager_type = getattr(main, "ConversationScopeLeaseManager", None)
    assert manager_type is not None
    manager = manager_type(idle_seconds=60, max_idle=2)
    closed: list[str] = []

    async def close_conversation(*, thread_id, mcp_session_id):
        assert thread_id == mcp_session_id
        closed.append(thread_id)

    runtime = SimpleNamespace(close_conversation=close_conversation)
    await manager.lease(runtime=runtime, owner_id="tab-a", scope_id="chat-1")
    await manager.lease(runtime=runtime, owner_id="tab-b", scope_id="chat-1")
    await manager.release("tab-a")
    await manager.lease(runtime=runtime, owner_id="tab-c", scope_id="chat-2")
    await manager.release("tab-c")
    await manager.lease(runtime=runtime, owner_id="tab-d", scope_id="chat-3")
    await manager.release("tab-d")
    await asyncio.sleep(0)
    assert closed == []  # chat-1 is still leased by tab-b.

    await manager.release("tab-b")
    await manager.lease(runtime=runtime, owner_id="tab-e", scope_id="chat-4")
    await manager.release("tab-e")
    await asyncio.sleep(0)
    assert closed == ["chat-2", "chat-3"]

    await manager.lease(runtime=runtime, owner_id="tab-f", scope_id="chat-1")
    assert closed == ["chat-2", "chat-3"]
    await manager.release("tab-f")
    await manager.aclose()


@pytest.mark.anyio
async def test_idle_scope_expiry_is_cancelled_on_reattach() -> None:
    """The prior idle timer may not close a newly leased conversation."""
    manager_type = getattr(main, "ConversationScopeLeaseManager", None)
    assert manager_type is not None
    manager = manager_type(idle_seconds=0.03, max_idle=4)
    closed: list[str] = []

    async def close_conversation(*, thread_id, mcp_session_id):
        closed.append(thread_id)

    runtime = SimpleNamespace(close_conversation=close_conversation)
    await manager.lease(runtime=runtime, owner_id="first", scope_id="saved-chat")
    await manager.release("first")
    await manager.lease(runtime=runtime, owner_id="second", scope_id="saved-chat")
    await asyncio.sleep(0.05)
    assert closed == []

    await manager.release("second")
    await asyncio.sleep(0.05)
    assert closed == ["saved-chat"]
    await manager.aclose()


@pytest.mark.anyio
async def test_stateless_scope_closes_when_last_tab_leaves() -> None:
    """A stateless runtime has no MCP transport to keep during idle time."""
    manager = main.ConversationScopeLeaseManager(idle_seconds=600, max_idle=4)
    closed: list[str] = []

    async def close_conversation(*, thread_id, mcp_session_id):
        closed.append(thread_id)

    runtime = SimpleNamespace(close_conversation=close_conversation)
    await manager.lease(
        runtime=runtime,
        owner_id="tab",
        scope_id="stateless-chat",
        retain_idle=False,
    )
    await manager.release("tab")
    await asyncio.sleep(0)
    assert closed == ["stateless-chat"]
    await manager.aclose()


@pytest.mark.anyio
async def test_tab_upgrades_turn_only_scope_to_idle_retention() -> None:
    """An early turn lease may not force immediate close after a tab attaches."""
    manager = main.ConversationScopeLeaseManager(idle_seconds=600, max_idle=4)
    closed: list[str] = []

    async def close_conversation(*, thread_id, mcp_session_id):
        closed.append(thread_id)

    runtime = SimpleNamespace(close_conversation=close_conversation)
    await manager.lease(
        runtime=runtime, owner_id="turn", scope_id="chat", retain_idle=False
    )
    await manager.lease(runtime=runtime, owner_id="tab", scope_id="chat", retain_idle=True)
    await manager.release("turn")
    await manager.release("tab")
    await asyncio.sleep(0)
    assert closed == []
    await manager.aclose()


@pytest.mark.anyio
async def test_runtime_shutdown_cancels_idle_expiry_without_conversation_close() -> None:
    """Runtime teardown owns MCP cleanup; idle timers must not fire afterward."""
    manager = main.ConversationScopeLeaseManager(idle_seconds=0.01, max_idle=4)
    closed: list[str] = []

    async def close_conversation(*, thread_id, mcp_session_id):
        closed.append(thread_id)

    runtime = SimpleNamespace(
        close_conversation=close_conversation,
        _exit_stack=AsyncExitStack(),
    )
    await manager.lease(runtime=runtime, owner_id="tab", scope_id="saved-chat")
    await manager.release("tab")
    await runtime._exit_stack.aclose()
    await asyncio.sleep(0.03)

    assert closed == []


@pytest.mark.anyio
async def test_failed_idle_close_is_retried_without_losing_scope() -> None:
    """A transient MCP close failure must not orphan an idle scope."""
    manager = main.ConversationScopeLeaseManager(
        idle_seconds=0.01, max_idle=4, retry_seconds=0.01, max_close_attempts=2
    )
    attempts: list[str] = []
    closed = asyncio.Event()

    async def close_conversation(*, thread_id, mcp_session_id):
        attempts.append(thread_id)
        if len(attempts) <= 2:
            raise RuntimeError("temporary transport failure")
        closed.set()

    runtime = SimpleNamespace(close_conversation=close_conversation)
    await manager.lease(
        runtime=runtime, owner_id="tab", scope_id="chat", retain_idle=False
    )
    await manager.release("tab")
    assert attempts == ["chat"]

    await asyncio.wait_for(closed.wait(), timeout=1)
    assert attempts == ["chat", "chat", "chat"]
    await manager.aclose()


@pytest.mark.anyio
async def test_settings_retarget_keeps_old_scope_until_active_turn_finishes(
    monkeypatch,
) -> None:
    """A tab's new settings must not close an in-flight old-thread turn."""
    manager = main.ConversationScopeLeaseManager(idle_seconds=600, max_idle=4)
    monkeypatch.setattr(main, "conversation_scopes", manager)
    closed: list[str] = []
    turn_started = asyncio.Event()
    finish_turn = asyncio.Event()

    async def close_conversation(*, thread_id, mcp_session_id):
        closed.append(thread_id)

    runtime = SimpleNamespace(
        close_conversation=close_conversation,
        config=SimpleNamespace(extensions=SimpleNamespace(mcp_stateful=False)),
    )

    class _TurnRunner:
        def __init__(self, *_args, **_kwargs):
            pass

        async def run(self, _request, _renderer):
            turn_started.set()
            await finish_turn.wait()
            return SimpleNamespace(status="completed", reflection=None, agent=None)

    monkeypatch.setattr(main, "TurnRunner", _TurnRunner)
    monkeypatch.setattr(main, "async_subagent_url_override", lambda: None)
    monkeypatch.setattr(main, "get_run_task_list", lambda **_kwargs: asyncio.sleep(0, result=None))
    await manager.lease(
        runtime=runtime, owner_id="tab", scope_id="old-thread", retain_idle=False
    )
    settings = main.AppSettings(
        model_name="test", reasoning_level="medium", thread_id="old-thread"
    )
    turn = asyncio.create_task(
        main._run_agent_turn(
            runtime=runtime,
            settings=settings,
            agent_prompt="hello",
            effective_reasoning_level="medium",
            effective_model_name="test",
            reasoning_level_is_explicit=False,
            mcp_session_id="old-thread",
        )
    )
    await asyncio.wait_for(turn_started.wait(), timeout=1)
    await manager.lease(
        runtime=runtime, owner_id="tab", scope_id="new-thread", retain_idle=False
    )
    await asyncio.sleep(0)
    assert closed == []

    finish_turn.set()
    await asyncio.wait_for(turn, timeout=1)
    await asyncio.sleep(0)
    assert closed == ["old-thread"]
    await manager.aclose()
