"""Regression tests for Chainlit socket disconnects during agent turns."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import pytest

import main


def _set_chat_session(monkeypatch, *, session, active, closed):
    values = {
        main.SESSION_ACTIVE_TURN_KEY: active,
        main.SESSION_SETTINGS_KEY: {"thread_id": "runtime-thread"},
    }

    async def close_conversation(*, thread_id, mcp_session_id):
        closed.append((thread_id, mcp_session_id))

    monkeypatch.setattr(main.cl, "context", SimpleNamespace(session=session))
    monkeypatch.setattr(main.cl, "user_session", SimpleNamespace(get=values.get))
    monkeypatch.setattr(
        main.AgentRuntime,
        "current",
        lambda: SimpleNamespace(close_conversation=close_conversation),
    )
    monkeypatch.setattr(main, "current_mcp_session_id", lambda: "mcp-session")


@pytest.mark.anyio
async def test_socket_disconnect_keeps_turn_and_resources_for_reconnect(monkeypatch):
    session = SimpleNamespace(socket_id="socket-1", to_clear=False)
    drafts = {"draft-1": ("runtime-thread", "follow up")}
    setattr(session, main.SESSION_INPUT_DRAFTS_KEY, drafts)
    closed: list[tuple[str | None, str | None]] = []
    finish_turn = asyncio.Event()
    active = asyncio.create_task(finish_turn.wait())
    _set_chat_session(monkeypatch, session=session, active=active, closed=closed)
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
async def test_expired_disconnected_session_stops_turn_and_closes_resources(
    monkeypatch,
):
    session = SimpleNamespace(socket_id="socket-1", to_clear=False)
    closed: list[tuple[str | None, str | None]] = []
    active = asyncio.create_task(asyncio.Event().wait())
    _set_chat_session(monkeypatch, session=session, active=active, closed=closed)
    monkeypatch.setattr(main.chainlit_config.project, "session_timeout", 0)

    await main.on_chat_end()
    assert not active.done()
    await asyncio.wait_for(
        getattr(session, main.SESSION_DISCONNECT_CLEANUP_KEY), timeout=1
    )

    assert active.cancelled()
    assert closed == [("runtime-thread", "mcp-session")]


@pytest.mark.anyio
async def test_explicit_chat_clear_stops_turn_immediately(monkeypatch):
    session = SimpleNamespace(socket_id="socket-1", to_clear=True)
    setattr(session, main.SESSION_INPUT_DRAFTS_KEY, {"draft-1": "pending"})
    closed: list[tuple[str | None, str | None]] = []
    active = asyncio.create_task(asyncio.Event().wait())
    _set_chat_session(monkeypatch, session=session, active=active, closed=closed)

    await main.on_chat_end()

    assert active.cancelled()
    assert closed == [("runtime-thread", "mcp-session")]
    assert getattr(session, main.SESSION_INPUT_DRAFTS_KEY) == {}
