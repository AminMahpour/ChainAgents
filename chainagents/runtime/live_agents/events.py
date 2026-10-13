"""In-process event bus for live agent lifecycle notifications."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

LiveAgentEventKind = Literal[
    "spawned", "turn_started", "turn_finished", "message", "stopped"
]


@dataclass(frozen=True)
class LiveAgentEvent:
    """A lifecycle event emitted by a live agent."""

    kind: LiveAgentEventKind
    session_id: str
    agent_id: str
    agent_name: str
    payload: Mapping[str, object]


NOTICE_BODY_LIMIT = 200


def format_live_agent_notice(event: LiveAgentEvent) -> str | None:
    """One-line terminal notice for a reply to the user or a stop, else ``None``.

    Replies show the first line of the body, truncated to ``NOTICE_BODY_LIMIT``
    characters.
    """
    if event.kind == "message" and event.payload.get("to") == "user":
        body = str(event.payload.get("body", "")).strip()
        first = body.splitlines()[0] if body else ""
        if len(first) > NOTICE_BODY_LIMIT:
            first = first[: NOTICE_BODY_LIMIT - 3] + "..."
        return f"{event.agent_name}: {first}"
    if event.kind == "stopped":
        return f"{event.agent_name} stopped ({event.payload.get('reason', 'unknown')})"
    return None


class LiveAgentEventBus:
    """Fan out live agent events to per-session asyncio queues."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[
            str, list[tuple[asyncio.Queue[LiveAgentEvent], asyncio.AbstractEventLoop]]
        ] = {}
        self._listeners: list[Callable[[LiveAgentEvent], None]] = []

    def add_listener(self, listener: Callable[[LiveAgentEvent], None]) -> None:
        """Register a synchronous callback run on every published event."""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[LiveAgentEvent], None]) -> None:
        """Unregister a callback added with ``add_listener``."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def subscribe(self, session_id: str) -> asyncio.Queue[LiveAgentEvent]:
        """Subscribe the running event loop to a session's events."""
        queue: asyncio.Queue[LiveAgentEvent] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subscribers.setdefault(session_id, []).append((queue, loop))
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[LiveAgentEvent]) -> None:
        with self._lock:
            entries = [
                entry
                for entry in self._subscribers.get(session_id, [])
                if entry[0] is not queue
            ]
            if entries:
                self._subscribers[session_id] = entries
            else:
                self._subscribers.pop(session_id, None)

    def publish(self, event: LiveAgentEvent) -> None:
        """Deliver without blocking; safe to call from any thread."""
        with self._lock:
            entries = list(self._subscribers.get(event.session_id, []))
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                logger.exception("Live agent event listener failed")
        for queue, loop in entries:
            if loop.is_closed():
                continue
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except RuntimeError:
                # The loop closed after the check; its subscriber is gone.
                self.unsubscribe(event.session_id, queue)
