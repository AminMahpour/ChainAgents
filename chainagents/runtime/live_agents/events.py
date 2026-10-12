"""In-process event bus for live agent lifecycle notifications."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

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


class LiveAgentEventBus:
    """Fan out live agent events to per-session asyncio queues."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[
            str, list[tuple[asyncio.Queue[LiveAgentEvent], asyncio.AbstractEventLoop]]
        ] = {}

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
        for queue, loop in entries:
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, event)
