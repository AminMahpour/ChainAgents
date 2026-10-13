"""Chainlit side panel that shows the conversation's live agents."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import datetime, UTC
from typing import TYPE_CHECKING, Any

import chainlit as cl
from chainlit.utils import utc_now

from chainagents.runtime.live_agents.events import LiveAgentEvent

if TYPE_CHECKING:
    from chainagents.runtime.live_agents.manager import LiveAgentManager

logger = logging.getLogger(__name__)

THREAD_HISTORY_LIMIT = 50
PANEL_ELEMENT_NAME = "LiveAgentsPanel"
PANEL_TITLE = "Live agents"
CLOSED_STEP_OUTPUT = "Panel closed before this turn finished."


async def _finish_step(step: Any, output: str) -> None:
    """Close a ``live:<name>`` activity step with ``output``."""
    step.output = output
    step.end = utc_now()
    await step.update()


def build_panel_props(
    agents: list[dict[str, object]],
    threads: Mapping[str, list[dict[str, object]]],
) -> dict[str, object]:
    """Build the ``LiveAgentsPanel`` props from manager rows and thread history.

    Threads are looked up by agent id (falling back to name) and capped at the
    last ``THREAD_HISTORY_LIMIT`` entries.
    """
    cards: list[dict[str, object]] = []
    for agent in agents:
        thread = threads.get(str(agent.get("id", "")))
        if thread is None:
            thread = threads.get(str(agent.get("name", "")), [])
        cards.append(
            {
                **agent,
                "thread": [dict(entry) for entry in list(thread)[-THREAD_HISTORY_LIMIT:]],
            }
        )
    return {"agents": cards}


class LiveAgentsPanel:
    """Subscribe to live agent events and keep the sidebar panel current."""

    def __init__(
        self,
        manager: LiveAgentManager,
        session_id: str,
        *,
        min_interval: float = 0.25,
        stream_activity: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.manager = manager
        self.session_id = session_id
        self.min_interval = min_interval
        self.stream_activity = stream_activity
        self._clock = clock
        self._threads: dict[str, deque[dict[str, object]]] = {}
        self._element: Any = None
        self._queue: asyncio.Queue[LiveAgentEvent] | None = None
        self._consumer: asyncio.Task[None] | None = None
        self._trailing: asyncio.Task[None] | None = None
        self._last_render: float | None = None
        self._dirty = False
        self._reopen = False
        self._steps: dict[str, Any] = {}

    @property
    def is_running(self) -> bool:
        """Whether the panel is started and still listening for events."""
        return self._consumer is not None and not self._consumer.done()

    async def start(self) -> None:
        """Subscribe to the session's events; render now if agents exist."""
        if self._consumer is not None:
            return
        self._queue = self.manager.events.subscribe(self.session_id)
        self._consumer = asyncio.create_task(self._consume())
        try:
            existing = self.manager.list(self.session_id)
        except Exception:
            existing = []
        if existing:
            self._request_render()

    async def aclose(self) -> None:
        """Stop listening, cancel any pending render and end open turn steps."""
        for task in (self._consumer, self._trailing):
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
        self._consumer = None
        self._trailing = None
        if self._queue is not None:
            self.manager.events.unsubscribe(self.session_id, self._queue)
            self._queue = None
        steps, self._steps = list(self._steps.values()), {}
        for step in steps:
            try:
                await _finish_step(step, CLOSED_STEP_OUTPUT)
            except Exception:
                # The Chainlit context may already be gone at session end.
                logger.debug("Could not finish a live agent step on close", exc_info=True)

    async def _consume(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                await self._handle(event)
            except Exception:
                logger.exception("Live agents panel failed to handle an event")

    async def _handle(self, event: LiveAgentEvent) -> None:
        if event.kind == "message":
            payload = event.payload
            thread = self._threads.setdefault(
                event.agent_id, deque(maxlen=THREAD_HISTORY_LIMIT)
            )
            thread.append(
                {
                    "from": str(payload.get("from", "")),
                    "to": str(payload.get("to", "")),
                    "body": str(payload.get("body", "")),
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
        elif event.kind == "spawned":
            # Reopen the sidebar if the user closed it before this spawn.
            self._reopen = True
        if self.stream_activity and event.kind in ("turn_started", "turn_finished"):
            await self._render_activity(event)
        self._request_render()

    async def _render_activity(self, event: LiveAgentEvent) -> None:
        if event.kind == "turn_started":
            step = cl.Step(name=f"live:{event.agent_name}", type="run")
            self._steps[event.agent_id] = step
            await step.send()
            return
        step = self._steps.pop(event.agent_id, None)
        if step is None:
            return
        turns = event.payload.get("turns")
        ok = bool(event.payload.get("ok", True))
        await _finish_step(step, f"Turn {turns} {'finished' if ok else 'failed'}.")

    def _request_render(self) -> None:
        """Render now if the interval has passed, else once it has (trailing)."""
        self._dirty = True
        if self._trailing is not None and not self._trailing.done():
            return  # The running render loop will pick up the new state.
        now = self._clock()
        if self._last_render is None or now - self._last_render >= self.min_interval:
            delay = 0.0
        else:
            delay = self.min_interval - (now - self._last_render)
        self._trailing = asyncio.create_task(self._render_loop(delay))

    async def _render_loop(self, delay: float) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        while self._dirty:
            self._dirty = False
            try:
                await self._render()
            except Exception:
                logger.exception("Live agents panel render failed")
            if self._dirty and self.min_interval > 0:
                await asyncio.sleep(self.min_interval)

    async def _render(self) -> None:
        self._last_render = self._clock()
        props = build_panel_props(
            self.manager.list(self.session_id, include_stopped=True),
            {key: list(entries) for key, entries in self._threads.items()},
        )
        if self._element is None or self._reopen:
            if self._element is None:
                self._element = cl.CustomElement(
                    name=PANEL_ELEMENT_NAME, props=props, display="side"
                )
            else:
                self._element.props = props
            self._reopen = False
            await cl.ElementSidebar.set_elements([self._element])
            await cl.ElementSidebar.set_title(PANEL_TITLE)
            return
        self._element.props = props
        # ``CustomElement.update()`` re-persists the element (a new session
        # file and, with a data layer, a new element row) on every call. The
        # sidebar copy was already sent once; re-emitting it carries the props.
        await cl.context.emitter.send_element(self._element.to_dict())
