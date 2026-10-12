"""Lifecycle manager for live agents: spawn, message, teardown and reaping."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from typing import Any

from chainagents.runtime.live_agents.config import (
    ResolvedLiveAgentSpec,
    SpawnRequest,
    resolve_spawn,
)
from chainagents.runtime.live_agents.events import (
    LiveAgentEvent,
    LiveAgentEventBus,
    LiveAgentEventKind,
)
from chainagents.runtime.live_agents.registry import (
    TERMINAL_STATUSES,
    LiveAgentRecord,
    LiveAgentRegistry,
)
from chainagents.runtime.live_agents.worker import LiveAgentWorker
from chainagents.runtime.messaging import AgentMessage, MessageBroker
from chainagents.runtime.types import LiveAgentsConfig

logger = logging.getLogger(__name__)

LiveAgentGraphFactory = Callable[[ResolvedLiveAgentSpec, str], Awaitable[Any]]

WORKER_TASK_PREFIX = "chainagents-live-agent-"
_MAIN_ADDRESS = "main"
_USER_ADDRESS = "user"
_REAPER_MAX_INTERVAL = 30


class LiveAgentManager:
    """Own live agent workers and their lifecycle for every session.

    All methods must be called from the same event loop: the registry is not
    locked.
    """

    def __init__(
        self,
        config: LiveAgentsConfig,
        broker: MessageBroker,
        graph_factory: LiveAgentGraphFactory,
        *,
        events: LiveAgentEventBus | None = None,
        clock: Callable[[], float] = time.monotonic,
        delete_thread: Callable[[str], Awaitable[None]] | None = None,
        turn_context: Callable[[LiveAgentRecord], AbstractContextManager[Any]]
        | None = None,
    ) -> None:
        self.config = config
        self.events = events if events is not None else LiveAgentEventBus()
        self._broker = broker
        self._graph_factory = graph_factory
        self._clock = clock
        self._delete_thread = delete_thread
        self._turn_context = turn_context
        self._registry = LiveAgentRegistry(config, clock=clock)
        self._workers: dict[str, asyncio.Task[None]] = {}
        # One teardown task per agent id; doubles as the "torn down" marker.
        self._teardowns: dict[str, asyncio.Task[LiveAgentRecord]] = {}
        self._background: set[asyncio.Task[Any]] = set()
        self._sessions: set[str] = set()
        # Sessions whose teardown is in progress; spawns into them are refused.
        self._closing_sessions: dict[str, int] = {}
        self._closed = False
        self._reaper: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ start

    def start(self) -> None:
        """Start the idle reaper; idempotent and a no-op outside a running loop."""
        if self._reaper is not None and not self._reaper.done():
            return
        if self.config.idle_timeout_seconds <= 0:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # started lazily on the first spawn
        self._reaper = loop.create_task(
            self._reap_forever(), name="chainagents-live-agents-reaper"
        )

    async def close(self) -> None:
        """Stop the reaper and tear down every session's live agents.

        Spawns are refused afterwards; a spawn still building its graph aborts.
        """
        self._closed = True
        reaper, self._reaper = self._reaper, None
        if reaper is not None:
            reaper.cancel()
            await asyncio.wait({reaper})
        for session_id in list(self._sessions):
            await self.teardown_session(session_id)
        if self._background:
            await asyncio.wait(set(self._background))

    async def _reap_forever(self) -> None:
        interval = min(_REAPER_MAX_INTERVAL, self.config.idle_timeout_seconds)
        while True:
            await asyncio.sleep(interval)
            try:
                await self.reap_idle()
            except Exception:
                logger.exception("Live agent idle reaper failed")

    # ------------------------------------------------------------------ spawn

    async def spawn(
        self,
        session_id: str,
        request: SpawnRequest,
        *,
        spawned_by: str = _MAIN_ADDRESS,
        parent_trace: Any = None,
    ) -> tuple[LiveAgentRecord, ResolvedLiveAgentSpec]:
        self._check_accepting(session_id)
        self.start()
        registry = self._registry
        spec = resolve_spawn(
            request,
            self.config,
            existing_names=registry.live_names(session_id),
            sequence=registry.next_sequence(session_id),
        )
        record = registry.reserve(session_id, spec, spawned_by=spawned_by)
        record.parent_trace = parent_trace
        try:
            graph = await self._graph_factory(spec, session_id)
        except BaseException:
            registry.discard(record)
            raise
        if not self._still_wanted(record):
            # Torn down, forgotten (session end) or the manager closed while the
            # graph was being built: never start a worker nobody can stop.
            if record.status not in TERMINAL_STATUSES:
                registry.discard(record)
            raise RuntimeError(f"Live agent '{record.name}' was stopped during spawn.")
        self._sessions.add(session_id)

        broker = self._broker
        broker.open(session_id, _MAIN_ADDRESS, name=_MAIN_ADDRESS, parent=None)
        broker.open(session_id, record.address, name=record.name, parent=_MAIN_ADDRESS)
        worker = LiveAgentWorker(
            record,
            graph,
            broker,
            self.events,
            recursion_limit=self.config.turn_recursion_limit,
            max_turns=self.config.max_turns_per_agent,
            clock=self._clock,
            turn_context=self._turn_context,
            on_finished=self._on_worker_finished,
        )
        task = asyncio.get_running_loop().create_task(
            worker.run(), name=f"{WORKER_TASK_PREFIX}{record.id}"
        )
        self._workers[record.id] = task
        task.add_done_callback(lambda done: self._on_worker_done(record, done))
        self._publish(
            "spawned", record, {**record.to_payload(), "address": record.address}
        )

        text = request.initial_message
        if text is not None and text.strip():
            try:
                if spawned_by == _USER_ADDRESS:
                    broker.send_from_user(session_id, record.address, text)
                else:
                    broker.send(session_id, _MAIN_ADDRESS, record.address, text)
            except ValueError:
                await self.teardown(session_id, record.id, reason="spawn_failed")
                raise
            self._publish(
                "message",
                record,
                {"from": spawned_by, "to": record.name, "body": text},
            )
        return record, spec

    def _check_accepting(self, session_id: str) -> None:
        if self._closed:
            raise RuntimeError("The live agent manager is closed.")
        if session_id in self._closing_sessions:
            raise RuntimeError("This conversation's live agents are shutting down.")

    def _still_wanted(self, record: LiveAgentRecord) -> bool:
        if self._closed or record.session_id in self._closing_sessions:
            return False
        if record.status != "idle":
            return False
        try:
            return self._registry.lookup(record.session_id, record.id) is record
        except KeyError:
            return False

    # ------------------------------------------------------------ inspection

    def list(
        self, session_id: str, *, include_stopped: bool = False
    ) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for record in self._registry.records(
            session_id, include_stopped=include_stopped
        ):
            try:
                pending = len(self._broker.pending(session_id, record.address))
            except ValueError:
                pending = 0
            payloads.append(record.to_payload(pending=pending))
        return payloads

    def get(self, session_id: str, ref: str) -> LiveAgentRecord:
        return self._registry.lookup(session_id, ref)

    def templates_payload(self) -> dict[str, object]:
        allow = self.config.allow
        return {
            "templates": [
                {
                    "name": t.name,
                    "description": t.description,
                    "model": t.model,
                    "tools": list(t.tools),
                    "skills": list(t.skills),
                    "mcp_servers": list(t.mcp_servers),
                }
                for t in self.config.templates
            ],
            "allow": {
                "applies_to": allow.applies_to,
                "models": list(allow.models),
                "mcp_servers": list(allow.mcp_servers),
                "skills": list(allow.skills),
                "tools": list(allow.tools),
                "allow_custom_prompt": allow.allow_custom_prompt,
            },
        }

    def roster(self, session_id: str) -> str:
        records = self._registry.records(session_id)
        if not records:
            return ""
        return "Live agents: " + ", ".join(f"{r.name}: {r.status}" for r in records)

    # -------------------------------------------------------------- messages

    def send_from_user(self, session_id: str, ref: str, body: str) -> AgentMessage:
        record = self.get(session_id, ref)
        message = self._broker.send_from_user(session_id, record.address, body)
        self._publish(
            "message", record, {"from": _USER_ADDRESS, "to": record.name, "body": body}
        )
        return message

    # -------------------------------------------------------------- teardown

    async def teardown(
        self, session_id: str, ref: str, *, reason: str = "requested"
    ) -> LiveAgentRecord:
        record = self._registry.lookup(session_id, ref)
        if record.status in TERMINAL_STATUSES and record.id not in self._teardowns:
            return record
        return await asyncio.shield(self._start_teardown(record, reason))

    async def teardown_session(self, session_id: str) -> None:
        # Refuse new spawns first, so the snapshot below covers every record.
        self._closing_sessions[session_id] = (
            self._closing_sessions.get(session_id, 0) + 1
        )
        try:
            records = self._registry.records(session_id, include_stopped=True)
            for record in records:
                if (
                    record.status not in TERMINAL_STATUSES
                    or record.id in self._teardowns
                ):
                    await asyncio.shield(self._start_teardown(record, "session_end"))
            self._registry.forget_session(session_id)
            self._sessions.discard(session_id)
            for record in records:
                self._teardowns.pop(record.id, None)
        finally:
            remaining = self._closing_sessions.pop(session_id) - 1
            if remaining:
                self._closing_sessions[session_id] = remaining

    async def reap_idle(self) -> list[LiveAgentRecord]:
        timeout = self.config.idle_timeout_seconds
        if timeout <= 0:
            return []
        now = self._clock()
        reaped: list[LiveAgentRecord] = []
        for session_id in list(self._sessions):
            for record in self._registry.records(session_id):
                if (
                    record.status == "idle"
                    and record.id not in self._teardowns
                    and now - record.last_active_at >= timeout
                ):
                    await self.teardown(session_id, record.id, reason="idle_timeout")
                    reaped.append(record)
        return reaped

    def _start_teardown(
        self, record: LiveAgentRecord, reason: str
    ) -> asyncio.Task[LiveAgentRecord]:
        """Return the record's single teardown task, creating it if needed."""
        task = self._teardowns.get(record.id)
        if task is None:
            task = asyncio.get_running_loop().create_task(
                self._do_teardown(record, reason),
                name=f"chainagents-live-agents-teardown-{record.id}",
            )
            self._teardowns[record.id] = task
            self._track(task)
        return task

    async def _do_teardown(
        self, record: LiveAgentRecord, reason: str
    ) -> LiveAgentRecord:
        record.status = "stopping"
        worker = self._workers.pop(record.id, None)
        if worker is not None and not worker.done():
            worker.cancel()
            await asyncio.wait({worker})
        self._broker.close(record.session_id, record.address)
        if self._delete_thread is not None and self.config.delete_history_on_teardown:
            try:
                await self._delete_thread(record.thread_id)
            except Exception:
                logger.warning(
                    "Could not delete history for live agent %s",
                    record.id,
                    exc_info=True,
                )
        record.status = "failed" if reason == "failed" else "stopped"
        record.stop_reason = reason
        record.last_active_at = self._clock()
        self._publish("stopped", record, {"reason": reason})
        return record

    async def _on_worker_finished(self, record: LiveAgentRecord, reason: str) -> None:
        # Called from inside the worker task: schedule, never await, the teardown
        # (it cancels and awaits the worker task itself).
        self._start_teardown(record, reason)

    def _on_worker_done(
        self, record: LiveAgentRecord, task: asyncio.Task[None]
    ) -> None:
        if self._workers.get(record.id) is task:
            del self._workers[record.id]
        reason = "worker_exited"
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "Live agent %s worker crashed",
                    record.id,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                reason = "failed"
        if record.id in self._teardowns or record.status in TERMINAL_STATUSES:
            return
        try:
            self._start_teardown(record, reason)
        except RuntimeError:  # event loop is shutting down
            self._broker.close(record.session_id, record.address)
            record.status = "failed" if reason == "failed" else "stopped"
            record.stop_reason = reason

    # --------------------------------------------------------------- helpers

    def _track(self, task: asyncio.Task[Any]) -> None:
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _publish(
        self,
        kind: LiveAgentEventKind,
        record: LiveAgentRecord,
        payload: dict[str, object],
    ) -> None:
        self.events.publish(
            LiveAgentEvent(
                kind=kind,
                session_id=record.session_id,
                agent_id=record.id,
                agent_name=record.name,
                payload=payload,
            )
        )
