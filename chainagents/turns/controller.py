"""Nonblocking, conversation-scoped user input scheduler."""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from typing import Any

from chainagents.runtime.messaging import MessageBroker
from chainagents.runtime.types import UserInputConfig


@dataclass
class InputJob:
    id: str
    session_id: str
    payload: Any
    run: Callable[[Any], Awaitable[Any]]
    followup_factory: Callable[[str], Any]
    context: Context = field(default_factory=copy_context, repr=False)
    status: str = "queued"
    result: Any = None
    error: str | None = None


@dataclass
class _Conversation:
    queue: deque[InputJob] = field(default_factory=deque)
    active: asyncio.Task[None] | None = None
    active_job: InputJob | None = None
    external_task: asyncio.Task[Any] | None = None
    external_waiters: set[asyncio.Task[Any]] = field(default_factory=set)
    paused: bool = False
    closing: bool = False
    jobs: dict[str, InputJob] = field(default_factory=dict)
    input_ids: dict[str, str] = field(default_factory=dict)
    steer_ids: dict[str, tuple[str, str]] = field(default_factory=dict)
    completed: deque[str] = field(default_factory=deque)
    changed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


class ConversationInputController:
    """Accept new input immediately while serializing agent turns per session."""

    def __init__(
        self,
        config: UserInputConfig,
        broker: MessageBroker,
        *,
        runner_serialized: bool = False,
    ) -> None:
        self.config = config
        self.broker = broker
        self.runner_serialized = runner_serialized
        self._sessions: dict[str, _Conversation] = {}

    def _session(self, session_id: str) -> _Conversation:
        return self._sessions.setdefault(session_id, _Conversation())

    def _reserved_followups(self, session_id: str) -> int:
        pending = self.broker.pending_user_steering_count(session_id)
        batch_size = self.broker.config.max_deliveries_per_step
        return (pending + batch_size - 1) // batch_size

    def submit(
        self,
        session_id: str,
        payload: Any,
        run: Callable[[Any], Awaitable[Any]],
        *,
        followup_factory: Callable[[str], Any],
        input_id: str | None = None,
        require_idle: bool = False,
    ) -> InputJob:
        if not self.config.enabled:
            raise ValueError("Nonblocking user input is disabled.")
        session = self._session(session_id)
        if input_id is not None and input_id in session.input_ids:
            return session.jobs[session.input_ids[input_id]]
        if session.closing:
            raise ValueError("Conversation is closing.")
        if require_idle and (
            session.paused
            or session.active is not None
            or session.external_task is not None
            or session.external_waiters
            or session.queue
        ):
            raise ValueError("Choose steer or queue while a conversation is busy or paused.")
        if len(session.queue) + self._reserved_followups(session_id) >= self.config.max_queued_turns:
            raise ValueError("Queued turn limit reached.")
        job = InputJob(uuid.uuid4().hex, session_id, payload, run, followup_factory)
        session.jobs[job.id] = job
        if input_id is not None:
            session.input_ids[input_id] = job.id
        session.queue.append(job)
        self._start_next(session_id)
        session.changed.set()
        return job

    def _start_next(self, session_id: str) -> None:
        session = self._session(session_id)
        if (
            session.paused
            or session.closing
            or session.active is not None
            or session.external_task is not None
            or session.external_waiters
            or not session.queue
        ):
            return
        job = session.queue.popleft()
        job.status = "waiting" if self.runner_serialized else "running"
        session.active_job = job
        self.broker.open(session_id, "main", name="main", parent=None)
        task = asyncio.create_task(
            self._run(job), name=f"agent-turn-{job.id}", context=job.context
        )
        session.active = task
        task.add_done_callback(
            lambda done: self._finalize_unstarted_job(session_id, session, job, done)
        )

    def _finalize_unstarted_job(
        self,
        session_id: str,
        session: _Conversation,
        job: InputJob,
        task: asyncio.Task[None],
    ) -> None:
        """Release the active slot when cancellation prevented ``_run`` from starting."""
        if self._sessions.get(session_id) is not session or session.active is not task:
            return
        if task.cancelled():
            job.status = "cancelled"
        else:
            error = task.exception()
            job.status = "failed"
            job.error = str(error) if error is not None else "Turn ended without finalizing."
        self._finish_job(session, job)

    async def _run(self, job: InputJob) -> None:
        session = self._session(job.session_id)
        try:
            job.result = await job.run(job.payload)
            job.status = "completed"
        except asyncio.CancelledError:
            job.status = "cancelled"
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
        finally:
            self._finish_job(session, job)

    def _finish_job(self, session: _Conversation, job: InputJob) -> None:
        # A late steering note becomes a visible follow-up ahead of queued turns.
        self.broker.recover(job.session_id, "main")
        steering = self.broker.take_late_user_steering(job.session_id)
        batch_size = self.broker.config.max_deliveries_per_step
        batches = [
            steering[index : index + batch_size]
            for index in range(0, len(steering), batch_size)
        ]
        for batch in reversed(batches):
            notes = "\n\n".join(item.body for item in batch)
            followup = InputJob(
                uuid.uuid4().hex,
                job.session_id,
                job.followup_factory(notes),
                job.run,
                job.followup_factory,
                context=job.context.copy(),
            )
            session.jobs[followup.id] = followup
            session.queue.appendleft(followup)
        job.payload = None
        job.followup_factory = lambda text: text
        session.completed.append(job.id)
        while len(session.completed) > self.config.max_completed_turns:
            expired_id = session.completed.popleft()
            session.jobs.pop(expired_id, None)
            for input_id, known_id in tuple(session.input_ids.items()):
                if known_id == expired_id:
                    session.input_ids.pop(input_id, None)
            for input_id, (_, known_id) in tuple(session.steer_ids.items()):
                if known_id == expired_id:
                    session.steer_ids.pop(input_id, None)
        session.active = None
        session.active_job = None
        self._start_next(job.session_id)
        session.changed.set()

    def steer(self, session_id: str, text: str, *, input_id: str | None = None) -> str:
        if not self.config.enabled:
            raise ValueError("Nonblocking user input is disabled.")
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("No active turn to steer.")
        if input_id is not None and input_id in session.steer_ids:
            return session.steer_ids[input_id][0]
        if (
            session.closing
            or session.paused
            or session.active is None
            or session.active_job is None
            or session.active_job.status != "running"
            or session.external_task is not None
        ):
            raise ValueError("No active turn to steer.")
        projected_steering = self.broker.pending_user_steering_count(session_id) + 1
        batch_size = self.broker.config.max_deliveries_per_step
        projected_followups = (projected_steering + batch_size - 1) // batch_size
        if len(session.queue) + projected_followups > self.config.max_queued_turns:
            raise ValueError("Queued turn limit reached; wait before steering this turn.")
        message_id = self.broker.send_user(session_id, text).id
        if input_id is not None:
            assert session.active_job is not None
            session.steer_ids[input_id] = (message_id, session.active_job.id)
        return message_id

    def stop(self, session_id: str) -> None:
        session = self._session(session_id)
        session.paused = True
        for task in tuple(session.external_waiters):
            task.cancel()
        if session.external_task is not None:
            session.external_task.cancel()
        if session.active is not None:
            session.active.cancel()
        session.changed.set()

    def resume(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None or session.closing:
            return
        session.paused = False
        self._start_next(session_id)
        session.changed.set()

    def status(self, session_id: str) -> dict[str, Any]:
        session = self._sessions.get(session_id)
        if session is None:
            return {
                "active_job_id": None,
                "external_active": False,
                "queued_job_ids": [],
                "paused": False,
            }
        return {
            "active_job_id": session.active_job.id if session.active_job else None,
            "external_active": session.external_task is not None or bool(session.external_waiters),
            "queued_job_ids": [job.id for job in session.queue],
            "paused": session.paused,
        }

    def has_session(self, session_id: str) -> bool:
        """Return whether this conversation has existing turn state."""
        return session_id in self._sessions

    def busy(self, session_id: str) -> bool:
        """Return whether a conversation has active, waiting, or queued input."""
        session = self._sessions.get(session_id)
        return bool(
            session is not None
            and (
                session.active is not None
                or session.external_task is not None
                or session.external_waiters
                or session.queue
            )
        )

    async def wait_drained(self, session_id: str) -> None:
        """Wait for all input work, including paused queues and external turns."""
        while True:
            session = self._sessions.get(session_id)
            if session is None or not self.busy(session_id):
                return
            # No await separates the state check from clearing the signal, so a
            # subsequent transition cannot be missed by the waiter.
            session.changed.clear()
            await session.changed.wait()

    def turn_waiting(self, session_id: str) -> bool:
        """Track an external turn before it can wait for the shared lock."""
        session = self._session(session_id)
        if session.paused or session.closing:
            return False
        task = asyncio.current_task()
        if task is not None and task is not session.active:
            session.external_waiters.add(task)
            session.changed.set()
        return True

    def turn_wait_finished(self, session_id: str) -> None:
        """Discard a waiter after lock acquisition or cancellation."""
        session = self._sessions.get(session_id)
        if session is not None:
            session.external_waiters.discard(asyncio.current_task())
            self._start_next(session_id)
            session.changed.set()

    def turn_started(self, session_id: str) -> bool:
        """Record the task that acquired the main-turn lock."""
        session = self._session(session_id)
        task = asyncio.current_task()
        session.external_waiters.discard(task)
        if session.paused or session.closing:
            return False
        if task is session.active:
            if session.active_job is not None:
                session.active_job.status = "running"
        else:
            session.external_task = task
        session.changed.set()
        return True

    def turn_finished(self, session_id: str) -> None:
        """Release an external turn before starting queued controller work."""
        session = self._sessions.get(session_id)
        if session is None:
            return
        task = asyncio.current_task()
        if task is session.active:
            if session.active_job is not None and session.active_job.status == "running":
                session.active_job.status = "finishing"
        elif task is session.external_task:
            session.external_task = None
            self._start_next(session_id)
        session.changed.set()

    def get(self, session_id: str, job_id: str) -> InputJob:
        try:
            return self._sessions[session_id].jobs[job_id]
        except KeyError as exc:
            raise ValueError("Turn is unavailable in this conversation.") from exc

    def find_input_id(self, session_id: str, input_id: str) -> InputJob | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        job_id = session.input_ids.get(input_id)
        return session.jobs.get(job_id) if job_id is not None else None

    async def wait_idle(self, session_id: str) -> None:
        while True:
            session = self._sessions.get(session_id)
            task = session.active if session is not None else None
            if task is None:
                return
            await asyncio.gather(task, return_exceptions=True)

    async def close_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session.closing = True
        session.paused = True
        current_task = asyncio.current_task()
        for task in tuple(session.external_waiters):
            if task is not current_task:
                task.cancel()
        if session.external_task is not None and session.external_task is not current_task:
            session.external_task.cancel()
        if session.active is not None:
            session.active.cancel()
        closing_tasks = [
            task for task in (session.active, session.external_task, *session.external_waiters)
            if task is not None and task is not current_task
        ]
        await asyncio.gather(
            *closing_tasks,
            return_exceptions=True,
        )
        if session.active_job is not None and session.active_job.status not in {"completed", "failed", "cancelled"}:
            session.active_job.status = "cancelled"
        for job in session.queue:
            job.status = "cancelled"
            job.payload = None
        session.queue.clear()
        self._sessions.pop(session_id, None)
        session.changed.set()

    async def close_all(self) -> None:
        for session_id in tuple(self._sessions):
            await self.close_session(session_id)
