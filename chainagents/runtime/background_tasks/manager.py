"""Process-local background execution for configured synchronous subagents."""

from __future__ import annotations

import asyncio
import builtins
import contextvars
import time
import uuid
import weakref
from collections.abc import AsyncIterator, Sequence

from contextlib import asynccontextmanager

from chainagents.events.stream import AgentStreamEvent
from chainagents.runtime.artifacts import LargeToolResultArtifactRegistry
from chainagents.runtime.background_tasks.context import (
    _CURRENT_BACKGROUND_INVOCATION_PATH,
    _CURRENT_BACKGROUND_SESSION_GENERATION,
    _CURRENT_BACKGROUND_SESSION_ID,
    _CURRENT_BACKGROUND_TASK_ID,
    BackgroundSessionGeneration,
)
from chainagents.runtime.background_tasks.models import (
    BackgroundCleanup,
    BackgroundRunner,
    BackgroundTaskActivity,
    BackgroundTaskSnapshot,
    BackgroundTaskStatus,
    TERMINAL_BACKGROUND_TASK_STATUSES,
    _BackgroundTaskRecord,
    _BackgroundTaskSubmission,
)
from chainagents.runtime.background_tasks.queues import (
    SUBSCRIBER_QUEUE_MAXSIZE,
    _put_activity_evicting_live,
    _put_evicting_oldest,
    await_preserving_cancellation,
)
from chainagents.runtime.types import BackgroundSubagentConfig


class BackgroundTaskManager:
    """Own process-local background subagent jobs for all conversations."""

    def __init__(
        self,
        config: BackgroundSubagentConfig,
        *,
        artifact_registry: LargeToolResultArtifactRegistry | None = None,
    ) -> None:
        self.config = config
        self.artifact_registry = artifact_registry
        self._lock = asyncio.Lock()
        self._records: dict[str, _BackgroundTaskRecord] = {}
        self._session_task_ids: dict[str, list[str]] = {}
        self._subscribers: dict[str, set[asyncio.Queue[BackgroundTaskSnapshot]]] = {}
        self._activity_subscribers: dict[
            str, set[asyncio.Queue[BackgroundTaskActivity]]
        ] = {}
        self._closing_sessions: set[str] = set()
        self._session_close_locks: dict[str, asyncio.Lock] = {}
        self._session_close_users: dict[str, int] = {}
        self._session_generations: weakref.WeakValueDictionary[
            str, BackgroundSessionGeneration
        ] = weakref.WeakValueDictionary()
        self._close_task: asyncio.Task[None] | None = None
        self._terminal_close = False
        self._closed = False

    def session_generation(self, session_id: str) -> BackgroundSessionGeneration:
        """Return the current lifecycle generation for one session."""
        normalized_session = session_id.strip()
        if not normalized_session:
            raise ValueError("Background tasks require a non-empty session ID.")
        if self._closed:
            raise RuntimeError("The background task manager is closed.")
        if normalized_session in self._closing_sessions:
            raise RuntimeError("The background task session is closing.")
        generation = self._session_generations.get(normalized_session)
        if generation is None:
            generation = BackgroundSessionGeneration(normalized_session)
            self._session_generations[normalized_session] = generation
        return generation

    def _validate_session_generation(
        self,
        session_id: str,
        expected: BackgroundSessionGeneration | None,
    ) -> None:
        if expected is None:
            return
        if (
            not expected.active
            or expected.session_id != session_id
            or self._session_generations.get(session_id) is not expected
        ):
            raise RuntimeError(
                "The background task session was closed; start a new foreground run."
            )

    async def spawn(
        self,
        *,
        session_id: str,
        agent_name: str,
        description: str,
        agent_path: tuple[str, ...],
        owner_path: tuple[str, ...] = (),
        runner: BackgroundRunner,
        parent_task_id: str | None = None,
        expected_session_generation: BackgroundSessionGeneration | None = None,
        cleanup: BackgroundCleanup | None = None,
    ) -> BackgroundTaskSnapshot:
        """Start a job immediately and return before the runner finishes."""
        snapshots = await self.spawn_batch(
            session_id=session_id,
            submissions=[
                _BackgroundTaskSubmission(
                    agent_name=agent_name,
                    description=description,
                    agent_path=agent_path,
                    runner=runner,
                    cleanup=cleanup,
                )
            ],
            owner_path=owner_path,
            parent_task_id=parent_task_id,
            expected_session_generation=expected_session_generation,
        )
        return snapshots[0]

    async def spawn_batch(
        self,
        *,
        session_id: str,
        submissions: Sequence[_BackgroundTaskSubmission],
        owner_path: tuple[str, ...] = (),
        parent_task_id: str | None = None,
        expected_session_generation: BackgroundSessionGeneration | None = None,
    ) -> builtins.list[BackgroundTaskSnapshot]:
        """Atomically admit and start an ordered batch of background jobs."""
        normalized_session = session_id.strip()
        if not normalized_session:
            raise ValueError("Background tasks require a non-empty session ID.")
        if not self.config.enabled:
            raise RuntimeError("Local background subagents are disabled.")
        if not submissions:
            raise ValueError("Background task batches require at least one submission.")

        async with self._lock:
            if self._closed:
                raise RuntimeError("The background task manager is closed.")
            if normalized_session in self._closing_sessions:
                raise RuntimeError("The background task session is closing.")
            self._validate_session_generation(
                normalized_session,
                expected_session_generation,
            )
            session_ids = self._session_task_ids.get(normalized_session, [])
            running_session = sum(
                self._records[task_id].status not in TERMINAL_BACKGROUND_TASK_STATUSES
                for task_id in session_ids
            )
            batch_size = len(submissions)
            if running_session + batch_size > self.config.max_running_per_session:
                raise RuntimeError(
                    "Background running limit reached for this session "
                    f"({self.config.max_running_per_session})."
                )
            running_total = sum(
                record.status not in TERMINAL_BACKGROUND_TASK_STATUSES
                for record in self._records.values()
            )
            if running_total + batch_size > self.config.max_running_total:
                raise RuntimeError(
                    "Background running limit reached for this process "
                    f"({self.config.max_running_total})."
                )
            # Validate the requested parent before eviction, and spare it: the new
            # children are not retained yet, so nothing else would stop eviction
            # from removing the very parent this spawn needs. Its own ancestors are
            # already spared as parents of a retained record.
            protected: set[str] | None = None
            if parent_task_id is not None:
                parent = self._records.get(parent_task_id)
                if parent is None or parent.session_id != normalized_session:
                    raise ValueError(
                        "Background parent task does not exist in this session."
                    )
                if parent.cancelling:
                    raise RuntimeError("Background parent task is cancelling.")
                protected = {parent.task_id}
            overflow = (
                len(session_ids) + batch_size - self.config.max_tasks_per_session
            )
            if overflow > 0:
                evictable = self._evictable_record_ids(
                    normalized_session,
                    overflow,
                    protected=protected,
                )
                if len(evictable) < overflow:
                    raise RuntimeError(
                        "Background retained task limit reached for this session "
                        f"({self.config.max_tasks_per_session})."
                    )
                self._forget_records(normalized_session, evictable)
                session_ids = self._session_task_ids.get(normalized_session, [])

            retained_ids = self._session_task_ids.setdefault(normalized_session, [])
            records: list[_BackgroundTaskRecord] = []
            artifact_handle = (
                self.artifact_registry.open_session(normalized_session)
                if self.artifact_registry is not None
                else None
            )
            for submission in submissions:
                task_id = f"bg-{uuid.uuid4().hex[:12]}"
                record = _BackgroundTaskRecord(
                    task_id=task_id,
                    session_id=normalized_session,
                    agent_name=submission.agent_name,
                    description=submission.description,
                    agent_path=submission.agent_path,
                    owner_path=owner_path,
                    session_generation=expected_session_generation,
                    artifact_handle=artifact_handle,
                    parent_task_id=parent_task_id,
                    status="pending",
                    created_at=time.time(),
                    completion=asyncio.Event(),
                    cleanup=submission.cleanup,
                )
                self._records[task_id] = record
                retained_ids.append(task_id)
                records.append(record)
            for record, submission in zip(records, submissions, strict=True):
                record.execution = asyncio.create_task(
                    self._execute(record, submission.runner),
                    name=f"chainagents-{record.task_id}",
                    context=contextvars.Context(),
                )
            return [record.snapshot() for record in records]

    def _evictable_record_ids(
        self,
        session_id: str,
        count: int,
        *,
        protected: set[str] | None = None,
    ) -> set[str]:
        """Select up to ``count`` of the oldest settled records in a session.

        Only terminal records that no waiter is blocked on, whose cleanup has
        completed, and that no retained record names as its parent are eligible,
        so ancestry, pending cleanup and in-flight waits stay intact. ``protected``
        additionally spares records the caller still needs, such as the parent a
        pending spawn names, which nothing retained points at yet. This only
        selects; ``_forget_records`` performs the removal once the caller knows
        the batch is admissible. Callers must hold ``self._lock``.
        """
        session_ids = self._session_task_ids.get(session_id)
        if not session_ids:
            return set()
        spared = protected or set()
        evicted: set[str] = set()
        while len(evicted) < count:
            parent_ids = {
                self._records[task_id].parent_task_id
                for task_id in session_ids
                if task_id not in evicted
            }
            candidate = next(
                (
                    task_id
                    for task_id in session_ids
                    if task_id not in evicted
                    and task_id not in spared
                    and task_id not in parent_ids
                    and self._is_evictable(self._records[task_id])
                ),
                None,
            )
            if candidate is None:
                break
            evicted.add(candidate)
        return evicted

    def _forget_records(self, session_id: str, task_ids: set[str]) -> None:
        """Drop selected records from a session's retention. Callers hold the lock."""
        if not task_ids:
            return
        session_ids = self._session_task_ids.get(session_id)
        if session_ids is not None:
            session_ids[:] = [
                task_id for task_id in session_ids if task_id not in task_ids
            ]
        for task_id in task_ids:
            self._records.pop(task_id, None)

    @staticmethod
    def _is_evictable(record: _BackgroundTaskRecord) -> bool:
        return (
            record.status in TERMINAL_BACKGROUND_TASK_STATUSES
            and record.waiters == 0
            and record.cleanup is None
            and record.cleanup_task is None
            and (record.cancel_task is None or record.cancel_task.done())
        )

    async def _execute(
        self,
        record: _BackgroundTaskRecord,
        runner: BackgroundRunner,
    ) -> None:
        task_token = _CURRENT_BACKGROUND_TASK_ID.set(record.task_id)
        session_token = _CURRENT_BACKGROUND_SESSION_ID.set(record.session_id)
        owner_token = _CURRENT_BACKGROUND_INVOCATION_PATH.set(record.owner_path)
        generation_token = _CURRENT_BACKGROUND_SESSION_GENERATION.set(
            record.session_generation
        )
        artifact_token = (
            self.artifact_registry.activate(record.artifact_handle)
            if self.artifact_registry is not None
            and record.artifact_handle is not None
            else None
        )
        try:
            async with self._lock:
                if record.status == "pending":
                    record.status = "running"
                elif record.status in TERMINAL_BACKGROUND_TASK_STATUSES:
                    return
            result = await runner(record.task_id)
        except asyncio.CancelledError:
            cleanup_error = await self._cleanup_record(record)
            await self._cancel_descendants(record.session_id, record.task_id)
            await self._finish(record, status="cancelled", error=cleanup_error)
            raise
        except Exception as exc:
            detail = " ".join(str(exc).split()).strip()
            error = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
            cleanup_error = await self._cleanup_record(record)
            if cleanup_error:
                error = f"{error}; {cleanup_error}"
            await self._cancel_descendants(record.session_id, record.task_id)
            await self._finish(record, status="error", error=error)
        else:
            cleanup_error = await self._cleanup_record(record)
            if record.cancelling:
                await self._cancel_descendants(record.session_id, record.task_id)
                await self._finish(
                    record,
                    status="cancelled",
                    error=cleanup_error,
                )
            elif cleanup_error:
                await self._cancel_descendants(record.session_id, record.task_id)
                await self._finish(record, status="error", error=cleanup_error)
            else:
                await self._finish(record, status="success", result=str(result))
        finally:
            if artifact_token is not None and self.artifact_registry is not None:
                self.artifact_registry.reset(artifact_token)
            _CURRENT_BACKGROUND_SESSION_GENERATION.reset(generation_token)
            _CURRENT_BACKGROUND_INVOCATION_PATH.reset(owner_token)
            _CURRENT_BACKGROUND_SESSION_ID.reset(session_token)
            _CURRENT_BACKGROUND_TASK_ID.reset(task_token)

    async def _cleanup_record(self, record: _BackgroundTaskRecord) -> str | None:
        async with self._lock:
            cleanup = record.cleanup
            cleanup_task = record.cleanup_task
            if cleanup is not None and cleanup_task is None:

                async def run_cleanup() -> None:
                    await cleanup(record.task_id)

                cleanup_task = asyncio.create_task(
                    run_cleanup(),
                    name=f"chainagents-cleanup-{record.task_id}",
                )
                record.cleanup_task = cleanup_task
        if cleanup is None:
            return None
        assert cleanup_task is not None
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            if cleanup_task.cancelled():
                async with self._lock:
                    if record.cleanup_task is cleanup_task:
                        record.cleanup_task = None
            raise
        except Exception as exc:
            async with self._lock:
                if record.cleanup_task is cleanup_task:
                    record.cleanup_task = None
            detail = " ".join(str(exc).split()).strip()
            summary = (
                f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
            )
            return f"Background checkpoint cleanup failed: {summary}"
        async with self._lock:
            if record.cleanup_task is cleanup_task:
                record.cleanup_task = None
                record.cleanup = None
        return None

    async def _finish(
        self,
        record: _BackgroundTaskRecord,
        *,
        status: BackgroundTaskStatus,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            if record.status in TERMINAL_BACKGROUND_TASK_STATUSES:
                return
            record.status = status
            record.result = result
            record.error = error
            record.completed_at = time.time()
            if record.completion is not None:
                record.completion.set()
            snapshot = record.snapshot()
            subscribers = tuple(self._subscribers.get(record.session_id, ()))
            activity_subscribers = tuple(
                self._activity_subscribers.get(record.session_id, ())
            )
        for completion_queue in subscribers:
            _put_evicting_oldest(completion_queue, snapshot)
        terminal_activity = BackgroundTaskActivity(
            task_id=record.task_id,
            session_id=record.session_id,
            agent_name=record.agent_name,
            description=record.description,
            snapshot=snapshot,
        )
        for activity_queue in activity_subscribers:
            _put_activity_evicting_live(activity_queue, terminal_activity)

    async def publish_activity(
        self,
        task_id: str,
        event: AgentStreamEvent,
    ) -> None:
        """Publish one live event to activity subscribers for its task session."""
        async with self._lock:
            record = self._records.get(task_id)
            if record is None:
                return
            subscribers = tuple(self._activity_subscribers.get(record.session_id, ()))
            activity = BackgroundTaskActivity(
                task_id=record.task_id,
                session_id=record.session_id,
                agent_name=record.agent_name,
                description=record.description,
                event=event,
            )
        for queue in subscribers:
            # Live events are best effort; drop new ones while a consumer lags.
            if not queue.full():
                queue.put_nowait(activity)

    def _is_visible(
        self,
        record: _BackgroundTaskRecord,
        scope_path: tuple[str, ...],
        owner_path: tuple[str, ...],
        ancestor_task_id: str | None,
    ) -> bool:
        if scope_path and record.agent_path[: len(scope_path)] != scope_path:
            return False
        if owner_path and record.owner_path[: len(owner_path)] != owner_path:
            return False
        if ancestor_task_id is None:
            return True
        parent_task_id = record.parent_task_id
        visited: set[str] = set()
        while parent_task_id is not None and parent_task_id not in visited:
            if parent_task_id == ancestor_task_id:
                return True
            visited.add(parent_task_id)
            parent = self._records.get(parent_task_id)
            parent_task_id = parent.parent_task_id if parent is not None else None
        return False

    async def list(
        self,
        session_id: str,
        *,
        scope_path: tuple[str, ...] = (),
        owner_path: tuple[str, ...] = (),
        ancestor_task_id: str | None = None,
        expected_session_generation: BackgroundSessionGeneration | None = None,
    ) -> builtins.list[BackgroundTaskSnapshot]:
        """List retained tasks visible to an agent scope."""
        async with self._lock:
            self._validate_session_generation(
                session_id,
                expected_session_generation,
            )
            return [
                record.snapshot()
                for task_id in self._session_task_ids.get(session_id, ())
                for record in [self._records[task_id]]
                if self._is_visible(
                    record,
                    scope_path,
                    owner_path,
                    ancestor_task_id,
                )
            ]

    async def get(
        self,
        session_id: str,
        task_id: str,
        *,
        scope_path: tuple[str, ...] = (),
        owner_path: tuple[str, ...] = (),
        ancestor_task_id: str | None = None,
        wait_seconds: float = 0,
        expected_session_generation: BackgroundSessionGeneration | None = None,
    ) -> BackgroundTaskSnapshot:
        """Return a visible task, optionally waiting for terminal state."""
        if wait_seconds < 0 or wait_seconds > 60:
            raise ValueError("wait_seconds must be between 0 and 60.")
        async with self._lock:
            self._validate_session_generation(
                session_id,
                expected_session_generation,
            )
            record = self._visible_record(
                session_id,
                task_id,
                scope_path,
                owner_path,
                ancestor_task_id,
            )
            completion = record.completion
            snapshot = record.snapshot()
            waiting = (
                bool(wait_seconds)
                and snapshot.status not in TERMINAL_BACKGROUND_TASK_STATUSES
                and completion is not None
            )
            if waiting:
                record.waiters += 1
        if waiting:
            assert completion is not None
            try:
                await asyncio.wait_for(completion.wait(), timeout=wait_seconds)
            except TimeoutError:
                pass
            finally:
                # Release the pin even when this caller is cancelled. A bare
                # decrement needs no lock: without a suspension point the loop
                # cannot interleave a reader, and awaiting the lock here could
                # itself be cancelled and leak the pin.
                record.waiters -= 1
            async with self._lock:
                self._validate_session_generation(
                    session_id,
                    expected_session_generation,
                )
                record = self._visible_record(
                    session_id,
                    task_id,
                    scope_path,
                    owner_path,
                    ancestor_task_id,
                )
                snapshot = record.snapshot()
        return snapshot

    async def wait_batch(
        self,
        session_id: str,
        task_ids: Sequence[str],
        *,
        scope_path: tuple[str, ...] = (),
        owner_path: tuple[str, ...] = (),
        ancestor_task_id: str | None = None,
        expected_session_generation: BackgroundSessionGeneration | None = None,
    ) -> builtins.list[BackgroundTaskSnapshot]:
        """Wait for an exact visible batch and return snapshots in input order."""
        async with self._lock:
            self._validate_session_generation(
                session_id,
                expected_session_generation,
            )
            records = [
                self._visible_record(
                    session_id,
                    task_id,
                    scope_path,
                    owner_path,
                    ancestor_task_id,
                )
                for task_id in task_ids
            ]
            completions = [
                record.completion
                for record in records
                if record.status not in TERMINAL_BACKGROUND_TASK_STATUSES
                and record.completion is not None
            ]
        if completions:
            await asyncio.shield(
                asyncio.gather(*(completion.wait() for completion in completions))
            )
        async with self._lock:
            return [record.snapshot() for record in records]

    def _visible_record(
        self,
        session_id: str,
        task_id: str,
        scope_path: tuple[str, ...],
        owner_path: tuple[str, ...],
        ancestor_task_id: str | None,
    ) -> _BackgroundTaskRecord:
        record = self._records.get(task_id)
        if (
            record is None
            or record.session_id != session_id
            or not self._is_visible(
                record,
                scope_path,
                owner_path,
                ancestor_task_id,
            )
        ):
            raise KeyError(
                f"Background task '{task_id}' is not visible in this session."
            )
        return record

    async def cancel(
        self,
        session_id: str,
        task_id: str,
        *,
        scope_path: tuple[str, ...] = (),
        owner_path: tuple[str, ...] = (),
        ancestor_task_id: str | None = None,
        expected_session_generation: BackgroundSessionGeneration | None = None,
        only_if_unfinished: bool = False,
    ) -> BackgroundTaskSnapshot:
        """Cancel a visible task and all of its descendants."""
        async with self._lock:
            self._validate_session_generation(
                session_id,
                expected_session_generation,
            )
            record = self._visible_record(
                session_id,
                task_id,
                scope_path,
                owner_path,
                ancestor_task_id,
            )
            if (
                only_if_unfinished
                and record.status in TERMINAL_BACKGROUND_TASK_STATUSES
            ):
                return record.snapshot()
            record.cancelling = True
            if record.cancel_task is None:
                record.cancel_task = asyncio.create_task(
                    self._cancel_record(record),
                    name=f"chainagents-cancel-{record.task_id}",
                )
            cancel_task = record.cancel_task
        return await await_preserving_cancellation(cancel_task)

    async def _cancel_record(
        self,
        record: _BackgroundTaskRecord,
    ) -> BackgroundTaskSnapshot:
        """Cancel and finalize a record independently of its caller."""
        async with self._lock:
            execution = record.execution
        if execution is not None and not execution.done():
            execution.cancel()
        await self._cancel_descendants(record.session_id, record.task_id)
        if execution is not None:
            await asyncio.gather(execution, return_exceptions=True)
        if record.status not in TERMINAL_BACKGROUND_TASK_STATUSES:
            cleanup_error = await self._cleanup_record(record)
            await self._finish(record, status="cancelled", error=cleanup_error)
        async with self._lock:
            return record.snapshot()

    async def _cancel_descendants(self, session_id: str, parent_task_id: str) -> None:
        async with self._lock:
            descendants: list[_BackgroundTaskRecord] = []
            descendant_ids: set[str] = set()
            parent_ids = {parent_task_id}
            while parent_ids:
                children = [
                    record
                    for record in self._records.values()
                    if record.session_id == session_id
                    and record.parent_task_id in parent_ids
                    and record.task_id not in descendant_ids
                ]
                descendants.extend(children)
                descendant_ids.update(record.task_id for record in children)
                parent_ids = {record.task_id for record in children}
            executions = [
                record.execution
                for record in descendants
                if record.execution is not None and not record.execution.done()
            ]
            for record in descendants:
                record.cancelling = True
        for execution in executions:
            execution.cancel()
        if executions:
            await asyncio.gather(*executions, return_exceptions=True)
        for record in descendants:
            if record.status not in TERMINAL_BACKGROUND_TASK_STATUSES:
                cleanup_error = await self._cleanup_record(record)
                await self._finish(record, status="cancelled", error=cleanup_error)

    async def wait_session(
        self,
        session_id: str,
    ) -> builtins.list[BackgroundTaskSnapshot]:
        """Wait until every task currently or subsequently running in a session ends."""
        while True:
            async with self._lock:
                executions = [
                    record.execution
                    for task_id in self._session_task_ids.get(session_id, ())
                    for record in [self._records[task_id]]
                    if record.execution is not None and not record.execution.done()
                ]
            if not executions:
                return await self.list(session_id)
            await asyncio.gather(*executions, return_exceptions=True)

    async def _close_session_tasks(self, session_id: str) -> None:
        """Cancel, await, and forget task records for a serialized session."""
        async with self._lock:
            records = [
                self._records[task_id]
                for task_id in self._session_task_ids.get(session_id, ())
                if task_id in self._records
            ]
            executions = [
                record.execution
                for record in records
                if record.execution is not None and not record.execution.done()
            ]
            for record in records:
                if record.status not in TERMINAL_BACKGROUND_TASK_STATUSES:
                    record.cancelling = True
        for execution in executions:
            execution.cancel()
        if executions:
            await asyncio.gather(*executions, return_exceptions=True)
        for record in records:
            if record.status not in TERMINAL_BACKGROUND_TASK_STATUSES:
                cleanup_error = await self._cleanup_record(record)
                await self._finish(record, status="cancelled", error=cleanup_error)
            else:
                await self._cleanup_record(record)
        async with self._lock:
            for record in records:
                self._records.pop(record.task_id, None)
            self._session_task_ids.pop(session_id, None)

    @asynccontextmanager
    async def closing_session(self, session_id: str) -> AsyncIterator[None]:
        """Close tasks and reject new work until dependent resources are released."""
        normalized_session = session_id.strip()
        if not normalized_session:
            raise ValueError("Background tasks require a non-empty session ID.")
        async with self._lock:
            close_lock = self._session_close_locks.setdefault(
                normalized_session,
                asyncio.Lock(),
            )
            self._session_close_users[normalized_session] = (
                self._session_close_users.get(normalized_session, 0) + 1
            )
            generation = self._session_generations.pop(normalized_session, None)
            if generation is not None:
                generation.active = False
            self._closing_sessions.add(normalized_session)
        try:
            async with close_lock:
                await self._close_session_tasks(normalized_session)
                yield
        finally:
            async with self._lock:
                users = self._session_close_users[normalized_session] - 1
                if users:
                    self._session_close_users[normalized_session] = users
                else:
                    self._session_close_users.pop(normalized_session, None)
                    self._session_close_locks.pop(normalized_session, None)
                    self._closing_sessions.discard(normalized_session)

    async def close_session(self, session_id: str) -> None:
        """Cancel, await, and forget all work owned by one conversation."""
        close_task = asyncio.create_task(
            self._close_session(session_id),
            name=f"chainagents-close-background-session-{session_id}",
        )
        await await_preserving_cancellation(close_task)

    async def _close_session(self, session_id: str) -> None:
        """Run a complete session close in its own cancellation scope."""
        async with self.closing_session(session_id):
            pass

    def subscribe(self, session_id: str) -> asyncio.Queue[BackgroundTaskSnapshot]:
        """Subscribe to terminal task snapshots for one conversation."""
        queue: asyncio.Queue[BackgroundTaskSnapshot] = asyncio.Queue(
            maxsize=SUBSCRIBER_QUEUE_MAXSIZE
        )
        self._subscribers.setdefault(session_id, set()).add(queue)
        return queue

    def unsubscribe(
        self,
        session_id: str,
        queue: asyncio.Queue[BackgroundTaskSnapshot],
    ) -> None:
        """Remove a previously registered completion subscriber."""
        subscribers = self._subscribers.get(session_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._subscribers.pop(session_id, None)

    def subscribe_activity(
        self,
        session_id: str,
    ) -> asyncio.Queue[BackgroundTaskActivity]:
        """Subscribe to ordered live and terminal task activity."""
        queue: asyncio.Queue[BackgroundTaskActivity] = asyncio.Queue(
            maxsize=SUBSCRIBER_QUEUE_MAXSIZE
        )
        self._activity_subscribers.setdefault(session_id, set()).add(queue)
        return queue

    def unsubscribe_activity(
        self,
        session_id: str,
        queue: asyncio.Queue[BackgroundTaskActivity],
    ) -> None:
        """Remove a previously registered task activity subscriber."""
        subscribers = self._activity_subscribers.get(session_id)
        if subscribers is None:
            return
        subscribers.discard(queue)
        if not subscribers:
            self._activity_subscribers.pop(session_id, None)

    async def close(self) -> None:
        """Cancel all work and prevent future spawns."""
        async with self._lock:
            self._terminal_close = True
            if self._close_task is None:
                self._closed = True
                session_ids = list(self._session_task_ids)
                self._close_task = asyncio.create_task(
                    self._close_all_sessions(session_ids),
                    name="chainagents-close-background-tasks",
                )
            close_task = self._close_task
        await await_preserving_cancellation(close_task)

    async def drain(self) -> None:
        """Cancel all work and reopen the manager for a later lifespan."""
        async with self._lock:
            if self._terminal_close:
                raise RuntimeError("The background task manager is closed.")
            if self._close_task is None:
                self._closed = True
                session_ids = list(self._session_task_ids)
                self._close_task = asyncio.create_task(
                    self._close_all_sessions(session_ids),
                    name="chainagents-drain-background-tasks",
                )
            close_task = self._close_task
        await await_preserving_cancellation(close_task)
        async with self._lock:
            if self._close_task is close_task:
                self._close_task = None
                self._closed = False

    async def _close_all_sessions(
        self,
        session_ids: builtins.list[str],
    ) -> None:
        """Close the manager's sessions and release completion subscribers."""
        try:
            results = await asyncio.gather(
                *(self.close_session(session_id) for session_id in session_ids),
                return_exceptions=True,
            )
        finally:
            self._subscribers.clear()
            self._activity_subscribers.clear()
        for result in results:
            if isinstance(result, BaseException):
                raise result
