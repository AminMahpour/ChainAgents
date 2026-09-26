"""Bounded queue helpers for background task subscribers."""

from __future__ import annotations

import asyncio

from chainagents.runtime.background_tasks.models import BackgroundTaskActivity

# Bound per-subscriber buffers so a stalled consumer cannot grow memory without limit.
SUBSCRIBER_QUEUE_MAXSIZE = 1000


def _put_evicting_oldest[T](queue: asyncio.Queue[T], item: T) -> None:
    """Enqueue a must-deliver item, discarding the oldest entry when full."""
    while queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    queue.put_nowait(item)


def _put_activity_evicting_live(
    queue: asyncio.Queue[BackgroundTaskActivity],
    item: BackgroundTaskActivity,
) -> None:
    """Enqueue a terminal activity, preferring to discard best-effort live events.

    Terminal snapshots close the rendered state of a task in the interfaces, so a
    full queue drops the oldest live event instead. Only when every queued entry
    is terminal does the oldest of those give way.
    """
    while queue.full():
        retained: list[BackgroundTaskActivity] = []
        dropped = False
        while True:
            try:
                queued = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not dropped and queued.snapshot is None:
                dropped = True
                continue
            retained.append(queued)
        if not dropped and retained:
            retained.pop(0)
        for queued in retained:
            queue.put_nowait(queued)
        if not retained and not dropped:
            break
    queue.put_nowait(item)


async def await_preserving_cancellation[T](task: asyncio.Task[T]) -> T:
    """Delay caller cancellation until a lifecycle task has finished."""
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc
    result = task.result()
    if cancellation is not None:
        raise cancellation
    return result
