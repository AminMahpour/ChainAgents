"""Contextvar-scoped identity for the currently executing background task."""

from __future__ import annotations

import contextvars


class BackgroundSessionGeneration:
    """Weakly tracked capability for one open background-task session."""

    __slots__ = ("__weakref__", "active", "session_id")

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.active = True


_CURRENT_BACKGROUND_TASK_ID: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar(
        "chainagents_background_task_id",
        default=None,
    )
)
_CURRENT_BACKGROUND_SESSION_ID: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar(
        "chainagents_background_session_id",
        default=None,
    )
)
_CURRENT_BACKGROUND_INVOCATION_PATH: contextvars.ContextVar[tuple[str, ...]] = (
    contextvars.ContextVar(
        "chainagents_background_invocation_path",
        default=(),
    )
)
_CURRENT_BACKGROUND_SESSION_GENERATION: contextvars.ContextVar[
    BackgroundSessionGeneration | None
] = contextvars.ContextVar(
    "chainagents_background_session_generation",
    default=None,
)


def current_background_task_id() -> str | None:
    """Return the task ID of the currently executing background subagent."""
    return _CURRENT_BACKGROUND_TASK_ID.get()


def current_background_session_id() -> str | None:
    """Return the conversation that owns the current background invocation."""
    return _CURRENT_BACKGROUND_SESSION_ID.get()


def current_background_invocation_path() -> tuple[str, ...]:
    """Return the ownership path of the current configured-agent invocation."""
    return _CURRENT_BACKGROUND_INVOCATION_PATH.get()


def current_background_session_generation() -> BackgroundSessionGeneration | None:
    """Return the lifecycle capability attached to the current graph run."""
    return _CURRENT_BACKGROUND_SESSION_GENERATION.get()
