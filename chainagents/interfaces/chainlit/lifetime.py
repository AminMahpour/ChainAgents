"""Keep conversation resources briefly available across Chainlit sessions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any
from weakref import WeakSet

logger = logging.getLogger(__name__)


@dataclass
class _Scope:
    runtime: Any
    scope_id: str
    retain_idle: bool = True
    owners: set[str] = field(default_factory=set)
    idle_since: float | None = None
    expiry_task: asyncio.Task[None] | None = None
    close_task: asyncio.Task[None] | None = None
    close_attempts: int = 0


class ConversationScopeLeaseManager:
    """Retain idle runtime scopes while any Chainlit tab may resume them."""

    def __init__(
        self,
        *,
        idle_seconds: float = 600,
        max_idle: int = 4,
        retry_seconds: float = 1,
        max_close_attempts: int = 3,
    ) -> None:
        self.idle_seconds = idle_seconds
        self.max_idle = max_idle
        self.retry_seconds = retry_seconds
        self.max_close_attempts = max_close_attempts
        self._lock = asyncio.Lock()
        self._scopes: dict[tuple[int, str], _Scope] = {}
        self._owner_scopes: dict[str, tuple[int, str]] = {}
        self._bound_runtimes: WeakSet[Any] = WeakSet()
        self._retired_runtimes: WeakSet[Any] = WeakSet()
        self._bound_unhashable_runtimes: list[Any] = []
        self._retired_unhashable_runtimes: list[Any] = []

    async def lease(
        self,
        *,
        runtime: Any,
        owner_id: str,
        scope_id: str,
        retain_idle: bool = True,
    ) -> None:
        """Attach one Chainlit session to the effective conversation scope."""
        if not owner_id or not scope_id:
            raise ValueError("A Chainlit owner and conversation scope are required")
        key = (id(runtime), scope_id)
        while True:
            async with self._lock:
                if self._runtime_known(
                    self._retired_runtimes,
                    self._retired_unhashable_runtimes,
                    runtime,
                ):
                    raise RuntimeError("Cannot lease a closed agent runtime")
                self._bind_runtime_locked(runtime)
                previous_key = self._owner_scopes.get(owner_id)
                if previous_key is not None and previous_key != key:
                    self._release_locked(owner_id)

                scope = self._scopes.get(key)
                if scope is not None and scope.close_task is not None:
                    closing = scope.close_task
                else:
                    if scope is None:
                        scope = _Scope(
                            runtime=runtime,
                            scope_id=scope_id,
                            retain_idle=retain_idle,
                        )
                        self._scopes[key] = scope
                    scope.retain_idle = scope.retain_idle or retain_idle
                    self._cancel_expiry(scope)
                    scope.owners.add(owner_id)
                    scope.idle_since = None
                    scope.close_attempts = 0
                    self._owner_scopes[owner_id] = key
                    self._enforce_idle_limit_locked()
                    return
            await asyncio.shield(closing)

    def _bind_runtime_locked(self, runtime: Any) -> None:
        if self._runtime_known(
            self._bound_runtimes, self._bound_unhashable_runtimes, runtime
        ):
            return
        exit_stack = getattr(runtime, "_exit_stack", None)
        push_callback = getattr(exit_stack, "push_async_callback", None)
        if callable(push_callback):
            push_callback(self.forget_runtime, runtime)
            self._remember_runtime(
                self._bound_runtimes, self._bound_unhashable_runtimes, runtime
            )

    @staticmethod
    def _runtime_known(weak: WeakSet[Any], fallback: list[Any], runtime: Any) -> bool:
        try:
            return runtime in weak
        except TypeError:
            return any(item is runtime for item in fallback)

    @staticmethod
    def _remember_runtime(weak: WeakSet[Any], fallback: list[Any], runtime: Any) -> None:
        try:
            weak.add(runtime)
        except TypeError:
            fallback.append(runtime)

    async def forget_runtime(self, runtime: Any) -> None:
        """Drop timers and leases before the runtime closes its own MCP pool."""
        runtime_id = id(runtime)
        async with self._lock:
            self._remember_runtime(
                self._retired_runtimes,
                self._retired_unhashable_runtimes,
                runtime,
            )
            try:
                self._bound_runtimes.discard(runtime)
            except TypeError:
                self._bound_unhashable_runtimes = [
                    item for item in self._bound_unhashable_runtimes if item is not runtime
                ]
            pending: list[asyncio.Task[None]] = []
            for key, scope in list(self._scopes.items()):
                if key[0] != runtime_id:
                    continue
                self._cancel_expiry(scope)
                for owner_id in scope.owners:
                    if self._owner_scopes.get(owner_id) == key:
                        self._owner_scopes.pop(owner_id)
                self._scopes.pop(key)
                if scope.close_task is not None:
                    pending.append(scope.close_task)
        if pending:
            await asyncio.gather(*(asyncio.shield(task) for task in pending))

    async def release(
        self, owner_id: str, *, only_if: Callable[[], bool] | None = None
    ) -> None:
        """Release one tab while retaining its owner-free scope for a short time."""
        async with self._lock:
            if only_if is not None and not only_if():
                return
            key = self._owner_scopes.get(owner_id)
            self._release_locked(owner_id)
            self._enforce_idle_limit_locked()
            scope = self._scopes.get(key) if key is not None else None
            closing = (
                scope.close_task
                if scope is not None and not scope.retain_idle
                else None
            )
        if closing is not None:
            await asyncio.shield(closing)

    def _release_locked(self, owner_id: str) -> None:
        key = self._owner_scopes.pop(owner_id, None)
        if key is None:
            return
        scope = self._scopes.get(key)
        if scope is None:
            return
        scope.owners.discard(owner_id)
        if scope.owners or scope.close_task is not None:
            return
        if not scope.retain_idle:
            self._start_close_locked(key, scope)
            return
        scope.idle_since = monotonic()
        scope.expiry_task = asyncio.create_task(
            self._expire_after_delay(key, scope, self.idle_seconds)
        )

    async def _expire_after_delay(
        self, key: tuple[int, str], scope: _Scope, delay: float
    ) -> None:
        try:
            await asyncio.sleep(delay)
            async with self._lock:
                if self._scopes.get(key) is scope and not scope.owners:
                    self._start_close_locked(key, scope)
        except asyncio.CancelledError:
            return

    def _cancel_expiry(self, scope: _Scope) -> None:
        task = scope.expiry_task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        scope.expiry_task = None

    def _enforce_idle_limit_locked(self) -> None:
        idle = sorted(
            (
                (key, scope)
                for key, scope in self._scopes.items()
                if not scope.owners and scope.close_task is None
            ),
            key=lambda item: item[1].idle_since or 0,
        )
        for key, scope in idle[: max(0, len(idle) - self.max_idle)]:
            self._start_close_locked(key, scope)

    def _start_close_locked(self, key: tuple[int, str], scope: _Scope) -> None:
        if scope.owners or scope.close_task is not None:
            return
        self._cancel_expiry(scope)
        scope.close_task = asyncio.create_task(self._close_scope(key, scope))

    async def _close_scope(self, key: tuple[int, str], scope: _Scope) -> None:
        failed = False
        try:
            await scope.runtime.close_conversation(
                thread_id=scope.scope_id,
                mcp_session_id=scope.scope_id,
            )
        except asyncio.CancelledError:
            failed = True
            raise
        except Exception:
            failed = True
            logger.exception("Failed to close idle conversation %s", scope.scope_id)
        finally:
            async with self._lock:
                if self._scopes.get(key) is scope:
                    scope.close_task = None
                    if failed:
                        scope.close_attempts += 1
                        if scope.close_attempts < self.max_close_attempts:
                            delay = self.retry_seconds * 2 ** (scope.close_attempts - 1)
                        else:
                            # Keep retrying at the idle cadence until either
                            # cleanup succeeds or runtime shutdown owns it.
                            delay = max(self.idle_seconds, self.retry_seconds)
                        scope.idle_since = monotonic()
                        scope.expiry_task = asyncio.create_task(
                            self._expire_after_delay(key, scope, delay)
                        )
                    else:
                        self._scopes.pop(key)

    async def aclose(self) -> None:
        """Close tracked scopes during application shutdown."""
        async with self._lock:
            self._owner_scopes.clear()
            for key, scope in self._scopes.items():
                scope.owners.clear()
                self._start_close_locked(key, scope)
            pending = [
                scope.close_task
                for scope in self._scopes.values()
                if scope.close_task is not None
            ]
        if pending:
            await asyncio.gather(*pending)
