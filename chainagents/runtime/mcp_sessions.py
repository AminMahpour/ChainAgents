"""MCP session pool: stateful transport ownership and tool discovery caching."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger("chainagents.runtime.core")

_MCP_DISCOVERY_WARNINGS: ContextVar[set[str] | None] = ContextVar(
    "mcp_discovery_warnings", default=None
)


def mcp_outage_warning(server_names: tuple[str, ...]) -> str:
    """Return a safe user-facing warning without exposing transport details."""
    names = ", ".join(server_names)
    return f"MCP server unavailable: {names}. Continuing with available tools."


class _MCPSessionOwner:
    """Enter and exit transport cancel scopes on the same long-lived task."""

    def __init__(self, context: Any) -> None:
        self._ready = asyncio.get_running_loop().create_future()
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(context))

    async def _run(self, context: Any) -> None:
        try:
            async with context as session:
                self._ready.set_result(session)
                await self._stop.wait()
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            else:
                raise

    async def session(self) -> Any:
        try:
            return await asyncio.shield(self._ready)
        except BaseException:
            await self.aclose()
            # Retrieve a startup error even when the requesting task was cancelled.
            if self._ready.done():
                self._ready.exception()
            raise

    async def aclose(self) -> None:
        self._stop.set()
        if not self._ready.done():
            self._task.cancel()
        await asyncio.shield(self._task)

    @property
    def terminal(self) -> bool:
        """Whether the transport task has exited and cannot be closed again."""
        return self._task.done()


class MCPSessionPool:
    """Own MCP client sessions, tool caches, and stateful transport teardown."""

    def __init__(self, extensions_provider: Callable[[], Any]) -> None:
        """Initialize the MCP session pool.

        Args:
            extensions_provider: Callable returning the current extensions
                configuration, resolved fresh on each use so runtime config
                reassignment (e.g. in tests) takes effect immediately.
        """
        self._extensions_provider = extensions_provider
        self._lock = asyncio.Lock()
        self._client: MultiServerMCPClient | None = None
        self._tools_cache: dict[tuple[str | None, tuple[str, ...]], list[Any]] = {}
        self._sessions: dict[tuple[str | None, str], Any] = {}
        self._session_owners: dict[tuple[str | None, str], _MCPSessionOwner] = {}
        self._inflight: dict[tuple[str | None, str], asyncio.Task[list[Any]]] = {}
        self._waiters: dict[asyncio.Task[list[Any]], int] = {}
        self._evicting_scopes: dict[str | None, asyncio.Future[None]] = {}
        self._evicting_all: asyncio.Future[None] | None = None

    @staticmethod
    async def _settle_cleanup(future: asyncio.Future[Any]) -> bool:
        """Wait for cleanup even if its caller is cancelled in the meantime."""
        interrupted = False
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                interrupted = True
            except BaseException:
                break
        return interrupted

    @property
    def client(self) -> MultiServerMCPClient | None:
        """Return the configured MCP client, or None when MCP is not configured."""
        return self._client

    @client.setter
    def client(self, value: MultiServerMCPClient | None) -> None:
        self._client = value

    def scope(
        self,
        *,
        mcp_session_id: str | None,
        thread_id: str | None = None,
    ) -> str | None:
        """Open or reuse MCP client resources for the current scope.

        Args:
            mcp_session_id: MCP session identifier.
            thread_id: Conversation thread identifier.

        Returns:
            The MCP scope result.
        """
        if not self._extensions_provider().mcp_stateful:
            return None

        candidate = str(mcp_session_id or "").strip()
        if candidate:
            return candidate

        fallback = str(thread_id or "").strip()
        return fallback or None

    async def _session_for_discovery(
        self,
        *,
        scope: str | None,
        server_name: str,
        client: MultiServerMCPClient,
    ) -> tuple[Any, _MCPSessionOwner | None]:
        """Find a live session or open one owned by this discovery task."""
        cache_key = (scope, server_name)
        session = self._sessions.get(cache_key)
        if session is not None:
            return session, None

        stale_owner = self._session_owners.get(cache_key)
        if stale_owner is not None:
            if not stale_owner.terminal:
                try:
                    await stale_owner.aclose()
                except Exception:
                    if not stale_owner.terminal:
                        raise
                    logger.exception("Failed to close terminal MCP session for %s", server_name)
            if self._session_owners.get(cache_key) is stale_owner:
                self._session_owners.pop(cache_key, None)

        owner = _MCPSessionOwner(client.session(server_name))
        # session() closes a failed or cancelled startup itself.
        session = await owner.session()
        return session, owner

    async def _discover_server(
        self,
        *,
        scope: str | None,
        server_name: str,
        client: MultiServerMCPClient,
        stateful: bool,
        tool_name_prefix: bool,
    ) -> list[Any]:
        """Load one server, committing only while this task still owns its key."""
        key = (scope, server_name)
        owner: _MCPSessionOwner | None = None
        try:
            if stateful:
                session, owner = await self._session_for_discovery(
                    scope=scope, server_name=server_name, client=client
                )
                loaded = await load_mcp_tools(
                    session,
                    callbacks=client.callbacks,
                    tool_interceptors=client.tool_interceptors,
                    server_name=server_name,
                    tool_name_prefix=tool_name_prefix,
                )
            else:
                loaded = await client.get_tools(server_name=server_name)

            tools = list(loaded)
            async with self._lock:
                # Eviction removes the task before cancelling it. A transport that
                # swallows cancellation still cannot restore the old resources.
                if self._inflight.get(key) is not asyncio.current_task():
                    raise asyncio.CancelledError
                if owner is not None:
                    self._sessions[key] = session
                    self._session_owners[key] = owner
                    owner = None
                self._tools_cache[(scope, (server_name,))] = tools
            return tools
        except BaseException as exc:
            if owner is not None and not owner.terminal:
                close_task = asyncio.create_task(owner.aclose())
                interrupted = await self._settle_cleanup(close_task)
                try:
                    close_task.result()
                except BaseException as close_error:
                    if isinstance(close_error, Exception):
                        logger.error(
                            "Failed to close MCP session for %s",
                            server_name,
                            exc_info=close_error,
                        )
                    if not owner.terminal:
                        async with self._lock:
                            if key not in self._session_owners:
                                self._session_owners[key] = owner
                if interrupted:
                    raise asyncio.CancelledError from None
            if isinstance(exc, Exception):
                logger.warning("MCP server %s is unavailable: %s", server_name, exc)
            raise

    async def tools_with_status(
        self,
        server_names: tuple[str, ...],
        *,
        thread_id: str | None = None,
        mcp_session_id: str | None = None,
    ) -> tuple[list[Any], tuple[str, ...]]:
        """Load available tools while reporting failed servers by name."""
        if not server_names or self._client is None:
            return [], ()

        tool_scope = self.scope(mcp_session_id=mcp_session_id, thread_id=thread_id)
        client = self._client
        extensions = self._extensions_provider()
        ready: dict[str, list[Any]] = {}
        pending: dict[str, asyncio.Task[list[Any]]] = {}
        while True:
            async with self._lock:
                barrier = self._evicting_all or self._evicting_scopes.get(tool_scope)
                if barrier is None:
                    for server_name in server_names:
                        if server_name in ready or server_name in pending:
                            continue
                        cache_key = (tool_scope, (server_name,))
                        cached = self._tools_cache.get(cache_key)
                        if cached is not None:
                            ready[server_name] = cached
                            continue
                        key = (tool_scope, server_name)
                        task = self._inflight.get(key)
                        if task is None or task.done():
                            task = asyncio.create_task(
                                self._discover_server(
                                    scope=tool_scope,
                                    server_name=server_name,
                                    client=client,
                                    stateful=extensions.mcp_stateful,
                                    tool_name_prefix=extensions.mcp_tool_name_prefix,
                                )
                            )
                            self._inflight[key] = task
                        pending[server_name] = task
                        self._waiters[task] = self._waiters.get(task, 0) + 1
                    break
            await asyncio.shield(barrier)

        try:
            outcomes = await asyncio.gather(
                *(asyncio.shield(task) for task in pending.values()),
                return_exceptions=True,
            )
        finally:
            abandoned: list[asyncio.Task[list[Any]]] = []
            async with self._lock:
                for server_name, task in pending.items():
                    remaining = self._waiters[task] - 1
                    if remaining:
                        self._waiters[task] = remaining
                        continue
                    self._waiters.pop(task)
                    key = (tool_scope, server_name)
                    if self._inflight.get(key) is task:
                        self._inflight.pop(key)
                    if not task.done():
                        task.cancel()
                        abandoned.append(task)
            if abandoned:
                # A cancelled caller must not return before its last transport
                # owner has had a chance to close.
                interrupted = await self._settle_cleanup(
                    asyncio.gather(*abandoned, return_exceptions=True)
                )
                if interrupted:
                    raise asyncio.CancelledError from None

        resolved: dict[str, list[Any] | BaseException] = dict(ready)
        resolved.update(zip(pending, outcomes, strict=True))
        tools: list[Any] = []
        failures: list[str] = []
        warnings = _MCP_DISCOVERY_WARNINGS.get()
        for server_name in server_names:
            outcome = resolved[server_name]
            if isinstance(outcome, Exception):
                failures.append(server_name)
                if warnings is not None:
                    warnings.add(server_name)
            elif isinstance(outcome, BaseException):
                raise outcome
            else:
                tools.extend(outcome)
        return tools, tuple(failures)

    async def evict_scope(
        self, scope: str
    ) -> list[tuple[tuple[str | None, str], _MCPSessionOwner]]:
        """Drop cached sessions and tools for a scope, returning owners to close."""
        return await self._evict_scope(scope, close_owners=False)

    async def evict_scope_and_close(self, scope: str) -> None:
        """Evict a scope and close its owners before allowing new discovery."""
        await self._evict_scope(scope, close_owners=True)

    async def _evict_scope(
        self, scope: str, *, close_owners: bool
    ) -> list[tuple[tuple[str | None, str], _MCPSessionOwner]]:
        while True:
            async with self._lock:
                existing = self._evicting_all or self._evicting_scopes.get(scope)
                if existing is None:
                    barrier = asyncio.get_running_loop().create_future()
                    self._evicting_scopes[scope] = barrier
                    tasks = [
                        task for key, task in self._inflight.items() if key[0] == scope
                    ]
                    self._inflight = {
                        key: task
                        for key, task in self._inflight.items()
                        if key[0] != scope
                    }
                    for task in tasks:
                        task.cancel()
                    owners = [
                        (key, owner)
                        for key, owner in self._session_owners.items()
                        if key[0] == scope
                    ]
                    self._sessions = {
                        key: session
                        for key, session in self._sessions.items()
                        if key[0] != scope
                    }
                    self._tools_cache = {
                        key: tools
                        for key, tools in self._tools_cache.items()
                        if key[0] != scope
                    }
                    break
            await asyncio.shield(existing)
        try:
            interrupted = False
            if tasks:
                interrupted = await self._settle_cleanup(
                    asyncio.gather(*tasks, return_exceptions=True)
                )
                async with self._lock:
                    owners.extend(
                        (key, owner)
                        for key, owner in self._session_owners.items()
                        if key[0] == scope
                        and all(existing is not owner for _, existing in owners)
                    )
            if close_owners or interrupted:
                close_task = asyncio.create_task(self.close_owner_entries(owners))
                close_interrupted = await self._settle_cleanup(close_task)
                close_task.result()
                if interrupted or close_interrupted:
                    raise asyncio.CancelledError
            return owners
        finally:
            async with self._lock:
                if self._evicting_scopes.get(scope) is barrier:
                    self._evicting_scopes.pop(scope)
                barrier.set_result(None)

    async def evict_all(
        self,
    ) -> list[tuple[tuple[str | None, str], _MCPSessionOwner]]:
        """Drop every cached session and tool entry, returning owners to close."""
        return await self._evict_all(close_owners=False)

    async def evict_all_and_close(self) -> None:
        """Evict and close every owner before allowing new discovery."""
        await self._evict_all(close_owners=True)

    async def _evict_all(
        self, *, close_owners: bool
    ) -> list[tuple[tuple[str | None, str], _MCPSessionOwner]]:
        while True:
            async with self._lock:
                existing = self._evicting_all
                if existing is None and self._evicting_scopes:
                    existing = next(iter(self._evicting_scopes.values()))
                if existing is None:
                    barrier = asyncio.get_running_loop().create_future()
                    self._evicting_all = barrier
                    tasks = list(self._inflight.values())
                    self._inflight.clear()
                    for task in tasks:
                        task.cancel()
                    owners = list(self._session_owners.items())
                    self._sessions.clear()
                    self._tools_cache.clear()
                    break
            await asyncio.shield(existing)
        try:
            interrupted = False
            if tasks:
                interrupted = await self._settle_cleanup(
                    asyncio.gather(*tasks, return_exceptions=True)
                )
                async with self._lock:
                    owners.extend(
                        (key, owner)
                        for key, owner in self._session_owners.items()
                        if all(existing is not owner for _, existing in owners)
                    )
            if close_owners or interrupted:
                close_task = asyncio.create_task(self.close_owner_entries(owners))
                close_interrupted = await self._settle_cleanup(close_task)
                close_task.result()
                if interrupted or close_interrupted:
                    raise asyncio.CancelledError
            return owners
        finally:
            async with self._lock:
                if self._evicting_all is barrier:
                    self._evicting_all = None
                barrier.set_result(None)

    async def close_owner_entries(
        self, owners: list[tuple[tuple[str | None, str], _MCPSessionOwner]]
    ) -> None:
        """Close owners independently and retain failures for later retry."""
        results = await asyncio.gather(
            *(owner.aclose() for _, owner in owners), return_exceptions=True
        )
        async with self._lock:
            for (key, owner), result in zip(owners, results, strict=True):
                should_release = not isinstance(result, BaseException) or getattr(
                    owner, "terminal", False
                )
                if should_release and self._session_owners.get(key) is owner:
                    self._session_owners.pop(key, None)
        for result in results:
            if isinstance(result, BaseException):
                raise result
