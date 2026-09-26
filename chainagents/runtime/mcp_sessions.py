"""MCP session pool: stateful transport ownership and tool discovery caching."""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import Any, Callable

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
        self.discovery_failed = False

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

    async def _stateful_session(
        self,
        *,
        server_name: str,
        thread_id: str | None,
        mcp_session_id: str | None,
    ) -> Any:
        """Return the cached MCP session for a Chainlit session.

        Args:
            server_name: The server name value.
            thread_id: Conversation thread identifier.
            mcp_session_id: MCP session identifier.

        Returns:
            The cached MCP session for a Chainlit session.

        Raises:
            RuntimeError: If the runtime is not in a usable state.
        """
        scope = self.scope(mcp_session_id=mcp_session_id, thread_id=thread_id)
        cache_key = (scope, server_name)
        session = self._sessions.get(cache_key)
        if session is not None:
            return session

        stale_owner = self._session_owners.get(cache_key)
        if stale_owner is not None:
            if not stale_owner.terminal:
                try:
                    await stale_owner.aclose()
                except Exception:
                    if not stale_owner.terminal:
                        raise
                    logger.exception("Failed to close terminal MCP session for %s", server_name)
            self._session_owners.pop(cache_key, None)

        if self._client is None:
            raise RuntimeError("MCP client is not initialized.")

        owner = _MCPSessionOwner(self._client.session(server_name))
        session = await owner.session()
        self._session_owners[cache_key] = owner
        self._sessions[cache_key] = session
        return session

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
        async with self._lock:
            tools: list[Any] = []
            failures: list[str] = []
            for server_name in server_names:
                cache_key = (tool_scope, (server_name,))
                cached = self._tools_cache.get(cache_key)
                if cached is not None:
                    tools.extend(cached)
                    continue
                session_key = (tool_scope, server_name)
                had_session = session_key in self._sessions
                extensions = self._extensions_provider()
                try:
                    if extensions.mcp_stateful:
                        session = await self._stateful_session(
                            server_name=server_name,
                            thread_id=thread_id,
                            mcp_session_id=mcp_session_id,
                        )
                        loaded = await load_mcp_tools(
                                session,
                                callbacks=self._client.callbacks,
                                tool_interceptors=self._client.tool_interceptors,
                                server_name=server_name,
                                tool_name_prefix=extensions.mcp_tool_name_prefix,
                            )
                    else:
                        loaded = await self._client.get_tools(server_name=server_name)
                except BaseException as exc:
                    if not had_session:
                        self._sessions.pop(session_key, None)
                        owner = self._session_owners.get(session_key)
                        if owner is not None:
                            try:
                                await owner.aclose()
                            except Exception:
                                logger.exception("Failed to close MCP session for %s", server_name)
                            else:
                                self._session_owners.pop(session_key, None)
                            if owner.terminal:
                                self._session_owners.pop(session_key, None)
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    if not isinstance(exc, Exception):
                        raise
                    logger.warning("MCP server %s is unavailable: %s", server_name, exc)
                    failures.append(server_name)
                    self.discovery_failed = True
                    warnings = _MCP_DISCOVERY_WARNINGS.get()
                    if warnings is not None:
                        warnings.add(server_name)
                    continue
                self._tools_cache[cache_key] = list(loaded)
                tools.extend(loaded)
            return tools, tuple(failures)

    async def evict_scope(
        self, scope: str
    ) -> list[tuple[tuple[str | None, str], _MCPSessionOwner]]:
        """Drop cached sessions and tools for a scope, returning owners to close."""
        async with self._lock:
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
            return owners

    async def evict_all(
        self,
    ) -> list[tuple[tuple[str | None, str], _MCPSessionOwner]]:
        """Drop every cached session and tool entry, returning owners to close."""
        async with self._lock:
            owners = list(self._session_owners.items())
            self._sessions.clear()
            self._tools_cache.clear()
            return owners

    async def close_owner_entries(
        self, owners: list[tuple[tuple[str | None, str], _MCPSessionOwner]]
    ) -> None:
        """Close owners independently and retain failures for later retry."""
        results = await asyncio.gather(
            *(owner.aclose() for _, owner in owners), return_exceptions=True
        )
        async with self._lock:
            for (key, owner), result in zip(owners, results, strict=True):
                if not isinstance(result, BaseException) or getattr(owner, "terminal", False):
                    if self._session_owners.get(key) is owner:
                        self._session_owners.pop(key, None)
        for result in results:
            if isinstance(result, BaseException):
                raise result
