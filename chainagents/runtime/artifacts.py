"""Session ownership and cleanup for offloaded large tool results."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterable

from deepagents.backends import BackendProtocol

_UNSCOPED_OWNER = "\0unscoped"


class LargeToolResultArtifactRegistry:
    """Track offloaded tool-result files until their owning session closes."""

    def __init__(self) -> None:
        self._guard = threading.RLock()
        self._paths: dict[str, dict[str, BackendProtocol]] = {}
        self._closing_sessions: set[str] = set()
        self._closed_sessions: set[str] = set()
        self._session_locks: dict[str, asyncio.Lock] = {}

    def open_session(self, session_id: str) -> None:
        """Allow a newly created runtime graph to own artifacts for a session."""
        normalized = session_id.strip()
        if not normalized:
            return
        with self._guard:
            if normalized in self._closing_sessions:
                return
            self._closed_sessions.discard(normalized)

    def register(
        self,
        session_id: str,
        path: str,
        backend: BackendProtocol,
    ) -> None:
        """Register an offload from synchronous middleware execution."""
        normalized = session_id.strip() or _UNSCOPED_OWNER
        with self._guard:
            if normalized not in self._closed_sessions:
                self._paths.setdefault(normalized, {})[path] = backend
                return
        self._delete_sync(path, backend)

    async def aregister(
        self,
        session_id: str,
        path: str,
        backend: BackendProtocol,
    ) -> None:
        """Register an offload from asynchronous middleware execution."""
        normalized = session_id.strip() or _UNSCOPED_OWNER
        with self._guard:
            if normalized not in self._closed_sessions:
                self._paths.setdefault(normalized, {})[path] = backend
                return
        await self._delete_async(path, backend)

    async def close_session(self, session_id: str) -> None:
        """Delete all known large tool results owned by one session."""
        normalized = session_id.strip()
        if not normalized:
            return
        with self._guard:
            lock = self._session_locks.setdefault(normalized, asyncio.Lock())
        async with lock:
            with self._guard:
                self._closing_sessions.add(normalized)
                self._closed_sessions.add(normalized)
                artifacts = self._paths.pop(normalized, {})
            try:
                await self._delete_many(artifacts.items())
            finally:
                with self._guard:
                    self._closing_sessions.discard(normalized)
        with self._guard:
            self._session_locks.pop(normalized, None)

    async def close(self) -> None:
        """Delete every artifact still owned by this registry."""
        with self._guard:
            session_ids = tuple(self._paths)
        if session_ids:
            await asyncio.gather(
                *(self.close_session(session_id) for session_id in session_ids)
            )

    @staticmethod
    def _missing(error: str | None) -> bool:
        return bool(error and "not found" in error.lower())

    @classmethod
    def _delete_sync(cls, path: str, backend: BackendProtocol) -> None:
        result = backend.delete(path)
        if result.error and not cls._missing(result.error):
            raise RuntimeError(
                f"Could not delete offloaded tool result '{path}': {result.error}"
            )

    @classmethod
    async def _delete_async(cls, path: str, backend: BackendProtocol) -> None:
        result = await backend.adelete(path)
        if result.error and not cls._missing(result.error):
            raise RuntimeError(
                f"Could not delete offloaded tool result '{path}': {result.error}"
            )

    @classmethod
    async def _delete_many(
        cls,
        artifacts: Iterable[tuple[str, BackendProtocol]],
    ) -> None:
        results = await asyncio.gather(
            *(cls._delete_async(path, backend) for path, backend in artifacts),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise ExceptionGroup("Large tool-result cleanup failed.", errors)
