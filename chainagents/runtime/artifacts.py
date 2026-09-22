"""Session-scoped storage and cleanup for offloaded large tool results."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import uuid
from dataclasses import dataclass, replace
from typing import Any

from deepagents.backends import BackendProtocol, CompositeBackend
from deepagents.backends.protocol import (
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
)


@dataclass(frozen=True)
class ArtifactSessionHandle:
    """One non-reusable generation of artifact ownership."""

    session_id: str
    token: str


_CURRENT_ARTIFACT_SESSION: contextvars.ContextVar[ArtifactSessionHandle | None] = (
    contextvars.ContextVar("chainagents_artifact_session", default=None)
)


class LargeToolResultArtifactRegistry:
    """Own physical large-result files until their session generation closes."""

    def __init__(self) -> None:
        self._guard = threading.RLock()
        self._unscoped = ArtifactSessionHandle("", f"unscoped-{uuid.uuid4().hex}")
        self._current: dict[str, ArtifactSessionHandle] = {}
        self._states: dict[ArtifactSessionHandle, str] = {self._unscoped: "open"}
        self._paths: dict[ArtifactSessionHandle, dict[str, BackendProtocol]] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._terminal = False

    def open_session(self, session_id: str) -> ArtifactSessionHandle:
        """Return the open generation for a session, creating it when needed."""
        normalized = session_id.strip()
        if not normalized:
            return self._unscoped
        with self._guard:
            if self._terminal:
                raise RuntimeError("Large tool-result artifact registry is closed.")
            current = self._current.get(normalized)
            if current is not None:
                state = self._states[current]
                if state == "open":
                    return current
                if state == "closing":
                    raise RuntimeError(f"Artifact session '{normalized}' is closing.")
            handle = ArtifactSessionHandle(normalized, uuid.uuid4().hex)
            self._current[normalized] = handle
            self._states[handle] = "open"
            return handle

    def activate(
        self, handle: ArtifactSessionHandle
    ) -> contextvars.Token[ArtifactSessionHandle | None]:
        """Activate one ownership generation in the current execution context."""
        return _CURRENT_ARTIFACT_SESSION.set(handle)

    @staticmethod
    def reset(token: contextvars.Token[ArtifactSessionHandle | None]) -> None:
        """Restore the preceding ownership context."""
        _CURRENT_ARTIFACT_SESSION.reset(token)

    def current_handle(self) -> ArtifactSessionHandle:
        """Return the active handle or the registry-owned unscoped fallback."""
        return _CURRENT_ARTIFACT_SESSION.get() or self._unscoped

    def register(
        self, handle: ArtifactSessionHandle, path: str, backend: BackendProtocol
    ) -> None:
        """Register a successful synchronous write or delete it when closed."""
        with self._guard:
            if not self._terminal and self._states.get(handle) == "open":
                self._paths.setdefault(handle, {})[path] = backend
                return
        try:
            self._delete_sync(path, backend)
        except BaseException:
            with self._guard:
                self._paths.setdefault(handle, {})[path] = backend
            raise

    async def aregister(
        self, handle: ArtifactSessionHandle, path: str, backend: BackendProtocol
    ) -> None:
        """Register a successful asynchronous write or delete it when closed."""
        with self._guard:
            if not self._terminal and self._states.get(handle) == "open":
                self._paths.setdefault(handle, {})[path] = backend
                return
        try:
            await self._delete_async(path, backend)
        except BaseException:
            with self._guard:
                self._paths.setdefault(handle, {})[path] = backend
            raise

    async def close_session(self, session_id: str) -> None:
        """Delete all artifacts for the current generation of one session."""
        normalized = session_id.strip()
        if not normalized:
            return
        with self._guard:
            lock = self._session_locks.setdefault(normalized, asyncio.Lock())
        async with lock:
            with self._guard:
                handle = self._current.get(normalized)
                if handle is None:
                    return
                self._states[handle] = "closing"
            await self._drain_handle(handle)
            with self._guard:
                if not self._paths.get(handle):
                    self._states[handle] = "closed"

    async def close(self) -> None:
        """Make the registry terminal and drain every known generation."""
        with self._guard:
            self._terminal = True
            handles = tuple(self._states)
            for handle in handles:
                self._states[handle] = "closing"
        results = await asyncio.gather(
            *(self._drain_handle(handle) for handle in handles),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise ExceptionGroup("Large tool-result cleanup failed.", errors)
        with self._guard:
            for handle in handles:
                if not self._paths.get(handle):
                    self._states[handle] = "closed"

    async def _drain_handle(self, handle: ArtifactSessionHandle) -> None:
        with self._guard:
            artifacts = tuple(self._paths.get(handle, {}).items())
        results = await asyncio.gather(
            *(self._delete_async(path, backend) for path, backend in artifacts),
            return_exceptions=True,
        )
        errors: list[Exception] = []
        with self._guard:
            owned = self._paths.get(handle)
            for (path, _backend), result in zip(artifacts, results, strict=True):
                if isinstance(result, Exception):
                    errors.append(result)
                elif owned is not None:
                    owned.pop(path, None)
            if owned == {}:
                self._paths.pop(handle, None)
        if errors:
            raise ExceptionGroup("Large tool-result cleanup failed.", errors)

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


class ArtifactTrackingBackend(CompositeBackend):
    """Namespace large-result files while preserving their logical paths."""

    def __init__(
        self, backend: CompositeBackend, registry: LargeToolResultArtifactRegistry
    ) -> None:
        super().__init__(
            default=backend.default,
            routes=backend.routes,
            artifacts_root=backend.artifacts_root,
        )
        self.backend = backend
        self.registry = registry
        self.artifacts_root = backend.artifacts_root.rstrip("/") or "/"
        self._logical_root = f"{self.artifacts_root.rstrip('/')}/large_tool_results"
        self._logical_prefix = f"{self._logical_root}/"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def _map(self, path: str | None) -> tuple[str | None, ArtifactSessionHandle | None]:
        if path is None or (
            path != self._logical_root and not path.startswith(self._logical_prefix)
        ):
            return path, None
        handle = self.registry.current_handle()
        relative = path.removeprefix(self._logical_root).lstrip("/")
        physical = (
            f"{self.artifacts_root.rstrip('/')}/session_tool_results/{handle.token}"
        )
        if relative:
            physical = f"{physical}/{relative}"
        return physical, handle

    @staticmethod
    def _restore_path(path: str, mapped: str | None, logical: str | None) -> str:
        if mapped is None or logical is None or not path.startswith(mapped):
            return path
        return f"{logical}{path.removeprefix(mapped)}"

    def ls(self, path: str) -> LsResult:
        mapped, _ = self._map(path)
        result = self.backend.ls(mapped or path)
        if result.entries is not None:
            result.entries = [
                {**entry, "path": self._restore_path(entry["path"], mapped, path)}
                for entry in result.entries
            ]
        return result

    async def als(self, path: str) -> LsResult:
        mapped, _ = self._map(path)
        result = await self.backend.als(mapped or path)
        if result.entries is not None:
            result.entries = [
                {**entry, "path": self._restore_path(entry["path"], mapped, path)}
                for entry in result.entries
            ]
        return result

    def read(self, file_path: str, offset: int = 0, limit: int = 2000):
        mapped, _ = self._map(file_path)
        return self.backend.read(mapped or file_path, offset=offset, limit=limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000):
        mapped, _ = self._map(file_path)
        return await self.backend.aread(mapped or file_path, offset=offset, limit=limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        mapped, _ = self._map(path)
        result = self.backend.grep(pattern, mapped, glob, max_count=max_count)
        if result.matches is not None:
            result.matches = [
                {**match, "path": self._restore_path(match["path"], mapped, path)}
                for match in result.matches
            ]
        return result

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        mapped, _ = self._map(path)
        result = await self.backend.agrep(pattern, mapped, glob, max_count=max_count)
        if result.matches is not None:
            result.matches = [
                {**match, "path": self._restore_path(match["path"], mapped, path)}
                for match in result.matches
            ]
        return result

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        mapped, _ = self._map(path)
        result = self.backend.glob(pattern, mapped)
        if result.matches is not None:
            result.matches = [
                {**entry, "path": self._restore_path(entry["path"], mapped, path)}
                for entry in result.matches
            ]
        return result

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        mapped, _ = self._map(path)
        result = await self.backend.aglob(pattern, mapped)
        if result.matches is not None:
            result.matches = [
                {**entry, "path": self._restore_path(entry["path"], mapped, path)}
                for entry in result.matches
            ]
        return result

    def write(self, file_path: str, content: str):
        mapped, handle = self._map(file_path)
        result = self.backend.write(mapped or file_path, content)
        if handle is not None and result.error is None:
            self.registry.register(handle, mapped or file_path, self.backend)
        if result.path is not None and mapped != file_path:
            result.path = file_path
        return result

    async def awrite(self, file_path: str, content: str):
        mapped, handle = self._map(file_path)

        async def complete_write():
            result = await self.backend.awrite(mapped or file_path, content)
            if handle is not None and result.error is None:
                await self.registry.aregister(handle, mapped or file_path, self.backend)
            if result.path is not None and mapped != file_path:
                result.path = file_path
            return result

        task = asyncio.create_task(complete_write())
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

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        mapped, _ = self._map(file_path)
        result = self.backend.edit(
            mapped or file_path, old_string, new_string, replace_all=replace_all
        )
        if result.path is not None and mapped != file_path:
            result.path = file_path
        return result

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        mapped, _ = self._map(file_path)
        result = await self.backend.aedit(
            mapped or file_path, old_string, new_string, replace_all=replace_all
        )
        if result.path is not None and mapped != file_path:
            result.path = file_path
        return result

    def delete(self, file_path: str):
        mapped, _ = self._map(file_path)
        result = self.backend.delete(mapped or file_path)
        if result.path is not None and mapped != file_path:
            result.path = file_path
        return result

    async def adelete(self, file_path: str):
        mapped, _ = self._map(file_path)
        result = await self.backend.adelete(mapped or file_path)
        if result.path is not None and mapped != file_path:
            result.path = file_path
        return result

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        mapped = [(self._map(path)[0] or path, content) for path, content in files]
        results = self.backend.upload_files(mapped)
        return [
            replace(result, path=files[index][0])
            for index, result in enumerate(results)
        ]

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        mapped = [(self._map(path)[0] or path, content) for path, content in files]
        results = await self.backend.aupload_files(mapped)
        return [
            replace(result, path=files[index][0])
            for index, result in enumerate(results)
        ]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        mapped = [self._map(path)[0] or path for path in paths]
        results = self.backend.download_files(mapped)
        return [
            replace(result, path=paths[index]) for index, result in enumerate(results)
        ]

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        mapped = [self._map(path)[0] or path for path in paths]
        results = await self.backend.adownload_files(mapped)
        return [
            replace(result, path=paths[index]) for index, result in enumerate(results)
        ]
