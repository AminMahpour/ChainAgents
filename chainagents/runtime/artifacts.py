"""Session-scoped storage and cleanup for offloaded large tool results."""

from __future__ import annotations

import asyncio
import contextvars
import posixpath
import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from deepagents.backends import BackendProtocol, CompositeBackend
from deepagents.backends.protocol import (
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    DeleteResult,
    EditResult,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)


_HIDDEN_ARTIFACT_ERROR = "Internal artifact storage is not accessible."


async def _await_preserving_cancellation[T](task: asyncio.Task[T]) -> T:
    """Finish one backend mutation before propagating caller cancellation."""
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


@dataclass(frozen=True)
class ArtifactSessionHandle:
    """One non-reusable generation of artifact ownership."""

    session_id: str
    token: str


class LargeToolResultArtifactRegistry:
    """Own physical large-result files until their session generation closes."""

    def __init__(self) -> None:
        self._guard = threading.RLock()
        self._current_context: contextvars.ContextVar[
            ArtifactSessionHandle | None
        ] = contextvars.ContextVar(
            f"chainagents_artifact_session_{uuid.uuid4().hex}",
            default=None,
        )
        self._unscoped = ArtifactSessionHandle("", f"unscoped-{uuid.uuid4().hex}")
        self._current: dict[str, ArtifactSessionHandle] = {}
        self._states: dict[ArtifactSessionHandle, str] = {self._unscoped: "open"}
        self._paths: dict[ArtifactSessionHandle, dict[str, BackendProtocol]] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_lock_users: dict[str, int] = {}
        self._close_lock = asyncio.Lock()
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
        return self._current_context.set(handle)

    def reset(self, token: contextvars.Token[ArtifactSessionHandle | None]) -> None:
        """Restore the preceding ownership context."""
        self._current_context.reset(token)

    def current_handle(self) -> ArtifactSessionHandle:
        """Return the active handle or the registry-owned unscoped fallback."""
        return self._current_context.get() or self._unscoped

    def owns(self, handle: ArtifactSessionHandle, path: str) -> bool:
        """Return whether one handle owns a registered cleanup target."""
        with self._guard:
            return path in self._paths.get(handle, {})

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
                self._states.setdefault(handle, "closing")
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
                self._states.setdefault(handle, "closing")
                self._paths.setdefault(handle, {})[path] = backend
            raise

    async def close_session(self, session_id: str) -> None:
        """Delete all artifacts for the current generation of one session."""
        normalized = session_id.strip()
        if not normalized:
            return
        with self._guard:
            lock = self._session_locks.setdefault(normalized, asyncio.Lock())
            self._session_lock_users[normalized] = (
                self._session_lock_users.get(normalized, 0) + 1
            )
        try:
            async with lock:
                with self._guard:
                    handle = self._current.get(normalized)
                    if handle is None:
                        return
                    self._states[handle] = "closing"
                await self._drain_handle(handle)
                with self._guard:
                    if not self._paths.get(handle):
                        if self._current.get(normalized) is handle:
                            self._current.pop(normalized, None)
                        self._states.pop(handle, None)
        finally:
            with self._guard:
                users = self._session_lock_users.get(normalized, 1) - 1
                if users:
                    self._session_lock_users[normalized] = users
                else:
                    self._session_lock_users.pop(normalized, None)
                    if normalized not in self._current:
                        self._session_locks.pop(normalized, None)

    async def close(self) -> None:
        """Make the registry terminal and drain every known generation."""
        await self._close(reopen=False)

    async def drain(self) -> None:
        """Drain every generation and reopen the registry for a later lifespan."""
        await self._close(reopen=True)

    async def _close(self, *, reopen: bool) -> None:
        async with self._close_lock:
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
                if reopen:
                    self._current.clear()
                    self._states.clear()
                    self._session_locks.clear()
                    self._session_lock_users.clear()
                    self._unscoped = ArtifactSessionHandle(
                        "",
                        f"unscoped-{uuid.uuid4().hex}",
                    )
                    self._states[self._unscoped] = "open"
                    self._terminal = False
                else:
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
        artifact_roots = {self._normalize_path(self.artifacts_root)}
        workspace_backend = backend.routes.get("/workspace/")
        workspace_root = getattr(workspace_backend, "cwd", None)
        if workspace_root is not None:
            try:
                relative_artifacts = Path(self.artifacts_root).relative_to(
                    Path(workspace_root)
                )
            except ValueError:
                pass
            else:
                relative_root = relative_artifacts.as_posix()
                artifact_roots.add(self._normalize_path(relative_root))
                artifact_roots.add(self._normalize_path(f"/workspace/{relative_root}"))
        self._artifact_roots = frozenset(artifact_roots)
        self._logical_roots = tuple(
            sorted(
                (
                    self._normalize_path(f"{root}/large_tool_results")
                    for root in self._artifact_roots
                ),
                key=len,
                reverse=True,
            )
        )
        self._hidden_roots = frozenset(
            self._normalize_path(f"{root}/session_tool_results")
            for root in self._artifact_roots
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    @staticmethod
    def _normalize_path(path: str) -> str:
        return posixpath.normpath(path.replace("\\", "/"))

    def _is_hidden_path(self, path: str | None) -> bool:
        if path is None:
            return False
        normalized = self._normalize_path(path)
        return any(
            normalized == root or normalized.startswith(f"{root}/")
            for root in self._hidden_roots
        )

    def _is_logical_path(self, path: str) -> bool:
        normalized = self._normalize_path(path)
        return any(
            normalized == root or normalized.startswith(f"{root}/")
            for root in self._logical_roots
        )

    def _search_includes_artifacts(self, path: str | None) -> bool:
        if path is None:
            return True
        normalized = self._normalize_path(path)
        if normalized == "/":
            return True
        if normalized == ".":
            return any(not root.startswith("/") for root in self._artifact_roots)
        prefix = f"{normalized.rstrip('/')}/"
        return any(
            normalized == root or root.startswith(prefix)
            for root in self._artifact_roots
        )

    def _restore_active_path(self, path: str) -> str:
        handle = self.registry.current_handle()
        if not self.registry.owns(handle, self._physical_root(handle)):
            return path
        normalized = self._normalize_path(path)
        trailing_slash = path.endswith("/")
        for artifact_root in self._artifact_roots:
            hidden_root = self._normalize_path(
                f"{artifact_root}/session_tool_results"
            )
            active_root = f"{hidden_root}/{handle.token}"
            logical_root = self._normalize_path(
                f"{artifact_root}/large_tool_results"
            )
            if normalized in (hidden_root, active_root):
                restored = logical_root
            elif normalized.startswith(f"{active_root}/"):
                restored = f"{logical_root}{normalized.removeprefix(active_root)}"
            else:
                continue
            if trailing_slash:
                return f"{restored}/"
            return restored
        return path

    def _visible_items(
        self,
        items: list[dict[str, Any]],
        *,
        hide_legacy: bool = False,
    ) -> list[dict[str, Any]]:
        visible: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            original = item["path"]
            restored = self._restore_active_path(original)
            if (
                hide_legacy
                and self._normalize_path(restored) == self._normalize_path(original)
                and self._is_logical_path(original)
            ):
                continue
            if self._is_hidden_path(restored) or restored in seen:
                continue
            seen.add(restored)
            visible.append({**item, "path": restored})
        return visible

    def _map(self, path: str | None) -> tuple[str | None, ArtifactSessionHandle | None]:
        if path is None:
            return path, None
        normalized = self._normalize_path(path)
        logical_root = next(
            (
                root
                for root in self._logical_roots
                if normalized == root or normalized.startswith(f"{root}/")
            ),
            None,
        )
        if logical_root is None:
            return path, None
        handle = self.registry.current_handle()
        relative = normalized.removeprefix(logical_root).lstrip("/")
        physical = self._physical_root(handle)
        if relative:
            physical = f"{physical}/{relative}"
        return physical, handle

    def _physical_root(self, handle: ArtifactSessionHandle) -> str:
        return (
            f"{self.artifacts_root.rstrip('/')}/session_tool_results/{handle.token}"
        )

    @staticmethod
    def _restore_path(path: str, mapped: str | None, logical: str | None) -> str:
        if mapped is None or logical is None or not path.startswith(mapped):
            return path
        return f"{logical}{path.removeprefix(mapped)}"

    def ls(self, path: str) -> LsResult:
        if self._is_hidden_path(path):
            return LsResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, _ = self._map(path)
        parent_search = mapped == path and self._search_includes_artifacts(path)
        result = self.backend.ls(mapped or path)
        if result.entries is not None:
            result.entries = cast(
                list[FileInfo],
                self._visible_items(
                    [
                        {
                            **entry,
                            "path": self._restore_path(entry["path"], mapped, path),
                        }
                        for entry in result.entries
                    ],
                    hide_legacy=parent_search,
                ),
            )
        return result

    async def als(self, path: str) -> LsResult:
        if self._is_hidden_path(path):
            return LsResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, _ = self._map(path)
        parent_search = mapped == path and self._search_includes_artifacts(path)
        result = await self.backend.als(mapped or path)
        if result.entries is not None:
            result.entries = cast(
                list[FileInfo],
                self._visible_items(
                    [
                        {
                            **entry,
                            "path": self._restore_path(entry["path"], mapped, path),
                        }
                        for entry in result.entries
                    ],
                    hide_legacy=parent_search,
                ),
            )
        return result

    def read(self, file_path: str, offset: int = 0, limit: int = 2000):
        if self._is_hidden_path(file_path):
            return ReadResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, _ = self._map(file_path)
        return self.backend.read(mapped or file_path, offset=offset, limit=limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000):
        if self._is_hidden_path(file_path):
            return ReadResult(error=_HIDDEN_ARTIFACT_ERROR)
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
        if self._is_hidden_path(path):
            return GrepResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, _ = self._map(path)
        parent_search = mapped == path and self._search_includes_artifacts(path)
        deferred_limit = parent_search and max_count is not None and max_count >= 0
        result = self.backend.grep(
            pattern,
            mapped,
            glob,
            max_count=None if deferred_limit else max_count,
        )
        if result.matches is not None:
            result.matches = cast(
                list[GrepMatch],
                self._visible_items(
                    [
                        {
                            **match,
                            "path": self._restore_path(match["path"], mapped, path),
                        }
                        for match in result.matches
                    ],
                    hide_legacy=parent_search,
                ),
            )
            if (
                deferred_limit
                and max_count is not None
                and len(result.matches) > max_count
            ):
                result.matches = result.matches[:max_count]
                result.truncated = True
        return result

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        if self._is_hidden_path(path):
            return GrepResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, _ = self._map(path)
        parent_search = mapped == path and self._search_includes_artifacts(path)
        deferred_limit = parent_search and max_count is not None and max_count >= 0
        result = await self.backend.agrep(
            pattern,
            mapped,
            glob,
            max_count=None if deferred_limit else max_count,
        )
        if result.matches is not None:
            result.matches = cast(
                list[GrepMatch],
                self._visible_items(
                    [
                        {
                            **match,
                            "path": self._restore_path(match["path"], mapped, path),
                        }
                        for match in result.matches
                    ],
                    hide_legacy=parent_search,
                ),
            )
            if (
                deferred_limit
                and max_count is not None
                and len(result.matches) > max_count
            ):
                result.matches = result.matches[:max_count]
                result.truncated = True
        return result

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        if self._is_hidden_path(path):
            return GlobResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, _ = self._map(path)
        parent_search = mapped == path and self._search_includes_artifacts(path)
        result = self.backend.glob(pattern, mapped)
        if result.matches is not None:
            result.matches = cast(
                list[FileInfo],
                self._visible_items(
                    [
                        {
                            **entry,
                            "path": self._restore_path(entry["path"], mapped, path),
                        }
                        for entry in result.matches
                    ],
                    hide_legacy=parent_search,
                ),
            )
        return result

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        if self._is_hidden_path(path):
            return GlobResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, _ = self._map(path)
        parent_search = mapped == path and self._search_includes_artifacts(path)
        result = await self.backend.aglob(pattern, mapped)
        if result.matches is not None:
            result.matches = cast(
                list[FileInfo],
                self._visible_items(
                    [
                        {
                            **entry,
                            "path": self._restore_path(entry["path"], mapped, path),
                        }
                        for entry in result.matches
                    ],
                    hide_legacy=parent_search,
                ),
            )
        return result

    def write(self, file_path: str, content: str):
        if self._is_hidden_path(file_path):
            return WriteResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, handle = self._map(file_path)
        result = self.backend.write(mapped or file_path, content)
        if handle is not None:
            self.registry.register(handle, self._physical_root(handle), self.backend)
        if result.path is not None and mapped != file_path:
            result.path = file_path
        return result

    async def awrite(self, file_path: str, content: str):
        if self._is_hidden_path(file_path):
            return WriteResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, handle = self._map(file_path)

        async def complete_write():
            result = await self.backend.awrite(mapped or file_path, content)
            if handle is not None:
                await self.registry.aregister(
                    handle,
                    self._physical_root(handle),
                    self.backend,
                )
            if result.path is not None and mapped != file_path:
                result.path = file_path
            return result

        return await _await_preserving_cancellation(
            asyncio.create_task(complete_write())
        )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ):
        if self._is_hidden_path(file_path):
            return EditResult(error=_HIDDEN_ARTIFACT_ERROR)
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
        if self._is_hidden_path(file_path):
            return EditResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, _ = self._map(file_path)
        result = await self.backend.aedit(
            mapped or file_path, old_string, new_string, replace_all=replace_all
        )
        if result.path is not None and mapped != file_path:
            result.path = file_path
        return result

    def delete(self, file_path: str):
        if self._is_hidden_path(file_path):
            return DeleteResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, _ = self._map(file_path)
        result = self.backend.delete(mapped or file_path)
        if result.path is not None and mapped != file_path:
            result.path = file_path
        return result

    async def adelete(self, file_path: str):
        if self._is_hidden_path(file_path):
            return DeleteResult(error=_HIDDEN_ARTIFACT_ERROR)
        mapped, _ = self._map(file_path)
        result = await self.backend.adelete(mapped or file_path)
        if result.path is not None and mapped != file_path:
            result.path = file_path
        return result

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse | None] = [None] * len(files)
        allowed: list[tuple[int, str, str, bytes, ArtifactSessionHandle | None]] = []
        for index, (path, content) in enumerate(files):
            if self._is_hidden_path(path):
                responses[index] = FileUploadResponse(
                    path=path,
                    error=_HIDDEN_ARTIFACT_ERROR,
                )
                continue
            mapped, handle = self._map(path)
            allowed.append((index, path, mapped or path, content, handle))
        results = self.backend.upload_files(
            [(mapped, content) for _, _, mapped, content, _ in allowed]
        )
        for (index, path, _mapped, _content, handle), result in zip(
            allowed,
            results,
            strict=True,
        ):
            if handle is not None:
                self.registry.register(
                    handle,
                    self._physical_root(handle),
                    self.backend,
                )
            responses[index] = replace(result, path=path)
        return cast(list[FileUploadResponse], responses)

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        responses: list[FileUploadResponse | None] = [None] * len(files)
        allowed: list[tuple[int, str, str, bytes, ArtifactSessionHandle | None]] = []
        for index, (path, content) in enumerate(files):
            if self._is_hidden_path(path):
                responses[index] = FileUploadResponse(
                    path=path,
                    error=_HIDDEN_ARTIFACT_ERROR,
                )
                continue
            mapped, handle = self._map(path)
            allowed.append((index, path, mapped or path, content, handle))

        async def complete_upload() -> list[FileUploadResponse]:
            results = await self.backend.aupload_files(
                [(mapped, content) for _, _, mapped, content, _ in allowed]
            )
            for (index, path, _mapped, _content, handle), result in zip(
                allowed,
                results,
                strict=True,
            ):
                if handle is not None:
                    await self.registry.aregister(
                        handle,
                        self._physical_root(handle),
                        self.backend,
                    )
                responses[index] = replace(result, path=path)
            return cast(list[FileUploadResponse], responses)

        return await _await_preserving_cancellation(
            asyncio.create_task(complete_upload())
        )

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse | None] = [None] * len(paths)
        allowed: list[tuple[int, str, str]] = []
        for index, path in enumerate(paths):
            if self._is_hidden_path(path):
                responses[index] = FileDownloadResponse(
                    path=path,
                    error=_HIDDEN_ARTIFACT_ERROR,
                )
                continue
            mapped, _ = self._map(path)
            allowed.append((index, path, mapped or path))
        results = self.backend.download_files([mapped for _, _, mapped in allowed])
        for (index, path, _mapped), result in zip(allowed, results, strict=True):
            responses[index] = replace(result, path=path)
        return cast(list[FileDownloadResponse], responses)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses: list[FileDownloadResponse | None] = [None] * len(paths)
        allowed: list[tuple[int, str, str]] = []
        for index, path in enumerate(paths):
            if self._is_hidden_path(path):
                responses[index] = FileDownloadResponse(
                    path=path,
                    error=_HIDDEN_ARTIFACT_ERROR,
                )
                continue
            mapped, _ = self._map(path)
            allowed.append((index, path, mapped or path))
        results = await self.backend.adownload_files(
            [mapped for _, _, mapped in allowed]
        )
        for (index, path, _mapped), result in zip(allowed, results, strict=True):
            responses[index] = replace(result, path=path)
        return cast(list[FileDownloadResponse], responses)
