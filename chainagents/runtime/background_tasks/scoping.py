"""Runnable wrappers that scope background task ownership to one invocation."""

from __future__ import annotations

import contextvars
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from langchain_core.runnables import Runnable, RunnableConfig

from chainagents.runtime.artifacts import (
    ArtifactSessionHandle,
    LargeToolResultArtifactRegistry,
)
from chainagents.runtime.background_tasks.context import (
    _CURRENT_BACKGROUND_INVOCATION_PATH,
    _CURRENT_BACKGROUND_SESSION_GENERATION,
    BackgroundSessionGeneration,
    current_background_invocation_path,
)

if TYPE_CHECKING:
    from chainagents.runtime.background_tasks.manager import BackgroundTaskManager


class _BackgroundInvocationScopedRunnable(Runnable[Any, Any]):
    """Give each configured subagent invocation a distinct task-tree scope."""

    def __init__(self, runnable: object) -> None:
        self.runnable = runnable

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        token = _CURRENT_BACKGROUND_INVOCATION_PATH.set(
            (*current_background_invocation_path(), uuid.uuid4().hex)
        )
        try:
            return self.runnable.invoke(input, config, **kwargs)  # type: ignore[attr-defined]
        finally:
            _CURRENT_BACKGROUND_INVOCATION_PATH.reset(token)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        token = _CURRENT_BACKGROUND_INVOCATION_PATH.set(
            (*current_background_invocation_path(), uuid.uuid4().hex)
        )
        try:
            return await self.runnable.ainvoke(  # type: ignore[attr-defined]
                input,
                config,
                **kwargs,
            )
        finally:
            _CURRENT_BACKGROUND_INVOCATION_PATH.reset(token)

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        token = _CURRENT_BACKGROUND_INVOCATION_PATH.set(
            (*current_background_invocation_path(), uuid.uuid4().hex)
        )
        try:
            async for chunk in self.runnable.astream(  # type: ignore[attr-defined]
                input,
                config,
                **kwargs,
            ):
                yield chunk
        finally:
            _CURRENT_BACKGROUND_INVOCATION_PATH.reset(token)


def scope_background_task_invocation(runnable: object) -> Runnable[Any, Any]:
    """Wrap a configured subagent with invocation-local task ownership."""
    return _BackgroundInvocationScopedRunnable(runnable)


class _BackgroundSessionScopedRunnable(Runnable[Any, Any]):
    """Attach a session lifecycle capability to one exported graph run."""

    def __init__(
        self,
        runnable: object,
        manager: "BackgroundTaskManager",
        on_session_open: Callable[[str], None] | None = None,
        artifact_registry: LargeToolResultArtifactRegistry | None = None,
        fixed_session_id: str | None = None,
        run_config_transform: Callable[[RunnableConfig | None], RunnableConfig] | None = None,
    ) -> None:
        self.runnable = runnable
        self.manager = manager
        self.on_session_open = on_session_open
        self.artifact_registry = artifact_registry
        self.fixed_session_id = fixed_session_id
        self.run_config_transform = run_config_transform

    def _prepare_config(self, config: RunnableConfig | None) -> RunnableConfig | None:
        if self.run_config_transform is None:
            return config
        return self.run_config_transform(config)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runnable, name)

    @property
    def InputType(self) -> Any:  # noqa: N802
        return self.runnable.InputType  # type: ignore[attr-defined]

    @property
    def OutputType(self) -> Any:  # noqa: N802
        return self.runnable.OutputType  # type: ignore[attr-defined]

    @property
    def config_specs(self) -> list[Any]:
        return list(self.runnable.config_specs)  # type: ignore[attr-defined]

    def get_input_schema(self, config: RunnableConfig | None = None) -> Any:
        return self.runnable.get_input_schema(config)  # type: ignore[attr-defined]

    def get_output_schema(self, config: RunnableConfig | None = None) -> Any:
        return self.runnable.get_output_schema(config)  # type: ignore[attr-defined]

    def get_graph(self, config: RunnableConfig | None = None) -> Any:
        return self.runnable.get_graph(config)  # type: ignore[attr-defined]

    def _set_generation(
        self,
        config: RunnableConfig | None,
    ) -> tuple[
        contextvars.Token[BackgroundSessionGeneration | None] | None,
        contextvars.Token[ArtifactSessionHandle | None] | None,
    ]:
        configurable = (config or {}).get("configurable", {})
        session_id = str(
            self.fixed_session_id or configurable.get("thread_id") or ""
        ).strip()
        if not session_id:
            return None, None
        generation = self._session_generation(session_id)
        artifact_token = None
        if self.artifact_registry is not None:
            handle = self.artifact_registry.open_session(session_id)
            artifact_token = self.artifact_registry.activate(handle)
        return (
            _CURRENT_BACKGROUND_SESSION_GENERATION.set(generation),
            artifact_token,
        )

    def _session_generation(
        self,
        session_id: str,
    ) -> BackgroundSessionGeneration:
        if self.on_session_open is not None:
            self.on_session_open(session_id)
        return self.manager.session_generation(session_id)

    @staticmethod
    def _reset_generation(
        tokens: tuple[
            contextvars.Token[BackgroundSessionGeneration | None] | None,
            contextvars.Token[ArtifactSessionHandle | None] | None,
        ],
        artifact_registry: LargeToolResultArtifactRegistry | None,
    ) -> None:
        token, artifact_token = tokens
        if artifact_token is not None and artifact_registry is not None:
            artifact_registry.reset(artifact_token)
        if token is not None:
            _CURRENT_BACKGROUND_SESSION_GENERATION.reset(token)

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        config = self._prepare_config(config)
        token = self._set_generation(config)
        try:
            return self.runnable.invoke(input, config, **kwargs)  # type: ignore[attr-defined]
        finally:
            self._reset_generation(token, self.artifact_registry)

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        config = self._prepare_config(config)
        token = self._set_generation(config)
        try:
            return await self.runnable.ainvoke(  # type: ignore[attr-defined]
                input,
                config,
                **kwargs,
            )
        finally:
            self._reset_generation(token, self.artifact_registry)

    def stream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        config = self._prepare_config(config)
        token = self._set_generation(config)
        try:
            yield from self.runnable.stream(  # type: ignore[attr-defined]
                input,
                config,
                **kwargs,
            )
        finally:
            self._reset_generation(token, self.artifact_registry)

    async def astream(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        config = self._prepare_config(config)
        token = self._set_generation(config)
        try:
            async for chunk in self.runnable.astream(  # type: ignore[attr-defined]
                input,
                config,
                **kwargs,
            ):
                yield chunk
        finally:
            self._reset_generation(token, self.artifact_registry)

    @overload
    def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]: ...

    @overload
    def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v3"],
        **kwargs: Any,
    ) -> Awaitable[Any]: ...

    def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2", "v3"] = "v2",
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any] | Awaitable[Any]:
        config = self._prepare_config(config)
        configurable = (config or {}).get("configurable", {})
        session_id = str(
            self.fixed_session_id or configurable.get("thread_id") or ""
        ).strip()
        generation = self._session_generation(session_id) if session_id else None
        artifact_handle = (
            self.artifact_registry.open_session(session_id)
            if session_id and self.artifact_registry is not None
            else None
        )
        if version == "v3":
            result = self.runnable.astream_events(  # type: ignore[attr-defined]
                input,
                config,
                version=version,
                **kwargs,
            )

            async def await_events() -> Any:
                token = _CURRENT_BACKGROUND_SESSION_GENERATION.set(generation)
                artifact_token = (
                    self.artifact_registry.activate(artifact_handle)
                    if artifact_handle is not None
                    and self.artifact_registry is not None
                    else None
                )
                try:
                    stream = await cast(Awaitable[Any], result)
                    graph_iterator = getattr(stream, "_graph_aiter", None)
                    if graph_iterator is not None:
                        stream._graph_aiter = (  # noqa: SLF001
                            _BackgroundSessionScopedAsyncIterator(
                                graph_iterator,
                                generation,
                                self.artifact_registry,
                                artifact_handle,
                            )
                        )
                    return stream
                finally:
                    if (
                        artifact_token is not None
                        and self.artifact_registry is not None
                    ):
                        self.artifact_registry.reset(artifact_token)
                    _CURRENT_BACKGROUND_SESSION_GENERATION.reset(token)

            return await_events()

        result = self.runnable.astream_events(  # type: ignore[attr-defined]
            input,
            config,
            version=version,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_names=exclude_names,
            exclude_types=exclude_types,
            exclude_tags=exclude_tags,
            **kwargs,
        )

        async def iterate_events() -> AsyncIterator[Any]:
            token = _CURRENT_BACKGROUND_SESSION_GENERATION.set(generation)
            artifact_token = (
                self.artifact_registry.activate(artifact_handle)
                if artifact_handle is not None and self.artifact_registry is not None
                else None
            )
            try:
                async for event in cast(AsyncIterator[Any], result):
                    yield event
            finally:
                if artifact_token is not None and self.artifact_registry is not None:
                    self.artifact_registry.reset(artifact_token)
                _CURRENT_BACKGROUND_SESSION_GENERATION.reset(token)

        return iterate_events()


class _BackgroundSessionScopedAsyncIterator:
    """Activate a session capability for each lazy v3 graph pull."""

    def __init__(
        self,
        iterator: AsyncIterator[Any],
        generation: BackgroundSessionGeneration | None,
        artifact_registry: LargeToolResultArtifactRegistry | None = None,
        artifact_handle: ArtifactSessionHandle | None = None,
    ) -> None:
        self.iterator = iterator
        self.generation = generation
        self.artifact_registry = artifact_registry
        self.artifact_handle = artifact_handle

    def __aiter__(self) -> "_BackgroundSessionScopedAsyncIterator":
        return self

    async def __anext__(self) -> Any:
        token = _CURRENT_BACKGROUND_SESSION_GENERATION.set(self.generation)
        artifact_token = (
            self.artifact_registry.activate(self.artifact_handle)
            if self.artifact_registry is not None and self.artifact_handle is not None
            else None
        )
        try:
            return await self.iterator.__anext__()
        finally:
            if artifact_token is not None and self.artifact_registry is not None:
                self.artifact_registry.reset(artifact_token)
            _CURRENT_BACKGROUND_SESSION_GENERATION.reset(token)

    async def aclose(self) -> None:
        close = getattr(self.iterator, "aclose", None)
        if close is None:
            return
        token = _CURRENT_BACKGROUND_SESSION_GENERATION.set(self.generation)
        artifact_token = (
            self.artifact_registry.activate(self.artifact_handle)
            if self.artifact_registry is not None and self.artifact_handle is not None
            else None
        )
        try:
            await close()
        finally:
            if artifact_token is not None and self.artifact_registry is not None:
                self.artifact_registry.reset(artifact_token)
            _CURRENT_BACKGROUND_SESSION_GENERATION.reset(token)


def scope_background_session_invocation(
    runnable: object,
    manager: "BackgroundTaskManager",
    *,
    on_session_open: Callable[[str], None] | None = None,
    artifact_registry: LargeToolResultArtifactRegistry | None = None,
    fixed_session_id: str | None = None,
    run_config_transform: Callable[[RunnableConfig | None], RunnableConfig] | None = None,
) -> Runnable[Any, Any]:
    """Wrap an exported graph with invocation-scoped session invalidation."""
    return _BackgroundSessionScopedRunnable(
        runnable,
        manager,
        on_session_open=on_session_open,
        artifact_registry=artifact_registry,
        fixed_session_id=fixed_session_id,
        run_config_transform=run_config_transform,
    )
