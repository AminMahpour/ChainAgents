"""Stateful agent, MCP, persistence, and RAG resource lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from weakref import WeakValueDictionary

from deepagents.backends import StoreBackend
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import (
    SSEConnection,
    StdioConnection,
    StreamableHttpConnection,
    WebsocketConnection,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore

import chainagents.runtime.artifacts as runtime_artifacts
import chainagents.runtime.background_tasks as runtime_background_tasks
import chainagents.runtime.commands as runtime_commands
import chainagents.runtime.constants as runtime_constants
import chainagents.runtime.graph as runtime_graph
import chainagents.runtime.mcp_sessions as runtime_mcp_sessions
import chainagents.runtime.middleware as runtime_middleware
import chainagents.runtime.models as runtime_models
import chainagents.runtime.messaging as runtime_messaging
import chainagents.runtime.rag_ops as runtime_rag_ops
import chainagents.runtime.tracing as runtime_tracing
from chainagents.rag.runtime import (
    RagStatus,
    RagUploadResult,
    UploadedRagFile,
    WorkspaceDocsRAG,
)
from chainagents.runtime.config import RuntimeConfig
from chainagents.runtime.constants import SYSTEM_PROMPT, ReasoningLevel
from chainagents.runtime.mcp_sessions import (
    MCPSessionPool,
    _MCPSessionOwner as _MCPSessionOwner,
    mcp_outage_warning as mcp_outage_warning,
)
from chainagents.runtime.reflection import (
    ReflectionProposal,
    append_reflection_lesson,
    canonical_reflection_lesson,
)
from chainagents.runtime.types import (
    AgentCacheKey,
    ChainlitCommandConfig,
    ModelDefaults,
    SubagentConfig,
)

logger = logging.getLogger("chainagents.runtime.core")


async def agent_with_mcp_status(
    runtime: Any, reasoning_level: ReasoningLevel, **kwargs: Any
) -> tuple[Any, tuple[str, ...]]:
    """Resolve an agent and warnings, including runtimes without status support."""
    method = getattr(runtime, "get_agent_with_status", None)
    if callable(method):
        return await method(reasoning_level, **kwargs)
    return await runtime.get_agent(reasoning_level, **kwargs), ()


async def _gather_or_cancel(*operations: Awaitable[Any]) -> list[Any]:
    """Stop sibling discovery and await cleanup if any branch fails."""
    tasks = [asyncio.ensure_future(operation) for operation in operations]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        cleanup = asyncio.gather(*tasks, return_exceptions=True)
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        cleanup.result()
        raise


class AgentRuntime:
    """Own configured agents, MCP sessions, persistence handles, and RAG state."""

    _instance: AgentRuntime | None = None
    _instance_lock = asyncio.Lock()

    def __init__(self, config: RuntimeConfig, *, project_root: Path | None = None) -> None:
        """Initialize the agent runtime instance.

        Args:
            config: Configuration object used by the operation.
            project_root: Project root used to resolve local paths.
        """
        from chainagents.turns.controller import ConversationInputController

        self.config = config
        self.langsmith_tracing = runtime_tracing.build_langsmith_tracing(config.langsmith)
        self.project_root = project_root or runtime_constants.PROJECT_ROOT
        self._exit_stack = AsyncExitStack()
        self._agent_lock = asyncio.Lock()
        self._agents: dict[AgentCacheKey, object] = {}
        self._agent_builds: dict[
            AgentCacheKey, asyncio.Future[tuple[Any, tuple[str, ...]]]
        ] = {}
        self._closing_agent_threads: dict[str, asyncio.Future[None]] = {}
        self._closing_agent_scopes: dict[str, asyncio.Future[None]] = {}
        self._closing_all_agents: asyncio.Future[None] | None = None
        self._turn_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._mcp_pool = MCPSessionPool(lambda: self.config.extensions)
        self._checkpointer: AsyncPostgresSaver | MemorySaver | None = None
        self._store: AsyncPostgresStore | InMemoryStore | None = None
        self._rag_service: WorkspaceDocsRAG | None = None
        self.large_tool_result_artifacts = (
            runtime_artifacts.LargeToolResultArtifactRegistry()
        )
        self._exit_stack.push_async_callback(self.close_all_mcp_sessions)
        self.background_tasks = runtime_background_tasks.BackgroundTaskManager(
            config.extensions.background_subagents,
            artifact_registry=self.large_tool_result_artifacts,
        )
        self._exit_stack.push_async_callback(self.background_tasks.close)
        broker_config = config.extensions.messaging
        if config.extensions.user_input.enabled and not broker_config.enabled:
            broker_config = replace(broker_config, enabled=True)
        self.message_broker = runtime_messaging.MessageBroker(broker_config)
        self.user_input = ConversationInputController(
            config.extensions.user_input, self.message_broker, runner_serialized=True
        )
        self._chainlit_commands, self._chainlit_command_notes = runtime_commands.build_chainlit_command_catalog(
            config.extensions,
            project_root=self.project_root,
        )

    async def save_reflection(self, proposal: ReflectionProposal) -> None:
        """Persist a confirmed lesson to configured memory and verify the full write.

        The runtime lock serializes reflection read/modify/write operations. Storage
        errors propagate as RuntimeError; cancellation remains retryable by callers.
        """
        config = self.config.extensions.agent_reflection
        if self.config.agent_state != "stateful" or not config.enabled:
            raise ValueError("Reflection saving requires an enabled stateful runtime.")
        if proposal.memory_file != config.memory_file:
            raise ValueError(
                "Reflection memory file does not match runtime configuration."
            )
        if (
            not proposal.lesson.strip()
            or len(proposal.lesson.strip()) > config.max_lesson_chars
        ):
            raise ValueError("Reflection lesson exceeds runtime validation limits.")
        lesson = canonical_reflection_lesson(proposal.lesson)
        # CompositeBackend strips the /memories prefix, retaining the leading /.
        path = config.memory_file.removeprefix("/memories")
        namespace = (self.config.extensions.agent_memory_namespace,)
        async with self._agent_lock:
            try:
                backend = StoreBackend(namespace=lambda _: namespace, store=self.store)

                async def read_content(*, allow_missing: bool = False) -> str:
                    parts: list[str] = []
                    offset = 0
                    while True:
                        result = await backend.aread(path, offset=offset)
                        if result.error:
                            # Do not mistake permission errors or invalid data for a
                            # missing file and overwrite existing memory.
                            if (
                                allow_missing
                                and offset == 0
                                and result.error == f"File '{path}' not found"
                                and await self.store.aget(namespace, path) is None
                            ):
                                return ""
                            raise RuntimeError("Reflection memory could not be read.")
                        data = result.file_data
                        if (
                            data is None
                            or not isinstance(data.get("content"), str)
                            or data.get("encoding", "utf-8") != "utf-8"
                        ):
                            raise RuntimeError("Reflection memory is not valid text.")
                        parts.append(data["content"])
                        if result.next_offset is None:
                            break
                        if result.next_offset <= offset:
                            raise RuntimeError(
                                "Reflection memory pagination did not advance."
                            )
                        offset = result.next_offset
                    # StoreBackend's text reader normalizes CR/CRLF. Recover the
                    # original string through the supported store API so adding a
                    # lesson does not rewrite existing memory's line endings.
                    item = await self.store.aget(namespace, path)
                    raw = item.value.get("content") if item is not None else None
                    if isinstance(raw, list) and all(
                        isinstance(line, str) for line in raw
                    ):
                        raw = "\n".join(raw)
                    if not isinstance(raw, str):
                        raise RuntimeError("Reflection memory is not valid text.")
                    normalized = (
                        raw.replace("\r\n", "\n").replace("\r", "\n")
                        if raw.strip()
                        else raw
                    )
                    if normalized != "".join(parts):
                        raise RuntimeError(
                            "Reflection memory changed while being read."
                        )
                    return raw

                existing = await read_content(allow_missing=True)
                updated = append_reflection_lesson(existing, lesson)
                if updated != existing:
                    result = await backend.awrite(path, updated)
                    if result.error:
                        raise RuntimeError("Reflection memory could not be written.")
                if await read_content() != updated:
                    raise RuntimeError(
                        "Reflection memory readback did not match the saved content."
                    )
            except Exception as exc:
                raise RuntimeError("Reflection persistence failed.") from exc

    @classmethod
    async def get(cls) -> AgentRuntime:
        """Get the agent runtime.

        Returns:
            The requested value.
        """
        async with cls._instance_lock:
            if cls._instance is None:
                instance = cls(RuntimeConfig.from_env())
                try:
                    await instance._initialize()
                except BaseException:
                    await instance.close()
                    raise
                cls._instance = instance
            return cls._instance

    @classmethod
    async def create(
        cls,
        config: RuntimeConfig | None = None,
        *,
        project_root: Path | None = None,
    ) -> AgentRuntime:
        """Create the agent runtime.

        Args:
            config: Configuration object used by the operation.
            project_root: Project root used to resolve local paths.

        Returns:
            The created the agent runtime.
        """
        instance = cls(config or RuntimeConfig.from_env(), project_root=project_root)
        try:
            await instance._initialize()
        except BaseException:
            await instance.close()
            raise
        return instance

    @classmethod
    def current(cls) -> AgentRuntime | None:
        """Return the current.

        Returns:
            The current.
        """
        return cls._instance

    def turn_lock(self, thread_id: str) -> asyncio.Lock:
        """Serialize main agent turns from all interfaces in one conversation."""
        return self._turn_locks.setdefault(thread_id, asyncio.Lock())

    async def conversation_busy(
        self, thread_id: str, *, include_paused_queue: bool = True
    ) -> bool:
        """Report input or background work, optionally excluding a paused queue."""
        if self.user_input.busy(
            thread_id, include_paused_queue=include_paused_queue
        ):
            return True
        tasks = await self.background_tasks.list(thread_id)
        return self.user_input.busy(
            thread_id, include_paused_queue=include_paused_queue
        ) or any(
            task.status not in runtime_background_tasks.TERMINAL_BACKGROUND_TASK_STATUSES
            for task in tasks
        )

    async def wait_conversation_idle(
        self, thread_id: str, *, include_paused_queue: bool = True
    ) -> None:
        """Wait for input and background work, optionally excluding a paused queue."""
        while True:
            await self.user_input.wait_drained(
                thread_id, include_paused_queue=include_paused_queue
            )
            await self.background_tasks.wait_session(thread_id)
            if not await self.conversation_busy(
                thread_id, include_paused_queue=include_paused_queue
            ):
                return

    @property
    def checkpointer(self) -> AsyncPostgresSaver | MemorySaver:
        """Return the initialized LangGraph checkpointer.

        Returns:
            The initialized LangGraph checkpointer.

        Raises:
            RuntimeError: If the runtime is not in a usable state.
        """
        if self._checkpointer is None:
            raise RuntimeError("Checkpointer is not initialized.")
        return self._checkpointer

    @property
    def store(self) -> AsyncPostgresStore | InMemoryStore:
        """Store the agent runtime.

        Returns:
            The stored value.

        Raises:
            RuntimeError: If the runtime is not in a usable state.
        """
        if self._store is None:
            raise RuntimeError("Store is not initialized.")
        return self._store

    @property
    def persistence_enabled(self) -> bool:
        """Return whether durable persistence is configured.

        Returns:
            Whether durable persistence is configured.
        """
        return (
            self.config.agent_state == "stateful"
            and self.config.persistence_mode == "postgres"
        )

    @property
    def rag_enabled(self) -> bool:
        """Return whether the RAG service is available.

        Returns:
            Whether the RAG service is available.
        """
        return runtime_rag_ops.rag_enabled(self)

    @property
    def chainlit_commands(self) -> tuple[ChainlitCommandConfig, ...]:
        """Return configured native Chainlit commands.

        Returns:
            Configured native Chainlit commands.
        """
        return self._chainlit_commands

    @property
    def chainlit_command_notes(self) -> tuple[str, ...]:
        """Return notes explaining configured Chainlit commands.

        Returns:
            Notes explaining configured Chainlit commands.
        """
        return self._chainlit_command_notes

    @property
    def rag_status(self) -> RagStatus:
        """Return the current RAG service status.

        Returns:
            The current RAG service status.
        """
        return runtime_rag_ops.rag_status(self)

    async def _initialize(self) -> None:
        """Initialize persistence, RAG, MCP clients, and configured agents."""
        if self.config.extensions.mcp_servers:
            # Parsed from user config as generic dicts; each entry's shape is
            # validated by MultiServerMCPClient itself at construction time.
            mcp_servers = cast(
                "dict[str, StdioConnection | SSEConnection | StreamableHttpConnection "
                "| WebsocketConnection]",
                self.config.extensions.mcp_servers,
            )
            self._mcp_pool.client = MultiServerMCPClient(
                mcp_servers,
                tool_name_prefix=self.config.extensions.mcp_tool_name_prefix,
            )

        if self.config.agent_state == "stateless":
            self._store = None
            self._checkpointer = None
        elif not self.config.database_url:
            self._store = InMemoryStore()
            self._checkpointer = MemorySaver()
        else:
            postgres_store = await self._exit_stack.enter_async_context(
                AsyncPostgresStore.from_conn_string(self.config.database_url)
            )
            await postgres_store.setup()
            self._store = postgres_store

            postgres_checkpointer = await self._exit_stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(self.config.database_url)
            )
            await postgres_checkpointer.setup()
            self._checkpointer = postgres_checkpointer

        if self.config.rag is not None:
            self._rag_service = WorkspaceDocsRAG(
                self.config.rag,
                project_root=self.project_root,
            )
            rag_status = await asyncio.to_thread(self._rag_service.ensure_ready)
            if not rag_status.ready and rag_status.reason:
                logger.warning("RAG initialization failed: %s", rag_status.reason)
        elif self.config.rag_requested and self.config.rag_error:
            logger.warning("RAG is configured but unavailable: %s", self.config.rag_error)

    async def _load_subagent_mcp_tools(
        self,
        *,
        thread_id: str | None,
        mcp_session_id: str | None,
    ) -> dict[tuple[str, ...], list[Any]]:
        """Load each sync subagent's own MCP tools, keyed by agent path."""
        registry = {
            subagent.name: subagent for subagent in self.config.extensions.subagents
        }
        requests: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

        def collect(subagent: SubagentConfig, agent_path: tuple[str, ...]) -> None:
            requests.append((agent_path, subagent.mcp_servers))
            for child in runtime_graph.nested_child_subagents(subagent, registry):
                collect(child, (*agent_path, child.name))

        for subagent in self.config.extensions.subagents:
            collect(subagent, (subagent.name,))
        loaded = await _gather_or_cancel(
            *(
                self._get_mcp_tools(
                    server_names,
                    thread_id=thread_id,
                    mcp_session_id=mcp_session_id,
                )
                for _, server_names in requests
            )
        )
        return {
            agent_path: tools
            for (agent_path, _), tools in zip(requests, loaded, strict=True)
        }

    def _agent_close_barrier(
        self, cache_key: AgentCacheKey
    ) -> asyncio.Future[None] | None:
        if self._closing_all_agents is not None:
            return self._closing_all_agents
        if cache_key.thread_id is not None:
            barrier = self._closing_agent_threads.get(cache_key.thread_id)
            if barrier is not None:
                return barrier
        if cache_key.mcp_scope is not None:
            return self._closing_agent_scopes.get(cache_key.mcp_scope)
        return None

    @asynccontextmanager
    async def _block_agent_builds(
        self, *, thread_id: str | None = None, mcp_scope: str | None = None,
        all_agents: bool = False,
    ):
        """Hold new graph builds outside resource teardown, without blocking others."""
        if not (all_agents or thread_id or mcp_scope):
            raise ValueError("An agent build barrier requires a target.")
        while True:
            async with self._agent_lock:
                if all_agents:
                    current = (
                        self._closing_all_agents
                        or next(iter(self._closing_agent_threads.values()), None)
                        or next(iter(self._closing_agent_scopes.values()), None)
                    )
                elif thread_id is not None:
                    current = self._closing_all_agents or self._closing_agent_threads.get(thread_id)
                else:
                    current = self._closing_all_agents or self._closing_agent_scopes.get(mcp_scope or "")
                if current is None:
                    barrier = asyncio.get_running_loop().create_future()
                    if all_agents:
                        self._closing_all_agents = barrier
                    elif thread_id is not None:
                        self._closing_agent_threads[thread_id] = barrier
                    else:
                        self._closing_agent_scopes[mcp_scope or ""] = barrier
                    active = [
                        future for key, future in self._agent_builds.items()
                        if all_agents
                        or (thread_id is not None and key.thread_id == thread_id)
                        or (mcp_scope is not None and key.mcp_scope == mcp_scope)
                    ]
                    break
            await asyncio.shield(current)
        try:
            if active:
                await asyncio.gather(
                    *(asyncio.shield(future) for future in active),
                    return_exceptions=True,
                )
            yield
        finally:
            async with self._agent_lock:
                if all_agents and self._closing_all_agents is barrier:
                    self._closing_all_agents = None
                elif thread_id is not None and self._closing_agent_threads.get(thread_id) is barrier:
                    self._closing_agent_threads.pop(thread_id)
                elif mcp_scope is not None and self._closing_agent_scopes.get(mcp_scope) is barrier:
                    self._closing_agent_scopes.pop(mcp_scope)
                if not barrier.done():
                    barrier.set_result(None)

    async def get_agent(
        self,
        reasoning_level: ReasoningLevel,
        *,
        model_name: str | None = None,
        reasoning_level_is_explicit: bool = False,
        thread_id: str | None = None,
        async_subagent_url_override: str | None = None,
        mcp_session_id: str | None = None,
    ):
        """Return the configured agent for a specific runtime context.

        Args:
            reasoning_level: The reasoning level value.
            model_name: The model name value.
            reasoning_level_is_explicit: Whether reasoning was set for this run.
            thread_id: Conversation thread identifier.
            async_subagent_url_override: The async subagent URL override value.
            mcp_session_id: MCP session identifier.

        Returns:
            The configured agent for a specific runtime context.
        """
        agent, _ = await self._get_agent_and_mcp_status(
            reasoning_level,
            model_name=model_name,
            reasoning_level_is_explicit=reasoning_level_is_explicit,
            thread_id=thread_id,
            async_subagent_url_override=async_subagent_url_override,
            mcp_session_id=mcp_session_id,
        )
        return agent

    async def _get_agent_and_mcp_status(
        self,
        reasoning_level: ReasoningLevel,
        *,
        model_name: str | None = None,
        reasoning_level_is_explicit: bool = False,
        thread_id: str | None = None,
        async_subagent_url_override: str | None = None,
        mcp_session_id: str | None = None,
    ) -> tuple[Any, tuple[str, ...]]:
        """Singleflight graph construction by cache key, including failure status."""
        selected_model = (
            str(model_name or self.config.model_name).strip()
            or self.config.model_name
        )
        selected_model_profile = runtime_models.resolve_runtime_model_profile(
            self.config,
            selected_model,
        )
        reasoning_level_is_explicit = (
            self.config.model_reasoning_override
            or reasoning_level_is_explicit
            or reasoning_level != self.config.default_reasoning
        )
        effective_reasoning_level = runtime_graph.reasoning_level_for_profile(
            selected_model_profile,
            reasoning_level,
            fallback_is_explicit=reasoning_level_is_explicit,
        )
        mcp_scope = self._mcp_pool.scope(
            mcp_session_id=mcp_session_id,
            thread_id=thread_id,
        )
        cache_key = AgentCacheKey(
            reasoning_level=effective_reasoning_level,
            reasoning_level_is_explicit=reasoning_level_is_explicit,
            model_name=selected_model,
            thread_id=thread_id,
            async_subagent_url_override=async_subagent_url_override,
            mcp_scope=mcp_scope,
        )
        while True:
            async with self._agent_lock:
                barrier = self._agent_close_barrier(cache_key)
                if barrier is None:
                    cached = self._agents.get(cache_key)
                    if cached is not None:
                        return cached, ()
                    build = self._agent_builds.get(cache_key)
                    owns_build = build is None
                    if build is None:
                        build = asyncio.get_running_loop().create_future()
                        self._agent_builds[cache_key] = build
            if barrier is not None:
                await asyncio.shield(barrier)
                continue
            assert build is not None
            if not owns_build:
                try:
                    return await asyncio.shield(build)
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling():
                        raise
                    # The original builder left; the surviving caller can
                    # retry after its transport cleanup has completed.
                    async with self._agent_lock:
                        if build.cancelled() and self._agent_builds.get(cache_key) is build:
                            self._agent_builds.pop(cache_key)
                    continue
            break

        assert build is not None
        warnings: set[str] = set()
        token = runtime_mcp_sessions._MCP_DISCOVERY_WARNINGS.set(warnings)
        try:
            raw_main_tools, subagent_mcp_tools = await _gather_or_cancel(
                self._build_main_tools(
                    thread_id=thread_id,
                    mcp_session_id=mcp_session_id,
                ),
                self._load_subagent_mcp_tools(
                    thread_id=thread_id,
                    mcp_session_id=mcp_session_id,
                ),
            )
            stateful = self.config.agent_state == "stateful"
            agent_kwargs = runtime_graph.build_agent_kwargs(
                self.config,
                tools=raw_main_tools,
                model_profile=selected_model_profile,
                reasoning_level=effective_reasoning_level,
                reasoning_level_is_explicit=reasoning_level_is_explicit,
                system_prompt=SYSTEM_PROMPT,
                custom_instruction=self.config.extensions.custom_instruction,
                rag_enabled=self._rag_service is not None,
                project_root=self.project_root,
                artifact_registry=self.large_tool_result_artifacts,
                include_async_subagents=True,
                model_name=selected_model,
                async_subagent_url_override=async_subagent_url_override,
                subagent_mcp_tools=subagent_mcp_tools,
                build_model=lambda level, profile: self._build_model(
                    level,
                    model_profile=profile,
                ),
                store=self.store if stateful else None,
                checkpointer=self.checkpointer if stateful else None,
                background_manager=(
                    self.background_tasks
                    if self.config.extensions.background_subagents.enabled
                    else None
                ),
                session_id=thread_id,
                langsmith_tracing=self.langsmith_tracing,
                messaging_broker=self.message_broker,
            )
            agent = runtime_middleware.create_deep_agent_with_configured_summarization(
                self.config,
                **agent_kwargs,
            )
            agent = runtime_background_tasks.scope_background_session_invocation(
                agent,
                self.background_tasks,
                artifact_registry=self.large_tool_result_artifacts,
                fixed_session_id=thread_id,
            )
            result = agent, tuple(sorted(warnings))
            async with self._agent_lock:
                if not warnings:
                    self._agents[cache_key] = agent
                build.set_result(result)
            return result
        except BaseException as exc:
            if not build.done():
                if isinstance(exc, asyncio.CancelledError):
                    build.cancel()
                else:
                    build.set_exception(exc)
                    # No waiters may exist. Mark the exception as observed while
                    # retaining it for any concurrent callers of this build.
                    build.exception()
            raise
        finally:
            runtime_mcp_sessions._MCP_DISCOVERY_WARNINGS.reset(token)
            async with self._agent_lock:
                if self._agent_builds.get(cache_key) is build:
                    self._agent_builds.pop(cache_key)

    async def get_agent_with_status(
        self, reasoning_level: ReasoningLevel, **kwargs: Any
    ) -> tuple[Any, tuple[str, ...]]:
        """Return an agent and MCP discovery failures for this build."""
        return await self._get_agent_and_mcp_status(reasoning_level, **kwargs)

    async def rebuild_rag_index(self) -> RagStatus:
        """Rebuild RAG index.

        Returns:
            The rebuilt object or status.
        """
        return await runtime_rag_ops.rebuild_rag_index(self)

    async def ingest_rag_uploads(
        self,
        *,
        thread_id: str,
        uploads: list[UploadedRagFile],
    ) -> RagUploadResult:
        """Ingest RAG uploads.

        Args:
            thread_id: Conversation thread identifier.
            uploads: Uploaded files supplied by the user.

        Returns:
            The ingest RAG uploads result.
        """
        return await runtime_rag_ops.ingest_rag_uploads(
            self, thread_id=thread_id, uploads=uploads
        )

    async def clone_rag_uploads(
        self,
        *,
        source_thread_id: str,
        target_thread_id: str,
    ) -> RagUploadResult:
        """Clone thread-scoped RAG uploads for a fresh conversation branch."""
        return await runtime_rag_ops.clone_rag_uploads(
            self,
            source_thread_id=source_thread_id,
            target_thread_id=target_thread_id,
        )

    def resolve_chainlit_command(self, name: str) -> ChainlitCommandConfig | None:
        """Resolve a native Chainlit command by name.

        Args:
            name: The name value.

        Returns:
            The matching command configuration, or None when no command matches.
        """
        normalized = runtime_commands.normalize_chainlit_command_name(name)
        if not normalized:
            return None
        for command in self.chainlit_commands:
            if command.name == normalized:
                return command
        return None

    async def invoke_mcp_tool_command(
        self,
        *,
        tool_name: str,
        raw_args: str,
        thread_id: str | None = None,
        mcp_session_id: str | None = None,
        server_name: str | None = None,
    ) -> Any:
        """Invoke a configured MCP tool command with parsed arguments.

        Args:
            tool_name: Name of the tool to invoke.
            raw_args: Raw argument text supplied with the command.
            thread_id: Conversation thread identifier.
            mcp_session_id: MCP session identifier.
            server_name: The server name value.

        Returns:
            The invoke MCP tool command result.

        Raises:
            ValueError: If the supplied value is invalid.
        """
        candidate_servers: tuple[str, ...]
        if server_name:
            candidate_servers = (server_name,)
        else:
            available_servers = self.config.extensions.mcp_servers or {}
            candidate_servers = tuple(available_servers.keys())

        tools, failures = await self.get_mcp_tools_with_status(
            candidate_servers,
            thread_id=thread_id,
            mcp_session_id=mcp_session_id,
        )
        selected_tool = next(
            (
                tool
                for tool in tools
                if str(getattr(tool, "name", "")).strip() == tool_name
            ),
            None,
        )
        if selected_tool is None:
            if failures:
                raise RuntimeError(mcp_outage_warning(failures))
            available = sorted(
                {
                    str(getattr(tool, "name", "")).strip()
                    for tool in tools
                    if str(getattr(tool, "name", "")).strip()
                }
            )
            raise ValueError(
                f"MCP tool '{tool_name}' is unavailable."
                + (f" Available tools: {available}" if available else "")
            )

        parsed_args: Any = {}
        raw_text = raw_args.strip()
        if raw_text:
            try:
                parsed_args = json.loads(raw_text)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Command arguments for MCP tool '{tool_name}' must be valid JSON."
                ) from None
        return await selected_tool.ainvoke(parsed_args)

    def _build_model(
        self,
        reasoning_level: ReasoningLevel,
        *,
        model_name: str | None = None,
        model_profile: ModelDefaults | None = None,
    ) -> Any:
        """Build the chat model for the current runtime settings.

        Args:
            reasoning_level: The reasoning level value.
            model_name: The model name value.

        Returns:
            The constructed the chat model for the current runtime settings.
        """
        if model_profile is not None:
            return runtime_models.build_model_for_profile(
                self.config,
                reasoning_level,
                model_profile,
            )
        return runtime_models.build_model(self.config, reasoning_level, model_name=model_name)

    async def _get_mcp_tools(
        self,
        server_names: tuple[str, ...],
        *,
        thread_id: str | None = None,
        mcp_session_id: str | None = None,
    ) -> list[Any]:
        """Load MCP tools for the active runtime context.

        Args:
            server_names: The server names value.
            thread_id: Conversation thread identifier.
            mcp_session_id: MCP session identifier.

        Returns:
            The requested value.
        """
        tools, _ = await self.get_mcp_tools_with_status(
            server_names, thread_id=thread_id, mcp_session_id=mcp_session_id
        )
        return tools

    async def get_mcp_tools_with_status(
        self,
        server_names: tuple[str, ...],
        *,
        thread_id: str | None = None,
        mcp_session_id: str | None = None,
    ) -> tuple[list[Any], tuple[str, ...]]:
        """Load available tools while reporting failed servers by name."""
        return await self._mcp_pool.tools_with_status(
            server_names, thread_id=thread_id, mcp_session_id=mcp_session_id
        )

    async def _build_main_tools(
        self,
        *,
        thread_id: str | None,
        mcp_session_id: str | None,
    ) -> list[Any]:
        """Build the main agent tool list for a runtime context.

        Args:
            thread_id: Conversation thread identifier.
            mcp_session_id: MCP session identifier.

        Returns:
            The constructed the main agent tool list for a runtime context.
        """
        mcp_tools = await self._get_mcp_tools(
            self.config.extensions.agent_mcp_servers,
            thread_id=thread_id,
            mcp_session_id=mcp_session_id,
        )
        return runtime_graph.build_main_tools(
            self.config,
            mcp_tools=mcp_tools,
            rag_service=self._rag_service,
            thread_id=thread_id,
        )

    async def _clear_agent_cache(self) -> None:
        """Clear cached agents after runtime tool state changes."""
        async with self._block_agent_builds(all_agents=True), self._agent_lock:
            self._agents.clear()

    async def close_mcp_session(self, mcp_session_id: str | None) -> None:
        """Close MCP session.

        Args:
            mcp_session_id: MCP session identifier.
        """
        mcp_scope = self._mcp_pool.scope(mcp_session_id=mcp_session_id)
        if mcp_scope is None:
            return

        async with self._block_agent_builds(mcp_scope=mcp_scope):
            async with self._agent_lock:
                self._agents = {
                    key: agent
                    for key, agent in self._agents.items()
                    if key.mcp_scope != mcp_scope
                }
            await self._mcp_pool.evict_scope_and_close(mcp_scope)

    async def close_conversation(
        self, *, thread_id: str | None, mcp_session_id: str | None = None
    ) -> None:
        """Release conversation graphs and any stateful MCP transport resources."""
        close_task = asyncio.create_task(
            self._close_conversation(
                thread_id=thread_id,
                mcp_session_id=mcp_session_id,
            ),
            name=f"chainagents-close-conversation-{thread_id or mcp_session_id}",
        )
        await runtime_background_tasks.await_preserving_cancellation(close_task)

    async def _close_conversation(
        self, *, thread_id: str | None, mcp_session_id: str | None = None
    ) -> None:
        """Complete conversation teardown independently of its caller."""
        if thread_id:
            async with (
                self._block_agent_builds(thread_id=thread_id),
                self.background_tasks.closing_session(thread_id),
            ):
                await self.user_input.close_session(thread_id)
                self.message_broker.close_session(thread_id)
                errors: list[BaseException] = []
                try:
                    await self.large_tool_result_artifacts.close_session(thread_id)
                except BaseException as exc:
                    errors.append(exc)
                try:
                    await self.close_mcp_session(mcp_session_id or thread_id)
                except BaseException as exc:
                    errors.append(exc)
                try:
                    async with self._agent_lock:
                        self._agents = {
                            key: agent for key, agent in self._agents.items()
                            if key.thread_id != thread_id
                        }
                except BaseException as exc:
                    errors.append(exc)
                if len(errors) == 1:
                    raise errors[0]
                if errors:
                    raise BaseExceptionGroup(
                        "Conversation resource cleanup failed.",
                        errors,
                    )
            return
        await self.close_mcp_session(mcp_session_id)

    async def close_all_mcp_sessions(self) -> None:
        """Close all MCP sessions."""
        async with self._block_agent_builds(all_agents=True):
            async with self._agent_lock:
                self._agents.clear()
            await self._mcp_pool.evict_all_and_close()

    async def close(self) -> None:
        """Close the agent runtime."""
        try:
            await self.user_input.close_all()
            await self.background_tasks.close()
        finally:
            try:
                await self.large_tool_result_artifacts.close()
            finally:
                try:
                    await self._exit_stack.aclose()
                finally:
                    try:
                        if self.langsmith_tracing is not None:
                            flush = asyncio.create_task(
                                asyncio.to_thread(self.langsmith_tracing.flush),
                                name="chainagents-flush-langsmith",
                            )
                            try:
                                await runtime_background_tasks.await_preserving_cancellation(
                                    flush
                                )
                            finally:
                                close = asyncio.create_task(
                                    asyncio.to_thread(self.langsmith_tracing.close),
                                    name="chainagents-close-langsmith",
                                )
                                await runtime_background_tasks.await_preserving_cancellation(
                                    close
                                )
                    finally:
                        self._checkpointer = None
                        self._store = None
                        self._mcp_pool.client = None
