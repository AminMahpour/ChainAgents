"""Stateful agent, MCP, persistence, and RAG resource lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
from contextvars import ContextVar
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from deepagents.backends import StoreBackend
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore

import chainagents.runtime.backends as runtime_backends
import chainagents.runtime.artifacts as runtime_artifacts
import chainagents.runtime.background_tasks as runtime_background_tasks
import chainagents.runtime.commands as runtime_commands
import chainagents.runtime.constants as runtime_constants
import chainagents.runtime.graph as runtime_graph
import chainagents.runtime.middleware as runtime_middleware
import chainagents.runtime.models as runtime_models
from chainagents.rag.runtime import (
    RagStatus,
    RagUploadResult,
    UploadedRagFile,
    WorkspaceDocsRAG,
    compose_rag_system_prompt,
    create_search_workspace_knowledge_tool,
)
from chainagents.runtime.config import RuntimeConfig
from chainagents.runtime.constants import SYSTEM_PROMPT, ReasoningLevel
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
_MCP_DISCOVERY_WARNINGS: ContextVar[set[str] | None] = ContextVar(
    "mcp_discovery_warnings", default=None
)


async def agent_with_mcp_status(
    runtime: Any, reasoning_level: ReasoningLevel, **kwargs: Any
) -> tuple[Any, tuple[str, ...]]:
    """Resolve an agent and warnings, including runtimes without status support."""
    method = getattr(runtime, "get_agent_with_status", None)
    if callable(method):
        return await method(reasoning_level, **kwargs)
    return await runtime.get_agent(reasoning_level, **kwargs), ()


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


class AgentRuntime:
    """Own configured agents, MCP sessions, persistence handles, and RAG state."""

    _instance: "AgentRuntime | None" = None
    _instance_lock = asyncio.Lock()

    def __init__(self, config: RuntimeConfig, *, project_root: Path | None = None) -> None:
        """Initialize the agent runtime instance.

        Args:
            config: Configuration object used by the operation.
            project_root: Project root used to resolve local paths.
        """
        self.config = config
        self.project_root = project_root or runtime_constants.PROJECT_ROOT
        self._exit_stack = AsyncExitStack()
        self._agent_lock = asyncio.Lock()
        self._mcp_lock = asyncio.Lock()
        self._agents: dict[AgentCacheKey, object] = {}
        self._mcp_client: MultiServerMCPClient | None = None
        self._mcp_tools_cache: dict[tuple[str | None, tuple[str, ...]], list[Any]] = {}
        self._mcp_discovery_failed = False
        self._mcp_sessions: dict[tuple[str | None, str], Any] = {}
        self._mcp_session_owners: dict[tuple[str | None, str], _MCPSessionOwner] = {}
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
    async def get(cls) -> "AgentRuntime":
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
    ) -> "AgentRuntime":
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
    def current(cls) -> "AgentRuntime | None":
        """Return the current.

        Returns:
            The current.
        """
        return cls._instance

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
        return self.config.rag_requested

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
        if self._rag_service is not None:
            return self._rag_service.snapshot()
        if self.config.rag_requested:
            return RagStatus.unavailable(
                reason=self.config.rag_error or "Knowledge index is unavailable.",
                persist_directory=(
                    self.config.rag.persist_directory if self.config.rag is not None else None
                ),
            )
        return RagStatus.disabled()

    async def _initialize(self) -> None:
        """Initialize persistence, RAG, MCP clients, and configured agents."""
        if self.config.extensions.mcp_servers:
            self._mcp_client = MultiServerMCPClient(
                self.config.extensions.mcp_servers,
                tool_name_prefix=self.config.extensions.mcp_tool_name_prefix,
            )

        if self.config.agent_state == "stateless":
            self._store = None
            self._checkpointer = None
        elif not self.config.database_url:
            self._store = InMemoryStore()
            self._checkpointer = MemorySaver()
        else:
            self._store = await self._exit_stack.enter_async_context(
                AsyncPostgresStore.from_conn_string(self.config.database_url)
            )
            await self.store.setup()

            self._checkpointer = await self._exit_stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(self.config.database_url)
            )
            await self.checkpointer.setup()

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

    async def _build_runtime_subagent_specs(
        self,
        *,
        reasoning_level: ReasoningLevel,
        reasoning_level_is_explicit: bool,
        selected_model_profile: ModelDefaults,
        backend: Any,
        inherited_tools: list[Any],
        sanitized_inherited_tools: list[Any],
        thread_id: str | None,
        mcp_session_id: str | None,
    ) -> list[Any]:
        """Build top-level sync subagent specs for a runtime context."""
        registry = {
            subagent.name: subagent for subagent in self.config.extensions.subagents
        }
        return [
            await self._build_runtime_sync_subagent_spec(
                subagent,
                registry=registry,
                reasoning_level=reasoning_level,
                reasoning_level_is_explicit=reasoning_level_is_explicit,
                inherited_model=selected_model_profile,
                backend=backend,
                inherited_tools=inherited_tools,
                sanitized_inherited_tools=sanitized_inherited_tools,
                thread_id=thread_id,
                mcp_session_id=mcp_session_id,
                agent_path=(subagent.name,),
            )
            for subagent in self.config.extensions.subagents
        ]

    async def _build_runtime_sync_subagent_spec(
        self,
        subagent: SubagentConfig,
        *,
        registry: dict[str, SubagentConfig],
        reasoning_level: ReasoningLevel,
        reasoning_level_is_explicit: bool,
        inherited_model: ModelDefaults,
        backend: Any,
        inherited_tools: list[Any],
        sanitized_inherited_tools: list[Any],
        thread_id: str | None,
        mcp_session_id: str | None,
        agent_path: tuple[str, ...],
    ) -> dict[str, Any]:
        """Build one sync subagent spec, compiling it when it has children."""
        effective_model = runtime_models.resolve_runtime_model_profile(
            self.config,
            subagent.model,
            inherited_model=inherited_model,
        )
        effective_reasoning_level = runtime_graph.reasoning_level_for_profile(
            effective_model,
            reasoning_level,
            fallback_is_explicit=reasoning_level_is_explicit,
        )
        raw_own_tools = await self._get_mcp_tools(
            subagent.mcp_servers,
            thread_id=thread_id,
            mcp_session_id=mcp_session_id,
        )
        own_tools = runtime_graph.sanitize_tools_for_model(
            effective_model.provider,
            raw_own_tools,
        )
        inherited_model_tools = runtime_graph.inherited_tools_for_model(
            inherited_tools=inherited_tools,
            sanitized_inherited_tools=sanitized_inherited_tools,
            inherited_provider=inherited_model.provider,
            effective_provider=effective_model.provider,
        )
        has_configured_own_tools = bool(subagent.mcp_servers)
        effective_tools = (
            own_tools
            if has_configured_own_tools
            else own_tools or inherited_model_tools
        )
        middleware = runtime_middleware.build_agent_middleware(
            backend=backend,
            config=self.config,
            reasoning_level=effective_reasoning_level,
            model_name=effective_model.name,
            source=subagent.name,
            project_root=self.project_root,
        )
        global_background_enabled = (
            self.config.extensions.background_subagents.enabled
        )
        child_subagents = runtime_graph.nested_child_subagents(
            subagent,
            registry,
        )
        target_background_enabled = global_background_enabled and subagent.background
        background_child_names = {
            child.name for child in child_subagents if child.background
        }
        caller_background_enabled = (
            global_background_enabled and bool(background_child_names)
        )
        if not child_subagents and not target_background_enabled:
            subagent_tools = own_tools
            if (
                not subagent_tools
                and subagent.model
                and effective_model.provider != inherited_model.provider
                and not has_configured_own_tools
            ):
                subagent_tools = inherited_model_tools
            subagent_model = (
                self._build_model(
                    effective_reasoning_level,
                    model_profile=effective_model,
                )
                if subagent.model
                else None
            )
            return subagent.to_deepagents_spec(
                tools=subagent_tools,
                middleware=middleware,
                model=subagent_model,
            )

        child_specs = [
            await self._build_runtime_sync_subagent_spec(
                child,
                registry=registry,
                reasoning_level=effective_reasoning_level,
                reasoning_level_is_explicit=reasoning_level_is_explicit,
                inherited_model=effective_model,
                backend=backend,
                inherited_tools=raw_own_tools if has_configured_own_tools else inherited_tools,
                sanitized_inherited_tools=effective_tools,
                thread_id=thread_id,
                mcp_session_id=mcp_session_id,
                agent_path=(*agent_path, child.name),
            )
            for child in child_subagents
        ]
        if not child_specs:
            middleware.append(
                runtime_middleware.DisableSubagentDelegationMiddleware()
            )
        background_tools = (
            runtime_background_tasks.create_background_task_tools(
                manager=self.background_tasks,
                subagents={
                    spec["name"]: spec["runnable"]
                    for spec in child_specs
                    if spec["name"] in background_child_names
                },
                agent_path=agent_path,
                recursion_limit=self.config.recursion_limit,
                session_generation=(
                    self.background_tasks.session_generation(thread_id)
                    if thread_id
                    else None
                ),
                batch_output_store=(
                    runtime_background_tasks.create_batch_result_output_store(
                        backend,
                        backend_prefix=(
                            runtime_backends.generated_outputs_route_prefix(
                                self.project_root
                            )
                        ),
                    )
                ),
                existing_tools=effective_tools,
            )
            if caller_background_enabled
            else []
        )
        runnable_kwargs: dict[str, Any] = {
            "model": self._build_model(
                effective_reasoning_level,
                model_profile=effective_model,
            ),
            "tools": [*effective_tools, *background_tools] or None,
            "system_prompt": subagent.system_prompt,
            "middleware": middleware,
            "backend": backend,
            "skills": list(subagent.skills) or None,
            "subagents": child_specs,
        }
        if self.config.agent_state == "stateful":
            runnable_kwargs["store"] = self.store
            runnable_kwargs["checkpointer"] = self.checkpointer
        runnable = runtime_middleware.create_deep_agent_with_configured_summarization(
            self.config,
            **runnable_kwargs,
        )
        return {
            "name": subagent.name,
            "description": subagent.description,
            "runnable": (
                runtime_background_tasks.scope_background_task_invocation(runnable)
                if caller_background_enabled
                else runnable
            ),
        }

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
        mcp_scope = self._mcp_scope(
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
        async with self._agent_lock:
            agent = self._agents.get(cache_key)
            if agent is None:
                self._mcp_discovery_failed = False
                model = self._build_model(
                    effective_reasoning_level,
                    model_profile=selected_model_profile,
                )
                rag_tool_enabled = self._rag_service is not None
                raw_main_tools = await self._build_main_tools(
                    thread_id=thread_id,
                    mcp_session_id=mcp_session_id,
                )
                main_tools = runtime_graph.sanitize_tools_for_model(
                    selected_model_profile.provider,
                    raw_main_tools,
                )
                backend = runtime_backends.build_deepagent_backend(
                    project_root=self.project_root,
                    include_memories=self.config.agent_state == "stateful",
                    memory_namespace=self.config.extensions.agent_memory_namespace,
                    artifact_registry=self.large_tool_result_artifacts,
                )
                middleware = runtime_middleware.build_agent_middleware(
                    backend=backend,
                    config=self.config,
                    reasoning_level=effective_reasoning_level,
                    model_name=selected_model,
                    source="main-agent",
                    project_root=self.project_root,
                )
                subagent_specs = await self._build_runtime_subagent_specs(
                    reasoning_level=effective_reasoning_level,
                    reasoning_level_is_explicit=reasoning_level_is_explicit,
                    selected_model_profile=selected_model_profile,
                    backend=backend,
                    inherited_tools=raw_main_tools,
                    sanitized_inherited_tools=main_tools,
                    thread_id=thread_id,
                    mcp_session_id=mcp_session_id,
                )
                local_subagent_specs = list(subagent_specs)
                subagent_specs.extend(
                    subagent.to_deepagents_spec(
                        url_override=async_subagent_url_override,
                    )
                    for subagent in self.config.extensions.async_subagents
                )
                background_subagent_names = {
                    subagent.name
                    for subagent in self.config.extensions.subagents
                    if subagent.background
                }
                background_subagents = {
                    spec["name"]: spec["runnable"]
                    for spec in local_subagent_specs
                    if spec["name"] in background_subagent_names
                }
                background_tools = (
                    runtime_background_tasks.create_background_task_tools(
                        manager=self.background_tasks,
                        subagents=background_subagents,
                        agent_path=(),
                        recursion_limit=self.config.recursion_limit,
                        session_generation=(
                            self.background_tasks.session_generation(thread_id)
                            if thread_id
                            else None
                        ),
                        batch_output_store=(
                            runtime_background_tasks.create_batch_result_output_store(
                                backend,
                                backend_prefix=(
                                    runtime_backends.generated_outputs_route_prefix(
                                        self.project_root
                                    )
                                ),
                            )
                        ),
                        existing_tools=main_tools,
                    )
                    if (
                        self.config.extensions.background_subagents.enabled
                        and background_subagents
                    )
                    else []
                )
                agent_kwargs: dict[str, Any] = {
                    "model": model,
                    "tools": [*main_tools, *background_tools] or None,
                    "system_prompt": compose_rag_system_prompt(
                        runtime_graph.compose_agent_system_prompt(
                            runtime_graph.system_prompt_for_agent_state(
                                SYSTEM_PROMPT,
                                self.config.agent_state,
                            ),
                            self.config.extensions.custom_instruction,
                            project_root=self.project_root,
                        ),
                        rag_enabled=rag_tool_enabled,
                    ),
                    "middleware": middleware,
                    "backend": backend,
                    "skills": list(self.config.extensions.skills) or None,
                    "subagents": subagent_specs or None,
                }
                memory_files = runtime_graph.stateful_agent_memory_files(self.config)
                if memory_files is not None:
                    agent_kwargs["memory"] = memory_files
                if self.config.agent_state == "stateful":
                    agent_kwargs["store"] = self.store
                    agent_kwargs["checkpointer"] = self.checkpointer
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
                if not self._mcp_discovery_failed:
                    self._agents[cache_key] = agent
            return agent

    async def get_agent_with_status(
        self, reasoning_level: ReasoningLevel, **kwargs: Any
    ) -> tuple[Any, tuple[str, ...]]:
        """Return an agent and MCP discovery failures for this build."""
        warnings: set[str] = set()
        token = _MCP_DISCOVERY_WARNINGS.set(warnings)
        try:
            agent = await self.get_agent(reasoning_level, **kwargs)
        finally:
            _MCP_DISCOVERY_WARNINGS.reset(token)
        return agent, tuple(sorted(warnings))

    async def rebuild_rag_index(self) -> RagStatus:
        """Rebuild RAG index.

        Returns:
            The rebuilt object or status.
        """
        if self._rag_service is None:
            if self.config.rag_requested:
                return RagStatus.unavailable(
                    reason=self.config.rag_error or "Knowledge index is unavailable.",
                    persist_directory=(
                        self.config.rag.persist_directory
                        if self.config.rag is not None
                        else None
                    ),
                )
            return RagStatus.disabled()

        status = await asyncio.to_thread(self._rag_service.rebuild)
        await self._clear_agent_cache()
        return status

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
        if self._rag_service is None:
            return RagUploadResult(
                thread_id=thread_id,
                reason=self.config.rag_error or "Knowledge index is unavailable.",
            )

        return await asyncio.to_thread(
            self._rag_service.ingest_uploaded_files,
            thread_id=thread_id,
            uploads=uploads,
        )

    async def clone_rag_uploads(
        self,
        *,
        source_thread_id: str,
        target_thread_id: str,
    ) -> RagUploadResult:
        """Clone thread-scoped RAG uploads for a fresh conversation branch."""
        if self._rag_service is None:
            return RagUploadResult(
                thread_id=target_thread_id,
                reason=self.config.rag_error or "Knowledge index is unavailable.",
            )
        return await asyncio.to_thread(
            self._rag_service.clone_thread_uploads,
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

    def _sanitize_tools_for_model(self, tools: list[Any]) -> list[Any]:
        """Sanitize tools for the active model provider.

        Args:
            tools: The tools value.

        Returns:
            The sanitized value.
        """
        return runtime_graph.sanitize_tools_for_model(self.config.model_provider, tools)

    @staticmethod
    def _tool_supports_openai_compatible_schema(tool: Any) -> bool:
        """Return whether a tool supports OpenAI-compatible schemas.

        Args:
            tool: The tool value.

        Returns:
            Whether a tool supports OpenAI-compatible schemas.
        """
        return runtime_graph.tool_supports_openai_compatible_schema(tool)

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

    def _mcp_scope(
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
        if not self.config.extensions.mcp_stateful:
            return None

        candidate = str(mcp_session_id or "").strip()
        if candidate:
            return candidate

        fallback = str(thread_id or "").strip()
        return fallback or None

    async def _get_stateful_mcp_session(
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
        scope = self._mcp_scope(
            mcp_session_id=mcp_session_id,
            thread_id=thread_id,
        )
        cache_key = (scope, server_name)
        session = self._mcp_sessions.get(cache_key)
        if session is not None:
            return session

        stale_owner = self._mcp_session_owners.get(cache_key)
        if stale_owner is not None:
            if not stale_owner.terminal:
                try:
                    await stale_owner.aclose()
                except Exception:
                    if not stale_owner.terminal:
                        raise
                    logger.exception("Failed to close terminal MCP session for %s", server_name)
            self._mcp_session_owners.pop(cache_key, None)

        if self._mcp_client is None:
            raise RuntimeError("MCP client is not initialized.")

        owner = _MCPSessionOwner(self._mcp_client.session(server_name))
        session = await owner.session()
        self._mcp_session_owners[cache_key] = owner
        self._mcp_sessions[cache_key] = session
        return session

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
        if not server_names or self._mcp_client is None:
            return [], ()

        tool_scope = self._mcp_scope(
            mcp_session_id=mcp_session_id,
            thread_id=thread_id,
        )
        async with self._mcp_lock:
            tools: list[Any] = []
            failures: list[str] = []
            for server_name in server_names:
                cache_key = (tool_scope, (server_name,))
                cached = self._mcp_tools_cache.get(cache_key)
                if cached is not None:
                    tools.extend(cached)
                    continue
                session_key = (tool_scope, server_name)
                had_session = session_key in self._mcp_sessions
                try:
                    if self.config.extensions.mcp_stateful:
                        session = await self._get_stateful_mcp_session(
                            server_name=server_name,
                            thread_id=thread_id,
                            mcp_session_id=mcp_session_id,
                        )
                        loaded = await load_mcp_tools(
                                session,
                                callbacks=self._mcp_client.callbacks,
                                tool_interceptors=self._mcp_client.tool_interceptors,
                                server_name=server_name,
                                tool_name_prefix=self.config.extensions.mcp_tool_name_prefix,
                            )
                    else:
                        loaded = await self._mcp_client.get_tools(server_name=server_name)
                except BaseException as exc:
                    if not had_session:
                        self._mcp_sessions.pop(session_key, None)
                        owner = self._mcp_session_owners.get(session_key)
                        if owner is not None:
                            try:
                                await owner.aclose()
                            except Exception:
                                logger.exception("Failed to close MCP session for %s", server_name)
                            else:
                                self._mcp_session_owners.pop(session_key, None)
                            if owner.terminal:
                                self._mcp_session_owners.pop(session_key, None)
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    if not isinstance(exc, Exception):
                        raise
                    logger.warning("MCP server %s is unavailable: %s", server_name, exc)
                    failures.append(server_name)
                    self._mcp_discovery_failed = True
                    warnings = _MCP_DISCOVERY_WARNINGS.get()
                    if warnings is not None:
                        warnings.add(server_name)
                    continue
                self._mcp_tools_cache[cache_key] = list(loaded)
                tools.extend(loaded)
            return tools, tuple(failures)

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
        tools = await self._get_mcp_tools(
            self.config.extensions.agent_mcp_servers,
            thread_id=thread_id,
            mcp_session_id=mcp_session_id,
        )
        tools = list(tools)
        if self.config.extensions.chainlit_generative_ui_enabled:
            tools.append(runtime_commands.create_render_chainlit_ui_tool())
        if self._rag_service is not None:
            tools.append(
                create_search_workspace_knowledge_tool(
                    self._rag_service,
                    thread_id=thread_id,
                )
            )
        return tools

    async def _clear_agent_cache(self) -> None:
        """Clear cached agents after runtime tool state changes."""
        async with self._agent_lock:
            self._agents.clear()

    async def close_mcp_session(self, mcp_session_id: str | None) -> None:
        """Close MCP session.

        Args:
            mcp_session_id: MCP session identifier.
        """
        mcp_scope = self._mcp_scope(mcp_session_id=mcp_session_id)
        if mcp_scope is None:
            return

        async with self._agent_lock:
            self._agents = {
                key: agent
                for key, agent in self._agents.items()
                if key.mcp_scope != mcp_scope
            }
            async with self._mcp_lock:
                owners = [
                    (key, owner)
                    for key, owner in self._mcp_session_owners.items()
                    if key[0] == mcp_scope
                ]
                self._mcp_sessions = {
                    key: session
                    for key, session in self._mcp_sessions.items()
                    if key[0] != mcp_scope
                }
                self._mcp_tools_cache = {
                    key: tools
                    for key, tools in self._mcp_tools_cache.items()
                    if key[0] != mcp_scope
                }

        await self._close_mcp_owner_entries(owners)

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
            async with self.background_tasks.closing_session(thread_id):
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

    async def _close_mcp_owner_entries(
        self, owners: list[tuple[tuple[str | None, str], _MCPSessionOwner]]
    ) -> None:
        """Close owners independently and retain failures for later retry."""
        results = await asyncio.gather(
            *(owner.aclose() for _, owner in owners), return_exceptions=True
        )
        async with self._mcp_lock:
            for (key, owner), result in zip(owners, results, strict=True):
                if not isinstance(result, BaseException) or getattr(owner, "terminal", False):
                    if self._mcp_session_owners.get(key) is owner:
                        self._mcp_session_owners.pop(key, None)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def close_all_mcp_sessions(self) -> None:
        """Close all MCP sessions."""
        async with self._agent_lock:
            async with self._mcp_lock:
                owners = list(self._mcp_session_owners.items())
                self._mcp_sessions.clear()
                self._mcp_tools_cache.clear()
                self._agents.clear()

        await self._close_mcp_owner_entries(owners)

    async def close(self) -> None:
        """Close the agent runtime."""
        try:
            await self.background_tasks.close()
        finally:
            try:
                await self.large_tool_result_artifacts.close()
            finally:
                try:
                    await self._exit_stack.aclose()
                finally:
                    self._checkpointer = None
                    self._store = None
                    self._mcp_client = None

    def _build_backend(self, runtime):
        """Build the Deep Agent backend for the current runtime settings.

        Args:
            runtime: Agent runtime used by the operation.

        Returns:
            The constructed the deep agent backend for the current runtime settings.
        """
        return runtime_backends.build_deepagent_backend(
            project_root=self.project_root,
            include_memories=runtime.config.agent_state == "stateful",
            memory_namespace=runtime.config.extensions.agent_memory_namespace,
            artifact_registry=self.large_tool_result_artifacts,
        )
