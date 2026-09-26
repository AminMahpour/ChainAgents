"""Stateful agent, MCP, persistence, and RAG resource lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from deepagents.backends import StoreBackend
from langchain_mcp_adapters.client import MultiServerMCPClient
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
import chainagents.runtime.mcp_sessions as runtime_mcp_sessions
import chainagents.runtime.middleware as runtime_middleware
import chainagents.runtime.models as runtime_models
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
        self.langsmith_tracing = runtime_tracing.build_langsmith_tracing(config.langsmith)
        self.project_root = project_root or runtime_constants.PROJECT_ROOT
        self._exit_stack = AsyncExitStack()
        self._agent_lock = asyncio.Lock()
        self._agents: dict[AgentCacheKey, object] = {}
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
            self._mcp_pool.client = MultiServerMCPClient(
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
        tools_by_path: dict[tuple[str, ...], list[Any]] = {}

        async def load(subagent: SubagentConfig, agent_path: tuple[str, ...]) -> None:
            tools_by_path[agent_path] = await self._get_mcp_tools(
                subagent.mcp_servers,
                thread_id=thread_id,
                mcp_session_id=mcp_session_id,
            )
            for child in runtime_graph.nested_child_subagents(subagent, registry):
                await load(child, (*agent_path, child.name))

        for subagent in self.config.extensions.subagents:
            await load(subagent, (subagent.name,))
        return tools_by_path

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
        async with self._agent_lock:
            agent = self._agents.get(cache_key)
            if agent is None:
                self._mcp_pool.discovery_failed = False
                raw_main_tools = await self._build_main_tools(
                    thread_id=thread_id,
                    mcp_session_id=mcp_session_id,
                )
                subagent_mcp_tools = await self._load_subagent_mcp_tools(
                    thread_id=thread_id,
                    mcp_session_id=mcp_session_id,
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
                if not self._mcp_pool.discovery_failed:
                    self._agents[cache_key] = agent
            return agent

    async def get_agent_with_status(
        self, reasoning_level: ReasoningLevel, **kwargs: Any
    ) -> tuple[Any, tuple[str, ...]]:
        """Return an agent and MCP discovery failures for this build."""
        warnings: set[str] = set()
        token = runtime_mcp_sessions._MCP_DISCOVERY_WARNINGS.set(warnings)
        try:
            agent = await self.get_agent(reasoning_level, **kwargs)
        finally:
            runtime_mcp_sessions._MCP_DISCOVERY_WARNINGS.reset(token)
        return agent, tuple(sorted(warnings))

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
        async with self._agent_lock:
            self._agents.clear()

    async def close_mcp_session(self, mcp_session_id: str | None) -> None:
        """Close MCP session.

        Args:
            mcp_session_id: MCP session identifier.
        """
        mcp_scope = self._mcp_pool.scope(mcp_session_id=mcp_session_id)
        if mcp_scope is None:
            return

        async with self._agent_lock:
            self._agents = {
                key: agent
                for key, agent in self._agents.items()
                if key.mcp_scope != mcp_scope
            }
            owners = await self._mcp_pool.evict_scope(mcp_scope)

        await self._mcp_pool.close_owner_entries(owners)

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

    async def close_all_mcp_sessions(self) -> None:
        """Close all MCP sessions."""
        async with self._agent_lock:
            owners = await self._mcp_pool.evict_all()
            self._agents.clear()

        await self._mcp_pool.close_owner_entries(owners)

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
