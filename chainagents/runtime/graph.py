"""Tool schema adaptation and static DeepAgents graph construction."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.utils.function_calling import convert_to_openai_tool

import chainagents.runtime.backends as runtime_backends
import chainagents.runtime.artifacts as runtime_artifacts
import chainagents.runtime.background_tasks as runtime_background_tasks
import chainagents.runtime.commands as runtime_commands
import chainagents.runtime.constants as runtime_constants
import chainagents.runtime.middleware as runtime_middleware
import chainagents.runtime.models as runtime_models
import chainagents.runtime.tracing as runtime_tracing
from chainagents.rag.runtime import (
    WorkspaceDocsRAG,
    compose_rag_system_prompt,
    create_search_workspace_knowledge_tool,
)
from chainagents.runtime.config import RuntimeConfig
from chainagents.runtime.constants import (
    AGENTS_MD_FILENAME,
    DEFAULT_REASONING_LEVEL,
    OPENAI_COMPATIBLE_MODEL_PROVIDERS,
    STATELESS_SYSTEM_PROMPT_MEMORY_LINE,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_MEMORY_LINE,
    AgentStateMode,
    ModelProvider,
    ReasoningLevel,
)
from chainagents.runtime.types import ModelDefaults, SubagentConfig

logger = logging.getLogger("chainagents.runtime.core")

_STATIC_BACKGROUND_TASK_MANAGERS: set[
    runtime_background_tasks.BackgroundTaskManager
] = set()
_STATIC_LARGE_TOOL_RESULT_ARTIFACTS = (
    runtime_artifacts.LargeToolResultArtifactRegistry()
)
_STATIC_LANGSMITH_TRACINGS: set[runtime_tracing.LangSmithTracing] = set()


def static_background_task_managers() -> tuple[
    runtime_background_tasks.BackgroundTaskManager, ...
]:
    """Return task managers owned by exported configured graphs."""
    return tuple(_STATIC_BACKGROUND_TASK_MANAGERS)


async def close_static_background_tasks() -> None:
    """Close exported-graph task managers and remaining result artifacts."""
    managers = tuple(_STATIC_BACKGROUND_TASK_MANAGERS)
    _STATIC_BACKGROUND_TASK_MANAGERS.clear()
    errors: list[BaseException] = []
    if managers:
        results = await asyncio.gather(
            *(manager.drain() for manager in managers),
            return_exceptions=True,
        )
        errors.extend(
            result for result in results if isinstance(result, BaseException)
        )
    try:
        await _STATIC_LARGE_TOOL_RESULT_ARTIFACTS.drain()
    except BaseException as exc:
        errors.append(exc)
    if _STATIC_LANGSMITH_TRACINGS:
        results = await asyncio.gather(
            *(
                asyncio.to_thread(tracing.flush)
                for tracing in tuple(_STATIC_LANGSMITH_TRACINGS)
            ),
            return_exceptions=True,
        )
        errors.extend(
            result for result in results if isinstance(result, BaseException)
        )
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("Exported graph cleanup failed.", errors)


async def close_static_background_session(session_id: str) -> None:
    """Close one session across exported task and artifact owners."""
    close_task = asyncio.create_task(
        _close_static_background_session(session_id),
        name=f"chainagents-close-static-session-{session_id}",
    )
    await runtime_background_tasks.await_preserving_cancellation(close_task)


async def _close_static_background_session(session_id: str) -> None:
    """Complete exported session teardown independently of its caller."""
    managers = static_background_task_managers()
    if managers:
        async with AsyncExitStack() as stack:
            for manager in managers:
                await stack.enter_async_context(manager.closing_session(session_id))
            await _STATIC_LARGE_TOOL_RESULT_ARTIFACTS.close_session(session_id)
        return
    await _STATIC_LARGE_TOOL_RESULT_ARTIFACTS.close_session(session_id)


def _get_static_background_task_manager(
    config: RuntimeConfig,
) -> runtime_background_tasks.BackgroundTaskManager:
    """Return the process-wide manager shared by exported graphs."""
    background_config = config.extensions.background_subagents
    if _STATIC_BACKGROUND_TASK_MANAGERS:
        manager = next(iter(_STATIC_BACKGROUND_TASK_MANAGERS))
        if manager.config != background_config:
            raise RuntimeError(
                "Exported graphs must use one background_subagents configuration."
            )
        return manager
    manager = runtime_background_tasks.BackgroundTaskManager(
        background_config,
        artifact_registry=_STATIC_LARGE_TOOL_RESULT_ARTIFACTS,
    )
    _STATIC_BACKGROUND_TASK_MANAGERS.add(manager)
    return manager


def load_agents_md_instruction(project_root: Path | None = None) -> str | None:
    """Load agents md instruction.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The loaded value.
    """
    agents_md_path = (project_root or runtime_constants.PROJECT_ROOT).resolve() / AGENTS_MD_FILENAME
    try:
        instruction = agents_md_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Failed to read %s: %s", agents_md_path, exc)
        return None
    return instruction or None


def compose_agent_system_prompt(
    base_prompt: str,
    custom_instruction: str | None,
    *,
    project_root: Path | None = None,
) -> str:
    """Compose agent system prompt.

    Args:
        base_prompt: The base prompt value.
        custom_instruction: The custom instruction value.
        project_root: Project root used to resolve local paths.

    Returns:
        The composed value.
    """
    sections = [base_prompt]
    agents_md_instruction = load_agents_md_instruction(project_root)
    if agents_md_instruction:
        sections.append(
            f"Repository instructions from {AGENTS_MD_FILENAME}:\n"
            f"{agents_md_instruction}"
        )

    instruction = (custom_instruction or "").strip()
    if not instruction:
        return "\n\n".join(sections)
    sections.append(
        "Custom user instruction from deepagent.toml:\n"
        f"{instruction}"
    )
    return "\n\n".join(sections)


def system_prompt_for_agent_state(
    base_prompt: str,
    agent_state: AgentStateMode,
) -> str:
    """Return the system prompt adjusted for configured agent state.

    Args:
        base_prompt: The base prompt value.
        agent_state: Whether the DeepAgents graph is stateful or stateless.

    Returns:
        The adjusted system prompt.
    """
    if agent_state == "stateful":
        return base_prompt
    return base_prompt.replace(
        SYSTEM_PROMPT_MEMORY_LINE,
        STATELESS_SYSTEM_PROMPT_MEMORY_LINE,
    )


def sanitize_tools_for_model(
    model_provider: ModelProvider,
    tools: list[Any],
) -> list[Any]:
    """Sanitize tools for model.

    Args:
        model_provider: The model provider value.
        tools: The tools value.

    Returns:
        The sanitized value.
    """
    if model_provider == "anthropic":
        return [normalize_anthropic_tool_schema(tool) for tool in tools]

    if model_provider not in OPENAI_COMPATIBLE_MODEL_PROVIDERS:
        return list(tools)

    compatible_tools: list[Any] = []
    skipped_tool_names: list[str] = []
    for candidate_tool in tools:
        if tool_supports_openai_compatible_schema(candidate_tool):
            compatible_tools.append(candidate_tool)
            continue
        skipped_tool_names.append(
            getattr(candidate_tool, "name", type(candidate_tool).__name__)
        )

    if skipped_tool_names:
        logger.warning(
            "Skipping %d tool(s) with non-object JSON schemas for OpenAI-compatible "
            "tool calling: %s",
            len(skipped_tool_names),
            ", ".join(skipped_tool_names),
        )

    return compatible_tools


def normalize_anthropic_tool_schema(tool: Any) -> Any:
    """Normalize tool schemas for Anthropic's stricter root object requirement.

    Args:
        tool: The tool value.

    Returns:
        The normalized tool value.
    """
    schema = getattr(tool, "args_schema", None)
    if not isinstance(schema, dict):
        return tool

    normalized_schema = normalize_json_object_schema_root(schema)
    if normalized_schema is schema:
        return tool

    if hasattr(tool, "model_copy"):
        return tool.model_copy(update={"args_schema": normalized_schema})

    try:
        cloned = copy.copy(tool)
        setattr(cloned, "args_schema", normalized_schema)
        return cloned
    except Exception:
        setattr(tool, "args_schema", normalized_schema)
        return tool


def normalize_json_object_schema_root(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure a JSON schema dict declares an object root when unspecified.

    Args:
        schema: The schema value.

    Returns:
        The normalized schema.
    """
    if schema.get("type") == "object":
        return schema

    if schema.get("type") is not None:
        return schema

    return {**schema, "type": "object"}


def tool_supports_openai_compatible_schema(tool: Any) -> bool:
    """Return whether a tool schema is OpenAI-compatible.

    Args:
        tool: The tool value.

    Returns:
        Whether a tool schema is OpenAI-compatible.
    """
    try:
        schema = convert_to_openai_tool(tool)
    except Exception:
        return False

    parameters = schema.get("function", {}).get("parameters")
    return isinstance(parameters, dict) and parameters.get("type") == "object"


def nested_child_subagents(
    subagent: SubagentConfig,
    registry: dict[str, SubagentConfig],
) -> tuple[SubagentConfig, ...]:
    """Return inline and referenced nested child subagents in config order."""
    return (
        *subagent.subagents,
        *(registry[name] for name in subagent.nested_subagent_names),
    )


def has_nested_child_subagents(subagent: SubagentConfig) -> bool:
    """Return whether a sync subagent exposes child subagents."""
    return bool(subagent.subagents or subagent.nested_subagent_names)


def has_background_subagent(subagents: tuple[SubagentConfig, ...]) -> bool:
    """Return whether a configured tree contains a background-capable agent."""
    return any(
        subagent.background or has_background_subagent(subagent.subagents)
        for subagent in subagents
    )


def inherited_tools_for_model(
    *,
    inherited_tools: list[Any],
    sanitized_inherited_tools: list[Any] | None = None,
    inherited_provider: ModelProvider,
    effective_provider: ModelProvider,
) -> list[Any]:
    """Return inherited tools adjusted for a subagent's effective model provider."""
    if effective_provider == inherited_provider:
        return list(
            sanitized_inherited_tools
            if sanitized_inherited_tools is not None
            else inherited_tools
        )
    return sanitize_tools_for_model(effective_provider, list(inherited_tools))


def reasoning_level_for_profile(
    model_profile: ModelDefaults,
    fallback: ReasoningLevel,
    *,
    fallback_is_explicit: bool = False,
) -> ReasoningLevel:
    """Return the reasoning level to use when building a profile-backed model."""
    if fallback_is_explicit:
        return fallback
    if "reasoning_effort" in model_profile.explicit_fields:
        return model_profile.reasoning_effort
    if (
        not model_profile.explicit_fields
        and model_profile.reasoning_effort != DEFAULT_REASONING_LEVEL
    ):
        return model_profile.reasoning_effort
    return fallback


ModelBuilder = Callable[[ReasoningLevel, ModelDefaults], Any]


@dataclass(frozen=True)
class _SubagentBuildContext:
    """Per-build settings shared by every sync subagent in one agent tree."""

    config: RuntimeConfig
    registry: dict[str, SubagentConfig]
    backend: Any
    project_root: Path | None
    reasoning_level_is_explicit: bool
    build_model: ModelBuilder
    subagent_mcp_tools: Mapping[tuple[str, ...], list[Any]] | None
    store: Any | None
    checkpointer: Any | None
    background_manager: runtime_background_tasks.BackgroundTaskManager | None
    session_id: str | None
    langsmith_tracing: runtime_tracing.LangSmithTracing | None

    def background_task_tools(
        self,
        *,
        subagents: dict[str, Any],
        agent_path: tuple[str, ...],
        existing_tools: list[Any],
    ) -> list[Any]:
        """Create task tools for one caller's background-capable children."""
        manager = self.background_manager
        if manager is None:
            return []
        return runtime_background_tasks.create_background_task_tools(
            manager=manager,
            subagents=subagents,
            agent_path=agent_path,
            recursion_limit=self.config.recursion_limit,
            session_generation=(
                manager.session_generation(self.session_id)
                if self.session_id
                else None
            ),
            batch_output_store=(
                runtime_background_tasks.create_batch_result_output_store(
                    self.backend,
                    backend_prefix=runtime_backends.generated_outputs_route_prefix(
                        self.project_root
                    ),
                )
            ),
            existing_tools=existing_tools,
            langsmith_tracing=self.langsmith_tracing,
        )


def _default_model_builder(config: RuntimeConfig) -> ModelBuilder:
    """Return a model builder that resolves models from runtime config."""
    return lambda reasoning_level, model_profile: (
        runtime_models.build_model_for_profile(config, reasoning_level, model_profile)
    )


def _build_sync_subagent_spec(
    context: _SubagentBuildContext,
    subagent: SubagentConfig,
    *,
    inherited_tools: list[Any],
    sanitized_inherited_tools: list[Any] | None,
    inherited_model: ModelDefaults,
    reasoning_level: ReasoningLevel,
    agent_path: tuple[str, ...],
) -> dict[str, Any]:
    """Build one sync subagent spec, compiling it when it has children."""
    config = context.config
    effective_model = runtime_models.resolve_runtime_model_profile(
        config,
        subagent.model,
        inherited_model=inherited_model,
    )
    effective_reasoning_level = reasoning_level_for_profile(
        effective_model,
        reasoning_level,
        fallback_is_explicit=context.reasoning_level_is_explicit,
    )
    # Without MCP discovery (exported graphs), configured servers add no tools
    # and the subagent inherits its caller's tools.
    has_configured_own_tools = (
        context.subagent_mcp_tools is not None and bool(subagent.mcp_servers)
    )
    raw_own_tools = (
        list(context.subagent_mcp_tools.get(agent_path, ()))
        if context.subagent_mcp_tools is not None
        else []
    )
    own_tools = sanitize_tools_for_model(effective_model.provider, raw_own_tools)
    inherited_model_tools = inherited_tools_for_model(
        inherited_tools=inherited_tools,
        sanitized_inherited_tools=sanitized_inherited_tools,
        inherited_provider=inherited_model.provider,
        effective_provider=effective_model.provider,
    )
    effective_tools = (
        own_tools
        if has_configured_own_tools
        else own_tools or inherited_model_tools
    )
    middleware = runtime_middleware.build_agent_middleware(
        backend=context.backend,
        config=config,
        reasoning_level=effective_reasoning_level,
        model_name=effective_model.name,
        source=subagent.name,
        project_root=context.project_root,
    )
    background_enabled = context.background_manager is not None
    child_subagents = nested_child_subagents(subagent, context.registry)
    target_background_enabled = background_enabled and subagent.background
    background_child_names = {
        child.name for child in child_subagents if child.background
    }
    caller_background_enabled = background_enabled and bool(background_child_names)
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
            context.build_model(effective_reasoning_level, effective_model)
            if subagent.model
            else None
        )
        return subagent.to_deepagents_spec(
            tools=subagent_tools,
            middleware=middleware,
            model=subagent_model,
        )

    child_specs = [
        _build_sync_subagent_spec(
            context,
            child,
            inherited_tools=(
                raw_own_tools if has_configured_own_tools else inherited_tools
            ),
            sanitized_inherited_tools=effective_tools,
            inherited_model=effective_model,
            reasoning_level=effective_reasoning_level,
            agent_path=(*agent_path, child.name),
        )
        for child in child_subagents
    ]
    if not child_specs:
        middleware.append(runtime_middleware.DisableSubagentDelegationMiddleware())
    background_tools = (
        context.background_task_tools(
            subagents={
                spec["name"]: spec["runnable"]
                for spec in child_specs
                if spec["name"] in background_child_names
            },
            agent_path=agent_path,
            existing_tools=effective_tools,
        )
        if caller_background_enabled
        else []
    )
    runnable_kwargs: dict[str, Any] = {
        "model": context.build_model(effective_reasoning_level, effective_model),
        "tools": [*effective_tools, *background_tools] or None,
        "system_prompt": subagent.system_prompt,
        "middleware": middleware,
        "backend": context.backend,
        "skills": list(subagent.skills) or None,
        "subagents": child_specs,
    }
    if context.store is not None:
        runnable_kwargs["store"] = context.store
    if context.checkpointer is not None:
        runnable_kwargs["checkpointer"] = context.checkpointer
    runnable = runtime_middleware.create_deep_agent_with_configured_summarization(
        config,
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


def _build_subagent_specs(
    context: _SubagentBuildContext,
    *,
    inherited_tools: list[Any],
    sanitized_inherited_tools: list[Any] | None,
    inherited_model: ModelDefaults,
    reasoning_level: ReasoningLevel,
    include_async_subagents: bool,
    async_subagent_url_override: str | None,
) -> list[Any]:
    """Build top-level sync specs, then async specs when requested."""
    subagent_specs: list[Any] = [
        _build_sync_subagent_spec(
            context,
            subagent,
            inherited_tools=list(inherited_tools),
            sanitized_inherited_tools=sanitized_inherited_tools,
            inherited_model=inherited_model,
            reasoning_level=reasoning_level,
            agent_path=(subagent.name,),
        )
        for subagent in context.config.extensions.subagents
    ]
    if include_async_subagents:
        subagent_specs.extend(
            subagent.to_deepagents_spec(url_override=async_subagent_url_override)
            for subagent in context.config.extensions.async_subagents
        )
    return subagent_specs


def build_subagent_specs(
    config: RuntimeConfig,
    *,
    backend: Any,
    project_root: Path | None,
    inherited_tools: list[Any],
    inherited_model: ModelDefaults,
    reasoning_level: ReasoningLevel,
    reasoning_level_is_explicit: bool,
    include_async_subagents: bool,
    sanitized_inherited_tools: list[Any] | None = None,
    async_subagent_url_override: str | None = None,
    subagent_mcp_tools: Mapping[tuple[str, ...], list[Any]] | None = None,
    build_model: ModelBuilder | None = None,
    store: Any | None = None,
    checkpointer: Any | None = None,
    background_manager: runtime_background_tasks.BackgroundTaskManager | None = None,
    session_id: str | None = None,
    langsmith_tracing: runtime_tracing.LangSmithTracing | None = None,
) -> list[Any]:
    """Build the subagent specs for one agent.

    Args:
        config: Configuration object used by the operation.
        backend: DeepAgents backend shared with compiled nested subgraphs.
        project_root: Project root used to resolve runtime middleware context.
        inherited_tools: Unsanitized tools inherited from the owning agent.
        inherited_model: Model profile of the owning agent.
        reasoning_level: Reasoning level of the owning agent.
        reasoning_level_is_explicit: Whether the reasoning level overrides profiles.
        include_async_subagents: Whether to include async subagents.
        sanitized_inherited_tools: Inherited tools already adapted to the owner's
            model provider, reused when a subagent keeps that provider.
        async_subagent_url_override: Agent Protocol URL for async subagents.
        subagent_mcp_tools: Each sync subagent's own MCP tools keyed by agent
            path, or None when MCP discovery is unavailable.
        build_model: Builds a chat model from a reasoning level and profile.
        store: LangGraph store for compiled subagents, if persistence is enabled.
        checkpointer: LangGraph checkpointer for compiled subagents.
        background_manager: Background task manager, or None when disabled.
        session_id: Session whose background generation scopes task tools.
        langsmith_tracing: LangSmith tracing for background task runs.

    Returns:
        The constructed subagent specs.
    """
    context = _SubagentBuildContext(
        config=config,
        registry={subagent.name: subagent for subagent in config.extensions.subagents},
        backend=backend,
        project_root=project_root,
        reasoning_level_is_explicit=reasoning_level_is_explicit,
        build_model=build_model or _default_model_builder(config),
        subagent_mcp_tools=subagent_mcp_tools,
        store=store,
        checkpointer=checkpointer,
        background_manager=background_manager,
        session_id=session_id,
        langsmith_tracing=langsmith_tracing,
    )
    return _build_subagent_specs(
        context,
        inherited_tools=inherited_tools,
        sanitized_inherited_tools=sanitized_inherited_tools,
        inherited_model=inherited_model,
        reasoning_level=reasoning_level,
        include_async_subagents=include_async_subagents,
        async_subagent_url_override=async_subagent_url_override,
    )


def build_graph_subagent_specs(
    config: RuntimeConfig,
    *,
    include_async_subagents: bool,
    backend: Any | None = None,
    project_root: Path | None = None,
    inherited_tools: list[Any] | None = None,
    background_manager: runtime_background_tasks.BackgroundTaskManager | None = None,
    artifact_registry: runtime_artifacts.LargeToolResultArtifactRegistry | None = None,
    langsmith_tracing: runtime_tracing.LangSmithTracing | None = None,
) -> list[Any]:
    """Build graph subagent specs for the configured default model.

    Args:
        config: Configuration object used by the operation.
        include_async_subagents: Whether to include async subagents.
        backend: DeepAgents backend shared with compiled nested subgraphs.
        project_root: Project root used to resolve runtime middleware context.
        inherited_tools: Tools inherited from the graph that owns these subagents.

    Returns:
        The constructed graph subagent specs.
    """
    resolved_backend = backend or runtime_backends.build_deepagent_backend(
        project_root=project_root,
        include_memories=config.agent_state == "stateful",
        memory_namespace=config.extensions.agent_memory_namespace,
        artifact_registry=artifact_registry,
    )
    inherited_model = runtime_models.resolve_runtime_model_profile(config)
    return build_subagent_specs(
        config,
        backend=resolved_backend,
        project_root=project_root,
        inherited_tools=list(inherited_tools or []),
        inherited_model=inherited_model,
        reasoning_level=reasoning_level_for_profile(
            inherited_model,
            config.default_reasoning,
            fallback_is_explicit=config.model_reasoning_override,
        ),
        reasoning_level_is_explicit=config.model_reasoning_override,
        include_async_subagents=include_async_subagents,
        background_manager=background_manager,
        langsmith_tracing=langsmith_tracing,
    )


def stateful_agent_memory_files(config: RuntimeConfig) -> list[str] | None:
    """Return startup memory files for stateful agents.

    Args:
        config: Configuration object used by the operation.

    Returns:
        The configured memory file paths, or None when startup memory is disabled.
    """
    if config.agent_state != "stateful" or not config.extensions.agent_memory_files:
        return None
    return list(config.extensions.agent_memory_files)


def build_main_tools(
    config: RuntimeConfig,
    *,
    mcp_tools: list[Any],
    rag_service: WorkspaceDocsRAG | None,
    thread_id: str | None = None,
) -> list[Any]:
    """Return the main agent's unsanitized tools.

    Args:
        config: Configuration object used by the operation.
        mcp_tools: Tools discovered from the main agent's MCP servers.
        rag_service: Workspace knowledge service, or None when RAG is off.
        thread_id: Conversation thread that scopes uploaded RAG documents.

    Returns:
        The main agent tool list.
    """
    tools = list(mcp_tools)
    if config.extensions.chainlit_generative_ui_enabled:
        tools.append(runtime_commands.create_render_chainlit_ui_tool())
    if rag_service is not None:
        tools.append(
            create_search_workspace_knowledge_tool(rag_service, thread_id=thread_id)
        )
    return tools


def build_agent_kwargs(
    config: RuntimeConfig,
    *,
    tools: list[Any],
    model_profile: ModelDefaults,
    reasoning_level: ReasoningLevel,
    reasoning_level_is_explicit: bool,
    system_prompt: str,
    custom_instruction: str | None,
    rag_enabled: bool,
    project_root: Path,
    artifact_registry: runtime_artifacts.LargeToolResultArtifactRegistry,
    include_async_subagents: bool,
    model_name: str | None = None,
    async_subagent_url_override: str | None = None,
    subagent_mcp_tools: Mapping[tuple[str, ...], list[Any]] | None = None,
    build_model: ModelBuilder | None = None,
    store: Any | None = None,
    checkpointer: Any | None = None,
    background_manager: runtime_background_tasks.BackgroundTaskManager | None = None,
    session_id: str | None = None,
    langsmith_tracing: runtime_tracing.LangSmithTracing | None = None,
) -> dict[str, Any]:
    """Build the DeepAgents keyword arguments for a configured main agent.

    Exported graphs and the live runtime share this assembly and differ only in
    the arguments they pass (MCP tools, persistence, sessions and model choice).

    Args:
        config: Configuration object used by the operation.
        tools: Unsanitized main agent tools (see ``build_main_tools``).
        model_profile: Resolved main agent model profile.
        reasoning_level: Effective main agent reasoning level.
        reasoning_level_is_explicit: Whether the reasoning level overrides profiles.
        system_prompt: Base system prompt before state, instruction and RAG notes.
        custom_instruction: Custom user instruction appended to the prompt.
        rag_enabled: Whether the workspace knowledge tool is available.
        project_root: Project root used to resolve local paths.
        artifact_registry: Registry for large tool result artifacts.
        include_async_subagents: Whether to include async subagents.
        model_name: Selected model name reported by the main agent middleware.
        async_subagent_url_override: Agent Protocol URL for async subagents.
        subagent_mcp_tools: Each sync subagent's own MCP tools keyed by agent
            path, or None when MCP discovery is unavailable.
        build_model: Builds a chat model from a reasoning level and profile.
        store: LangGraph store, if persistence is enabled.
        checkpointer: LangGraph checkpointer, if persistence is enabled.
        background_manager: Background task manager, or None when disabled.
        session_id: Session whose background generation scopes task tools.
        langsmith_tracing: LangSmith tracing for background task runs.

    Returns:
        Keyword arguments for ``create_deep_agent_with_configured_summarization``.
    """
    model_builder = build_model or _default_model_builder(config)
    model = model_builder(reasoning_level, model_profile)
    main_tools = sanitize_tools_for_model(model_profile.provider, tools)
    backend = runtime_backends.build_deepagent_backend(
        project_root=project_root,
        include_memories=config.agent_state == "stateful",
        memory_namespace=config.extensions.agent_memory_namespace,
        artifact_registry=artifact_registry,
    )
    middleware = runtime_middleware.build_agent_middleware(
        backend=backend,
        config=config,
        reasoning_level=reasoning_level,
        model_name=model_name,
        source="main-agent",
        project_root=project_root,
    )
    context = _SubagentBuildContext(
        config=config,
        registry={subagent.name: subagent for subagent in config.extensions.subagents},
        backend=backend,
        project_root=project_root,
        reasoning_level_is_explicit=reasoning_level_is_explicit,
        build_model=model_builder,
        subagent_mcp_tools=subagent_mcp_tools,
        store=store,
        checkpointer=checkpointer,
        background_manager=background_manager,
        session_id=session_id,
        langsmith_tracing=langsmith_tracing,
    )
    subagent_specs = _build_subagent_specs(
        context,
        inherited_tools=tools,
        sanitized_inherited_tools=main_tools,
        inherited_model=model_profile,
        reasoning_level=reasoning_level,
        include_async_subagents=include_async_subagents,
        async_subagent_url_override=async_subagent_url_override,
    )
    background_subagent_names = {
        subagent.name
        for subagent in config.extensions.subagents
        if subagent.background
    }
    background_subagents = {
        spec["name"]: spec["runnable"]
        for spec in subagent_specs
        if "runnable" in spec and spec["name"] in background_subagent_names
    }
    background_tools = (
        context.background_task_tools(
            subagents=background_subagents,
            agent_path=(),
            existing_tools=main_tools,
        )
        if background_subagents
        else []
    )
    agent_kwargs: dict[str, Any] = {
        "model": model,
        "tools": [*main_tools, *background_tools] or None,
        "system_prompt": compose_rag_system_prompt(
            compose_agent_system_prompt(
                system_prompt_for_agent_state(system_prompt, config.agent_state),
                custom_instruction,
                project_root=project_root,
            ),
            rag_enabled=rag_enabled,
        ),
        "middleware": middleware,
        "backend": backend,
        "skills": list(config.extensions.skills) or None,
        "subagents": subagent_specs or None,
    }
    memory_files = stateful_agent_memory_files(config)
    if memory_files is not None:
        agent_kwargs["memory"] = memory_files
    if store is not None:
        agent_kwargs["store"] = store
    if checkpointer is not None:
        agent_kwargs["checkpointer"] = checkpointer
    return agent_kwargs


def create_configured_graph(
    *,
    include_async_subagents: bool,
    system_prompt: str = SYSTEM_PROMPT,
    apply_custom_instruction: bool = False,
) -> Any:
    """Create configured graph.

    Args:
        include_async_subagents: Whether to include async subagents.
        system_prompt: The system prompt value.
        apply_custom_instruction: The apply custom instruction value.

    Returns:
        The created configured graph.
    """
    config = RuntimeConfig.from_env()
    langsmith_tracing = runtime_tracing.build_langsmith_tracing(config.langsmith)
    if langsmith_tracing is not None:
        _STATIC_LANGSMITH_TRACINGS.add(langsmith_tracing)
    rag_service = (
        WorkspaceDocsRAG(config.rag, project_root=runtime_constants.PROJECT_ROOT)
        if config.rag is not None
        else None
    )
    if rag_service is None and config.rag_requested and config.rag_error:
        logger.warning("RAG is configured but unavailable: %s", config.rag_error)
    main_model_profile = runtime_models.resolve_runtime_model_profile(config)
    session_manager = _get_static_background_task_manager(config)
    agent_kwargs = build_agent_kwargs(
        config,
        tools=build_main_tools(config, mcp_tools=[], rag_service=rag_service),
        model_profile=main_model_profile,
        reasoning_level=reasoning_level_for_profile(
            main_model_profile,
            config.default_reasoning,
            fallback_is_explicit=config.model_reasoning_override,
        ),
        reasoning_level_is_explicit=config.model_reasoning_override,
        system_prompt=system_prompt,
        custom_instruction=(
            config.extensions.custom_instruction if apply_custom_instruction else None
        ),
        rag_enabled=rag_service is not None,
        project_root=runtime_constants.PROJECT_ROOT,
        artifact_registry=_STATIC_LARGE_TOOL_RESULT_ARTIFACTS,
        include_async_subagents=include_async_subagents,
        background_manager=(
            session_manager
            if config.extensions.background_subagents.enabled
            else None
        ),
        langsmith_tracing=langsmith_tracing,
    )
    graph = runtime_middleware.create_deep_agent_with_configured_summarization(
        config,
        **agent_kwargs,
    )
    return runtime_background_tasks.scope_background_session_invocation(
        graph,
        session_manager,
        on_session_open=lambda _session_id: _STATIC_BACKGROUND_TASK_MANAGERS.add(
            session_manager
        ),
        artifact_registry=_STATIC_LARGE_TOOL_RESULT_ARTIFACTS,
        run_config_transform=(
            langsmith_tracing.with_callback if langsmith_tracing is not None else None
        ),
    )
