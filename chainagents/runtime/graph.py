"""Tool schema adaptation and static DeepAgents graph construction."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

from langchain_core.utils.function_calling import convert_to_openai_tool

import chainagents.runtime.backends as runtime_backends
import chainagents.runtime.commands as runtime_commands
import chainagents.runtime.constants as runtime_constants
import chainagents.runtime.middleware as runtime_middleware
import chainagents.runtime.models as runtime_models
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


def build_static_sync_subagent_spec(
    config: RuntimeConfig,
    subagent: SubagentConfig,
    *,
    registry: dict[str, SubagentConfig],
    backend: Any,
    inherited_tools: list[Any],
    reasoning_level: ReasoningLevel,
    inherited_model: ModelDefaults,
    project_root: Path | None,
    reasoning_level_is_explicit: bool = False,
) -> dict[str, Any]:
    """Build a sync subagent spec for configured graph creation."""
    effective_model = runtime_models.resolve_runtime_model_profile(
        config,
        subagent.model,
        inherited_model=inherited_model,
    )
    effective_reasoning_level = reasoning_level_for_profile(
        effective_model,
        reasoning_level,
        fallback_is_explicit=reasoning_level_is_explicit,
    )
    inherited_model_tools = inherited_tools_for_model(
        inherited_tools=inherited_tools,
        inherited_provider=inherited_model.provider,
        effective_provider=effective_model.provider,
    )
    effective_tools = inherited_model_tools
    middleware = runtime_middleware.build_agent_middleware(
        backend=backend,
        config=config,
        reasoning_level=effective_reasoning_level,
        model_name=effective_model.name,
        source=subagent.name,
        project_root=project_root,
    )
    if not has_nested_child_subagents(subagent):
        subagent_model = (
            runtime_models.build_model_for_profile(
                config,
                effective_reasoning_level,
                effective_model,
            )
            if subagent.model
            else None
        )
        subagent_tools = (
            inherited_model_tools
            if subagent.model and effective_model.provider != inherited_model.provider
            else []
        )
        return subagent.to_deepagents_spec(
            tools=subagent_tools,
            middleware=middleware,
            model=subagent_model,
        )

    child_specs = [
        build_static_sync_subagent_spec(
            config,
            child,
            registry=registry,
            backend=backend,
            inherited_tools=effective_tools,
            reasoning_level=effective_reasoning_level,
            reasoning_level_is_explicit=reasoning_level_is_explicit,
            inherited_model=effective_model,
            project_root=project_root,
        )
        for child in nested_child_subagents(subagent, registry)
    ]
    runnable_kwargs: dict[str, Any] = {
        "model": runtime_models.build_model_for_profile(
            config,
            effective_reasoning_level,
            effective_model,
        ),
        "tools": effective_tools or None,
        "system_prompt": subagent.system_prompt,
        "middleware": middleware,
        "backend": backend,
        "skills": list(subagent.skills) or None,
        "subagents": child_specs or None,
    }
    runnable = runtime_middleware.create_deep_agent_with_configured_summarization(
        config,
        **runnable_kwargs,
    )
    return {
        "name": subagent.name,
        "description": subagent.description,
        "runnable": runnable,
    }


def build_graph_subagent_specs(
    config: RuntimeConfig,
    *,
    include_async_subagents: bool,
    backend: Any | None = None,
    project_root: Path | None = None,
    inherited_tools: list[Any] | None = None,
) -> list[Any]:
    """Build graph subagent specs.

    Args:
        config: Configuration object used by the operation.
        include_async_subagents: Whether to include async subagents.
        backend: DeepAgents backend shared with compiled nested subgraphs.
        project_root: Project root used to resolve runtime middleware context.
        inherited_tools: Tools inherited from the graph that owns these subagents.

    Returns:
        The constructed graph subagent specs.
    """
    registry = {subagent.name: subagent for subagent in config.extensions.subagents}
    resolved_backend = backend or runtime_backends.build_deepagent_backend(
        project_root=project_root,
        include_memories=config.agent_state == "stateful",
        memory_namespace=config.extensions.agent_memory_namespace,
    )
    inherited_model = runtime_models.resolve_runtime_model_profile(config)
    graph_reasoning_level = reasoning_level_for_profile(
        inherited_model,
        config.default_reasoning,
        fallback_is_explicit=config.model_reasoning_override,
    )
    subagent_specs: list[Any] = [
        build_static_sync_subagent_spec(
            config,
            subagent,
            registry=registry,
            backend=resolved_backend,
            inherited_tools=list(inherited_tools or []),
            reasoning_level=graph_reasoning_level,
            reasoning_level_is_explicit=config.model_reasoning_override,
            inherited_model=inherited_model,
            project_root=project_root,
        )
        for subagent in config.extensions.subagents
    ]
    if include_async_subagents:
        subagent_specs.extend(
            subagent.to_deepagents_spec()
            for subagent in config.extensions.async_subagents
        )
    return subagent_specs


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
    backend = runtime_backends.build_deepagent_backend(
        include_memories=config.agent_state == "stateful",
        memory_namespace=config.extensions.agent_memory_namespace,
    )
    tools: list[Any] = []
    if config.extensions.chainlit_generative_ui_enabled:
        tools.append(runtime_commands.create_render_chainlit_ui_tool())
    if config.rag is not None:
        tools.append(
            create_search_workspace_knowledge_tool(
                WorkspaceDocsRAG(config.rag, project_root=runtime_constants.PROJECT_ROOT)
            )
        )
    else:
        if config.rag_requested and config.rag_error:
            logger.warning("RAG is configured but unavailable: %s", config.rag_error)
    main_model_profile = runtime_models.resolve_runtime_model_profile(config)
    main_tools = sanitize_tools_for_model(main_model_profile.provider, tools)
    main_reasoning_level = reasoning_level_for_profile(
        main_model_profile,
        config.default_reasoning,
        fallback_is_explicit=config.model_reasoning_override,
    )
    subagent_specs = build_graph_subagent_specs(
        config,
        include_async_subagents=include_async_subagents,
        backend=backend,
        project_root=runtime_constants.PROJECT_ROOT,
        inherited_tools=main_tools,
    )
    agent_kwargs: dict[str, Any] = {
        "model": runtime_models.build_model_for_profile(
            config,
            main_reasoning_level,
            main_model_profile,
        ),
        "tools": main_tools or None,
        "system_prompt": compose_rag_system_prompt(
            compose_agent_system_prompt(
                system_prompt_for_agent_state(system_prompt, config.agent_state),
                (
                    config.extensions.custom_instruction
                    if apply_custom_instruction
                    else None
                ),
                project_root=runtime_constants.PROJECT_ROOT,
            ),
            rag_enabled=config.rag is not None,
        ),
        "middleware": runtime_middleware.build_agent_middleware(
            backend=backend,
            config=config,
            reasoning_level=main_reasoning_level,
            source="main-agent",
            project_root=runtime_constants.PROJECT_ROOT,
        ),
        "backend": backend,
        "skills": list(config.extensions.skills) or None,
        "subagents": subagent_specs or None,
    }
    memory_files = stateful_agent_memory_files(config)
    if memory_files is not None:
        agent_kwargs["memory"] = memory_files
    return runtime_middleware.create_deep_agent_with_configured_summarization(config, **agent_kwargs)
