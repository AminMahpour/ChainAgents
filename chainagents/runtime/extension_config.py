"""MCP, skills, subagent, and Chainlit extension configuration."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

import chainagents.runtime.constants as runtime_constants
import chainagents.runtime.model_config as runtime_model_config
from chainagents.runtime.constants import (
    AGENT_MEMORY_NAMESPACE_RE,
    DEFAULT_AGENT_MEMORY_FILES,
    DEFAULT_AGENT_MEMORY_NAMESPACE,
    DEFAULT_AGENT_STATE,
    DEFAULT_RECURSION_LIMIT,
    AgentStateMode,
)
from chainagents.runtime.reflection import normalize_reflection_config
from chainagents.runtime.types import (
    AsyncSubagentConfig,
    ChainlitCommandConfig,
    ChainlitStarterConfig,
    ExtensionsConfig,
    SubagentConfig,
)


def normalize_agent_state(value: Any | None) -> AgentStateMode:
    """Normalize agent state mode.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized agent state mode.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None or str(value).strip() == "":
        return DEFAULT_AGENT_STATE
    candidate = str(value).strip().lower().replace("-", "_")
    if candidate == "stateful":
        return "stateful"
    if candidate == "stateless":
        return "stateless"
    raise ValueError(
        "The top-level 'agent.state' config must be 'stateful' or 'stateless'."
    )


def normalize_agent_memory_namespace(value: Any | None) -> str:
    """Normalize the shared agent memory namespace.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized agent memory namespace.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return DEFAULT_AGENT_MEMORY_NAMESPACE
    if not isinstance(value, str):
        raise ValueError(
            "The top-level 'agent.memory_namespace' config must be a non-empty string."
        )
    candidate = value.strip()
    if not candidate:
        raise ValueError(
            "The top-level 'agent.memory_namespace' config must be a non-empty string."
        )
    if AGENT_MEMORY_NAMESPACE_RE.fullmatch(candidate) is None:
        raise ValueError(
            "The top-level 'agent.memory_namespace' config may only contain "
            "alphanumeric characters, hyphens, underscores, dots, @, +, colons, "
            "and tildes."
        )
    return candidate


def normalize_agent_memory_files(value: Any | None) -> tuple[str, ...]:
    """Normalize startup memory files loaded by DeepAgents.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized memory file paths.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return DEFAULT_AGENT_MEMORY_FILES
    if not isinstance(value, list):
        raise ValueError(
            "The top-level 'agent.memory_files' config must be an array of /memories/ paths."
        )

    memory_files: list[str] = []
    for index, raw_path in enumerate(value, start=1):
        if not isinstance(raw_path, str):
            raise ValueError(
                f"The top-level 'agent.memory_files' entry #{index} must be a string."
            )
        memory_path = raw_path.strip()
        if not memory_path.startswith("/memories/") or memory_path == "/memories/":
            raise ValueError(
                "The top-level 'agent.memory_files' entries must be absolute "
                "/memories/ file paths."
            )
        memory_files.append(memory_path)
    return tuple(memory_files)


def normalize_recursion_limit(
    value: Any | None,
    *,
    default: int = DEFAULT_RECURSION_LIMIT,
    field_name: str = "recursion_limit",
) -> int:
    """Normalize recursion limit.

    Args:
        value: Value to normalize, convert, or serialize.
        default: Fallback value used when no explicit value is available.
        field_name: The field name value.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None or str(value).strip() == "":
        return default

    try:
        recursion_limit = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc

    if recursion_limit <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return recursion_limit


def resolve_local_path(path_value: str, base_dir: Path) -> Path:
    """Resolve local path.

    Args:
        path_value: The path value value.
        base_dir: The base dir value.

    Returns:
        The resolved local path.
    """
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def normalize_mcp_transport(value: str) -> str:
    """Normalize MCP transport.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized value.
    """
    transport = value.strip().lower()
    if transport == "streamable-http":
        return "streamable_http"
    return transport


def normalize_skill_source_path(path_value: str, base_dir: Path) -> str:
    """Normalize skill source path.

    Args:
        path_value: The path value value.
        base_dir: The base dir value.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    raw = path_value.strip()
    if not raw:
        raise ValueError("Skill source paths cannot be empty.")

    normalized = raw.replace("\\", "/")
    if normalized.startswith("/"):
        path = PurePosixPath(normalized).as_posix()
        return path if path.endswith("/") else f"{path}/"

    resolved = resolve_local_path(raw, base_dir)
    try:
        relative = resolved.relative_to(runtime_constants.PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Skill source path '{path_value}' must stay inside the project root "
            f"({runtime_constants.PROJECT_ROOT}) or be given as an explicit virtual path like /workspace/skills/."
        ) from exc

    virtual_path = (PurePosixPath("/workspace") / PurePosixPath(relative.as_posix())).as_posix()
    return virtual_path if virtual_path.endswith("/") else f"{virtual_path}/"


def normalize_mcp_server_config(raw_server: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Normalize MCP server config.

    Args:
        raw_server: Raw server to process.
        base_dir: The base dir value.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    server = dict(raw_server)
    transport = normalize_mcp_transport(str(server.get("transport", "")).strip())
    if not transport:
        raise ValueError("Each MCP server must define a non-empty 'transport'.")
    server["transport"] = transport

    if "command" in server:
        server["command"] = str(server["command"]).strip()
    if "args" in server:
        server["args"] = [str(arg) for arg in server.get("args", [])]
    if "cwd" in server and server["cwd"]:
        server["cwd"] = str(resolve_local_path(str(server["cwd"]), base_dir))
    if "headers" in server and server["headers"] is not None:
        server["headers"] = {str(k): str(v) for k, v in server["headers"].items()}
    if "env" in server and server["env"] is not None:
        server["env"] = {str(k): str(v) for k, v in server["env"].items()}
    if "url" in server:
        server["url"] = str(server["url"]).strip()

    return server


def normalize_string_mapping(
    value: Any | None,
    *,
    field_name: str,
) -> dict[str, str] | None:
    """Normalize string mapping.

    Args:
        value: Value to normalize, convert, or serialize.
        field_name: The field name value.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be a table/object.")
    return {str(key): str(raw_value) for key, raw_value in value.items()}


def parse_async_subagent_config(
    raw_subagent: dict[str, Any],
    *,
    index: int,
    source_name: str,
) -> AsyncSubagentConfig:
    """Parse async subagent config.

    Args:
        raw_subagent: Raw subagent to process.
        index: The index value.
        source_name: The source name value.

    Returns:
        The parsed async subagent config.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    name = str(raw_subagent.get("name", "")).strip()
    description = str(raw_subagent.get("description", "")).strip()
    graph_id = str(raw_subagent.get("graph_id", "")).strip()
    if not name or not description or not graph_id:
        raise ValueError(
            f"{source_name} entry #{index} must include non-empty "
            "'name', 'description', and 'graph_id'."
        )

    unsupported_fields = sorted(
        field
        for field in (
            "system_prompt",
            "system_prompt_file",
            "skills",
            "mcp_servers",
            "model",
        )
        if field in raw_subagent
    )
    if unsupported_fields:
        raise ValueError(
            f"Async subagent '{name}' cannot define sync-only field(s): "
            f"{', '.join(unsupported_fields)}."
        )

    return AsyncSubagentConfig(
        name=name,
        description=description,
        graph_id=graph_id,
        url=runtime_model_config.normalize_optional_string(
            runtime_model_config.normalize_model_base_url(raw_subagent.get("url"))
        ),
        headers=normalize_string_mapping(
            raw_subagent.get("headers"),
            field_name=f"async subagent '{name}' headers",
        ),
    )


def parse_sync_subagent_config(
    raw_subagent: dict[str, Any],
    *,
    index: int,
    base_dir: Path,
    mcp_servers: dict[str, dict[str, Any]],
    parent_name: str | None = None,
) -> SubagentConfig:
    """Parse sync subagent config.

    Args:
        raw_subagent: Raw subagent to process.
        index: The index value.
        base_dir: The base dir value.
        mcp_servers: The MCP servers value.
        parent_name: Parent subagent name for nested entries.

    Returns:
        The parsed sync subagent config.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    name = str(raw_subagent.get("name", "")).strip()
    description = str(raw_subagent.get("description", "")).strip()
    if not name or not description:
        raise ValueError(
            f"Subagent entry #{index} must include non-empty 'name' and 'description'."
        )
    if parent_name and "graph_id" in raw_subagent:
        raise ValueError(
            f"Subagent '{parent_name}' nested async subagents are not supported; "
            f"nested subagent '{name or index}' defines 'graph_id'."
        )

    inline_prompt = raw_subagent.get("system_prompt")
    prompt_file = raw_subagent.get("system_prompt_file")
    if inline_prompt and prompt_file:
        raise ValueError(
            f"Subagent '{name}' cannot define both 'system_prompt' and 'system_prompt_file'."
        )
    if prompt_file:
        prompt_path = resolve_local_path(str(prompt_file), base_dir)
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    else:
        system_prompt = str(inline_prompt or "").strip()
    if not system_prompt:
        raise ValueError(
            f"Subagent '{name}' must include 'system_prompt' or 'system_prompt_file'."
        )

    raw_subagent_skill_paths = raw_subagent.get("skills", [])
    subagent_skill_paths = tuple(
        normalize_skill_source_path(str(path_value), base_dir)
        for path_value in raw_subagent_skill_paths
    )
    raw_subagent_mcp_servers = tuple(
        str(server_name).strip()
        for server_name in raw_subagent.get("mcp_servers", [])
        if str(server_name).strip()
    )
    for server_name in raw_subagent_mcp_servers:
        if server_name not in mcp_servers:
            raise ValueError(
                f"Subagent '{name}' references unknown MCP server '{server_name}'. "
                f"Defined servers: {sorted(mcp_servers)}"
            )

    nested_subagent_names = normalize_required_string_list(
        raw_subagent.get("nested_subagents", []),
        field_name=f"subagent '{name}' nested_subagents",
    )
    raw_nested_subagents = raw_subagent.get("subagents", [])
    if not isinstance(raw_nested_subagents, list):
        raise ValueError(
            f"Subagent '{name}' nested 'subagents' config must be an array of tables."
        )
    nested_subagents: list[SubagentConfig] = []
    for nested_index, raw_nested_subagent in enumerate(raw_nested_subagents, start=1):
        if not isinstance(raw_nested_subagent, dict):
            raise ValueError(
                f"Subagent '{name}' nested subagent entry #{nested_index} must be a table/object."
            )
        nested_subagents.append(
            parse_sync_subagent_config(
                raw_nested_subagent,
                index=nested_index,
                base_dir=base_dir,
                mcp_servers=mcp_servers,
                parent_name=name,
            )
        )

    model = str(raw_subagent.get("model", "")).strip() or None
    return SubagentConfig(
        name=name,
        description=description,
        system_prompt=system_prompt,
        skills=subagent_skill_paths,
        mcp_servers=raw_subagent_mcp_servers,
        model=model,
        nested_subagent_names=nested_subagent_names,
        subagents=tuple(nested_subagents),
    )


def normalize_required_string_list(
    value: Any,
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Normalize a config field that must be a list of non-empty strings."""
    if not isinstance(value, list):
        raise ValueError(f"'{field_name}' must be a list of strings.")

    items: list[str] = []
    for index, raw_item in enumerate(value, start=1):
        item = str(raw_item).strip()
        if not item:
            raise ValueError(
                f"'{field_name}' entry #{index} must be a non-empty string."
            )
        items.append(item)
    return tuple(items)


def validate_subagent_names(
    subagents: tuple[SubagentConfig, ...],
    async_subagents: tuple[AsyncSubagentConfig, ...],
) -> None:
    """Validate top-level subagent name uniqueness across sync and async specs."""
    seen_names: set[str] = set()
    for subagent in (*subagents, *async_subagents):
        if subagent.name in seen_names:
            raise ValueError(
                f"Top-level subagent name '{subagent.name}' is defined more than once."
            )
        seen_names.add(subagent.name)


def validate_nested_subagent_references(
    subagents: tuple[SubagentConfig, ...],
) -> None:
    """Validate nested sync subagent references and cycles."""
    registry = {subagent.name: subagent for subagent in subagents}
    for subagent in subagents:
        validate_nested_subagent_reference_tree(
            subagent,
            registry=registry,
            path=(subagent.name,),
        )


def validate_nested_subagent_reference_tree(
    subagent: SubagentConfig,
    *,
    registry: dict[str, SubagentConfig],
    path: tuple[str, ...],
) -> None:
    """Validate one subagent's direct children and referenced descendants."""
    direct_child_names: set[str] = set()
    for child in subagent.subagents:
        if child.name in direct_child_names:
            raise ValueError(
                f"Subagent '{subagent.name}' has duplicate nested child subagent "
                f"'{child.name}'."
            )
        direct_child_names.add(child.name)

    for referenced_name in subagent.nested_subagent_names:
        if referenced_name not in registry:
            raise ValueError(
                f"Subagent '{subagent.name}' references unknown nested subagent "
                f"'{referenced_name}'. Defined subagents: {sorted(registry)}"
            )
        if referenced_name in direct_child_names:
            raise ValueError(
                f"Subagent '{subagent.name}' has duplicate nested child subagent "
                f"'{referenced_name}'."
            )
        if referenced_name in path:
            cycle = " -> ".join((*path, referenced_name))
            raise ValueError(f"nested subagent cycle detected: {cycle}")
        direct_child_names.add(referenced_name)

    for child in subagent.subagents:
        validate_nested_subagent_reference_tree(
            child,
            registry=registry,
            path=(*path, child.name),
        )
    for referenced_name in subagent.nested_subagent_names:
        validate_nested_subagent_reference_tree(
            registry[referenced_name],
            registry=registry,
            path=(*path, referenced_name),
        )


def parse_agent_custom_instruction(
    agent_section: dict[str, Any],
    base_dir: Path,
) -> str | None:
    """Parse inline or file-based main-agent custom instruction."""
    custom_instruction = runtime_model_config.normalize_optional_string(
        agent_section.get("custom_instruction")
    )
    custom_instruction_file = runtime_model_config.normalize_optional_string(
        agent_section.get("custom_instruction_file")
    )
    if custom_instruction and custom_instruction_file:
        raise ValueError(
            "The top-level 'agent' config cannot define both "
            "'custom_instruction' and 'custom_instruction_file'."
        )
    if not custom_instruction_file:
        return custom_instruction
    instruction_path = resolve_local_path(custom_instruction_file, base_dir)
    return instruction_path.read_text(encoding="utf-8").strip() or None


def parse_extensions_config(raw_config: dict[str, Any], config_path: Path) -> ExtensionsConfig:
    """Parse extensions config.

    Args:
        raw_config: Raw config to process.
        config_path: Path to the config.

    Returns:
        The parsed extensions config.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    base_dir = config_path.parent
    mcp_section = raw_config.get("mcp", {})
    if mcp_section and not isinstance(mcp_section, dict):
        raise ValueError("The top-level 'mcp' config must be a table/object.")

    agent_section = raw_config.get("agent", {})
    if agent_section and not isinstance(agent_section, dict):
        raise ValueError("The top-level 'agent' config must be a table/object.")

    recursion_limit = normalize_recursion_limit(
        agent_section.get("recursion_limit"),
        field_name="The top-level 'agent.recursion_limit' config",
    )
    agent_state = normalize_agent_state(agent_section.get("state"))
    agent_memory_namespace = normalize_agent_memory_namespace(
        agent_section.get("memory_namespace")
    )
    agent_memory_files = normalize_agent_memory_files(agent_section.get("memory_files"))
    raw_delete_tool_enabled = agent_section.get("delete_tool_enabled", False)
    if not isinstance(raw_delete_tool_enabled, bool):
        raise ValueError(
            "The top-level 'agent.delete_tool_enabled' config must be a boolean."
        )
    raw_execute_tool_enabled = agent_section.get("execute_tool_enabled", False)
    if not isinstance(raw_execute_tool_enabled, bool):
        raise ValueError(
            "The top-level 'agent.execute_tool_enabled' config must be a boolean."
        )
    agent_reflection = normalize_reflection_config(
        agent_section.get("reflection"),
        agent_state=agent_state,
    )
    agent_model = runtime_model_config.normalize_optional_string(agent_section.get("model"))
    raw_mcp_servers = mcp_section.get("servers", {})
    mcp_servers: dict[str, dict[str, Any]] = {}
    for name, raw_server in raw_mcp_servers.items():
        if not isinstance(raw_server, dict):
            raise ValueError(f"MCP server '{name}' must be a table/object.")
        mcp_servers[str(name)] = normalize_mcp_server_config(raw_server, base_dir)

    raw_skill_paths = agent_section.get("skills", [])
    custom_instruction = parse_agent_custom_instruction(agent_section, base_dir)
    raw_summarization_middleware_enabled = agent_section.get(
        "summarization_middleware_enabled",
        False,
    )
    raw_summarization_trigger_tokens = agent_section.get("summarization_trigger_tokens")
    raw_summarization_keep_tokens = agent_section.get("summarization_keep_tokens")
    if not isinstance(raw_summarization_middleware_enabled, bool):
        raise ValueError(
            "The top-level 'agent.summarization_middleware_enabled' config must be a boolean."
        )
    if raw_summarization_trigger_tokens is not None and (
        not isinstance(raw_summarization_trigger_tokens, int)
        or raw_summarization_trigger_tokens <= 0
    ):
        raise ValueError(
            "The top-level 'agent.summarization_trigger_tokens' config must be a positive integer."
        )
    if raw_summarization_keep_tokens is not None and (
        not isinstance(raw_summarization_keep_tokens, int)
        or raw_summarization_keep_tokens <= 0
    ):
        raise ValueError(
            "The top-level 'agent.summarization_keep_tokens' config must be a positive integer."
        )
    skill_paths = tuple(
        normalize_skill_source_path(str(path_value), base_dir)
        for path_value in raw_skill_paths
    )
    raw_agent_mcp_servers = tuple(
        str(server_name).strip()
        for server_name in agent_section.get("mcp_servers", [])
        if str(server_name).strip()
    )
    for server_name in raw_agent_mcp_servers:
        if server_name not in mcp_servers:
            raise ValueError(
                f"Agent references unknown MCP server '{server_name}'. "
                f"Defined servers: {sorted(mcp_servers)}"
            )

    raw_subagents = raw_config.get("subagents", [])
    if not isinstance(raw_subagents, list):
        raise ValueError("The top-level 'subagents' config must be an array of tables.")
    subagents: list[SubagentConfig] = []
    async_subagents: list[AsyncSubagentConfig] = []
    for index, raw_subagent in enumerate(raw_subagents, start=1):
        if not isinstance(raw_subagent, dict):
            raise ValueError(f"Subagent entry #{index} must be a table/object.")
        if "graph_id" in raw_subagent:
            async_subagents.append(
                parse_async_subagent_config(
                    raw_subagent,
                    index=index,
                    source_name="Subagent",
                )
            )
            continue
        subagents.append(
            parse_sync_subagent_config(
                raw_subagent,
                index=index,
                base_dir=base_dir,
                mcp_servers=mcp_servers,
            )
        )

    raw_async_subagents = raw_config.get("async_subagents", [])
    if not isinstance(raw_async_subagents, list):
        raise ValueError(
            "The top-level 'async_subagents' config must be an array of tables."
        )
    for index, raw_async_subagent in enumerate(raw_async_subagents, start=1):
        if not isinstance(raw_async_subagent, dict):
            raise ValueError(f"Async subagent entry #{index} must be a table/object.")
        async_subagents.append(
            parse_async_subagent_config(
                raw_async_subagent,
                index=index,
                source_name="Async subagent",
            )
        )

    validate_subagent_names(tuple(subagents), tuple(async_subagents))
    validate_nested_subagent_references(tuple(subagents))

    chainlit_section = raw_config.get("chainlit", {})
    if chainlit_section and not isinstance(chainlit_section, dict):
        raise ValueError("The top-level 'chainlit' config must be a table/object.")

    raw_chainlit_commands = chainlit_section.get("commands", [])
    if not isinstance(raw_chainlit_commands, list):
        raise ValueError("The top-level 'chainlit.commands' config must be an array of tables.")
    raw_chainlit_starters = chainlit_section.get("starters", [])
    if not isinstance(raw_chainlit_starters, list):
        raise ValueError("The top-level 'chainlit.starters' config must be an array of tables.")
    raw_reasoning_mode_enabled = chainlit_section.get("reasoning_mode_enabled", True)
    raw_reasoning_steps_enabled = chainlit_section.get("reasoning_steps_enabled", True)
    raw_tool_steps_enabled = chainlit_section.get("tool_steps_enabled", True)
    raw_model_mode_enabled = chainlit_section.get("model_mode_enabled", True)
    raw_startup_status_enabled = chainlit_section.get("startup_status_enabled", True)
    raw_chronological_ui_enabled = chainlit_section.get("chronological_ui_enabled", True)
    raw_generative_ui_enabled = chainlit_section.get("generative_ui_enabled", True)
    if not isinstance(raw_reasoning_mode_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.reasoning_mode_enabled' config must be a boolean."
        )
    if not isinstance(raw_reasoning_steps_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.reasoning_steps_enabled' config must be a boolean."
        )
    if not isinstance(raw_tool_steps_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.tool_steps_enabled' config must be a boolean."
        )
    if not isinstance(raw_model_mode_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.model_mode_enabled' config must be a boolean."
        )
    if not isinstance(raw_startup_status_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.startup_status_enabled' config must be a boolean."
        )
    if not isinstance(raw_chronological_ui_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.chronological_ui_enabled' config must be a boolean."
        )
    if not isinstance(raw_generative_ui_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.generative_ui_enabled' config must be a boolean."
        )

    chainlit_commands: list[ChainlitCommandConfig] = []
    seen_commands: set[str] = set()
    for index, raw_chainlit_command in enumerate(raw_chainlit_commands, start=1):
        if not isinstance(raw_chainlit_command, dict):
            raise ValueError(
                f"Chainlit command entry #{index} must be a table/object."
            )
        name = str(raw_chainlit_command.get("name", "")).strip().lstrip("/").lower()
        description = str(raw_chainlit_command.get("description", "")).strip()
        target = str(raw_chainlit_command.get("target", "")).strip().lower()
        value = str(raw_chainlit_command.get("value", "")).strip()
        template = runtime_model_config.normalize_optional_string(raw_chainlit_command.get("template"))
        mcp_server = runtime_model_config.normalize_optional_string(raw_chainlit_command.get("mcp_server"))
        if not name or " " in name:
            raise ValueError(
                f"Chainlit command entry #{index} must define a slash-compatible 'name' with no spaces."
            )
        if name in seen_commands:
            raise ValueError(f"Chainlit command '/{name}' is defined more than once.")
        if not description:
            raise ValueError(f"Chainlit command '/{name}' must include a non-empty 'description'.")
        if target not in {"prompt", "subagent", "mcp_tool"}:
            raise ValueError(
                f"Chainlit command '/{name}' target must be one of: prompt, subagent, mcp_tool."
            )
        if not value:
            raise ValueError(f"Chainlit command '/{name}' must include a non-empty 'value'.")
        if target == "subagent":
            valid_subagent_names = {subagent.name for subagent in subagents}
            if value not in valid_subagent_names:
                raise ValueError(
                    f"Chainlit command '/{name}' references unknown subagent '{value}'. "
                    f"Defined subagents: {sorted(valid_subagent_names)}"
                )
        if target == "mcp_tool" and mcp_server and mcp_server not in mcp_servers:
            raise ValueError(
                f"Chainlit command '/{name}' references unknown MCP server '{mcp_server}'. "
                f"Defined servers: {sorted(mcp_servers)}"
            )
        chainlit_commands.append(
            ChainlitCommandConfig(
                name=name,
                description=description,
                target=target,  # type: ignore[arg-type]
                value=value,
                template=template,
                mcp_server=mcp_server,
            )
        )
        seen_commands.add(name)

    chainlit_starters: list[ChainlitStarterConfig] = []
    for index, raw_chainlit_starter in enumerate(raw_chainlit_starters, start=1):
        if not isinstance(raw_chainlit_starter, dict):
            raise ValueError(
                f"Chainlit starter entry #{index} must be a table/object."
            )
        label = str(raw_chainlit_starter.get("label", "")).strip()
        message = str(raw_chainlit_starter.get("message", "")).strip()
        command = runtime_model_config.normalize_optional_string(raw_chainlit_starter.get("command"))
        icon = runtime_model_config.normalize_optional_string(raw_chainlit_starter.get("icon"))
        if not label:
            raise ValueError(
                f"Chainlit starter entry #{index} must include a non-empty 'label'."
            )
        if not message:
            raise ValueError(
                f"Chainlit starter '{label}' must include a non-empty 'message'."
            )
        chainlit_starters.append(
            ChainlitStarterConfig(
                label=label,
                message=message,
                command=command,
                icon=icon,
            )
        )

    return ExtensionsConfig(
        config_path=config_path,
        mcp_tool_name_prefix=bool(mcp_section.get("tool_name_prefix", True)),
        mcp_stateful=bool(mcp_section.get("stateful", False)),
        agent_state=agent_state,
        agent_memory_namespace=agent_memory_namespace,
        agent_memory_files=agent_memory_files,
        delete_tool_enabled=raw_delete_tool_enabled,
        execute_tool_enabled=raw_execute_tool_enabled,
        agent_reflection=agent_reflection,
        agent_model=agent_model,
        recursion_limit=recursion_limit,
        mcp_servers=mcp_servers or None,
        skills=skill_paths,
        agent_mcp_servers=raw_agent_mcp_servers,
        subagents=tuple(subagents),
        async_subagents=tuple(async_subagents),
        chainlit_commands=tuple(chainlit_commands),
        chainlit_starters=tuple(chainlit_starters),
        chainlit_model_mode_enabled=raw_model_mode_enabled,
        chainlit_reasoning_mode_enabled=raw_reasoning_mode_enabled,
        chainlit_reasoning_steps_enabled=raw_reasoning_steps_enabled,
        chainlit_tool_steps_enabled=raw_tool_steps_enabled,
        chainlit_startup_status_enabled=raw_startup_status_enabled,
        chainlit_chronological_ui_enabled=raw_chronological_ui_enabled,
        chainlit_generative_ui_enabled=raw_generative_ui_enabled,
        summarization_middleware_enabled=raw_summarization_middleware_enabled,
        summarization_trigger_tokens=raw_summarization_trigger_tokens,
        summarization_keep_tokens=raw_summarization_keep_tokens,
        custom_instruction=custom_instruction,
    )
