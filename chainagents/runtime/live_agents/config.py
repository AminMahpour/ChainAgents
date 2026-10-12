"""Parsing and validation for the ``[agent.live_agents]`` config section."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from chainagents.runtime.constants import DEFAULT_DEEPAGENT_FILESYSTEM_TOOLS
from chainagents.runtime.types import (
    LiveAgentAllowlist,
    LiveAgentsConfig,
    LiveAgentTemplate,
)

LIVE_AGENT_COMMAND_NAMES: frozenset[str] = frozenset({"agents", "spawn", "msg", "kill"})

_PREFIX = "agent.live_agents"
_LIMIT_FIELDS = (
    "max_live_per_session",
    "max_live_total",
    "max_turns_per_agent",
    "turn_recursion_limit",
)
_BOOL_FIELDS = (
    "delete_history_on_teardown",
    "notify_main_on_user_actions",
    "stream_activity",
)
_ALLOW_LISTS = ("models", "mcp_servers", "skills", "tools")
_TEMPLATE_FIELDS = {
    "name",
    "description",
    "system_prompt",
    "system_prompt_file",
    "model",
    "tools",
    "skills",
    "mcp_servers",
}


def live_agent_filesystem_tools(
    *, delete_tool_enabled: bool, execute_tool_enabled: bool
) -> tuple[str, ...]:
    """Return the filesystem tool names a live agent may be granted."""
    tools = list(DEFAULT_DEEPAGENT_FILESYSTEM_TOOLS)
    if delete_tool_enabled:
        tools.append("delete")
    if execute_tool_enabled:
        tools.append("execute")
    return tuple(tools)


def _string_list(value: Any, *, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"The top-level '{path}' config must be an array of strings.")
    items: list[str] = []
    for item in value:
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"The top-level '{path}' config cannot contain empty strings.")
        if stripped not in items:
            items.append(stripped)
    return tuple(items)


def _check_members(
    items: tuple[str, ...],
    *,
    valid: Any,
    kind: str,
    path: str,
) -> None:
    for item in items:
        if item not in valid:
            raise ValueError(
                f"The top-level '{path}' config references unknown {kind} '{item}'. "
                f"Valid: {sorted(valid)}"
            )


def parse_live_agents_config(
    value: Any,
    *,
    base_dir: Path,
    mcp_servers: Mapping[str, Any],
    messaging_enabled: bool,
    delete_tool_enabled: bool,
    execute_tool_enabled: bool,
) -> LiveAgentsConfig:
    """Parse and validate the opt-in ``[agent.live_agents]`` section."""
    # Imported lazily: extension_config imports this module.
    from chainagents.runtime.extension_config import (
        normalize_skill_source_path,
        resolve_local_path,
    )

    if value is None:
        return LiveAgentsConfig()
    if not isinstance(value, dict):
        raise ValueError(f"The top-level '{_PREFIX}' config must be a table/object.")

    defaults = LiveAgentsConfig()
    known = {
        "enabled",
        "idle_timeout_seconds",
        "allow",
        "templates",
        *_LIMIT_FIELDS,
        *_BOOL_FIELDS,
    }
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"Unknown '{_PREFIX}' config field(s): {', '.join(unknown)}.")

    values: dict[str, Any] = {}
    for field_name in ("enabled", *_BOOL_FIELDS):
        raw = value.get(field_name, getattr(defaults, field_name))
        if not isinstance(raw, bool):
            raise ValueError(
                f"The top-level '{_PREFIX}.{field_name}' config must be a boolean."
            )
        values[field_name] = raw
    for field_name in _LIMIT_FIELDS:
        raw = value.get(field_name, getattr(defaults, field_name))
        if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
            raise ValueError(
                f"The top-level '{_PREFIX}.{field_name}' config must be a positive integer."
            )
        values[field_name] = raw
    idle = value.get("idle_timeout_seconds", defaults.idle_timeout_seconds)
    if not isinstance(idle, int) or isinstance(idle, bool) or idle < 0:
        raise ValueError(
            f"The top-level '{_PREFIX}.idle_timeout_seconds' config must be a "
            "non-negative integer (0 disables the idle timeout)."
        )
    values["idle_timeout_seconds"] = idle

    if values["enabled"] and not messaging_enabled:
        raise ValueError(
            f"'{_PREFIX}.enabled' requires 'agent.messaging.enabled' to be true."
        )

    valid_tools = live_agent_filesystem_tools(
        delete_tool_enabled=delete_tool_enabled,
        execute_tool_enabled=execute_tool_enabled,
    )

    # Allowlist.
    raw_allow = value.get("allow")
    if raw_allow is None:
        allow = LiveAgentAllowlist()
    else:
        allow_path = f"{_PREFIX}.allow"
        if not isinstance(raw_allow, dict):
            raise ValueError(f"The top-level '{allow_path}' config must be a table/object.")
        unknown_allow = sorted(
            set(raw_allow) - {"applies_to", "allow_custom_prompt", *_ALLOW_LISTS}
        )
        if unknown_allow:
            raise ValueError(
                f"Unknown '{allow_path}' config field(s): {', '.join(unknown_allow)}."
            )
        applies_to = raw_allow.get("applies_to", "all")
        if applies_to not in ("all", "overrides"):
            raise ValueError(
                f"The top-level '{allow_path}.applies_to' config must be 'all' or 'overrides'."
            )
        allow_custom_prompt = raw_allow.get("allow_custom_prompt", True)
        if not isinstance(allow_custom_prompt, bool):
            raise ValueError(
                f"The top-level '{allow_path}.allow_custom_prompt' config must be a boolean."
            )
        allow_models = _string_list(raw_allow.get("models"), path=f"{allow_path}.models")
        allow_servers = _string_list(
            raw_allow.get("mcp_servers"), path=f"{allow_path}.mcp_servers"
        )
        allow_skills = tuple(
            normalize_skill_source_path(item, base_dir)
            for item in _string_list(raw_allow.get("skills"), path=f"{allow_path}.skills")
        )
        allow_tools = _string_list(raw_allow.get("tools"), path=f"{allow_path}.tools")
        _check_members(
            allow_servers, valid=mcp_servers, kind="MCP server", path=f"{allow_path}.mcp_servers"
        )
        _check_members(allow_tools, valid=valid_tools, kind="tool", path=f"{allow_path}.tools")
        allow = LiveAgentAllowlist(
            applies_to=cast("Any", applies_to),
            models=allow_models,
            mcp_servers=allow_servers,
            skills=allow_skills,
            tools=allow_tools,
            allow_custom_prompt=allow_custom_prompt,
        )

    # Templates.
    raw_templates = value.get("templates", [])
    if not isinstance(raw_templates, list):
        raise ValueError(f"The top-level '{_PREFIX}.templates' config must be an array of tables.")
    templates: list[LiveAgentTemplate] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_templates, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Live agent template entry #{index} must be a table/object.")
        name = str(raw.get("name", "")).strip()
        description = str(raw.get("description", "")).strip()
        if not name or not description:
            raise ValueError(
                f"Live agent template entry #{index} must include non-empty "
                "'name' and 'description'."
            )
        if name in seen:
            raise ValueError(f"Live agent template '{name}' is defined more than once.")
        seen.add(name)
        unknown_tpl = sorted(set(raw) - _TEMPLATE_FIELDS)
        if unknown_tpl:
            raise ValueError(
                f"Unknown live agent template '{name}' field(s): {', '.join(unknown_tpl)}."
            )
        inline_prompt = raw.get("system_prompt")
        prompt_file = raw.get("system_prompt_file")
        if inline_prompt and prompt_file:
            raise ValueError(
                f"Live agent template '{name}' cannot define both "
                "'system_prompt' and 'system_prompt_file'."
            )
        if prompt_file:
            prompt_path = resolve_local_path(str(prompt_file), base_dir)
            system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        else:
            system_prompt = str(inline_prompt or "").strip()
        if not system_prompt:
            raise ValueError(
                f"Live agent template '{name}' must include "
                "'system_prompt' or 'system_prompt_file'."
            )
        tpl_path = f"{_PREFIX}.templates[{name}]"
        model = raw.get("model")
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise ValueError(
                    f"The top-level '{tpl_path}.model' config must be a nonempty string."
                )
            model = model.strip()
        tools = _string_list(raw.get("tools"), path=f"{tpl_path}.tools")
        skills = tuple(
            normalize_skill_source_path(item, base_dir)
            for item in _string_list(raw.get("skills"), path=f"{tpl_path}.skills")
        )
        servers = _string_list(raw.get("mcp_servers"), path=f"{tpl_path}.mcp_servers")
        _check_members(servers, valid=mcp_servers, kind="MCP server", path=f"{tpl_path}.mcp_servers")
        _check_members(tools, valid=valid_tools, kind="tool", path=f"{tpl_path}.tools")
        if allow.applies_to == "all":
            _check_members(
                tools, valid=allow.tools, kind="tool outside allowlist", path=f"{tpl_path}.tools"
            )
            _check_members(
                skills,
                valid=allow.skills,
                kind="skill outside allowlist",
                path=f"{tpl_path}.skills",
            )
            _check_members(
                servers,
                valid=allow.mcp_servers,
                kind="MCP server outside allowlist",
                path=f"{tpl_path}.mcp_servers",
            )
        templates.append(
            LiveAgentTemplate(
                name=name,
                description=description,
                system_prompt=system_prompt,
                model=model,
                tools=tools,
                skills=skills,
                mcp_servers=servers,
            )
        )

    return LiveAgentsConfig(allow=allow, templates=tuple(templates), **values)


def validate_live_agent_models(
    config: LiveAgentsConfig, model_profiles: Mapping[str, Any]
) -> None:
    """Check that referenced model profiles exist and respect the allowlist."""
    for model in config.allow.models:
        if model not in model_profiles:
            raise ValueError(
                f"The top-level '{_PREFIX}.allow.models' config references unknown "
                f"model profile '{model}'. Defined profiles: {sorted(model_profiles)}"
            )
    for template in config.templates:
        if template.model is None:
            continue
        if template.model not in model_profiles:
            raise ValueError(
                f"Live agent template '{template.name}' references unknown "
                f"model profile '{template.model}'. Defined profiles: {sorted(model_profiles)}"
            )
        if config.allow.applies_to == "all" and template.model not in config.allow.models:
            raise ValueError(
                f"Live agent template '{template.name}' uses model '{template.model}' "
                f"outside '{_PREFIX}.allow.models'."
            )
