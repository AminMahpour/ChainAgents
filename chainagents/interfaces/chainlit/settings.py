"""Resolve Chainlit chat settings, modes, and model/reasoning defaults."""

from __future__ import annotations

from typing import Any

import chainlit as cl
from chainlit.input_widget import Select, Switch, TextInput

from chainagents.runtime import (
    DEFAULT_REASONING_LEVEL,
    AppSettings,
    ReasoningLevel,
    RuntimeConfig,
    normalize_reasoning_level,
    reasoning_level_for_profile,
    resolve_runtime_model_profile,
)

SESSION_SETTINGS_KEY = "agent_settings"
SESSION_MCP_SESSION_ID_KEY = "mcp_session_id"


def current_chainlit_thread_id() -> str:
    """Return the current chainlit thread ID.

    Returns:
        The current chainlit thread ID.
    """
    try:
        session = cl.context.session
    except Exception:
        return ""
    thread_id = getattr(session, "thread_id", None) or getattr(session, "id", None)
    return str(thread_id or "").strip()


def current_chainlit_session_id() -> str:
    """Return the current chainlit session ID.

    Returns:
        The current chainlit session ID.
    """
    try:
        session = cl.context.session
    except Exception:
        return ""
    return str(getattr(session, "id", None) or "").strip()


def store_mcp_session_id() -> str:
    """Store MCP session ID.

    Returns:
        The stored value.
    """
    session_id = current_chainlit_session_id() or current_chainlit_thread_id()
    cl.user_session.set(SESSION_MCP_SESSION_ID_KEY, session_id)
    return session_id


def current_mcp_session_id() -> str:
    """Return the current MCP session ID.

    Returns:
        The current MCP session ID.
    """
    session_id = str(cl.user_session.get(SESSION_MCP_SESSION_ID_KEY) or "").strip()
    if session_id:
        return session_id
    return store_mcp_session_id()


def settings_payload(settings: AppSettings) -> dict[str, Any]:
    """Build a serializable payload from Chainlit chat settings.

    Args:
        settings: The settings value.

    Returns:
        The constructed a serializable payload from chainlit chat settings.
    """
    return {
        "model_name": settings.model_name,
        "reasoning_level": settings.reasoning_level,
        "thread_id": settings.thread_id,
        "show_reasoning_stream": settings.show_reasoning_stream,
        "show_tool_calls": settings.show_tool_calls,
    }


def store_settings(settings: AppSettings) -> None:
    """Store settings.

    Args:
        settings: The settings value.
    """
    cl.user_session.set(SESSION_SETTINGS_KEY, settings_payload(settings))


def resolve_model_name(
    value: Any | None,
    *,
    available_models: tuple[str, ...],
    default: str,
) -> str:
    """Resolve model name.

    Args:
        value: Value to normalize, convert, or serialize.
        available_models: The available models value.
        default: Fallback value used when no explicit value is available.

    Returns:
        The resolved model name.
    """
    candidate = str(value or "").strip()
    if candidate in available_models:
        return candidate
    return default


def coerce_bool_setting(value: Any | None, *, default: bool) -> bool:
    """Coerce a raw Chainlit setting value into a boolean."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    candidate = str(value).strip().lower()
    if candidate in {"1", "true", "yes", "on", "enabled"}:
        return True
    if candidate in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def build_chat_settings(
    settings: AppSettings,
    *,
    available_models: tuple[str, ...],
    model_mode_enabled: bool = True,
) -> cl.ChatSettings:
    """Build chat settings.

    Args:
        settings: The settings value.
        available_models: The available models value.
        model_mode_enabled: The model mode enabled value.

    Returns:
        The constructed chat settings.
    """
    reasoning_levels = ["low", "medium", "high"]
    inputs: list[Any] = []
    if model_mode_enabled:
        inputs.append(
            Select(
                id="model_name",
                label="Model",
                values=list(available_models),
                initial_index=available_models.index(settings.model_name),
                description="Select a configured model for this chat session.",
            )
        )
    inputs.extend(
        [
            Select(
                id="reasoning_level",
                label="Reasoning Level",
                values=reasoning_levels,
                initial_index=reasoning_levels.index(settings.reasoning_level),
                description=(
                    "Controls the configured model's reasoning setting when the active "
                    "provider supports it."
                ),
            ),
            TextInput(
                id="thread_id",
                label="LangGraph Thread ID",
                initial=settings.thread_id,
                description=(
                    "Defaults to the current Chainlit thread. Override it only if you want "
                    "to point this chat at a different persisted LangGraph thread."
                ),
            ),
            Switch(
                id="show_reasoning_stream",
                label="Show Reasoning Stream",
                initial=settings.show_reasoning_stream,
                description="Show streamed reasoning panels and reasoning task entries.",
            ),
            Switch(
                id="show_tool_calls",
                label="Show Tool Calls",
                initial=settings.show_tool_calls,
                description="Show streamed tool-call panels and tool task entries.",
            ),
        ]
    )
    return cl.ChatSettings(inputs)


def build_modes(
    settings: AppSettings,
    *,
    available_models: tuple[str, ...],
    model_mode_enabled: bool = True,
    reasoning_mode_enabled: bool = True,
) -> list[cl.Mode]:
    """Build modes.

    Args:
        settings: The settings value.
        available_models: The available models value.
        model_mode_enabled: The model mode enabled value.
        reasoning_mode_enabled: The reasoning mode enabled value.

    Returns:
        The constructed modes.
    """
    reasoning_levels = ["low", "medium", "high"]
    modes: list[cl.Mode] = []
    if model_mode_enabled:
        modes.append(
            cl.Mode(
                id="model_name",
                name="Model",
                options=[
                    cl.ModeOption(
                        id=model_name,
                        name=model_name,
                        description="Use this model for the current message.",
                        icon="bot",
                        default=model_name == settings.model_name,
                    )
                    for model_name in available_models
                ],
            )
        )
    if reasoning_mode_enabled:
        modes.append(
            cl.Mode(
                id="reasoning_level",
                name="Reasoning",
                options=[
                    cl.ModeOption(
                        id=level,
                        name=level.capitalize(),
                        description=(
                            "Deeper reasoning with higher latency"
                            if level == "high"
                            else (
                                "Balanced quality and speed"
                                if level == "medium"
                                else "Fastest responses with lighter reasoning"
                            )
                        ),
                        icon=(
                            "brain"
                            if level == "high"
                            else ("sparkles" if level == "medium" else "zap")
                        ),
                        default=level == settings.reasoning_level,
                    )
                    for level in reasoning_levels
                ],
            )
        )
    return modes


async def publish_modes(
    settings: AppSettings,
    *,
    available_models: tuple[str, ...],
    model_mode_enabled: bool = True,
    reasoning_mode_enabled: bool = True,
) -> None:
    """Publish modes.

    Args:
        settings: The settings value.
        available_models: The available models value.
        model_mode_enabled: The model mode enabled value.
        reasoning_mode_enabled: The reasoning mode enabled value.
    """
    try:
        await cl.context.emitter.set_modes(
            build_modes(
                settings,
                available_models=available_models,
                model_mode_enabled=model_mode_enabled,
                reasoning_mode_enabled=reasoning_mode_enabled,
            )
        )
    except Exception as exc:
        message = str(exc).lower()
        missing_modes_column = "modes" in message and (
            ("column" in message and "does not exist" in message)
            or "no such column" in message
        )
        if not missing_modes_column:
            raise


def coerce_settings(
    raw_settings: AppSettings | dict[str, Any] | None,
    *,
    default_model_name: str,
    available_models: tuple[str, ...],
    default_reasoning_level: ReasoningLevel = DEFAULT_REASONING_LEVEL,
    runtime_config: RuntimeConfig | None = None,
    show_reasoning_stream_default: bool = True,
    show_tool_calls_default: bool = True,
) -> AppSettings:
    """Coerce settings.

    Args:
        raw_settings: Raw settings to process.
        default_model_name: The default model name value.
        available_models: The available models value.
        default_reasoning_level: Reasoning default when settings omit it.
        runtime_config: Runtime config used to derive profile-aware defaults.
        show_reasoning_stream_default: Default reasoning stream visibility.
        show_tool_calls_default: Default tool-call visibility.

    Returns:
        The coerced value.
    """
    if raw_settings is None:
        raw_settings = {}
    if isinstance(raw_settings, AppSettings):
        return AppSettings(
            model_name=resolve_model_name(
                raw_settings.model_name,
                available_models=available_models,
                default=default_model_name,
            ),
            reasoning_level=normalize_reasoning_level(raw_settings.reasoning_level),
            thread_id=raw_settings.thread_id,
            show_reasoning_stream=coerce_bool_setting(
                getattr(raw_settings, "show_reasoning_stream", None),
                default=show_reasoning_stream_default,
            ),
            show_tool_calls=coerce_bool_setting(
                getattr(raw_settings, "show_tool_calls", None),
                default=show_tool_calls_default,
            ),
        )
    model_name = resolve_model_name(
        raw_settings.get("model_name"),
        available_models=available_models,
        default=default_model_name,
    )
    reasoning_default = (
        default_reasoning_level_for_model(runtime_config, model_name)
        if runtime_config is not None
        else default_reasoning_level
    )
    reasoning_level = normalize_reasoning_level(
        raw_settings.get("reasoning_level", reasoning_default),
        default=reasoning_default,
    )
    thread_id = str(
        raw_settings.get("thread_id") or current_chainlit_thread_id()
    ).strip()
    if not thread_id:
        thread_id = current_chainlit_thread_id()
    return AppSettings(
        model_name=model_name,
        reasoning_level=reasoning_level,
        thread_id=thread_id.strip(),
        show_reasoning_stream=coerce_bool_setting(
            raw_settings.get("show_reasoning_stream"),
            default=show_reasoning_stream_default,
        ),
        show_tool_calls=coerce_bool_setting(
            raw_settings.get("show_tool_calls"),
            default=show_tool_calls_default,
        ),
    )


def resolve_reasoning_level_for_message(
    message: cl.Message,
    settings: AppSettings,
    *,
    reasoning_mode_enabled: bool = True,
) -> ReasoningLevel:
    """Resolve reasoning level for message.

    Args:
        message: Chainlit message or LangChain message to process.
        settings: The settings value.
        reasoning_mode_enabled: The reasoning mode enabled value.

    Returns:
        The resolved reasoning level for message.
    """
    if not reasoning_mode_enabled:
        return settings.reasoning_level
    raw_modes = getattr(message, "modes", None)
    if not isinstance(raw_modes, dict):
        return settings.reasoning_level
    return normalize_reasoning_level(
        raw_modes.get("reasoning_level"),
        default=settings.reasoning_level,
    )


def message_has_reasoning_level_override(
    message: cl.Message,
    *,
    reasoning_mode_enabled: bool = True,
) -> bool:
    """Return whether a message explicitly selected a reasoning level."""
    if not reasoning_mode_enabled:
        return False
    raw_modes = getattr(message, "modes", None)
    return isinstance(raw_modes, dict) and raw_modes.get("reasoning_level") is not None


def default_reasoning_level_for_model(
    config: RuntimeConfig,
    model_name: str | None,
) -> ReasoningLevel:
    """Return the profile-aware reasoning default for a Chainlit model choice."""
    model_profile = resolve_runtime_model_profile(config, model_name)
    return reasoning_level_for_profile(
        model_profile,
        config.default_reasoning,
        fallback_is_explicit=config.model_reasoning_override,
    )


def settings_reasoning_level_is_explicit(
    config: RuntimeConfig,
    settings: AppSettings,
    model_name: str | None = None,
) -> bool:
    """Return whether chat settings override the selected model's reasoning default."""
    selected_model = model_name if model_name is not None else settings.model_name
    return (
        normalize_reasoning_level(settings.reasoning_level)
        != default_reasoning_level_for_model(config, selected_model)
    )


def resolve_model_name_for_message(
    message: cl.Message,
    settings: AppSettings,
    *,
    available_models: tuple[str, ...],
    model_mode_enabled: bool = True,
) -> str:
    """Resolve model name for message.

    Args:
        message: Chainlit message or LangChain message to process.
        settings: The settings value.
        available_models: The available models value.
        model_mode_enabled: The model mode enabled value.

    Returns:
        The resolved model name for message.
    """
    if not model_mode_enabled:
        return settings.model_name
    raw_modes = getattr(message, "modes", None)
    if not isinstance(raw_modes, dict):
        return settings.model_name
    return resolve_model_name(
        raw_modes.get("model_name"),
        available_models=available_models,
        default=settings.model_name,
    )
