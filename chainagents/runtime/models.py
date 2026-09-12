"""Model selection and configured provider construction."""

from __future__ import annotations

import inspect
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

import chainagents.runtime.model_config as runtime_model_config
from chainagents.runtime.config import RuntimeConfig
from chainagents.runtime.constants import (
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_REASONING_LEVEL,
    DEFAULT_TEMPERATURE,
    ModelThinking,
    ReasoningLevel,
)
from chainagents.runtime.providers import (
    AnthropicDefaultQueryChatAnthropic,
    OpenAICompatibleChatOpenAI,
    SnowflakeCortexChatOpenAI,
)
from chainagents.runtime.types import ModelDefaults


def runtime_default_model_profile(config: RuntimeConfig) -> ModelDefaults:
    """Return the default model profile represented by flattened runtime fields."""
    provider = runtime_model_config.normalize_model_provider(
        getattr(config, "model_provider", None),
        default=DEFAULT_MODEL_PROVIDER,
    )
    default_base_url = (
        DEFAULT_ANTHROPIC_BASE_URL
        if provider == "anthropic"
        else (DEFAULT_OLLAMA_BASE_URL if provider == "ollama" else "")
    )
    return ModelDefaults(
        provider=provider,
        base_url=str(getattr(config, "model_base_url", None) or default_base_url),
        endpoint_query=tuple(getattr(config, "model_endpoint_query", ())),
        name=str(
            getattr(config, "model_default_name", None)
            or getattr(config, "model_name", DEFAULT_MODEL)
        ),
        api_key=getattr(config, "model_api_key", None),
        models=tuple(
            getattr(config, "model_default_choices", ())
            or getattr(config, "model_choices", ())
        ),
        name_is_explicit=True,
        reasoning_effort=runtime_model_config.normalize_reasoning_level(
            getattr(config, "default_reasoning", DEFAULT_REASONING_LEVEL),
        ),
        thinking=runtime_model_config.normalize_model_thinking(getattr(config, "model_thinking", None)),
        temperature=runtime_model_config.normalize_model_temperature(
            getattr(config, "model_temperature", DEFAULT_TEMPERATURE)
        ),
        repeat_penalty=runtime_model_config.normalize_repeat_penalty(
            getattr(config, "model_repeat_penalty", None)
        ),
        disable_streaming=runtime_model_config.normalize_disable_streaming(
            getattr(config, "model_disable_streaming", False)
        ),
        modalities=runtime_model_config.normalize_model_modalities(
            list(getattr(config, "model_modalities", ("text",))),
        ),
        cross_provider_base_url=getattr(
            config,
            "model_cross_provider_base_url",
            None,
        ),
        cross_provider_endpoint_url=getattr(
            config,
            "model_cross_provider_endpoint_url",
            None,
        ),
        cross_provider_endpoint_query=tuple(
            getattr(config, "model_cross_provider_endpoint_query", ())
        ),
        runtime_override_fields=(
            frozenset(
                {
                    field_name
                    for field_name, enabled in (
                        ("base_url", getattr(config, "model_base_url_override", False)),
                        (
                            "cross_provider_base_url",
                            bool(
                                getattr(
                                    config,
                                    "model_cross_provider_base_url_override",
                                    False,
                                )
                                and not getattr(
                                    config,
                                    "model_cross_provider_endpoint_url",
                                    None,
                                )
                            ),
                        ),
                        (
                            "cross_provider_endpoint_url",
                            bool(
                                getattr(
                                    config,
                                    "model_cross_provider_base_url_override",
                                    False,
                                )
                                and getattr(
                                    config,
                                    "model_cross_provider_endpoint_url",
                                    None,
                                )
                            ),
                        ),
                        (
                            "temperature",
                            getattr(config, "model_temperature_override", False),
                        ),
                        (
                            "disable_streaming",
                            getattr(
                                config,
                                "model_disable_streaming_override",
                                False,
                            ),
                        ),
                    )
                    if enabled
                }
            )
        ),
    )


def resolve_runtime_model_profile(
    config: RuntimeConfig,
    model_name: str | None = None,
    *,
    inherited_model: ModelDefaults | None = None,
) -> ModelDefaults:
    """Resolve a runtime profile-or-model reference."""
    if model_name is not None:
        model_ref = model_name
    elif inherited_model is not None:
        model_ref = None
    else:
        model_ref = config.model_name
    return runtime_model_config.resolve_model_profile_defaults(
        runtime_default_model_profile(config),
        getattr(config, "model_profiles", {}),
        model_ref,
        inherited_model=inherited_model,
    )


def model_api_key_for_profile(
    config: RuntimeConfig,
    model_profile: ModelDefaults,
) -> str | None:
    """Return the effective API key for a resolved model profile."""
    if config.model_api_key_override:
        return config.model_api_key_override
    if model_profile.provider == "anthropic":
        provider_key = runtime_model_config.normalize_optional_string(os.getenv("ANTHROPIC_API_KEY"))
        if provider_key:
            return provider_key
    if model_profile.provider == "snowflake_cortex":
        provider_key = runtime_model_config.normalize_optional_string(os.getenv("SNOWFLAKE_PAT"))
        if provider_key:
            return provider_key
    generic_key = runtime_model_config.normalize_optional_string(os.getenv("DEEPAGENT_MODEL_API_KEY"))
    if generic_key:
        return generic_key
    if model_profile.api_key:
        return model_profile.api_key
    if model_profile.provider == config.model_provider and config.model_api_key:
        return config.model_api_key
    return None


def build_model(
    config: RuntimeConfig,
    reasoning_level: ReasoningLevel,
    *,
    model_name: str | None = None,
    model_profile: ModelDefaults | None = None,
) -> Any:
    """Build model.

    Args:
        config: Configuration object used by the operation.
        reasoning_level: The reasoning level value.
        model_name: The model name or profile reference.
        model_profile: Already resolved model profile settings.

    Returns:
        The constructed model.
    """
    resolved_profile = model_profile or resolve_runtime_model_profile(
        config,
        model_name,
    )
    selected_model = resolved_profile.name
    if resolved_profile.provider == "ollama":
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "base_url": resolved_profile.base_url,
            "reasoning": reasoning_level,
            "temperature": resolved_profile.temperature,
            "disable_streaming": resolved_profile.disable_streaming,
        }
        if resolved_profile.repeat_penalty is not None:
            kwargs["repeat_penalty"] = resolved_profile.repeat_penalty
        return ChatOllama(**kwargs)

    api_key = model_api_key_for_profile(config, resolved_profile)
    if resolved_profile.provider == "anthropic":
        if not api_key:
            raise ValueError(
                "Anthropic runtime requires DEEPAGENT_MODEL_API_KEY, "
                "ANTHROPIC_API_KEY, or [model].api_key."
            )
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "base_url": resolved_profile.base_url,
            "temperature": resolved_profile.temperature,
            "effort": reasoning_level,
            "disable_streaming": resolved_profile.disable_streaming,
        }
        if should_enable_anthropic_adaptive_thinking(
            selected_model,
            resolved_profile.thinking,
        ):
            kwargs["thinking"] = {"type": "adaptive"}
        kwargs["api_key"] = api_key
        default_query = runtime_model_config.model_endpoint_query_to_dict(resolved_profile.endpoint_query)
        if default_query:
            kwargs["default_query"] = default_query
            return AnthropicDefaultQueryChatAnthropic(**kwargs)
        return ChatAnthropic(**kwargs)

    if resolved_profile.provider == "snowflake_cortex":
        if not api_key:
            raise ValueError(
                "Snowflake Cortex runtime requires a CLI API key, SNOWFLAKE_PAT, "
                "DEEPAGENT_MODEL_API_KEY, or [model].api_key."
            )
        kwargs = {
            "model": selected_model,
            "base_url": resolved_profile.base_url,
            "api_key": api_key,
            "temperature": resolved_profile.temperature,
            "disable_streaming": resolved_profile.disable_streaming,
            "extra_body": {"reasoning": {"effort": reasoning_level}},
        }
        default_query = runtime_model_config.model_endpoint_query_to_dict(resolved_profile.endpoint_query)
        if default_query:
            kwargs["default_query"] = default_query
        return SnowflakeCortexChatOpenAI(**kwargs)

    kwargs: dict[str, Any] = {
        "model": selected_model,
        "base_url": resolved_profile.base_url,
        "api_key": api_key or "deepagent",
        "temperature": resolved_profile.temperature,
        "disable_streaming": resolved_profile.disable_streaming,
    }
    default_query = runtime_model_config.model_endpoint_query_to_dict(resolved_profile.endpoint_query)
    if default_query:
        kwargs["default_query"] = default_query
    return OpenAICompatibleChatOpenAI(**kwargs)


def build_model_for_profile(
    config: RuntimeConfig,
    reasoning_level: ReasoningLevel,
    model_profile: ModelDefaults,
) -> Any:
    """Build a model from resolved profile settings."""
    try:
        parameters = inspect.signature(build_model).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "model_profile" in parameters:
        return build_model(
            config,
            reasoning_level,
            model_profile=model_profile,
        )
    return build_model(config, reasoning_level, model_name=model_profile.name)


def should_enable_anthropic_adaptive_thinking(
    model_name: str,
    thinking: ModelThinking,
) -> bool:
    """Return whether adaptive thinking should be enabled for Anthropic.

    Args:
        model_name: The model name value.
        thinking: The configured thinking mode.

    Returns:
        Whether adaptive thinking should be enabled.
    """
    if thinking == "disabled":
        return False
    if thinking == "adaptive":
        return True
    return anthropic_model_supports_adaptive_thinking(model_name)


def anthropic_model_supports_adaptive_thinking(model_name: str) -> bool:
    """Return whether an Anthropic model supports adaptive thinking.

    Args:
        model_name: The model name value.

    Returns:
        Whether the model supports adaptive thinking.
    """
    normalized = model_name.lower()
    return any(
        marker in normalized
        for marker in (
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-opus-4-7",
            "claude-opus-4-8",
        )
    )
