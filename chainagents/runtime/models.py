"""Model selection and configured provider construction."""

from __future__ import annotations

import inspect
import os
import re
import warnings
from collections.abc import Mapping
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import (
    _get_default_model_profile as anthropic_model_profile,
)
from langchain_aws import ChatAnthropicBedrock, ChatBedrockConverse
from langchain_ollama import ChatOllama

import chainagents.runtime.model_config as runtime_model_config
from chainagents.runtime.config import RuntimeConfig
from chainagents.runtime.constants import (
    BEDROCK_INFERENCE_PROFILE_PREFIXES,
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
    EndpointChatAnthropicBedrock,
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
        max_tokens=runtime_model_config.normalize_model_max_tokens(
            getattr(config, "model_max_tokens", None)
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
        if resolved_profile.max_tokens is not None:
            kwargs["num_predict"] = resolved_profile.max_tokens
        return ChatOllama(**kwargs)

    if resolved_profile.provider == "bedrock":
        return build_bedrock_model(resolved_profile, reasoning_level)
    if resolved_profile.provider == "anthropic_bedrock":
        return build_anthropic_bedrock_model(resolved_profile, reasoning_level)

    api_key = model_api_key_for_profile(config, resolved_profile)
    if resolved_profile.provider == "anthropic":
        if not api_key:
            raise ValueError(
                "Anthropic runtime requires DEEPAGENT_MODEL_API_KEY, "
                "ANTHROPIC_API_KEY, or [model].api_key."
            )
        kwargs = {
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
        if resolved_profile.max_tokens is not None:
            kwargs["max_tokens"] = resolved_profile.max_tokens
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
        if resolved_profile.max_tokens is not None:
            kwargs["max_completion_tokens"] = resolved_profile.max_tokens
        default_query = runtime_model_config.model_endpoint_query_to_dict(resolved_profile.endpoint_query)
        if default_query:
            kwargs["default_query"] = default_query
        return SnowflakeCortexChatOpenAI(**kwargs)

    kwargs = {
        "model": selected_model,
        "base_url": resolved_profile.base_url,
        "api_key": api_key or "deepagent",
        "temperature": resolved_profile.temperature,
        "disable_streaming": resolved_profile.disable_streaming,
    }
    if resolved_profile.max_tokens is not None:
        kwargs["max_completion_tokens"] = resolved_profile.max_tokens
    default_query = runtime_model_config.model_endpoint_query_to_dict(resolved_profile.endpoint_query)
    if default_query:
        kwargs["default_query"] = default_query
    return OpenAICompatibleChatOpenAI(**kwargs)


def build_bedrock_model(
    model_profile: ModelDefaults,
    reasoning_level: ReasoningLevel,
) -> ChatBedrockConverse:
    """Build an Amazon Bedrock Converse model.

    Credentials and region come from the standard AWS chain (environment,
    shared config and profiles, SSO, or instance roles), so no API key is
    passed here.

    Args:
        model_profile: Resolved model profile settings.
        reasoning_level: The reasoning level value.

    Returns:
        The constructed Bedrock model.
    """
    validate_bedrock_temperature(model_profile.temperature)
    kwargs: dict[str, Any] = {
        "model_id": model_profile.name,
        "temperature": model_profile.temperature,
    }
    if model_profile.disable_streaming or "disable_streaming" in (
        model_profile.explicit_fields | model_profile.runtime_override_fields
    ):
        kwargs["disable_streaming"] = model_profile.disable_streaming
    # Otherwise langchain-aws picks the per-model default, e.g. "tool_calling"
    # for Bedrock models that cannot stream tool use.
    if model_profile.name.startswith("arn:"):
        kwargs.update(bedrock_arn_model_metadata(model_profile.name))
    if model_profile.base_url:
        kwargs["endpoint_url"] = model_profile.base_url
    if model_profile.max_tokens is not None:
        kwargs["max_tokens"] = model_profile.max_tokens
    if model_profile.thinking == "disabled":
        disabled = anthropic_bedrock_disabled_thinking_kwargs(
            anthropic_bedrock_base_model(model_profile.name)
        )
        if disabled:
            kwargs["additional_model_request_fields"] = disabled
    else:
        kwargs["reasoning_effort"] = reasoning_level
    with warnings.catch_warnings():
        if model_profile.thinking == "auto":
            # langchain-aws ignores reasoning_effort for models without
            # configurable reasoning; in auto mode that is expected, not noise.
            warnings.filterwarnings(
                "ignore",
                message="reasoning_effort is not supported",
            )
        model = ChatBedrockConverse(**kwargs)
    thinking = (model.additional_model_request_fields or {}).get("thinking")
    if isinstance(thinking, Mapping) and thinking.get("type") != "disabled":
        # Claude rejects non-default sampling temperatures while thinking.
        model.temperature = None
    return model


def validate_bedrock_temperature(temperature: float) -> None:
    """Reject temperatures outside the 0..1 range Bedrock accepts.

    Args:
        temperature: The configured sampling temperature.

    Raises:
        ValueError: If the temperature is outside 0..1.
    """
    if not 0 <= temperature <= 1:
        raise ValueError(
            f"Amazon Bedrock temperature must be between 0 and 1; got {temperature}."
        )


def build_anthropic_bedrock_model(
    model_profile: ModelDefaults,
    reasoning_level: ReasoningLevel,
) -> ChatAnthropicBedrock:
    """Build Claude on Amazon Bedrock through the Anthropic Messages API.

    Uses the Anthropic SDK's Bedrock client, so requests keep the Anthropic
    Messages format (thinking, effort, Anthropic content blocks) while
    credentials and region come from the standard AWS chain.

    Args:
        model_profile: Resolved model profile settings.
        reasoning_level: The reasoning level value.

    Returns:
        The constructed Claude-on-Bedrock model.
    """
    validate_bedrock_temperature(model_profile.temperature)
    validate_anthropic_bedrock_model_id(model_profile.name)
    base_model = anthropic_bedrock_base_model(model_profile.name)
    kwargs: dict[str, Any] = {
        "model": model_profile.name,
        "temperature": model_profile.temperature,
        "disable_streaming": model_profile.disable_streaming,
    }
    if model_profile.thinking == "disabled":
        kwargs.update(anthropic_bedrock_disabled_thinking_kwargs(base_model))
    else:
        # Claude rejects output_config.effort on models without effort support
        # (Claude 3.x, Haiku 4.5), so effort is only sent where it is accepted.
        # langchain-anthropic also turns effort into adaptive thinking on models
        # that support it, which is why it is skipped when thinking is disabled.
        if anthropic_model_profile(base_model).get("reasoning_effort_levels"):
            kwargs["effort"] = reasoning_level
        if should_enable_anthropic_adaptive_thinking(
            model_profile.name,
            model_profile.thinking,
        ):
            kwargs["thinking"] = {"type": "adaptive"}
    if model_profile.max_tokens is not None:
        kwargs["max_tokens"] = model_profile.max_tokens
    if os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        # The Anthropic SDK rejects a Bedrock API key combined with SigV4
        # credentials, which langchain-aws would otherwise read from the env.
        kwargs.update(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
        )
    model: ChatAnthropicBedrock
    if model_profile.base_url:
        model = EndpointChatAnthropicBedrock(
            bedrock_endpoint_url=model_profile.base_url,
            **kwargs,
        )
    else:
        model = ChatAnthropicBedrock(**kwargs)
    thinking_enabled = (
        "thinking" in kwargs and kwargs["thinking"].get("type") != "disabled"
    )
    if thinking_enabled or (
        "effort" in kwargs
        and "xhigh" in ((model.profile or {}).get("reasoning_effort_levels") or ())
    ):
        # Claude rejects non-default sampling temperatures while thinking.
        model.temperature = None
    return model


def anthropic_bedrock_base_model(model_id: str) -> str:
    """Return the Anthropic model name behind a Bedrock Claude model ID.

    Strips the ARN path, cross-region geography prefix, ``anthropic.`` vendor
    prefix, Bedrock version suffix and release date, so
    ``us.anthropic.claude-haiku-4-5-20251001-v1:0`` becomes ``claude-haiku-4-5``.

    Args:
        model_id: The Bedrock model ID, inference-profile ID or ARN.

    Returns:
        The Anthropic model name, or the stripped ID if it is not recognized.
    """
    parts = model_id.rsplit("/", 1)[-1].split(".")
    if len(parts) >= 3 and parts[0] in BEDROCK_INFERENCE_PROFILE_PREFIXES:
        parts = parts[1:]
    if len(parts) >= 2 and parts[0] == "anthropic":
        parts = parts[1:]
    name = ".".join(parts)
    name = re.sub(r"-v\d+(:\d+)?$", "", name)
    return re.sub(r"-\d{8}$", "", name)


def validate_anthropic_bedrock_model_id(model_id: str) -> None:
    """Reject Bedrock model IDs from vendors other than Anthropic.

    The Anthropic Messages API only serves Claude, so a model such as
    ``amazon.nova-pro-v1:0`` would fail on its first request. IDs without a
    vendor prefix, such as application inference-profile ARNs, are allowed.

    Args:
        model_id: The Bedrock model ID, inference-profile ID or ARN.

    Raises:
        ValueError: If the ID names a non-Anthropic model.
    """
    parts = model_id.rsplit("/", 1)[-1].split(".")
    if len(parts) >= 3 and parts[0] in BEDROCK_INFERENCE_PROFILE_PREFIXES:
        parts = parts[1:]
    if len(parts) >= 2 and parts[0] != "anthropic":
        raise ValueError(
            'provider = "anthropic_bedrock" only supports Anthropic Claude '
            f'models; got {model_id}. Use provider = "bedrock" for other models.'
        )


def anthropic_bedrock_disabled_thinking_kwargs(base_model: str) -> dict[str, Any]:
    """Return the request settings that turn thinking off for a Claude model.

    Opus 5 and Sonnet 5 think adaptively unless thinking is explicitly
    disabled, while Opus 5.5, Sonnet 5.5 and Fable 5 cannot run without
    thinking. Older models do not think unless asked, so omitting the setting
    is enough.

    Args:
        base_model: The Anthropic model name.

    Returns:
        The ``thinking`` request setting, or an empty dict when none is needed.

    Raises:
        ValueError: If the model cannot disable thinking.
    """
    if base_model.startswith(("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5")):
        raise ValueError(
            f'{base_model} cannot run with thinking = "disabled"; use "auto" '
            'or "adaptive" and lower reasoning_effort instead.'
        )
    if base_model.startswith(("claude-opus-5", "claude-sonnet-5")):
        return {"thinking": {"type": "disabled"}}
    return {}


def bedrock_arn_model_metadata(model_arn: str) -> dict[str, str]:
    """Derive the provider and base model langchain-aws needs for a model ARN.

    langchain-aws cannot infer the model family from an ARN. Foundation-model
    and system inference-profile ARNs end in a model ID (for example
    ``.../inference-profile/us.anthropic.claude-sonnet-5``), so both values can
    be read from that suffix.

    Args:
        model_arn: The Bedrock model ARN.

    Returns:
        The ``provider`` and ``base_model`` constructor arguments.

    Raises:
        ValueError: If the ARN does not end in a recognizable model ID.
    """
    model_id = model_arn.rsplit("/", 1)[-1]
    parts = model_id.split(".")
    # Cross-region inference-profile IDs carry a geography prefix (us., eu., ...).
    if len(parts) >= 3 and parts[0] in BEDROCK_INFERENCE_PROFILE_PREFIXES:
        parts = parts[1:]
    if len(parts) < 2 or not parts[0] or ":" in parts[0]:
        raise ValueError(
            "Amazon Bedrock model ARNs must end in a model ID, such as a "
            "foundation-model or system inference-profile ARN. For application "
            "inference profiles or provisioned models, use the underlying "
            "model or inference-profile ID as [model].name instead."
        )
    return {"provider": parts[0], "base_model": ".".join(parts)}


def build_model_for_profile(
    config: RuntimeConfig,
    reasoning_level: ReasoningLevel,
    model_profile: ModelDefaults,
) -> Any:
    """Build a model from resolved profile settings."""
    try:
        parameters: Mapping[str, inspect.Parameter] = inspect.signature(
            build_model
        ).parameters
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
