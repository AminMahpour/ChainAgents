"""Provider settings, endpoint normalization, and model profile parsing."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from chainagents.runtime.constants import (
    ANTHROPIC_MESSAGES_PATH_SUFFIX,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_MODEL_THINKING,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_OLLAMA_PORT,
    DEFAULT_REASONING_LEVEL,
    DEFAULT_TEMPERATURE,
    OPENAI_CHAT_COMPLETIONS_PATH_SUFFIX,
    OPENAI_COMPATIBLE_MODEL_PROVIDERS,
    OPENAI_RESPONSES_PATH_SUFFIX,
    SNOWFLAKE_CORTEX_BASE_PATH,
    SNOWFLAKE_CORTEX_CHAT_COMPLETIONS_PATH,
    SNOWFLAKE_CORTEX_HOST_SUFFIXES,
    DisableStreaming,
    ModelModality,
    ModelProvider,
    ModelThinking,
    ReasoningLevel,
)
from chainagents.runtime.types import ModelDefaults


def normalize_reasoning_level(
    value: str | None,
    *,
    default: ReasoningLevel = DEFAULT_REASONING_LEVEL,
) -> ReasoningLevel:
    """Normalize reasoning level.

    Args:
        value: Value to normalize, convert, or serialize.
        default: Fallback value used when no explicit value is available.

    Returns:
        The normalized value.
    """
    candidate = (value or default).strip().lower()
    if candidate not in {"low", "medium", "high"}:
        return default
    return candidate  # type: ignore[return-value]


def normalize_disable_streaming(value: Any | None) -> DisableStreaming:
    """Normalize the LangChain disable_streaming model option.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized disable streaming value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    candidate = str(value).strip().lower().replace("-", "_")
    if not candidate:
        return False
    if candidate in {"true", "1", "yes", "on"}:
        return True
    if candidate in {"false", "0", "no", "off"}:
        return False
    if candidate == "tool_calling":
        return "tool_calling"
    raise ValueError(
        "Model disable_streaming must be a boolean or 'tool_calling'."
    )


def normalize_model_thinking(value: Any | None) -> ModelThinking:
    """Normalize Anthropic model thinking configuration.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized thinking mode.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return DEFAULT_MODEL_THINKING
    candidate = str(value).strip().lower().replace("-", "_")
    if not candidate:
        return DEFAULT_MODEL_THINKING
    if candidate not in {"auto", "adaptive", "disabled"}:
        raise ValueError(
            "model.thinking must be one of 'auto', 'adaptive', or 'disabled'."
        )
    return candidate  # type: ignore[return-value]


def normalize_disable_streaming_for_tool_calls(value: Any | None) -> bool:
    """Normalize whether to disable streaming only when tools are bound.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        Whether streaming should be disabled for tool-calling requests.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    normalized = normalize_disable_streaming(value)
    if normalized == "tool_calling":
        return True
    if isinstance(normalized, bool):
        return normalized
    return False


def parse_model_disable_streaming(raw_model: dict[str, Any]) -> DisableStreaming:
    """Parse model streaming-disabling settings.

    Args:
        raw_model: Raw model config table.

    Returns:
        The parsed LangChain disable_streaming value.
    """
    if "disable_streaming" in raw_model:
        return normalize_disable_streaming(raw_model.get("disable_streaming"))
    if normalize_disable_streaming_for_tool_calls(
        raw_model.get("disable_streaming_for_tool_calls")
    ):
        return "tool_calling"
    return False


def normalize_model_provider(
    value: Any | None,
    *,
    default: ModelProvider = DEFAULT_MODEL_PROVIDER,
) -> ModelProvider:
    """Normalize model provider.

    Args:
        value: Value to normalize, convert, or serialize.
        default: Fallback value used when no explicit value is available.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    raw_candidate = str(value or default).strip().lower()
    candidate = raw_candidate.replace("-", "_")
    if not candidate:
        return default
    if candidate == "claude":
        candidate = "anthropic"
    if candidate == "snowflake_cortex" and raw_candidate != candidate:
        raise ValueError("The Snowflake Cortex provider must be 'snowflake_cortex'.")
    if candidate not in {"ollama", "openai_compatible", "snowflake_cortex", "anthropic"}:
        raise ValueError(
            "The model provider must be 'ollama', 'openai_compatible', "
            "'snowflake_cortex', 'anthropic', or 'claude'."
        )
    return candidate  # type: ignore[return-value]


def format_model_provider(provider: ModelProvider) -> str:
    """Format model provider.

    Args:
        provider: The provider value.

    Returns:
        The formatted value.
    """
    if provider == "openai_compatible":
        return "OpenAI-compatible"
    if provider == "snowflake_cortex":
        return "Snowflake Cortex"
    if provider == "anthropic":
        return "Anthropic Claude"
    return "Ollama"


def normalize_model_endpoint(value: str | None) -> str:
    """Normalize model endpoint.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized value.
    """
    candidate = (value or DEFAULT_OLLAMA_ENDPOINT).strip()
    if not candidate:
        candidate = DEFAULT_OLLAMA_ENDPOINT
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def normalize_model_port(value: Any | None) -> int:
    """Normalize model port.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized value.
    """
    if value is None:
        return DEFAULT_OLLAMA_PORT

    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_OLLAMA_PORT

    if 1 <= port <= 65535:
        return port
    return DEFAULT_OLLAMA_PORT


def normalize_model_temperature(value: Any | None) -> float:
    """Normalize model temperature.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized value.
    """
    if value is None:
        return DEFAULT_TEMPERATURE

    try:
        temperature = float(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_TEMPERATURE

    if not math.isfinite(temperature):
        return DEFAULT_TEMPERATURE
    return temperature


def normalize_repeat_penalty(value: Any | None) -> float | None:
    """Normalize repeat penalty.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        repeat_penalty = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("Model repeat_penalty must be a finite number.") from exc
    if not math.isfinite(repeat_penalty):
        raise ValueError("Model repeat_penalty must be a finite number.")
    if repeat_penalty < 0:
        raise ValueError("Model repeat_penalty must be greater than or equal to 0.")
    return repeat_penalty


def normalize_model_base_url(
    value: Any | None,
    *,
    default: str | None = None,
    required_message: str | None = None,
) -> str:
    """Normalize model base URL.

    Args:
        value: Value to normalize, convert, or serialize.
        default: Fallback value used when no explicit value is available.
        required_message: The required message value.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    candidate = str(value or default or "").strip()
    if not candidate:
        if required_message:
            raise ValueError(required_message)
        return ""
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def normalize_openai_endpoint_url(
    value: Any | None,
    *,
    required_message: str | None = None,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Normalize openai endpoint URL.

    Args:
        value: Value to normalize, convert, or serialize.
        required_message: The required message value.

    Returns:
        The normalized value.
    """
    candidate = normalize_model_base_url(
        value,
        required_message=required_message,
    )
    parsed = urlsplit(candidate)
    path = parsed.path.rstrip("/")
    for suffix in (
        OPENAI_CHAT_COMPLETIONS_PATH_SUFFIX,
        OPENAI_RESPONSES_PATH_SUFFIX,
    ):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break

    base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")
    return base_url, tuple(parse_qsl(parsed.query, keep_blank_values=True))


def normalize_snowflake_cortex_endpoint_url(
    value: Any | None,
    *,
    full_endpoint: bool,
    required_message: str | None = None,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Validate and normalize a Snowflake Cortex API base or full endpoint URL."""
    candidate = str(value or "").strip()
    if not candidate:
        raise ValueError(
            required_message
            or "Snowflake Cortex model config must define a non-empty endpoint URL."
        )

    parsed = urlsplit(candidate)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not hostname:
        raise ValueError("Snowflake Cortex endpoints must use an absolute HTTPS URL.")
    if not any(hostname.endswith(suffix) for suffix in SNOWFLAKE_CORTEX_HOST_SUFFIXES):
        raise ValueError(
            "Snowflake Cortex endpoints must use a Snowflake account hostname."
        )
    if parsed.fragment:
        raise ValueError("Snowflake Cortex endpoints must not include a fragment.")
    if not full_endpoint and parsed.query:
        raise ValueError(
            "Snowflake Cortex base URLs must not include query parameters."
        )

    expected_path = (
        SNOWFLAKE_CORTEX_CHAT_COMPLETIONS_PATH
        if full_endpoint
        else SNOWFLAKE_CORTEX_BASE_PATH
    )
    path = parsed.path[:-1] if parsed.path.endswith("/") else parsed.path
    if path != expected_path:
        kind = "endpoint URL" if full_endpoint else "base URL"
        raise ValueError(
            f"Snowflake Cortex {kind} must use the path '{expected_path}'."
        )

    base_url = urlunsplit((parsed.scheme, parsed.netloc, SNOWFLAKE_CORTEX_BASE_PATH, "", ""))
    endpoint_query = (
        tuple(parse_qsl(parsed.query, keep_blank_values=True)) if full_endpoint else ()
    )
    return base_url, endpoint_query


def normalize_anthropic_endpoint_url(
    value: Any | None,
    *,
    required_message: str | None = None,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Normalize Anthropic endpoint URL to the API base URL.

    Args:
        value: Value to normalize, convert, or serialize.
        required_message: The required message value.

    Returns:
        The normalized Anthropic API base URL and query params.
    """
    candidate = normalize_model_base_url(
        value,
        required_message=required_message,
    )
    parsed = urlsplit(candidate)
    path = parsed.path.rstrip("/")
    if path.endswith(ANTHROPIC_MESSAGES_PATH_SUFFIX):
        path = path[: -len(ANTHROPIC_MESSAGES_PATH_SUFFIX)].rstrip("/")

    base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", "")).rstrip("/")
    return base_url, tuple(parse_qsl(parsed.query, keep_blank_values=True))


def model_endpoint_query_to_dict(
    query: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    """Parse endpoint query parameters into a dictionary.

    Args:
        query: Search query text.

    Returns:
        The parsed endpoint query parameters into a dictionary.
    """
    values: dict[str, object] = {}
    for key, value in query:
        existing = values.get(key)
        if existing is None:
            values[key] = value
        elif isinstance(existing, list):
            existing.append(value)
        else:
            values[key] = [existing, value]
    return values


def normalize_optional_string(value: Any | None) -> str | None:
    """Normalize optional string.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized value.
    """
    candidate = str(value or "").strip()
    return candidate or None


def compose_base_url(endpoint: str | None, port: int) -> str:
    """Compose base URL.

    Args:
        endpoint: The endpoint value.
        port: The port value.

    Returns:
        The composed value.
    """
    parsed = urlsplit(normalize_model_endpoint(endpoint))
    hostname = parsed.hostname
    if hostname is None:
        return DEFAULT_OLLAMA_BASE_URL

    auth = ""
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        auth = f"{auth}@"

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    netloc = f"{auth}{hostname}:{port}"
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme or "http", netloc, path, parsed.query, parsed.fragment))


def _parse_model_names(value: Any, *, field_name: str) -> tuple[str, ...]:
    """Parse a TOML model list while preserving order and removing duplicates."""
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"The {field_name} config must be an array of strings.")
    parsed_models: list[str] = []
    for raw_candidate in value:
        candidate = str(raw_candidate or "").strip()
        if candidate and candidate not in parsed_models:
            parsed_models.append(candidate)
    return tuple(parsed_models)


def normalize_model_modalities(
    value: Any | None,
    *,
    default: tuple[ModelModality, ...] = ("text",),
    field_name: str = "[model].modalities",
) -> tuple[ModelModality, ...]:
    """Normalize declared model input modalities with a safe text-only default."""
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError(f"The {field_name} config must be an array of strings.")

    modalities: list[ModelModality] = []
    for raw_modality in value:
        if not isinstance(raw_modality, str):
            raise ValueError(
                f"The {field_name} config may only contain 'text' and 'image'."
            )
        modality = raw_modality.strip().lower()
        if modality not in {"text", "image"}:
            raise ValueError(
                f"The {field_name} config may only contain 'text' and 'image'."
            )
        if modality not in modalities:
            modalities.append(modality)  # type: ignore[arg-type]
    if "text" not in modalities:
        raise ValueError(f"The {field_name} config must include 'text'.")
    return tuple(modalities)


def parse_model_profile_defaults(
    raw_model: dict[str, Any],
    *,
    base: ModelDefaults | None = None,
    field_prefix: str = "[model]",
) -> ModelDefaults:
    """Parse one model default/profile table."""
    if raw_model and not isinstance(raw_model, dict):
        raise ValueError(
            f"The top-level '{field_prefix.strip('[]')}' config must be a table/object."
        )

    explicit_fields: set[str] = set()
    base_provider = base.provider if base is not None else DEFAULT_MODEL_PROVIDER
    provider_is_explicit = "provider" in raw_model
    if provider_is_explicit:
        explicit_fields.add("provider")
    provider = normalize_model_provider(
        raw_model.get("provider"),
        default=base_provider,
    )
    provider_changed = bool(
        base is not None and provider_is_explicit and provider != base.provider
    )

    raw_models = raw_model.get("models") if "models" in raw_model else None
    parsed_models = _parse_model_names(raw_models, field_name=f"{field_prefix}.models")
    if raw_models is not None:
        explicit_fields.add("models")
    if raw_models is None and base is not None and not provider_changed:
        parsed_models = base.models

    raw_name = str(raw_model.get("name", "")).strip() if "name" in raw_model else ""
    if raw_name:
        explicit_fields.add("name")
    if raw_name:
        name = raw_name
    elif raw_models is not None and parsed_models:
        name = parsed_models[0]
        explicit_fields.add("name")
    elif base is not None and not provider_changed:
        name = base.name
    elif parsed_models:
        name = parsed_models[0]
    else:
        name = DEFAULT_MODEL if provider == "ollama" else ""

    if provider in {*OPENAI_COMPATIBLE_MODEL_PROVIDERS, "anthropic"} and not name:
        provider_label = (
            "OpenAI-compatible"
            if provider == "openai_compatible"
            else ("Snowflake Cortex" if provider == "snowflake_cortex" else "Anthropic")
        )
        raise ValueError(
            f"{provider_label} model config must define a non-empty 'name' or 'models'."
        )

    endpoint_query: tuple[tuple[str, str], ...] = ()
    raw_base_url = normalize_optional_string(raw_model.get("base_url"))
    raw_endpoint_url = raw_model.get("endpoint_url")
    has_endpoint_url = normalize_optional_string(raw_endpoint_url) is not None
    endpoint_is_explicit = bool(
        raw_base_url is not None
        or has_endpoint_url
        or "endpoint" in raw_model
        or "port" in raw_model
    )
    if endpoint_is_explicit:
        explicit_fields.update({"base_url", "endpoint_query"})
    inherits_endpoint = bool(
        base is not None
        and not provider_changed
        and raw_base_url is None
        and not has_endpoint_url
        and "endpoint" not in raw_model
        and "port" not in raw_model
    )
    if inherits_endpoint:
        base_url = base.base_url
        endpoint_query = base.endpoint_query
    elif provider == "ollama":
        if raw_base_url:
            base_url = normalize_model_base_url(
                raw_base_url,
                default=DEFAULT_OLLAMA_BASE_URL,
            )
        else:
            base_url = compose_base_url(
                raw_model.get("endpoint"),
                normalize_model_port(raw_model.get("port")),
            )
    elif provider == "snowflake_cortex":
        required_message = (
            "Snowflake Cortex model config must define a non-empty "
            "'base_url' or 'endpoint_url'."
        )
        if has_endpoint_url:
            base_url, endpoint_query = normalize_snowflake_cortex_endpoint_url(
                raw_endpoint_url,
                full_endpoint=True,
                required_message=required_message,
            )
        else:
            base_url, endpoint_query = normalize_snowflake_cortex_endpoint_url(
                raw_model.get("base_url"),
                full_endpoint=False,
                required_message=required_message,
            )
    elif provider == "openai_compatible":
        required_message = (
            f"{format_model_provider(provider)} model config must define a non-empty "
            "'base_url' or 'endpoint_url'."
        )
        if has_endpoint_url:
            base_url, endpoint_query = normalize_openai_endpoint_url(
                raw_endpoint_url,
                required_message=required_message,
            )
        else:
            base_url = normalize_model_base_url(
                raw_model.get("base_url"),
                required_message=required_message,
            )
    else:
        if has_endpoint_url:
            base_url, endpoint_query = normalize_anthropic_endpoint_url(raw_endpoint_url)
        else:
            base_url = normalize_model_base_url(
                raw_model.get("base_url"),
                default=DEFAULT_ANTHROPIC_BASE_URL,
            )

    api_key = (
        normalize_optional_string(raw_model.get("api_key"))
        if "api_key" in raw_model
        else (base.api_key if base is not None and not provider_changed else None)
    )
    if "api_key" in raw_model:
        explicit_fields.add("api_key")
    reasoning_effort = (
        normalize_reasoning_level(
            raw_model.get("reasoning_effort"),
            default=(
                base.reasoning_effort
                if base is not None
                else DEFAULT_REASONING_LEVEL
            ),
        )
        if "reasoning_effort" in raw_model or base is None
        else base.reasoning_effort
    )
    if "reasoning_effort" in raw_model:
        explicit_fields.add("reasoning_effort")
    thinking = (
        normalize_model_thinking(raw_model.get("thinking"))
        if "thinking" in raw_model or base is None
        else base.thinking
    )
    if "thinking" in raw_model:
        explicit_fields.add("thinking")
    temperature = (
        normalize_model_temperature(
            raw_model.get("temperature", raw_model.get("tempreature"))
        )
        if "temperature" in raw_model or "tempreature" in raw_model or base is None
        else base.temperature
    )
    if "temperature" in raw_model or "tempreature" in raw_model:
        explicit_fields.add("temperature")
    repeat_penalty = (
        normalize_repeat_penalty(raw_model.get("repeat_penalty"))
        if "repeat_penalty" in raw_model or base is None
        else base.repeat_penalty
    )
    if "repeat_penalty" in raw_model:
        explicit_fields.add("repeat_penalty")
    disable_streaming = (
        parse_model_disable_streaming(raw_model)
        if (
            "disable_streaming" in raw_model
            or "disable_streaming_for_tool_calls" in raw_model
            or base is None
        )
        else base.disable_streaming
    )
    if (
        "disable_streaming" in raw_model
        or "disable_streaming_for_tool_calls" in raw_model
    ):
        explicit_fields.add("disable_streaming")
    modalities = normalize_model_modalities(
        raw_model.get("modalities") if "modalities" in raw_model else None,
        default=(base.modalities if base is not None else ("text",)),
        field_name=f"{field_prefix}.modalities",
    )
    if "modalities" in raw_model:
        explicit_fields.add("modalities")

    return ModelDefaults(
        provider=provider,
        base_url=base_url,
        endpoint_query=endpoint_query,
        name=name,
        api_key=api_key,
        models=parsed_models,
        name_is_explicit=bool(raw_name or parsed_models),
        reasoning_effort=reasoning_effort,
        thinking=thinking,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        disable_streaming=disable_streaming,
        modalities=modalities,
        explicit_fields=frozenset(explicit_fields),
    )


def parse_model_profiles(
    raw_model: dict[str, Any],
    *,
    base: ModelDefaults,
) -> dict[str, ModelDefaults]:
    """Parse named model profiles from the [model.profiles] TOML table."""
    raw_profiles = raw_model.get("profiles", {})
    if raw_profiles in ({}, None):
        return {}
    if not isinstance(raw_profiles, dict):
        raise ValueError("The [model].profiles config must be a table/object.")

    profiles: dict[str, ModelDefaults] = {}
    for raw_name, raw_profile in raw_profiles.items():
        profile_name = str(raw_name).strip()
        if not profile_name:
            raise ValueError("Model profile names must be non-empty strings.")
        if not isinstance(raw_profile, dict):
            raise ValueError(
                f"Model profile '{profile_name}' must be a table/object."
            )
        profiles[profile_name] = parse_model_profile_defaults(
            raw_profile,
            base=base,
            field_prefix=f"[model.profiles.{profile_name}]",
        )
    return profiles


def resolve_model_profile_defaults(
    default_model: ModelDefaults,
    model_profiles: dict[str, ModelDefaults],
    model_ref: str | None,
    *,
    inherited_model: ModelDefaults | None = None,
) -> ModelDefaults:
    """Resolve a profile-or-raw-model reference into concrete model settings."""
    base_model = inherited_model or default_model
    selected_ref = normalize_optional_string(model_ref)
    if selected_ref and selected_ref in model_profiles:
        return rebase_model_profile_defaults(model_profiles[selected_ref], base_model)
    if selected_ref:
        return replace(
            base_model,
            name=selected_ref,
            models=(),
            name_is_explicit=True,
            explicit_fields=base_model.explicit_fields | frozenset({"name", "models"}),
        )
    return base_model


def rebase_model_profile_defaults(
    model_profile: ModelDefaults,
    base_model: ModelDefaults,
) -> ModelDefaults:
    """Apply runtime base fields to profile values inherited from parsed defaults."""
    explicit_fields = model_profile.explicit_fields
    runtime_override_fields = base_model.runtime_override_fields
    if model_profile.provider != base_model.provider:
        updates: dict[str, Any] = {}
        if "cross_provider_endpoint_url" in runtime_override_fields:
            cross_provider_endpoint_url = (
                base_model.cross_provider_endpoint_url or ""
            )
            if model_profile.provider == "anthropic":
                cross_provider_base_url, cross_provider_endpoint_query = (
                    normalize_anthropic_endpoint_url(
                        cross_provider_endpoint_url,
                        required_message=(
                            "The Anthropic model endpoint URL cannot be empty."
                        ),
                    )
                )
            elif model_profile.provider == "snowflake_cortex":
                cross_provider_base_url, cross_provider_endpoint_query = (
                    normalize_snowflake_cortex_endpoint_url(
                        cross_provider_endpoint_url,
                        full_endpoint=True,
                        required_message=(
                            "The Snowflake Cortex model endpoint URL cannot be empty."
                        ),
                    )
                )
            elif model_profile.provider == "openai_compatible":
                cross_provider_base_url, cross_provider_endpoint_query = (
                    normalize_openai_endpoint_url(
                        cross_provider_endpoint_url,
                        required_message="The model endpoint URL cannot be empty.",
                    )
                )
            else:
                raise ValueError(
                    "DEEPAGENT_MODEL_ENDPOINT_URL can only target "
                    "provider-switched Anthropic or OpenAI-compatible profiles."
                )
            updates["base_url"] = cross_provider_base_url
            updates["endpoint_query"] = cross_provider_endpoint_query
        elif "cross_provider_base_url" in runtime_override_fields:
            cross_provider_base_url = (
                base_model.cross_provider_base_url or base_model.base_url
            )
            if model_profile.provider == "snowflake_cortex":
                cross_provider_base_url, _ = normalize_snowflake_cortex_endpoint_url(
                    cross_provider_base_url,
                    full_endpoint=False,
                    required_message=(
                        "The Snowflake Cortex model base URL cannot be empty."
                    ),
                )
            updates["base_url"] = cross_provider_base_url
            updates["endpoint_query"] = (
                base_model.cross_provider_endpoint_query
                or base_model.endpoint_query
            )
        for field_name in ("temperature", "disable_streaming"):
            if field_name in runtime_override_fields:
                updates[field_name] = getattr(base_model, field_name)
        if model_profile.runtime_override_fields != runtime_override_fields:
            updates["runtime_override_fields"] = runtime_override_fields
        if model_profile.cross_provider_base_url != base_model.cross_provider_base_url:
            updates["cross_provider_base_url"] = base_model.cross_provider_base_url
        if (
            model_profile.cross_provider_endpoint_url
            != base_model.cross_provider_endpoint_url
        ):
            updates["cross_provider_endpoint_url"] = (
                base_model.cross_provider_endpoint_url
            )
        if (
            model_profile.cross_provider_endpoint_query
            != base_model.cross_provider_endpoint_query
        ):
            updates["cross_provider_endpoint_query"] = (
                base_model.cross_provider_endpoint_query
            )
        if not updates:
            return model_profile
        return replace(model_profile, **updates)

    updates: dict[str, Any] = {}
    if "name" not in explicit_fields:
        updates["name"] = base_model.name
        updates["name_is_explicit"] = base_model.name_is_explicit
    if "models" not in explicit_fields:
        updates["models"] = base_model.models
    if (
        "base_url" in runtime_override_fields
        or "base_url" not in explicit_fields
    ):
        updates["base_url"] = base_model.base_url
        updates["endpoint_query"] = base_model.endpoint_query
    if "api_key" not in explicit_fields:
        updates["api_key"] = base_model.api_key
    for field_name in (
        "reasoning_effort",
        "thinking",
        "temperature",
        "repeat_penalty",
        "disable_streaming",
        "modalities",
    ):
        if (
            field_name in runtime_override_fields
            or field_name not in explicit_fields
        ):
            updates[field_name] = getattr(base_model, field_name)
    if model_profile.runtime_override_fields != runtime_override_fields:
        updates["runtime_override_fields"] = runtime_override_fields
    if model_profile.cross_provider_base_url != base_model.cross_provider_base_url:
        updates["cross_provider_base_url"] = base_model.cross_provider_base_url
    if (
        model_profile.cross_provider_endpoint_url
        != base_model.cross_provider_endpoint_url
    ):
        updates["cross_provider_endpoint_url"] = base_model.cross_provider_endpoint_url
    if (
        model_profile.cross_provider_endpoint_query
        != base_model.cross_provider_endpoint_query
    ):
        updates["cross_provider_endpoint_query"] = (
            base_model.cross_provider_endpoint_query
        )

    if not updates:
        return model_profile
    return replace(model_profile, **updates)


def parse_model_defaults(raw_config: dict[str, Any]) -> ModelDefaults:
    """Parse model defaults.

    Args:
        raw_config: Raw config to process.

    Returns:
        The parsed model defaults.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    raw_model = raw_config.get("model", {})
    if raw_model and not isinstance(raw_model, dict):
        raise ValueError("The top-level 'model' config must be a table/object.")
    return parse_model_profile_defaults(raw_model, field_prefix="[model]")
