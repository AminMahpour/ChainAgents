"""TOML and environment loading into resolved runtime configuration."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chainagents.runtime.constants as runtime_constants
import chainagents.runtime.extension_config as runtime_extension_config
import chainagents.runtime.model_config as runtime_model_config
from chainagents.rag.runtime import (
    RagConfig,
    ResolvedRagConfig,
    parse_rag_config,
    resolve_rag_config,
)
from chainagents.runtime.constants import (
    DEFAULT_AGENT_STATE,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_EXTENSIONS_CONFIG,
    DEFAULT_MODEL_THINKING,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_RECURSION_LIMIT,
    OPENAI_COMPATIBLE_MODEL_PROVIDERS,
    AgentStateMode,
    DisableStreaming,
    ModelModality,
    ModelProvider,
    ModelThinking,
    PersistenceMode,
    ReasoningLevel,
)
from chainagents.runtime.types import (
    ExtensionsConfig,
    FileConfig,
    LangfuseConfig,
    LangSmithConfig,
    ModelDefaults,
    RuntimeConfigOverrides,
)


def parse_langfuse_config(raw_config: dict[str, Any]) -> LangfuseConfig:
    """Parse Langfuse tracing configuration.

    Args:
        raw_config: Raw config to process.

    Returns:
        The parsed Langfuse tracing configuration.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    raw_langfuse = raw_config.get("langfuse", {})
    if raw_langfuse and not isinstance(raw_langfuse, dict):
        raise ValueError("The top-level 'langfuse' config must be a table/object.")

    raw_enabled = raw_langfuse.get("enabled", False)
    if not isinstance(raw_enabled, bool):
        raise ValueError("The top-level 'langfuse.enabled' config must be a boolean.")
    return LangfuseConfig(enabled=raw_enabled)


def parse_langsmith_config(raw_config: dict[str, Any]) -> LangSmithConfig:
    """Parse optional LangSmith tracing settings from a top-level table."""
    raw_langsmith = raw_config.get("langsmith", {})
    if not isinstance(raw_langsmith, dict):
        raise ValueError("The top-level 'langsmith' config must be a table/object.")
    enabled = raw_langsmith.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("The top-level 'langsmith.enabled' config must be a boolean.")
    project = raw_langsmith.get("project")
    if project is not None:
        if not isinstance(project, str) or not project.strip():
            raise ValueError("The top-level 'langsmith.project' config must be a nonempty string.")
        project = project.strip()
    mode = raw_langsmith.get("background_trace_mode", "linked")
    if mode not in ("linked", "separate"):
        raise ValueError("The top-level 'langsmith.background_trace_mode' config must be 'linked' or 'separate'.")
    return LangSmithConfig(enabled=enabled, project=project, background_trace_mode=mode)


def load_file_config(config_path: str | Path | None = None) -> FileConfig:
    """Load file config.

    Args:
        config_path: Path to the config.

    Returns:
        The loaded value.
    """
    config_name = (
        str(config_path).strip()
        if config_path is not None
        else os.getenv("DEEPAGENT_CONFIG", DEFAULT_EXTENSIONS_CONFIG).strip()
    )
    resolved_config_path = runtime_extension_config.resolve_local_path(
        config_name or DEFAULT_EXTENSIONS_CONFIG,
        runtime_constants.PROJECT_ROOT,
    )
    if not resolved_config_path.exists():
        return FileConfig(
            model=ModelDefaults(),
            extensions=ExtensionsConfig(config_path=None),
            model_profiles={},
            langfuse=LangfuseConfig(),
            langsmith=LangSmithConfig(),
            rag=RagConfig(),
        )

    with resolved_config_path.open("rb") as fh:
        raw_config = tomllib.load(fh)

    model_defaults = runtime_model_config.parse_model_defaults(raw_config)
    raw_model = raw_config.get("model", {})
    return FileConfig(
        model=model_defaults,
        extensions=runtime_extension_config.parse_extensions_config(raw_config, resolved_config_path),
        model_profiles=runtime_model_config.parse_model_profiles(raw_model, base=model_defaults),
        langfuse=parse_langfuse_config(raw_config),
        langsmith=parse_langsmith_config(raw_config),
        rag=parse_rag_config(raw_config, resolved_config_path),
    )


def load_extensions_config(config_path: str | Path | None = None) -> ExtensionsConfig:
    # Keep the previous public helper for existing imports and tests.
    """Load extensions config.

    Args:
        config_path: Path to the config.

    Returns:
        The loaded value.
    """
    return load_file_config(config_path).extensions


_ENDPOINT_URL_PROVIDERS = frozenset({"anthropic", "snowflake_cortex", "openai_compatible"})


@dataclass(frozen=True)
class _ModelInputs:
    """Raw model settings gathered from runtime overrides and the environment.

    Attributes:
        provider: Resolved runtime model provider.
        provider_override: Explicit provider override, if any.
        name: Generic model name override (``DEEPAGENT_MODEL_NAME``).
        base_url: Generic base URL override (``DEEPAGENT_MODEL_BASE_URL``).
        base_url_from_env: Whether ``base_url`` came from the environment
            rather than an explicit runtime override.
        endpoint_url: Generic full endpoint URL override.
        reasoning: Generic reasoning level override.
        name_alias: Ollama-only ``OLLAMA_MODEL`` alias.
        base_url_alias: Ollama-only ``OLLAMA_BASE_URL`` alias.
        reasoning_alias: Ollama-only ``OLLAMA_REASONING`` alias.
    """

    provider: ModelProvider
    provider_override: str | None
    name: str
    base_url: str
    base_url_from_env: bool
    endpoint_url: str
    reasoning: str
    name_alias: str
    base_url_alias: str
    reasoning_alias: str


@dataclass(frozen=True)
class _ModelSelection:
    """Model name and profile chosen from the raw inputs and file config.

    Attributes:
        provider_changed: Whether an explicit override switched providers.
        model_name: Selected model or profile name.
        name_override: Runtime model name override, or an empty string.
        default_name: Runtime default model name before profile selection.
        override_profile: Profile named by ``name_override``, if any.
    """

    provider_changed: bool
    model_name: str
    name_override: str
    default_name: str
    override_profile: ModelDefaults | None


@dataclass(frozen=True)
class _ModelEndpoints:
    """Resolved primary and cross-provider model endpoints.

    Attributes:
        base_url: Primary model base URL.
        endpoint_query: Primary model endpoint query parameters.
        base_url_override: Whether the endpoint was overridden at runtime.
        cross_provider_base_url: Generic endpoint for provider-switched profiles.
        cross_provider_endpoint_url: Unnormalized generic full endpoint.
        cross_provider_endpoint_query: Query for provider-switched profiles.
    """

    base_url: str
    endpoint_query: tuple[tuple[str, str], ...]
    base_url_override: bool
    cross_provider_base_url: str | None
    cross_provider_endpoint_url: str | None
    cross_provider_endpoint_query: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _RagResolution:
    """Outcome of resolving the optional RAG configuration."""

    requested: bool
    rag: ResolvedRagConfig | None = None
    error: str | None = None


def _resolve_database_url(overrides: RuntimeConfigOverrides) -> str | None:
    """Resolve the database URL from overrides or ``DATABASE_URL``."""
    if overrides.disable_database:
        return None
    if overrides.database_url is not None:
        return runtime_model_config.normalize_optional_string(overrides.database_url)
    return os.getenv("DATABASE_URL", "").strip() or None


def _collect_model_inputs(
    overrides: RuntimeConfigOverrides,
    model_defaults: ModelDefaults,
) -> _ModelInputs:
    """Gather raw model settings from overrides, ``DEEPAGENT_*`` and Ollama aliases."""
    provider_override = runtime_model_config.normalize_optional_string(
        overrides.model_provider
    )
    if provider_override is None:
        provider_override = runtime_model_config.normalize_optional_string(
            os.getenv("DEEPAGENT_MODEL_PROVIDER")
        )
    provider = runtime_model_config.normalize_model_provider(
        provider_override,
        default=model_defaults.provider,
    )
    override_base_url = runtime_model_config.normalize_optional_string(overrides.model_base_url)
    env_base_url = runtime_model_config.normalize_optional_string(
        os.getenv("DEEPAGENT_MODEL_BASE_URL")
    )
    is_ollama = provider == "ollama"
    return _ModelInputs(
        provider=provider,
        provider_override=provider_override,
        name=(
            runtime_model_config.normalize_optional_string(overrides.model_name)
            or os.getenv("DEEPAGENT_MODEL_NAME", "").strip()
        ),
        base_url=override_base_url or env_base_url or "",
        base_url_from_env=override_base_url is None and env_base_url is not None,
        endpoint_url=(
            runtime_model_config.normalize_optional_string(overrides.model_endpoint_url)
            or os.getenv("DEEPAGENT_MODEL_ENDPOINT_URL", "").strip()
        ),
        reasoning=(
            runtime_model_config.normalize_optional_string(overrides.reasoning_level)
            or os.getenv("DEEPAGENT_MODEL_REASONING", "").strip()
        ),
        name_alias=os.getenv("OLLAMA_MODEL", "").strip() if is_ollama else "",
        base_url_alias=os.getenv("OLLAMA_BASE_URL", "").strip() if is_ollama else "",
        reasoning_alias=os.getenv("OLLAMA_REASONING", "").strip() if is_ollama else "",
    )


def _select_model_profile(inputs: _ModelInputs, file_config: FileConfig) -> _ModelSelection:
    """Select the model name and any profile named by the runtime override."""
    model_defaults = file_config.model
    provider_changed = (
        bool(inputs.provider_override) and inputs.provider != model_defaults.provider
    )
    model_name = (
        inputs.name
        or inputs.name_alias
        or (
            file_config.extensions.agent_model
            if not provider_changed
            else None
        )
        or model_defaults.name
    )
    name_override = inputs.name or inputs.name_alias
    return _ModelSelection(
        provider_changed=provider_changed,
        model_name=model_name,
        name_override=name_override,
        default_name=(
            model_defaults.name
            if name_override in file_config.model_profiles
            else name_override or model_defaults.name
        ),
        override_profile=file_config.model_profiles.get(name_override or ""),
    )


def _validate_provider_switch(
    inputs: _ModelInputs,
    selection: _ModelSelection,
    model_defaults: ModelDefaults,
) -> bool:
    """Reject provider switches and model names that cannot resolve safely.

    Returns:
        Whether only the selected profile's own endpoint satisfies a provider
        switch, in which case the file's default model choices are dropped.

    Raises:
        ValueError: If the provider override conflicts with the selected
            profile, leaves the new provider without an endpoint, or lacks a
            required model name.
    """
    model_provider = inputs.provider
    provider_changed = selection.provider_changed
    override_profile = selection.override_profile
    if (
        inputs.provider_override
        and override_profile is not None
        and override_profile.provider != model_provider
    ):
        raise ValueError(
            f"Model provider override '{model_provider}' does not match "
            f"selected profile '{selection.name_override}' provider "
            f"'{override_profile.provider}'."
        )
    profile_endpoint_satisfies_provider_switch = bool(
        override_profile is not None
        and override_profile.provider == model_provider
        and override_profile.base_url
    )
    endpoint_url_satisfies_provider_switch = (
        model_provider in OPENAI_COMPATIBLE_MODEL_PROVIDERS
        and bool(inputs.endpoint_url)
    )
    profile_endpoint_only_satisfies_provider_switch = (
        provider_changed
        and profile_endpoint_satisfies_provider_switch
        and not inputs.base_url
        and not endpoint_url_satisfies_provider_switch
    )
    provider_switch_requires_url = (
        provider_changed
        and model_provider in {"ollama", *OPENAI_COMPATIBLE_MODEL_PROVIDERS}
        and not inputs.base_url
        and not endpoint_url_satisfies_provider_switch
        and not profile_endpoint_satisfies_provider_switch
    )
    if provider_switch_requires_url:
        required_url_env = "DEEPAGENT_MODEL_BASE_URL"
        if model_provider in OPENAI_COMPATIBLE_MODEL_PROVIDERS:
            required_url_env = (
                "DEEPAGENT_MODEL_BASE_URL or DEEPAGENT_MODEL_ENDPOINT_URL"
            )
        raise ValueError(
            "Switching model providers via DEEPAGENT_MODEL_PROVIDER also requires "
            f"{required_url_env} so the new provider does not inherit an "
            "incompatible endpoint."
        )

    if (
        provider_changed
        and model_provider == "anthropic"
        and inputs.base_url
        and inputs.base_url_from_env
        and not inputs.endpoint_url
    ):
        raise ValueError(
            "Switching model providers to Anthropic with "
            "DEEPAGENT_MODEL_BASE_URL is ambiguous. Remove stale "
            "DEEPAGENT_MODEL_BASE_URL, pass --base-url explicitly, or use "
            "DEEPAGENT_MODEL_ENDPOINT_URL or --endpoint-url with the "
            "Anthropic /v1/messages path for proxy endpoints."
        )

    if (
        model_provider in OPENAI_COMPATIBLE_MODEL_PROVIDERS
        and not inputs.name
        and not model_defaults.name_is_explicit
    ):
        provider_label = runtime_model_config.format_model_provider(model_provider)
        raise ValueError(
            f"{provider_label} runtime must define DEEPAGENT_MODEL_NAME "
            "or set a non-empty [model].name in deepagent.toml."
        )
    if (
        model_provider == "anthropic"
        and not inputs.name
        and (provider_changed or not model_defaults.name_is_explicit)
    ):
        raise ValueError(
            "Anthropic runtime must define DEEPAGENT_MODEL_NAME "
            "or set a non-empty [model].name in deepagent.toml."
        )
    return profile_endpoint_only_satisfies_provider_switch


def _model_choices(
    inputs: _ModelInputs,
    selection: _ModelSelection,
    file_config: FileConfig,
    profile_endpoint_only_satisfies_provider_switch: bool,
) -> tuple[str, ...]:
    """Build the ordered, de-duplicated model choices for the runtime."""
    model_defaults = file_config.model
    default_model_choices = (
        ()
        if profile_endpoint_only_satisfies_provider_switch
        else (model_defaults.name, *model_defaults.models)
    )
    profile_choices = tuple(
        profile_name
        for profile_name, profile in file_config.model_profiles.items()
        if not (selection.provider_changed and inputs.provider_override)
        or profile.provider == inputs.provider
    )
    return tuple(
        dict.fromkeys(
            [
                selection.model_name,
                *default_model_choices,
                *profile_choices,
            ]
        )
    )


def _normalize_endpoint_for_provider(
    provider: ModelProvider,
    url: str,
    *,
    full_endpoint: bool,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Normalize a provider endpoint URL into its base URL and query.

    Args:
        provider: Provider whose endpoint rules apply.
        url: Base URL (``full_endpoint=False``, Snowflake Cortex only) or
            full endpoint URL to normalize.
        full_endpoint: Whether ``url`` is a full endpoint URL.

    Returns:
        The normalized base URL and endpoint query parameters.

    Raises:
        ValueError: If the URL is invalid or the provider has no full
            endpoint URL support.
    """
    if provider == "snowflake_cortex":
        return runtime_model_config.normalize_snowflake_cortex_endpoint_url(
            url,
            full_endpoint=full_endpoint,
            required_message=(
                "The Snowflake Cortex model endpoint URL cannot be empty."
                if full_endpoint
                else "The Snowflake Cortex model base URL cannot be empty."
            ),
        )
    if full_endpoint and provider == "anthropic":
        return runtime_model_config.normalize_anthropic_endpoint_url(
            url,
            required_message="The Anthropic model endpoint URL cannot be empty.",
        )
    if full_endpoint and provider == "openai_compatible":
        return runtime_model_config.normalize_openai_endpoint_url(
            url,
            required_message="The model endpoint URL cannot be empty.",
        )
    raise ValueError(
        "DEEPAGENT_MODEL_ENDPOINT_URL can only target "
        "provider-switched Anthropic or OpenAI-compatible profiles."
    )


def _resolve_cross_provider_endpoint(
    inputs: _ModelInputs,
    active_provider: ModelProvider,
    cross_provider_profile_selected: bool,
) -> tuple[str | None, tuple[tuple[str, str], ...]]:
    """Resolve the generic endpoint deferred to provider-switched profiles."""
    base_url_only = bool(inputs.base_url) and not inputs.endpoint_url
    base_url = inputs.base_url if base_url_only else None
    endpoint_query: tuple[tuple[str, str], ...] = ()
    if base_url_only and active_provider == "snowflake_cortex":
        base_url, endpoint_query = _normalize_endpoint_for_provider(
            "snowflake_cortex",
            inputs.base_url,
            full_endpoint=False,
        )
    elif cross_provider_profile_selected and inputs.base_url:
        base_url = runtime_model_config.normalize_model_base_url(
            inputs.base_url,
            required_message="The model base URL cannot be empty.",
        )
    if inputs.endpoint_url and cross_provider_profile_selected:
        base_url, endpoint_query = _normalize_endpoint_for_provider(
            active_provider,
            inputs.endpoint_url,
            full_endpoint=True,
        )
    return base_url, endpoint_query


def _resolve_primary_endpoint(
    inputs: _ModelInputs,
    model_defaults: ModelDefaults,
    cross_provider_profile_selected: bool,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Resolve the runtime provider's own base URL and endpoint query."""
    model_provider = inputs.provider
    if cross_provider_profile_selected and (inputs.base_url or inputs.endpoint_url):
        return model_defaults.base_url, model_defaults.endpoint_query
    if inputs.endpoint_url and model_provider in _ENDPOINT_URL_PROVIDERS:
        return _normalize_endpoint_for_provider(
            model_provider,
            inputs.endpoint_url,
            full_endpoint=True,
        )
    if model_provider == "anthropic":
        defaults_are_anthropic = model_defaults.provider == "anthropic"
        base_url = runtime_model_config.normalize_model_base_url(
            inputs.base_url or (model_defaults.base_url if defaults_are_anthropic else ""),
            default=DEFAULT_ANTHROPIC_BASE_URL,
        )
        return base_url, model_defaults.endpoint_query if defaults_are_anthropic else ()

    selected_base_url = inputs.base_url or inputs.base_url_alias or model_defaults.base_url
    endpoint_query = model_defaults.endpoint_query
    if model_provider == "snowflake_cortex":
        if inputs.base_url:
            base_url, endpoint_query = _normalize_endpoint_for_provider(
                model_provider,
                selected_base_url,
                full_endpoint=False,
            )
        else:
            base_url = model_defaults.base_url
    else:
        base_url = runtime_model_config.normalize_model_base_url(
            selected_base_url,
            required_message="The model base URL cannot be empty.",
        )
    if inputs.base_url or inputs.base_url_alias:
        endpoint_query = ()
    return base_url, endpoint_query


def _resolve_model_endpoints(
    inputs: _ModelInputs,
    file_config: FileConfig,
    model_name: str,
) -> _ModelEndpoints:
    """Resolve cross-provider endpoints first, then the primary endpoint."""
    model_defaults = file_config.model
    active_model_defaults = runtime_model_config.resolve_model_profile_defaults(
        model_defaults,
        file_config.model_profiles,
        model_name,
    )
    cross_provider_profile_selected = bool(
        model_name in file_config.model_profiles
        and active_model_defaults.provider != inputs.provider
    )
    cross_base_url, cross_endpoint_query = _resolve_cross_provider_endpoint(
        inputs,
        active_model_defaults.provider,
        cross_provider_profile_selected,
    )
    base_url, endpoint_query = _resolve_primary_endpoint(
        inputs,
        model_defaults,
        cross_provider_profile_selected,
    )
    return _ModelEndpoints(
        base_url=base_url,
        endpoint_query=endpoint_query,
        base_url_override=bool(
            inputs.base_url or inputs.base_url_alias or inputs.endpoint_url
        ),
        cross_provider_base_url=cross_base_url,
        cross_provider_endpoint_url=inputs.endpoint_url or None,
        cross_provider_endpoint_query=cross_endpoint_query,
    )


def _resolve_api_key(
    provider: ModelProvider,
    *,
    override: str | None,
    use_override: bool,
    default_api_key: str | None,
) -> str | None:
    """Resolve a model API key for ``provider``.

    Args:
        provider: Provider whose provider-specific env key applies.
        override: Normalized runtime API key override.
        use_override: Whether ``override`` wins outright, even when empty.
        default_api_key: Configured key used when no env key is set.

    Returns:
        The resolved API key, if any.
    """
    if use_override:
        return override
    if provider == "anthropic":
        provider_specific_api_key = runtime_model_config.normalize_optional_string(
            os.getenv("ANTHROPIC_API_KEY")
        )
    elif provider == "snowflake_cortex":
        provider_specific_api_key = runtime_model_config.normalize_optional_string(
            os.getenv("SNOWFLAKE_PAT")
        )
    else:
        provider_specific_api_key = None
    return (
        provider_specific_api_key
        or runtime_model_config.normalize_optional_string(os.getenv("DEEPAGENT_MODEL_API_KEY"))
        or default_api_key
    )


@dataclass(frozen=True)
class _RuntimeApiKey:
    """Runtime default API key and the explicit override it came from, if any."""

    value: str | None
    override: str | None


def _resolve_runtime_api_key(
    overrides: RuntimeConfigOverrides,
    model_provider: ModelProvider,
    model_defaults: ModelDefaults,
) -> _RuntimeApiKey:
    """Resolve the runtime default model's API key; an explicit override wins."""
    override = (
        runtime_model_config.normalize_optional_string(overrides.model_api_key)
        if overrides.model_api_key is not None
        else None
    )
    return _RuntimeApiKey(
        value=_resolve_api_key(
            model_provider,
            override=override,
            use_override=overrides.model_api_key is not None,
            default_api_key=(
                model_defaults.api_key
                if model_defaults.provider == model_provider
                else None
            ),
        ),
        override=override,
    )


def _validate_active_api_key(
    active_runtime_model: ModelDefaults,
    model_api_key_override: str | None,
) -> None:
    """Require an API key for the active model when its provider needs one."""
    active_runtime_api_key = _resolve_api_key(
        active_runtime_model.provider,
        override=model_api_key_override,
        # A blank override falls back here, unlike the runtime default key.
        use_override=bool(model_api_key_override),
        default_api_key=active_runtime_model.api_key,
    )
    if active_runtime_model.provider == "anthropic" and not active_runtime_api_key:
        raise ValueError(
            "Anthropic runtime requires DEEPAGENT_MODEL_API_KEY, "
            "ANTHROPIC_API_KEY, or [model].api_key."
        )
    if active_runtime_model.provider == "snowflake_cortex" and not active_runtime_api_key:
        raise ValueError(
            "Snowflake Cortex runtime requires a CLI API key, SNOWFLAKE_PAT, "
            "DEEPAGENT_MODEL_API_KEY, or [model].api_key."
        )


@dataclass(frozen=True)
class _GenerationSettings:
    """Resolved sampling, streaming and reasoning settings.

    Attributes:
        temperature: Model temperature.
        temperature_override: Whether temperature was overridden at runtime.
        disable_streaming: Streaming mode.
        disable_streaming_override: Whether streaming was overridden at runtime.
        reasoning: Default reasoning level.
        reasoning_override: Whether reasoning was overridden at runtime.
    """

    temperature: float
    temperature_override: bool
    disable_streaming: DisableStreaming
    disable_streaming_override: bool
    reasoning: ReasoningLevel
    reasoning_override: bool


def _resolve_disable_streaming(
    overrides: RuntimeConfigOverrides,
    model_defaults: ModelDefaults,
) -> tuple[DisableStreaming, bool]:
    """Resolve the streaming mode and whether it was overridden at runtime."""
    raw_disable_streaming = os.getenv("DEEPAGENT_MODEL_DISABLE_STREAMING")
    raw_disable_streaming_for_tool_calls = os.getenv(
        "DEEPAGENT_MODEL_DISABLE_STREAMING_FOR_TOOL_CALLS"
    )
    model_disable_streaming_override = bool(
        overrides.model_disable_streaming is not None
        or raw_disable_streaming is not None
        or raw_disable_streaming_for_tool_calls is not None
    )
    model_disable_streaming: DisableStreaming
    if overrides.model_disable_streaming is not None:
        model_disable_streaming = runtime_model_config.normalize_disable_streaming(
            overrides.model_disable_streaming
        )
    elif raw_disable_streaming is not None:
        model_disable_streaming = runtime_model_config.normalize_disable_streaming(raw_disable_streaming)
    elif raw_disable_streaming_for_tool_calls is not None:
        if runtime_model_config.normalize_disable_streaming_for_tool_calls(
            raw_disable_streaming_for_tool_calls
        ):
            model_disable_streaming = "tool_calling"
        else:
            model_disable_streaming = False
    else:
        model_disable_streaming = model_defaults.disable_streaming
    return model_disable_streaming, model_disable_streaming_override


def _resolve_generation_settings(
    overrides: RuntimeConfigOverrides,
    inputs: _ModelInputs,
    model_defaults: ModelDefaults,
) -> _GenerationSettings:
    """Resolve temperature, then streaming, then reasoning, in that order."""
    temperature = (
        runtime_model_config.normalize_model_temperature(overrides.model_temperature)
        if overrides.model_temperature is not None
        else model_defaults.temperature
    )
    disable_streaming, disable_streaming_override = _resolve_disable_streaming(
        overrides, model_defaults
    )
    return _GenerationSettings(
        temperature=temperature,
        temperature_override=overrides.model_temperature is not None,
        disable_streaming=disable_streaming,
        disable_streaming_override=disable_streaming_override,
        reasoning=runtime_model_config.normalize_reasoning_level(
            inputs.reasoning or inputs.reasoning_alias,
            default=model_defaults.reasoning_effort,
        ),
        reasoning_override=bool(inputs.reasoning or inputs.reasoning_alias),
    )


def _resolve_recursion_limit(
    overrides: RuntimeConfigOverrides,
    file_config: FileConfig,
) -> int:
    """Resolve the graph recursion limit from overrides, env, or the file config."""
    return runtime_extension_config.normalize_recursion_limit(
        (
            overrides.recursion_limit
            if overrides.recursion_limit is not None
            else os.getenv("DEEPAGENT_RECURSION_LIMIT")
        ),
        default=file_config.extensions.recursion_limit,
        field_name="DEEPAGENT_RECURSION_LIMIT",
    )


def _runtime_override_fields(
    endpoints: _ModelEndpoints,
    generation: _GenerationSettings,
) -> frozenset[str]:
    """Name the runtime default model fields that runtime overrides pinned."""
    return frozenset(
        {
            field_name
            for field_name, enabled in (
                ("base_url", endpoints.base_url_override),
                (
                    "cross_provider_base_url",
                    bool(
                        endpoints.cross_provider_base_url
                        and not endpoints.cross_provider_endpoint_url
                    ),
                ),
                (
                    "cross_provider_endpoint_url",
                    bool(endpoints.cross_provider_endpoint_url),
                ),
                ("temperature", generation.temperature_override),
                ("disable_streaming", generation.disable_streaming_override),
            )
            if enabled
        }
    )


def _build_runtime_default_model(
    model_defaults: ModelDefaults,
    model_provider: ModelProvider,
    selection: _ModelSelection,
    endpoints: _ModelEndpoints,
    api_key: str | None,
    generation: _GenerationSettings,
) -> ModelDefaults:
    """Build the runtime default model that named profiles inherit from."""
    return ModelDefaults(
        provider=model_provider,
        base_url=endpoints.base_url,
        endpoint_query=endpoints.endpoint_query,
        name=selection.default_name,
        api_key=api_key,
        models=model_defaults.models,
        name_is_explicit=True,
        reasoning_effort=generation.reasoning,
        thinking=model_defaults.thinking,
        temperature=generation.temperature,
        max_tokens=model_defaults.max_tokens,
        repeat_penalty=model_defaults.repeat_penalty,
        disable_streaming=generation.disable_streaming,
        modalities=model_defaults.modalities,
        cross_provider_base_url=endpoints.cross_provider_base_url,
        cross_provider_endpoint_url=endpoints.cross_provider_endpoint_url,
        cross_provider_endpoint_query=endpoints.cross_provider_endpoint_query,
        runtime_override_fields=_runtime_override_fields(endpoints, generation),
    )


def _resolve_rag(
    rag_config: RagConfig,
    disable_rag: bool,
    active_runtime_model: ModelDefaults,
    runtime_default_model: ModelDefaults,
) -> _RagResolution:
    """Resolve RAG settings against the embedding provider's model endpoint."""
    if not (rag_config.enabled and not disable_rag):
        return _RagResolution(requested=False)
    rag_embedding_provider = rag_config.embedding.provider
    if rag_embedding_provider == "auto" or rag_embedding_provider == active_runtime_model.provider:
        rag_model_provider = active_runtime_model.provider
        rag_model_base_url = active_runtime_model.base_url
    elif rag_embedding_provider == runtime_default_model.provider:
        rag_model_provider = runtime_default_model.provider
        rag_model_base_url = runtime_default_model.base_url
    elif rag_embedding_provider == "ollama":
        rag_model_provider = "ollama"
        rag_model_base_url = DEFAULT_OLLAMA_BASE_URL
    else:
        rag_model_provider = rag_embedding_provider
        rag_model_base_url = ""
    try:
        rag = resolve_rag_config(
            rag_config,
            model_provider=rag_model_provider,
            model_base_url=rag_model_base_url,
        )
    except ValueError as exc:
        return _RagResolution(requested=True, error=str(exc))
    return _RagResolution(requested=True, rag=rag)


@dataclass(frozen=True)
class RuntimeConfig:
    """Hold resolved runtime configuration and factory helpers.

    Attributes:
        database_url: URL for the database.
        model_provider: The model provider value.
        model_name: The model name value.
        model_choices: The model choices value.
        model_base_url: URL for the model base.
        model_api_key: The model API key value.
        model_temperature: The model temperature value.
        model_max_tokens: Maximum output tokens for one model response.
        default_reasoning: The default reasoning value.
        persistence_mode: The persistence mode value.
        agent_state: Whether the DeepAgents graph is stateful or stateless.
        extensions: The extensions value.
        langfuse: Langfuse tracing configuration.
        model_repeat_penalty: The model repeat penalty value.
        recursion_limit: The recursion limit value.
        rag_requested: The RAG requested value.
        rag: The RAG value.
        rag_error: The RAG error value.
        model_endpoint_query: The model endpoint query value.
        model_disable_streaming: Whether to disable model streaming.
        model_thinking: Anthropic thinking mode.
        model_profiles: Named model profiles available by profile name.
        model_api_key_override: Explicit runtime API key override.
        model_default_name: Runtime default model name before agent/profile selection.
        model_default_choices: Runtime default model list before profile names are added.
        model_reasoning_override: Whether reasoning was explicitly overridden at runtime.
        model_base_url_override: Whether the model endpoint was overridden at runtime.
        model_cross_provider_base_url_override: Whether a generic runtime endpoint
            override may apply across profile provider boundaries.
        model_cross_provider_base_url: Generic runtime endpoint for provider-switched
            profiles.
        model_cross_provider_endpoint_url: Unnormalized generic full endpoint for
            provider-switched profiles.
        model_cross_provider_endpoint_query: Generic runtime endpoint query for
            provider-switched profiles.
        model_temperature_override: Whether temperature was overridden at runtime.
        model_disable_streaming_override: Whether streaming was overridden at runtime.
    """

    database_url: str | None
    model_provider: ModelProvider
    model_name: str
    model_choices: tuple[str, ...]
    model_base_url: str
    model_api_key: str | None
    model_temperature: float
    default_reasoning: ReasoningLevel
    persistence_mode: PersistenceMode
    extensions: ExtensionsConfig
    model_max_tokens: int | None = None
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    langsmith: LangSmithConfig = field(default_factory=LangSmithConfig)
    agent_state: AgentStateMode = DEFAULT_AGENT_STATE
    model_repeat_penalty: float | None = None
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    rag_requested: bool = False
    rag: ResolvedRagConfig | None = None
    rag_error: str | None = None
    model_endpoint_query: tuple[tuple[str, str], ...] = ()
    model_disable_streaming: DisableStreaming = False
    model_thinking: ModelThinking = DEFAULT_MODEL_THINKING
    model_modalities: tuple[ModelModality, ...] = ("text",)
    model_profiles: dict[str, ModelDefaults] = field(default_factory=dict)
    model_api_key_override: str | None = None
    model_default_name: str | None = None
    model_default_choices: tuple[str, ...] = ()
    model_reasoning_override: bool = False
    model_base_url_override: bool = False
    model_cross_provider_base_url_override: bool = False
    model_cross_provider_base_url: str | None = None
    model_cross_provider_endpoint_url: str | None = None
    model_cross_provider_endpoint_query: tuple[tuple[str, str], ...] = ()
    model_temperature_override: bool = False
    model_disable_streaming_override: bool = False

    @classmethod
    def from_env(
        cls,
        overrides: RuntimeConfigOverrides | None = None,
    ) -> RuntimeConfig:
        """Create this object from environment.

        Args:
            overrides: The overrides value.

        Returns:
            The created this object from environment.

        Raises:
            ValueError: If the supplied value is invalid.
        """
        overrides = overrides or RuntimeConfigOverrides()
        file_config = load_file_config(overrides.config_path)
        model_defaults = file_config.model
        database_url = _resolve_database_url(overrides)
        inputs = _collect_model_inputs(overrides, model_defaults)
        selection = _select_model_profile(inputs, file_config)
        profile_endpoint_only = _validate_provider_switch(inputs, selection, model_defaults)
        model_choices = _model_choices(inputs, selection, file_config, profile_endpoint_only)
        endpoints = _resolve_model_endpoints(inputs, file_config, selection.model_name)
        api_key = _resolve_runtime_api_key(overrides, inputs.provider, model_defaults)
        generation = _resolve_generation_settings(overrides, inputs, model_defaults)
        recursion_limit = _resolve_recursion_limit(overrides, file_config)
        runtime_default_model = _build_runtime_default_model(
            model_defaults, inputs.provider, selection, endpoints, api_key.value, generation
        )
        active_runtime_model = runtime_model_config.resolve_model_profile_defaults(
            runtime_default_model, file_config.model_profiles, selection.model_name
        )
        _validate_active_api_key(active_runtime_model, api_key.override)
        rag = _resolve_rag(
            file_config.rag, overrides.disable_rag, active_runtime_model, runtime_default_model
        )

        return cls(
            database_url=database_url,
            model_provider=inputs.provider,
            model_name=selection.model_name,
            model_choices=model_choices,
            model_base_url=endpoints.base_url,
            model_api_key=api_key.value,
            model_temperature=generation.temperature,
            model_max_tokens=model_defaults.max_tokens,
            model_repeat_penalty=model_defaults.repeat_penalty,
            default_reasoning=generation.reasoning,
            persistence_mode="postgres" if database_url else "memory",
            agent_state=file_config.extensions.agent_state,
            extensions=file_config.extensions,
            langfuse=file_config.langfuse,
            langsmith=file_config.langsmith,
            recursion_limit=recursion_limit,
            rag_requested=rag.requested,
            rag=rag.rag,
            rag_error=rag.error,
            model_endpoint_query=endpoints.endpoint_query,
            model_disable_streaming=generation.disable_streaming,
            model_thinking=model_defaults.thinking,
            model_modalities=model_defaults.modalities,
            model_profiles=file_config.model_profiles,
            model_api_key_override=api_key.override,
            model_default_name=selection.default_name,
            model_default_choices=model_defaults.models,
            model_reasoning_override=generation.reasoning_override,
            model_base_url_override=endpoints.base_url_override,
            model_cross_provider_base_url_override=bool(
                endpoints.cross_provider_base_url or endpoints.cross_provider_endpoint_url
            ),
            model_cross_provider_base_url=endpoints.cross_provider_base_url,
            model_cross_provider_endpoint_url=endpoints.cross_provider_endpoint_url,
            model_cross_provider_endpoint_query=endpoints.cross_provider_endpoint_query,
            model_temperature_override=generation.temperature_override,
            model_disable_streaming_override=generation.disable_streaming_override,
        )
