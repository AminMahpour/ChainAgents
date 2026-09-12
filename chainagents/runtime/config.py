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
    langfuse: LangfuseConfig = LangfuseConfig()
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
    ) -> "RuntimeConfig":
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
        if overrides.disable_database:
            database_url = None
        elif overrides.database_url is not None:
            database_url = runtime_model_config.normalize_optional_string(overrides.database_url)
        else:
            database_url = os.getenv("DATABASE_URL", "").strip() or None
        model_provider_override = runtime_model_config.normalize_optional_string(
            overrides.model_provider
        )
        if model_provider_override is None:
            model_provider_override = runtime_model_config.normalize_optional_string(
                os.getenv("DEEPAGENT_MODEL_PROVIDER")
            )
        model_provider = runtime_model_config.normalize_model_provider(
            model_provider_override,
            default=model_defaults.provider,
        )
        generic_model_name = (
            runtime_model_config.normalize_optional_string(overrides.model_name)
            or os.getenv("DEEPAGENT_MODEL_NAME", "").strip()
        )
        override_model_base_url = runtime_model_config.normalize_optional_string(overrides.model_base_url)
        env_model_base_url = runtime_model_config.normalize_optional_string(
            os.getenv("DEEPAGENT_MODEL_BASE_URL")
        )
        generic_model_base_url = override_model_base_url or env_model_base_url or ""
        generic_model_base_url_override = bool(generic_model_base_url)
        generic_model_base_url_from_env = (
            override_model_base_url is None and env_model_base_url is not None
        )
        generic_model_endpoint_url = (
            runtime_model_config.normalize_optional_string(overrides.model_endpoint_url)
            or os.getenv("DEEPAGENT_MODEL_ENDPOINT_URL", "").strip()
        )
        generic_model_reasoning = (
            runtime_model_config.normalize_optional_string(overrides.reasoning_level)
            or os.getenv("DEEPAGENT_MODEL_REASONING", "").strip()
        )
        model_name_alias = (
            os.getenv("OLLAMA_MODEL", "").strip()
            if model_provider == "ollama"
            else ""
        )
        model_base_url_alias = (
            os.getenv("OLLAMA_BASE_URL", "").strip()
            if model_provider == "ollama"
            else ""
        )
        model_reasoning_alias = (
            os.getenv("OLLAMA_REASONING", "").strip()
            if model_provider == "ollama"
            else ""
        )

        provider_changed = (
            bool(model_provider_override) and model_provider != model_defaults.provider
        )
        model_name = (
            generic_model_name
            or model_name_alias
            or (
                file_config.extensions.agent_model
                if not provider_changed
                else None
            )
            or model_defaults.name
        )
        model_name_override = generic_model_name or model_name_alias
        model_default_name = (
            model_defaults.name
            if model_name_override in file_config.model_profiles
            else model_name_override or model_defaults.name
        )
        selected_override_profile = file_config.model_profiles.get(
            model_name_override or ""
        )
        if (
            model_provider_override
            and selected_override_profile is not None
            and selected_override_profile.provider != model_provider
        ):
            raise ValueError(
                f"Model provider override '{model_provider}' does not match "
                f"selected profile '{model_name_override}' provider "
                f"'{selected_override_profile.provider}'."
            )
        profile_endpoint_satisfies_provider_switch = bool(
            selected_override_profile is not None
            and selected_override_profile.provider == model_provider
            and selected_override_profile.base_url
        )
        endpoint_url_satisfies_provider_switch = (
            model_provider in OPENAI_COMPATIBLE_MODEL_PROVIDERS
            and bool(generic_model_endpoint_url)
        )
        profile_endpoint_only_satisfies_provider_switch = (
            provider_changed
            and profile_endpoint_satisfies_provider_switch
            and not generic_model_base_url
            and not endpoint_url_satisfies_provider_switch
        )
        provider_switch_requires_url = (
            provider_changed
            and model_provider in {"ollama", *OPENAI_COMPATIBLE_MODEL_PROVIDERS}
            and not generic_model_base_url
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
            and generic_model_base_url
            and generic_model_base_url_from_env
            and not generic_model_endpoint_url
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
            and not generic_model_name
            and not model_defaults.name_is_explicit
        ):
            provider_label = runtime_model_config.format_model_provider(model_provider)
            raise ValueError(
                f"{provider_label} runtime must define DEEPAGENT_MODEL_NAME "
                "or set a non-empty [model].name in deepagent.toml."
            )
        if (
            model_provider == "anthropic"
            and not generic_model_name
            and (provider_changed or not model_defaults.name_is_explicit)
        ):
            raise ValueError(
                "Anthropic runtime must define DEEPAGENT_MODEL_NAME "
                "or set a non-empty [model].name in deepagent.toml."
            )

        default_model_choices = (
            ()
            if profile_endpoint_only_satisfies_provider_switch
            else (model_defaults.name, *model_defaults.models)
        )
        profile_choices = tuple(
            profile_name
            for profile_name, profile in file_config.model_profiles.items()
            if not (provider_changed and model_provider_override)
            or profile.provider == model_provider
        )
        model_choices = tuple(
            dict.fromkeys(
                [
                    model_name,
                    *default_model_choices,
                    *profile_choices,
                ]
            )
        )
        active_model_defaults = runtime_model_config.resolve_model_profile_defaults(
            model_defaults,
            file_config.model_profiles,
            model_name,
        )
        cross_provider_profile_selected = bool(
            model_name in file_config.model_profiles
            and active_model_defaults.provider != model_provider
        )
        cross_provider_model_base_url = (
            generic_model_base_url
            if generic_model_base_url_override and not generic_model_endpoint_url
            else None
        )
        cross_provider_model_endpoint_url = generic_model_endpoint_url or None
        cross_provider_model_endpoint_query: tuple[tuple[str, str], ...] = ()
        model_endpoint_query = model_defaults.endpoint_query
        if (
            generic_model_base_url_override
            and not generic_model_endpoint_url
            and active_model_defaults.provider == "snowflake_cortex"
        ):
            (
                cross_provider_model_base_url,
                cross_provider_model_endpoint_query,
            ) = runtime_model_config.normalize_snowflake_cortex_endpoint_url(
                generic_model_base_url,
                full_endpoint=False,
                required_message="The Snowflake Cortex model base URL cannot be empty.",
            )
        elif cross_provider_profile_selected and generic_model_base_url_override:
            cross_provider_model_base_url = runtime_model_config.normalize_model_base_url(
                generic_model_base_url,
                required_message="The model base URL cannot be empty.",
            )
        if (
            generic_model_endpoint_url
            and cross_provider_profile_selected
        ):
            if active_model_defaults.provider == "anthropic":
                (
                    cross_provider_model_base_url,
                    cross_provider_model_endpoint_query,
                ) = runtime_model_config.normalize_anthropic_endpoint_url(
                    generic_model_endpoint_url,
                    required_message=(
                        "The Anthropic model endpoint URL cannot be empty."
                    ),
                )
            elif active_model_defaults.provider == "snowflake_cortex":
                (
                    cross_provider_model_base_url,
                    cross_provider_model_endpoint_query,
                ) = runtime_model_config.normalize_snowflake_cortex_endpoint_url(
                    generic_model_endpoint_url,
                    full_endpoint=True,
                    required_message="The Snowflake Cortex model endpoint URL cannot be empty.",
                )
            elif active_model_defaults.provider == "openai_compatible":
                (
                    cross_provider_model_base_url,
                    cross_provider_model_endpoint_query,
                ) = runtime_model_config.normalize_openai_endpoint_url(
                    generic_model_endpoint_url,
                    required_message="The model endpoint URL cannot be empty.",
                )
            else:
                raise ValueError(
                    "DEEPAGENT_MODEL_ENDPOINT_URL can only target "
                    "provider-switched Anthropic or OpenAI-compatible profiles."
                )
        if cross_provider_profile_selected and (
            generic_model_base_url or generic_model_endpoint_url
        ):
            model_base_url = model_defaults.base_url
            model_endpoint_query = model_defaults.endpoint_query
        elif model_provider == "anthropic":
            if generic_model_endpoint_url:
                model_base_url, model_endpoint_query = runtime_model_config.normalize_anthropic_endpoint_url(
                    generic_model_endpoint_url,
                    required_message="The Anthropic model endpoint URL cannot be empty.",
                )
            else:
                model_base_url = runtime_model_config.normalize_model_base_url(
                    (
                        generic_model_base_url
                        or (
                            model_defaults.base_url
                            if model_defaults.provider == "anthropic"
                            else ""
                        )
                    ),
                    default=DEFAULT_ANTHROPIC_BASE_URL,
                )
                model_endpoint_query = (
                    model_defaults.endpoint_query
                    if model_defaults.provider == "anthropic"
                    else ()
                )
        elif model_provider == "snowflake_cortex" and generic_model_endpoint_url:
            model_base_url, model_endpoint_query = runtime_model_config.normalize_snowflake_cortex_endpoint_url(
                generic_model_endpoint_url,
                full_endpoint=True,
                required_message="The Snowflake Cortex model endpoint URL cannot be empty.",
            )
        elif model_provider == "openai_compatible" and generic_model_endpoint_url:
            model_base_url, model_endpoint_query = runtime_model_config.normalize_openai_endpoint_url(
                generic_model_endpoint_url,
                required_message="The model endpoint URL cannot be empty.",
            )
        else:
            selected_base_url = (
                generic_model_base_url
                or model_base_url_alias
                or model_defaults.base_url
            )
            if model_provider == "snowflake_cortex":
                if generic_model_base_url:
                    (
                        model_base_url,
                        model_endpoint_query,
                    ) = runtime_model_config.normalize_snowflake_cortex_endpoint_url(
                        selected_base_url,
                        full_endpoint=False,
                        required_message=(
                            "The Snowflake Cortex model base URL cannot be empty."
                        ),
                    )
                else:
                    model_base_url = model_defaults.base_url
            else:
                model_base_url = runtime_model_config.normalize_model_base_url(
                    selected_base_url,
                    required_message="The model base URL cannot be empty.",
                )
            if generic_model_base_url or model_base_url_alias:
                model_endpoint_query = ()
        model_api_key_override = (
            runtime_model_config.normalize_optional_string(overrides.model_api_key)
            if overrides.model_api_key is not None
            else None
        )
        if overrides.model_api_key is not None:
            model_api_key = model_api_key_override
        else:
            provider_specific_api_key = (
                runtime_model_config.normalize_optional_string(os.getenv("ANTHROPIC_API_KEY"))
                if model_provider == "anthropic"
                else (
                    runtime_model_config.normalize_optional_string(os.getenv("SNOWFLAKE_PAT"))
                    if model_provider == "snowflake_cortex"
                    else None
                )
            )
            model_default_api_key = (
                model_defaults.api_key
                if model_defaults.provider == model_provider
                else None
            )
            model_api_key = (
                provider_specific_api_key
                or runtime_model_config.normalize_optional_string(os.getenv("DEEPAGENT_MODEL_API_KEY"))
                or model_default_api_key
            )
        model_temperature = (
            runtime_model_config.normalize_model_temperature(overrides.model_temperature)
            if overrides.model_temperature is not None
            else model_defaults.temperature
        )
        model_temperature_override = overrides.model_temperature is not None
        model_repeat_penalty = model_defaults.repeat_penalty
        raw_disable_streaming = os.getenv("DEEPAGENT_MODEL_DISABLE_STREAMING")
        raw_disable_streaming_for_tool_calls = os.getenv(
            "DEEPAGENT_MODEL_DISABLE_STREAMING_FOR_TOOL_CALLS"
        )
        model_disable_streaming_override = bool(
            overrides.model_disable_streaming is not None
            or raw_disable_streaming is not None
            or raw_disable_streaming_for_tool_calls is not None
        )
        if overrides.model_disable_streaming is not None:
            model_disable_streaming = runtime_model_config.normalize_disable_streaming(
                overrides.model_disable_streaming
            )
        else:
            if raw_disable_streaming is not None:
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
        default_reasoning = runtime_model_config.normalize_reasoning_level(
            generic_model_reasoning or model_reasoning_alias,
            default=model_defaults.reasoning_effort,
        )
        model_reasoning_override = bool(generic_model_reasoning or model_reasoning_alias)
        recursion_limit = runtime_extension_config.normalize_recursion_limit(
            (
                overrides.recursion_limit
                if overrides.recursion_limit is not None
                else os.getenv("DEEPAGENT_RECURSION_LIMIT")
            ),
            default=file_config.extensions.recursion_limit,
            field_name="DEEPAGENT_RECURSION_LIMIT",
        )
        runtime_default_model = ModelDefaults(
            provider=model_provider,
            base_url=model_base_url,
            endpoint_query=model_endpoint_query,
            name=model_default_name,
            api_key=model_api_key,
            models=model_defaults.models,
            name_is_explicit=True,
            reasoning_effort=default_reasoning,
            thinking=model_defaults.thinking,
            temperature=model_temperature,
            repeat_penalty=model_repeat_penalty,
            disable_streaming=model_disable_streaming,
            modalities=model_defaults.modalities,
            cross_provider_base_url=cross_provider_model_base_url,
            cross_provider_endpoint_url=cross_provider_model_endpoint_url,
            cross_provider_endpoint_query=cross_provider_model_endpoint_query,
            runtime_override_fields=(
                frozenset(
                    {
                        field_name
                        for field_name, enabled in (
                            (
                                "base_url",
                                bool(
                                    generic_model_base_url
                                    or model_base_url_alias
                                    or generic_model_endpoint_url
                                ),
                            ),
                            (
                                "cross_provider_base_url",
                                bool(
                                    cross_provider_model_base_url
                                    and not cross_provider_model_endpoint_url
                                ),
                            ),
                            (
                                "cross_provider_endpoint_url",
                                bool(cross_provider_model_endpoint_url),
                            ),
                            ("temperature", model_temperature_override),
                            ("disable_streaming", model_disable_streaming_override),
                        )
                        if enabled
                    }
                )
            ),
        )
        active_runtime_model = runtime_model_config.resolve_model_profile_defaults(
            runtime_default_model,
            file_config.model_profiles,
            model_name,
        )
        active_runtime_provider_key = (
            runtime_model_config.normalize_optional_string(os.getenv("ANTHROPIC_API_KEY"))
            if active_runtime_model.provider == "anthropic"
            else (
                runtime_model_config.normalize_optional_string(os.getenv("SNOWFLAKE_PAT"))
                if active_runtime_model.provider == "snowflake_cortex"
                else None
            )
        )
        active_runtime_api_key = (
            model_api_key_override
            or active_runtime_provider_key
            or runtime_model_config.normalize_optional_string(os.getenv("DEEPAGENT_MODEL_API_KEY"))
            or active_runtime_model.api_key
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
        rag_requested = file_config.rag.enabled and not overrides.disable_rag
        rag = None
        rag_error = None
        if rag_requested:
            rag_embedding_provider = file_config.rag.embedding.provider
            if rag_embedding_provider == "auto":
                rag_model_provider = active_runtime_model.provider
                rag_model_base_url = active_runtime_model.base_url
            elif rag_embedding_provider == active_runtime_model.provider:
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
                    file_config.rag,
                    model_provider=rag_model_provider,
                    model_base_url=rag_model_base_url,
                )
            except ValueError as exc:
                rag_error = str(exc)

        return cls(
            database_url=database_url,
            model_provider=model_provider,
            model_name=model_name,
            model_choices=model_choices,
            model_base_url=model_base_url,
            model_api_key=model_api_key,
            model_temperature=model_temperature,
            model_repeat_penalty=model_repeat_penalty,
            default_reasoning=default_reasoning,
            persistence_mode="postgres" if database_url else "memory",
            agent_state=file_config.extensions.agent_state,
            extensions=file_config.extensions,
            langfuse=file_config.langfuse,
            recursion_limit=recursion_limit,
            rag_requested=rag_requested,
            rag=rag,
            rag_error=rag_error,
            model_endpoint_query=model_endpoint_query,
            model_disable_streaming=model_disable_streaming,
            model_thinking=model_defaults.thinking,
            model_modalities=model_defaults.modalities,
            model_profiles=file_config.model_profiles,
            model_api_key_override=model_api_key_override,
            model_default_name=model_default_name,
            model_default_choices=model_defaults.models,
            model_reasoning_override=model_reasoning_override,
            model_base_url_override=(
                bool(
                    generic_model_base_url
                    or model_base_url_alias
                    or generic_model_endpoint_url
                )
            ),
            model_cross_provider_base_url_override=bool(
                cross_provider_model_base_url or cross_provider_model_endpoint_url
            ),
            model_cross_provider_base_url=cross_provider_model_base_url,
            model_cross_provider_endpoint_url=cross_provider_model_endpoint_url,
            model_cross_provider_endpoint_query=(
                cross_provider_model_endpoint_query
            ),
            model_temperature_override=model_temperature_override,
            model_disable_streaming_override=model_disable_streaming_override,
        )
