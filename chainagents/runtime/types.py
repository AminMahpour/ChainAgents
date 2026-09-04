"""Configuration value objects and shared runtime records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from deepagents import AsyncSubAgent
from langchain.agents.middleware.types import AgentMiddleware

from chainagents.rag.runtime import RagConfig
from chainagents.runtime.constants import (
    DEFAULT_AGENT_MEMORY_FILES,
    DEFAULT_AGENT_MEMORY_NAMESPACE,
    DEFAULT_AGENT_STATE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PROVIDER,
    DEFAULT_MODEL_THINKING,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_REASONING_LEVEL,
    DEFAULT_RECURSION_LIMIT,
    DEFAULT_TEMPERATURE,
    AgentStateMode,
    DisableStreaming,
    ModelModality,
    ModelProvider,
    ModelThinking,
    ReasoningLevel,
)
from chainagents.runtime.reflection import ReflectionConfig


@dataclass(frozen=True)
class SubagentConfig:
    """Describe a synchronous subagent from the deepagent configuration.

    Attributes:
        name: The name value.
        description: The description value.
        system_prompt: The system prompt value.
        skills: The skills value.
        mcp_servers: The MCP servers value.
        model: Model name or model object used by the runtime.
        nested_subagent_names: Top-level sync subagent names exposed to this subagent.
        subagents: Inline private sync subagents exposed to this subagent.
    """

    name: str
    description: str
    system_prompt: str
    skills: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()
    model: str | None = None
    nested_subagent_names: tuple[str, ...] = ()
    subagents: tuple["SubagentConfig", ...] = ()

    def to_deepagents_spec(
        self,
        *,
        tools: list[Any] | None = None,
        middleware: list[AgentMiddleware[Any, Any, Any]] | None = None,
        model: Any | None = None,
    ) -> dict[str, Any]:
        """Convert this object to deepagents spec.

        Args:
            tools: The tools value.
            middleware: The middleware value.
            model: Resolved model object for this subagent.

        Returns:
            The converted value.
        """
        spec: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
        }
        if self.skills:
            spec["skills"] = list(self.skills)
        if tools:
            spec["tools"] = tools
        if middleware:
            spec["middleware"] = list(middleware)
        if model is not None:
            spec["model"] = model
        elif self.model:
            spec["model"] = self.model
        return spec


@dataclass(frozen=True)
class AsyncSubagentConfig:
    """Describe an async subagent that runs through the Agent Protocol.

    Attributes:
        name: The name value.
        description: The description value.
        graph_id: Graph identifier.
        url: The URL value.
        headers: The headers value.
    """

    name: str
    description: str
    graph_id: str
    url: str | None = None
    headers: dict[str, str] | None = None

    def to_deepagents_spec(
        self,
        *,
        url_override: str | None = None,
    ) -> AsyncSubAgent:
        """Convert this object to deepagents spec.

        Args:
            url_override: Agent Protocol URL override, if one is configured.

        Returns:
            The converted value.
        """
        spec: AsyncSubAgent = {
            "name": self.name,
            "description": self.description,
            "graph_id": self.graph_id,
        }
        url = self.url or url_override
        if url:
            spec["url"] = url
        if self.headers:
            spec["headers"] = dict(self.headers)
        return spec


@dataclass(frozen=True)
class ChainlitCommandConfig:
    """Describe a native Chainlit command backed by a configured target.

    Attributes:
        name: The name value.
        description: The description value.
        target: The target value.
        value: Value to normalize, convert, or serialize.
        template: Template string applied to command input.
        mcp_server: The MCP server value.
        source: The source value.
    """

    name: str
    description: str
    target: Literal["prompt", "subagent", "mcp_tool", "skill"]
    value: str
    template: str | None = None
    mcp_server: str | None = None
    source: Literal["config", "agent_skill", "subagent_skill"] = "config"


@dataclass(frozen=True)
class ChainlitStarterConfig:
    """Describe a Chainlit starter prompt exposed at thread start.

    Attributes:
        label: Starter label shown in the Chainlit UI.
        message: Message sent when the starter is selected.
        command: Optional Chainlit command associated with the starter.
        icon: Optional icon name shown by Chainlit.
    """

    label: str
    message: str
    command: str | None = None
    icon: str | None = None


@dataclass(frozen=True)
class SkillCommandMetadata:
    """Track metadata required to expose a configured skill as a command.

    Attributes:
        name: The name value.
        description: The description value.
        path: Filesystem path to read or write.
        source: The source value.
        owner: The owner value.
    """

    name: str
    description: str
    path: str
    source: Literal["agent_skill", "subagent_skill"]
    owner: str | None = None

    @property
    def label(self) -> str:
        """Return the display label for a skill command.

        Returns:
            The display label for a skill command.
        """
        if self.source == "agent_skill":
            return f"main agent skill `{self.path}`"
        if self.owner:
            return f"subagent `{self.owner}` skill `{self.path}`"
        return f"subagent skill `{self.path}`"

    def to_chainlit_command(self) -> ChainlitCommandConfig:
        """Convert this object to chainlit command.

        Returns:
            The converted value.
        """
        return ChainlitCommandConfig(
            name=self.name,
            description=self.description,
            target="skill",
            value=self.path,
            source=self.source,
        )


@dataclass(frozen=True)
class LangfuseConfig:
    """Store Langfuse tracing configuration parsed from TOML.

    Attributes:
        enabled: Whether to attach the Langfuse LangChain callback handler.
    """

    enabled: bool = False


@dataclass(frozen=True)
class ExtensionsConfig:
    """Store optional runtime extension settings parsed from configuration.

    Attributes:
        config_path: Path to the config.
        mcp_tool_name_prefix: The MCP tool name prefix value.
        mcp_stateful: The MCP stateful value.
        agent_state: Whether the DeepAgents graph is stateful or stateless.
        agent_memory_namespace: Shared StoreBackend namespace for /memories/.
        agent_memory_files: Startup memory files loaded into the agent prompt.
        delete_tool_enabled: Whether to expose DeepAgents' recursive delete tool.
        execute_tool_enabled: Whether to expose DeepAgents' execute tool.
        agent_reflection: Correction reflection workflow configuration.
        agent_model: Optional main-agent model profile or raw model name.
        recursion_limit: The recursion limit value.
        mcp_servers: The MCP servers value.
        skills: The skills value.
        agent_mcp_servers: The agent MCP servers value.
        subagents: The subagents value.
        async_subagents: Async subagent configurations available for monitoring.
        chainlit_commands: The chainlit commands value.
        chainlit_starters: The chainlit starters value.
        chainlit_model_mode_enabled: The chainlit model mode enabled value.
        chainlit_reasoning_mode_enabled: The chainlit reasoning mode enabled value.
        chainlit_reasoning_steps_enabled: The chainlit reasoning steps enabled value.
        chainlit_tool_steps_enabled: The chainlit tool steps enabled value.
        chainlit_startup_status_enabled: The chainlit startup status enabled value.
        chainlit_chronological_ui_enabled: The chainlit chronological UI enabled value.
        chainlit_generative_ui_enabled: Whether Chainlit generated UI is enabled.
        summarization_middleware_enabled: The summarization middleware enabled value.
        summarization_trigger_tokens: The summarization trigger tokens value.
        summarization_keep_tokens: The summarization keep tokens value.
        custom_instruction: Inline or file-loaded main-agent custom instruction.
    """

    config_path: Path | None
    mcp_tool_name_prefix: bool = True
    mcp_stateful: bool = False
    agent_state: AgentStateMode = DEFAULT_AGENT_STATE
    agent_memory_namespace: str = DEFAULT_AGENT_MEMORY_NAMESPACE
    agent_memory_files: tuple[str, ...] = DEFAULT_AGENT_MEMORY_FILES
    delete_tool_enabled: bool = False
    execute_tool_enabled: bool = False
    agent_reflection: ReflectionConfig = ReflectionConfig()
    agent_model: str | None = None
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    mcp_servers: dict[str, dict[str, Any]] | None = None
    skills: tuple[str, ...] = ()
    agent_mcp_servers: tuple[str, ...] = ()
    subagents: tuple[SubagentConfig, ...] = ()
    async_subagents: tuple[AsyncSubagentConfig, ...] = ()
    chainlit_commands: tuple[ChainlitCommandConfig, ...] = ()
    chainlit_starters: tuple[ChainlitStarterConfig, ...] = ()
    chainlit_model_mode_enabled: bool = True
    chainlit_reasoning_mode_enabled: bool = True
    chainlit_reasoning_steps_enabled: bool = True
    chainlit_tool_steps_enabled: bool = True
    chainlit_startup_status_enabled: bool = True
    chainlit_chronological_ui_enabled: bool = True
    chainlit_generative_ui_enabled: bool = True
    summarization_middleware_enabled: bool = False
    summarization_trigger_tokens: int | None = None
    summarization_keep_tokens: int | None = None
    custom_instruction: str | None = None

    @property
    def enabled(self) -> bool:
        """Return whether optional runtime extensions are configured.

        Returns:
            True when at least one optional extension is configured; otherwise, False.
        """
        return bool(
            self.skills
            or self.delete_tool_enabled
            or self.execute_tool_enabled
            or self.agent_mcp_servers
            or self.subagents
            or self.async_subagents
            or self.chainlit_commands
            or self.chainlit_starters
        )


@dataclass(frozen=True)
class ModelDefaults:
    """Store resolved model provider defaults for the runtime.

    Attributes:
        provider: The provider value.
        base_url: URL for the base.
        endpoint_query: The endpoint query value.
        name: The name value.
        api_key: The API key value.
        models: The models value.
        name_is_explicit: The name is explicit value.
        reasoning_effort: The reasoning effort value.
        thinking: The Anthropic thinking mode.
        temperature: The temperature value.
        repeat_penalty: The repeat penalty value.
        disable_streaming: Whether to disable model streaming.
        cross_provider_base_url: Runtime endpoint override for provider-switched
            profiles.
        cross_provider_endpoint_url: Unnormalized full runtime endpoint override for
            provider-switched profiles.
        cross_provider_endpoint_query: Runtime endpoint query for provider-switched
            profiles.
        explicit_fields: Profile fields explicitly set in TOML.
        runtime_override_fields: Fields explicitly overridden at runtime.
    """

    provider: ModelProvider = DEFAULT_MODEL_PROVIDER
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    endpoint_query: tuple[tuple[str, str], ...] = ()
    name: str = DEFAULT_MODEL
    api_key: str | None = None
    models: tuple[str, ...] = ()
    name_is_explicit: bool = False
    reasoning_effort: ReasoningLevel = DEFAULT_REASONING_LEVEL
    thinking: ModelThinking = DEFAULT_MODEL_THINKING
    temperature: float = DEFAULT_TEMPERATURE
    repeat_penalty: float | None = None
    disable_streaming: DisableStreaming = False
    modalities: tuple[ModelModality, ...] = ("text",)
    cross_provider_base_url: str | None = None
    cross_provider_endpoint_url: str | None = None
    cross_provider_endpoint_query: tuple[tuple[str, str], ...] = ()
    explicit_fields: frozenset[str] = field(
        default_factory=frozenset,
        compare=False,
        repr=False,
    )
    runtime_override_fields: frozenset[str] = field(
        default_factory=frozenset,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class FileConfig:
    """Store resolved virtual file-system settings for the runtime.

    Attributes:
        model: Model name or model object used by the runtime.
        model_profiles: Named model profiles available to agents.
        extensions: The extensions value.
        langfuse: Langfuse tracing configuration.
        rag: The RAG value.
    """

    model: ModelDefaults
    extensions: ExtensionsConfig
    model_profiles: dict[str, ModelDefaults] = field(default_factory=dict)
    langfuse: LangfuseConfig = LangfuseConfig()
    rag: RagConfig = RagConfig()


@dataclass(frozen=True)
class RuntimeConfigOverrides:
    """Capture CLI-provided runtime settings that override defaults.

    Attributes:
        config_path: Path to the config.
        database_url: URL for the database.
        disable_database: The disable database value.
        model_provider: The model provider value.
        model_name: The model name value.
        model_base_url: URL for the model base.
        model_endpoint_url: URL for the model endpoint.
        model_api_key: The model API key value.
        model_temperature: The model temperature value.
        reasoning_level: The reasoning level value.
        recursion_limit: The recursion limit value.
        disable_rag: The disable RAG value.
        model_disable_streaming: Whether to disable model streaming.
    """

    config_path: str | Path | None = None
    database_url: str | None = None
    disable_database: bool = False
    model_provider: str | None = None
    model_name: str | None = None
    model_base_url: str | None = None
    model_endpoint_url: str | None = None
    model_api_key: str | None = None
    model_temperature: float | None = None
    reasoning_level: str | None = None
    recursion_limit: int | None = None
    disable_rag: bool = False
    model_disable_streaming: DisableStreaming | None = None


@dataclass(frozen=True)
class AppSettings:
    """Store user-selected Chainlit settings for a chat session.

    Attributes:
        reasoning_level: The reasoning level value.
        thread_id: Conversation thread identifier.
        model_name: The model name value.
        show_reasoning_stream: Whether to show streamed reasoning UI.
        show_tool_calls: Whether to show streamed tool-call UI.
    """

    reasoning_level: ReasoningLevel
    thread_id: str
    model_name: str
    show_reasoning_stream: bool = True
    show_tool_calls: bool = True


@dataclass(frozen=True)
class AgentCacheKey:
    """Identify a graph by its model, conversation, and transport scope."""

    reasoning_level: ReasoningLevel
    reasoning_level_is_explicit: bool
    model_name: str
    thread_id: str | None
    async_subagent_url_override: str | None
    mcp_scope: str | None
