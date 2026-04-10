from __future__ import annotations

import asyncio
import logging
import math
import os
import tomllib
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from deepagents import AsyncSubAgent, create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore


ModelProvider = Literal["ollama", "openai_compatible"]
ReasoningLevel = Literal["low", "medium", "high"]
PersistenceMode = Literal["memory", "postgres"]
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_MODEL_PROVIDER: ModelProvider = "ollama"
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_REASONING_LEVEL: ReasoningLevel = "medium"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_EXTENSIONS_CONFIG = "deepagent.toml"
PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""
You are a local workspace deep agent running inside a Chainlit UI.

Workspace contract:
- Use `/workspace/` for real project files. This route maps to `{PROJECT_ROOT}`.
- Use `/memories/` for agent memory. Persistence depends on runtime configuration.
- Use any other absolute path only for ephemeral scratch work.

Operating constraints:
- You do not have host shell execution.
- Read existing files before editing them.
- Keep edits scoped to the user request.
- When you finish, explain the result clearly and concisely.
- For non-trivial, multi-step work, call `write_todos` early and keep it updated as you progress so the UI can reflect your current plan and progress.
- If you expect to use multiple tools or perform more than two distinct steps, create a todo list before proceeding with the main work.

Availability questions:
- When asked what skills are available, answer from the actually loaded Skills section in your system prompt, not from generic world knowledge or broad capabilities.
- If no skills are listed there, say that no explicit Deep Agent skills are currently configured.
- Do not invent skills, MCP servers, or subagents that are not currently configured for this runtime.
""".strip()


def normalize_reasoning_level(
    value: str | None,
    *,
    default: ReasoningLevel = DEFAULT_REASONING_LEVEL,
) -> ReasoningLevel:
    candidate = (value or default).strip().lower()
    if candidate not in {"low", "medium", "high"}:
        return default
    return candidate  # type: ignore[return-value]


def normalize_model_provider(
    value: Any | None,
    *,
    default: ModelProvider = DEFAULT_MODEL_PROVIDER,
) -> ModelProvider:
    candidate = str(value or default).strip().lower().replace("-", "_")
    if not candidate:
        return default
    if candidate not in {"ollama", "openai_compatible"}:
        raise ValueError(
            "The model provider must be 'ollama' or 'openai_compatible'."
        )
    return candidate  # type: ignore[return-value]


def format_model_provider(provider: ModelProvider) -> str:
    if provider == "openai_compatible":
        return "OpenAI-compatible"
    return "Ollama"


def normalize_model_endpoint(value: str | None) -> str:
    candidate = (value or DEFAULT_OLLAMA_ENDPOINT).strip()
    if not candidate:
        candidate = DEFAULT_OLLAMA_ENDPOINT
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def normalize_model_port(value: Any | None) -> int:
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
    if value is None:
        return DEFAULT_TEMPERATURE

    try:
        temperature = float(str(value).strip())
    except (TypeError, ValueError):
        return DEFAULT_TEMPERATURE

    if not math.isfinite(temperature):
        return DEFAULT_TEMPERATURE
    return temperature


def normalize_model_base_url(
    value: Any | None,
    *,
    default: str | None = None,
    required_message: str | None = None,
) -> str:
    candidate = str(value or default or "").strip()
    if not candidate:
        if required_message:
            raise ValueError(required_message)
        return ""
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def normalize_optional_string(value: Any | None) -> str | None:
    candidate = str(value or "").strip()
    return candidate or None


def compose_base_url(endpoint: str | None, port: int) -> str:
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


def resolve_local_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def normalize_mcp_transport(value: str) -> str:
    transport = value.strip().lower()
    if transport == "streamable-http":
        return "streamable_http"
    return transport


def normalize_skill_source_path(path_value: str, base_dir: Path) -> str:
    raw = path_value.strip()
    if not raw:
        raise ValueError("Skill source paths cannot be empty.")

    normalized = raw.replace("\\", "/")
    if normalized.startswith("/"):
        path = PurePosixPath(normalized).as_posix()
        return path if path.endswith("/") else f"{path}/"

    resolved = resolve_local_path(raw, base_dir)
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Skill source path '{path_value}' must stay inside the project root "
            f"({PROJECT_ROOT}) or be given as an explicit virtual path like /workspace/skills/."
        ) from exc

    virtual_path = (PurePosixPath("/workspace") / PurePosixPath(relative.as_posix())).as_posix()
    return virtual_path if virtual_path.endswith("/") else f"{virtual_path}/"


def normalize_mcp_server_config(raw_server: dict[str, Any], base_dir: Path) -> dict[str, Any]:
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
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be a table/object.")
    return {str(key): str(raw_value) for key, raw_value in value.items()}


@dataclass(frozen=True)
class SubagentConfig:
    name: str
    description: str
    system_prompt: str
    skills: tuple[str, ...] = ()
    mcp_servers: tuple[str, ...] = ()
    model: str | None = None

    def to_deepagents_spec(self, *, tools: list[Any] | None = None) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
        }
        if self.skills:
            spec["skills"] = list(self.skills)
        if tools:
            spec["tools"] = tools
        if self.model:
            spec["model"] = self.model
        return spec


@dataclass(frozen=True)
class AsyncSubagentConfig:
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
class ExtensionsConfig:
    config_path: Path | None
    mcp_tool_name_prefix: bool = True
    mcp_servers: dict[str, dict[str, Any]] | None = None
    skills: tuple[str, ...] = ()
    agent_mcp_servers: tuple[str, ...] = ()
    subagents: tuple[SubagentConfig, ...] = ()
    async_subagents: tuple[AsyncSubagentConfig, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(
            self.skills
            or self.agent_mcp_servers
            or self.subagents
            or self.async_subagents
        )


@dataclass(frozen=True)
class ModelDefaults:
    provider: ModelProvider = DEFAULT_MODEL_PROVIDER
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    name: str = DEFAULT_MODEL
    api_key: str | None = None
    name_is_explicit: bool = False
    reasoning_effort: ReasoningLevel = DEFAULT_REASONING_LEVEL
    temperature: float = DEFAULT_TEMPERATURE


@dataclass(frozen=True)
class FileConfig:
    model: ModelDefaults
    extensions: ExtensionsConfig


def parse_model_defaults(raw_config: dict[str, Any]) -> ModelDefaults:
    raw_model = raw_config.get("model", {})
    if raw_model and not isinstance(raw_model, dict):
        raise ValueError("The top-level 'model' config must be a table/object.")

    provider = normalize_model_provider(raw_model.get("provider"))
    raw_name = str(raw_model.get("name", "")).strip()
    if provider == "openai_compatible" and not raw_name:
        raise ValueError(
            "OpenAI-compatible model config must define a non-empty 'name'."
        )

    raw_base_url = str(raw_model.get("base_url", "")).strip()
    if provider == "ollama":
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
    else:
        base_url = normalize_model_base_url(
            raw_model.get("base_url"),
            required_message=(
                "OpenAI-compatible model config must define a non-empty 'base_url'."
            ),
        )

    return ModelDefaults(
        provider=provider,
        base_url=base_url,
        name=raw_name or DEFAULT_MODEL,
        api_key=normalize_optional_string(raw_model.get("api_key")),
        name_is_explicit=bool(raw_name),
        reasoning_effort=normalize_reasoning_level(
            raw_model.get("reasoning_effort"),
            default=DEFAULT_REASONING_LEVEL,
        ),
        temperature=normalize_model_temperature(
            raw_model.get("temperature", raw_model.get("tempreature"))
        ),
    )


def parse_async_subagent_config(
    raw_subagent: dict[str, Any],
    *,
    index: int,
    source_name: str,
) -> AsyncSubagentConfig:
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
        url=normalize_optional_string(
            normalize_model_base_url(raw_subagent.get("url"))
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
) -> SubagentConfig:
    name = str(raw_subagent.get("name", "")).strip()
    description = str(raw_subagent.get("description", "")).strip()
    if not name or not description:
        raise ValueError(
            f"Subagent entry #{index} must include non-empty 'name' and 'description'."
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

    model = str(raw_subagent.get("model", "")).strip() or None
    return SubagentConfig(
        name=name,
        description=description,
        system_prompt=system_prompt,
        skills=subagent_skill_paths,
        mcp_servers=raw_subagent_mcp_servers,
        model=model,
    )


def parse_extensions_config(raw_config: dict[str, Any], config_path: Path) -> ExtensionsConfig:
    base_dir = config_path.parent
    mcp_section = raw_config.get("mcp", {})
    if mcp_section and not isinstance(mcp_section, dict):
        raise ValueError("The top-level 'mcp' config must be a table/object.")

    agent_section = raw_config.get("agent", {})
    if agent_section and not isinstance(agent_section, dict):
        raise ValueError("The top-level 'agent' config must be a table/object.")

    raw_mcp_servers = mcp_section.get("servers", {})
    mcp_servers: dict[str, dict[str, Any]] = {}
    for name, raw_server in raw_mcp_servers.items():
        if not isinstance(raw_server, dict):
            raise ValueError(f"MCP server '{name}' must be a table/object.")
        mcp_servers[str(name)] = normalize_mcp_server_config(raw_server, base_dir)

    raw_skill_paths = agent_section.get("skills", [])
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

    return ExtensionsConfig(
        config_path=config_path,
        mcp_tool_name_prefix=bool(mcp_section.get("tool_name_prefix", True)),
        mcp_servers=mcp_servers or None,
        skills=skill_paths,
        agent_mcp_servers=raw_agent_mcp_servers,
        subagents=tuple(subagents),
        async_subagents=tuple(async_subagents),
    )


def load_file_config() -> FileConfig:
    config_name = os.getenv("DEEPAGENT_CONFIG", DEFAULT_EXTENSIONS_CONFIG).strip()
    config_path = resolve_local_path(config_name or DEFAULT_EXTENSIONS_CONFIG, PROJECT_ROOT)
    if not config_path.exists():
        return FileConfig(
            model=ModelDefaults(),
            extensions=ExtensionsConfig(config_path=None),
        )

    with config_path.open("rb") as fh:
        raw_config = tomllib.load(fh)

    return FileConfig(
        model=parse_model_defaults(raw_config),
        extensions=parse_extensions_config(raw_config, config_path),
    )


def load_extensions_config() -> ExtensionsConfig:
    # Keep the previous public helper for existing imports and tests.
    return load_file_config().extensions


@dataclass(frozen=True)
class RuntimeConfig:
    database_url: str | None
    model_provider: ModelProvider
    model_name: str
    model_base_url: str
    model_api_key: str | None
    model_temperature: float
    default_reasoning: ReasoningLevel
    persistence_mode: PersistenceMode
    extensions: ExtensionsConfig

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        file_config = load_file_config()
        model_defaults = file_config.model
        database_url = os.getenv("DATABASE_URL", "").strip() or None
        model_provider_override = normalize_optional_string(
            os.getenv("DEEPAGENT_MODEL_PROVIDER")
        )
        model_provider = normalize_model_provider(
            model_provider_override,
            default=model_defaults.provider,
        )
        generic_model_name = os.getenv("DEEPAGENT_MODEL_NAME", "").strip()
        generic_model_base_url = os.getenv("DEEPAGENT_MODEL_BASE_URL", "").strip()
        generic_model_reasoning = os.getenv("DEEPAGENT_MODEL_REASONING", "").strip()
        model_name_alias = os.getenv("OLLAMA_MODEL", "").strip() if model_provider == "ollama" else ""
        model_base_url_alias = (
            os.getenv("OLLAMA_BASE_URL", "").strip() if model_provider == "ollama" else ""
        )
        model_reasoning_alias = (
            os.getenv("OLLAMA_REASONING", "").strip() if model_provider == "ollama" else ""
        )

        if (
            model_provider_override
            and model_provider != model_defaults.provider
            and not generic_model_base_url
        ):
            raise ValueError(
                "Switching model providers via DEEPAGENT_MODEL_PROVIDER also requires "
                "DEEPAGENT_MODEL_BASE_URL so the new provider does not inherit an "
                "incompatible endpoint."
            )

        if (
            model_provider == "openai_compatible"
            and not generic_model_name
            and not model_defaults.name_is_explicit
        ):
            raise ValueError(
                "OpenAI-compatible runtime must define DEEPAGENT_MODEL_NAME "
                "or set a non-empty [model].name in deepagent.toml."
            )

        model_name = generic_model_name or model_name_alias or model_defaults.name
        model_base_url = normalize_model_base_url(
            generic_model_base_url or model_base_url_alias or model_defaults.base_url,
            required_message="The model base URL cannot be empty.",
        )
        model_api_key = (
            normalize_optional_string(os.getenv("DEEPAGENT_MODEL_API_KEY"))
            or model_defaults.api_key
        )
        default_reasoning = normalize_reasoning_level(
            generic_model_reasoning or model_reasoning_alias,
            default=model_defaults.reasoning_effort,
        )

        return cls(
            database_url=database_url,
            model_provider=model_provider,
            model_name=model_name,
            model_base_url=model_base_url,
            model_api_key=model_api_key,
            model_temperature=model_defaults.temperature,
            default_reasoning=default_reasoning,
            persistence_mode="postgres" if database_url else "memory",
            extensions=file_config.extensions,
        )


def build_model(config: RuntimeConfig, reasoning_level: ReasoningLevel) -> Any:
    if config.model_provider == "ollama":
        return ChatOllama(
            model=config.model_name,
            base_url=config.model_base_url,
            reasoning=reasoning_level,
            temperature=config.model_temperature,
        )

    kwargs: dict[str, Any] = {
        "model": config.model_name,
        "base_url": config.model_base_url,
        "api_key": config.model_api_key or "deepagent",
        "temperature": config.model_temperature,
    }
    return ChatOpenAI(**kwargs)


def build_deepagent_backend() -> CompositeBackend:
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(
                root_dir=str(PROJECT_ROOT),
                virtual_mode=True,
            ),
            "/memories/": StoreBackend(),
        },
    )


def build_graph_subagent_specs(
    config: RuntimeConfig,
    *,
    include_async_subagents: bool,
) -> list[Any]:
    subagent_specs: list[Any] = [
        subagent.to_deepagents_spec()
        for subagent in config.extensions.subagents
    ]
    if include_async_subagents:
        subagent_specs.extend(
            subagent.to_deepagents_spec()
            for subagent in config.extensions.async_subagents
        )
    return subagent_specs


def create_configured_graph(
    *,
    include_async_subagents: bool,
    system_prompt: str = SYSTEM_PROMPT,
) -> Any:
    config = RuntimeConfig.from_env()
    subagent_specs = build_graph_subagent_specs(
        config,
        include_async_subagents=include_async_subagents,
    )
    return create_deep_agent(
        model=build_model(config, config.default_reasoning),
        system_prompt=system_prompt,
        backend=build_deepagent_backend(),
        skills=list(config.extensions.skills) or None,
        subagents=subagent_specs or None,
    )


@dataclass(frozen=True)
class AppSettings:
    reasoning_level: ReasoningLevel
    thread_id: str


class AgentRuntime:
    _instance: "AgentRuntime | None" = None
    _instance_lock = asyncio.Lock()

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._exit_stack = AsyncExitStack()
        self._agent_lock = asyncio.Lock()
        self._agents: dict[tuple[ReasoningLevel, str | None], object] = {}
        self._mcp_client: MultiServerMCPClient | None = None
        self._mcp_tools_cache: dict[tuple[str, ...], list[Any]] = {}
        self._checkpointer: AsyncPostgresSaver | MemorySaver | None = None
        self._store: AsyncPostgresStore | InMemoryStore | None = None

    @classmethod
    async def get(cls) -> "AgentRuntime":
        async with cls._instance_lock:
            if cls._instance is None:
                instance = cls(RuntimeConfig.from_env())
                await instance._initialize()
                cls._instance = instance
            return cls._instance

    @property
    def checkpointer(self) -> AsyncPostgresSaver | MemorySaver:
        if self._checkpointer is None:
            raise RuntimeError("Checkpointer is not initialized.")
        return self._checkpointer

    @property
    def store(self) -> AsyncPostgresStore | InMemoryStore:
        if self._store is None:
            raise RuntimeError("Store is not initialized.")
        return self._store

    @property
    def persistence_enabled(self) -> bool:
        return self.config.persistence_mode == "postgres"

    async def _initialize(self) -> None:
        if self.config.extensions.mcp_servers:
            self._mcp_client = MultiServerMCPClient(
                self.config.extensions.mcp_servers,
                tool_name_prefix=self.config.extensions.mcp_tool_name_prefix,
            )

        if not self.config.database_url:
            self._store = InMemoryStore()
            self._checkpointer = MemorySaver()
            return

        self._store = await self._exit_stack.enter_async_context(
            AsyncPostgresStore.from_conn_string(self.config.database_url)
        )
        await self.store.setup()

        self._checkpointer = await self._exit_stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(self.config.database_url)
        )
        await self.checkpointer.setup()

    async def get_agent(
        self,
        reasoning_level: ReasoningLevel,
        *,
        async_subagent_url_override: str | None = None,
    ):
        cache_key = (reasoning_level, async_subagent_url_override)
        async with self._agent_lock:
            agent = self._agents.get(cache_key)
            if agent is None:
                model = self._build_model(reasoning_level)
                main_tools = self._sanitize_tools_for_model(
                    await self._get_mcp_tools(self.config.extensions.agent_mcp_servers)
                )
                subagent_specs: list[Any] = [
                    subagent.to_deepagents_spec(
                        tools=self._sanitize_tools_for_model(
                            await self._get_mcp_tools(subagent.mcp_servers)
                        )
                    )
                    for subagent in self.config.extensions.subagents
                ]
                subagent_specs.extend(
                    subagent.to_deepagents_spec(
                        url_override=async_subagent_url_override,
                    )
                    for subagent in self.config.extensions.async_subagents
                )
                agent = create_deep_agent(
                    model=model,
                    tools=main_tools or None,
                    system_prompt=SYSTEM_PROMPT,
                    backend=self._build_backend,
                    store=self.store,
                    checkpointer=self.checkpointer,
                    skills=list(self.config.extensions.skills) or None,
                    subagents=subagent_specs or None,
                )
                self._agents[cache_key] = agent
            return agent

    def _sanitize_tools_for_model(self, tools: list[Any]) -> list[Any]:
        if self.config.model_provider != "openai_compatible":
            return list(tools)

        compatible_tools: list[Any] = []
        skipped_tool_names: list[str] = []
        for tool in tools:
            if self._tool_supports_openai_compatible_schema(tool):
                compatible_tools.append(tool)
                continue
            skipped_tool_names.append(getattr(tool, "name", type(tool).__name__))

        if skipped_tool_names:
            logger.warning(
                "Skipping %d tool(s) with non-object JSON schemas for OpenAI-compatible "
                "tool calling: %s",
                len(skipped_tool_names),
                ", ".join(skipped_tool_names),
            )

        return compatible_tools

    @staticmethod
    def _tool_supports_openai_compatible_schema(tool: Any) -> bool:
        try:
            schema = convert_to_openai_tool(tool)
        except Exception:
            return False

        parameters = schema.get("function", {}).get("parameters")
        return isinstance(parameters, dict) and parameters.get("type") == "object"

    def _build_model(self, reasoning_level: ReasoningLevel) -> Any:
        return build_model(self.config, reasoning_level)

    async def _get_mcp_tools(self, server_names: tuple[str, ...]) -> list[Any]:
        if not server_names or self._mcp_client is None:
            return []

        cache_key = tuple(server_names)
        cached = self._mcp_tools_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        tools: list[Any] = []
        for server_name in cache_key:
            tools.extend(await self._mcp_client.get_tools(server_name=server_name))

        self._mcp_tools_cache[cache_key] = tools
        return list(tools)

    def _build_backend(self, runtime):
        return build_deepagent_backend()
