from __future__ import annotations

import asyncio
import math
import os
import tomllib
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore


ReasoningLevel = Literal["low", "medium", "high"]
PersistenceMode = Literal["memory", "postgres"]
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_REASONING_LEVEL: ReasoningLevel = "medium"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_EXTENSIONS_CONFIG = "deepagent.toml"
PROJECT_ROOT = Path(__file__).resolve().parent

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


def compose_base_url(endpoint: str, port: int) -> str:
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
class ExtensionsConfig:
    config_path: Path | None
    mcp_tool_name_prefix: bool = True
    mcp_servers: dict[str, dict[str, Any]] | None = None
    skills: tuple[str, ...] = ()
    agent_mcp_servers: tuple[str, ...] = ()
    subagents: tuple[SubagentConfig, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.skills or self.agent_mcp_servers or self.subagents)


@dataclass(frozen=True)
class ModelDefaults:
    endpoint: str = DEFAULT_OLLAMA_ENDPOINT
    port: int = DEFAULT_OLLAMA_PORT
    name: str = DEFAULT_MODEL
    reasoning_effort: ReasoningLevel = DEFAULT_REASONING_LEVEL
    temperature: float = DEFAULT_TEMPERATURE

    @property
    def base_url(self) -> str:
        return compose_base_url(self.endpoint, self.port)


@dataclass(frozen=True)
class FileConfig:
    model: ModelDefaults
    extensions: ExtensionsConfig


def parse_model_defaults(raw_config: dict[str, Any]) -> ModelDefaults:
    raw_model = raw_config.get("model", {})
    if raw_model and not isinstance(raw_model, dict):
        raise ValueError("The top-level 'model' config must be a table/object.")

    return ModelDefaults(
        endpoint=normalize_model_endpoint(raw_model.get("endpoint")),
        port=normalize_model_port(raw_model.get("port")),
        name=str(raw_model.get("name", DEFAULT_MODEL)).strip() or DEFAULT_MODEL,
        reasoning_effort=normalize_reasoning_level(
            raw_model.get("reasoning_effort"),
            default=DEFAULT_REASONING_LEVEL,
        ),
        temperature=normalize_model_temperature(
            raw_model.get("temperature", raw_model.get("tempreature"))
        ),
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
    if raw_subagents and not isinstance(raw_subagents, list):
        raise ValueError("The top-level 'subagents' config must be an array of tables.")
    subagents: list[SubagentConfig] = []
    for index, raw_subagent in enumerate(raw_subagents, start=1):
        if not isinstance(raw_subagent, dict):
            raise ValueError(f"Subagent entry #{index} must be a table/object.")

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
        subagents.append(
            SubagentConfig(
                name=name,
                description=description,
                system_prompt=system_prompt,
                skills=subagent_skill_paths,
                mcp_servers=raw_subagent_mcp_servers,
                model=model,
            )
        )

    return ExtensionsConfig(
        config_path=config_path,
        mcp_tool_name_prefix=bool(mcp_section.get("tool_name_prefix", True)),
        mcp_servers=mcp_servers or None,
        skills=skill_paths,
        agent_mcp_servers=raw_agent_mcp_servers,
        subagents=tuple(subagents),
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
    ollama_model: str
    ollama_base_url: str
    ollama_temperature: float
    default_reasoning: ReasoningLevel
    persistence_mode: PersistenceMode
    extensions: ExtensionsConfig

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        file_config = load_file_config()
        model_defaults = file_config.model
        database_url = os.getenv("DATABASE_URL", "").strip() or None
        ollama_model = os.getenv("OLLAMA_MODEL", "").strip() or model_defaults.name
        ollama_base_url = (
            os.getenv("OLLAMA_BASE_URL", "").strip() or model_defaults.base_url
        )
        default_reasoning = normalize_reasoning_level(
            os.getenv("OLLAMA_REASONING", "").strip(),
            default=model_defaults.reasoning_effort,
        )

        return cls(
            database_url=database_url,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
            ollama_temperature=model_defaults.temperature,
            default_reasoning=default_reasoning,
            persistence_mode="postgres" if database_url else "memory",
            extensions=file_config.extensions,
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
        self._agents: dict[ReasoningLevel, object] = {}
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

    async def get_agent(self, reasoning_level: ReasoningLevel):
        async with self._agent_lock:
            agent = self._agents.get(reasoning_level)
            if agent is None:
                model = ChatOllama(
                    model=self.config.ollama_model,
                    base_url=self.config.ollama_base_url,
                    reasoning=reasoning_level,
                    temperature=self.config.ollama_temperature,
                )
                main_tools = await self._get_mcp_tools(self.config.extensions.agent_mcp_servers)
                agent = create_deep_agent(
                    model=model,
                    tools=main_tools or None,
                    system_prompt=SYSTEM_PROMPT,
                    backend=self._build_backend,
                    store=self.store,
                    checkpointer=self.checkpointer,
                    skills=list(self.config.extensions.skills) or None,
                    subagents=[
                        subagent.to_deepagents_spec(
                            tools=await self._get_mcp_tools(subagent.mcp_servers)
                        )
                        for subagent in self.config.extensions.subagents
                    ]
                    or None,
                )
                self._agents[reasoning_level] = agent
            return agent

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
        return CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/workspace/": FilesystemBackend(
                    root_dir=str(PROJECT_ROOT),
                    virtual_mode=True,
                ),
                "/memories/": StoreBackend(runtime),
            },
        )
