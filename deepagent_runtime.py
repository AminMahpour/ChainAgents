from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import tomllib
from collections.abc import Awaitable, Callable
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
from deepagents.middleware.skills import _list_skills
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command
from rag_runtime import (
    RagConfig,
    RagStatus,
    RagUploadResult,
    ResolvedRagConfig,
    UploadedRagFile,
    WorkspaceDocsRAG,
    compose_rag_system_prompt,
    create_search_workspace_knowledge_tool,
    parse_rag_config,
    resolve_rag_config,
)


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
DEEPAGENT_ARTIFACTS_DIRECTORY = Path(".files/deepagent")
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


def deepagent_artifacts_root(project_root: Path | None = None) -> Path:
    root = (project_root or PROJECT_ROOT).resolve()
    return root / DEEPAGENT_ARTIFACTS_DIRECTORY


def deepagent_artifacts_route_prefix(project_root: Path | None = None) -> str:
    return f"{deepagent_artifacts_root(project_root).as_posix().rstrip('/')}/"


def summarize_tool_exception(exc: Exception, *, limit: int = 400) -> str:
    detail = " ".join(str(exc).split()).strip()
    if not detail:
        return exc.__class__.__name__
    summary = detail
    if detail != exc.__class__.__name__:
        summary = f"{exc.__class__.__name__}: {detail}"
    if len(summary) > limit:
        return f"{summary[: limit - 3].rstrip()}..."
    return summary


class ToolExecutionResilienceMiddleware(AgentMiddleware[Any, Any, Any]):
    def _error_tool_message(
        self,
        request: ToolCallRequest,
        exc: Exception,
    ) -> ToolMessage:
        tool_name = (
            str(request.tool_call.get("name") or getattr(request.tool, "name", "tool")).strip()
            or "tool"
        )
        tool_call_id = str(request.tool_call.get("id") or tool_name)
        summary = summarize_tool_exception(exc)
        logger.exception(
            "Tool call failed without aborting the run: %s (%s)",
            tool_name,
            tool_call_id,
            exc_info=exc,
        )
        return ToolMessage(
            content=(
                f"Tool execution failed for `{tool_name}`: {summary}\n\n"
                "The tool error was returned without aborting the run. "
                "Adjust the tool inputs or continue with another approach."
            ),
            name=tool_name,
            tool_call_id=tool_call_id,
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        try:
            return handler(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._error_tool_message(request, exc)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        try:
            return await handler(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._error_tool_message(request, exc)


def build_agent_middleware() -> list[AgentMiddleware[Any, Any, Any]]:
    return [ToolExecutionResilienceMiddleware()]


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

    def to_deepagents_spec(
        self,
        *,
        tools: list[Any] | None = None,
        middleware: list[AgentMiddleware[Any, Any, Any]] | None = None,
    ) -> dict[str, Any]:
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
class ChainlitCommandConfig:
    name: str
    description: str
    target: Literal["prompt", "subagent", "mcp_tool", "skill"]
    value: str
    template: str | None = None
    mcp_server: str | None = None
    source: Literal["config", "agent_skill", "subagent_skill"] = "config"


@dataclass(frozen=True)
class SkillCommandMetadata:
    name: str
    description: str
    path: str
    source: Literal["agent_skill", "subagent_skill"]
    owner: str | None = None

    @property
    def label(self) -> str:
        if self.source == "agent_skill":
            return f"main agent skill `{self.path}`"
        if self.owner:
            return f"subagent `{self.owner}` skill `{self.path}`"
        return f"subagent skill `{self.path}`"

    def to_chainlit_command(self) -> ChainlitCommandConfig:
        return ChainlitCommandConfig(
            name=self.name,
            description=self.description,
            target="skill",
            value=self.path,
            source=self.source,
        )


def virtual_workspace_path_to_local(path_value: str, project_root: Path | None = None) -> str:
    normalized = path_value.strip().replace("\\", "/")
    workspace_prefix = "/workspace"
    if normalized != workspace_prefix and not normalized.startswith(f"{workspace_prefix}/"):
        return path_value

    root = (project_root or PROJECT_ROOT).resolve()
    relative = PurePosixPath(normalized.removeprefix(workspace_prefix).lstrip("/"))
    local_path = (root / Path(*relative.parts)).resolve()
    try:
        local_path.relative_to(root)
    except ValueError:
        return path_value
    return str(local_path)


@dataclass(frozen=True)
class ExtensionsConfig:
    config_path: Path | None
    mcp_tool_name_prefix: bool = True
    mcp_stateful: bool = False
    mcp_servers: dict[str, dict[str, Any]] | None = None
    skills: tuple[str, ...] = ()
    agent_mcp_servers: tuple[str, ...] = ()
    subagents: tuple[SubagentConfig, ...] = ()
    async_subagents: tuple[AsyncSubagentConfig, ...] = ()
    chainlit_commands: tuple[ChainlitCommandConfig, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(
            self.skills
            or self.agent_mcp_servers
            or self.subagents
            or self.async_subagents
            or self.chainlit_commands
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
    rag: RagConfig = RagConfig()


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

    chainlit_section = raw_config.get("chainlit", {})
    if chainlit_section and not isinstance(chainlit_section, dict):
        raise ValueError("The top-level 'chainlit' config must be a table/object.")

    raw_chainlit_commands = chainlit_section.get("commands", [])
    if not isinstance(raw_chainlit_commands, list):
        raise ValueError("The top-level 'chainlit.commands' config must be an array of tables.")

    chainlit_commands: list[ChainlitCommandConfig] = []
    seen_commands: set[str] = set()
    for index, raw_chainlit_command in enumerate(raw_chainlit_commands, start=1):
        if not isinstance(raw_chainlit_command, dict):
            raise ValueError(
                f"Chainlit command entry #{index} must be a table/object."
            )
        name = str(raw_chainlit_command.get("name", "")).strip().lstrip("/").lower()
        description = str(raw_chainlit_command.get("description", "")).strip()
        target = str(raw_chainlit_command.get("target", "")).strip().lower()
        value = str(raw_chainlit_command.get("value", "")).strip()
        template = normalize_optional_string(raw_chainlit_command.get("template"))
        mcp_server = normalize_optional_string(raw_chainlit_command.get("mcp_server"))
        if not name or " " in name:
            raise ValueError(
                f"Chainlit command entry #{index} must define a slash-compatible 'name' with no spaces."
            )
        if name in seen_commands:
            raise ValueError(f"Chainlit command '/{name}' is defined more than once.")
        if not description:
            raise ValueError(f"Chainlit command '/{name}' must include a non-empty 'description'.")
        if target not in {"prompt", "subagent", "mcp_tool"}:
            raise ValueError(
                f"Chainlit command '/{name}' target must be one of: prompt, subagent, mcp_tool."
            )
        if not value:
            raise ValueError(f"Chainlit command '/{name}' must include a non-empty 'value'.")
        if target == "subagent":
            valid_subagent_names = {subagent.name for subagent in subagents}
            if value not in valid_subagent_names:
                raise ValueError(
                    f"Chainlit command '/{name}' references unknown subagent '{value}'. "
                    f"Defined subagents: {sorted(valid_subagent_names)}"
                )
        if target == "mcp_tool" and mcp_server and mcp_server not in mcp_servers:
            raise ValueError(
                f"Chainlit command '/{name}' references unknown MCP server '{mcp_server}'. "
                f"Defined servers: {sorted(mcp_servers)}"
            )
        chainlit_commands.append(
            ChainlitCommandConfig(
                name=name,
                description=description,
                target=target,  # type: ignore[arg-type]
                value=value,
                template=template,
                mcp_server=mcp_server,
            )
        )
        seen_commands.add(name)

    return ExtensionsConfig(
        config_path=config_path,
        mcp_tool_name_prefix=bool(mcp_section.get("tool_name_prefix", True)),
        mcp_stateful=bool(mcp_section.get("stateful", False)),
        mcp_servers=mcp_servers or None,
        skills=skill_paths,
        agent_mcp_servers=raw_agent_mcp_servers,
        subagents=tuple(subagents),
        async_subagents=tuple(async_subagents),
        chainlit_commands=tuple(chainlit_commands),
    )


def load_file_config() -> FileConfig:
    config_name = os.getenv("DEEPAGENT_CONFIG", DEFAULT_EXTENSIONS_CONFIG).strip()
    config_path = resolve_local_path(config_name or DEFAULT_EXTENSIONS_CONFIG, PROJECT_ROOT)
    if not config_path.exists():
        return FileConfig(
            model=ModelDefaults(),
            extensions=ExtensionsConfig(config_path=None),
            rag=RagConfig(),
        )

    with config_path.open("rb") as fh:
        raw_config = tomllib.load(fh)

    return FileConfig(
        model=parse_model_defaults(raw_config),
        extensions=parse_extensions_config(raw_config, config_path),
        rag=parse_rag_config(raw_config, config_path),
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
    rag_requested: bool = False
    rag: ResolvedRagConfig | None = None
    rag_error: str | None = None

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
        rag_requested = file_config.rag.enabled
        rag = None
        rag_error = None
        if rag_requested:
            try:
                rag = resolve_rag_config(
                    file_config.rag,
                    model_provider=model_provider,
                    model_base_url=model_base_url,
                )
            except ValueError as exc:
                rag_error = str(exc)

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
            rag_requested=rag_requested,
            rag=rag,
            rag_error=rag_error,
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


def build_deepagent_backend(*, project_root: Path | None = None) -> CompositeBackend:
    resolved_project_root = project_root or PROJECT_ROOT
    artifacts_root = deepagent_artifacts_root(resolved_project_root)
    return CompositeBackend(
        default=StateBackend(),
        routes={
            deepagent_artifacts_route_prefix(resolved_project_root): FilesystemBackend(
                root_dir=str(artifacts_root),
                virtual_mode=True,
            ),
            "/workspace/": FilesystemBackend(
                root_dir=str(resolved_project_root),
                virtual_mode=True,
            ),
            "/memories/": StoreBackend(),
        },
        artifacts_root=str(artifacts_root),
    )


def normalize_chainlit_command_name(value: str) -> str:
    return value.strip().lstrip("/").lower()


def _load_skill_command_bucket(
    *,
    backend: CompositeBackend,
    source_paths: tuple[str, ...],
    source: Literal["agent_skill", "subagent_skill"],
    project_root: Path | None = None,
    owner: str | None = None,
) -> tuple[SkillCommandMetadata, ...]:
    commands_by_name: dict[str, SkillCommandMetadata] = {}
    for source_path in source_paths:
        try:
            source_skills = _list_skills(backend, source_path)
        except Exception as exc:
            logger.warning(
                "Failed to load skills from '%s' for Chainlit command generation: %s",
                source_path,
                exc,
            )
            continue

        for skill in source_skills:
            command_name = normalize_chainlit_command_name(str(skill["name"]))
            if not command_name or " " in command_name:
                logger.warning(
                    "Skipping skill '%s' from %s because it is not slash-command compatible.",
                    skill["name"],
                    skill["path"],
                )
                continue

            metadata = SkillCommandMetadata(
                name=command_name,
                description=str(skill["description"]).strip(),
                path=virtual_workspace_path_to_local(str(skill["path"]), project_root),
                source=source,
                owner=owner,
            )
            previous = commands_by_name.pop(command_name, None)
            if previous is not None:
                logger.warning(
                    "Auto skill command '/%s' from %s overrides %s.",
                    command_name,
                    metadata.label,
                    previous.label,
                )
            commands_by_name[command_name] = metadata
    return tuple(commands_by_name.values())


def _resolve_chainlit_project_root(
    *,
    backend: CompositeBackend | None,
    project_root: Path | None,
) -> Path:
    if project_root is not None:
        return project_root

    if backend is not None:
        workspace_backend = backend.routes.get("/workspace/")
        if isinstance(workspace_backend, FilesystemBackend):
            return workspace_backend.cwd

    return PROJECT_ROOT


def build_chainlit_command_catalog(
    extensions: ExtensionsConfig,
    *,
    backend: CompositeBackend | None = None,
    project_root: Path | None = None,
) -> tuple[tuple[ChainlitCommandConfig, ...], tuple[str, ...]]:
    resolved_project_root = _resolve_chainlit_project_root(
        backend=backend,
        project_root=project_root,
    )
    backend = backend or build_deepagent_backend(project_root=resolved_project_root)
    notes: list[str] = []
    merged_commands = list(extensions.chainlit_commands)
    explicit_names = {command.name: command for command in extensions.chainlit_commands}

    main_skill_commands = _load_skill_command_bucket(
        backend=backend,
        source_paths=extensions.skills,
        source="agent_skill",
        project_root=resolved_project_root,
    )
    subagent_commands_by_name: dict[str, SkillCommandMetadata] = {}
    for subagent in extensions.subagents:
        for metadata in _load_skill_command_bucket(
            backend=backend,
            source_paths=subagent.skills,
            source="subagent_skill",
            project_root=resolved_project_root,
            owner=subagent.name,
        ):
            previous = subagent_commands_by_name.pop(metadata.name, None)
            if previous is not None:
                logger.warning(
                    "Auto skill command '/%s' from %s overrides %s.",
                    metadata.name,
                    metadata.label,
                    previous.label,
                )
            subagent_commands_by_name[metadata.name] = metadata
    subagent_skill_commands = tuple(subagent_commands_by_name.values())

    winner_by_name: dict[str, ChainlitCommandConfig | SkillCommandMetadata] = {
        command.name: command for command in merged_commands
    }

    for metadata in main_skill_commands:
        explicit = explicit_names.get(metadata.name)
        if explicit is not None:
            note = (
                f"`/{metadata.name}` from {metadata.label} is hidden by explicit "
                f"Chainlit command `/{explicit.name}`."
            )
            notes.append(note)
            logger.warning(note)
            continue
        merged_commands.append(metadata.to_chainlit_command())
        winner_by_name[metadata.name] = metadata

    for metadata in subagent_skill_commands:
        winner = winner_by_name.get(metadata.name)
        if winner is None:
            merged_commands.append(metadata.to_chainlit_command())
            winner_by_name[metadata.name] = metadata
            continue

        if isinstance(winner, ChainlitCommandConfig):
            note = (
                f"`/{metadata.name}` from {metadata.label} is hidden by explicit "
                f"Chainlit command `/{winner.name}`."
            )
        else:
            note = f"`/{metadata.name}` from {metadata.label} is hidden by {winner.label}."
        notes.append(note)
        logger.warning(note)

    return tuple(merged_commands), tuple(notes)


def sanitize_tools_for_model(
    model_provider: ModelProvider,
    tools: list[Any],
) -> list[Any]:
    if model_provider != "openai_compatible":
        return list(tools)

    compatible_tools: list[Any] = []
    skipped_tool_names: list[str] = []
    for tool in tools:
        if tool_supports_openai_compatible_schema(tool):
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


def tool_supports_openai_compatible_schema(tool: Any) -> bool:
    try:
        schema = convert_to_openai_tool(tool)
    except Exception:
        return False

    parameters = schema.get("function", {}).get("parameters")
    return isinstance(parameters, dict) and parameters.get("type") == "object"


def build_graph_subagent_specs(
    config: RuntimeConfig,
    *,
    include_async_subagents: bool,
) -> list[Any]:
    middleware = build_agent_middleware()
    subagent_specs: list[Any] = [
        subagent.to_deepagents_spec(middleware=middleware)
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
    tools: list[Any] = []
    if config.rag is not None:
        tools.append(
            create_search_workspace_knowledge_tool(
                WorkspaceDocsRAG(config.rag, project_root=PROJECT_ROOT)
            )
        )
    else:
        if config.rag_requested and config.rag_error:
            logger.warning("RAG is configured but unavailable: %s", config.rag_error)
    return create_deep_agent(
        model=build_model(config, config.default_reasoning),
        tools=sanitize_tools_for_model(config.model_provider, tools) or None,
        system_prompt=compose_rag_system_prompt(
            system_prompt,
            rag_enabled=config.rag is not None,
        ),
        middleware=build_agent_middleware(),
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

    def __init__(self, config: RuntimeConfig, *, project_root: Path | None = None) -> None:
        self.config = config
        self.project_root = project_root or PROJECT_ROOT
        self._exit_stack = AsyncExitStack()
        self._agent_lock = asyncio.Lock()
        self._mcp_lock = asyncio.Lock()
        self._agents: dict[
            tuple[ReasoningLevel, str | None, str | None, str | None],
            object,
        ] = {}
        self._mcp_client: MultiServerMCPClient | None = None
        self._mcp_tools_cache: dict[tuple[str | None, tuple[str, ...]], list[Any]] = {}
        self._mcp_sessions: dict[tuple[str | None, str], Any] = {}
        self._mcp_session_stacks: dict[str | None, AsyncExitStack] = {}
        self._checkpointer: AsyncPostgresSaver | MemorySaver | None = None
        self._store: AsyncPostgresStore | InMemoryStore | None = None
        self._rag_service: WorkspaceDocsRAG | None = None
        self._exit_stack.push_async_callback(self.close_all_mcp_sessions)
        self._chainlit_commands, self._chainlit_command_notes = build_chainlit_command_catalog(
            config.extensions,
            project_root=self.project_root,
        )

    @classmethod
    async def get(cls) -> "AgentRuntime":
        async with cls._instance_lock:
            if cls._instance is None:
                instance = cls(RuntimeConfig.from_env())
                await instance._initialize()
                cls._instance = instance
            return cls._instance

    @classmethod
    def current(cls) -> "AgentRuntime | None":
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

    @property
    def rag_enabled(self) -> bool:
        return self.config.rag_requested

    @property
    def chainlit_commands(self) -> tuple[ChainlitCommandConfig, ...]:
        return self._chainlit_commands

    @property
    def chainlit_command_notes(self) -> tuple[str, ...]:
        return self._chainlit_command_notes

    @property
    def rag_status(self) -> RagStatus:
        if self._rag_service is not None:
            return self._rag_service.snapshot()
        if self.config.rag_requested:
            return RagStatus.unavailable(
                reason=self.config.rag_error or "Knowledge index is unavailable.",
                persist_directory=(
                    self.config.rag.persist_directory if self.config.rag is not None else None
                ),
            )
        return RagStatus.disabled()

    async def _initialize(self) -> None:
        if self.config.extensions.mcp_servers:
            self._mcp_client = MultiServerMCPClient(
                self.config.extensions.mcp_servers,
                tool_name_prefix=self.config.extensions.mcp_tool_name_prefix,
            )

        if not self.config.database_url:
            self._store = InMemoryStore()
            self._checkpointer = MemorySaver()
        else:
            self._store = await self._exit_stack.enter_async_context(
                AsyncPostgresStore.from_conn_string(self.config.database_url)
            )
            await self.store.setup()

            self._checkpointer = await self._exit_stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(self.config.database_url)
            )
            await self.checkpointer.setup()

        if self.config.rag is not None:
            self._rag_service = WorkspaceDocsRAG(
                self.config.rag,
                project_root=self.project_root,
            )
            rag_status = await asyncio.to_thread(self._rag_service.ensure_ready)
            if not rag_status.ready and rag_status.reason:
                logger.warning("RAG initialization failed: %s", rag_status.reason)
        elif self.config.rag_requested and self.config.rag_error:
            logger.warning("RAG is configured but unavailable: %s", self.config.rag_error)

    async def get_agent(
        self,
        reasoning_level: ReasoningLevel,
        *,
        thread_id: str | None = None,
        async_subagent_url_override: str | None = None,
        mcp_session_id: str | None = None,
    ):
        mcp_scope = self._mcp_scope(
            mcp_session_id=mcp_session_id,
            thread_id=thread_id,
        )
        cache_key = (
            reasoning_level,
            thread_id,
            async_subagent_url_override,
            mcp_scope,
        )
        async with self._agent_lock:
            agent = self._agents.get(cache_key)
            if agent is None:
                model = self._build_model(reasoning_level)
                rag_tool_enabled = self._rag_service is not None
                main_tools = sanitize_tools_for_model(
                    self.config.model_provider,
                    await self._build_main_tools(
                        thread_id=thread_id,
                        mcp_session_id=mcp_session_id,
                    ),
                )
                middleware = build_agent_middleware()
                subagent_specs: list[Any] = [
                    subagent.to_deepagents_spec(
                        tools=sanitize_tools_for_model(
                            self.config.model_provider,
                            await self._get_mcp_tools(
                                subagent.mcp_servers,
                                thread_id=thread_id,
                                mcp_session_id=mcp_session_id,
                            ),
                        ),
                        middleware=middleware,
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
                    system_prompt=compose_rag_system_prompt(
                        SYSTEM_PROMPT,
                        rag_enabled=rag_tool_enabled,
                    ),
                    middleware=middleware,
                    backend=self._build_backend,
                    store=self.store,
                    checkpointer=self.checkpointer,
                    skills=list(self.config.extensions.skills) or None,
                    subagents=subagent_specs or None,
                )
                self._agents[cache_key] = agent
            return agent

    async def rebuild_rag_index(self) -> RagStatus:
        if self._rag_service is None:
            if self.config.rag_requested:
                return RagStatus.unavailable(
                    reason=self.config.rag_error or "Knowledge index is unavailable.",
                    persist_directory=(
                        self.config.rag.persist_directory
                        if self.config.rag is not None
                        else None
                    ),
                )
            return RagStatus.disabled()

        status = await asyncio.to_thread(self._rag_service.rebuild)
        await self._clear_agent_cache()
        return status

    async def ingest_rag_uploads(
        self,
        *,
        thread_id: str,
        uploads: list[UploadedRagFile],
    ) -> RagUploadResult:
        if self._rag_service is None:
            return RagUploadResult(
                thread_id=thread_id,
                reason=self.config.rag_error or "Knowledge index is unavailable.",
            )

        return await asyncio.to_thread(
            self._rag_service.ingest_uploaded_files,
            thread_id=thread_id,
            uploads=uploads,
        )

    def resolve_chainlit_command(self, name: str) -> ChainlitCommandConfig | None:
        normalized = normalize_chainlit_command_name(name)
        if not normalized:
            return None
        for command in self.chainlit_commands:
            if command.name == normalized:
                return command
        return None

    async def invoke_mcp_tool_command(
        self,
        *,
        tool_name: str,
        raw_args: str,
        thread_id: str | None = None,
        mcp_session_id: str | None = None,
        server_name: str | None = None,
    ) -> Any:
        candidate_servers: tuple[str, ...]
        if server_name:
            candidate_servers = (server_name,)
        else:
            available_servers = self.config.extensions.mcp_servers or {}
            candidate_servers = tuple(available_servers.keys())

        tools = await self._get_mcp_tools(
            candidate_servers,
            thread_id=thread_id,
            mcp_session_id=mcp_session_id,
        )
        selected_tool = next(
            (
                tool
                for tool in tools
                if str(getattr(tool, "name", "")).strip() == tool_name
            ),
            None,
        )
        if selected_tool is None:
            available = sorted(
                {
                    str(getattr(tool, "name", "")).strip()
                    for tool in tools
                    if str(getattr(tool, "name", "")).strip()
                }
            )
            raise ValueError(
                f"MCP tool '{tool_name}' is unavailable."
                + (f" Available tools: {available}" if available else "")
            )

        parsed_args: Any = {}
        raw_text = raw_args.strip()
        if raw_text:
            try:
                parsed_args = json.loads(raw_text)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Command arguments for MCP tool '{tool_name}' must be valid JSON."
                ) from None
        return await selected_tool.ainvoke(parsed_args)

    def _sanitize_tools_for_model(self, tools: list[Any]) -> list[Any]:
        return sanitize_tools_for_model(self.config.model_provider, tools)

    @staticmethod
    def _tool_supports_openai_compatible_schema(tool: Any) -> bool:
        return tool_supports_openai_compatible_schema(tool)

    def _build_model(self, reasoning_level: ReasoningLevel) -> Any:
        return build_model(self.config, reasoning_level)

    def _mcp_scope(
        self,
        *,
        mcp_session_id: str | None,
        thread_id: str | None = None,
    ) -> str | None:
        if not self.config.extensions.mcp_stateful:
            return None

        candidate = str(mcp_session_id or "").strip()
        if candidate:
            return candidate

        fallback = str(thread_id or "").strip()
        return fallback or None

    async def _get_stateful_mcp_session(
        self,
        *,
        server_name: str,
        thread_id: str | None,
        mcp_session_id: str | None,
    ) -> Any:
        scope = self._mcp_scope(
            mcp_session_id=mcp_session_id,
            thread_id=thread_id,
        )
        cache_key = (scope, server_name)
        session = self._mcp_sessions.get(cache_key)
        if session is not None:
            return session

        if self._mcp_client is None:
            raise RuntimeError("MCP client is not initialized.")

        stack = self._mcp_session_stacks.get(scope)
        if stack is None:
            stack = AsyncExitStack()
            self._mcp_session_stacks[scope] = stack

        session = await stack.enter_async_context(self._mcp_client.session(server_name))
        self._mcp_sessions[cache_key] = session
        return session

    async def _get_mcp_tools(
        self,
        server_names: tuple[str, ...],
        *,
        thread_id: str | None = None,
        mcp_session_id: str | None = None,
    ) -> list[Any]:
        if not server_names or self._mcp_client is None:
            return []

        tool_scope = self._mcp_scope(
            mcp_session_id=mcp_session_id,
            thread_id=thread_id,
        )
        cache_key = (tool_scope, tuple(server_names))

        async with self._mcp_lock:
            cached = self._mcp_tools_cache.get(cache_key)
            if cached is not None:
                return list(cached)

            tools: list[Any] = []
            for server_name in cache_key[1]:
                if self.config.extensions.mcp_stateful:
                    session = await self._get_stateful_mcp_session(
                        server_name=server_name,
                        thread_id=thread_id,
                        mcp_session_id=mcp_session_id,
                    )
                    tools.extend(
                        await load_mcp_tools(
                            session,
                            callbacks=self._mcp_client.callbacks,
                            tool_interceptors=self._mcp_client.tool_interceptors,
                            server_name=server_name,
                            tool_name_prefix=self.config.extensions.mcp_tool_name_prefix,
                        )
                    )
                    continue

                tools.extend(await self._mcp_client.get_tools(server_name=server_name))

            self._mcp_tools_cache[cache_key] = tools
            return list(tools)

    async def _build_main_tools(
        self,
        *,
        thread_id: str | None,
        mcp_session_id: str | None,
    ) -> list[Any]:
        tools = await self._get_mcp_tools(
            self.config.extensions.agent_mcp_servers,
            thread_id=thread_id,
            mcp_session_id=mcp_session_id,
        )
        if self._rag_service is not None:
            tools = list(tools)
            tools.append(
                create_search_workspace_knowledge_tool(
                    self._rag_service,
                    thread_id=thread_id,
                )
            )
        return tools

    async def _clear_agent_cache(self) -> None:
        async with self._agent_lock:
            self._agents.clear()

    async def close_mcp_session(self, mcp_session_id: str | None) -> None:
        mcp_scope = self._mcp_scope(mcp_session_id=mcp_session_id)
        if mcp_scope is None:
            return

        async with self._agent_lock:
            self._agents = {
                key: agent
                for key, agent in self._agents.items()
                if len(key) < 4 or key[3] != mcp_scope
            }
            async with self._mcp_lock:
                stack = self._mcp_session_stacks.pop(mcp_scope, None)
                self._mcp_sessions = {
                    key: session
                    for key, session in self._mcp_sessions.items()
                    if key[0] != mcp_scope
                }
                self._mcp_tools_cache = {
                    key: tools
                    for key, tools in self._mcp_tools_cache.items()
                    if key[0] != mcp_scope
                }

        if stack is not None:
            await stack.aclose()

    async def close_all_mcp_sessions(self) -> None:
        async with self._agent_lock:
            async with self._mcp_lock:
                stacks = list(self._mcp_session_stacks.values())
                self._mcp_session_stacks.clear()
                self._mcp_sessions.clear()
                self._mcp_tools_cache.clear()
                self._agents.clear()

        for stack in stacks:
            await stack.aclose()

    def _build_backend(self, runtime):
        return build_deepagent_backend(project_root=self.project_root)
