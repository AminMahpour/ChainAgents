"""Build and configure the LangChain Deep Agent runtime used by ChainAgents."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import re
import threading
import tomllib
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from chainagents.util.langchain_warnings import install_langchain_warning_filters

install_langchain_warning_filters()

from deepagents import AsyncSubAgent, create_deep_agent
from deepagents.backends import (
    BackendProtocol,
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.skills import _list_skills
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph.ui import push_ui_message
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.types import Command
from chainagents.rag.runtime import (
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
from chainagents.runtime.reflection import (
    ReflectionConfig,
    normalize_reflection_config,
)


ModelProvider = Literal["ollama", "openai_compatible", "snowflake_cortex", "anthropic"]
ReasoningLevel = Literal["low", "medium", "high"]
ModelModality = Literal["text", "image"]
DisableStreaming = bool | Literal["tool_calling"]
ModelThinking = Literal["auto", "adaptive", "disabled"]
PersistenceMode = Literal["memory", "postgres"]
AgentStateMode = Literal["stateful", "stateless"]
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_MODEL_PROVIDER: ModelProvider = "ollama"
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_REASONING_LEVEL: ReasoningLevel = "medium"
DEFAULT_MODEL_THINKING: ModelThinking = "auto"
DEFAULT_AGENT_STATE: AgentStateMode = "stateful"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_EXTENSIONS_CONFIG = "deepagent.toml"
DEFAULT_RECURSION_LIMIT = 100
DEFAULT_AGENT_MEMORY_NAMESPACE = "filesystem"
DEFAULT_AGENT_MEMORY_FILES = ("/memories/AGENTS.md",)
DEFAULT_DEEPAGENT_FILESYSTEM_TOOLS = (
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
)
AGENT_MEMORY_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9\-_.@+:~]+$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEEPAGENT_ARTIFACTS_DIRECTORY = Path(".files/deepagent")
GENERATED_OUTPUTS_DIRECTORY = Path(".files/outputs")
AGENTS_MD_FILENAME = "AGENTS.md"
logger = logging.getLogger(__name__)
_DEEPAGENTS_SUMMARIZATION_FACTORY_LOCK = threading.RLock()
OPENAI_CHAT_COMPLETIONS_PATH_SUFFIX = "/chat/completions"
OPENAI_RESPONSES_PATH_SUFFIX = "/responses"
ANTHROPIC_MESSAGES_PATH_SUFFIX = "/v1/messages"
OPENAI_COMPATIBLE_MODEL_PROVIDERS = frozenset({"openai_compatible", "snowflake_cortex"})
SNOWFLAKE_CORTEX_CANONICAL_TOOL_CALL_ID_RE = re.compile(r"^call_[0-9a-f]{24}$")
SNOWFLAKE_CORTEX_BASE_PATH = "/api/v2/cortex/v1"
SNOWFLAKE_CORTEX_CHAT_COMPLETIONS_PATH = (
    f"{SNOWFLAKE_CORTEX_BASE_PATH}{OPENAI_CHAT_COMPLETIONS_PATH_SUFFIX}"
)
SNOWFLAKE_CORTEX_HOST_SUFFIXES = (
    ".snowflakecomputing.com",
    ".snowflakecomputing.cn",
)

# Anthropic reasoning is not represented by OpenAI-style `delta` keys here.
# LangChain Anthropic maps Claude `thinking_delta` and `signature_delta` stream
# events into structured `thinking` content blocks.
OPENAI_COMPATIBLE_REASONING_DELTA_KEYS = (
    "reasoning_content",
    "reasoning",
    "reasoning_text",
    "reasoning_details",
)
SUMMARIZATION_STATUS_EVENT_KIND = "summarization_status"
GENERATIVE_UI_COMPONENT_NAME = "GeneratedPanel"
SYSTEM_PROMPT_MEMORY_LINE = (
    "- Use `/memories/` for agent memory. Persistence depends on runtime configuration."
)
STATELESS_SYSTEM_PROMPT_MEMORY_LINE = "- Agent memory is disabled for this runtime."

SYSTEM_PROMPT = f"""
You are a local workspace deep agent running inside a Chainlit UI.

Workspace contract:
- Use `/workspace/` for real project files. This route maps to `{PROJECT_ROOT}`.
- Write downloadable generated files under `/workspace/.files/outputs/`, which maps to `{PROJECT_ROOT / GENERATED_OUTPUTS_DIRECTORY}`.
{SYSTEM_PROMPT_MEMORY_LINE}
- Use any other absolute path only for ephemeral scratch work.

Operating constraints:
- You do not have host shell execution.
- Read existing files before editing them.
- Keep edits scoped to the user request.
- When you finish, explain the result clearly and concisely.
- For non-trivial, multi-step work, call `write_todos` early and keep it updated as you progress so the UI can reflect your current plan and progress.
- If you expect to use multiple tools or perform more than two distinct steps, create a todo list before proceeding with the main work.
- Actively use `render_chainlit_ui` in Chainlit for interactive or structured answers. When the tool is available, default to a compact GeneratedPanel for summaries, facts, checklists, comparisons, status updates, choices, and next-step action buttons. Still provide the normal text answer. Do not describe generated panels as above or below the answer; use non-positional wording such as "the generated panel" or "the panel actions." For simple one-sentence answers or when the tool is absent, answer in text only.

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


def normalize_agent_state(value: Any | None) -> AgentStateMode:
    """Normalize agent state mode.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized agent state mode.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None or str(value).strip() == "":
        return DEFAULT_AGENT_STATE
    candidate = str(value).strip().lower().replace("-", "_")
    if candidate == "stateful":
        return "stateful"
    if candidate == "stateless":
        return "stateless"
    raise ValueError(
        "The top-level 'agent.state' config must be 'stateful' or 'stateless'."
    )


def normalize_agent_memory_namespace(value: Any | None) -> str:
    """Normalize the shared agent memory namespace.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized agent memory namespace.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return DEFAULT_AGENT_MEMORY_NAMESPACE
    if not isinstance(value, str):
        raise ValueError(
            "The top-level 'agent.memory_namespace' config must be a non-empty string."
        )
    candidate = value.strip()
    if not candidate:
        raise ValueError(
            "The top-level 'agent.memory_namespace' config must be a non-empty string."
        )
    if AGENT_MEMORY_NAMESPACE_RE.fullmatch(candidate) is None:
        raise ValueError(
            "The top-level 'agent.memory_namespace' config may only contain "
            "alphanumeric characters, hyphens, underscores, dots, @, +, colons, "
            "and tildes."
        )
    return candidate


def normalize_agent_memory_files(value: Any | None) -> tuple[str, ...]:
    """Normalize startup memory files loaded by DeepAgents.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized memory file paths.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return DEFAULT_AGENT_MEMORY_FILES
    if not isinstance(value, list):
        raise ValueError(
            "The top-level 'agent.memory_files' config must be an array of /memories/ paths."
        )

    memory_files: list[str] = []
    for index, raw_path in enumerate(value, start=1):
        if not isinstance(raw_path, str):
            raise ValueError(
                f"The top-level 'agent.memory_files' entry #{index} must be a string."
            )
        memory_path = raw_path.strip()
        if not memory_path.startswith("/memories/") or memory_path == "/memories/":
            raise ValueError(
                "The top-level 'agent.memory_files' entries must be absolute "
                "/memories/ file paths."
            )
        memory_files.append(memory_path)
    return tuple(memory_files)


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


def _first_openai_compatible_delta(chunk: dict[str, Any]) -> dict[str, Any]:
    """Return the first delta object from an OpenAI-compatible chunk.

    Args:
        chunk: Streamed event chunk to normalize.

    Returns:
        The first delta object from an OpenAI-compatible chunk.
    """
    choices = chunk.get("choices", [])
    if not choices:
        nested_chunk = chunk.get("chunk")
        if isinstance(nested_chunk, dict):
            choices = nested_chunk.get("choices", [])

    if not isinstance(choices, list) or not choices:
        return {}

    choice = choices[0]
    if not isinstance(choice, dict):
        return {}

    delta = choice.get("delta")
    if isinstance(delta, dict):
        return delta
    return {}


def _openai_compatible_reasoning_delta(chunk: dict[str, Any]) -> Any:
    """Return reasoning content from an OpenAI-compatible delta.

    Args:
        chunk: Streamed event chunk to normalize.

    Returns:
        Reasoning content from an OpenAI-compatible delta.
    """
    delta = _first_openai_compatible_delta(chunk)
    for key in OPENAI_COMPATIBLE_REASONING_DELTA_KEYS:
        value = delta.get(key)
        if value not in (None, ""):
            return value
    return None


class OpenAICompatibleChatOpenAI(ChatOpenAI):
    """Adapt OpenAI-compatible chat chunks while preserving reasoning deltas."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict | None,
    ):
        """Convert provider chunks while preserving OpenAI-compatible reasoning.

        Args:
            chunk: Streamed event chunk to normalize.
            default_chunk_class: The default chunk class value.
            base_generation_info: The base generation info value.

        Returns:
            The convert chunk to generation chunk result.
        """
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None

        reasoning_delta = _openai_compatible_reasoning_delta(chunk)
        if reasoning_delta is None or not isinstance(
            generation_chunk.message,
            AIMessageChunk,
        ):
            return generation_chunk

        generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning_delta
        return generation_chunk


class SnowflakeCortexChatOpenAI(OpenAICompatibleChatOpenAI):
    """Adapt Snowflake Cortex Chat Completions tool-call IDs."""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Return a copied Chat Completions payload with canonical tool-call IDs."""
        payload = copy.deepcopy(
            super()._get_request_payload(input_, stop=stop, **kwargs)
        )
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        pending_ids: set[str] = set()
        pending_canonical_ids: set[str] = set()
        canonical_ids: dict[str, str] = {}
        for message in messages:
            if not isinstance(message, dict):
                if pending_ids:
                    raise ValueError("incomplete tool-call batch before a new non-tool message")
                continue
            role = message.get("role")
            tool_calls = message.get("tool_calls")
            if role == "assistant" and tool_calls:
                if pending_ids:
                    raise ValueError("incomplete tool-call batch before a new assistant batch")
                if not isinstance(tool_calls, list):
                    raise ValueError("assistant tool calls must be a list")
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        raise ValueError("assistant tool call must be an object")
                    raw_id = tool_call.get("id")
                    if not isinstance(raw_id, str) or not raw_id:
                        raise ValueError("empty tool call ID")
                    if raw_id in pending_ids:
                        raise ValueError("duplicate tool call ID within a batch")
                    canonical_id = (
                        raw_id
                        if SNOWFLAKE_CORTEX_CANONICAL_TOOL_CALL_ID_RE.fullmatch(raw_id)
                        else f"call_{hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:24]}"
                    )
                    if canonical_id in pending_canonical_ids:
                        raise ValueError("duplicate canonical tool call ID within a batch")
                    pending_ids.add(raw_id)
                    pending_canonical_ids.add(canonical_id)
                    canonical_ids[raw_id] = canonical_id
                    tool_call["id"] = canonical_id
                continue

            if role == "tool":
                raw_id = message.get("tool_call_id")
                if not isinstance(raw_id, str) or not raw_id:
                    raise ValueError("empty tool response ID")
                if raw_id not in pending_ids:
                    raise ValueError("unmatched tool response ID")
                message["tool_call_id"] = canonical_ids[raw_id]
                pending_ids.remove(raw_id)
                if not pending_ids:
                    pending_canonical_ids.clear()
                    canonical_ids.clear()
                continue

            if pending_ids:
                raise ValueError("incomplete tool-call batch before a new non-tool message")

        if pending_ids:
            raise ValueError("incomplete tool-call batch at payload end")
        return payload


class AnthropicDefaultQueryChatAnthropic(ChatAnthropic):
    """ChatAnthropic variant that forwards endpoint query params to the SDK."""

    default_query: dict[str, object] | None = None

    @cached_property
    def _client_params(self) -> dict[str, Any]:
        """Return Anthropic client params with optional default query values."""
        params = super()._client_params.copy()
        if self.default_query:
            params["default_query"] = self.default_query
        return params


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


def normalize_recursion_limit(
    value: Any | None,
    *,
    default: int = DEFAULT_RECURSION_LIMIT,
    field_name: str = "recursion_limit",
) -> int:
    """Normalize recursion limit.

    Args:
        value: Value to normalize, convert, or serialize.
        default: Fallback value used when no explicit value is available.
        field_name: The field name value.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None or str(value).strip() == "":
        return default

    try:
        recursion_limit = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc

    if recursion_limit <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return recursion_limit


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


def deepagent_artifacts_root(project_root: Path | None = None) -> Path:
    """Return the local directory used for stored tool artifacts.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The local directory used for stored tool artifacts.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    return root / DEEPAGENT_ARTIFACTS_DIRECTORY


def deepagent_artifacts_route_prefix(project_root: Path | None = None) -> str:
    """Return the URL route prefix for stored tool artifacts.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The URL route prefix for stored tool artifacts.
    """
    return f"{deepagent_artifacts_root(project_root).as_posix().rstrip('/')}/"


def generated_outputs_root(project_root: Path | None = None) -> Path:
    """Return the local directory used for downloadable generated outputs.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The local directory used for downloadable generated outputs.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    return root / GENERATED_OUTPUTS_DIRECTORY


def generated_outputs_route_prefix(project_root: Path | None = None) -> str:
    """Return the URL route prefix for downloadable generated outputs.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The URL route prefix for downloadable generated outputs.
    """
    return f"{generated_outputs_root(project_root).as_posix().rstrip('/')}/"


def summarize_tool_exception(exc: Exception, *, limit: int = 400) -> str:
    """Summarize tool exception.

    Args:
        exc: The exc value.
        limit: The limit value.

    Returns:
        The summary value.
    """
    detail = " ".join(str(exc).split()).strip()
    if not detail:
        return exc.__class__.__name__
    summary = detail
    if detail != exc.__class__.__name__:
        summary = f"{exc.__class__.__name__}: {detail}"
    if len(summary) > limit:
        return f"{summary[: limit - 3].rstrip()}..."
    return summary


WORKSPACE_PATH_TOOL_ARG_KEYS = {
    "destination",
    "dest",
    "dst",
    "path",
    "paths",
    "source",
    "src",
}


def _map_workspace_tool_path_value(value: Any, project_root: Path) -> Any:
    """Map one virtual workspace path value to a local path.

    Args:
        value: Value to normalize, convert, or serialize.
        project_root: Project root used to resolve local paths.

    Returns:
        The mapped value.
    """
    if isinstance(value, str):
        return virtual_workspace_path_to_local(value, project_root)
    if isinstance(value, list):
        return [_map_workspace_tool_path_value(item, project_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_workspace_tool_path_value(item, project_root) for item in value)
    return value


def map_workspace_paths_in_tool_args(args: Any, project_root: Path | None = None) -> Any:
    """Map workspace paths in tool args.

    Args:
        args: Parsed command-line arguments.
        project_root: Project root used to resolve local paths.

    Returns:
        The mapped value.
    """
    if not isinstance(args, dict):
        return args

    root = (project_root or PROJECT_ROOT).resolve()
    mapped = dict(args)
    for key, value in args.items():
        if str(key).lower() in WORKSPACE_PATH_TOOL_ARG_KEYS:
            mapped[key] = _map_workspace_tool_path_value(value, root)
    return mapped


class ToolExecutionResilienceMiddleware(AgentMiddleware[Any, Any, Any]):
    """Wrap tool execution with workspace path mapping and recoverable errors."""

    def __init__(self, *, project_root: Path | None = None) -> None:
        """Initialize the tool execution resilience middleware instance.

        Args:
            project_root: Project root used to resolve local paths.
        """
        self.project_root = (project_root or PROJECT_ROOT).resolve()

    def _map_workspace_path_args(self, request: ToolCallRequest) -> None:
        """Map virtual workspace paths inside tool-call arguments.

        Args:
            request: The request value.
        """
        args = request.tool_call.get("args")
        mapped_args = map_workspace_paths_in_tool_args(args, self.project_root)
        if mapped_args is not args:
            request.tool_call["args"] = mapped_args

    def _error_tool_message(
        self,
        request: ToolCallRequest,
        exc: Exception,
    ) -> ToolMessage:
        """Build a ToolMessage describing a recoverable tool failure.

        Args:
            request: The request value.
            exc: The exc value.

        Returns:
            The constructed a toolmessage describing a recoverable tool failure.
        """
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
        """Wrap synchronous tool calls with path mapping and error handling.

        Args:
            request: The request value.
            handler: The handler value.

        Returns:
            The wrap tool call result.
        """
        try:
            self._map_workspace_path_args(request)
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
        """Wrap asynchronous tool calls with path mapping and error handling.

        Args:
            request: The request value.
            handler: The handler value.

        Returns:
            The awrap tool call result.
        """
        try:
            self._map_workspace_path_args(request)
            return await handler(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._error_tool_message(request, exc)


class SummarizationStatusMiddleware(AgentMiddleware[Any, Any, Any]):
    """Emit stream events when conversation history summarization is about to run."""

    def __init__(self, inner: AgentMiddleware[Any, Any, Any], *, source: str = "main-agent") -> None:
        """Initialize the summarization status middleware instance.

        Args:
            inner: The inner value.
            source: The source value.
        """
        super().__init__()
        self.inner = inner
        self.source = source

    @property
    def name(self) -> str:
        """Return the wrapped summarization middleware name.

        Returns:
            The middleware name exposed to LangGraph.
        """
        return str(getattr(self.inner, "name", "SummarizationMiddleware"))

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Run middleware logic before a model invocation.

        Args:
            state: Runtime state to inspect or update.
            runtime: Agent runtime used by the operation.

        Returns:
            The before model result.
        """
        will_summarize = self._will_summarize(state)
        if will_summarize:
            self._emit(runtime, "started", "Conversation summarization triggered.")
        try:
            result = self.inner.before_model(state, runtime)
        except Exception:
            if will_summarize:
                self._emit(runtime, "failed", "Conversation summarization failed.")
            raise
        if result is not None:
            if not will_summarize:
                self._emit(runtime, "started", "Conversation summarization triggered.")
            self._emit(runtime, "completed", "Conversation summarization completed.")
        return result

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Run asynchronous middleware logic before a model invocation.

        Args:
            state: Runtime state to inspect or update.
            runtime: Agent runtime used by the operation.

        Returns:
            The abefore model result.
        """
        will_summarize = self._will_summarize(state)
        if will_summarize:
            self._emit(runtime, "started", "Conversation summarization triggered.")
        try:
            result = await self.inner.abefore_model(state, runtime)
        except Exception:
            if will_summarize:
                self._emit(runtime, "failed", "Conversation summarization failed.")
            raise
        if result is not None:
            if not will_summarize:
                self._emit(runtime, "started", "Conversation summarization triggered.")
            self._emit(runtime, "completed", "Conversation summarization completed.")
        return result

    def _will_summarize(self, state: Any) -> bool:
        """Return whether the next model call will trigger summarization.

        Args:
            state: Runtime state to inspect or update.

        Returns:
            Whether the next model call will trigger summarization.
        """
        try:
            messages = state["messages"]
            ensure_ids = getattr(self.inner, "_ensure_message_ids", None)
            if callable(ensure_ids):
                ensure_ids(messages)
            token_counter = getattr(self.inner, "token_counter")
            should_summarize = getattr(self.inner, "_should_summarize")
            determine_cutoff = getattr(self.inner, "_determine_cutoff_index")
            total_tokens = token_counter(messages)
            return bool(
                should_summarize(messages, total_tokens)
                and determine_cutoff(messages) > 0
            )
        except Exception:
            return False

    def _emit(self, runtime: Any, status: str, message: str) -> None:
        """Emit a custom stream event through the configured writer.

        Args:
            runtime: Agent runtime used by the operation.
            status: The status value.
            message: Chainlit message or LangChain message to process.
        """
        stream_writer = getattr(runtime, "stream_writer", None)
        if not callable(stream_writer):
            return
        try:
            stream_writer(
                {
                    "kind": SUMMARIZATION_STATUS_EVENT_KIND,
                    "status": status,
                    "source": self.source,
                    "message": message,
                }
            )
        except Exception as exc:
            logger.debug("Failed to emit summarization status event: %s", exc)


def _build_summarization_middleware(
    *,
    config: RuntimeConfig,
    reasoning_level: ReasoningLevel,
    model_name: str | None = None,
    source: str = "main-agent",
) -> AgentMiddleware[Any, Any, Any] | None:
    """Build summarization middleware when the extension is enabled.

    Args:
        config: Configuration object used by the operation.
        reasoning_level: The reasoning level value.
        model_name: The model name value.
        source: The source value.

    Returns:
        The constructed summarization middleware when the extension is enabled.
    """
    try:
        from langchain.agents.middleware import SummarizationMiddleware
    except Exception:
        logger.warning(
            "Summarization middleware is enabled but unavailable in the installed LangChain version."
        )
        return None

    kwargs: dict[str, Any] = {}
    signature = inspect.signature(SummarizationMiddleware)
    if "model" in signature.parameters:
        kwargs["model"] = build_model(
            config,
            reasoning_level,
            model_name=model_name,
        )
    if config.extensions.summarization_trigger_tokens is not None:
        if "max_tokens_before_summary" in signature.parameters:
            kwargs["max_tokens_before_summary"] = (
                config.extensions.summarization_trigger_tokens
            )
        elif "trigger" in signature.parameters:
            kwargs["trigger"] = ("tokens", config.extensions.summarization_trigger_tokens)
    if config.extensions.summarization_keep_tokens is not None and "keep" in signature.parameters:
        kwargs["keep"] = ("tokens", config.extensions.summarization_keep_tokens)

    try:
        middleware = SummarizationMiddleware(**kwargs)
    except Exception as exc:
        logger.warning(
            "Failed to initialize summarization middleware; continuing without it: %s",
            exc,
        )
        return None
    return SummarizationStatusMiddleware(middleware, source=source)


def _build_deepagents_summarization_factory(
    config: RuntimeConfig,
) -> Callable[[Any, Any], AgentMiddleware[Any, Any, Any]] | None:
    """Build a DeepAgents summarization factory honoring configured thresholds.

    Args:
        config: Configuration object used by the operation.

    Returns:
        A replacement DeepAgents summarization factory, or None when the
        built-in defaults should be used unchanged.
    """
    trigger_tokens = config.extensions.summarization_trigger_tokens
    keep_tokens = config.extensions.summarization_keep_tokens
    if trigger_tokens is None and keep_tokens is None:
        return None

    def factory(model: Any, backend: Any) -> AgentMiddleware[Any, Any, Any]:
        """Create DeepAgents summarization middleware with configured thresholds."""
        from deepagents.middleware.summarization import (
            SummarizationMiddleware,
            compute_summarization_defaults,
        )

        defaults = compute_summarization_defaults(model)
        trigger = (
            ("tokens", trigger_tokens)
            if trigger_tokens is not None
            else defaults["trigger"]
        )
        keep = ("tokens", keep_tokens) if keep_tokens is not None else defaults["keep"]
        return SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=trigger,
            keep=keep,
            trim_tokens_to_summarize=None,
            truncate_args_settings=defaults["truncate_args_settings"],
        )

    return factory


def create_deep_agent_with_configured_summarization(
    config: RuntimeConfig,
    **kwargs: Any,
) -> Any:
    """Create a DeepAgents graph while applying configured summarization thresholds.

    Args:
        config: Configuration object used by the operation.
        kwargs: Keyword arguments passed to create_deep_agent.

    Returns:
        The created DeepAgents graph.
    """
    summarization_factory = _build_deepagents_summarization_factory(config)
    if summarization_factory is None:
        return create_deep_agent(**kwargs)

    import deepagents.graph as deepagents_graph

    with _DEEPAGENTS_SUMMARIZATION_FACTORY_LOCK:
        original_factory = deepagents_graph.create_summarization_middleware
        deepagents_graph.create_summarization_middleware = summarization_factory
        try:
            return create_deep_agent(**kwargs)
        finally:
            deepagents_graph.create_summarization_middleware = original_factory


def build_agent_middleware(
    *,
    backend: BackendProtocol,
    config: RuntimeConfig | None = None,
    reasoning_level: ReasoningLevel | None = None,
    model_name: str | None = None,
    source: str = "main-agent",
    project_root: Path | None = None,
) -> list[AgentMiddleware[Any, Any, Any]]:
    """Build agent middleware.

    Args:
        backend: Concrete DeepAgents backend used by filesystem middleware.
        config: Configuration object used by the operation.
        reasoning_level: The reasoning level value.
        model_name: The model name value.
        source: The source value.
        project_root: Project root used to resolve local paths.

    Returns:
        The constructed agent middleware.
    """
    # DeepAgents owns summarization middleware in its base main-agent and
    # sync-subagent stacks. Passing another SummarizationMiddleware here creates
    # duplicate middleware names that LangChain rejects during agent creation.
    middleware: list[AgentMiddleware[Any, Any, Any]] = [TodoListMiddleware()]
    filesystem_tools = list(DEFAULT_DEEPAGENT_FILESYSTEM_TOOLS)
    if config is not None:
        if config.extensions.delete_tool_enabled:
            filesystem_tools.append("delete")
        if config.extensions.execute_tool_enabled:
            filesystem_tools.append("execute")
    middleware.append(
        FilesystemMiddleware(
            backend=backend,
            tools=filesystem_tools,
        )
    )
    middleware.append(ToolExecutionResilienceMiddleware(project_root=project_root))
    return middleware


def resolve_local_path(path_value: str, base_dir: Path) -> Path:
    """Resolve local path.

    Args:
        path_value: The path value value.
        base_dir: The base dir value.

    Returns:
        The resolved local path.
    """
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def normalize_mcp_transport(value: str) -> str:
    """Normalize MCP transport.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized value.
    """
    transport = value.strip().lower()
    if transport == "streamable-http":
        return "streamable_http"
    return transport


def normalize_skill_source_path(path_value: str, base_dir: Path) -> str:
    """Normalize skill source path.

    Args:
        path_value: The path value value.
        base_dir: The base dir value.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
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
    """Normalize MCP server config.

    Args:
        raw_server: Raw server to process.
        base_dir: The base dir value.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
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
    """Normalize string mapping.

    Args:
        value: Value to normalize, convert, or serialize.
        field_name: The field name value.

    Returns:
        The normalized value.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"'{field_name}' must be a table/object.")
    return {str(key): str(raw_value) for key, raw_value in value.items()}


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


def virtual_workspace_path_to_local(path_value: str, project_root: Path | None = None) -> str:
    """Convert a virtual workspace path into a local filesystem path.

    Args:
        path_value: The path value value.
        project_root: Project root used to resolve local paths.

    Returns:
        The virtual workspace path to local result.
    """
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


class RenderChainlitUIInput(BaseModel):
    """Define the schema for generated Chainlit UI panel requests."""

    title: str = Field(
        ...,
        min_length=1,
        description="Short title shown at the top of the generated UI panel.",
    )
    summary: str | None = Field(
        default=None,
        description="Optional concise markdown-style summary for the panel body.",
    )
    facts: dict[str, Any] | None = Field(
        default=None,
        description="Optional key-value facts to display in a compact grid.",
    )
    items: list[Any] | None = Field(
        default=None,
        description=(
            "Optional short list items to display in order. Prefer strings; "
            "use actions, not items, for prompt buttons."
        ),
    )
    table: dict[str, Any] | None = Field(
        default=None,
        description="Optional small table with columns and rows.",
    )
    actions: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional prompt buttons with label and prompt values.",
    )
    id: str | None = Field(
        default=None,
        description="Optional stable panel id. Reusing it updates the existing panel.",
    )


def _normalize_generated_ui_props(
    *,
    title: str,
    summary: str | None = None,
    facts: dict[str, Any] | None = None,
    items: list[Any] | None = None,
    table: dict[str, Any] | None = None,
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return props accepted by the bundled GeneratedPanel custom element."""
    props: dict[str, Any] = {"title": title.strip()}
    normalized_actions: list[dict[str, str]] = []
    action_keys: set[tuple[str, str]] = set()

    def action_parts(value: Any) -> tuple[str, str] | None:
        if not isinstance(value, dict):
            return None
        label = str(value.get("label") or "").strip()
        prompt = str(value.get("prompt") or "").strip()
        if label and prompt:
            return label, prompt
        return None

    def append_action(value: Any) -> None:
        parts = action_parts(value)
        if parts is None:
            return
        label, prompt = parts
        if parts not in action_keys:
            normalized_actions.append({"label": label, "prompt": prompt})
            action_keys.add(parts)

    def item_text(value: Any) -> str:
        if isinstance(value, dict):
            for key in ("label", "text", "title", "value", "name"):
                text = str(value.get(key) or "").strip()
                if text:
                    return text
            return ""
        return str(value).strip()

    if summary is not None and summary.strip():
        props["summary"] = summary.strip()
    if facts:
        props["facts"] = {str(key): value for key, value in facts.items()}
    if items:
        normalized_items: list[str] = []
        for item in items:
            append_action(item)
            if action_parts(item) is not None:
                continue
            text = item_text(item)
            if text:
                normalized_items.append(text)
        if normalized_items:
            props["items"] = normalized_items
    if table:
        props["table"] = table
    if actions:
        for action in actions:
            append_action(action)
    if normalized_actions:
        props["actions"] = normalized_actions
    return props


def create_render_chainlit_ui_tool() -> Any:
    """Create the built-in tool that emits LangGraph UI messages for Chainlit."""

    @tool(
        "render_chainlit_ui",
        args_schema=RenderChainlitUIInput,
        return_direct=False,
    )
    def render_chainlit_ui(
        title: str,
        summary: str | None = None,
        facts: dict[str, Any] | None = None,
        items: list[Any] | None = None,
        table: dict[str, Any] | None = None,
        actions: list[dict[str, Any]] | None = None,
        id: str | None = None,
    ) -> dict[str, Any]:
        """Render a whitelisted Chainlit generated UI panel for the current answer."""
        props = _normalize_generated_ui_props(
            title=title,
            summary=summary,
            facts=facts,
            items=items,
            table=table,
            actions=actions,
        )
        ui_message = push_ui_message(
            GENERATIVE_UI_COMPONENT_NAME,
            props,
            id=(id.strip() if isinstance(id, str) and id.strip() else None),
            metadata={"source": "main-agent"},
            state_key=None,
        )
        return {
            "rendered": True,
            "component": ui_message["name"],
            "id": ui_message["id"],
        }

    return render_chainlit_ui


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


def parse_async_subagent_config(
    raw_subagent: dict[str, Any],
    *,
    index: int,
    source_name: str,
) -> AsyncSubagentConfig:
    """Parse async subagent config.

    Args:
        raw_subagent: Raw subagent to process.
        index: The index value.
        source_name: The source name value.

    Returns:
        The parsed async subagent config.

    Raises:
        ValueError: If the supplied value is invalid.
    """
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
    parent_name: str | None = None,
) -> SubagentConfig:
    """Parse sync subagent config.

    Args:
        raw_subagent: Raw subagent to process.
        index: The index value.
        base_dir: The base dir value.
        mcp_servers: The MCP servers value.
        parent_name: Parent subagent name for nested entries.

    Returns:
        The parsed sync subagent config.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    name = str(raw_subagent.get("name", "")).strip()
    description = str(raw_subagent.get("description", "")).strip()
    if not name or not description:
        raise ValueError(
            f"Subagent entry #{index} must include non-empty 'name' and 'description'."
        )
    if parent_name and "graph_id" in raw_subagent:
        raise ValueError(
            f"Subagent '{parent_name}' nested async subagents are not supported; "
            f"nested subagent '{name or index}' defines 'graph_id'."
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

    nested_subagent_names = normalize_required_string_list(
        raw_subagent.get("nested_subagents", []),
        field_name=f"subagent '{name}' nested_subagents",
    )
    raw_nested_subagents = raw_subagent.get("subagents", [])
    if not isinstance(raw_nested_subagents, list):
        raise ValueError(
            f"Subagent '{name}' nested 'subagents' config must be an array of tables."
        )
    nested_subagents: list[SubagentConfig] = []
    for nested_index, raw_nested_subagent in enumerate(raw_nested_subagents, start=1):
        if not isinstance(raw_nested_subagent, dict):
            raise ValueError(
                f"Subagent '{name}' nested subagent entry #{nested_index} must be a table/object."
            )
        nested_subagents.append(
            parse_sync_subagent_config(
                raw_nested_subagent,
                index=nested_index,
                base_dir=base_dir,
                mcp_servers=mcp_servers,
                parent_name=name,
            )
        )

    model = str(raw_subagent.get("model", "")).strip() or None
    return SubagentConfig(
        name=name,
        description=description,
        system_prompt=system_prompt,
        skills=subagent_skill_paths,
        mcp_servers=raw_subagent_mcp_servers,
        model=model,
        nested_subagent_names=nested_subagent_names,
        subagents=tuple(nested_subagents),
    )


def normalize_required_string_list(
    value: Any,
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Normalize a config field that must be a list of non-empty strings."""
    if not isinstance(value, list):
        raise ValueError(f"'{field_name}' must be a list of strings.")

    items: list[str] = []
    for index, raw_item in enumerate(value, start=1):
        item = str(raw_item).strip()
        if not item:
            raise ValueError(
                f"'{field_name}' entry #{index} must be a non-empty string."
            )
        items.append(item)
    return tuple(items)


def validate_subagent_names(
    subagents: tuple[SubagentConfig, ...],
    async_subagents: tuple[AsyncSubagentConfig, ...],
) -> None:
    """Validate top-level subagent name uniqueness across sync and async specs."""
    seen_names: set[str] = set()
    for subagent in (*subagents, *async_subagents):
        if subagent.name in seen_names:
            raise ValueError(
                f"Top-level subagent name '{subagent.name}' is defined more than once."
            )
        seen_names.add(subagent.name)


def validate_nested_subagent_references(
    subagents: tuple[SubagentConfig, ...],
) -> None:
    """Validate nested sync subagent references and cycles."""
    registry = {subagent.name: subagent for subagent in subagents}
    for subagent in subagents:
        validate_nested_subagent_reference_tree(
            subagent,
            registry=registry,
            path=(subagent.name,),
        )


def validate_nested_subagent_reference_tree(
    subagent: SubagentConfig,
    *,
    registry: dict[str, SubagentConfig],
    path: tuple[str, ...],
) -> None:
    """Validate one subagent's direct children and referenced descendants."""
    direct_child_names: set[str] = set()
    for child in subagent.subagents:
        if child.name in direct_child_names:
            raise ValueError(
                f"Subagent '{subagent.name}' has duplicate nested child subagent "
                f"'{child.name}'."
            )
        direct_child_names.add(child.name)

    for referenced_name in subagent.nested_subagent_names:
        if referenced_name not in registry:
            raise ValueError(
                f"Subagent '{subagent.name}' references unknown nested subagent "
                f"'{referenced_name}'. Defined subagents: {sorted(registry)}"
            )
        if referenced_name in direct_child_names:
            raise ValueError(
                f"Subagent '{subagent.name}' has duplicate nested child subagent "
                f"'{referenced_name}'."
            )
        if referenced_name in path:
            cycle = " -> ".join((*path, referenced_name))
            raise ValueError(f"nested subagent cycle detected: {cycle}")
        direct_child_names.add(referenced_name)

    for child in subagent.subagents:
        validate_nested_subagent_reference_tree(
            child,
            registry=registry,
            path=(*path, child.name),
        )
    for referenced_name in subagent.nested_subagent_names:
        validate_nested_subagent_reference_tree(
            registry[referenced_name],
            registry=registry,
            path=(*path, referenced_name),
        )


def parse_agent_custom_instruction(
    agent_section: dict[str, Any],
    base_dir: Path,
) -> str | None:
    """Parse inline or file-based main-agent custom instruction."""
    custom_instruction = normalize_optional_string(
        agent_section.get("custom_instruction")
    )
    custom_instruction_file = normalize_optional_string(
        agent_section.get("custom_instruction_file")
    )
    if custom_instruction and custom_instruction_file:
        raise ValueError(
            "The top-level 'agent' config cannot define both "
            "'custom_instruction' and 'custom_instruction_file'."
        )
    if not custom_instruction_file:
        return custom_instruction
    instruction_path = resolve_local_path(custom_instruction_file, base_dir)
    return instruction_path.read_text(encoding="utf-8").strip() or None


def parse_extensions_config(raw_config: dict[str, Any], config_path: Path) -> ExtensionsConfig:
    """Parse extensions config.

    Args:
        raw_config: Raw config to process.
        config_path: Path to the config.

    Returns:
        The parsed extensions config.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    base_dir = config_path.parent
    mcp_section = raw_config.get("mcp", {})
    if mcp_section and not isinstance(mcp_section, dict):
        raise ValueError("The top-level 'mcp' config must be a table/object.")

    agent_section = raw_config.get("agent", {})
    if agent_section and not isinstance(agent_section, dict):
        raise ValueError("The top-level 'agent' config must be a table/object.")

    recursion_limit = normalize_recursion_limit(
        agent_section.get("recursion_limit"),
        field_name="The top-level 'agent.recursion_limit' config",
    )
    agent_state = normalize_agent_state(agent_section.get("state"))
    agent_memory_namespace = normalize_agent_memory_namespace(
        agent_section.get("memory_namespace")
    )
    agent_memory_files = normalize_agent_memory_files(agent_section.get("memory_files"))
    raw_delete_tool_enabled = agent_section.get("delete_tool_enabled", False)
    if not isinstance(raw_delete_tool_enabled, bool):
        raise ValueError(
            "The top-level 'agent.delete_tool_enabled' config must be a boolean."
        )
    raw_execute_tool_enabled = agent_section.get("execute_tool_enabled", False)
    if not isinstance(raw_execute_tool_enabled, bool):
        raise ValueError(
            "The top-level 'agent.execute_tool_enabled' config must be a boolean."
        )
    agent_reflection = normalize_reflection_config(
        agent_section.get("reflection"),
        agent_state=agent_state,
    )
    agent_model = normalize_optional_string(agent_section.get("model"))
    raw_mcp_servers = mcp_section.get("servers", {})
    mcp_servers: dict[str, dict[str, Any]] = {}
    for name, raw_server in raw_mcp_servers.items():
        if not isinstance(raw_server, dict):
            raise ValueError(f"MCP server '{name}' must be a table/object.")
        mcp_servers[str(name)] = normalize_mcp_server_config(raw_server, base_dir)

    raw_skill_paths = agent_section.get("skills", [])
    custom_instruction = parse_agent_custom_instruction(agent_section, base_dir)
    raw_summarization_middleware_enabled = agent_section.get(
        "summarization_middleware_enabled",
        False,
    )
    raw_summarization_trigger_tokens = agent_section.get("summarization_trigger_tokens")
    raw_summarization_keep_tokens = agent_section.get("summarization_keep_tokens")
    if not isinstance(raw_summarization_middleware_enabled, bool):
        raise ValueError(
            "The top-level 'agent.summarization_middleware_enabled' config must be a boolean."
        )
    if raw_summarization_trigger_tokens is not None and (
        not isinstance(raw_summarization_trigger_tokens, int)
        or raw_summarization_trigger_tokens <= 0
    ):
        raise ValueError(
            "The top-level 'agent.summarization_trigger_tokens' config must be a positive integer."
        )
    if raw_summarization_keep_tokens is not None and (
        not isinstance(raw_summarization_keep_tokens, int)
        or raw_summarization_keep_tokens <= 0
    ):
        raise ValueError(
            "The top-level 'agent.summarization_keep_tokens' config must be a positive integer."
        )
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

    validate_subagent_names(tuple(subagents), tuple(async_subagents))
    validate_nested_subagent_references(tuple(subagents))

    chainlit_section = raw_config.get("chainlit", {})
    if chainlit_section and not isinstance(chainlit_section, dict):
        raise ValueError("The top-level 'chainlit' config must be a table/object.")

    raw_chainlit_commands = chainlit_section.get("commands", [])
    if not isinstance(raw_chainlit_commands, list):
        raise ValueError("The top-level 'chainlit.commands' config must be an array of tables.")
    raw_chainlit_starters = chainlit_section.get("starters", [])
    if not isinstance(raw_chainlit_starters, list):
        raise ValueError("The top-level 'chainlit.starters' config must be an array of tables.")
    raw_reasoning_mode_enabled = chainlit_section.get("reasoning_mode_enabled", True)
    raw_reasoning_steps_enabled = chainlit_section.get("reasoning_steps_enabled", True)
    raw_tool_steps_enabled = chainlit_section.get("tool_steps_enabled", True)
    raw_model_mode_enabled = chainlit_section.get("model_mode_enabled", True)
    raw_startup_status_enabled = chainlit_section.get("startup_status_enabled", True)
    raw_chronological_ui_enabled = chainlit_section.get("chronological_ui_enabled", True)
    raw_generative_ui_enabled = chainlit_section.get("generative_ui_enabled", True)
    if not isinstance(raw_reasoning_mode_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.reasoning_mode_enabled' config must be a boolean."
        )
    if not isinstance(raw_reasoning_steps_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.reasoning_steps_enabled' config must be a boolean."
        )
    if not isinstance(raw_tool_steps_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.tool_steps_enabled' config must be a boolean."
        )
    if not isinstance(raw_model_mode_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.model_mode_enabled' config must be a boolean."
        )
    if not isinstance(raw_startup_status_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.startup_status_enabled' config must be a boolean."
        )
    if not isinstance(raw_chronological_ui_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.chronological_ui_enabled' config must be a boolean."
        )
    if not isinstance(raw_generative_ui_enabled, bool):
        raise ValueError(
            "The top-level 'chainlit.generative_ui_enabled' config must be a boolean."
        )

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

    chainlit_starters: list[ChainlitStarterConfig] = []
    for index, raw_chainlit_starter in enumerate(raw_chainlit_starters, start=1):
        if not isinstance(raw_chainlit_starter, dict):
            raise ValueError(
                f"Chainlit starter entry #{index} must be a table/object."
            )
        label = str(raw_chainlit_starter.get("label", "")).strip()
        message = str(raw_chainlit_starter.get("message", "")).strip()
        command = normalize_optional_string(raw_chainlit_starter.get("command"))
        icon = normalize_optional_string(raw_chainlit_starter.get("icon"))
        if not label:
            raise ValueError(
                f"Chainlit starter entry #{index} must include a non-empty 'label'."
            )
        if not message:
            raise ValueError(
                f"Chainlit starter '{label}' must include a non-empty 'message'."
            )
        chainlit_starters.append(
            ChainlitStarterConfig(
                label=label,
                message=message,
                command=command,
                icon=icon,
            )
        )

    return ExtensionsConfig(
        config_path=config_path,
        mcp_tool_name_prefix=bool(mcp_section.get("tool_name_prefix", True)),
        mcp_stateful=bool(mcp_section.get("stateful", False)),
        agent_state=agent_state,
        agent_memory_namespace=agent_memory_namespace,
        agent_memory_files=agent_memory_files,
        delete_tool_enabled=raw_delete_tool_enabled,
        execute_tool_enabled=raw_execute_tool_enabled,
        agent_reflection=agent_reflection,
        agent_model=agent_model,
        recursion_limit=recursion_limit,
        mcp_servers=mcp_servers or None,
        skills=skill_paths,
        agent_mcp_servers=raw_agent_mcp_servers,
        subagents=tuple(subagents),
        async_subagents=tuple(async_subagents),
        chainlit_commands=tuple(chainlit_commands),
        chainlit_starters=tuple(chainlit_starters),
        chainlit_model_mode_enabled=raw_model_mode_enabled,
        chainlit_reasoning_mode_enabled=raw_reasoning_mode_enabled,
        chainlit_reasoning_steps_enabled=raw_reasoning_steps_enabled,
        chainlit_tool_steps_enabled=raw_tool_steps_enabled,
        chainlit_startup_status_enabled=raw_startup_status_enabled,
        chainlit_chronological_ui_enabled=raw_chronological_ui_enabled,
        chainlit_generative_ui_enabled=raw_generative_ui_enabled,
        summarization_middleware_enabled=raw_summarization_middleware_enabled,
        summarization_trigger_tokens=raw_summarization_trigger_tokens,
        summarization_keep_tokens=raw_summarization_keep_tokens,
        custom_instruction=custom_instruction,
    )


def load_agents_md_instruction(project_root: Path | None = None) -> str | None:
    """Load agents md instruction.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The loaded value.
    """
    agents_md_path = (project_root or PROJECT_ROOT).resolve() / AGENTS_MD_FILENAME
    try:
        instruction = agents_md_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("Failed to read %s: %s", agents_md_path, exc)
        return None
    return instruction or None


def compose_agent_system_prompt(
    base_prompt: str,
    custom_instruction: str | None,
    *,
    project_root: Path | None = None,
) -> str:
    """Compose agent system prompt.

    Args:
        base_prompt: The base prompt value.
        custom_instruction: The custom instruction value.
        project_root: Project root used to resolve local paths.

    Returns:
        The composed value.
    """
    sections = [base_prompt]
    agents_md_instruction = load_agents_md_instruction(project_root)
    if agents_md_instruction:
        sections.append(
            f"Repository instructions from {AGENTS_MD_FILENAME}:\n"
            f"{agents_md_instruction}"
        )

    instruction = (custom_instruction or "").strip()
    if not instruction:
        return "\n\n".join(sections)
    sections.append(
        "Custom user instruction from deepagent.toml:\n"
        f"{instruction}"
    )
    return "\n\n".join(sections)


def system_prompt_for_agent_state(
    base_prompt: str,
    agent_state: AgentStateMode,
) -> str:
    """Return the system prompt adjusted for configured agent state.

    Args:
        base_prompt: The base prompt value.
        agent_state: Whether the DeepAgents graph is stateful or stateless.

    Returns:
        The adjusted system prompt.
    """
    if agent_state == "stateful":
        return base_prompt
    return base_prompt.replace(
        SYSTEM_PROMPT_MEMORY_LINE,
        STATELESS_SYSTEM_PROMPT_MEMORY_LINE,
    )


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
    resolved_config_path = resolve_local_path(
        config_name or DEFAULT_EXTENSIONS_CONFIG,
        PROJECT_ROOT,
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

    model_defaults = parse_model_defaults(raw_config)
    raw_model = raw_config.get("model", {})
    return FileConfig(
        model=model_defaults,
        extensions=parse_extensions_config(raw_config, resolved_config_path),
        model_profiles=parse_model_profiles(raw_model, base=model_defaults),
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
            database_url = normalize_optional_string(overrides.database_url)
        else:
            database_url = os.getenv("DATABASE_URL", "").strip() or None
        model_provider_override = normalize_optional_string(
            overrides.model_provider
        )
        if model_provider_override is None:
            model_provider_override = normalize_optional_string(
                os.getenv("DEEPAGENT_MODEL_PROVIDER")
            )
        model_provider = normalize_model_provider(
            model_provider_override,
            default=model_defaults.provider,
        )
        generic_model_name = (
            normalize_optional_string(overrides.model_name)
            or os.getenv("DEEPAGENT_MODEL_NAME", "").strip()
        )
        override_model_base_url = normalize_optional_string(overrides.model_base_url)
        env_model_base_url = normalize_optional_string(
            os.getenv("DEEPAGENT_MODEL_BASE_URL")
        )
        generic_model_base_url = override_model_base_url or env_model_base_url or ""
        generic_model_base_url_override = bool(generic_model_base_url)
        generic_model_base_url_from_env = (
            override_model_base_url is None and env_model_base_url is not None
        )
        generic_model_endpoint_url = (
            normalize_optional_string(overrides.model_endpoint_url)
            or os.getenv("DEEPAGENT_MODEL_ENDPOINT_URL", "").strip()
        )
        generic_model_reasoning = (
            normalize_optional_string(overrides.reasoning_level)
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
            provider_label = format_model_provider(model_provider)
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
        active_model_defaults = resolve_model_profile_defaults(
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
            ) = normalize_snowflake_cortex_endpoint_url(
                generic_model_base_url,
                full_endpoint=False,
                required_message="The Snowflake Cortex model base URL cannot be empty.",
            )
        elif cross_provider_profile_selected and generic_model_base_url_override:
            cross_provider_model_base_url = normalize_model_base_url(
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
                ) = normalize_anthropic_endpoint_url(
                    generic_model_endpoint_url,
                    required_message=(
                        "The Anthropic model endpoint URL cannot be empty."
                    ),
                )
            elif active_model_defaults.provider == "snowflake_cortex":
                (
                    cross_provider_model_base_url,
                    cross_provider_model_endpoint_query,
                ) = normalize_snowflake_cortex_endpoint_url(
                    generic_model_endpoint_url,
                    full_endpoint=True,
                    required_message="The Snowflake Cortex model endpoint URL cannot be empty.",
                )
            elif active_model_defaults.provider == "openai_compatible":
                (
                    cross_provider_model_base_url,
                    cross_provider_model_endpoint_query,
                ) = normalize_openai_endpoint_url(
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
                model_base_url, model_endpoint_query = normalize_anthropic_endpoint_url(
                    generic_model_endpoint_url,
                    required_message="The Anthropic model endpoint URL cannot be empty.",
                )
            else:
                model_base_url = normalize_model_base_url(
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
            model_base_url, model_endpoint_query = normalize_snowflake_cortex_endpoint_url(
                generic_model_endpoint_url,
                full_endpoint=True,
                required_message="The Snowflake Cortex model endpoint URL cannot be empty.",
            )
        elif model_provider == "openai_compatible" and generic_model_endpoint_url:
            model_base_url, model_endpoint_query = normalize_openai_endpoint_url(
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
                    ) = normalize_snowflake_cortex_endpoint_url(
                        selected_base_url,
                        full_endpoint=False,
                        required_message=(
                            "The Snowflake Cortex model base URL cannot be empty."
                        ),
                    )
                else:
                    model_base_url = model_defaults.base_url
            else:
                model_base_url = normalize_model_base_url(
                    selected_base_url,
                    required_message="The model base URL cannot be empty.",
                )
            if generic_model_base_url or model_base_url_alias:
                model_endpoint_query = ()
        model_api_key_override = (
            normalize_optional_string(overrides.model_api_key)
            if overrides.model_api_key is not None
            else None
        )
        if overrides.model_api_key is not None:
            model_api_key = model_api_key_override
        else:
            provider_specific_api_key = (
                normalize_optional_string(os.getenv("ANTHROPIC_API_KEY"))
                if model_provider == "anthropic"
                else (
                    normalize_optional_string(os.getenv("SNOWFLAKE_PAT"))
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
                or normalize_optional_string(os.getenv("DEEPAGENT_MODEL_API_KEY"))
                or model_default_api_key
            )
        model_temperature = (
            normalize_model_temperature(overrides.model_temperature)
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
            model_disable_streaming = normalize_disable_streaming(
                overrides.model_disable_streaming
            )
        else:
            if raw_disable_streaming is not None:
                model_disable_streaming = normalize_disable_streaming(raw_disable_streaming)
            elif raw_disable_streaming_for_tool_calls is not None:
                if normalize_disable_streaming_for_tool_calls(
                    raw_disable_streaming_for_tool_calls
                ):
                    model_disable_streaming = "tool_calling"
                else:
                    model_disable_streaming = False
            else:
                model_disable_streaming = model_defaults.disable_streaming
        default_reasoning = normalize_reasoning_level(
            generic_model_reasoning or model_reasoning_alias,
            default=model_defaults.reasoning_effort,
        )
        model_reasoning_override = bool(generic_model_reasoning or model_reasoning_alias)
        recursion_limit = normalize_recursion_limit(
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
        active_runtime_model = resolve_model_profile_defaults(
            runtime_default_model,
            file_config.model_profiles,
            model_name,
        )
        active_runtime_provider_key = (
            normalize_optional_string(os.getenv("ANTHROPIC_API_KEY"))
            if active_runtime_model.provider == "anthropic"
            else (
                normalize_optional_string(os.getenv("SNOWFLAKE_PAT"))
                if active_runtime_model.provider == "snowflake_cortex"
                else None
            )
        )
        active_runtime_api_key = (
            model_api_key_override
            or active_runtime_provider_key
            or normalize_optional_string(os.getenv("DEEPAGENT_MODEL_API_KEY"))
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


def _import_langfuse_callback_handler() -> type[Any]:
    """Import Langfuse's LangChain callback handler on demand.

    Returns:
        The Langfuse LangChain callback handler type.

    Raises:
        RuntimeError: If Langfuse support is enabled but unavailable.
    """
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as exc:
        raise RuntimeError(
            "Langfuse tracing is enabled but the 'langfuse' package is not installed. "
            "Run `uv sync` to install project dependencies."
        ) from exc
    return CallbackHandler


def build_langfuse_callback_handler(config: RuntimeConfig) -> Any | None:
    """Build a Langfuse callback handler when tracing is enabled.

    Args:
        config: Configuration object used by the operation.

    Returns:
        A Langfuse callback handler, or None when disabled.
    """
    langfuse = getattr(config, "langfuse", LangfuseConfig())
    if not langfuse.enabled:
        return None
    handler_cls = _import_langfuse_callback_handler()
    return handler_cls()


def shutdown_langfuse_client(config: RuntimeConfig) -> bool:
    """Shut down Langfuse's buffered client when tracing is enabled.

    Langfuse batches events in background workers, so short-lived CLI processes
    need an explicit shutdown before process exit to avoid dropping traces.

    Args:
        config: Configuration object used by the operation.

    Returns:
        True when Langfuse tracing was enabled and shutdown was requested.
    """
    langfuse = getattr(config, "langfuse", LangfuseConfig())
    if not langfuse.enabled:
        return False

    from langfuse import get_client

    get_client().shutdown()
    return True


def build_langgraph_run_config(
    config: RuntimeConfig,
    *,
    thread_id: str,
) -> dict[str, Any]:
    """Build LangGraph run config shared by all ChainAgents entrypoints.

    Args:
        config: Configuration object used by the operation.
        thread_id: Conversation thread identifier.

    Returns:
        A LangGraph configuration dictionary for the run.
    """
    run_config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": config.recursion_limit,
    }
    langfuse_handler = build_langfuse_callback_handler(config)
    if langfuse_handler is not None:
        run_config["callbacks"] = [langfuse_handler]
        run_config["metadata"] = {"langfuse_session_id": thread_id}
        run_config["tags"] = ["chainagents"]
    return run_config


def runtime_default_model_profile(config: RuntimeConfig) -> ModelDefaults:
    """Return the default model profile represented by flattened runtime fields."""
    provider = normalize_model_provider(
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
        reasoning_effort=normalize_reasoning_level(
            getattr(config, "default_reasoning", DEFAULT_REASONING_LEVEL),
        ),
        thinking=normalize_model_thinking(getattr(config, "model_thinking", None)),
        temperature=normalize_model_temperature(
            getattr(config, "model_temperature", DEFAULT_TEMPERATURE)
        ),
        repeat_penalty=normalize_repeat_penalty(
            getattr(config, "model_repeat_penalty", None)
        ),
        disable_streaming=normalize_disable_streaming(
            getattr(config, "model_disable_streaming", False)
        ),
        modalities=normalize_model_modalities(
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
    return resolve_model_profile_defaults(
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
        provider_key = normalize_optional_string(os.getenv("ANTHROPIC_API_KEY"))
        if provider_key:
            return provider_key
    if model_profile.provider == "snowflake_cortex":
        provider_key = normalize_optional_string(os.getenv("SNOWFLAKE_PAT"))
        if provider_key:
            return provider_key
    generic_key = normalize_optional_string(os.getenv("DEEPAGENT_MODEL_API_KEY"))
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
        default_query = model_endpoint_query_to_dict(resolved_profile.endpoint_query)
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
        default_query = model_endpoint_query_to_dict(resolved_profile.endpoint_query)
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
    default_query = model_endpoint_query_to_dict(resolved_profile.endpoint_query)
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


def build_deepagent_backend(
    *,
    project_root: Path | None = None,
    include_memories: bool = True,
    memory_namespace: str = DEFAULT_AGENT_MEMORY_NAMESPACE,
) -> CompositeBackend:
    """Build deepagent backend.

    Args:
        project_root: Project root used to resolve local paths.
        include_memories: Whether to expose the /memories/ store route.
        memory_namespace: Shared StoreBackend namespace for /memories/.

    Returns:
        The constructed deepagent backend.
    """
    resolved_project_root = project_root or PROJECT_ROOT
    artifacts_root = deepagent_artifacts_root(resolved_project_root)
    outputs_root = generated_outputs_root(resolved_project_root)
    routes = {
        deepagent_artifacts_route_prefix(resolved_project_root): FilesystemBackend(
            root_dir=str(artifacts_root),
            virtual_mode=True,
        ),
        generated_outputs_route_prefix(resolved_project_root): FilesystemBackend(
            root_dir=str(outputs_root),
            virtual_mode=True,
        ),
        "/workspace/": FilesystemBackend(
            root_dir=str(resolved_project_root),
            virtual_mode=True,
        ),
    }
    if include_memories:
        routes["/memories/"] = StoreBackend(
            namespace=lambda _runtime: (memory_namespace,)
        )
    return CompositeBackend(
        default=StateBackend(),
        routes=routes,
        artifacts_root=str(artifacts_root),
    )


def normalize_chainlit_command_name(value: str) -> str:
    """Normalize chainlit command name.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized value.
    """
    return value.strip().lstrip("/").lower()


def _load_skill_command_bucket(
    *,
    backend: CompositeBackend,
    source_paths: tuple[str, ...],
    source: Literal["agent_skill", "subagent_skill"],
    project_root: Path | None = None,
    owner: str | None = None,
) -> tuple[SkillCommandMetadata, ...]:
    """Load skill command metadata from one configured source bucket.

    Args:
        backend: The backend value.
        source_paths: Paths to the source.
        source: The source value.
        project_root: Project root used to resolve local paths.
        owner: The owner value.

    Returns:
        The loaded value.
    """
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
    """Resolve the project root used for Chainlit command discovery.

    Args:
        backend: The backend value.
        project_root: Project root used to resolve local paths.

    Returns:
        The resolved the project root used for chainlit command discovery.
    """
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
    """Build chainlit command catalog.

    Args:
        extensions: The extensions value.
        backend: The backend value.
        project_root: Project root used to resolve local paths.

    Returns:
        The constructed chainlit command catalog.
    """
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
    """Sanitize tools for model.

    Args:
        model_provider: The model provider value.
        tools: The tools value.

    Returns:
        The sanitized value.
    """
    if model_provider == "anthropic":
        return [normalize_anthropic_tool_schema(tool) for tool in tools]

    if model_provider not in OPENAI_COMPATIBLE_MODEL_PROVIDERS:
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


def normalize_anthropic_tool_schema(tool: Any) -> Any:
    """Normalize tool schemas for Anthropic's stricter root object requirement.

    Args:
        tool: The tool value.

    Returns:
        The normalized tool value.
    """
    schema = getattr(tool, "args_schema", None)
    if not isinstance(schema, dict):
        return tool

    normalized_schema = normalize_json_object_schema_root(schema)
    if normalized_schema is schema:
        return tool

    if hasattr(tool, "model_copy"):
        return tool.model_copy(update={"args_schema": normalized_schema})

    try:
        cloned = copy.copy(tool)
        setattr(cloned, "args_schema", normalized_schema)
        return cloned
    except Exception:
        setattr(tool, "args_schema", normalized_schema)
        return tool


def normalize_json_object_schema_root(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure a JSON schema dict declares an object root when unspecified.

    Args:
        schema: The schema value.

    Returns:
        The normalized schema.
    """
    if schema.get("type") == "object":
        return schema

    if schema.get("type") is not None:
        return schema

    return {**schema, "type": "object"}


def tool_supports_openai_compatible_schema(tool: Any) -> bool:
    """Return whether a tool schema is OpenAI-compatible.

    Args:
        tool: The tool value.

    Returns:
        Whether a tool schema is OpenAI-compatible.
    """
    try:
        schema = convert_to_openai_tool(tool)
    except Exception:
        return False

    parameters = schema.get("function", {}).get("parameters")
    return isinstance(parameters, dict) and parameters.get("type") == "object"


def nested_child_subagents(
    subagent: SubagentConfig,
    registry: dict[str, SubagentConfig],
) -> tuple[SubagentConfig, ...]:
    """Return inline and referenced nested child subagents in config order."""
    return (
        *subagent.subagents,
        *(registry[name] for name in subagent.nested_subagent_names),
    )


def has_nested_child_subagents(subagent: SubagentConfig) -> bool:
    """Return whether a sync subagent exposes child subagents."""
    return bool(subagent.subagents or subagent.nested_subagent_names)


def inherited_tools_for_model(
    *,
    inherited_tools: list[Any],
    sanitized_inherited_tools: list[Any] | None = None,
    inherited_provider: ModelProvider,
    effective_provider: ModelProvider,
) -> list[Any]:
    """Return inherited tools adjusted for a subagent's effective model provider."""
    if effective_provider == inherited_provider:
        return list(
            sanitized_inherited_tools
            if sanitized_inherited_tools is not None
            else inherited_tools
        )
    return sanitize_tools_for_model(effective_provider, list(inherited_tools))


def reasoning_level_for_profile(
    model_profile: ModelDefaults,
    fallback: ReasoningLevel,
    *,
    fallback_is_explicit: bool = False,
) -> ReasoningLevel:
    """Return the reasoning level to use when building a profile-backed model."""
    if fallback_is_explicit:
        return fallback
    if "reasoning_effort" in model_profile.explicit_fields:
        return model_profile.reasoning_effort
    if (
        not model_profile.explicit_fields
        and model_profile.reasoning_effort != DEFAULT_REASONING_LEVEL
    ):
        return model_profile.reasoning_effort
    return fallback


def build_static_sync_subagent_spec(
    config: RuntimeConfig,
    subagent: SubagentConfig,
    *,
    registry: dict[str, SubagentConfig],
    backend: Any,
    inherited_tools: list[Any],
    reasoning_level: ReasoningLevel,
    inherited_model: ModelDefaults,
    project_root: Path | None,
    reasoning_level_is_explicit: bool = False,
) -> dict[str, Any]:
    """Build a sync subagent spec for configured graph creation."""
    effective_model = resolve_runtime_model_profile(
        config,
        subagent.model,
        inherited_model=inherited_model,
    )
    effective_reasoning_level = reasoning_level_for_profile(
        effective_model,
        reasoning_level,
        fallback_is_explicit=reasoning_level_is_explicit,
    )
    inherited_model_tools = inherited_tools_for_model(
        inherited_tools=inherited_tools,
        inherited_provider=inherited_model.provider,
        effective_provider=effective_model.provider,
    )
    effective_tools = inherited_model_tools
    middleware = build_agent_middleware(
        backend=backend,
        config=config,
        reasoning_level=effective_reasoning_level,
        model_name=effective_model.name,
        source=subagent.name,
        project_root=project_root,
    )
    if not has_nested_child_subagents(subagent):
        subagent_model = (
            build_model_for_profile(
                config,
                effective_reasoning_level,
                effective_model,
            )
            if subagent.model
            else None
        )
        subagent_tools = (
            inherited_model_tools
            if subagent.model and effective_model.provider != inherited_model.provider
            else []
        )
        return subagent.to_deepagents_spec(
            tools=subagent_tools,
            middleware=middleware,
            model=subagent_model,
        )

    child_specs = [
        build_static_sync_subagent_spec(
            config,
            child,
            registry=registry,
            backend=backend,
            inherited_tools=effective_tools,
            reasoning_level=effective_reasoning_level,
            reasoning_level_is_explicit=reasoning_level_is_explicit,
            inherited_model=effective_model,
            project_root=project_root,
        )
        for child in nested_child_subagents(subagent, registry)
    ]
    runnable_kwargs: dict[str, Any] = {
        "model": build_model_for_profile(
            config,
            effective_reasoning_level,
            effective_model,
        ),
        "tools": effective_tools or None,
        "system_prompt": subagent.system_prompt,
        "middleware": middleware,
        "backend": backend,
        "skills": list(subagent.skills) or None,
        "subagents": child_specs or None,
    }
    runnable = create_deep_agent_with_configured_summarization(
        config,
        **runnable_kwargs,
    )
    return {
        "name": subagent.name,
        "description": subagent.description,
        "runnable": runnable,
    }


def build_graph_subagent_specs(
    config: RuntimeConfig,
    *,
    include_async_subagents: bool,
    backend: Any | None = None,
    project_root: Path | None = None,
    inherited_tools: list[Any] | None = None,
) -> list[Any]:
    """Build graph subagent specs.

    Args:
        config: Configuration object used by the operation.
        include_async_subagents: Whether to include async subagents.
        backend: DeepAgents backend shared with compiled nested subgraphs.
        project_root: Project root used to resolve runtime middleware context.
        inherited_tools: Tools inherited from the graph that owns these subagents.

    Returns:
        The constructed graph subagent specs.
    """
    registry = {subagent.name: subagent for subagent in config.extensions.subagents}
    resolved_backend = backend or build_deepagent_backend(
        project_root=project_root,
        include_memories=config.agent_state == "stateful",
        memory_namespace=config.extensions.agent_memory_namespace,
    )
    inherited_model = resolve_runtime_model_profile(config)
    graph_reasoning_level = reasoning_level_for_profile(
        inherited_model,
        config.default_reasoning,
        fallback_is_explicit=config.model_reasoning_override,
    )
    subagent_specs: list[Any] = [
        build_static_sync_subagent_spec(
            config,
            subagent,
            registry=registry,
            backend=resolved_backend,
            inherited_tools=list(inherited_tools or []),
            reasoning_level=graph_reasoning_level,
            reasoning_level_is_explicit=config.model_reasoning_override,
            inherited_model=inherited_model,
            project_root=project_root,
        )
        for subagent in config.extensions.subagents
    ]
    if include_async_subagents:
        subagent_specs.extend(
            subagent.to_deepagents_spec()
            for subagent in config.extensions.async_subagents
        )
    return subagent_specs


def stateful_agent_memory_files(config: RuntimeConfig) -> list[str] | None:
    """Return startup memory files for stateful agents.

    Args:
        config: Configuration object used by the operation.

    Returns:
        The configured memory file paths, or None when startup memory is disabled.
    """
    if config.agent_state != "stateful" or not config.extensions.agent_memory_files:
        return None
    return list(config.extensions.agent_memory_files)


def create_configured_graph(
    *,
    include_async_subagents: bool,
    system_prompt: str = SYSTEM_PROMPT,
    apply_custom_instruction: bool = False,
) -> Any:
    """Create configured graph.

    Args:
        include_async_subagents: Whether to include async subagents.
        system_prompt: The system prompt value.
        apply_custom_instruction: The apply custom instruction value.

    Returns:
        The created configured graph.
    """
    config = RuntimeConfig.from_env()
    backend = build_deepagent_backend(
        include_memories=config.agent_state == "stateful",
        memory_namespace=config.extensions.agent_memory_namespace,
    )
    tools: list[Any] = []
    if config.extensions.chainlit_generative_ui_enabled:
        tools.append(create_render_chainlit_ui_tool())
    if config.rag is not None:
        tools.append(
            create_search_workspace_knowledge_tool(
                WorkspaceDocsRAG(config.rag, project_root=PROJECT_ROOT)
            )
        )
    else:
        if config.rag_requested and config.rag_error:
            logger.warning("RAG is configured but unavailable: %s", config.rag_error)
    main_model_profile = resolve_runtime_model_profile(config)
    main_tools = sanitize_tools_for_model(main_model_profile.provider, tools)
    main_reasoning_level = reasoning_level_for_profile(
        main_model_profile,
        config.default_reasoning,
        fallback_is_explicit=config.model_reasoning_override,
    )
    subagent_specs = build_graph_subagent_specs(
        config,
        include_async_subagents=include_async_subagents,
        backend=backend,
        project_root=PROJECT_ROOT,
        inherited_tools=main_tools,
    )
    agent_kwargs: dict[str, Any] = {
        "model": build_model_for_profile(
            config,
            main_reasoning_level,
            main_model_profile,
        ),
        "tools": main_tools or None,
        "system_prompt": compose_rag_system_prompt(
            compose_agent_system_prompt(
                system_prompt_for_agent_state(system_prompt, config.agent_state),
                (
                    config.extensions.custom_instruction
                    if apply_custom_instruction
                    else None
                ),
                project_root=PROJECT_ROOT,
            ),
            rag_enabled=config.rag is not None,
        ),
        "middleware": build_agent_middleware(
            backend=backend,
            config=config,
            reasoning_level=main_reasoning_level,
            source="main-agent",
            project_root=PROJECT_ROOT,
        ),
        "backend": backend,
        "skills": list(config.extensions.skills) or None,
        "subagents": subagent_specs or None,
    }
    memory_files = stateful_agent_memory_files(config)
    if memory_files is not None:
        agent_kwargs["memory"] = memory_files
    return create_deep_agent_with_configured_summarization(config, **agent_kwargs)


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


class _MCPSessionOwner:
    """Enter and exit transport cancel scopes on the same long-lived task."""

    def __init__(self, context: Any) -> None:
        self._ready = asyncio.get_running_loop().create_future()
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run(context))

    async def _run(self, context: Any) -> None:
        try:
            async with context as session:
                self._ready.set_result(session)
                await self._stop.wait()
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            else:
                raise

    async def session(self) -> Any:
        try:
            return await asyncio.shield(self._ready)
        except BaseException:
            await self.aclose()
            # Retrieve a startup error even when the requesting task was cancelled.
            if self._ready.done():
                self._ready.exception()
            raise

    async def aclose(self) -> None:
        self._stop.set()
        if not self._ready.done():
            self._task.cancel()
        await asyncio.shield(self._task)


class AgentRuntime:
    """Own configured agents, MCP sessions, persistence handles, and RAG state."""

    _instance: "AgentRuntime | None" = None
    _instance_lock = asyncio.Lock()

    def __init__(self, config: RuntimeConfig, *, project_root: Path | None = None) -> None:
        """Initialize the agent runtime instance.

        Args:
            config: Configuration object used by the operation.
            project_root: Project root used to resolve local paths.
        """
        self.config = config
        self.project_root = project_root or PROJECT_ROOT
        self._exit_stack = AsyncExitStack()
        self._agent_lock = asyncio.Lock()
        self._mcp_lock = asyncio.Lock()
        self._agents: dict[AgentCacheKey, object] = {}
        self._mcp_client: MultiServerMCPClient | None = None
        self._mcp_tools_cache: dict[tuple[str | None, tuple[str, ...]], list[Any]] = {}
        self._mcp_sessions: dict[tuple[str | None, str], Any] = {}
        self._mcp_session_owners: dict[tuple[str | None, str], _MCPSessionOwner] = {}
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
        """Get the agent runtime.

        Returns:
            The requested value.
        """
        async with cls._instance_lock:
            if cls._instance is None:
                instance = cls(RuntimeConfig.from_env())
                try:
                    await instance._initialize()
                except BaseException:
                    await instance.close()
                    raise
                cls._instance = instance
            return cls._instance

    @classmethod
    async def create(
        cls,
        config: RuntimeConfig | None = None,
        *,
        project_root: Path | None = None,
    ) -> "AgentRuntime":
        """Create the agent runtime.

        Args:
            config: Configuration object used by the operation.
            project_root: Project root used to resolve local paths.

        Returns:
            The created the agent runtime.
        """
        instance = cls(config or RuntimeConfig.from_env(), project_root=project_root)
        try:
            await instance._initialize()
        except BaseException:
            await instance.close()
            raise
        return instance

    @classmethod
    def current(cls) -> "AgentRuntime | None":
        """Return the current.

        Returns:
            The current.
        """
        return cls._instance

    @property
    def checkpointer(self) -> AsyncPostgresSaver | MemorySaver:
        """Return the initialized LangGraph checkpointer.

        Returns:
            The initialized LangGraph checkpointer.

        Raises:
            RuntimeError: If the runtime is not in a usable state.
        """
        if self._checkpointer is None:
            raise RuntimeError("Checkpointer is not initialized.")
        return self._checkpointer

    @property
    def store(self) -> AsyncPostgresStore | InMemoryStore:
        """Store the agent runtime.

        Returns:
            The stored value.

        Raises:
            RuntimeError: If the runtime is not in a usable state.
        """
        if self._store is None:
            raise RuntimeError("Store is not initialized.")
        return self._store

    @property
    def persistence_enabled(self) -> bool:
        """Return whether durable persistence is configured.

        Returns:
            Whether durable persistence is configured.
        """
        return (
            self.config.agent_state == "stateful"
            and self.config.persistence_mode == "postgres"
        )

    @property
    def rag_enabled(self) -> bool:
        """Return whether the RAG service is available.

        Returns:
            Whether the RAG service is available.
        """
        return self.config.rag_requested

    @property
    def chainlit_commands(self) -> tuple[ChainlitCommandConfig, ...]:
        """Return configured native Chainlit commands.

        Returns:
            Configured native Chainlit commands.
        """
        return self._chainlit_commands

    @property
    def chainlit_command_notes(self) -> tuple[str, ...]:
        """Return notes explaining configured Chainlit commands.

        Returns:
            Notes explaining configured Chainlit commands.
        """
        return self._chainlit_command_notes

    @property
    def rag_status(self) -> RagStatus:
        """Return the current RAG service status.

        Returns:
            The current RAG service status.
        """
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
        """Initialize persistence, RAG, MCP clients, and configured agents."""
        if self.config.extensions.mcp_servers:
            self._mcp_client = MultiServerMCPClient(
                self.config.extensions.mcp_servers,
                tool_name_prefix=self.config.extensions.mcp_tool_name_prefix,
            )

        if self.config.agent_state == "stateless":
            self._store = None
            self._checkpointer = None
        elif not self.config.database_url:
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

    async def _build_runtime_subagent_specs(
        self,
        *,
        reasoning_level: ReasoningLevel,
        reasoning_level_is_explicit: bool,
        selected_model_profile: ModelDefaults,
        backend: Any,
        inherited_tools: list[Any],
        sanitized_inherited_tools: list[Any],
        thread_id: str | None,
        mcp_session_id: str | None,
    ) -> list[Any]:
        """Build top-level sync subagent specs for a runtime context."""
        registry = {
            subagent.name: subagent for subagent in self.config.extensions.subagents
        }
        return [
            await self._build_runtime_sync_subagent_spec(
                subagent,
                registry=registry,
                reasoning_level=reasoning_level,
                reasoning_level_is_explicit=reasoning_level_is_explicit,
                inherited_model=selected_model_profile,
                backend=backend,
                inherited_tools=inherited_tools,
                sanitized_inherited_tools=sanitized_inherited_tools,
                thread_id=thread_id,
                mcp_session_id=mcp_session_id,
            )
            for subagent in self.config.extensions.subagents
        ]

    async def _build_runtime_sync_subagent_spec(
        self,
        subagent: SubagentConfig,
        *,
        registry: dict[str, SubagentConfig],
        reasoning_level: ReasoningLevel,
        reasoning_level_is_explicit: bool,
        inherited_model: ModelDefaults,
        backend: Any,
        inherited_tools: list[Any],
        sanitized_inherited_tools: list[Any],
        thread_id: str | None,
        mcp_session_id: str | None,
    ) -> dict[str, Any]:
        """Build one sync subagent spec, compiling it when it has children."""
        effective_model = resolve_runtime_model_profile(
            self.config,
            subagent.model,
            inherited_model=inherited_model,
        )
        effective_reasoning_level = reasoning_level_for_profile(
            effective_model,
            reasoning_level,
            fallback_is_explicit=reasoning_level_is_explicit,
        )
        raw_own_tools = await self._get_mcp_tools(
            subagent.mcp_servers,
            thread_id=thread_id,
            mcp_session_id=mcp_session_id,
        )
        own_tools = sanitize_tools_for_model(
            effective_model.provider,
            raw_own_tools,
        )
        inherited_model_tools = inherited_tools_for_model(
            inherited_tools=inherited_tools,
            sanitized_inherited_tools=sanitized_inherited_tools,
            inherited_provider=inherited_model.provider,
            effective_provider=effective_model.provider,
        )
        has_configured_own_tools = bool(subagent.mcp_servers)
        effective_tools = (
            own_tools
            if has_configured_own_tools
            else own_tools or inherited_model_tools
        )
        middleware = build_agent_middleware(
            backend=backend,
            config=self.config,
            reasoning_level=effective_reasoning_level,
            model_name=effective_model.name,
            source=subagent.name,
            project_root=self.project_root,
        )
        if not has_nested_child_subagents(subagent):
            subagent_tools = own_tools
            if (
                not subagent_tools
                and subagent.model
                and effective_model.provider != inherited_model.provider
                and not has_configured_own_tools
            ):
                subagent_tools = inherited_model_tools
            subagent_model = (
                self._build_model(
                    effective_reasoning_level,
                    model_profile=effective_model,
                )
                if subagent.model
                else None
            )
            return subagent.to_deepagents_spec(
                tools=subagent_tools,
                middleware=middleware,
                model=subagent_model,
            )

        child_specs = [
            await self._build_runtime_sync_subagent_spec(
                child,
                registry=registry,
                reasoning_level=effective_reasoning_level,
                reasoning_level_is_explicit=reasoning_level_is_explicit,
                inherited_model=effective_model,
                backend=backend,
                inherited_tools=raw_own_tools if has_configured_own_tools else inherited_tools,
                sanitized_inherited_tools=effective_tools,
                thread_id=thread_id,
                mcp_session_id=mcp_session_id,
            )
            for child in nested_child_subagents(subagent, registry)
        ]
        runnable_kwargs: dict[str, Any] = {
            "model": self._build_model(
                effective_reasoning_level,
                model_profile=effective_model,
            ),
            "tools": effective_tools or None,
            "system_prompt": subagent.system_prompt,
            "middleware": middleware,
            "backend": backend,
            "skills": list(subagent.skills) or None,
            "subagents": child_specs or None,
        }
        if self.config.agent_state == "stateful":
            runnable_kwargs["store"] = self.store
            runnable_kwargs["checkpointer"] = self.checkpointer
        runnable = create_deep_agent_with_configured_summarization(
            self.config,
            **runnable_kwargs,
        )
        return {
            "name": subagent.name,
            "description": subagent.description,
            "runnable": runnable,
        }

    async def get_agent(
        self,
        reasoning_level: ReasoningLevel,
        *,
        model_name: str | None = None,
        reasoning_level_is_explicit: bool = False,
        thread_id: str | None = None,
        async_subagent_url_override: str | None = None,
        mcp_session_id: str | None = None,
    ):
        """Return the configured agent for a specific runtime context.

        Args:
            reasoning_level: The reasoning level value.
            model_name: The model name value.
            reasoning_level_is_explicit: Whether reasoning was set for this run.
            thread_id: Conversation thread identifier.
            async_subagent_url_override: The async subagent URL override value.
            mcp_session_id: MCP session identifier.

        Returns:
            The configured agent for a specific runtime context.
        """
        selected_model = (
            str(model_name or self.config.model_name).strip()
            or self.config.model_name
        )
        selected_model_profile = resolve_runtime_model_profile(
            self.config,
            selected_model,
        )
        reasoning_level_is_explicit = (
            self.config.model_reasoning_override
            or reasoning_level_is_explicit
            or reasoning_level != self.config.default_reasoning
        )
        effective_reasoning_level = reasoning_level_for_profile(
            selected_model_profile,
            reasoning_level,
            fallback_is_explicit=reasoning_level_is_explicit,
        )
        mcp_scope = self._mcp_scope(
            mcp_session_id=mcp_session_id,
            thread_id=thread_id,
        )
        cache_key = AgentCacheKey(
            reasoning_level=effective_reasoning_level,
            reasoning_level_is_explicit=reasoning_level_is_explicit,
            model_name=selected_model,
            thread_id=thread_id,
            async_subagent_url_override=async_subagent_url_override,
            mcp_scope=mcp_scope,
        )
        async with self._agent_lock:
            agent = self._agents.get(cache_key)
            if agent is None:
                model = self._build_model(
                    effective_reasoning_level,
                    model_profile=selected_model_profile,
                )
                rag_tool_enabled = self._rag_service is not None
                raw_main_tools = await self._build_main_tools(
                    thread_id=thread_id,
                    mcp_session_id=mcp_session_id,
                )
                main_tools = sanitize_tools_for_model(
                    selected_model_profile.provider,
                    raw_main_tools,
                )
                backend = build_deepagent_backend(
                    project_root=self.project_root,
                    include_memories=self.config.agent_state == "stateful",
                    memory_namespace=self.config.extensions.agent_memory_namespace,
                )
                middleware = build_agent_middleware(
                    backend=backend,
                    config=self.config,
                    reasoning_level=effective_reasoning_level,
                    model_name=selected_model,
                    source="main-agent",
                    project_root=self.project_root,
                )
                subagent_specs = await self._build_runtime_subagent_specs(
                    reasoning_level=effective_reasoning_level,
                    reasoning_level_is_explicit=reasoning_level_is_explicit,
                    selected_model_profile=selected_model_profile,
                    backend=backend,
                    inherited_tools=raw_main_tools,
                    sanitized_inherited_tools=main_tools,
                    thread_id=thread_id,
                    mcp_session_id=mcp_session_id,
                )
                subagent_specs.extend(
                    subagent.to_deepagents_spec(
                        url_override=async_subagent_url_override,
                    )
                    for subagent in self.config.extensions.async_subagents
                )
                agent_kwargs: dict[str, Any] = {
                    "model": model,
                    "tools": main_tools or None,
                    "system_prompt": compose_rag_system_prompt(
                        compose_agent_system_prompt(
                            system_prompt_for_agent_state(
                                SYSTEM_PROMPT,
                                self.config.agent_state,
                            ),
                            self.config.extensions.custom_instruction,
                            project_root=self.project_root,
                        ),
                        rag_enabled=rag_tool_enabled,
                    ),
                    "middleware": middleware,
                    "backend": backend,
                    "skills": list(self.config.extensions.skills) or None,
                    "subagents": subagent_specs or None,
                }
                memory_files = stateful_agent_memory_files(self.config)
                if memory_files is not None:
                    agent_kwargs["memory"] = memory_files
                if self.config.agent_state == "stateful":
                    agent_kwargs["store"] = self.store
                    agent_kwargs["checkpointer"] = self.checkpointer
                agent = create_deep_agent_with_configured_summarization(
                    self.config,
                    **agent_kwargs,
                )
                self._agents[cache_key] = agent
            return agent

    async def rebuild_rag_index(self) -> RagStatus:
        """Rebuild RAG index.

        Returns:
            The rebuilt object or status.
        """
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
        """Ingest RAG uploads.

        Args:
            thread_id: Conversation thread identifier.
            uploads: Uploaded files supplied by the user.

        Returns:
            The ingest RAG uploads result.
        """
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

    async def clone_rag_uploads(
        self,
        *,
        source_thread_id: str,
        target_thread_id: str,
    ) -> RagUploadResult:
        """Clone thread-scoped RAG uploads for a fresh conversation branch."""
        if self._rag_service is None:
            return RagUploadResult(
                thread_id=target_thread_id,
                reason=self.config.rag_error or "Knowledge index is unavailable.",
            )
        return await asyncio.to_thread(
            self._rag_service.clone_thread_uploads,
            source_thread_id=source_thread_id,
            target_thread_id=target_thread_id,
        )

    def resolve_chainlit_command(self, name: str) -> ChainlitCommandConfig | None:
        """Resolve a native Chainlit command by name.

        Args:
            name: The name value.

        Returns:
            The matching command configuration, or None when no command matches.
        """
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
        """Invoke a configured MCP tool command with parsed arguments.

        Args:
            tool_name: Name of the tool to invoke.
            raw_args: Raw argument text supplied with the command.
            thread_id: Conversation thread identifier.
            mcp_session_id: MCP session identifier.
            server_name: The server name value.

        Returns:
            The invoke MCP tool command result.

        Raises:
            ValueError: If the supplied value is invalid.
        """
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
        """Sanitize tools for the active model provider.

        Args:
            tools: The tools value.

        Returns:
            The sanitized value.
        """
        return sanitize_tools_for_model(self.config.model_provider, tools)

    @staticmethod
    def _tool_supports_openai_compatible_schema(tool: Any) -> bool:
        """Return whether a tool supports OpenAI-compatible schemas.

        Args:
            tool: The tool value.

        Returns:
            Whether a tool supports OpenAI-compatible schemas.
        """
        return tool_supports_openai_compatible_schema(tool)

    def _build_model(
        self,
        reasoning_level: ReasoningLevel,
        *,
        model_name: str | None = None,
        model_profile: ModelDefaults | None = None,
    ) -> Any:
        """Build the chat model for the current runtime settings.

        Args:
            reasoning_level: The reasoning level value.
            model_name: The model name value.

        Returns:
            The constructed the chat model for the current runtime settings.
        """
        if model_profile is not None:
            return build_model_for_profile(
                self.config,
                reasoning_level,
                model_profile,
            )
        return build_model(self.config, reasoning_level, model_name=model_name)

    def _mcp_scope(
        self,
        *,
        mcp_session_id: str | None,
        thread_id: str | None = None,
    ) -> str | None:
        """Open or reuse MCP client resources for the current scope.

        Args:
            mcp_session_id: MCP session identifier.
            thread_id: Conversation thread identifier.

        Returns:
            The MCP scope result.
        """
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
        """Return the cached MCP session for a Chainlit session.

        Args:
            server_name: The server name value.
            thread_id: Conversation thread identifier.
            mcp_session_id: MCP session identifier.

        Returns:
            The cached MCP session for a Chainlit session.

        Raises:
            RuntimeError: If the runtime is not in a usable state.
        """
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

        owner = _MCPSessionOwner(self._mcp_client.session(server_name))
        session = await owner.session()
        self._mcp_session_owners[cache_key] = owner
        self._mcp_sessions[cache_key] = session
        return session

    async def _get_mcp_tools(
        self,
        server_names: tuple[str, ...],
        *,
        thread_id: str | None = None,
        mcp_session_id: str | None = None,
    ) -> list[Any]:
        """Load MCP tools for the active runtime context.

        Args:
            server_names: The server names value.
            thread_id: Conversation thread identifier.
            mcp_session_id: MCP session identifier.

        Returns:
            The requested value.
        """
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

            existing_sessions = set(self._mcp_sessions)
            try:
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

            except BaseException:
                owners = []
                for key in set(self._mcp_sessions) - existing_sessions:
                    self._mcp_sessions.pop(key)
                    owners.append(self._mcp_session_owners.pop(key))
                await self._close_mcp_owners(owners)
                raise

            self._mcp_tools_cache[cache_key] = tools
            return list(tools)

    async def _build_main_tools(
        self,
        *,
        thread_id: str | None,
        mcp_session_id: str | None,
    ) -> list[Any]:
        """Build the main agent tool list for a runtime context.

        Args:
            thread_id: Conversation thread identifier.
            mcp_session_id: MCP session identifier.

        Returns:
            The constructed the main agent tool list for a runtime context.
        """
        tools = await self._get_mcp_tools(
            self.config.extensions.agent_mcp_servers,
            thread_id=thread_id,
            mcp_session_id=mcp_session_id,
        )
        tools = list(tools)
        if self.config.extensions.chainlit_generative_ui_enabled:
            tools.append(create_render_chainlit_ui_tool())
        if self._rag_service is not None:
            tools.append(
                create_search_workspace_knowledge_tool(
                    self._rag_service,
                    thread_id=thread_id,
                )
            )
        return tools

    async def _clear_agent_cache(self) -> None:
        """Clear cached agents after runtime tool state changes."""
        async with self._agent_lock:
            self._agents.clear()

    async def close_mcp_session(self, mcp_session_id: str | None) -> None:
        """Close MCP session.

        Args:
            mcp_session_id: MCP session identifier.
        """
        mcp_scope = self._mcp_scope(mcp_session_id=mcp_session_id)
        if mcp_scope is None:
            return

        async with self._agent_lock:
            self._agents = {
                key: agent
                for key, agent in self._agents.items()
                if key.mcp_scope != mcp_scope
            }
            async with self._mcp_lock:
                owners = [
                    self._mcp_session_owners.pop(key)
                    for key in list(self._mcp_session_owners)
                    if key[0] == mcp_scope
                ]
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

        await self._close_mcp_owners(owners)

    async def close_conversation(
        self, *, thread_id: str | None, mcp_session_id: str | None = None
    ) -> None:
        """Release conversation graphs and any stateful MCP transport resources."""
        await self.close_mcp_session(mcp_session_id or thread_id)
        if thread_id:
            async with self._agent_lock:
                self._agents = {
                    key: agent for key, agent in self._agents.items()
                    if key.thread_id != thread_id or key.mcp_scope is not None
                }

    @staticmethod
    async def _close_mcp_owners(owners: list[_MCPSessionOwner]) -> None:
        # Each owner closes independently; one broken transport must not leak others.
        results = await asyncio.gather(
            *(owner.aclose() for owner in owners), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def close_all_mcp_sessions(self) -> None:
        """Close all MCP sessions."""
        async with self._agent_lock:
            async with self._mcp_lock:
                owners = list(self._mcp_session_owners.values())
                self._mcp_session_owners.clear()
                self._mcp_sessions.clear()
                self._mcp_tools_cache.clear()
                self._agents.clear()

        await self._close_mcp_owners(owners)

    async def close(self) -> None:
        """Close the agent runtime."""
        try:
            await self._exit_stack.aclose()
        finally:
            self._checkpointer = None
            self._store = None
            self._mcp_client = None

    def _build_backend(self, runtime):
        """Build the Deep Agent backend for the current runtime settings.

        Args:
            runtime: Agent runtime used by the operation.

        Returns:
            The constructed the deep agent backend for the current runtime settings.
        """
        return build_deepagent_backend(
            project_root=self.project_root,
            include_memories=runtime.config.agent_state == "stateful",
            memory_namespace=runtime.config.extensions.agent_memory_namespace,
        )
