"""Shared runtime defaults, workspace resolution, and prompt constants."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

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


def _resolve_default_project_root(
    *,
    module_file: Path,
    working_directory: Path | None = None,
) -> Path:
    """Resolve the default user workspace for source and installed runtimes.

    Args:
        module_file: Path to this runtime module.
        working_directory: Process working directory used by installed packages.

    Returns:
        The source checkout root, or the user's working directory for an install.
    """
    source_root = module_file.resolve().parents[2]
    if (source_root / "pyproject.toml").is_file() and (
        source_root / "chainagents"
    ).is_dir():
        return source_root
    return (working_directory or Path.cwd()).resolve()


PROJECT_ROOT = _resolve_default_project_root(module_file=Path(__file__))
DEEPAGENT_ARTIFACTS_DIRECTORY = Path(".files/deepagent")
GENERATED_OUTPUTS_DIRECTORY = Path(".files/outputs")
AGENTS_MD_FILENAME = "AGENTS.md"
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
