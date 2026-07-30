"""Run the Chainlit UI for the configured ChainAgents runtime."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import secrets
import traceback
from collections.abc import Iterable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from chainagents.util.langchain_warnings import install_langchain_warning_filters

install_langchain_warning_filters()

import chainlit as cl
from chainlit.input_widget import Select, Switch, TextInput
from chainlit.types import ThreadDict

from chainagents.commands.native import (
    ParsedNativeCommand,
    build_skill_command_prompt,
    dumps_tool_result,
    parse_native_command,
    resolve_native_command,
    resolve_runtime_command,
)
from chainagents.interfaces.chainlit.async_tasks import AsyncTaskNotifier, async_subagent_url_override
from chainagents.interfaces.chainlit.bridge import ChainlitEventBridge, RunTaskList
from chainagents.interfaces.chainlit.persistence import chainlit_data_layer_enabled, create_chainlit_data_layer
from chainagents.runtime import (
    DEFAULT_REASONING_LEVEL,
    AgentRuntime,
    AppSettings,
    ChainlitStarterConfig,
    ReasoningLevel,
    RuntimeConfig,
    build_langgraph_run_config,
    format_model_provider,
    normalize_reasoning_level,
    reasoning_level_for_profile,
    resolve_runtime_model_profile,
)
from chainagents.runtime.reflection import (
    ReflectionCollector,
    ReflectionProposal,
    format_reflection_proposal,
    reflection_save_prompt,
)
from chainagents.rag.runtime import UploadedRagFile
from chainagents.exports.response import (
    DOWNLOAD_MARKDOWN_ACTION,
    DOWNLOAD_PDF_ACTION,
    send_markdown_export,
    send_pdf_export,
)


SESSION_SETTINGS_KEY = "agent_settings"
SESSION_TASK_LIST_KEY = "run_task_list"
SESSION_ASYNC_TASK_NOTIFIER_KEY = "async_task_notifier"
SESSION_MCP_SESSION_ID_KEY = "mcp_session_id"
SESSION_GENERATED_UI_ELEMENTS_KEY = "generated_ui_elements"
REBUILD_RAG_INDEX_ACTION = "rebuild_knowledge_index"
UPLOAD_RAG_FILE_ACTION = "upload_rag_file"
REFLECTION_SAVE_ACTION = "save_reflection_lesson"
REFLECTION_DISMISS_ACTION = "dismiss_reflection_lesson"
RAG_UPLOAD_ACCEPT = {
    "text/plain": [
        ".csv",
        ".json",
        ".log",
        ".md",
        ".py",
        ".rst",
        ".text",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    ],
    "application/json": [".json"],
    "text/markdown": [".md"],
    "application/octet-stream": [
        ".csv",
        ".json",
        ".log",
        ".md",
        ".py",
        ".rst",
        ".text",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    ],
}
IMAGE_UPLOAD_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VISION_IMAGE_MIME_TYPE_BY_EXTENSION = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
VISION_IMAGE_MIME_TYPES = frozenset(VISION_IMAGE_MIME_TYPE_BY_EXTENSION.values())
VISION_IMAGE_MIME_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
}
GENERIC_UPLOAD_MIME_TYPES = {"", "application/octet-stream"}


def load_chainlit_auth_users(
    *,
    raw_users: str | None = None,
    legacy_username: str | None = None,
    legacy_password: str | None = None,
) -> dict[str, str]:
    """Load Chainlit password users from environment-compatible values.

    Args:
        raw_users: JSON object mapping usernames to passwords.
        legacy_username: Single-user username fallback.
        legacy_password: Single-user password fallback.

    Returns:
        A mapping of usernames to passwords.

    Raises:
        ValueError: If CHAINLIT_AUTH_USERS is set but invalid.
    """
    if raw_users is None:
        raw_users = os.getenv("CHAINLIT_AUTH_USERS", "")
    raw_users = raw_users.strip()
    if raw_users:
        try:
            parsed_users = json.loads(raw_users)
        except json.JSONDecodeError as exc:
            raise ValueError("CHAINLIT_AUTH_USERS must contain valid JSON") from exc

        if not isinstance(parsed_users, dict):
            raise ValueError(
                "CHAINLIT_AUTH_USERS must be a JSON object mapping usernames to passwords"
            )

        auth_users: dict[str, str] = {}
        for username, password in parsed_users.items():
            if not isinstance(username, str) or not username.strip():
                raise ValueError(
                    "CHAINLIT_AUTH_USERS usernames must be non-empty strings"
                )
            if not isinstance(password, str) or not password:
                raise ValueError(
                    "CHAINLIT_AUTH_USERS passwords must be non-empty strings"
                )
            auth_users[username] = password

        if not auth_users:
            raise ValueError("CHAINLIT_AUTH_USERS must define at least one user")
        return auth_users

    if legacy_username is None:
        legacy_username = os.getenv("CHAINLIT_AUTH_USERNAME", "")
    if legacy_password is None:
        legacy_password = os.getenv("CHAINLIT_AUTH_PASSWORD", "")

    legacy_username = legacy_username.strip()
    legacy_password = legacy_password.strip()
    if legacy_username and legacy_password:
        return {legacy_username: legacy_password}
    return {}


def authenticate_chainlit_user(
    username: str,
    password: str,
    auth_users: Mapping[str, str] | None = None,
) -> cl.User | None:
    """Authenticate a Chainlit user from configured password settings.

    Args:
        username: The username value.
        password: The password value.
        auth_users: Optional configured users mapping.

    Returns:
        A Chainlit user when credentials match, otherwise None.
    """
    configured_users = AUTH_USERS if auth_users is None else auth_users
    configured_password = configured_users.get(username)
    if configured_password is None:
        return None
    if not secrets.compare_digest(
        password.encode("utf-8"),
        configured_password.encode("utf-8"),
    ):
        return None
    return cl.User(
        identifier=username,
        display_name=username,
        metadata={"provider": "credentials"},
    )


AUTH_USERS = load_chainlit_auth_users()
AUTH_SECRET = os.getenv("CHAINLIT_AUTH_SECRET", "").strip()
AUTH_ENABLED = bool(AUTH_SECRET and AUTH_USERS)


if chainlit_data_layer_enabled():

    @cl.data_layer
    def configured_chainlit_data_layer():
        """Return the Postgres-backed Chainlit data layer with schema bootstrap."""
        return create_chainlit_data_layer()


def build_chainlit_starters(
    starter_configs: Iterable[ChainlitStarterConfig],
) -> list[cl.Starter]:
    """Build Chainlit starter objects from runtime config.

    Args:
        starter_configs: Configured starter definitions.

    Returns:
        Chainlit starter objects.
    """
    return [
        cl.Starter(
            label=starter.label,
            message=starter.message,
            command=starter.command,
            icon=starter.icon,
        )
        for starter in starter_configs
    ]


@cl.set_starters
async def configured_chainlit_starters(
    user: cl.User | None = None,
    language: str | None = None,
) -> list[cl.Starter]:
    """Return configured Chainlit starters.

    Args:
        user: Authenticated Chainlit user, if any.
        language: Active UI language, if any.

    Returns:
        Configured Chainlit starter objects.
    """
    _ = (user, language)
    runtime = AgentRuntime.current()
    extensions = (
        runtime.config.extensions
        if runtime is not None
        else RuntimeConfig.from_env().extensions
    )
    return build_chainlit_starters(extensions.chainlit_starters)


def current_chainlit_thread_id() -> str:
    """Return the current chainlit thread ID.

    Returns:
        The current chainlit thread ID.
    """
    try:
        session = cl.context.session
    except Exception:
        return ""
    thread_id = getattr(session, "thread_id", None) or getattr(session, "id", None)
    return str(thread_id or "").strip()


def current_chainlit_session_id() -> str:
    """Return the current chainlit session ID.

    Returns:
        The current chainlit session ID.
    """
    try:
        session = cl.context.session
    except Exception:
        return ""
    return str(getattr(session, "id", None) or "").strip()


def store_mcp_session_id() -> str:
    """Store MCP session ID.

    Returns:
        The stored value.
    """
    session_id = current_chainlit_session_id() or current_chainlit_thread_id()
    cl.user_session.set(SESSION_MCP_SESSION_ID_KEY, session_id)
    return session_id


def current_mcp_session_id() -> str:
    """Return the current MCP session ID.

    Returns:
        The current MCP session ID.
    """
    session_id = str(cl.user_session.get(SESSION_MCP_SESSION_ID_KEY) or "").strip()
    if session_id:
        return session_id
    return store_mcp_session_id()


def settings_payload(settings: AppSettings) -> dict[str, Any]:
    """Build a serializable payload from Chainlit chat settings.

    Args:
        settings: The settings value.

    Returns:
        The constructed a serializable payload from chainlit chat settings.
    """
    return {
        "model_name": settings.model_name,
        "reasoning_level": settings.reasoning_level,
        "thread_id": settings.thread_id,
        "show_reasoning_stream": settings.show_reasoning_stream,
        "show_tool_calls": settings.show_tool_calls,
    }


def store_settings(settings: AppSettings) -> None:
    """Store settings.

    Args:
        settings: The settings value.
    """
    cl.user_session.set(SESSION_SETTINGS_KEY, settings_payload(settings))


def build_langgraph_config(
    settings: AppSettings,
    *,
    runtime_config: RuntimeConfig,
    project_root: Path,
) -> dict[str, Any]:
    """Build the LangGraph run configuration for a Chainlit thread.

    Args:
        settings: The settings value.
        runtime_config: Resolved runtime config.
        project_root: Active runtime project root.

    Returns:
        A LangGraph configuration dictionary for the thread.
    """
    return build_langgraph_run_config(
        runtime_config,
        thread_id=settings.thread_id,
        project_root=project_root,
    )


def build_rag_action() -> cl.Action:
    """Build RAG action.

    Returns:
        The constructed rag action.
    """
    return cl.Action(
        name=REBUILD_RAG_INDEX_ACTION,
        payload={},
        label="Rebuild Knowledge Index",
        tooltip="Rebuild the local documentation RAG index.",
        icon="refresh-cw",
    )


def build_upload_rag_action() -> cl.Action:
    """Build upload RAG action.

    Returns:
        The constructed upload rag action.
    """
    return cl.Action(
        name=UPLOAD_RAG_FILE_ACTION,
        payload={},
        label="Upload File For RAG",
        tooltip="Upload a text file and add it to this chat thread's knowledge index.",
        icon="paperclip",
    )


def rag_actions() -> list[cl.Action]:
    """Return Chainlit action buttons for RAG workflows.

    Returns:
        Chainlit action buttons for RAG workflows.
    """
    return [build_rag_action(), build_upload_rag_action()]


def reflection_actions() -> list[cl.Action]:
    """Return Chainlit actions for reflection confirmation."""
    return [
        cl.Action(
            name=REFLECTION_SAVE_ACTION,
            payload={"value": "save"},
            label="Save lesson",
            tooltip="Ask the agent to save this lesson into long-term memory.",
            icon="save",
        ),
        cl.Action(
            name=REFLECTION_DISMISS_ACTION,
            payload={"value": "dismiss"},
            label="Dismiss",
            tooltip="Do not save this reflection lesson.",
            icon="x",
        ),
    ]


async def ask_to_save_reflection_lesson(
    *,
    runtime: AgentRuntime,
    settings: AppSettings,
    proposal: ReflectionProposal,
    reasoning_level: ReasoningLevel,
    model_name: str,
    async_url_override: str | None,
    mcp_session_id: str | None,
) -> None:
    """Ask the Chainlit user whether to save a reflection lesson."""
    response = await cl.AskActionMessage(
        content=format_reflection_proposal(proposal),
        actions=reflection_actions(),
        author="System",
        timeout=90,
        raise_on_timeout=False,
    ).send()
    if not response:
        return
    payload = response.get("payload") if isinstance(response, dict) else None
    if not isinstance(payload, dict) or payload.get("value") != "save":
        return
    await save_reflection_lesson(
        runtime=runtime,
        settings=settings,
        proposal=proposal,
        reasoning_level=reasoning_level,
        model_name=model_name,
        async_url_override=async_url_override,
        mcp_session_id=mcp_session_id,
    )


async def save_reflection_lesson(
    *,
    runtime: AgentRuntime,
    settings: AppSettings,
    proposal: ReflectionProposal,
    reasoning_level: ReasoningLevel,
    model_name: str,
    async_url_override: str | None,
    mcp_session_id: str | None,
) -> None:
    """Ask the configured agent to save a confirmed reflection lesson."""
    reflection_thread_id = f"{settings.thread_id}:reflection"
    agent = await runtime.get_agent(
        reasoning_level,
        model_name=model_name,
        thread_id=reflection_thread_id,
        async_subagent_url_override=async_url_override,
        mcp_session_id=mcp_session_id,
    )
    payload = {
        "messages": [
            {
                "role": "user",
                "content": reflection_save_prompt(proposal),
            }
        ]
    }
    config = build_langgraph_run_config(
        runtime.config,
        thread_id=reflection_thread_id,
        project_root=runtime.project_root,
    )
    await agent.ainvoke(payload, config=config)
    await cl.Message(
        content=f"Saved lesson to `{proposal.memory_file}`.",
        author="System",
    ).send()


def build_native_command_specs(runtime: AgentRuntime) -> list[dict[str, Any]]:
    """Build native command specs.

    Args:
        runtime: Agent runtime used by the operation.

    Returns:
        The constructed native command specs.
    """
    icon_by_target = {
        "prompt": "square-pen",
        "subagent": "bot",
        "mcp_tool": "wrench",
        "skill": "book-open",
    }
    return [
        {
            "id": command.name,
            "description": command.description,
            "icon": icon_by_target.get(command.target, "terminal"),
            "button": False,
            "persistent": True,
        }
        for command in runtime.chainlit_commands
    ]


async def publish_native_commands(runtime: AgentRuntime) -> None:
    """Publish native commands.

    Args:
        runtime: Agent runtime used by the operation.
    """
    await cl.context.emitter.set_commands(build_native_command_specs(runtime))


def rag_status_line(runtime: AgentRuntime) -> str:
    """Render one status line for the RAG service.

    Args:
        runtime: Agent runtime used by the operation.

    Returns:
        The RAG status line result.
    """
    status = runtime.rag_status
    if not status.enabled:
        return "- RAG: disabled\n"
    if status.ready:
        return (
            f"- RAG: ready (`{status.file_count}` files, "
            f"`{status.chunk_count}` chunks)\n"
        )

    reason = (status.reason or "unknown error").strip()
    if len(reason) > 160:
        reason = f"{reason[:157].rstrip()}..."
    return f"- RAG: unavailable; {reason}\n"


def message_uploads(message: cl.Message) -> list[tuple[Path, str, str]]:
    """Return readable files attached to a Chainlit message.

    Args:
        message: Chainlit message or LangChain message to process.

    Returns:
        Tuples of path, display name, and MIME type for attached files.
    """
    uploads: list[tuple[Path, str, str]] = []
    for element in getattr(message, "elements", []) or []:
        raw_path = getattr(element, "path", None)
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.exists() or not path.is_file():
            continue
        name = str(getattr(element, "name", "") or path.name).strip() or path.name
        mime_type = uploaded_file_mime_type(element, path=path, name=name)
        uploads.append((path, name, mime_type))
    return uploads


def uploaded_file_mime_type(element: Any, *, path: Path, name: str) -> str:
    """Resolve the MIME type for an uploaded Chainlit file."""
    for attr in ("mime", "mime_type", "content_type"):
        raw_mime = getattr(element, attr, None)
        if isinstance(raw_mime, str) and "/" in raw_mime:
            return raw_mime.split(";", 1)[0].strip().lower()

    for candidate in (name, path.name):
        guessed_type, _ = mimetypes.guess_type(candidate)
        if guessed_type:
            return guessed_type.lower()
    return ""


def is_image_upload(path: Path, mime_type: str) -> bool:
    """Return whether an uploaded file should be sent as an image attachment."""
    if mime_type.startswith("image/"):
        return True
    return path.suffix.lower() in IMAGE_UPLOAD_EXTENSIONS


def provider_safe_image_mime_type(path: Path, mime_type: str) -> str | None:
    """Return a vision-provider-safe MIME type for an uploaded image."""
    normalized_mime = VISION_IMAGE_MIME_ALIASES.get(mime_type, mime_type)
    if normalized_mime in VISION_IMAGE_MIME_TYPES:
        return normalized_mime

    inferred_mime = VISION_IMAGE_MIME_TYPE_BY_EXTENSION.get(path.suffix.lower())
    if inferred_mime and normalized_mime in GENERIC_UPLOAD_MIME_TYPES:
        return inferred_mime
    return None


def message_uploaded_rag_files(message: cl.Message) -> list[UploadedRagFile]:
    """Build the message for uploaded RAG files.

    Args:
        message: Chainlit message or LangChain message to process.

    Returns:
        The constructed the message for uploaded rag files.
    """
    uploads: list[UploadedRagFile] = []
    for path, name, mime_type in message_uploads(message):
        if is_image_upload(path, mime_type):
            continue
        uploads.append(UploadedRagFile(path=path, name=name))
    return uploads


def message_uploaded_image_names(message: cl.Message) -> tuple[str, ...]:
    """Return names for image files attached to a Chainlit message."""
    return tuple(
        name
        for path, name, mime_type in message_uploads(message)
        if provider_safe_image_mime_type(path, mime_type) is not None
    )


def unsupported_uploaded_image_names(message: cl.Message) -> tuple[str, ...]:
    """Return names for image files that cannot be sent to vision providers."""
    return tuple(
        name
        for path, name, mime_type in message_uploads(message)
        if is_image_upload(path, mime_type)
        and provider_safe_image_mime_type(path, mime_type) is None
    )


def message_uploaded_image_parts(message: cl.Message) -> list[dict[str, Any]]:
    """Build multimodal content parts for uploaded Chainlit images.

    Args:
        message: Chainlit message or LangChain message to process.

    Returns:
        OpenAI-compatible image content parts backed by data URLs.
    """
    parts: list[dict[str, Any]] = []
    for path, _name, mime_type in message_uploads(message):
        image_mime_type = provider_safe_image_mime_type(path, mime_type)
        if image_mime_type is None:
            continue
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            continue
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime_type};base64,{encoded}"},
            }
        )
    return parts


def chainlit_prompt_text(
    content: str,
    *,
    image_names: tuple[str, ...],
    prompt_note: str,
) -> str:
    """Build the text part of a Chainlit user message sent to the agent."""
    if not image_names:
        return f"{content}{prompt_note}"

    prompt = content.strip()
    if not prompt and image_names:
        prompt = "Extract any visible text from the attached image(s)."

    attached = ", ".join(f"`{name}`" for name in image_names)
    prompt = (
        f"{prompt}\n\n"
        f"Attached image file(s): {attached}. "
        "Use the image content directly when answering."
    )

    return f"{prompt}{prompt_note}"


def chainlit_user_message_content(
    prompt: str,
    *,
    image_parts: list[dict[str, Any]],
) -> str | list[dict[str, Any]]:
    """Build the multimodal user message content sent from Chainlit."""
    if not image_parts:
        return prompt
    return [{"type": "text", "text": prompt}, *image_parts]


def unsupported_uploaded_images_message(image_names: tuple[str, ...]) -> str:
    """Build a user-facing note for unsupported image uploads."""
    names = ", ".join(f"`{name}`" for name in image_names)
    return (
        "Some uploaded images were not attached to the agent request because their "
        "formats are not supported by the configured vision providers.\n\n"
        f"- Unsupported: {names}\n"
        "- Supported image formats: PNG, JPEG, WEBP, GIF"
    )


def upload_result_prompt_note(added_files: tuple[str, ...]) -> str:
    """Build the prompt note describing uploaded RAG files.

    Args:
        added_files: The added files value.

    Returns:
        The constructed the prompt note describing uploaded rag files.
    """
    if not added_files:
        return ""
    file_list = ", ".join(f"`{name}`" for name in added_files)
    return (
        "\n\nUploaded files are available in this thread's knowledge index: "
        f"{file_list}. Use `search_workspace_knowledge` if the user refers to them."
    )


def upload_result_message(upload_result) -> str:
    """Build the Chainlit message for a RAG upload result.

    Args:
        upload_result: The upload result value.

    Returns:
        The constructed the chainlit message for a rag upload result.
    """
    if upload_result.added_files:
        added = ", ".join(f"`{name}`" for name in upload_result.added_files)
        content = (
            "Uploaded file(s) added to this thread's knowledge index.\n\n"
            f"- Added: {added}\n"
            f"- Uploaded files indexed for this thread: `{upload_result.indexed_files}`\n"
            f"- Uploaded chunks indexed for this thread: `{upload_result.chunk_count}`"
        )
        if upload_result.rejected_files:
            rejected = ", ".join(f"`{name}`" for name in upload_result.rejected_files)
            content += f"\n- Rejected: {rejected}"
        return content

    if upload_result.rejected_files:
        rejected = ", ".join(f"`{name}`" for name in upload_result.rejected_files)
        return f"No supported text files were added to RAG. Rejected: {rejected}"

    return upload_result.reason or "No files were added to RAG."

async def handle_native_command(
    *,
    runtime: AgentRuntime,
    settings: AppSettings,
    parsed: ParsedNativeCommand,
    mcp_session_id: str | None = None,
) -> str | None:
    """Handle a native slash command selected in Chainlit.

    Args:
        runtime: Agent runtime used by the operation.
        settings: The settings value.
        parsed: Parsed native command details.
        mcp_session_id: MCP session identifier.

    Returns:
        The handle native command result.
    """
    result = await resolve_runtime_command(
        runtime=runtime,
        parsed=parsed,
        thread_id=settings.thread_id,
        mcp_session_id=mcp_session_id,
    )
    if result.target == "unknown":
        return None

    if result.target == "mcp_tool":
        await cl.Message(
            author="System",
            content=(
                f"Ran `/{result.command_name}` ({result.description}).\n\n"
                "Tool result:\n```json\n"
                f"{dumps_tool_result(result.tool_result)}\n"
                "```"
            ),
        ).send()
        return ""

    return result.prompt


async def ask_for_rag_upload() -> list[UploadedRagFile]:
    """Ask for for RAG upload.

    Returns:
        The prompt or response used to ask the user.
    """
    files = await cl.AskFileMessage(
        content=(
            "Upload text-based files for this chat thread's knowledge index.\n\n"
            "Accepted examples: `.md`, `.txt`, `.rst`, `.json`, `.toml`, `.yaml`, `.yml`, `.csv`, `.log`, `.py`."
        ),
        accept=RAG_UPLOAD_ACCEPT,
        max_size_mb=25,
        max_files=5,
        timeout=300,
        raise_on_timeout=False,
    ).send()
    if not files:
        return []
    return [
        UploadedRagFile(path=Path(file.path), name=file.name)
        for file in files
        if Path(file.path).exists()
    ]


def resolve_model_name(
    value: Any | None,
    *,
    available_models: tuple[str, ...],
    default: str,
) -> str:
    """Resolve model name.

    Args:
        value: Value to normalize, convert, or serialize.
        available_models: The available models value.
        default: Fallback value used when no explicit value is available.

    Returns:
        The resolved model name.
    """
    candidate = str(value or "").strip()
    if candidate in available_models:
        return candidate
    return default


def coerce_bool_setting(value: Any | None, *, default: bool) -> bool:
    """Coerce a raw Chainlit setting value into a boolean."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    candidate = str(value).strip().lower()
    if candidate in {"1", "true", "yes", "on", "enabled"}:
        return True
    if candidate in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def build_chat_settings(
    settings: AppSettings,
    *,
    available_models: tuple[str, ...],
    model_mode_enabled: bool = True,
) -> cl.ChatSettings:
    """Build chat settings.

    Args:
        settings: The settings value.
        available_models: The available models value.
        model_mode_enabled: The model mode enabled value.

    Returns:
        The constructed chat settings.
    """
    reasoning_levels = ["low", "medium", "high"]
    inputs: list[Any] = []
    if model_mode_enabled:
        inputs.append(
            Select(
                id="model_name",
                label="Model",
                values=list(available_models),
                initial_index=available_models.index(settings.model_name),
                description="Select a configured model for this chat session.",
            )
        )
    inputs.extend(
        [
            Select(
                id="reasoning_level",
                label="Reasoning Level",
                values=reasoning_levels,
                initial_index=reasoning_levels.index(settings.reasoning_level),
                description=(
                    "Controls the configured model's reasoning setting when the active "
                    "provider supports it."
                ),
            ),
            TextInput(
                id="thread_id",
                label="LangGraph Thread ID",
                initial=settings.thread_id,
                description=(
                    "Defaults to the current Chainlit thread. Override it only if you want "
                    "to point this chat at a different persisted LangGraph thread."
                ),
            ),
            Switch(
                id="show_reasoning_stream",
                label="Show Reasoning Stream",
                initial=settings.show_reasoning_stream,
                description="Show streamed reasoning panels and reasoning task entries.",
            ),
            Switch(
                id="show_tool_calls",
                label="Show Tool Calls",
                initial=settings.show_tool_calls,
                description="Show streamed tool-call panels and tool task entries.",
            ),
        ]
    )
    return cl.ChatSettings(inputs)


def build_modes(
    settings: AppSettings,
    *,
    available_models: tuple[str, ...],
    model_mode_enabled: bool = True,
    reasoning_mode_enabled: bool = True,
) -> list[cl.Mode]:
    """Build modes.

    Args:
        settings: The settings value.
        available_models: The available models value.
        model_mode_enabled: The model mode enabled value.
        reasoning_mode_enabled: The reasoning mode enabled value.

    Returns:
        The constructed modes.
    """
    reasoning_levels = ["low", "medium", "high"]
    modes: list[cl.Mode] = []
    if model_mode_enabled:
        modes.append(
            cl.Mode(
                id="model_name",
                name="Model",
                options=[
                    cl.ModeOption(
                        id=model_name,
                        name=model_name,
                        description="Use this model for the current message.",
                        icon="bot",
                        default=model_name == settings.model_name,
                    )
                    for model_name in available_models
                ],
            )
        )
    if reasoning_mode_enabled:
        modes.append(
            cl.Mode(
                id="reasoning_level",
                name="Reasoning",
                options=[
                    cl.ModeOption(
                        id=level,
                        name=level.capitalize(),
                        description=(
                            "Deeper reasoning with higher latency"
                            if level == "high"
                            else (
                                "Balanced quality and speed"
                                if level == "medium"
                                else "Fastest responses with lighter reasoning"
                            )
                        ),
                        icon=(
                            "brain"
                            if level == "high"
                            else ("sparkles" if level == "medium" else "zap")
                        ),
                        default=level == settings.reasoning_level,
                    )
                    for level in reasoning_levels
                ],
            )
        )
    return modes


async def publish_modes(
    settings: AppSettings,
    *,
    available_models: tuple[str, ...],
    model_mode_enabled: bool = True,
    reasoning_mode_enabled: bool = True,
) -> None:
    """Publish modes.

    Args:
        settings: The settings value.
        available_models: The available models value.
        model_mode_enabled: The model mode enabled value.
        reasoning_mode_enabled: The reasoning mode enabled value.
    """
    try:
        await cl.context.emitter.set_modes(
            build_modes(
                settings,
                available_models=available_models,
                model_mode_enabled=model_mode_enabled,
                reasoning_mode_enabled=reasoning_mode_enabled,
            )
        )
    except Exception as exc:
        message = str(exc).lower()
        missing_modes_column = "modes" in message and (
            ("column" in message and "does not exist" in message)
            or "no such column" in message
        )
        if not missing_modes_column:
            raise


def coerce_settings(
    raw_settings: AppSettings | dict[str, Any] | None,
    *,
    default_model_name: str,
    available_models: tuple[str, ...],
    default_reasoning_level: ReasoningLevel = DEFAULT_REASONING_LEVEL,
    runtime_config: RuntimeConfig | None = None,
    show_reasoning_stream_default: bool = True,
    show_tool_calls_default: bool = True,
) -> AppSettings:
    """Coerce settings.

    Args:
        raw_settings: Raw settings to process.
        default_model_name: The default model name value.
        available_models: The available models value.
        default_reasoning_level: Reasoning default when settings omit it.
        runtime_config: Runtime config used to derive profile-aware defaults.
        show_reasoning_stream_default: Default reasoning stream visibility.
        show_tool_calls_default: Default tool-call visibility.

    Returns:
        The coerced value.
    """
    if raw_settings is None:
        raw_settings = {}
    if isinstance(raw_settings, AppSettings):
        return AppSettings(
            model_name=resolve_model_name(
                raw_settings.model_name,
                available_models=available_models,
                default=default_model_name,
            ),
            reasoning_level=normalize_reasoning_level(raw_settings.reasoning_level),
            thread_id=raw_settings.thread_id,
            show_reasoning_stream=coerce_bool_setting(
                getattr(raw_settings, "show_reasoning_stream", None),
                default=show_reasoning_stream_default,
            ),
            show_tool_calls=coerce_bool_setting(
                getattr(raw_settings, "show_tool_calls", None),
                default=show_tool_calls_default,
            ),
        )
    model_name = resolve_model_name(
        raw_settings.get("model_name"),
        available_models=available_models,
        default=default_model_name,
    )
    reasoning_default = (
        default_reasoning_level_for_model(runtime_config, model_name)
        if runtime_config is not None
        else default_reasoning_level
    )
    reasoning_level = normalize_reasoning_level(
        raw_settings.get("reasoning_level", reasoning_default),
        default=reasoning_default,
    )
    thread_id = str(
        raw_settings.get("thread_id") or current_chainlit_thread_id()
    ).strip()
    if not thread_id:
        thread_id = current_chainlit_thread_id()
    return AppSettings(
        model_name=model_name,
        reasoning_level=reasoning_level,
        thread_id=thread_id.strip(),
        show_reasoning_stream=coerce_bool_setting(
            raw_settings.get("show_reasoning_stream"),
            default=show_reasoning_stream_default,
        ),
        show_tool_calls=coerce_bool_setting(
            raw_settings.get("show_tool_calls"),
            default=show_tool_calls_default,
        ),
    )


def resolve_reasoning_level_for_message(
    message: cl.Message,
    settings: AppSettings,
    *,
    reasoning_mode_enabled: bool = True,
) -> str:
    """Resolve reasoning level for message.

    Args:
        message: Chainlit message or LangChain message to process.
        settings: The settings value.
        reasoning_mode_enabled: The reasoning mode enabled value.

    Returns:
        The resolved reasoning level for message.
    """
    if not reasoning_mode_enabled:
        return settings.reasoning_level
    raw_modes = getattr(message, "modes", None)
    if not isinstance(raw_modes, dict):
        return settings.reasoning_level
    return normalize_reasoning_level(
        raw_modes.get("reasoning_level"),
        default=settings.reasoning_level,
    )


def message_has_reasoning_level_override(
    message: cl.Message,
    *,
    reasoning_mode_enabled: bool = True,
) -> bool:
    """Return whether a message explicitly selected a reasoning level."""
    if not reasoning_mode_enabled:
        return False
    raw_modes = getattr(message, "modes", None)
    return isinstance(raw_modes, dict) and raw_modes.get("reasoning_level") is not None


def default_reasoning_level_for_model(
    config: RuntimeConfig,
    model_name: str | None,
) -> ReasoningLevel:
    """Return the profile-aware reasoning default for a Chainlit model choice."""
    model_profile = resolve_runtime_model_profile(config, model_name)
    return reasoning_level_for_profile(
        model_profile,
        config.default_reasoning,
        fallback_is_explicit=config.model_reasoning_override,
    )


def settings_reasoning_level_is_explicit(
    config: RuntimeConfig,
    settings: AppSettings,
    model_name: str | None = None,
) -> bool:
    """Return whether chat settings override the selected model's reasoning default."""
    selected_model = model_name if model_name is not None else settings.model_name
    return (
        normalize_reasoning_level(settings.reasoning_level)
        != default_reasoning_level_for_model(config, selected_model)
    )


def resolve_model_name_for_message(
    message: cl.Message,
    settings: AppSettings,
    *,
    available_models: tuple[str, ...],
    model_mode_enabled: bool = True,
) -> str:
    """Resolve model name for message.

    Args:
        message: Chainlit message or LangChain message to process.
        settings: The settings value.
        available_models: The available models value.
        model_mode_enabled: The model mode enabled value.

    Returns:
        The resolved model name for message.
    """
    if not model_mode_enabled:
        return settings.model_name
    raw_modes = getattr(message, "modes", None)
    if not isinstance(raw_modes, dict):
        return settings.model_name
    return resolve_model_name(
        raw_modes.get("model_name"),
        available_models=available_models,
        default=settings.model_name,
    )


if AUTH_ENABLED:

    @cl.password_auth_callback
    def password_auth_callback(username: str, password: str) -> cl.User | None:
        """Authenticate a Chainlit user from configured password settings.

        Args:
            username: The username value.
            password: The password value.

        Returns:
            The password auth callback result.
        """
        return authenticate_chainlit_user(username, password)


async def get_runtime_or_notify() -> AgentRuntime | None:
    """Return the runtime or notify the user that startup failed.

    Returns:
        The runtime or notify the user that startup failed.
    """
    try:
        return await AgentRuntime.get()
    except Exception as exc:
        await cl.Message(content=f"Startup error: {exc}", author="System").send()
        return None


async def get_run_task_list(
    *,
    reasoning_steps_enabled: bool = True,
    tool_steps_enabled: bool = True,
) -> RunTaskList:
    """Return the per-session Chainlit run task list.

    Returns:
        The per-session Chainlit run task list.
    """
    run_task_list = cl.user_session.get(SESSION_TASK_LIST_KEY)
    if isinstance(run_task_list, RunTaskList):
        run_task_list.configure(
            reasoning_steps_enabled=reasoning_steps_enabled,
            tool_steps_enabled=tool_steps_enabled,
        )
        return run_task_list

    run_task_list = await RunTaskList.create(
        reasoning_steps_enabled=reasoning_steps_enabled,
        tool_steps_enabled=tool_steps_enabled,
    )
    cl.user_session.set(SESSION_TASK_LIST_KEY, run_task_list)
    return run_task_list


def get_generated_ui_elements() -> dict[str, cl.CustomElement]:
    """Return the per-session generated UI element registry."""
    generated_ui_elements = cl.user_session.get(SESSION_GENERATED_UI_ELEMENTS_KEY)
    if isinstance(generated_ui_elements, dict):
        return generated_ui_elements

    generated_ui_elements = {}
    cl.user_session.set(SESSION_GENERATED_UI_ELEMENTS_KEY, generated_ui_elements)
    return generated_ui_elements


def get_async_task_notifier(
    *,
    agent: Any,
    runtime: AgentRuntime,
    url_override: str | None,
) -> AsyncTaskNotifier | None:
    """Return the per-session async task notifier.

    Args:
        agent: Agent or runtime object used for the operation.
        runtime: Agent runtime used by the operation.
        url_override: Agent Protocol URL override, if one is configured.

    Returns:
        The per-session async task notifier.
    """
    if not runtime.config.extensions.async_subagents:
        return None

    notifier = cl.user_session.get(SESSION_ASYNC_TASK_NOTIFIER_KEY)
    if (
        isinstance(notifier, AsyncTaskNotifier)
        and notifier.matches(agent=agent, url_override=url_override)
    ):
        return notifier

    if isinstance(notifier, AsyncTaskNotifier):
        notifier.cancel()

    notifier = AsyncTaskNotifier(
        agent=agent,
        async_subagents=runtime.config.extensions.async_subagents,
        url_override=url_override,
    )
    cl.user_session.set(SESSION_ASYNC_TASK_NOTIFIER_KEY, notifier)
    return notifier


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize Chainlit session state when a chat starts."""
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    store_mcp_session_id()
    await publish_native_commands(runtime)
    extensions = runtime.config.extensions
    settings = AppSettings(
        model_name=runtime.config.model_name,
        reasoning_level=default_reasoning_level_for_model(
            runtime.config,
            runtime.config.model_name,
        ),
        thread_id=current_chainlit_thread_id(),
        show_reasoning_stream=extensions.chainlit_reasoning_steps_enabled,
        show_tool_calls=extensions.chainlit_tool_steps_enabled,
    )
    run_task_list = await get_run_task_list(
        reasoning_steps_enabled=settings.show_reasoning_stream,
        tool_steps_enabled=settings.show_tool_calls,
    )
    await run_task_list.show_ready()
    store_settings(settings)
    await publish_modes(
        settings,
        available_models=runtime.config.model_choices,
        model_mode_enabled=runtime.config.extensions.chainlit_model_mode_enabled,
        reasoning_mode_enabled=runtime.config.extensions.chainlit_reasoning_mode_enabled,
    )
    await build_chat_settings(
        settings,
        available_models=runtime.config.model_choices,
        model_mode_enabled=runtime.config.extensions.chainlit_model_mode_enabled,
    ).send()
    persistence_line = (
        "- Persistence: Postgres-backed LangGraph checkpoints and `/memories/`\n"
        if runtime.persistence_enabled
        else "- Persistence: in-memory only for this process; set `DATABASE_URL` to enable durable checkpoints and `/memories/`\n"
    )
    history_line = (
        "- History bar: enabled for authenticated users\n"
        if runtime.persistence_enabled and AUTH_ENABLED
        else (
            "- History bar: disabled; set `DATABASE_URL`, `CHAINLIT_AUTH_SECRET`, "
            "and `CHAINLIT_AUTH_USERS` (or legacy `CHAINLIT_AUTH_USERNAME` / "
            "`CHAINLIT_AUTH_PASSWORD`) to enable native Chainlit history\n"
        )
    )
    configured_command_count = len(extensions.chainlit_commands)
    skill_command_count = sum(
        1 for command in runtime.chainlit_commands if command.target == "skill"
    )
    mcp_session_mode_line = (
        "- MCP session mode: stateful for this Chainlit session; cleaned up when the session ends\n"
        if extensions.mcp_stateful
        else "- MCP session mode: stateless; a new MCP session is created for each tool call\n"
    )
    extensions_line = (
        f"- Skill sources: `{len(extensions.skills)}`\n"
        f"- MCP servers: `{len(extensions.mcp_servers or {})}`\n"
        f"{mcp_session_mode_line}"
        f"- Custom subagents: `{len(extensions.subagents)}`\n"
        f"- Async subagents: `{len(extensions.async_subagents)}`\n"
        f"- Configured commands: `{configured_command_count}`\n"
        f"- Skill-backed commands: `{skill_command_count}`\n"
        f"- Native commands: `{len(runtime.chainlit_commands)}`\n"
        f"- Starters: `{len(extensions.chainlit_starters)}`\n"
    )
    if runtime.chainlit_commands:
        command_lines = "\n".join(
            f"  - `/{command.name}` ({command.target}): {command.description}"
            for command in runtime.chainlit_commands
        )
        extensions_line += f"Available commands:\n{command_lines}\n"
    if runtime.chainlit_command_notes:
        note_lines = "\n".join(f"  - {note}" for note in runtime.chainlit_command_notes)
        extensions_line += f"Command notes:\n{note_lines}\n"
    if extensions.config_path is not None:
        extensions_line += f"- Extensions config: `{extensions.config_path.name}`\n"
    if runtime.config.extensions.chainlit_startup_status_enabled:
        startup_model_profile = resolve_runtime_model_profile(runtime.config)
        startup_message = cl.Message(
            content=(
                "Workspace agent ready.\n\n"
                f"- Model provider: `{format_model_provider(startup_model_profile.provider)}`\n"
                f"- Model: `{runtime.config.model_name}`\n"
                f"- Thread ID: `{settings.thread_id}`\n"
                f"{persistence_line}"
                f"{history_line}"
                f"{rag_status_line(runtime)}"
                f"{extensions_line}"
                "- Real repo files live under `/workspace/`\n"
                "- Agent memory is available under `/memories/`"
            ),
            author="System",
        )
        if runtime.rag_enabled:
            startup_message.actions = rag_actions()
        await startup_message.send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    """Restore Chainlit session state when a chat resumes.

    Args:
        thread: The thread value.
    """
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    mcp_session_id = store_mcp_session_id()
    await publish_native_commands(runtime)

    extensions = runtime.config.extensions
    metadata = thread.get("metadata") or {}
    raw_settings = (
        metadata.get(SESSION_SETTINGS_KEY) if isinstance(metadata, dict) else None
    )
    settings = coerce_settings(
        raw_settings,
        default_model_name=runtime.config.model_name,
        available_models=runtime.config.model_choices,
        runtime_config=runtime.config,
        show_reasoning_stream_default=extensions.chainlit_reasoning_steps_enabled,
        show_tool_calls_default=extensions.chainlit_tool_steps_enabled,
    )
    run_task_list = await get_run_task_list(
        reasoning_steps_enabled=settings.show_reasoning_stream,
        tool_steps_enabled=settings.show_tool_calls,
    )
    await run_task_list.show_ready()
    store_settings(settings)
    await publish_modes(
        settings,
        available_models=runtime.config.model_choices,
        model_mode_enabled=runtime.config.extensions.chainlit_model_mode_enabled,
        reasoning_mode_enabled=runtime.config.extensions.chainlit_reasoning_mode_enabled,
    )
    await build_chat_settings(
        settings,
        available_models=runtime.config.model_choices,
        model_mode_enabled=runtime.config.extensions.chainlit_model_mode_enabled,
    ).send()
    async_url_override = async_subagent_url_override()
    agent = await runtime.get_agent(
        settings.reasoning_level,
        model_name=settings.model_name,
        reasoning_level_is_explicit=settings_reasoning_level_is_explicit(
            runtime.config,
            settings,
            settings.model_name,
        ),
        thread_id=settings.thread_id,
        async_subagent_url_override=async_url_override,
        mcp_session_id=mcp_session_id,
    )
    async_task_notifier = get_async_task_notifier(
        agent=agent,
        runtime=runtime,
        url_override=async_url_override,
    )
    if async_task_notifier is not None:
        with suppress(Exception):
            await async_task_notifier.schedule_from_state(thread_id=settings.thread_id)


@cl.on_settings_update
async def on_settings_update(raw_settings: dict[str, Any]) -> None:
    """Persist updated Chainlit chat settings for the current session.

    Args:
        raw_settings: Raw settings to process.
    """
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    settings = coerce_settings(
        raw_settings,
        default_model_name=runtime.config.model_name,
        available_models=runtime.config.model_choices,
        runtime_config=runtime.config,
        show_reasoning_stream_default=(
            runtime.config.extensions.chainlit_reasoning_steps_enabled
        ),
        show_tool_calls_default=runtime.config.extensions.chainlit_tool_steps_enabled,
    )
    store_settings(settings)
    run_task_list = cl.user_session.get(SESSION_TASK_LIST_KEY)
    if isinstance(run_task_list, RunTaskList):
        run_task_list.configure(
            reasoning_steps_enabled=settings.show_reasoning_stream,
            tool_steps_enabled=settings.show_tool_calls,
        )
    await publish_modes(
        settings,
        available_models=runtime.config.model_choices,
        model_mode_enabled=runtime.config.extensions.chainlit_model_mode_enabled,
        reasoning_mode_enabled=runtime.config.extensions.chainlit_reasoning_mode_enabled,
    )


@cl.action_callback(DOWNLOAD_MARKDOWN_ACTION)
async def download_response_markdown(action: cl.Action) -> None:
    """Download response markdown.

    Args:
        action: The action value.
    """
    await send_markdown_export(action)


@cl.action_callback(DOWNLOAD_PDF_ACTION)
async def download_response_pdf(action: cl.Action) -> None:
    """Download response PDF.

    Args:
        action: The action value.
    """
    await send_pdf_export(action)


@cl.action_callback(REBUILD_RAG_INDEX_ACTION)
async def rebuild_knowledge_index(action: cl.Action) -> None:
    """Rebuild knowledge index.

    Args:
        action: The action value.
    """
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return

    status = await runtime.rebuild_rag_index()
    if status.ready:
        content = (
            "Knowledge index rebuilt.\n\n"
            f"- Files indexed: `{status.file_count}`\n"
            f"- Chunks indexed: `{status.chunk_count}`"
        )
    elif status.enabled:
        content = f"Knowledge index rebuild failed: {status.reason or 'unknown error'}"
    else:
        content = "RAG is currently disabled in `deepagent.toml`."

    message = cl.Message(content=content, author="System")
    if runtime.rag_enabled:
        message.actions = rag_actions()
    await message.send()


@cl.action_callback(UPLOAD_RAG_FILE_ACTION)
async def upload_rag_file(action: cl.Action) -> None:
    """Ingest one uploaded file into thread-scoped RAG storage.

    Args:
        action: The action value.
    """
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return

    settings = coerce_settings(
        cl.user_session.get(SESSION_SETTINGS_KEY),
        default_model_name=runtime.config.model_name,
        available_models=runtime.config.model_choices,
        runtime_config=runtime.config,
        show_reasoning_stream_default=(
            runtime.config.extensions.chainlit_reasoning_steps_enabled
        ),
        show_tool_calls_default=runtime.config.extensions.chainlit_tool_steps_enabled,
    )
    uploads = await ask_for_rag_upload()
    if not uploads:
        await cl.Message(content="No files were uploaded.", author="System").send()
        return

    upload_result = await runtime.ingest_rag_uploads(
        thread_id=settings.thread_id,
        uploads=uploads,
    )
    message = cl.Message(
        content=upload_result_message(upload_result),
        author="System",
    )
    if runtime.rag_enabled:
        message.actions = rag_actions()
    await message.send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle a Chainlit user message by streaming the agent response.

    Args:
        message: Chainlit message or LangChain message to process.
    """
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    settings = coerce_settings(
        cl.user_session.get(SESSION_SETTINGS_KEY),
        default_model_name=runtime.config.model_name,
        available_models=runtime.config.model_choices,
        show_reasoning_stream_default=(
            runtime.config.extensions.chainlit_reasoning_steps_enabled
        ),
        show_tool_calls_default=runtime.config.extensions.chainlit_tool_steps_enabled,
    )
    effective_reasoning_level = resolve_reasoning_level_for_message(
        message,
        settings,
        reasoning_mode_enabled=runtime.config.extensions.chainlit_reasoning_mode_enabled,
    )
    effective_model_name = resolve_model_name_for_message(
        message,
        settings,
        available_models=runtime.config.model_choices,
        model_mode_enabled=runtime.config.extensions.chainlit_model_mode_enabled,
    )
    mcp_session_id = current_mcp_session_id()
    run_task_list = await get_run_task_list(
        reasoning_steps_enabled=settings.show_reasoning_stream,
        tool_steps_enabled=settings.show_tool_calls,
    )
    uploaded_files = message_uploaded_rag_files(message)
    uploaded_image_parts = message_uploaded_image_parts(message)
    uploaded_image_names = message_uploaded_image_names(message)
    unsupported_image_names = unsupported_uploaded_image_names(message)
    prompt_note = ""
    if uploaded_files:
        upload_result = await runtime.ingest_rag_uploads(
            thread_id=settings.thread_id,
            uploads=uploaded_files,
        )
        prompt_note = upload_result_prompt_note(upload_result.added_files)
        upload_message = cl.Message(
            content=upload_result_message(upload_result),
            author="System",
        )
        if runtime.rag_enabled:
            upload_message.actions = rag_actions()
        await upload_message.send()

    if unsupported_image_names:
        await cl.Message(
            content=unsupported_uploaded_images_message(unsupported_image_names),
            author="System",
        ).send()

    parsed_command = resolve_native_command(
        raw_text=message.content,
        selected_command=getattr(message, "command", None),
    )
    slash_command_from_text = parse_native_command(message.content)
    if (
        not message.content.strip()
        and uploaded_files
        and not uploaded_image_parts
        and parsed_command is None
    ):
        return
    if (
        not message.content.strip()
        and not uploaded_files
        and not uploaded_image_parts
        and parsed_command is None
    ):
        return

    if parsed_command is not None:
        try:
            transformed_prompt = await handle_native_command(
                runtime=runtime,
                settings=settings,
                parsed=parsed_command,
                mcp_session_id=mcp_session_id,
            )
        except Exception as exc:
            await cl.Message(
                author="System",
                content=f"Native command `/{parsed_command.command_name}` failed: {exc}",
            ).send()
            return
        if transformed_prompt is None:
            if slash_command_from_text is None:
                await cl.Message(
                    author="System",
                    content=(
                        f"Unknown command `/{parsed_command.command_name}`.\n"
                        "Use a configured command from startup or send a normal prompt."
                    ),
                ).send()
                return
        else:
            if not transformed_prompt.strip():
                return
            message.content = transformed_prompt

    agent_prompt = chainlit_prompt_text(
        message.content,
        image_names=uploaded_image_names,
        prompt_note=prompt_note,
    )
    async_url_override = async_subagent_url_override()
    reasoning_level_is_explicit = (
        message_has_reasoning_level_override(
            message,
            reasoning_mode_enabled=runtime.config.extensions.chainlit_reasoning_mode_enabled,
        )
        or settings_reasoning_level_is_explicit(
            runtime.config,
            settings,
            effective_model_name,
        )
    )
    agent = await runtime.get_agent(
        effective_reasoning_level,
        model_name=effective_model_name,
        reasoning_level_is_explicit=reasoning_level_is_explicit,
        thread_id=settings.thread_id,
        async_subagent_url_override=async_url_override,
        mcp_session_id=mcp_session_id,
    )
    async_task_notifier = get_async_task_notifier(
        agent=agent,
        runtime=runtime,
        url_override=async_url_override,
    )
    bridge = ChainlitEventBridge(
        prompt=agent_prompt,
        run_task_list=run_task_list,
        chronological_ui_enabled=runtime.config.extensions.chainlit_chronological_ui_enabled,
        reasoning_steps_enabled=settings.show_reasoning_stream,
        tool_steps_enabled=settings.show_tool_calls,
        generative_ui_enabled=runtime.config.extensions.chainlit_generative_ui_enabled,
        generated_ui_elements=get_generated_ui_elements(),
        reflection_collector=ReflectionCollector.from_runtime_config(
            runtime.config,
            prompt=agent_prompt,
        ),
    )
    await bridge.start()

    config = build_langgraph_config(
        settings,
        runtime_config=runtime.config,
        project_root=runtime.project_root,
    )
    payload = {
        "messages": [
            {
                "role": "user",
                "content": chainlit_user_message_content(
                    agent_prompt,
                    image_parts=uploaded_image_parts,
                ),
            }
        ]
    }
    stream = agent.astream_events(
        payload,
        config=config,
        version="v2",
        stream_mode=["messages", "updates", "custom"],
        subgraphs=True,
    )

    try:
        while True:
            try:
                part = await anext(stream)
            except StopAsyncIteration:
                break
            await bridge.handle_event(part)
    except asyncio.CancelledError:
        with suppress(Exception):
            await stream.aclose()
        return
    except Exception as exc:
        with suppress(Exception):
            await stream.aclose()
        details = traceback.format_exc(limit=10)
        with suppress(Exception):
            await bridge.fail(exc, details)
        proposal = bridge.reflection_proposal()
        if proposal is not None:
            with suppress(Exception):
                await ask_to_save_reflection_lesson(
                    runtime=runtime,
                    settings=settings,
                    proposal=proposal,
                    reasoning_level=effective_reasoning_level,
                    model_name=effective_model_name,
                    async_url_override=async_url_override,
                    mcp_session_id=mcp_session_id,
                )
        return
    finally:
        with suppress(Exception):
            await stream.aclose()

    await bridge.finish()
    proposal = bridge.reflection_proposal()
    if proposal is not None:
        await ask_to_save_reflection_lesson(
            runtime=runtime,
            settings=settings,
            proposal=proposal,
            reasoning_level=effective_reasoning_level,
            model_name=effective_model_name,
            async_url_override=async_url_override,
            mcp_session_id=mcp_session_id,
        )
    if async_task_notifier is not None:
        with suppress(Exception):
            await async_task_notifier.schedule_from_state(thread_id=settings.thread_id)


@cl.on_chat_end
async def on_chat_end() -> None:
    """Clean up runtime resources when the Chainlit chat ends."""
    notifier = cl.user_session.get(SESSION_ASYNC_TASK_NOTIFIER_KEY)
    if isinstance(notifier, AsyncTaskNotifier):
        notifier.cancel()
