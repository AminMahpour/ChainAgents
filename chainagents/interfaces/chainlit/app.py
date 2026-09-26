"""Run the Chainlit UI for the configured ChainAgents runtime."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from collections.abc import Iterable, Mapping
from contextlib import nullcontext, suppress
from typing import Any

from chainagents.util.langchain_warnings import install_langchain_warning_filters

install_langchain_warning_filters()

import chainlit as cl
from chainlit.config import config as chainlit_config
from chainlit.types import ThreadDict

from chainagents.interfaces.chainlit.async_tasks import (
    AsyncTaskNotifier,
    LocalBackgroundTaskNotifier,
    async_subagent_url_override,
)
from chainagents.interfaces.chainlit.bridge import ChainlitEventBridge, RunTaskList
from chainagents.interfaces.chainlit.persistence import chainlit_data_layer_enabled, create_chainlit_data_layer
from chainagents.interfaces.chainlit.renderer import ChainlitTurnRenderer
from chainagents.interfaces.chainlit.settings import (
    SESSION_MCP_SESSION_ID_KEY,  # noqa: F401
    SESSION_SETTINGS_KEY,
    build_chat_settings,
    coerce_settings,
    current_chainlit_thread_id,
    current_mcp_session_id,
    default_reasoning_level_for_model,
    message_has_reasoning_level_override,
    publish_modes,
    resolve_model_name_for_message,
    resolve_reasoning_level_for_message,
    settings_payload,  # noqa: F401
    settings_reasoning_level_is_explicit,
    store_mcp_session_id,
    store_settings,
)
from chainagents.interfaces.chainlit.uploads import (
    REBUILD_RAG_INDEX_ACTION,
    UPLOAD_RAG_FILE_ACTION,
    ask_for_rag_upload,
    message_uploaded_image_names,
    message_uploaded_image_parts,
    message_uploaded_rag_files,
    rag_actions,
    unsupported_uploaded_image_names,
    unsupported_uploaded_images_message,
    upload_result_message,
    upload_result_prompt_note,
)
from chainagents.runtime import (
    AgentRuntime,
    AppSettings,
    ChainlitStarterConfig,
    ReasoningLevel,
    RuntimeConfig,
    format_model_provider,
    resolve_runtime_model_profile,
)
from chainagents.runtime.reflection import (
    ReflectionProposal,
    format_reflection_proposal,
    reflection_save_prompt as reflection_save_prompt,
)
from chainagents.turns import TurnRequest, TurnRunner
from chainagents.exports.response import (
    DOWNLOAD_MARKDOWN_ACTION,
    DOWNLOAD_PDF_ACTION,
    RUN_RESPONSE_ACTION,
    resolve_response_action,
    restore_response_export_actions,
    send_markdown_export,
    send_pdf_export,
)

logger = logging.getLogger(__name__)

SESSION_TASK_LIST_KEY = "run_task_list"
SESSION_ASYNC_TASK_NOTIFIER_KEY = "async_task_notifier"
SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY = "local_background_task_notifier"
SESSION_GENERATED_UI_ELEMENTS_KEY = "generated_ui_elements"
SESSION_ACTIVE_TURN_KEY = "active_agent_turn"
REFLECTION_SAVE_ACTION = "save_reflection_lesson"
REFLECTION_DISMISS_ACTION = "dismiss_reflection_lesson"
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


async def start_local_background_notifier(
    *,
    runtime: AgentRuntime,
    session_id: str,
    reasoning_steps_enabled: bool,
    tool_steps_enabled: bool,
) -> None:
    """Start one local background completion subscriber for this chat."""
    existing = cl.user_session.get(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY)
    if isinstance(existing, LocalBackgroundTaskNotifier):
        await existing.aclose()
    if not runtime.config.extensions.background_subagents.enabled:
        return
    notifier = LocalBackgroundTaskNotifier(
        manager=runtime.background_tasks,
        session_id=session_id,
        reasoning_steps_enabled=reasoning_steps_enabled,
        tool_steps_enabled=tool_steps_enabled,
    )
    notifier.start()
    cl.user_session.set(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY, notifier)


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
    await start_local_background_notifier(
        runtime=runtime,
        session_id=settings.thread_id,
        reasoning_steps_enabled=settings.show_reasoning_stream,
        tool_steps_enabled=settings.show_tool_calls,
    )
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
    await start_local_background_notifier(
        runtime=runtime,
        session_id=settings.thread_id,
        reasoning_steps_enabled=settings.show_reasoning_stream,
        tool_steps_enabled=settings.show_tool_calls,
    )
    run_task_list = await get_run_task_list(
        reasoning_steps_enabled=settings.show_reasoning_stream,
        tool_steps_enabled=settings.show_tool_calls,
    )
    await run_task_list.show_ready()
    store_settings(settings)
    await _restore_saved_response_actions(thread, runtime)
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


async def _restore_saved_response_actions(
    thread: ThreadDict,
    runtime: AgentRuntime,
) -> None:
    """Reattach actions after Chainlit restores persisted response steps."""
    for action in restore_response_export_actions(
        thread,
        response_actions=runtime.config.extensions.chainlit_response_actions,
    ):
        await action.send(for_id=action.forId or "")


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
    local_notifier = cl.user_session.get(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY)
    if (
        isinstance(local_notifier, LocalBackgroundTaskNotifier)
        and local_notifier.session_id != settings.thread_id
    ):
        await local_notifier.aclose()
        await runtime.close_conversation(
            thread_id=local_notifier.session_id,
            mcp_session_id=current_mcp_session_id() or None,
        )
    store_settings(settings)
    if (
        not isinstance(local_notifier, LocalBackgroundTaskNotifier)
        or local_notifier.session_id != settings.thread_id
    ):
        await start_local_background_notifier(
            runtime=runtime,
            session_id=settings.thread_id,
            reasoning_steps_enabled=settings.show_reasoning_stream,
            tool_steps_enabled=settings.show_tool_calls,
        )
    else:
        local_notifier.configure(
            reasoning_steps_enabled=settings.show_reasoning_stream,
            tool_steps_enabled=settings.show_tool_calls,
        )
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


def _claim_active_turn() -> bool:
    """Reserve this Chainlit session for one agent turn."""
    active = cl.user_session.get(SESSION_ACTIVE_TURN_KEY)
    if isinstance(active, asyncio.Task) and not active.done():
        return False
    cl.user_session.set(SESSION_ACTIVE_TURN_KEY, asyncio.current_task())
    return True


def _release_active_turn() -> None:
    """Release the session turn only when owned by this task."""
    if cl.user_session.get(SESSION_ACTIVE_TURN_KEY) is asyncio.current_task():
        cl.user_session.set(SESSION_ACTIVE_TURN_KEY, None)


async def _send_turn_busy() -> None:
    await cl.Message(
        content="The agent is already responding. Your request was not sent; please resend it after this turn finishes.",
        author="System",
    ).send()


@cl.action_callback(RUN_RESPONSE_ACTION)
async def run_response_action(action: cl.Action) -> None:
    """Run a configured response action without creating a user message."""
    if not _claim_active_turn():
        await _send_turn_busy()
        return
    try:
        runtime = await get_runtime_or_notify()
        if runtime is None:
            return
        resolved = resolve_response_action(
            action, runtime.config.extensions.chainlit_response_actions
        )
        if resolved is None:
            await cl.Message(
                content="This response action is no longer available.", author="System"
            ).send()
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
        active_task = asyncio.current_task()
        previous_task = cl.context.session.current_task
        cl.context.session.current_task = active_task
        try:
            await cl.context.emitter.task_start()
            try:
                await _run_agent_turn(
                    runtime=runtime,
                    settings=settings,
                    agent_prompt=resolved.prompt,
                    effective_reasoning_level=settings.reasoning_level,
                    effective_model_name=settings.model_name,
                    reasoning_level_is_explicit=settings_reasoning_level_is_explicit(
                        runtime.config, settings, settings.model_name
                    ),
                    mcp_session_id=current_mcp_session_id(),
                    resolve_commands=False,
                    display_prompt="",
                    export_label=resolved.label,
                )
            finally:
                await cl.context.emitter.task_end()
        finally:
            if cl.context.session.current_task is active_task:
                cl.context.session.current_task = previous_task
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Response action failed")
        await cl.Message(
            content="The response action failed. Please try again.", author="System"
        ).send()
    finally:
        _release_active_turn()


@cl.on_stop
async def on_stop() -> None:
    """Stop the active agent turn, including an action callback turn."""
    active = cl.user_session.get(SESSION_ACTIVE_TURN_KEY)
    if isinstance(active, asyncio.Task) and active is not asyncio.current_task():
        active.cancel()


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
    # Chainlit wraps this callback in its ``on_message`` run step; swallowing a
    # stopped turn here keeps that step from being marked as an error.
    with suppress(asyncio.CancelledError):
        await _handle_message(message)


_chainlit_message_callback = chainlit_config.code.on_message


async def _guarded_chainlit_message_callback(message: cl.Message) -> None:
    """Reject a busy message before Chainlit creates its on_message run step."""
    if not _claim_active_turn():
        await _send_turn_busy()
        return
    try:
        assert _chainlit_message_callback is not None
        await _chainlit_message_callback(message)
    finally:
        _release_active_turn()


chainlit_config.code.on_message = _guarded_chainlit_message_callback


async def _handle_message(message: cl.Message) -> None:
    """Prepare a normal user request for the shared agent turn."""
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
    await get_run_task_list(
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
    await _run_agent_turn(
        runtime=runtime,
        settings=settings,
        agent_prompt=message.content,
        effective_reasoning_level=effective_reasoning_level,
        effective_model_name=effective_model_name,
        reasoning_level_is_explicit=reasoning_level_is_explicit,
        mcp_session_id=mcp_session_id,
        selected_command=getattr(message, "command", None),
        uploaded_image_parts=uploaded_image_parts,
        uploaded_image_names=uploaded_image_names,
        prompt_note=prompt_note,
    )


async def _run_agent_turn(
    *,
    runtime: AgentRuntime,
    settings: AppSettings,
    agent_prompt: str,
    effective_reasoning_level: ReasoningLevel,
    effective_model_name: str,
    reasoning_level_is_explicit: bool,
    mcp_session_id: str | None,
    selected_command: str | None = None,
    uploaded_image_parts: list[dict[str, Any]] | None = None,
    uploaded_image_names: tuple[str, ...] = (),
    prompt_note: str = "",
    resolve_commands: bool = True,
    display_prompt: str | None = None,
    export_label: str = "",
) -> None:
    """Run one ordinary or response-action turn through the shared runner.

    ``agent_prompt`` is the raw text, resolved as a native command when
    ``resolve_commands`` is set. ``CancelledError`` propagates to the
    outermost callback.
    """
    async_url_override = async_subagent_url_override()
    run_task_list = await get_run_task_list(
        reasoning_steps_enabled=settings.show_reasoning_stream,
        tool_steps_enabled=settings.show_tool_calls,
    )

    def build_bridge(prompt: str) -> ChainlitEventBridge:
        return ChainlitEventBridge(
            prompt=prompt,
            run_task_list=run_task_list,
            chronological_ui_enabled=runtime.config.extensions.chainlit_chronological_ui_enabled,
            reasoning_steps_enabled=settings.show_reasoning_stream,
            tool_steps_enabled=settings.show_tool_calls,
            generative_ui_enabled=runtime.config.extensions.chainlit_generative_ui_enabled,
            generated_ui_elements=get_generated_ui_elements(),
            display_prompt=display_prompt,
            export_label=export_label,
            response_actions=runtime.config.extensions.chainlit_response_actions,
        )

    request = TurnRequest(
        prompt=agent_prompt,
        thread_id=settings.thread_id,
        model_name=effective_model_name,
        reasoning_level=effective_reasoning_level,
        selected_command=selected_command,
        reasoning_level_is_explicit=reasoning_level_is_explicit,
        content_parts=tuple(uploaded_image_parts or ()),
        image_names=uploaded_image_names,
        prompt_note=prompt_note,
        async_subagent_url=async_url_override,
        mcp_session_id=mcp_session_id,
        resolve_commands=resolve_commands,
    )
    renderer = ChainlitTurnRenderer(build_bridge, prompt=agent_prompt)
    result = await TurnRunner(runtime, sanitize_errors=False).run(request, renderer)

    if result.reflection is not None:
        # A failed turn has already reported its error; never raise a second one.
        with suppress(Exception) if result.status == "failed" else nullcontext():
            await ask_to_save_reflection_lesson(
                runtime=runtime,
                settings=settings,
                proposal=result.reflection,
                reasoning_level=effective_reasoning_level,
                model_name=effective_model_name,
                async_url_override=async_url_override,
                mcp_session_id=mcp_session_id,
            )
    if result.status == "completed" and result.agent is not None:
        async_task_notifier = get_async_task_notifier(
            agent=result.agent,
            runtime=runtime,
            url_override=async_url_override,
        )
        if async_task_notifier is not None:
            with suppress(Exception):
                await async_task_notifier.schedule_from_state(thread_id=settings.thread_id)


def reflection_actions(*, retry: bool = False) -> list[cl.Action]:
    """Return Chainlit actions for reflection confirmation."""
    return [
        cl.Action(
            name=REFLECTION_SAVE_ACTION,
            payload={"value": "retry" if retry else "save"},
            label="Retry" if retry else "Save lesson",
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
    retry = False
    while True:
        response = await cl.AskActionMessage(
            content=format_reflection_proposal(proposal),
            actions=reflection_actions(retry=retry),
            author="System",
            timeout=90,
            raise_on_timeout=False,
        ).send()
        if not response:
            return
        payload = response.get("payload") if isinstance(response, dict) else None
        expected_action = "retry" if retry else "save"
        if not isinstance(payload, dict) or payload.get("value") != expected_action:
            return
        if await save_reflection_lesson(
            runtime=runtime,
            settings=settings,
            proposal=proposal,
            reasoning_level=reasoning_level,
            model_name=model_name,
            async_url_override=async_url_override,
            mcp_session_id=mcp_session_id,
        ):
            return
        # Chainlit consumes each prompt's actions. Keep this proposal and wait
        # for a fresh user selection before attempting storage again.
        retry = True


async def save_reflection_lesson(
    *,
    runtime: AgentRuntime,
    settings: AppSettings,
    proposal: ReflectionProposal,
    reasoning_level: ReasoningLevel,
    model_name: str,
    async_url_override: str | None,
    mcp_session_id: str | None,
) -> bool:
    """Return success only after verified persistence, allowing an explicit retry."""
    try:
        await runtime.save_reflection(proposal)
    except Exception:
        await cl.Message(
            content="Reflection could not be saved. Please retry.",
            author="System",
        ).send()
        return False
    await cl.Message(
        content=f"Saved lesson to `{proposal.memory_file}`.",
        author="System",
    ).send()
    return True


@cl.on_chat_end
async def on_chat_end() -> None:
    """Clean up runtime resources when the Chainlit chat ends."""
    active = cl.user_session.get(SESSION_ACTIVE_TURN_KEY)
    if isinstance(active, asyncio.Task) and active is not asyncio.current_task():
        active.cancel()
        with suppress(asyncio.CancelledError):
            await active
    notifier = cl.user_session.get(SESSION_ASYNC_TASK_NOTIFIER_KEY)
    if isinstance(notifier, AsyncTaskNotifier):
        notifier.cancel()
    local_notifier = cl.user_session.get(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY)
    if isinstance(local_notifier, LocalBackgroundTaskNotifier):
        await local_notifier.aclose()

    runtime = AgentRuntime.current()
    if runtime is not None:
        raw_settings = cl.user_session.get(SESSION_SETTINGS_KEY)
        thread_id = (
            str(raw_settings.get("thread_id") or "").strip()
            if isinstance(raw_settings, dict)
            else ""
        ) or current_chainlit_thread_id()
        await runtime.close_conversation(
            thread_id=thread_id or None,
            mcp_session_id=current_mcp_session_id() or None,
        )
