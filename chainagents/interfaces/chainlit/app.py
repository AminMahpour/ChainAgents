"""Run the Chainlit UI for the configured ChainAgents runtime."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import secrets
from collections import deque
from collections.abc import Iterable, Mapping
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import Any

from chainagents.util.langchain_warnings import install_langchain_warning_filters

install_langchain_warning_filters()

import chainlit as cl
from chainlit.config import config as chainlit_config
from chainlit.types import ThreadDict
from chainlit.user_session import user_sessions as chainlit_user_sessions

from chainagents.interfaces.chainlit.async_tasks import (
    AsyncTaskNotifier,
    LocalBackgroundTaskNotifier,
    async_subagent_url_override,
    retain_missed_local_task_results,
)
from chainagents.interfaces.chainlit.bridge import ChainlitEventBridge, RunTaskList
from chainagents.interfaces.chainlit.lifetime import ConversationScopeLeaseManager
from chainagents.interfaces.chainlit.persistence import chainlit_data_layer_enabled, create_chainlit_data_layer
from chainagents.interfaces.chainlit.renderer import ChainlitTurnRenderer
from chainagents.interfaces.chainlit.settings import (
    SESSION_MCP_SESSION_ID_KEY,  # noqa: F401
    SESSION_SETTINGS_KEY,
    build_chat_settings,
    coerce_settings,
    current_chainlit_session_id,
    current_chainlit_thread_id,
    current_mcp_session_id,
    default_reasoning_level_for_model,
    message_has_reasoning_level_override,
    publish_modes,
    resolve_model_name_for_message,
    resolve_reasoning_level_for_message,
    settings_payload,  # noqa: F401
    settings_reasoning_level_is_explicit,
    store_mcp_session_id,  # noqa: F401
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
chainlit_socket: Any = importlib.import_module("chainlit.socket")

SESSION_TASK_LIST_KEY = "run_task_list"
SESSION_ASYNC_TASK_NOTIFIER_KEY = "async_task_notifier"
SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY = "local_background_task_notifier"
SESSION_GENERATED_UI_ELEMENTS_KEY = "generated_ui_elements"
SESSION_INPUT_DRAFTS_KEY = "pending_input_drafts"
STEER_INPUT_ACTION = "steer_busy_input"
QUEUE_INPUT_ACTION = "queue_busy_input"
STOP_INPUT_ACTION = "stop_active_input"
RESUME_INPUT_ACTION = "resume_queued_input"
SESSION_ACTIVE_TURN_KEY = "active_agent_turn"
SESSION_DISCONNECT_CLEANUP_KEY = "pending_disconnect_cleanup"
SESSION_DISCONNECTED_OUTPUT_KEY = "pending_disconnected_output"
SESSION_RESUME_WARMUP_KEY = "pending_resume_warmup"
SESSION_RECONNECT_RECOVERY_KEY = "pending_reconnect_recovery"
SESSION_RECONNECT_RECOVERY_TASK_KEY = "pending_reconnect_recovery_task"
SESSION_PENDING_LOCAL_RECONCILIATION_KEY = "pending_local_reconciliation"
REFLECTION_SAVE_ACTION = "save_reflection_lesson"
REFLECTION_DISMISS_ACTION = "dismiss_reflection_lesson"
conversation_scopes = ConversationScopeLeaseManager(idle_seconds=600, max_idle=4)
_detached_conversations: dict[tuple[int, str], _DetachedConversation] = {}


@dataclass
class _DetachedConversation:
    runtime: AgentRuntime
    scopes: ConversationScopeLeaseManager
    thread_id: str
    owner_id: str
    foreground_tasks: set[asyncio.Task[Any]]
    cleared_sessions: dict[str, Any]
    task: asyncio.Task[None] | None = None
    accepting_work: bool = True


def _unfinished_foreground_tasks(*tasks: Any) -> set[asyncio.Task[Any]]:
    current = asyncio.current_task()
    return {
        task
        for task in tasks
        if isinstance(task, asyncio.Task) and task is not current and not task.done()
    }


async def _conversation_has_work(
    runtime: AgentRuntime, thread_id: str, foreground_tasks: set[asyncio.Task[Any]]
) -> bool:
    if _unfinished_foreground_tasks(*foreground_tasks):
        return True
    busy = getattr(runtime, "conversation_busy", None)
    return (
        bool(await busy(thread_id, include_paused_queue=False))
        if callable(busy)
        else False
    )


async def _retain_missed_background_notices(
    runtime: AgentRuntime, thread_id: str
) -> None:
    background_tasks = getattr(runtime, "background_tasks", None)
    if background_tasks is not None:
        await retain_missed_local_task_results(background_tasks, thread_id)


async def _retain_conversation_until_idle(
    runtime: AgentRuntime,
    thread_id: str,
    foreground_tasks: set[asyncio.Task[Any]],
    *,
    cleared_session: Any | None = None,
) -> None:
    """Hold one runtime lease while work outlives its Chainlit tab."""
    key = (id(runtime), thread_id)
    existing = _detached_conversations.get(key)
    if (
        existing is not None
        and existing.accepting_work
        and existing.task is not None
        and not existing.task.done()
    ):
        existing.foreground_tasks.update(foreground_tasks)
        if cleared_session is not None and getattr(cleared_session, "id", None):
            existing.cleared_sessions[cleared_session.id] = cleared_session
        return

    owner_id = f"detached:{secrets.token_hex(16)}"
    extensions = getattr(getattr(runtime, "config", None), "extensions", None)
    await conversation_scopes.lease(
        runtime=runtime,
        owner_id=owner_id,
        scope_id=thread_id,
        retain_idle=getattr(extensions, "mcp_stateful", True),
    )
    cleared_sessions = (
        {cleared_session.id: cleared_session}
        if cleared_session is not None and getattr(cleared_session, "id", None)
        else {}
    )
    entry = _DetachedConversation(
        runtime, conversation_scopes, thread_id, owner_id,
        foreground_tasks, cleared_sessions,
    )
    _detached_conversations[key] = entry

    async def release_when_idle() -> None:
        try:
            while True:
                try:
                    pending = _unfinished_foreground_tasks(*entry.foreground_tasks)
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    wait_idle = getattr(runtime, "wait_conversation_idle", None)
                    if callable(wait_idle):
                        await wait_idle(thread_id, include_paused_queue=False)
                    if not await _conversation_has_work(
                        runtime, thread_id, entry.foreground_tasks
                    ):
                        await _retain_missed_background_notices(runtime, thread_id)
                        if not await _conversation_has_work(
                            runtime, thread_id, entry.foreground_tasks
                        ):
                            break
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "Failed to wait for conversation %s to become idle", thread_id
                    )
                    await asyncio.sleep(1)
        finally:
            # A later clear needs its own lease once this owner starts releasing.
            # The task is still pending during scope close and orphan cleanup.
            entry.accepting_work = False
            try:
                await entry.scopes.release(owner_id)
            finally:
                if _detached_conversations.get(key) is entry:
                    _detached_conversations.pop(key, None)
            # Chainlit removes the old user_session after on_chat_end returns.
            # Preserved turns can re-create it, so discard that orphan only
            # after Chainlit has deleted the WebsocketSession itself.
            for cleared in entry.cleared_sessions.values():
                while chainlit_socket.WebsocketSession.get_by_id(cleared.id) is cleared:  # noqa: ASYNC110
                    await asyncio.sleep(0.05)
                if chainlit_socket.WebsocketSession.get_by_id(cleared.id) is None:
                    chainlit_user_sessions.pop(cleared.id, None)

    entry.task = asyncio.create_task(release_when_idle())


@dataclass(frozen=True)
class _PendingLocalReconciliation:
    notifier: LocalBackgroundTaskNotifier
    restored_message_ids: frozenset[str]


@dataclass(frozen=True)
class _ReconnectRecovery:
    disconnected_socket_id: str
    runtime: AgentRuntime | None
    raw_settings: dict[str, Any] | None
    async_notifier: Any
    local_notifier: Any


def _conversation_owner_id(session: Any | None = None) -> str:
    """Identify a Chainlit tab independently of its reconnecting socket."""
    session_id = current_chainlit_session_id()
    if session_id:
        return session_id
    if session is None:
        with suppress(Exception):
            session = cl.context.session
    return f"chainlit-session-{id(session if session is not None else cl.user_session)}"


def _cancel_resume_warmup(session: Any | None) -> asyncio.Task[None] | None:
    """Stop a pending resume warm-up before its session is retargeted or closed."""
    task = getattr(session, SESSION_RESUME_WARMUP_KEY, None)
    if isinstance(task, asyncio.Task) and not task.done():
        task.cancel()
        return task
    return None


def _start_resume_warmup(
    *, runtime: AgentRuntime, settings: AppSettings, mcp_session_id: str
) -> None:
    """Restore async-subagent polling after the resume callback returns."""
    try:
        session = cl.context.session
    except Exception:
        return
    _cancel_resume_warmup(session)

    async def warm_up() -> None:
        try:
            url_override = async_subagent_url_override()
            agent = await runtime.get_agent(
                settings.reasoning_level,
                model_name=settings.model_name,
                reasoning_level_is_explicit=settings_reasoning_level_is_explicit(
                    runtime.config,
                    settings,
                    settings.model_name,
                ),
                thread_id=settings.thread_id,
                async_subagent_url_override=url_override,
                mcp_session_id=mcp_session_id,
            )
            current = asyncio.current_task()
            if (
                current is None
                or current.cancelling()
                or getattr(session, SESSION_RESUME_WARMUP_KEY, None) is not current
            ):
                return
            notifier = get_async_task_notifier(
                agent=agent,
                runtime=runtime,
                url_override=url_override,
            )
            if notifier is not None:
                await notifier.schedule_from_state(thread_id=settings.thread_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to restore async tasks for Chainlit thread %s", settings.thread_id)

    setattr(session, SESSION_RESUME_WARMUP_KEY, asyncio.create_task(warm_up()))


def _schedule_reconnect_recovery(session: Any) -> None:
    """Restore a reused Chainlit session after expired-socket cleanup raced it."""
    recovery = getattr(session, SESSION_RECONNECT_RECOVERY_KEY, None)
    if not isinstance(recovery, _ReconnectRecovery):
        return
    if getattr(session, "to_clear", False):
        delattr(session, SESSION_RECONNECT_RECOVERY_KEY)
        return
    if getattr(session, "socket_id", None) == recovery.disconnected_socket_id:
        return
    existing = getattr(session, SESSION_RECONNECT_RECOVERY_TASK_KEY, None)
    if isinstance(existing, asyncio.Task) and not existing.done():
        return

    async def recover() -> None:
        try:
            cleanup = getattr(session, SESSION_DISCONNECT_CLEANUP_KEY, None)
            if (
                isinstance(cleanup, asyncio.Task)
                and cleanup is not asyncio.current_task()
                and not cleanup.done()
            ):
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
            if (
                getattr(session, SESSION_RECONNECT_RECOVERY_KEY, None) is recovery
                and not getattr(session, "to_clear", False)
                and not isinstance(
                    getattr(session, SESSION_DISCONNECTED_OUTPUT_KEY, None),
                    DisconnectedOutput,
                )
                and getattr(session, "socket_id", None)
                != recovery.disconnected_socket_id
            ):
                await _restore_reconnected_observers(session, recovery)
        except Exception:
            logger.exception("Failed to restore observers after Chainlit reconnect")

    setattr(
        session,
        SESSION_RECONNECT_RECOVERY_TASK_KEY,
        asyncio.create_task(recover()),
    )


async def _restore_reconnected_observers(
    session: Any, recovery: _ReconnectRecovery
) -> None:
    runtime = AgentRuntime.current() or recovery.runtime
    if runtime is None:
        return
    extensions = runtime.config.extensions
    raw_settings = cl.user_session.get(SESSION_SETTINGS_KEY)
    if not isinstance(raw_settings, dict):
        raw_settings = recovery.raw_settings
    settings = coerce_settings(
        raw_settings,
        default_model_name=runtime.config.model_name,
        available_models=runtime.config.model_choices,
        runtime_config=runtime.config,
        show_reasoning_stream_default=extensions.chainlit_reasoning_steps_enabled,
        show_tool_calls_default=extensions.chainlit_tool_steps_enabled,
    )
    await conversation_scopes.lease(
        runtime=runtime,
        owner_id=_conversation_owner_id(session),
        scope_id=settings.thread_id,
        retain_idle=getattr(extensions, "mcp_stateful", True),
    )

    current_local = cl.user_session.get(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY)
    if current_local is recovery.local_notifier:
        cl.user_session.set(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY, None)
        current_local = None
    if (
        current_local is None
        and extensions.background_subagents.enabled
    ):
        await start_local_background_notifier(
            runtime=runtime,
            session_id=settings.thread_id,
            reasoning_steps_enabled=settings.show_reasoning_stream,
            tool_steps_enabled=settings.show_tool_calls,
        )

    current_async = cl.user_session.get(SESSION_ASYNC_TASK_NOTIFIER_KEY)
    if current_async is recovery.async_notifier:
        cl.user_session.set(SESSION_ASYNC_TASK_NOTIFIER_KEY, None)
        current_async = None
    if current_async is None and extensions.async_subagents:
        _start_resume_warmup(
            runtime=runtime,
            settings=settings,
            mcp_session_id=settings.thread_id,
        )
    if getattr(session, SESSION_RECONNECT_RECOVERY_KEY, None) is recovery:
        delattr(session, SESSION_RECONNECT_RECOVERY_KEY)


class DisconnectedOutput:
    """Keep socket events emitted while a Chainlit session is disconnected."""

    def __init__(self) -> None:
        self.events: deque[tuple[str, Any]] = deque()
        self.live_emit: Any = None

    async def queue(self, event: str, data: Any) -> None:
        self.events.append((event, data))

    def reconnect(self, session: Any) -> None:
        # Chainlit replaces emit during session restoration. Keep queuing until
        # its connection_successful handler has finished initializing the UI.
        self.live_emit = session.emit
        session.emit = self.queue

    async def flush(self, session: Any) -> None:
        if self.live_emit is None:
            return
        while self.events:
            event, data = self.events[0]
            await self.live_emit(event, data)
            self.events.popleft()
        session.emit = self.live_emit
        delattr(session, SESSION_DISCONNECTED_OUTPUT_KEY)


_chainlit_restore_existing_session = chainlit_socket.restore_existing_session
_chainlit_connection_successful = chainlit_socket.connection_successful


def _restore_session_with_output_replay(
    sid: str,
    session_id: str,
    emit_fn: Any,
    emit_call_fn: Any,
    environ: Any,
    user: Any = None,
) -> bool:
    restored = _chainlit_restore_existing_session(
        sid, session_id, emit_fn, emit_call_fn, environ, user=user
    )
    if restored:
        session = chainlit_socket.WebsocketSession.get_by_id(session_id)
        pending = getattr(session, SESSION_DISCONNECTED_OUTPUT_KEY, None)
        if isinstance(pending, DisconnectedOutput):
            pending.reconnect(session)
    return restored


async def _connection_successful_with_output_replay(sid: str) -> None:
    await _chainlit_connection_successful(sid)
    session = chainlit_socket.WebsocketSession.get(sid)
    pending = getattr(session, SESSION_DISCONNECTED_OUTPUT_KEY, None)
    if isinstance(pending, DisconnectedOutput):
        await pending.flush(session)
    local_reconciliation = getattr(
        session, SESSION_PENDING_LOCAL_RECONCILIATION_KEY, None
    )
    if isinstance(local_reconciliation, _PendingLocalReconciliation):
        delattr(session, SESSION_PENDING_LOCAL_RECONCILIATION_KEY)
        local_notifier = local_reconciliation.notifier
        if cl.user_session.get(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY) is local_notifier:
            try:
                await local_notifier.reconcile_terminal_tasks(
                    restored_message_ids=local_reconciliation.restored_message_ids
                )
            except Exception:
                logger.exception("Failed to reconcile local background tasks after resume")
    _schedule_reconnect_recovery(session)


chainlit_socket.restore_existing_session = _restore_session_with_output_replay
chainlit_socket.sio.on(
    "connection_successful", handler=_connection_successful_with_output_replay
)


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
    try:
        session = cl.context.session
    except Exception:
        session = None
    notifier = LocalBackgroundTaskNotifier(
        manager=runtime.background_tasks,
        session_id=session_id,
        reasoning_steps_enabled=reasoning_steps_enabled,
        tool_steps_enabled=tool_steps_enabled,
        delivery_allowed=lambda: not getattr(session, "to_clear", False),
    )
    notifier.start()
    cl.user_session.set(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY, notifier)


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize Chainlit session state when a chat starts."""
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
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
    await conversation_scopes.lease(
        runtime=runtime,
        owner_id=_conversation_owner_id(),
        scope_id=settings.thread_id,
        retain_idle=getattr(extensions, "mcp_stateful", True),
    )
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
        "- MCP session mode: stateful for this conversation; idle resources expire after 10 minutes\n"
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
    await publish_native_commands(runtime)

    extensions = runtime.config.extensions
    metadata = thread.get("metadata") or {}
    raw_settings = (
        metadata.get(SESSION_SETTINGS_KEY) if isinstance(metadata, dict) else None
    )
    raw_settings = dict(raw_settings) if isinstance(raw_settings, dict) else {}
    if not raw_settings.get("thread_id") and thread.get("id"):
        raw_settings["thread_id"] = thread["id"]
    settings = coerce_settings(
        raw_settings,
        default_model_name=runtime.config.model_name,
        available_models=runtime.config.model_choices,
        runtime_config=runtime.config,
        show_reasoning_stream_default=extensions.chainlit_reasoning_steps_enabled,
        show_tool_calls_default=extensions.chainlit_tool_steps_enabled,
    )
    store_settings(settings)
    await conversation_scopes.lease(
        runtime=runtime,
        owner_id=_conversation_owner_id(),
        scope_id=settings.thread_id,
        retain_idle=getattr(extensions, "mcp_stateful", True),
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
    local_notifier = cl.user_session.get(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY)
    if isinstance(local_notifier, LocalBackgroundTaskNotifier):
        restored_message_ids = frozenset(
            str(step["id"])
            for step in thread.get("steps", [])
            if isinstance(step, dict) and step.get("id")
        )
        setattr(
            cl.context.session,
            SESSION_PENDING_LOCAL_RECONCILIATION_KEY,
            _PendingLocalReconciliation(local_notifier, restored_message_ids),
        )
    if extensions.async_subagents:
        _start_resume_warmup(
            runtime=runtime,
            settings=settings,
            mcp_session_id=settings.thread_id,
        )


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
    previous_settings = cl.user_session.get(SESSION_SETTINGS_KEY)
    previous_thread_id = (
        str(previous_settings.get("thread_id") or "").strip()
        if isinstance(previous_settings, dict)
        else ""
    )
    try:
        session = cl.context.session
    except Exception:
        session = None
    if previous_thread_id and previous_thread_id != settings.thread_id:
        old_local_notifier = cl.user_session.get(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY)
        if (
            isinstance(old_local_notifier, LocalBackgroundTaskNotifier)
            and old_local_notifier.session_id == previous_thread_id
        ):
            old_local_notifier.detach()
    warmup = _cancel_resume_warmup(session)
    if warmup is not None:
        with suppress(asyncio.CancelledError):
            await warmup
    old_thread_busy = False
    if previous_thread_id and previous_thread_id != settings.thread_id:
        foreground_tasks = _unfinished_foreground_tasks(
            cl.user_session.get(SESSION_ACTIVE_TURN_KEY),
            getattr(session, "current_task", None),
        )
        old_thread_busy = await _conversation_has_work(
            runtime, previous_thread_id, foreground_tasks
        )
        if old_thread_busy:
            await _retain_conversation_until_idle(
                runtime, previous_thread_id, foreground_tasks
            )
        else:
            await _retain_missed_background_notices(runtime, previous_thread_id)
    if previous_thread_id and previous_thread_id != settings.thread_id:
        async_notifier = cl.user_session.get(SESSION_ASYNC_TASK_NOTIFIER_KEY)
        if isinstance(async_notifier, AsyncTaskNotifier):
            async_notifier.cancel()
            cl.user_session.set(SESSION_ASYNC_TASK_NOTIFIER_KEY, None)
    local_notifier = cl.user_session.get(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY)
    if (
        isinstance(local_notifier, LocalBackgroundTaskNotifier)
        and local_notifier.session_id != settings.thread_id
    ):
        if old_thread_busy:
            await local_notifier.aclose_for_handoff()
        else:
            await local_notifier.aclose()
        cl.user_session.set(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY, None)
    store_settings(settings)
    await conversation_scopes.lease(
        runtime=runtime,
        owner_id=_conversation_owner_id(session),
        scope_id=settings.thread_id,
        retain_idle=getattr(runtime.config.extensions, "mcp_stateful", True),
    )
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
    if previous_thread_id and previous_thread_id != settings.thread_id:
        current_local = cl.user_session.get(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY)
        if isinstance(current_local, LocalBackgroundTaskNotifier):
            await current_local.reconcile_terminal_tasks()
    await publish_modes(
        settings,
        available_models=runtime.config.model_choices,
        model_mode_enabled=runtime.config.extensions.chainlit_model_mode_enabled,
        reasoning_mode_enabled=runtime.config.extensions.chainlit_reasoning_mode_enabled,
    )
    if warmup is not None and getattr(runtime.config.extensions, "async_subagents", ()):
        _start_resume_warmup(
            runtime=runtime,
            settings=settings,
            mcp_session_id=settings.thread_id,
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
    with suppress(Exception):
        if getattr(cl.context.session, "to_clear", False):
            return
    if cl.user_session.get(SESSION_ACTIVE_TURN_KEY) is asyncio.current_task():
        cl.user_session.set(SESSION_ACTIVE_TURN_KEY, None)


async def _send_turn_busy() -> None:
    await cl.Message(
        content="The agent is already responding. Your request was not sent; please resend it after this turn finishes.",
        author="System",
    ).send()


def _submit_nonblocking_message(
    runtime: AgentRuntime,
    thread_id: str,
    message: cl.Message,
    *,
    settings: AppSettings | None = None,
) -> str:
    """Schedule a Chainlit turn while its socket callback returns promptly."""
    settings_snapshot = settings or coerce_settings(
        cl.user_session.get(SESSION_SETTINGS_KEY),
        default_model_name=runtime.config.model_name,
        available_models=runtime.config.model_choices,
        show_reasoning_stream_default=(
            runtime.config.extensions.chainlit_reasoning_steps_enabled
        ),
        show_tool_calls_default=runtime.config.extensions.chainlit_tool_steps_enabled,
    )

    async def run(item: tuple[cl.Message, bool]) -> None:
        current, show_user = item
        if show_user:
            current = await cl.Message(content=current.content, author="User").send()
        async with cl.Step(name="on_message", type="run", parent_id=current.id) as step:
            step.input = current.content
            await _handle_message(current, settings_override=settings_snapshot)

    job = runtime.user_input.submit(
        thread_id,
        (message, False),
        run,
        followup_factory=lambda text: (cl.Message(content=text), True),
        input_id=str(message.id) if getattr(message, "id", None) else None,
    )
    return job.id


def _current_input_thread_id(runtime: AgentRuntime) -> str:
    """Resolve the configured LangGraph thread from this Chainlit connection."""
    settings = coerce_settings(
        cl.user_session.get(SESSION_SETTINGS_KEY),
        default_model_name=runtime.config.model_name,
        available_models=runtime.config.model_choices,
    )
    return settings.thread_id


async def _offer_busy_input(runtime: AgentRuntime, thread_id: str, message: cl.Message) -> None:
    draft_id = secrets.token_hex(12)
    drafts = getattr(cl.context.session, SESSION_INPUT_DRAFTS_KEY, None)
    if drafts is None:
        drafts = {}
        setattr(cl.context.session, SESSION_INPUT_DRAFTS_KEY, drafts)
    if len(drafts) >= runtime.config.extensions.user_input.max_queued_turns:
        await cl.Message(
            content="Too many pending input choices. Choose one before sending another prompt.",
            author="System",
        ).send()
        return
    drafts[draft_id] = (thread_id, message)
    await cl.Message(
        content="The agent is working. Choose how to send this input.",
        author="System",
        actions=[
            cl.Action(name=STEER_INPUT_ACTION, payload={"draft_id": draft_id}, label="Steer active turn"),
            cl.Action(name=QUEUE_INPUT_ACTION, payload={"draft_id": draft_id}, label="Queue next turn"),
        ],
    ).send()


@cl.action_callback(STEER_INPUT_ACTION)
@cl.action_callback(QUEUE_INPUT_ACTION)
async def _submit_busy_input(action: cl.Action) -> None:
    drafts = getattr(cl.context.session, SESSION_INPUT_DRAFTS_KEY, {})
    draft = drafts.pop(str(action.payload.get("draft_id", "")), None)
    if draft is None:
        await cl.Message(content="This input choice expired. Please resend the prompt.", author="System").send()
        return
    thread_id, message = draft
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    if thread_id != _current_input_thread_id(runtime):
        await cl.Message(content="This input choice belongs to another conversation.", author="System").send()
        return
    try:
        if action.name == STEER_INPUT_ACTION:
            if message.elements:
                raise ValueError("Steering accepts text only; queue this turn to include attachments.")
            runtime.user_input.steer(thread_id, message.content)
            response = "Steering note sent to the active turn."
        else:
            job_id = _submit_nonblocking_message(runtime, thread_id, message)
            response = f"Queued turn `{job_id}`."
        await cl.Message(content=response, author="System").send()
    except ValueError as exc:
        await cl.Message(content=str(exc), author="System").send()


@cl.action_callback(STOP_INPUT_ACTION)
async def _stop_nonblocking_input(action: cl.Action) -> None:
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    thread_id = str(action.payload.get("thread_id", "")).strip()
    if not thread_id or thread_id != _current_input_thread_id(runtime):
        await cl.Message(content="This Stop action belongs to another conversation.", author="System").send()
        return
    runtime.user_input.stop(thread_id)
    await cl.Message(
        content="Active turn stopped. Queued turns are paused.",
        author="System",
        actions=[cl.Action(
            name=RESUME_INPUT_ACTION,
            payload={"thread_id": thread_id},
            label="Resume queue",
        )],
    ).send()


@cl.action_callback(RESUME_INPUT_ACTION)
async def _resume_nonblocking_input(action: cl.Action) -> None:
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    thread_id = str(action.payload.get("thread_id", "")).strip()
    if not thread_id or thread_id != _current_input_thread_id(runtime):
        await cl.Message(content="This Resume action belongs to another conversation.", author="System").send()
        return
    runtime.user_input.resume(thread_id)
    await cl.Message(content="Queued turns resumed.", author="System").send()


@cl.action_callback(RUN_RESPONSE_ACTION)
async def run_response_action(action: cl.Action) -> None:
    """Run a configured response action without creating a user message."""
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    if getattr(getattr(runtime.config.extensions, "user_input", None), "enabled", False):
        resolved = resolve_response_action(
            action, runtime.config.extensions.chainlit_response_actions
        )
        if resolved is None:
            await cl.Message(content="This response action is no longer available.", author="System").send()
            return
        settings = coerce_settings(
            cl.user_session.get(SESSION_SETTINGS_KEY),
            default_model_name=runtime.config.model_name,
            available_models=runtime.config.model_choices,
        )

        async def run(item: tuple[str, bool]) -> None:
            prompt, is_followup = item
            if is_followup:
                await cl.Message(content=prompt, author="User").send()
            await _run_agent_turn(
                runtime=runtime,
                settings=settings,
                agent_prompt=prompt,
                effective_reasoning_level=settings.reasoning_level,
                effective_model_name=settings.model_name,
                reasoning_level_is_explicit=settings_reasoning_level_is_explicit(
                    runtime.config, settings, settings.model_name
                ),
                mcp_session_id=current_mcp_session_id(),
                resolve_commands=False,
                display_prompt=prompt if is_followup else "",
                export_label="" if is_followup else resolved.label,
            )

        try:
            job = runtime.user_input.submit(
                settings.thread_id, (resolved.prompt, False), run,
                followup_factory=lambda text: (text, True),
            )
            await cl.Message(content=f"Response action turn `{job.id}`: {job.status}.", author="System").send()
        except ValueError as exc:
            await cl.Message(content=str(exc), author="System").send()
        return
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
    runtime = AgentRuntime.current()
    if runtime is not None and getattr(getattr(runtime.config.extensions, "user_input", None), "enabled", False):
        thread_id = _current_input_thread_id(runtime)
        if thread_id:
            runtime.user_input.stop(thread_id)
        return
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
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    if getattr(getattr(runtime.config.extensions, "user_input", None), "enabled", False):
        settings = coerce_settings(
            cl.user_session.get(SESSION_SETTINGS_KEY),
            default_model_name=runtime.config.model_name,
            available_models=runtime.config.model_choices,
            show_reasoning_stream_default=(
                runtime.config.extensions.chainlit_reasoning_steps_enabled
            ),
            show_tool_calls_default=runtime.config.extensions.chainlit_tool_steps_enabled,
        )
        thread_id = settings.thread_id
        status = runtime.user_input.status(thread_id)
        if status["active_job_id"] is not None:
            await _offer_busy_input(runtime, thread_id, message)
        else:
            try:
                job_id = _submit_nonblocking_message(runtime, thread_id, message, settings=settings)
                await cl.Message(
                    content=f"Working on turn `{job_id}`.", author="System",
                    actions=[cl.Action(
                        name=STOP_INPUT_ACTION,
                        payload={"thread_id": thread_id},
                        label="Stop",
                    )],
                ).send()
            except ValueError as exc:
                await cl.Message(content=str(exc), author="System").send()
        return
    if not _claim_active_turn():
        await _send_turn_busy()
        return
    try:
        assert _chainlit_message_callback is not None
        await _chainlit_message_callback(message)
    finally:
        _release_active_turn()


chainlit_config.code.on_message = _guarded_chainlit_message_callback


async def _handle_message(
    message: cl.Message, *, settings_override: AppSettings | None = None
) -> None:
    """Prepare a normal user request for the shared agent turn."""
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    settings = settings_override or coerce_settings(
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
    turn_owner_id = f"turn:{secrets.token_hex(16)}"
    await conversation_scopes.lease(
        runtime=runtime,
        owner_id=turn_owner_id,
        scope_id=settings.thread_id,
        retain_idle=False,
    )
    try:
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
    finally:
        await conversation_scopes.release(turn_owner_id)


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
    """Clean up a cleared chat or an expired disconnected session."""
    try:
        session = cl.context.session
    except Exception:
        session = None
    active = cl.user_session.get(SESSION_ACTIVE_TURN_KEY)
    owner_id = _conversation_owner_id(session)
    raw_settings = cl.user_session.get(SESSION_SETTINGS_KEY)
    raw_settings = dict(raw_settings) if isinstance(raw_settings, dict) else None
    runtime = AgentRuntime.current()
    thread_id = (
        str(raw_settings.get("thread_id") or "").strip()
        if raw_settings is not None
        else str(getattr(session, "thread_id", "") or "").strip()
    )

    async def close_chat(
        *, expected_socket_id: str | None = None, preserve_work: bool = False
    ) -> None:
        tasks = {active, getattr(session, "current_task", None)}
        # Snapshot current watchers before awaiting a cancelled turn: Chainlit's
        # own timeout may remove user_session while cancellation renders UI.
        notifier = cl.user_session.get(SESSION_ASYNC_TASK_NOTIFIER_KEY)
        local_notifier = cl.user_session.get(SESSION_LOCAL_BACKGROUND_NOTIFIER_KEY)
        def still_disconnected() -> bool:
            return expected_socket_id is None or (
                getattr(session, "socket_id", None) == expected_socket_id
            )
        if not still_disconnected():
            return
        if not preserve_work:
            for task in tasks:
                if (
                    isinstance(task, asyncio.Task)
                    and task is not asyncio.current_task()
                    and not task.done()
                ):
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
        if not still_disconnected():
            return
        resume_warmup = _cancel_resume_warmup(session)
        if resume_warmup is not None:
            with suppress(asyncio.CancelledError):
                await resume_warmup
        if not still_disconnected():
            return
        if expected_socket_id is not None:
            setattr(
                session,
                SESSION_RECONNECT_RECOVERY_KEY,
                _ReconnectRecovery(
                    disconnected_socket_id=expected_socket_id,
                    runtime=runtime,
                    raw_settings=raw_settings,
                    async_notifier=notifier,
                    local_notifier=local_notifier,
                ),
            )
        if isinstance(notifier, AsyncTaskNotifier):
            notifier.cancel()
        if isinstance(local_notifier, LocalBackgroundTaskNotifier):
            if preserve_work:
                await local_notifier.aclose_for_handoff()
            else:
                await local_notifier.aclose()
        if not still_disconnected():
            _schedule_reconnect_recovery(session)
            return
        with suppress(Exception):
            setattr(session, SESSION_INPUT_DRAFTS_KEY, {})
        await conversation_scopes.release(owner_id, only_if=still_disconnected)
        if not still_disconnected():
            _schedule_reconnect_recovery(session)

    previous_cleanup = getattr(session, SESSION_DISCONNECT_CLEANUP_KEY, None)
    if isinstance(previous_cleanup, asyncio.Task) and not previous_cleanup.done():
        previous_cleanup.cancel()

    if session is None or getattr(session, "to_clear", False) or not getattr(session, "socket_id", None):
        foreground_tasks = _unfinished_foreground_tasks(
            active, getattr(session, "current_task", None)
        )
        preserve_work = bool(
            runtime is not None
            and thread_id
            and await _conversation_has_work(runtime, thread_id, foreground_tasks)
        )
        if preserve_work and runtime is not None:
            await _retain_conversation_until_idle(
                runtime,
                thread_id,
                foreground_tasks,
                cleared_session=session if getattr(session, "to_clear", False) else None,
            )
        elif runtime is not None and thread_id:
            await _retain_missed_background_notices(runtime, thread_id)
        await close_chat(preserve_work=preserve_work)
        return

    if callable(getattr(session, "emit", None)):
        socket_session: Any = session
        pending_output = getattr(session, SESSION_DISCONNECTED_OUTPUT_KEY, None)
        if not isinstance(pending_output, DisconnectedOutput):
            pending_output = DisconnectedOutput()
            setattr(session, SESSION_DISCONNECTED_OUTPUT_KEY, pending_output)
        socket_session.emit = pending_output.queue

    # Chainlit retains disconnected sessions for this timeout and changes the
    # socket ID when a browser reconnects to the same session.
    disconnected_socket_id = session.socket_id

    async def close_after_timeout() -> None:
        await asyncio.sleep(chainlit_config.project.session_timeout)
        if session.socket_id != disconnected_socket_id:
            return
        try:
            await close_chat(expected_socket_id=disconnected_socket_id)
        except Exception:
            logger.exception("Failed to clean up an expired Chainlit session")

    setattr(
        session,
        SESSION_DISCONNECT_CLEANUP_KEY,
        asyncio.create_task(close_after_timeout()),
    )
