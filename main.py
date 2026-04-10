from __future__ import annotations

import asyncio
import os
import secrets
import traceback
from contextlib import suppress
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select, TextInput
from chainlit.types import ThreadDict

from async_task_notifications import AsyncTaskNotifier, async_subagent_url_override
from chainlit_bridge import ChainlitEventBridge, RunTaskList
from deepagent_runtime import (
    DEFAULT_REASONING_LEVEL,
    AgentRuntime,
    AppSettings,
    format_model_provider,
    normalize_reasoning_level,
)
from response_exports import (
    DOWNLOAD_MARKDOWN_ACTION,
    DOWNLOAD_PDF_ACTION,
    send_markdown_export,
    send_pdf_export,
)


SESSION_SETTINGS_KEY = "agent_settings"
SESSION_TASK_LIST_KEY = "run_task_list"
SESSION_ASYNC_TASK_NOTIFIER_KEY = "async_task_notifier"
AUTH_USERNAME = os.getenv("CHAINLIT_AUTH_USERNAME", "").strip()
AUTH_PASSWORD = os.getenv("CHAINLIT_AUTH_PASSWORD", "").strip()
AUTH_SECRET = os.getenv("CHAINLIT_AUTH_SECRET", "").strip()
AUTH_ENABLED = bool(AUTH_SECRET and AUTH_USERNAME and AUTH_PASSWORD)


def current_chainlit_thread_id() -> str:
    try:
        session = cl.context.session
    except Exception:
        return ""
    thread_id = getattr(session, "thread_id", None) or getattr(session, "id", None)
    return str(thread_id or "").strip()


def settings_payload(settings: AppSettings) -> dict[str, str]:
    return {
        "reasoning_level": settings.reasoning_level,
        "thread_id": settings.thread_id,
    }


def store_settings(settings: AppSettings) -> None:
    cl.user_session.set(SESSION_SETTINGS_KEY, settings_payload(settings))


def build_chat_settings(settings: AppSettings) -> cl.ChatSettings:
    reasoning_levels = ["low", "medium", "high"]
    return cl.ChatSettings(
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
        ]
    )


def coerce_settings(raw_settings: AppSettings | dict[str, Any] | None) -> AppSettings:
    if raw_settings is None:
        raw_settings = {}
    if isinstance(raw_settings, AppSettings):
        return raw_settings
    reasoning_level = normalize_reasoning_level(
        raw_settings.get("reasoning_level", DEFAULT_REASONING_LEVEL)
    )
    thread_id = str(
        raw_settings.get("thread_id") or current_chainlit_thread_id()
    ).strip()
    if not thread_id:
        thread_id = current_chainlit_thread_id()
    return AppSettings(reasoning_level=reasoning_level, thread_id=thread_id.strip())


if AUTH_ENABLED:

    @cl.password_auth_callback
    def password_auth_callback(username: str, password: str) -> cl.User | None:
        if not (
            secrets.compare_digest(username, AUTH_USERNAME)
            and secrets.compare_digest(password, AUTH_PASSWORD)
        ):
            return None

        return cl.User(
            identifier=AUTH_USERNAME,
            display_name=AUTH_USERNAME,
            metadata={"provider": "credentials"},
        )


async def get_runtime_or_notify() -> AgentRuntime | None:
    try:
        return await AgentRuntime.get()
    except Exception as exc:
        await cl.Message(content=f"Startup error: {exc}", author="System").send()
        return None


async def get_run_task_list() -> RunTaskList:
    run_task_list = cl.user_session.get(SESSION_TASK_LIST_KEY)
    if isinstance(run_task_list, RunTaskList):
        return run_task_list

    run_task_list = await RunTaskList.create()
    cl.user_session.set(SESSION_TASK_LIST_KEY, run_task_list)
    return run_task_list


def get_async_task_notifier(
    *,
    agent: Any,
    runtime: AgentRuntime,
    url_override: str | None,
) -> AsyncTaskNotifier | None:
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
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    run_task_list = await get_run_task_list()
    await run_task_list.show_ready()
    settings = AppSettings(
        reasoning_level=runtime.config.default_reasoning,
        thread_id=current_chainlit_thread_id(),
    )
    store_settings(settings)
    await build_chat_settings(settings).send()
    persistence_line = (
        "- Persistence: Postgres-backed LangGraph checkpoints and `/memories/`\n"
        if runtime.persistence_enabled
        else "- Persistence: in-memory only for this process; set `DATABASE_URL` to enable durable checkpoints and `/memories/`\n"
    )
    history_line = (
        "- History bar: enabled for authenticated users\n"
        if runtime.persistence_enabled and AUTH_ENABLED
        else "- History bar: disabled; set `DATABASE_URL`, `CHAINLIT_AUTH_SECRET`, `CHAINLIT_AUTH_USERNAME`, and `CHAINLIT_AUTH_PASSWORD` to enable native Chainlit history\n"
    )
    extensions = runtime.config.extensions
    extensions_line = (
        f"- Skill sources: `{len(extensions.skills)}`\n"
        f"- MCP servers: `{len(extensions.mcp_servers or {})}`\n"
        f"- Custom subagents: `{len(extensions.subagents)}`\n"
        f"- Async subagents: `{len(extensions.async_subagents)}`\n"
    )
    if extensions.config_path is not None:
        extensions_line += f"- Extensions config: `{extensions.config_path.name}`\n"
    await cl.Message(
        content=(
            "Workspace agent ready.\n\n"
            f"- Model provider: `{format_model_provider(runtime.config.model_provider)}`\n"
            f"- Model: `{runtime.config.model_name}`\n"
            f"- Thread ID: `{settings.thread_id}`\n"
            f"{persistence_line}"
            f"{history_line}"
            f"{extensions_line}"
            "- Real repo files live under `/workspace/`\n"
            "- Agent memory is available under `/memories/`"
        ),
        author="System",
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return

    run_task_list = await get_run_task_list()
    await run_task_list.show_ready()

    metadata = thread.get("metadata") or {}
    raw_settings = (
        metadata.get(SESSION_SETTINGS_KEY) if isinstance(metadata, dict) else None
    )
    settings = coerce_settings(raw_settings)
    store_settings(settings)
    await build_chat_settings(settings).send()
    async_url_override = async_subagent_url_override()
    agent = await runtime.get_agent(
        settings.reasoning_level,
        async_subagent_url_override=async_url_override,
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
    settings = coerce_settings(raw_settings)
    store_settings(settings)


@cl.action_callback(DOWNLOAD_MARKDOWN_ACTION)
async def download_response_markdown(action: cl.Action) -> None:
    await send_markdown_export(action)


@cl.action_callback(DOWNLOAD_PDF_ACTION)
async def download_response_pdf(action: cl.Action) -> None:
    await send_pdf_export(action)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    settings = coerce_settings(cl.user_session.get(SESSION_SETTINGS_KEY))
    runtime = await get_runtime_or_notify()
    if runtime is None:
        return
    run_task_list = await get_run_task_list()
    async_url_override = async_subagent_url_override()
    agent = await runtime.get_agent(
        settings.reasoning_level,
        async_subagent_url_override=async_url_override,
    )
    async_task_notifier = get_async_task_notifier(
        agent=agent,
        runtime=runtime,
        url_override=async_url_override,
    )
    bridge = ChainlitEventBridge(prompt=message.content, run_task_list=run_task_list)
    await bridge.start()

    config = {"configurable": {"thread_id": settings.thread_id}}
    payload = {"messages": [{"role": "user", "content": message.content}]}
    stream = agent.astream(
        payload,
        config=config,
        stream_mode=["messages", "updates"],
        subgraphs=True,
        version="v2",
    )

    try:
        while True:
            try:
                part = await anext(stream)
            except StopAsyncIteration:
                break
            await bridge.handle_part(part)
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
        return
    finally:
        with suppress(Exception):
            await stream.aclose()

    await bridge.finish()
    if async_task_notifier is not None:
        with suppress(Exception):
            await async_task_notifier.schedule_from_state(thread_id=settings.thread_id)


@cl.on_chat_end
async def on_chat_end() -> None:
    notifier = cl.user_session.get(SESSION_ASYNC_TASK_NOTIFIER_KEY)
    if isinstance(notifier, AsyncTaskNotifier):
        notifier.cancel()
