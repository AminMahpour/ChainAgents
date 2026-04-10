from __future__ import annotations

import asyncio
import os
from typing import Any

import chainlit as cl
from langgraph_sdk import get_client

from deepagent_runtime import AsyncSubagentConfig


DEFAULT_POLL_SECONDS = 5.0
DEFAULT_AGENT_PROTOCOL_URL = "http://127.0.0.1:2024"
TERMINAL_STATUSES = {"success", "error", "cancelled", "interrupted", "timeout"}


def async_subagent_url_override() -> str | None:
    for key in (
        "CHAINLIT_ASYNC_SUBAGENT_URL",
        "LANGGRAPH_SERVER_URL",
        "LANGGRAPH_API_URL",
    ):
        value = os.getenv(key, "").strip()
        if value:
            return value.rstrip("/")
    return DEFAULT_AGENT_PROTOCOL_URL


def async_task_poll_seconds() -> float:
    try:
        seconds = float(os.getenv("CHAINLIT_ASYNC_TASK_POLL_SECONDS", "").strip())
    except ValueError:
        return DEFAULT_POLL_SECONDS
    return seconds if seconds > 0 else DEFAULT_POLL_SECONDS


def resolved_headers(spec: AsyncSubagentConfig) -> dict[str, str]:
    headers = dict(spec.headers or {})
    if "x-auth-scheme" not in headers:
        headers["x-auth-scheme"] = "langsmith"
    return headers


def task_values(snapshot: Any) -> dict[str, Any]:
    values = getattr(snapshot, "values", {}) or {}
    async_tasks = values.get("async_tasks") if isinstance(values, dict) else None
    return async_tasks if isinstance(async_tasks, dict) else {}


def result_from_thread_values(thread_values: dict[str, Any]) -> str:
    messages = thread_values.get("messages", [])
    if not messages:
        return "(completed with no output messages)"
    last = messages[-1]
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(getattr(last, "content", last))


def format_task_result(result: dict[str, Any]) -> str:
    status = result.get("status", "unknown")
    agent_name = result.get("agent_name", "async subagent")
    task_id = result.get("task_id", "")
    prefix = f"Async subagent `{agent_name}` finished with status `{status}`."
    if task_id:
        prefix += f"\n\nTask ID: `{task_id}`"

    if status == "success":
        return f"{prefix}\n\n{result.get('result', '')}".strip()
    if result.get("error"):
        return f"{prefix}\n\nError: {result['error']}"
    return prefix


class AsyncTaskNotifier:
    def __init__(
        self,
        *,
        agent: Any,
        async_subagents: tuple[AsyncSubagentConfig, ...],
        url_override: str | None,
    ) -> None:
        self.agent = agent
        self.async_subagents = {subagent.name: subagent for subagent in async_subagents}
        self.url_override = url_override
        self.poll_seconds = async_task_poll_seconds()
        self.watched_task_ids: set[str] = set()
        self.notified_task_ids: set[str] = set()
        self.tasks: set[asyncio.Task[Any]] = set()

    def matches(self, *, agent: Any, url_override: str | None) -> bool:
        return self.agent is agent and self.url_override == url_override

    async def schedule_from_state(self, *, thread_id: str) -> None:
        if not self.async_subagents:
            return

        snapshot = await self.agent.aget_state(
            {"configurable": {"thread_id": thread_id}},
        )
        for task_id, task in task_values(snapshot).items():
            if not isinstance(task, dict):
                continue
            status = str(task.get("status", "")).lower()
            if status in TERMINAL_STATUSES or task_id in self.watched_task_ids:
                continue
            self.watched_task_ids.add(task_id)
            monitor_task = asyncio.create_task(self._watch_task(dict(task)))
            self.tasks.add(monitor_task)
            monitor_task.add_done_callback(self.tasks.discard)

    def cancel(self) -> None:
        for task in self.tasks:
            task.cancel()
        self.tasks.clear()

    async def _watch_task(self, task: dict[str, Any]) -> None:
        task_id = str(task.get("task_id") or task.get("thread_id") or "").strip()
        agent_name = str(task.get("agent_name") or "").strip()
        run_id = str(task.get("run_id") or "").strip()
        thread_id = str(task.get("thread_id") or task_id).strip()
        if not task_id or not agent_name or not run_id or not thread_id:
            return

        subagent = self.async_subagents.get(agent_name)
        if subagent is None:
            return

        url = subagent.url or self.url_override
        if not url:
            await self._notify_once(
                task_id,
                (
                    f"Async task `{task_id}` is running, but Chainlit realtime "
                    "monitoring needs an Agent Protocol `url`. Omitted `url` uses "
                    "ASGI transport only inside LangGraph Agent Server."
                ),
            )
            return

        client = get_client(url=url, headers=resolved_headers(subagent))
        try:
            failures = 0
            while True:
                try:
                    run = await client.runs.get(thread_id=thread_id, run_id=run_id)
                    failures = 0
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    if failures >= 3:
                        await self._notify_once(
                            task_id,
                            f"Could not monitor async task `{task_id}`: {exc}",
                        )
                        return
                    await asyncio.sleep(self.poll_seconds)
                    continue

                status = str(run.get("status", "")).lower()
                if status in TERMINAL_STATUSES:
                    result: dict[str, Any] = {
                        "task_id": task_id,
                        "agent_name": agent_name,
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "status": status,
                    }
                    if status == "success":
                        thread = await client.threads.get(thread_id=thread_id)
                        values = thread.get("values") or {}
                        result["result"] = result_from_thread_values(values)
                    elif run.get("error"):
                        result["error"] = str(run["error"])
                    await self._notify_once(task_id, format_task_result(result))
                    return

                await asyncio.sleep(self.poll_seconds)
        finally:
            await client.aclose()

    async def _notify_once(self, task_id: str, content: str) -> None:
        if task_id in self.notified_task_ids:
            return
        self.notified_task_ids.add(task_id)
        await cl.Message(content=content, author="Async subagent").send()
