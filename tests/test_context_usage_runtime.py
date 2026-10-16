"""Context usage measurement wired into every agent the runtime builds."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import chainagents.runtime.core as deepagent_runtime
import chainagents.runtime.models as runtime_models
from chainagents.events.stream import AGENT_STREAM_MODES, AgentStreamEventAdapter
from chainagents.runtime.live_agents.config import SpawnRequest

OLLAMA_MODEL = """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "qwen3:32b"
context_window = 32768

[rag]
enabled = false
"""

SCOUT = """
[[subagents]]
name = "scout"
description = "Small helper"
system_prompt = "Help."
"""

USAGE = {"input_tokens": 1200, "output_tokens": 10, "total_tokens": 1210}


def _ai(content: str = "", **kwargs: Any) -> AIMessage:
    return AIMessage(content=content, usage_metadata=USAGE, **kwargs)


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return _ai(tool_calls=[{"name": name, "args": args, "id": call_id}])


class RoutedModel(FakeMessagesListChatModel):
    """Answers each call from the conversation it sees, so agents can interleave."""

    route: Callable[[list[BaseMessage]], AIMessage] = Field(exclude=True)

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
        return ChatResult(generations=[ChatGeneration(message=self.route(list(messages)))])


def _first_human(messages: list[BaseMessage]) -> str:
    return next(str(m.content) for m in messages if isinstance(m, HumanMessage))


def _route(messages: list[BaseMessage]) -> AIMessage:
    if isinstance(messages[-1], ToolMessage):
        return _ai("main done")
    prompt = _first_human(messages)
    if prompt == "delegate":
        return _call("task", {"description": "look around", "subagent_type": "scout"}, "c1")
    if prompt == "background":
        return _call(
            "spawn_background_task",
            {"description": "look around", "subagent_type": "scout"},
            "c1",
        )
    return _ai("reply")


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> RoutedModel:
    model = RoutedModel(responses=[], route=_route)
    monkeypatch.setattr(runtime_models, "build_model", lambda *args, **kwargs: model)
    return model


async def _runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toml: str
) -> deepagent_runtime.AgentRuntime:
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(toml.strip() + "\n", encoding="utf-8")
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    return await deepagent_runtime.AgentRuntime.create(project_root=tmp_path)


async def _turn(
    runtime: deepagent_runtime.AgentRuntime, thread_id: str, prompt: str
) -> list[Any]:
    """Run one main-agent turn the way the turn runner streams it."""
    agent = await runtime.get_agent("medium", thread_id=thread_id)
    adapter = AgentStreamEventAdapter(prompt=prompt)
    events: list[Any] = []
    async for raw_event in agent.astream_events(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": thread_id}},
        version="v2",
        stream_mode=AGENT_STREAM_MODES,
        subgraphs=True,
    ):
        events.extend(adapter.events_from_raw_event(raw_event))
    return events


async def _until(predicate: Callable[[], bool]) -> None:
    async def poll() -> None:
        while not predicate():  # noqa: ASYNC110
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), 5)


@pytest.mark.anyio
async def test_main_and_subagent_snapshots_reach_the_registry(
    tmp_path, monkeypatch, fake_model
) -> None:
    runtime = await _runtime(tmp_path, monkeypatch, OLLAMA_MODEL + SCOUT)
    try:
        events = await _turn(runtime, "thread-1", "delegate")

        snapshots = {item.agent: item for item in runtime.context_usage.list("thread-1")}
        assert set(snapshots) == {"main", "scout"}
        assert snapshots["main"].kind == "main"
        assert snapshots["scout"].kind == "subagent"
        assert snapshots["main"].used == 1200 and snapshots["main"].exact
        usage_events = [event for event in events if event.kind == "context_usage"]
        assert usage_events
        assert {event.context_usage["agent"] for event in usage_events} == {"main"}
    finally:
        await runtime.close()


@pytest.mark.anyio
async def test_concurrent_threads_record_separately(
    tmp_path, monkeypatch, fake_model
) -> None:
    runtime = await _runtime(tmp_path, monkeypatch, OLLAMA_MODEL)
    try:
        await asyncio.gather(
            _turn(runtime, "thread-a", "hello"), _turn(runtime, "thread-b", "hi")
        )

        for thread_id in ("thread-a", "thread-b"):
            snapshots = runtime.context_usage.list(thread_id)
            assert [(item.agent, item.kind) for item in snapshots] == [("main", "main")]
    finally:
        await runtime.close()


@pytest.mark.anyio
async def test_close_conversation_clears_context_usage(
    tmp_path, monkeypatch, fake_model
) -> None:
    runtime = await _runtime(tmp_path, monkeypatch, OLLAMA_MODEL)
    try:
        await _turn(runtime, "thread-1", "hello")
        await _turn(runtime, "thread-2", "hello")
        assert runtime.context_usage.list("thread-1")

        await runtime.close_conversation(thread_id="thread-1")

        assert runtime.context_usage.list("thread-1") == []
        assert runtime.context_usage.list("thread-2")
    finally:
        await runtime.close()


@pytest.mark.anyio
async def test_background_task_snapshot_belongs_to_owning_session_and_finishes(
    tmp_path, monkeypatch, fake_model
) -> None:
    toml = (
        OLLAMA_MODEL
        + """
[agent.background_subagents]
enabled = true
"""
        + SCOUT
        + "background = true\n"
    )
    runtime = await _runtime(tmp_path, monkeypatch, toml)
    try:
        await _turn(runtime, "thread-1", "background")

        def background_entries() -> list[Any]:
            return [
                item
                for item in runtime.context_usage.list("thread-1")
                if item.kind == "background"
            ]

        await _until(
            lambda: any(item.status == "finished" for item in background_entries())
        )
        [entry] = background_entries()
        assert entry.agent == "scout"
        assert entry.used == 1200
        assert runtime.context_usage.list("thread-1")[0].agent == "main"
    finally:
        await runtime.close()


@pytest.mark.anyio
async def test_live_agent_snapshot_is_marked_stopped(
    tmp_path, monkeypatch, fake_model
) -> None:
    toml = (
        OLLAMA_MODEL
        + """
[agent.messaging]
enabled = true

[agent.live_agents]
enabled = true
idle_timeout_seconds = 0

[[agent.live_agents.templates]]
name = "researcher"
description = "Reads sources."
system_prompt = "Research."
"""
    )
    runtime = await _runtime(tmp_path, monkeypatch, toml)
    assert runtime.live_agents is not None
    try:
        record, _spec = await runtime.live_agents.spawn(
            "thread-1", SpawnRequest(template="researcher")
        )
        runtime.live_agents.send_from_user("thread-1", record.id, "ping")
        await _until(lambda: bool(runtime.context_usage.list("thread-1")))
        [entry] = runtime.context_usage.list("thread-1")
        assert (entry.agent, entry.kind, entry.status) == (record.name, "live", "active")

        await runtime.live_agents.teardown("thread-1", record.id)

        [entry] = runtime.context_usage.list("thread-1")
        assert entry.status == "stopped"
    finally:
        await runtime.close()
