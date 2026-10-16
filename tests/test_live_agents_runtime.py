"""Live agents wired into the main agent and the runtime lifecycle."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field
from test_deepagent_runtime_rag import make_runtime_config
from test_live_agents_worker import FakeGraph

import chainagents.runtime.lifecycle as core
from chainagents.runtime.artifacts import LargeToolResultArtifactRegistry
from chainagents.runtime.graph import build_agent_kwargs
from chainagents.runtime.live_agents.config import SpawnRequest
from chainagents.runtime.live_agents.manager import LiveAgentManager
from chainagents.runtime.live_agents.tools import LIVE_AGENT_TOOL_NAMES, LIVE_AGENTS_PROMPT
from chainagents.runtime.messaging import MessageBroker
from chainagents.runtime.middleware import (
    create_deep_agent_with_configured_summarization,
)
from chainagents.runtime.types import (
    ExtensionsConfig,
    LiveAgentsConfig,
    LiveAgentTemplate,
    MessagingConfig,
    ModelDefaults,
)

TEMPLATES = (
    LiveAgentTemplate(
        name="researcher", description="Reads sources.", system_prompt="Research."
    ),
)


class RecordingModel(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)
    bound: list = Field(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bound.extend(getattr(item, "name", None) for item in tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


def _config(tmp_path, *, live: bool):
    return replace(
        make_runtime_config(
            tmp_path,
            extensions=ExtensionsConfig(
                config_path=None,
                messaging=MessagingConfig(enabled=True),
                live_agents=LiveAgentsConfig(
                    enabled=live, idle_timeout_seconds=0, templates=TEMPLATES
                ),
            ),
        ),
        rag=None,
        rag_requested=False,
        agent_state="stateless",
    )


def _kwargs(config, tmp_path, model, broker, manager=None):
    return build_agent_kwargs(
        config,
        tools=[],
        model_profile=ModelDefaults(provider="ollama", name="fake"),
        reasoning_level="medium",
        reasoning_level_is_explicit=False,
        system_prompt="You are helpful.",
        custom_instruction=None,
        rag_enabled=False,
        project_root=tmp_path,
        artifact_registry=LargeToolResultArtifactRegistry(),
        include_async_subagents=False,
        build_model=lambda _level, _profile: model,
        session_id="s",
        messaging_broker=broker,
        live_agent_manager=manager,
    )


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


def test_tools_absent_when_disabled(tmp_path):
    config = _config(tmp_path, live=False)
    broker = MessageBroker(config.extensions.messaging)
    model = RecordingModel(responses=[AIMessage(content="done")])
    kwargs = _kwargs(config, tmp_path, model, broker)
    names = {getattr(t, "name", None) for t in kwargs["tools"] or []}
    assert not names & LIVE_AGENT_TOOL_NAMES
    assert LIVE_AGENTS_PROMPT not in kwargs["system_prompt"]


@pytest.mark.anyio
async def test_end_to_end_spawn_message_reply_teardown(tmp_path):
    config = _config(tmp_path, live=True)
    broker = MessageBroker(config.extensions.messaging)
    released = asyncio.Event()

    async def reply_after_main_turn() -> str:
        # Hold the reply until the main turn ends so it arrives asynchronously.
        await released.wait()
        return "pong"

    live_graph = FakeGraph([reply_after_main_turn])

    async def factory(_spec, _session_id):
        return live_graph

    manager = LiveAgentManager(config.extensions.live_agents, broker, factory)
    model = RecordingModel(
        responses=[
            _call("spawn_live_agent", {"template": "researcher"}, "c1"),
            _call(
                "send_agent_message",
                {"recipient": "researcher-1", "message": "ping"},
                "c2",
            ),
            AIMessage(content="Asked the researcher."),
            _call("teardown_live_agent", {"agent": "researcher-1"}, "c3"),
            AIMessage(content="All done."),
        ]
    )
    kwargs = _kwargs(config, tmp_path, model, broker, manager)
    assert LIVE_AGENTS_PROMPT in kwargs["system_prompt"]
    graph = create_deep_agent_with_configured_summarization(config, **kwargs)
    run_config = {"configurable": {"thread_id": "s"}}
    try:
        await graph.ainvoke(
            {"messages": [HumanMessage(content="Start a researcher")]}, run_config
        )
        assert set(model.bound) >= LIVE_AGENT_TOOL_NAMES
        record = manager.get("s", "researcher-1")
        assert not broker.pending("s", "main")
        released.set()

        async def reply_arrives() -> None:
            while not broker.pending("s", "main"):  # noqa: ASYNC110
                await asyncio.sleep(0.005)

        await asyncio.wait_for(reply_arrives(), 2)
        assert "ping" in str(live_graph.calls)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Any news?")]}, run_config
        )
        replies = [
            m.content
            for m in model.seen[3]
            if isinstance(m, HumanMessage)
            and m.content.startswith(f"Message from {record.address}")
        ]
        assert len(replies) == 1 and "researcher-1" in replies[0]
        assert replies[0].endswith("pong")
        assert "Live agents: researcher-1: idle" in str(model.seen[3][0].content)
        assert record.status == "stopped"
        assert result["messages"][-1].content == "All done."
    finally:
        await manager.close()


@pytest.mark.anyio
async def test_close_conversation_tears_down_live_agents(tmp_path):
    config = replace(_config(tmp_path, live=True), agent_state="stateful")
    runtime = core.AgentRuntime(config, project_root=tmp_path)
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    async def stub_graph(_spec, _session_id):
        return FakeGraph()

    runtime._build_live_agent_graph = stub_graph
    assert runtime.live_agents is not None
    records = [
        (await runtime.live_agents.spawn("s", SpawnRequest(template="researcher")))[0]
        for _ in range(2)
    ]
    statuses_at_close: list[list[str]] = []
    close_session = runtime.message_broker.close_session

    def spy(session_id: str) -> None:
        statuses_at_close.append([r.status for r in records])
        close_session(session_id)

    runtime.message_broker.close_session = spy
    try:
        await runtime.close_conversation(thread_id="s")
        assert [r.status for r in records] == ["stopped", "stopped"]
        assert statuses_at_close == [["stopped", "stopped"]]
        assert "s" not in runtime.message_broker._participants
        assert runtime.live_agents.list("s", include_stopped=True) == []
    finally:
        await runtime.close()


def test_runtime_live_agents_is_none_when_disabled(tmp_path):
    runtime = core.AgentRuntime(_config(tmp_path, live=False), project_root=tmp_path)
    assert runtime.live_agents is None


@pytest.mark.anyio
async def test_runtime_builds_live_graph_from_spec(tmp_path, monkeypatch):
    config = replace(_config(tmp_path, live=True), agent_state="stateful")
    runtime = core.AgentRuntime(config, project_root=tmp_path)
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()
    built: dict[str, Any] = {}
    monkeypatch.setattr(
        runtime, "_build_model", lambda level, **kw: ("model", level, kw)
    )
    monkeypatch.setattr(
        core.runtime_graph,
        "build_live_agent_graph",
        lambda _config, **kwargs: built.update(kwargs) or "graph",
    )
    spec = (
        await runtime.live_agents.spawn("s", SpawnRequest(template="researcher"))
    )[1]
    try:
        assert built["spec"] == spec
        assert built["session_id"] == "s"
        assert built["mcp_tools"] == []
        assert built["broker"] is runtime.message_broker
        assert built["checkpointer"] is runtime.checkpointer
        assert built["store"] is runtime.store
        assert built["model"][1] == built["reasoning_level"]
    finally:
        await runtime.close()


class PerCallRecordingModel(FakeMessagesListChatModel):
    bindings: list = Field(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        self.bindings.append({getattr(item, "name", None) for item in tools})
        return self


@pytest.mark.anyio
async def test_general_purpose_delegate_lacks_live_agent_tools(tmp_path):
    config = _config(tmp_path, live=True)
    broker = MessageBroker(config.extensions.messaging)

    async def factory(_spec, _session_id):
        return FakeGraph()

    manager = LiveAgentManager(config.extensions.live_agents, broker, factory)
    model = PerCallRecordingModel(
        responses=[
            _call(
                "task",
                {"description": "look around", "subagent_type": "general-purpose"},
                "c1",
            ),
            AIMessage(content="delegate done"),
            AIMessage(content="main done"),
        ]
    )
    kwargs = _kwargs(config, tmp_path, model, broker, manager)
    graph = create_deep_agent_with_configured_summarization(config, **kwargs)
    try:
        await graph.ainvoke(
            {"messages": [HumanMessage(content="delegate")]},
            {"configurable": {"thread_id": "s"}},
        )
        main_bindings = [b for b in model.bindings if "task" in b]
        delegate_bindings = [b for b in model.bindings if "task" not in b]
        assert main_bindings and all(
            b >= LIVE_AGENT_TOOL_NAMES for b in main_bindings
        )
        assert delegate_bindings
        assert all(not b & LIVE_AGENT_TOOL_NAMES for b in delegate_bindings)
        assert all("send_agent_message" in b for b in delegate_bindings)
    finally:
        await manager.close()


@pytest.mark.anyio
async def test_stateless_runtime_gives_live_agents_a_shared_checkpointer(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, live=True)
    assert config.agent_state == "stateless"
    runtime = core.AgentRuntime(config, project_root=tmp_path)
    model = RecordingModel(
        responses=[AIMessage(content="first"), AIMessage(content="second")]
    )
    monkeypatch.setattr(runtime, "_build_model", lambda level, **kw: model)
    built: list[Any] = []
    real_build = core.runtime_graph.build_live_agent_graph

    def spy(*args, **kwargs):
        built.append(kwargs["checkpointer"])
        return real_build(*args, **kwargs)

    monkeypatch.setattr(core.runtime_graph, "build_live_agent_graph", spy)
    try:
        _record, spec = await runtime.live_agents.spawn(
            "s", SpawnRequest(template="researcher")
        )
        graph_one = await runtime._build_live_agent_graph(spec, "s")
        graph_two = await runtime._build_live_agent_graph(spec, "s")
        assert built[0] is not None and built[0] is built[1]
        run_config = {"configurable": {"thread_id": "live-thread"}}
        await graph_one.ainvoke(
            {"messages": [HumanMessage(content="remember zebra")]}, run_config
        )
        await graph_two.ainvoke(
            {"messages": [HumanMessage(content="what animal?")]}, run_config
        )
        assert "remember zebra" in str(model.seen[-1])
        await runtime._delete_live_agent_thread("live-thread")
        assert built[0].get_tuple(run_config) is None
    finally:
        await runtime.close()


@pytest.mark.anyio
async def test_runtime_live_agent_turns_keep_artifact_scope_and_langfuse(
    tmp_path, monkeypatch
):
    """Isolated live agent workers still get the conversation's scope and tracing."""
    config = replace(
        _config(tmp_path, live=True),
        agent_state="stateful",
        langfuse=replace(make_runtime_config(tmp_path).langfuse, enabled=True),
    )
    tracer = object()
    monkeypatch.setattr(
        core.runtime_tracing, "build_langfuse_callback_handler", lambda _config: tracer
    )
    runtime = core.AgentRuntime(config, project_root=tmp_path)
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()
    artifacts = runtime.large_tool_result_artifacts
    seen: list[object] = []

    async def capture() -> str:
        seen.append(artifacts.current_handle())
        return "findings"

    graph = FakeGraph([capture])

    async def stub_graph(_spec, _session_id):
        return graph

    runtime._build_live_agent_graph = stub_graph
    assert runtime.live_agents is not None
    try:
        await runtime.live_agents.spawn(
            "s", SpawnRequest(template="researcher", initial_message="look")
        )
        for _ in range(200):
            if graph.calls:
                break
            await asyncio.sleep(0.01)
        assert seen == [artifacts.open_session("s")]
        assert graph.calls[0][1]["callbacks"] == [tracer]
    finally:
        await runtime.close()
