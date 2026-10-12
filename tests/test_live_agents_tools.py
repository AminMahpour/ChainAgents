"""Main-agent live agent tools and roster middleware."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import SystemMessage
from test_live_agents_worker import FakeGraph

from chainagents.runtime.live_agents.config import SpawnRequest
from chainagents.runtime.live_agents.manager import LiveAgentManager
from chainagents.runtime.live_agents.tools import (
    LIVE_AGENT_TOOL_NAMES,
    LiveAgentRosterMiddleware,
    create_live_agent_tools,
    validate_live_agent_tool_names,
)
from chainagents.runtime.messaging import MessageBroker
from chainagents.runtime.types import (
    LiveAgentAllowlist,
    LiveAgentsConfig,
    LiveAgentTemplate,
    MessagingConfig,
)

SESSION = "s"


def make_manager(**config: Any) -> LiveAgentManager:
    live_config = LiveAgentsConfig(
        enabled=True,
        idle_timeout_seconds=0,
        allow=LiveAgentAllowlist(tools=("read_file", "ls")),
        templates=(
            LiveAgentTemplate(
                name="researcher",
                description="Reads sources.",
                system_prompt="Research.",
                tools=("read_file",),
            ),
        ),
        **config,
    )
    broker = MessageBroker(MessagingConfig(enabled=True))

    async def factory(_spec, _session_id):
        return FakeGraph()

    return LiveAgentManager(live_config, broker, factory)


def tools_by_name(manager: LiveAgentManager) -> dict[str, Any]:
    return {
        t.name: t for t in create_live_agent_tools(manager, fixed_session_id=SESSION)
    }


RUNTIME = SimpleNamespace(config={})


@pytest.mark.anyio
async def test_spawn_tool_returns_resolved_spec():
    manager = make_manager()
    tools = tools_by_name(manager)
    try:
        result = await tools["spawn_live_agent"].coroutine(
            template="researcher", add_tools=["ls"], runtime=RUNTIME
        )
        assert result["name"] == "researcher-1"
        assert result["status"] == "idle"
        assert result["address"].startswith("live:researcher-1:")
        assert result["id"] in result["address"]
        assert result["resolved"] == {
            "model": None,
            "tools": ["read_file", "ls"],
            "skills": [],
            "mcp_servers": [],
        }
    finally:
        await manager.close()


@pytest.mark.anyio
async def test_spawn_tool_returns_error_for_disallowed_tool():
    manager = make_manager()
    tools = tools_by_name(manager)
    try:
        result = await tools["spawn_live_agent"].coroutine(
            template="researcher", add_tools=["write_file"], runtime=RUNTIME
        )
        assert set(result) == {"error"}
        assert "write_file" in result["error"]
        assert manager.list(SESSION) == []
    finally:
        await manager.close()


@pytest.mark.anyio
async def test_list_and_teardown_tools():
    manager = make_manager()
    tools = tools_by_name(manager)
    try:
        templates = tools["list_live_agent_templates"].func(runtime=RUNTIME)
        assert [t["name"] for t in templates["templates"]] == ["researcher"]
        assert templates["allow"]["tools"] == ["read_file", "ls"]

        await tools["spawn_live_agent"].coroutine(
            template="researcher", name="alpha", runtime=RUNTIME
        )
        listed = tools["list_live_agents"].func(runtime=RUNTIME)
        assert [(a["name"], a["status"]) for a in listed] == [("alpha", "idle")]

        stopped = await tools["teardown_live_agent"].coroutine(
            agent="alpha", reason="done", runtime=RUNTIME
        )
        assert stopped["name"] == "alpha"
        assert stopped["status"] == "stopped"
        assert stopped["stop_reason"] == "done"
        assert tools["list_live_agents"].func(runtime=RUNTIME) == []
        assert [
            a["status"]
            for a in tools["list_live_agents"].func(
                include_stopped=True, runtime=RUNTIME
            )
        ] == ["stopped"]

        missing = await tools["teardown_live_agent"].coroutine(
            agent="ghost", runtime=RUNTIME
        )
        assert set(missing) == {"error"}
        assert "ghost" in missing["error"]
    finally:
        await manager.close()


def test_tool_names_match_reserved_names():
    manager = make_manager()
    assert {t.name for t in create_live_agent_tools(manager)} == LIVE_AGENT_TOOL_NAMES


@pytest.mark.parametrize("name", sorted(LIVE_AGENT_TOOL_NAMES))
def test_reserved_tool_name_collision_raises(name):
    with pytest.raises(ValueError, match=name):
        validate_live_agent_tool_names([SimpleNamespace(name=name)])
    validate_live_agent_tool_names([SimpleNamespace(name="other_tool")])


@pytest.mark.anyio
async def test_roster_middleware_injects_roster_only_when_live():
    manager = make_manager()
    middleware = LiveAgentRosterMiddleware(manager, fixed_session_id=SESSION)
    seen: list[ModelRequest] = []

    def handler(request):
        seen.append(request)
        return "response"

    async def ahandler(request):
        seen.append(request)
        return "response"

    request = ModelRequest(
        model=FakeListChatModel(responses=["x"]),
        messages=[],
        system_message=SystemMessage(content="Base prompt."),
        runtime=SimpleNamespace(config={}),
    )
    try:
        assert middleware.wrap_model_call(request, handler) == "response"
        assert seen[-1] is request

        await manager.spawn(
            SESSION,
            SpawnRequest(template="researcher"),
        )
        middleware.wrap_model_call(request, handler)
        text = seen[-1].system_message.text
        assert text.startswith("Base prompt.")
        assert "Live agents: researcher-1: idle" in text

        await middleware.awrap_model_call(request, ahandler)
        assert "Live agents: researcher-1: idle" in seen[-1].system_message.text
        assert request.system_message.text == "Base prompt."

        await manager.teardown(SESSION, "researcher-1")
        await middleware.awrap_model_call(request, ahandler)
        assert seen[-1] is request
    finally:
        await manager.close()
