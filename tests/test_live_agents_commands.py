"""Tests for the /agents, /spawn, /msg and /kill slash commands."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_live_agents_manager import SESSION, Harness, until
from test_live_agents_worker import FakeGraph

from chainagents.commands.native import ParsedNativeCommand, resolve_runtime_command
from chainagents.runtime.commands import build_chainlit_command_catalog
from chainagents.runtime.backends import build_deepagent_backend
from chainagents.runtime.config import RuntimeConfig
from chainagents.runtime.live_agents.commands import (
    format_idle,
    handle_live_agent_command,
)
from chainagents.runtime.types import ExtensionsConfig, LiveAgentsConfig
from chainagents.turns.renderer import BaseTurnRenderer
from chainagents.turns.runner import TurnRequest, TurnRunner


async def run(h: Harness, name: str, args: str = "") -> str:
    return await handle_live_agent_command(
        h.manager, command_name=name, raw_args=args, session_id=SESSION
    )


@pytest.mark.anyio
async def test_agents_empty() -> None:
    h = Harness()
    assert await run(h, "agents") == "No live agents."


@pytest.mark.anyio
async def test_agents_table() -> None:
    h = Harness()
    await run(h, "spawn", "researcher")
    h.now += 125
    table = await run(h, "agents")
    # manager.now() uses the harness clock, not time.monotonic
    lines = table.splitlines()
    assert lines[0] == "| name | template | status | turns | idle |"
    assert lines[2] == "| researcher-1 | researcher | idle | 0 | 2m |"
    await h.manager.close()


def test_format_idle() -> None:
    assert [format_idle(s) for s in (42, 180, 3900, 7200)] == [
        "42s",
        "3m",
        "1h5m",
        "2h",
    ]


@pytest.mark.anyio
async def test_spawn_parses_name_and_instructions() -> None:
    h = Harness()
    out = await run(h, "spawn", "coder fixer -- be terse -- really")
    assert out.startswith("Spawned `fixer` (coder")
    spec = h.built[0][0]
    assert spec.name == "fixer"
    assert "be terse -- really" in spec.system_prompt
    await h.manager.close()


@pytest.mark.anyio
async def test_spawn_usage_error() -> None:
    h = Harness()
    for args in ("", "a b c", "-- only"):
        with pytest.raises(ValueError, match="Usage: /spawn"):
            await run(h, "spawn", args)
    with pytest.raises(ValueError) as info:
        await run(h, "spawn", "nosuchtemplate")
    assert "Usage:" not in str(info.value)  # manager rejections are not usage errors


@pytest.mark.anyio
async def test_msg_routes_to_agent_as_user() -> None:
    h = Harness(graphs=[FakeGraph(["pong"])])
    await run(h, "spawn", "researcher bob")
    out = await run(h, "msg", "bob hello there")
    assert out == "Sent to `bob`. The reply will appear when ready."
    await until(lambda: h.manager.list(SESSION)[0]["turns"] == 1)
    graph = h.built[0][2]
    assert "hello there" in graph.calls[0][0]["messages"][0].content
    assert "from user" in graph.calls[0][0]["messages"][0].content
    await h.manager.close()


@pytest.mark.anyio
async def test_kill_one_and_all() -> None:
    h = Harness()
    await run(h, "spawn", "researcher a")
    await run(h, "spawn", "researcher b")
    await run(h, "spawn", "coder c")
    assert await run(h, "kill", "a") == "Stopped `a`."
    assert [r["name"] for r in h.manager.list(SESSION)] == ["b", "c"]
    assert await run(h, "kill", "all") == "Stopped 2 agents."
    await run(h, "spawn", "coder d")
    assert await run(h, "kill", "all") == "Stopped 1 agent."
    assert await run(h, "kill", "all") == "Stopped 0 agents."
    await h.manager.close()


@pytest.mark.anyio
async def test_unknown_agent_is_value_error() -> None:
    h = Harness()
    with pytest.raises(ValueError, match="Unknown live agent `ghost`"):
        await run(h, "msg", "ghost hi")
    with pytest.raises(ValueError, match="Usage: /kill"):
        await run(h, "kill", "ghost")
    with pytest.raises(ValueError, match="Usage: /msg"):
        await run(h, "msg", "ghost")


@pytest.mark.anyio
async def test_fyi_sent_to_main_when_enabled() -> None:
    h = Harness(graphs=[FakeGraph(["pong"])])
    await run(h, "spawn", "coder coder-2")
    assert h.inbox("main") == ["FYI: user spawned `coder-2` (template coder)."]
    await run(h, "msg", "coder-2 ping")
    await until(lambda: len(h.inbox("main")) == 3)
    assert h.inbox("main")[1] == "FYI: user sent a message to `coder-2`: ping"
    assert h.inbox("main")[2] == "FYI: `coder-2` replied to user: pong"
    await run(h, "kill", "coder-2")
    assert h.inbox("main")[-1] == "FYI: user stopped `coder-2`."
    await h.manager.close()


@pytest.mark.anyio
async def test_no_fyi_when_disabled() -> None:
    h = Harness(graphs=[FakeGraph(["pong"])], notify_main_on_user_actions=False)
    await run(h, "spawn", "coder c")
    await run(h, "msg", "c ping")
    await until(lambda: h.manager.list(SESSION)[0]["turns"] == 1)
    await run(h, "kill", "c")
    assert h.inbox("main") == []
    await h.manager.close()


@pytest.mark.anyio
async def test_reply_forwarding_survives_missing_main_and_stops_on_close() -> None:
    h = Harness(graphs=[FakeGraph(["pong"])])
    await run(h, "spawn", "coder c")
    h.broker.close(SESSION, "main")
    await run(h, "msg", "c ping")  # FYI to closed main is logged, not raised
    await until(lambda: h.manager.list(SESSION)[0]["turns"] == 1)
    await h.manager.close()
    h.broker.open(SESSION, "main", name="main", parent=None)
    from chainagents.runtime.live_agents.events import LiveAgentEvent

    h.events.publish(
        LiveAgentEvent("message", SESSION, "x", "c", {"to": "user", "body": "late"})
    )
    assert h.inbox("main") == []


@pytest.mark.anyio
async def test_resolve_runtime_command_returns_live_agent_target() -> None:
    h = Harness()
    runtime = SimpleNamespace(live_agents=h.manager)
    result = await resolve_runtime_command(
        runtime=runtime,
        parsed=ParsedNativeCommand("agents", ""),
        thread_id=SESSION,
    )
    assert result.target == "live_agent"
    assert result.command_name == "agents"
    assert result.tool_result == "No live agents."


@pytest.mark.anyio
async def test_disabled_commands_fall_through_to_catalog() -> None:
    runtime = SimpleNamespace(live_agents=None, resolve_chainlit_command=lambda n: None)
    result = await resolve_runtime_command(
        runtime=runtime,
        parsed=ParsedNativeCommand("kill", "all"),
        thread_id=SESSION,
    )
    assert result.target == "unknown"


class _Renderer(BaseTurnRenderer):
    def __init__(self) -> None:
        self.command_results: list[Any] = []
        self.errors: list[Any] = []

    async def on_command_result(self, command_result: Any) -> None:
        self.command_results.append(command_result)

    async def on_command_error(self, error: Any, status: int) -> None:
        self.errors.append((error, status))


class _Runtime:
    config = RuntimeConfig.from_env()

    def __init__(self, manager: Any) -> None:
        self.live_agents = manager

    async def get_agent(self, *_a: Any, **_k: Any) -> Any:
        raise AssertionError("the agent must not be built for live agent commands")


@pytest.mark.anyio
async def test_runner_renders_live_agent_command_without_running_agent() -> None:
    h = Harness()
    renderer = _Renderer()
    request = TurnRequest(
        prompt="/agents",
        thread_id=SESSION,
        model_name="fake",
        reasoning_level="medium",
    )
    result = await TurnRunner(_Runtime(h.manager)).run(request, renderer)
    assert result.status == "completed"
    assert result.response == "No live agents."
    assert renderer.command_results[0].target == "live_agent"

    bad = replace(request, prompt="/kill")
    result = await TurnRunner(_Runtime(h.manager)).run(bad, renderer)
    assert result.status == "command_error"
    assert renderer.errors[0][1] == 422


def test_skill_command_named_kill_hidden_with_note(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "kill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: kill\ndescription: Kill things.\n---\n\n# kill\n",
        encoding="utf-8",
    )

    def build(enabled: bool):
        extensions = ExtensionsConfig(
            config_path=None,
            skills=("/workspace/skills/",),
            live_agents=LiveAgentsConfig(enabled=enabled),
        )
        return build_chainlit_command_catalog(
            extensions,
            backend=build_deepagent_backend(project_root=tmp_path),
            project_root=tmp_path,
        )

    commands, notes = build(True)
    assert commands == ()
    assert len(notes) == 1 and "`/kill`" in notes[0]
    commands, notes = build(False)
    assert [c.name for c in commands] == ["kill"]
    assert notes == ()


def test_native_command_specs_include_live_agent_commands_when_enabled() -> None:
    from chainagents.interfaces.chainlit.app import build_native_command_specs

    on = SimpleNamespace(chainlit_commands=(), live_agents=object())
    off = SimpleNamespace(chainlit_commands=(), live_agents=None)
    assert [s["id"] for s in build_native_command_specs(on)] == [
        "agents",
        "spawn",
        "msg",
        "kill",
    ]
    assert build_native_command_specs(off) == []


def test_renderers_emit_live_agent_markdown_raw() -> None:
    import asyncio
    import io

    from chainagents.commands.native import RuntimeCommandResult
    from chainagents.interfaces.chainlit.renderer import native_command_output_message
    from chainagents.interfaces.cli.render import CliEventRenderer

    text = "| name | status |\n| --- | --- |\n| a | idle |"
    result = RuntimeCommandResult(
        target="live_agent", command_name="agents", tool_result=text
    )
    assert native_command_output_message(result) == text

    out = io.StringIO()
    renderer = CliEventRenderer(
        stdout=out,
        stderr=io.StringIO(),
        stream=False,
        json_output=False,
        show_reasoning=False,
        show_tools=False,
    )
    asyncio.run(renderer.on_command_result(result))
    assert out.getvalue() == text + "\n"
