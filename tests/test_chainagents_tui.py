"""Test the Textual ChainAgents TUI."""

from __future__ import annotations

import asyncio
import io
import os
from types import SimpleNamespace
from typing import Any

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Input, Markdown, RichLog

import chainagents_cli
from chainagents_tui import (
    DEFAULT_TUI_THREAD_ID,
    ChainAgentsTuiApp,
    TUI_SIDE_PANEL_WIDTH,
    capture_mcp_stdio_stderr,
    run_tui,
)


class _Token:
    type = "AIMessageChunk"
    additional_kwargs: dict[str, str] = {}
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, content: str = "") -> None:
        self.content = content


class _ReasoningToken:
    type = "AIMessageChunk"
    content = ""
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, reasoning: str) -> None:
        self.additional_kwargs = {"reasoning_content": reasoning}


class _ToolCallChunkToken:
    type = "AIMessageChunk"
    content = ""
    additional_kwargs: dict[str, str] = {}

    def __init__(self, chunk: dict[str, str]) -> None:
        self.tool_call_chunks = [chunk]


class _ToolMessage:
    type = "tool"
    name = "read_file"
    status = "success"
    tool_call_id = "call-1"

    def __init__(self, content: str) -> None:
        self.content = content


def _raw_event(chunk: object) -> dict[str, object]:
    return {"event": "on_chain_stream", "data": {"chunk": chunk}}


def _args(**overrides: Any) -> SimpleNamespace:
    values = {
        "thread_id": None,
        "reasoning": None,
        "model": None,
        "async_subagent_url": None,
        "mcp_session_id": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeAgent:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.events = events
        self.payload: dict[str, Any] | None = None
        self.config: dict[str, Any] | None = None

    def astream_events(self, payload, *, config, version, stream_mode, subgraphs):
        self.payload = payload
        self.config = config

        async def events():
            for event in self.events:
                yield event

        return events()


class _QueuedAgent:
    def __init__(self, runs: list[list[dict[str, object]]]) -> None:
        self.runs = runs

    def astream_events(self, payload, *, config, version, stream_mode, subgraphs):
        events_for_run = self.runs.pop(0)

        async def events():
            for event in events_for_run:
                yield event

        return events()


class _BlockingAgent:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    def astream_events(self, payload, *, config, version, stream_mode, subgraphs):
        async def events():
            self.started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            if False:
                yield None

        return events()


class _FakeRuntime:
    def __init__(self, agent: _FakeAgent | _BlockingAgent) -> None:
        self.agent = agent
        self.config = SimpleNamespace(
            default_reasoning="medium",
            model_name="fake-model",
            recursion_limit=100,
        )
        self.commands: dict[str, Any] = {}

    async def get_agent(self, *args, **kwargs):
        return self.agent

    def resolve_chainlit_command(self, name: str):
        return self.commands.get(name)

    async def invoke_mcp_tool_command(self, **kwargs):
        return {"ok": True, "kwargs": kwargs}

    @property
    def chainlit_commands(self):
        return tuple(self.commands.values())


@pytest.mark.anyio
async def test_run_tui_leaves_app_stderr_visible(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    runtime = _FakeRuntime(_FakeAgent([]))
    runtime.project_root = tmp_path

    async def fake_run_async(self):
        os.write(2, b"Textual terminal output\n")
        return 0

    monkeypatch.setattr(ChainAgentsTuiApp, "run_async", fake_run_async)

    code = await run_tui(runtime, _args())

    assert code == 0
    assert "Textual terminal output" in capfd.readouterr().err
    log_path = tmp_path / ".files" / "tui-stderr.log"
    assert not log_path.exists() or "Textual terminal output" not in log_path.read_text(
        encoding="utf-8"
    )


def test_capture_mcp_stdio_stderr_routes_stdio_client_errlog(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_mcp_adapters import sessions

    captured_errlog = None

    def fake_stdio_client(server, errlog=None):
        nonlocal captured_errlog
        captured_errlog = errlog
        errlog.write("Secure MCP Filesystem Server running on stdio\n")
        errlog.flush()
        return "stdio-context"

    monkeypatch.setattr(sessions, "stdio_client", fake_stdio_client)
    log_path = tmp_path / ".files" / "tui-stderr.log"

    with capture_mcp_stdio_stderr(log_path):
        assert sessions.stdio_client("server") == "stdio-context"

    assert captured_errlog is not None
    assert "Secure MCP Filesystem Server running on stdio" in log_path.read_text(
        encoding="utf-8"
    )


def test_cli_parses_tui_flag() -> None:
    args = chainagents_cli.parse_args(["--tui"])

    assert args.tui is True
    assert args.thread_id is None


def test_tui_title_is_chainagents() -> None:
    app = ChainAgentsTuiApp(runtime=_FakeRuntime(_FakeAgent([])), args=_args())

    assert app.title == "ChainAgents"


def test_tui_side_panel_width_is_wide_enough() -> None:
    assert TUI_SIDE_PANEL_WIDTH == 56


@pytest.mark.anyio
async def test_cli_rejects_one_shot_prompt_in_tui_mode() -> None:
    args = chainagents_cli.parse_args(["--tui", "--prompt", "hello"])

    code = await chainagents_cli.run_cli(
        args,
        runtime=_FakeRuntime(_FakeAgent([])),  # type: ignore[arg-type]
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        stdin=io.StringIO("should not read"),
    )

    assert code == 2


@pytest.mark.anyio
async def test_cli_rejects_unsupported_one_shot_flags_in_tui_mode() -> None:
    args = chainagents_cli.parse_args(["--tui", "--json", "--status"])
    stderr = io.StringIO()

    code = await chainagents_cli.run_cli(
        args,
        runtime=_FakeRuntime(_FakeAgent([])),  # type: ignore[arg-type]
        stdout=io.StringIO(),
        stderr=stderr,
        stdin=io.StringIO(""),
    )

    assert code == 2
    assert "unsupported flags in TUI mode: --status, --json" in stderr.getvalue()


@pytest.mark.anyio
async def test_tui_mounts_expected_panes_and_prompt() -> None:
    app = ChainAgentsTuiApp(
        runtime=_FakeRuntime(_FakeAgent([])),
        args=chainagents_cli.parse_args(["--tui"]),
    )

    async with app.run_test():
        assert app.thread_id == DEFAULT_TUI_THREAD_ID
        assert app.query_one("#conversation", VerticalScroll)
        assert app.query_one("#reasoning", RichLog)
        assert app.query_one("#tools", RichLog)
        assert app.query_one("#prompt", Input)


@pytest.mark.anyio
async def test_tui_submits_prompt_and_streams_response() -> None:
    agent = _FakeAgent([
        _raw_event(((), "messages", (_Token("Hello from agent"), {}))),
    ])
    app = ChainAgentsTuiApp(runtime=_FakeRuntime(agent), args=_args())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "hello"
        await pilot.press("enter")
        await pilot.pause()

    assert agent.payload == {"messages": [{"role": "user", "content": "hello"}]}
    assert agent.config == {
        "configurable": {"thread_id": DEFAULT_TUI_THREAD_ID},
        "recursion_limit": 100,
    }
    assert app.conversation_entries == [
        ("You", "hello"),
        ("Assistant", "Hello from agent"),
    ]


@pytest.mark.anyio
async def test_tui_adds_configured_rubric_to_agent_payload() -> None:
    """Verify configured rubrics are passed through TUI agent invocations."""
    agent = _FakeAgent([])
    runtime = _FakeRuntime(agent)
    runtime.config.extensions = SimpleNamespace(
        rubric=SimpleNamespace(
            enabled=True,
            rubric="- The answer satisfies the configured TUI rubric.",
        )
    )
    app = ChainAgentsTuiApp(runtime=runtime, args=_args())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "hello"
        await pilot.press("enter")
        await pilot.pause()

    assert agent.payload == {
        "messages": [{"role": "user", "content": "hello"}],
        "rubric": "- The answer satisfies the configured TUI rubric.",
    }


@pytest.mark.anyio
async def test_tui_renders_assistant_response_as_markdown_widget() -> None:
    agent = _FakeAgent(
        [
            _raw_event(((), "messages", (_Token("## Heading\n\n"), {}))),
            _raw_event(((), "messages", (_Token("- item"), {}))),
        ]
    )
    app = ChainAgentsTuiApp(runtime=_FakeRuntime(agent), args=_args())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "format this"
        await pilot.press("enter")
        await pilot.pause()

        markdown = app.query_one("#conversation Markdown", Markdown)

    assert markdown.source == "## Heading\n\n- item"


@pytest.mark.anyio
async def test_tui_streams_reasoning_and_tool_activity_to_side_panes() -> None:
    agent = _FakeAgent(
        [
            _raw_event(((), "messages", (_ReasoningToken("thinking"), {}))),
            _raw_event(
                (
                    (),
                    "messages",
                    (
                        _ToolCallChunkToken(
                            {
                                "id": "call-1",
                                "name": "read_file",
                                "args": '{"path":"README.md"}',
                            }
                        ),
                        {},
                    ),
                )
            ),
            _raw_event(((), "messages", (_ToolMessage("file content"), {}))),
        ]
    )
    app = ChainAgentsTuiApp(runtime=_FakeRuntime(agent), args=_args())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "inspect"
        await pilot.press("enter")
        await pilot.pause()

    assert app.reasoning_entries == [("main-agent", "thinking")]
    assert app.tool_entries == [
        "main-agent read_file start {\"path\":\"README.md\"}",
        "main-agent read_file success file content",
    ]


@pytest.mark.anyio
async def test_tui_accumulates_reasoning_deltas_in_one_entry() -> None:
    agent = _FakeAgent(
        [
            _raw_event(((), "messages", (_ReasoningToken("thinking"), {}))),
            _raw_event(((), "messages", (_ReasoningToken("thinking through"), {}))),
            _raw_event(
                ((), "messages", (_ReasoningToken("thinking through details"), {}))
            ),
        ]
    )
    app = ChainAgentsTuiApp(runtime=_FakeRuntime(agent), args=_args())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "inspect"
        await pilot.press("enter")
        await pilot.pause()

    assert app.reasoning_entries == [("main-agent", "thinking through details")]


@pytest.mark.anyio
async def test_tui_starts_new_reasoning_entry_for_each_prompt() -> None:
    agent = _QueuedAgent(
        [
            [
                _raw_event(((), "messages", (_ReasoningToken("first"), {}))),
                _raw_event(((), "messages", (_ReasoningToken("first run"), {}))),
            ],
            [
                _raw_event(((), "messages", (_ReasoningToken("second"), {}))),
                _raw_event(((), "messages", (_ReasoningToken("second run"), {}))),
            ],
        ]
    )
    app = ChainAgentsTuiApp(runtime=_FakeRuntime(agent), args=_args())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "first"
        await pilot.press("enter")
        await pilot.pause()

        prompt.value = "second"
        await pilot.press("enter")
        await pilot.pause()

    assert app.reasoning_entries == [
        ("main-agent", "first run"),
        ("main-agent", "second run"),
    ]


@pytest.mark.anyio
async def test_tui_applies_prompt_slash_commands() -> None:
    agent = _FakeAgent([
        _raw_event(((), "messages", (_Token("summary"), {}))),
    ])
    runtime = _FakeRuntime(agent)
    runtime.commands["summarize"] = SimpleNamespace(
        name="summarize",
        description="Summarize",
        target="prompt",
        value="Summarize the input",
        template="Summarize this:\n{input}",
    )
    app = ChainAgentsTuiApp(runtime=runtime, args=_args())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/summarize notes"
        await pilot.press("enter")
        await pilot.pause()

    assert agent.payload == {
        "messages": [{"role": "user", "content": "Summarize this:\nnotes"}]
    }


@pytest.mark.anyio
async def test_tui_shows_filtered_slash_commands_while_typing() -> None:
    runtime = _FakeRuntime(_FakeAgent([]))
    runtime.commands["ask-researcher"] = SimpleNamespace(
        name="ask-researcher",
        description="Delegate to repo-researcher.",
        target="subagent",
        value="repo-researcher",
        template="{input}",
    )
    runtime.commands["summarize"] = SimpleNamespace(
        name="summarize",
        description="Summarize input.",
        target="prompt",
        value="Summarize",
        template="{input}",
    )
    app = ChainAgentsTuiApp(runtime=runtime, args=_args())

    async with app.run_test():
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/su"
        app.refresh_command_help(prompt.value)

        assert app.command_help_visible is True
        assert app.visible_command_names == ["summarize"]
        assert "/summarize - Summarize input." in app.command_help_text


@pytest.mark.anyio
async def test_tui_tab_completes_first_matching_slash_command() -> None:
    runtime = _FakeRuntime(_FakeAgent([]))
    runtime.commands["summarize"] = SimpleNamespace(
        name="summarize",
        description="Summarize input.",
        target="prompt",
        value="Summarize",
        template="{input}",
    )
    app = ChainAgentsTuiApp(runtime=runtime, args=_args())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "/su"
        app.refresh_command_help(prompt.value)

        await pilot.press("tab")

        assert prompt.value == "/summarize "
        assert app.command_help_visible is False


@pytest.mark.anyio
async def test_tui_ctrl_c_cancels_active_run() -> None:
    agent = _BlockingAgent()
    app = ChainAgentsTuiApp(runtime=_FakeRuntime(agent), args=_args())

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "wait"
        await pilot.press("enter")
        await agent.started.wait()

        assert prompt.disabled is True

        await app.action_cancel_or_quit()
        await pilot.pause()

        assert agent.cancelled is True
        assert prompt.disabled is False
        assert app.status_message == "Cancelled."
