from __future__ import annotations

from types import SimpleNamespace

import main
import pytest


def test_resolve_native_command_prefers_explicit_slash_text() -> None:
    parsed = main.resolve_native_command(
        raw_text="/summarize hello world",
        selected_command="ask-researcher",
    )

    assert parsed == main.ParsedNativeCommand(
        command_name="summarize",
        raw_args="hello world",
    )


def test_resolve_native_command_uses_selected_command_input() -> None:
    parsed = main.resolve_native_command(
        raw_text="hello world",
        selected_command="summarize",
    )

    assert parsed == main.ParsedNativeCommand(
        command_name="summarize",
        raw_args="hello world",
    )


def test_resolve_native_command_returns_none_without_command() -> None:
    parsed = main.resolve_native_command(
        raw_text="hello world",
        selected_command=None,
    )

    assert parsed is None


class _DummyRuntime:
    def __init__(self) -> None:
        self.invocation: dict[str, str | None] | None = None
        self.command = SimpleNamespace(
            name="repo-readme",
            description="Read repository README",
            target="mcp_tool",
            value="repo_readme",
            template='{"path":"README.md"}',
            mcp_server="repo",
        )

    def resolve_chainlit_command(self, command_name: str):
        if command_name == self.command.name:
            return self.command
        return None

    async def invoke_mcp_tool_command(
        self,
        *,
        tool_name: str,
        raw_args: str,
        thread_id: str,
        server_name: str | None = None,
    ):
        self.invocation = {
            "tool_name": tool_name,
            "raw_args": raw_args,
            "thread_id": thread_id,
            "server_name": server_name,
        }
        return {"ok": True}


class _DummyMessage:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def send(self):
        return None


@pytest.mark.asyncio
async def test_handle_native_command_applies_template_for_mcp_tool(monkeypatch) -> None:
    runtime = _DummyRuntime()
    settings = SimpleNamespace(thread_id="thread-1")
    monkeypatch.setattr(main.cl, "Message", _DummyMessage)

    result = await main.handle_native_command(
        runtime=runtime,
        settings=settings,
        parsed=main.ParsedNativeCommand(command_name="repo-readme", raw_args=""),
    )

    assert result == ""
    assert runtime.invocation is not None
    assert runtime.invocation["raw_args"] == '{"path":"README.md"}'
