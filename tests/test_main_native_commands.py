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


def test_resolve_reasoning_level_for_message_defaults_to_settings() -> None:
    message = SimpleNamespace(content="hello")
    settings = SimpleNamespace(reasoning_level="medium")

    resolved = main.resolve_reasoning_level_for_message(message, settings)

    assert resolved == "medium"


def test_resolve_reasoning_level_for_message_uses_mode_override() -> None:
    message = SimpleNamespace(content="hello", modes={"reasoning_level": "high"})
    settings = SimpleNamespace(reasoning_level="medium")

    resolved = main.resolve_reasoning_level_for_message(message, settings)

    assert resolved == "high"


def test_resolve_reasoning_level_for_message_falls_back_to_settings_default() -> None:
    message = SimpleNamespace(content="hello", modes={})
    settings = SimpleNamespace(reasoning_level="low")

    resolved = main.resolve_reasoning_level_for_message(message, settings)

    assert resolved == "low"


def test_resolve_reasoning_level_for_message_ignores_override_when_disabled() -> None:
    message = SimpleNamespace(content="hello", modes={"reasoning_level": "high"})
    settings = SimpleNamespace(reasoning_level="low")

    resolved = main.resolve_reasoning_level_for_message(
        message,
        settings,
        reasoning_mode_enabled=False,
    )

    assert resolved == "low"


def test_resolve_model_name_for_message_uses_mode_override() -> None:
    message = SimpleNamespace(content="hello", modes={"model_name": "gemma4:27b"})
    settings = SimpleNamespace(model_name="gpt-oss:20b")

    resolved = main.resolve_model_name_for_message(
        message,
        settings,
        available_models=("gpt-oss:20b", "gemma4:27b"),
    )

    assert resolved == "gemma4:27b"


def test_resolve_model_name_for_message_falls_back_to_settings() -> None:
    message = SimpleNamespace(content="hello", modes={"model_name": "unknown"})
    settings = SimpleNamespace(model_name="gpt-oss:20b")

    resolved = main.resolve_model_name_for_message(
        message,
        settings,
        available_models=("gpt-oss:20b", "gemma4:27b"),
    )

    assert resolved == "gpt-oss:20b"


class _DummyRuntime:
    def __init__(self, command=None) -> None:
        self.invocation: dict[str, str | None] | None = None
        self.command = command or SimpleNamespace(
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
        mcp_session_id: str | None = None,
        server_name: str | None = None,
    ):
        self.invocation = {
            "tool_name": tool_name,
            "raw_args": raw_args,
            "thread_id": thread_id,
            "mcp_session_id": mcp_session_id,
            "server_name": server_name,
        }
        return {"ok": True}


class _DummyMessage:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def send(self):
        return None


@pytest.mark.anyio
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
    assert runtime.invocation["mcp_session_id"] is None


def test_build_skill_command_prompt_requires_skill_and_request() -> None:
    prompt = main.build_skill_command_prompt(
        skill_name="reviewer",
        skill_path="/workspace/skills/reviewer/SKILL.md",
        raw_args="inspect this diff",
    )

    assert "Use the configured `reviewer` skill" in prompt
    assert "Read `/workspace/skills/reviewer/SKILL.md` before taking any other action" in prompt
    assert "Skill usage is mandatory" in prompt
    assert "User request:\ninspect this diff" in prompt


def test_build_skill_command_prompt_without_request_asks_for_task() -> None:
    prompt = main.build_skill_command_prompt(
        skill_name="reviewer",
        skill_path="/workspace/skills/reviewer/SKILL.md",
        raw_args="",
    )

    assert "briefly explain what it does and ask the user for the specific task" in prompt


@pytest.mark.anyio
async def test_handle_native_command_returns_forced_skill_prompt() -> None:
    runtime = _DummyRuntime(
        command=SimpleNamespace(
            name="reviewer",
            description="Review code for bugs",
            target="skill",
            value="/workspace/skills/reviewer/SKILL.md",
            template=None,
            mcp_server=None,
        )
    )
    settings = SimpleNamespace(thread_id="thread-1")

    result = await main.handle_native_command(
        runtime=runtime,
        settings=settings,
        parsed=main.ParsedNativeCommand(
            command_name="reviewer",
            raw_args="inspect this diff",
        ),
    )

    assert result is not None
    assert "Use the configured `reviewer` skill" in result
    assert "User request:\ninspect this diff" in result


@pytest.mark.anyio
async def test_handle_native_command_without_skill_args_requests_clarification() -> None:
    runtime = _DummyRuntime(
        command=SimpleNamespace(
            name="reviewer",
            description="Review code for bugs",
            target="skill",
            value="/workspace/skills/reviewer/SKILL.md",
            template=None,
            mcp_server=None,
        )
    )
    settings = SimpleNamespace(thread_id="thread-1")

    result = await main.handle_native_command(
        runtime=runtime,
        settings=settings,
        parsed=main.ParsedNativeCommand(
            command_name="reviewer",
            raw_args="",
        ),
    )

    assert result is not None
    assert "briefly explain what it does and ask the user for the specific task" in result


@pytest.mark.anyio
async def test_handle_native_command_uses_selected_skill_command_input() -> None:
    runtime = _DummyRuntime(
        command=SimpleNamespace(
            name="reviewer",
            description="Review code for bugs",
            target="skill",
            value="/workspace/skills/reviewer/SKILL.md",
            template=None,
            mcp_server=None,
        )
    )
    settings = SimpleNamespace(thread_id="thread-1")
    parsed = main.resolve_native_command(
        raw_text="inspect this diff",
        selected_command="reviewer",
    )

    assert parsed == main.ParsedNativeCommand(
        command_name="reviewer",
        raw_args="inspect this diff",
    )

    result = await main.handle_native_command(
        runtime=runtime,
        settings=settings,
        parsed=parsed,
    )

    assert result is not None
    assert "User request:\ninspect this diff" in result
