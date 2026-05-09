"""Test native Chainlit command and mode handling in the main app."""

from __future__ import annotations

from types import SimpleNamespace

import agent_commands
import main
import pytest


def test_resolve_native_command_prefers_explicit_slash_text() -> None:
    """Verify that resolve native command prefers explicit slash text."""
    parsed = agent_commands.resolve_native_command(
        raw_text="/summarize hello world",
        selected_command="ask-researcher",
    )

    assert parsed == agent_commands.ParsedNativeCommand(
        command_name="summarize",
        raw_args="hello world",
    )


def test_resolve_native_command_uses_selected_command_input() -> None:
    """Verify that resolve native command uses selected command input."""
    parsed = agent_commands.resolve_native_command(
        raw_text="hello world",
        selected_command="summarize",
    )

    assert parsed == agent_commands.ParsedNativeCommand(
        command_name="summarize",
        raw_args="hello world",
    )


def test_resolve_native_command_returns_none_without_command() -> None:
    """Verify that resolve native command returns none without command."""
    parsed = agent_commands.resolve_native_command(
        raw_text="hello world",
        selected_command=None,
    )

    assert parsed is None


def test_resolve_reasoning_level_for_message_defaults_to_settings() -> None:
    """Verify that resolve reasoning level for message defaults to settings."""
    message = SimpleNamespace(content="hello")
    settings = SimpleNamespace(reasoning_level="medium")

    resolved = main.resolve_reasoning_level_for_message(message, settings)

    assert resolved == "medium"


def test_resolve_reasoning_level_for_message_uses_mode_override() -> None:
    """Verify that resolve reasoning level for message uses mode override."""
    message = SimpleNamespace(content="hello", modes={"reasoning_level": "high"})
    settings = SimpleNamespace(reasoning_level="medium")

    resolved = main.resolve_reasoning_level_for_message(message, settings)

    assert resolved == "high"


def test_resolve_reasoning_level_for_message_falls_back_to_settings_default() -> None:
    """Verify that resolve reasoning level for message falls back to settings default."""
    message = SimpleNamespace(content="hello", modes={})
    settings = SimpleNamespace(reasoning_level="low")

    resolved = main.resolve_reasoning_level_for_message(message, settings)

    assert resolved == "low"


def test_resolve_reasoning_level_for_message_ignores_override_when_disabled() -> None:
    """Verify that resolve reasoning level for message ignores override when disabled."""
    message = SimpleNamespace(content="hello", modes={"reasoning_level": "high"})
    settings = SimpleNamespace(reasoning_level="low")

    resolved = main.resolve_reasoning_level_for_message(
        message,
        settings,
        reasoning_mode_enabled=False,
    )

    assert resolved == "low"


def test_resolve_model_name_for_message_uses_mode_override() -> None:
    """Verify that resolve model name for message uses mode override."""
    message = SimpleNamespace(content="hello", modes={"model_name": "gemma4:27b"})
    settings = SimpleNamespace(model_name="gpt-oss:20b")

    resolved = main.resolve_model_name_for_message(
        message,
        settings,
        available_models=("gpt-oss:20b", "gemma4:27b"),
    )

    assert resolved == "gemma4:27b"


def test_resolve_model_name_for_message_falls_back_to_settings() -> None:
    """Verify that resolve model name for message falls back to settings."""
    message = SimpleNamespace(content="hello", modes={"model_name": "unknown"})
    settings = SimpleNamespace(model_name="gpt-oss:20b")

    resolved = main.resolve_model_name_for_message(
        message,
        settings,
        available_models=("gpt-oss:20b", "gemma4:27b"),
    )

    assert resolved == "gpt-oss:20b"


def test_resolve_model_name_for_message_ignores_override_when_disabled() -> None:
    """Verify that resolve model name for message ignores override when disabled."""
    message = SimpleNamespace(content="hello", modes={"model_name": "gemma4:27b"})
    settings = SimpleNamespace(model_name="gpt-oss:20b")

    resolved = main.resolve_model_name_for_message(
        message,
        settings,
        available_models=("gpt-oss:20b", "gemma4:27b"),
        model_mode_enabled=False,
    )

    assert resolved == "gpt-oss:20b"


def test_build_langgraph_config_includes_recursion_limit() -> None:
    """Verify that build langgraph config includes recursion limit."""
    settings = SimpleNamespace(thread_id="thread-1")

    config = main.build_langgraph_config(settings, recursion_limit=100)

    assert config == {
        "configurable": {"thread_id": "thread-1"},
        "recursion_limit": 100,
    }


@pytest.mark.anyio
async def test_publish_modes_ignores_missing_modes_column_error(monkeypatch) -> None:
    """Verify that publish modes ignores missing modes column error.

    Args:
        monkeypatch: The monkeypatch value.
    """
    class _Emitter:
        """Provide an internal helper for emitter."""

        async def set_modes(self, _modes):
            """Record requested Chainlit mode definitions on the test emitter.

            Args:
                _modes: The modes value.

            Raises:
                RuntimeError: If the runtime is not in a usable state.
            """
            raise RuntimeError('column "modes" does not exist')

    monkeypatch.setattr(main.cl, "context", SimpleNamespace(emitter=_Emitter()))

    await main.publish_modes(
        SimpleNamespace(model_name="gpt-oss:20b", reasoning_level="medium"),
        available_models=("gpt-oss:20b",),
    )


@pytest.mark.anyio
async def test_publish_modes_raises_for_unrelated_errors(monkeypatch) -> None:
    """Verify that publish modes raises for unrelated errors.

    Args:
        monkeypatch: The monkeypatch value.
    """
    class _Emitter:
        """Provide an internal helper for emitter."""

        async def set_modes(self, _modes):
            """Record requested Chainlit mode definitions on the test emitter.

            Args:
                _modes: The modes value.

            Raises:
                RuntimeError: If the runtime is not in a usable state.
            """
            raise RuntimeError("boom")

    monkeypatch.setattr(main.cl, "context", SimpleNamespace(emitter=_Emitter()))

    with pytest.raises(RuntimeError, match="boom"):
        await main.publish_modes(
            SimpleNamespace(model_name="gpt-oss:20b", reasoning_level="medium"),
            available_models=("gpt-oss:20b",),
        )


class _DummyRuntime:
    """Provide a test double for dummy runtime."""

    def __init__(self, command=None) -> None:
        """Initialize the dummy runtime instance.

        Args:
            command: Configured command to render or execute.
        """
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
        """Resolve chainlit command.

        Args:
            command_name: Name of the native command.

        Returns:
            The resolved chainlit command.
        """
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
        """Invoke a configured MCP tool command with parsed arguments.

        Args:
            tool_name: Name of the tool to invoke.
            raw_args: Raw argument text supplied with the command.
            thread_id: Conversation thread identifier.
            mcp_session_id: MCP session identifier.
            server_name: The server name value.

        Returns:
            The invoke MCP tool command result.
        """
        self.invocation = {
            "tool_name": tool_name,
            "raw_args": raw_args,
            "thread_id": thread_id,
            "mcp_session_id": mcp_session_id,
            "server_name": server_name,
        }
        return {"ok": True}


class _DummyMessage:
    """Provide a test double for dummy message."""

    def __init__(self, **kwargs):
        """Initialize the dummy message instance.

        Args:
            kwargs: The kwargs value.
        """
        self.kwargs = kwargs

    async def send(self):
        """Record that the dummy Chainlit message was sent.

        Returns:
            The sent message or element.
        """
        return None


@pytest.mark.anyio
async def test_handle_native_command_applies_template_for_mcp_tool(monkeypatch) -> None:
    """Verify that handle native command applies template for MCP tool.

    Args:
        monkeypatch: The monkeypatch value.
    """
    runtime = _DummyRuntime()
    settings = SimpleNamespace(thread_id="thread-1")
    monkeypatch.setattr(main.cl, "Message", _DummyMessage)

    result = await main.handle_native_command(
        runtime=runtime,
        settings=settings,
        parsed=agent_commands.ParsedNativeCommand(command_name="repo-readme", raw_args=""),
    )

    assert result == ""
    assert runtime.invocation is not None
    assert runtime.invocation["raw_args"] == '{"path":"README.md"}'
    assert runtime.invocation["mcp_session_id"] is None


def test_build_skill_command_prompt_requires_skill_and_request() -> None:
    """Verify that build skill command prompt requires skill and request."""
    prompt = agent_commands.build_skill_command_prompt(
        skill_name="reviewer",
        skill_path="/workspace/skills/reviewer/SKILL.md",
        raw_args="inspect this diff",
    )

    assert "Use the configured `reviewer` skill" in prompt
    assert "Read `/workspace/skills/reviewer/SKILL.md` before taking any other action" in prompt
    assert "Skill usage is mandatory" in prompt
    assert "User request:\ninspect this diff" in prompt


def test_build_skill_command_prompt_without_request_asks_for_task() -> None:
    """Verify that build skill command prompt without request asks for task."""
    prompt = agent_commands.build_skill_command_prompt(
        skill_name="reviewer",
        skill_path="/workspace/skills/reviewer/SKILL.md",
        raw_args="",
    )

    assert "briefly explain what it does and ask the user for the specific task" in prompt


@pytest.mark.anyio
async def test_handle_native_command_returns_forced_skill_prompt() -> None:
    """Verify that handle native command returns forced skill prompt."""
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
        parsed=agent_commands.ParsedNativeCommand(
            command_name="reviewer",
            raw_args="inspect this diff",
        ),
    )

    assert result is not None
    assert "Use the configured `reviewer` skill" in result
    assert "User request:\ninspect this diff" in result


@pytest.mark.anyio
async def test_handle_native_command_without_skill_args_requests_clarification() -> None:
    """Verify that handle native command without skill args requests clarification."""
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
        parsed=agent_commands.ParsedNativeCommand(
            command_name="reviewer",
            raw_args="",
        ),
    )

    assert result is not None
    assert "briefly explain what it does and ask the user for the specific task" in result


@pytest.mark.anyio
async def test_handle_native_command_uses_selected_skill_command_input() -> None:
    """Verify that handle native command uses selected skill command input."""
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
    parsed = agent_commands.resolve_native_command(
        raw_text="inspect this diff",
        selected_command="reviewer",
    )

    assert parsed == agent_commands.ParsedNativeCommand(
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
