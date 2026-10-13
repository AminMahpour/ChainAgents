"""Parse and resolve native slash commands for the ChainAgents runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from chainagents.runtime.live_agents.commands import handle_live_agent_command
from chainagents.runtime.live_agents.config import LIVE_AGENT_COMMAND_NAMES


class ParsedNativeCommand(NamedTuple):
    """Represent a parsed native command and its raw argument text.

    Attributes:
        command_name: Name of the native command.
        raw_args: Raw argument text supplied with the command.
    """

    command_name: str
    raw_args: str


@dataclass(frozen=True)
class RuntimeCommandResult:
    """Describe how a resolved runtime command should be handled.

    Attributes:
        target: The target value.
        command_name: Name of the native command.
        description: The description value.
        prompt: The prompt value.
        tool_result: The tool result value.
    """

    target: Literal["unknown", "prompt", "mcp_tool", "live_agent"]
    command_name: str
    description: str = ""
    prompt: str | None = None
    tool_result: Any = None


def parse_native_command(raw_text: str) -> ParsedNativeCommand | None:
    """Parse native command.

    Args:
        raw_text: Raw user text to parse.

    Returns:
        The parsed native command.
    """
    text = raw_text.strip()
    if not text.startswith("/"):
        return None
    tokens = text[1:].split(None, 1)
    if not tokens:
        return None
    return ParsedNativeCommand(
        command_name=tokens[0].strip().lower(),
        raw_args=tokens[1].strip() if len(tokens) > 1 else "",
    )


def resolve_native_command(
    *,
    raw_text: str,
    selected_command: str | None = None,
) -> ParsedNativeCommand | None:
    """Resolve native command.

    Args:
        raw_text: Raw user text to parse.
        selected_command: Command selected through a separate interface field, if any.

    Returns:
        The resolved native command.
    """
    command_name = str(selected_command or "").strip().lstrip("/").lower()
    if command_name:
        return ParsedNativeCommand(
            command_name=command_name,
            raw_args=raw_text.strip(),
        )

    return parse_native_command(raw_text)


def resolve_live_agent_command(
    runtime: Any,
    *,
    raw_text: str,
    selected_command: str | None = None,
) -> ParsedNativeCommand | None:
    """Return the live agent command in ``raw_text``, if ``runtime`` runs one.

    Live agent commands (``/agents``, ``/spawn``, ``/msg``, ``/kill``) never
    run the main agent, so interfaces execute them immediately, even while a
    main turn is busy, instead of steering or queueing them.
    """
    if getattr(runtime, "live_agents", None) is None:
        return None
    parsed = resolve_native_command(raw_text=raw_text, selected_command=selected_command)
    if parsed is None or parsed.command_name not in LIVE_AGENT_COMMAND_NAMES:
        return None
    return parsed


async def run_live_agent_command(
    runtime: Any, parsed: ParsedNativeCommand, *, thread_id: str
) -> str:
    """Run a live agent command for ``thread_id`` and return its Markdown output.

    Raises:
        ValueError: On usage errors, unknown agents and manager rejections.
    """
    return await handle_live_agent_command(
        runtime.live_agents,
        command_name=parsed.command_name,
        raw_args=parsed.raw_args,
        session_id=thread_id,
    )


def apply_native_template(template: str | None, raw_args: str) -> str:
    """Apply native template.

    Args:
        template: Template string applied to command input.
        raw_args: Raw argument text supplied with the command.

    Returns:
        The transformed value.
    """
    if template is None:
        return raw_args.strip()
    return template.replace("{input}", raw_args.strip()).strip()


def build_skill_command_prompt(*, skill_name: str, skill_path: str, raw_args: str) -> str:
    """Build skill command prompt.

    Args:
        skill_name: The skill name value.
        skill_path: Path to the skill.
        raw_args: Raw argument text supplied with the command.

    Returns:
        The constructed skill command prompt.
    """
    prelude = (
        f"Use the configured `{skill_name}` skill for this request.\n"
        f"Read `{skill_path}` before taking any other action and follow it for this entire turn.\n"
        "Skill usage is mandatory for this request.\n"
    )
    request = raw_args.strip()
    if request:
        return (
            f"{prelude}\n"
            "After reading the skill, complete the user's request below.\n\n"
            f"User request:\n{request}"
        ).strip()
    return (
        f"{prelude}\n"
        "After reading the skill, briefly explain what it does and ask the user for the specific task."
    ).strip()


async def resolve_runtime_command(
    *,
    runtime: Any,
    parsed: ParsedNativeCommand,
    thread_id: str,
    mcp_session_id: str | None = None,
) -> RuntimeCommandResult:
    """Resolve runtime command.

    Args:
        runtime: Agent runtime used by the operation.
        parsed: Parsed native command details.
        thread_id: Conversation thread identifier.
        mcp_session_id: MCP session identifier.

    Returns:
        The resolved runtime command.
    """
    live_agents = getattr(runtime, "live_agents", None)
    if live_agents is not None and parsed.command_name in LIVE_AGENT_COMMAND_NAMES:
        return RuntimeCommandResult(
            target="live_agent",
            command_name=parsed.command_name,
            tool_result=await run_live_agent_command(
                runtime, parsed, thread_id=thread_id
            ),
        )

    command = runtime.resolve_chainlit_command(parsed.command_name)
    if command is None:
        return RuntimeCommandResult(
            target="unknown",
            command_name=parsed.command_name,
        )

    if command.target == "skill":
        return RuntimeCommandResult(
            target="prompt",
            command_name=command.name,
            description=command.description,
            prompt=build_skill_command_prompt(
                skill_name=command.name,
                skill_path=command.value,
                raw_args=parsed.raw_args,
            ),
        )

    if command.target == "mcp_tool":
        tool_raw_args = apply_native_template(command.template, parsed.raw_args)
        result = await runtime.invoke_mcp_tool_command(
            tool_name=command.value,
            raw_args=tool_raw_args,
            thread_id=thread_id,
            mcp_session_id=mcp_session_id,
            server_name=command.mcp_server,
        )
        return RuntimeCommandResult(
            target="mcp_tool",
            command_name=command.name,
            description=command.description,
            tool_result=result,
        )

    transformed = apply_native_template(command.template, parsed.raw_args)
    if command.target == "prompt":
        prompt = transformed or command.value
    else:
        prompt = (
            f"Delegate this request to the configured `{command.value}` subagent.\n\n"
            f"Command: `/{command.name}`\n"
            f"Description: {command.description}\n\n"
            f"User request:\n{transformed or parsed.raw_args or command.value}"
        ).strip()

    return RuntimeCommandResult(
        target="prompt",
        command_name=command.name,
        description=command.description,
        prompt=prompt,
    )


def command_result_text(result: RuntimeCommandResult) -> str:
    """Return display text for a command result.

    Live agent output is Markdown and is shown raw; other tool results are
    serialized as stable JSON.
    """
    if result.target == "live_agent":
        return str(result.tool_result)
    return dumps_tool_result(result.tool_result)


def dumps_tool_result(value: Any) -> str:
    """Serialize a tool result as stable JSON for display.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        A stable JSON string suitable for display.
    """
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, default=str)
