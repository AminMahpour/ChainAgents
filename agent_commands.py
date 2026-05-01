from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple


class ParsedNativeCommand(NamedTuple):
    command_name: str
    raw_args: str


@dataclass(frozen=True)
class RuntimeCommandResult:
    target: Literal["unknown", "prompt", "mcp_tool"]
    command_name: str
    description: str = ""
    prompt: str | None = None
    tool_result: Any = None


def parse_native_command(raw_text: str) -> ParsedNativeCommand | None:
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
    parsed = parse_native_command(raw_text)
    if parsed is not None:
        return parsed

    command_name = str(selected_command or "").strip().lstrip("/").lower()
    if not command_name:
        return None

    return ParsedNativeCommand(
        command_name=command_name,
        raw_args=raw_text.strip(),
    )


def apply_native_template(template: str | None, raw_args: str) -> str:
    if template is None:
        return raw_args.strip()
    return template.replace("{input}", raw_args.strip()).strip()


def build_skill_command_prompt(*, skill_name: str, skill_path: str, raw_args: str) -> str:
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


def dumps_tool_result(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, default=str)
