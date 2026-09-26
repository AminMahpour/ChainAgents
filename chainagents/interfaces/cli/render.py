"""Render CLI turn output: the Rich-backed stream renderer and status printers."""

from __future__ import annotations

import json
import traceback
from typing import Any, TextIO

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from chainagents.commands.native import RuntimeCommandResult, dumps_tool_result
from chainagents.events.stream import AgentStreamEvent, stringify_content
from chainagents.rag.runtime import RagStatus, RagUploadResult
from chainagents.runtime import (
    AgentRuntime,
    format_model_provider,
    resolve_runtime_model_profile,
)
from chainagents.runtime.reflection import ReflectionProposal, format_reflection_proposal
from chainagents.turns import BaseTurnRenderer, TurnCommandError, TurnResult

TOOL_RESULT_PREVIEW_CHARS = 200
CLI_PANEL_BOX = box.HEAVY
CLI_TABLE_BOX = box.SIMPLE_HEAVY
CLI_PANEL_PADDING = (0, 1)


def truncate_tool_result_content(value: Any) -> str:
    """Truncate tool result content.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The truncated value.
    """
    content = stringify_content(value).strip()
    if len(content) <= TOOL_RESULT_PREVIEW_CHARS:
        return content
    return content[:TOOL_RESULT_PREVIEW_CHARS]


def pretty_tool_call_args(value: Any) -> str:
    """Format tool call args.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The formatted display value.
    """
    content = stringify_content(value).strip()
    if not content:
        return ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    return json.dumps(parsed, indent=2, sort_keys=True, ensure_ascii=True)


def cli_console(file: TextIO) -> Console:
    """Create the Rich console used by the CLI renderer.

    Args:
        file: The file value.

    Returns:
        The created the rich console used by the cli renderer.
    """
    return Console(
        file=file,
        highlight=False,
        soft_wrap=True,
    )


def cli_panel(renderable: Any, *, title: str, border_style: str) -> Panel:
    """Create a Rich panel with ChainAgents CLI styling.

    Args:
        renderable: The renderable value.
        title: The title value.
        border_style: The border style value.

    Returns:
        The created a rich panel with chainagents cli styling.
    """
    return Panel(
        renderable,
        title=title,
        title_align="left",
        border_style=border_style,
        box=CLI_PANEL_BOX,
        padding=CLI_PANEL_PADDING,
    )


def cli_kv_table() -> Table:
    """Create a two-column Rich table for CLI key-value output.

    Returns:
        The created a two-column rich table for cli key-value output.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column()
    return table


def is_command_output(result: TurnResult) -> bool:
    """Return whether ``result`` is MCP-tool command output, not an agent run."""
    return (
        result.command_result is not None
        and result.command_result.target == "mcp_tool"
    )


def generated_file_paths(result: TurnResult) -> list[str]:
    """Return the local paths of the turn's validated generated files."""
    return [
        str(descriptor.path) if descriptor.path is not None else descriptor.name
        for descriptor in result.generated_files
    ]


class CliEventRenderer(BaseTurnRenderer):
    """Render one shared-runner agent turn to the CLI's stdout and stderr."""

    def __init__(
        self,
        *,
        stdout: TextIO,
        stderr: TextIO,
        stream: bool,
        json_output: bool,
        show_reasoning: bool,
        show_tools: bool,
    ) -> None:
        """Initialize the CLI event renderer instance.

        Args:
            stdout: The stdout value.
            stderr: The stderr value.
            stream: The stream value.
            json_output: The JSON output value.
            show_reasoning: The show reasoning value.
            show_tools: The show tools value.
        """
        self.stdout = stdout
        self.stderr = stderr
        self.stream = stream
        self.json_output = json_output
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self.response_buffer = ""
        self.reasoning_line_source: str | None = None
        self.stdout_console = cli_console(stdout)
        self.stderr_console = cli_console(stderr)

    async def on_event(self, event: AgentStreamEvent) -> None:
        """Render one normalized agent stream event."""
        if event.kind == "mcp_status":
            print(event.text, file=self.stderr)
        elif event.kind == "response_delta":
            self._stream_response_delta(event.text)
        elif event.kind == "reasoning_delta":
            self._stream_reasoning_delta(event.source, event.text)
        elif event.kind == "tool_call":
            self._stream_tool_call_event(event)
        elif event.kind == "tool_result":
            self._complete_tool_event(event)
        elif event.kind == "summarization_status":
            self._stream_summarization_status(event)

    async def on_command_result(self, result: RuntimeCommandResult) -> None:
        """Print MCP-tool command output as raw JSON on stdout."""
        print(dumps_tool_result(result.tool_result), file=self.stdout)

    async def on_command_error(self, exc: TurnCommandError, status: int) -> None:
        """Print a failed or unknown native command to stderr."""
        if exc.unknown:
            print(f"Unknown command /{exc.command_name}.", file=self.stderr)
        else:
            print(f"Command /{exc.command_name} failed: {exc.message}", file=self.stderr)

    async def on_complete(self, result: TurnResult) -> None:
        """Print the response, generated files, reflection and failure, in order."""
        if result.status == "command_error":
            return
        command_output = is_command_output(result)
        if result.status == "completed" and not command_output:
            self._finish_response()
        else:
            self._close_reasoning_line()
            self._end_streamed_response_line()
        # A successful JSON agent turn carries the files in its payload instead.
        if result.generated_files and not (
            self.json_output and result.status == "completed" and not command_output
        ):
            self._print_generated_files(result)
        if result.reflection is not None:
            self.print_reflection_proposal(result.reflection)
        if result.error is not None:
            exc = result.error
            print(f"{type(exc).__name__}: {exc}", file=self.stderr)
            if self.show_tools or self.show_reasoning:
                print(
                    "".join(traceback.format_exception(exc, limit=10)),
                    file=self.stderr,
                )

    def _finish_response(self) -> None:
        """Close streamed output, or print the buffered response when not streaming."""
        self._close_reasoning_line()
        if self.json_output:
            return
        if self.stream:
            self._end_streamed_response_line()
            return
        if self.response_buffer:
            self.stdout_console.print(Text(self.response_buffer, style="bright_white"))

    def _end_streamed_response_line(self) -> None:
        """Terminate partially streamed stdout text with a newline."""
        if (
            self.stream
            and not self.json_output
            and self.response_buffer
            and not self.response_buffer.endswith("\n")
        ):
            self.stdout_console.print()

    def _print_generated_files(self, result: TurnResult) -> None:
        """List the turn's generated file paths on stderr."""
        print("Generated files:", file=self.stderr)
        for path in generated_file_paths(result):
            print(f"  {path}", file=self.stderr)

    def print_reflection_proposal(self, proposal: ReflectionProposal) -> None:
        """Print a reflection proposal to stderr for human CLI users."""
        if self.json_output:
            return
        self._close_reasoning_line()
        self.stderr_console.print(
            cli_panel(
                Text(format_reflection_proposal(proposal), style="white"),
                title="Reflection Proposal",
                border_style="cyan",
            )
        )

    def _stream_summarization_status(self, event: AgentStreamEvent) -> None:
        """Render a summarization status update."""
        if self.json_output:
            return

        self._close_reasoning_line()
        body = Text.assemble(
            ("status: ", "dim"),
            (event.status, "bold cyan"),
            ("\nsource: ", "dim"),
            (event.source, "cyan"),
        )
        if event.text:
            body.append("\nmessage: ", style="dim")
            body.append(event.text, style="white")
        self.stderr_console.print(
            cli_panel(
                body,
                title="Summarization",
                border_style="cyan",
            )
        )

    def _stream_response_delta(self, text: str) -> None:
        """Stream final response text into the active output target.

        Args:
            text: Text content to process.
        """
        if not text:
            return
        self.response_buffer += text
        if self.stream and not self.json_output:
            self.stdout_console.print(Text(text, style="bright_white"), end="")

    def _stream_reasoning_delta(self, source: str, text: str) -> None:
        """Stream reasoning text into the active output target.

        Args:
            source: The source value.
            text: Text content to process.
        """
        if not text:
            return
        if self.show_reasoning:
            if self.reasoning_line_source != source:
                self._close_reasoning_line()
                self.stderr_console.print(
                    Text(f"[reasoning:{source}] ", style="bold magenta"),
                    end="",
                )
                self.reasoning_line_source = source
            self.stderr_console.print(Text(text, style="magenta"), end="")

    def _close_reasoning_line(self) -> None:
        """Close the active reasoning line before rendering other output."""
        if self.reasoning_line_source is None:
            return
        self.stderr_console.print()
        self.reasoning_line_source = None

    def _stream_tool_call_event(self, event: AgentStreamEvent) -> None:
        """Render a streamed tool call and its accumulated arguments."""
        if not self.show_tools:
            return

        self._close_reasoning_line()
        body = Text.assemble(
            ("status: ", "dim"),
            (event.status, "bold yellow"),
            ("\nsource: ", "dim"),
            (event.source, "cyan"),
            ("\ntool: ", "dim"),
            (event.tool_name, "bold white"),
        )
        args = pretty_tool_call_args(event.tool_args)
        if args:
            body.append("\nargs: ", style="dim yellow")
            body.append(args, style="yellow")
        self.stderr_console.print(
            cli_panel(
                body,
                title="Tool Call",
                border_style="yellow",
            )
        )

    def _complete_tool_event(self, event: AgentStreamEvent) -> None:
        """Render the final status and output for a completed tool call."""
        if not self.show_tools:
            return
        content = truncate_tool_result_content(event.tool_result)
        self._close_reasoning_line()
        status_style = "bold red" if event.status.lower() == "error" else "bold green"
        body = Text.assemble(
            ("status: ", "dim"),
            (event.status, status_style),
            ("\nsource: ", "dim"),
            (event.source, "cyan"),
            ("\ntool: ", "dim"),
            (event.tool_name, "bold white"),
        )
        if content:
            body.append("\nresult: ", style="dim yellow")
            body.append(content, style="yellow")
        self.stderr_console.print(
            cli_panel(
                body,
                title="Tool Result",
                border_style="red" if event.status.lower() == "error" else "green",
            )
        )


def rag_status_payload(status: RagStatus) -> dict[str, Any]:
    """Build the JSON payload for CLI RAG status output.

    Args:
        status: The status value.

    Returns:
        The constructed the json payload for cli rag status output.
    """
    return {
        "enabled": status.enabled,
        "ready": status.ready,
        "file_count": status.file_count,
        "chunk_count": status.chunk_count,
        "reason": status.reason,
        "persist_directory": str(status.persist_directory) if status.persist_directory else None,
    }


def upload_result_payload(result: RagUploadResult) -> dict[str, Any]:
    """Build the JSON payload for CLI upload results.

    Args:
        result: Result payload to format or inspect.

    Returns:
        The constructed the json payload for cli upload results.
    """
    return {
        "thread_id": result.thread_id,
        "success": result.success,
        "added_files": list(result.added_files),
        "indexed_files": result.indexed_files,
        "chunk_count": result.chunk_count,
        "rejected_files": list(result.rejected_files),
        "reason": result.reason,
    }


def runtime_status_payload(runtime: AgentRuntime) -> dict[str, Any]:
    """Build the JSON payload for CLI runtime status output.

    Args:
        runtime: Agent runtime used by the operation.

    Returns:
        The constructed the json payload for cli runtime status output.
    """
    extensions = runtime.config.extensions
    active_model = resolve_runtime_model_profile(runtime.config)
    return {
        "model_provider": active_model.provider,
        "model_provider_label": format_model_provider(active_model.provider),
        "model": runtime.config.model_name,
        "model_choices": list(runtime.config.model_choices),
        "model_base_url": active_model.base_url,
        "reasoning": runtime.config.default_reasoning,
        "model_disable_streaming": runtime.config.model_disable_streaming,
        "agent_state": runtime.config.agent_state,
        "recursion_limit": runtime.config.recursion_limit,
        "persistence": runtime.config.persistence_mode,
        "rag": rag_status_payload(runtime.rag_status),
        "extensions_config": str(extensions.config_path) if extensions.config_path else None,
        "skill_sources": list(extensions.skills),
        "mcp_servers": sorted((extensions.mcp_servers or {}).keys()),
        "agent_mcp_servers": list(extensions.agent_mcp_servers),
        "sync_subagents": [subagent.name for subagent in extensions.subagents],
        "async_subagents": [subagent.name for subagent in extensions.async_subagents],
        "commands": [command.name for command in runtime.chainlit_commands],
    }


def rag_status_text(rag: dict[str, Any]) -> Text:
    """Render RAG status as human-readable CLI text.

    Args:
        rag: The RAG value.

    Returns:
        The RAG status text result.
    """
    if rag["enabled"] and rag["ready"]:
        text = Text("ready", style="bold green")
        text.append(f" ({rag['file_count']} files, {rag['chunk_count']} chunks)")
        return text
    if rag["enabled"]:
        text = Text("unavailable", style="bold yellow")
        text.append(f" ({rag['reason'] or 'unknown error'})", style="yellow")
        return text
    return Text("disabled", style="dim")


def print_runtime_status(
    runtime: AgentRuntime,
    *,
    stdout: TextIO,
    json_output: bool,
) -> None:
    """Print runtime status.

    Args:
        runtime: Agent runtime used by the operation.
        stdout: The stdout value.
        json_output: The JSON output value.
    """
    payload = runtime_status_payload(runtime)
    if json_output:
        print(json.dumps({"status": payload}, indent=2, sort_keys=True), file=stdout)
        return

    table = cli_kv_table()
    table.add_row(
        "Model provider",
        Text(str(payload["model_provider_label"]), "bright_white"),
    )
    table.add_row("Model", Text(str(payload["model"]), "bright_white"))
    table.add_row(
        "Base URL",
        Text(str(payload["model_base_url"] or "not set"), "white"),
    )
    table.add_row("Reasoning", Text(str(payload["reasoning"]), "cyan"))
    table.add_row(
        "Disable streaming",
        Text(str(payload["model_disable_streaming"]), "white"),
    )
    table.add_row("Agent state", Text(str(payload["agent_state"]), "white"))
    table.add_row("Recursion limit", Text(str(payload["recursion_limit"]), "white"))
    table.add_row("Persistence", Text(str(payload["persistence"]), "white"))
    table.add_row("RAG", rag_status_text(payload["rag"]))
    table.add_row("Commands", Text(str(len(payload["commands"])), "bold cyan"))
    cli_console(stdout).print(
        cli_panel(
            table,
            title="ChainAgents Runtime",
            border_style="bright_cyan",
        )
    )


def print_command_list(
    runtime: AgentRuntime,
    *,
    stdout: TextIO,
    json_output: bool,
) -> None:
    """Print command list.

    Args:
        runtime: Agent runtime used by the operation.
        stdout: The stdout value.
        json_output: The JSON output value.
    """
    commands = [
        {
            "name": command.name,
            "description": command.description,
            "target": command.target,
            "value": command.value,
            "source": command.source,
        }
        for command in runtime.chainlit_commands
    ]
    if json_output:
        print(
            json.dumps(
                {
                    "commands": commands,
                    "notes": list(runtime.chainlit_command_notes),
                },
                indent=2,
                sort_keys=True,
            ),
            file=stdout,
        )
        return

    console = cli_console(stdout)
    if not commands:
        console.print(
            cli_panel(
                Text("No configured commands.", style="dim"),
                title="Commands",
                border_style="bright_black",
            )
        )
    else:
        table = Table(
            box=CLI_TABLE_BOX,
            border_style="bright_black",
            header_style="bold cyan",
            expand=True,
            show_lines=False,
        )
        table.add_column("Command", style="bold bright_white", no_wrap=True)
        table.add_column("Target", style="cyan", no_wrap=True)
        table.add_column("Description", style="white")
        table.add_column("Source", style="dim", no_wrap=True)
        for command in commands:
            table.add_row(
                f"/{command['name']}",
                str(command["target"]),
                str(command["description"] or "-"),
                str(command["source"] or "-"),
            )
        console.print(
            cli_panel(
                table,
                title=f"Commands ({len(commands)})",
                border_style="cyan",
            )
        )
    if runtime.chainlit_command_notes:
        notes = Text()
        for index, note in enumerate(runtime.chainlit_command_notes):
            if index:
                notes.append("\n")
            notes.append("note: ", style="dim yellow")
            notes.append(str(note), style="yellow")
        console.print(
            cli_panel(
                notes,
                title="Command Notes",
                border_style="yellow",
            )
        )


def print_rag_status(
    *,
    status: RagStatus,
    action: str,
    stdout: TextIO,
    json_output: bool,
) -> None:
    """Print RAG status.

    Args:
        status: The status value.
        action: The action value.
        stdout: The stdout value.
        json_output: The JSON output value.
    """
    payload = rag_status_payload(status)
    if json_output:
        print(json.dumps({action: payload}, indent=2, sort_keys=True), file=stdout)
        return
    if status.ready:
        body = Text(f"{action}: ready", style="bold green")
        body.append(f" ({status.file_count} files, {status.chunk_count} chunks)")
        border_style = "green"
    elif status.enabled:
        body = Text(f"{action}: unavailable", style="bold yellow")
        body.append(f" ({status.reason or 'unknown error'})", style="yellow")
        border_style = "yellow"
    else:
        body = Text(f"{action}: disabled", style="dim")
        border_style = "bright_black"
    cli_console(stdout).print(
        cli_panel(
            body,
            title="RAG",
            border_style=border_style,
        )
    )


def print_upload_result(
    result: RagUploadResult,
    *,
    stdout: TextIO,
    json_output: bool,
) -> None:
    """Print upload result.

    Args:
        result: Result payload to format or inspect.
        stdout: The stdout value.
        json_output: The JSON output value.
    """
    payload = upload_result_payload(result)
    if json_output:
        print(json.dumps({"upload_rag": payload}, indent=2, sort_keys=True), file=stdout)
        return
    body = Text()
    if result.added_files:
        body.append("upload-rag: added ", style="bold green")
        body.append(", ".join(result.added_files), style="bright_white")
        body.append(f" ({result.indexed_files} files, {result.chunk_count} chunks)")
    elif result.reason:
        body.append(f"upload-rag: {result.reason}", style="yellow")
    if result.rejected_files:
        if body.plain:
            body.append("\n")
        body.append("upload-rag: rejected ", style="bold yellow")
        body.append(", ".join(result.rejected_files), style="yellow")
    if body.plain:
        cli_console(stdout).print(
            cli_panel(
                body,
                title="Upload RAG",
                border_style="green" if result.added_files else "yellow",
            )
        )
