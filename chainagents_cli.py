#!/usr/bin/env python3
"""Provide the terminal CLI for ChainAgents prompts and runtime commands."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import sys
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent_commands import (
    dumps_tool_result,
    parse_native_command,
    resolve_native_command,
    resolve_runtime_command,
)
from deepagent_runtime import (
    AgentRuntime,
    AppSettings,
    ReasoningLevel,
    RuntimeConfig,
    RuntimeConfigOverrides,
    format_model_provider,
    normalize_reasoning_level,
)
from rag_runtime import RagStatus, RagUploadResult, UploadedRagFile


DEFAULT_CLI_THREAD_ID = "cli"
TOOL_RESULT_PREVIEW_CHARS = 200
SUMMARIZATION_STATUS_KIND = "summarization_status"
LANGGRAPH_STREAM_MODES = {
    "values",
    "updates",
    "custom",
    "messages",
    "checkpoints",
    "tasks",
    "debug",
}
CLI_PANEL_BOX = box.HEAVY
CLI_TABLE_BOX = box.SIMPLE_HEAVY
CLI_PANEL_PADDING = (0, 1)


def build_parser() -> argparse.ArgumentParser:
    """Build parser.

    Returns:
        The constructed parser.
    """
    parser = argparse.ArgumentParser(
        prog="chainagents",
        description="Run the ChainAgents DeepAgent runtime without the Chainlit UI.",
    )
    parser.add_argument("prompt_parts", nargs="*", metavar="PROMPT")
    parser.add_argument("--prompt", help="Prompt to send to the agent.")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read the prompt from stdin.",
    )
    parser.add_argument(
        "--photo",
        action="append",
        default=[],
        metavar="PATH",
        help="Attach an image file to the prompt for vision-capable models. May be repeated.",
    )

    parser.add_argument("--config", help="Path to deepagent.toml.")
    parser.add_argument("--database-url", help="Postgres URL for durable state.")
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="Force in-memory state even when DATABASE_URL is set.",
    )
    parser.add_argument("--provider", help="Model provider: ollama or openai_compatible.")
    parser.add_argument("--base-url", help="Model server base URL.")
    parser.add_argument(
        "--endpoint-url",
        help=(
            "OpenAI-compatible full endpoint URL. "
            "Use this for non-standard /chat/completions paths."
        ),
    )
    parser.add_argument("--model", help="Model name to run.")
    parser.add_argument("--api-key", help="API key for OpenAI-compatible model servers.")
    parser.add_argument("--temperature", type=float, help="Model temperature.")
    parser.add_argument(
        "--reasoning",
        choices=("low", "medium", "high"),
        help="Reasoning effort for this run.",
    )
    parser.add_argument(
        "--thread-id",
        default=DEFAULT_CLI_THREAD_ID,
        help=f"LangGraph thread ID. Defaults to {DEFAULT_CLI_THREAD_ID!r}.",
    )
    parser.add_argument("--recursion-limit", type=int, help="LangGraph recursion limit.")

    parser.add_argument("--async-subagent-url", help="Override URL for async subagents.")
    parser.add_argument("--mcp-session-id", help="Session scope for stateful MCP servers.")
    parser.add_argument(
        "--command",
        help="Run a configured command using the prompt as command input.",
    )

    parser.add_argument(
        "--rebuild-rag",
        action="store_true",
        help="Rebuild the configured workspace documentation RAG index.",
    )
    parser.add_argument(
        "--upload-rag",
        action="append",
        default=[],
        metavar="PATH",
        help="Add a file to the current thread's uploaded RAG index.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG for this CLI process.",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Print resolved runtime status.",
    )
    parser.add_argument(
        "--list-commands",
        action="store_true",
        help="List configured commands and exit unless a prompt is also provided.",
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream the final response as it is produced.",
    )
    parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Print streamed reasoning traces to stderr.",
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Print tool call traces to stderr.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args.

    Args:
        argv: The argv value.

    Returns:
        The parsed args.
    """
    return build_parser().parse_args(argv)


def runtime_overrides_from_args(args: argparse.Namespace) -> RuntimeConfigOverrides:
    """Build runtime override values from parsed CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The constructed runtime override values from parsed cli arguments.
    """
    return RuntimeConfigOverrides(
        config_path=args.config,
        database_url=args.database_url,
        disable_database=args.no_database,
        model_provider=args.provider,
        model_name=args.model,
        model_base_url=args.base_url,
        model_endpoint_url=args.endpoint_url,
        model_api_key=args.api_key,
        model_temperature=args.temperature,
        reasoning_level=args.reasoning,
        recursion_limit=args.recursion_limit,
        disable_rag=args.no_rag,
    )


def prompt_from_args(
    args: argparse.Namespace,
    *,
    stdin: TextIO,
    parser: argparse.ArgumentParser | None = None,
) -> str | None:
    """Resolve the prompt text supplied through CLI arguments.

    Args:
        args: Parsed command-line arguments.
        stdin: The stdin value.
        parser: The parser value.

    Returns:
        The resolved the prompt text supplied through cli arguments.

    Raises:
        ValueError: If the supplied value is invalid.
    """
    prompt_sources = sum(
        1
        for enabled in (
            bool(args.prompt),
            bool(args.prompt_parts),
            bool(args.stdin),
        )
        if enabled
    )
    if prompt_sources > 1:
        message = "provide only one prompt source: positional PROMPT, --prompt, or --stdin"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    if args.stdin:
        return stdin.read()
    if args.prompt is not None:
        return args.prompt
    if args.prompt_parts:
        return " ".join(args.prompt_parts)
    return None


def photo_content_parts(paths: list[str], *, stderr: TextIO) -> list[dict[str, Any]] | None:
    """Build multimodal content parts for uploaded CLI photos.

    Args:
        paths: Filesystem paths to inspect.
        stderr: The stderr value.

    Returns:
        The constructed multimodal content parts for uploaded cli photos.
    """
    parts: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            print(f"photo: file does not exist: {path}", file=stderr)
            return None

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            print(f"photo: unsupported image type: {path}", file=stderr)
            return None

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )
    return parts


def user_message_content(prompt: str, photos: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Build the user message payload sent from the CLI.

    Args:
        prompt: The prompt value.
        photos: The photos value.

    Returns:
        The constructed the user message payload sent from the cli.
    """
    if not photos:
        return prompt
    return [{"type": "text", "text": prompt}, *photos]


def langgraph_part_from_event_chunk(chunk: Any) -> dict[str, Any] | None:
    """Normalize stream event chunks into LangGraph part metadata.

    Args:
        chunk: Streamed event chunk to normalize.

    Returns:
        The langgraph part from event chunk result.
    """
    if isinstance(chunk, dict):
        mode = chunk.get("type")
        if isinstance(mode, str) and mode in LANGGRAPH_STREAM_MODES and "data" in chunk:
            return chunk
        return None

    if not isinstance(chunk, tuple):
        return None

    if len(chunk) == 3:
        ns, mode, data = chunk
    elif len(chunk) == 2:
        first, data = chunk
        if isinstance(first, tuple):
            ns = first
            mode = "values"
        else:
            ns = ()
            mode = first
    else:
        return None

    if not isinstance(mode, str) or mode not in LANGGRAPH_STREAM_MODES:
        return None

    if isinstance(ns, tuple):
        namespace = ns
    elif ns in (None, ""):
        namespace = ()
    else:
        return None

    return {"type": mode, "ns": namespace, "data": data}


def stringify_content(value: Any) -> str:
    """Convert content.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The string representation.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(stringify_content(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "reasoning", "content"):
            nested = value.get(key)
            if isinstance(nested, (str, list, dict)):
                return stringify_content(nested)
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    return str(value)


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


def reasoning_text_from_token(token: Any) -> str:
    """Extract reasoning text from a streamed model token.

    Args:
        token: Streamed model token to inspect.

    Returns:
        The extracted reasoning text from a streamed model token.
    """
    if hasattr(token, "additional_kwargs"):
        text = stringify_content(token.additional_kwargs.get("reasoning_content"))
        if text:
            return text
    if hasattr(token, "reasoning_content"):
        return stringify_content(token.reasoning_content)
    return ""


def namespace_label(ns: tuple[str, ...], metadata: dict[str, Any]) -> str:
    """Return a display label for a tool namespace.

    Args:
        ns: The ns value.
        metadata: The metadata value.

    Returns:
        A display label for a tool namespace.
    """
    agent_name = metadata.get("lc_agent_name")
    if agent_name:
        return str(agent_name)
    if not ns:
        return "main-agent"

    labels: list[str] = []
    for segment in ns:
        if segment.startswith("tools:"):
            labels.append(f"subagent {segment.split(':', 1)[1]}")
            continue
        labels.append(segment.split(":", 1)[0])
    return " / ".join(labels)


def iter_messages(value: Any) -> list[Any]:
    """Iterate over messages.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        An iterator over the matching values.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        if "messages" in value:
            return iter_messages(value["messages"])
        if "value" in value:
            return iter_messages(value["value"])
        return [value]
    if isinstance(value, (str, bytes)):
        return []

    for attr in ("value", "messages", "data"):
        if hasattr(value, attr):
            messages = iter_messages(getattr(value, attr))
            if messages:
                return messages

    try:
        return list(value)
    except TypeError:
        return [value]


def messages_from_node_data(data: Any) -> list[Any]:
    """Extract message objects from LangGraph node update payloads.

    Args:
        data: Payload data to inspect.

    Returns:
        The extracted message objects from langgraph node update payloads.
    """
    if data is None:
        return []
    if isinstance(data, dict):
        return iter_messages(data.get("messages"))
    return iter_messages(data)


def is_assistant_message(message: Any) -> bool:
    """Return whether assistant message.

    Args:
        message: Chainlit message or LangChain message to process.

    Returns:
        Whether assistant message.
    """
    return getattr(message, "type", None) in {"ai", "AIMessageChunk"}


def message_text(message: Any) -> str:
    """Build the message for text.

    Args:
        message: Chainlit message or LangChain message to process.

    Returns:
        The constructed the message for text.
    """
    return stringify_content(getattr(message, "content", "")).strip()


def assistant_messages_for_current_prompt(messages: list[Any], prompt: str) -> list[Any]:
    """Return assistant messages produced after the current prompt began.

    Args:
        messages: The messages value.
        prompt: The prompt value.

    Returns:
        Assistant messages produced after the current prompt began.
    """
    prompt_text = prompt.strip()
    current_prompt_index = -1
    for index, message in enumerate(messages):
        if getattr(message, "type", None) != "human":
            continue
        if message_text(message) == prompt_text:
            current_prompt_index = index

    if current_prompt_index < 0:
        return []

    return [
        message
        for message in messages[current_prompt_index + 1 :]
        if is_assistant_message(message)
    ]


class CliEventRenderer:
    """Represent CLI event renderer."""

    def __init__(
        self,
        *,
        prompt: str,
        stdout: TextIO,
        stderr: TextIO,
        stream: bool,
        json_output: bool,
        show_reasoning: bool,
        show_tools: bool,
    ) -> None:
        """Initialize the CLI event renderer instance.

        Args:
            prompt: The prompt value.
            stdout: The stdout value.
            stderr: The stderr value.
            stream: The stream value.
            json_output: The JSON output value.
            show_reasoning: The show reasoning value.
            show_tools: The show tools value.
        """
        self.prompt = prompt
        self.stdout = stdout
        self.stderr = stderr
        self.stream = stream
        self.json_output = json_output
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self.response_buffer = ""
        self.response_streamed_from_messages = False
        self.reasoning_buffers: dict[str, str] = {}
        self.reasoning_line_source: str | None = None
        self.tool_names: dict[str, str] = {}
        self.tool_args_buffers: dict[str, str] = {}
        self.tool_call_started: set[str] = set()
        self.completed_tool_results: set[tuple[str, str, str]] = set()
        self.stdout_console = cli_console(stdout)
        self.stderr_console = cli_console(stderr)

    async def handle_event(self, event: dict[str, Any]) -> None:
        """Handle one raw LangGraph stream event.

        Args:
            event: LangGraph stream event to process.
        """
        if event.get("event") != "on_chain_stream":
            return
        if event.get("parent_ids"):
            return

        data = event.get("data")
        if not isinstance(data, dict):
            return

        part = langgraph_part_from_event_chunk(data.get("chunk"))
        if part is None:
            return

        kind = part["type"]
        if kind == "messages":
            self._handle_message_chunk(part)
        elif kind == "updates":
            self._handle_update_chunk(part)
        elif kind == "custom":
            self._handle_custom_chunk(part)

    def finish(self) -> str:
        """Finish the CLI event renderer.

        Returns:
            The finish result.
        """
        self._close_reasoning_line()
        if self.json_output:
            return self.response_buffer
        if self.stream:
            if self.response_buffer and not self.response_buffer.endswith("\n"):
                self.stdout_console.print()
            return self.response_buffer
        if self.response_buffer:
            self.stdout_console.print(Text(self.response_buffer, style="bright_white"))
        return self.response_buffer

    def _handle_message_chunk(self, part: dict[str, Any]) -> None:
        """Handle message chunk.

        Args:
            part: The part value.
        """
        token, metadata = part["data"]
        metadata = metadata if isinstance(metadata, dict) else {}
        ns = tuple(part.get("ns", ()))
        source = namespace_label(ns, metadata)
        is_main_source = not ns

        reasoning_text = reasoning_text_from_token(token)
        if reasoning_text:
            self._stream_reasoning(source, reasoning_text)

        tool_call_chunks = getattr(token, "tool_call_chunks", None) or []
        if tool_call_chunks:
            for chunk in tool_call_chunks:
                self._stream_tool_call(source, chunk)

        if getattr(token, "type", None) == "tool":
            self._complete_tool(source, token)
            return

        content_text = stringify_content(getattr(token, "content", ""))
        if is_main_source and content_text and not tool_call_chunks:
            self.response_streamed_from_messages = True
            self._stream_response(content_text)

    def _handle_update_chunk(self, part: dict[str, Any]) -> None:
        """Handle update chunk.

        Args:
            part: The part value.
        """
        ns = tuple(part.get("ns", ()))
        source = namespace_label(ns, {"lc_agent_name": None})

        for node_name, data in part["data"].items():
            if node_name != "tools":
                if not ns and not self.response_streamed_from_messages:
                    assistant_messages = assistant_messages_for_current_prompt(
                        messages_from_node_data(data),
                        self.prompt,
                    )
                    if assistant_messages:
                        content_text = stringify_content(
                            getattr(assistant_messages[-1], "content", "")
                        )
                        if content_text:
                            self._stream_response(content_text)
                continue

            for message in messages_from_node_data(data):
                if getattr(message, "type", None) == "tool":
                    self._complete_tool(source, message)

    def _handle_custom_chunk(self, part: dict[str, Any]) -> None:
        """Handle custom chunk.

        Args:
            part: The part value.
        """
        data = part.get("data")
        if not isinstance(data, dict):
            return
        if data.get("kind") != SUMMARIZATION_STATUS_KIND:
            return
        if self.json_output:
            return

        status = str(data.get("status") or "triggered").strip() or "triggered"
        source = str(data.get("source") or "main-agent").strip() or "main-agent"
        message = str(data.get("message") or "Conversation summarization triggered.").strip()
        self._close_reasoning_line()
        body = Text.assemble(
            ("status: ", "dim"),
            (status, "bold cyan"),
            ("\nsource: ", "dim"),
            (source, "cyan"),
        )
        if message:
            body.append("\nmessage: ", style="dim")
            body.append(message, style="white")
        self.stderr_console.print(
            cli_panel(
                body,
                title="Summarization",
                border_style="cyan",
            )
        )

    def _stream_response(self, text: str) -> None:
        """Stream final response text into the active output target.

        Args:
            text: Text content to process.
        """
        delta = text[len(self.response_buffer) :] if text.startswith(self.response_buffer) else text
        if not delta:
            return
        self.response_buffer += delta
        if self.stream and not self.json_output:
            self.stdout_console.print(Text(delta, style="bright_white"), end="")

    def _stream_reasoning(self, source: str, text: str) -> None:
        """Stream reasoning text into the active output target.

        Args:
            source: The source value.
            text: Text content to process.
        """
        previous = self.reasoning_buffers.get(source, "")
        delta = text[len(previous) :] if text.startswith(previous) else text
        if not delta:
            return
        self.reasoning_buffers[source] = previous + delta
        if self.show_reasoning:
            if self.reasoning_line_source != source:
                self._close_reasoning_line()
                self.stderr_console.print(
                    Text(f"[reasoning:{source}] ", style="bold magenta"),
                    end="",
                )
                self.reasoning_line_source = source
            self.stderr_console.print(Text(delta, style="magenta"), end="")

    def _close_reasoning_line(self) -> None:
        """Close the active reasoning line before rendering other output."""
        if self.reasoning_line_source is None:
            return
        self.stderr_console.print()
        self.reasoning_line_source = None

    def _stream_tool_call(self, source: str, chunk: dict[str, Any]) -> None:
        """Render a streamed tool call and its accumulated arguments.

        Args:
            source: The source value.
            chunk: Streamed event chunk to normalize.
        """
        call_id = str(chunk.get("id") or f"{source}:{chunk.get('index', '0')}")
        tool_name = str(chunk.get("name") or self.tool_names.get(call_id) or "tool")
        self.tool_names[call_id] = tool_name
        args_delta = chunk.get("args")
        if args_delta:
            self.tool_args_buffers[call_id] = self.tool_args_buffers.get(call_id, "") + str(
                args_delta
            )

        if not self.show_tools:
            return
        if not chunk.get("name") and call_id not in self.tool_call_started:
            return

        self._close_reasoning_line()
        status = "start" if call_id not in self.tool_call_started else "update"
        self.tool_call_started.add(call_id)
        body = Text.assemble(
            ("status: ", "dim"),
            (status, "bold yellow"),
            ("\nsource: ", "dim"),
            (source, "cyan"),
            ("\ntool: ", "dim"),
            (tool_name, "bold white"),
        )
        args = pretty_tool_call_args(self.tool_args_buffers.get(call_id))
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

    def _complete_tool(self, source: str, tool_message: Any) -> None:
        """Render the final status and output for a completed tool call.

        Args:
            source: The source value.
            tool_message: The tool message value.
        """
        if not self.show_tools:
            return
        name = str(getattr(tool_message, "name", "") or "tool")
        status = str(getattr(tool_message, "status", "") or "done")
        content = truncate_tool_result_content(getattr(tool_message, "content", ""))
        result_key = self._tool_result_key(
            source=source,
            name=name,
            tool_message=tool_message,
            content=content,
        )
        if result_key in self.completed_tool_results:
            return
        self.completed_tool_results.add(result_key)

        self._close_reasoning_line()
        status_style = "bold red" if status.lower() == "error" else "bold green"
        body = Text.assemble(
            ("status: ", "dim"),
            (status, status_style),
            ("\nsource: ", "dim"),
            (source, "cyan"),
            ("\ntool: ", "dim"),
            (name, "bold white"),
        )
        if content:
            body.append("\nresult: ", style="dim yellow")
            body.append(content, style="yellow")
        self.stderr_console.print(
            cli_panel(
                body,
                title="Tool Result",
                border_style="red" if status.lower() == "error" else "green",
            )
        )

    def _tool_result_key(
        self,
        *,
        source: str,
        name: str,
        tool_message: Any,
        content: str,
    ) -> tuple[str, str, str]:
        """Build a stable key for deduplicating streamed tool results.

        Args:
            source: The source value.
            name: The name value.
            tool_message: The tool message value.
            content: Message or document content to process.

        Returns:
            The constructed a stable key for deduplicating streamed tool results.
        """
        stable_id = str(
            getattr(tool_message, "tool_call_id", None)
            or getattr(tool_message, "id", None)
            or ""
        ).strip()
        if stable_id:
            return (source, "id", stable_id)
        return (source, name, content)


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
    return {
        "model_provider": runtime.config.model_provider,
        "model_provider_label": format_model_provider(runtime.config.model_provider),
        "model": runtime.config.model_name,
        "model_choices": list(runtime.config.model_choices),
        "model_base_url": runtime.config.model_base_url,
        "reasoning": runtime.config.default_reasoning,
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


async def ingest_uploads(
    runtime: AgentRuntime,
    *,
    paths: list[str],
    thread_id: str,
    stdout: TextIO,
    stderr: TextIO,
    json_output: bool,
    emit_output: bool = True,
) -> RagUploadResult | None:
    """Ingest uploads.

    Args:
        runtime: Agent runtime used by the operation.
        paths: Filesystem paths to inspect.
        thread_id: Conversation thread identifier.
        stdout: The stdout value.
        stderr: The stderr value.
        json_output: The JSON output value.
        emit_output: The emit output value.

    Returns:
        The ingest uploads result.
    """
    if not paths:
        return None

    uploads: list[UploadedRagFile] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            print(f"upload-rag: file does not exist: {path}", file=stderr)
            return RagUploadResult(
                thread_id=thread_id,
                success=False,
                reason=f"file does not exist: {path}",
            )
        uploads.append(UploadedRagFile(path=path, name=path.name))

    result = await runtime.ingest_rag_uploads(thread_id=thread_id, uploads=uploads)
    if emit_output:
        print_upload_result(result, stdout=stdout, json_output=json_output)
    return result


async def run_agent_prompt(
    runtime: AgentRuntime,
    args: argparse.Namespace,
    *,
    prompt: str,
    stdout: TextIO,
    stderr: TextIO,
    emit_json: bool = True,
) -> int | dict[str, Any]:
    """Stream one CLI prompt through the configured agent.

    Args:
        runtime: Agent runtime used by the operation.
        args: Parsed command-line arguments.
        prompt: The prompt value.
        stdout: The stdout value.
        stderr: The stderr value.
        emit_json: The emit JSON value.

    Returns:
        A process-style status code or JSON-compatible response payload.
    """
    thread_id = str(args.thread_id or DEFAULT_CLI_THREAD_ID).strip() or DEFAULT_CLI_THREAD_ID
    reasoning_level: ReasoningLevel = normalize_reasoning_level(
        args.reasoning,
        default=runtime.config.default_reasoning,
    )
    settings = AppSettings(
        model_name=args.model or runtime.config.model_name,
        reasoning_level=reasoning_level,
        thread_id=thread_id,
    )

    parsed_command = resolve_native_command(
        raw_text=prompt,
        selected_command=args.command,
    )
    slash_command_from_text = parse_native_command(prompt)
    if parsed_command is not None:
        try:
            command_result = await resolve_runtime_command(
                runtime=runtime,
                parsed=parsed_command,
                thread_id=settings.thread_id,
                mcp_session_id=args.mcp_session_id,
            )
        except Exception as exc:
            print(
                f"Command /{parsed_command.command_name} failed: {exc}",
                file=stderr,
            )
            return 1

        if command_result.target == "unknown":
            if slash_command_from_text is not None or args.command:
                print(
                    f"Unknown command /{parsed_command.command_name}.",
                    file=stderr,
                )
                return 2
        elif command_result.target == "mcp_tool":
            print(dumps_tool_result(command_result.tool_result), file=stdout)
            return 0
        elif command_result.prompt is not None:
            prompt = command_result.prompt
            if not prompt.strip():
                return 0

    photos = photo_content_parts(args.photo, stderr=stderr)
    if photos is None:
        return 1

    agent = await runtime.get_agent(
        settings.reasoning_level,
        model_name=settings.model_name,
        thread_id=settings.thread_id,
        async_subagent_url_override=args.async_subagent_url,
        mcp_session_id=args.mcp_session_id,
    )
    payload = {
        "messages": [{"role": "user", "content": user_message_content(prompt, photos)}]
    }
    config = {
        "configurable": {"thread_id": settings.thread_id},
        "recursion_limit": runtime.config.recursion_limit,
    }
    renderer = CliEventRenderer(
        prompt=prompt,
        stdout=stdout,
        stderr=stderr,
        stream=args.stream,
        json_output=args.json_output,
        show_reasoning=args.show_reasoning,
        show_tools=args.show_tools,
    )
    stream = agent.astream_events(
        payload,
        config=config,
        version="v2",
        stream_mode=["messages", "updates", "custom"],
        subgraphs=True,
    )

    try:
        while True:
            try:
                event = await anext(stream)
            except StopAsyncIteration:
                break
            await renderer.handle_event(event)
    except asyncio.CancelledError:
        with suppress(Exception):
            await stream.aclose()
        raise
    except Exception as exc:
        with suppress(Exception):
            await stream.aclose()
        print(f"{type(exc).__name__}: {exc}", file=stderr)
        if args.show_tools or args.show_reasoning:
            print(traceback.format_exc(limit=10), file=stderr)
        return 1
    finally:
        with suppress(Exception):
            await stream.aclose()

    response = renderer.finish()
    if args.json_output:
        payload = {
            "response": response,
            "thread_id": settings.thread_id,
            "model": settings.model_name,
            "reasoning": settings.reasoning_level,
        }
        if emit_json:
            print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
            return 0
        return {"prompt": payload}
    return 0


async def interactive_repl(
    runtime: AgentRuntime,
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Run the interactive CLI prompt loop.

    Args:
        runtime: Agent runtime used by the operation.
        args: Parsed command-line arguments.
        stdout: The stdout value.
        stderr: The stderr value.

    Returns:
        The interactive REPL result.
    """
    print("ChainAgents CLI. Press Ctrl-D to exit.", file=stderr)
    while True:
        try:
            prompt = input("chainagents> ")
        except EOFError:
            print("", file=stderr)
            return 0
        except KeyboardInterrupt:
            print("", file=stderr)
            return 130
        if not prompt.strip():
            continue
        code = await run_agent_prompt(
            runtime,
            args,
            prompt=prompt,
            stdout=stdout,
            stderr=stderr,
        )
        if code not in (0,):
            return code


async def run_cli(
    args: argparse.Namespace,
    *,
    runtime: AgentRuntime,
    stdout: TextIO,
    stderr: TextIO,
    stdin: TextIO,
    parser: argparse.ArgumentParser | None = None,
) -> int:
    """Run the CLI with parsed arguments and runtime configuration.

    Args:
        args: Parsed command-line arguments.
        runtime: Agent runtime used by the operation.
        stdout: The stdout value.
        stderr: The stderr value.
        stdin: The stdin value.
        parser: The parser value.

    Returns:
        The command result.
    """
    prompt = prompt_from_args(args, stdin=stdin, parser=parser)
    has_prompt = bool(prompt and prompt.strip())

    json_actions: dict[str, Any] = {}

    if args.photo and not has_prompt:
        print("photo: provide a prompt with --photo.", file=stderr)
        return 2

    if args.status:
        if args.json_output:
            json_actions["status"] = runtime_status_payload(runtime)
        else:
            print_runtime_status(runtime, stdout=stdout, json_output=False)
    if args.list_commands:
        if args.json_output:
            json_actions["commands"] = [
                {
                    "name": command.name,
                    "description": command.description,
                    "target": command.target,
                    "value": command.value,
                    "source": command.source,
                }
                for command in runtime.chainlit_commands
            ]
            json_actions["notes"] = list(runtime.chainlit_command_notes)
        else:
            print_command_list(runtime, stdout=stdout, json_output=False)
    if args.rebuild_rag:
        status = await runtime.rebuild_rag_index()
        if args.json_output:
            json_actions["rebuild_rag"] = rag_status_payload(status)
        else:
            print_rag_status(
                status=status,
                action="rebuild_rag",
                stdout=stdout,
                json_output=False,
            )

    upload_result = await ingest_uploads(
        runtime,
        paths=args.upload_rag,
        thread_id=args.thread_id,
        stdout=stdout,
        stderr=stderr,
        json_output=False,
        emit_output=not args.json_output,
    )
    if args.upload_rag and args.json_output and upload_result is not None:
        json_actions["upload_rag"] = upload_result_payload(upload_result)

    if upload_result is not None and upload_result.reason is not None:
        return 1

    if has_prompt:
        prompt_result = await run_agent_prompt(
            runtime,
            args,
            prompt=prompt or "",
            stdout=stdout,
            stderr=stderr,
            emit_json=not args.json_output,
        )
        if args.json_output:
            if isinstance(prompt_result, dict):
                json_actions.update(prompt_result)
                print(json.dumps(json_actions, indent=2, sort_keys=True), file=stdout)
                return 0
            return int(prompt_result)
        return int(prompt_result)

    if args.status or args.list_commands or args.rebuild_rag or args.upload_rag:
        if args.json_output:
            print(json.dumps(json_actions, indent=2, sort_keys=True), file=stdout)
        return 0

    return await interactive_repl(
        runtime,
        args,
        stdout=stdout,
        stderr=stderr,
    )


async def async_main(argv: list[str] | None = None) -> int:
    """Run the asynchronous command-line entry point.

    Args:
        argv: The argv value.

    Returns:
        The async main result.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    config = RuntimeConfig.from_env(runtime_overrides_from_args(args))
    runtime = await AgentRuntime.create(config)
    try:
        return await run_cli(
            args,
            runtime=runtime,
            stdout=sys.stdout,
            stderr=sys.stderr,
            stdin=sys.stdin,
            parser=parser,
        )
    finally:
        await runtime.close()


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point.

    Args:
        argv: The argv value.

    Returns:
        The main result.
    """
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
