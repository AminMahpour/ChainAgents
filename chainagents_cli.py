#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO

from rich import box
from rich.console import Console
from rich.panel import Panel
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
LANGGRAPH_STREAM_MODES = {
    "values",
    "updates",
    "custom",
    "messages",
    "checkpoints",
    "tasks",
    "debug",
}


def build_parser() -> argparse.ArgumentParser:
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

    parser.add_argument("--config", help="Path to deepagent.toml.")
    parser.add_argument("--database-url", help="Postgres URL for durable state.")
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="Force in-memory state even when DATABASE_URL is set.",
    )
    parser.add_argument("--provider", help="Model provider: ollama or openai_compatible.")
    parser.add_argument("--base-url", help="Model server base URL.")
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
    return build_parser().parse_args(argv)


def runtime_overrides_from_args(args: argparse.Namespace) -> RuntimeConfigOverrides:
    return RuntimeConfigOverrides(
        config_path=args.config,
        database_url=args.database_url,
        disable_database=args.no_database,
        model_provider=args.provider,
        model_name=args.model,
        model_base_url=args.base_url,
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


def langgraph_part_from_event_chunk(chunk: Any) -> dict[str, Any] | None:
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
    content = stringify_content(value).strip()
    if len(content) <= TOOL_RESULT_PREVIEW_CHARS:
        return content
    return content[:TOOL_RESULT_PREVIEW_CHARS]


def reasoning_text_from_token(token: Any) -> str:
    if hasattr(token, "additional_kwargs"):
        text = stringify_content(token.additional_kwargs.get("reasoning_content"))
        if text:
            return text
    if hasattr(token, "reasoning_content"):
        return stringify_content(token.reasoning_content)
    return ""


def namespace_label(ns: tuple[str, ...], metadata: dict[str, Any]) -> str:
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
    if data is None:
        return []
    if isinstance(data, dict):
        return iter_messages(data.get("messages"))
    return iter_messages(data)


def is_assistant_message(message: Any) -> bool:
    return getattr(message, "type", None) in {"ai", "AIMessageChunk"}


def message_text(message: Any) -> str:
    return stringify_content(getattr(message, "content", "")).strip()


def assistant_messages_for_current_prompt(messages: list[Any], prompt: str) -> list[Any]:
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
        self.completed_tool_results: set[tuple[str, str, str]] = set()
        self.stdout_console = Console(
            file=stdout,
            highlight=False,
            soft_wrap=True,
        )
        self.stderr_console = Console(
            file=stderr,
            highlight=False,
            soft_wrap=True,
        )

    async def handle_event(self, event: dict[str, Any]) -> None:
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

    def finish(self) -> str:
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

    def _stream_response(self, text: str) -> None:
        delta = text[len(self.response_buffer) :] if text.startswith(self.response_buffer) else text
        if not delta:
            return
        self.response_buffer += delta
        if self.stream and not self.json_output:
            self.stdout_console.print(Text(delta, style="bright_white"), end="")

    def _stream_reasoning(self, source: str, text: str) -> None:
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
        if self.reasoning_line_source is None:
            return
        self.stderr_console.print()
        self.reasoning_line_source = None

    def _stream_tool_call(self, source: str, chunk: dict[str, Any]) -> None:
        call_id = str(chunk.get("id") or f"{source}:{chunk.get('index', '0')}")
        tool_name = str(chunk.get("name") or self.tool_names.get(call_id) or "tool")
        self.tool_names[call_id] = tool_name
        if self.show_tools and chunk.get("name"):
            self._close_reasoning_line()
            self.stderr_console.print(
                Panel(
                    Text.assemble(
                        ("status: ", "dim"),
                        ("start", "bold yellow"),
                        ("\nsource: ", "dim"),
                        (source, "cyan"),
                        ("\ntool: ", "dim"),
                        (tool_name, "bold white"),
                    ),
                    title="Tool Call",
                    title_align="left",
                    border_style="yellow",
                    box=box.ASCII,
                )
            )

    def _complete_tool(self, source: str, tool_message: Any) -> None:
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
            Panel(
                body,
                title="Tool Result",
                title_align="left",
                border_style="red" if status.lower() == "error" else "green",
                box=box.ASCII,
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
        stable_id = str(
            getattr(tool_message, "tool_call_id", None)
            or getattr(tool_message, "id", None)
            or ""
        ).strip()
        if stable_id:
            return (source, "id", stable_id)
        return (source, name, content)


def rag_status_payload(status: RagStatus) -> dict[str, Any]:
    return {
        "enabled": status.enabled,
        "ready": status.ready,
        "file_count": status.file_count,
        "chunk_count": status.chunk_count,
        "reason": status.reason,
        "persist_directory": str(status.persist_directory) if status.persist_directory else None,
    }


def upload_result_payload(result: RagUploadResult) -> dict[str, Any]:
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


def print_runtime_status(
    runtime: AgentRuntime,
    *,
    stdout: TextIO,
    json_output: bool,
) -> None:
    payload = runtime_status_payload(runtime)
    if json_output:
        print(json.dumps({"status": payload}, indent=2, sort_keys=True), file=stdout)
        return

    print("ChainAgents runtime", file=stdout)
    print(f"- Model provider: {payload['model_provider_label']}", file=stdout)
    print(f"- Model: {payload['model']}", file=stdout)
    print(f"- Reasoning: {payload['reasoning']}", file=stdout)
    print(f"- Recursion limit: {payload['recursion_limit']}", file=stdout)
    print(f"- Persistence: {payload['persistence']}", file=stdout)
    rag = payload["rag"]
    if rag["enabled"] and rag["ready"]:
        print(
            f"- RAG: ready ({rag['file_count']} files, {rag['chunk_count']} chunks)",
            file=stdout,
        )
    elif rag["enabled"]:
        print(f"- RAG: unavailable ({rag['reason'] or 'unknown error'})", file=stdout)
    else:
        print("- RAG: disabled", file=stdout)
    print(f"- Commands: {len(payload['commands'])}", file=stdout)


def print_command_list(
    runtime: AgentRuntime,
    *,
    stdout: TextIO,
    json_output: bool,
) -> None:
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

    if not commands:
        print("No configured commands.", file=stdout)
    else:
        for command in commands:
            print(
                f"/{command['name']} ({command['target']}): {command['description']}",
                file=stdout,
            )
    for note in runtime.chainlit_command_notes:
        print(f"note: {note}", file=stdout)


def print_rag_status(
    *,
    status: RagStatus,
    action: str,
    stdout: TextIO,
    json_output: bool,
) -> None:
    payload = rag_status_payload(status)
    if json_output:
        print(json.dumps({action: payload}, indent=2, sort_keys=True), file=stdout)
        return
    if status.ready:
        print(
            f"{action}: ready ({status.file_count} files, {status.chunk_count} chunks)",
            file=stdout,
        )
    elif status.enabled:
        print(f"{action}: unavailable ({status.reason or 'unknown error'})", file=stdout)
    else:
        print(f"{action}: disabled", file=stdout)


def print_upload_result(
    result: RagUploadResult,
    *,
    stdout: TextIO,
    json_output: bool,
) -> None:
    payload = upload_result_payload(result)
    if json_output:
        print(json.dumps({"upload_rag": payload}, indent=2, sort_keys=True), file=stdout)
        return
    if result.added_files:
        print(
            "upload-rag: added "
            f"{', '.join(result.added_files)} "
            f"({result.indexed_files} files, {result.chunk_count} chunks)",
            file=stdout,
        )
    elif result.reason:
        print(f"upload-rag: {result.reason}", file=stdout)
    if result.rejected_files:
        print(f"upload-rag: rejected {', '.join(result.rejected_files)}", file=stdout)


async def ingest_uploads(
    runtime: AgentRuntime,
    *,
    paths: list[str],
    thread_id: str,
    stdout: TextIO,
    stderr: TextIO,
    json_output: bool,
) -> RagUploadResult | None:
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

    agent = await runtime.get_agent(
        settings.reasoning_level,
        model_name=settings.model_name,
        thread_id=settings.thread_id,
        async_subagent_url_override=args.async_subagent_url,
        mcp_session_id=args.mcp_session_id,
    )
    payload = {"messages": [{"role": "user", "content": prompt}]}
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
        stream_mode=["messages", "updates"],
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
    prompt = prompt_from_args(args, stdin=stdin, parser=parser)
    has_prompt = bool(prompt and prompt.strip())

    json_actions: dict[str, Any] = {}

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
        json_output=not args.json_output,
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
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
