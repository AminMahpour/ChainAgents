#!/usr/bin/env python3
"""Provide the terminal CLI for ChainAgents prompts and runtime commands."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import sys
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, TextIO

from chainagents.runtime.background_tasks import BackgroundTaskSnapshot
from chainagents.runtime import (
    AgentRuntime,
    ReasoningLevel,
    RuntimeConfig,
    RuntimeConfigOverrides,
    shutdown_langfuse_client,
    normalize_reasoning_level,
)
from chainagents.turns import TurnRequest, TurnRunner
from chainagents.rag.runtime import RagUploadResult, UploadedRagFile

from chainagents.interfaces.cli.args import (
    DEFAULT_CLI_THREAD_ID,
    build_parser,
    parse_args,  # noqa: F401
)
from chainagents.interfaces.cli.configure import (
    CONFIGURE_PROMPTS,  # noqa: F401
    parse_config_prompt_value,  # noqa: F401
    resolve_configure_config_path,
    run_configure_command,
)
from chainagents.interfaces.cli.render import (
    CliEventRenderer,
    generated_file_paths,
    is_command_output,
    print_command_list,
    print_rag_status,
    print_runtime_status,
    print_upload_result,
    rag_status_payload,
    runtime_status_payload,
    truncate_tool_result_content,  # noqa: F401
    upload_result_payload,
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

    def _resolve_uploads() -> tuple[list[UploadedRagFile], Path | None]:
        resolved: list[UploadedRagFile] = []
        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists() or not path.is_file():
                return resolved, path
            resolved.append(UploadedRagFile(path=path, name=path.name))
        return resolved, None

    uploads, missing_path = await asyncio.to_thread(_resolve_uploads)
    if missing_path is not None:
        print(f"upload-rag: file does not exist: {missing_path}", file=stderr)
        return RagUploadResult(
            thread_id=thread_id,
            success=False,
            reason=f"file does not exist: {missing_path}",
        )

    result = await runtime.ingest_rag_uploads(thread_id=thread_id, uploads=uploads)
    if emit_output:
        print_upload_result(result, stdout=stdout, json_output=json_output)
    return result


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
        model_disable_streaming=(
            "tool_calling" if args.disable_streaming_for_tool_calls else None
        ),
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


async def run_agent_prompt(
    runtime: AgentRuntime,
    args: argparse.Namespace,
    *,
    prompt: str,
    stdout: TextIO,
    stderr: TextIO,
    emit_json: bool = True,
) -> int | dict[str, Any]:
    """Run one CLI prompt through the shared turn runner.

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
    model_name = args.model or runtime.config.model_name
    photos = photo_content_parts(args.photo, stderr=stderr)
    if photos is None:
        return 1

    renderer = CliEventRenderer(
        stdout=stdout,
        stderr=stderr,
        stream=args.stream,
        json_output=args.json_output,
        show_reasoning=args.show_reasoning,
        show_tools=args.show_tools,
    )
    result = await TurnRunner(runtime, sanitize_errors=False).run(
        TurnRequest(
            prompt=prompt,
            thread_id=thread_id,
            model_name=model_name,
            reasoning_level=reasoning_level,
            selected_command=args.command,
            reasoning_level_is_explicit=args.reasoning is not None,
            content_parts=tuple(photos),
            async_subagent_url=args.async_subagent_url,
            mcp_session_id=args.mcp_session_id,
        ),
        renderer,
    )
    if result.command_error is not None:
        return 2 if result.command_error.unknown else 1
    if result.status == "failed":
        return 1
    if result.status == "skipped" or is_command_output(result) or not args.json_output:
        return 0

    payload: dict[str, Any] = {
        "response": result.response,
        "thread_id": thread_id,
        "model": model_name,
        "reasoning": reasoning_level,
    }
    if result.reflection is not None:
        payload["reflection_proposal"] = result.reflection.to_payload()
    if result.generated_files:
        payload["generated_files"] = generated_file_paths(result)
    if emit_json:
        print(json.dumps(payload, indent=2, sort_keys=True), file=stdout)
        return 0
    return {"prompt": payload}


async def _read_terminal_line(
    *,
    stdin: TextIO,
    stdout: TextIO,
    prompt: str,
) -> str:
    """Read one line without blocking the event loop or its default executor."""
    stdout.write(prompt)
    stdout.flush()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def deliver_result(line: str) -> None:
        if not future.done():
            future.set_result(line)

    def deliver_error(exc: BaseException) -> None:
        if not future.done():
            future.set_exception(exc)

    def read_in_daemon_thread() -> None:
        try:
            line = stdin.readline()
        except BaseException as exc:  # pragma: no cover - device-specific failures
            try:
                loop.call_soon_threadsafe(deliver_error, exc)
            except RuntimeError:
                return
        else:
            try:
                loop.call_soon_threadsafe(deliver_result, line)
            except RuntimeError:
                return

    threading.Thread(target=read_in_daemon_thread, daemon=True).start()

    line = await future
    if line == "":
        raise EOFError
    return line.rstrip("\r\n")


async def interactive_repl(
    runtime: AgentRuntime,
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    stderr: TextIO,
    stdin: TextIO,
) -> int:
    """Run the interactive CLI prompt loop.

    Args:
        runtime: Agent runtime used by the operation.
        args: Parsed command-line arguments.
        stdout: The stdout value.
        stderr: The stderr value.
        stdin: The stdin value.

    Returns:
        The interactive REPL result.
    """
    thread_id = str(args.thread_id or DEFAULT_CLI_THREAD_ID).strip() or DEFAULT_CLI_THREAD_ID
    queue = runtime.background_tasks.subscribe(thread_id)

    async def print_completions() -> None:
        while True:
            snapshot = await queue.get()
            print(format_background_task_notice(snapshot), file=stderr)

    notice_task = asyncio.create_task(print_completions())
    print("ChainAgents CLI. Press Ctrl-D to exit.", file=stderr)
    try:
        while True:
            try:
                prompt = await _read_terminal_line(
                    stdin=stdin,
                    stdout=stdout,
                    prompt="chainagents> ",
                )
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
    finally:
        runtime.background_tasks.unsubscribe(thread_id, queue)
        notice_task.cancel()
        with suppress(asyncio.CancelledError):
            await notice_task


def format_background_task_notice(
    snapshot: BackgroundTaskSnapshot,
    *,
    include_result: bool = False,
) -> str:
    """Format one terminal local task notice for a CLI delivery mode."""
    message = (
        f"Background subagent {snapshot.agent_name} finished with status "
        f"{snapshot.status}.\nTask ID: {snapshot.task_id}"
    )
    if snapshot.error:
        return f"{message}\nError: {snapshot.error}"
    if include_result and snapshot.result:
        return f"{message}\n{snapshot.result}"
    return message


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
    if args.configure:
        return run_configure_command(
            config_path=resolve_configure_config_path(args.config),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )

    if args.tui:
        if args.prompt or args.prompt_parts or args.stdin:
            print("tui: start the TUI without a one-shot prompt.", file=stderr)
            return 2
        unsupported_tui_flags = []
        if args.photo:
            unsupported_tui_flags.append("--photo")
        if args.command:
            unsupported_tui_flags.append("--command")
        if args.rebuild_rag:
            unsupported_tui_flags.append("--rebuild-rag")
        if args.upload_rag:
            unsupported_tui_flags.append("--upload-rag")
        if args.status:
            unsupported_tui_flags.append("--status")
        if args.list_commands:
            unsupported_tui_flags.append("--list-commands")
        if args.json_output:
            unsupported_tui_flags.append("--json")
        if unsupported_tui_flags:
            print(
                "tui: unsupported flags in TUI mode: "
                + ", ".join(unsupported_tui_flags),
                file=stderr,
            )
            return 2
        from chainagents.interfaces.tui.app import run_tui

        return await run_tui(runtime, args)

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

    thread_id = str(args.thread_id or DEFAULT_CLI_THREAD_ID).strip() or DEFAULT_CLI_THREAD_ID
    upload_result = await ingest_uploads(
        runtime,
        paths=args.upload_rag,
        thread_id=thread_id,
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
        background_snapshots = []
        if isinstance(prompt_result, dict) or int(prompt_result) == 0:
            background_snapshots = await runtime.background_tasks.wait_session(thread_id)
        if args.json_output:
            if isinstance(prompt_result, dict):
                json_actions.update(prompt_result)
                json_actions["background_tasks"] = [
                    snapshot.to_payload() for snapshot in background_snapshots
                ]
                print(json.dumps(json_actions, indent=2, sort_keys=True), file=stdout)
                return 0
            return int(prompt_result)
        for snapshot in background_snapshots:
            print(
                format_background_task_notice(snapshot, include_result=True),
                file=stderr,
            )
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
        stdin=stdin,
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
    if args.configure:
        return run_configure_command(
            config_path=resolve_configure_config_path(args.config),
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

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
        try:
            await runtime.close()
        finally:
            shutdown_langfuse_client(config)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line entry point.

    Args:
        argv: The argv value.

    Returns:
        The main result.
    """
    try:
        return asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
