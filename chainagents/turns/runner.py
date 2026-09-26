"""Run one agent turn through the shared command, stream and finish flow."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from chainagents.commands.native import (
    RuntimeCommandResult,
    dumps_tool_result,
    resolve_native_command,
    resolve_runtime_command,
)
from chainagents.events.stream import (
    AGENT_STREAM_MODES,
    AgentStreamEvent,
    AgentStreamEventAdapter,
)
from chainagents.exports.generated_files import (
    GeneratedFileDescriptor,
    generated_file_descriptors,
    generated_file_paths_from_text,
    generated_file_paths_from_tool_args,
    generated_file_paths_from_tool_result,
)
from chainagents.interfaces.uploads import prompt_with_images
from chainagents.runtime.constants import ReasoningLevel
from chainagents.runtime.lifecycle import agent_with_mcp_status, mcp_outage_warning
from chainagents.runtime.reflection import ReflectionCollector, ReflectionProposal
from chainagents.runtime.tracing import build_langgraph_run_config
from chainagents.turns.renderer import TurnRenderer


logger = logging.getLogger(__name__)

DEFAULT_BACKEND_ERROR = "Agent operation failed. Please retry."
TurnStatus = Literal["completed", "skipped", "command_error", "failed"]


@dataclass(frozen=True)
class TurnRequest:
    """Front-end input for one agent turn.

    ``prompt`` is the raw user text used for native command resolution. After
    resolution the agent prompt is decorated with ``image_names`` and
    ``prompt_note`` (see ``prompt_with_images``) and sent with
    ``content_parts`` (image parts) after ``history``. ``run_config_extras``
    keys shallowly override the keys of ``build_langgraph_run_config`` (for
    example a ``callbacks`` or ``metadata`` entry replaces the tracing value).
    """

    prompt: str
    thread_id: str
    model_name: str
    reasoning_level: ReasoningLevel
    selected_command: str | None = None
    reasoning_level_is_explicit: bool = False
    content_parts: tuple[dict[str, Any], ...] = ()
    history: tuple[dict[str, Any], ...] = ()
    image_names: tuple[str, ...] = ()
    prompt_note: str = ""
    async_subagent_url: str | None = None
    mcp_session_id: str | None = None
    run_config_extras: Mapping[str, Any] = field(default_factory=dict)


class TurnCommandError(Exception):
    """A failed or unknown native command.

    ``message`` is user-safe when the runner sanitises errors, otherwise it is
    the real exception text; the original exception is ``__cause__``.
    """

    def __init__(
        self,
        message: str,
        *,
        command_name: str,
        status: int,
        unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.command_name = command_name
        self.status = status
        self.unknown = unknown


@dataclass
class TurnResult:
    """Outcome of one ``TurnRunner.run`` call that was not cancelled.

    ``status`` is ``completed`` (agent run or MCP-tool command succeeded),
    ``skipped`` (a command resolved to a blank prompt without images, so no
    agent ran), ``command_error`` or ``failed`` (the agent run raised).
    """

    status: TurnStatus
    prompt: str
    response: str = ""
    command_result: RuntimeCommandResult | None = None
    command_error: TurnCommandError | None = None
    error: Exception | None = None
    generated_files: list[GeneratedFileDescriptor] = field(default_factory=list)
    reflection: ReflectionProposal | None = None
    agent: Any | None = None

    @property
    def ok(self) -> bool:
        """Return whether the turn ended without a command or agent error."""
        return self.status in {"completed", "skipped"}


class TurnRunner:
    """Own the canonical flow of one agent turn for every front end."""

    def __init__(self, runtime: Any, *, sanitize_errors: bool = True) -> None:
        """Bind the runner to ``runtime``.

        ``sanitize_errors`` replaces command-error messages and the
        reflection-proposal failure text with fixed messages (logging the real
        exception). Keep it on wherever errors cross the network (the API);
        local front ends pass False to show the real exception text.
        """
        self.runtime = runtime
        self.sanitize_errors = sanitize_errors

    async def run(self, request: TurnRequest, renderer: TurnRenderer) -> TurnResult:
        """Run one turn, reporting output to ``renderer``.

        Command and agent failures are reported to the renderer and returned in
        the result; they are not raised. A renderer callback that raises while
        events stream counts as an agent failure, but one that raises during
        the finish steps (``on_generated_files``, ``on_reflection``,
        ``on_error``, ``on_complete``) or a command outcome propagates out of
        ``run``. ``CancelledError`` is re-raised after ``renderer.on_cancelled()``.
        """
        try:
            return await self._run(request, renderer)
        except asyncio.CancelledError:
            try:
                await renderer.on_cancelled()
            except Exception:
                logger.exception("Turn renderer failed while handling cancellation.")
            raise

    async def _run(self, request: TurnRequest, renderer: TurnRenderer) -> TurnResult:
        prompt = request.prompt
        command_result: RuntimeCommandResult | None = None
        parsed = resolve_native_command(
            raw_text=request.prompt,
            selected_command=request.selected_command,
        )
        if parsed is not None:
            command_error: TurnCommandError | None = None
            try:
                command_result = await resolve_runtime_command(
                    runtime=self.runtime,
                    parsed=parsed,
                    thread_id=request.thread_id,
                    mcp_session_id=request.mcp_session_id,
                )
            except ValueError as exc:
                command_error = TurnCommandError(
                    (
                        safe_command_validation_error(exc)
                        if self.sanitize_errors
                        else str(exc)
                    ),
                    command_name=parsed.command_name,
                    status=422,
                )
                command_error.__cause__ = exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                command_error = TurnCommandError(
                    safe_backend_error(exc) if self.sanitize_errors else str(exc),
                    command_name=parsed.command_name,
                    status=500,
                )
                command_error.__cause__ = exc
            # resolve_native_command only parses a selected command or typed
            # ``/name`` text, so an unknown parsed command is always an error.
            if command_result is not None and command_result.target == "unknown":
                command_error = TurnCommandError(
                    f"Unknown command `/{command_result.command_name}`.",
                    command_name=command_result.command_name,
                    status=422,
                    unknown=True,
                )
            if command_error is not None:
                await renderer.on_command_error(command_error, command_error.status)
                result = TurnResult(
                    status="command_error",
                    prompt=request.prompt,
                    command_error=command_error,
                )
                await renderer.on_complete(result)
                return result
            if command_result is not None and command_result.target == "mcp_tool":
                return await self._finish_command_output(
                    request, command_result, renderer
                )
            if command_result is not None:
                prompt = (command_result.prompt or "").strip()

        if not prompt.strip() and not request.content_parts:
            result = TurnResult(
                status="skipped", prompt="", command_result=command_result
            )
            await renderer.on_complete(result)
            return result

        prompt = prompt_with_images(
            prompt,
            image_names=request.image_names,
            prompt_note=request.prompt_note,
        )
        return await self._run_agent(request, prompt, command_result, renderer)

    async def _finish_command_output(
        self,
        request: TurnRequest,
        command_result: RuntimeCommandResult,
        renderer: TurnRenderer,
    ) -> TurnResult:
        """Treat MCP-tool output as the turn response and finish the turn."""
        await renderer.on_command_result(command_result)
        result = TurnResult(
            status="completed",
            prompt=request.prompt,
            response=dumps_tool_result(command_result.tool_result),
            command_result=command_result,
        )
        # The command output is not an agent response, so it is not recorded;
        # the proposal can still come from the user's own text.
        collector = ReflectionCollector.from_runtime_config(
            self.runtime.config,
            prompt=request.prompt,
        )
        await self._finish(result, collector, [], renderer)
        return result

    async def _run_agent(
        self,
        request: TurnRequest,
        prompt: str,
        command_result: RuntimeCommandResult | None,
        renderer: TurnRenderer,
    ) -> TurnResult:
        result = TurnResult(
            status="completed", prompt=prompt, command_result=command_result
        )
        collector = ReflectionCollector.from_runtime_config(
            self.runtime.config,
            prompt=prompt,
        )
        tracker = _GeneratedPathTracker()
        response_parts: list[str] = []
        try:
            agent, mcp_failures = await agent_with_mcp_status(
                self.runtime,
                request.reasoning_level,
                model_name=request.model_name,
                reasoning_level_is_explicit=request.reasoning_level_is_explicit,
                thread_id=request.thread_id,
                async_subagent_url_override=request.async_subagent_url,
                mcp_session_id=request.mcp_session_id,
            )
            result.agent = agent
            if mcp_failures:
                await renderer.on_event(
                    AgentStreamEvent(
                        kind="mcp_status",
                        source="mcp",
                        status="warning",
                        text=mcp_outage_warning(mcp_failures),
                    )
                )
            async with aclosing(self._agent_events(agent, request, prompt)) as events:
                async for event in events:
                    collector.record_event(event)
                    tracker.record(event)
                    if event.kind == "response_delta":
                        response_parts.append(event.text)
                    await renderer.on_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A sanitised proposal may cross the network, so omit backend text.
            collector.mark_run_failed(
                RuntimeError("Agent operation failed.") if self.sanitize_errors else exc
            )
            result.status = "failed"
            result.error = exc
        result.response = "".join(response_parts)
        await self._finish(result, collector, tracker.paths, renderer)
        return result

    async def _agent_events(
        self,
        agent: Any,
        request: TurnRequest,
        prompt: str,
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """Stream normalized events for one agent run, always closing the stream."""
        payload = {
            "messages": [
                *request.history,
                {
                    "role": "user",
                    "content": (
                        [{"type": "text", "text": prompt}, *request.content_parts]
                        if request.content_parts
                        else prompt
                    ),
                },
            ]
        }
        config = build_langgraph_run_config(
            self.runtime.config,
            thread_id=request.thread_id,
            langsmith_tracing=getattr(self.runtime, "langsmith_tracing", None),
        )
        config.update(request.run_config_extras)
        adapter = AgentStreamEventAdapter(prompt=prompt)
        stream = agent.astream_events(
            payload,
            config=config,
            version="v2",
            stream_mode=AGENT_STREAM_MODES,
            subgraphs=True,
        )
        try:
            async for raw_event in stream:
                for event in adapter.events_from_raw_event(raw_event):
                    yield event
        finally:
            with suppress(Exception):
                await stream.aclose()

    async def _finish(
        self,
        result: TurnResult,
        collector: ReflectionCollector,
        generated_paths: list[str],
        renderer: TurnRenderer,
    ) -> None:
        """Emit generated files, reflection, the failure (if any) and completion."""
        raw_paths = [
            *generated_paths,
            *generated_file_paths_from_text(result.response),
        ]
        if raw_paths:
            result.generated_files = generated_file_descriptors(
                raw_paths,
                project_root=self._project_root(),
            )
        if result.generated_files:
            await renderer.on_generated_files(result.generated_files)
        result.reflection = collector.build_proposal()
        if result.reflection is not None:
            await renderer.on_reflection(result.reflection)
        if result.error is not None:
            await renderer.on_error(result.error)
        await renderer.on_complete(result)

    def _project_root(self) -> Path | None:
        project_root = getattr(self.runtime, "project_root", None)
        return Path(project_root) if project_root is not None else None


class _GeneratedPathTracker:
    """Collect generated-file paths from successful write tool calls."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self._tool_calls: dict[str, tuple[str, str]] = {}

    def record(self, event: AgentStreamEvent) -> None:
        if event.kind == "tool_call":
            if event.previous_tool_call_id:
                self._tool_calls.pop(event.previous_tool_call_id, None)
            self._tool_calls[event.tool_call_id] = (event.tool_name, event.tool_args)
        elif event.kind == "tool_result":
            tool_name, tool_args = self._tool_calls.pop(
                event.tool_call_id,
                (event.tool_name, ""),
            )
            if event.status.lower() != "error":
                self.paths.extend(
                    generated_file_paths_from_tool_args(tool_name, tool_args)
                )
                self.paths.extend(
                    generated_file_paths_from_tool_result(tool_name, event.tool_result)
                )


def safe_backend_error(exc: Exception, message: str = DEFAULT_BACKEND_ERROR) -> str:
    """Log ``exc`` with its traceback and return a fixed user-safe message."""
    logger.error(message, exc_info=(type(exc), exc, exc.__traceback__))
    return message


def safe_command_validation_error(exc: ValueError) -> str:
    """Return a user-safe message for a native command ``ValueError``."""
    # Runtime command JSON validation has a stable correction message. Other
    # ValueErrors can originate from an MCP backend and must not be echoed.
    message = str(exc)
    if message.startswith("Command arguments") and message.endswith(
        "must be valid JSON."
    ):
        return "Command arguments must be valid JSON."
    return safe_backend_error(exc, "Command arguments could not be validated.")
