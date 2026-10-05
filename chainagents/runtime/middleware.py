"""Tool resilience, summarization, and agent middleware construction."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest, hook_config
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.errors import GraphBubbleUp, GraphRecursionError
from langgraph.types import Command

import chainagents.runtime.backends as runtime_backends
import chainagents.runtime.constants as runtime_constants
import chainagents.runtime.models as runtime_models
from chainagents.runtime.config import RuntimeConfig
from chainagents.runtime.constants import (
    DEFAULT_DEEPAGENT_FILESYSTEM_TOOLS,
    SUMMARIZATION_STATUS_EVENT_KIND,
    ReasoningLevel,
)

logger = logging.getLogger("chainagents.runtime.core")


_DEEPAGENTS_SUMMARIZATION_FACTORY_LOCK = threading.RLock()
_TOKEN_LIMIT_RETRY_MARKER = "chainagents_token_limit_retry"

# Exceptions that must propagate instead of becoming retryable tool errors:
# cancellation, LangGraph control flow (interrupts, parent commands), and an
# exhausted step budget. Swallowing a GraphRecursionError would let a parent
# delegate again with a fresh budget.
NON_RECOVERABLE_TOOL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.CancelledError,
    GraphBubbleUp,
    GraphRecursionError,
)
REPEATED_TOOL_RESULT_LIMIT = 10


class TokenLimitedToolCallMiddleware(AgentMiddleware[Any, Any, Any]):
    """Prevent execution of model tool calls cut off by an output-token cap."""

    @hook_config(can_jump_to=["model", "end"])
    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        message = messages[-1]
        metadata = message.response_metadata
        token_limited = (
            metadata.get("finish_reason") in {"length", "max_tokens"}
            or metadata.get("stop_reason") == "max_tokens"
            or metadata.get("done_reason") == "length"
            or (
                metadata.get("status") == "incomplete"
                and isinstance(metadata.get("incomplete_details"), dict)
                and metadata["incomplete_details"].get("reason") == "max_output_tokens"
            )
        )
        raw_calls = message.additional_kwargs.get("tool_calls")
        if not token_limited or not (message.tool_calls or message.invalid_tool_calls or raw_calls):
            return None

        # Drop the incomplete assistant turn before another model request. This
        # also prevents any valid call in the same truncated batch from running.
        replacement: list[Any] = [RemoveMessage(id=message.id)] if message.id else []
        already_retried = False
        for prior in reversed(messages[:-1]):
            if isinstance(prior, HumanMessage):
                already_retried = bool(
                    prior.additional_kwargs.get(_TOKEN_LIMIT_RETRY_MARKER)
                )
                break
        if already_retried or not message.id:
            replacement.append(
                AIMessage(
                    content=(
                        "The model reached its output-token limit while generating a tool call "
                        "again. The incomplete tool call was not executed. Increase the model "
                        "output-token limit or use a smaller request."
                    )
                )
            )
            return {"messages": replacement, "jump_to": "end"}

        replacement.append(
            HumanMessage(
                content=(
                    "Your previous tool call was incomplete because you reached the "
                    "output-token limit. It was not executed. Shorten or split the tool "
                    "call; do not repeat the same long call. You have one retry."
                ),
                additional_kwargs={_TOKEN_LIMIT_RETRY_MARKER: True},
            )
        )
        return {"messages": replacement, "jump_to": "model"}


_OFFLOADED_TOOL_RESULT_PREFIX = "Tool result too large, the result of this tool call "


def _tool_message_text(content: Any) -> str | None:
    """Return a stable string form of ToolMessage content for comparison.

    Returns None for a result DeepAgents offloaded to a file: its stub keeps
    only a head/tail preview, so equal stubs do not prove equal results.
    """
    text = content if isinstance(content, str) else json.dumps(
        content, sort_keys=True, default=str
    )
    if text.startswith(_OFFLOADED_TOOL_RESULT_PREFIX):
        return None
    return text


# Synthetic HumanMessages that are not a new user turn: media DeepAgents'
# read_file appends after its result, and the token-limit retry notice.
_SYNTHETIC_HUMAN_MESSAGE_MARKERS = ("read_file_media_result", _TOKEN_LIMIT_RETRY_MARKER)


def _is_user_turn(message: Any) -> bool:
    """Return whether ``message`` starts a new user turn."""
    return isinstance(message, HumanMessage) and not any(
        message.additional_kwargs.get(marker) for marker in _SYNTHETIC_HUMAN_MESSAGE_MARKERS
    )


class RepeatedToolResultGuardMiddleware(AgentMiddleware[Any, Any, Any]):
    """End a run stuck calling one tool with identical arguments and results.

    The guard reads message history instead of holding instance state, so one
    instance is safe across concurrent runs. Only the current turn (messages
    after the latest human message) is inspected, and repeats count even when
    interleaved with other calls. The count for a tool and its arguments
    restarts whenever the result changes, so polling that eventually reports
    a new status is treated as progress, as is the same result for different
    arguments.
    """

    def __init__(self, *, limit: int = REPEATED_TOOL_RESULT_LIMIT) -> None:
        super().__init__()
        self.limit = limit

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        turn_start = 0
        for index in range(len(messages) - 1, -1, -1):
            if _is_user_turn(messages[index]):
                turn_start = index + 1
                break

        call_args: dict[str, tuple[str, str]] = {}
        # Latest result and its run length for each (tool, arguments) pair.
        streaks: dict[tuple[str, str], tuple[str | None, int]] = {}
        for message in messages[turn_start:]:
            if isinstance(message, AIMessage):
                for tool_call in message.tool_calls:
                    call_id = tool_call.get("id")
                    if call_id:
                        call_args[call_id] = (
                            str(tool_call.get("name") or ""),
                            json.dumps(tool_call.get("args"), sort_keys=True, default=str),
                        )
                continue
            if not isinstance(message, ToolMessage):
                continue
            call = call_args.get(message.tool_call_id)
            if call is None:
                continue
            result = _tool_message_text(message.content)
            if result is None:
                # An offloaded result cannot be compared; leave the streak as is.
                continue
            previous_result, previous_count = streaks.get(call, (None, 0))
            count = previous_count + 1 if previous_result == result else 1
            streaks[call] = (result, count)
            if count < self.limit:
                continue
            tool_name, arguments = call
            logger.warning(
                "Stopping a non-converging run: %s was called %d times with "
                "arguments %s and returned the same result each time.",
                tool_name,
                count,
                arguments[:200],
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"Stopped: `{tool_name}` was called {count} times with "
                            "identical arguments and returned the same result each time, "
                            "so the run was not making progress."
                        )
                    )
                ],
                "jump_to": "end",
            }
        return None


class DisableSubagentDelegationMiddleware(AgentMiddleware[Any, Any, Any]):
    """Replace DeepAgents' implicit subagent middleware with an empty slot.

    DeepAgents adds a general-purpose delegate when a compiled agent has no
    configured synchronous children. Its supported middleware merge replaces
    entries by name, so this no-op slot preserves a true leaf agent without
    changing harness profiles shared by parent agents using the same model.
    """

    name = "SubAgentMiddleware"


def summarize_tool_exception(exc: Exception, *, limit: int = 400) -> str:
    """Summarize tool exception.

    Args:
        exc: The exc value.
        limit: The limit value.

    Returns:
        The summary value.
    """
    detail = " ".join(str(exc).split()).strip()
    if not detail:
        return exc.__class__.__name__
    summary = detail
    if detail != exc.__class__.__name__:
        summary = f"{exc.__class__.__name__}: {detail}"
    if len(summary) > limit:
        return f"{summary[: limit - 3].rstrip()}..."
    return summary


class ToolExecutionResilienceMiddleware(AgentMiddleware[Any, Any, Any]):
    """Wrap tool execution with workspace path mapping and recoverable errors."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        virtual_filesystem_tools: bool = True,
    ) -> None:
        """Initialize the tool execution resilience middleware instance.

        Args:
            project_root: Project root used to resolve local paths.
            virtual_filesystem_tools: Whether the filesystem tools' backend
                routes ``/workspace/`` itself. When it does not (for example
                a plain non-virtual filesystem backend), their ``/workspace/``
                paths are mapped to real paths like any other tool's.
        """
        self.project_root = (project_root or runtime_constants.PROJECT_ROOT).resolve()
        self.virtual_filesystem_tools = virtual_filesystem_tools

    def _map_workspace_path_args(self, request: ToolCallRequest) -> None:
        """Map virtual workspace paths inside tool-call arguments.

        Args:
            request: The request value.
        """
        args = request.tool_call.get("args")
        tool_name = str(request.tool_call.get("name") or "")
        if (
            self.virtual_filesystem_tools
            and tool_name in runtime_backends.BACKEND_FILESYSTEM_TOOLS
        ):
            mapped_args = runtime_backends.map_backend_tool_paths_to_virtual(
                args, self.project_root
            )
        else:
            mapped_args = runtime_backends.map_workspace_paths_in_tool_args(
                args, self.project_root
            )
        if mapped_args is not args:
            request.tool_call["args"] = mapped_args

    def _error_tool_message(
        self,
        request: ToolCallRequest,
        exc: Exception,
    ) -> ToolMessage:
        """Build a ToolMessage describing a recoverable tool failure.

        Args:
            request: The request value.
            exc: The exc value.

        Returns:
            The constructed a toolmessage describing a recoverable tool failure.
        """
        tool_name = (
            str(request.tool_call.get("name") or getattr(request.tool, "name", "tool")).strip()
            or "tool"
        )
        tool_call_id = str(request.tool_call.get("id") or tool_name)
        summary = summarize_tool_exception(exc)
        logger.exception(
            "Tool call failed without aborting the run: %s (%s)",
            tool_name,
            tool_call_id,
            exc_info=exc,
        )
        return ToolMessage(
            content=(
                f"Tool execution failed for `{tool_name}`: {summary}\n\n"
                "The tool error was returned without aborting the run. "
                "Adjust the tool inputs or continue with another approach."
            ),
            name=tool_name,
            tool_call_id=tool_call_id,
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Wrap synchronous tool calls with path mapping and error handling.

        Args:
            request: The request value.
            handler: The handler value.

        Returns:
            The wrap tool call result.
        """
        try:
            self._map_workspace_path_args(request)
            return handler(request)
        except NON_RECOVERABLE_TOOL_EXCEPTIONS:
            raise
        except Exception as exc:
            return self._error_tool_message(request, exc)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Wrap asynchronous tool calls with path mapping and error handling.

        Args:
            request: The request value.
            handler: The handler value.

        Returns:
            The awrap tool call result.
        """
        try:
            self._map_workspace_path_args(request)
            return await handler(request)
        except NON_RECOVERABLE_TOOL_EXCEPTIONS:
            raise
        except Exception as exc:
            return self._error_tool_message(request, exc)


class SummarizationStatusMiddleware(AgentMiddleware[Any, Any, Any]):
    """Emit stream events when conversation history summarization is about to run."""

    def __init__(self, inner: AgentMiddleware[Any, Any, Any], *, source: str = "main-agent") -> None:
        """Initialize the summarization status middleware instance.

        Args:
            inner: The inner value.
            source: The source value.
        """
        super().__init__()
        self.inner = inner
        self.source = source

    @property
    def name(self) -> str:
        """Return the wrapped summarization middleware name.

        Returns:
            The middleware name exposed to LangGraph.
        """
        return str(getattr(self.inner, "name", "SummarizationMiddleware"))

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Run middleware logic before a model invocation.

        Args:
            state: Runtime state to inspect or update.
            runtime: Agent runtime used by the operation.

        Returns:
            The before model result.
        """
        will_summarize = self._will_summarize(state)
        if will_summarize:
            self._emit(runtime, "started", "Conversation summarization triggered.")
        try:
            result = self.inner.before_model(state, runtime)
        except Exception:
            if will_summarize:
                self._emit(runtime, "failed", "Conversation summarization failed.")
            raise
        if result is not None:
            if not will_summarize:
                self._emit(runtime, "started", "Conversation summarization triggered.")
            self._emit(runtime, "completed", "Conversation summarization completed.")
        return result

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Run asynchronous middleware logic before a model invocation.

        Args:
            state: Runtime state to inspect or update.
            runtime: Agent runtime used by the operation.

        Returns:
            The abefore model result.
        """
        will_summarize = self._will_summarize(state)
        if will_summarize:
            self._emit(runtime, "started", "Conversation summarization triggered.")
        try:
            result = await self.inner.abefore_model(state, runtime)
        except Exception:
            if will_summarize:
                self._emit(runtime, "failed", "Conversation summarization failed.")
            raise
        if result is not None:
            if not will_summarize:
                self._emit(runtime, "started", "Conversation summarization triggered.")
            self._emit(runtime, "completed", "Conversation summarization completed.")
        return result

    def _will_summarize(self, state: Any) -> bool:
        """Return whether the next model call will trigger summarization.

        Args:
            state: Runtime state to inspect or update.

        Returns:
            Whether the next model call will trigger summarization.
        """
        try:
            messages = state["messages"]
            ensure_ids = getattr(self.inner, "_ensure_message_ids", None)
            if callable(ensure_ids):
                ensure_ids(messages)
            # Duck-typed on purpose: `self.inner` may be langchain's or
            # deepagents' SummarizationMiddleware (constructed dynamically by
            # name in `create_summarization_middleware`/the deepagents
            # factory below), which are unrelated classes that happen to
            # share this shape. getattr keeps mypy happy without pinning to
            # either concrete type; a missing attribute on some other inner
            # middleware falls through to the `except` below, same as before.
            token_counter = getattr(self.inner, "token_counter", None)
            should_summarize = getattr(self.inner, "_should_summarize", None)
            determine_cutoff = getattr(self.inner, "_determine_cutoff_index", None)
            if token_counter is None or should_summarize is None or determine_cutoff is None:
                return False
            total_tokens = token_counter(messages)
            return bool(
                should_summarize(messages, total_tokens)
                and determine_cutoff(messages) > 0
            )
        except Exception:
            return False

    def _emit(self, runtime: Any, status: str, message: str) -> None:
        """Emit a custom stream event through the configured writer.

        Args:
            runtime: Agent runtime used by the operation.
            status: The status value.
            message: Chainlit message or LangChain message to process.
        """
        stream_writer = getattr(runtime, "stream_writer", None)
        if not callable(stream_writer):
            return
        try:
            stream_writer(
                {
                    "kind": SUMMARIZATION_STATUS_EVENT_KIND,
                    "status": status,
                    "source": self.source,
                    "message": message,
                }
            )
        except Exception as exc:
            logger.debug("Failed to emit summarization status event: %s", exc)


def _build_summarization_middleware(
    *,
    config: RuntimeConfig,
    reasoning_level: ReasoningLevel,
    model_name: str | None = None,
    source: str = "main-agent",
) -> AgentMiddleware[Any, Any, Any] | None:
    """Build summarization middleware when the extension is enabled.

    Args:
        config: Configuration object used by the operation.
        reasoning_level: The reasoning level value.
        model_name: The model name value.
        source: The source value.

    Returns:
        The constructed summarization middleware when the extension is enabled.
    """
    try:
        from langchain.agents.middleware import SummarizationMiddleware
    except Exception:
        logger.warning(
            "Summarization middleware is enabled but unavailable in the installed LangChain version."
        )
        return None

    kwargs: dict[str, Any] = {}
    signature = inspect.signature(SummarizationMiddleware)
    if "model" in signature.parameters:
        kwargs["model"] = runtime_models.build_model(
            config,
            reasoning_level,
            model_name=model_name,
        )
    if config.extensions.summarization_trigger_tokens is not None:
        if "max_tokens_before_summary" in signature.parameters:
            kwargs["max_tokens_before_summary"] = (
                config.extensions.summarization_trigger_tokens
            )
        elif "trigger" in signature.parameters:
            kwargs["trigger"] = ("tokens", config.extensions.summarization_trigger_tokens)
    if config.extensions.summarization_keep_tokens is not None and "keep" in signature.parameters:
        kwargs["keep"] = ("tokens", config.extensions.summarization_keep_tokens)

    try:
        middleware = SummarizationMiddleware(**kwargs)
    except Exception as exc:
        logger.warning(
            "Failed to initialize summarization middleware; continuing without it: %s",
            exc,
        )
        return None
    return SummarizationStatusMiddleware(middleware, source=source)


def _build_deepagents_summarization_factory(
    config: RuntimeConfig,
) -> Callable[[Any, Any], AgentMiddleware[Any, Any, Any]] | None:
    """Build a DeepAgents summarization factory honoring configured thresholds.

    Args:
        config: Configuration object used by the operation.

    Returns:
        A replacement DeepAgents summarization factory, or None when the
        built-in defaults should be used unchanged.
    """
    trigger_tokens = config.extensions.summarization_trigger_tokens
    keep_tokens = config.extensions.summarization_keep_tokens
    if trigger_tokens is None and keep_tokens is None:
        return None

    def factory(model: Any, backend: Any) -> AgentMiddleware[Any, Any, Any]:
        """Create DeepAgents summarization middleware with configured thresholds."""
        from langchain.agents.middleware.summarization import ContextSize

        from deepagents.middleware.summarization import (
            SummarizationMiddleware,
            compute_summarization_defaults,
        )

        defaults = compute_summarization_defaults(model)
        trigger: ContextSize = (
            ("tokens", trigger_tokens)
            if trigger_tokens is not None
            else defaults["trigger"]
        )
        keep: ContextSize = (
            ("tokens", keep_tokens) if keep_tokens is not None else defaults["keep"]
        )
        return SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=trigger,
            keep=keep,
            trim_tokens_to_summarize=None,
            truncate_args_settings=defaults["truncate_args_settings"],
        )

    return factory


def create_deep_agent_with_configured_summarization(
    config: RuntimeConfig,
    **kwargs: Any,
) -> Any:
    """Create a DeepAgents graph while applying configured summarization thresholds.

    Args:
        config: Configuration object used by the operation.
        kwargs: Keyword arguments passed to create_deep_agent.

    Returns:
        The created DeepAgents graph.
    """
    import deepagents.graph as deepagents_graph

    summarization_factory = _build_deepagents_summarization_factory(config)

    with _DEEPAGENTS_SUMMARIZATION_FACTORY_LOCK:
        original_factory = deepagents_graph.create_summarization_middleware
        original_profile = deepagents_graph._harness_profile_for_model

        def profile_with_token_guard(model: Any, spec: str | None) -> Any:
            # DeepAgents builds its implicit general-purpose delegate from the
            # harness profile, rather than inheriting caller middleware.
            profile = original_profile(model, spec)

            def middleware_for_stacks() -> list[AgentMiddleware[Any, Any, Any]]:
                existing = profile.materialize_extra_middleware()
                if not any(isinstance(item, TokenLimitedToolCallMiddleware) for item in existing):
                    existing.append(TokenLimitedToolCallMiddleware())
                if not any(
                    isinstance(item, RepeatedToolResultGuardMiddleware) for item in existing
                ):
                    existing.append(RepeatedToolResultGuardMiddleware())
                return existing

            return dataclasses.replace(profile, extra_middleware=middleware_for_stacks)

        if summarization_factory is not None:
            # This replacement factory intentionally narrows the signature to
            # the two positional arguments deepagents actually passes at this
            # call site; the module-level function's broader keyword-only
            # defaults are not exercised here.
            deepagents_graph.create_summarization_middleware = (
                summarization_factory  # type: ignore[assignment]
            )
        deepagents_graph._harness_profile_for_model = profile_with_token_guard
        try:
            return create_deep_agent(**kwargs)
        finally:
            deepagents_graph.create_summarization_middleware = original_factory
            deepagents_graph._harness_profile_for_model = original_profile


def build_agent_middleware(
    *,
    backend: BackendProtocol,
    config: RuntimeConfig | None = None,
    reasoning_level: ReasoningLevel | None = None,
    model_name: str | None = None,
    source: str = "main-agent",
    project_root: Path | None = None,
) -> list[AgentMiddleware[Any, Any, Any]]:
    """Build agent middleware.

    Args:
        backend: Concrete DeepAgents backend used by filesystem middleware.
        config: Configuration object used by the operation.
        reasoning_level: The reasoning level value.
        model_name: The model name value.
        source: The source value.
        project_root: Project root used to resolve local paths.

    Returns:
        The constructed agent middleware.
    """
    # DeepAgents owns summarization middleware in its base main-agent and
    # sync-subagent stacks. Passing another SummarizationMiddleware here creates
    # duplicate middleware names that LangChain rejects during agent creation.
    middleware: list[AgentMiddleware[Any, Any, Any]] = [TodoListMiddleware()]
    filesystem_tools: list[Any] = list(DEFAULT_DEEPAGENT_FILESYSTEM_TOOLS)
    if config is not None:
        if config.extensions.delete_tool_enabled:
            filesystem_tools.append("delete")
        if config.extensions.execute_tool_enabled:
            filesystem_tools.append("execute")
    filesystem_middleware = FilesystemMiddleware(
        backend=backend,
        tools=filesystem_tools,
    )
    middleware.append(filesystem_middleware)
    middleware.append(TokenLimitedToolCallMiddleware())
    middleware.append(RepeatedToolResultGuardMiddleware())
    middleware.append(
        ToolExecutionResilienceMiddleware(
            project_root=project_root,
            virtual_filesystem_tools=runtime_backends.backend_routes_workspace(backend),
        )
    )
    return middleware
