"""Optional Langfuse and LangSmith tracing for LangGraph runs."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig

from chainagents.runtime.config import RuntimeConfig
from chainagents.runtime.types import LangfuseConfig, LangSmithConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LangSmithParentReference:
    """The minimal parent identity safe to carry into an isolated task."""

    dotted_order: str
    run_id: str
    trace_id: str
    project: str


def _same_langsmith_destination(first: Any, second: Any) -> bool:
    """Recognize the same writable endpoint without assuming client identity."""
    if first is second:
        return True
    try:
        first_url, second_url = first.api_url, second.api_url
        first_key, second_key = first.api_key, second.api_key
        return bool(first_url and first_key) and (
            first_url == second_url
            and first_key == second_key
            and first.workspace_id == second.workspace_id
        )
    except (AttributeError, TypeError):
        return False


def _import_langsmith_client() -> type[Any]:
    """Import the SDK only when LangSmith tracing is enabled."""
    try:
        from langsmith import Client
    except ImportError as exc:
        raise RuntimeError(
            "LangSmith tracing is enabled but 'langsmith' is not installed. "
            "Run `uv sync` to install project dependencies."
        ) from exc
    return Client


class LangSmithTracing:
    """Own a LangSmith client shared by one agent runtime or exported graph."""

    def __init__(self, config: LangSmithConfig, *, client: Any | None = None) -> None:
        self.config = config
        self.project = (
            config.project or os.getenv("LANGSMITH_PROJECT", "").strip() or "chainagents"
        )
        self.client = client if client is not None else _import_langsmith_client()()
        self._owns_client = client is None

    def new_callback(self) -> Any:
        """Create one LangChain callback handler for a foreground invocation."""
        from langchain_core.tracers.langchain import LangChainTracer

        return LangChainTracer(client=self.client, project_name=self.project)

    def with_callback(self, config: RunnableConfig | None) -> RunnableConfig:
        """Add the runtime callback unless this invocation already has its tracer."""
        from langchain_core.runnables.config import ensure_config, merge_configs
        from langchain_core.tracers.langchain import LangChainTracer

        effective_config = ensure_config(config)
        callbacks = effective_config.get("callbacks")
        handlers = getattr(callbacks, "handlers", callbacks)
        if isinstance(handlers, (list, tuple)) and any(
            isinstance(handler, LangChainTracer)
            and _same_langsmith_destination(handler.client, self.client)
            and handler.project_name == self.project
            for handler in handlers
        ):
            return effective_config
        return merge_configs(effective_config, {"callbacks": [self.new_callback()]})

    def capture_parent(self, run_config: RunnableConfig) -> LangSmithParentReference | None:
        """Capture only trace identity, without carrying runnable callbacks or context."""
        from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
        from langchain_core.tracers.langchain import LangChainTracer
        from langsmith import RunTree, get_current_run_tree

        try:
            parent = None
            callbacks = run_config.get("callbacks")
            if isinstance(callbacks, (CallbackManager, AsyncCallbackManager)):
                owned = next(
                    (
                        handler
                        for handler in callbacks.handlers
                        if isinstance(handler, LangChainTracer)
                        and _same_langsmith_destination(handler.client, self.client)
                        and handler.project_name == self.project
                    ),
                    None,
                )
                if owned is not None:
                    selected_callbacks = callbacks.copy()
                    selected_callbacks.handlers = [owned]
                    parent = RunTree.from_runnable_config(
                        {**run_config, "callbacks": selected_callbacks}
                    )
            if parent is None:
                current = get_current_run_tree()
                if (
                    current is not None
                    and _same_langsmith_destination(current.ls_client, self.client)
                    and current.session_name == self.project
                ):
                    parent = current
        except Exception:
            logger.warning("Could not capture the LangSmith parent run", exc_info=True)
            return None
        if parent is None:
            return None
        return LangSmithParentReference(
            dotted_order=parent.dotted_order,
            run_id=str(parent.id),
            trace_id=str(parent.trace_id),
            project=parent.session_name or self.project,
        )

    @contextmanager
    def background_scope(
        self,
        parent: LangSmithParentReference | None,
        *,
        mode: Literal["linked", "separate"],
    ) -> Iterator[None]:
        """Trace one isolated background invocation with an explicit client."""
        from langsmith import RunTree, tracing_context

        stack = ExitStack()
        try:
            use_linked_parent = mode == "linked" and parent is not None
            linked_parent: RunTree | Literal[False] = (
                RunTree.from_dotted_order(
                    parent.dotted_order,
                    client=self.client,
                    project_name=parent.project,
                )
                if use_linked_parent and parent is not None
                else False
            )
            stack.enter_context(
                tracing_context(
                    enabled=True,
                    client=self.client,
                    project_name=(
                        parent.project
                        if use_linked_parent and parent is not None
                        else self.project
                    ),
                    parent=linked_parent,
                )
            )
        except Exception:
            logger.warning("Could not start LangSmith background tracing", exc_info=True)
        with stack:
            yield

    def flush(self) -> None:
        """Submit buffered traces before a short-lived process exits."""
        self.client.flush()

    def close(self) -> None:
        """Close the client if this runtime created it."""
        if self._owns_client:
            self.client.close()


def build_langsmith_tracing(config: LangSmithConfig) -> LangSmithTracing | None:
    """Build tracing only for an explicitly enabled integration."""
    return LangSmithTracing(config) if config.enabled else None


def _import_langfuse_callback_handler() -> type[Any]:
    """Import Langfuse's LangChain callback handler on demand.

    Returns:
        The Langfuse LangChain callback handler type.

    Raises:
        RuntimeError: If Langfuse support is enabled but unavailable.
    """
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError as exc:
        raise RuntimeError(
            "Langfuse tracing is enabled but the 'langfuse' package is not installed. "
            "Run `uv sync` to install project dependencies."
        ) from exc
    return CallbackHandler


def build_langfuse_callback_handler(config: RuntimeConfig) -> Any | None:
    """Build a Langfuse callback handler when tracing is enabled.

    Args:
        config: Configuration object used by the operation.

    Returns:
        A Langfuse callback handler, or None when disabled.
    """
    langfuse = getattr(config, "langfuse", LangfuseConfig())
    if not langfuse.enabled:
        return None
    handler_cls = _import_langfuse_callback_handler()
    return handler_cls()


def shutdown_langfuse_client(config: RuntimeConfig) -> bool:
    """Shut down Langfuse's buffered client when tracing is enabled.

    Langfuse batches events in background workers, so short-lived CLI processes
    need an explicit shutdown before process exit to avoid dropping traces.

    Args:
        config: Configuration object used by the operation.

    Returns:
        True when Langfuse tracing was enabled and shutdown was requested.
    """
    langfuse = getattr(config, "langfuse", LangfuseConfig())
    if not langfuse.enabled:
        return False

    from langfuse import get_client

    get_client().shutdown()
    return True


def build_langgraph_run_config(
    config: RuntimeConfig,
    *,
    thread_id: str,
    langsmith_tracing: LangSmithTracing | None = None,
) -> dict[str, Any]:
    """Build LangGraph run config shared by all ChainAgents entrypoints.

    Args:
        config: Configuration object used by the operation.
        thread_id: Conversation thread identifier.

    Returns:
        A LangGraph configuration dictionary for the run.
    """
    run_config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": config.recursion_limit,
    }
    callbacks: list[Any] = []
    langfuse_handler = build_langfuse_callback_handler(config)
    if langfuse_handler is not None:
        callbacks.append(langfuse_handler)
        run_config["metadata"] = {"langfuse_session_id": thread_id}
        run_config["tags"] = ["chainagents"]
    if langsmith_tracing is not None:
        callbacks.append(langsmith_tracing.new_callback())
        run_config.setdefault("metadata", {})["session_id"] = thread_id
        run_config["tags"] = ["chainagents"]
    if callbacks:
        run_config["callbacks"] = callbacks
    return run_config
