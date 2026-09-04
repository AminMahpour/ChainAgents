"""Optional Langfuse tracing and LangGraph run configuration."""

from __future__ import annotations

from typing import Any

from chainagents.runtime.config import RuntimeConfig
from chainagents.runtime.types import LangfuseConfig


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
    langfuse_handler = build_langfuse_callback_handler(config)
    if langfuse_handler is not None:
        run_config["callbacks"] = [langfuse_handler]
        run_config["metadata"] = {"langfuse_session_id": thread_id}
        run_config["tags"] = ["chainagents"]
    return run_config
