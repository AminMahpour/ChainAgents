"""Install narrow warning filters for known third-party startup warnings."""

from __future__ import annotations

import warnings

try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
except ImportError:  # pragma: no cover - defensive fallback for dependency changes.
    LangChainPendingDeprecationWarning = Warning


ALLOWED_OBJECTS_WARNING_MESSAGE = (
    r"The default value of `allowed_objects` will change in a future version\..*"
)
LANGGRAPH_CHECKPOINT_SERDE_MODULES = (
    r"langgraph\.checkpoint\.serde\.(encrypted|jsonplus)"
)


def install_langchain_warning_filters() -> None:
    """Hide LangChain's import-time Reviver warning from LangGraph internals."""
    warnings.filterwarnings(
        "ignore",
        message=ALLOWED_OBJECTS_WARNING_MESSAGE,
        category=LangChainPendingDeprecationWarning,
        module=LANGGRAPH_CHECKPOINT_SERDE_MODULES,
    )

