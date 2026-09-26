"""Compatibility wrapper for the moved RAG runtime.

Deprecated: this root-level module will be removed in a future release.
Import from ``chainagents.rag.runtime`` instead.
"""

from __future__ import annotations

from chainagents.rag import runtime as _module
from chainagents.util.compat import alias_module

alias_module(__name__, _module)
