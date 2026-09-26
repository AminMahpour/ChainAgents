"""Compatibility wrapper for moved LangChain warning filters.

Deprecated: this root-level module will be removed in a future release.
Import from ``chainagents.util.langchain_warnings`` instead.
"""

from __future__ import annotations

from chainagents.util import langchain_warnings as _module
from chainagents.util.compat import alias_module

alias_module(__name__, _module)
