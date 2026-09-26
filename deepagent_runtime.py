"""Compatibility wrapper for the moved ChainAgents runtime.

Deprecated: this root-level module will be removed in a future release.
Import from ``chainagents.runtime.core`` instead.
"""

from __future__ import annotations

from chainagents.runtime import core as _module
from chainagents.util.compat import alias_module

alias_module(__name__, _module)
