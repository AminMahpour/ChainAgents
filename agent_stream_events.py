"""Compatibility wrapper for the moved ChainAgents stream event helpers.

Deprecated: this root-level module will be removed in a future release.
Import from ``chainagents.events.stream`` instead.
"""

from __future__ import annotations

from chainagents.events import stream as _module
from chainagents.util.compat import alias_module

alias_module(__name__, _module)
