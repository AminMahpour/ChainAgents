"""Compatibility wrapper for the moved Chainlit event bridge.

Deprecated: this root-level module will be removed in a future release.
Import from ``chainagents.interfaces.chainlit.bridge`` instead.
"""

from __future__ import annotations

from chainagents.interfaces.chainlit import bridge as _module
from chainagents.util.compat import alias_module

alias_module(__name__, _module)
