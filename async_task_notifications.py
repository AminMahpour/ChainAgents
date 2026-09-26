"""Compatibility wrapper for moved Chainlit async task notifications.

Deprecated: this root-level module will be removed in a future release.
Import from ``chainagents.interfaces.chainlit.async_tasks`` instead.
"""

from __future__ import annotations

from chainagents.interfaces.chainlit import async_tasks as _module
from chainagents.util.compat import alias_module

alias_module(__name__, _module)
