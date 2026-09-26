"""Compatibility wrapper for the moved ChainAgents CLI.

Deprecated: this root-level module will be removed in a future release.
Import from ``chainagents.interfaces.cli.app`` instead.
"""

from __future__ import annotations

from chainagents.interfaces.cli import app as _module
from chainagents.util.compat import alias_module

alias_module(__name__, _module)
