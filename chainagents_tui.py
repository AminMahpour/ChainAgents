"""Compatibility wrapper for the moved ChainAgents TUI."""

from __future__ import annotations

import sys as _sys

from chainagents.interfaces.tui import app as _module

_sys.modules[__name__] = _module

