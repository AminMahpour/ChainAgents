"""Compatibility wrapper for the moved ChainAgents CLI."""

from __future__ import annotations

import sys as _sys

from chainagents.interfaces.cli import app as _module

_sys.modules[__name__] = _module

