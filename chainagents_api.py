"""Compatibility wrapper for the moved ChainAgents FastAPI app."""

from __future__ import annotations

import sys as _sys

from chainagents.interfaces.api import app as _module

_sys.modules[__name__] = _module

