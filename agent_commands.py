"""Compatibility wrapper for the moved ChainAgents command helpers."""

from __future__ import annotations

import sys as _sys

from chainagents.commands import native as _module

_sys.modules[__name__] = _module

