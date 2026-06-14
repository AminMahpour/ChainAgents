"""Compatibility wrapper for the moved ChainAgents runtime."""

from __future__ import annotations

import sys as _sys

from chainagents.runtime import core as _module

_sys.modules[__name__] = _module

