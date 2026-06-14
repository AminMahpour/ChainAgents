"""Compatibility wrapper for the moved ChainAgents stream event helpers."""

from __future__ import annotations

import sys as _sys

from chainagents.events import stream as _module

_sys.modules[__name__] = _module

