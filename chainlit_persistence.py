"""Compatibility wrapper for the moved Chainlit persistence helpers."""

from __future__ import annotations

import sys as _sys

from chainagents.interfaces.chainlit import persistence as _module

_sys.modules[__name__] = _module

