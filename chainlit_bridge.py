"""Compatibility wrapper for the moved Chainlit event bridge."""

from __future__ import annotations

import sys as _sys

from chainagents.interfaces.chainlit import bridge as _module

_sys.modules[__name__] = _module

