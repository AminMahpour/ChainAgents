"""Compatibility wrapper for moved Chainlit async task notifications."""

from __future__ import annotations

import sys as _sys

from chainagents.interfaces.chainlit import async_tasks as _module

_sys.modules[__name__] = _module

