"""Compatibility wrapper for the moved LangGraph app exports."""

from __future__ import annotations

import sys as _sys

from chainagents.langgraph import app as _module

_sys.modules[__name__] = _module

