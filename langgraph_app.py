"""Compatibility wrapper for the moved LangGraph Agent Server exports."""

from __future__ import annotations

import sys as _sys

from chainagents.langgraph import app as _module

supervisor = _module.supervisor
async_researcher = _module.async_researcher

__all__ = ["supervisor", "async_researcher"]

_sys.modules[__name__] = _module
