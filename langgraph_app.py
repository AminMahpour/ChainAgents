"""Compatibility wrapper for the moved LangGraph app exports."""

from __future__ import annotations

import sys as _sys

from chainagents.langgraph import app as _module

supervisor = _module.supervisor
async_researcher = _module.async_researcher

__all__ = ["async_researcher", "supervisor"]

_sys.modules[__name__] = _module
