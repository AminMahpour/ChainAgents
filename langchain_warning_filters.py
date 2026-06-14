"""Compatibility wrapper for moved LangChain warning filters."""

from __future__ import annotations

import sys as _sys

from chainagents.util import langchain_warnings as _module

_sys.modules[__name__] = _module

