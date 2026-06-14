"""Compatibility wrapper for the moved RAG runtime."""

from __future__ import annotations

import sys as _sys

from chainagents.rag import runtime as _module

_sys.modules[__name__] = _module

