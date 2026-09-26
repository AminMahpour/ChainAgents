"""Helper for deprecating root-level compatibility shim modules."""

from __future__ import annotations

import sys
import warnings
from types import ModuleType


def alias_module(name: str, target: ModuleType) -> None:
    """Warn that ``name`` is deprecated, then alias it to ``target`` in sys.modules."""
    warnings.warn(
        f"'{name}' is deprecated; import '{target.__name__}'",
        DeprecationWarning,
        stacklevel=3,
    )
    sys.modules[name] = target
