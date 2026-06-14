"""Compatibility entrypoint for Chainlit.

Keep this root module so `chainlit run main.py -w` continues to import and
register the Chainlit callbacks from the packaged app.
"""

from __future__ import annotations

import sys as _sys

from chainagents.interfaces.chainlit import app as _module

_sys.modules[__name__] = _module

