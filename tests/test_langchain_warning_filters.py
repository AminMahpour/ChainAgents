"""Test third-party warning handling for runtime startup."""

from __future__ import annotations

import os
import subprocess
import sys


def test_deepagent_runtime_import_suppresses_allowed_objects_warning() -> None:
    """Verify that startup hides LangChain's import-time allowed_objects warning."""
    env = {
        **os.environ,
        "PYTHONPATH": os.getcwd(),
    }

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "default",
            "-c",
            "import deepagent_runtime",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "allowed_objects" not in result.stderr
