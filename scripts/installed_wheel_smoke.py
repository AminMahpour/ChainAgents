"""Exercise an installed ChainAgents wheel from a user working directory."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path


PUBLIC_MODULES = (
    "chainagents.runtime",
    "chainagents.interfaces.api.app",
    "chainagents.interfaces.cli.app",
    "chainagents.interfaces.chainlit.bridge",
    "chainagents.rag.runtime",
)
LEGACY_MODULES = (
    "agent_commands",
    "agent_stream_events",
    "chainagents_api",
    "chainagents_cli",
    "chainagents_tui",
    "chainlit_bridge",
    "chainlit_persistence",
    "deepagent_runtime",
    "main",
    "rag_runtime",
    "response_exports",
)


def main() -> None:
    """Validate imports, config discovery, and workspace resolution."""
    working_directory = Path.cwd().resolve()
    expected_config_path = working_directory / "deepagent.toml"
    expected_config_path.write_text(
        """[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "gpt-oss:20b"

[agent]
mcp_servers = []

[rag]
enabled = false
""",
        encoding="utf-8",
    )
    for module_name in (*PUBLIC_MODULES, *LEGACY_MODULES):
        importlib.import_module(module_name)

    from chainagents.interfaces.cli.app import resolve_configure_config_path
    from chainagents.runtime import AgentRuntime, RuntimeConfig
    from chainagents.runtime import core as runtime_core

    config = RuntimeConfig.from_env()
    runtime_requirements = importlib.metadata.requires("ChainAgents") or []

    assert runtime_core.PROJECT_ROOT == working_directory
    assert runtime_core.PROJECT_ROOT.name != "site-packages"
    assert not any(
        requirement.lower().startswith("pytest")
        for requirement in runtime_requirements
    )
    assert config.extensions.config_path == expected_config_path
    assert config.model_provider == "ollama"
    assert config.model_name == "gpt-oss:20b"
    assert resolve_configure_config_path(None) == expected_config_path

    explicit_project_root = working_directory / "explicit-workspace"
    runtime = AgentRuntime(config, project_root=explicit_project_root)
    assert Path(runtime.project_root) == explicit_project_root

    print(
        "installed wheel smoke passed:",
        {
            "cwd": os.fspath(working_directory),
            "project_root": os.fspath(runtime_core.PROJECT_ROOT),
            "config": os.fspath(config.extensions.config_path),
        },
    )


if __name__ == "__main__":
    main()
