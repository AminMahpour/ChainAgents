"""Smoke tests for package and legacy module imports."""

from __future__ import annotations

import tomllib
from pathlib import Path

from chainagents.runtime import core as runtime_core


def dependency_name(requirement: str) -> str:
    """Return the normalized package name from a dependency requirement."""
    for delimiter in ("[", ">", "=", "<"):
        requirement = requirement.split(delimiter, 1)[0]
    return requirement.strip()


def test_package_imports_expose_preferred_runtime_and_interface_paths() -> None:
    """Verify that new package paths expose the public reorganization surface."""
    from chainagents.events.stream import AgentStreamEventAdapter
    from chainagents.interfaces.api.app import create_app
    from chainagents.interfaces.chainlit.bridge import ChainlitEventBridge
    from chainagents.interfaces.cli.app import build_parser
    from chainagents.rag.runtime import RagStatus
    from chainagents.runtime import RuntimeConfig

    assert RuntimeConfig is not None
    assert ChainlitEventBridge is not None
    assert AgentStreamEventAdapter is not None
    assert create_app is not None
    assert build_parser is not None
    assert RagStatus is not None


def test_legacy_imports_alias_moved_modules() -> None:
    """Verify that old top-level import paths still resolve to moved modules."""
    import agent_commands
    import agent_stream_events
    import chainagents_api
    import chainagents_cli
    import chainagents_tui
    import chainlit_bridge
    import chainlit_persistence
    import deepagent_runtime
    import main
    import rag_runtime
    import response_exports

    assert deepagent_runtime.__name__ == "chainagents.runtime.core"
    assert chainlit_bridge.__name__ == "chainagents.interfaces.chainlit.bridge"
    assert main.__name__ == "chainagents.interfaces.chainlit.app"
    assert chainagents_cli.__name__ == "chainagents.interfaces.cli.app"
    assert chainagents_api.__name__ == "chainagents.interfaces.api.app"
    assert chainagents_tui.__name__ == "chainagents.interfaces.tui.app"
    assert rag_runtime.__name__ == "chainagents.rag.runtime"
    assert response_exports.__name__ == "chainagents.exports.response"
    assert agent_stream_events.__name__ == "chainagents.events.stream"
    assert agent_commands.__name__ == "chainagents.commands.native"
    assert chainlit_persistence.__name__ == "chainagents.interfaces.chainlit.persistence"


def test_default_dependencies_do_not_include_chromadb() -> None:
    """Verify default dependencies avoid the vulnerable ChromaDB stack."""
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependency_names = {
        dependency_name(requirement)
        for requirement in project["project"]["dependencies"]
    }

    assert "chromadb" not in dependency_names
    assert "langchain-chroma" not in dependency_names


def test_default_project_root_keeps_source_checkout(tmp_path: Path) -> None:
    """Keep source checkouts as the workspace even when launched elsewhere."""
    source_root = tmp_path / "source"
    (source_root / "chainagents/runtime").mkdir(parents=True)
    (source_root / "pyproject.toml").write_text(
        '[project]\nname = "ChainAgents"\n',
        encoding="utf-8",
    )
    working_directory = tmp_path / "user-workspace"
    working_directory.mkdir()

    resolved = runtime_core._resolve_default_project_root(
        module_file=source_root / "chainagents/runtime/core.py",
        working_directory=working_directory,
    )

    assert resolved == source_root.resolve()


def test_default_project_root_uses_user_cwd_for_installed_package(
    tmp_path: Path,
) -> None:
    """Avoid treating an installed package's site-packages as the workspace."""
    site_packages = tmp_path / "venv/lib/python3.12/site-packages"
    working_directory = tmp_path / "user-workspace"
    working_directory.mkdir()

    resolved = runtime_core._resolve_default_project_root(
        module_file=site_packages / "chainagents/runtime/core.py",
        working_directory=working_directory,
    )

    assert resolved == working_directory.resolve()
