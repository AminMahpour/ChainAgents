"""Smoke tests for package and legacy module imports."""

from __future__ import annotations


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

