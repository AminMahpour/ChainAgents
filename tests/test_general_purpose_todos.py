"""DeepAgents' general-purpose delegate keeps ChainAgents' todo list tool."""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

import langchain.agents
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
import pytest

import chainagents.runtime.core as deepagent_runtime

BASE_CONFIG = """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "qwen3:32b"

[rag]
enabled = false
"""

LIVE_AGENTS_CONFIG = """
[agent.messaging]
enabled = true

[agent.live_agents]
enabled = true

[[agent.live_agents.templates]]
name = "researcher"
description = "Researches."
system_prompt = "Research."
"""


@pytest.mark.parametrize("live_agents", [False, True], ids=["implicit", "explicit"])
def test_general_purpose_delegate_offers_write_todos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_agents: bool
) -> None:
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        BASE_CONFIG + (LIVE_AGENTS_CONFIG if live_agents else ""), encoding="utf-8"
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))

    stacks: list[tuple[str | None, list[str]]] = []
    original = langchain.agents.create_agent

    def recording_create_agent(*args: Any, **kwargs: Any) -> Any:
        names = [item.name for item in kwargs.get("middleware") or ()]
        stacks.append((kwargs.get("name"), names))
        return original(*args, **kwargs)

    for target in (
        "langchain.agents.create_agent",
        "deepagents.graph.create_agent",
        "deepagents.middleware.subagents.create_agent",
    ):
        monkeypatch.setattr(target, recording_create_agent)

    runtime = deepagent_runtime.AgentRuntime(deepagent_runtime.RuntimeConfig.from_env())
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()
    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    general_purpose = [names for name, names in stacks if name == "general-purpose"]
    assert general_purpose, stacks
    for names in general_purpose:
        assert names.count("TodoListMiddleware") == 1, names
    for name, names in stacks:
        duplicates = [item for item, count in Counter(names).items() if count > 1]
        assert not duplicates, (name, duplicates)
