"""Check that exported graphs and the live runtime share one agent assembly."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import chainagents.runtime.config as runtime_config
import chainagents.runtime.constants as runtime_constants
import chainagents.runtime.graph as runtime_graph
import chainagents.runtime.middleware as runtime_middleware
import chainagents.runtime.models as runtime_models
from chainagents.runtime import core
from chainagents.runtime.types import (
    AsyncSubagentConfig,
    BackgroundSubagentConfig,
    ExtensionsConfig,
    SubagentConfig,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from test_deepagent_runtime_rag import make_runtime_config


def _assembly_config(tmp_path: Path, **overrides: Any) -> runtime_config.RuntimeConfig:
    """Return a config that exercises every branch both assembly paths share."""
    extensions = ExtensionsConfig(
        config_path=None,
        background_subagents=BackgroundSubagentConfig(enabled=True),
        custom_instruction="Answer briefly.",
        subagents=(
            SubagentConfig(
                name="researcher",
                description="Researches.",
                system_prompt="Research.",
                background=True,
            ),
            SubagentConfig(
                name="manager",
                description="Coordinates reviewers.",
                system_prompt="Manage.",
                subagents=(
                    SubagentConfig(
                        name="reviewer",
                        description="Reviews.",
                        system_prompt="Review.",
                        background=True,
                    ),
                ),
            ),
            SubagentConfig(
                name="writer",
                description="Writes.",
                system_prompt="Write.",
            ),
        ),
        async_subagents=(
            AsyncSubagentConfig(
                name="remote",
                description="Runs remotely.",
                graph_id="remote",
                url="http://127.0.0.1:2024",
            ),
        ),
    )
    config = replace(
        make_runtime_config(tmp_path, extensions=extensions),
        rag=None,
        rag_requested=False,
    )
    return replace(config, **overrides)


def _summarize(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Reduce DeepAgents kwargs to comparable, identity-free values."""
    return {
        "keys": list(kwargs),
        "tools": [tool.name for tool in kwargs.get("tools") or ()],
        "system_prompt": kwargs["system_prompt"],
        "middleware": [type(item).__name__ for item in kwargs["middleware"]],
        "skills": kwargs["skills"],
        "subagents": [
            (spec["name"], sorted(spec)) for spec in kwargs["subagents"] or ()
        ],
    }


def _build_both(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: runtime_config.RuntimeConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return every create_deep_agent call from the static and live paths."""
    calls: list[dict[str, Any]] = []

    def fake_create_deep_agent(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(runtime_middleware, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(runtime_models, "build_model", lambda *a, **kw: object())
    monkeypatch.setattr(runtime_constants, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        runtime_config.RuntimeConfig, "from_env", staticmethod(lambda: config)
    )

    runtime_graph.create_configured_graph(
        include_async_subagents=True,
        apply_custom_instruction=True,
    )
    asyncio.run(runtime_graph.close_static_background_tasks())
    static_calls = list(calls)
    calls.clear()

    runtime = core.AgentRuntime(config, project_root=tmp_path)
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    async def exercise() -> None:
        try:
            await runtime.get_agent(config.default_reasoning, thread_id="thread-1")
        finally:
            await runtime.close()

    asyncio.run(exercise())
    return static_calls, list(calls)


def test_static_and_live_assembly_build_the_same_agent_without_mcp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without MCP or persistence, both paths must hand DeepAgents the same kwargs."""
    config = _assembly_config(tmp_path, agent_state="stateless")

    static_calls, live_calls = _build_both(tmp_path, monkeypatch, config)

    # Compiled subagents (researcher, reviewer, manager) then the main agent.
    assert len(static_calls) == len(live_calls) == 4
    assert [_summarize(kwargs) for kwargs in static_calls] == [
        _summarize(kwargs) for kwargs in live_calls
    ]
    main = _summarize(live_calls[-1])
    assert main["tools"][0] == "render_chainlit_ui"
    assert "spawn_background_task" in main["tools"]
    # The manager's own task tools come from the nested background reviewer.
    assert "spawn_background_task" in _summarize(live_calls[-2])["tools"]
    assert [name for name, _ in main["subagents"]] == [
        "researcher",
        "manager",
        "writer",
        "remote",
    ]
    assert "Answer briefly." in main["system_prompt"]


def test_live_assembly_adds_only_persistence_when_stateful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stateful live agents differ from exported graphs only by store and checkpointer."""
    config = _assembly_config(tmp_path, agent_state="stateful")

    static_calls, live_calls = _build_both(tmp_path, monkeypatch, config)

    assert len(static_calls) == len(live_calls)
    for static_kwargs, live_kwargs in zip(static_calls, live_calls, strict=True):
        assert set(live_kwargs) - set(static_kwargs) == {"store", "checkpointer"}
        assert set(static_kwargs) <= set(live_kwargs)
        static_summary = _summarize(static_kwargs)
        live_summary = _summarize(live_kwargs)
        live_summary["keys"] = [
            key for key in live_summary["keys"] if key not in {"store", "checkpointer"}
        ]
        assert static_summary == live_summary


def test_live_assembly_ignores_background_flag_when_background_is_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A background subagent under a disabled manager builds as a plain subagent."""
    config = _assembly_config(
        tmp_path,
        agent_state="stateless",
        extensions=ExtensionsConfig(
            config_path=None,
            subagents=(
                SubagentConfig(
                    name="researcher",
                    description="Researches.",
                    system_prompt="Research.",
                    background=True,
                ),
            ),
        ),
    )

    static_calls, live_calls = _build_both(tmp_path, monkeypatch, config)

    assert [_summarize(kwargs) for kwargs in static_calls] == [
        _summarize(kwargs) for kwargs in live_calls
    ]
    assert "runnable" not in live_calls[-1]["subagents"][0]
