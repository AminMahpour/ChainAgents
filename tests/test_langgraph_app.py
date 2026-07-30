"""Test Agent Server graph exports."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

import chainagents.runtime as runtime_package
from chainagents.runtime import core as runtime_core
from chainagents.runtime.token_usage import TokenUsageFileCallbackHandler


class _State(TypedDict):
    value: int


def _compiled_test_graph(observed_configs: list[RunnableConfig]):
    """Build a real minimal graph for exercising the exported factory."""

    def capture_config(state: _State, config: RunnableConfig) -> _State:
        observed_configs.append(config)
        return state

    builder = StateGraph(_State)
    builder.add_node("capture_config", capture_config)
    builder.add_edge(START, "capture_config")
    builder.add_edge("capture_config", END)
    return builder.compile()


def _load_langgraph_app() -> ModuleType:
    """Load the deployment module without sharing its process-global module cache."""
    module_path = Path(runtime_core.__file__).parents[1] / "langgraph" / "app.py"
    spec = importlib.util.spec_from_file_location(
        "_chainagents_test_langgraph_app",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agent_server_exports_add_request_scoped_token_logging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exporting raw shared graphs must bypass request-scoped usage logging."""
    observed_configs: list[RunnableConfig] = []
    monkeypatch.setattr(runtime_core, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runtime_package, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        runtime_package,
        "create_configured_graph",
        lambda **_kwargs: _compiled_test_graph(observed_configs),
    )

    module = _load_langgraph_app()
    run_configs = [
        {
            "configurable": {"thread_id": "supervisor-thread"},
            "recursion_limit": 42,
            "tags": ["server"],
        },
        {
            "configurable": {"thread_id": "researcher-thread"},
            "recursion_limit": 43,
            "tags": ["server"],
        },
    ]
    configured_graphs = [
        module.supervisor(run_configs[0]),
        module.async_researcher(run_configs[1]),
    ]

    callbacks = [graph.config["callbacks"][0] for graph in configured_graphs]
    assert all(
        isinstance(callback, TokenUsageFileCallbackHandler)
        for callback in callbacks
    )
    assert callbacks[0] is not callbacks[1]

    for graph, run_config in zip(configured_graphs, run_configs, strict=True):
        graph.invoke({"value": 1}, config=run_config)

    records = [
        json.loads(line)
        for line in (tmp_path / ".files" / "token-usage.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(records) == 2
    assert [
        {
            "thread_id": config["configurable"]["thread_id"],
            "recursion_limit": config["recursion_limit"],
            "tags": config["tags"],
        }
        for config in observed_configs
    ] == [
        {
            "thread_id": "supervisor-thread",
            "recursion_limit": 42,
            "tags": ["server"],
        },
        {
            "thread_id": "researcher-thread",
            "recursion_limit": 43,
            "tags": ["server"],
        },
    ]
