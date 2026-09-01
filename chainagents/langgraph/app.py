"""Expose configured LangGraph agents for Agent Server deployment."""

from __future__ import annotations

import os

from chainagents.runtime import (
    PROJECT_ROOT,
    SYSTEM_PROMPT,
    RuntimeConfig,
    build_runtime_backend_bundle,
    create_configured_graph,
)


os.environ.pop("__LANGGRAPH_DEFER_LOOPBACK_TRANSPORT", None)

ASYNC_RESEARCHER_PROMPT = """
You are an async research subagent.

Focus on longer background research and codebase analysis tasks. Return concise
findings with concrete file paths or sources when relevant.
""".strip()

_runtime_config = RuntimeConfig.from_env()
_backend_bundle = build_runtime_backend_bundle(
    backend_config=_runtime_config.backend,
    project_root=PROJECT_ROOT,
    include_memories=_runtime_config.agent_state == "stateful",
    memory_namespace=_runtime_config.extensions.agent_memory_namespace,
)

supervisor = create_configured_graph(
    include_async_subagents=True,
    system_prompt=SYSTEM_PROMPT,
    apply_custom_instruction=True,
    config=_runtime_config,
    backend=_backend_bundle.backend,
    backend_metadata=_backend_bundle.metadata,
)

async_researcher = create_configured_graph(
    include_async_subagents=False,
    system_prompt=ASYNC_RESEARCHER_PROMPT,
    config=_runtime_config,
    backend=_backend_bundle.backend,
    backend_metadata=_backend_bundle.metadata,
)
