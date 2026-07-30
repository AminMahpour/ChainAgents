"""Expose configured LangGraph agents for Agent Server deployment."""

from __future__ import annotations

import os

from chainagents.runtime import (
    SYSTEM_PROMPT,
    build_agent_server_graph_factory,
    create_configured_graph,
)


os.environ.pop("__LANGGRAPH_DEFER_LOOPBACK_TRANSPORT", None)

ASYNC_RESEARCHER_PROMPT = """
You are an async research subagent.

Focus on longer background research and codebase analysis tasks. Return concise
findings with concrete file paths or sources when relevant.
""".strip()


supervisor = build_agent_server_graph_factory(
    create_configured_graph(
        include_async_subagents=True,
        system_prompt=SYSTEM_PROMPT,
        apply_custom_instruction=True,
    )
)

async_researcher = build_agent_server_graph_factory(
    create_configured_graph(
        include_async_subagents=False,
        system_prompt=ASYNC_RESEARCHER_PROMPT,
    )
)
