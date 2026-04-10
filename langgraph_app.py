from __future__ import annotations

import os

from deepagent_runtime import SYSTEM_PROMPT, create_configured_graph


os.environ.pop("__LANGGRAPH_DEFER_LOOPBACK_TRANSPORT", None)

ASYNC_RESEARCHER_PROMPT = """
You are an async research subagent.

Focus on longer background research and codebase analysis tasks. Return concise
findings with concrete file paths or sources when relevant.
""".strip()


supervisor = create_configured_graph(
    include_async_subagents=True,
    system_prompt=SYSTEM_PROMPT,
)

async_researcher = create_configured_graph(
    include_async_subagents=False,
    system_prompt=ASYNC_RESEARCHER_PROMPT,
)
