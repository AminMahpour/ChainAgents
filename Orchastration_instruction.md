# Main Agent Orchestration Instructions

This project can make the main agent behave as an orchestrator by keeping direct
MCP access off the main agent and assigning tool access to focused subagents.

## Configuration Pattern

Define MCP servers globally, but do not attach them to the main agent:

```toml
[mcp.servers.repo]
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem@2025.8.21", "."]
cwd = "."

[agent]
recursion_limit = 200
skills = ["skills"]
# Do not set mcp_servers here if the main agent should only orchestrate.
```

Attach MCP servers only to subagents that need those tools:

```toml
[[subagents]]
name = "repo-researcher"
description = "Researches the repository and reports findings to the supervisor."
system_prompt_file = "prompts/repo-researcher.md"
skills = ["skills"]
mcp_servers = ["repo"]
```

With this setup, the main agent can delegate to `repo-researcher`, while only
that subagent receives the `repo` MCP tools.

## Main Agent Instruction

Use either `AGENTS.md` or `[agent].custom_instruction` in `deepagent.toml` to
make the orchestration behavior explicit:

```toml
[agent]
custom_instruction = """
Act as the supervisor/orchestrator.
Do not perform repository inspection directly when a configured subagent can do it.
Delegate codebase research, file reading, and tool-heavy work to the appropriate subagent.
Synthesize subagent results and ask follow-up questions only when needed.
"""
```

## RAG Consideration

If `[rag].enabled = true`, the main agent still receives the
`search_workspace_knowledge` tool. For a stricter router/synthesizer role,
disable RAG:

```toml
[rag]
enabled = false
```

## Caveats

- This removes MCP tools from the main agent, but DeepAgents may still provide
  built-in planning or workspace behavior depending on backend/runtime behavior.
- Async subagents cannot declare `mcp_servers` in this local config. Their MCP
  access must be configured in the remote graph.
- Subagent descriptions should be narrow and explicit so the main agent can
  route work reliably.
