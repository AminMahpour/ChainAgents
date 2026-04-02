# Workspace Deep Agent

Local-first LangChain Deep Agent UI running on Ollama.

## Available Surfaces

- Final assistant response streams into the main chat message.
- Raw model reasoning is shown in Chain of Thought steps.
- Tool calls and tool outputs are rendered as native Chainlit tool steps.

## Workspace Contract

- Real repo files are available under `/workspace/`.
- Memory is available under `/memories/`.
- Reuse the `LangGraph Thread ID` setting to continue a persisted thread.

## Model Defaults

- `deepagent.toml` can define `[model]` with `endpoint`, `port`, `name`, and `reasoning_effort`
- if `deepagent.toml` is missing, the runtime defaults to `http://127.0.0.1:11434`, `gpt-oss:20b`, and `medium`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_REASONING` still override the TOML defaults when set

## Optional Persistence

- Set `DATABASE_URL` to enable durable LangGraph checkpoints and `/memories/`.
- Leave it unset to run in-memory only for the current process.

## Optional Extensions

- Use `deepagent.toml` to add skills, MCP servers, and custom subagents.
- Each subagent can have its own `skills` and `mcp_servers`.
