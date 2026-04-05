# Workspace Deep Agent

Local-first LangChain Deep Agent UI running on Ollama or any OpenAI-compatible server.

## Available Surfaces

- Final assistant response streams into the main chat message.
- Raw model reasoning is shown in Chain of Thought steps.
- Tool calls and tool outputs are rendered as native Chainlit tool steps.
- Each completed assistant response includes Markdown and PDF download buttons.
- Completed reasoning and tool steps auto-collapse based on `chainlit.toml`.

## Workspace Contract

- Real repo files are available under `/workspace/`.
- Memory is available under `/memories/`.
- Reuse the `LangGraph Thread ID` setting to continue a persisted thread.

## Model Defaults

- `deepagent.toml` can define `[model]` with `provider`, `base_url`, `temperature`, `name`, optional `api_key`, and `reasoning_effort`
- if `deepagent.toml` is missing, the runtime defaults to `http://127.0.0.1:11434`, `gpt-oss:20b`, and `medium`
- `DEEPAGENT_MODEL_*` env vars override the TOML defaults, and `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_REASONING` remain available as Ollama-only compatibility aliases

## Optional Persistence

- Set `DATABASE_URL` to enable durable LangGraph checkpoints and `/memories/`.
- Leave it unset to run in-memory only for the current process.

## Optional Extensions

- Use `deepagent.toml` to add skills, MCP servers, and custom subagents.
- Each subagent can have its own `skills` and `mcp_servers`.

## App Config

- `.chainlit/config.toml` controls native Chainlit settings.
- `chainlit.toml` controls app-owned UI settings such as `[steps].auto_collapse_delay_seconds`.
