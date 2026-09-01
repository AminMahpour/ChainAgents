# Workspace Deep Agent

Local-first LangChain Deep Agent UI running on Ollama, OpenAI-compatible servers, or
Snowflake Cortex.

## Available Surfaces

- Final assistant response streams into the main chat message.
- Raw model reasoning is shown in Chain of Thought steps.
- Tool calls and tool outputs are rendered as native Chainlit tool steps.
- Todo updates from `write_todos` are rendered as the current Chainlit task list.
- Each completed assistant response includes Markdown and WeasyPrint-backed PDF download buttons.
- Completed reasoning and tool steps auto-collapse based on `chainlit.toml`.

## Workspace Contract

- Real repo files are available under `/workspace/`.
- Memory is available under `/memories/`.
- A repo-root `AGENTS.md` file is automatically included in the main agent system prompt when present.
- Reuse the `LangGraph Thread ID` setting to continue a persisted thread.

## Model Defaults

- `deepagent.toml` can define `[model]` with `provider`, `base_url` or OpenAI-compatible `endpoint_url`, `temperature`, `name`, optional `api_key`, and `reasoning_effort`
- if `deepagent.toml` is missing, the runtime defaults to `http://127.0.0.1:11434`, `gpt-oss:20b`, and `medium`
- `DEEPAGENT_MODEL_*` env vars override the TOML defaults, and `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_REASONING` remain available as Ollama-only compatibility aliases
- Snowflake Cortex uses the exact `snowflake_cortex` provider value (no aliases) and
  requires a key. The credential order is CLI `--api-key`, `SNOWFLAKE_PAT`,
  `DEEPAGENT_MODEL_API_KEY`, then `[model].api_key`.
- Use `https://<account-identifier>.snowflakecomputing.com/api/v2/cortex/v1` as the
  Cortex base URL, or the full
  `https://<account-identifier>.snowflakecomputing.com/api/v2/cortex/v1/chat/completions`
  endpoint URL.
- Cortex requires `[rag.embedding].provider` to be `ollama` or
  `openai_compatible`, with an appropriate embedding model and base URL, when RAG is
  enabled; `auto` is not valid. Tool-call IDs are normalized only for Cortex outbound
  requests.
- DeepAgents manages conversation summarization in its base agent stack.
- DeepAgents 0.7 no longer adds todo middleware by default, so ChainAgents
  explicitly restores it to preserve `write_todos`, `todos` state, and the
  planning prompt.

## Optional Persistence

- Set `DATABASE_URL` to enable durable LangGraph checkpoints and `/memories/`.
- Leave it unset to run in-memory only for the current process.

## Optional Extensions

- Use `deepagent.toml` to add skills, MCP servers, custom subagents, and async subagents.
- Recursive `delete` is disabled by default. Set
  `[agent].delete_tool_enabled = true` to expose it to the main agent and local
  synchronous subagents; remote async graphs are configured independently.
- Command `execute` is also disabled by default. Set
  `[agent].execute_tool_enabled = true` to expose it independently; the local
  backend must implement compatible sandbox execution.
- Use `[chainlit].commands` in `deepagent.toml` to add slash commands that can rewrite prompts, delegate to configured subagents, or invoke MCP tools directly.
- Each sync subagent can have its own `skills` and `mcp_servers`.
- Async subagents are Agent Protocol background jobs configured with `graph_id` and optional `url`/`headers`.
- Omit async subagent `url` only when running the co-deployed graphs from `langgraph.json`; Chainlit-only runs need an HTTP `url`.
- Chainlit uses `http://127.0.0.1:2024` as the default HTTP fallback for launch and completion notifications; override it with `CHAINLIT_ASYNC_SUBAGENT_URL`.

## App Config

- `.chainlit/config.toml` controls native Chainlit settings.
- `chainlit.toml` controls app-owned UI settings such as `[steps].auto_collapse_delay_seconds`.
