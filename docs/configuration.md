# Configuration

ChainAgents is configured through a TOML file plus environment variables. The
config path defaults to `deepagent.toml` in the project root and can be
overridden with `DEEPAGENT_CONFIG`. If the file is missing, the app falls back
to built-in model defaults and runs without extra skills, MCP servers, or
custom subagents.

See [deepagent.toml.example](https://github.com/AminMahpour/ChainAgents/blob/master/deepagent.toml.example)
in the repository for a fully commented reference file.

## Model

The `[model]` table selects the provider and endpoint. Supported providers:

| Provider | Notes |
| --- | --- |
| `ollama` | Local Ollama server, e.g. `base_url = "http://127.0.0.1:11434"` |
| `openai_compatible` | Any OpenAI-compatible server (LM Studio, vLLM, OpenAI) |
| `anthropic` | Claude models; needs `ANTHROPIC_API_KEY` or `DEEPAGENT_MODEL_API_KEY` |
| `snowflake_cortex` | Snowflake Cortex endpoint; needs a Snowflake PAT |

Example:

```toml
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "gpt-oss:20b"
reasoning_effort = "medium"
```

At launch, the supported `DEEPAGENT_MODEL_*` environment overrides
(`DEEPAGENT_MODEL_PROVIDER`, `DEEPAGENT_MODEL_BASE_URL`,
`DEEPAGENT_MODEL_ENDPOINT_URL`, `DEEPAGENT_MODEL_NAME`,
`DEEPAGENT_MODEL_REASONING`, `DEEPAGENT_MODEL_API_KEY`) replace the matching
`[model]` values. Other model settings (such as `temperature`, `max_tokens`,
`models`, `modalities`, or `thinking`) are only set in the file.

## Agent runtime

The `[agent]` table controls the Deep Agent runtime:

- `recursion_limit` — LangGraph recursion limit for long tool-heavy runs.
- `state` — set to `"stateless"` to skip opening LangGraph checkpoint and
  store handles even when `DATABASE_URL` is set.
- `delete_tool_enabled` — recursive file deletion is **disabled** unless this
  is `true`.
- `execute_tool_enabled` — command execution is **disabled** unless this is
  `true`.
- `custom_instruction` / `custom_instruction_file` — extra system
  instructions, inline or from a file under `prompts/`.

## Skills

Skills are reusable behaviors the agent loads on demand. They live under
`skills/` as directories with a `SKILL.md` file, and are referenced from the
TOML config via a `skills` list. See the README's *Add Skills* section for the
directory layout.

## MCP servers

MCP (Model Context Protocol) servers are configured as **named tables** under
`[mcp.servers]`, one table per server:

```toml
[mcp.servers.docs]
transport = "streamable_http"
url = "http://127.0.0.1:9000/mcp"
```

Declaring a server only registers it — its tools are not exposed until you
attach it to an agent by name via `[agent].mcp_servers` (default `[]`):

```toml
[agent]
mcp_servers = ["docs"]
```

Individual subagents can instead attach servers through their own
`mcp_servers` list. MCP sessions are managed by
`chainagents.runtime.mcp_sessions`.

## Subagents

Subagents delegate tasks with isolated context windows and are configured as
`[[subagents]]` entries. Their execution modes:

- **Synchronous** (default) — the parent agent waits for the result.
- **Local background** — additionally requires `background = true` on the
  entry *and* `[agent.background_subagents].enabled = true`; these run as
  process-local background tasks with completion notifications in the UI.
- **Remote async** — Agent Protocol jobs declared under `[[async_subagents]]`
  (or as a `[[subagents]]` entry carrying a `graph_id`).

Subagents can also be **nested** — a subagent can declare its own children,
either private inline definitions or reuse of a top-level subagent — and each
subagent may pin its own `model`.

## RAG

Workspace documentation RAG (index, uploads, and a search tool) is disabled
until configured in the `[rag]` table. With Ollama embeddings, pull an
embedding model such as `nomic-embed-text` first.

## Chainlit app behavior

App-specific Chainlit switches — model selection visibility, per-message
reasoning overrides (`reasoning_mode_enabled`), and reasoning panel display
(`reasoning_steps_enabled`) — live in the top-level `[chainlit]` table of
`deepagent.toml`. The root `chainlit.toml` holds only native Chainlit
settings (currently the `[steps]` table), and `.chainlit/config.toml` is the
native Chainlit config.

## Tracing

Optional tracing integrations are **disabled by default** and need both an
enable switch and credentials:

- **Langfuse** — set `[langfuse].enabled = true` in `deepagent.toml`, then
  `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optionally
  `LANGFUSE_BASE_URL`.
- **LangSmith** — set `[langsmith].enabled = true`, then provide
  `LANGSMITH_API_KEY` (the authentication credential the SDK client is
  built from) plus `LANGSMITH_PROJECT`; `LANGSMITH_ENDPOINT` and
  `LANGSMITH_WORKSPACE_ID` are optional routing settings for non-default
  regions/workspaces.

Setting only the environment variables leaves tracing off; no traces are
exported until the matching table is enabled.

## Security

The HTTP API's authentication and rate-limiting model is documented in
[API_SECURITY.md](API_SECURITY.md).
