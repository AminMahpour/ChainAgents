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
reasoning = "medium"
```

Every `[model]` value can be overridden at launch with the matching
`DEEPAGENT_MODEL_*` environment variable (see
[Getting Started](getting_started.md#environment-variables)).

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

MCP (Model Context Protocol) servers are configured as `[[mcp.servers]]`
entries and expose their tools to the agent. MCP sessions are managed by
`chainagents.runtime.mcp_sessions`.

## Subagents

Subagents delegate tasks with isolated context windows and are configured as
`[[subagent]]` entries. They can be:

- **Synchronous or asynchronous** — async subagents run as local background
  tasks with completion notifications in the UI.
- **Nested** — a subagent can declare its own children, either private inline
  definitions or reuse of a top-level subagent.
- **Model-specific** — each subagent may pin its own `model`.

## RAG

Workspace documentation RAG (index, uploads, and a search tool) is disabled
until configured in the `[rag]` table. With Ollama embeddings, pull an
embedding model such as `nomic-embed-text` first.

## Chainlit app behavior

Chainlit UI behavior is configured in `chainlit.toml` (app-specific options
such as model selection visibility, per-message reasoning overrides, and
reasoning panel display) and `.chainlit/config.toml` (native Chainlit
config).

## Tracing

Optional tracing integrations:

- **Langfuse** — set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
  optionally `LANGFUSE_BASE_URL`.
- **LangSmith** — set the standard `LANGSMITH_*` environment variables
  (`LANGSMITH_PROJECT`, and optionally `LANGSMITH_ENDPOINT` and
  `LANGSMITH_WORKSPACE_ID` for non-default regions/workspaces).

## Security

The HTTP API's authentication and rate-limiting model is documented in
[API_SECURITY.md](API_SECURITY.md).
