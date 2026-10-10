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
| `bedrock` | Amazon Bedrock Converse API; uses the standard AWS credential chain and `AWS_REGION`, with an optional `endpoint_url` override |
| `anthropic_bedrock` | Claude on Amazon Bedrock through the Anthropic Messages API; same AWS credentials, region and endpoint handling as `bedrock` |

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
`DEEPAGENT_MODEL_REASONING`, `DEEPAGENT_MODEL_API_KEY`,
`DEEPAGENT_MODEL_DISABLE_STREAMING`, and
`DEEPAGENT_MODEL_DISABLE_STREAMING_FOR_TOOL_CALLS`) replace the matching
`[model]` values. `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_REASONING`
still work as Ollama-only compatibility aliases. Other model settings (such as
`temperature`, `max_tokens`, `models`, `modalities`, or `thinking`) are only
set in the file.

## Agent runtime

The `[agent]` table controls the Deep Agent runtime:

- `recursion_limit` — LangGraph recursion limit for long tool-heavy runs.
- `state` — set to `"stateless"` to skip opening LangGraph checkpoint and
  store handles even when `DATABASE_URL` is set.
- `delete_tool_enabled` — recursive file deletion is **disabled** unless this
  is `true`.
- `execute_tool_enabled` — command execution is **disabled** unless this is
  `true`. The flag only exposes the `execute` tool; the default ChainAgents
  backend is not execution-capable, so running commands also requires a
  compatible execution-capable sandbox backend.
- `custom_instruction` / `custom_instruction_file` — extra system
  instructions, inline or from a file (typically under `prompts/`).
  Relative `custom_instruction_file` paths resolve from the directory of
  the active config file, not the repository root, so use an absolute path
  when `DEEPAGENT_CONFIG` points outside the repo.

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
`chainagents.runtime.mcp_sessions`. With `[mcp].stateful = true`, Chainlit
shares a conversation's MCP scope across saved-chat resumes. After its last
session leaves, that scope stays available for 10 minutes; at most four idle
conversation scopes are retained. Server discovery runs concurrently while
preserving each agent's configured tool order.

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
either private inline definitions or reuse of a top-level subagent.
Synchronous subagents may pin their own `model`; remote async subagents
reject it, since their model is configured on the remote graph.

## RAG

Workspace documentation RAG (index, uploads, and a search tool) is disabled
until explicitly enabled in the `[rag]` table:

```toml
[rag]
enabled = true

[rag.embedding]
provider = "ollama"
```

With Ollama embeddings, pull an embedding model such as `nomic-embed-text`
first. `provider = "auto"` follows the active chat-model provider and is only
valid for Ollama and OpenAI-compatible chat models — it is rejected for
Anthropic and Snowflake Cortex, which must set `ollama` or
`openai_compatible` explicitly with an appropriate `model` and `base_url`
(and `api_key` when needed). OpenAI-compatible embeddings also require an
explicit `[rag.embedding].model`.

## Chainlit app behavior

App-specific Chainlit switches — model selection visibility, per-message
reasoning overrides (`reasoning_mode_enabled`), and reasoning panel display
(`reasoning_steps_enabled`) — live in the top-level `[chainlit]` table of
`deepagent.toml`. The root `chainlit.toml` holds app-specific UI behavior the
Chainlit bridge owns (currently the `[steps]` auto-collapse delay, defaulting
to 3 seconds), and `.chainlit/config.toml` is the native Chainlit config.

## Tracing

Optional tracing integrations are **disabled by default** and need both an
enable switch and credentials:

- **Langfuse** — set `[langfuse].enabled = true` in `deepagent.toml`, then
  `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and optionally
  `LANGFUSE_BASE_URL`.
- **LangSmith** — set `[langsmith].enabled = true`, then provide
  `LANGSMITH_API_KEY` (the authentication credential the SDK client is
  built from). The TOML `project` setting takes precedence over
  `LANGSMITH_PROJECT`, and when neither is set, traces go to the
  `chainagents` project. `LANGSMITH_ENDPOINT` and `LANGSMITH_WORKSPACE_ID`
  are optional routing settings for non-default regions/workspaces.

Setting only the environment variables leaves tracing off; no traces are
exported until the matching table is enabled.

## Security

The HTTP API's authentication and request-size/resource limits are
documented in [API_SECURITY.md](API_SECURITY.md). These bounds are not
rate limits; a public deployment requires an external rate limiter.
