# Workspace Deep Agent

This project runs a local-first LangChain Deep Agent behind a Chainlit UI.

The app is wired for:

- `ChatOllama` or `ChatOpenAI` with a configurable local model backend
- native Chainlit streaming for reasoning, tool calls, and final response
- config-driven synchronous and async DeepAgents subagents
- per-response download buttons for Markdown and PDF exports
- Postgres-backed LangGraph checkpoints and durable `/memories/` when `DATABASE_URL` is set
- repo files mounted for the agent under `/workspace/`

## Environment

Set these variables before starting the app if you want environment-based overrides:

```bash
export DATABASE_URL="postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=disable"
export DEEPAGENT_MODEL_PROVIDER="ollama"
export DEEPAGENT_MODEL_BASE_URL="http://127.0.0.1:11434"
export DEEPAGENT_MODEL_NAME="gpt-oss:20b"
export DEEPAGENT_MODEL_REASONING="medium"
# export DEEPAGENT_MODEL_API_KEY="optional-for-secured-openai-compatible-servers"
export DEEPAGENT_CONFIG="deepagent.toml"
export CHAINLIT_AUTH_SECRET="replace-with-a-long-random-string"
export CHAINLIT_AUTH_USERNAME="admin"
export CHAINLIT_AUTH_PASSWORD="change-me"
```

`DATABASE_URL` is optional now:

- when set, LangGraph checkpoints and `/memories/` are persisted in Postgres
- when unset, the app falls back to in-memory persistence for the current process only

`DEEPAGENT_CONFIG` is optional:

- defaults to `deepagent.toml` in the project root
- if the file is missing, the app falls back to built-in model defaults and runs without extra skills, MCP servers, or custom subagents

`DEEPAGENT_MODEL_*` variables are optional:

- they override the matching `[model]` values in `deepagent.toml`
- `DEEPAGENT_MODEL_API_KEY` is only needed for secured OpenAI-compatible servers
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_REASONING` remain supported as Ollama-only compatibility aliases

`CHAINLIT_AUTH_SECRET`, `CHAINLIT_AUTH_USERNAME`, and `CHAINLIT_AUTH_PASSWORD` are optional:

- when all three are set, the app enables Chainlit password authentication
- together with `DATABASE_URL`, that unlocks the native Chainlit history bar and chat resume UI
- when they are unset, the app stays unauthenticated and the history bar remains unavailable

## Optional: Install Postgres

You only need Postgres if you want durable LangGraph checkpoints and `/memories/`.
If `DATABASE_URL` is unset, the app runs fully in memory for the current process.

This repo includes a Compose file for a local Postgres instance:

```bash
docker compose up -d postgres
```

Point the app at that database:

```bash
export DATABASE_URL="postgresql://chainagents:chainagents@127.0.0.1:5432/chainagents?sslmode=disable"
```

Optional verification:

```bash
docker compose exec postgres psql -U chainagents -d chainagents -c "select 1;"
```

Notes:

- The Compose file lives at [compose.yaml](compose.yaml) and creates a persistent `postgres-data` volume.
- If you already have Postgres installed locally, create an empty database and set `DATABASE_URL` to that instance instead.
- No separate migration step is required for this app. On startup it calls the LangGraph Postgres store and checkpointer `setup()` routines automatically.

## Optional: Enable Native Chainlit History

Chainlit only shows its built-in history sidebar when both persistence and authentication are enabled.

This app includes a simple password-based auth callback driven by environment variables:

```bash
export CHAINLIT_AUTH_SECRET="replace-with-a-long-random-string"
export CHAINLIT_AUTH_USERNAME="admin"
export CHAINLIT_AUTH_PASSWORD="change-me"
```

With both `DATABASE_URL` and the `CHAINLIT_AUTH_*` variables set:

- users can sign in through Chainlit's native auth screen
- the history sidebar can list and reopen prior chats
- resumed chats default the LangGraph thread ID to the persisted Chainlit thread ID for that conversation

If you leave auth disabled, Chainlit can still persist thread records in Postgres, but the native history bar will stay hidden.

## Setup

Install dependencies, then either pull an Ollama model or point `deepagent.toml` at an OpenAI-compatible server such as LM Studio:

```bash
uv sync
ollama pull gpt-oss:20b
```

If you are using LM Studio or another OpenAI-compatible server instead of Ollama, skip `ollama pull`, load a model in that server, and set `[model].provider = "openai_compatible"` with the server's `base_url`.

If you enable workspace-docs RAG with Ollama embeddings, also pull an embedding model such as:

```bash
ollama pull nomic-embed-text
```

This repo now includes a live [deepagent.toml](deepagent.toml) with:

- model defaults for provider, base URL, model name, and reasoning effort
- a real `repo` MCP server pinned to `npx @modelcontextprotocol/server-filesystem@2025.8.21`
- a `repo-researcher` subagent using [prompts/repo-researcher.md](prompts/repo-researcher.md)
- the repo-local `skills/` source for both the main agent and the subagent

If the MCP package is not already cached on your machine, `npx` may download it on first use.

## Run

Start the Chainlit app:

```bash
chainlit run main.py -w
```

## Model Config

You can keep the model defaults in `deepagent.toml`:

```toml
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
temperature = 0
name = "gpt-oss:20b"
reasoning_effort = "medium"
```

For LM Studio or another OpenAI-compatible server:

```toml
[model]
provider = "openai_compatible"
base_url = "http://127.0.0.1:1234/v1"
temperature = 0
name = "your-loaded-model-id"
reasoning_effort = "medium"
# api_key = "optional"
```

Notes:

- `provider` selects `ChatOllama` or `ChatOpenAI`.
- Preferred shared fields are `base_url`, `name`, `temperature`, and `reasoning_effort`.
- `api_key` is optional and only used for `provider = "openai_compatible"`. When omitted, the runtime sends a placeholder token that local servers like LM Studio accept.
- Legacy Ollama `endpoint` and `port` are still accepted when `provider = "ollama"` or omitted.
- `reasoning_effort` sets the default Chainlit reasoning level for new chats. Ollama uses that level directly; OpenAI-compatible servers may ignore it.
- `DEEPAGENT_MODEL_PROVIDER`, `DEEPAGENT_MODEL_BASE_URL`, `DEEPAGENT_MODEL_NAME`, `DEEPAGENT_MODEL_API_KEY`, and `DEEPAGENT_MODEL_REASONING` override the TOML defaults when set.
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_REASONING` still work as Ollama-only compatibility aliases.

## Optional: Enable Workspace Docs RAG

The app can build a local-first RAG index over repo documentation and expose it to the main agent as the `search_workspace_knowledge` tool.

Example config:

```toml
[rag]
enabled = true
persist_directory = ".rag"
include_globs = ["README.md", "chainlit.md", "prompts/**/*.md", "skills/**/*.md"]
exclude_globs = ["AGENTS.md", "AGENT.md"]
chunk_size = 1200
chunk_overlap = 200
top_k = 4

[rag.embedding]
provider = "auto"
```

Notes:

- The default corpus is docs-only: `README.md`, `chainlit.md`, `prompts/**/*.md`, and `skills/**/*.md`.
- The persisted local index lives under [`.rag/`](.rag/) and is safe to delete and rebuild.
- With `provider = "auto"`, the embedding backend follows the active chat-model provider.
- For Ollama, the default embedding model is `nomic-embed-text`.
- For OpenAI-compatible embeddings, set `[rag.embedding].model` explicitly.
- On startup, the UI reports whether RAG is ready and how many files/chunks were indexed.
- The startup message includes a `Rebuild Knowledge Index` action so you can refresh the index after documentation changes.
- The startup message also includes `Upload File For RAG`, which lets you add text-based files to the current chat thread's knowledge index.
- Composer file attachments are enabled for text-based uploads; attached files are automatically ingested into the current thread's RAG store before the model responds.
- Uploaded files are thread-scoped and persist under `.rag/uploads/`, so they do not leak into other chat threads.

## Chainlit App Config

This repo also includes an app-specific `chainlit.toml` for UI behavior that the bridge owns:

```toml
[steps]
auto_collapse_delay_seconds = 3
```

Notes:

- `chainlit.toml` is separate from Chainlit's native [`.chainlit/config.toml`](.chainlit/config.toml).
- `[steps].auto_collapse_delay_seconds` controls how long completed reasoning and tool steps stay expanded before auto-collapsing.
- If `chainlit.toml` is missing or invalid, the app falls back to `3` seconds.

## Add Skills

The runtime now supports Deep Agents skill sources through `deepagent.toml`.

1. Create a skill source directory in the repo, for example:

```text
skills/
├── repo-docs/
│   └── SKILL.md
└── reviewer/
    └── SKILL.md
```

2. Add the source directory to `deepagent.toml`:

```toml
[agent]
skills = ["skills"]
```

Notes:

- Relative paths in `deepagent.toml` are resolved from the config file location.
- Relative skill paths are automatically mapped into the Deep Agents virtual filesystem as `/workspace/...`.
- Each skill source directory should contain one or more skill folders, and each skill folder must contain `SKILL.md`.
- You can also use explicit virtual paths such as `"/workspace/skills/"` if you prefer.
- Every loaded skill is also exposed as a Chainlit slash command using the skill `name`, for example `reviewer` becomes `/reviewer`.
- Running a skill-backed slash command forces the main agent to read that skill's `SKILL.md` and apply it for that request.
- Skills loaded through `[agent].skills` and sync `[[subagents]].skills` are both considered for slash commands, but explicit `[chainlit].commands` take precedence on name collisions.

Minimal `SKILL.md` example:

```md
---
name: reviewer
description: Use this skill when reviewing code changes for bugs and missing tests.
---

# reviewer

When asked to review code:
1. Read the relevant files first.
2. Focus on bugs, regressions, and missing tests.
3. Return concise findings with file references.
```

## Add Subagents

Custom subagents are also loaded from `deepagent.toml`, and each subagent can have its own `skills` and `mcp_servers`.

Example:

```toml
[mcp]
tool_name_prefix = true
stateful = true

[mcp.servers.repo]
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem@2025.8.21", "."]
cwd = "."

[agent]
skills = ["skills"]
mcp_servers = ["repo"]

[[subagents]]
name = "repo-researcher"
description = "Researches the codebase and produces concise implementation guidance."
system_prompt_file = "prompts/repo-researcher.md"
skills = ["skills/research"]
mcp_servers = ["repo"]

[[subagents]]
name = "reviewer"
description = "Reviews proposed changes for bugs and regressions."
system_prompt = """
You are a strict code reviewer.
Focus on correctness, regressions, and missing tests.
Keep findings concise and actionable.
"""
skills = ["skills/reviewer"]
mcp_servers = ["repo"]
# model = "gpt-oss:20b"
```

Supported subagent fields:

- `name`: required
- `description`: required
- `system_prompt` or `system_prompt_file`: one is required
- `skills`: optional list of skill source paths for that subagent
- `mcp_servers`: optional list of MCP server names to attach to that subagent
- `model`: optional model override

## Chainlit Native Commands

You can configure slash-style commands that run from the Chainlit composer before the model call.
Chainlit also auto-generates slash commands for loaded skills.

Place this config in `deepagent.toml` or whatever file `DEEPAGENT_CONFIG` points to.
Do not put it in the app UI file `chainlit.toml` or Chainlit's native [`.chainlit/config.toml`](.chainlit/config.toml).

Example:

```toml
[chainlit]
commands = [
  { name = "ask-researcher", description = "Delegate to repo-researcher.", target = "subagent", value = "repo-researcher", template = "{input}" },
  { name = "repo-readme", description = "Run an MCP tool directly.", target = "mcp_tool", value = "repo_read_file", mcp_server = "repo", template = "{\"path\":\"README.md\"}" },
  { name = "summarize", description = "Apply a prompt template.", target = "prompt", value = "Summarize the input", template = "Summarize this:\n{input}" }
]
```

`target` modes:

- `prompt`: rewrites the user prompt before sending it to the agent.
- `subagent`: rewrites the user prompt to direct the runtime to delegate via the configured subagent.
- `mcp_tool`: invokes the configured MCP tool directly and returns tool output in chat.

Notes:

- The `[chainlit]` table for native commands belongs in `deepagent.toml`, alongside `[model]`, `[agent]`, `[mcp]`, `[[subagents]]`, and `[[async_subagents]]`.
- Command `name` is invoked as `/<name>` and must be unique.
- `template` is optional and may include `{input}`.
- For `mcp_tool`, user command arguments must be valid JSON, e.g. `/repo-readme {"path":"README.md"}`.
- Each discovered skill also becomes `/<skill-name>` automatically. For example, a skill with `name: reviewer` is available as `/reviewer`.
- Skill-backed commands always force the main agent to read the selected `SKILL.md` first and use it for that turn.
- If a configured `[chainlit].commands` entry and a skill share the same slash name, the configured command wins.

## Add Async Subagents

Async subagents are loaded from `deepagent.toml` as background Agent Protocol jobs. They are useful for long-running or remote work where the main agent should return a task ID immediately and let you check, update, cancel, or list tasks later.

Example:

```toml
[[async_subagents]]
name = "remote-researcher"
description = "Runs longer research jobs in the background on an Agent Protocol server."
graph_id = "researcher"
# Omit url for ASGI transport in a co-deployed LangGraph setup.
# Set url for HTTP transport to a remote Agent Protocol server.
# url = "https://researcher-deployment.langsmith.dev"
# headers = { Authorization = "Bearer ${RESEARCHER_TOKEN}" }
```

Supported async subagent fields:

- `name`: required
- `description`: required
- `graph_id`: required graph or assistant ID on the Agent Protocol server
- `url`: optional remote Agent Protocol server URL; omit for ASGI transport in co-deployed LangGraph setups
- `headers`: optional request headers for remote/self-hosted Agent Protocol servers

For compatibility with DeepAgents' native discriminator, a `[[subagents]]` entry with a `graph_id` is also treated as an async subagent. Async subagents cannot define sync-only fields such as `system_prompt`, `skills`, `mcp_servers`, or `model`; those capabilities are configured on the remote graph.

This repo includes a LangGraph co-deployment entrypoint for ASGI transport:

- [langgraph.json](langgraph.json) registers `supervisor` and `async-researcher`
- [langgraph_app.py](langgraph_app.py) exports both graphs
- omit `url` in `deepagent.toml` when running through LangGraph Agent Server

Run the co-deployed Agent Protocol server with enough worker capacity for the supervisor plus background tasks:

```bash
uv run --with "langgraph-cli[inmem]" langgraph dev --n-jobs-per-worker 10
```

ASGI transport is only available in this LangGraph server path. If you launch the UI with `chainlit run main.py -w`, use HTTP transport instead by setting `url = "http://127.0.0.1:2024"` on the async subagent.

Chainlit also starts a background notifier for launched async tasks. It polls the Agent Protocol server and posts a message when a task reaches `success`, `error`, `cancelled`, `interrupted`, or `timeout`. If `deepagent.toml` omits `url` for ASGI co-deployment, Chainlit defaults to `http://127.0.0.1:2024`, the usual `langgraph dev` URL. Override it with:

```bash
export CHAINLIT_ASYNC_SUBAGENT_URL="http://127.0.0.1:2024"
```

Optional:

```bash
export CHAINLIT_ASYNC_TASK_POLL_SECONDS="5"
```

## Add MCP Servers

MCP servers are defined once in `deepagent.toml` and then attached by name to the main agent or any subagent.

Example:

```toml
[mcp]
tool_name_prefix = true
stateful = true

[mcp.servers.repo]
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem@2025.8.21", "."]
cwd = "."

[mcp.servers.docs]
transport = "http"
url = "http://localhost:8000/mcp"

[mcp.servers.github]
transport = "sse"
url = "http://localhost:8080/sse"
headers = { Authorization = "Bearer ${GITHUB_TOKEN}" }

[agent]
mcp_servers = ["repo"]

[[subagents]]
name = "repo-researcher"
description = "Researches the repo and docs."
system_prompt = "Use the repo and docs MCP servers to answer questions."
mcp_servers = ["repo", "docs"]

[[subagents]]
name = "release-assistant"
description = "Works with repository metadata and hosted services."
system_prompt = "Use the GitHub MCP server when release metadata is needed."
mcp_servers = ["github"]
```

Supported MCP config fields:

- top-level `[mcp] tool_name_prefix = true|false`
- top-level `[mcp] stateful = true|false`
- `[mcp.servers.<name>] transport`
- for `stdio`: `command`, `args`, optional `cwd`, optional `env`
- for `http`, `streamable_http`, `streamable-http`: `url`, optional `headers`
- for `sse`: `url`, optional `headers`
- for `websocket`: `url`

Notes:

- `mcp_servers` on `[agent]` attaches those MCP tools to the main agent.
- `mcp_servers` on `[[subagents]]` attaches those MCP tools only to that subagent.
- Skills and MCP servers are independent. You can use neither, either, or both on any subagent.
- Relative `cwd` values are resolved from the location of `deepagent.toml`.
- `tool_name_prefix = true` is recommended when multiple MCP servers expose overlapping tool names.
- `stateful = true` keeps MCP sessions open per LangGraph thread while the app process is running.
- `stateful = false` recreates the MCP session for every tool call.

Current scope of this config support:

- it supports Deep Agents built-in tool surface plus config-driven skills and MCP tools
- it supports config-driven sync subagents and async Agent Protocol subagents
- it does not yet provide a config-driven registry for custom Python tools per subagent beyond MCP
- if you need custom Python tools, extend [deepagent_runtime.py](deepagent_runtime.py)

See [deepagent.toml.example](deepagent.toml.example) for a complete example.

## Workspace Contract

- `/workspace/` maps to this repo on disk.
- `/memories/` is durable across LangGraph threads only when `DATABASE_URL` is configured.
- any other absolute path is treated as ephemeral scratch space by the deep agent backend.

## Notes

- Native Chainlit history is available when both `DATABASE_URL` and the `CHAINLIT_AUTH_*` variables are configured.
- If `DATABASE_URL` is set but authentication is not configured, Chainlit still persists thread records, but they are not browseable from the UI.
- When `DATABASE_URL` is unset, thread IDs only persist while the process stays alive.
- When `DATABASE_URL` is set, durable state is available through LangGraph thread IDs. You can reuse a thread ID from the chat settings panel to continue the same checkpointed thread.
- MCP stateful sessions are process-local. They survive tool calls in the same thread, but not an app restart.
- On startup, the UI shows how many skill sources, MCP servers, custom subagents, and async subagents were loaded from `deepagent.toml`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
