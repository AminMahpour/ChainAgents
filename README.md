# Workspace Deep Agent

This project runs a local-first LangChain Deep Agent behind a Chainlit UI.

The app is wired for:

- `ChatOllama` with `gpt-oss:20b`
- native Chainlit streaming for reasoning, tool calls, and final response
- Postgres-backed LangGraph checkpoints and durable `/memories/` when `DATABASE_URL` is set
- repo files mounted for the agent under `/workspace/`

## Environment

Set these variables before starting the app:

```bash
export DATABASE_URL="postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=disable"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export OLLAMA_MODEL="gpt-oss:20b"
export OLLAMA_REASONING="medium"
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
- if the file is missing, the app runs without extra skills, MCP servers, or custom subagents

`CHAINLIT_AUTH_SECRET`, `CHAINLIT_AUTH_USERNAME`, and `CHAINLIT_AUTH_PASSWORD` are optional:

- when all three are set, the app enables Chainlit password authentication
- together with `DATABASE_URL`, that unlocks the native Chainlit history bar and chat resume UI
- when they are unset, the app stays unauthenticated and the history bar remains unavailable

## Optional: Install Postgres

You only need Postgres if you want durable LangGraph checkpoints and `/memories/`.
If `DATABASE_URL` is unset, the app runs fully in memory for the current process.

The quickest local setup is Docker:

```bash
docker run --name chainagents-postgres \
  -e POSTGRES_USER=chainagents \
  -e POSTGRES_PASSWORD=chainagents \
  -e POSTGRES_DB=chainagents \
  -p 5432:5432 \
  -d postgres:17
```

Point the app at that database:

```bash
export DATABASE_URL="postgresql://chainagents:chainagents@127.0.0.1:5432/chainagents?sslmode=disable"
```

Optional verification:

```bash
docker exec -it chainagents-postgres psql -U chainagents -d chainagents -c "select 1;"
```

Notes:

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

Install dependencies and ensure Ollama has the model:

```bash
uv sync
ollama pull gpt-oss:20b
```

This repo now includes a live [deepagent.toml](deepagent.toml) with:

- a real `repo` MCP server pinned to `npx @modelcontextprotocol/server-filesystem@2025.8.21`
- a `repo-researcher` subagent using [prompts/repo-researcher.md](prompts/repo-researcher.md)
- the repo-local `skills/` source for both the main agent and the subagent

If the MCP package is not already cached on your machine, `npx` may download it on first use.

## Run

Start the Chainlit app:

```bash
chainlit run main.py -w
```

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

## Add MCP Servers

MCP servers are defined once in `deepagent.toml` and then attached by name to the main agent or any subagent.

Example:

```toml
[mcp]
tool_name_prefix = true

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

Current scope of this config support:

- it supports Deep Agents built-in tool surface plus config-driven skills and MCP tools
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
- On startup, the UI shows how many skill sources, MCP servers, and custom subagents were loaded from `deepagent.toml`.
