
<div align="center">
<img src="logo.png" width="300px"/>

</div>

<div align="center">

[![CI](https://img.shields.io/github/actions/workflow/status/AminMahpour/ChainAgents/ci.yml?branch=master&label=CI&logo=githubactions&logoColor=white)](https://github.com/AminMahpour/ChainAgents/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
[![License](https://img.shields.io/github/license/AminMahpour/ChainAgents?color=blue)](LICENSE)
![Chainlit](https://img.shields.io/badge/Chainlit-2.8%2B-FF5A5F)
![LangChain](https://img.shields.io/badge/LangChain-powered-1C3C3C)

**This hobby, personal project powers a highly configurable, local-first LangChain DeepAgent. We also provide interface to the agent though Chainlit, CLI and TUI.**
</div>

**Implemented features include:**

- **Sub-agents** — delegate tasks to agents with isolated context windows
- **Filesystem** — read, write, edit, or search over pluggable local, sandboxed, or remote backends
- **Context management** — summarize long threads and offload tool outputs to disk
- **Persistent memory** — pluggable state and store backends for cross-session recall
- **Skills** — reusable behaviors the agent can load on demand
- **Tools** — bring your own functions or any MCP server

Highlights
----------

- `ChatOllama`, `ChatOpenAI`, or `ChatAnthropic` with configurable model backends
- native Chainlit streaming for reasoning, tool calls, and final response
- optional Langfuse tracing through the LangChain callback handler
- Chainlit image uploads sent to vision-capable models as photo attachments for OCR or image analysis
- Chainlit OCR/image uploads accept PNG, JPEG, WEBP, and GIF files
- config-driven synchronous and async DeepAgents subagents
- DeepAgents `>=0.7.0,<0.8.0` with explicit todo planning and safe filesystem defaults
- per-response download buttons for Markdown and PDF exports
- Postgres-backed LangGraph checkpoints and durable `/memories/` when `DATABASE_URL` is set
- repo files mounted for the agent under `/workspace/`
- Chainlit Modes support for per-message reasoning selection (`Low`, `Medium`, `High`)

## Environment

Set these variables before starting the app if you want environment-based overrides:

```bash
export DATABASE_URL="postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=disable"
export DEEPAGENT_MODEL_PROVIDER="ollama"
export DEEPAGENT_MODEL_BASE_URL="http://127.0.0.1:11434"
# export DEEPAGENT_MODEL_ENDPOINT_URL="https://api.example.test/custom/v1/messages"
# export DEEPAGENT_MODEL_DISABLE_STREAMING_FOR_TOOL_CALLS="true"
export DEEPAGENT_MODEL_NAME="gpt-oss:20b"
export DEEPAGENT_MODEL_REASONING="medium"
export DEEPAGENT_RECURSION_LIMIT="200"
# export DEEPAGENT_MODEL_API_KEY="optional-for-secured-openai-compatible-servers"
# export ANTHROPIC_API_KEY="required-for-provider-anthropic-unless-DEEPAGENT_MODEL_API_KEY-is-set"
export DEEPAGENT_CONFIG="deepagent.toml"
export CHAINLIT_AUTH_SECRET="replace-with-a-long-random-string"
export CHAINLIT_AUTH_USERS='{"admin":"change-me","alice":"alice-password"}'
# export LANGFUSE_PUBLIC_KEY="pk-lf-..."
# export LANGFUSE_SECRET_KEY="sk-lf-..."
# export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

`DATABASE_URL` is optional now:

- when set, LangGraph checkpoints and `/memories/` are persisted in Postgres
- when unset, the app falls back to in-memory persistence for the current process only
- if `[agent].state = "stateless"`, LangGraph checkpoint and store handles are not opened or passed to the agent graph even when `DATABASE_URL` is set

`DEEPAGENT_CONFIG` is optional:

- defaults to `deepagent.toml` in the project root
- if the file is missing, the app falls back to built-in model defaults and runs without extra skills, MCP servers, or custom subagents

`DEEPAGENT_MODEL_*` variables are optional:

- they override the matching `[model]` values in `deepagent.toml`
- `DEEPAGENT_MODEL_API_KEY` is used for secured OpenAI-compatible servers and can also supply the Anthropic API key when `ANTHROPIC_API_KEY` is unset
- `ANTHROPIC_API_KEY` is read first when `provider = "anthropic"` or `provider = "claude"`, so stale generic keys do not override the Claude credential
- when switching to Anthropic with `DEEPAGENT_MODEL_PROVIDER`, unset stale `DEEPAGENT_MODEL_BASE_URL`; use `DEEPAGENT_MODEL_ENDPOINT_URL` with the `/v1/messages` path for env-based Anthropic proxy switches, or pass `--base-url` explicitly from the CLI
- `DEEPAGENT_MODEL_DISABLE_STREAMING` accepts `true`, `false`, or `tool_calling`; `DEEPAGENT_MODEL_DISABLE_STREAMING_FOR_TOOL_CALLS=true` is a convenience alias for `tool_calling`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_REASONING` remain supported as Ollama-only compatibility aliases

`DEEPAGENT_RECURSION_LIMIT` is optional:

- it overrides `[agent].recursion_limit` in `deepagent.toml`
- it controls the maximum LangGraph steps for a single agent run
- raise it when long tool-heavy Deep Agent runs hit `GraphRecursionError`

`CHAINLIT_AUTH_SECRET` and Chainlit user credentials are optional:

- when `CHAINLIT_AUTH_SECRET` and `CHAINLIT_AUTH_USERS` are set, the app enables Chainlit password authentication for each configured user
- `CHAINLIT_AUTH_USERS` must be a JSON object mapping usernames to passwords, e.g. `{"admin":"change-me","alice":"alice-password"}`
- the legacy `CHAINLIT_AUTH_USERNAME` and `CHAINLIT_AUTH_PASSWORD` pair still works for a single user when `CHAINLIT_AUTH_USERS` is unset
- together with `DATABASE_URL`, that unlocks the native Chainlit history bar and chat resume UI
- when auth credentials are unset, the app stays unauthenticated and the history bar remains unavailable

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
- No separate migration step is required for this app. On startup it calls the LangGraph Postgres store/checkpointer `setup()` routines and creates any missing Chainlit persistence tables (`"User"`, `"Thread"`, `"Step"`, `"Feedback"`, and `"Element"`) automatically.
- If you manage the Chainlit schema externally, set `CHAINLIT_SCHEMA_BOOTSTRAP=false` before launching the app to skip the automatic Chainlit table bootstrap.

## Optional: Enable Native Chainlit History

Chainlit only shows its built-in history sidebar when both persistence and authentication are enabled.

This app includes a simple password-based auth callback driven by environment variables:

```bash
export CHAINLIT_AUTH_SECRET="replace-with-a-long-random-string"
export CHAINLIT_AUTH_USERS='{"admin":"change-me","alice":"alice-password"}'
```

For compatibility, a single user can still be configured with:

```bash
export CHAINLIT_AUTH_SECRET="replace-with-a-long-random-string"
export CHAINLIT_AUTH_USERNAME="admin"
export CHAINLIT_AUTH_PASSWORD="change-me"
```

With `DATABASE_URL`, `CHAINLIT_AUTH_SECRET`, and either `CHAINLIT_AUTH_USERS` or the legacy username/password pair set:

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

The dependency manifests require DeepAgents `>=0.7.0,<0.8.0`; `uv sync`
installs the matching locked release.

PDF downloads are rendered with WeasyPrint. `uv sync` installs the Python package,
but WeasyPrint also needs native rendering libraries. On macOS, install them with:

```bash
brew install weasyprint
```

On Linux, install the Pango packages listed in the
[WeasyPrint installation guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation)
for your distribution before starting the app.

If you are using LM Studio or another OpenAI-compatible server instead of Ollama, skip `ollama pull`, load a model in that server, and set `[model].provider = "openai_compatible"` with the server's `base_url`.
If you are using Claude through Anthropic, set `[model].provider = "anthropic"` and provide `ANTHROPIC_API_KEY` or `DEEPAGENT_MODEL_API_KEY`.

If you enable workspace-docs RAG with Ollama embeddings, also pull an embedding model such as:

```bash
ollama pull nomic-embed-text
```

This repo now includes a live [deepagent.toml](deepagent.toml) with:

- model defaults for provider, base URL, model name, and reasoning effort
- a higher LangGraph recursion limit for longer tool-heavy Deep Agent runs
- recursive file deletion disabled unless `[agent].delete_tool_enabled = true`
- command execution disabled unless `[agent].execute_tool_enabled = true`
- a real `repo` MCP server pinned to `npx @modelcontextprotocol/server-filesystem@2025.8.21`
- a `repo-researcher` subagent using [prompts/repo-researcher.md](prompts/repo-researcher.md)
- the repo-local `skills/` source for both the main agent and the subagent

If the MCP package is not already cached on your machine, `npx` may download it on first use.

## Run

Start the Chainlit app:

```bash
chainlit run main.py -w
```

Start the FastAPI server:

```bash
uv run chainagents-api --host 127.0.0.1 --port 8000
```

The API uses the same `deepagent.toml` and environment settings as the Chainlit
and CLI entrypoints. Useful endpoints include:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/status
THREAD_ID="api-$(uuidgen)"
curl -X POST http://127.0.0.1:8000/api/agent/invoke \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Summarize this repository\",\"thread_id\":\"$THREAD_ID\"}"
curl -N -X POST http://127.0.0.1:8000/api/agent/stream \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Summarize this repository\",\"thread_id\":\"$THREAD_ID\"}"
```

Run the same underlying agent from a terminal without the Chainlit UI:

```bash
uv run chainagents --prompt "Summarize this repository" --thread-id cli
```

Start the full-screen terminal UI:

```bash
uv run chainagents --tui
```

The TUI defaults to thread ID `tui`, keeps the prompt box at the bottom, shows
the conversation in the main pane with Markdown-formatted assistant responses,
and splits reasoning and tool activity in the right sidebar. The prompt editor
supports multiple lines: press Shift+Enter to insert a newline and Enter to send
the complete prompt. Type `/` to show configured slash commands, and press Tab
to complete the first matching command. Stdio MCP server diagnostics are
written to `.files/tui-stderr.log` in TUI mode so they do not corrupt the
full-screen interface.

Useful CLI examples:

```bash
uv run chainagents --status --no-rag
uv run chainagents --configure
uv run chainagents --tui --reasoning high
uv run chainagents --list-commands
uv run chainagents --command ask-researcher --prompt "Find the config entrypoints"
uv run chainagents --stdin --model gemma4:26b --reasoning high < prompt.txt
uv run chainagents --rebuild-rag
uv run chainagents --upload-rag notes.md --prompt "Use my uploaded notes"
uv run chainagents --photo scene.jpg --prompt "Describe this photo"
```

Run `uv run chainagents --help` for all runtime flags, including model provider,
base URL, endpoint URL, API key, temperature, persistence, MCP session scope,
async subagent URL, RAG controls, photo attachments, streaming, reasoning traces,
tool traces, and JSON output.

## Project Structure

Core Python code lives under the `chainagents/` package. The root-level Python
files are compatibility wrappers and entrypoints so existing imports and commands
continue to work.

```text
chainagents/
  runtime/              Core DeepAgents runtime, model setup, config parsing,
                        MCP/tool loading, persistence backends, and Langfuse.
  interfaces/
    chainlit/           Chainlit callbacks, UI bridge, auth, persistence,
                        uploads, async task notifications, and chat settings.
    cli/                Terminal CLI parser, status output, command execution,
                        upload handling, and event rendering.
    tui/                Full-screen Textual terminal UI.
    api/                FastAPI application, request schemas, and streaming API.
  events/               Shared LangGraph stream normalization used by all
                        interfaces.
  commands/             Native slash-command parsing and dispatch helpers.
  rag/                  Workspace documentation RAG config, index, uploads,
                        and search tool.
  exports/              Markdown and PDF response export helpers.
  langgraph/            Agent Server graph exports.
  util/                 Shared utility helpers.
```

Runtime assets stay at the repository root because they are user/configuration
content rather than importable Python package code:

- `deepagent.toml` and `deepagent.toml.example`: model, agent, MCP, RAG,
  Chainlit, Langfuse, and subagent configuration.
- `skills/`: Deep Agents skill sources referenced from TOML as `skills`.
- `prompts/`: prompt files referenced by configured subagents.
- `public/` and `.chainlit/`: Chainlit static assets and native Chainlit config.
- `tests/`: regression tests for runtime, interfaces, RAG, exports, and events.

Compatibility wrappers such as `main.py`, `deepagent_runtime.py`,
`chainlit_bridge.py`, `chainagents_cli.py`, `chainagents_api.py`,
`rag_runtime.py`, and `response_exports.py` import the moved package modules.
Prefer new code under `chainagents/`, but keep the wrappers until external users
no longer rely on the old import paths.

## Model Config

You can keep the model defaults in `deepagent.toml`:

```toml
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
temperature = 0
repeat_penalty = 1.1
name = "gpt-oss:20b"
models = ["gpt-oss:20b", "gemma4:27b"]
reasoning_effort = "medium"
# Disable streaming only for requests that include tools, which can help
# model servers that emit malformed streamed tool-call chunks.
disable_streaming_for_tool_calls = false
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

For OpenAI-compatible servers with a non-standard full chat-completions endpoint:

```toml
[model]
provider = "openai_compatible"
endpoint_url = "https://api.example.test/openai/deployments/local/chat/completions?api-version=2026-01-01"
name = "your-loaded-model-id"
# api_key = "optional"
```

For Claude through Anthropic:

```toml
[model]
provider = "anthropic"
temperature = 0
name = "claude-sonnet-4-6"
models = ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5-20251001"]
reasoning_effort = "medium"
thinking = "auto"
# api_key = "optional-if-ANTHROPIC_API_KEY-or-DEEPAGENT_MODEL_API_KEY-is-set"
# base_url = "https://api.anthropic.com"
# endpoint_url = "https://claude-proxy.example/proxy/v1/messages"
```

Named model profiles let the main agent, Chainlit mode picker, and sync
subagents use different provider settings from the same config file:

```toml
[model]
provider = "openai_compatible"
base_url = "http://127.0.0.1:1234/v1"
name = "local-default"
models = ["local-default"]

[model.profiles.fast-local]
name = "local-fast"
temperature = 0.1
reasoning_effort = "low"

[model.profiles.claude-reviewer]
provider = "anthropic"
name = "claude-sonnet-4-6"
thinking = "auto"
# api_key = "optional-if-ANTHROPIC_API_KEY-or-DEEPAGENT_MODEL_API_KEY-is-set"

[agent]
model = "fast-local"

[[subagents]]
name = "reviewer"
description = "Reviews proposed changes."
system_prompt = "Review for bugs, regressions, and missing tests."
model = "claude-reviewer"
```

Notes:

- `provider` selects `ChatOllama`, `ChatOpenAI`, or `ChatAnthropic`.
- `provider = "claude"` is accepted as an alias for `provider = "anthropic"`.
- Preferred shared fields are `base_url`, `name`, `temperature`, and `reasoning_effort`.
- `repeat_penalty` is optional and currently applies to `provider = "ollama"`; when omitted, Ollama defaults are used.
- `disable_streaming = "tool_calling"` or `disable_streaming_for_tool_calls = true` bypasses model streaming only when tools are attached to the request; use this for providers that have trouble streaming tool-call chunks. `disable_streaming = true` disables model streaming for all requests.
- `endpoint_url` is an override for full non-standard model endpoint URLs. OpenAI-compatible paths ending in `/chat/completions` or `/responses` are normalized to the client base URL and query parameters are forwarded as OpenAI client default query parameters. Anthropic paths ending in `/v1/messages` are normalized to the Claude client base URL and query parameters are forwarded as Anthropic client default query parameters.
- `models` is an optional list of model IDs surfaced in Chainlit settings and modes so users can switch models per session or per message.
- `[model.profiles.<name>]` defines a named profile. Profiles inherit omitted fields from `[model]` when they keep the same provider; profiles that switch to `openai_compatible` must provide `base_url` or `endpoint_url`, and profiles that switch to `anthropic` default to `https://api.anthropic.com` unless `base_url` or `endpoint_url` is set.
- Profile names are surfaced in Chainlit settings and modes alongside `[model].models`. When a selected value matches a profile name, the full profile is used; otherwise the value is treated as a raw model name using the inherited/default provider settings.
- `[agent].model` optionally sets the main/supervisor agent's default profile or raw model name. CLI and environment model overrides still take precedence.
- `api_key` is optional for `provider = "openai_compatible"`; when omitted, the runtime sends a placeholder token that local servers like LM Studio accept.
- Anthropic requires an API key from `ANTHROPIC_API_KEY`, `DEEPAGENT_MODEL_API_KEY`, or `api_key`; when multiple are set, `ANTHROPIC_API_KEY` takes precedence over the generic key.
- When switching from another provider to Anthropic through environment or CLI overrides, provide Anthropic credentials through `ANTHROPIC_API_KEY`, `DEEPAGENT_MODEL_API_KEY`, or `--api-key`; the runtime will not reuse an `api_key` from another provider's TOML config.
- Legacy Ollama `endpoint` and `port` are still accepted when `provider = "ollama"` or omitted.
- `reasoning_effort` sets the default Chainlit reasoning level for new chats. Ollama uses that level directly, Anthropic maps it to Claude `effort`, and OpenAI-compatible servers may ignore it.
- `thinking` controls Anthropic adaptive thinking: `auto` enables it only for known supported Claude models, `adaptive` always sends `thinking = {"type": "adaptive"}`, and `disabled` never sends a thinking parameter.
- `DEEPAGENT_MODEL_PROVIDER`, `DEEPAGENT_MODEL_BASE_URL`, `DEEPAGENT_MODEL_ENDPOINT_URL`, `DEEPAGENT_MODEL_NAME`, `DEEPAGENT_MODEL_API_KEY`, `DEEPAGENT_MODEL_REASONING`, `DEEPAGENT_MODEL_DISABLE_STREAMING`, and `DEEPAGENT_MODEL_DISABLE_STREAMING_FOR_TOOL_CALLS` override the TOML defaults when set.
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and `OLLAMA_REASONING` still work as Ollama-only compatibility aliases.

## Token Usage Log

ChainAgents always appends one aggregate record per agent request to
`.files/token-usage.jsonl`. The log covers Chainlit, CLI, TUI, API, Agent
Server graphs, and nested subagent model calls. Records contain the UTC
timestamp, an opaque request identifier, completion status (`success`, `error`,
or `cancelled`), and aggregate input, output, and total token counts. Streaming
OpenAI-compatible models request provider usage metadata so completed model
calls contribute to the aggregate.

The log never includes prompts, responses, tool arguments, or per-model
breakdowns. It also omits conversation thread IDs so an agent-readable log
cannot disclose reusable checkpoint identifiers between users. The `.files/`
directory is ignored by Git.

## Optional: Enable Langfuse Tracing

Langfuse tracing is disabled by default. To enable it, set your Langfuse
credentials in the environment and turn on the TOML option:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

```toml
[langfuse]
enabled = true
```

When enabled, ChainAgents attaches Langfuse's LangChain callback handler to
Chainlit, CLI, TUI, and API agent runs. The LangGraph thread ID is also passed
as the Langfuse session ID.

## Agent Runtime Config

The `[agent]` table configures main-agent runtime behavior:

```toml
[agent]
state = "stateful"
delete_tool_enabled = false
execute_tool_enabled = false
recursion_limit = 200
memory_namespace = "filesystem"
memory_files = ["/memories/AGENTS.md"]
skills = ["skills"]
mcp_servers = ["repo"]
# custom_instruction = "Always ask clarifying questions before editing files."
# custom_instruction_file = "prompts/ui_promots.md"

[agent.reflection]
enabled = true
memory_file = "/memories/AGENTS.md"
max_lesson_chars = 700
tool_failure_mode = "unrecovered"
```

Notes:

- `state = "stateful"` passes the configured LangGraph store and checkpointer to DeepAgents so thread IDs can continue conversation state. `state = "stateless"` omits those state handles and does not expose `/memories/` when building the agent graph.
- `delete_tool_enabled = false` preserves the pre-0.7 filesystem surface. Set it to `true` only when the main agent and local synchronous subagents should receive DeepAgents 0.7's recursive `delete` tool. Remote async graphs have their own configuration.
- `execute_tool_enabled = false` keeps command execution out of the tool surface. Set it to `true` only when the main agent and local synchronous subagents should receive DeepAgents 0.7's `execute` tool. The default ChainAgents backend is not execution-capable, so opting in exposes the tool for a compatible sandbox backend but does not grant host-shell access by itself. Remote async graphs have their own configuration.
- `recursion_limit` is the LangGraph step limit for one agent run.
- The built-in default is `100`; this repo's `deepagent.toml` sets it to `200`.
- `DEEPAGENT_RECURSION_LIMIT` overrides this value when set.
- Increase it for long tool-heavy runs that hit `GraphRecursionError`; lower it if you want runaway loops to stop sooner.
- `memory_namespace` is the shared agent-scoped `StoreBackend` namespace for `/memories/`. ChainAgents passes a concrete backend instance and the explicit namespace tuple `(memory_namespace,)`, as required by DeepAgents 0.7. The default is `filesystem` to preserve existing memory data from earlier configs. Use only letters, numbers, hyphens, underscores, dots, `@`, `+`, colons, and tildes.
- `memory_files` lists `/memories/` files DeepAgents loads into the startup memory prompt. The default is `["/memories/AGENTS.md"]`; set it to `[]` to keep the memory route without startup memory loading.
- `custom_instruction` appends an inline instruction to the **main/supervisor** agent system prompt.
- `custom_instruction_file` loads that appended instruction from a UTF-8 text file. Relative paths are resolved from the active `deepagent.toml`; use either `custom_instruction` or `custom_instruction_file`, not both. This repo uses `prompts/ui_promots.md` to encourage active Chainlit generated UI panels and next-step action buttons.
- `[agent.reflection]` is opt-in. When enabled for stateful agents, ChainAgents proposes a compact lesson for `memory_file` after correction phrases such as "that was wrong" or after unrecovered tool failures. Chainlit asks with Save/Dismiss before writing through the agent; CLI, TUI, and API expose the proposal without mutating memory.

### DeepAgents 0.7 Compatibility

DeepAgents 0.7 no longer installs `TodoListMiddleware` by default. ChainAgents
adds `langchain.agents.middleware.TodoListMiddleware` explicitly to every local
main, synchronous, and nested agent stack. This preserves the `write_todos`
tool, the `todos` state channel, the planning prompt, and Chainlit task-list
rendering. Separately deployed async graphs must restore todo middleware in
their own runtime if they rely on the same behavior.

ChainAgents uses concrete `BackendProtocol` instances throughout. Backend
integrations should use the current `ls(path)`, `glob(pattern, path=None)`,
`grep(pattern, path=None, glob=None, max_count=None)`, and
`read(file_path, offset=0, limit=2000) -> ReadResult` contracts. Consume
`ReadResult.file_data` and its metadata fields rather than parsing rendered
`read_file` text.

For rendered tool output, empty `ls` and `glob` results are the string
`No files found`, not `[]`. `read_file` line numbers no longer use a fixed-width
`cat -n` gutter, so callers must not parse text by fixed character columns.
ChainAgents does not parse any of these rendered filesystem outputs.

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
- `AGENTS.md` and `AGENT.md` stay out of RAG because `AGENTS.md` is loaded directly into the main agent prompt when present.
- The persisted local index lives under [`.rag/`](.rag/) and is safe to delete and rebuild.
- With `provider = "auto"`, the embedding backend follows the active chat-model provider.
- For Ollama, the default embedding model is `nomic-embed-text`.
- For OpenAI-compatible embeddings, set `[rag.embedding].model` explicitly.
- For Anthropic chat models, set `[rag.embedding].provider` explicitly to `ollama` or `openai_compatible`; `auto` cannot infer a Claude embedding backend.
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
state = "stateful"
recursion_limit = 200
skills = ["skills"]
mcp_servers = ["repo"]
summarization_trigger_tokens = 6000
summarization_keep_tokens = 2400

[[subagents]]
name = "repo-researcher"
description = "Researches the codebase and produces concise implementation guidance."
system_prompt_file = "prompts/repo-researcher.md"
skills = ["skills/research"]
mcp_servers = ["repo"]
nested_subagents = ["reviewer"]

[[subagents.subagents]]
name = "repo-planner"
description = "Turns repository findings into an implementation plan."
system_prompt = "Use repository context to produce a concise implementation plan."
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
- `model`: optional profile name or raw model name. Profile names can switch provider settings and tool-schema handling for that sync subagent. Raw model names inherit the parent/default provider settings.
- `nested_subagents`: optional list of top-level sync subagent names exposed as children of this subagent
- `[[subagents.subagents]]`: optional inline private sync child subagents under a parent subagent

### Nested Subagents

Nested subagents let a synchronous subagent delegate work to its own synchronous
child agents. They are useful when a coordinator needs focused helpers but the
main agent should not necessarily see every helper directly.

ChainAgents supports two nesting patterns:

- use `[[subagents.subagents]]` for a private inline child available only to its
  parent
- use `nested_subagents = ["name"]` to reuse a top-level synchronous subagent as
  a child while keeping it available to the main agent

#### Add a Private Inline Child

Place `[[subagents.subagents]]` immediately after its parent
`[[subagents]]` entry:

```toml
[[subagents]]
name = "research-manager"
description = "Coordinates repository research and planning."
system_prompt = "Delegate focused work and synthesize the results."

[[subagents.subagents]]
name = "repo-planner"
description = "Turns repository findings into an implementation plan."
system_prompt = "Produce a concise, actionable implementation plan."
```

In this configuration:

- `repo-planner` is available to `research-manager`
- `repo-planner` is not exposed directly to the main agent
- the child can use the normal synchronous subagent fields, including `skills`,
  `mcp_servers`, and `model`

Use an inline child when it is an implementation detail of one parent.

#### Reuse a Top-Level Subagent as a Child

Define the child as a normal top-level `[[subagents]]` entry and reference its
name from the parent:

```toml
[[subagents]]
name = "research-manager"
description = "Coordinates repository research and review."
system_prompt = "Delegate research and ask the reviewer to check the result."
nested_subagents = ["reviewer"]

[[subagents]]
name = "reviewer"
description = "Reviews proposed changes for bugs and regressions."
system_prompt = "Return concise findings with actionable file references."
```

In this configuration:

- `reviewer` remains available directly to the main agent
- `research-manager` can also delegate to `reviewer`
- more than one parent can reference the same top-level subagent
- each referenced name must exactly match a top-level synchronous subagent

Use a reference when a role should be shared by the main agent or multiple
parents. A parent can expose several shared children, for example
`nested_subagents = ["planner", "reviewer"]`.

#### Configuration and Inheritance

Inline children accept the same fields as other synchronous subagents:
`name`, `description`, `system_prompt` or `system_prompt_file`, `skills`,
`mcp_servers`, `model`, and their own nested children. A referenced child uses
the configuration from its top-level `[[subagents]]` entry wherever it is
reused. When `model` is omitted, the child continues with its parent/default
model configuration.

#### Limitations and Validation

Nested subagents are synchronous only. Async Agent Protocol subagents must
remain top-level `[[async_subagents]]` entries.

Configuration loading rejects:

- a `nested_subagents` name that does not match a top-level synchronous
  subagent
- duplicate direct child names, including an inline child and a referenced
  child with the same name
- reference cycles such as `manager -> reviewer -> manager`
- a nested child that defines `graph_id`, because that represents an async
  subagent

As a rule of thumb, use an inline child for a parent-private specialist, a
referenced child for a shared synchronous role, and `[[async_subagents]]` for
remote or background Agent Protocol work.

Main `[agent]` additions:

- `state`: optional agent state mode. Use `stateful` for checkpointed conversation state, or `stateless` to build the DeepAgents graph without a LangGraph store, checkpointer, or `/memories/` route. Defaults to `stateful`.
- `recursion_limit`: optional positive integer LangGraph step limit for a single agent run. Defaults to `100` unless overridden by `DEEPAGENT_RECURSION_LIMIT`.
- `memory_namespace`: optional non-empty namespace for agent-scoped `/memories/` storage. Defaults to `filesystem`; allowed characters are letters, numbers, `-`, `_`, `.`, `@`, `+`, `:`, and `~`.
- `memory_files`: optional list of absolute `/memories/` file paths loaded into the DeepAgents startup memory prompt. Defaults to `["/memories/AGENTS.md"]`; use `[]` to disable startup memory loading.
- `delete_tool_enabled`: optional boolean controlling DeepAgents 0.7's recursive `delete` tool for the main agent and local synchronous subagents. Defaults to `false`.
- `execute_tool_enabled`: optional boolean controlling DeepAgents 0.7's `execute` tool for the main agent and local synchronous subagents. Defaults to `false`.
- `model`: optional profile name or raw model name for the main/supervisor agent. CLI and environment model overrides take precedence.
- `[agent.reflection]`: optional correction-learning workflow. `enabled = true` requires `state = "stateful"` and a `memory_file` under `/memories/`; `max_lesson_chars` limits proposal size; `tool_failure_mode = "unrecovered"` only proposes lessons for failed tool calls that do not produce a later final response.
- `AGENTS.md`: optional repo-root file that is automatically appended to the **main/supervisor** agent system prompt when present. It is not applied to separately configured async graph prompts.
- `custom_instruction`: optional string appended to the **main/supervisor** agent system prompt. This setting does **not** get applied to separately configured prompts such as the `async_researcher` graph prompt.
- `custom_instruction_file`: optional UTF-8 text file loaded as the main-agent custom instruction. Relative paths resolve from the active `deepagent.toml`. This is mutually exclusive with `custom_instruction`.
- ChainAgents explicitly restores `TodoListMiddleware` for the main agent and local sync subagents because DeepAgents 0.7 no longer includes it by default.
- DeepAgents still provides its own summarization middleware in the main agent and sync subagents.
- `summarization_trigger_tokens`: optional positive integer token threshold for DeepAgents' built-in summarization middleware.
- `summarization_keep_tokens`: optional positive integer token budget to keep after DeepAgents summarizes conversation history.
- Legacy `summarization_middleware_enabled` entries are still parsed for compatibility, but ChainAgents no longer injects a second summarization middleware.

## Chainlit Native Commands

You can configure slash-style commands that run from the Chainlit composer before the model call.
Chainlit also auto-generates slash commands for loaded skills.

Place this config in `deepagent.toml` or whatever file `DEEPAGENT_CONFIG` points to.
Do not put it in the app UI file `chainlit.toml` or Chainlit's native [`.chainlit/config.toml`](.chainlit/config.toml).

Example:

```toml
[chainlit]
# Set false to hide model selection in chat settings and Modes.
model_mode_enabled = true
# Set false to disable per-message reasoning overrides from the Modes picker.
reasoning_mode_enabled = true
# Set false to hide streamed reasoning step panels and reasoning task entries.
reasoning_steps_enabled = true
# Set false to hide streamed tool step panels and tool task entries.
tool_steps_enabled = true
# Set false to hide the initial startup status message ("Workspace agent ready...").
startup_status_enabled = true
# Set false to hide generated Chainlit CustomElement panels and remove the render tool.
generative_ui_enabled = true
# Set false to keep legacy non-chronological streaming order in Chainlit.
chronological_ui_enabled = true
commands = [
  { name = "ask-researcher", description = "Delegate to repo-researcher.", target = "subagent", value = "repo-researcher", template = "{input}" },
  { name = "repo-readme", description = "Run an MCP tool directly.", target = "mcp_tool", value = "repo_read_file", mcp_server = "repo", template = "{\"path\":\"README.md\"}" },
  { name = "summarize", description = "Apply a prompt template.", target = "prompt", value = "Summarize the input", template = "Summarize this:\n{input}" }
]
starters = [
  { label = "Explain this repo", message = "Explain the architecture of this repository and identify the most important files.", command = "ask-researcher", icon = "book-open" },
  { label = "Review current changes", message = "Review the current working tree changes for bugs, regressions, and missing tests." }
]
```

`target` modes:

- `prompt`: rewrites the user prompt before sending it to the agent.
- `subagent`: rewrites the user prompt to direct the runtime to delegate via the configured subagent.
- `mcp_tool`: invokes the configured MCP tool directly and returns tool output in chat.

Notes:

- The `[chainlit]` table for native commands belongs in `deepagent.toml`, alongside `[model]`, `[agent]`, `[mcp]`, `[[subagents]]`, and `[[async_subagents]]`.
- `[chainlit].model_mode_enabled = false` hides the Model selector in chat settings and the Model mode group, and ignores per-message model overrides from UI modes.
- `[chainlit].reasoning_mode_enabled = false` hides the Reasoning mode group and ignores per-message reasoning overrides from UI modes.
- `[chainlit].reasoning_steps_enabled = false` hides streamed reasoning `cl.Step` panels and reasoning task-list entries while preserving model reasoning settings.
- `[chainlit].tool_steps_enabled = false` hides streamed tool `cl.Step` panels and tool task-list entries while preserving tool execution.
- `[chainlit].startup_status_enabled = false` disables the initial startup status message that summarizes runtime configuration.
- `[chainlit].generative_ui_enabled = false` hides generated Chainlit CustomElement panels and removes the `render_chainlit_ui` tool from the main agent.
- `[chainlit].chronological_ui_enabled = false` disables chronological UI ordering so response tokens stream immediately and reasoning steps are not force-rolled at tool boundaries.
- Command `name` is invoked as `/<name>` and must be unique.
- `template` is optional and may include `{input}`.
- For `mcp_tool`, user command arguments must be valid JSON, e.g. `/repo-readme {"path":"README.md"}`.
- Each discovered skill also becomes `/<skill-name>` automatically. For example, a skill with `name: reviewer` is available as `/reviewer`.
- Skill-backed commands always force the main agent to read the selected `SKILL.md` first and use it for that turn.
- If a configured `[chainlit].commands` entry and a skill share the same slash name, the configured command wins.
- `starters` define starter prompts shown by Chainlit before the first message in a thread.
- Starter `label` and `message` are required. Starter `command` and `icon` are optional.

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
- [langgraph_app.py](langgraph_app.py) exports request-scoped graph factories
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
- `mcp_servers` on `[[subagents.subagents]]` attaches those MCP tools only to that nested child subagent.
- Skills and MCP servers are independent. You can use neither, either, or both on any subagent.
- Relative `cwd` values are resolved from the location of `deepagent.toml`.
- `tool_name_prefix = true` is recommended when multiple MCP servers expose overlapping tool names.
- `stateful = true` keeps MCP sessions open per LangGraph thread while the app process is running.
- `stateful = false` recreates the MCP session for every tool call.

Current scope of this config support:

- it exposes `ls`, `read_file`, `write_file`, `edit_file`, `glob`, and `grep`
  by default, plus `write_todos`, subagent tools, config-driven skills, and MCP
  tools
- it exposes recursive `delete` only when
  `[agent].delete_tool_enabled = true`
- it exposes `execute` only when `[agent].execute_tool_enabled = true`; the
  configured backend must also implement sandbox execution
- it supports config-driven sync subagents and async Agent Protocol subagents
- it does not yet provide a config-driven registry for custom Python tools per subagent beyond MCP
- if you need custom Python tools, extend [chainagents/runtime/core.py](chainagents/runtime/core.py)

See [deepagent.toml.example](deepagent.toml.example) for a complete example.

## Workspace Contract

- `/workspace/` maps to this repo on disk.
- `/memories/` is available in stateful mode under the configured agent-scoped namespace and durable across LangGraph threads only when `DATABASE_URL` is configured.
- any other absolute path is treated as ephemeral scratch space by the deep agent backend.

## Notes

- Native Chainlit history is available when `DATABASE_URL`, `CHAINLIT_AUTH_SECRET`, and either `CHAINLIT_AUTH_USERS` or the legacy username/password pair are configured.
- If `DATABASE_URL` is set but authentication is not configured, Chainlit still persists thread records, but they are not browseable from the UI.
- When `DATABASE_URL` is unset, thread IDs only persist while the process stays alive.
- When `DATABASE_URL` is set, durable state is available through LangGraph thread IDs. You can reuse a thread ID from the chat settings panel to continue the same checkpointed thread.
- When `[agent].state = "stateless"`, thread IDs still identify requests and MCP/RAG scopes, but the agent graph does not checkpoint conversation state, receive a LangGraph store, or expose `/memories/`.
- MCP stateful sessions are process-local. They survive tool calls in the same thread, but not an app restart.
- On startup, the UI shows how many skill sources, MCP servers, custom subagents, and async subagents were loaded from `deepagent.toml`.

## License

This hobby, personal project is made available under the MIT License and is
built with open source Python libraries. See [LICENSE](LICENSE) for the full
text.
