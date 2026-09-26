# Architecture

Core Python code lives under the `chainagents/` package. Root-level Python
files (`main.py`, `deepagent_runtime.py`, `chainlit_bridge.py`, …) are
compatibility wrappers and entrypoints kept for existing import paths; all of
them except `main.py` and `langgraph_app.py` emit a `DeprecationWarning` and
will be removed in a future release. Prefer importing from the package, e.g.
`chainagents.runtime.core` instead of `deepagent_runtime`.

## Package layout

```text
chainagents/
  runtime/              Core DeepAgents runtime, model setup, config parsing,
                        MCP/tool loading, persistence backends, and tracing.
  interfaces/
    chainlit/           Chainlit callbacks, UI bridge, auth, persistence,
                        uploads, async task notifications, and chat settings.
    cli/                Terminal CLI parser, status output, command execution,
                        upload handling, and event rendering.
    tui/                Full-screen Textual terminal UI.
    api/                FastAPI application, request schemas, and streaming API.
  turns/                Shared TurnRunner: one agent turn (commands, uploads,
                        streaming, generated files) used by every interface.
  events/               Shared LangGraph stream normalization used by all
                        interfaces.
  commands/             Native slash-command parsing and dispatch helpers.
  rag/                  Workspace documentation RAG config, index, uploads,
                        and search tool.
  exports/              Markdown and PDF response export helpers.
  langgraph/            Agent Server graph exports.
  util/                 Shared utility helpers.
```

## Layering

The design separates the agent runtime from the user-facing surfaces:

- **Interfaces** (Chainlit, CLI, TUI, API) only translate user input into
  calls into the shared turn pipeline, and render the resulting event
  stream back to the user.
- **Turns** — {py:mod}`chainagents.turns.runner` executes exactly one agent
  turn end to end, so every interface gets identical command handling,
  upload processing, streaming, and generated-file behavior.
- **Events** — LangGraph's raw stream is normalized once in
  {py:mod}`chainagents.events.stream`, and all interfaces consume the
  normalized stream.
- **Runtime** — {py:mod}`chainagents.runtime.core` builds the DeepAgents
  graph from `deepagent.toml`: model provider, middleware, filesystem
  backends, skills, MCP tools, subagents, and persistence.

## Runtime subsystems

Notable modules inside {py:mod}`chainagents.runtime`:

- `config` / `model_config` / `extension_config` — TOML parsing and model
  provider selection.
- `core` / `graph` — agent graph assembly.
- `backends` — pluggable filesystem state and store backends.
- `background_tasks` — queues, context scoping, and lifecycle for
  asynchronous background subagents.
- `mcp_sessions` — MCP server session management.
- `tracing` — Langfuse and LangSmith callback wiring.
- `rag_ops` — RAG indexing and search operations.

## Non-Python assets

Runtime assets stay at the repository root because they are
user/configuration content rather than importable package code:

- `deepagent.toml` and `deepagent.toml.example` — configuration.
- `skills/` — Deep Agents skill sources referenced from the TOML.
- `prompts/` — prompt files referenced by configured subagents.
- `public/` and `.chainlit/` — Chainlit static assets and native config.
- `tests/` — regression tests for runtime, interfaces, RAG, exports, and
  events.

## Development workflow

```bash
uv sync --locked
uv run ruff check chainagents *.py scripts/*.py
uv run mypy
uv run pytest
bash scripts/verify-installed-wheel.sh
```

The wheel check builds the real distribution, installs its locked runtime
dependencies in a fresh virtual environment, and runs import, CLI, API, and
configuration smoke checks from a temporary user working directory.

## Building these docs

```bash
uv sync --group dev
uv run sphinx-build -b html docs docs/_build/html
```
