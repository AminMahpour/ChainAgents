# Getting Started

ChainAgents is a local-first LangChain Deep Agent. This guide gets you from a
fresh clone to a working chat in about five minutes.

## Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** — the Python package installer used by this project
- **[Ollama](https://ollama.com)** (optional, for running models locally)
- **[Docker](https://www.docker.com)** (optional, for Postgres persistence)

## Install

```bash
git clone git@github.com:AminMahpour/ChainAgents.git
cd ChainAgents
uv sync
```

This installs the locked DeepAgents `0.7.17` release. PDF exports use
WeasyPrint, which also needs native rendering libraries — on macOS install
them with `brew install weasyprint`; on Linux, see the
[WeasyPrint installation guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#installation).

## Pick a model provider

- **Ollama (local, free):** `ollama pull gpt-oss:20b`
- **LM Studio / OpenAI-compatible:** load a model, start the local server
  (usually `http://localhost:1234`), and set
  `[model].provider = "openai_compatible"` with the server's `base_url`
  *and* a `[model].name` matching the model ID your server exposes (the
  default name targets Ollama and will otherwise 404).
- **OpenAI:** the checked-in default points at Ollama, so switch providers
  explicitly: set `[model].provider = "openai_compatible"`, a
  `[model].base_url` (or `endpoint_url`) targeting OpenAI, and an OpenAI
  `[model].name` in `deepagent.toml` (or the matching `DEEPAGENT_MODEL_*`
  overrides), and provide your key via `DEEPAGENT_MODEL_API_KEY` or
  `[model].api_key`.
- **Anthropic:** set `[model].provider = "anthropic"` and a `[model].name`,
  and provide your key via `ANTHROPIC_API_KEY`,
  `DEEPAGENT_MODEL_API_KEY`, or `[model].api_key`.
- **Snowflake Cortex:** set `[model].provider = "snowflake_cortex"` with the
  Cortex `endpoint_url` and `[model].name`, and provide a Snowflake PAT via
  `SNOWFLAKE_PAT`, `DEEPAGENT_MODEL_API_KEY`, or `[model].api_key`.

See [Configuration](configuration.md) for the full set of options.

If you plan to enable workspace-docs RAG with Ollama embeddings, also pull an
embedding model:

```bash
ollama pull nomic-embed-text
```

## Run

```bash
# Chainlit web UI
uv run chainlit run main.py -w

# Terminal CLI
uv run chainagents

# Full-screen terminal UI
uv run chainagents --tui

# FastAPI server
uv run chainagents-api --host 127.0.0.1 --port 8000
```

The Chainlit app is then available at `http://localhost:8000` by default.

## Persistence

`DATABASE_URL` is optional:

- **When set**, LangGraph checkpoints and `/memories/` are persisted in
  Postgres — but only in stateful mode. With `[agent].state = "stateless"`
  in `deepagent.toml`, no checkpoint or store handles are opened at all and
  nothing is persisted, even with `DATABASE_URL` set. A local Postgres is
  available via `compose.yaml`:

  ```bash
  docker compose up -d
  export DATABASE_URL="postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=disable"
  ```

- **When unset**, the app falls back to in-memory persistence for the current
  process only.

## Environment variables

Set these before starting the app for environment-based overrides:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Postgres connection for persistence |
| `DEEPAGENT_CONFIG` | Path to the config file (default: `deepagent.toml`) |
| `DEEPAGENT_MODEL_PROVIDER` | `ollama`, `openai_compatible`, `anthropic`, or `snowflake_cortex` |
| `DEEPAGENT_MODEL_BASE_URL` | Model server base URL |
| `DEEPAGENT_MODEL_NAME` | Model name |
| `DEEPAGENT_MODEL_REASONING` | Reasoning effort (`low` / `medium` / `high`) |
| `DEEPAGENT_MODEL_API_KEY` | API key for secured servers |
| `ANTHROPIC_API_KEY` | Anthropic-specific key (alternative to `DEEPAGENT_MODEL_API_KEY`) |
| `SNOWFLAKE_PAT` | Snowflake Cortex-specific key (alternative to `DEEPAGENT_MODEL_API_KEY`) |
| `DEEPAGENT_RECURSION_LIMIT` | LangGraph recursion limit |
| `CHAINLIT_AUTH_SECRET` | Long random string for Chainlit auth |
| `CHAINLIT_AUTH_USERS` | JSON map of Chainlit usernames to passwords |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | Optional Langfuse tracing |

The supported environment overrides are `DEEPAGENT_MODEL_PROVIDER`,
`DEEPAGENT_MODEL_BASE_URL`, `DEEPAGENT_MODEL_ENDPOINT_URL`,
`DEEPAGENT_MODEL_NAME`, `DEEPAGENT_MODEL_REASONING`, and
`DEEPAGENT_MODEL_API_KEY`, each overriding the matching `[model]` value in
`deepagent.toml`. Other model settings (such as `temperature`, `max_tokens`,
or `thinking`) remain file-configured.

## Next steps

- [Configuration](configuration.md) — model, agent, skills, MCP, and subagent config
- [Interfaces](interfaces.md) — Chainlit, CLI, TUI, and HTTP API details
- [Architecture](architecture.md) — how the codebase is organized
