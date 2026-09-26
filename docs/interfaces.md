# Interfaces

Every interface shares the same runtime and the same `deepagent.toml`
configuration. The Chainlit, CLI, TUI, and FastAPI interfaces each drive one
agent turn through the shared {py:mod}`chainagents.turns.runner` pipeline:
slash commands, uploads, streaming events, and generated files. (The
LangGraph Agent Server path is different — it exports graphs built directly
by `create_configured_graph` and does not use the turn runner, so
slash-command dispatch, upload ingestion, and generated-file collection are
not provided there.)

## Chainlit web UI

Start the app from the repository root:

```bash
uv run chainlit run main.py -w
```

The entrypoint is the root-level `main.py` wrapper, which registers the
callbacks in {py:mod}`chainagents.interfaces.chainlit.app`.

Notable UI features:

- Native streaming of reasoning steps, tool calls, and the final response.
- Image uploads (PNG, JPEG, WEBP, GIF) sent to vision-capable models as photo
  attachments for OCR or image analysis.
- Per-response download buttons for Markdown and PDF exports.
- Configurable response actions that ask the agent without showing a user
  prompt.
- Chainlit Modes for per-message reasoning selection (`Low`, `Medium`,
  `High`).
- Notifications when asynchronous background subagents finish.
- Optional authentication via `CHAINLIT_AUTH_SECRET` and
  `CHAINLIT_AUTH_USERS`.
- Optional native Chainlit history, which needs all three: `DATABASE_URL`
  set, plus both `CHAINLIT_AUTH_SECRET` and `CHAINLIT_AUTH_USERS` — the
  history bar stays hidden when authentication is off.

App-specific UI switches live in the `[chainlit]` table of
`deepagent.toml`; see
[Configuration](configuration.md#chainlit-app-behavior).

## CLI

```bash
uv run chainagents --prompt "Summarize this repository" --thread-id cli
```

The CLI exposes the agent without a UI: status output, slash-command
execution, upload handling, and rendered streaming events. Useful flags:

```bash
uv run chainagents --tui             # full-screen terminal UI
uv run chainagents --status          # print resolved runtime status
uv run chainagents --configure       # interactively write deepagent.toml
uv run chainagents --list-commands   # list configured commands
```

## TUI

The TUI is a full-screen [Textual](https://textual.textualize.io/) terminal
UI with Markdown rendering, built on the same turn runner as the other
interfaces.

## HTTP API

```bash
uv run chainagents-api --host 127.0.0.1 --port 8000
```

A FastAPI application with a streaming endpoint. It uses the same
`deepagent.toml` and environment settings as the Chainlit app.
[API_SECURITY.md](API_SECURITY.md) covers authentication and the
request-size/resource limits — note these are not rate limits, so an
external rate limiter is required in front of a public deployment.

## LangGraph Agent Server

`langgraph.json` registers the `supervisor` and `async-researcher` graphs
exported by `langgraph_app.py` (backed by
{py:mod}`chainagents.langgraph.app`), plus a custom HTTP app, so the agent can
also be served through the LangGraph development server and Studio. Start it
with enough worker capacity for the supervisor plus background tasks:

```bash
uv run --with "langgraph-cli[inmem]" langgraph dev --n-jobs-per-worker 10
```
