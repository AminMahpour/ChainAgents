# Interfaces

Every interface shares the same runtime and the same `deepagent.toml`
configuration. Each drives one agent turn through the shared
{py:mod}`chainagents.turns.runner` pipeline: slash commands, uploads,
streaming events, and generated files.

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
- Optional native Chainlit history backed by Postgres when `DATABASE_URL`
  is set.

UI behavior is tuned in `chainlit.toml` and `.chainlit/config.toml`; see
[Configuration](configuration.md#chainlit-app-behavior).

## CLI

```bash
uv run chainagents "summarize the notes in /workspace"
```

The CLI exposes the agent without a UI: status output, slash-command
execution, upload handling, and rendered streaming events. Useful
subcommands:

```bash
uv run chainagents tui         # full-screen terminal UI
uv run chainagents status      # check agent status
uv run chainagents configure   # interactively write deepagent.toml
uv run chainagents commands    # list available commands
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
`deepagent.toml` and environment settings as the Chainlit app. Authentication,
rate limiting, and deployment guidance are covered in
[API_SECURITY.md](API_SECURITY.md).

## LangGraph Agent Server

`langgraph.json` points at {py:mod}`chainagents.langgraph.app` so the agent can
also be served through the LangGraph development server and Studio.
