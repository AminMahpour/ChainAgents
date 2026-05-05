# Agent Notes

This file is for local, untracked notes that help future agent sessions work in this repository.

## Project overview

- This repo runs a local-first LangChain Deep Agent behind a Chainlit UI.
- The main app entrypoint is `main.py`.
- Core runtime code lives primarily in `deepagent_runtime.py`, `chainlit_bridge.py`, and `langgraph_app.py`.

## Key config files

- `deepagent.toml`: model, MCP, skill, and subagent configuration.
- `chainlit.toml`: app-specific Chainlit UI behavior.
- `.chainlit/config.toml`: native Chainlit config.
- `compose.yaml`: optional local Postgres for persistence.

## Local development

- Install dependencies with `uv sync`.
- Run the app with `chainlit run main.py -w`.
- `DATABASE_URL` is optional. When unset, persistence is in-memory only.
- Skills live under `skills/`.
- Prompt files live under `prompts/`.

## Notes for future agent work

- Prefer reading `README.md` first for current setup and runtime expectations.
- Treat this file as local workspace context. It should not be committed.
- Add or update notes here when repo-specific workflow details matter for future sessions.
