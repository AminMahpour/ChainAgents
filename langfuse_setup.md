# Langfuse Setup for ChainAgents

This is the easiest way to run Langfuse with ChainAgents using Docker Compose:
run Langfuse with its official Compose stack, then point ChainAgents at that
Langfuse instance.

ChainAgents already supports Langfuse tracing through Langfuse's LangChain
callback handler. The local `compose.yaml` in this repository only starts the
optional ChainAgents Postgres service; it does not currently include a
ChainAgents app container or a Langfuse stack.

## 1. Start Langfuse

Use the official Langfuse Docker Compose deployment:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
```

Before starting it, update the secrets marked `CHANGEME` in the Langfuse
Compose file or provide equivalent `.env` values.

Then start Langfuse:

```bash
docker compose up -d
```

Open Langfuse at:

```text
http://localhost:3000
```

Create a project in Langfuse and copy its public and secret API keys.

Reference: https://langfuse.com/self-hosting/deployment/docker-compose

## 2. Enable Langfuse in ChainAgents

In `deepagent.toml`, enable the Langfuse block:

```toml
[langfuse]
enabled = true
```

## 3. Start ChainAgents with Langfuse Credentials

From the ChainAgents repository root:

```bash
cd /Users/amin/pythonProjects/ChainAgents

export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="http://localhost:3000"

uv run chainlit run main.py -w
```

Send one message in Chainlit, then check the Langfuse trace table. ChainAgents
passes the LangGraph thread ID as the Langfuse session ID for Chainlit, CLI,
TUI, and API runs.

## Optional: ChainAgents Postgres Persistence

If you also want durable ChainAgents checkpoints and `/memories/`, you can run
the repository's Postgres service:

```bash
docker compose up -d postgres
```

However, Langfuse's official Compose file also binds Postgres to host port
`5432`. If both stacks run on the same machine, change the ChainAgents
Postgres host port in `compose.yaml` from:

```yaml
ports:
  - "5432:5432"
```

to:

```yaml
ports:
  - "5433:5432"
```

Then point ChainAgents at that database:

```bash
export DATABASE_URL="postgresql://chainagents:chainagents@127.0.0.1:5433/chainagents?sslmode=disable"
```

## Notes

- For Langfuse Cloud, use `LANGFUSE_BASE_URL="https://cloud.langfuse.com"`.
- For a fully single-command deployment, ChainAgents still needs a Docker
  deployment pass: add a `Dockerfile`, `.dockerignore`, app service, Node/npm
  support for the existing `npx` MCP server, and Compose wiring from the app
  service to `http://langfuse-web:3000`.
