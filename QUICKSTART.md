# Quick Start Guide

A friendly introduction to ChainAgents. If you're looking for the full documentation, see [README.md](README.md).

## What is this?

ChainAgents is a local-first LangChain DeepAgent project that powers a highly configurable AI assistant. You can interact with it through:

- **Chainlit UI** — a web-based chat interface
- **CLI** — terminal-based command-line interface
- **TUI** — a full-screen terminal UI with Markdown formatting

Everything runs locally on your machine by default.

## Prerequisites

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** — the fast Python package installer
- **[Ollama](https://ollama.com)** (optional, for running models locally)
- **[Docker](https://www.docker.com)** (optional, for Postgres persistence)

## 5-Minute Setup

### 1. Clone and install

```bash
git clone git@github.com:AminMahPour/ChainAgents.git
cd ChainAgents
uv sync
```

This installs the locked DeepAgents 0.7 release. ChainAgents explicitly keeps
its todo-planning middleware, so multi-step work can still use `write_todos`
and appear as a task list in Chainlit.

### 2. Pick a model provider

**Option A: Ollama (local, free)**

```bash
ollama install   # if not already installed
ollama pull gpt-oss:20b
```

**Option B: LM Studio (local, OpenAI-compatible)**

1. Download [LM Studio](https://lmstudio.ai/) and load a model
2. Start its local server (usually on `http://localhost:1234`)
3. Skip the `ollama pull` step below

**Option C: OpenAI or Anthropic (cloud)**

Set your API key:

```bash
export DEEPAGENT_MODEL_API_KEY="your-openai-key"
# or
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 3. Run the app

```bash
chainlit run main.py -w
```

Open your browser at `http://localhost:8000`. You should see a "Workspace agent ready" message.

## Your First Chat

Try one of these prompts:

- `"Summarize this repository"`
- `"What does this project do?"`
- `"Tell me about the available skills"`

## Choosing a Model Provider

| Provider | Cost | Speed | Privacy | Setup |
|---|---|---|---|---|
| **Ollama** | Free | Depends on hardware | Full privacy | `ollama pull <model>` |
| **LM Studio** | Free | Depends on hardware | Full privacy | Start local server |
| **OpenAI** | Pay-per-token | Fast | Data sent to OpenAI | Set `DEEPAGENT_MODEL_API_KEY` |
| **Anthropic** | Pay-per-token | Fast | Data sent to Anthropic | Set `ANTHROPIC_API_KEY` |

To switch providers, edit `deepagent.toml`:

```toml
[model]
provider = "ollama"        # or "openai_compatible" or "anthropic"
name = "gpt-oss:20b"       # change model name
base_url = "http://127.0.0.1:11434"   # for Ollama or LM Studio
```

Then restart the app.

## Common Next Steps

### Enable RAG (document search)

Edit `deepagent.toml`:

```toml
[rag]
enabled = true
include_globs = ["README.md", "prompts/**/*.md", "skills/**/*.md"]
```

### Allow recursive file deletion

DeepAgents 0.7 includes a recursive `delete` tool, but ChainAgents disables it
by default. Opt in only when the main agent and local synchronous subagents
should be able to remove files:

```toml
[agent]
delete_tool_enabled = true
```

Leave the setting unset or `false` to retain the safer default filesystem
surface.

### Add a Skill

1. Create a folder: `mkdir -p skills/my-skill`
2. Add `SKILL.md`:

```markdown
---
name: my-skill
description: Use this skill when you need to do X.
---

# my-skill

When asked to do something:
1. First step...
2. Second step...
```

3. Add to `deepagent.toml`:

```toml
[agent]
skills = ["skills"]
```

### Use Postgres for Chat History

```bash
docker compose up -d postgres
export DATABASE_URL="postgresql://chainagents:chainagents@127.0.0.1:5432/chainagents?sslmode=disable"
```

### Enable Authentication

```bash
export CHAINLIT_AUTH_SECRET="a-long-random-string"
export CHAINLIT_AUTH_USERNAME="admin"
export CHAINLIT_AUTH_PASSWORD="change-me"
```

With both `DATABASE_URL` and auth set, you get sign-in and chat history in the UI.

## CLI Quick Reference

```bash
# Chat without the UI
uv run chainagents --prompt "Hello, agent!"

# Terminal TUI
uv run chainagents --tui

# Check agent status
uv run chainagents --status

# Configure deepagent.toml interactively
uv run chainagents --configure

# List available commands
uv run chainagents --list-commands

# Delegate to a subagent
uv run chainagents --command ask-researcher --prompt "Research this topic"
```

Run `uv run chainagents --help` for all options.

## Troubleshooting

### "Model not found" or connection errors

- For Ollama: run `ollama list` to see installed models, and `ollama serve` to ensure it's running.
- For LM Studio: verify the local server is running and update `base_url` in `deepagent.toml`.
- For cloud providers: check that your API key is set and valid.

### Chainlit history doesn't appear

Make sure both `DATABASE_URL` and all three `CHAINLIT_AUTH_*` variables are set.

### RAG says "not ready" on startup

Make sure the embedding model is available. For Ollama: `ollama pull nomic-embed-text`.

### The app crashes on startup

Check that `deepagent.toml` exists and is valid TOML. If it's missing, the app falls back to built-in defaults but may not load any skills or MCP servers.

## Where to Go Next

- [README.md](README.md) — Full feature documentation
- [deepagent.toml.example](deepagent.toml.example) — Complete config reference
- [skills/](skills/) — Example skill definitions
- [prompts/](prompts/) — Subagent prompt templates
- [AGENTS.md](AGENTS.md) — Developer notes for this repo
