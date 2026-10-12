# Live Agents: On-Demand, Long-Lived Subagents

Date: 2026-10-12
Status: Draft for review

## Goal

Let the main agent spawn subagents at runtime, converse with them over many
turns, and tear them down when finished. Subagents are defined on the fly from
TOML templates and may be customized within a TOML allowlist. The user can see
and control them from Chainlit (side panel + slash commands), the TUI and the
HTTP API.

## Decisions

| Topic | Decision |
|---|---|
| Definition | TOML templates as defaults; the main agent may customize within a TOML allowlist. |
| Allowlist scope | Switch `allow.applies_to`: `"all"` (default; templates must also fit the allowlist) or `"overrides"` (allowlist only limits spawn-time additions). |
| Execution model | Mailbox worker: own persistent thread; each inbox batch triggers one turn; idle between turns. |
| Visibility | Agent tools + user slash commands + Chainlit side panel. |
| Nesting | None. Only the main agent (or the user) spawns. Live agents may message each other. |
| Durability | Process-local. Restart ends live agents. Histories may persist in the checkpointer but are not resumed. |
| Architecture | New `LiveAgentManager` built on the existing `MessageBroker`, separate from `BackgroundTaskManager`. |

Rejected alternatives:

- deepagents `AsyncSubAgent` over a local LangGraph server: requires a server,
  `update_async_task` interrupts/restarts rather than queueing, no peer
  messaging, and Chainlit streaming would require polling.
- A "persistent" flag on background tasks: conflates one-shot task semantics
  (batches, result formats, `get_background_task` results) with long-lived
  agents.

## 1. Configuration

```toml
[agent.live_agents]
enabled = false
max_live_per_session = 4
max_live_total = 16
idle_timeout_seconds = 1800     # 0 disables the idle timeout
max_turns_per_agent = 50
turn_recursion_limit = 100
delete_history_on_teardown = true
notify_main_on_user_actions = true
stream_activity = false         # stream turn activity as nested Chainlit steps

[agent.live_agents.allow]
applies_to = "all"              # "all" | "overrides"
models = ["fast-local"]         # [model.profiles] names
mcp_servers = []
skills = []
tools = ["read_file", "ls", "grep", "glob", "write_file", "edit_file"]
allow_custom_prompt = true

[[agent.live_agents.templates]]
name = "researcher"
description = "Reads sources and reports findings."
system_prompt_file = "prompts/live/researcher.md"   # or system_prompt
model = "fast-local"            # optional; defaults to the main agent model
skills = []
mcp_servers = []
tools = ["read_file", "ls", "grep", "glob"]

[chainlit]
live_agents_panel_enabled = true
```

Field vocabulary (`system_prompt`/`system_prompt_file`, `model`, `skills`,
`mcp_servers`) matches `[[subagents]]`.

### Startup validation

Startup fails with a descriptive `ValueError` when:

- `enabled = true` and `[agent.messaging].enabled` is not `true`.
- A template name is duplicated, empty, or lacks a description.
- A template defines both `system_prompt` and `system_prompt_file`, or the
  prompt file does not exist.
- A referenced model profile, skill path or MCP server does not exist (in the
  allowlist or in a template).
- `applies_to = "all"` and a template grants a model, tool, skill or MCP server
  outside the allowlist.
- `applies_to` is not `"all"` or `"overrides"`; a limit is not a positive
  integer (`idle_timeout_seconds` may be `0`).
- A configured tool or MCP tool collides with a reserved live-agent tool name,
  or a skill / `[chainlit].commands` entry collides with a reserved slash
  command.

### Spawn resolution

A spawn request carries `template?`, `name?`, `instructions?`, `model?`,
`add_tools?`, `remove_tools?`, `add_skills?`, `add_mcp_servers?`,
`initial_message?`. Resolution produces a `ResolvedLiveAgentSpec`
(`name`, `template`, `system_prompt`, `model`, `tools`, `skills`,
`mcp_servers`):

1. Base: the named template, or, when no template is given, an empty base (no
   tools, no skills, no MCP servers, main agent model). A template-less spawn
   requires `allow_custom_prompt = true` and non-empty `instructions`.
2. Apply additions; each added item must be in the matching allowlist.
   Removals are always permitted. A `model` override must be in
   `allow.models`.
3. `instructions`: appended under `## Task from supervisor` when a template is
   used; used as the full system prompt otherwise. When
   `allow_custom_prompt = false`, `instructions` is still accepted with a
   template (it is task framing, not a prompt replacement).
4. `name`: defaults to `<template>-<n>` (or `agent-<n>`); must match
   `[a-z0-9][a-z0-9_-]{0,39}` and be unique among the session's non-stopped
   agents.
5. Any violation returns an error naming the rejected value and the permitted
   values; the spawn does not occur.

Global gates always apply: `delete`/`execute` tools are only available if
`delete_tool_enabled`/`execute_tool_enabled` are set; live agents never receive
live-agent management tools or background task tools.

## 2. Runtime architecture

Package `chainagents/runtime/live_agents/`:

| Module | Responsibility |
|---|---|
| `config.py` | Parse/validate `[agent.live_agents]` into `LiveAgentsConfig`; `resolve_spawn(request, config) -> ResolvedLiveAgentSpec`. Pure, no I/O beyond prompt-file reads at load time. |
| `registry.py` | `LiveAgentRecord` (id, name, template, status, address, thread_id, turns, consecutive_failures, created_at, last_active_at); per-session bookkeeping; limit enforcement; lookup by name or id. |
| `worker.py` | `LiveAgentWorker`: one asyncio task per agent running the mailbox loop. |
| `manager.py` | `LiveAgentManager`: `spawn`, `list`, `get`, `send_from_user`, `teardown`, `teardown_session`; builds graphs; owns workers; emits events. |
| `events.py` | `LiveAgentEvent` dataclasses (`spawned`, `turn_started`, `turn_finished`, `message`, `stopped`) and a subscriber registry. |
| `tools.py` | Main-agent tools (Section 3). |

The manager is created once per process alongside `BackgroundTaskManager` and
scopes everything by the same session id. Graphs are built with the existing
runtime graph builder from the resolved spec, using the shared checkpointer
and store, and with the messaging middleware/tools attached.

### Lifecycle

Statuses: `idle`, `running`, `stopping`, `stopped`, `failed`.

1. **Spawn**: resolve spec, check limits, build graph, open broker address
   `live:<name>:<id>` with parent `main`, start worker, status `idle`, emit
   `spawned`. Deliver `initial_message` (sender = spawner) if given.
2. **Turn**: worker awaits `broker.wait(address)`, claims pending messages,
   sets `running`, emits `turn_started`, and invokes the graph on
   `thread_id = live-<session>-<id>` with the claimed messages as human
   messages (formatted `Message from <sender> (id <id>):\n<body>`). The final
   assistant text is sent back via `broker.send` to each distinct sender of
   the claimed batch. Status returns to `idle`, `turns += 1`, emit
   `turn_finished`.
3. **Messages during a turn**: stay in the inbox and form the next batch. The
   agent can use messaging tools mid-turn (e.g. ask `main` a question).
4. **Teardown** triggers: `teardown_live_agent`, `/kill`, panel button, idle
   timeout, `max_turns_per_agent` reached (after the final reply), session end
   (Chainlit chat end, CLI run end, API session deletion). Steps: status
   `stopping`; cancel the worker (aborting any in-flight turn); close the
   broker address and notify senders of undelivered messages; delete the
   checkpointer thread if `delete_history_on_teardown`; status `stopped`; emit
   `stopped` with reason. A stopped record is retained for listing until the
   session ends.

### Errors

- A failed turn sends `[error] <exception summary>` to the batch senders,
  increments `consecutive_failures`, and returns to `idle`. A successful turn
  resets the counter.
- Three consecutive failures set status `failed` and tear the agent down with
  reason `failed`.
- A spawn failure during graph build releases the reserved slot and name.
- Turn recursion is bounded by `turn_recursion_limit`.

### Concurrency and tracing

One turn at a time per agent; agents run concurrently up to
`max_live_per_session` / `max_live_total`. Each turn is traced as a child of
the main agent's run via the existing LangSmith parent-capture mechanism.

## 3. Agent tools

Main agent only, when `enabled = true`:

| Tool | Args | Returns |
|---|---|---|
| `list_live_agent_templates` | — | Templates (name, description, default model/tools/skills/MCP) and allowlist. |
| `spawn_live_agent` | as in Spawn resolution | `{id, name, address, status, resolved: {model, tools, skills, mcp_servers}}` or an error. |
| `list_live_agents` | `include_stopped?` | Per agent: id, name, template, status, turns, last_active_at, pending count. |
| `teardown_live_agent` | `agent`, `reason?` | Final status. |

Messaging uses the existing tools unchanged (`send_agent_message`,
`wait_for_agent_messages`, `list_agent_recipients`, `get_agent_message`);
recipients may be addressed by live agent name. Live agents receive their
resolved tools plus the messaging tools.

Prompting: when enabled, a short block is appended to the main agent system
prompt explaining live agents, async replies, and the expectation to tear
agents down when done. While agents are live, a one-line roster
(`name: status`) is injected into each main-agent model request.

## 4. User control and visibility

### Slash commands (Chainlit and TUI)

Handled before the agent runs, via the native command parser:

| Command | Effect |
|---|---|
| `/agents` | Table of the session's live agents. |
| `/spawn <template> [name] [-- instructions]` | Spawn through the same resolver and allowlist. |
| `/msg <agent> <text>` | Send as sender `user`; the reply is delivered to the user. |
| `/kill <agent>` / `/kill all` | Tear down. |

When `notify_main_on_user_actions = true`, each user action leaves a short FYI
in the main agent's inbox (spawn, kill, and message/reply summaries).

### Chainlit panel

- `public/elements/LiveAgentsPanel.jsx` rendered in the element sidebar,
  opened automatically on the first spawn; controlled by
  `[chainlit].live_agents_panel_enabled`.
- One card per agent: name, template, model, status badge, turns, and message
  thread (main↔agent, user↔agent, agent↔agent) with timestamps.
- Card actions: Message (inline input, equivalent to `/msg`) and Shut down
  (confirm).
- A Chainlit listener subscribes to manager events and re-renders panel props,
  throttled to ~4 updates/second. Props are built by a pure function from
  manager state.
- When `[agent.live_agents].stream_activity = true`, turn activity is also
  rendered as nested steps using the existing background-activity renderer.

### TUI, CLI, HTTP API

- TUI: slash commands and one-line status notices on reply/stop.
- CLI (one-shot): all live agents are torn down at run end.
- HTTP API: `GET /sessions/{id}/live-agents`,
  `POST /sessions/{id}/live-agents/{agent}/messages`,
  `DELETE /sessions/{id}/live-agents/{agent}`.

## 5. Testing

pytest, fake chat models, no network.

- `config.py`: every validation error; `applies_to` both modes; resolution
  cases (template only; add/remove within and outside allowlist; model
  override; instructions append vs replace; template-less rejected when
  `allow_custom_prompt = false`; name defaults and validation).
- `registry.py` / `manager.py` with a stub graph builder: limits, name
  uniqueness, lookup, teardown mid-turn cancels, session teardown, idle timeout
  and turn cap via injected clock.
- `worker.py` with a scripted fake model: message → one turn → reply to sender;
  history persists across turns; mid-turn messages form the next batch;
  failure path and three-strike `failed`.
- Tools/graph: reserved-name collisions; tools absent when disabled; live
  agents lack management and background tools; end-to-end spawn → message →
  reply injected into main model input → teardown.
- Interfaces: slash command parsing and FYI notes; HTTP routes; panel props
  builder.
- Docs: README section and commented example in `deepagent.toml.example`.

## Out of scope

- Resuming live agents after process restart.
- Nested spawning by live agents.
- Free-running (non-mailbox) agent loops.
- Remote/distributed workers.
