# Context Window Usage and Summarization Visibility — Design

Date: 2026-10-15
Status: Draft for review
Repos: ChainAgents (backend), SparxUI (frontend)
Depends on: per-profile context window and summarization budget (PR #126)

## Goal

Show SparxUI users how full each agent's context window is, what fills it,
when summarization will run, and when it last ran.

Success criteria:

- While a turn streams, a gauge next to the chat input shows the main agent's
  context usage against its window and updates after every main-agent model
  call.
- The gauge's popup breaks usage into system prompt, tool definitions,
  conversation summary, messages and tool results, plus free space, and marks
  where summarization triggers.
- The popup lists every agent in the conversation (main agent, sync and
  background subagents, live agents) with its own usage against its own window.
- When DeepAgents summarizes the main agent's history, the conversation shows a
  notice and the gauge drops.
- Reopening a conversation restores the gauge.
- Older backends, OpenAgents and the Chainlit/CLI/TUI interfaces are unaffected.

Non-goals: per-tool or per-message sizes, cost estimates, historical charts,
changing summarization behavior, showing context usage in Chainlit, the CLI or
the TUI.

## Decisions

| Topic | Decision |
|---|---|
| Breakdown | Main categories: `system`, `tools`, `summary`, `messages`, `tool_results`, plus free space. |
| Scope | Main agent on the gauge; every agent in a per-agent list. |
| Delivery | Main agent over the stream (`context_usage` events); all agents through a process-local registry and `GET /api/context-usage`. |
| Accuracy | Total from the server's `usage_metadata.input_tokens` when present; categories estimated with DeepAgents' token counter and scaled to that total. |
| Summarization notice | Detected from a new summary message at the head of the request; streamed for the main agent only. |

Rejected alternatives:

- Stream only: background subagents and live agents never appear in the main
  stream, and nothing survives a reload.
- Endpoint only: the gauge lags behind model calls and SparxUI polls during
  every turn.

## 1. Measurement (ChainAgents)

New module `chainagents/runtime/context_usage.py` with
`ContextUsageMiddleware`, `ContextUsageSnapshot` and `ContextUsageRegistry`.

### Middleware

`ContextUsageMiddleware(registry, extensions)` is added to every agent stack
built by `build_agent_middleware` (main agent, configured subagents, live
agents). DeepAgents' `general-purpose` delegate inherits main-agent middleware
only when it replaces one of the delegate's default slots, so the middleware is
added to the delegate explicitly: through the harness-profile extra middleware
for the implicit delegate, and through ChainAgents' explicit delegate spec when
live agents are enabled. Each stack holds exactly one instance. DeepAgents
merges it after its core stack, so it sees the request after summarization and
argument truncation. Middleware in DeepAgents' tail (skills, memory, prompt
caching) runs inside it; the `system` remainder rule below accounts for what
they add.

The middleware resolves which agent is calling at call time, because one
instance can serve several agents (the `general-purpose` delegate inherits the
main agent's instances, and a background run reuses its sync subagent graph):

- `kind`: `configurable["chainagents_agent_kind"]` when set (`background` or
  `live`); otherwise `main` when the run metadata has no `lc_agent_name`, and
  `subagent` when it does.
- `agent`: `configurable["chainagents_agent_name"]` when set (background and
  live runs), else `main` for the main agent, else `lc_agent_name` (DeepAgents
  sets it to the subagent's name).
- Registry key: `configurable["chainagents_agent_instance"]` when set (the
  background task id or live agent id), else `kind:agent`.
- Nested calls: the run-config identity keys describe only the run's own agent,
  but LangGraph passes `configurable` on to nested `task` calls. When the
  metadata's `lc_agent_name` is present and differs from
  `chainagents_agent_name`, the caller is a nested subagent: `kind` is
  `subagent`, `agent` is `lc_agent_name`, and the key is
  `<chainagents_agent_instance>:<lc_agent_name>`. The session still comes
  from `chainagents_session_id`.
- Only `kind == "main"` emits stream events.

On each `awrap_model_call` / `wrap_model_call`:

1. Estimate tokens per category from the request:
   - `system`: `request.system_message`.
   - `tools`: `request.tools`.
   - `summary`: messages whose `additional_kwargs["lc_source"] == "summarization"`.
   - `tool_results`: `ToolMessage`s.
   - `messages`: all other messages.

   Counting uses `langchain_core.messages.utils.count_tokens_approximately`,
   the counter DeepAgents' summarization uses, applied per category.
2. Call the handler.
3. If the response's `AIMessage.usage_metadata["input_tokens"]` is a positive
   integer, set `exact = true` and make the categories sum to it: `tools`,
   `summary`, `messages` and `tool_results` keep their estimates, scaled down
   proportionally only if together they exceed the total, and `system` is the
   remainder. The remainder also covers prompt sections added inside this
   middleware (skills, memory) and provider formatting. Otherwise keep the
   estimates and set `exact = false`.
4. Build the snapshot, store it in the registry, and when
   `emit_stream_events` is true, write it to the LangGraph custom stream writer
   as `{"kind": "context_usage", "snapshot": {...}}`.
5. Summarization detection: if the first non-system message is a summary
   message whose content hash differs from the last one recorded for this
   session and agent, set `last_summarized_at_utc` to now. For the main agent,
   also emit the existing `summarization_status` custom event with
   `status = "completed"`, `source = agent_label` and the message
   "Conversation summarized to fit the context window." The first summary seen
   after a process restart also counts as new; this is acceptable because the
   notice is informational.

Any exception inside steps 1, 3, 4 or 5 is caught and logged at debug level;
the model call's result or exception passes through unchanged. Step 2 is the
only step whose exceptions propagate.

### Snapshot

`ContextUsageSnapshot` (frozen dataclass, `to_payload()` for JSON):

| Field | Type | Meaning |
|---|---|---|
| `agent` | str | Display label: `main`, the subagent name, or the live agent name. |
| `kind` | str | `main`, `subagent`, `background` or `live`. |
| `model` | str | Model name. |
| `window` | int \| null | `model.profile["max_input_tokens"]`, or null. |
| `used` | int | Total input tokens of the last call. |
| `categories` | object | `system`, `tools`, `summary`, `messages`, `tool_results` (ints). |
| `trigger` | int \| null | Effective summarization trigger in tokens. |
| `keep` | int \| null | Effective tokens kept after summarization. |
| `exact` | bool | Whether `used` came from the server. |
| `at_utc` | str | ISO 8601 UTC time of the call. |
| `last_summarized_at_utc` | str \| null | ISO 8601 UTC time summarization last ran for this agent. |
| `status` | str | `active`, or `finished`/`stopped` for ended background and live agents. |

Effective thresholds follow the precedence of PR #126 for each value: the
model profile's budget keys, then `[agent].summarization_trigger_tokens` /
`summarization_keep_tokens`, then DeepAgents' defaults. Fraction defaults are
converted to tokens with the window (`0.85 × window` and `0.10 × window`,
rounded down). With no window and no configured value, the trigger is the
DeepAgents fallback of 170,000 tokens and `keep` is null because the fallback
keeps a message count, not tokens. The threshold resolution is shared with the
summarization factory in `middleware.py` so the two cannot diverge.

## 2. Delivery (ChainAgents)

### Stream event

`AgentStreamEventAdapter._events_from_custom_chunk` maps
`{"kind": "context_usage"}` custom chunks to a new `AgentStreamEvent` kind
`context_usage` with a new optional field `context_usage: dict | None`.
`AGENT_STREAM_EVENT_KINDS` gains the kind. The NDJSON renderer forwards it
unchanged. The Chainlit, CLI and TUI renderers ignore it. Because only the
main agent emits stream events, subagent snapshots never reach the stream.

### Registry

`ContextUsageRegistry` is created once per process alongside
`BackgroundTaskManager` and `LiveAgentManager` and exposed as
`AgentRuntime.context_usage`.

- Keyed by session id, then the agent key described in section 1.
- Session id resolution: the middleware reads
  `configurable["chainagents_session_id"]` and falls back to
  `configurable["thread_id"]`. The main agent and sync subagents run on the
  conversation's thread, so the fallback is correct for them. Background
  subagent runs (thread `{session}:background:{task}`) and live agent turns
  (thread `live-{session}-{id}`) set `chainagents_session_id` to the owning
  conversation's session, plus `chainagents_agent_kind`,
  `chainagents_agent_name` and `chainagents_agent_instance`, in their run
  config.
- At most 50 agents per session; the least recently updated entry is dropped.
- `close_conversation(thread_id)` clears the session, so the existing
  `DELETE /api/background-tasks?thread_id=…` cleanup removes it.
- When a background task finishes or a live agent stops, its entry's `status`
  becomes `finished` or `stopped`; the entry stays until the session closes or
  the cap evicts it.
- All methods run on the event loop thread; like the live agent registry, it is
  not locked.

### Endpoint and status

- `GET /api/context-usage?thread_id=…` returns
  `{"agents": [snapshot, …]}`: the main agent first, then the others by
  `at_utc`, newest first. It is protected like other `/api` routes and returns
  an empty list for unknown sessions.
- `GET /api/status` adds `features.context_usage: true`.
- Backends without the route return 404, which SparxUI treats as unsupported.

## 3. SparxUI

### Data

- New types: `ChainAgentsContextUsage` (snapshot) and
  `ChainAgentsContextCategories`. `ChainAgentsStreamEvent` gains
  `context_usage?: ChainAgentsContextUsage`.
- `reduceAgentEvent` handles `context_usage` by recording the snapshot as the
  message's `contextUsage`. `DatabaseMessage` gains
  `contextUsage?: ChainAgentsContextUsage`, persisted in IndexedDB as a
  non-indexed field (no schema version change). Malformed snapshots are
  ignored.
- The gauge's data is the `contextUsage` of the latest assistant message on
  the active branch that has one, including the live streaming state.
- `ChainAgentsService.listContextUsage(threadId, signal)` calls the endpoint;
  a 404 marks the feature unsupported.
- `runtimeStore.supportsContextUsage` is true when the status reports
  `features.context_usage`.

### Gauge

`AgentContextGauge.svelte` (new) in the chat form actions row:

- Reuses `ContextGaugeDial` with `colorLevelFromPercent(used / window)`;
  percent is clamped to 100 for the dial.
- With an unknown window, the dial shows no percent.
- Hidden until the active conversation has a snapshot.
- Opens its popup with the existing `gauge-popup` interaction helpers (click,
  hover, keyboard).

### Popup

`AgentContextPopup.svelte` (new):

1. Header: `used / window tokens · percent`, with "estimated" when
   `exact` is false and a warning tint when `used > window`. With an unknown
   window: `used tokens` and the hint "Set context_window in the model profile
   to see the limit."
2. Stacked bar: system, tools, summary, messages, tool results and free space,
   each with a legend row showing tokens and percent of the window. A tick
   marks `trigger`, with the line "Summarizes at {trigger} · keeps {keep}".
   Without a window the bar is scaled to `used` and shows no free space or tick.
3. "Last summarized {local time}" when `last_summarized_at_utc` is set.
4. Agents section, only when `runtimeStore.supportsContextUsage`: fetched when
   the popup opens and every 3 seconds while it stays open during a streaming
   turn. Each row shows the agent label, model, a small usage bar with its
   percent of that agent's window, the last summarization time, and a muted
   style for `finished`/`stopped` agents. A 404 hides the section; other
   errors show "Couldn't load agents" with a retry button.

Popup data shaping (percentages, free space, tick position, clamping, unknown
window) is a pure function in `src/lib/utils/context-usage.ts`.

### Summarization notice

No new UI: the existing reducer already renders `summarization_status` as an
info status line, which now shows the backend's message.

## 4. Testing

ChainAgents (pytest, fake chat models, no network):

- Middleware: categorization including summary and tool messages; scaling to
  `input_tokens` with exact totals; `exact = false` fallback; threshold
  resolution for profile, global, fraction-default and unknown-window cases;
  summarization detection fires once per new summary; failures inside
  measurement never fail the model call.
- Registry: per-session and per-agent storage, eviction at 50, clearing on
  `close_conversation`, status updates for finished background tasks and
  stopped live agents.
- Stream adapter: `context_usage` events; non-main agents not streamed.
- API: endpoint payload and ordering, 404 when unavailable, status flag.
- End to end: `get_agent` with a subagent on another profile; the main
  agent's snapshots stream and both agents appear in the registry with their
  own windows.

SparxUI (vitest):

- Reducer stores `contextUsage` and history replay restores it.
- Latest-snapshot selection on the active branch.
- `context-usage.ts` shaping: category percentages, free space, trigger tick,
  clamping, unknown window.
- `listContextUsage` request and 404 handling.

Live check: the GLM probe with `context_window = 16384` and a trigger of 8,000
over 10 turns. Verify the NDJSON stream carries one `context_usage` event per
main-agent call with totals matching the server's `input_tokens`, a
`summarization_status` event when summarization runs, and that
`/api/context-usage` lists a subagent with its own window. The user verifies
the gauge and popup visually in a browser.

## 5. Delivery

- ChainAgents: branch `feat/context-usage`, based on
  `feat/profile-context-window`, opened as a PR stacked on #126.
- SparxUI: branch `dev/context-usage-gauge`, its own PR.
- Merge order: ChainAgents first. SparxUI degrades gracefully against older
  backends, so the order is not strictly required.

## Out of scope

- Context usage in Chainlit, the CLI or the TUI.
- Per-tool or per-message breakdowns.
- Persisting the registry across restarts.
- Streaming subagent snapshots.
