# Clarifying Questions Before Delegation — Design

Date: 2026-10-05
Status: Implemented

## Goal

Before delegating work to subagents, the main agent judges whether the user's
request is ambiguous in ways that would change what the subagents do (scope,
sources, output format). If it is, the agent pauses the run, asks the user one
focused question, and resumes the same run with the answer. Clear requests
proceed straight to delegation.

Success criteria:

- The question appears in Chainlit, the CLI REPL, the TUI, and the HTTP API.
- The user's answer resumes the paused run (same graph state, same tool call),
  not a fresh turn.
- Subagents never pause for user input.
- The feature is opt-in and changes nothing when disabled.

Non-goals: subagent-initiated questions, answer timeouts, forced questions on
every request, approve/edit/reject of planned delegations.

## Approach

A main-agent-only `ask_user` tool that calls LangGraph `interrupt()`, plus
prompt guidance on when to use it. The model decides whether a request is
ambiguous. Interrupt detection and resume live in the shared `TurnRunner`, so
every interface gets the same behavior; each renderer only decides how to show
the question.

Rejected alternatives:

- A middleware gate that intercepts the first `task` call each turn: guarantees
  a question but asks on clear requests too, contradicting "only when
  ambiguous".
- LangChain `HumanInTheLoopMiddleware` on `task`: provides approval of a planned
  call, not clarification of the request.

## Configuration

New opt-in section, following the existing `[agent.messaging]` /
`[agent.user_input]` pattern:

```toml
[agent.clarification]
# Let the main agent pause and ask one clarifying question before delegating
# ambiguous work to subagents. Requires agent.state = "stateful".
enabled = false
```

- `ClarificationConfig(enabled: bool = False)` in `chainagents/runtime/types.py`,
  added to `ExtensionsConfig`.
- Parsed with `_normalize_opt_in_config` in
  `chainagents/runtime/extension_config.py`; unknown keys are rejected.
- When `enabled = true` and `agent.state = "stateless"`, the tool is not
  registered and a warning is logged once at startup (resume needs a
  checkpointer).
- Documented in `deepagent.toml` and `README.md`.

## Components

### 1. `ask_user` tool (`chainagents/runtime/clarification.py`, new)

```python
ask_user(question: str, options: list[str] | None = None) -> str
```

- Calls `interrupt({"kind": "clarification", "question": ..., "options": [...]})`
  and returns the resume value (the user's answer) as the tool result text.
- If `current_background_task_id()` is set, returns an error string
  ("ask_user is unavailable in background tasks; proceed with your best
  judgment") instead of interrupting, as a defense in depth.
- Empty `question` returns an error string; `options` is optional, trimmed,
  de-duplicated, capped at 6 entries.

Registration (`chainagents/runtime/graph.py`): the tool is attached through a
`ClarificationMiddleware` appended to the main agent's middleware only.
DeepAgents copies the main agent's `tools` argument into subagents that declare
no tools of their own (including the built-in general-purpose subagent), but
never middleware tools, so attaching the tool through middleware keeps it
main-agent only without rewriting subagent tool lists. Verified by test.

### 2. Prompt guidance

When the tool is registered, one line is appended to the main agent's system
prompt:

> If a request is ambiguous in ways that would change what subagents do
> (scope, sources, output), call `ask_user` once with one focused question
> before delegating. Otherwise proceed without asking.

### 3. Stream event

- New `StreamEventKind` literal `clarification_requested` in
  `chainagents/events/stream.py`.
- The event uses existing `AgentStreamEvent` fields; no dataclass change:
  `source="main-agent"`, `text` = the question,
  `ui_props = {"options": [...], "interrupt_id": "<id>"}`.
- The event is synthesized by the runner (below), not by the stream adapter.

### 4. `TurnRunner` (`chainagents/turns/runner.py`)

- New `TurnStatus` value `"awaiting_input"`.
- After the agent stream ends without error, call `agent.aget_state(config)`.
  If `state.interrupts` contains a clarification interrupt, emit one
  `clarification_requested` event per pending interrupt and finish with status
  `awaiting_input`. With that status the runner skips reflection; generated-file
  collection still runs. Renderers receive the status in `on_complete` and do
  not render a final answer.
- At the start of an agent turn (after native-command resolution), call
  `aget_state(config)`. If a clarification interrupt is pending, the input
  payload becomes `Command(resume=answer)` instead of a new user message. With
  more than one pending interrupt, resume with `{interrupt_id: answer}` for
  each (same answer).
- Native slash commands are resolved before the pending check and do not
  consume the pending question.
- `aget_state` is skipped when the runtime is stateless.

### 5. Renderers

- **Chainlit** (`interfaces/chainlit/renderer.py`, `bridge.py`): show the
  question as an agent message. If options exist, attach one action button per
  option; clicking submits that option as the user's next message. Typing a
  free-text reply also works. On `awaiting_input`, do not send the
  final-response message or the reflection prompt.
- **CLI** (`interfaces/cli/render.py`, `app.py`): print the question and a
  numbered option list. The next prompt on the thread is the answer. A paused
  turn exits 0 with a note to answer with the next prompt on the same thread
  (across processes this needs the Postgres checkpointer). JSON output includes
  `status: "awaiting_input"` and the pending `clarifications`.
- A bare option number (for example `2`) is mapped to that option by the
  runner, so every interface supports it.
- **TUI** (`interfaces/tui/app.py`): show the question and options, set the
  status line to "Waiting for your answer", and re-enable the prompt box. A
  bare number selects an option.
- **HTTP API** (`interfaces/api/app.py`): the event passes through the generic
  `_event_payload`. Only for a paused turn, the stream's `done` payload adds
  `status: "awaiting_input"` and `/api/agent/invoke` adds `status` and
  `clarifications`, so existing payloads are unchanged. `/api/agent/input`
  results include `clarifications`. Posting the next input to the same thread
  resumes it. No new endpoint.

## Data flow

1. User sends a request. The main agent calls `ask_user`.
2. `interrupt()` raises; the root graph suppresses it and checkpoints the
   pending interrupt. The stream ends normally.
3. `TurnRunner` sees the pending interrupt via `aget_state`, emits
   `clarification_requested`, and finishes with `awaiting_input`.
4. The interface shows the question. The user replies.
5. The next turn on that thread finds the pending interrupt and sends
   `Command(resume=answer)`. `ask_user` returns the answer; the agent continues
   and delegates.

## Error handling and edge cases

- In-memory checkpointer: a pending question is lost on restart; the next
  message starts a normal turn. Postgres persists it across restarts.
- Stateless runtime: tool not registered (see Configuration).
- Background tasks: tool returns an error string instead of interrupting.
- An agent error during a resumed turn is handled like any other turn error.
- The user-input controller's queued/steer behavior is unchanged: an
  interrupted turn releases the lock, and the answer arrives as the next
  submitted turn.
- `RepeatedToolResultGuardMiddleware` (on another branch) counts since the last
  human message; one question per request will not approach its limit.

## Testing

- Config parsing: default disabled, `enabled = true`, unknown key rejected.
- Tool: interrupt payload shape; background-task error; empty question error;
  options normalization.
- Registration: main agent has `ask_user` when enabled; no subagent (compiled
  or leaf) has it; absent when disabled or stateless.
- `TurnRunner` end to end with a fake model and in-memory checkpointer: turn 1
  calls `ask_user` → `clarification_requested` event + `awaiting_input`; turn 2
  answer resumes and the tool result equals the answer; a native command while
  pending does not consume it.
- Renderers: CLI prints question/options and JSON status; API stream emits the
  event and `done.status == "awaiting_input"`; Chainlit renderer skips the final
  answer and reflection on `awaiting_input`.
