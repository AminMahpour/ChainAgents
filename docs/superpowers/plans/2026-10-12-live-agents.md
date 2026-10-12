# Live Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the main agent (and the user) spawn long-lived, template-based subagents at runtime, converse with them through the existing message broker, and tear them down.

**Architecture:** A new `chainagents/runtime/live_agents/` package holds config parsing and spawn resolution (pure), a registry, an event bus, a per-agent mailbox worker and a `LiveAgentManager`. `AgentRuntime` owns one manager per process, builds each live agent's graph with the existing deep-agent factory, and tears agents down when a conversation closes. Interfaces (slash commands via `TurnRunner`, HTTP API, Chainlit panel, TUI, CLI) are thin layers over the manager.

**Tech Stack:** Python 3.12, LangChain/LangGraph, deepagents 0.7.24, Chainlit, FastAPI, Textual, pytest + anyio.

**Spec:** `docs/superpowers/specs/2026-10-12-live-agents-design.md`

## Global Constraints

- Branch: `feature/live-agents` (already created; spec committed as `874729d`).
- Feature is off by default: `[agent.live_agents].enabled = false`.
- `enabled = true` requires `[agent.messaging].enabled = true` (startup `ValueError`).
- Defaults: `max_live_per_session = 4`, `max_live_total = 16`, `idle_timeout_seconds = 1800` (`0` disables), `max_turns_per_agent = 50`, `turn_recursion_limit = 100`, `delete_history_on_teardown = true`, `notify_main_on_user_actions = true`, `stream_activity = false`.
- Allowlist: `applies_to = "all"` (default) or `"overrides"`; `allow_custom_prompt = true` default.
- Agent names match `^[a-z0-9][a-z0-9_-]{0,39}$`, unique among a session's non-stopped agents; default `<template>-<n>` or `agent-<n>`.
- Broker address: `live:<name>:<id>`; checkpointer thread: `live-<session_id>-<id>`; `id` = `uuid4().hex[:12]`.
- Three consecutive failed turns → status `failed` and teardown.
- Reserved tool names: `list_live_agent_templates`, `spawn_live_agent`, `list_live_agents`, `teardown_live_agent`.
- Reserved slash commands: `agents`, `spawn`, `msg`, `kill`.
- `[chainlit].live_agents_panel_enabled = true` default.
- No nesting: live agents never receive live-agent or background-task tools.
- Live agents exist only in `AgentRuntime` (Chainlit, TUI, CLI, HTTP API). The exported LangGraph graph (`create_configured_graph`) does not get live-agent tools.
- Async tests use `@pytest.mark.anyio` (no pytest-asyncio, no conftest). Fake models subclass `FakeMessagesListChatModel` with `bind_tools` returning `self`, as in `tests/test_agent_messaging_graph.py:26`.
- Run tests with `uv run pytest <path> -v`. Full suite (`uv run pytest -q`) must stay green after every task.

## Review Focus

1. **Teardown while a turn is mid tool-call:** cancellation must abort the turn, recover inflight messages, close the address, and leave no orphaned asyncio task. Pinned in Task 5 (`test_teardown_mid_turn_cancels_and_marks_undeliverable`).
2. **Message to a just-stopped agent by name:** must fail with "Recipient is unavailable" (the broker already does this once the address is closed) rather than queue forever. Pinned in Task 5 (`test_send_after_teardown_is_rejected`).
3. **Duplicate names after teardown:** a stopped agent's name must be reusable. Pinned in Task 3 (`test_name_reusable_after_stop`).
4. **Conversation close with live agents:** `close_conversation` must stop every live agent before the broker session is dropped. Pinned in Task 7 (`test_close_conversation_tears_down_live_agents`).
5. **Model emits no final text** (only tool calls, or empty content): the reply must still be sent, as `(no reply)`, so the sender is not left waiting. Pinned in Task 4 (`test_empty_final_text_sends_placeholder`).

---

### Task 1: Config types and parsing

**Files:**
- Create: `chainagents/runtime/live_agents/__init__.py`, `chainagents/runtime/live_agents/config.py`
- Modify: `chainagents/runtime/types.py` (new dataclasses; add `live_agents: LiveAgentsConfig` to `ExtensionsConfig` near line 357; include `live_agents.enabled` in its `enabled` property; add `chainlit_live_agents_panel_enabled: bool = True`)
- Modify: `chainagents/runtime/extension_config.py` (`parse_extensions_config` ~line 705: parse `agent_section.get("live_agents")`; `[chainlit]` parsing ~line 829: `live_agents_panel_enabled`; reject `[[chainlit.commands]]` named `agents|spawn|msg|kill`)
- Modify: `chainagents/runtime/config.py:119-128` (after `model_profiles` are parsed, call `validate_live_agent_models`)
- Test: `tests/test_live_agents_config.py`

**Interfaces:**
- Produces (in `types.py`, all `@dataclass(frozen=True)`):
  - `LiveAgentTemplate(name: str, description: str, system_prompt: str, model: str | None = None, tools: tuple[str, ...] = (), skills: tuple[str, ...] = (), mcp_servers: tuple[str, ...] = ())`
  - `LiveAgentAllowlist(applies_to: Literal["all", "overrides"] = "all", models: tuple[str, ...] = (), mcp_servers: tuple[str, ...] = (), skills: tuple[str, ...] = (), tools: tuple[str, ...] = (), allow_custom_prompt: bool = True)`
  - `LiveAgentsConfig(enabled=False, max_live_per_session=4, max_live_total=16, idle_timeout_seconds=1800, max_turns_per_agent=50, turn_recursion_limit=100, delete_history_on_teardown=True, notify_main_on_user_actions=True, stream_activity=False, allow: LiveAgentAllowlist = LiveAgentAllowlist(), templates: tuple[LiveAgentTemplate, ...] = ())`
- Produces (in `live_agents/config.py`):
  - `LIVE_AGENT_COMMAND_NAMES: frozenset[str] = frozenset({"agents", "spawn", "msg", "kill"})`
  - `live_agent_filesystem_tools(*, delete_tool_enabled: bool, execute_tool_enabled: bool) -> tuple[str, ...]` — `DEFAULT_DEEPAGENT_FILESYSTEM_TOOLS` plus `delete`/`execute` when enabled.
  - `parse_live_agents_config(value: Any, *, base_dir: Path, mcp_servers: Mapping[str, Any], messaging_enabled: bool, delete_tool_enabled: bool, execute_tool_enabled: bool) -> LiveAgentsConfig`
  - `validate_live_agent_models(config: LiveAgentsConfig, model_profiles: Mapping[str, Any]) -> None`
- Skill paths go through `normalize_skill_source_path(path, base_dir)` (`extension_config.py:274`) in both `allow.skills` and template `skills`, so comparisons use normalized values. Prompt files resolve with `resolve_local_path` as in `parse_sync_subagent_config`.

- [ ] **Step 1: Write failing tests** in `tests/test_live_agents_config.py`:
  - `test_defaults_when_section_missing`: `parse_live_agents_config(None, ...)` equals `LiveAgentsConfig()`.
  - `test_enabled_requires_messaging`: `{"enabled": True}` with `messaging_enabled=False` raises `ValueError` matching `"agent.messaging"`.
  - `test_rejects_unknown_fields`: `{"bogus": 1}` raises matching `"Unknown 'agent.live_agents'"`.
  - `test_limits_must_be_positive_ints` (parametrized over the five limit fields with `0`, `-1`, `"4"`); `idle_timeout_seconds = 0` is accepted.
  - `test_applies_to_must_be_all_or_overrides`.
  - `test_template_requires_name_and_description`, `test_template_names_unique`, `test_template_prompt_and_prompt_file_exclusive`, `test_template_prompt_file_loaded` (tmp_path file content becomes `system_prompt`).
  - `test_unknown_mcp_server_rejected` (in allow and in template), `test_unknown_tool_rejected` (`"delete"` rejected when `delete_tool_enabled=False`, accepted when `True`).
  - `test_template_outside_allowlist_rejected_when_applies_to_all` and `test_template_outside_allowlist_accepted_when_overrides` (template tool `write_file` not in `allow.tools`).
  - `test_validate_models_rejects_unknown_profile` (`allow.models=("nope",)`, `model_profiles={}`) and template `model` likewise; `"all"` mode rejects a template model missing from `allow.models`.
  - `test_reserved_chainlit_command_rejected`: `parse_extensions_config` with `[[chainlit.commands]] name = "kill"` raises.
  - `test_extensions_config_exposes_live_agents_and_panel_flag`.
- [ ] **Step 2:** Run `uv run pytest tests/test_live_agents_config.py -v` — expect import errors / failures.
- [ ] **Step 3:** Implement the dataclasses and functions above. Error messages name the `agent.live_agents.<field>` path, matching the style of `_normalize_opt_in_config` (`extension_config.py:35`).
- [ ] **Step 4:** Run the test file — PASS. Run `uv run pytest -q` — all green.
- [ ] **Step 5:** Commit `feat(live-agents): parse [agent.live_agents] config`.

### Task 2: Spawn resolution

**Files:**
- Modify: `chainagents/runtime/live_agents/config.py`
- Test: `tests/test_live_agents_resolve.py`

**Interfaces:**
- Consumes: Task 1 dataclasses.
- Produces:
  - `@dataclass(frozen=True) class SpawnRequest(template: str | None = None, name: str | None = None, instructions: str | None = None, model: str | None = None, add_tools: tuple[str, ...] = (), remove_tools: tuple[str, ...] = (), add_skills: tuple[str, ...] = (), add_mcp_servers: tuple[str, ...] = (), initial_message: str | None = None)`
  - `@dataclass(frozen=True) class ResolvedLiveAgentSpec(name: str, template: str | None, system_prompt: str, model: str | None, tools: tuple[str, ...], skills: tuple[str, ...], mcp_servers: tuple[str, ...])`
  - `class LiveAgentSpawnError(ValueError)`
  - `resolve_spawn(request: SpawnRequest, config: LiveAgentsConfig, *, existing_names: Collection[str], sequence: int) -> ResolvedLiveAgentSpec`
  - `SUPERVISOR_TASK_HEADING = "## Task from supervisor"`

- [ ] **Step 1: Write failing tests:**
  - `test_template_only_copies_template_fields`.
  - `test_add_tool_in_allowlist` / `test_add_tool_outside_allowlist_lists_permitted` (`str(err)` contains the rejected tool and every allowed tool).
  - `test_remove_tool_always_allowed` (even a tool not in the template: no error, absent from result).
  - `test_model_override_must_be_allowed`.
  - `test_add_skill_and_mcp_server_checked` (skills normalized like config).
  - `test_instructions_appended_under_heading`: `system_prompt == f"{template_prompt}\n\n## Task from supervisor\n{instructions}"`.
  - `test_templateless_uses_instructions_as_prompt` (tools `()`, model `None`).
  - `test_templateless_requires_instructions` and `test_templateless_rejected_when_custom_prompt_disallowed`.
  - `test_instructions_with_template_accepted_when_custom_prompt_disallowed`.
  - `test_unknown_template_lists_templates`.
  - `test_default_name_uses_template_and_sequence` (`researcher-3`), `test_default_name_without_template` (`agent-3`).
  - `test_invalid_name_rejected` (`"Bad Name"`, `"-x"`, 41 chars) and `test_duplicate_name_rejected` (`existing_names={"r1"}`).
  - Result tuples keep template order then additions, de-duplicated.
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Implement per spec §1 "Spawn resolution". Allowlist checks for additions apply in both `applies_to` modes.
- [ ] **Step 4:** Run — PASS; full suite green.
- [ ] **Step 5:** Commit `feat(live-agents): resolve spawn requests against templates and allowlist`.

### Task 3: Registry and event bus

**Files:**
- Create: `chainagents/runtime/live_agents/registry.py`, `chainagents/runtime/live_agents/events.py`
- Test: `tests/test_live_agents_registry.py`

**Interfaces:**
- Produces (`registry.py`):
  - `LiveAgentStatus = Literal["idle", "running", "stopping", "stopped", "failed"]`
  - `@dataclass class LiveAgentRecord(id: str, session_id: str, name: str, template: str | None, model: str | None, address: str, thread_id: str, status: LiveAgentStatus, created_at: float, last_active_at: float, spawned_by: str, turns: int = 0, consecutive_failures: int = 0, stop_reason: str | None = None, parent_trace: Any = None)` with `to_payload(self, *, pending: int = 0) -> dict[str, object]` (keys: `id, name, template, model, status, turns, created_at, last_active_at, pending, stop_reason`).
  - `class LiveAgentLimitError(ValueError)`
  - `class LiveAgentRegistry(config: LiveAgentsConfig, *, clock: Callable[[], float] = time.monotonic)`:
    - `reserve(self, session_id: str, spec: ResolvedLiveAgentSpec, *, spawned_by: str) -> LiveAgentRecord` — checks `max_live_per_session`/`max_live_total` over non-stopped records, generates `id`, `address`, `thread_id` per Global Constraints, status `idle`.
    - `discard(self, record: LiveAgentRecord) -> None` — removes a reserved record entirely (failed spawn).
    - `lookup(self, session_id: str, ref: str) -> LiveAgentRecord` — non-stopped name match first, then id (any status); `KeyError` otherwise.
    - `records(self, session_id: str, *, include_stopped: bool = False) -> list[LiveAgentRecord]` (creation order).
    - `live_names(self, session_id: str) -> set[str]`, `next_sequence(self, session_id: str) -> int` (1-based count of all records ever reserved in the session).
    - `forget_session(self, session_id: str) -> None`.
- Produces (`events.py`):
  - `LiveAgentEventKind = Literal["spawned", "turn_started", "turn_finished", "message", "stopped"]`
  - `@dataclass(frozen=True) class LiveAgentEvent(kind: LiveAgentEventKind, session_id: str, agent_id: str, agent_name: str, payload: Mapping[str, object])`
  - `class LiveAgentEventBus`: `subscribe(session_id: str) -> asyncio.Queue[LiveAgentEvent]`, `unsubscribe(session_id: str, queue) -> None`, `publish(event: LiveAgentEvent) -> None` (non-blocking `put_nowait`; thread-safe via `loop.call_soon_threadsafe` like `MessageBroker.send`).

- [ ] **Step 1: Write failing tests:** `test_reserve_enforces_session_limit`, `test_reserve_enforces_total_limit`, `test_lookup_by_name_then_id`, `test_name_reusable_after_stop` (record set `stopped`; `live_names` excludes it; lookup by name raises `KeyError`, by id returns it), `test_next_sequence_counts_all`, `test_payload_shape`, `test_event_bus_delivers_to_session_subscribers_only` (anyio).
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS; full suite green.
- [ ] **Step 5:** Commit `feat(live-agents): add registry and event bus`.

### Task 4: Mailbox worker

**Files:**
- Create: `chainagents/runtime/live_agents/worker.py`
- Test: `tests/test_live_agents_worker.py`

**Interfaces:**
- Consumes: `MessageBroker` (`messaging.py:72`: `wait`, `claim`, `acknowledge`, `recover`, `send`), contextvars `_CURRENT_AGENT_ADDRESS` / `_CURRENT_MESSAGING_SESSION` (`messaging.py:23-26`), Task 3 types.
- Produces:
  - `NO_REPLY_TEXT = "(no reply)"`; `MAX_CONSECUTIVE_FAILURES = 3`
  - `class LiveAgentWorker(record: LiveAgentRecord, graph: Any, broker: MessageBroker, events: LiveAgentEventBus, *, recursion_limit: int, max_turns: int, clock: Callable[[], float], turn_context: Callable[[LiveAgentRecord], AbstractContextManager[Any]] | None = None, on_finished: Callable[[LiveAgentRecord, str], Awaitable[None]])`
    - `async run(self) -> None` — loop: `await broker.wait(session, address, 30)`; `claim`; if empty continue; run a turn; `acknowledge`. Calls `on_finished(record, reason)` with `"max_turns"` after the reply of turn `max_turns`, or `"failed"` after the third consecutive failure. On `CancelledError` calls `broker.recover` and re-raises.
    - `async run_turn(self, messages: list[AgentMessage]) -> str` — sets the two contextvars to `record.address` / `record.session_id`, wraps in `turn_context(record)` when given, invokes `graph.ainvoke({"messages": [HumanMessage(content=f"Message from {m.sender} (id {m.id}):\n{m.body}") ...]}, {"configurable": {"thread_id": record.thread_id}, "recursion_limit": recursion_limit})`, returns the text of the last `AIMessage` (string content, or joined text blocks), or `NO_REPLY_TEXT` if empty.
  - Reply routing: for each distinct `sender` of the batch, in first-seen order: if `sender == "user"`, publish a `message` event (`{"from": name, "to": "user", "body": text}`) instead of `broker.send`; otherwise `broker.send(session, record.address, sender, text)` and publish a `message` event. On failure the body is `f"[error] {type(exc).__name__}: {exc}"` (truncated to the broker's `max_message_chars`).
  - Status transitions and events: `running` + `turn_started` before invoking; `idle` + `turn_finished` (`{"turns": n, "ok": bool}`) after; updates `last_active_at`, `turns`, `consecutive_failures`.
  - The live agent graph has **no** `AgentMessageMiddleware`: the worker owns the inbox, so messages arriving mid-turn stay pending for the next batch.

- [ ] **Step 1: Write failing tests** (anyio; a scripted fake graph object exposing `async ainvoke(input, config)` that records calls and returns `{"messages": [AIMessage(...)]}`; a real `MessageBroker(MessagingConfig(enabled=True))` with `main` and the agent address opened):
  - `test_message_triggers_one_turn_and_reply_reaches_sender`.
  - `test_thread_id_is_stable_across_turns` (both calls use `record.thread_id`).
  - `test_mid_turn_messages_form_next_batch` (fake graph blocks on an `asyncio.Event`; a second message sent meanwhile is in the second call's input, not the first).
  - `test_user_sender_reply_published_not_sent`.
  - `test_multiple_senders_each_get_reply`.
  - `test_failure_sends_error_and_stays_idle`.
  - `test_three_failures_call_on_finished_failed`.
  - `test_max_turns_calls_on_finished_after_reply`.
  - `test_empty_final_text_sends_placeholder` (Review Focus 5).
  - `test_contextvars_set_during_turn` (fake graph reads `_CURRENT_AGENT_ADDRESS.get()`).
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS; full suite green.
- [ ] **Step 5:** Commit `feat(live-agents): add mailbox worker`.

### Task 5: Manager

**Files:**
- Create: `chainagents/runtime/live_agents/manager.py`
- Test: `tests/test_live_agents_manager.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `LiveAgentGraphFactory = Callable[[ResolvedLiveAgentSpec, str], Awaitable[Any]]` — `(spec, session_id) -> compiled graph`.
  - `class LiveAgentManager(config: LiveAgentsConfig, broker: MessageBroker, graph_factory: LiveAgentGraphFactory, *, events: LiveAgentEventBus | None = None, clock: Callable[[], float] = time.monotonic, delete_thread: Callable[[str], Awaitable[None]] | None = None, turn_context: Callable[[LiveAgentRecord], AbstractContextManager[Any]] | None = None)`
    - `config` and `events` attributes (public).
    - `async spawn(self, session_id: str, request: SpawnRequest, *, spawned_by: str = "main", parent_trace: Any = None) -> tuple[LiveAgentRecord, ResolvedLiveAgentSpec]` — resolve (Task 2) → `registry.reserve` → `await graph_factory(spec, session_id)` (on exception: `registry.discard`, re-raise) → `broker.open(session, "main", name="main", parent=None)` and `broker.open(session, record.address, name=record.name, parent="main")` → start worker task named `chainagents-live-agent-{id}` → publish `spawned` → if `initial_message`: `broker.send(session, spawned_by_address, record.address, text)` where `spawned_by_address` is `"main"`, or for `"user"` a direct enqueue via a user-originated send (open `"user"` temporarily as `MessageBroker.send_user` does, then close it).
    - `list(self, session_id: str, *, include_stopped: bool = False) -> list[dict[str, object]]` — payloads with `pending=len(broker.pending(...))` for open addresses.
    - `get(self, session_id: str, ref: str) -> LiveAgentRecord`
    - `send_from_user(self, session_id: str, ref: str, body: str) -> AgentMessage` — user-originated send as above; publishes a `message` event (`{"from": "user", "to": name, "body": body}`).
    - `async teardown(self, session_id: str, ref: str, *, reason: str = "requested") -> LiveAgentRecord` — status `stopping`; cancel and await the worker task; `broker.close(session, address)`; `await delete_thread(thread_id)` when configured and enabled (errors logged, not raised); status `stopped` (or keep `failed` when reason is `"failed"`); publish `stopped` (`{"reason": reason}`). Idempotent for already-stopped records.
    - `async teardown_session(self, session_id: str) -> None` — tear down every non-stopped record, then `registry.forget_session`.
    - `async reap_idle(self) -> list[LiveAgentRecord]` — tears down `idle` records whose `clock() - last_active_at >= idle_timeout_seconds` (no-op when `0`), reason `"idle_timeout"`.
    - `start(self) -> None` / `async close(self) -> None` — background reaper task every `min(30, idle_timeout_seconds)` seconds; `close` cancels it and tears down all sessions.
    - `templates_payload(self) -> dict[str, object]` — `{"templates": [...], "allow": {...}}`.
    - `roster(self, session_id: str) -> str` — `""` when none live, else `"Live agents: researcher-1: idle, coder-2: running"`.
    - Worker `on_finished` callback → `teardown(..., reason=reason)` scheduled as a separate task (the worker must not cancel itself while awaiting).

- [ ] **Step 1: Write failing tests** (anyio; graph factory returns the scripted fake graph from Task 4; fake clock):
  - `test_spawn_opens_address_and_starts_idle`.
  - `test_spawn_initial_message_runs_first_turn`.
  - `test_spawn_graph_failure_releases_slot_and_name`.
  - `test_spawn_beyond_session_limit_raises_limit_error`.
  - `test_teardown_mid_turn_cancels_and_marks_undeliverable` (Review Focus 1: a pending message to the agent ends `undeliverable`; worker task is done; no tasks named `chainagents-live-agent-*` remain).
  - `test_send_after_teardown_is_rejected` (Review Focus 2: `broker.send(..., "main", name, ...)` raises `ValueError` "Recipient is unavailable").
  - `test_teardown_deletes_history_when_configured` and `test_teardown_keeps_history_when_disabled`.
  - `test_teardown_session_stops_all`.
  - `test_reap_idle_uses_timeout` (advance fake clock; `0` disables).
  - `test_max_turns_tears_down_with_reason`.
  - `test_send_from_user_reply_published_as_event`.
  - `test_roster_lists_live_agents`.
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS; full suite green.
- [ ] **Step 5:** Commit `feat(live-agents): add LiveAgentManager`.

### Task 6: Live agent graph construction

**Files:**
- Modify: `chainagents/runtime/middleware.py:660` (`build_agent_middleware` gains `filesystem_tools: Sequence[str] | None = None`; when given, it replaces the computed list passed to `FilesystemMiddleware`)
- Modify: `chainagents/runtime/graph.py` (new `build_live_agent_graph`)
- Test: `tests/test_live_agents_graph.py`

**Interfaces:**
- Produces: `build_live_agent_graph(config: RuntimeConfig, *, spec: ResolvedLiveAgentSpec, model: Any, model_name: str | None, reasoning_level: ReasoningLevel, mcp_tools: list[Any], backend: Any, project_root: Path, broker: MessageBroker, session_id: str, store: Any | None = None, checkpointer: Any | None = None) -> Any`
  - Middleware: `build_agent_middleware(backend=backend, config=config, reasoning_level=reasoning_level, model_name=model_name, source=f"live:{spec.name}", project_root=project_root, filesystem_tools=spec.tools)`.
  - Tools: `sanitize_tools_for_model(mcp_tools, ...)` (as `_build_sync_subagent_spec` does) + `create_agent_message_tools(broker, fixed_session_id=session_id)`; call `validate_agent_message_tool_names(mcp_tools)` first.
  - Graph: `runtime_middleware.create_deep_agent_with_configured_summarization(config, model=model, tools=tools, system_prompt=spec.system_prompt, middleware=middleware, backend=backend, skills=list(spec.skills) or None, subagents=[], store=..., checkpointer=...)` (store/checkpointer only when not `None`, as at `graph.py:608-624`).
  - No `AgentMessageMiddleware`, no background task tools, no live-agent tools.
- If deepagents' own stack adds filesystem tools regardless of `FilesystemMiddleware(tools=...)`, the test in Step 1 will expose it; then restrict via deepagents' tool-exclusion middleware (`deepagents/middleware/_tool_exclusion.py`) instead and keep the same function signature.

- [ ] **Step 1: Write failing tests** (pattern from `tests/test_agent_messaging_graph.py::test_opted_in_leaf_subagent_runs_as_compiled_participant`; capture the tool names the fake model receives through `bind_tools`):
  - `test_live_graph_exposes_only_resolved_filesystem_tools` (spec tools `("read_file", "ls")` → bound tool names include those, exclude `write_file`, `edit_file`, `glob`, `grep`).
  - `test_live_graph_has_messaging_tools_and_no_management_tools` (includes `send_agent_message`; excludes every `BACKGROUND_TASK_TOOL_NAMES` and `LIVE_AGENT_TOOL_NAMES` entry — the latter constant is created in Task 7, so import it from `chainagents.runtime.live_agents.tools` once Task 7 lands; in this task assert on the literal names).
  - `test_live_graph_uses_spec_system_prompt`.
  - `test_build_agent_middleware_default_tools_unchanged` (no `filesystem_tools` → same list as before).
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS; full suite green.
- [ ] **Step 5:** Commit `feat(live-agents): build restricted live agent graphs`.

### Task 7: Main-agent tools, prompt, and runtime wiring

**Files:**
- Create: `chainagents/runtime/live_agents/tools.py`
- Modify: `chainagents/runtime/graph.py:846` (`build_agent_kwargs` gains `live_agent_manager: LiveAgentManager | None = None`; when set: validate names, append tools, append `LIVE_AGENTS_PROMPT` to the system prompt, add `LiveAgentRosterMiddleware`)
- Modify: `chainagents/runtime/lifecycle.py` (create manager after the broker at ~line 139 when `config.extensions.live_agents.enabled`; `push_async_callback(manager.close)`; `manager.start()` on initialize; add `_build_live_agent_graph`; pass `live_agent_manager=` at ~line 673-702; in `_close_conversation` ~line 990 `await self.live_agents.teardown_session(thread_id)` before `message_broker.close_session`; expose `AgentRuntime.live_agents: LiveAgentManager | None`)
- Test: `tests/test_live_agents_tools.py`, `tests/test_live_agents_runtime.py`

**Interfaces:**
- Produces (`tools.py`):
  - `LIVE_AGENT_TOOL_NAMES = frozenset({"list_live_agent_templates", "spawn_live_agent", "list_live_agents", "teardown_live_agent"})`
  - `validate_live_agent_tool_names(existing_tools: Iterable[object]) -> None` (same matching as `validate_agent_message_tool_names`).
  - `create_live_agent_tools(manager: LiveAgentManager, *, fixed_session_id: str | None = None, langsmith_tracing: LangSmithTracing | None = None) -> list[Any]` — four `@tool` functions using `ToolRuntime`; session id via `messaging._session_id(runtime, fixed_session_id)`. `spawn_live_agent` args: `template: str | None = None, name: str | None = None, instructions: str | None = None, model: str | None = None, add_tools: list[str] | None = None, remove_tools: list[str] | None = None, add_skills: list[str] | None = None, add_mcp_servers: list[str] | None = None, initial_message: str | None = None`; captures `langsmith_tracing.capture_parent(runtime.config)` as `parent_trace`; returns `{"id", "name", "address", "status", "resolved": {"model", "tools", "skills", "mcp_servers"}}`; `LiveAgentSpawnError`/`LiveAgentLimitError` return `{"error": str(exc)}`. `teardown_live_agent(agent: str, reason: str | None = None)` returns the record payload; unknown agent returns `{"error": ...}`.
  - `LIVE_AGENTS_PROMPT: str` — explains: live agents persist across turns; message them with `send_agent_message` by name; replies arrive asynchronously as messages on a later step or via `wait_for_agent_messages`; call `list_live_agent_templates` before spawning; tear agents down when their work is done.
  - `class LiveAgentRosterMiddleware(AgentMiddleware)` — `wrap_model_call`/`awrap_model_call` append `manager.roster(session)` to the system message when non-empty (use `append_to_system_message` from `deepagents.middleware._utils`, as `async_subagents.py` does).
- Produces (`lifecycle.py`): `async _build_live_agent_graph(self, spec: ResolvedLiveAgentSpec, session_id: str) -> Any` — model profile via `resolve_runtime_model_profile(self.config, spec.model)`, model via `self._build_model(level, model_profile=profile)` with the profile's default reasoning level, MCP tools via `self._get_mcp_tools(spec.mcp_servers, thread_id=session_id, mcp_session_id=session_id)`, backend built the same way `build_agent_kwargs` builds it, `checkpointer`/`store` only when stateful, then `build_live_agent_graph(...)`. `delete_thread` = checkpointer's `adelete_thread` when present. `turn_context` = `langsmith_tracing.background_scope(record.parent_trace, mode=config.langsmith.background_trace_mode)` when tracing is configured.

- [ ] **Step 1: Write failing tests:**
  - `tests/test_live_agents_tools.py`: `test_spawn_tool_returns_resolved_spec`, `test_spawn_tool_returns_error_for_disallowed_tool`, `test_list_and_teardown_tools`, `test_reserved_tool_name_collision_raises` (parametrized over the four names), `test_roster_middleware_injects_roster_only_when_live`.
  - `tests/test_live_agents_runtime.py`:
    - `test_tools_absent_when_disabled` (`build_agent_kwargs` without manager → none of the four names).
    - `test_end_to_end_spawn_message_reply_teardown`: main graph built via `build_agent_kwargs(..., build_model=..., session_id="s", messaging_broker=broker, live_agent_manager=manager)` with a scripted main model that calls `spawn_live_agent` then `send_agent_message`; the manager's graph factory returns a scripted live graph replying `"pong"`; after `await` on the worker reply, a second main `ainvoke` shows `Message from researcher-1` … `pong` in the main model's input; then `teardown_live_agent` stops it.
    - `test_close_conversation_tears_down_live_agents` (Review Focus 4: `AgentRuntime` with live agents enabled and a stub graph factory; `close_conversation(thread_id="s")` leaves every record `stopped` and the broker session empty).
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS; full suite green (in particular `tests/test_agent_assembly.py` static-vs-live equivalence must still pass with the feature disabled).
- [ ] **Step 5:** Commit `feat(live-agents): wire manager and tools into the main agent`.

### Task 8: Slash commands

**Files:**
- Create: `chainagents/runtime/live_agents/commands.py`
- Modify: `chainagents/runtime/live_agents/manager.py` (FYI forwarding of user-directed replies to `main`)
- Modify: `chainagents/commands/native.py:23` (`RuntimeCommandResult.target` adds `"live_agent"`) and `:130` (`resolve_runtime_command`: when `parsed.command_name in LIVE_AGENT_COMMAND_NAMES` and `getattr(runtime, "live_agents", None)` is set, return `RuntimeCommandResult(target="live_agent", command_name=..., tool_result=text)` before the catalog lookup)
- Modify: `chainagents/turns/runner.py:283` (treat `"live_agent"` like `"mcp_tool"` → `_finish_command_output`)
- Modify: `chainagents/runtime/commands.py:318-353` (skill-derived commands named in `LIVE_AGENT_COMMAND_NAMES` are hidden with a note when live agents are enabled, like existing collisions)
- Modify: `chainagents/interfaces/chainlit/app.py:693` (`build_native_command_specs` adds the four commands with descriptions when enabled)
- Test: `tests/test_live_agents_commands.py`

**Interfaces:**
- Produces: `async handle_live_agent_command(manager: LiveAgentManager, *, command_name: str, raw_args: str, session_id: str) -> str` — returns Markdown:
  - `agents` → table `| name | template | status | turns | idle |`, or `No live agents.`
  - `spawn <template> [name] [-- instructions]` → `Spawned \`name\` (template, model).` Parse: split on the first ` -- `; the left side is `template [name]`.
  - `msg <agent> <text>` → `Sent to \`agent\`. The reply will appear when ready.`
  - `kill <agent>` / `kill all` → `Stopped \`agent\`.` / `Stopped N agents.`
  - Usage errors and `KeyError`/`ValueError` from the manager raise `ValueError` with a usage line (the runner maps it to a 422 command error).
  - When `manager.config.notify_main_on_user_actions`: after `spawn`, `kill` and `msg`, send a FYI to `main` with the broker's user-originated send (e.g. `"FYI: user spawned \`coder-2\` (template coder)."`). For `msg`, the reply FYI is sent by a manager hook when the worker publishes the `to="user"` reply: add `notify_main: bool` handling in `LiveAgentManager` (subscribe internally to its own `message` events with `to == "user"` and forward `"FYI: \`name\` replied to user: <body>"` to `main`).

- [ ] **Step 1: Write failing tests:** `test_agents_table`, `test_agents_empty`, `test_spawn_parses_name_and_instructions`, `test_spawn_usage_error`, `test_msg_routes_to_agent_as_user`, `test_kill_one_and_all`, `test_unknown_agent_is_value_error`, `test_fyi_sent_to_main_when_enabled`, `test_no_fyi_when_disabled`, `test_resolve_runtime_command_returns_live_agent_target`, `test_runner_renders_live_agent_command_without_running_agent` (fake renderer records `on_command_result`; the agent is never built), `test_skill_command_named_kill_hidden_with_note`.
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS; full suite green.
- [ ] **Step 5:** Commit `feat(live-agents): add /agents, /spawn, /msg and /kill commands`.

### Task 9: HTTP API

**Files:**
- Modify: `chainagents/interfaces/api/app.py` (routes inside `create_app`, near the background-task routes at ~538-600)
- Test: `tests/test_live_agents_api.py`

**Interfaces:**
- Routes (404 when live agents are disabled or the agent is unknown; 422 for `ValueError`):
  - `GET /api/live-agents?thread_id=...&include_stopped=false` → `{"agents": manager.list(...)}`
  - `POST /api/live-agents/{agent}/messages?thread_id=...` body `{"body": str}` → `{"message_id": id}` via `send_from_user`
  - `DELETE /api/live-agents/{agent}?thread_id=...` → record payload via `teardown(..., reason="user")`
  - Paths follow the existing `/api/background-tasks?thread_id=` convention rather than the spec's illustrative `/sessions/{id}/…`.

- [ ] **Step 1: Write failing tests** with `TestClient` and a fake runtime exposing `live_agents` (pattern: `tests/test_chainagents_api.py:205`): list, message, delete, 404 when disabled, 404 unknown agent, 422 on empty body.
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS; full suite green.
- [ ] **Step 5:** Commit `feat(live-agents): expose live agents over the HTTP API`.

### Task 10: Chainlit panel

**Files:**
- Create: `chainagents/interfaces/chainlit/live_agents.py`, `public/elements/LiveAgentsPanel.jsx`
- Modify: `chainagents/interfaces/chainlit/app.py` (`on_chat_start`/`on_chat_resume`: start the panel listener when the runtime has live agents and `chainlit_live_agents_panel_enabled`; stop it in `on_chat_end`; action callbacks for panel buttons)
- Test: `tests/test_chainlit_live_agents_panel.py`

**Interfaces:**
- Produces (`live_agents.py`):
  - `build_panel_props(agents: list[dict[str, object]], threads: Mapping[str, list[dict[str, object]]]) -> dict[str, object]` — pure; `{"agents": [{...payload, "thread": [{"from", "to", "body", "at"}...]}]}`; threads capped at the last 50 entries per agent.
  - `class LiveAgentsPanel(manager: LiveAgentManager, session_id: str, *, min_interval: float = 0.25, stream_activity: bool = False)` — `async start()`, `async aclose()`; subscribes to `manager.events`; keeps per-agent thread history from `message` events; on any event, re-renders (throttled to `min_interval`); first render creates `cl.CustomElement(name="LiveAgentsPanel", props=..., display="side")` and opens it with `cl.ElementSidebar.set_elements([element])` + `cl.ElementSidebar.set_title("Live agents")`; later renders assign `element.props` and `await element.update()`.
  - Panel buttons call the JSX global `callAction({name: "live_agent_message", payload: {agent, body}})` and `callAction({name: "live_agent_kill", payload: {agent}})`; `app.py` registers `@cl.action_callback` handlers that call `manager.send_from_user` / `manager.teardown(..., reason="user")`.
  - When `stream_activity` is true, `turn_started`/`turn_finished` events render as `cl.Step(name=f"live:{name}", type="run")` entries.
- JSX (`LiveAgentsPanel.jsx`): `export default function LiveAgentsPanel()` reading global `props.agents`; same no-import, Tailwind/shadcn class conventions as `public/elements/GeneratedPanel.jsx`; card per agent with name, template, model, status badge, turns, scrollable thread, an inline message input + Send button, and a Shut down button guarded by `window.confirm`.

- [ ] **Step 1: Write failing tests:** `test_build_panel_props_merges_threads`, `test_build_panel_props_caps_thread_length`, `test_panel_throttles_renders` (fake clock / patched `cl.CustomElement` and `cl.ElementSidebar`; three events within 0.25s → at most two renders), `test_action_callbacks_call_manager`.
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS; full suite green.
- [ ] **Step 5:** Manual check: run `uv run chainlit run main.py -w` with live agents and messaging enabled and a template configured; ask the agent to spawn one; confirm the sidebar opens and updates, Message and Shut down work. Record what was checked in the commit message body.
- [ ] **Step 6:** Commit `feat(live-agents): add Chainlit live agents panel`.

### Task 11: TUI notices and CLI teardown

**Files:**
- Modify: `chainagents/interfaces/tui/app.py` (subscribe to `runtime.live_agents.events` for the session alongside `_watch_background_tasks` at ~249; show one-line notices for `message` events with `to == "user"` and for `stopped`; unsubscribe in `on_unmount`)
- Modify: `chainagents/interfaces/cli/app.py:595-619` (one-shot path: after the run, `await runtime.live_agents.teardown_session(thread_id)` when enabled; REPL tears down on exit)
- Test: extend `tests/test_chainagents_tui.py` and `tests/test_chainagents_cli.py` (or the existing TUI/CLI test files — use the file that already tests `_watch_background_tasks` / `wait_session`)

- [ ] **Step 1: Write failing tests:** `test_tui_shows_live_agent_reply_notice`, `test_tui_shows_live_agent_stopped_notice`, `test_cli_one_shot_tears_down_live_agents`.
- [ ] **Step 2:** Run — FAIL. **Step 3:** Implement. **Step 4:** Run — PASS; full suite green.
- [ ] **Step 5:** Commit `feat(live-agents): TUI notices and CLI teardown`.

### Task 12: Documentation

**Files:**
- Modify: `README.md` (new `### Live Agents` section after `### Local Background Subagents` ending ~line 1160: config reference, templates/allowlist and `applies_to`, tools, slash commands, panel, API routes, limits and teardown triggers; add `/agents`, `/spawn`, `/msg`, `/kill` to `## Chainlit Native Commands`; add an `[agent.live_agents]` bullet near line 958)
- Modify: `deepagent.toml.example` (commented `[agent.live_agents]`, `[agent.live_agents.allow]`, one `[[agent.live_agents.templates]]`, plus `[agent.messaging]` since it is required; `live_agents_panel_enabled` under `[chainlit]`)
- Modify: `deepagent.toml` (add the disabled `[agent.live_agents]` block with comments, matching the other opt-in sections)
- Modify: `docs/configuration.md`, `docs/interfaces.md`, `docs/architecture.md` (short sections pointing to the README)
- Create: `prompts/live/researcher.md` (example template prompt referenced by the example config)

- [ ] **Step 1:** Write the docs.
- [ ] **Step 2:** Verify the example parses: `uv run python -c "from chainagents.runtime.config import load_file_config; load_file_config('deepagent.toml.example')"` with the live agents block uncommented in a temp copy — no error. Build docs: `uv run sphinx-build -b html docs docs/_build/html` — no new warnings.
- [ ] **Step 3:** Run `uv run pytest -q` — all green.
- [ ] **Step 4:** Commit `docs(live-agents): document configuration, commands and panel`.
