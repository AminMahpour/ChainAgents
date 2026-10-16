# Context Window Usage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report each agent's context-window usage, breakdown and summarization from ChainAgents, and show it in SparxUI as a gauge with a breakdown popup and a per-agent list.

**Architecture:** A `ContextUsageMiddleware` in every ChainAgents agent stack measures each model call, stores a snapshot per agent in a process-local registry, and streams the main agent's snapshots as `context_usage` NDJSON events. `GET /api/context-usage` serves every agent's latest snapshot. SparxUI stores the main agent's snapshot on the assistant message and renders it with the reused `ContextGaugeDial`.

**Tech Stack:** Python 3.12, LangChain/LangGraph middleware, DeepAgents 0.7, FastAPI, pytest (ChainAgents); Svelte 5, TypeScript, Dexie, vitest (SparxUI).

**Spec:** `docs/superpowers/specs/2026-10-15-context-usage-design.md`

## Global Constraints

- ChainAgents work happens on branch `feat/context-usage` (based on `feat/profile-context-window`, PR #126); SparxUI work on a new branch `dev/context-usage-gauge` from `master`.
- Categories are exactly `system`, `tools`, `summary`, `messages`, `tool_results`.
- Run-config keys are exactly `chainagents_session_id`, `chainagents_agent_kind`, `chainagents_agent_name`, `chainagents_agent_instance`.
- `kind` values: `main`, `subagent`, `background`, `live`. `status` values: `active`, `finished`, `stopped`.
- Stream event kind and payload field are both `context_usage`; the custom-writer chunk is `{"kind": "context_usage", "snapshot": {...}}`.
- Registry cap: 50 agents per session, evicting the least recently updated.
- Default thresholds when only the window is known: trigger `floor(0.85 × window)`, keep `floor(0.10 × window)`. No window and nothing configured: trigger `170000`, keep `null`.
- Summarization notice text: `Conversation summarized to fit the context window.`
- Timestamps: ISO 8601 UTC with `timespec="seconds"`.
- Measurement failures never fail a model call; log at debug level.
- ChainAgents CI: `uv run ruff check chainagents *.py scripts/*.py tests`, `uv run mypy`, `uv run pytest` (5 `test_response_exports` PDF tests fail locally without Pango; ignore those).
- SparxUI checks: `npm run check`, `npx eslint <files>`, `npx vitest run --project=unit`, `npm run build`.
- Commit author `Amin Mahpour <amin.mahpour@gmail.com>`; SparxUI commits use gitmoji prefixes (`✨ feat(...)`, `🐛 fix(...)`), ChainAgents commits use `feat(context-usage): ...`.

## Review Focus

- A model server that returns no `usage_metadata` (some OpenAI-compatible and Ollama setups): the snapshot must still appear with `exact = false`, and the popup must say "estimated". Test: Task 3, `test_missing_usage_keeps_estimates`.
- Two conversations streaming at once: snapshots must never cross sessions. Test: Task 2, `test_sessions_are_isolated`; Task 5, `test_concurrent_threads_record_separately`.
- Conversations created before this feature, or with a malformed stored `contextUsage`: the gauge stays hidden instead of crashing. Test: Task 7, `ignores malformed stored snapshots`.
- Switching branches in a conversation: the gauge must show the active branch's latest snapshot, not the newest overall. Test: Task 7, `uses the active branch snapshot`.
- After compaction the summary message heads every later request: the notice must fire once per new summary, not on every call. Test: Task 3, `test_summary_notice_fires_once_per_new_summary`.

---

## ChainAgents (branch `feat/context-usage`)

### Task 1: Shared summarization threshold resolution

**Files:**
- Create: `chainagents/runtime/summarization_budget.py`
- Modify: `chainagents/runtime/middleware.py` (`_build_deepagents_summarization_factory`)
- Test: `tests/test_summarization_budget.py`

**Interfaces:**
- Consumes: `runtime_models.SUMMARIZATION_TRIGGER_TOKENS_KEY`, `SUMMARIZATION_KEEP_TOKENS_KEY` (PR #126).
- Produces:
  - `configured_thresholds(model: Any, extensions: ExtensionsConfig) -> tuple[int | None, int | None]`: profile keys on `model.profile`, else `extensions.summarization_trigger_tokens` / `summarization_keep_tokens`, per value.
  - `effective_threshold_tokens(model: Any, extensions: ExtensionsConfig) -> tuple[int | None, int | None]`: `configured_thresholds`, then window defaults (`floor(0.85 × w)`, `floor(0.10 × w)`) when `model.profile["max_input_tokens"]` is an int, else `(170000, None)` for a missing trigger.
  - `context_window(model: Any) -> int | None`.

- [ ] **Step 1: Write failing tests** in `tests/test_summarization_budget.py`, using `FakeListChatModel` with `profile` set:
  - `test_profile_tokens_win_over_global`: profile trigger 6000/keep 1000, global 50000/9000 → `effective_threshold_tokens == (6000, 1000)`.
  - `test_global_tokens_apply_without_profile_budget`: profile `{"max_input_tokens": 200000}`, global 50000/9000 → `(50000, 9000)`.
  - `test_window_defaults_use_fractions`: profile `{"max_input_tokens": 16384}`, no global → `(13926, 1638)`.
  - `test_unknown_window_uses_fallback_trigger`: no profile, no global → `(170000, None)`.
  - `test_configured_thresholds_ignore_defaults`: profile `{"max_input_tokens": 16384}` → `configured_thresholds == (None, None)`.
- [ ] **Step 2: Run** `uv run pytest tests/test_summarization_budget.py -v` → FAIL (module missing).
- [ ] **Step 3: Implement** the three functions in `summarization_budget.py`. Then change `_build_deepagents_summarization_factory` to call `configured_thresholds(model, config.extensions)` in place of its inline profile/global lookup; behavior is unchanged.
- [ ] **Step 4: Run** `uv run pytest tests/test_summarization_budget.py tests/test_model_context_budget.py -v` → PASS.
- [ ] **Step 5: Commit** `feat(context-usage): share summarization threshold resolution`.

### Task 2: Snapshot, estimation and registry

**Files:**
- Create: `chainagents/runtime/context_usage.py`
- Test: `tests/test_context_usage.py`

**Interfaces:**
- Produces (all in `chainagents/runtime/context_usage.py`):
  - `CATEGORY_NAMES: tuple[str, ...] = ("system", "tools", "summary", "messages", "tool_results")`
  - `MAX_AGENTS_PER_SESSION = 50`
  - Run-config key constants `SESSION_ID_KEY`, `AGENT_KIND_KEY`, `AGENT_NAME_KEY`, `AGENT_INSTANCE_KEY` with the Global Constraints values.
  - `is_summary_message(message: Any) -> bool` (`additional_kwargs.get("lc_source") == "summarization"`).
  - `estimate_categories(system_message: Any | None, tools: Sequence[Any], messages: Sequence[Any]) -> dict[str, int]`: `count_tokens_approximately` per category. Tools are counted by passing them to `count_tokens_approximately([], tools=tools)`.
  - `scale_to_total(estimates: dict[str, int], total: int) -> dict[str, int]`: the spec's remainder rule. Non-system categories are scaled with largest-remainder rounding only when their sum exceeds `total`; `system = total − sum(others)`.
  - `@dataclass(frozen=True) ContextUsageSnapshot` with the spec's fields (no `agent_path`), and `to_payload() -> dict[str, Any]`.
  - `ContextUsageRegistry` methods:
    - `record(session_id: str, key: str, snapshot: ContextUsageSnapshot) -> None`
    - `note_summary(session_id: str, key: str, digest: str) -> bool`: True when the digest is new for that agent.
    - `last_summarized(session_id: str, key: str) -> str | None`
    - `mark_ended(session_id: str, key: str, status: str) -> None`
    - `list(session_id: str) -> list[ContextUsageSnapshot]`: main first, then `at_utc` descending.
    - `clear(session_id: str) -> None`

- [ ] **Step 1: Write failing tests:**
  - `test_estimate_categories_splits_summary_tool_and_other_messages`: a `SystemMessage`, one tool, a summary `HumanMessage(additional_kwargs={"lc_source": "summarization"})`, a `HumanMessage`, an `AIMessage` and a `ToolMessage` → every category > 0, `summary` counts only the summary message, `tool_results` only the `ToolMessage`.
  - `test_scale_to_total_makes_system_the_remainder`: `{"system": 100, "tools": 200, "summary": 0, "messages": 300, "tool_results": 100}`, total 1000 → `{"system": 400, "tools": 200, "summary": 0, "messages": 300, "tool_results": 100}`.
  - `test_scale_to_total_shrinks_others_when_they_exceed_total`: same estimates, total 300 → values sum to 300, `system == 0`, `messages ≥ tools ≥ tool_results`.
  - `test_registry_orders_main_first_then_newest`.
  - `test_registry_evicts_least_recently_updated_beyond_cap`: 51 agents → 50 listed, the first-updated one gone.
  - `test_sessions_are_isolated`: records in `s1` never appear in `list("s2")`; `clear("s1")` leaves `s2` intact.
  - `test_note_summary_is_true_once_per_digest`.
  - `test_mark_ended_updates_status`: `stopped` appears in `list()`.
  - `test_snapshot_payload_has_spec_fields`: payload keys equal the spec's field set.
- [ ] **Step 2: Run** `uv run pytest tests/test_context_usage.py -v` → FAIL.
- [ ] **Step 3: Implement** the module as listed. The registry is a `dict[str, OrderedDict[str, ContextUsageSnapshot]]` plus a summary-digest map; `record` moves the key to the end of the session's `OrderedDict` and evicts from the front.
- [ ] **Step 4: Run** the tests → PASS.
- [ ] **Step 5: Commit** `feat(context-usage): add snapshots, estimation and registry`.

### Task 3: ContextUsageMiddleware

**Files:**
- Modify: `chainagents/runtime/context_usage.py`
- Test: `tests/test_context_usage_middleware.py`

**Interfaces:**
- Consumes: Task 1 `effective_threshold_tokens`, `context_window`; Task 2 everything.
- Produces: `class ContextUsageMiddleware(AgentMiddleware)` with `__init__(self, *, registry: ContextUsageRegistry, extensions: ExtensionsConfig)`, `name == "ChainAgentsContextUsage"`, and `wrap_model_call` / `awrap_model_call`. It reads `langgraph.config.get_config()` for `configurable` and `metadata`, and writes events with `langgraph.config.get_stream_writer()`. Both are guarded: outside a graph run they raise, and the middleware then records and emits nothing.

- [ ] **Step 1: Write failing tests.** Call `awrap_model_call` directly with a `ModelRequest` built from a `FakeListChatModel` whose `profile` is `{"max_input_tokens": 16384}`. The handler returns a `ModelResponse` whose `AIMessage` has `usage_metadata`. Patch `get_config` and `get_stream_writer` in the module.
  - `test_main_agent_snapshot_is_recorded_and_streamed`: config without `lc_agent_name` → one `{"kind": "context_usage", "snapshot": ...}` write with `agent == "main"`, `kind == "main"`, `used == input_tokens`, `exact is True`, `window == 16384`, `trigger == 13926`, `keep == 1638`. The registry `list(thread_id)` has it.
  - `test_subagent_is_recorded_not_streamed`: metadata `{"lc_agent_name": "scout"}` → `agent == "scout"`, `kind == "subagent"`, no stream write.
  - `test_background_run_uses_owning_session`: configurable `{"thread_id": "s1:background:t1", "chainagents_session_id": "s1", "chainagents_agent_kind": "background", "chainagents_agent_name": "researcher", "chainagents_agent_instance": "t1"}` → recorded under `s1` with key `t1`, `kind == "background"`, not streamed.
  - `test_missing_usage_keeps_estimates`: no `usage_metadata` → `exact is False`, `used == sum(categories)`.
  - `test_summary_notice_fires_once_per_new_summary`: two calls with the same summary message, then one with a different summary. Exactly two `summarization_status` writes (`status == "completed"`, message per Global Constraints), and `last_summarized_at_utc` is set.
  - `test_measurement_errors_do_not_fail_the_call`: `estimate_categories` patched to raise → the handler's response is returned unchanged.
  - `test_model_errors_propagate`: handler raises `RuntimeError` → the same error propagates.
- [ ] **Step 2: Run** `uv run pytest tests/test_context_usage_middleware.py -v` → FAIL.
- [ ] **Step 3: Implement** the middleware. Identity resolution follows spec §1 exactly. The summary digest is the SHA-256 of the first non-system message's text when `is_summary_message` holds. A sync `wrap_model_call` mirrors the async one.
- [ ] **Step 4: Run** the tests → PASS.
- [ ] **Step 5: Commit** `feat(context-usage): measure model calls per agent`.

### Task 4: Stream event plumbing

**Files:**
- Modify: `chainagents/events/stream.py` (`StreamEventKind`, `AgentStreamEvent`, `_events_from_custom_chunk`)
- Modify: `chainagents/interfaces/api/app.py` (`_event_payload`)
- Test: `tests/test_context_usage_stream.py`

**Interfaces:**
- Produces: `StreamEventKind` gains `"context_usage"`; `AgentStreamEvent.context_usage: dict[str, Any] | None = None`. `_event_payload` drops the `context_usage` key for every other kind.

- [ ] **Step 1: Write failing tests:**
  - `test_custom_chunk_becomes_context_usage_event`: `AgentStreamEventAdapter` fed `{"type": "custom", "ns": (), "data": {"kind": "context_usage", "snapshot": {"used": 5}}}` → one event with `kind == "context_usage"` and `context_usage == {"used": 5}`.
  - `test_malformed_context_usage_chunk_is_ignored`: `snapshot` not a dict → no events.
  - `test_ndjson_payload_carries_context_usage_only_for_its_kind`: `_event_payload` of a `context_usage` event includes the key; a `response_delta` payload does not.
  - `test_cli_and_tui_ignore_context_usage`: rendering a `context_usage` event through each renderer's `on_event` writes nothing. Locate the renderers with `grep -n "def on_event" chainagents/interfaces`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** Add the explicit no-op branch to any renderer whose `on_event` does not already ignore unknown kinds.
- [ ] **Step 4: Run** the tests plus `uv run pytest tests/test_chainagents_api.py -q` → PASS.
- [ ] **Step 5: Commit** `feat(context-usage): stream main-agent context usage events`.

### Task 5: Runtime wiring

**Files:**
- Modify: `chainagents/runtime/lifecycle.py` (create `self.context_usage = ContextUsageRegistry()` next to `background_tasks`; `_close_conversation` calls `self.context_usage.clear(thread_id)`; pass the registry into graph builds)
- Modify: `chainagents/runtime/middleware.py` (`build_agent_middleware(..., context_usage: ContextUsageRegistry | None = None)` appends `ContextUsageMiddleware` last when set)
- Modify: `chainagents/runtime/graph.py` (`_SubagentBuildContext.context_usage`, `build_agent_kwargs(..., context_usage=...)`, `build_live_agent_graph(..., context_usage=...)`; forward to every `build_agent_middleware` call)
- Modify: `chainagents/runtime/background_tasks/tools.py` (`run_child` configurable gains the four keys: kind `background`, name = subagent name, instance = `task_id`, session = `session_id`)
- Modify: `chainagents/runtime/background_tasks/manager.py` (constructor gains `on_task_finished: Callable[[str, str, str], None] | None = None`, called from `_finish` with `(session_id, task_id, status)`; lifecycle maps every terminal status to `finished` via `registry.mark_ended`)
- Modify: `chainagents/runtime/live_agents/worker.py` (turn configurable gains the four keys: kind `live`, name = `record.name`, instance = `record.id`, session = `record.session_id`)
- Modify: `chainagents/runtime/live_agents/manager.py` (constructor gains `on_stopped: Callable[[str, str], None] | None = None`, called after teardown with `(session_id, agent_id)`; lifecycle calls `mark_ended(..., "stopped")`)
- Test: `tests/test_context_usage_runtime.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces: `AgentRuntime.context_usage: ContextUsageRegistry`.

- [ ] **Step 1: Write failing tests**, modeled on `test_each_agent_summarizes_with_its_own_profile_budget` in `tests/test_model_context_budget.py` (TOML via `DEEPAGENT_CONFIG`, `AgentRuntime.create`). The agent uses a `FakeMessagesListChatModel` returning messages with `usage_metadata`, patched in via `runtime_models.build_model`.
  - `test_main_and_subagent_snapshots_reach_the_registry`: a turn where the main model calls `task` for subagent `scout` → `runtime.context_usage.list(thread)` holds `main` and `scout` entries, and the stream yields `context_usage` events for `main` only.
  - `test_concurrent_threads_record_separately`: two threads run with `asyncio.gather` → each thread lists only its own main entry.
  - `test_close_conversation_clears_context_usage`.
  - `test_background_task_snapshot_belongs_to_owning_session_and_finishes`: a background subagent run → entry under the conversation session with `kind == "background"`, and `status == "finished"` once the task completes.
  - `test_live_agent_snapshot_is_marked_stopped`: spawn, message and tear down a live agent → entry with `kind == "live"`, `status == "stopped"`.
- [ ] **Step 2: Run** `uv run pytest tests/test_context_usage_runtime.py -v` → FAIL.
- [ ] **Step 3: Implement** the wiring listed under Files.
- [ ] **Step 4: Run** the new tests, then the full CI commands → PASS (except the known PDF tests).
- [ ] **Step 5: Commit** `feat(context-usage): wire measurement into every agent`.

### Task 6: API endpoint, status flag, docs and PR

**Files:**
- Modify: `chainagents/interfaces/api/app.py` (`GET /api/context-usage`; `RuntimeFeatureFlags.context_usage: bool = False`, set to `getattr(active_runtime, "context_usage", None) is not None`)
- Modify: `docs/interfaces.md` (HTTP API section: the endpoint, the `context_usage` event and the run-config keys)
- Test: `tests/test_context_usage_api.py`; update the expected `features` dict in `tests/test_chainagents_api.py::test_status_reports_runtime_configuration` with `"context_usage": False`

- [ ] **Step 1: Write failing tests:**
  - `test_context_usage_endpoint_lists_agents`: fake runtime with a registry holding main and subagent snapshots → `200`, `{"agents": [main_payload, sub_payload]}`.
  - `test_context_usage_endpoint_unknown_session_is_empty` → `{"agents": []}`.
  - `test_context_usage_requires_thread_id` → `422`.
  - `test_context_usage_disabled_returns_404`: runtime without `context_usage` → `404`.
  - `test_status_reports_context_usage_feature` → `features.context_usage is True`.
  - `test_context_usage_endpoint_requires_auth_in_token_mode`: with `CHAINAGENTS_API_TOKEN` set, no bearer → `401`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement**, following `list_live_agents` in `app.py`.
- [ ] **Step 4: Run** the full CI commands → PASS.
- [ ] **Step 5: Commit** `feat(context-usage): serve per-agent context usage over the API`, push, and open a PR against `feat/profile-context-window` titled `feat(context-usage): report context window usage and summarization per agent`. The body links the spec and lists the tests.

---

## SparxUI (branch `dev/context-usage-gauge`)

### Task 7: Types, reducer and persistence

**Files:**
- Modify: `src/lib/types/chainagents.ts` (`ChainAgentsContextCategories`, `ChainAgentsContextUsage` mirroring the snapshot payload; `ChainAgentsStreamEvent.context_usage?`; `features.context_usage?: boolean`)
- Modify: `src/lib/types/database.d.ts` (`DatabaseMessage.contextUsage?: ChainAgentsContextUsage`)
- Modify: `src/lib/types/index.ts` (exports)
- Create: `src/lib/utils/context-usage.ts`: `parseContextUsage(value: unknown): ChainAgentsContextUsage | null` and `latestContextUsage(messages: DatabaseMessage[]): ChainAgentsContextUsage | null` (the last assistant message on the given path with a valid `contextUsage`)
- Modify: `src/lib/stores/agent-messages.ts` (`LiveAgentMessageState` gains `contextUsage`)
- Modify: `src/lib/stores/chat/index.svelte.ts` (stream loop: on `context_usage`, set a local `contextUsage = parseContextUsage(event.context_usage) ?? contextUsage`; include it in `mirrorAndPersist`'s `updates` and in `publishLiveAgentMessage`)
- Modify: `src/lib/utils/chainagents-persistence.ts` (`normalizeMessageRecord` drops an invalid `contextUsage` via `parseContextUsage`)
- Test: `tests/unit/context-usage.test.ts`

- [ ] **Step 1: Write failing tests:**
  - `parseContextUsage accepts a valid snapshot`, and rejects missing `used`, non-object `categories` and negative numbers (→ `null`).
  - `latestContextUsage returns the newest assistant snapshot`.
  - `uses the active branch snapshot`: two sibling assistant branches, each with a snapshot, and messages filtered to one branch with `filterByLeafNodeId` → that branch's snapshot.
  - `ignores malformed stored snapshots`: `normalizeMessageRecord({..., contextUsage: {used: "x"}})` has no `contextUsage`.
- [ ] **Step 2: Run** `npx vitest run --project=unit tests/unit/context-usage.test.ts` → FAIL.
- [ ] **Step 3: Implement** as listed.
- [ ] **Step 4: Run** the test file, then `npm run check` → PASS, 0 errors.
- [ ] **Step 5: Commit** `✨ feat(chainagents): store context usage snapshots on messages`.

### Task 8: Popup data shaping

**Files:**
- Modify: `src/lib/utils/context-usage.ts`
- Test: `tests/unit/context-usage.test.ts`

**Interfaces:**
- Produces: `shapeContextUsage(usage: ChainAgentsContextUsage): ContextUsageView`, where
  - `ContextUsageView = { percent: number | null; dialPercent: number | null; overflow: boolean; scale: number; segments: Array<{ key: CategoryKey | 'free'; label: string; tokens: number; percent: number }>; triggerPercent: number | null }`
  - `scale = window ?? used`
  - labels are `System prompt`, `Tools`, `Summary`, `Messages`, `Tool results` and `Free`
  - `dialPercent` is clamped to `[0, 100]`
  - there is no `free` segment and `triggerPercent` is `null` when `window` is `null`

- [ ] **Step 1: Write failing tests:**
  - `shapes categories and free space against the window`: window 16384, used 12000 → `percent ≈ 73.2`, a `free` segment of 4384 tokens, and `triggerPercent ≈ 85`.
  - `clamps the dial and flags overflow`: used 20000, window 16384 → `dialPercent === 100`, `overflow === true`, no `free` segment.
  - `scales to used when the window is unknown`: `percent === null`, `scale === used`, `triggerPercent === null`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `shapeContextUsage`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `✨ feat(chainagents): shape context usage for display`.

### Task 9: Service and capability flag

**Files:**
- Modify: `src/lib/services/chainagents.service.ts` (`listContextUsage(threadId: string, signal?: AbortSignal): Promise<ChainAgentsContextUsage[]>`, filtering entries through `parseContextUsage`; place it to satisfy `perfectionist/sort-classes`)
- Modify: `src/lib/stores/runtime.svelte.ts` (`get supportsContextUsage(): boolean`, true when `supportsBackgroundTasks && status.features.context_usage === true`)
- Test: `tests/unit/chainagents-context-usage-service.test.ts`

- [ ] **Step 1: Write failing tests**, styled like `tests/unit/chainagents-live-agents.test.ts`:
  - `lists agent snapshots`: URL `/api/context-usage?thread_id=t+1`, `cache: 'no-store'`.
  - `drops malformed entries`.
  - `reports a disabled backend as a 404 error`: a `ChainAgentsHttpError` with `status === 404`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `✨ feat(chainagents): fetch per-agent context usage`.

### Task 10: Gauge and popup components

**Files:**
- Create: `src/lib/components/app/chat/ChatForm/ChatFormContextGauge/AgentContextGauge.svelte`. It reuses `ContextGaugeDial` with `colorLevelFromPercent(view.dialPercent)` and `gauge-popup.svelte` triggers, and is hidden when `latestContextUsage(conversationsStore.activeMessages)`, merged with the live streaming state from `liveAgentMessages`, is `null`.
- Create: `src/lib/components/app/chat/ChatForm/ChatFormContextGauge/AgentContextPopup.svelte`, implementing spec §3 Popup items 1–4. The agent list polls `listContextUsage` every 3000 ms while open and a turn streams, hides on a 404, and otherwise shows "Couldn't load agents" with a Retry button.
- Modify: `src/lib/components/app/chat/ChatForm/ChatFormActions/ChatFormActions.svelte` (render `AgentContextGauge` when `runtimeStore.supportsBackgroundTasks`)
- Modify: `src/lib/components/app/chat/ChatForm/ChatFormContextGauge/index.ts` if that folder exports components through one

- [ ] **Step 1: Implement** both components. Copy: the header `{used} / {window} tokens · {percent}%`; "estimated" when `exact` is false; with an unknown window, `{used} tokens` plus "Set context_window in the model profile to see the limit."; `Summarizes at {trigger} · keeps {keep}`; `Last summarized {local time}`. Format numbers with `toLocaleString()`.
- [ ] **Step 2: Verify** `npm run check` (0 errors), `npx eslint` on the touched files, `npx vitest run --project=unit` (all pass) and `npm run build` (succeeds).
- [ ] **Step 3: Commit** `✨ feat(chainagents): show context window gauge and per-agent usage`.

### Task 11: Live verification and SparxUI PR

- [ ] **Step 1:** Start ChainAgents from `feat/context-usage` in token mode with the probe config: GLM at `10.0.1.181:8888`, `context_window = 16384`, trigger 8000, keep 1500, and one sync subagent with `model` on a second profile using `context_window = 32768`. Then start SparxUI with `CHAINAGENTS_ORIGIN=http://127.0.0.1:8000 npx vite dev --host 0.0.0.0`.
- [ ] **Step 2:** Through the proxy, stream 10 turns with `curl`. Verify:
  - one `context_usage` event per main-agent model call, whose `used` equals the server's `input_tokens`
  - a `summarization_status` event with the notice text once summarization runs
  - `/api/context-usage` listing `main` and the subagent, each against its own window

  Record the numbers for the PR.
- [ ] **Step 3:** Ask the user to open `http://10.0.1.144:5173` and check the gauge, popup and agent list visually.
- [ ] **Step 4:** Push `dev/context-usage-gauge` and open a SparxUI PR titled `✨ feat(chainagents): context window gauge and per-agent usage`, linking the ChainAgents PR and the live numbers.
