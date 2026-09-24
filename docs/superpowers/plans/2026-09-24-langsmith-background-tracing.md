# LangSmith Background Tracing Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task under Sol at high effort. Steps use checkbox syntax for tracking.

**Status:** Approved for implementation on 2026-09-24.

**Goal:** Make local background subagents observable in LangSmith, with configurable parent linkage and independent discovery by task ID.

**Architecture:** Add optional LangSmith configuration and a runtime-owned tracing client. Capture a small parent trace reference at submission time, then establish a new LangSmith tracing scope inside the existing isolated background job. Trace the graph's model and tool runs once, whether the background graph is invoked or streamed.

**Tech stack:** Python 3.12+, installed LangSmith 0.14.0, LangChain callbacks, LangGraph, asyncio, pytest, UV.

**Spec:** The design contract below is the specification for this draft.

## Design contract

Recommended configuration:

```toml
[langsmith]
enabled = false
project = "chainagents"
background_trace_mode = "linked" # "linked" or "separate"
```

Enabling this integration traces foreground runs and local background subagents. `linked` attaches a background graph run beneath its spawning run when a LangSmith parent is available. It remains searchable by background task ID, session ID, and agent name. `separate` gives each background task a new root trace, with originating parent run and trace IDs in metadata for correlation. Both modes record one execution tree; a linked child is a run within the parent trace, not an additional top-level trace.

If no LangSmith parent is available, `linked` falls back to a separate root and records `background_trace_link = "parent_unavailable"`. Nested jobs link to their immediate spawning background run when available. Batch siblings get distinct run IDs and task metadata.

Credentials come from the SDK's existing environment settings: `LANGSMITH_API_KEY`, optional `LANGSMITH_ENDPOINT`, and optional `LANGSMITH_WORKSPACE_ID`. No secrets go into TOML. Project precedence is a configured nonempty project, then `LANGSMITH_PROJECT`, then `chainagents`. The example explicitly chooses `chainagents`.

`enabled` controls the new ChainAgents integration. When false, ChainAgents creates no LangSmith client or callback for this feature. Existing SDK tracing enabled externally through environment variables or caller callbacks remains caller-controlled. Do not mutate process environment variables to turn tracing on or off.

Use this stable metadata on the background graph invocation:

```python
{
    "session_id": session_id,
    "background_task_id": task_id,
    "background_parent_task_id": parent_task_id,
    "background_agent": subagent_type,
    "background_agent_path": list(agent_path),
    "background_trace_mode": mode,
    "background_trace_link": "linked" | "separate" | "parent_unavailable",
    "originating_parent_run_id": parent_run_id,
    "originating_parent_trace_id": parent_trace_id,
}
```

Omit absent optional values. Use `run_name = f"background/{subagent_type}/{task_id}"`, a fresh UUID for `run_id`, and tags `chainagents` and `background-subagent`. Keep the existing checkpoint thread ID `f"{session_id}:background:{task_id}"`; observability must not change graph state ownership. Graph inputs, outputs, tools, model calls, and available usage metadata are recorded by normal LangChain tracing. UI reasoning/tool visibility switches do not control tracing.

The runtime background-task record remains authoritative for cancellation and cleanup status. The trace describes graph execution; a cleanup failure after successful graph completion need not rewrite the graph run as a model failure. Document that distinction.

## Global constraints

- This draft covers local background subagents, including nested delegation and batches, across Chainlit, CLI, TUI, API, and exported graphs.
- Keep `asyncio.create_task(..., context=contextvars.Context())` and the existing session/artifact scoping behavior.
- Transfer only a trace reference and allowlisted application metadata. Do not copy the parent callback manager, Chainlit context, arbitrary baggage, runnable config, or contextvars context into the background task.
- Preserve Langfuse and caller callbacks. Add LangSmith callbacks without replacing either or creating duplicate LangSmith runs.
- Keep SDK imports and client creation lazy when the integration is disabled.
- Declare `langsmith>=0.14.0` directly in `pyproject.toml`, and regenerate the lock with UV.
- Start runtime implementation on a separate feature branch after checking the current master and PR107 state. Keep this draft out of PR107.
- Planning uses Astra at extra-high effort; implementation uses Sol at high effort.

## Verified feasibility

A local characterization using the installed SDK passed with outbound HTTP blocked. A foreground `RunnableLambda` finished; a second task was then started with a fresh `contextvars.Context()`. Reconstructing the parent reference and entering `tracing_context` gave the background child the parent's trace ID and parent run ID. Setting `parent=False` produced a new root whose trace ID equaled its own run ID.

The SDK provides `RunTree.from_runnable_config(config)` to recover a LangChain callback parent. Capture its `dotted_order`, run ID, trace ID, and project while submitting the task. If that returns no parent, use `get_current_run_tree()` only when a current SDK run exists. Reconstruct a parent with the explicit runtime-owned client. Passing only a string into `tracing_context` can initialize the SDK's default client; the explicit `RunTree` avoids that unwanted client initialization.

```python
parent = (
    RunTree.from_dotted_order(
        reference.dotted_order,
        client=tracing.client,
        project_name=reference.project,
    )
    if mode == "linked" and reference is not None
    else False
)
with tracing_context(
    enabled=True,
    client=tracing.client,
    project_name=reference.project if parent is not False else tracing.project,
    parent=parent,
):
    result = await child.ainvoke(state, child_config)
```

For `astream`, keep the same scope active through the complete `async for` and stream closure. A finished foreground turn must not invalidate the captured trace reference or close a client still used by background jobs.

## Review focus

1. Delayed children and two overlapping sessions must attach to the correct originating run, even after the parent finishes.
2. Result-only, reasoning-only, and nonstreaming child execution must have the same tracing behavior; UI streaming is an independent feature.
3. Missing parent context must create a usable independent trace, rather than dropping the task or inventing a parent.
4. SDK export failures must not retry or rerun the agent, and cancellation must keep its existing task-manager behavior.
5. Exported graphs and lazy v3 streams must preserve their invocation/iteration contract and submit buffered traces at application teardown.

## Task 1: Configured foreground tracing and client ownership

**Files:** `chainagents/runtime/types.py`, `chainagents/runtime/config.py`, `chainagents/runtime/tracing.py`, `chainagents/runtime/lifecycle.py`, `chainagents/runtime/graph.py`, `chainagents/runtime/core.py`, shared run-config callers, `pyproject.toml`, `uv.lock`, `deepagent.toml`, `README.md`; new `tests/test_langsmith_tracing.py`.

**Interfaces:** Add frozen `LangSmithConfig(enabled: bool = False, project: str | None = None, background_trace_mode: Literal["linked", "separate"] = "linked")`. Add it to `FileConfig` and `RuntimeConfig`. Add `LangSmithTracing`, owning one explicit `langsmith.Client` and a resolved project, with `new_callback()`, `flush()`, and `close()` methods. Runtime creation and static graph construction own service instances; multiple sessions share the owning runtime's client, but parent references are per submission.

- [ ] Add configuration tests for omitted settings, both modes, project precedence, non-table values, nonboolean `enabled`, nonstring/empty project, and unknown modes. Check the exact field name in error messages.

```python
@pytest.mark.parametrize("mode", ["linked", "separate"])
def test_langsmith_trace_mode_config(mode):
    parsed = parse_langsmith_config({"langsmith": {
        "enabled": True, "project": "test-project", "background_trace_mode": mode,
    }})
    assert parsed.enabled is True
    assert parsed.project == "test-project"
    assert parsed.background_trace_mode == mode
```

- [ ] Run the new configuration tests and confirm they fail before implementing the parser and config propagation.
- [ ] Implement lazy service creation for enabled configurations. Use an explicit `Client()` and `LangChainTracer(client=client, project_name=project)`. Append the callback in the shared run config path and the static graph invocation path. Make any new helper argument optional for existing public callers, and make client ownership explicit for all built-in callers. Avoid adding a second equivalent LangSmith tracer if invocation callbacks already supply one for the same destination.
- [ ] Test actual callback events with a recording client transport: one foreground root, preserved Langfuse/custom handlers, disabled integration creates no client, and two successive conversations do not share run IDs.
- [ ] Add the direct dependency using UV, document the three TOML fields and environment credentials, run focused tests, and commit this deliverable.

## Task 2: Isolated linked and separate background traces

**Files:** `chainagents/runtime/tracing.py`, `chainagents/runtime/background_tasks.py`, `chainagents/runtime/lifecycle.py`, `chainagents/runtime/graph.py`, `tests/test_langsmith_tracing.py`, `tests/test_background_tasks.py`, `tests/test_langgraph_background.py`.

**Interfaces:** Add frozen `LangSmithParentReference(dotted_order: str, run_id: str, trace_id: str, project: str)`. Add `LangSmithTracing.capture_parent(run_config: dict[str, Any]) -> LangSmithParentReference | None` and `LangSmithTracing.background_scope(parent: LangSmithParentReference | None, *, mode: Literal["linked", "separate"]) -> ContextManager[None]`. Pass optional `langsmith_tracing` and the configured mode into every `create_background_task_tools` construction in lifecycle and static graph builders.

- [ ] Port the verified local characterization into `tests/test_langsmith_tracing.py`, using real `RunnableLambda`, `LangChainTracer`, callback managers, and `RunTree` behavior. Stub only SDK transport and block outbound HTTP.

```python
# After a foreground run has ended and reference was captured inside it:
async def background():
    with tracing.background_scope(reference, mode="linked"):
        return await child.ainvoke("payload", config={"run_id": child_id})

await asyncio.create_task(background(), context=contextvars.Context())
assert recorded[child_id].trace_id == foreground_id
assert recorded[child_id].parent_run_id == foreground_id
```

- [ ] Add the separate-mode test with `recorded[child_id].trace_id == child_id` and `parent_run_id is None`; add linked-without-parent fallback and assert correlation metadata is absent rather than fabricated.
- [ ] Add submission-level tests for streaming on/off, nested tasks, ordered batch results, two interleaved sessions, and a child that starts after the foreground run completes. Assert task/session/agent metadata and the correct parent run in the recorded SDK payloads.
- [ ] Add a sentinel UI callback and unrelated ContextVar to the parent; assert no background invocation reaches the UI callback and the unrelated value is absent in the child. Preserve existing artifact/session isolation tests.
- [ ] Capture the reference synchronously in `build_submission` while `ToolRuntime.config` is available. Store only the small immutable reference and task identity in the closure. Inside `run_child`, construct the existing child config plus `run_name`, UUID `run_id`, tags, and allowlisted metadata. Wrap both execution branches with the new scope. Do not copy `runtime.config['callbacks']`.
- [ ] Run the new regressions and the existing background and exported-graph tests. Commit this deliverable.

## Task 3: Lifecycle behavior and publication checks

**Files:** `chainagents/runtime/tracing.py`, `chainagents/runtime/lifecycle.py`, `chainagents/runtime/graph.py`, relevant CLI/API shutdown paths, `tests/test_langsmith_tracing.py`, `tests/test_runtime_lifecycle.py`, `tests/test_agent_cli.py`, `tests/test_langgraph_background.py`, `README.md`.

**Interfaces:** `flush()` calls the owned client's supported flush operation after task draining; `close()` closes only the owned client after runtime terminal shutdown. Exported graph teardown flushes without permanently disabling a reusable graph/service. Do not close a process-global SDK client supplied by someone else.

- [ ] Add an ordering regression that starts a background child, initiates shutdown, and checks that child completion precedes trace flush/client close. Use events instead of sleeps.
- [ ] Add a callback-export failure regression and verify the child runner executes once and returns its original result. Catch tracing setup/export failures narrowly; never catch an agent failure and run the agent again without tracing.

```python
await runtime.close()
assert lifecycle_events.index("child-finished") < lifecycle_events.index("trace-flush")
assert lifecycle_events.index("trace-flush") < lifecycle_events.index("client-close")
```

- [ ] Add cancellation and graph-exception regressions asserting original manager status and one graph invocation. Add exported-graph second-lifespan and lazy-v3-stream tests for tracing scope/client reuse.
- [ ] Wire flush/close into the runtime lifecycle after background-task drain and into exported graph teardown. Preserve Langfuse shutdown behavior.
- [ ] Extend README examples with both modes, task-ID filtering, no-parent fallback, environment precedence, and graph-execution versus cleanup status semantics.
- [ ] Run `.venv/bin/pytest -q`, `.venv/bin/ruff check chainagents *.py scripts/*.py`, `.venv/bin/mypy`, and `git diff --check`. Report any warnings. Use the dependency-managed environment on the implementation branch.
- [ ] When credentials are available, perform a live smoke run for each mode and inspect the LangSmith run tree and task metadata. If unavailable, explicitly report SDK/transport verification only; do not claim a live dashboard check.
- [ ] Commit, push, and create a separate draft PR after implementation authorization. Attach the PR, inspect CI on its full head SHA, and address review findings. Do not merge without the user's request.

## Sources

- [LangSmith tracing with LangChain](https://docs.langchain.com/langsmith/trace-with-langchain): callback/context integration, projects, run identity, and metadata.
- [LangSmith distributed tracing](https://docs.langchain.com/langsmith/distributed-tracing): parent context propagation.
- [Tracing without environment variables](https://docs.langchain.com/langsmith/trace-without-env-vars): scoped enablement and explicit clients.
- Installed SDK inspected and characterized on 2026-09-24: `RunTree.from_runnable_config`, `RunTree.from_dotted_order`, `tracing_context`, and `Client.flush/close`.
