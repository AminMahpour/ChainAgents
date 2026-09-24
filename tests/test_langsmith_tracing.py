"""LangSmith configuration and tracing boundaries."""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import contextmanager
from dataclasses import replace
from uuid import uuid4
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langsmith import Client
import langsmith

import pytest
from langchain_core.tracers.langchain import LangChainTracer

from chainagents.runtime import config as runtime_config
from chainagents.runtime import lifecycle as runtime_lifecycle
from chainagents.runtime import graph as runtime_graph
from chainagents.runtime import tracing as runtime_tracing
from chainagents.runtime.background_tasks import (
    BackgroundTaskManager,
    create_background_task_tools,
    current_background_session_generation,
    scope_background_session_invocation,
)
from chainagents.runtime.types import BackgroundSubagentConfig, LangSmithConfig, LangfuseConfig


@pytest.mark.parametrize("mode", ["linked", "separate"])
def test_parse_langsmith_trace_mode(mode: str) -> None:
    parsed = runtime_config.parse_langsmith_config(
        {
            "langsmith": {
                "enabled": True,
                "project": "test-project",
                "background_trace_mode": mode,
            }
        }
    )
    assert parsed.enabled is True
    assert parsed.project == "test-project"
    assert parsed.background_trace_mode == mode


def test_parse_langsmith_defaults_are_disabled() -> None:
    parsed = runtime_config.parse_langsmith_config({})
    assert parsed.enabled is False
    assert parsed.project is None
    assert parsed.background_trace_mode == "linked"


@pytest.mark.parametrize(
    ("value", "field"),
    [
        ("yes", "langsmith"),
        ({"enabled": "yes"}, "langsmith.enabled"),
        ({"project": 3}, "langsmith.project"),
        ({"project": "  "}, "langsmith.project"),
        ({"background_trace_mode": "unknown"}, "langsmith.background_trace_mode"),
    ],
)
def test_parse_langsmith_rejects_invalid_fields(value: object, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        runtime_config.parse_langsmith_config({"langsmith": value})


def test_langsmith_tracing_uses_configured_project_and_client(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_PROJECT", "environment-project")
    client = object()
    tracing = runtime_tracing.LangSmithTracing(
        LangSmithConfig(enabled=True, project="configured-project"), client=client
    )
    callback = tracing.new_callback()
    assert isinstance(callback, LangChainTracer)
    assert callback.project_name == "configured-project"
    assert callback.client is client


def test_langsmith_tracing_uses_environment_then_default_project(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_PROJECT", "environment-project")
    configured = runtime_tracing.LangSmithTracing(
        LangSmithConfig(enabled=True), client=object()
    )
    assert configured.project == "environment-project"
    monkeypatch.delenv("LANGSMITH_PROJECT")
    default = runtime_tracing.LangSmithTracing(
        LangSmithConfig(enabled=True), client=object()
    )
    assert default.project == "chainagents"


def test_disabled_langsmith_runtime_does_not_create_client(monkeypatch) -> None:
    def forbidden_client() -> object:
        raise AssertionError("disabled tracing created a client")

    monkeypatch.setattr(runtime_tracing, "_import_langsmith_client", forbidden_client)
    assert runtime_tracing.build_langsmith_tracing(LangSmithConfig()) is None


def test_file_config_propagates_langsmith_to_runtime(tmp_path, monkeypatch) -> None:
    path = tmp_path / "deepagent.toml"
    path.write_text(
        '[langsmith]\nenabled = true\nproject = "runtime-project"\n'
        'background_trace_mode = "separate"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(path))
    config = runtime_config.RuntimeConfig.from_env()
    assert config.langsmith == LangSmithConfig(
        enabled=True,
        project="runtime-project",
        background_trace_mode="separate",
    )


def test_runtime_owns_enabled_langsmith_tracing(tmp_path, monkeypatch) -> None:
    path = tmp_path / "deepagent.toml"
    path.write_text('[langsmith]\nenabled = true\n', encoding="utf-8")
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(path))
    marker = object()
    monkeypatch.setattr(runtime_tracing, "build_langsmith_tracing", lambda config: marker)
    runtime = runtime_lifecycle.AgentRuntime(runtime_config.RuntimeConfig.from_env())
    assert runtime.langsmith_tracing is marker


def test_run_config_preserves_langfuse_while_adding_langsmith(monkeypatch) -> None:
    class FakeLangfuseHandler:
        pass

    monkeypatch.setattr(
        runtime_tracing,
        "_import_langfuse_callback_handler",
        lambda: FakeLangfuseHandler,
    )
    config = runtime_config.RuntimeConfig.from_env()
    config = replace(config, langfuse=LangfuseConfig(enabled=True))
    tracing = runtime_tracing.LangSmithTracing(
        LangSmithConfig(enabled=True), client=object()
    )
    run_config = runtime_tracing.build_langgraph_run_config(
        config, thread_id="conversation-1", langsmith_tracing=tracing
    )
    callbacks = run_config["callbacks"]
    assert len(callbacks) == 2
    assert isinstance(callbacks[0], FakeLangfuseHandler)
    assert isinstance(callbacks[1], LangChainTracer)
    assert run_config["metadata"] == {
        "langfuse_session_id": "conversation-1",
        "session_id": "conversation-1",
    }
    assert run_config["tags"] == ["chainagents"]


def test_chainlit_run_config_attaches_runtime_langsmith_client() -> None:
    from chainagents.interfaces.chainlit.app import build_langgraph_config
    from chainagents.runtime.types import AppSettings

    tracing = runtime_tracing.LangSmithTracing(
        LangSmithConfig(enabled=True), client=object()
    )
    settings = AppSettings(
        model_name="test-model", reasoning_level="medium", thread_id="conversation-1"
    )
    run_config = build_langgraph_config(
        settings,
        recursion_limit=50,
        runtime_config=runtime_config.RuntimeConfig.from_env(),
        langsmith_tracing=tracing,
    )
    callback = run_config["callbacks"][0]
    assert isinstance(callback, LangChainTracer)
    assert callback.client is tracing.client


def test_foreground_invocations_have_distinct_traces_without_network() -> None:
    client = Client(
        api_key="local-test-key",
        api_url="http://127.0.0.1:1",
        auto_batch_tracing=False,
    )
    tracing = runtime_tracing.LangSmithTracing(
        LangSmithConfig(enabled=True), client=client
    )
    created: list[dict[str, object]] = []

    def record_create(
        self: Client, name: str, inputs: object, run_type: str, **kwargs: object
    ) -> None:
        created.append({"name": name, **kwargs})

    async def run_twice() -> None:
        graph = RunnableLambda(lambda value: value)
        config = runtime_config.RuntimeConfig.from_env()
        for session in ("session-one", "session-two"):
            await graph.ainvoke(
                "prompt",
                config=runtime_tracing.build_langgraph_run_config(
                    config, thread_id=session, langsmith_tracing=tracing
                ),
            )

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", record_create),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        asyncio.run(run_twice())
    assert len(created) == 2
    assert created[0]["id"] != created[1]["id"]
    assert all(item.get("parent_run_id") is None for item in created)


@pytest.mark.parametrize("caller_tracer", ["none", "same_client", "equivalent_client"])
def test_exported_graph_foreground_run_is_traced(
    tmp_path, monkeypatch, caller_tracer: str
) -> None:
    path = tmp_path / "deepagent.toml"
    path.write_text('[langsmith]\nenabled = true\n', encoding="utf-8")
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(path))
    client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    monkeypatch.setattr(runtime_tracing, "_import_langsmith_client", lambda: lambda: client)
    monkeypatch.setattr(
        runtime_graph.runtime_middleware,
        "create_deep_agent_with_configured_summarization",
        lambda *args, **kwargs: RunnableLambda(lambda value: value),
    )
    created: list[str] = []

    def record_create(
        self: Client, name: str, inputs: object, run_type: str, **kwargs: object
    ) -> None:
        created.append(name)

    async def invoke() -> None:
        graph = runtime_graph.create_configured_graph(include_async_subagents=False)
        config: dict[str, object] = {"configurable": {"thread_id": "static-session"}}
        if caller_tracer != "none":
            caller_client = (
                client
                if caller_tracer == "same_client"
                else Client(
                    api_key="local-test-key",
                    api_url="http://127.0.0.1:1",
                    auto_batch_tracing=False,
                )
            )
            config["callbacks"] = [
                LangChainTracer(client=caller_client, project_name="chainagents")
            ]
        await graph.ainvoke(
            {"messages": []}, config=config
        )

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", record_create),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        asyncio.run(invoke())
    assert len(created) == 1


def test_exported_graph_deduplicates_ambient_owned_callback() -> None:
    client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    tracing = runtime_tracing.LangSmithTracing(LangSmithConfig(enabled=True), client=client)
    manager = BackgroundTaskManager(BackgroundSubagentConfig(enabled=True))
    inner = scope_background_session_invocation(
        RunnableLambda(lambda value: value),
        manager,
        run_config_transform=tracing.with_callback,
    )
    created: list[object] = []

    async def outer_body(value: str) -> str:
        return await inner.ainvoke(value)

    def record_create(
        self: Client, name: str, inputs: object, run_type: str, **kwargs: object
    ) -> None:
        created.append(kwargs["id"])

    async def execute() -> None:
        await RunnableLambda(outer_body).ainvoke(
            "payload", config={"callbacks": [tracing.new_callback()]}
        )
        await manager.close()

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", record_create),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        asyncio.run(execute())
    assert len(created) == len(set(created))


@pytest.mark.parametrize("mode", ["linked", "separate"])
def test_background_scope_uses_captured_parent_after_foreground_finishes(mode: str) -> None:
    client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    tracing = runtime_tracing.LangSmithTracing(
        LangSmithConfig(enabled=True, background_trace_mode=mode), client=client
    )
    parent_id, child_id = uuid4(), uuid4()
    created: dict[object, dict[str, object]] = {}
    references: list[object] = []

    def parent_body(value: str, config: dict[str, object]) -> str:
        references.append(tracing.capture_parent(config))
        return value

    def record_create(
        self: Client, name: str, inputs: object, run_type: str, **kwargs: object
    ) -> None:
        created[kwargs["id"]] = {"name": name, **kwargs}

    async def execute() -> None:
        await RunnableLambda(parent_body).ainvoke(
            "prompt", config={"callbacks": [tracing.new_callback()], "run_id": parent_id}
        )

        async def background() -> None:
            with tracing.background_scope(references[0], mode=mode):
                await RunnableLambda(lambda value: value).ainvoke(
                    "payload", config={"run_id": child_id}
                )

        await asyncio.create_task(background(), context=contextvars.Context())

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", record_create),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        asyncio.run(execute())

    assert references[0] is not None
    assert created[child_id]["trace_id"] == (parent_id if mode == "linked" else child_id)
    assert created[child_id].get("parent_run_id") == (
        parent_id if mode == "linked" else None
    )


def test_background_scope_without_parent_creates_separate_root() -> None:
    client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    tracing = runtime_tracing.LangSmithTracing(LangSmithConfig(enabled=True), client=client)
    child_id = uuid4()
    created: list[dict[str, object]] = []

    def record_create(
        self: Client, name: str, inputs: object, run_type: str, **kwargs: object
    ) -> None:
        created.append(kwargs)

    async def execute() -> None:
        with tracing.background_scope(None, mode="linked"):
            await RunnableLambda(lambda value: value).ainvoke(
                "payload", config={"run_id": child_id}
            )

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", record_create),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        asyncio.run(execute())
    assert created[0]["id"] == child_id
    assert created[0]["trace_id"] == child_id
    assert created[0].get("parent_run_id") is None


@pytest.mark.parametrize("equivalent_client_only", [False, True])
def test_capture_parent_chooses_runtime_tracer_over_caller_project(
    equivalent_client_only: bool,
) -> None:
    owned_client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    caller_client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    tracing = runtime_tracing.LangSmithTracing(
        LangSmithConfig(enabled=True, project="configured-project"),
        client=owned_client,
    )
    parent_id = uuid4()
    references: list[object] = []

    def capture(value: str, config: dict[str, object]) -> str:
        references.append(tracing.capture_parent(config))
        return value

    async def execute() -> None:
        callbacks = (
            [LangChainTracer(client=caller_client, project_name="configured-project")]
            if equivalent_client_only
            else [
                LangChainTracer(client=caller_client, project_name="caller-project"),
                tracing.new_callback(),
            ]
        )
        await RunnableLambda(capture).ainvoke(
            "prompt",
            config={
                "callbacks": callbacks,
                "run_id": parent_id,
            },
        )

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", lambda *args, **kwargs: None),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        asyncio.run(execute())
    assert references[0] is not None
    assert references[0].run_id == str(parent_id)
    assert references[0].project == "configured-project"


@pytest.mark.parametrize("mode", ["linked", "separate"])
@pytest.mark.parametrize("stream_activity", [False, True])
def test_background_tool_traces_task_identity_and_parent(
    mode: str, stream_activity: bool
) -> None:
    client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    tracing = runtime_tracing.LangSmithTracing(
        LangSmithConfig(enabled=True, background_trace_mode=mode), client=client
    )
    manager = BackgroundTaskManager(
        BackgroundSubagentConfig(enabled=True, stream_activity=stream_activity)
    )
    parent_id = uuid4()
    created: dict[object, dict[str, object]] = {}
    private_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "unrelated_parent_context", default=None
    )

    class Child:
        async def ainvoke(self, state: object, config: dict[str, object]) -> object:
            assert private_context.get() is None
            assert "callbacks" not in config
            return await RunnableLambda(
                lambda value: {"messages": [AIMessage(content="completed")]}
            ).ainvoke(state, config=config)

        async def astream(self, state: object, config: dict[str, object], **kwargs: object):
            result = await self.ainvoke(state, config)
            yield ("values", result)

    spawn = next(
        tool
        for tool in create_background_task_tools(
            manager=manager,
            subagents={"researcher": Child()},
            agent_path=(),
            recursion_limit=30,
            langsmith_tracing=tracing,
        )
        if tool.name == "spawn_background_task"
    )

    def record_create(
        self: Client, name: str, inputs: object, run_type: str, **kwargs: object
    ) -> None:
        created[kwargs["id"]] = {"name": name, **kwargs}

    async def parent_body(value: str, config: dict[str, object]) -> str:
        private_context.set("parent-only")
        runtime = ToolRuntime(
            state={},
            context=None,
            config=config,
            stream_writer=lambda _: None,
            tool_call_id="call-1",
            store=None,
        )
        await spawn.coroutine("investigate", "researcher", runtime)
        return value

    async def execute() -> object:
        await RunnableLambda(parent_body).ainvoke(
            "prompt",
            config={
                "callbacks": [tracing.new_callback()],
                "run_id": parent_id,
                "configurable": {"thread_id": "session-one"},
            },
        )
        snapshots = await manager.wait_session("session-one")
        await manager.close()
        assert snapshots[0].status == "success"
        return snapshots[0]

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", record_create),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        snapshot = asyncio.run(execute())

    child = next(payload for run_id, payload in created.items() if run_id != parent_id)
    assert child["trace_id"] == (parent_id if mode == "linked" else child["id"])
    assert child.get("parent_run_id") == (parent_id if mode == "linked" else None)
    metadata = child["extra"]["metadata"]
    assert metadata["session_id"] == "session-one"
    assert metadata["background_task_id"] == snapshot.task_id
    assert metadata["background_agent"] == "researcher"
    assert metadata["background_agent_path"] == ["researcher"]
    assert metadata["background_trace_mode"] == mode
    assert metadata["background_trace_link"] == mode
    assert child["name"] == f"background/researcher/{snapshot.task_id}"


def test_runtime_flushes_and_closes_after_background_task_teardown(
    tmp_path, monkeypatch
) -> None:
    events: list[str] = []

    class RecordingTracing:
        def flush(self) -> None:
            events.append("trace-flush")

        def close(self) -> None:
            events.append("client-close")

    monkeypatch.setattr(
        runtime_tracing, "build_langsmith_tracing", lambda config: RecordingTracing()
    )
    config = runtime_config.RuntimeConfig.from_env()
    config = replace(
        config,
        extensions=replace(
            config.extensions,
            background_subagents=BackgroundSubagentConfig(enabled=True),
        ),
    )
    runtime = runtime_lifecycle.AgentRuntime(config, project_root=tmp_path)

    async def execute() -> None:
        async def runner(task_id: str) -> str:
            try:
                await asyncio.Event().wait()
            finally:
                events.append("child-finished")
            return task_id

        await runtime.background_tasks.spawn(
            session_id="session-one",
            agent_name="researcher",
            description="work",
            agent_path=("researcher",),
            runner=runner,
        )
        await asyncio.sleep(0)
        await runtime.close()

    asyncio.run(execute())
    assert events == ["child-finished", "trace-flush", "client-close"]


def test_static_graph_teardown_flushes_without_closing_reusable_client(monkeypatch) -> None:
    events: list[str] = []

    class RecordingTracing:
        def flush(self) -> None:
            events.append("flush")

        def close(self) -> None:
            events.append("close")

    tracing = RecordingTracing()
    monkeypatch.setattr(runtime_graph, "_STATIC_LANGSMITH_TRACINGS", {tracing})

    async def execute() -> None:
        await runtime_graph.close_static_background_tasks()
        await runtime_graph.close_static_background_tasks()

    asyncio.run(execute())
    assert events == ["flush", "flush"]


@pytest.mark.filterwarnings("ignore:The v3 streaming protocol.*")
def test_exported_v3_stream_keeps_tracing_and_session_scope() -> None:
    client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    tracing = runtime_tracing.LangSmithTracing(LangSmithConfig(enabled=True), client=client)
    manager = BackgroundTaskManager(BackgroundSubagentConfig(enabled=True))
    generations: list[object] = []
    created: list[object] = []

    async def inspect(state: dict[str, object]) -> dict[str, object]:
        generations.append(current_background_session_generation())
        return state

    builder = StateGraph(dict)
    builder.add_node("inspect", inspect)
    builder.add_edge(START, "inspect")
    builder.add_edge("inspect", END)
    graph = scope_background_session_invocation(
        builder.compile(), manager, run_config_transform=tracing.with_callback
    )

    def record_create(
        self: Client, name: str, inputs: object, run_type: str, **kwargs: object
    ) -> None:
        created.append(kwargs["id"])

    async def execute() -> None:
        run = await graph.astream_events(
            {},
            {
                "configurable": {"thread_id": "static-session"},
                "callbacks": [tracing.new_callback()],
            },
            version="v3",
        )
        await run.output()
        assert generations == [manager.session_generation("static-session")]
        await manager.close()

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", record_create),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        asyncio.run(execute())
    assert created
    assert len(created) == len(set(created))


def test_background_tracing_setup_failure_does_not_rerun_child(monkeypatch) -> None:
    tracing = runtime_tracing.LangSmithTracing(
        LangSmithConfig(enabled=True), client=object()
    )
    calls: list[object] = []

    @contextmanager
    def broken_tracing_context(**kwargs: object):
        raise RuntimeError("trace transport unavailable")
        yield

    monkeypatch.setattr(langsmith, "tracing_context", broken_tracing_context)

    class Child:
        async def ainvoke(self, state: object, config: object) -> object:
            calls.append(state)
            return {"messages": [AIMessage(content="completed")]}

    async def execute() -> None:
        manager = BackgroundTaskManager(BackgroundSubagentConfig(enabled=True))
        spawn = next(
            tool
            for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": Child()},
                agent_path=(),
                recursion_limit=30,
                langsmith_tracing=tracing,
            )
            if tool.name == "spawn_background_task"
        )
        runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "session-one"}},
            stream_writer=lambda _: None,
            tool_call_id="call-1",
            store=None,
        )
        snapshot = await spawn.coroutine("investigate", "researcher", runtime)
        result = await manager.get("session-one", snapshot["task_id"], wait_seconds=1)
        assert result.status == "success"
        assert result.result == "completed"
        await manager.close()

    asyncio.run(execute())
    assert len(calls) == 1


def test_nested_background_task_links_to_immediate_parent_run() -> None:
    client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    tracing = runtime_tracing.LangSmithTracing(LangSmithConfig(enabled=True), client=client)
    manager = BackgroundTaskManager(
        BackgroundSubagentConfig(
            enabled=True, max_running_per_session=4, max_running_total=4
        )
    )
    created: dict[object, dict[str, object]] = {}

    def runtime_for(config: dict[str, object]) -> ToolRuntime:
        return ToolRuntime(
            state={}, context=None, config=config, stream_writer=lambda _: None,
            tool_call_id="nested-call", store=None,
        )

    leaf = RunnableLambda(lambda value: {"messages": [AIMessage(content="leaf done")]})
    nested_spawn = next(
        tool for tool in create_background_task_tools(
            manager=manager, subagents={"leaf": leaf}, agent_path=("manager",),
            recursion_limit=30, langsmith_tracing=tracing,
        ) if tool.name == "spawn_background_task"
    )

    async def parent_body(value: object, config: dict[str, object]) -> object:
        await nested_spawn.coroutine("leaf work", "leaf", runtime_for(config))
        return {"messages": [AIMessage(content="manager done")]}

    parent = RunnableLambda(parent_body)
    main_spawn = next(
        tool for tool in create_background_task_tools(
            manager=manager, subagents={"manager": parent}, agent_path=(),
            recursion_limit=30, langsmith_tracing=tracing,
        ) if tool.name == "spawn_background_task"
    )

    async def root_body(value: object, config: dict[str, object]) -> object:
        await main_spawn.coroutine("manager work", "manager", runtime_for(config))
        return value

    def record_create(
        self: Client, name: str, inputs: object, run_type: str, **kwargs: object
    ) -> None:
        created[kwargs["id"]] = kwargs

    async def execute() -> object:
        await RunnableLambda(root_body).ainvoke(
            "prompt",
            config={
                "callbacks": [tracing.new_callback()],
                "configurable": {"thread_id": "nested-session"},
            },
        )
        snapshots = await manager.wait_session("nested-session")
        await manager.close()
        return snapshots

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", record_create),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        snapshots = asyncio.run(execute())
    assert len(snapshots) == 2
    assert all(snapshot.status == "success" for snapshot in snapshots)
    manager_run = next(
        payload for payload in created.values()
        if payload["extra"]["metadata"].get("background_agent") == "manager"
    )
    leaf_run = next(
        payload for payload in created.values()
        if payload["extra"]["metadata"].get("background_agent") == "leaf"
    )
    assert leaf_run["parent_run_id"] == manager_run["id"]
    assert leaf_run["trace_id"] == manager_run["trace_id"]
    assert leaf_run["extra"]["metadata"]["background_parent_task_id"] == next(
        snapshot.task_id for snapshot in snapshots if snapshot.agent_name == "manager"
    )


def test_background_batches_keep_session_parentage_and_input_order() -> None:
    client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    tracing = runtime_tracing.LangSmithTracing(LangSmithConfig(enabled=True), client=client)
    manager = BackgroundTaskManager(
        BackgroundSubagentConfig(
            enabled=True, batch_result_format="json",
            max_running_per_session=4, max_running_total=8,
        )
    )
    created: dict[object, dict[str, object]] = {}
    child = RunnableLambda(
        lambda state: {"messages": [AIMessage(content=state["messages"][0].content)]}
    )
    batch = next(
        tool for tool in create_background_task_tools(
            manager=manager, subagents={"researcher": child}, agent_path=(),
            recursion_limit=30, langsmith_tracing=tracing,
        ) if tool.name == "run_subagent_batch"
    )

    async def parent_body(session: str, config: dict[str, object]) -> object:
        runtime = ToolRuntime(
            state={}, context=None, config=config, stream_writer=lambda _: None,
            tool_call_id=f"batch-{session}", store=None,
        )
        return await batch.coroutine(
            [
                {"description": f"{session}-first", "subagent_type": "researcher"},
                {"description": f"{session}-second", "subagent_type": "researcher"},
            ],
            runtime,
        )

    def record_create(
        self: Client, name: str, inputs: object, run_type: str, **kwargs: object
    ) -> None:
        created[kwargs["id"]] = kwargs

    async def execute() -> list[object]:
        results = await asyncio.gather(
            *(
                RunnableLambda(parent_body).ainvoke(
                    session,
                    config={
                        "callbacks": [tracing.new_callback()],
                        "configurable": {"thread_id": session},
                    },
                )
                for session in ("session-one", "session-two")
            )
        )
        await manager.close()
        return results

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", record_create),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        results = asyncio.run(execute())

    assert len(results) == 2
    for session, result in zip(("session-one", "session-two"), results, strict=True):
        assert [item["description"] for item in result["results"]] == [
            f"{session}-first", f"{session}-second"
        ]
        parents = [
            payload for payload in created.values()
            if payload["extra"]["metadata"].get("thread_id") == session
            and "background_task_id" not in payload["extra"]["metadata"]
        ]
        assert len(parents) == 1
        children = [
            payload for payload in created.values()
            if payload["extra"]["metadata"].get("session_id") == session
        ]
        assert len(children) == 2
        assert len({payload["id"] for payload in children}) == 2
        assert all(payload["parent_run_id"] == parents[0]["id"] for payload in children)


def test_background_graph_error_keeps_manager_status_and_runs_once() -> None:
    client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    tracing = runtime_tracing.LangSmithTracing(LangSmithConfig(enabled=True), client=client)
    calls = 0

    async def failing_graph(state: object) -> object:
        nonlocal calls
        calls += 1
        raise ValueError("model failed")

    async def execute() -> None:
        manager = BackgroundTaskManager(BackgroundSubagentConfig(enabled=True))
        spawn = next(
            tool for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": RunnableLambda(failing_graph)},
                agent_path=(), recursion_limit=30, langsmith_tracing=tracing,
            ) if tool.name == "spawn_background_task"
        )
        runtime = ToolRuntime(
            state={}, context=None,
            config={"configurable": {"thread_id": "session-one"}},
            stream_writer=lambda _: None, tool_call_id="call-1", store=None,
        )
        task = await spawn.coroutine("investigate", "researcher", runtime)
        result = await manager.get("session-one", task["task_id"], wait_seconds=1)
        assert result.status == "error"
        assert "model failed" in (result.error or "")
        await manager.close()

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", lambda *args, **kwargs: None),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        asyncio.run(execute())
    assert calls == 1


def test_background_trace_preserves_cancellation_status() -> None:
    client = Client(
        api_key="local-test-key", api_url="http://127.0.0.1:1", auto_batch_tracing=False
    )
    tracing = runtime_tracing.LangSmithTracing(LangSmithConfig(enabled=True), client=client)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    calls = 0

    async def blocking_graph(state: object) -> object:
        nonlocal calls
        calls += 1
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def execute() -> None:
        manager = BackgroundTaskManager(BackgroundSubagentConfig(enabled=True))
        spawn = next(
            tool for tool in create_background_task_tools(
                manager=manager,
                subagents={"researcher": RunnableLambda(blocking_graph)},
                agent_path=(), recursion_limit=30, langsmith_tracing=tracing,
            ) if tool.name == "spawn_background_task"
        )
        runtime = ToolRuntime(
            state={}, context=None,
            config={"configurable": {"thread_id": "session-one"}},
            stream_writer=lambda _: None, tool_call_id="call-1", store=None,
        )
        task = await spawn.coroutine("investigate", "researcher", runtime)
        await started.wait()
        result = await manager.cancel("session-one", task["task_id"])
        await cancelled.wait()
        assert result.status == "cancelled"
        await manager.close()

    with (
        patch("requests.Session.request", side_effect=AssertionError("network forbidden")),
        patch.object(Client, "create_run", lambda *args, **kwargs: None),
        patch.object(Client, "update_run", lambda *args, **kwargs: None),
    ):
        asyncio.run(execute())
    assert calls == 1
