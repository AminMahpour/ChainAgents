"""Chainlit live agents side panel: props, throttled rendering, actions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from chainagents.interfaces.chainlit import app
from chainagents.interfaces.chainlit import live_agents as panel_module
from chainagents.interfaces.chainlit.live_agents import (
    THREAD_HISTORY_LIMIT,
    LiveAgentsPanel,
    build_panel_props,
)
from chainagents.runtime.live_agents.events import LiveAgentEvent, LiveAgentEventBus


def _agent(agent_id: str, name: str, **extra: object) -> dict[str, object]:
    return {
        "id": agent_id,
        "name": name,
        "template": "researcher",
        "model": "fake",
        "status": "idle",
        "turns": 0,
        **extra,
    }


def _entry(index: int) -> dict[str, object]:
    return {"from": "user", "to": "scout", "body": f"m{index}", "at": f"t{index}"}


def test_build_panel_props_merges_threads() -> None:
    agents = [_agent("a1", "scout"), _agent("a2", "critic", status="running")]
    threads = {"a1": [_entry(1), _entry(2)]}

    props = build_panel_props(agents, threads)

    assert props == {
        "agents": [
            {**agents[0], "thread": [_entry(1), _entry(2)]},
            {**agents[1], "thread": []},
        ]
    }
    # Pure: inputs are not mutated.
    assert "thread" not in agents[0]


def test_build_panel_props_caps_thread_length() -> None:
    entries = [_entry(index) for index in range(THREAD_HISTORY_LIMIT + 20)]

    props = build_panel_props([_agent("a1", "scout")], {"a1": entries})

    thread = props["agents"][0]["thread"]  # type: ignore[index]
    assert THREAD_HISTORY_LIMIT == 50
    assert len(thread) == 50
    assert thread[0]["body"] == "m20"
    assert thread[-1]["body"] == f"m{THREAD_HISTORY_LIMIT + 19}"


class _FakeElement:
    instances: ClassVar[list[_FakeElement]] = []

    def __init__(self, *, name: str, props: dict[str, Any], display: str) -> None:
        self.name = name
        self.props = props
        self.display = display
        self.for_id = None
        self.updates: list[dict[str, Any]] = []
        _FakeElement.instances.append(self)

    def to_dict(self) -> dict[str, Any]:
        return {"element": self, "props": self.props}


async def _record_element_update(element_dict: dict[str, Any]) -> None:
    element_dict["element"].updates.append(element_dict["props"])


_FAKE_CONTEXT = SimpleNamespace(
    emitter=SimpleNamespace(send_element=_record_element_update)
)


class _FakeSidebar:
    def __init__(self) -> None:
        self.elements: list[list[Any]] = []
        self.titles: list[str] = []

    async def set_elements(self, elements: list[Any], key: str | None = None) -> None:
        self.elements.append(list(elements))

    async def set_title(self, title: str) -> None:
        self.titles.append(title)


def _event(kind: str, payload: dict[str, object] | None = None) -> LiveAgentEvent:
    return LiveAgentEvent(
        kind=kind,  # type: ignore[arg-type]
        session_id="thread-1",
        agent_id="a1",
        agent_name="scout",
        payload=payload or {},
    )


@pytest.mark.anyio
async def test_panel_throttles_renders(monkeypatch) -> None:
    _FakeElement.instances = []
    sidebar = _FakeSidebar()
    monkeypatch.setattr(panel_module.cl, "CustomElement", _FakeElement)
    monkeypatch.setattr(panel_module.cl, "context", _FAKE_CONTEXT)
    monkeypatch.setattr(panel_module.cl, "ElementSidebar", sidebar)
    bus = LiveAgentEventBus()
    rows: list[dict[str, object]] = []
    manager = SimpleNamespace(
        events=bus,
        list=lambda session_id, include_stopped=False: list(rows),
    )
    panel = LiveAgentsPanel(manager, "thread-1", min_interval=0.25)  # type: ignore[arg-type]
    await panel.start()
    try:
        await asyncio.sleep(0.02)
        assert _FakeElement.instances == []  # Nothing to show before a spawn.
        rows.append(_agent("a1", "scout"))
        bus.publish(_event("spawned"))
        await asyncio.sleep(0.02)
        bus.publish(_event("message", {"from": "main", "to": "scout", "body": "hi"}))
        bus.publish(_event("message", {"from": "scout", "to": "main", "body": "yo"}))
        await asyncio.sleep(0.1)
        # First event renders immediately; the rest are coalesced.
        assert len(_FakeElement.instances) == 1
        element = _FakeElement.instances[0]
        assert element.name == "LiveAgentsPanel"
        assert element.display == "side"
        assert sidebar.elements == [[element]]
        assert sidebar.titles == ["Live agents"]
        assert element.updates == []

        await asyncio.sleep(0.35)
        # Exactly one trailing render carrying the latest state.
        assert len(element.updates) == 1
        thread = element.props["agents"][0]["thread"]
        assert [entry["body"] for entry in thread] == ["hi", "yo"]
        assert all(entry["at"] for entry in thread)
    finally:
        await panel.aclose()
    assert bus._subscribers == {}


@pytest.mark.anyio
async def test_panel_streams_turn_activity_as_steps(monkeypatch) -> None:
    _FakeElement.instances = []
    monkeypatch.setattr(panel_module.cl, "CustomElement", _FakeElement)
    monkeypatch.setattr(panel_module.cl, "context", _FAKE_CONTEXT)
    monkeypatch.setattr(panel_module.cl, "ElementSidebar", _FakeSidebar())
    steps: list[Any] = []

    class Step:
        def __init__(self, *, name: str, type: str) -> None:
            self.name = name
            self.type = type
            self.output = ""
            self.sent = False
            self.updated = False
            steps.append(self)

        async def send(self) -> Step:
            self.sent = True
            return self

        async def update(self) -> None:
            self.updated = True

    monkeypatch.setattr(panel_module.cl, "Step", Step)
    bus = LiveAgentEventBus()
    manager = SimpleNamespace(
        events=bus,
        list=lambda session_id, include_stopped=False: [_agent("a1", "scout")],
    )
    panel = LiveAgentsPanel(  # type: ignore[arg-type]
        manager, "thread-1", min_interval=0.0, stream_activity=True
    )
    await panel.start()
    try:
        bus.publish(_event("turn_started"))
        bus.publish(_event("turn_finished", {"turns": 1, "ok": True}))
        await asyncio.sleep(0.05)
    finally:
        await panel.aclose()
    assert len(steps) == 1
    assert steps[0].name == "live:scout"
    assert steps[0].type == "run"
    assert steps[0].sent and steps[0].updated


class _Session:
    def __init__(self, values: dict[str, Any]) -> None:
        self.values = values

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


@pytest.mark.anyio
async def test_action_callbacks_call_manager(monkeypatch) -> None:
    calls: list[tuple[Any, ...]] = []
    notes: list[str] = []

    class Manager:
        config = SimpleNamespace(notify_main_on_user_actions=True)

        def get(self, session_id: str, ref: str) -> Any:
            if ref == "ghost":
                raise KeyError(ref)
            return SimpleNamespace(id="a1", name="scout")

        def send_from_user(self, session_id: str, ref: str, body: str) -> None:
            calls.append(("send", session_id, ref, body))

        async def teardown(self, session_id: str, ref: str, *, reason: str) -> None:
            calls.append(("teardown", session_id, ref, reason))

        def notify_main(self, session_id: str, text: str) -> None:
            notes.append(text)

    runtime = SimpleNamespace(
        live_agents=Manager(),
        config=SimpleNamespace(model_name="m", model_choices=("m",)),
    )

    async def get_runtime():
        return runtime

    sent: list[str] = []

    class Message:
        def __init__(self, content: str, **_kwargs: Any) -> None:
            self.content = content

        async def send(self) -> None:
            sent.append(self.content)

    monkeypatch.setattr(app, "get_runtime_or_notify", get_runtime)
    monkeypatch.setattr(
        app.cl,
        "user_session",
        _Session({app.SESSION_SETTINGS_KEY: {"model": "m", "thread_id": "thread-1"}}),
    )
    monkeypatch.setattr(app.cl, "Message", Message)

    await app.live_agent_message_action(
        app.cl.Action(
            name=app.LIVE_AGENT_MESSAGE_ACTION,
            payload={"agent": "a1", "body": "status?"},
        )
    )
    await app.live_agent_kill_action(
        app.cl.Action(name=app.LIVE_AGENT_KILL_ACTION, payload={"agent": "a1"})
    )
    await app.live_agent_kill_action(
        app.cl.Action(name=app.LIVE_AGENT_KILL_ACTION, payload={"agent": "ghost"})
    )

    assert calls == [
        ("send", "thread-1", "a1", "status?"),
        ("teardown", "thread-1", "a1", "user"),
    ]
    assert notes == [
        "FYI: user sent a message to `scout`: status?",
        "FYI: user stopped `scout`.",
    ]
    assert len(sent) == 1 and "ghost" in sent[0]
    assert app.LIVE_AGENT_MESSAGE_ACTION == "live_agent_message"
    assert app.LIVE_AGENT_KILL_ACTION == "live_agent_kill"


@pytest.mark.anyio
async def test_panel_rerender_does_not_persist_element(monkeypatch) -> None:
    import chainlit.element as chainlit_element

    persisted: list[str] = []
    created: list[Any] = []
    emitted: list[dict[str, Any]] = []

    async def persist_file(**kwargs: Any) -> dict[str, str]:
        persisted.append(kwargs["name"])
        return {"id": f"file-{len(persisted)}"}

    async def send_element(element_dict: dict[str, Any]) -> None:
        emitted.append(element_dict)

    async def create_element(element: Any) -> None:
        created.append(element)

    fake_context = SimpleNamespace(
        session=SimpleNamespace(persist_file=persist_file, thread_id="thread-1"),
        emitter=SimpleNamespace(send_element=send_element),
    )
    monkeypatch.setattr(chainlit_element, "context", fake_context)
    monkeypatch.setattr(panel_module.cl, "context", fake_context)
    monkeypatch.setattr(
        chainlit_element,
        "get_data_layer",
        lambda: SimpleNamespace(create_element=create_element),
    )
    monkeypatch.setattr(panel_module.cl, "ElementSidebar", _FakeSidebar())
    bus = LiveAgentEventBus()
    manager = SimpleNamespace(
        events=bus,
        list=lambda session_id, include_stopped=False: [_agent("a1", "scout")],
    )
    panel = LiveAgentsPanel(manager, "thread-1", min_interval=0.0)  # type: ignore[arg-type]
    await panel._render()  # Opens the sidebar.
    await panel._handle(_event("message", {"from": "user", "to": "scout", "body": "hi"}))
    await panel._render()
    await panel._render()
    await asyncio.sleep(0)
    assert persisted == []
    assert created == []
    assert len(emitted) >= 2
    assert emitted[-1]["props"]["agents"][0]["thread"][0]["body"] == "hi"
    await panel.aclose()


def _panel_runtime(bus: LiveAgentEventBus) -> Any:
    manager = SimpleNamespace(
        events=bus,
        list=lambda session_id, include_stopped=False: [],
        config=SimpleNamespace(stream_activity=False),
    )
    extensions = SimpleNamespace(
        background_subagents=SimpleNamespace(enabled=False),
        async_subagents=(),
        mcp_stateful=True,
        chainlit_live_agents_panel_enabled=True,
        chainlit_reasoning_steps_enabled=False,
        chainlit_tool_steps_enabled=False,
    )
    config = SimpleNamespace(
        extensions=extensions, model_name="m", model_choices=("m",)
    )
    return SimpleNamespace(live_agents=manager, config=config)


@pytest.mark.anyio
async def test_reconnect_restore_restarts_closed_live_agents_panel(monkeypatch) -> None:
    bus = LiveAgentEventBus()
    runtime = _panel_runtime(bus)
    old_panel = LiveAgentsPanel(runtime.live_agents, "thread-1")  # type: ignore[arg-type]
    await old_panel.start()
    await old_panel.aclose()  # Closed by the disconnect cleanup.
    session_values: dict[str, Any] = {app.SESSION_LIVE_AGENTS_PANEL_KEY: old_panel}
    monkeypatch.setattr(app.cl, "user_session", _Session(session_values))
    monkeypatch.setattr(app.AgentRuntime, "current", staticmethod(lambda: runtime))
    monkeypatch.setattr(
        app,
        "coerce_settings",
        lambda *_args, **_kwargs: SimpleNamespace(
            thread_id="thread-1", show_reasoning_stream=False, show_tool_calls=False
        ),
    )

    async def lease(**_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(app.conversation_scopes, "lease", lease)
    recovery = app._ReconnectRecovery(
        disconnected_socket_id="old",
        runtime=runtime,  # type: ignore[arg-type]
        raw_settings={"thread_id": "thread-1"},
        async_notifier=None,
        local_notifier=None,
    )
    await app._restore_reconnected_observers(SimpleNamespace(), recovery)
    restarted = session_values[app.SESSION_LIVE_AGENTS_PANEL_KEY]
    try:
        assert isinstance(restarted, LiveAgentsPanel)
        assert restarted is not old_panel
        assert restarted.is_running
        assert not old_panel.is_running
        assert len(bus._subscribers["thread-1"]) == 1
    finally:
        await restarted.aclose()


@pytest.mark.anyio
async def test_closed_panel_is_not_treated_as_current(monkeypatch) -> None:
    bus = LiveAgentEventBus()
    runtime = _panel_runtime(bus)
    panel = LiveAgentsPanel(runtime.live_agents, "thread-1")  # type: ignore[arg-type]
    monkeypatch.setattr(
        app.cl, "user_session", _Session({app.SESSION_LIVE_AGENTS_PANEL_KEY: panel})
    )
    await panel.start()
    assert app._live_agents_panel_is_current("thread-1")
    assert not app._live_agents_panel_is_current("thread-2")
    await panel.aclose()
    assert not app._live_agents_panel_is_current("thread-1")
