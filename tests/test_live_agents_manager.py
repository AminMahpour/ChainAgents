"""Tests for the live agent manager."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from test_live_agents_worker import FakeGraph

from chainagents.runtime.live_agents.config import (
    LiveAgentSpawnError,
    ResolvedLiveAgentSpec,
    SpawnRequest,
)
from chainagents.runtime.live_agents.events import LiveAgentEvent, LiveAgentEventBus
from chainagents.runtime.live_agents.manager import LiveAgentManager
from chainagents.runtime.live_agents.registry import LiveAgentLimitError
from chainagents.runtime.messaging import MessageBroker
from chainagents.runtime.types import (
    LiveAgentsConfig,
    LiveAgentTemplate,
    MessagingConfig,
)

SESSION = "s1"
TASK_PREFIX = "chainagents-live-agent-"


def make_config(**kwargs: Any) -> LiveAgentsConfig:
    return LiveAgentsConfig(
        enabled=True,
        templates=(
            LiveAgentTemplate(
                name="researcher", description="Reads sources.", system_prompt="p"
            ),
            LiveAgentTemplate(
                name="coder", description="Writes code.", system_prompt="c"
            ),
        ),
        **kwargs,
    )


class Harness:
    def __init__(
        self,
        *,
        graphs: list[Any] | None = None,
        delete: bool = False,
        delete_error: bool = False,
        **config: Any,
    ) -> None:
        self.broker = MessageBroker(MessagingConfig(enabled=True))
        self.events = LiveAgentEventBus()
        self.now = 1000.0
        self.graphs = list(graphs or [])
        self.built: list[tuple[ResolvedLiveAgentSpec, str, Any]] = []
        self.deleted: list[str] = []
        self.delete_error = delete_error

        async def delete_thread(thread_id: str) -> None:
            self.deleted.append(thread_id)
            if self.delete_error:
                raise RuntimeError("checkpointer down")

        self.manager = LiveAgentManager(
            make_config(**config),
            self.broker,
            self._factory,
            events=self.events,
            clock=lambda: self.now,
            delete_thread=delete_thread if delete else None,
        )
        self.queue: asyncio.Queue[LiveAgentEvent] | None = None

    async def _factory(self, spec: ResolvedLiveAgentSpec, session_id: str) -> Any:
        graph = self.graphs.pop(0) if self.graphs else FakeGraph()
        if isinstance(graph, BaseException):
            raise graph
        self.built.append((spec, session_id, graph))
        return graph

    def subscribe(self) -> None:
        self.queue = self.events.subscribe(SESSION)

    def drain(self) -> list[LiveAgentEvent]:
        assert self.queue is not None
        items: list[LiveAgentEvent] = []
        while not self.queue.empty():
            items.append(self.queue.get_nowait())
        return items

    def inbox(self, address: str) -> list[str]:
        return [m.body for m in self.broker.pending(SESSION, address)]


async def until(predicate, limit: float = 2.0) -> None:
    async def _poll() -> None:
        while not predicate():  # noqa: ASYNC110 - polling manager state
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), limit)


def live_tasks() -> list[asyncio.Task[Any]]:
    return [
        t
        for t in asyncio.all_tasks()
        if t.get_name().startswith(TASK_PREFIX) and not t.done()
    ]


def blocking_step(started: asyncio.Event) -> Any:
    async def _step() -> str:
        started.set()
        await asyncio.Event().wait()
        return "never"

    return _step


@pytest.mark.anyio
async def test_spawn_opens_address_and_starts_idle() -> None:
    h = Harness()
    h.subscribe()
    record, spec = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))

    assert spec.name == "researcher-1"
    assert record.name == "researcher-1"
    assert record.status == "idle"
    assert record.spawned_by == "main"
    recipients = {r.address: r for r in h.broker.recipients(SESSION, "main")}
    assert recipients[record.address].name == "researcher-1"
    assert recipients[record.address].parent == "main"
    assert [t.get_name() for t in live_tasks()] == [f"{TASK_PREFIX}{record.id}"]
    await asyncio.sleep(0)
    events = h.drain()
    assert [e.kind for e in events] == ["spawned"]
    assert events[0].agent_id == record.id
    assert h.built[0][1] == SESSION
    assert h.manager.get(SESSION, "researcher-1") is record
    payload = h.manager.list(SESSION)
    assert payload[0]["name"] == "researcher-1"
    assert payload[0]["pending"] == 0
    await h.manager.close()
    assert live_tasks() == []


@pytest.mark.anyio
async def test_spawn_initial_message_runs_first_turn() -> None:
    graph = FakeGraph(["findings"])
    h = Harness(graphs=[graph])
    record, _ = await h.manager.spawn(
        SESSION, SpawnRequest(template="researcher", initial_message="look into X")
    )
    await until(lambda: bool(h.inbox("main")))

    assert h.inbox("main") == ["findings"]
    texts = [m.content for m in graph.calls[0][0]["messages"]]
    assert len(texts) == 1
    assert texts[0].startswith("Message from main (id ")
    assert texts[0].endswith("\nlook into X")
    assert record.turns == 1
    await h.manager.close()


@pytest.mark.anyio
async def test_spawn_initial_message_from_main_published_once() -> None:
    h = Harness(graphs=[FakeGraph(["findings"])])
    h.subscribe()
    record, _ = await h.manager.spawn(
        SESSION, SpawnRequest(template="researcher", initial_message="look into X")
    )
    await until(lambda: record.turns == 1)
    await asyncio.sleep(0)
    payloads = [e.payload for e in h.drain() if e.kind == "message"]
    assert payloads == [
        {"from": "main", "to": "researcher-1", "body": "look into X"},
        {"from": "researcher-1", "to": "main", "body": "findings"},
    ]
    await h.manager.close()


@pytest.mark.anyio
async def test_spawn_initial_message_from_user_does_not_leave_user_open() -> None:
    graph = FakeGraph(["hi user"])
    h = Harness(graphs=[graph])
    h.subscribe()
    record, _ = await h.manager.spawn(
        SESSION,
        SpawnRequest(template="researcher", initial_message="hello"),
        spawned_by="user",
    )
    await until(lambda: record.turns == 1)

    assert graph.calls[0][0]["messages"][0].content.startswith("Message from user")
    addresses = {r.address for r in h.broker.recipients(SESSION, "main")}
    assert "user" not in addresses
    await asyncio.sleep(0)
    replies = [e.payload for e in h.drain() if e.kind == "message"]
    assert {"from": "researcher-1", "to": "user", "body": "hi user"} in replies
    await h.manager.close()


@pytest.mark.anyio
async def test_spawn_graph_failure_releases_slot_and_name() -> None:
    h = Harness(graphs=[RuntimeError("build failed")], max_live_per_session=1)
    with pytest.raises(RuntimeError, match="build failed"):
        await h.manager.spawn(SESSION, SpawnRequest(template="researcher", name="x"))

    assert h.manager.list(SESSION, include_stopped=True) == []
    assert live_tasks() == []
    record, _ = await h.manager.spawn(
        SESSION, SpawnRequest(template="researcher", name="x")
    )
    assert record.name == "x"
    await h.manager.close()


@pytest.mark.anyio
async def test_spawn_beyond_session_limit_raises_limit_error() -> None:
    h = Harness(max_live_per_session=1)
    await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    with pytest.raises(LiveAgentLimitError):
        await h.manager.spawn(SESSION, SpawnRequest(template="coder"))
    assert len(h.built) == 1
    assert len(live_tasks()) == 1
    await h.manager.close()


@pytest.mark.anyio
async def test_spawn_rejects_invalid_request() -> None:
    h = Harness()
    with pytest.raises(LiveAgentSpawnError):
        await h.manager.spawn(SESSION, SpawnRequest(template="missing"))
    assert h.built == []
    await h.manager.close()


@pytest.mark.anyio
async def test_teardown_mid_turn_cancels_and_marks_undeliverable() -> None:
    started = asyncio.Event()
    h = Harness(graphs=[FakeGraph([blocking_step(started)])])
    h.subscribe()
    record, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    first = h.broker.send(SESSION, "main", record.address, "do the thing")
    await asyncio.wait_for(started.wait(), 2)
    assert record.status == "running"
    second = h.broker.send(SESSION, "main", record.address, "and another")
    assert h.manager.list(SESSION)[0]["pending"] == 1

    result = await h.manager.teardown(SESSION, record.name)

    assert result is record
    assert record.status == "stopped"
    assert record.stop_reason == "requested"
    assert first.status == "undeliverable"
    assert second.status == "undeliverable"
    assert live_tasks() == []
    assert not any(t.get_name().startswith(TASK_PREFIX) for t in asyncio.all_tasks())
    await asyncio.sleep(0)
    stopped = [e for e in h.drain() if e.kind == "stopped"]
    assert len(stopped) == 1
    assert stopped[0].payload == {"reason": "requested"}
    # Idempotent for an already-stopped record.
    again = await h.manager.teardown(SESSION, record.id)
    assert again is record
    await asyncio.sleep(0)
    assert [e for e in h.drain() if e.kind == "stopped"] == []
    await h.manager.close()


@pytest.mark.anyio
async def test_send_after_teardown_is_rejected() -> None:
    h = Harness()
    record, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    await h.manager.teardown(SESSION, record.name)

    with pytest.raises(ValueError, match="Recipient is unavailable"):
        h.broker.send(SESSION, "main", record.name, "are you there?")
    with pytest.raises(ValueError, match="Recipient is unavailable"):
        h.manager.send_from_user(SESSION, record.id, "hello?")
    await h.manager.close()


@pytest.mark.anyio
async def test_teardown_deletes_history_when_configured() -> None:
    h = Harness(delete=True, delete_error=True)
    record, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    await h.manager.teardown(SESSION, record.name, reason="done")

    assert h.deleted == [record.thread_id]
    assert record.status == "stopped"
    assert record.stop_reason == "done"
    await h.manager.close()


@pytest.mark.anyio
async def test_teardown_keeps_history_when_disabled() -> None:
    h = Harness(delete=True, delete_history_on_teardown=False)
    record, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    await h.manager.teardown(SESSION, record.name)

    assert h.deleted == []
    assert record.status == "stopped"
    await h.manager.close()


@pytest.mark.anyio
async def test_teardown_session_stops_all() -> None:
    h = Harness()
    a, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    b, _ = await h.manager.spawn(SESSION, SpawnRequest(template="coder"))
    other, _ = await h.manager.spawn("s2", SpawnRequest(template="coder"))

    await h.manager.teardown_session(SESSION)

    assert a.status == "stopped"
    assert b.status == "stopped"
    assert h.manager.list(SESSION, include_stopped=True) == []
    assert {r.address for r in h.broker.recipients(SESSION, "main")} == set()
    assert other.status == "idle"
    assert [t.get_name() for t in live_tasks()] == [f"{TASK_PREFIX}{other.id}"]
    await h.manager.close()
    assert other.status == "stopped"
    assert live_tasks() == []


@pytest.mark.anyio
async def test_reap_idle_uses_timeout() -> None:
    h = Harness(idle_timeout_seconds=10)
    record, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    h.now += 9
    assert await h.manager.reap_idle() == []
    h.now += 1
    assert await h.manager.reap_idle() == [record]
    assert record.status == "stopped"
    assert record.stop_reason == "idle_timeout"
    await h.manager.close()

    disabled = Harness(idle_timeout_seconds=0)
    kept, _ = await disabled.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    disabled.now += 1_000_000
    assert await disabled.manager.reap_idle() == []
    assert kept.status == "idle"
    await disabled.manager.close()


@pytest.mark.anyio
async def test_reap_idle_skips_running_agents() -> None:
    started = asyncio.Event()
    h = Harness(graphs=[FakeGraph([blocking_step(started)])], idle_timeout_seconds=5)
    record, _ = await h.manager.spawn(
        SESSION, SpawnRequest(template="researcher", initial_message="go")
    )
    await asyncio.wait_for(started.wait(), 2)
    h.now += 100
    assert await h.manager.reap_idle() == []
    assert record.status == "running"
    await h.manager.close()
    assert record.status == "stopped"


@pytest.mark.anyio
async def test_max_turns_tears_down_with_reason() -> None:
    h = Harness(graphs=[FakeGraph(["final answer"])], max_turns_per_agent=1)
    h.subscribe()
    record, _ = await h.manager.spawn(
        SESSION, SpawnRequest(template="researcher", initial_message="one job")
    )
    await until(lambda: record.status == "stopped")

    assert record.stop_reason == "max_turns"
    assert h.inbox("main") == ["final answer"]
    assert live_tasks() == []
    await asyncio.sleep(0)
    stopped = [e for e in h.drain() if e.kind == "stopped"]
    assert [e.payload for e in stopped] == [{"reason": "max_turns"}]
    await h.manager.close()


@pytest.mark.anyio
async def test_three_failures_tear_down_as_failed() -> None:
    graph = FakeGraph(default=RuntimeError("boom"))
    h = Harness(graphs=[graph], max_live_per_session=1)
    record, _ = await h.manager.spawn(
        SESSION, SpawnRequest(template="researcher", name="r")
    )
    for _ in range(3):
        turns = record.turns
        h.broker.send(SESSION, "main", record.address, "try")
        await until(lambda turns=turns: record.turns == turns + 1)

    await until(lambda: record.stop_reason is not None)
    assert record.status == "failed"
    assert record.stop_reason == "failed"
    assert live_tasks() == []
    # Failed is terminal: it frees the slot and the name and leaves the roster.
    assert h.manager.roster(SESSION) == ""
    assert h.manager.list(SESSION) == []
    assert h.manager.list(SESSION, include_stopped=True)[0]["status"] == "failed"
    assert await h.manager.teardown(SESSION, record.id) is record
    assert record.status == "failed"
    replacement, _ = await h.manager.spawn(
        SESSION, SpawnRequest(template="researcher", name="r")
    )
    assert h.manager.get(SESSION, "r") is replacement
    await h.manager.close()
    assert record.status == "failed"


@pytest.mark.anyio
async def test_worker_exit_without_teardown_is_finalized() -> None:
    h = Harness()
    h.subscribe()
    record, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    await asyncio.sleep(0.01)
    h.broker.close_session(SESSION)

    await until(lambda: record.status == "stopped")
    assert record.stop_reason == "worker_exited"
    assert live_tasks() == []
    await asyncio.sleep(0)
    stopped = [e for e in h.drain() if e.kind == "stopped"]
    assert [e.payload for e in stopped] == [{"reason": "worker_exited"}]
    await h.manager.close()


@pytest.mark.anyio
async def test_worker_crash_is_logged_and_finalized_as_failed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    h = Harness()

    def broken_claim(session_id: str, address: str) -> list[Any]:
        raise RuntimeError("claim exploded")

    monkeypatch.setattr(h.broker, "claim", broken_claim)
    record, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    h.broker.send(SESSION, "main", record.address, "hi")

    await until(lambda: record.stop_reason is not None)
    assert record.status == "failed"
    assert record.stop_reason == "failed"
    assert "claim exploded" in caplog.text
    await h.manager.close()


@pytest.mark.anyio
async def test_send_from_user_reply_published_as_event() -> None:
    h = Harness(graphs=[FakeGraph(["answer for user"])])
    h.subscribe()
    record, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    message = h.manager.send_from_user(SESSION, "researcher-1", "what is up?")

    assert message.sender == "user"
    assert message.recipient == record.address
    await until(lambda: record.turns == 1)
    await asyncio.sleep(0)
    payloads = [e.payload for e in h.drain() if e.kind == "message"]
    assert payloads == [
        {"from": "user", "to": "researcher-1", "body": "what is up?"},
        {"from": "researcher-1", "to": "user", "body": "answer for user"},
    ]
    assert "user" not in {r.address for r in h.broker.recipients(SESSION, "main")}
    await h.manager.close()


@pytest.mark.anyio
async def test_send_from_user_keeps_existing_user_participant() -> None:
    h = Harness()
    h.broker.open(SESSION, "user", name="user", parent=None)
    record, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    h.manager.send_from_user(SESSION, record.name, "hello")

    assert "user" in {r.address for r in h.broker.recipients(SESSION, "main")}
    await h.manager.close()


@pytest.mark.anyio
async def test_roster_lists_live_agents() -> None:
    started = asyncio.Event()
    h = Harness(graphs=[FakeGraph(), FakeGraph([blocking_step(started)])])
    assert h.manager.roster(SESSION) == ""
    r1, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    await h.manager.spawn(
        SESSION, SpawnRequest(template="coder", initial_message="build it")
    )
    await asyncio.wait_for(started.wait(), 2)

    assert (
        h.manager.roster(SESSION) == "Live agents: researcher-1: idle, coder-2: running"
    )
    await h.manager.teardown(SESSION, r1.name)
    assert h.manager.roster(SESSION) == "Live agents: coder-2: running"
    await h.manager.close()


@pytest.mark.anyio
async def test_templates_payload() -> None:
    h = Harness()
    payload = h.manager.templates_payload()
    assert [t["name"] for t in payload["templates"]] == ["researcher", "coder"]
    assert payload["allow"]["applies_to"] == "all"
    assert payload["allow"]["allow_custom_prompt"] is True
    await h.manager.close()


@pytest.mark.anyio
async def test_start_is_idempotent_and_lazy() -> None:
    h = Harness(idle_timeout_seconds=10)
    assert h.manager._reaper is None
    await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    reaper = h.manager._reaper
    assert reaper is not None
    h.manager.start()
    assert h.manager._reaper is reaper
    await h.manager.close()
    assert reaper.done()
    assert h.manager._reaper is None


def gated_factory(h: Harness) -> tuple[asyncio.Event, asyncio.Event]:
    """Make the graph build block; return (building, release) events."""
    building, release = asyncio.Event(), asyncio.Event()
    original = h._factory

    async def slow_factory(spec: ResolvedLiveAgentSpec, session_id: str) -> Any:
        building.set()
        await release.wait()
        return await original(spec, session_id)

    h.manager._graph_factory = slow_factory
    return building, release


@pytest.mark.anyio
async def test_teardown_session_during_spawn_leaves_no_worker() -> None:
    h = Harness()
    building, release = gated_factory(h)
    spawn = asyncio.create_task(
        h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    )
    await building.wait()
    await h.manager.teardown_session(SESSION)
    release.set()

    with pytest.raises(RuntimeError, match="stopped during spawn"):
        await spawn
    assert live_tasks() == []
    assert h.manager.list(SESSION, include_stopped=True) == []
    await h.manager.close()


@pytest.mark.anyio
async def test_spawn_reserved_during_teardown_session_does_not_start_worker() -> None:
    h = Harness(delete=True)
    existing, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    deleting, gate = asyncio.Event(), asyncio.Event()

    async def slow_delete(thread_id: str) -> None:
        deleting.set()
        await gate.wait()

    h.manager._delete_thread = slow_delete
    closing = asyncio.create_task(h.manager.teardown_session(SESSION))
    await deleting.wait()  # teardown_session is mid-await on the snapshot

    with pytest.raises(RuntimeError, match="shutting down"):
        await h.manager.spawn(SESSION, SpawnRequest(template="coder"))
    gate.set()
    await closing

    assert existing.status == "stopped"
    assert live_tasks() == []
    assert h.manager.list(SESSION, include_stopped=True) == []
    # The session accepts spawns again once its teardown has finished.
    again, _ = await h.manager.spawn(SESSION, SpawnRequest(template="coder"))
    assert h.manager.get(SESSION, again.name) is again
    await h.manager.close()
    assert live_tasks() == []


@pytest.mark.anyio
async def test_spawn_building_during_close_does_not_start_worker() -> None:
    h = Harness()
    building, release = gated_factory(h)
    spawn = asyncio.create_task(
        h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    )
    await building.wait()
    await h.manager.close()
    release.set()

    with pytest.raises(RuntimeError, match="stopped during spawn"):
        await spawn
    assert live_tasks() == []
    assert h.manager.list(SESSION, include_stopped=True) == []
    with pytest.raises(RuntimeError, match="manager is closed"):
        await h.manager.spawn(SESSION, SpawnRequest(template="coder"))
    assert live_tasks() == []


@pytest.mark.anyio
async def test_teardown_bounds_a_hung_history_delete(monkeypatch, caplog) -> None:
    import chainagents.runtime.live_agents.manager as manager_module

    monkeypatch.setattr(manager_module, "DELETE_THREAD_TIMEOUT_SECONDS", 0.05)
    h = Harness()
    hung = asyncio.Event()

    async def delete_thread(_thread_id: str) -> None:
        await hung.wait()  # Never set: a hung checkpointer.

    h.manager._delete_thread = delete_thread
    record, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    with caplog.at_level("WARNING", logger=manager_module.__name__):
        await asyncio.wait_for(
            h.manager.teardown(SESSION, record.name, reason="done"), 2
        )
    assert record.status == "stopped"
    assert "Timed out deleting history" in caplog.text
    await h.manager.close()


def test_publish_drops_subscriber_whose_loop_closed() -> None:
    bus = LiveAgentEventBus()

    class ClosingLoop:
        def is_closed(self) -> bool:
            return False  # Closes between the check and the call.

        def call_soon_threadsafe(self, *_args: Any) -> None:
            raise RuntimeError("Event loop is closed")

    class OpenLoop:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def is_closed(self) -> bool:
            return False

        def call_soon_threadsafe(self, callback: Any, event: Any) -> None:
            self.calls.append(event)

    stale_queue: asyncio.Queue[LiveAgentEvent] = asyncio.Queue()
    live_queue: asyncio.Queue[LiveAgentEvent] = asyncio.Queue()
    open_loop = OpenLoop()
    bus._subscribers[SESSION] = [
        (stale_queue, ClosingLoop()),  # type: ignore[list-item]
        (live_queue, open_loop),  # type: ignore[list-item]
    ]
    event = LiveAgentEvent(
        kind="spawned", session_id=SESSION, agent_id="a", agent_name="a", payload={}
    )
    bus.publish(event)
    assert open_loop.calls == [event]
    assert [queue for queue, _ in bus._subscribers[SESSION]] == [live_queue]


@pytest.mark.anyio
async def test_notify_main_fyi_is_not_user_steering() -> None:
    from types import SimpleNamespace

    from chainagents.runtime.live_agents.manager import LIVE_AGENTS_NOTICE_ADDRESS
    from chainagents.runtime.messaging import AgentMessageMiddleware

    h = Harness()
    h.broker.open(SESSION, "main", name="main", parent=None)
    h.manager.notify_main(SESSION, "FYI: user spawned `x`.")

    pending = h.broker.pending(SESSION, "main")
    assert [m.sender for m in pending] == [LIVE_AGENTS_NOTICE_ADDRESS]
    assert LIVE_AGENTS_NOTICE_ADDRESS == "live-agents"
    assert h.broker.pending_user_steering_count(SESSION) == 0
    assert h.broker.take_late_user_steering(SESSION) == []
    assert h.inbox("main") == ["FYI: user spawned `x`."]
    # The notice participant is not left open as a recipient agents could use.
    assert LIVE_AGENTS_NOTICE_ADDRESS not in {
        r.address for r in h.broker.recipients(SESSION, "main")
    }

    middleware = AgentMessageMiddleware(h.broker, fixed_session_id=SESSION)
    update = middleware.before_model({}, SimpleNamespace(config={}))
    assert update is not None
    content = update["messages"][0].content
    assert content.startswith("Message from live-agents (id ")
    assert content.endswith(":\nFYI: user spawned `x`.")
    await h.manager.close()


@pytest.mark.anyio
async def test_slow_graph_build_does_not_make_agent_instantly_reapable() -> None:
    h = Harness(idle_timeout_seconds=10)
    building, release = gated_factory(h)
    spawn = asyncio.create_task(
        h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
    )
    await building.wait()
    h.now += 15  # The build takes longer than the idle timeout.
    release.set()
    record, _ = await spawn
    assert record.last_active_at == h.now
    assert await h.manager.reap_idle() == []
    assert record.status == "idle"
    h.now += 10
    assert await h.manager.reap_idle() == [record]
    await h.manager.close()


@pytest.mark.anyio
async def test_terminal_records_are_capped_per_session(monkeypatch) -> None:
    from chainagents.runtime.live_agents import registry as registry_module

    assert registry_module.MAX_RETAINED_TERMINAL_RECORDS == 50
    monkeypatch.setattr(registry_module, "MAX_RETAINED_TERMINAL_RECORDS", 3)
    h = Harness()
    keeper, _ = await h.manager.spawn(SESSION, SpawnRequest(template="coder"))
    stopped = []
    for _ in range(5):
        record, _ = await h.manager.spawn(SESSION, SpawnRequest(template="researcher"))
        h.now += 1
        await h.manager.teardown(SESSION, record.id)
        stopped.append(record)

    rows = h.manager.list(SESSION, include_stopped=True)
    assert [r["id"] for r in rows] == [keeper.id] + [r.id for r in stopped[-3:]]
    assert keeper.status == "idle"  # Live records are never evicted.
    for evicted in stopped[:2]:
        assert evicted.id not in h.manager._teardowns
        with pytest.raises(KeyError):
            h.manager.get(SESSION, evicted.id)
    assert all(r.id in h.manager._teardowns for r in stopped[-3:])
    await h.manager.close()
