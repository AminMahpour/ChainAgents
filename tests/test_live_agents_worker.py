"""Tests for the live agent mailbox worker."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from chainagents.runtime.live_agents.events import LiveAgentEvent, LiveAgentEventBus
from chainagents.runtime.live_agents.registry import LiveAgentRecord
from chainagents.runtime.live_agents.worker import (
    MAX_CONSECUTIVE_FAILURES,
    NO_REPLY_TEXT,
    LiveAgentWorker,
)
from chainagents.runtime.messaging import (
    _CURRENT_AGENT_ADDRESS,
    _CURRENT_MESSAGING_SESSION,
    MessageBroker,
)
from chainagents.runtime.types import MessagingConfig

SESSION = "s1"
ADDRESS = "live:helper:abc123"


class FakeGraph:
    """Scripted graph: each entry is a reply string, an exception, or a callable."""

    def __init__(self, script: list[Any] | None = None, default: Any = "ok") -> None:
        self.script = list(script or [])
        self.default = default
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.observed: list[tuple[str, str | None]] = []

    async def ainvoke(self, input: dict[str, Any], config: dict[str, Any]) -> Any:
        self.calls.append((input, config))
        self.observed.append(
            (_CURRENT_AGENT_ADDRESS.get(), _CURRENT_MESSAGING_SESSION.get())
        )
        step = self.script.pop(0) if self.script else self.default
        if callable(step):
            step = await step()
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, AIMessage):
            return {"messages": [step]}
        return {"messages": [AIMessage(content=step)]}


def make_record() -> LiveAgentRecord:
    return LiveAgentRecord(
        id="abc123",
        session_id=SESSION,
        name="helper",
        template="researcher",
        model=None,
        address=ADDRESS,
        thread_id=f"live-{SESSION}-abc123",
        status="idle",
        created_at=0.0,
        last_active_at=0.0,
        spawned_by="main",
    )


class Harness:
    def __init__(self, graph: FakeGraph, *, max_turns: int = 50) -> None:
        self.broker = MessageBroker(MessagingConfig(enabled=True))
        self.broker.open(SESSION, "main", name="main", parent=None)
        self.broker.open(SESSION, ADDRESS, name="helper", parent="main")
        self.events = LiveAgentEventBus()
        self.record = make_record()
        self.graph = graph
        self.finished: list[str] = []
        self.now = 100.0
        self.worker = LiveAgentWorker(
            self.record,
            graph,
            self.broker,
            self.events,
            recursion_limit=7,
            max_turns=max_turns,
            clock=lambda: self.now,
            on_finished=self._on_finished,
        )
        self.queue: asyncio.Queue[LiveAgentEvent] | None = None
        self.task: asyncio.Task[None] | None = None

    async def _on_finished(self, record: LiveAgentRecord, reason: str) -> None:
        self.finished.append(reason)

    def start(self) -> None:
        self.queue = self.events.subscribe(SESSION)
        self.task = asyncio.create_task(self.worker.run())

    async def stop(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await self.task

    def drain_events(self) -> list[LiveAgentEvent]:
        assert self.queue is not None
        items: list[LiveAgentEvent] = []
        while not self.queue.empty():
            items.append(self.queue.get_nowait())
        return items

    def inbox(self, address: str) -> list[str]:
        return [m.body for m in self.broker.pending(SESSION, address)]


async def run_until(harness: Harness, predicate, limit: float = 2.0) -> None:
    async def _poll() -> None:
        while not predicate():  # noqa: ASYNC110 - polling broker/record state
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), limit)


def human_texts(call: tuple[dict[str, Any], dict[str, Any]]) -> list[str]:
    return [m.content for m in call[0]["messages"]]


@pytest.mark.anyio
async def test_message_triggers_one_turn_and_reply_reaches_sender() -> None:
    h = Harness(FakeGraph(["hello back"]))
    h.start()
    sent = h.broker.send(SESSION, "main", ADDRESS, "hi there")
    await run_until(h, lambda: bool(h.inbox("main")))
    await h.stop()

    assert len(h.graph.calls) == 1
    assert human_texts(h.graph.calls[0]) == [
        f"Message from main (id {sent.id}):\nhi there"
    ]
    assert h.graph.calls[0][1]["recursion_limit"] == 7
    replies = h.broker.pending(SESSION, "main")
    assert [(r.sender, r.body) for r in replies] == [(ADDRESS, "hello back")]
    assert h.broker.get(SESSION, sent.id).status == "delivered"
    assert h.record.turns == 1
    assert h.record.status == "idle"
    assert h.record.last_active_at == 100.0
    kinds = [e.kind for e in h.drain_events()]
    assert kinds == ["message", "turn_started", "message", "turn_finished"]


@pytest.mark.anyio
async def test_thread_id_is_stable_across_turns() -> None:
    h = Harness(FakeGraph(["one", "two"]))
    h.start()
    h.broker.send(SESSION, "main", ADDRESS, "first")
    await run_until(h, lambda: len(h.inbox("main")) == 1)
    h.broker.send(SESSION, "main", ADDRESS, "second")
    await run_until(h, lambda: len(h.inbox("main")) == 2)
    await h.stop()

    assert len(h.graph.calls) == 2
    thread_ids = [c[1]["configurable"]["thread_id"] for c in h.graph.calls]
    assert thread_ids == [h.record.thread_id, h.record.thread_id]
    assert h.record.turns == 2


@pytest.mark.anyio
async def test_mid_turn_messages_form_next_batch() -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def blocked() -> str:
        entered.set()
        await gate.wait()
        return "first reply"

    h = Harness(FakeGraph([blocked, "second reply"]))
    h.start()
    h.broker.send(SESSION, "main", ADDRESS, "first")
    await asyncio.wait_for(entered.wait(), 2)
    assert h.record.status == "running"
    h.broker.send(SESSION, "main", ADDRESS, "second")
    gate.set()
    await run_until(h, lambda: len(h.inbox("main")) == 2)
    await h.stop()

    assert len(h.graph.calls) == 2
    first, second = (human_texts(c) for c in h.graph.calls)
    assert len(first) == 1 and first[0].endswith("\nfirst")
    assert len(second) == 1 and second[0].endswith("\nsecond")
    assert h.inbox("main") == ["first reply", "second reply"]


@pytest.mark.anyio
async def test_user_sender_reply_published_not_sent() -> None:
    h = Harness(FakeGraph(["for you"]))
    h.start()
    h.broker.open(SESSION, "user", name="user", parent=None)
    h.broker.send(SESSION, "user", ADDRESS, "hey agent")
    await run_until(h, lambda: h.record.turns == 1 and h.record.status == "idle")
    await asyncio.sleep(0.01)
    await h.stop()

    assert h.broker.pending(SESSION, "user") == []
    message_events = [e for e in h.drain_events() if e.kind == "message"]
    assert [dict(e.payload) for e in message_events] == [
        {"from": "helper", "to": "user", "body": "for you"}
    ]
    assert message_events[0].agent_id == "abc123"
    assert message_events[0].agent_name == "helper"


@pytest.mark.anyio
async def test_multiple_senders_each_get_reply() -> None:
    h = Harness(FakeGraph(["shared reply"]))
    h.broker.open(SESSION, "other", name="other", parent="main")
    h.broker.send(SESSION, "main", ADDRESS, "a")
    h.broker.send(SESSION, "other", ADDRESS, "b")
    h.broker.send(SESSION, "main", ADDRESS, "c")
    h.start()
    await run_until(h, lambda: bool(h.inbox("main")) and bool(h.inbox("other")))
    await asyncio.sleep(0.01)
    await h.stop()

    assert len(h.graph.calls) == 1
    assert len(human_texts(h.graph.calls[0])) == 3
    assert h.inbox("main") == ["shared reply"]
    assert h.inbox("other") == ["shared reply"]
    message_events = [e for e in h.drain_events() if e.kind == "message"]
    assert [e.payload["to"] for e in message_events] == [
        "helper", "helper", "helper", "main", "other"
    ]


@pytest.mark.anyio
async def test_failure_sends_error_and_stays_idle() -> None:
    h = Harness(FakeGraph([RuntimeError("boom")]))
    h.start()
    h.broker.send(SESSION, "main", ADDRESS, "do it")
    await run_until(h, lambda: bool(h.inbox("main")))
    await asyncio.sleep(0.01)
    await h.stop()

    assert h.inbox("main") == ["[error] RuntimeError: boom"]
    assert h.record.status == "idle"
    assert h.record.consecutive_failures == 1
    assert h.record.turns == 1
    assert h.finished == []
    finished = [e for e in h.drain_events() if e.kind == "turn_finished"]
    assert [dict(e.payload) for e in finished] == [{"turns": 1, "ok": False}]


@pytest.mark.anyio
async def test_success_resets_consecutive_failures() -> None:
    h = Harness(FakeGraph([RuntimeError("boom"), "fine"]))
    h.start()
    h.broker.send(SESSION, "main", ADDRESS, "one")
    await run_until(h, lambda: len(h.inbox("main")) == 1)
    h.broker.send(SESSION, "main", ADDRESS, "two")
    await run_until(h, lambda: len(h.inbox("main")) == 2)
    await asyncio.sleep(0.01)
    await h.stop()

    assert h.record.consecutive_failures == 0


@pytest.mark.anyio
async def test_error_body_truncated_to_max_message_chars() -> None:
    h = Harness(FakeGraph([RuntimeError("x" * 20000)]))
    h.start()
    h.broker.send(SESSION, "main", ADDRESS, "do it")
    await run_until(h, lambda: bool(h.inbox("main")))
    await h.stop()

    (body,) = h.inbox("main")
    assert body.startswith("[error] RuntimeError: xxx")
    assert len(body) == h.broker.config.max_message_chars


@pytest.mark.anyio
async def test_three_failures_call_on_finished_failed() -> None:
    assert MAX_CONSECUTIVE_FAILURES == 3
    h = Harness(FakeGraph(default=RuntimeError("boom")))
    h.start()
    for i in range(3):
        h.broker.send(SESSION, "main", ADDRESS, f"try {i}")
        await run_until(h, lambda i=i: len(h.inbox("main")) == i + 1)
    assert h.task is not None
    await asyncio.wait_for(h.task, 2)

    assert h.finished == ["failed"]
    assert h.record.status == "failed"
    assert h.record.consecutive_failures == 3
    assert len(h.inbox("main")) == 3


@pytest.mark.anyio
async def test_max_turns_calls_on_finished_after_reply() -> None:
    h = Harness(FakeGraph(["r1", "r2"]), max_turns=2)
    h.start()
    h.broker.send(SESSION, "main", ADDRESS, "one")
    await run_until(h, lambda: len(h.inbox("main")) == 1)
    assert h.finished == []
    h.broker.send(SESSION, "main", ADDRESS, "two")
    assert h.task is not None
    await asyncio.wait_for(h.task, 2)

    assert h.finished == ["max_turns"]
    assert h.inbox("main") == ["r1", "r2"]
    assert h.record.turns == 2


@pytest.mark.anyio
async def test_empty_final_text_sends_placeholder() -> None:
    tool_only = AIMessage(
        content="",
        tool_calls=[{"name": "noop", "args": {}, "id": "call-1"}],
    )
    h = Harness(FakeGraph([tool_only, AIMessage(content=[])]))
    h.start()
    h.broker.send(SESSION, "main", ADDRESS, "one")
    await run_until(h, lambda: len(h.inbox("main")) == 1)
    h.broker.send(SESSION, "main", ADDRESS, "two")
    await run_until(h, lambda: len(h.inbox("main")) == 2)
    await h.stop()

    assert h.inbox("main") == [NO_REPLY_TEXT, NO_REPLY_TEXT]
    assert NO_REPLY_TEXT == "(no reply)"


@pytest.mark.anyio
async def test_text_blocks_are_joined() -> None:
    blocks = AIMessage(
        content=[
            {"type": "text", "text": "part one"},
            {"type": "tool_use", "id": "x", "name": "noop", "input": {}},
            {"type": "text", "text": "part two"},
        ]
    )
    h = Harness(FakeGraph([blocks]))
    text = await h.worker.run_turn(
        [h.broker.send(SESSION, "main", ADDRESS, "hi")]
    )
    assert "part one" in text and "part two" in text
    assert "tool_use" not in text


@pytest.mark.anyio
async def test_contextvars_set_during_turn() -> None:
    h = Harness(FakeGraph(["ok"]))
    entered: list[Any] = []

    from contextlib import contextmanager

    @contextmanager
    def turn_context(record: LiveAgentRecord):
        entered.append(record.id)
        yield

    h.worker = LiveAgentWorker(
        h.record,
        h.graph,
        h.broker,
        h.events,
        recursion_limit=7,
        max_turns=50,
        clock=lambda: h.now,
        turn_context=turn_context,
        on_finished=h._on_finished,
    )
    message = h.broker.send(SESSION, "main", ADDRESS, "hi")
    text = await h.worker.run_turn([message])

    assert text == "ok"
    assert h.graph.observed == [(ADDRESS, SESSION)]
    assert entered == ["abc123"]
    assert _CURRENT_AGENT_ADDRESS.get() == "main"
    assert _CURRENT_MESSAGING_SESSION.get() is None


@pytest.mark.anyio
async def test_cancel_mid_turn_recovers_inflight_messages() -> None:
    entered = asyncio.Event()

    async def hang() -> str:
        entered.set()
        await asyncio.Event().wait()
        return "never"

    h = Harness(FakeGraph([hang]))
    h.start()
    sent = h.broker.send(SESSION, "main", ADDRESS, "hi")
    await asyncio.wait_for(entered.wait(), 2)
    assert h.broker.get(SESSION, sent.id).status == "inflight"
    await h.stop()

    assert h.broker.get(SESSION, sent.id).status == "pending"
    assert h.finished == []


@pytest.mark.anyio
async def test_reply_to_closed_sender_is_skipped_and_others_still_replied() -> None:
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def blocked() -> str:
        entered.set()
        await gate.wait()
        return "reply"

    h = Harness(FakeGraph([blocked]))
    h.broker.open(SESSION, "other", name="other", parent="main")
    h.broker.send(SESSION, "other", ADDRESS, "a")
    h.broker.send(SESSION, "main", ADDRESS, "b")
    h.start()
    await asyncio.wait_for(entered.wait(), 2)
    h.broker.close(SESSION, "other")
    gate.set()
    await run_until(h, lambda: bool(h.inbox("main")))
    await asyncio.sleep(0.01)

    assert h.task is not None and not h.task.done()
    assert h.record.status == "idle"
    await h.stop()
    assert h.inbox("main") == ["reply"]


@pytest.mark.anyio
async def test_claimed_messages_published_as_inbound_events() -> None:
    h = Harness(FakeGraph(["done"]))
    h.broker.open(SESSION, "other", name="other", parent="main")
    h.broker.open(SESSION, "user", name="user", parent=None)
    h.broker.send(SESSION, "main", ADDRESS, "from main")
    h.broker.send(SESSION, "other", ADDRESS, "from peer")
    h.broker.send(SESSION, "user", ADDRESS, "from user")
    h.start()
    await run_until(h, lambda: h.record.turns == 1 and h.record.status == "idle")
    await asyncio.sleep(0.01)
    await h.stop()

    events = h.drain_events()
    assert [e.kind for e in events][:3] == ["message", "message", "turn_started"]
    inbound = [dict(e.payload) for e in events if e.kind == "message"][:2]
    # User messages are already published by the manager, so they are skipped.
    assert inbound == [
        {"from": "main", "to": "helper", "body": "from main"},
        {"from": "other", "to": "helper", "body": "from peer"},
    ]
    assert all(e.agent_id == "abc123" for e in events)


@pytest.mark.anyio
async def test_reply_to_user_is_recorded_as_last_reply() -> None:
    h = Harness(FakeGraph(["not for user", "for you"]))
    h.start()
    assert h.record.last_reply is None
    h.broker.send(SESSION, "main", ADDRESS, "from main")
    await run_until(h, lambda: h.record.turns == 1 and h.record.status == "idle")
    assert h.record.last_reply is None  # Replies to agents are not recorded.
    h.now = 150.0
    h.broker.open(SESSION, "user", name="user", parent=None)
    h.broker.send(SESSION, "user", ADDRESS, "hey agent")
    await run_until(h, lambda: h.record.turns == 2 and h.record.status == "idle")
    await h.stop()

    assert h.record.last_reply == {"body": "for you", "at": 150.0}
    assert h.record.to_payload()["last_reply"] == {"body": "for you", "at": 150.0}
