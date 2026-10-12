import asyncio

import pytest

from chainagents.runtime.live_agents.config import ResolvedLiveAgentSpec
from chainagents.runtime.live_agents.events import LiveAgentEvent, LiveAgentEventBus
from chainagents.runtime.live_agents.registry import (
    LiveAgentLimitError,
    LiveAgentRegistry,
)
from chainagents.runtime.types import LiveAgentsConfig


def spec(name="worker", template=None):
    return ResolvedLiveAgentSpec(
        name=name,
        template=template,
        system_prompt="p",
        model="m",
        tools=(),
        skills=(),
        mcp_servers=(),
    )


def make(**kwargs):
    ticks = iter(range(1, 1000))
    return LiveAgentRegistry(LiveAgentsConfig(**kwargs), clock=lambda: float(next(ticks)))


def test_reserve_enforces_session_limit():
    reg = make(max_live_per_session=2)
    reg.reserve("s", spec("a"), spawned_by="user")
    reg.reserve("s", spec("b"), spawned_by="user")
    with pytest.raises(LiveAgentLimitError):
        reg.reserve("s", spec("c"), spawned_by="user")
    reg.reserve("other", spec("c"), spawned_by="user")


def test_reserve_enforces_total_limit():
    reg = make(max_live_per_session=4, max_live_total=2)
    reg.reserve("s1", spec("a"), spawned_by="user")
    reg.reserve("s2", spec("b"), spawned_by="user")
    with pytest.raises(LiveAgentLimitError):
        reg.reserve("s3", spec("c"), spawned_by="user")


def test_reserve_fields_and_discard():
    reg = make()
    rec = reg.reserve("s", spec("a"), spawned_by="main")
    assert len(rec.id) == 12
    assert rec.address == f"live:a:{rec.id}"
    assert rec.thread_id == f"live-s-{rec.id}"
    assert rec.status == "idle"
    assert rec.spawned_by == "main"
    reg.discard(rec)
    assert reg.records("s", include_stopped=True) == []


def test_lookup_by_name_then_id():
    reg = make()
    rec = reg.reserve("s", spec("a"), spawned_by="user")
    assert reg.lookup("s", "a") is rec
    assert reg.lookup("s", rec.id) is rec
    with pytest.raises(KeyError):
        reg.lookup("s", "missing")
    with pytest.raises(KeyError):
        reg.lookup("other", "a")


def test_name_reusable_after_stop():
    reg = make()
    old = reg.reserve("s", spec("a"), spawned_by="user")
    old.status = "stopped"
    assert reg.live_names("s") == set()
    with pytest.raises(KeyError):
        reg.lookup("s", "a")
    assert reg.lookup("s", old.id) is old
    new = reg.reserve("s", spec("a"), spawned_by="user")
    assert reg.lookup("s", "a") is new
    assert reg.records("s") == [new]
    assert reg.records("s", include_stopped=True) == [old, new]


def test_failed_agent_frees_slot_and_name():
    reg = make(max_live_per_session=1, max_live_total=1)
    old = reg.reserve("s", spec("a"), spawned_by="main")
    old.status = "failed"
    assert reg.live_names("s") == set()
    assert reg.records("s") == []
    with pytest.raises(KeyError):
        reg.lookup("s", "a")
    assert reg.lookup("s", old.id) is old
    new = reg.reserve("s", spec("a"), spawned_by="main")
    assert reg.lookup("s", "a") is new
    assert reg.records("s") == [new]
    assert reg.records("s", include_stopped=True) == [old, new]


def test_next_sequence_counts_all():
    reg = make()
    assert reg.next_sequence("s") == 1
    rec = reg.reserve("s", spec("a"), spawned_by="user")
    rec.status = "stopped"
    assert reg.next_sequence("s") == 2
    reg.forget_session("s")
    assert reg.next_sequence("s") == 1


def test_payload_shape():
    reg = make()
    rec = reg.reserve("s", spec("a", "tpl"), spawned_by="user")
    assert rec.to_payload(pending=3) == {
        "id": rec.id,
        "name": "a",
        "template": "tpl",
        "model": "m",
        "status": "idle",
        "turns": 0,
        "created_at": rec.created_at,
        "last_active_at": rec.last_active_at,
        "pending": 3,
        "stop_reason": None,
        "last_reply": None,
    }


@pytest.mark.anyio
async def test_event_bus_delivers_to_session_subscribers_only():
    bus = LiveAgentEventBus()
    q1 = bus.subscribe("s1")
    q2 = bus.subscribe("s2")
    event = LiveAgentEvent("spawned", "s1", "id", "a", {"x": 1})
    bus.publish(event)
    assert await asyncio.wait_for(q1.get(), 1) is event
    await asyncio.sleep(0.01)
    assert q2.empty()
    bus.unsubscribe("s1", q1)
    bus.publish(event)
    await asyncio.sleep(0.01)
    assert q1.empty()
