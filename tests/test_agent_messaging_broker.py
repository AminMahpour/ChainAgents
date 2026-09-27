import asyncio

import pytest

from chainagents.runtime.messaging import MessageBroker
from chainagents.runtime.types import MessagingConfig


def test_broker_routes_only_live_opted_in_recipients_and_bounds_queues():
    broker = MessageBroker(MessagingConfig(enabled=True, max_pending_per_recipient=1))
    broker.open("s", "main", name="main", parent=None)
    broker.open("s", "researcher:one", name="researcher", parent="main")
    first = broker.send("s", "main", "researcher:one", "first")
    assert first.status == "pending"
    with pytest.raises(ValueError, match="pending limit"):
        broker.send("s", "main", "researcher:one", "second")
    assert broker.deliver("s", "researcher:one") == [first]
    assert broker.get("s", first.id).status == "delivered"
    broker.close("s", "researcher:one")
    with pytest.raises(ValueError, match="unavailable"):
        broker.send("s", "main", "researcher:one", "late")


def test_broker_isolates_sessions_and_rejects_unregistered_senders():
    broker = MessageBroker(MessagingConfig(enabled=True))
    broker.open("a", "main", name="main", parent=None)
    broker.open("b", "main", name="main", parent=None)
    broker.open("b", "worker:two", name="worker", parent="main")
    with pytest.raises(ValueError, match="unavailable"):
        broker.send("a", "main", "worker:two", "secret")
    with pytest.raises(ValueError, match="unavailable"):
        broker.send("b", "worker:one", "main", "secret")


def test_wait_wakes_when_a_sync_tool_sends_from_another_thread():
    broker = MessageBroker(MessagingConfig(enabled=True))
    broker.open("s", "main", name="main", parent=None)
    broker.open("s", "worker:one", name="worker", parent="main")

    async def exercise():
        async def send_later():
            await asyncio.sleep(0.01)
            await asyncio.to_thread(broker.send, "s", "main", "worker:one", "hello")

        task = asyncio.create_task(send_later())
        messages = await broker.wait("s", "worker:one", 1)
        await task
        return messages

    assert [item.body for item in asyncio.run(exercise())] == ["hello"]


def test_parallel_waiters_both_wake_for_one_recipient():
    broker = MessageBroker(MessagingConfig(enabled=True))
    broker.open("s", "main", name="main", parent=None)
    broker.open("s", "worker:one", name="worker", parent="main")

    async def exercise():
        first = asyncio.create_task(broker.wait("s", "worker:one", 1))
        second = asyncio.create_task(broker.wait("s", "worker:one", 1))
        await asyncio.sleep(0)
        broker.send("s", "main", "worker:one", "hello")
        return await asyncio.gather(first, second)

    assert [[item.body for item in group] for group in asyncio.run(exercise())] == [
        ["hello"],
        ["hello"],
    ]


def test_claim_is_recoverable_until_model_step_acknowledges_it():
    broker = MessageBroker(MessagingConfig(enabled=True))
    broker.open("s", "main", name="main", parent=None)
    broker.open("s", "worker:one", name="worker", parent="main")
    message = broker.send("s", "main", "worker:one", "hello")
    assert broker.claim("s", "worker:one") == [message]
    assert broker.get("s", message.id).status == "inflight"
    broker.recover("s", "worker:one")
    assert broker.pending("s", "worker:one") == [message]
    broker.claim("s", "worker:one")
    broker.acknowledge("s", "worker:one")
    assert broker.get("s", message.id).status == "delivered"
