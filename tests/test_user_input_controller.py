import asyncio

from chainagents.turns.controller import ConversationInputController
from chainagents.runtime.messaging import MessageBroker
from chainagents.runtime.types import MessagingConfig, UserInputConfig


def test_queue_runs_one_turn_at_a_time_and_stop_pauses_until_resume():
    async def exercise():
        broker = MessageBroker(MessagingConfig(enabled=True))
        controller = ConversationInputController(UserInputConfig(enabled=True), broker)
        gate = asyncio.Event()
        started = []

        async def run(text):
            started.append(text)
            if text == "one":
                await gate.wait()

        first = controller.submit("s", "one", run, followup_factory=lambda text: text)
        second = controller.submit("s", "two", run, followup_factory=lambda text: text)
        third = controller.submit(
            "s", "ignored", run, followup_factory=lambda text: text, input_id="same"
        )
        assert (
            third.id
            == controller.submit(
                "s", "ignored", run, followup_factory=lambda text: text, input_id="same"
            ).id
        )
        await asyncio.sleep(0)
        assert started == ["one"]
        controller.stop("s")
        await asyncio.sleep(0)
        assert first.status == "cancelled"
        assert second.status == "queued"
        assert started == ["one"]
        controller.resume("s")
        await controller.wait_idle("s")
        assert started == ["one", "two", "ignored"]

    asyncio.run(exercise())


def test_late_steering_becomes_followup_before_queued_turn():
    async def exercise():
        broker = MessageBroker(MessagingConfig(enabled=True))
        controller = ConversationInputController(UserInputConfig(enabled=True), broker)
        gate = asyncio.Event()
        seen = []

        async def run(text):
            seen.append(text)
            if text == "one":
                await gate.wait()

        controller.submit(
            "s", "one", run, followup_factory=lambda text: f"followup:{text}"
        )
        controller.submit("s", "two", run, followup_factory=lambda text: text)
        await asyncio.sleep(0)
        controller.steer("s", "change direction")
        gate.set()
        await controller.wait_idle("s")
        assert seen == ["one", "followup:change direction", "two"]

    asyncio.run(exercise())


def test_agent_message_does_not_wake_idle_main_as_user_followup():
    async def exercise():
        broker = MessageBroker(MessagingConfig(enabled=True))
        controller = ConversationInputController(UserInputConfig(enabled=True), broker)
        gate = asyncio.Event()
        seen = []

        async def run(text):
            seen.append(text)
            await gate.wait()

        controller.submit("s", "original", run, followup_factory=lambda text: text)
        await asyncio.sleep(0)
        broker.open("s", "worker:one", name="worker", parent="main")
        sent = broker.send("s", "worker:one", "main", "peer note")
        gate.set()
        await controller.wait_idle("s")
        assert seen == ["original"]
        assert broker.pending("s", "main") == [sent]

    asyncio.run(exercise())


def test_completed_jobs_release_payload_and_expire_old_results():
    async def exercise():
        broker = MessageBroker(MessagingConfig(enabled=True))
        controller = ConversationInputController(
            UserInputConfig(enabled=True, max_completed_turns=1), broker
        )

        async def run(payload):
            return len(payload)

        first = controller.submit(
            "s", "large upload", run, followup_factory=lambda text: text,
            input_id="first",
        )
        await controller.wait_idle("s")
        assert first.payload is None
        assert first.result == 12

        second = controller.submit(
            "s", "next", run, followup_factory=lambda text: text,
            input_id="second",
        )
        await controller.wait_idle("s")
        assert second.payload is None
        assert controller.find_input_id("s", "first") is None
        assert controller.find_input_id("s", "second") is second
        try:
            controller.get("s", first.id)
        except ValueError:
            pass
        else:
            raise AssertionError("expired result was retained")

    asyncio.run(exercise())


def test_pending_steering_reserves_queue_capacity():
    async def exercise():
        broker = MessageBroker(
            MessagingConfig(enabled=True, max_deliveries_per_step=2)
        )
        controller = ConversationInputController(
            UserInputConfig(enabled=True, max_queued_turns=1), broker
        )
        gate = asyncio.Event()
        seen = []

        async def run(text):
            seen.append(text)
            if text == "first":
                await gate.wait()

        controller.submit("s", "first", run, followup_factory=lambda text: text)
        await asyncio.sleep(0)
        controller.steer("s", "one")
        controller.steer("s", "two")
        try:
            controller.submit("s", "queued", run, followup_factory=lambda text: text)
        except ValueError:
            pass
        else:
            raise AssertionError("queue exceeded reserved capacity")
        try:
            controller.steer("s", "three")
        except ValueError:
            pass
        else:
            raise AssertionError("steering exceeded reserved capacity")
        gate.set()
        await controller.wait_idle("s")
        assert seen == ["first", "one\n\ntwo"]

    asyncio.run(exercise())
