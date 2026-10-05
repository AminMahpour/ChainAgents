import asyncio
from contextvars import ContextVar

import pytest
from types import SimpleNamespace

from chainagents.turns.controller import ConversationInputController
from chainagents.turns.runner import TurnRequest, TurnRunner
from chainagents.runtime.messaging import MessageBroker
from chainagents.runtime.types import MessagingConfig, UserInputConfig


def test_queued_job_keeps_its_submitter_context():
    async def exercise():
        marker: ContextVar[str] = ContextVar("submitter")
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        release = asyncio.Event()
        seen = []

        async def run(label):
            seen.append((label, marker.get()))
            if label == "first":
                await release.wait()

        marker.set("first submitter")
        controller.submit("s", "first", run, followup_factory=lambda text: text)
        await asyncio.sleep(0)
        marker.set("second submitter")
        controller.submit("s", "second", run, followup_factory=lambda text: text)
        release.set()
        await controller.wait_idle("s")
        assert seen == [("first", "first submitter"), ("second", "second submitter")]

    asyncio.run(exercise())


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


def test_stop_before_task_starts_releases_active_slot():
    async def exercise():
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        started = []

        async def run(text):
            started.append(text)

        first = controller.submit(
            "s", "first", run, followup_factory=lambda text: text, input_id="first"
        )
        second = controller.submit("s", "second", run, followup_factory=lambda text: text)
        controller.stop("s")
        await asyncio.wait_for(controller.wait_idle("s"), timeout=1)
        assert first.status == "cancelled"
        assert first.payload is None
        assert controller.find_input_id("s", "first") is first
        assert controller.status("s")["active_job_id"] is None
        controller.resume("s")
        await controller.wait_idle("s")
        assert second.status == "completed"
        assert started == ["second"]

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


def test_wait_drained_waits_for_paused_queued_turn_until_resume():
    async def exercise():
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        controller.stop("s")
        assert controller.busy("s") is False
        ran = asyncio.Event()

        async def run(_payload):
            ran.set()

        controller.submit("s", "queued", run, followup_factory=lambda text: text)
        assert controller.busy("s") is True
        waiter = asyncio.create_task(controller.wait_drained("s"))
        await asyncio.sleep(0)
        assert not waiter.done()
        assert not ran.is_set()

        controller.resume("s")
        await asyncio.wait_for(waiter, timeout=1)
        assert ran.is_set()
        assert controller.busy("s") is False

    asyncio.run(exercise())


def test_detached_drain_ignores_paused_queue_but_waits_for_resumed_turn():
    async def exercise():
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def run(_payload):
            started.set()
            await release.wait()

        controller.stop("s")
        job = controller.submit(
            "s", "queued", run, followup_factory=lambda text: text
        )
        assert controller.busy("s") is True
        assert controller.busy("s", include_paused_queue=False) is False
        await asyncio.wait_for(
            controller.wait_drained("s", include_paused_queue=False), timeout=1
        )
        assert job.status == "queued"

        controller.resume("s")
        await started.wait()
        assert controller.busy("s", include_paused_queue=False) is True
        drained = asyncio.create_task(
            controller.wait_drained("s", include_paused_queue=False)
        )
        await asyncio.sleep(0)
        assert not drained.done()
        release.set()
        await asyncio.wait_for(drained, timeout=1)
        assert job.status == "completed"

    asyncio.run(exercise())


def test_detached_drain_waits_for_cancelled_active_turn_before_paused_queue():
    async def exercise():
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        first_started = asyncio.Event()
        release_cancellation = asyncio.Event()

        async def run(payload):
            if payload == "first":
                first_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    await release_cancellation.wait()

        first = controller.submit(
            "s", "first", run, followup_factory=lambda text: text
        )
        await first_started.wait()
        second = controller.submit(
            "s", "second", run, followup_factory=lambda text: text
        )
        controller.stop("s")
        assert controller.busy("s", include_paused_queue=False) is True
        drained = asyncio.create_task(
            controller.wait_drained("s", include_paused_queue=False)
        )
        await asyncio.sleep(0)
        assert not drained.done()

        release_cancellation.set()
        await asyncio.wait_for(drained, timeout=1)
        assert first.status == "cancelled"
        assert second.status == "queued"
        assert controller.busy("s", include_paused_queue=False) is False

    asyncio.run(exercise())


def test_wait_drained_waits_for_external_turn_and_waiter():
    async def exercise():
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        external_started = asyncio.Event()
        release_external = asyncio.Event()
        waiter_started = asyncio.Event()
        release_waiter = asyncio.Event()

        async def external_turn():
            assert controller.turn_started("s") is True
            external_started.set()
            await release_external.wait()
            controller.turn_finished("s")

        async def waiting_turn():
            assert controller.turn_waiting("s") is True
            waiter_started.set()
            await release_waiter.wait()
            controller.turn_wait_finished("s")

        external = asyncio.create_task(external_turn())
        waiting = asyncio.create_task(waiting_turn())
        await external_started.wait()
        await waiter_started.wait()
        assert controller.busy("s") is True
        drained = asyncio.create_task(controller.wait_drained("s"))
        await asyncio.sleep(0)
        assert not drained.done()

        release_external.set()
        await external
        await asyncio.sleep(0)
        assert not drained.done()

        release_waiter.set()
        await waiting
        await asyncio.wait_for(drained, timeout=1)
        assert controller.busy("s") is False

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


def test_expired_steering_id_can_be_reused_for_a_later_turn():
    async def exercise():
        broker = MessageBroker(MessagingConfig(enabled=True))
        controller = ConversationInputController(
            UserInputConfig(enabled=True, max_completed_turns=1), broker
        )
        first_gate = asyncio.Event()
        third_gate = asyncio.Event()

        async def run(text):
            if text == "first":
                await first_gate.wait()
            elif text == "third":
                await third_gate.wait()

        controller.submit("s", "first", run, followup_factory=lambda text: text)
        await asyncio.sleep(0)
        first_message = controller.steer("s", "first note", input_id="same-key")
        assert controller.steer("s", "ignored retry", input_id="same-key") == first_message
        broker.deliver("s", "main")
        first_gate.set()
        await controller.wait_idle("s")

        controller.submit("s", "second", run, followup_factory=lambda text: text)
        await controller.wait_idle("s")
        controller.submit("s", "third", run, followup_factory=lambda text: text)
        await asyncio.sleep(0)
        later_message = controller.steer("s", "later note", input_id="same-key")
        assert later_message != first_message
        assert broker.get("s", later_message).body == "later note"
        broker.deliver("s", "main")
        third_gate.set()
        await controller.wait_idle("s")

    asyncio.run(exercise())


def test_legacy_turn_blocks_controller_start_and_steering(monkeypatch):
    async def exercise():
        broker = MessageBroker(MessagingConfig(enabled=True))
        controller = ConversationInputController(UserInputConfig(enabled=True), broker)
        lock = asyncio.Lock()
        runtime = SimpleNamespace(
            config=SimpleNamespace(extensions=SimpleNamespace(
                user_input=UserInputConfig(enabled=True),
                messaging=MessagingConfig(enabled=False),
            )),
            user_input=controller,
            turn_lock=lambda _session_id: lock,
        )
        legacy_started = asyncio.Event()
        release_legacy = asyncio.Event()
        seen: list[str] = []

        async def run(_runner, request, _renderer):
            seen.append(request.prompt)
            if request.prompt == "legacy":
                legacy_started.set()
                await release_legacy.wait()
            return request.prompt

        def request(prompt):
            return TurnRequest(
                prompt=prompt, thread_id="s", model_name="fake", reasoning_level="medium"
            )

        renderer = SimpleNamespace(on_cancelled=lambda: asyncio.sleep(0))

        async def managed(prompt):
            return await TurnRunner(runtime).run(request(prompt), renderer)

        monkeypatch.setattr(TurnRunner, "_run", run)
        legacy = asyncio.create_task(TurnRunner(runtime).run(request("legacy"), renderer))
        await legacy_started.wait()
        assert controller.status("s")["external_active"] is True
        job = controller.submit("s", "queued", managed, followup_factory=lambda text: text)
        assert job.status == "queued"
        assert controller.status("s")["active_job_id"] is None
        with pytest.raises(ValueError, match="No active turn"):
            controller.steer("s", "must not reach legacy")
        release_legacy.set()
        await legacy
        await controller.wait_idle("s")
        assert seen == ["legacy", "queued"]
        assert job.status == "completed"

    asyncio.run(exercise())


def test_stop_cancels_running_external_turn(monkeypatch):
    async def exercise():
        broker = MessageBroker(MessagingConfig(enabled=True))
        controller = ConversationInputController(UserInputConfig(enabled=True), broker)
        lock = asyncio.Lock()
        runtime = SimpleNamespace(
            config=SimpleNamespace(extensions=SimpleNamespace(
                user_input=UserInputConfig(enabled=True),
                messaging=MessagingConfig(enabled=False),
            )),
            user_input=controller,
            turn_lock=lambda _session_id: lock,
        )
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def run(_runner, _request, _renderer):
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        monkeypatch.setattr(TurnRunner, "_run", run)
        request = TurnRequest(
            prompt="legacy", thread_id="s", model_name="fake", reasoning_level="medium"
        )
        renderer = SimpleNamespace(on_cancelled=lambda: asyncio.sleep(0))
        legacy = asyncio.create_task(TurnRunner(runtime).run(request, renderer))
        await started.wait()
        controller.stop("s")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(legacy, timeout=1)
        assert cancelled.is_set()
        assert controller.status("s") == {
            "active_job_id": None, "external_active": False,
            "queued_job_ids": [], "paused": True,
        }

    asyncio.run(exercise())


def test_stop_cancels_external_turn_waiting_for_same_lock(monkeypatch):
    async def exercise():
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        lock = asyncio.Lock()
        runtime = SimpleNamespace(
            config=SimpleNamespace(extensions=SimpleNamespace(
                user_input=UserInputConfig(enabled=True),
                messaging=MessagingConfig(enabled=False),
            )),
            user_input=controller,
            turn_lock=lambda _session_id: lock,
        )
        first_started = asyncio.Event()
        second_started = asyncio.Event()

        async def run(_runner, request, _renderer):
            if request.prompt == "first":
                first_started.set()
            else:
                second_started.set()
            await asyncio.Future()

        monkeypatch.setattr(TurnRunner, "_run", run)
        command_errors = []

        async def on_command_error(exc, status):
            command_errors.append((exc.message, status))

        renderer = SimpleNamespace(
            on_cancelled=lambda: asyncio.sleep(0),
            on_command_error=on_command_error,
        )

        def request(prompt):
            return TurnRequest(
                prompt=prompt, thread_id="s", model_name="fake", reasoning_level="medium"
            )

        first = asyncio.create_task(TurnRunner(runtime).run(request("first"), renderer))
        await first_started.wait()
        second = asyncio.create_task(TurnRunner(runtime).run(request("second"), renderer))
        await asyncio.sleep(0)
        controller.stop("s")
        try:
            for task in (first, second):
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)
            assert not second_started.is_set()
            assert controller.status("s")["paused"] is True
            third = await asyncio.wait_for(
                TurnRunner(runtime).run(request("third"), renderer), timeout=1
            )
            assert third.command_error is not None
            assert third.command_error.status == 409
            assert command_errors == [
                ("Conversation is paused. Resume before starting a turn.", 409)
            ]
        finally:
            for task in (first, second):
                task.cancel()
            await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(exercise())


def test_external_lock_waiters_count_as_busy():
    async def exercise():
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        assert controller.turn_waiting("s") is True
        assert controller.status("s")["external_active"] is True
        controller.turn_wait_finished("s")
        assert controller.status("s")["external_active"] is False

    asyncio.run(exercise())


def test_resume_cannot_restart_queued_job_during_close():
    async def exercise():
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        running = asyncio.Event()
        unwinding = asyncio.Event()
        finish_unwind = asyncio.Event()
        queued_started = asyncio.Event()

        async def run(value):
            if value == "active":
                running.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    unwinding.set()
                    await finish_unwind.wait()
                    raise
            queued_started.set()

        controller.submit("s", "active", run, followup_factory=lambda text: text)
        queued = controller.submit("s", "queued", run, followup_factory=lambda text: text)
        await running.wait()
        closing = asyncio.create_task(controller.close_session("s"))
        await unwinding.wait()
        controller.resume("s")
        assert controller.status("s")["paused"] is True
        finish_unwind.set()
        await closing
        await asyncio.sleep(0)
        assert not queued_started.is_set()
        assert queued.status == "cancelled"

    asyncio.run(exercise())


def test_resume_unknown_thread_does_not_create_session():
    controller = ConversationInputController(
        UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
    )
    for index in range(100):
        controller.resume(f"unknown-{index}")
    assert controller._sessions == {}


def test_read_only_turn_lookups_do_not_create_sessions():
    controller = ConversationInputController(
        UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
    )
    for index in range(100):
        session_id = f"unknown-{index}"
        assert controller.status(session_id) == {
            "active_job_id": None, "external_active": False,
            "queued_job_ids": [], "paused": False,
        }
        assert controller.find_input_id(session_id, "request") is None
        with pytest.raises(ValueError, match="Turn is unavailable"):
            controller.get(session_id, "turn")
    assert controller._sessions == {}


def test_close_session_cancels_queued_jobs_retained_by_event_consumers():
    async def exercise():
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        async def seed(_payload):
            return None

        controller.submit("s", "seed", seed, followup_factory=lambda text: text)
        await controller.wait_idle("s")
        controller.stop("s")

        async def run(_payload):
            raise AssertionError("paused jobs must not run")

        job = controller.submit("s", "queued", run, followup_factory=lambda text: text)
        assert job.status == "queued"
        await controller.close_session("s")
        assert job.status == "cancelled"
        assert "s" not in controller._sessions

    asyncio.run(exercise())


def test_queued_turn_cannot_be_steered_while_legacy_waiter_owns_lock(monkeypatch):
    async def exercise():
        broker = MessageBroker(MessagingConfig(enabled=True))
        controller = ConversationInputController(UserInputConfig(enabled=True), broker)
        lock = asyncio.Lock()
        runtime = SimpleNamespace(
            config=SimpleNamespace(extensions=SimpleNamespace(
                user_input=UserInputConfig(enabled=True),
                messaging=MessagingConfig(enabled=False),
            )),
            user_input=controller,
            turn_lock=lambda _session_id: lock,
        )
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release_first = asyncio.Event()
        release_second = asyncio.Event()
        renderer = SimpleNamespace(on_cancelled=lambda: asyncio.sleep(0))

        async def run(_runner, request, _renderer):
            if request.prompt == "first":
                first_started.set()
                await release_first.wait()
            elif request.prompt == "second":
                second_started.set()
                await release_second.wait()
            return request.prompt

        def request(prompt):
            return TurnRequest(
                prompt=prompt, thread_id="s", model_name="fake", reasoning_level="medium"
            )

        async def managed(prompt):
            return await TurnRunner(runtime).run(request(prompt), renderer)

        monkeypatch.setattr(TurnRunner, "_run", run)
        first = asyncio.create_task(TurnRunner(runtime).run(request("first"), renderer))
        await first_started.wait()
        second = asyncio.create_task(TurnRunner(runtime).run(request("second"), renderer))
        await asyncio.sleep(0)
        controller.submit("s", "managed", managed, followup_factory=lambda text: text)
        try:
            release_first.set()
            await second_started.wait()
            assert controller.status("s")["external_active"] is True
            with pytest.raises(ValueError, match="No active turn"):
                controller.steer("s", "must not reach second")
        finally:
            release_first.set()
            release_second.set()
            await asyncio.gather(first, second)
            await controller.wait_idle("s")

    asyncio.run(exercise())


def test_managed_turn_stays_waiting_until_runner_acquires_lock(monkeypatch):
    async def exercise():
        broker = MessageBroker(MessagingConfig(enabled=True))
        controller = ConversationInputController(
            UserInputConfig(enabled=True), broker, runner_serialized=True
        )
        lock = asyncio.Lock()
        runtime = SimpleNamespace(
            config=SimpleNamespace(extensions=SimpleNamespace(
                user_input=UserInputConfig(enabled=True),
                messaging=MessagingConfig(enabled=False),
            )),
            user_input=controller,
            turn_lock=lambda _session_id: lock,
        )
        started = asyncio.Event()
        release = asyncio.Event()
        renderer = SimpleNamespace(on_cancelled=lambda: asyncio.sleep(0))

        async def run(_runner, request, _renderer):
            started.set()
            await release.wait()
            return request.prompt

        async def managed(prompt):
            return await TurnRunner(runtime).run(
                TurnRequest(
                    prompt=prompt, thread_id="s", model_name="fake",
                    reasoning_level="medium",
                ),
                renderer,
            )

        monkeypatch.setattr(TurnRunner, "_run", run)
        job = controller.submit("s", "managed", managed, followup_factory=lambda text: text)
        assert job.status == "waiting"
        try:
            await started.wait()
            assert job.status == "running"
        finally:
            release.set()
            await controller.wait_idle("s")
        assert job.status == "completed"

    asyncio.run(exercise())
