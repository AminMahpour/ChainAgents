"""Replay missed local background completions when Chainlit resumes a chat."""

from __future__ import annotations

import asyncio

import pytest

from chainagents.interfaces.chainlit import async_tasks
from chainagents.interfaces.chainlit.async_tasks import LocalBackgroundTaskNotifier
from chainagents.runtime.background_tasks import BackgroundTaskManager
from chainagents.runtime.types import BackgroundSubagentConfig


def _manager(
    *, stream_activity: bool = False, max_tasks_per_session: int = 100
) -> BackgroundTaskManager:
    return BackgroundTaskManager(
        BackgroundSubagentConfig(
            enabled=True,
            stream_activity=stream_activity,
            max_tasks_per_session=max_tasks_per_session,
        )
    )


async def _finish(manager: BackgroundTaskManager, session_id: str) -> str:
    async def runner(task_id: str) -> str:
        return f"result for {task_id}"

    spawned = await manager.spawn(
        session_id=session_id,
        agent_name="researcher",
        description="research",
        agent_path=("researcher",),
        runner=runner,
    )
    finished = await manager.get(session_id, spawned.task_id, wait_seconds=1)
    assert finished.status == "success"
    return spawned.task_id


def test_reconcile_replays_only_missed_session_notice_once_across_notifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        manager = _manager()
        missed_id = await _finish(manager, "session-a")
        other_id = await _finish(manager, "session-b")
        sent: list[str] = []

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                assert author == "Background subagent"
                self.content = content

            async def send(self) -> None:
                sent.append(self.content)

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message", Message
        )
        first = LocalBackgroundTaskNotifier(manager=manager, session_id="session-a")
        first.start()
        await first.reconcile_terminal_tasks()
        await first.aclose()

        resumed = LocalBackgroundTaskNotifier(manager=manager, session_id="session-a")
        resumed.start()
        await resumed.reconcile_terminal_tasks()

        assert len(sent) == 1
        assert f"Task ID: `{missed_id}`" in sent[0]
        assert other_id not in sent[0]

        await resumed.aclose()
        await manager.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("stream_activity", [False, True])
def test_reconcile_and_live_terminal_event_send_one_notice(
    monkeypatch: pytest.MonkeyPatch, stream_activity: bool
) -> None:
    async def exercise() -> None:
        manager = _manager(stream_activity=stream_activity)
        sent: list[str] = []
        send_started = asyncio.Event()
        release_send = asyncio.Event()

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                self.content = content

            async def send(self) -> None:
                send_started.set()
                await release_send.wait()
                sent.append(self.content)

        class Step:
            def __init__(self, *, name: str, type: str, **kwargs: object) -> None:
                self.id = "background-step"
                self.name = name
                self.type = type
                self.end = None
                self.output = ""

            async def send(self) -> None:
                pass

            async def update(self) -> None:
                pass

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message", Message
        )
        monkeypatch.setattr("chainagents.interfaces.chainlit.async_tasks.cl.Step", Step)
        notifier = LocalBackgroundTaskNotifier(manager=manager, session_id="session-a")
        notifier.start()
        task_id = await _finish(manager, "session-a")

        reconcile = asyncio.create_task(notifier.reconcile_terminal_tasks())
        await asyncio.wait_for(send_started.wait(), timeout=1)
        release_send.set()
        await asyncio.wait_for(reconcile, timeout=1)
        await asyncio.sleep(0.1)

        assert len(sent) == 1
        assert f"Task ID: `{task_id}`" in sent[0]
        await notifier.aclose()
        await manager.close()

    asyncio.run(exercise())


def test_failed_replay_notice_can_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        manager = _manager()
        task_id = await _finish(manager, "session-a")
        attempts = 0
        sent: list[str] = []

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                self.content = content

            async def send(self) -> None:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("temporary send failure")
                sent.append(self.content)

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message", Message
        )
        notifier = LocalBackgroundTaskNotifier(manager=manager, session_id="session-a")
        notifier.start()
        await notifier.reconcile_terminal_tasks()
        await notifier.reconcile_terminal_tasks()

        assert attempts == 2
        assert len(sent) == 1
        assert f"Task ID: `{task_id}`" in sent[0]
        await notifier.aclose()
        await manager.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("detach_during_send", [False, True])
def test_clearing_chat_does_not_mark_terminal_notice_delivered(
    monkeypatch: pytest.MonkeyPatch, detach_during_send: bool
) -> None:
    async def exercise() -> None:
        manager = _manager()
        finish = asyncio.Event()
        attached = True
        sent: list[str] = []

        async def runner(_task_id: str) -> str:
            await finish.wait()
            return "finished"

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                assert author == "Background subagent"
                self.content = content

            async def send(self) -> None:
                nonlocal attached
                sent.append(self.content)
                if detach_during_send:
                    attached = False

        monkeypatch.setattr(async_tasks.cl, "Message", Message)
        old = LocalBackgroundTaskNotifier(
            manager=manager,
            session_id="session-a",
            delivery_allowed=lambda: attached,
        )
        old.start()
        spawned = await manager.spawn(
            session_id="session-a",
            agent_name="researcher",
            description="research",
            agent_path=("researcher",),
            runner=runner,
        )
        if not detach_during_send:
            attached = False  # Chainlit has set to_clear before on_chat_end.
        finish.set()
        await manager.wait_session("session-a")
        await old.reconcile_terminal_tasks()
        old.cancel()
        await async_tasks.retain_missed_local_task_results(manager, "session-a")
        await manager.close_session("session-a")

        attached = True
        resumed = LocalBackgroundTaskNotifier(manager=manager, session_id="session-a")
        resumed.start()
        await resumed.reconcile_terminal_tasks()
        assert len(sent) == (2 if detach_during_send else 1)
        assert f"Task ID: `{spawned.task_id}`" in sent[-1]
        await resumed.aclose()
        await manager.close()

    asyncio.run(exercise())


def test_handoff_replays_missed_notice_after_manager_closes_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        manager = _manager()
        task_id = await _finish(manager, "session-a")
        await async_tasks.retain_missed_local_task_results(manager, "session-a")
        await manager.close_session("session-a")
        assert await manager.list("session-a") == []

        sent: list[str] = []

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                self.content = content

            async def send(self) -> None:
                sent.append(self.content)

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message", Message
        )
        resumed = LocalBackgroundTaskNotifier(manager=manager, session_id="session-a")
        resumed.start()
        await resumed.reconcile_terminal_tasks()
        await resumed.reconcile_terminal_tasks()

        assert len(sent) == 1
        assert f"Task ID: `{task_id}`" in sent[0]
        await resumed.aclose()
        await manager.close()

    asyncio.run(exercise())


def test_handoff_does_not_retain_already_announced_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        manager = _manager()
        delivered = asyncio.Event()
        sent: list[str] = []

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                self.content = content

            async def send(self) -> None:
                sent.append(self.content)
                delivered.set()

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message", Message
        )
        active = LocalBackgroundTaskNotifier(manager=manager, session_id="session-a")
        active.start()
        await _finish(manager, "session-a")
        await asyncio.wait_for(delivered.wait(), timeout=1)
        await active.aclose()

        await async_tasks.retain_missed_local_task_results(manager, "session-a")
        await manager.close_session("session-a")
        resumed = LocalBackgroundTaskNotifier(manager=manager, session_id="session-a")
        resumed.start()
        await resumed.reconcile_terminal_tasks()

        assert len(sent) == 1
        await resumed.aclose()
        await manager.close()

    asyncio.run(exercise())


def test_completion_state_is_bounded_after_manager_prunes_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        manager = _manager(max_tasks_per_session=2)
        sent_count = 0
        delivered = asyncio.Event()

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                pass

            async def send(self) -> None:
                nonlocal sent_count
                sent_count += 1
                delivered.set()

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message", Message
        )
        notifier = LocalBackgroundTaskNotifier(manager=manager, session_id="session-a")
        notifier.start()
        for index in range(1010):
            delivered.clear()
            await _finish(manager, "session-a")
            await asyncio.wait_for(delivered.wait(), timeout=1)
            assert sent_count == index + 1

        state = async_tasks._local_completion_states[manager]["session-a"]
        assert len(state.notified_task_ids) <= 1002
        await notifier.aclose()
        await manager.close()

    asyncio.run(exercise())


def test_old_handoffs_are_evicted_before_recent_missed_notices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        manager = _manager()
        monkeypatch.setattr(async_tasks, "MAX_STORED_SESSIONS", 3, raising=False)
        ids: dict[str, str] = {}
        for index in range(5):
            session_id = f"session-{index}"
            ids[session_id] = await _finish(manager, session_id)
            await async_tasks.retain_missed_local_task_results(manager, session_id)
            await manager.close_session(session_id)

        sessions = async_tasks._local_completion_states[manager]
        assert len(sessions) <= 3
        assert "session-4" in sessions

        sent: list[str] = []

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                self.content = content

            async def send(self) -> None:
                sent.append(self.content)

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message", Message
        )
        resumed = LocalBackgroundTaskNotifier(manager=manager, session_id="session-4")
        resumed.start()
        await resumed.reconcile_terminal_tasks()
        assert len(sent) == 1
        assert f"Task ID: `{ids['session-4']}`" in sent[0]
        await resumed.aclose()
        await manager.close()

    asyncio.run(exercise())


def test_recent_live_notice_keeps_dedup_state_fresh_after_long_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def exercise() -> None:
        manager = _manager()
        sent: list[str] = []
        delivered = asyncio.Event()

        class Message:
            def __init__(self, *, content: str, author: str) -> None:
                self.content = content

            async def send(self) -> None:
                sent.append(self.content)
                delivered.set()

        monkeypatch.setattr(
            "chainagents.interfaces.chainlit.async_tasks.cl.Message", Message
        )
        active = LocalBackgroundTaskNotifier(manager=manager, session_id="session-a")
        active.start()
        state = async_tasks._local_completion_states[manager]["session-a"]
        state.last_touched -= async_tasks.MISSED_NOTICE_RETENTION_SECONDS + 1
        await _finish(manager, "session-a")
        await asyncio.wait_for(delivered.wait(), timeout=1)
        await active.aclose()

        resumed = LocalBackgroundTaskNotifier(manager=manager, session_id="session-a")
        resumed.start()
        await resumed.reconcile_terminal_tasks()
        assert len(sent) == 1
        await resumed.aclose()
        await manager.close()

    asyncio.run(exercise())
