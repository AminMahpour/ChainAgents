"""Background task activity in the HTTP API's NDJSON stream."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from chainagents.events.stream import AgentStreamEvent
from chainagents.interfaces.api import app as chainagents_api
from chainagents.runtime.background_tasks.models import BackgroundTaskActivity

THREAD = "thread-1"


class _FakeManager:
    """Activity subscriptions of a BackgroundTaskManager, nothing else."""

    def __init__(self, *, stream_activity: bool = True) -> None:
        self.config = SimpleNamespace(stream_activity=stream_activity)
        self.queues: list[asyncio.Queue[BackgroundTaskActivity]] = []
        self.unsubscribed: list[str] = []

    def subscribe_activity(self, session_id: str) -> asyncio.Queue[BackgroundTaskActivity]:
        queue: asyncio.Queue[BackgroundTaskActivity] = asyncio.Queue()
        self.queues.append(queue)
        return queue

    def unsubscribe_activity(
        self, session_id: str, queue: asyncio.Queue[BackgroundTaskActivity]
    ) -> None:
        self.unsubscribed.append(session_id)


def _activity(task_id: str, kind: str, *, source: str = "scout", **fields: Any) -> BackgroundTaskActivity:
    return BackgroundTaskActivity(
        task_id=task_id,
        session_id=THREAD,
        agent_name="scout",
        description="research",
        event=AgentStreamEvent(kind=kind, source=source, **fields),  # type: ignore[arg-type]
    )


class _ScriptedRunner:
    """A turn that publishes background activity while it runs."""

    def __init__(self, manager: _FakeManager, activities: list[BackgroundTaskActivity]) -> None:
        self.manager = manager
        self.activities = activities

    async def run(self, request: Any, renderer: Any) -> None:
        for activity in self.activities:
            for queue in self.manager.queues:
                queue.put_nowait(activity)
        # Let the forwarder drain before the turn finishes.
        while any(not queue.empty() for queue in self.manager.queues):  # noqa: ASYNC110 - polling the forwarder's queue
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        await renderer.on_event(AgentStreamEvent(kind="response_delta", source="main-agent", text="done"))


def _context() -> chainagents_api.AgentRunContext:
    return chainagents_api.AgentRunContext(
        prompt="go",
        thread_id=THREAD,
        model_name="fake",
        reasoning_level="medium",
        reasoning_level_is_explicit=False,
        async_subagent_url=None,
        mcp_session_id=None,
        history=(),
        source_thread_id=None,
    )


async def _stream(manager: _FakeManager | None, activities: list[BackgroundTaskActivity]) -> list[dict[str, Any]]:
    renderer = chainagents_api.NdjsonRenderer(
        _context(),
        issue_reflection_token=lambda *_: "token",
        background_activity=manager,
    )
    runner = _ScriptedRunner(manager or _FakeManager(), activities)
    return [json.loads(line) async for line in renderer.lines(runner, object())]  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_background_activity_streams_with_one_label_per_task() -> None:
    manager = _FakeManager()
    events = await _stream(
        manager,
        [
            _activity("t1", "reasoning_delta", text="plan A"),
            _activity("t2", "reasoning_delta", text="plan B"),
            _activity("t1", "tool_call", tool_call_id="c1", tool_name="write_todos", tool_args="{}"),
            _activity("t2", "tool_result", source="scout / helper", tool_call_id="c2", tool_name="ls", tool_result="ok"),
        ],
    )

    forwarded = [(e["kind"], e["source"]) for e in events if e["source"] != "main-agent"]
    assert forwarded == [
        ("reasoning_delta", "scout"),
        ("reasoning_delta", "scout 2"),
        ("tool_call", "scout"),
        ("tool_result", "scout 2 / helper"),
    ]
    assert events[-1] == {**events[-1], "kind": "response_delta", "source": "main-agent"}
    assert manager.unsubscribed == [THREAD]


@pytest.mark.anyio
async def test_background_responses_and_terminal_snapshots_are_not_forwarded() -> None:
    manager = _FakeManager()
    terminal = BackgroundTaskActivity(task_id="t1", session_id=THREAD, agent_name="scout", description="research")
    events = await _stream(
        manager,
        [_activity("t1", "response_delta", text="child answer"), terminal],
    )

    assert [e["source"] for e in events] == ["main-agent"]


@pytest.mark.anyio
async def test_no_subscription_without_background_activity() -> None:
    events = await _stream(None, [_activity("t1", "reasoning_delta", text="plan")])

    assert [e["source"] for e in events] == ["main-agent"]


def test_stream_activity_switch_selects_the_manager() -> None:
    on = _FakeManager(stream_activity=True)
    off = _FakeManager(stream_activity=False)

    assert chainagents_api._background_activity_manager(SimpleNamespace(background_tasks=on)) is on
    assert chainagents_api._background_activity_manager(SimpleNamespace(background_tasks=off)) is None
    assert chainagents_api._background_activity_manager(SimpleNamespace()) is None
