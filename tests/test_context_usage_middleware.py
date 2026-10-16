"""Tests for the context usage middleware."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import chainagents.runtime.context_usage as runtime_context_usage
from chainagents.runtime.constants import SUMMARIZATION_STATUS_EVENT_KIND
from chainagents.runtime.context_usage import (
    CATEGORY_NAMES,
    ContextUsageMiddleware,
    ContextUsageRegistry,
)
from chainagents.runtime.types import ExtensionsConfig

THREAD_ID = "thread-1"
NOTICE = "Conversation summarized to fit the context window."


class _Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.writes: list[dict[str, Any]] = []
        self.config: dict[str, Any] = {"configurable": {"thread_id": THREAD_ID}, "metadata": {}}
        self.registry = ContextUsageRegistry()
        self.middleware = ContextUsageMiddleware(
            registry=self.registry, extensions=ExtensionsConfig(config_path=None)
        )
        monkeypatch.setattr(runtime_context_usage, "get_config", lambda: self.config)
        monkeypatch.setattr(runtime_context_usage, "get_stream_writer", lambda: self.writes.append)

    def call(
        self,
        messages: list[Any] | None = None,
        usage: dict[str, int] | None = None,
    ) -> ModelResponse[Any]:
        model = FakeListChatModel(responses=["ok"])
        model.profile = {"max_input_tokens": 16384}
        request = ModelRequest(
            model=model,
            messages=messages or [HumanMessage(content="hello there")],
            system_message=SystemMessage(content="You are a helpful assistant."),
            tools=[],
        )
        response = ModelResponse(result=[AIMessage(content="ok", usage_metadata=usage)])

        async def handler(_: ModelRequest[Any]) -> ModelResponse[Any]:
            return response

        result = asyncio.run(self.middleware.awrap_model_call(request, handler))
        assert result is response
        return result

    def kinds(self, kind: str) -> list[dict[str, Any]]:
        return [write for write in self.writes if write.get("kind") == kind]


def _usage(input_tokens: int = 500) -> dict[str, int]:
    return {"input_tokens": input_tokens, "output_tokens": 5, "total_tokens": input_tokens + 5}


def _summary(text: str) -> HumanMessage:
    return HumanMessage(content=text, additional_kwargs={"lc_source": "summarization"})


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Harness]:
    yield _Harness(monkeypatch)


def test_main_agent_snapshot_is_recorded_and_streamed(harness: _Harness) -> None:
    harness.call(usage=_usage(500))

    events = harness.kinds("context_usage")
    assert len(events) == 1
    snapshot = events[0]["snapshot"]
    assert snapshot["agent"] == "main"
    assert snapshot["kind"] == "main"
    assert snapshot["used"] == 500
    assert snapshot["exact"] is True
    assert snapshot["window"] == 16384
    assert snapshot["trigger"] == 13926
    assert snapshot["keep"] == 1638
    assert snapshot["status"] == "active"
    assert sum(snapshot["categories"].values()) == 500
    recorded = harness.registry.list(THREAD_ID)
    assert [item.to_payload() for item in recorded] == [snapshot]


def test_subagent_is_recorded_not_streamed(harness: _Harness) -> None:
    harness.config["metadata"] = {"lc_agent_name": "scout"}

    harness.call(usage=_usage())

    assert harness.writes == []
    [snapshot] = harness.registry.list(THREAD_ID)
    assert snapshot.agent == "scout"
    assert snapshot.kind == "subagent"


def test_background_run_uses_owning_session(harness: _Harness) -> None:
    harness.config["configurable"] = {
        "thread_id": "s1:background:t1",
        "chainagents_session_id": "s1",
        "chainagents_agent_kind": "background",
        "chainagents_agent_name": "researcher",
        "chainagents_agent_instance": "t1",
    }
    harness.config["metadata"] = {"lc_agent_name": "researcher"}
    recorded: list[tuple[str, str]] = []
    original_record = harness.registry.record

    def spy(session_id: str, key: str, snapshot: Any) -> None:
        recorded.append((session_id, key))
        original_record(session_id, key, snapshot)

    harness.registry.record = spy  # type: ignore[method-assign]

    harness.call(usage=_usage())

    assert recorded == [("s1", "t1")]
    assert harness.writes == []
    [snapshot] = harness.registry.list("s1")
    assert snapshot.kind == "background"
    assert snapshot.agent == "researcher"
    assert harness.registry.list("s1:background:t1") == []


def test_nested_subagent_in_background_run_gets_its_own_entry(harness: _Harness) -> None:
    harness.config["configurable"] = {
        "thread_id": "s1:background:t1",
        "chainagents_session_id": "s1",
        "chainagents_agent_kind": "background",
        "chainagents_agent_name": "researcher",
        "chainagents_agent_instance": "t1",
    }
    harness.call(usage=_usage(500))
    harness.config["metadata"] = {"lc_agent_name": "helper"}

    harness.call(usage=_usage(700))

    assert harness.writes == []
    snapshots = {item.agent: item for item in harness.registry.list("s1")}
    assert set(snapshots) == {"researcher", "helper"}
    assert (snapshots["researcher"].kind, snapshots["researcher"].used) == ("background", 500)
    assert (snapshots["helper"].kind, snapshots["helper"].used) == ("subagent", 700)
    harness.registry.mark_ended("s1", "t1:helper", "finished")
    statuses = {item.agent: item.status for item in harness.registry.list("s1")}
    assert statuses == {"researcher": "active", "helper": "finished"}


def test_missing_usage_keeps_estimates(harness: _Harness) -> None:
    harness.call(usage=None)

    [snapshot] = harness.registry.list(THREAD_ID)
    assert snapshot.exact is False
    assert set(snapshot.categories) == set(CATEGORY_NAMES)
    assert snapshot.used == sum(snapshot.categories.values())
    assert snapshot.used > 0


def test_summary_notice_fires_once_per_new_summary(harness: _Harness) -> None:
    first = [_summary("Summary one."), HumanMessage(content="next question")]
    second = [_summary("Summary two."), HumanMessage(content="another question")]

    harness.call(first, usage=_usage())
    harness.call(first, usage=_usage())
    harness.call(second, usage=_usage())

    notices = harness.kinds(SUMMARIZATION_STATUS_EVENT_KIND)
    assert notices == [
        {
            "kind": SUMMARIZATION_STATUS_EVENT_KIND,
            "status": "completed",
            "source": "main",
            "message": NOTICE,
        }
    ] * 2
    snapshots = [event["snapshot"] for event in harness.kinds("context_usage")]
    assert all(item["last_summarized_at_utc"] for item in snapshots)
    assert snapshots[0]["categories"]["summary"] > 0


def test_measurement_errors_do_not_fail_the_call(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any) -> dict[str, int]:
        raise ValueError("boom")

    monkeypatch.setattr(runtime_context_usage, "estimate_categories", broken)

    harness.call(usage=_usage())

    assert harness.writes == []


def test_model_errors_propagate(harness: _Harness) -> None:
    model = FakeListChatModel(responses=["ok"])
    request = ModelRequest(model=model, messages=[HumanMessage(content="hi")])
    error = RuntimeError("model down")

    async def handler(_: ModelRequest[Any]) -> ModelResponse[Any]:
        raise error

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(harness.middleware.awrap_model_call(request, handler))
    assert raised.value is error
    assert harness.writes == []


def test_sync_wrap_model_call_mirrors_async(harness: _Harness) -> None:
    model = FakeListChatModel(responses=["ok"])
    model.profile = {"max_input_tokens": 16384}
    request = ModelRequest(model=model, messages=[HumanMessage(content="hi")])
    response = AIMessage(content="ok", usage_metadata=_usage(42))

    result = harness.middleware.wrap_model_call(request, lambda _: response)

    assert result is response
    [event] = harness.kinds("context_usage")
    assert event["snapshot"]["used"] == 42


def test_no_graph_context_records_nothing(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def outside() -> Any:
        raise RuntimeError("Called get_config outside of a runnable context")

    monkeypatch.setattr(runtime_context_usage, "get_config", outside)

    harness.call(usage=_usage())

    assert harness.writes == []
    assert harness.registry.list(THREAD_ID) == []
