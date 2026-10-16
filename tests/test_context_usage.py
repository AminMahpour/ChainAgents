"""Tests for context usage snapshots, estimation and the registry."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from chainagents.runtime.context_usage import (
    CATEGORY_NAMES,
    MAX_AGENTS_PER_SESSION,
    ContextUsageRegistry,
    ContextUsageSnapshot,
    estimate_categories,
    scale_to_total,
)


@tool
def lookup(query: str) -> str:
    """Look something up in the knowledge base."""
    return query


def _snapshot(
    agent: str = "main",
    kind: str = "main",
    at_utc: str = "2026-10-15T10:00:00+00:00",
    **overrides: Any,
) -> ContextUsageSnapshot:
    values: dict[str, Any] = {
        "agent": agent,
        "kind": kind,
        "model": "m",
        "window": 1000,
        "used": 10,
        "categories": dict.fromkeys(CATEGORY_NAMES, 2),
        "trigger": 850,
        "keep": 100,
        "exact": True,
        "at_utc": at_utc,
        "last_summarized_at_utc": None,
        "status": "active",
    }
    values.update(overrides)
    return ContextUsageSnapshot(**values)


def test_estimate_categories_splits_summary_tool_and_other_messages() -> None:
    summary = HumanMessage(
        content="Summary of earlier conversation " * 5,
        additional_kwargs={"lc_source": "summarization"},
    )
    tool_result = ToolMessage(content="result " * 20, tool_call_id="1")
    estimates = estimate_categories(
        SystemMessage(content="You are helpful."),
        [lookup],
        [summary, HumanMessage(content="hello there"), AIMessage(content="hi"), tool_result],
    )
    assert set(estimates) == set(CATEGORY_NAMES)
    assert all(value > 0 for value in estimates.values())
    assert estimates["summary"] > 0
    only_summary = estimate_categories(None, [], [summary])
    assert estimates["summary"] == only_summary["summary"]
    only_tool_result = estimate_categories(None, [], [tool_result])
    assert estimates["tool_results"] == only_tool_result["tool_results"]
    assert only_summary["messages"] == 0
    assert only_tool_result["messages"] == 0


def test_scale_to_total_makes_system_the_remainder() -> None:
    estimates = {"system": 100, "tools": 200, "summary": 0, "messages": 300, "tool_results": 100}
    assert scale_to_total(estimates, 1000) == {
        "system": 400,
        "tools": 200,
        "summary": 0,
        "messages": 300,
        "tool_results": 100,
    }


def test_scale_to_total_shrinks_others_when_they_exceed_total() -> None:
    estimates = {"system": 100, "tools": 200, "summary": 0, "messages": 300, "tool_results": 100}
    scaled = scale_to_total(estimates, 300)
    assert sum(scaled.values()) == 300
    assert scaled["system"] == 0
    assert scaled["messages"] >= scaled["tools"] >= scaled["tool_results"]


def test_registry_orders_main_first_then_newest() -> None:
    registry = ContextUsageRegistry()
    registry.record("s", "a", _snapshot("a", "subagent", "2026-10-15T10:00:01+00:00"))
    registry.record("s", "main:main", _snapshot("main", "main", "2026-10-15T09:00:00+00:00"))
    registry.record("s", "b", _snapshot("b", "background", "2026-10-15T10:00:05+00:00"))
    assert [item.agent for item in registry.list("s")] == ["main", "b", "a"]


def test_registry_evicts_least_recently_updated_beyond_cap() -> None:
    registry = ContextUsageRegistry()
    for index in range(MAX_AGENTS_PER_SESSION + 1):
        registry.record("s", f"k{index}", _snapshot(f"a{index}", "subagent"))
    agents = {item.agent for item in registry.list("s")}
    assert len(agents) == MAX_AGENTS_PER_SESSION
    assert "a0" not in agents
    assert "a1" in agents


def test_sessions_are_isolated() -> None:
    registry = ContextUsageRegistry()
    registry.record("s1", "k", _snapshot("one"))
    registry.record("s2", "k", _snapshot("two"))
    assert [item.agent for item in registry.list("s2")] == ["two"]
    registry.clear("s1")
    assert registry.list("s1") == []
    assert [item.agent for item in registry.list("s2")] == ["two"]


def test_note_summary_is_true_once_per_digest() -> None:
    registry = ContextUsageRegistry()
    assert registry.last_summarized("s", "k") is None
    assert registry.note_summary("s", "k", "d1") is True
    assert registry.note_summary("s", "k", "d1") is False
    assert registry.last_summarized("s", "k") is not None
    assert registry.note_summary("s", "k", "d2") is True
    registry.clear("s")
    assert registry.last_summarized("s", "k") is None
    assert registry.note_summary("s", "k", "d2") is True


def test_mark_ended_updates_status() -> None:
    registry = ContextUsageRegistry()
    registry.record("s", "k", _snapshot("bg", "background"))
    registry.mark_ended("s", "k", "stopped")
    registry.mark_ended("s", "missing", "stopped")
    assert [item.status for item in registry.list("s")] == ["stopped"]


def test_snapshot_payload_has_spec_fields() -> None:
    snapshot = _snapshot()
    payload = snapshot.to_payload()
    assert set(payload) == {
        "agent",
        "kind",
        "model",
        "window",
        "used",
        "categories",
        "trigger",
        "keep",
        "exact",
        "at_utc",
        "last_summarized_at_utc",
        "status",
    }
    assert payload["categories"] == snapshot.categories
    assert payload["categories"] is not snapshot.categories


def test_mark_ended_also_ends_the_instances_nested_children() -> None:
    reg = ContextUsageRegistry()
    reg.record("s", "t1", _snapshot("scout", "background"))
    reg.record("s", "t1:helper", _snapshot("helper", "subagent"))
    reg.record("s", "t10", _snapshot("other", "background"))

    reg.mark_ended("s", "t1", "finished")

    statuses = {s.agent: s.status for s in reg.list("s")}
    assert statuses == {"scout": "finished", "helper": "finished", "other": "active"}


def test_eviction_drops_the_evicted_agents_summary_state() -> None:
    reg = ContextUsageRegistry()
    reg.record("s", "a0", _snapshot("a0", "subagent"))
    assert reg.note_summary("s", "a0", "digest") is True
    for index in range(1, 51):
        reg.record("s", f"a{index}", _snapshot(f"a{index}", "subagent"))

    assert all(s.agent != "a0" for s in reg.list("s"))
    assert reg.last_summarized("s", "a0") is None
    assert reg.note_summary("s", "a0", "digest") is True


def test_model_name_reads_bedrock_model_id() -> None:
    from types import SimpleNamespace

    from chainagents.runtime.context_usage import _model_name

    bedrock = SimpleNamespace(model_id="us.anthropic.claude-sonnet-4-5-v1:0")

    assert _model_name(bedrock) == "us.anthropic.claude-sonnet-4-5-v1:0"
    assert _model_name(SimpleNamespace(model_name="gpt", model_id="x")) == "gpt"


def test_snapshots_from_the_same_second_list_newest_first() -> None:
    reg = ContextUsageRegistry()
    same_second = "2026-10-15T10:00:00+00:00"
    reg.record("s", "a", _snapshot("first", "subagent", at_utc=same_second))
    reg.record("s", "b", _snapshot("second", "subagent", at_utc=same_second))

    assert [s.agent for s in reg.list("s")] == ["second", "first"]
