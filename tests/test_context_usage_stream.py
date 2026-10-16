"""Context-usage stream events: adapter mapping, NDJSON payload and renderers."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest

from chainagents.events.stream import AgentStreamEvent, AgentStreamEventAdapter
from chainagents.interfaces.api.app import _event_payload
from chainagents.interfaces.cli.render import CliEventRenderer
from chainagents.interfaces.tui.app import TuiRenderer


def _custom_event(data: object) -> dict[str, object]:
    return {
        "event": "on_chain_stream",
        "parent_ids": [],
        "data": {"chunk": {"type": "custom", "ns": (), "data": data}},
    }


def test_custom_chunk_becomes_context_usage_event() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    events = adapter.events_from_raw_event(
        _custom_event({"kind": "context_usage", "snapshot": {"used": 5}})
    )

    assert len(events) == 1
    assert events[0].kind == "context_usage"
    assert events[0].source == "main-agent"
    assert events[0].context_usage == {"used": 5}


@pytest.mark.parametrize("snapshot", ["nope", None, [1]])
def test_malformed_context_usage_chunk_is_ignored(snapshot: object) -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    events = adapter.events_from_raw_event(
        _custom_event({"kind": "context_usage", "snapshot": snapshot})
    )

    assert events == []


def test_ndjson_payload_carries_context_usage_only_for_its_kind() -> None:
    context = MagicMock(thread_id="t", model_name="m", reasoning_level="low")
    usage = AgentStreamEvent(
        kind="context_usage", source="main-agent", context_usage={"used": 5}
    )
    delta = AgentStreamEvent(kind="response_delta", source="main-agent", text="hi")

    assert _event_payload(usage, context)["context_usage"] == {"used": 5}
    assert "context_usage" not in _event_payload(delta, context)


@pytest.mark.anyio
async def test_cli_and_tui_ignore_context_usage() -> None:
    event = AgentStreamEvent(
        kind="context_usage", source="main-agent", context_usage={"used": 5}
    )
    stdout, stderr = io.StringIO(), io.StringIO()
    cli = CliEventRenderer(
        stdout=stdout,
        stderr=stderr,
        stream=True,
        json_output=False,
        show_reasoning=True,
        show_tools=True,
    )
    await cli.on_event(event)
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""

    app = MagicMock()
    await TuiRenderer(app).on_event(event)
    assert app.method_calls == []
