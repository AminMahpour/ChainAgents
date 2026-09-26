"""Test the shared agent turn runner with a scripted agent and a recording renderer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chainagents.runtime.reflection import ReflectionConfig
from chainagents.turns import BaseTurnRenderer, TurnRequest, TurnRunner


class _Token:
    """Minimal streamed AI message chunk."""

    type = "AIMessageChunk"
    additional_kwargs: dict[str, str] = {}
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, content: str = "") -> None:
        self.content = content


def _raw(chunk: object) -> dict[str, object]:
    return {"event": "on_chain_stream", "data": {"chunk": chunk}}


def _token(text: str) -> dict[str, object]:
    return _raw(((), "messages", (_Token(text), {})))


def _write_file_events(path: str) -> list[dict[str, object]]:
    call = _Token()
    call.tool_call_chunks = [
        {
            "id": "call-1",
            "name": "write_file",
            "args": json.dumps({"file_path": path, "content": "x"}),
        }
    ]
    result = SimpleNamespace(
        type="tool",
        name="write_file",
        status="success",
        tool_call_id="call-1",
        content="ok",
    )
    return [_raw(((), "messages", (call, {}))), _raw(((), "messages", (result, {})))]


class _FakeStream:
    """Async iterator over scripted raw events that records closing."""

    def __init__(
        self,
        events: list[dict[str, object]],
        *,
        error: Exception | None = None,
        block: bool = False,
    ) -> None:
        self.events = list(events)
        self.error = error
        self.block = block
        self.closed = False
        self.started = asyncio.Event()

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> dict[str, object]:
        if self.events:
            return self.events.pop(0)
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _FakeAgent:
    def __init__(self, stream: _FakeStream) -> None:
        self.stream = stream
        self.payload: dict[str, Any] | None = None
        self.config: dict[str, Any] | None = None
        self.stream_mode: list[str] | None = None

    def astream_events(self, payload, *, config, version, stream_mode, subgraphs):
        self.payload = payload
        self.config = config
        self.stream_mode = stream_mode
        return self.stream


class _FakeRuntime:
    def __init__(self, agent: _FakeAgent, project_root: Path) -> None:
        self.agent = agent
        self.project_root = project_root
        self.agent_requests: list[dict[str, Any]] = []
        self.command_requests: list[dict[str, Any]] = []
        self.command_error: Exception | None = None
        self.commands: dict[str, Any] = {
            "lookup": SimpleNamespace(
                name="lookup",
                description="Look up",
                target="mcp_tool",
                value="lookup",
                template=None,
                mcp_server="docs",
            ),
            "review": SimpleNamespace(
                name="review",
                description="Review",
                target="prompt",
                value="Review it",
                template="Review {input}",
                mcp_server=None,
            ),
        }
        self.config = SimpleNamespace(
            recursion_limit=50,
            extensions=SimpleNamespace(agent_reflection=ReflectionConfig(enabled=True)),
        )

    async def get_agent(self, *args, **kwargs):
        self.agent_requests.append({"args": args, "kwargs": kwargs})
        return self.agent

    def resolve_chainlit_command(self, name: str):
        return self.commands.get(name)

    async def invoke_mcp_tool_command(self, **kwargs):
        self.command_requests.append(kwargs)
        if self.command_error is not None:
            raise self.command_error
        return {"echo": kwargs["raw_args"]}


class _RecordingRenderer(BaseTurnRenderer):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def kinds(self) -> list[str]:
        return [name for name, _value in self.calls]

    async def on_event(self, event):
        self.calls.append(("event", event))

    async def on_command_result(self, result):
        self.calls.append(("command_result", result))

    async def on_command_error(self, exc, status):
        self.calls.append(("command_error", (exc, status)))

    async def on_generated_files(self, files):
        self.calls.append(("generated_files", files))

    async def on_reflection(self, proposal):
        self.calls.append(("reflection", proposal))

    async def on_cancelled(self):
        self.calls.append(("cancelled", None))

    async def on_error(self, exc):
        self.calls.append(("error", exc))

    async def on_complete(self, result):
        self.calls.append(("complete", result))


def _request(prompt: str, **kwargs: Any) -> TurnRequest:
    values: dict[str, Any] = {
        "prompt": prompt,
        "thread_id": "thread-1",
        "model_name": "fake-model",
        "reasoning_level": "medium",
    }
    values.update(kwargs)
    return TurnRequest(**values)


def _run(runtime: _FakeRuntime, request: TurnRequest):
    renderer = _RecordingRenderer()
    result = asyncio.run(TurnRunner(runtime).run(request, renderer))
    return result, renderer


def _make_runtime(
    tmp_path: Path,
    events: list[dict[str, object]] | None = None,
    **stream_kwargs: Any,
) -> _FakeRuntime:
    stream = _FakeStream(events if events is not None else [], **stream_kwargs)
    return _FakeRuntime(_FakeAgent(stream), tmp_path)


def test_prompt_turn_streams_events_and_completes(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, [_token("Hello"), _token(" world")])

    result, renderer = _run(
        runtime,
        _request(
            "hello",
            reasoning_level="high",
            reasoning_level_is_explicit=True,
            mcp_session_id="thread-1",
            history=({"role": "assistant", "content": "earlier"},),
            run_config_extras={"tags": ["extra"]},
        ),
    )

    assert result.status == "completed" and result.ok
    assert result.response == "Hello world"
    assert result.agent is runtime.agent
    assert renderer.kinds() == ["event", "event", "complete"]
    assert [event.text for name, event in renderer.calls if name == "event"] == [
        "Hello",
        " world",
    ]
    assert runtime.agent_requests == [
        {
            "args": ("high",),
            "kwargs": {
                "model_name": "fake-model",
                "reasoning_level_is_explicit": True,
                "thread_id": "thread-1",
                "async_subagent_url_override": None,
                "mcp_session_id": "thread-1",
            },
        }
    ]
    assert runtime.agent.payload == {
        "messages": [
            {"role": "assistant", "content": "earlier"},
            {"role": "user", "content": "hello"},
        ]
    }
    assert runtime.agent.config == {
        "configurable": {"thread_id": "thread-1"},
        "recursion_limit": 50,
        "tags": ["extra"],
    }
    assert runtime.agent.stream_mode == ["messages", "updates", "custom"]
    assert runtime.agent.stream.closed


def test_image_parts_and_prompt_decoration_build_one_payload(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, [_token("Seen")])
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}

    result, _renderer = _run(
        runtime,
        _request(
            "",
            content_parts=(image,),
            image_names=("scan.png",),
            prompt_note="\n\nNote.",
        ),
    )

    text = (
        "Extract any visible text from the attached image(s).\n\n"
        "Attached image file(s): `scan.png`. Use the image content directly "
        "when answering.\n\nNote."
    )
    assert result.prompt == text
    assert runtime.agent.payload == {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": text}, image]}
        ]
    }


def test_prompt_command_transforms_prompt(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, [_token("ok")])

    result, _renderer = _run(runtime, _request("/review the diff"))

    assert result.status == "completed"
    assert result.command_result is not None
    assert result.prompt == "Review the diff"
    assert runtime.agent.payload == {
        "messages": [{"role": "user", "content": "Review the diff"}]
    }


def test_mcp_command_reports_output_without_running_agent(tmp_path: Path) -> None:
    (tmp_path / ".files" / "outputs").mkdir(parents=True)
    (tmp_path / ".files" / "outputs" / "a.txt").write_text("x")
    runtime = _make_runtime(tmp_path)

    result, renderer = _run(
        runtime,
        _request(".files/outputs/a.txt", selected_command="lookup"),
    )

    assert result.status == "completed"
    assert json.loads(result.response) == {"echo": ".files/outputs/a.txt"}
    assert renderer.kinds() == ["command_result", "generated_files", "complete"]
    assert renderer.calls[0][1].tool_result == {"echo": ".files/outputs/a.txt"}
    assert [file.name for file in result.generated_files] == ["a.txt"]
    assert runtime.agent_requests == []
    assert runtime.command_requests[0]["thread_id"] == "thread-1"


def test_unknown_slash_command_is_an_error(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    result, renderer = _run(runtime, _request("/nope please"))

    assert result.status == "command_error" and not result.ok
    assert renderer.kinds() == ["command_error", "complete"]
    exc, status = renderer.calls[0][1]
    assert status == 422 and exc.unknown and exc.command_name == "nope"
    assert exc.message == "Unknown command `/nope`."
    assert runtime.agent_requests == []


def test_unknown_selected_command_is_an_error(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    result, _renderer = _run(runtime, _request("plain", selected_command="nope"))

    assert result.command_error is not None and result.command_error.unknown


def test_plain_text_is_a_prompt_not_a_command(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, [_token("hi")])

    result, renderer = _run(runtime, _request("nope please"))

    assert result.status == "completed"
    assert "command_error" not in renderer.kinds()
    assert runtime.command_requests == []


@pytest.mark.parametrize(
    ("error", "status", "message"),
    [
        (
            ValueError("Command arguments for lookup must be valid JSON."),
            422,
            "Command arguments must be valid JSON.",
        ),
        (
            ValueError("credential=supersecret"),
            422,
            "Command arguments could not be validated.",
        ),
        (
            RuntimeError("credential=supersecret"),
            500,
            "Agent operation failed. Please retry.",
        ),
    ],
)
def test_command_errors_are_classified_and_sanitised(
    tmp_path: Path, error: Exception, status: int, message: str
) -> None:
    runtime = _make_runtime(tmp_path)
    runtime.command_error = error

    result, renderer = _run(runtime, _request("/lookup q"))

    assert renderer.kinds() == ["command_error", "complete"]
    exc, reported_status = renderer.calls[0][1]
    assert reported_status == status == exc.status
    assert str(exc) == message and not exc.unknown
    assert exc.__cause__ is error
    assert result.command_error is exc
    assert runtime.agent_requests == []


def test_blank_command_prompt_skips_agent(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    runtime.commands["blank"] = SimpleNamespace(
        name="blank",
        description="",
        target="prompt",
        value="",
        template="",
        mcp_server=None,
    )

    result, renderer = _run(runtime, _request("/blank"))

    assert result.status == "skipped" and result.ok
    assert renderer.kinds() == ["complete"]
    assert runtime.agent_requests == []


def test_failure_still_emits_generated_files_and_reflection(tmp_path: Path) -> None:
    output = tmp_path / ".files" / "outputs" / "report.csv"
    output.parent.mkdir(parents=True)
    output.write_text("a,b\n")
    error = RuntimeError("credential=supersecret")
    runtime = _make_runtime(
        tmp_path,
        _write_file_events("/workspace/.files/outputs/report.csv"),
        error=error,
    )

    result, renderer = _run(runtime, _request("That was wrong, fix it"))

    assert result.status == "failed" and result.error is error
    assert renderer.kinds()[-4:] == [
        "generated_files",
        "reflection",
        "error",
        "complete",
    ]
    assert [file.path for file in result.generated_files] == [output]
    assert result.reflection is not None
    assert "supersecret" not in json.dumps(result.reflection.to_payload())
    assert renderer.calls[-2][1] is error
    assert runtime.agent.stream.closed


def test_cancellation_notifies_renderer_reraises_and_closes_stream(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path, [_token("partial")], block=True)
    renderer = _RecordingRenderer()

    async def exercise() -> None:
        task = asyncio.create_task(TurnRunner(runtime).run(_request("hi"), renderer))
        await runtime.agent.stream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert renderer.kinds() == ["event", "cancelled"]
    assert runtime.agent.stream.closed


def test_unsanitised_runner_keeps_real_error_text(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    runtime.command_error = RuntimeError("backend exploded")
    renderer = _RecordingRenderer()

    result = asyncio.run(
        TurnRunner(runtime, sanitize_errors=False).run(
            _request("/lookup q"), renderer
        )
    )

    assert result.command_error is not None
    assert result.command_error.message == "backend exploded"
    assert result.command_error.status == 500

    runtime.command_error = ValueError("bad args")
    result = asyncio.run(
        TurnRunner(runtime, sanitize_errors=False).run(
            _request("/lookup q"), _RecordingRenderer()
        )
    )
    assert result.command_error is not None
    assert result.command_error.message == "bad args"
    assert result.command_error.status == 422


@pytest.mark.parametrize(
    ("sanitize", "expected", "hidden"),
    [
        (True, "Agent operation failed.", "credential=supersecret"),
        (False, "credential=supersecret", None),
    ],
)
def test_failure_reflection_text_follows_sanitize_mode(
    tmp_path: Path, sanitize: bool, expected: str, hidden: str | None
) -> None:
    error = RuntimeError("credential=supersecret")
    call = _Token()
    call.tool_call_chunks = [{"id": "call-1", "name": "read_file", "args": "{}"}]
    failed = SimpleNamespace(
        type="tool",
        name="read_file",
        status="error",
        tool_call_id="call-1",
        content="",
    )
    runtime = _make_runtime(
        tmp_path,
        [_raw(((), "messages", (call, {}))), _raw(((), "messages", (failed, {})))],
        error=error,
    )

    result = asyncio.run(
        TurnRunner(runtime, sanitize_errors=sanitize).run(
            _request("hi"), _RecordingRenderer()
        )
    )

    assert result.error is error
    assert result.reflection is not None
    payload = json.dumps(result.reflection.to_payload())
    assert expected in payload
    if hidden is not None:
        assert hidden not in payload


def test_mcp_outage_is_reported_as_mcp_status_event(tmp_path: Path) -> None:
    class _DegradedRuntime(_FakeRuntime):
        async def get_agent_with_status(self, *args, **kwargs):
            self.agent_requests.append({"args": args, "kwargs": kwargs})
            return self.agent, ("docs",)

    runtime = _DegradedRuntime(_FakeAgent(_FakeStream([_token("Ready")])), tmp_path)

    result, renderer = _run(runtime, _request("hello"))

    events = [value for name, value in renderer.calls if name == "event"]
    assert events[0].kind == "mcp_status"
    assert events[0].status == "warning"
    assert events[0].text == (
        "MCP server unavailable: docs. Continuing with available tools."
    )
    assert result.response == "Ready"


def test_mcp_command_can_emit_reflection_from_user_text(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)

    result, renderer = _run(runtime, _request("/lookup That was wrong, fix it"))

    assert renderer.kinds() == ["command_result", "reflection", "complete"]
    assert result.reflection is not None
    assert result.reflection.reason == "correction"
    assert runtime.agent_requests == []


def test_agent_start_follows_mcp_status_with_final_prompt(tmp_path: Path) -> None:
    class _DegradedRuntime(_FakeRuntime):
        async def get_agent_with_status(self, *args, **kwargs):
            return self.agent, ("docs",)

    class _StartRecorder(_RecordingRenderer):
        async def on_agent_start(self, prompt):
            self.calls.append(("agent_start", prompt))

    runtime = _DegradedRuntime(_FakeAgent(_FakeStream([_token("Ready")])), tmp_path)
    renderer = _StartRecorder()

    asyncio.run(TurnRunner(runtime).run(_request("/review the diff"), renderer))

    assert renderer.kinds() == ["event", "agent_start", "event", "complete"]
    assert renderer.calls[0][1].kind == "mcp_status"
    assert renderer.calls[1][1] == "Review the diff"


def test_resolve_commands_off_sends_slash_text_as_prompt(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, [_token("Done")])

    result, renderer = _run(
        runtime, _request("/lookup keep this literal", resolve_commands=False)
    )

    assert result.status == "completed"
    assert runtime.command_requests == []
    assert runtime.agent.payload == {
        "messages": [{"role": "user", "content": "/lookup keep this literal"}]
    }
