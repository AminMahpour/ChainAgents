from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import chainagents_cli
from deepagent_runtime import RuntimeConfig
from rag_runtime import RagStatus, RagUploadResult


def test_cli_parses_prompt_and_runtime_flags() -> None:
    args = chainagents_cli.parse_args(
        [
            "--prompt",
            "hello",
            "--photo",
            "image.png",
            "--model",
            "gemma4:26b",
            "--reasoning",
            "high",
            "--thread-id",
            "thread-1",
            "--no-stream",
            "--json",
        ]
    )

    assert chainagents_cli.prompt_from_args(args, stdin=io.StringIO("")) == "hello"
    assert args.photo == ["image.png"]
    assert args.model == "gemma4:26b"
    assert args.reasoning == "high"
    assert args.thread_id == "thread-1"
    assert args.stream is False
    assert args.json_output is True


def test_runtime_overrides_from_cli_args_take_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://config.example:11434"
name = "config-model"
temperature = 0.1
reasoning_effort = "low"

[agent]
recursion_limit = 20
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://env/db")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "env-model")
    monkeypatch.setenv("DEEPAGENT_MODEL_REASONING", "medium")

    args = chainagents_cli.parse_args(
        [
            "--config",
            str(config_path),
            "--database-url",
            "postgresql://cli/db",
            "--provider",
            "openai_compatible",
            "--base-url",
            "http://cli.example/v1",
            "--model",
            "cli-model",
            "--api-key",
            "cli-key",
            "--temperature",
            "0.7",
            "--reasoning",
            "high",
            "--recursion-limit",
            "77",
            "--no-rag",
            "--status",
        ]
    )

    config = RuntimeConfig.from_env(chainagents_cli.runtime_overrides_from_args(args))

    assert config.database_url == "postgresql://cli/db"
    assert config.model_provider == "openai_compatible"
    assert config.model_base_url == "http://cli.example/v1"
    assert config.model_name == "cli-model"
    assert config.model_api_key == "cli-key"
    assert config.model_temperature == 0.7
    assert config.default_reasoning == "high"
    assert config.recursion_limit == 77
    assert config.rag_requested is False


class _FakeMcpRuntime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            default_reasoning="medium",
            model_name="fake-model",
            recursion_limit=100,
        )
        self.invocation: dict[str, str | None] | None = None

    def resolve_chainlit_command(self, name: str):
        if name != "repo-readme":
            return None
        return SimpleNamespace(
            name="repo-readme",
            description="Read README",
            target="mcp_tool",
            value="repo_read_file",
            template='{"path": "README.md"}',
            mcp_server="repo",
        )

    async def invoke_mcp_tool_command(
        self,
        *,
        tool_name: str,
        raw_args: str,
        thread_id: str,
        mcp_session_id: str | None,
        server_name: str | None,
    ):
        self.invocation = {
            "tool_name": tool_name,
            "raw_args": raw_args,
            "thread_id": thread_id,
            "mcp_session_id": mcp_session_id,
            "server_name": server_name,
        }
        return {"ok": True}


@pytest.mark.anyio
async def test_cli_command_invokes_configured_mcp_tool() -> None:
    args = chainagents_cli.parse_args(
        [
            "--prompt",
            "ignored",
            "--command",
            "repo-readme",
            "--thread-id",
            "thread-1",
            "--mcp-session-id",
            "session-1",
        ]
    )
    runtime = _FakeMcpRuntime()
    stdout = io.StringIO()

    code = await chainagents_cli.run_agent_prompt(
        runtime,  # type: ignore[arg-type]
        args,
        prompt="ignored",
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    assert '"ok": true' in stdout.getvalue()
    assert runtime.invocation == {
        "tool_name": "repo_read_file",
        "raw_args": '{"path": "README.md"}',
        "thread_id": "thread-1",
        "mcp_session_id": "session-1",
        "server_name": "repo",
    }


class _FakeRagRuntime:
    def __init__(self) -> None:
        self.rebuilt = False
        self.uploaded: list[str] = []

    async def rebuild_rag_index(self) -> RagStatus:
        self.rebuilt = True
        return RagStatus.ready_status(
            file_count=1,
            chunk_count=2,
            persist_directory=Path(".rag"),
        )

    async def ingest_rag_uploads(self, *, thread_id: str, uploads):
        self.uploaded = [upload.name for upload in uploads]
        return RagUploadResult(
            thread_id=thread_id,
            added_files=tuple(self.uploaded),
            indexed_files=1,
            chunk_count=3,
        )


class _CaptureAgent:
    def __init__(self) -> None:
        self.payload = None
        self.config = None

    def astream_events(self, payload, *, config, version, stream_mode, subgraphs):
        self.payload = payload
        self.config = config

        async def events():
            if False:
                yield None

        return events()


class _FakePromptRuntime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            default_reasoning="medium",
            model_name="fake-model",
            recursion_limit=100,
        )
        self.agent = _CaptureAgent()

    async def get_agent(self, *args, **kwargs):
        return self.agent


@pytest.mark.anyio
async def test_cli_runs_rag_actions_without_prompt(tmp_path: Path) -> None:
    upload = tmp_path / "notes.md"
    upload.write_text("# Notes\n", encoding="utf-8")
    args = chainagents_cli.parse_args(
        [
            "--rebuild-rag",
            "--upload-rag",
            str(upload),
            "--thread-id",
            "thread-1",
        ]
    )
    runtime = _FakeRagRuntime()
    stdout = io.StringIO()

    code = await chainagents_cli.run_cli(
        args,
        runtime=runtime,  # type: ignore[arg-type]
        stdout=stdout,
        stderr=io.StringIO(),
        stdin=io.StringIO(""),
    )

    assert code == 0
    assert runtime.rebuilt is True
    assert runtime.uploaded == ["notes.md"]
    assert "rebuild_rag: ready" in stdout.getvalue()
    assert "upload-rag: added notes.md" in stdout.getvalue()


@pytest.mark.anyio
async def test_cli_photo_attaches_image_content_to_agent_payload(tmp_path: Path) -> None:
    photo = tmp_path / "scene.png"
    photo.write_bytes(b"\x89PNG\r\n")
    args = chainagents_cli.parse_args(
        [
            "--prompt",
            "Describe this scene",
            "--photo",
            str(photo),
            "--no-stream",
        ]
    )
    runtime = _FakePromptRuntime()

    code = await chainagents_cli.run_agent_prompt(
        runtime,  # type: ignore[arg-type]
        args,
        prompt="Describe this scene",
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 0
    content = runtime.agent.payload["messages"][0]["content"]
    expected_image_url = (
        "data:image/png;base64,"
        + base64.b64encode(photo.read_bytes()).decode("ascii")
    )
    assert content == [
        {"type": "text", "text": "Describe this scene"},
        {"type": "image_url", "image_url": {"url": expected_image_url}},
    ]


@pytest.mark.anyio
async def test_cli_photo_requires_prompt(tmp_path: Path) -> None:
    photo = tmp_path / "scene.png"
    photo.write_bytes(b"\x89PNG\r\n")
    args = chainagents_cli.parse_args(["--photo", str(photo)])
    stderr = io.StringIO()

    code = await chainagents_cli.run_cli(
        args,
        runtime=_FakePromptRuntime(),  # type: ignore[arg-type]
        stdout=io.StringIO(),
        stderr=stderr,
        stdin=io.StringIO(""),
    )

    assert code == 2
    assert "provide a prompt with --photo" in stderr.getvalue()


@pytest.mark.anyio
async def test_cli_json_combines_multiple_actions(tmp_path: Path) -> None:
    upload = tmp_path / "notes.md"
    upload.write_text("# Notes\n", encoding="utf-8")
    args = chainagents_cli.parse_args(
        [
            "--json",
            "--status",
            "--list-commands",
            "--rebuild-rag",
            "--upload-rag",
            str(upload),
            "--thread-id",
            "thread-1",
        ]
    )
    runtime = _FakeRagRuntime()
    runtime.config = SimpleNamespace(  # type: ignore[attr-defined]
        model_provider="ollama",
        model_name="fake-model",
        model_choices=["fake-model"],
        model_base_url=None,
        default_reasoning="medium",
        recursion_limit=50,
        persistence_mode="memory",
        extensions=SimpleNamespace(
            config_path=None,
            skills=[],
            mcp_servers={},
            agent_mcp_servers=[],
            subagents=[],
            async_subagents=[],
        ),
    )
    runtime.rag_status = RagStatus.ready_status(  # type: ignore[attr-defined]
        file_count=1,
        chunk_count=2,
        persist_directory=Path(".rag"),
    )
    runtime.chainlit_commands = []  # type: ignore[attr-defined]
    runtime.chainlit_command_notes = []  # type: ignore[attr-defined]
    stdout = io.StringIO()

    code = await chainagents_cli.run_cli(
        args,
        runtime=runtime,  # type: ignore[arg-type]
        stdout=stdout,
        stderr=io.StringIO(),
        stdin=io.StringIO(""),
    )

    assert code == 0
    payload = json.loads(stdout.getvalue())
    assert set(payload.keys()) == {"status", "commands", "notes", "rebuild_rag", "upload_rag"}


class _Token:
    type = "AIMessageChunk"
    additional_kwargs: dict[str, str] = {}
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, content: str) -> None:
        self.content = content


class _ReasoningToken:
    type = "AIMessageChunk"
    content = ""
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, reasoning: str = "thinking") -> None:
        self.additional_kwargs = {"reasoning_content": reasoning}


class _ToolCallToken:
    type = "AIMessageChunk"
    content = ""
    additional_kwargs: dict[str, str] = {}
    tool_call_chunks = [
        {
            "id": "call-1",
            "name": "read_file",
            "args": '{"path":"README.md"}',
        }
    ]


class _ToolMessage:
    type = "tool"
    name = "read_file"
    status = "success"
    tool_call_id = "call-1"

    def __init__(self, content: str) -> None:
        self.content = content


@pytest.mark.anyio
async def test_cli_event_renderer_streams_final_response() -> None:
    stdout = io.StringIO()
    renderer = chainagents_cli.CliEventRenderer(
        prompt="hello",
        stdout=stdout,
        stderr=io.StringIO(),
        stream=True,
        json_output=False,
        show_reasoning=False,
        show_tools=False,
    )

    await renderer.handle_event(
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (_Token("Hello"), {}),
                ),
            },
        }
    )
    await renderer.handle_event(
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (_Token("Hello world"), {}),
                ),
            },
        }
    )

    response = renderer.finish()

    assert response == "Hello world"
    assert stdout.getvalue() == "Hello world\n"


@pytest.mark.anyio
async def test_cli_event_renderer_formats_reasoning_trace() -> None:
    stderr = io.StringIO()
    renderer = chainagents_cli.CliEventRenderer(
        prompt="hello",
        stdout=io.StringIO(),
        stderr=stderr,
        stream=True,
        json_output=False,
        show_reasoning=True,
        show_tools=False,
    )

    await renderer.handle_event(
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (_ReasoningToken(), {}),
                ),
            },
        }
    )

    assert "[reasoning:main-agent] thinking" in stderr.getvalue()


@pytest.mark.anyio
async def test_cli_event_renderer_appends_reasoning_chunks_inline() -> None:
    stderr = io.StringIO()
    renderer = chainagents_cli.CliEventRenderer(
        prompt="hello",
        stdout=io.StringIO(),
        stderr=stderr,
        stream=True,
        json_output=False,
        show_reasoning=True,
        show_tools=False,
    )

    for reasoning in ("thinking", "thinking through", "thinking through details"):
        await renderer.handle_event(
            {
                "event": "on_chain_stream",
                "data": {
                    "chunk": (
                        (),
                        "messages",
                        (_ReasoningToken(reasoning), {}),
                    ),
                },
            }
        )
    renderer.finish()

    output = stderr.getvalue()
    assert output.count("[reasoning:main-agent]") == 1
    assert "[reasoning:main-agent] thinking through details\n" in output


@pytest.mark.anyio
async def test_cli_event_renderer_boxes_tool_call_start() -> None:
    stderr = io.StringIO()
    renderer = chainagents_cli.CliEventRenderer(
        prompt="hello",
        stdout=io.StringIO(),
        stderr=stderr,
        stream=True,
        json_output=False,
        show_reasoning=False,
        show_tools=True,
    )

    await renderer.handle_event(
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (_ToolCallToken(), {}),
                ),
            },
        }
    )

    output = stderr.getvalue()
    assert "Tool Call" in output
    assert "status: start" in output
    assert "source: main-agent" in output
    assert "tool: read_file" in output
    assert "+" in output
    assert "|" in output


@pytest.mark.anyio
async def test_cli_event_renderer_truncates_tool_results_to_200_characters() -> None:
    stderr = io.StringIO()
    renderer = chainagents_cli.CliEventRenderer(
        prompt="hello",
        stdout=io.StringIO(),
        stderr=stderr,
        stream=True,
        json_output=False,
        show_reasoning=False,
        show_tools=True,
    )
    long_result = ("abcdefghijklmnopqrstuvwxyz" * 8) + "TAIL"

    assert chainagents_cli.truncate_tool_result_content(long_result) == long_result[:200]

    await renderer.handle_event(
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (_ToolMessage(long_result), {}),
                ),
            },
        }
    )

    output = stderr.getvalue()
    assert "Tool Result" in output
    assert "status: success" in output
    assert "source: main-agent" in output
    assert "tool: read_file" in output
    assert "result: " in output
    assert "TAIL" not in output
    assert "+" in output
    assert "|" in output


@pytest.mark.anyio
async def test_cli_event_renderer_deduplicates_tool_results_from_stream_modes() -> None:
    stderr = io.StringIO()
    renderer = chainagents_cli.CliEventRenderer(
        prompt="hello",
        stdout=io.StringIO(),
        stderr=stderr,
        stream=True,
        json_output=False,
        show_reasoning=False,
        show_tools=True,
    )
    tool_message = _ToolMessage("same result")

    await renderer.handle_event(
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (tool_message, {}),
                ),
            },
        }
    )
    await renderer.handle_event(
        {
            "event": "on_chain_stream",
            "data": {
                "chunk": (
                    (),
                    "updates",
                    {"tools": {"messages": [tool_message]}},
                ),
            },
        }
    )

    output = stderr.getvalue()
    assert output.count("Tool Result") == 1
    assert output.count("same result") == 1
