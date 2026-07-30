"""Test the ChainAgents CLI argument parsing, commands, and streaming renderer."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import chainagents_cli
from chainagents.runtime import core as runtime_core
from chainagents.runtime.token_usage import TokenUsageFileCallbackHandler
from deepagent_runtime import RuntimeConfig
from rag_runtime import RagStatus, RagUploadResult


def test_cli_parses_prompt_and_runtime_flags() -> None:
    """Verify that CLI parses prompt and runtime flags."""
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
            "--disable-streaming-for-tool-calls",
            "--json",
        ]
    )

    assert chainagents_cli.prompt_from_args(args, stdin=io.StringIO("")) == "hello"
    assert args.photo == ["image.png"]
    assert args.model == "gemma4:26b"
    assert args.reasoning == "high"
    assert args.thread_id == "thread-1"
    assert args.stream is False
    assert args.disable_streaming_for_tool_calls is True
    assert args.json_output is True


def test_cli_parses_configure_flag() -> None:
    """Verify that CLI parses the interactive configuration flag."""
    args = chainagents_cli.parse_args(["--configure", "--config", "custom.toml"])

    assert args.configure is True
    assert args.config == "custom.toml"


def test_configure_command_updates_known_toml_values(tmp_path: Path) -> None:
    """Verify the interactive config command updates known TOML values."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
# Keep this comment.
provider = "ollama"
base_url = "http://old.example:11434"
name = "old-model"
temperature = 0.1
reasoning_effort = "low"

[agent]
state = "stateful"
recursion_limit = 20

[rag]
enabled = false

[rag.embedding]
provider = "auto"

[langfuse]
enabled = false

[chainlit]
model_mode_enabled = false
reasoning_mode_enabled = false
reasoning_steps_enabled = false
tool_steps_enabled = false
startup_status_enabled = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    answers = "\n".join(
        [
            "openai_compatible",
            "http://127.0.0.1:1234/v1",
            "local-model",
            "high",
            "0.2",
            "stateless",
            "250",
            "yes",
            "openai_compatible",
            "text-embedding-3-large",
            "http://127.0.0.1:1234/v1",
            "yes",
            "yes",
            "no",
            "yes",
            "no",
            "yes",
        ]
    )
    stdout = io.StringIO()

    code = chainagents_cli.run_configure_command(
        config_path=config_path,
        stdin=io.StringIO(answers),
        stdout=stdout,
        stderr=io.StringIO(),
    )

    assert code == 0
    written = config_path.read_text(encoding="utf-8")
    assert "# Keep this comment." in written
    assert 'provider = "openai_compatible"' in written
    assert 'name = "local-model"' in written
    assert "Configuration written" in stdout.getvalue()

    import tomllib

    parsed = tomllib.loads(written)
    assert parsed["model"]["base_url"] == "http://127.0.0.1:1234/v1"
    assert parsed["model"]["temperature"] == 0.2
    assert parsed["model"]["reasoning_effort"] == "high"
    assert parsed["agent"]["state"] == "stateless"
    assert parsed["agent"]["recursion_limit"] == 250
    assert parsed["rag"]["enabled"] is True
    assert parsed["rag"]["embedding"]["provider"] == "openai_compatible"
    assert parsed["rag"]["embedding"]["model"] == "text-embedding-3-large"
    assert parsed["rag"]["embedding"]["base_url"] == "http://127.0.0.1:1234/v1"
    assert parsed["langfuse"]["enabled"] is True
    assert parsed["chainlit"]["model_mode_enabled"] is True
    assert parsed["chainlit"]["reasoning_mode_enabled"] is False
    assert parsed["chainlit"]["reasoning_steps_enabled"] is True
    assert parsed["chainlit"]["tool_steps_enabled"] is False
    assert parsed["chainlit"]["startup_status_enabled"] is True


def test_configure_command_stops_section_at_array_table(tmp_path: Path) -> None:
    """Verify array-of-table headers bound scalar section updates."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[rag.embedding]
provider = "openai_compatible"
# model = "text-embedding-3-small"
# base_url = "http://127.0.0.1:11434"

[[async_subagents]]
name = "async-researcher"
description = "Runs longer research jobs."
graph_id = "async-researcher"
url = "http://127.0.0.1:2024"

[[subagents]]
name = "repo-researcher"
description = "Researches the current repository."
system_prompt_file = "prompts/repo-researcher.md"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    answers = "\n".join(
        [
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "text-embedding-3-large",
            "http://127.0.0.1:1234/v1",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    )

    code = chainagents_cli.run_configure_command(
        config_path=config_path,
        stdin=io.StringIO(answers),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 0

    import tomllib

    written = config_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(written)
    assert parsed["rag"]["embedding"]["model"] == "text-embedding-3-large"
    assert parsed["rag"]["embedding"]["base_url"] == "http://127.0.0.1:1234/v1"
    assert "model" not in parsed["async_subagents"][0]
    assert "base_url" not in parsed["async_subagents"][0]


def test_configure_command_appends_missing_sections(tmp_path: Path) -> None:
    """Verify the interactive config command appends missing TOML sections."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text("[model]\nname = \"old-model\"\n", encoding="utf-8")
    answers = "\n".join([""] * 17)

    code = chainagents_cli.run_configure_command(
        config_path=config_path,
        stdin=io.StringIO(answers),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 0
    import tomllib

    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["model"]["provider"] == "ollama"
    assert parsed["model"]["name"] == "old-model"
    assert parsed["agent"]["state"] == "stateful"
    assert parsed["rag"]["enabled"] is False
    assert parsed["rag"]["embedding"]["provider"] == "auto"
    assert parsed["langfuse"]["enabled"] is False
    assert parsed["chainlit"]["startup_status_enabled"] is True


def test_configure_command_reprompts_invalid_choice(tmp_path: Path) -> None:
    """Verify the interactive config command validates choices."""
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text("", encoding="utf-8")
    answers = "\n".join(["bad-provider", "ollama", *([""] * 16)])
    stderr = io.StringIO()

    code = chainagents_cli.run_configure_command(
        config_path=config_path,
        stdin=io.StringIO(answers),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert code == 0
    assert "Choose one of" in stderr.getvalue()


@pytest.mark.anyio
async def test_run_cli_configure_honors_deepagent_config_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify configure writes the runtime env config path."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_path = project_root / "prod.toml"
    config_path.write_text("[model]\nname = \"prod-model\"\n", encoding="utf-8")
    cwd = tmp_path / "workdir"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(chainagents_cli, "PROJECT_ROOT", project_root, raising=False)
    monkeypatch.setenv("DEEPAGENT_CONFIG", "prod.toml")
    args = chainagents_cli.parse_args(["--configure"])

    code = await chainagents_cli.run_cli(
        args,
        runtime=object(),
        stdin=io.StringIO("\n".join([""] * len(chainagents_cli.CONFIGURE_PROMPTS))),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 0
    assert config_path.exists()
    assert not (cwd / "deepagent.toml").exists()

    import tomllib

    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["model"]["name"] == "prod-model"
    assert parsed["agent"]["state"] == "stateful"


@pytest.mark.anyio
async def test_run_cli_configure_resolves_relative_config_against_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify relative --config paths use runtime path resolution."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_path = project_root / "custom.toml"
    config_path.write_text("[model]\nname = \"custom-model\"\n", encoding="utf-8")
    cwd = project_root / "subdir"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(chainagents_cli, "PROJECT_ROOT", project_root, raising=False)
    args = chainagents_cli.parse_args(["--configure", "--config", "custom.toml"])

    code = await chainagents_cli.run_cli(
        args,
        runtime=object(),
        stdin=io.StringIO("\n".join([""] * len(chainagents_cli.CONFIGURE_PROMPTS))),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 0
    assert config_path.exists()
    assert not (cwd / "custom.toml").exists()

    import tomllib

    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["model"]["name"] == "custom-model"
    assert parsed["agent"]["state"] == "stateful"


def test_runtime_overrides_from_cli_args_take_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that runtime overrides from CLI args take precedence.

    Args:
        tmp_path: Path to the tmp.
        monkeypatch: The monkeypatch value.
    """
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
            "--disable-streaming-for-tool-calls",
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
    assert config.model_disable_streaming == "tool_calling"
    assert config.default_reasoning == "high"
    assert config.recursion_limit == 77
    assert config.rag_requested is False


@pytest.mark.anyio
async def test_async_main_shuts_down_langfuse_after_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CLI shutdown flushes Langfuse after the runtime closes.

    Args:
        monkeypatch: The monkeypatch value.
    """
    config = RuntimeConfig.from_env()
    events: list[str] = []

    class FakeRuntime:
        """Represent the runtime created by the CLI."""

        async def close(self) -> None:
            """Record runtime close calls."""
            events.append("close")

    async def fake_create(runtime_config: RuntimeConfig) -> FakeRuntime:
        assert runtime_config is config
        events.append("create")
        return FakeRuntime()

    async def fake_run_cli(*args, runtime: FakeRuntime, **kwargs) -> int:
        assert isinstance(runtime, FakeRuntime)
        events.append("run")
        return 0

    monkeypatch.setattr(
        chainagents_cli.RuntimeConfig,
        "from_env",
        classmethod(lambda cls, overrides=None: config),
    )
    monkeypatch.setattr(chainagents_cli.AgentRuntime, "create", fake_create)
    monkeypatch.setattr(chainagents_cli, "run_cli", fake_run_cli)
    monkeypatch.setattr(
        chainagents_cli,
        "shutdown_langfuse_client",
        lambda runtime_config: events.append("shutdown"),
    )

    assert await chainagents_cli.async_main(["--status"]) == 0
    assert events == ["create", "run", "close", "shutdown"]


def test_cli_endpoint_url_override_supplies_openai_default_query(
    tmp_path: Path,
) -> None:
    """Verify that CLI endpoint URL override supplies openai default query.

    Args:
        tmp_path: Path to the tmp.
    """
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[model]
provider = "ollama"
base_url = "http://config.example:11434"
name = "config-model"
""".strip(),
        encoding="utf-8",
    )
    args = chainagents_cli.parse_args(
        [
            "--config",
            str(config_path),
            "--provider",
            "openai_compatible",
            "--endpoint-url",
            "https://api.example.test/proxy/chat/completions?api-version=2026-01-01",
            "--model",
            "cli-model",
            "--status",
        ]
    )

    config = RuntimeConfig.from_env(chainagents_cli.runtime_overrides_from_args(args))

    assert config.model_provider == "openai_compatible"
    assert config.model_base_url == "https://api.example.test/proxy"
    assert config.model_endpoint_query == (("api-version", "2026-01-01"),)


class _FakeMcpRuntime:
    """Provide a test double for fake MCP runtime."""

    def __init__(self) -> None:
        """Initialize the fake MCP runtime instance."""
        self.config = SimpleNamespace(
            default_reasoning="medium",
            model_name="fake-model",
            recursion_limit=100,
        )
        self.invocation: dict[str, str | None] | None = None

    def resolve_chainlit_command(self, name: str):
        """Resolve chainlit command.

        Args:
            name: The name value.

        Returns:
            The resolved chainlit command.
        """
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
        """Invoke a configured MCP tool command with parsed arguments.

        Args:
            tool_name: Name of the tool to invoke.
            raw_args: Raw argument text supplied with the command.
            thread_id: Conversation thread identifier.
            mcp_session_id: MCP session identifier.
            server_name: The server name value.

        Returns:
            The invoke MCP tool command result.
        """
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
    """Verify that CLI command invokes configured MCP tool."""
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
    """Provide a test double for fake RAG runtime."""

    def __init__(self) -> None:
        """Initialize the fake RAG runtime instance."""
        self.rebuilt = False
        self.uploaded: list[str] = []

    async def rebuild_rag_index(self) -> RagStatus:
        """Rebuild RAG index.

        Returns:
            The rebuilt object or status.
        """
        self.rebuilt = True
        return RagStatus.ready_status(
            file_count=1,
            chunk_count=2,
            persist_directory=Path(".rag"),
        )

    async def ingest_rag_uploads(self, *, thread_id: str, uploads):
        """Ingest RAG uploads.

        Args:
            thread_id: Conversation thread identifier.
            uploads: Uploaded files supplied by the user.

        Returns:
            The ingest RAG uploads result.
        """
        self.uploaded = [upload.name for upload in uploads]
        return RagUploadResult(
            thread_id=thread_id,
            added_files=tuple(self.uploaded),
            indexed_files=1,
            chunk_count=3,
        )


class _CaptureAgent:
    """Provide a test double for capture agent."""

    def __init__(self) -> None:
        """Initialize the capture agent instance."""
        self.payload = None
        self.config = None

    def astream_events(self, payload, *, config, version, stream_mode, subgraphs):
        """Yield fake stream events for CLI renderer tests.

        Args:
            payload: The payload value.
            config: Configuration object used by the operation.
            version: The version value.
            stream_mode: The stream mode value.
            subgraphs: The subgraphs value.

        Returns:
            The astream events result.
        """
        self.payload = payload
        self.config = config

        async def events():
            """Provide events behavior.

            Yields:
                Values produced by events.
            """
            if False:
                yield None

        return events()


class _CancelledPromptAgent:
    """Simulate a CLI stream cancelled before LangGraph emits a terminal event."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.root_run_id = uuid4()

    def astream_events(self, payload, *, config, version, stream_mode, subgraphs):
        async def events():
            for callback in config["callbacks"]:
                callback.on_chain_start(
                    {},
                    payload,
                    run_id=self.root_run_id,
                )
            self.started.set()
            await asyncio.sleep(60)
            if False:
                yield None

        return events()


class _FakePromptRuntime:
    """Provide a test double for fake prompt runtime."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        agent=None,
    ) -> None:
        """Initialize the fake prompt runtime instance."""
        self.project_root = project_root or Path.cwd()
        self.config = SimpleNamespace(
            default_reasoning="medium",
            model_name="fake-model",
            recursion_limit=100,
        )
        self.agent = agent or _CaptureAgent()

    async def get_agent(self, *args, **kwargs):
        """Return the fake prompt agent used by CLI tests.

        Args:
            args: Parsed command-line arguments.
            kwargs: The kwargs value.

        Returns:
            The fake prompt agent.
        """
        return self.agent


@pytest.mark.anyio
async def test_cli_runs_rag_actions_without_prompt(tmp_path: Path) -> None:
    """Verify that CLI runs RAG actions without prompt.

    Args:
        tmp_path: Path to the tmp.
    """
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
async def test_cli_photo_attaches_image_content_to_agent_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that CLI photo attaches image content to agent payload.

    Args:
        tmp_path: Path to the tmp.
    """
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
    runtime_root = tmp_path / "runtime-root"
    fallback_root = tmp_path / "fallback-root"
    monkeypatch.setattr(runtime_core, "PROJECT_ROOT", fallback_root)
    runtime = _FakePromptRuntime(project_root=runtime_root)

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
    assert len(runtime.agent.config["callbacks"]) == 1
    assert isinstance(
        runtime.agent.config["callbacks"][0],
        TokenUsageFileCallbackHandler,
    )
    runtime.agent.config["callbacks"][0].on_chain_end({}, run_id=uuid4())
    assert (runtime_root / ".files" / "token-usage.jsonl").exists()
    assert not (fallback_root / ".files" / "token-usage.jsonl").exists()


@pytest.mark.anyio
async def test_cancelled_cli_prompt_records_token_usage(tmp_path: Path) -> None:
    """Closing a cancelled CLI stream must preserve its usage record."""
    agent = _CancelledPromptAgent()
    runtime = _FakePromptRuntime(project_root=tmp_path, agent=agent)
    args = chainagents_cli.parse_args(["--prompt", "wait"])
    task = asyncio.create_task(
        chainagents_cli.run_agent_prompt(
            runtime,  # type: ignore[arg-type]
            args,
            prompt="wait",
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
    )
    await agent.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    record = json.loads(
        (tmp_path / ".files" / "token-usage.jsonl").read_text(encoding="utf-8")
    )
    assert record["request_id"] == str(agent.root_run_id)
    assert record["status"] == "cancelled"


@pytest.mark.anyio
async def test_cli_photo_requires_prompt(tmp_path: Path) -> None:
    """Verify that CLI photo requires prompt.

    Args:
        tmp_path: Path to the tmp.
    """
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
    """Verify that CLI JSON combines multiple actions.

    Args:
        tmp_path: Path to the tmp.
    """
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
        model_disable_streaming=False,
        agent_state="stateful",
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
    """Provide an internal helper for token.

    Attributes:
        type: The type value.
        additional_kwargs: The additional kwargs value.
        tool_call_chunks: The tool call chunks value.
    """

    type = "AIMessageChunk"
    additional_kwargs: dict[str, str] = {}
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, content: str) -> None:
        """Initialize the token instance.

        Args:
            content: Message or document content to process.
        """
        self.content = content


class _AnthropicThinkingToken:
    """Provide an internal helper for Anthropic thinking token."""

    type = "AIMessageChunk"
    additional_kwargs: dict[str, str] = {}
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, thinking: str) -> None:
        """Initialize the Anthropic thinking token instance.

        Args:
            thinking: The thinking delta value.
        """
        self.content = [
            {
                "type": "thinking",
                "thinking": thinking,
                "index": 0,
            }
        ]


def test_reasoning_text_from_token_extracts_anthropic_thinking_block() -> None:
    """Verify that Anthropic thinking content blocks are treated as reasoning."""
    token = _AnthropicThinkingToken("checking Claude reasoning")

    assert chainagents_cli.reasoning_text_from_token(token) == "checking Claude reasoning"


def test_stringify_content_omits_anthropic_thinking_block() -> None:
    """Verify that Anthropic thinking content blocks do not render as answer text."""
    token = _AnthropicThinkingToken("hidden reasoning")

    assert chainagents_cli.stringify_content(token.content) == ""


class _ReasoningToken:
    """Provide an internal helper for reasoning token.

    Attributes:
        type: The type value.
        content: Message or document content to process.
        tool_call_chunks: The tool call chunks value.
    """

    type = "AIMessageChunk"
    content = ""
    tool_call_chunks: list[dict[str, str]] = []

    def __init__(self, reasoning: str = "thinking") -> None:
        """Initialize the reasoning token instance.

        Args:
            reasoning: The reasoning value.
        """
        self.additional_kwargs = {"reasoning_content": reasoning}


class _ToolCallToken:
    """Provide an internal helper for tool call token.

    Attributes:
        type: The type value.
        content: Message or document content to process.
        additional_kwargs: The additional kwargs value.
        tool_call_chunks: The tool call chunks value.
    """

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


class _ToolCallChunkToken:
    """Provide an internal helper for tool call chunk token.

    Attributes:
        type: The type value.
        content: Message or document content to process.
        additional_kwargs: The additional kwargs value.
    """

    type = "AIMessageChunk"
    content = ""
    additional_kwargs: dict[str, str] = {}

    def __init__(self, chunk: dict[str, str]) -> None:
        """Initialize the tool call chunk token instance.

        Args:
            chunk: Streamed event chunk to normalize.
        """
        self.tool_call_chunks = [chunk]


class _ToolMessage:
    """Provide an internal helper for tool message.

    Attributes:
        type: The type value.
        name: The name value.
        status: The status value.
        tool_call_id: Tool call identifier.
    """

    type = "tool"
    name = "read_file"
    status = "success"
    tool_call_id = "call-1"

    def __init__(self, content: str) -> None:
        """Initialize the tool message instance.

        Args:
            content: Message or document content to process.
        """
        self.content = content


@pytest.mark.anyio
async def test_cli_event_renderer_streams_final_response() -> None:
    """Verify that CLI event renderer streams final response."""
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
    """Verify that CLI event renderer formats reasoning trace."""
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
    """Verify that CLI event renderer appends reasoning chunks inline."""
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
async def test_cli_event_renderer_uses_block_panel_for_tool_call_start() -> None:
    """Verify that CLI event renderer uses block panel for tool call start."""
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
    assert "args: " in output
    assert '"path": "README.md"' in output
    assert "+" not in output
    assert "|" not in output
    assert "━" in output
    assert "┃" in output


@pytest.mark.anyio
async def test_cli_event_renderer_shows_summarization_status() -> None:
    """Verify that CLI event renderer shows summarization status."""
    stderr = io.StringIO()
    renderer = chainagents_cli.CliEventRenderer(
        prompt="hello",
        stdout=io.StringIO(),
        stderr=stderr,
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
                    "custom",
                    {
                        "kind": "summarization_status",
                        "status": "started",
                        "source": "main-agent",
                        "message": "Conversation summarization triggered.",
                    },
                ),
            },
        }
    )

    output = stderr.getvalue()
    assert "Summarization" in output
    assert "status: started" in output
    assert "source: main-agent" in output
    assert "Conversation summarization triggered." in output


@pytest.mark.anyio
async def test_cli_event_renderer_accumulates_tool_args_across_chunks() -> None:
    """Verify that CLI event renderer accumulates tool args across chunks."""
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
            "data": {"chunk": ((), "messages", (_ToolCallChunkToken({"id": "call-9", "name": "read_file", "args": '{"path":"REA'}), {}))},
        }
    )
    await renderer.handle_event(
        {
            "event": "on_chain_stream",
            "data": {"chunk": ((), "messages", (_ToolCallChunkToken({"id": "call-9", "args": 'DME.md"}'}), {}))},
        }
    )

    output = stderr.getvalue()
    assert "Tool Call" in output
    assert "status: start" in output
    assert "status: update" in output
    assert '"path": "README.md"' in output


@pytest.mark.anyio
async def test_cli_event_renderer_truncates_tool_results_to_200_characters() -> None:
    """Verify that CLI event renderer truncates tool results to 200 characters."""
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
    assert "+" not in output
    assert "|" not in output
    assert "━" in output
    assert "┃" in output


@pytest.mark.anyio
async def test_cli_event_renderer_deduplicates_tool_results_from_stream_modes() -> None:
    """Verify that CLI event renderer deduplicates tool results from stream modes."""
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
