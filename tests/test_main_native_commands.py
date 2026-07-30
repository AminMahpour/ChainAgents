"""Test native Chainlit command and mode handling in the main app."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import agent_commands
import main
import pytest
from chainagents.runtime import core as runtime_core
from deepagent_runtime import (
    AppSettings,
    ChainlitStarterConfig,
    ExtensionsConfig,
    ModelDefaults,
    RuntimeConfig,
    TokenUsageFileCallbackHandler,
)
from chainagents.runtime.reflection import ReflectionProposal


def test_load_chainlit_auth_users_parses_json_map() -> None:
    """Verify that Chainlit auth can load multiple users from JSON."""
    users = main.load_chainlit_auth_users(
        raw_users='{"admin":"change-me","alice":"alice-password"}',
    )

    assert users == {"admin": "change-me", "alice": "alice-password"}


def test_load_chainlit_auth_users_uses_legacy_credentials() -> None:
    """Verify that legacy single-user Chainlit auth settings still work."""
    users = main.load_chainlit_auth_users(
        raw_users="",
        legacy_username="admin",
        legacy_password="change-me",
    )

    assert users == {"admin": "change-me"}


def test_load_chainlit_auth_users_prefers_json_over_legacy_credentials() -> None:
    """Verify that CHAINLIT_AUTH_USERS takes precedence over legacy settings."""
    users = main.load_chainlit_auth_users(
        raw_users='{"alice":"alice-password"}',
        legacy_username="admin",
        legacy_password="change-me",
    )

    assert users == {"alice": "alice-password"}


@pytest.mark.parametrize(
    ("raw_users", "message"),
    (
        ("not-json", "must contain valid JSON"),
        ('["admin"]', "must be a JSON object"),
        ("{}", "must define at least one user"),
        ('{"":"password"}', "usernames must be non-empty strings"),
        ('{"admin":123}', "passwords must be non-empty strings"),
        ('{"admin":""}', "passwords must be non-empty strings"),
    ),
)
def test_load_chainlit_auth_users_rejects_invalid_json_map(
    raw_users: str,
    message: str,
) -> None:
    """Verify that invalid Chainlit auth user config fails clearly."""
    with pytest.raises(ValueError, match=message):
        main.load_chainlit_auth_users(raw_users=raw_users)


def test_settings_payload_includes_stream_visibility_toggles() -> None:
    """Verify stream visibility toggles are persisted with chat settings."""
    settings = AppSettings(
        model_name="gpt-oss:20b",
        reasoning_level="medium",
        thread_id="thread-1",
        show_reasoning_stream=True,
        show_tool_calls=False,
    )

    assert main.settings_payload(settings) == {
        "model_name": "gpt-oss:20b",
        "reasoning_level": "medium",
        "thread_id": "thread-1",
        "show_reasoning_stream": True,
        "show_tool_calls": False,
    }


def test_coerce_settings_reads_stream_visibility_toggles() -> None:
    """Verify raw Chainlit settings include stream visibility toggles."""
    settings = main.coerce_settings(
        {
            "model_name": "gpt-oss:20b",
            "reasoning_level": "high",
            "thread_id": "thread-1",
            "show_reasoning_stream": False,
            "show_tool_calls": False,
        },
        default_model_name="gpt-oss:20b",
        available_models=("gpt-oss:20b",),
    )

    assert settings.show_reasoning_stream is False
    assert settings.show_tool_calls is False


def test_build_chat_settings_includes_stream_visibility_switches() -> None:
    """Verify Chainlit settings expose reasoning and tool-call stream toggles."""
    chat_settings = main.build_chat_settings(
        AppSettings(
            model_name="gpt-oss:20b",
            reasoning_level="medium",
            thread_id="thread-1",
            show_reasoning_stream=False,
            show_tool_calls=True,
        ),
        available_models=("gpt-oss:20b",),
    )

    inputs_by_id = {
        input_widget.id: input_widget for input_widget in chat_settings.inputs
    }

    assert inputs_by_id["show_reasoning_stream"].type == "switch"
    assert inputs_by_id["show_reasoning_stream"].initial is False
    assert inputs_by_id["show_tool_calls"].type == "switch"
    assert inputs_by_id["show_tool_calls"].initial is True


def test_authenticate_chainlit_user_accepts_configured_users() -> None:
    """Verify that Chainlit auth accepts any configured user."""
    users = {"admin": "change-me", "alice": "alice-password"}

    admin = main.authenticate_chainlit_user("admin", "change-me", users)
    alice = main.authenticate_chainlit_user("alice", "alice-password", users)

    assert admin is not None
    assert admin.identifier == "admin"
    assert admin.display_name == "admin"
    assert admin.metadata == {"provider": "credentials"}
    assert alice is not None
    assert alice.identifier == "alice"
    assert alice.display_name == "alice"
    assert alice.metadata == {"provider": "credentials"}


def test_authenticate_chainlit_user_accepts_non_ascii_password() -> None:
    """Verify that Chainlit auth supports non-ASCII JSON passwords."""
    users = {"alice": "påsswörd"}

    user = main.authenticate_chainlit_user("alice", "påsswörd", users)

    assert user is not None
    assert user.identifier == "alice"


def test_authenticate_chainlit_user_rejects_invalid_credentials() -> None:
    """Verify that Chainlit auth rejects unknown users and wrong passwords."""
    users = {"admin": "change-me", "alice": "alice-password"}

    assert main.authenticate_chainlit_user("admin", "wrong", users) is None
    assert main.authenticate_chainlit_user("unknown", "change-me", users) is None


def test_resolve_native_command_prefers_explicit_slash_text() -> None:
    """Verify that resolve native command prefers explicit slash text."""
    parsed = agent_commands.resolve_native_command(
        raw_text="/summarize hello world",
        selected_command="ask-researcher",
    )

    assert parsed == agent_commands.ParsedNativeCommand(
        command_name="summarize",
        raw_args="hello world",
    )


def test_build_chainlit_starters_maps_config_to_chainlit_starters() -> None:
    """Verify that starter config maps to Chainlit Starter objects."""
    starters = main.build_chainlit_starters(
        (
            ChainlitStarterConfig(
                label="Explain repo",
                message="Explain this repository.",
                icon="book-open",
            ),
            ChainlitStarterConfig(
                label="Review diff",
                message="Review the current changes.",
                command="review",
            ),
        )
    )

    assert len(starters) == 2
    assert starters[0].label == "Explain repo"
    assert starters[0].message == "Explain this repository."
    assert starters[0].icon == "book-open"
    assert starters[0].command is None
    assert starters[1].label == "Review diff"
    assert starters[1].message == "Review the current changes."
    assert starters[1].command == "review"
    assert starters[1].icon is None


@pytest.mark.anyio
async def test_configured_chainlit_starters_reads_current_runtime(monkeypatch) -> None:
    """Verify that the Chainlit starter callback reads current runtime config."""
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            extensions=SimpleNamespace(
                chainlit_starters=(
                    ChainlitStarterConfig(
                        label="Explain repo",
                        message="Explain this repository.",
                    ),
                )
            )
        )
    )

    async def fail_get_runtime():
        raise AssertionError("starter callback should not initialize the runtime")

    monkeypatch.setattr(main.AgentRuntime, "current", lambda: runtime)
    monkeypatch.setattr(main.AgentRuntime, "get", fail_get_runtime)

    starters = await main.configured_chainlit_starters()

    assert len(starters) == 1
    assert starters[0].label == "Explain repo"
    assert starters[0].message == "Explain this repository."


def test_resolve_native_command_uses_selected_command_input() -> None:
    """Verify that resolve native command uses selected command input."""
    parsed = agent_commands.resolve_native_command(
        raw_text="hello world",
        selected_command="summarize",
    )

    assert parsed == agent_commands.ParsedNativeCommand(
        command_name="summarize",
        raw_args="hello world",
    )


def test_resolve_native_command_returns_none_without_command() -> None:
    """Verify that resolve native command returns none without command."""
    parsed = agent_commands.resolve_native_command(
        raw_text="hello world",
        selected_command=None,
    )

    assert parsed is None


def test_resolve_reasoning_level_for_message_defaults_to_settings() -> None:
    """Verify that resolve reasoning level for message defaults to settings."""
    message = SimpleNamespace(content="hello")
    settings = SimpleNamespace(reasoning_level="medium")

    resolved = main.resolve_reasoning_level_for_message(message, settings)

    assert resolved == "medium"


def test_resolve_reasoning_level_for_message_uses_mode_override() -> None:
    """Verify that resolve reasoning level for message uses mode override."""
    message = SimpleNamespace(content="hello", modes={"reasoning_level": "high"})
    settings = SimpleNamespace(reasoning_level="medium")

    resolved = main.resolve_reasoning_level_for_message(message, settings)

    assert resolved == "high"


def test_message_has_reasoning_level_override_tracks_explicit_mode() -> None:
    """Verify Chainlit reasoning modes preserve explicit per-message choices."""
    message = SimpleNamespace(content="hello", modes={"reasoning_level": "medium"})

    assert main.message_has_reasoning_level_override(message)


def test_default_reasoning_level_for_model_uses_profile_default() -> None:
    """Verify Chainlit defaults settings to the selected profile reasoning."""
    config = RuntimeConfig(
        database_url=None,
        model_provider="ollama",
        model_name="reviewer",
        model_choices=("reviewer",),
        model_base_url="http://127.0.0.1:11434",
        model_api_key=None,
        model_temperature=0.0,
        default_reasoning="low",
        persistence_mode="memory",
        extensions=ExtensionsConfig(config_path=None),
        model_profiles={
            "reviewer": ModelDefaults(
                provider="ollama",
                base_url="http://127.0.0.1:11434",
                name="review-model",
                reasoning_effort="high",
                explicit_fields=frozenset({"name", "reasoning_effort"}),
            )
        },
    )

    assert main.default_reasoning_level_for_model(config, "reviewer") == "high"


def test_settings_reasoning_level_is_explicit_for_profile_mismatch() -> None:
    """Verify Chainlit settings can override a profile reasoning default."""
    config = RuntimeConfig(
        database_url=None,
        model_provider="ollama",
        model_name="reviewer",
        model_choices=("reviewer",),
        model_base_url="http://127.0.0.1:11434",
        model_api_key=None,
        model_temperature=0.0,
        default_reasoning="low",
        persistence_mode="memory",
        extensions=ExtensionsConfig(config_path=None),
        model_profiles={
            "reviewer": ModelDefaults(
                provider="ollama",
                base_url="http://127.0.0.1:11434",
                name="review-model",
                reasoning_effort="high",
                explicit_fields=frozenset({"name", "reasoning_effort"}),
            )
        },
    )
    settings = AppSettings(
        model_name="reviewer",
        reasoning_level="low",
        thread_id="thread-1",
    )

    assert main.settings_reasoning_level_is_explicit(
        config,
        settings,
        "reviewer",
    )


def test_resolve_reasoning_level_for_message_falls_back_to_settings_default() -> None:
    """Verify that resolve reasoning level for message falls back to settings default."""
    message = SimpleNamespace(content="hello", modes={})
    settings = SimpleNamespace(reasoning_level="low")

    resolved = main.resolve_reasoning_level_for_message(message, settings)

    assert resolved == "low"


def test_resolve_reasoning_level_for_message_ignores_override_when_disabled() -> None:
    """Verify that resolve reasoning level for message ignores override when disabled."""
    message = SimpleNamespace(content="hello", modes={"reasoning_level": "high"})
    settings = SimpleNamespace(reasoning_level="low")

    resolved = main.resolve_reasoning_level_for_message(
        message,
        settings,
        reasoning_mode_enabled=False,
    )

    assert resolved == "low"


def test_resolve_model_name_for_message_uses_mode_override() -> None:
    """Verify that resolve model name for message uses mode override."""
    message = SimpleNamespace(content="hello", modes={"model_name": "gemma4:27b"})
    settings = SimpleNamespace(model_name="gpt-oss:20b")

    resolved = main.resolve_model_name_for_message(
        message,
        settings,
        available_models=("gpt-oss:20b", "gemma4:27b"),
    )

    assert resolved == "gemma4:27b"


def test_resolve_model_name_for_message_falls_back_to_settings() -> None:
    """Verify that resolve model name for message falls back to settings."""
    message = SimpleNamespace(content="hello", modes={"model_name": "unknown"})
    settings = SimpleNamespace(model_name="gpt-oss:20b")

    resolved = main.resolve_model_name_for_message(
        message,
        settings,
        available_models=("gpt-oss:20b", "gemma4:27b"),
    )

    assert resolved == "gpt-oss:20b"


def test_resolve_model_name_for_message_ignores_override_when_disabled() -> None:
    """Verify that resolve model name for message ignores override when disabled."""
    message = SimpleNamespace(content="hello", modes={"model_name": "gemma4:27b"})
    settings = SimpleNamespace(model_name="gpt-oss:20b")

    resolved = main.resolve_model_name_for_message(
        message,
        settings,
        available_models=("gpt-oss:20b", "gemma4:27b"),
        model_mode_enabled=False,
    )

    assert resolved == "gpt-oss:20b"


def test_build_langgraph_config_includes_token_logging_under_runtime_root(
    tmp_path: Path,
) -> None:
    """Verify Chainlit's config helper cannot bypass token usage logging."""
    settings = SimpleNamespace(thread_id="thread-1")

    config = main.build_langgraph_config(
        settings,
        runtime_config=SimpleNamespace(recursion_limit=100),
        project_root=tmp_path,
    )

    assert len(config["callbacks"]) == 1
    assert isinstance(config["callbacks"][0], TokenUsageFileCallbackHandler)
    assert {key: value for key, value in config.items() if key != "callbacks"} == {
        "configurable": {"thread_id": "thread-1"},
        "recursion_limit": 100,
    }
    config["callbacks"][0].on_chain_end({}, run_id=uuid4())
    assert (tmp_path / ".files" / "token-usage.jsonl").exists()


def test_message_uploaded_rag_files_skips_image_uploads(tmp_path) -> None:
    """Verify that RAG uploads ignore Chainlit image attachments."""
    notes = tmp_path / "notes.md"
    notes.write_text("# Notes\n", encoding="utf-8")
    photo = tmp_path / "receipt.jpg"
    photo.write_bytes(b"fake jpeg bytes")
    message = SimpleNamespace(
        elements=[
            SimpleNamespace(path=str(notes), name="notes.md", mime="text/markdown"),
            SimpleNamespace(path=str(photo), name="receipt.jpg", mime="image/jpeg"),
        ]
    )

    uploads = main.message_uploaded_rag_files(message)

    assert [upload.name for upload in uploads] == ["notes.md"]


@pytest.mark.anyio
async def test_cancelled_chainlit_message_records_token_usage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Returning from Chainlit cancellation must not drop the usage record."""

    class _Agent:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.root_run_id = uuid4()

        def astream_events(
            self,
            payload,
            *,
            config,
            version,
            stream_mode,
            subgraphs,
        ):
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

    class _Bridge:
        def __init__(self, **_kwargs) -> None:
            pass

        async def start(self) -> None:
            pass

        async def handle_event(self, _event) -> None:
            pass

    agent = _Agent()
    extensions = SimpleNamespace(
        chainlit_reasoning_steps_enabled=True,
        chainlit_tool_steps_enabled=True,
        chainlit_reasoning_mode_enabled=True,
        chainlit_model_mode_enabled=True,
        chainlit_chronological_ui_enabled=False,
        chainlit_generative_ui_enabled=False,
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            model_name="fake-model",
            model_choices=("fake-model",),
            default_reasoning="medium",
            recursion_limit=100,
            extensions=extensions,
        ),
        project_root=tmp_path,
    )

    async def get_agent(*_args, **_kwargs):
        return agent

    async def get_runtime():
        runtime.get_agent = get_agent
        return runtime

    async def get_task_list(**_kwargs):
        return None

    settings = AppSettings(
        model_name="fake-model",
        reasoning_level="medium",
        thread_id="thread-1",
    )
    monkeypatch.setattr(main, "get_runtime_or_notify", get_runtime)
    monkeypatch.setattr(main, "coerce_settings", lambda *_args, **_kwargs: settings)
    monkeypatch.setattr(main, "get_run_task_list", get_task_list)
    monkeypatch.setattr(main, "ChainlitEventBridge", _Bridge)
    monkeypatch.setattr(main, "current_mcp_session_id", lambda: None)
    monkeypatch.setattr(main, "get_generated_ui_elements", lambda: {})
    monkeypatch.setattr(main, "get_async_task_notifier", lambda **_kwargs: None)
    monkeypatch.setattr(
        main,
        "settings_reasoning_level_is_explicit",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(main.cl.user_session, "get", lambda _key: settings)
    message = SimpleNamespace(
        content="wait",
        command=None,
        elements=[],
        modes={},
    )
    task = asyncio.create_task(main.on_message(message))
    await agent.started.wait()

    task.cancel()
    await task

    record = json.loads(
        (tmp_path / ".files" / "token-usage.jsonl").read_text(encoding="utf-8")
    )
    assert record["request_id"] == str(agent.root_run_id)
    assert record["status"] == "cancelled"


def test_message_uploaded_image_parts_builds_data_url_parts(tmp_path) -> None:
    """Verify that Chainlit image attachments become multimodal data URL parts."""
    photo = tmp_path / "receipt.jpg"
    photo.write_bytes(b"fake jpeg bytes")
    message = SimpleNamespace(
        elements=[
            SimpleNamespace(path=str(photo), name="receipt.jpg", mime="image/jpeg"),
        ]
    )

    image_parts = main.message_uploaded_image_parts(message)

    assert image_parts == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,ZmFrZSBqcGVnIGJ5dGVz"},
        }
    ]


def test_message_uploaded_image_parts_skips_unsupported_provider_formats(tmp_path) -> None:
    """Verify that unsupported image formats are not sent to vision providers."""
    heic = tmp_path / "receipt.heic"
    heic.write_bytes(b"fake heic bytes")
    tiff = tmp_path / "scan.tiff"
    tiff.write_bytes(b"fake tiff bytes")
    message = SimpleNamespace(
        elements=[
            SimpleNamespace(path=str(heic), name="receipt.heic", mime="image/heic"),
            SimpleNamespace(path=str(tiff), name="scan.tiff", mime="image/tiff"),
        ]
    )

    assert main.message_uploaded_image_parts(message) == []
    assert main.message_uploaded_image_names(message) == ()
    assert main.unsupported_uploaded_image_names(message) == (
        "receipt.heic",
        "scan.tiff",
    )


def test_message_uploaded_image_parts_uses_safe_mime_for_octet_stream_image(
    tmp_path,
) -> None:
    """Verify that safe image extensions override generic upload MIME types."""
    photo = tmp_path / "receipt.png"
    photo.write_bytes(b"fake png bytes")
    message = SimpleNamespace(
        elements=[
            SimpleNamespace(
                path=str(photo),
                name="receipt.png",
                mime="application/octet-stream",
            ),
        ]
    )

    image_parts = main.message_uploaded_image_parts(message)

    assert image_parts == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,ZmFrZSBwbmcgYnl0ZXM="},
        }
    ]


def test_chainlit_user_message_content_includes_uploaded_images() -> None:
    """Verify that agent payload content includes text and uploaded image parts."""
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
    }

    content = main.chainlit_user_message_content(
        "OCR this receipt.",
        image_parts=[image_part],
    )

    assert content == [{"type": "text", "text": "OCR this receipt."}, image_part]


def test_chainlit_prompt_text_defaults_to_ocr_for_image_only_upload() -> None:
    """Verify that image-only messages ask the agent to extract visible text."""
    prompt = main.chainlit_prompt_text(
        "",
        image_names=("receipt.jpg",),
        prompt_note="",
    )

    assert "Extract any visible text" in prompt
    assert "receipt.jpg" in prompt


def test_chainlit_prompt_text_preserves_text_without_images() -> None:
    """Verify that non-image prompts keep their existing text shape."""
    prompt = main.chainlit_prompt_text(
        "  keep my spacing  ",
        image_names=(),
        prompt_note="\n\nRAG note",
    )

    assert prompt == "  keep my spacing  \n\nRAG note"


@pytest.mark.anyio
async def test_publish_modes_ignores_missing_modes_column_error(monkeypatch) -> None:
    """Verify that publish modes ignores missing modes column error.

    Args:
        monkeypatch: The monkeypatch value.
    """
    class _Emitter:
        """Provide an internal helper for emitter."""

        async def set_modes(self, _modes):
            """Record requested Chainlit mode definitions on the test emitter.

            Args:
                _modes: The modes value.

            Raises:
                RuntimeError: If the runtime is not in a usable state.
            """
            raise RuntimeError('column "modes" does not exist')

    monkeypatch.setattr(main.cl, "context", SimpleNamespace(emitter=_Emitter()))

    await main.publish_modes(
        SimpleNamespace(model_name="gpt-oss:20b", reasoning_level="medium"),
        available_models=("gpt-oss:20b",),
    )


@pytest.mark.anyio
async def test_publish_modes_raises_for_unrelated_errors(monkeypatch) -> None:
    """Verify that publish modes raises for unrelated errors.

    Args:
        monkeypatch: The monkeypatch value.
    """
    class _Emitter:
        """Provide an internal helper for emitter."""

        async def set_modes(self, _modes):
            """Record requested Chainlit mode definitions on the test emitter.

            Args:
                _modes: The modes value.

            Raises:
                RuntimeError: If the runtime is not in a usable state.
            """
            raise RuntimeError("boom")

    monkeypatch.setattr(main.cl, "context", SimpleNamespace(emitter=_Emitter()))

    with pytest.raises(RuntimeError, match="boom"):
        await main.publish_modes(
            SimpleNamespace(model_name="gpt-oss:20b", reasoning_level="medium"),
            available_models=("gpt-oss:20b",),
        )


class _DummyRuntime:
    """Provide a test double for dummy runtime."""

    def __init__(self, command=None) -> None:
        """Initialize the dummy runtime instance.

        Args:
            command: Configured command to render or execute.
        """
        self.invocation: dict[str, str | None] | None = None
        self.command = command or SimpleNamespace(
            name="repo-readme",
            description="Read repository README",
            target="mcp_tool",
            value="repo_readme",
            template='{"path":"README.md"}',
            mcp_server="repo",
        )

    def resolve_chainlit_command(self, command_name: str):
        """Resolve chainlit command.

        Args:
            command_name: Name of the native command.

        Returns:
            The resolved chainlit command.
        """
        if command_name == self.command.name:
            return self.command
        return None

    async def invoke_mcp_tool_command(
        self,
        *,
        tool_name: str,
        raw_args: str,
        thread_id: str,
        mcp_session_id: str | None = None,
        server_name: str | None = None,
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


class _DummyMessage:
    """Provide a test double for dummy message."""

    def __init__(self, **kwargs):
        """Initialize the dummy message instance.

        Args:
            kwargs: The kwargs value.
        """
        self.kwargs = kwargs

    async def send(self):
        """Record that the dummy Chainlit message was sent.

        Returns:
            The sent message or element.
        """
        return None


@pytest.mark.anyio
async def test_ask_to_save_reflection_lesson_uses_ask_action_message(
    monkeypatch,
) -> None:
    """Verify Chainlit reflection confirmation uses AskActionMessage."""
    captured: dict[str, object] = {}
    saved: dict[str, object] = {}

    class _AskActionMessage:
        """Capture AskActionMessage construction and return Save."""

        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def send(self):
            return {"payload": {"value": "save"}}

    async def fake_save_reflection_lesson(**kwargs):
        saved.update(kwargs)

    monkeypatch.setattr(main.cl, "AskActionMessage", _AskActionMessage)
    monkeypatch.setattr(main, "save_reflection_lesson", fake_save_reflection_lesson)

    proposal = ReflectionProposal(
        reason="correction",
        memory_file="/memories/AGENTS.md",
        lesson="- Correction: remember the generated output directory.",
        trigger="That was wrong.",
    )
    settings = AppSettings(
        model_name="fake-model",
        reasoning_level="medium",
        thread_id="thread-1",
    )

    await main.ask_to_save_reflection_lesson(
        runtime=SimpleNamespace(),
        settings=settings,
        proposal=proposal,
        reasoning_level="high",
        model_name="other-model",
        async_url_override="http://async.example",
        mcp_session_id="mcp-session",
    )

    assert "Correction reflection" in str(captured["content"])
    actions = captured["actions"]
    assert [action.payload["value"] for action in actions] == ["save", "dismiss"]
    assert saved["proposal"] == proposal
    assert saved["settings"] == settings
    assert saved["reasoning_level"] == "high"
    assert saved["model_name"] == "other-model"
    assert saved["async_url_override"] == "http://async.example"
    assert saved["mcp_session_id"] == "mcp-session"


@pytest.mark.anyio
async def test_ask_to_save_reflection_lesson_dismiss_does_not_save(
    monkeypatch,
) -> None:
    """Verify Chainlit reflection dismissal does not write memory."""
    saved = False

    class _AskActionMessage:
        """Return Dismiss from AskActionMessage."""

        def __init__(self, **kwargs):
            pass

        async def send(self):
            return {"payload": {"value": "dismiss"}}

    async def fake_save_reflection_lesson(**kwargs):
        nonlocal saved
        saved = True

    monkeypatch.setattr(main.cl, "AskActionMessage", _AskActionMessage)
    monkeypatch.setattr(main, "save_reflection_lesson", fake_save_reflection_lesson)

    await main.ask_to_save_reflection_lesson(
        runtime=SimpleNamespace(),
        settings=AppSettings(
            model_name="fake-model",
            reasoning_level="medium",
            thread_id="thread-1",
        ),
        proposal=ReflectionProposal(
            reason="correction",
            memory_file="/memories/AGENTS.md",
            lesson="- Correction: remember the generated output directory.",
            trigger="That was wrong.",
        ),
        reasoning_level="medium",
        model_name="fake-model",
        async_url_override=None,
        mcp_session_id=None,
    )

    assert saved is False


@pytest.mark.anyio
async def test_save_reflection_lesson_invokes_agent_in_reflection_thread(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify confirmed reflections are saved by a hidden agent run."""
    captured: dict[str, object] = {}
    runtime_root = tmp_path / "runtime-root"
    fallback_root = tmp_path / "fallback-root"
    monkeypatch.setattr(runtime_core, "PROJECT_ROOT", fallback_root)

    class _Agent:
        async def ainvoke(self, payload, *, config):
            captured["payload"] = payload
            captured["config"] = config
            return {"messages": []}

    class _Runtime:
        config = SimpleNamespace(recursion_limit=100)
        project_root = runtime_root

        async def get_agent(self, *args, **kwargs):
            captured["agent_args"] = args
            captured["agent_kwargs"] = kwargs
            return _Agent()

    messages: list[_DummyMessage] = []

    class _Message(_DummyMessage):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            messages.append(self)

    monkeypatch.setattr(main.cl, "Message", _Message)

    proposal = ReflectionProposal(
        reason="correction",
        memory_file="/memories/AGENTS.md",
        lesson="- Correction: remember the generated output directory.",
        trigger="That was wrong.",
    )

    await main.save_reflection_lesson(
        runtime=_Runtime(),
        settings=AppSettings(
            model_name="fake-model",
            reasoning_level="medium",
            thread_id="thread-1",
        ),
        proposal=proposal,
        reasoning_level="high",
        model_name="other-model",
        async_url_override=None,
        mcp_session_id="mcp-session",
    )

    assert captured["agent_args"] == ("high",)
    assert captured["agent_kwargs"] == {
        "model_name": "other-model",
        "thread_id": "thread-1:reflection",
        "async_subagent_url_override": None,
        "mcp_session_id": "mcp-session",
    }
    config = captured["config"]
    assert len(config["callbacks"]) == 1
    assert isinstance(config["callbacks"][0], TokenUsageFileCallbackHandler)
    assert {key: value for key, value in config.items() if key != "callbacks"} == {
        "configurable": {"thread_id": "thread-1:reflection"},
        "recursion_limit": 100,
    }
    config["callbacks"][0].on_chain_end({}, run_id=uuid4())
    assert (runtime_root / ".files" / "token-usage.jsonl").exists()
    assert not (fallback_root / ".files" / "token-usage.jsonl").exists()
    prompt = captured["payload"]["messages"][0]["content"]
    assert "Target memory file: /memories/AGENTS.md" in prompt
    assert "Lessons learned from corrections" in prompt
    assert messages[-1].kwargs["content"] == "Saved lesson to `/memories/AGENTS.md`."


@pytest.mark.anyio
async def test_handle_native_command_applies_template_for_mcp_tool(monkeypatch) -> None:
    """Verify that handle native command applies template for MCP tool.

    Args:
        monkeypatch: The monkeypatch value.
    """
    runtime = _DummyRuntime()
    settings = SimpleNamespace(thread_id="thread-1")
    monkeypatch.setattr(main.cl, "Message", _DummyMessage)

    result = await main.handle_native_command(
        runtime=runtime,
        settings=settings,
        parsed=agent_commands.ParsedNativeCommand(command_name="repo-readme", raw_args=""),
    )

    assert result == ""
    assert runtime.invocation is not None
    assert runtime.invocation["raw_args"] == '{"path":"README.md"}'
    assert runtime.invocation["mcp_session_id"] is None


def test_build_skill_command_prompt_requires_skill_and_request() -> None:
    """Verify that build skill command prompt requires skill and request."""
    prompt = agent_commands.build_skill_command_prompt(
        skill_name="reviewer",
        skill_path="/workspace/skills/reviewer/SKILL.md",
        raw_args="inspect this diff",
    )

    assert "Use the configured `reviewer` skill" in prompt
    assert "Read `/workspace/skills/reviewer/SKILL.md` before taking any other action" in prompt
    assert "Skill usage is mandatory" in prompt
    assert "User request:\ninspect this diff" in prompt


def test_build_skill_command_prompt_without_request_asks_for_task() -> None:
    """Verify that build skill command prompt without request asks for task."""
    prompt = agent_commands.build_skill_command_prompt(
        skill_name="reviewer",
        skill_path="/workspace/skills/reviewer/SKILL.md",
        raw_args="",
    )

    assert "briefly explain what it does and ask the user for the specific task" in prompt


@pytest.mark.anyio
async def test_handle_native_command_returns_forced_skill_prompt() -> None:
    """Verify that handle native command returns forced skill prompt."""
    runtime = _DummyRuntime(
        command=SimpleNamespace(
            name="reviewer",
            description="Review code for bugs",
            target="skill",
            value="/workspace/skills/reviewer/SKILL.md",
            template=None,
            mcp_server=None,
        )
    )
    settings = SimpleNamespace(thread_id="thread-1")

    result = await main.handle_native_command(
        runtime=runtime,
        settings=settings,
        parsed=agent_commands.ParsedNativeCommand(
            command_name="reviewer",
            raw_args="inspect this diff",
        ),
    )

    assert result is not None
    assert "Use the configured `reviewer` skill" in result
    assert "User request:\ninspect this diff" in result


@pytest.mark.anyio
async def test_handle_native_command_without_skill_args_requests_clarification() -> None:
    """Verify that handle native command without skill args requests clarification."""
    runtime = _DummyRuntime(
        command=SimpleNamespace(
            name="reviewer",
            description="Review code for bugs",
            target="skill",
            value="/workspace/skills/reviewer/SKILL.md",
            template=None,
            mcp_server=None,
        )
    )
    settings = SimpleNamespace(thread_id="thread-1")

    result = await main.handle_native_command(
        runtime=runtime,
        settings=settings,
        parsed=agent_commands.ParsedNativeCommand(
            command_name="reviewer",
            raw_args="",
        ),
    )

    assert result is not None
    assert "briefly explain what it does and ask the user for the specific task" in result


@pytest.mark.anyio
async def test_handle_native_command_uses_selected_skill_command_input() -> None:
    """Verify that handle native command uses selected skill command input."""
    runtime = _DummyRuntime(
        command=SimpleNamespace(
            name="reviewer",
            description="Review code for bugs",
            target="skill",
            value="/workspace/skills/reviewer/SKILL.md",
            template=None,
            mcp_server=None,
        )
    )
    settings = SimpleNamespace(thread_id="thread-1")
    parsed = agent_commands.resolve_native_command(
        raw_text="inspect this diff",
        selected_command="reviewer",
    )

    assert parsed == agent_commands.ParsedNativeCommand(
        command_name="reviewer",
        raw_args="inspect this diff",
    )

    result = await main.handle_native_command(
        runtime=runtime,
        settings=settings,
        parsed=parsed,
    )

    assert result is not None
    assert "User request:\ninspect this diff" in result
