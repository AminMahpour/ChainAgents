"""Tests for the Amazon Bedrock model provider."""

from __future__ import annotations

import io
import tomllib
from pathlib import Path
from typing import ClassVar

import pytest
from langchain_aws import ChatBedrockConverse

import chainagents.runtime.core as deepagent_runtime
import chainagents.runtime.graph as runtime_graph
from chainagents.events.stream import (
    AgentStreamEvent,
    AgentStreamEventAdapter,
    reasoning_text_from_token,
    stringify_content,
)
from chainagents.interfaces.cli import app as chainagents_cli
from chainagents.interfaces.cli.args import parse_model_provider_argument
from chainagents.runtime.model_config import (
    format_model_provider,
    normalize_bedrock_endpoint_url,
    normalize_model_provider,
)


@pytest.fixture(autouse=True)
def _isolated_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DEEPAGENT_MODEL_PROVIDER",
        "DEEPAGENT_MODEL_NAME",
        "DEEPAGENT_MODEL_BASE_URL",
        "DEEPAGENT_MODEL_ENDPOINT_URL",
        "DEEPAGENT_MODEL_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(body.strip(), encoding="utf-8")
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))


@pytest.mark.parametrize("value", ["bedrock", "Bedrock", "aws_bedrock", "aws-bedrock", "amazon_bedrock"])
def test_normalize_model_provider_accepts_bedrock_aliases(value: str) -> None:
    assert normalize_model_provider(value) == "bedrock"
    assert parse_model_provider_argument(value) == "bedrock"
    assert format_model_provider("bedrock") == "Amazon Bedrock"


def test_normalize_bedrock_endpoint_url_defaults_to_https_and_allows_empty() -> None:
    assert normalize_bedrock_endpoint_url(None) == ""
    assert normalize_bedrock_endpoint_url("  ") == ""
    assert (
        normalize_bedrock_endpoint_url("vpce-123.bedrock-runtime.us-east-1.vpce.amazonaws.com/")
        == "https://vpce-123.bedrock-runtime.us-east-1.vpce.amazonaws.com"
    )


def test_runtime_config_builds_bedrock_model_from_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
temperature = 0.3
max_tokens = 2048
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert config.model_provider == "bedrock"
    assert config.model_name == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert config.model_base_url == ""
    assert isinstance(model, ChatBedrockConverse)
    assert model.model_id == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert model.client.meta.region_name == "us-east-1"
    assert model.endpoint_url is None
    assert model.temperature == 0.3
    assert model.max_tokens == 2048
    # No configurable reasoning on this model, so nothing is added.
    assert model.additional_model_request_fields is None


def test_bedrock_model_applies_reasoning_effort_and_drops_temperature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "us.anthropic.claude-opus-5-5"
endpoint_url = "https://bedrock-runtime.vpce.example"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "high")

    assert isinstance(model, ChatBedrockConverse)
    assert model.endpoint_url == "https://bedrock-runtime.vpce.example"
    assert model.additional_model_request_fields == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    assert model.temperature is None


def test_bedrock_model_skips_reasoning_when_thinking_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "us.amazon.nova-2-lite-v1:0"
thinking = "disabled"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "high")

    assert model.reasoning_effort is None
    assert model.additional_model_request_fields is None
    # High reasoning would drop temperature; with thinking off it is kept.
    assert model.temperature == pytest.approx(0.00001)


def test_bedrock_model_does_not_require_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-model"
""",
    )
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "bedrock")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "amazon.nova-pro-v1:0")

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.model_provider == "bedrock"
    assert config.model_base_url == ""
    assert isinstance(deepagent_runtime.build_model(config, "low"), ChatBedrockConverse)


def test_bedrock_switch_requires_model_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "ollama"
name = "local-model"
""",
    )
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "bedrock")

    with pytest.raises(ValueError, match="Amazon Bedrock runtime must define"):
        deepagent_runtime.RuntimeConfig.from_env()


def test_bedrock_switch_rejects_stale_env_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "ollama"
name = "local-model"
""",
    )
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "bedrock")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "amazon.nova-pro-v1:0")
    monkeypatch.setenv("DEEPAGENT_MODEL_BASE_URL", "http://127.0.0.1:11434")

    with pytest.raises(ValueError, match="Amazon Bedrock"):
        deepagent_runtime.RuntimeConfig.from_env()


def test_bedrock_profile_builds_from_ollama_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-model"

[model.profiles.bedrock-nova]
provider = "bedrock"
name = "amazon.nova-pro-v1:0"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium", model_name="bedrock-nova")

    assert isinstance(model, ChatBedrockConverse)
    assert model.model_id == "amazon.nova-pro-v1:0"
    assert model.endpoint_url is None


def test_bedrock_profile_requires_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
""",
    )

    with pytest.raises(ValueError, match="Amazon Bedrock model config must define"):
        deepagent_runtime.RuntimeConfig.from_env()


def test_bedrock_tool_sanitization_adds_object_type_to_schema() -> None:
    class _Tool:
        name = "lookup"
        args_schema: ClassVar[dict[str, object]] = {"properties": {"q": {"type": "string"}}}

    [tool] = runtime_graph.sanitize_tools_for_model("bedrock", [_Tool()])

    assert tool.args_schema["type"] == "object"


class _BedrockToken:
    type = "AIMessageChunk"
    additional_kwargs: ClassVar[dict[str, str]] = {}
    tool_call_chunks: ClassVar[list[dict[str, str]]] = []

    def __init__(self, content: object) -> None:
        self.content = content


def test_reasoning_text_from_token_extracts_bedrock_reasoning_blocks() -> None:
    token = _BedrockToken(
        [
            {
                "type": "reasoning_content",
                "reasoning_content": {"type": "text", "text": "weighing options"},
                "index": 0,
            },
            {
                "type": "reasoning_content",
                "reasoning_content": {"signature": "sig"},
                "index": 0,
            },
        ]
    )

    assert reasoning_text_from_token(token) == "weighing options"
    assert stringify_content(token.content) == ""


def test_reasoning_text_from_token_extracts_standard_reasoning_blocks() -> None:
    token = _BedrockToken(
        [
            {"type": "reasoning", "reasoning": "weighing options"},
            {"type": "text", "text": "Final answer"},
        ]
    )

    assert reasoning_text_from_token(token) == "weighing options"
    assert stringify_content(token.content) == "Final answer"


def test_adapter_separates_bedrock_reasoning_from_response_text() -> None:
    adapter = AgentStreamEventAdapter(prompt="hello")

    events = adapter.events_from_raw_event(
        {
            "event": "on_chain_stream",
            "parent_ids": [],
            "data": {
                "chunk": (
                    (),
                    "messages",
                    (
                        _BedrockToken(
                            [
                                {
                                    "type": "reasoning_content",
                                    "reasoning_content": {"type": "text", "text": "thinking"},
                                },
                                {"type": "text", "text": "Final answer"},
                            ]
                        ),
                        {},
                    ),
                )
            },
        }
    )

    assert events == [
        AgentStreamEvent(kind="reasoning_delta", source="main-agent", text="thinking"),
        AgentStreamEvent(kind="response_delta", source="main-agent", text="Final answer"),
    ]


def test_configure_command_writes_bedrock_without_base_url(tmp_path: Path) -> None:
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        '[model]\nprovider = "ollama"\nbase_url = "http://127.0.0.1:11434"\n',
        encoding="utf-8",
    )
    answers = "\n".join(
        [
            "bedrock",
            "",
            "us.anthropic.claude-sonnet-5",
            *([""] * (len(chainagents_cli.CONFIGURE_PROMPTS) - 3)),
        ]
    )

    code = chainagents_cli.run_configure_command(
        config_path=config_path,
        stdin=io.StringIO(answers),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 0
    model = tomllib.loads(config_path.read_text(encoding="utf-8"))["model"]
    assert model["provider"] == "bedrock"
    assert model["name"] == "us.anthropic.claude-sonnet-5"
    assert "base_url" not in model


def test_configure_command_requires_model_name_for_bedrock(tmp_path: Path) -> None:
    config_path = tmp_path / "deepagent.toml"
    original = '[model]\nprovider = "ollama"\nname = "local"\n'
    config_path.write_text(original, encoding="utf-8")
    answers = "\n".join(["bedrock", *([""] * (len(chainagents_cli.CONFIGURE_PROMPTS) - 1))])
    stderr = io.StringIO()

    code = chainagents_cli.run_configure_command(
        config_path=config_path,
        stdin=io.StringIO(answers),
        stdout=io.StringIO(),
        stderr=stderr,
    )

    assert code == 1
    assert config_path.read_text(encoding="utf-8") == original
    assert "Amazon Bedrock requires an explicit model name" in stderr.getvalue()


@pytest.mark.parametrize(
    ("model_arn", "provider", "base_model"),
    [
        (
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-5-5",
            "anthropic",
            "anthropic.claude-opus-5-5",
        ),
        (
            "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0",
            "amazon",
            "amazon.nova-pro-v1:0",
        ),
    ],
)
def test_bedrock_model_arn_supplies_provider_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_arn: str,
    provider: str,
    base_model: str,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        f"""
[model]
provider = "bedrock"
name = "{model_arn}"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "high")

    assert isinstance(model, ChatBedrockConverse)
    assert model.model_id == model_arn
    assert model.provider == provider
    assert model.base_model_id == base_model


def test_bedrock_opaque_model_arn_is_rejected_with_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    with pytest.raises(ValueError, match="application inference profiles"):
        deepagent_runtime.build_model(config, "medium")


def test_bedrock_model_keeps_langchain_aws_streaming_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "meta.llama3-1-70b-instruct-v1:0"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert model.disable_streaming == "tool_calling"


def test_bedrock_model_honors_explicit_streaming_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "meta.llama3-1-70b-instruct-v1:0"
disable_streaming = true
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert model.disable_streaming is True


def test_configure_command_clears_model_choices_on_provider_switch(tmp_path: Path) -> None:
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        '[model]\nprovider = "ollama"\nname = "gpt-oss:20b"\nmodels = ["gpt-oss:20b"]\n',
        encoding="utf-8",
    )
    answers = "\n".join(
        [
            "bedrock",
            "",
            "amazon.nova-pro-v1:0",
            *([""] * (len(chainagents_cli.CONFIGURE_PROMPTS) - 3)),
        ]
    )

    code = chainagents_cli.run_configure_command(
        config_path=config_path,
        stdin=io.StringIO(answers),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )

    assert code == 0
    model = tomllib.loads(config_path.read_text(encoding="utf-8"))["model"]
    assert "models" not in model


def test_bedrock_model_rejects_out_of_range_temperature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "amazon.nova-pro-v1:0"
temperature = 1.5
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    with pytest.raises(ValueError, match="between 0 and 1"):
        deepagent_runtime.build_model(config, "medium")


@pytest.mark.parametrize(
    "value", ["anthropic_bedrock", "anthropic-bedrock", "bedrock_anthropic", "claude_bedrock"]
)
def test_normalize_model_provider_accepts_anthropic_bedrock_aliases(value: str) -> None:
    assert normalize_model_provider(value) == "anthropic_bedrock"
    assert parse_model_provider_argument(value) == "anthropic_bedrock"
    assert format_model_provider("anthropic_bedrock") == "Anthropic Claude on Amazon Bedrock"


def test_runtime_config_builds_anthropic_bedrock_model_from_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_aws import ChatAnthropicBedrock

    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "us.anthropic.claude-opus-4-8-v1"
temperature = 0.2
max_tokens = 4096
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "high")

    assert config.model_provider == "anthropic_bedrock"
    assert config.model_base_url == ""
    assert type(model) is ChatAnthropicBedrock
    assert model.model == "us.anthropic.claude-opus-4-8-v1"
    # Adaptive thinking is on, so the configured temperature is dropped.
    assert model.temperature is None
    assert model.max_tokens == 4096
    assert model.effort == "high"
    assert model.thinking == {"type": "adaptive"}
    assert type(model._client).__name__ == "AnthropicBedrock"
    assert model._client.aws_region == "us-east-1"


def test_anthropic_bedrock_honors_disabled_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "us.anthropic.claude-opus-4-8-v1"
thinking = "disabled"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert model.thinking is None


def test_anthropic_bedrock_forwards_custom_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chainagents.runtime.providers import EndpointChatAnthropicBedrock

    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "us.anthropic.claude-sonnet-4-6"
endpoint_url = "vpce-0123.bedrock-runtime.us-east-1.vpce.amazonaws.com"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert isinstance(model, EndpointChatAnthropicBedrock)
    assert str(model._client.base_url).rstrip("/") == (
        "https://vpce-0123.bedrock-runtime.us-east-1.vpce.amazonaws.com"
    )


def test_anthropic_bedrock_switch_needs_no_api_key_or_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "local-model"
""",
    )
    monkeypatch.setenv("DEEPAGENT_MODEL_PROVIDER", "anthropic_bedrock")
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "us.anthropic.claude-sonnet-4-6")

    config = deepagent_runtime.RuntimeConfig.from_env()

    assert config.model_provider == "anthropic_bedrock"
    assert config.model_base_url == ""


def test_anthropic_bedrock_profile_requires_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
""",
    )

    with pytest.raises(
        ValueError,
        match="Anthropic Claude on Amazon Bedrock model config must define",
    ):
        deepagent_runtime.RuntimeConfig.from_env()


def test_anthropic_bedrock_tool_sanitization_adds_object_type_to_schema() -> None:
    class _Tool:
        name = "lookup"
        args_schema: ClassVar[dict[str, object]] = {"properties": {"q": {"type": "string"}}}

    [tool] = runtime_graph.sanitize_tools_for_model("anthropic_bedrock", [_Tool()])

    assert tool.args_schema["type"] == "object"


def test_anthropic_bedrock_skips_effort_when_thinking_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "us.anthropic.claude-opus-4-8-v1"
thinking = "disabled"
temperature = 0.2
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "high")

    assert model.thinking is None
    assert model.reasoning_effort is None
    assert model.temperature == 0.2


def test_anthropic_bedrock_drops_temperature_when_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "us.anthropic.claude-sonnet-4-6"
temperature = 0.2
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert model.thinking == {"type": "adaptive"}
    assert model.temperature is None


def test_anthropic_bedrock_prefers_bearer_token_over_env_sigv4_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "us.anthropic.claude-sonnet-4-6"
""",
    )
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key")

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert model.aws_access_key_id is None
    assert model._client.api_key == "bedrock-api-key"


def test_anthropic_bedrock_rejects_out_of_range_temperature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "us.anthropic.claude-sonnet-4-6"
thinking = "disabled"
temperature = 1.5
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    with pytest.raises(ValueError, match="between 0 and 1"):
        deepagent_runtime.build_model(config, "medium")


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5"),
        ("anthropic.claude-3-5-sonnet-20240620-v1:0", "claude-3-5-sonnet"),
        ("anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),
        ("global.anthropic.claude-opus-5-5", "claude-opus-5-5"),
        ("in.anthropic.claude-sonnet-5", "claude-sonnet-5"),
        (
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/"
            "us.anthropic.claude-sonnet-5",
            "claude-sonnet-5",
        ),
    ],
)
def test_anthropic_bedrock_base_model_strips_bedrock_decorations(
    model_id: str,
    expected: str,
) -> None:
    from chainagents.runtime.models import anthropic_bedrock_base_model

    assert anthropic_bedrock_base_model(model_id) == expected


def test_anthropic_bedrock_skips_effort_for_models_without_effort_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
temperature = 0.2
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "high")

    assert model.reasoning_effort is None
    assert model.thinking is None
    assert model.temperature == 0.2


def test_anthropic_bedrock_sends_disabled_thinking_to_claude_5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "us.anthropic.claude-sonnet-5"
thinking = "disabled"
temperature = 0.2
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "high")

    assert model.thinking == {"type": "disabled"}
    assert model.reasoning_effort is None
    assert model.temperature == 0.2


@pytest.mark.parametrize("provider", ["bedrock", "anthropic_bedrock"])
@pytest.mark.parametrize(
    "name",
    [
        "global.anthropic.claude-opus-5-5",
        "global.anthropic.claude-sonnet-5-5",
        "us.anthropic.claude-fable-5",
    ],
)
def test_bedrock_rejects_disabled_thinking_on_always_thinking_claude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    name: str,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        f"""
[model]
provider = "{provider}"
name = "{name}"
thinking = "disabled"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    with pytest.raises(ValueError, match="cannot run with thinking"):
        deepagent_runtime.build_model(config, "medium")


def test_bedrock_converse_sends_disabled_thinking_to_claude_5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "us.anthropic.claude-sonnet-5"
thinking = "disabled"
temperature = 0.2
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "high")

    assert model.additional_model_request_fields == {"thinking": {"type": "disabled"}}
    assert model.temperature == 0.2


@pytest.mark.parametrize(
    "name",
    [
        "amazon.nova-pro-v1:0",
        "us.meta.llama3-1-70b-instruct-v1:0",
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0",
    ],
)
def test_anthropic_bedrock_rejects_non_claude_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        f"""
[model]
provider = "anthropic_bedrock"
name = "{name}"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    with pytest.raises(ValueError, match="only supports Anthropic Claude"):
        deepagent_runtime.build_model(config, "medium")


def test_anthropic_bedrock_rejects_opaque_application_inference_profile_arn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    with pytest.raises(ValueError, match="underlying Claude model"):
        deepagent_runtime.build_model(config, "medium")


def test_anthropic_bedrock_arn_uses_anthropic_profile_for_thinking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-5"
temperature = 0.2
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "high")

    assert model.reasoning_effort == "high"
    assert "xhigh" in model.profile["reasoning_effort_levels"]
    assert model.temperature is None


def test_anthropic_bedrock_defaults_max_tokens_from_model_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_anthropic.chat_models import _get_default_model_profile

    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "anthropic_bedrock"
name = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    expected = _get_default_model_profile("claude-haiku-4-5")["max_output_tokens"]
    assert expected > 4096
    assert model.max_tokens == expected


def test_bedrock_supplies_metadata_for_geo_prefixes_langchain_aws_lacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "in.anthropic.claude-opus-5"
temperature = 0.2
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "high")

    assert model.provider == "anthropic"
    assert model.base_model_id == "anthropic.claude-opus-5"
    assert "thinking" in model.additional_model_request_fields
    assert model.temperature is None


def test_bedrock_nova_replaces_zero_temperature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "us.amazon.nova-pro-v1:0"
temperature = 0
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    model = deepagent_runtime.build_model(config, "medium")

    assert model.temperature == pytest.approx(0.00001)


def test_bedrock_nova_drops_temperature_at_high_reasoning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "us.amazon.nova-2-lite-v1:0"
temperature = 0.3
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    high = deepagent_runtime.build_model(config, "high")
    medium = deepagent_runtime.build_model(config, "medium")

    assert high.additional_model_request_fields["reasoningConfig"]["maxReasoningEffort"] == "high"
    assert high.temperature is None
    assert medium.temperature == 0.3


def test_bedrock_rejects_disabled_thinking_for_gpt_oss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "bedrock"
name = "openai.gpt-oss-120b-1:0"
thinking = "disabled"
""",
    )

    config = deepagent_runtime.RuntimeConfig.from_env()
    with pytest.raises(ValueError, match="reasoning_effort"):
        deepagent_runtime.build_model(config, "medium")
