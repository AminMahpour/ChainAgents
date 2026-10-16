"""Per-profile context window and summarization budget."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
import pytest

import chainagents.runtime.core as deepagent_runtime
import chainagents.runtime.middleware as runtime_middleware
from chainagents.runtime.models import (
    SUMMARIZATION_KEEP_TOKENS_KEY,
    SUMMARIZATION_TRIGGER_TOKENS_KEY,
)
from chainagents.runtime.types import ExtensionsConfig

OLLAMA_MODEL = """
[model]
provider = "ollama"
base_url = "http://127.0.0.1:11434"
name = "qwen3:32b"
"""


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toml: str) -> Any:
    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(toml.strip() + "\n", encoding="utf-8")
    monkeypatch.setenv("DEEPAGENT_CONFIG", str(config_path))
    return deepagent_runtime.RuntimeConfig.from_env()


# ------------------------------------------------------------------ parsing


def test_profiles_parse_and_inherit_context_settings(tmp_path, monkeypatch) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        OLLAMA_MODEL
        + """
context_window = 32768
summarization_trigger_tokens = 24000
summarization_keep_tokens = 4000

[model.profiles.inherits]
temperature = 0.2

[model.profiles.small]
name = "qwen3:4b"
context_window = 8192
summarization_trigger_tokens = 6000
summarization_keep_tokens = 1000
""",
    )

    default = deepagent_runtime.resolve_runtime_model_profile(config)
    inherits = deepagent_runtime.resolve_runtime_model_profile(config, "inherits")
    small = deepagent_runtime.resolve_runtime_model_profile(config, "small")

    assert (default.context_window, default.summarization_trigger_tokens) == (32768, 24000)
    assert default.summarization_keep_tokens == 4000
    assert (inherits.context_window, inherits.summarization_trigger_tokens) == (32768, 24000)
    assert (small.context_window, small.summarization_trigger_tokens) == (8192, 6000)
    assert small.summarization_keep_tokens == 1000


def test_profile_with_its_own_window_does_not_inherit_the_base_budget(
    tmp_path, monkeypatch
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        OLLAMA_MODEL
        + """
context_window = 32768
summarization_trigger_tokens = 24000
summarization_keep_tokens = 4000

[model.profiles.small]
name = "qwen3:4b"
context_window = 8192
""",
    )

    small = deepagent_runtime.resolve_runtime_model_profile(config, "small")

    assert small.context_window == 8192
    assert small.summarization_trigger_tokens is None
    assert small.summarization_keep_tokens is None


def test_profiles_for_other_models_do_not_inherit_context_settings(
    tmp_path, monkeypatch
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        OLLAMA_MODEL
        + """
context_window = 32768
summarization_trigger_tokens = 24000

[model.profiles.other-local]
name = "qwen3:4b"

[model.profiles.claude]
provider = "anthropic"
name = "claude-sonnet-4-6"
api_key = "x"
""",
    )

    for name in ("other-local", "claude"):
        profile = deepagent_runtime.resolve_runtime_model_profile(config, name)
        assert profile.context_window is None, name
        assert profile.summarization_trigger_tokens is None, name


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ("context_window = 0", r"context_window.*positive integer"),
        ('context_window = "big"', r"context_window.*positive integer"),
        ("summarization_trigger_tokens = -5", r"summarization_trigger_tokens.*positive"),
        ("summarization_keep_tokens = true", r"summarization_keep_tokens.*positive"),
        (
            "context_window = 8192\nsummarization_trigger_tokens = 8192",
            r"summarization_trigger_tokens.*less than.*context_window",
        ),
        (
            "summarization_trigger_tokens = 4000\nsummarization_keep_tokens = 4000",
            r"summarization_keep_tokens.*less than.*summarization_trigger_tokens",
        ),
        (
            "context_window = 8192\nsummarization_keep_tokens = 9000",
            r"summarization_keep_tokens.*less than.*context_window",
        ),
    ],
)
def test_invalid_context_settings_fail_at_startup(
    tmp_path, monkeypatch, fields: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(
            tmp_path,
            monkeypatch,
            OLLAMA_MODEL + f"\n[model.profiles.small]\n{fields}\n",
        )


def test_global_trigger_must_fit_each_profile_window(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError, match=r"\[agent\]\.summarization_trigger_tokens.*small"):
        _config(
            tmp_path,
            monkeypatch,
            OLLAMA_MODEL
            + """
[model.profiles.small]
context_window = 8192

[agent]
summarization_trigger_tokens = 150000
""",
        )


def test_profile_budget_may_replace_an_oversized_global_trigger(tmp_path, monkeypatch) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        OLLAMA_MODEL
        + """
[model.profiles.small]
context_window = 8192
summarization_trigger_tokens = 6000

[agent]
summarization_trigger_tokens = 150000
""",
    )

    assert config.extensions.summarization_trigger_tokens == 150000


# ------------------------------------------------------------ model building


def test_ollama_context_window_sets_num_ctx_and_model_metadata(tmp_path, monkeypatch) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        OLLAMA_MODEL
        + """
context_window = 32768
summarization_trigger_tokens = 24000
summarization_keep_tokens = 4000
""",
    )

    model = deepagent_runtime.build_model(config, "medium")

    assert model.num_ctx == 32768
    assert model.profile["max_input_tokens"] == 32768
    assert model.profile[SUMMARIZATION_TRIGGER_TOKENS_KEY] == 24000
    assert model.profile[SUMMARIZATION_KEEP_TOKENS_KEY] == 4000


def test_models_without_context_settings_are_unchanged(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, monkeypatch, OLLAMA_MODEL)

    model = deepagent_runtime.build_model(config, "medium")

    assert model.num_ctx is None
    assert not (model.profile or {}).get(SUMMARIZATION_TRIGGER_TOKENS_KEY)


def test_openai_compatible_context_window_reaches_model_metadata(tmp_path, monkeypatch) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        """
[model]
provider = "openai_compatible"
base_url = "http://127.0.0.1:1234/v1"
name = "local-model"
api_key = "x"
context_window = 65536
""",
    )

    model = deepagent_runtime.build_model(config, "medium")

    assert model.profile["max_input_tokens"] == 65536


# ------------------------------------------------------------- summarization


def _model(**profile: Any) -> FakeListChatModel:
    model = FakeListChatModel(responses=["ok"])
    model.profile = profile or None
    return model


class _CapturingSummarizer(AgentMiddleware):
    created: ClassVar[list[dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return "SummarizationMiddleware"

    def __init__(self, model, *, backend, trigger=None, keep=None, **kwargs) -> None:
        type(self).created.append({"model": model, "trigger": trigger, "keep": keep})


@pytest.fixture
def summarizers(monkeypatch) -> list[dict[str, Any]]:
    _CapturingSummarizer.created = []
    monkeypatch.setattr(
        "deepagents.middleware.summarization.SummarizationMiddleware",
        _CapturingSummarizer,
    )
    return _CapturingSummarizer.created


def _factory(extensions: ExtensionsConfig, original=None):
    config = type("Config", (), {"extensions": extensions})()
    return runtime_middleware._build_deepagents_summarization_factory(
        config, original or (lambda model, backend: ("original", model))
    )


@pytest.mark.parametrize(
    ("profile", "global_tokens", "expected"),
    [
        # Profile budget wins over the global [agent] thresholds.
        (
            {SUMMARIZATION_TRIGGER_TOKENS_KEY: 6000, SUMMARIZATION_KEEP_TOKENS_KEY: 1000},
            (50000, 9000),
            (("tokens", 6000), ("tokens", 1000)),
        ),
        # Global thresholds apply when the profile sets none.
        ({"max_input_tokens": 200000}, (50000, 9000), (("tokens", 50000), ("tokens", 9000))),
        # A known window alone uses DeepAgents' fraction defaults.
        ({"max_input_tokens": 8192}, (None, None), None),
        # Partial budgets fill the rest from the next source.
        (
            {SUMMARIZATION_TRIGGER_TOKENS_KEY: 6000, "max_input_tokens": 8192},
            (None, None),
            (("tokens", 6000), ("fraction", 0.10)),
        ),
    ],
)
def test_summarization_budget_precedence(
    summarizers, profile, global_tokens, expected
) -> None:
    trigger, keep = global_tokens
    factory = _factory(
        ExtensionsConfig(
            config_path=None,
            summarization_trigger_tokens=trigger,
            summarization_keep_tokens=keep,
        )
    )

    result = factory(_model(**profile), object())

    if expected is None:
        assert result[0] == "original"
    else:
        assert (summarizers[-1]["trigger"], summarizers[-1]["keep"]) == expected


def test_unconfigured_models_keep_deepagents_default_summarization(summarizers) -> None:
    model = _model()

    result = _factory(ExtensionsConfig(config_path=None))(model, object())

    assert result == ("original", model)
    assert summarizers == []


def test_each_agent_summarizes_with_its_own_profile_budget(
    tmp_path, monkeypatch, summarizers
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        OLLAMA_MODEL
        + """
context_window = 32768
summarization_trigger_tokens = 24000
summarization_keep_tokens = 4000

[model.profiles.small]
name = "qwen3:4b"
context_window = 8192
summarization_trigger_tokens = 6000
summarization_keep_tokens = 1000

[rag]
enabled = false

[[subagents]]
name = "scout"
description = "Small helper"
system_prompt = "Help."
model = "small"
""",
    )
    runtime = deepagent_runtime.AgentRuntime(config)
    runtime._store = InMemoryStore()
    runtime._checkpointer = MemorySaver()

    asyncio.run(runtime.get_agent("medium", thread_id="thread-1"))

    budgets = {
        (item["model"].model, item["trigger"], item["keep"]) for item in summarizers
    }
    assert ("qwen3:32b", ("tokens", 24000), ("tokens", 4000)) in budgets
    assert ("qwen3:4b", ("tokens", 6000), ("tokens", 1000)) in budgets


# ------------------------------------------------- model switches and defaults


SWITCHABLE = (
    OLLAMA_MODEL
    + """
models = ["qwen3:32b", "qwen3:4b"]
context_window = 32768
summarization_trigger_tokens = 24000
summarization_keep_tokens = 4000
"""
)


def test_raw_model_choice_does_not_inherit_the_default_models_context(
    tmp_path, monkeypatch
) -> None:
    config = _config(tmp_path, monkeypatch, SWITCHABLE)

    other = deepagent_runtime.resolve_runtime_model_profile(config, "qwen3:4b")
    model = deepagent_runtime.build_model(config, "medium", model_name="qwen3:4b")

    assert (other.context_window, other.summarization_trigger_tokens) == (None, None)
    assert other.summarization_keep_tokens is None
    assert model.num_ctx is None


def test_model_override_does_not_inherit_the_file_models_context(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("DEEPAGENT_MODEL_NAME", "qwen3:4b")
    config = _config(tmp_path, monkeypatch, SWITCHABLE)

    default = deepagent_runtime.resolve_runtime_model_profile(config)

    assert default.name == "qwen3:4b"
    assert (default.context_window, default.summarization_trigger_tokens) == (None, None)
    assert deepagent_runtime.build_model(config, "medium").num_ctx is None


def test_same_model_choice_keeps_its_context(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, monkeypatch, SWITCHABLE)

    same = deepagent_runtime.resolve_runtime_model_profile(config, "qwen3:32b")

    assert (same.context_window, same.summarization_trigger_tokens) == (32768, 24000)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        # Default keep (10% = 1000) is not below an explicit trigger of 500.
        (
            "context_window = 10000\nsummarization_trigger_tokens = 500",
            r"keep \(1000, default 10% of context_window\).*less than.*trigger \(500",
        ),
        # Explicit keep of 9000 is not below the default trigger (85% = 8500).
        (
            "context_window = 10000\nsummarization_keep_tokens = 9000",
            r"keep \(9000.*less than.*trigger \(8500, default 85% of context_window\)",
        ),
    ],
)
def test_validation_uses_default_fractions_for_omitted_thresholds(
    tmp_path, monkeypatch, fields: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(tmp_path, monkeypatch, OLLAMA_MODEL + f"\n[model.profiles.small]\n{fields}\n")


def test_global_keep_can_complete_a_profile_budget(tmp_path, monkeypatch) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        OLLAMA_MODEL
        + """
[model.profiles.small]
context_window = 10000
summarization_trigger_tokens = 500

[agent]
summarization_keep_tokens = 300
""",
    )

    assert config.extensions.summarization_keep_tokens == 300
