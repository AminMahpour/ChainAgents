"""Shared summarization threshold resolution."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from chainagents.runtime.models import (
    SUMMARIZATION_KEEP_TOKENS_KEY,
    SUMMARIZATION_TRIGGER_TOKENS_KEY,
)
from chainagents.runtime.summarization_budget import (
    configured_thresholds,
    context_window,
    effective_threshold_tokens,
)
from chainagents.runtime.types import ExtensionsConfig


def _model(profile: dict[str, Any] | None) -> FakeListChatModel:
    model = FakeListChatModel(responses=["ok"])
    model.profile = profile
    return model


def test_profile_tokens_win_over_global() -> None:
    model = _model(
        {SUMMARIZATION_TRIGGER_TOKENS_KEY: 6000, SUMMARIZATION_KEEP_TOKENS_KEY: 1000}
    )
    extensions = ExtensionsConfig(
        config_path=None,
        summarization_trigger_tokens=50000,
        summarization_keep_tokens=9000,
    )
    assert effective_threshold_tokens(model, extensions) == (6000, 1000)


def test_global_tokens_apply_without_profile_budget() -> None:
    model = _model({"max_input_tokens": 200000})
    extensions = ExtensionsConfig(
        config_path=None,
        summarization_trigger_tokens=50000,
        summarization_keep_tokens=9000,
    )
    assert effective_threshold_tokens(model, extensions) == (50000, 9000)


def test_window_defaults_use_fractions() -> None:
    model = _model({"max_input_tokens": 16384})
    assert effective_threshold_tokens(model, ExtensionsConfig(config_path=None)) == (
        13926,
        1638,
    )


def test_unknown_window_uses_fallback_trigger() -> None:
    assert effective_threshold_tokens(
        _model(None), ExtensionsConfig(config_path=None)
    ) == (170000, None)


def test_configured_thresholds_ignore_defaults() -> None:
    model = _model({"max_input_tokens": 16384})
    assert configured_thresholds(model, ExtensionsConfig(config_path=None)) == (
        None,
        None,
    )


def test_context_window_rejects_non_int_and_bool() -> None:
    assert context_window(_model({"max_input_tokens": 16384})) == 16384
    assert context_window(_model({"max_input_tokens": True})) is None
    assert context_window(_model({"max_input_tokens": "16384"})) is None
    assert context_window(_model(None)) is None


def test_trigger_is_clamped_to_the_reserved_input_budget() -> None:
    """DeepAgents summarizes once input exceeds 95% of the window minus output."""
    from types import SimpleNamespace

    model = SimpleNamespace(profile={"max_input_tokens": 16384}, max_tokens=4096)

    trigger, keep = effective_threshold_tokens(model, ExtensionsConfig(config_path=None))

    assert trigger == 16384 * 95 // 100 - 4096
    assert keep == 1638
