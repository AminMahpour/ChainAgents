"""Resolve the summarization thresholds that apply to a chat model."""

from __future__ import annotations

from typing import Any

import chainagents.runtime.models as runtime_models
from chainagents.runtime.types import ExtensionsConfig

TRIGGER_FRACTION_PERCENT = 85
KEEP_FRACTION_PERCENT = 10
FALLBACK_TRIGGER_TOKENS = 170000


def _profile(model: Any) -> dict[str, Any]:
    """Return the model profile as a dict, treating anything else as empty."""
    profile = getattr(model, "profile", None)
    return profile if isinstance(profile, dict) else {}


def context_window(model: Any) -> int | None:
    """Return the model's ``max_input_tokens`` when it is a real int, else None."""
    value = _profile(model).get("max_input_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def configured_thresholds(
    model: Any, extensions: ExtensionsConfig
) -> tuple[int | None, int | None]:
    """Return explicitly configured (trigger, keep) tokens.

    For each value the model profile wins over the global extensions setting.
    """
    profile = _profile(model)
    trigger = profile.get(runtime_models.SUMMARIZATION_TRIGGER_TOKENS_KEY)
    keep = profile.get(runtime_models.SUMMARIZATION_KEEP_TOKENS_KEY)
    if trigger is None:
        trigger = extensions.summarization_trigger_tokens
    if keep is None:
        keep = extensions.summarization_keep_tokens
    return trigger, keep


def effective_threshold_tokens(
    model: Any, extensions: ExtensionsConfig
) -> tuple[int | None, int | None]:
    """Return the (trigger, keep) tokens summarization will actually use.

    Configured values win; otherwise defaults are floor fractions of the context
    window, or a fixed fallback trigger (and no keep) when the window is unknown.
    """
    trigger, keep = configured_thresholds(model, extensions)
    window = context_window(model)
    if window is not None:
        if trigger is None:
            trigger = window * TRIGGER_FRACTION_PERCENT // 100
        if keep is None:
            keep = window * KEEP_FRACTION_PERCENT // 100
    elif trigger is None:
        trigger = FALLBACK_TRIGGER_TOKENS
    return trigger, keep
