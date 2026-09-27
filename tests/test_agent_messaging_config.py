"""Configuration contracts for local agent messaging and live user input."""

from __future__ import annotations

from pathlib import Path

import pytest

from chainagents.runtime.extension_config import (
    normalize_messaging_config,
    normalize_user_input_config,
    parse_sync_subagent_config,
)


def test_messaging_and_input_are_disabled_by_default() -> None:
    assert normalize_messaging_config(None).enabled is False
    assert normalize_user_input_config(None).enabled is False


def test_messaging_and_input_limits_are_configurable() -> None:
    messaging = normalize_messaging_config(
        {
            "enabled": True,
            "max_message_chars": 120,
            "max_pending_per_recipient": 3,
            "max_messages_per_session": 5,
            "max_deliveries_per_step": 2,
        }
    )
    user_input = normalize_user_input_config(
        {"enabled": True, "max_queued_turns": 4, "max_completed_turns": 6}
    )
    assert messaging.max_message_chars == 120
    assert messaging.max_deliveries_per_step == 2
    assert user_input.max_queued_turns == 4
    assert user_input.max_completed_turns == 6


@pytest.mark.parametrize(
    "field", ["enabled", "max_message_chars", "max_pending_per_recipient"]
)
def test_invalid_messaging_values_are_rejected(field: str) -> None:
    with pytest.raises(ValueError, match=f"agent.messaging.{field}"):
        normalize_messaging_config({field: 1 if field == "enabled" else False})


def test_subagent_messaging_opt_in_is_independent_of_background_and_parent() -> None:
    raw = {
        "name": "parent",
        "description": "Parent",
        "system_prompt": "Parent prompt",
        "subagents": [
            {
                "name": "child",
                "description": "Child",
                "system_prompt": "Child prompt",
                "messaging": True,
            }
        ],
    }
    parsed = parse_sync_subagent_config(
        raw, index=1, base_dir=Path("."), mcp_servers={}
    )
    assert parsed.messaging is False
    assert parsed.subagents[0].messaging is True
    assert parsed.subagents[0].background is False


def test_invalid_subagent_messaging_is_rejected() -> None:
    with pytest.raises(ValueError, match="subagent 'child' messaging"):
        parse_sync_subagent_config(
            {
                "name": "child",
                "description": "Child",
                "system_prompt": "Prompt",
                "messaging": 1,
            },
            index=1,
            base_dir=Path("."),
            mcp_servers={},
        )
