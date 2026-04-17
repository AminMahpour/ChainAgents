from __future__ import annotations

import main


def test_resolve_native_command_prefers_explicit_slash_text() -> None:
    parsed = main.resolve_native_command(
        raw_text="/summarize hello world",
        selected_command="ask-researcher",
    )

    assert parsed == main.ParsedNativeCommand(
        command_name="summarize",
        raw_args="hello world",
    )


def test_resolve_native_command_uses_selected_command_input() -> None:
    parsed = main.resolve_native_command(
        raw_text="hello world",
        selected_command="summarize",
    )

    assert parsed == main.ParsedNativeCommand(
        command_name="summarize",
        raw_args="hello world",
    )


def test_resolve_native_command_returns_none_without_command() -> None:
    parsed = main.resolve_native_command(
        raw_text="hello world",
        selected_command=None,
    )

    assert parsed is None
