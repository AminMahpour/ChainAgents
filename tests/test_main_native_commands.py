from __future__ import annotations

import asyncio

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


def test_handle_native_command_skill_target_builds_skill_prompt() -> None:
    class DummyRuntime:
        def resolve_chainlit_command(self, name: str):
            assert name == "review-skill"
            return type(
                "Command",
                (),
                {
                    "name": "review-skill",
                    "description": "Run reviewer skill",
                    "target": "skill",
                    "value": "reviewer",
                    "template": "Please review: {input}",
                },
            )()

    settings = type("Settings", (), {"thread_id": "thread-1"})()
    parsed = main.ParsedNativeCommand(
        command_name="review-skill",
        raw_args="main.py",
    )

    transformed = asyncio.run(
        main.handle_native_command(
            runtime=DummyRuntime(),  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            parsed=parsed,
        )
    )

    assert transformed is not None
    assert "Use the configured `reviewer` skill" in transformed
    assert "Please review: main.py" in transformed
