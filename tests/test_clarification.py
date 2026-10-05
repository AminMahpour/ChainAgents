"""Tests for main-agent clarifying questions that pause and resume a run."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt

from chainagents.runtime.background_tasks.context import _CURRENT_BACKGROUND_TASK_ID
from chainagents.runtime.clarification import (
    CLARIFICATION_SYSTEM_PROMPT,
    ClarificationMiddleware,
    ask_user,
    normalize_clarification_options,
    pending_clarifications,
)
from chainagents.runtime.extension_config import normalize_clarification_config
from chainagents.runtime.reflection import ReflectionConfig
from chainagents.runtime.types import ClarificationConfig
from chainagents.turns.renderer import BaseTurnRenderer
from chainagents.turns.runner import TurnRequest, TurnRunner


# --- config -------------------------------------------------------------------


def test_clarification_config_defaults_to_disabled() -> None:
    assert normalize_clarification_config(None) == ClarificationConfig(enabled=False)


def test_clarification_config_accepts_enabled() -> None:
    assert normalize_clarification_config({"enabled": True}).enabled is True


@pytest.mark.parametrize("value", [{"enabled": "yes"}, {"enabled": True, "timeout": 5}, []])
def test_clarification_config_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(ValueError):
        normalize_clarification_config(value)


# --- tool -----------------------------------------------------------------------


def test_options_are_trimmed_deduplicated_and_capped() -> None:
    options = [" Python ", "Python", "", "Go", "Rust", "C", "Java", "Kotlin", "Swift"]

    assert normalize_clarification_options(options) == [
        "Python",
        "Go",
        "Rust",
        "C",
        "Java",
        "Kotlin",
    ]


def test_ask_user_rejects_an_empty_question() -> None:
    assert "non-empty question" in ask_user.invoke({"question": "   "})


def test_ask_user_does_not_pause_background_tasks() -> None:
    token = _CURRENT_BACKGROUND_TASK_ID.set("task-1")
    try:
        result = ask_user.invoke({"question": "Which repo?"})
    finally:
        _CURRENT_BACKGROUND_TASK_ID.reset(token)

    assert "unavailable in background tasks" in result


def test_pending_clarifications_ignores_other_interrupts() -> None:
    interrupts = (
        Interrupt(value={"kind": "clarification", "question": "Which?", "options": ["a"]}, id="i1"),
        Interrupt(value={"action_requests": []}, id="i2"),
    )

    pending = pending_clarifications(interrupts)

    assert [(p.interrupt_id, p.question, p.options) for p in pending] == [
        ("i1", "Which?", ("a",))
    ]


# --- registration -------------------------------------------------------------


def _agent_kwargs(tmp_path: Path, *, enabled: bool, agent_state: str = "stateful"):
    from dataclasses import replace

    from chainagents.runtime import graph as runtime_graph
    from chainagents.runtime.artifacts import LargeToolResultArtifactRegistry
    from chainagents.runtime.config import RuntimeConfig
    from chainagents.runtime.types import ModelDefaults, SubagentConfig

    config = RuntimeConfig.from_env()
    config = replace(
        config,
        agent_state=agent_state,
        extensions=replace(
            config.extensions,
            clarification=ClarificationConfig(enabled=enabled),
            subagents=(
                SubagentConfig(
                    name="researcher",
                    description="Research things.",
                    system_prompt="Research.",
                ),
            ),
            agent_mcp_servers=(),
            mcp_servers=None,
        ),
    )
    return runtime_graph.build_agent_kwargs(
        config,
        tools=[],
        model_profile=ModelDefaults(provider="ollama", name="fake"),
        reasoning_level="medium",
        reasoning_level_is_explicit=False,
        system_prompt="Base prompt.",
        custom_instruction=None,
        rag_enabled=False,
        project_root=tmp_path,
        artifact_registry=LargeToolResultArtifactRegistry(),
        include_async_subagents=False,
        build_model=lambda *_args: FakeMessagesListChatModel(responses=[]),
    )


def _tool_names(items: Any) -> set[str]:
    return {getattr(item, "name", None) for item in items or ()}


def test_main_agent_gets_ask_user_but_subagents_do_not(tmp_path: Path) -> None:
    kwargs = _agent_kwargs(tmp_path, enabled=True)

    assert any(isinstance(item, ClarificationMiddleware) for item in kwargs["middleware"])
    assert CLARIFICATION_SYSTEM_PROMPT in kwargs["system_prompt"]
    assert "ask_user" not in _tool_names(kwargs["tools"])
    for spec in kwargs["subagents"]:
        assert "ask_user" not in _tool_names(spec.get("tools"))
        assert not any(
            isinstance(item, ClarificationMiddleware) for item in spec.get("middleware", ())
        )


@pytest.mark.parametrize(("enabled", "agent_state"), [(False, "stateful"), (True, "stateless")])
def test_ask_user_absent_when_disabled_or_stateless(
    tmp_path: Path, enabled: bool, agent_state: str
) -> None:
    kwargs = _agent_kwargs(tmp_path, enabled=enabled, agent_state=agent_state)

    assert not any(isinstance(item, ClarificationMiddleware) for item in kwargs["middleware"])
    assert CLARIFICATION_SYSTEM_PROMPT not in kwargs["system_prompt"]


# --- TurnRunner end to end ----------------------------------------------------


class _ToolAwareFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class _Runtime:
    def __init__(self, agent: Any, project_root: Path, *, enabled: bool = True) -> None:
        self.agent = agent
        self.project_root = project_root
        self.config = SimpleNamespace(
            recursion_limit=50,
            agent_state="stateful",
            extensions=SimpleNamespace(
                agent_reflection=ReflectionConfig(enabled=True),
                clarification=ClarificationConfig(enabled=enabled),
            ),
        )

    async def get_agent(self, *args, **kwargs):
        return self.agent

    def resolve_chainlit_command(self, name: str):
        return None


class _Renderer(BaseTurnRenderer):
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.reflections: list[Any] = []
        self.completed: list[Any] = []

    async def on_event(self, event):
        self.events.append(event)

    async def on_reflection(self, proposal):
        self.reflections.append(proposal)

    async def on_complete(self, result):
        self.completed.append(result)


def _clarifying_agent(delegations: list[str]):
    def task(description: str) -> str:
        """Delegate work to a subagent."""
        delegations.append(description)
        return "subagent report"

    model = _ToolAwareFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "ask_user",
                        "args": {"question": "Which module?", "options": ["api", "cli"]},
                        "id": "call-ask",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "task", "args": {"description": "review cli"}, "id": "call-task"}
                ],
            ),
            AIMessage(content="Reviewed the cli module."),
        ]
    )
    return create_agent(
        model,
        tools=[task],
        middleware=[ClarificationMiddleware()],
        checkpointer=MemorySaver(),
    )


def _request(prompt: str) -> TurnRequest:
    return TurnRequest(
        prompt=prompt,
        thread_id="thread-1",
        model_name="fake",
        reasoning_level="medium",
    )


def test_turn_pauses_for_a_clarification_and_resumes_with_the_answer(tmp_path: Path) -> None:
    delegations: list[str] = []
    agent = _clarifying_agent(delegations)
    runtime = _Runtime(agent, tmp_path)

    async def exercise():
        first_renderer = _Renderer()
        first = await TurnRunner(runtime).run(_request("Review the code"), first_renderer)
        delegated_before_answer = list(delegations)
        second_renderer = _Renderer()
        second = await TurnRunner(runtime).run(_request("cli"), second_renderer)
        state = await agent.aget_state({"configurable": {"thread_id": "thread-1"}})
        return first, first_renderer, delegated_before_answer, second, state

    first, first_renderer, delegated_before_answer, second, state = asyncio.run(exercise())

    assert first.status == "awaiting_input" and first.ok
    assert delegated_before_answer == []
    asked = [e for e in first_renderer.events if e.kind == "clarification_requested"]
    assert len(asked) == 1
    assert asked[0].text == "Which module?"
    assert asked[0].ui_props["options"] == ["api", "cli"]
    assert first.clarifications[0].question == "Which module?"
    assert first_renderer.reflections == []

    assert second.status == "completed"
    assert second.clarifications == []
    assert delegations == ["review cli"]
    assert "Reviewed the cli module." in second.response
    answers = [
        m.content
        for m in state.values["messages"]
        if isinstance(m, ToolMessage) and m.tool_call_id == "call-ask"
    ]
    assert answers == ["cli"]
    # The answer resumed the paused call; it was not added as a new user message.
    assert [m.content for m in state.values["messages"] if m.type == "human"] == [
        "Review the code"
    ]


def test_native_command_does_not_consume_a_pending_question(tmp_path: Path) -> None:
    delegations: list[str] = []
    agent = _clarifying_agent(delegations)
    runtime = _Runtime(agent, tmp_path)

    async def exercise():
        await TurnRunner(runtime).run(_request("Review the code"), _Renderer())
        await TurnRunner(runtime).run(_request("/unknown-command"), _Renderer())
        return await agent.aget_state({"configurable": {"thread_id": "thread-1"}})

    state = asyncio.run(exercise())

    assert pending_clarifications(state.interrupts)
    assert delegations == []


def test_runner_ignores_interrupts_when_clarification_is_disabled(tmp_path: Path) -> None:
    delegations: list[str] = []
    runtime = _Runtime(_clarifying_agent(delegations), tmp_path, enabled=False)

    result = asyncio.run(TurnRunner(runtime).run(_request("Review the code"), _Renderer()))

    assert result.status == "completed"
    assert result.clarifications == []


# --- interfaces ---------------------------------------------------------------


def _pending(options: tuple[str, ...] = ("api", "cli")):
    from chainagents.runtime.clarification import PendingClarification

    return PendingClarification(interrupt_id="i1", question="Which module?", options=options)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("2", "cli"), (" 1 ", "api"), ("3", "3"), ("0", "0"), ("the cli one", "the cli one")],
)
def test_bare_option_number_selects_that_option(text: str, expected: str) -> None:
    from chainagents.turns.runner import clarification_answer

    assert clarification_answer(text, _pending()) == expected


def _question_event():
    from chainagents.events.stream import AgentStreamEvent

    return AgentStreamEvent(
        kind="clarification_requested",
        source="main-agent",
        text="Which module?",
        ui_props={"options": ["api", "cli"], "interrupt_id": "i1"},
    )


def test_cli_renderer_prints_question_and_numbered_options() -> None:
    import io

    from chainagents.interfaces.cli.render import CliEventRenderer
    from chainagents.turns.runner import TurnResult

    stdout, stderr = io.StringIO(), io.StringIO()
    renderer = CliEventRenderer(
        stdout=stdout,
        stderr=stderr,
        stream=True,
        json_output=False,
        show_reasoning=False,
        show_tools=False,
    )

    async def exercise() -> None:
        await renderer.on_event(_question_event())
        await renderer.on_complete(TurnResult(status="awaiting_input", prompt="x"))

    asyncio.run(exercise())

    output = stdout.getvalue()
    assert "Which module?" in output
    assert "1. api" in output and "2. cli" in output


def test_cli_json_renderer_skips_the_question_panel() -> None:
    import io

    from chainagents.interfaces.cli.render import CliEventRenderer

    stdout = io.StringIO()
    renderer = CliEventRenderer(
        stdout=stdout,
        stderr=io.StringIO(),
        stream=False,
        json_output=True,
        show_reasoning=False,
        show_tools=False,
    )

    asyncio.run(renderer.on_event(_question_event()))

    assert stdout.getvalue() == ""


def test_api_payloads_report_a_paused_turn_only_when_paused() -> None:
    from chainagents.interfaces.api.app import _clarifications_payload, _done_payload
    from chainagents.turns.runner import TurnResult

    context = SimpleNamespace(thread_id="t", model_name="m", reasoning_level="medium")
    result = TurnResult(status="awaiting_input", prompt="x", clarifications=[_pending()])

    assert "status" not in _done_payload(context)  # type: ignore[arg-type]
    assert _done_payload(context, status="awaiting_input")["status"] == "awaiting_input"  # type: ignore[arg-type]
    assert _clarifications_payload(result) == [
        {"interrupt_id": "i1", "question": "Which module?", "options": ["api", "cli"]}
    ]


def test_chainlit_question_message_offers_one_action_per_option(monkeypatch) -> None:
    from chainagents.interfaces.chainlit import renderer as chainlit_renderer
    from chainagents.interfaces.chainlit.renderer import (
        CLARIFICATION_ANSWER_ACTION,
        clarification_message,
    )

    monkeypatch.setattr(chainlit_renderer.cl, "Message", SimpleNamespace)
    message = clarification_message(_question_event())

    assert message.content.startswith("Which module?")
    assert [(a.name, a.label, a.payload) for a in message.actions] == [
        (CLARIFICATION_ANSWER_ACTION, "api", {"answer": "api"}),
        (CLARIFICATION_ANSWER_ACTION, "cli", {"answer": "cli"}),
    ]
