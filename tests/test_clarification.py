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
        (CLARIFICATION_ANSWER_ACTION, "api", {"answer": "api", "interrupt_id": "i1"}),
        (CLARIFICATION_ANSWER_ACTION, "cli", {"answer": "cli", "interrupt_id": "i1"}),
    ]


# --- review fixes -------------------------------------------------------------


def test_queued_turn_waits_while_a_question_is_pending() -> None:
    from chainagents.runtime.messaging import MessageBroker
    from chainagents.runtime.types import MessagingConfig, UserInputConfig
    from chainagents.turns.controller import ConversationInputController

    async def exercise() -> list[str]:
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        gate = asyncio.Event()
        started: list[str] = []

        async def run(text: str) -> None:
            started.append(text)
            if text == "request":
                await gate.wait()
                controller.clarification_requested("s")
            elif text == "answer":
                controller.clarification_answered("s")

        controller.submit("s", "request", run, followup_factory=lambda t: t)
        await asyncio.sleep(0)
        controller.submit("s", "queued", run, followup_factory=lambda t: t)
        gate.set()
        for _ in range(5):
            await asyncio.sleep(0)
        held = list(started)
        assert not controller.busy("s", include_paused_queue=False)
        controller.submit("s", "answer", run, followup_factory=lambda t: t)
        await controller.wait_idle("s")
        return [*held, "|", *started[len(held):]]

    assert asyncio.run(exercise()) == ["request", "|", "answer", "queued"]


def test_refused_answer_frees_the_answer_slot() -> None:
    from chainagents.runtime.messaging import MessageBroker
    from chainagents.runtime.types import MessagingConfig, UserInputConfig
    from chainagents.turns.controller import ConversationInputController

    async def exercise() -> list[str]:
        controller = ConversationInputController(
            UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
        )
        gate = asyncio.Event()
        started: list[str] = []

        async def run(text: str) -> None:
            started.append(text)
            if text == "request":
                await gate.wait()
                controller.clarification_requested("s")
            elif text == "answer":
                controller.clarification_answered("s")

        controller.submit("s", "request", run, followup_factory=lambda t: t)
        await asyncio.sleep(0)
        controller.submit("s", "queued", run, followup_factory=lambda t: t)
        gate.set()
        for _ in range(5):
            await asyncio.sleep(0)
        assert started == ["request"]
        controller.submit("s", "/refused-command", run, followup_factory=lambda t: t)
        await controller.wait_idle("s")
        controller.submit("s", "answer", run, followup_factory=lambda t: t)
        await controller.wait_idle("s")
        return started

    assert asyncio.run(exercise()) == ["request", "/refused-command", "answer", "queued"]


def test_prompt_command_does_not_answer_a_pending_question(tmp_path: Path) -> None:
    delegations: list[str] = []
    agent = _clarifying_agent(delegations)
    runtime = _Runtime(agent, tmp_path)
    runtime.resolve_chainlit_command = lambda name: (  # type: ignore[method-assign]
        SimpleNamespace(
            name="review",
            description="Review",
            target="prompt",
            value="Review it",
            template="Review {input}",
            mcp_server=None,
        )
        if name == "review"
        else None
    )

    async def exercise():
        await TurnRunner(runtime).run(_request("Review the code"), _Renderer())
        result = await TurnRunner(runtime).run(_request("/review the diff"), _Renderer())
        state = await agent.aget_state({"configurable": {"thread_id": "thread-1"}})
        return result, state

    result, state = asyncio.run(exercise())

    assert result.status == "command_error"
    assert result.command_error is not None
    assert "Answer it before running `/review`" in result.command_error.message
    assert pending_clarifications(state.interrupts)
    assert delegations == []


def test_reply_with_attachments_is_refused_instead_of_dropped(tmp_path: Path) -> None:
    delegations: list[str] = []
    agent = _clarifying_agent(delegations)
    runtime = _Runtime(agent, tmp_path)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}

    async def exercise():
        await TurnRunner(runtime).run(_request("Review the code"), _Renderer())
        request = TurnRequest(
            prompt="cli",
            thread_id="thread-1",
            model_name="fake",
            reasoning_level="medium",
            content_parts=(image,),
        )
        result = await TurnRunner(runtime).run(request, _Renderer())
        state = await agent.aget_state({"configurable": {"thread_id": "thread-1"}})
        return result, state

    result, state = asyncio.run(exercise())

    assert result.status == "command_error"
    assert result.command_error is not None
    assert "text only" in result.command_error.message
    assert pending_clarifications(state.interrupts)


def test_turns_are_serialized_per_thread_when_clarification_is_enabled(
    tmp_path: Path,
) -> None:
    runtime = _Runtime(_clarifying_agent([]), tmp_path)
    locks: list[str] = []

    def turn_lock(key: str) -> asyncio.Lock:
        locks.append(key)
        return asyncio.Lock()

    runtime.turn_lock = turn_lock  # type: ignore[attr-defined]
    runtime.config.extensions.mcp_stateful = False

    asyncio.run(TurnRunner(runtime).run(_request("Review the code"), _Renderer()))

    assert locks == ["thread-1"]


def test_tool_calls_alongside_ask_user_are_dropped(tmp_path: Path) -> None:
    delegations: list[str] = []

    def task(description: str) -> str:
        """Delegate work to a subagent."""
        delegations.append(description)
        return "report"

    model = _ToolAwareFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "task", "args": {"description": "guess"}, "id": "call-task"},
                    {"name": "ask_user", "args": {"question": "Which?"}, "id": "call-ask"},
                ],
            ),
        ]
    )
    agent = create_agent(
        model, tools=[task], middleware=[ClarificationMiddleware()], checkpointer=MemorySaver()
    )
    config = {"configurable": {"thread_id": "t"}}

    async def exercise():
        await agent.ainvoke({"messages": [{"role": "user", "content": "do it"}]}, config)
        return await agent.aget_state(config)

    state = asyncio.run(exercise())

    assert delegations == []
    assert [p.question for p in pending_clarifications(state.interrupts)] == ["Which?"]
    last_ai = [m for m in state.values["messages"] if isinstance(m, AIMessage)][-1]
    assert [call["name"] for call in last_ai.tool_calls] == ["ask_user"]


def test_keep_only_tool_call_strips_provider_tool_blocks() -> None:
    from chainagents.runtime.clarification import _keep_only_tool_call

    ask = {"name": "ask_user", "args": {"question": "Which?"}, "id": "a", "type": "tool_call"}
    message = AIMessage(
        id="m1",
        content=[
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "t", "name": "task", "input": {}},
            {"type": "tool_use", "id": "a", "name": "ask_user", "input": {}},
        ],
        tool_calls=[
            {"name": "task", "args": {}, "id": "t", "type": "tool_call"},
            ask,
        ],
        additional_kwargs={"tool_calls": [{"id": "t"}, {"id": "a"}]},
    )

    kept = _keep_only_tool_call(message, ask)  # type: ignore[arg-type]

    assert kept.id == "m1"
    assert [call["id"] for call in kept.tool_calls] == ["a"]
    assert [block.get("id") for block in kept.content if isinstance(block, dict)] == [None, "a"]
    assert kept.additional_kwargs["tool_calls"] == [{"id": "a"}]


def test_api_stream_carries_clarification_options() -> None:
    from chainagents.interfaces.api.app import _done_payload, _event_payload

    context = SimpleNamespace(thread_id="t", model_name="m", reasoning_level="medium")
    event_payload = _event_payload(_question_event(), context)  # type: ignore[arg-type]
    done = _done_payload(
        context,  # type: ignore[arg-type]
        status="awaiting_input",
        clarifications=[{"interrupt_id": "i1", "question": "Which module?", "options": ["api"]}],
    )

    assert event_payload["options"] == ["api", "cli"]
    assert event_payload["interrupt_id"] == "i1"
    assert done["clarifications"][0]["interrupt_id"] == "i1"


def test_cli_prints_buffered_preamble_before_the_question() -> None:
    import io

    from chainagents.events.stream import AgentStreamEvent
    from chainagents.interfaces.cli.render import CliEventRenderer
    from chainagents.turns.runner import TurnResult

    stdout = io.StringIO()
    renderer = CliEventRenderer(
        stdout=stdout,
        stderr=io.StringIO(),
        stream=False,
        json_output=False,
        show_reasoning=False,
        show_tools=False,
    )

    async def exercise() -> None:
        await renderer.on_event(
            AgentStreamEvent(kind="response_delta", source="main-agent", text="Let me check.")
        )
        await renderer.on_event(_question_event())
        await renderer.on_complete(TurnResult(status="awaiting_input", prompt="x"))

    asyncio.run(exercise())

    output = stdout.getvalue()
    assert output.count("Let me check.") == 1
    assert output.index("Let me check.") < output.index("Which module?")


def test_chainlit_option_buttons_expire_once_answered(monkeypatch) -> None:
    from chainagents.interfaces.chainlit import renderer as chainlit_renderer

    store: dict[str, Any] = {}
    monkeypatch.setattr(
        chainlit_renderer.cl,
        "user_session",
        SimpleNamespace(get=store.get, set=store.__setitem__),
    )
    removed: list[str] = []

    class _Action:
        def __init__(self, label: str) -> None:
            self.label = label

        async def remove(self) -> None:
            removed.append(self.label)

    store[chainlit_renderer.SESSION_PENDING_CLARIFICATIONS_KEY] = {
        "i1": [_Action("api"), _Action("cli")]
    }

    async def exercise() -> tuple[bool, bool]:
        first = await chainlit_renderer.expire_clarification_actions("i1")
        stale = await chainlit_renderer.expire_clarification_actions("i1")
        return first, stale

    first, stale = asyncio.run(exercise())

    assert (first, stale) == (True, False)
    assert removed == ["api", "cli"]
