"""Configured Chainlit response actions use the normal agent turn."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest

import main
from chainagents.exports import response as exports
from chainagents.runtime.types import ChainlitResponseActionConfig
from chainagents.runtime.types import MessagingConfig, UserInputConfig
from chainagents.runtime.messaging import MessageBroker
from chainagents.turns.controller import ConversationInputController

chainlit_socket = importlib.import_module("chainlit.socket")
chainlit_callbacks = importlib.import_module("chainlit.callbacks")


class Session:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {
            exports.RESPONSE_EXPORTS_SESSION_KEY: {
                "earlier": {
                    "prompt": "Original request",
                    "response_text": "Earlier answer",
                    "basename": "original",
                }
            }
        }

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


@pytest.mark.anyio
async def test_queued_chainlit_prompt_keeps_submitted_settings(monkeypatch) -> None:
    session = Session()
    session.set(main.SESSION_SETTINGS_KEY, {"model": "original"})
    broker = MessageBroker(MessagingConfig(enabled=True))
    controller = ConversationInputController(UserInputConfig(enabled=True), broker)
    runtime = SimpleNamespace(
        user_input=controller,
        config=SimpleNamespace(
            model_name="original", model_choices=("original", "changed"),
            extensions=SimpleNamespace(
                chainlit_reasoning_steps_enabled=True,
                chainlit_tool_steps_enabled=True,
            ),
        ),
    )
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    seen: list[tuple[str, str]] = []

    class Message:
        def __init__(self, content: str):
            self.id = content
            self.content = content

    class Step:
        def __init__(self, **_kwargs):
            self.input = ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    async def get_runtime():
        return runtime

    async def handle(message, *, settings_override=None):
        model = (
            settings_override.model_name
            if settings_override is not None
            else session.get(main.SESSION_SETTINGS_KEY)["model"]
        )
        seen.append((message.content, model))
        if message.content == "first":
            first_started.set()
            await release_first.wait()

    monkeypatch.setattr(main.cl, "user_session", session)
    monkeypatch.setattr(main.cl, "Step", Step)
    monkeypatch.setattr(main, "get_runtime_or_notify", get_runtime)
    monkeypatch.setattr(main, "_handle_message", handle)
    monkeypatch.setattr(
        main, "coerce_settings",
        lambda raw, **_kwargs: SimpleNamespace(thread_id="s", model_name=raw["model"]),
    )

    main._submit_nonblocking_message(runtime, "s", Message("first"))
    await first_started.wait()
    main._submit_nonblocking_message(runtime, "s", Message("second"))
    session.set(main.SESSION_SETTINGS_KEY, {"model": "changed"})
    release_first.set()
    await controller.wait_idle("s")
    assert seen == [("first", "original"), ("second", "original")]


@pytest.mark.anyio
async def test_nonblocking_chainlit_prompt_keeps_configured_rendering_defaults(monkeypatch) -> None:
    session = Session()
    controller = ConversationInputController(
        UserInputConfig(enabled=True), MessageBroker(MessagingConfig(enabled=True))
    )
    runtime = SimpleNamespace(
        user_input=controller,
        config=SimpleNamespace(
            model_name="model", model_choices=("model",),
            extensions=SimpleNamespace(
                user_input=UserInputConfig(enabled=True),
                chainlit_reasoning_steps_enabled=False,
                chainlit_tool_steps_enabled=False,
            ),
        ),
    )
    seen: list[tuple[bool, bool]] = []

    class Message:
        def __init__(self, content: str, author: str = "User", actions=None):
            self.id = content
            self.content = content
            self.author = author
            self.actions = actions or []

        async def send(self):
            return self

    class Step:
        def __init__(self, **_kwargs):
            self.input = ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    async def get_runtime():
        return runtime

    async def handle(_message, *, settings_override):
        seen.append((settings_override.show_reasoning_stream, settings_override.show_tool_calls))

    monkeypatch.setattr(main.cl, "user_session", session)
    monkeypatch.setattr(main.cl, "context", SimpleNamespace(session=SimpleNamespace(thread_id="s")))
    monkeypatch.setattr(main.cl, "Message", Message)
    monkeypatch.setattr(main.cl, "Step", Step)
    monkeypatch.setattr(main, "get_runtime_or_notify", get_runtime)
    monkeypatch.setattr(main, "_handle_message", handle)

    await main._guarded_chainlit_message_callback(Message("hello"))
    await controller.wait_idle("s")
    assert seen == [(False, False)]


@pytest.mark.anyio
async def test_chainlit_accepts_busy_input_and_offers_steer_or_queue(monkeypatch) -> None:
    session = Session()
    broker = MessageBroker(MessagingConfig(enabled=True))
    user_input = ConversationInputController(UserInputConfig(enabled=True), broker)
    runtime = SimpleNamespace(
        user_input=user_input,
        config=SimpleNamespace(model_name="model", model_choices=("model",),
            extensions=SimpleNamespace(
                user_input=UserInputConfig(enabled=True),
                chainlit_reasoning_steps_enabled=True,
                chainlit_tool_steps_enabled=True,
            )),
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: list[str] = []
    notices: list[Any] = []
    run_steps: list[tuple[str, str, str]] = []

    class Message:
        def __init__(self, content: str, author: str = "User", actions=None):
            self.id = content
            self.content = content
            self.author = author
            self.actions = actions or []
            self.elements = []

        async def send(self):
            notices.append(self)
            return self

    class Step:
        def __init__(self, *, name, type, parent_id):
            self.name = name
            self.type = type
            self.parent_id = parent_id
            self.input = ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            run_steps.append((self.name, self.parent_id, self.input))
            return False

    async def get_runtime():
        return runtime

    async def handle(message, *, settings_override=None):
        seen.append(message.content)
        if message.content == "first":
            entered.set()
            await release.wait()

    monkeypatch.setattr(main.cl, "user_session", session)
    monkeypatch.setattr(main.cl, "Message", Message)
    monkeypatch.setattr(main.cl, "Step", Step)
    monkeypatch.setattr(main.cl, "context", SimpleNamespace(session=SimpleNamespace(thread_id="ui-chat")))
    monkeypatch.setattr(main, "get_runtime_or_notify", get_runtime)
    monkeypatch.setattr(main, "coerce_settings", lambda *_args, **_kwargs: SimpleNamespace(thread_id="s"))
    monkeypatch.setattr(main, "_handle_message", handle)

    await main._guarded_chainlit_message_callback(Message("first"))
    await entered.wait()
    await main._guarded_chainlit_message_callback(Message("second"))
    assert seen == ["first"]
    assert [action.label for action in notices[-1].actions] == ["Steer active turn", "Queue next turn"]
    draft_id = notices[-1].actions[1].payload["draft_id"]
    await main._submit_busy_input(SimpleNamespace(name=main.QUEUE_INPUT_ACTION, payload={"draft_id": draft_id}))
    release.set()
    await user_input.wait_idle("s")
    assert seen == ["first", "second"]
    assert run_steps == [
        ("on_message", "first", "first"),
        ("on_message", "second", "second"),
    ]
    await main._stop_nonblocking_input(SimpleNamespace(payload={"thread_id": "another-chat"}))
    assert user_input.status("s")["paused"] is False
    assert user_input.status("another-chat")["paused"] is False
    await main._resume_nonblocking_input(SimpleNamespace(payload={"thread_id": "another-chat"}))
    assert user_input.status("s")["paused"] is False
    await main._stop_nonblocking_input(SimpleNamespace(payload={"thread_id": "s"}))
    assert user_input.status("s")["paused"] is True
    await main._resume_nonblocking_input(SimpleNamespace(payload={"thread_id": "s"}))
    assert user_input.status("s")["paused"] is False


@pytest.mark.anyio
async def test_late_steering_followup_after_response_action_is_visible(monkeypatch) -> None:
    session = Session()
    broker = MessageBroker(MessagingConfig(enabled=True))
    user_input = ConversationInputController(UserInputConfig(enabled=True), broker)
    action_config = ChainlitResponseActionConfig(
        name="summarize", label="Summarize", prompt="Summarize {response}"
    )
    runtime = SimpleNamespace(
        user_input=user_input,
        config=SimpleNamespace(
            model_name="model", model_choices=("model",),
            extensions=SimpleNamespace(
                user_input=UserInputConfig(enabled=True),
                chainlit_response_actions=(action_config,),
            ),
        ),
    )
    settings = SimpleNamespace(
        thread_id="s", model_name="model", reasoning_level="medium"
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    turns: list[dict[str, Any]] = []
    messages: list[tuple[str, str]] = []

    class Message:
        def __init__(self, content: str, author: str = "Assistant"):
            self.content = content
            self.author = author

        async def send(self):
            messages.append((self.author, self.content))

    async def get_runtime():
        return runtime

    async def run_turn(**kwargs):
        turns.append(kwargs)
        if len(turns) == 1:
            entered.set()
            await release.wait()

    monkeypatch.setattr(main.cl, "user_session", session)
    monkeypatch.setattr(main.cl, "Message", Message)
    monkeypatch.setattr(main, "get_runtime_or_notify", get_runtime)
    monkeypatch.setattr(main, "coerce_settings", lambda *_args, **_kwargs: settings)
    monkeypatch.setattr(main, "current_mcp_session_id", lambda: "s")
    monkeypatch.setattr(main, "settings_reasoning_level_is_explicit", lambda *_args: False)
    monkeypatch.setattr(main, "_run_agent_turn", run_turn)

    await main.run_response_action(SimpleNamespace(
        forId="earlier", payload={"response_id": "earlier", "action_name": "summarize"}
    ))
    await entered.wait()
    user_input.steer("s", "late correction")
    release.set()
    await user_input.wait_idle("s")
    assert turns[1]["display_prompt"] == "late correction"
    assert turns[1]["export_label"] == ""
    assert ("User", "late correction") in messages


@pytest.mark.anyio
async def test_response_action_runs_hidden_turn_with_clicked_response(monkeypatch) -> None:
    session = Session()
    action_config = ChainlitResponseActionConfig(
        name="summarize", label="Summarize", prompt="Summarize {prompt}: {response}"
    )
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            model_name="model", model_choices=("model",),
            extensions=SimpleNamespace(
                chainlit_reasoning_steps_enabled=True,
                chainlit_tool_steps_enabled=True,
                chainlit_response_actions=(action_config,),
            ),
        )
    )
    settings = SimpleNamespace(
        model_name="model", reasoning_level="medium", thread_id="thread-1",
        show_reasoning_stream=True, show_tool_calls=True,
    )
    emissions: list[str] = []
    turns: list[dict[str, Any]] = []
    messages: list[tuple[str, str]] = []
    fail_next = False

    async def get_runtime():
        return runtime

    async def run_turn(**kwargs: Any) -> None:
        nonlocal fail_next
        turns.append(kwargs)
        if fail_next:
            fail_next = False
            raise RuntimeError("agent unavailable")

    class Message:
        def __init__(self, content: str, author: str = "Assistant") -> None:
            self.content = content
            self.author = author

        async def send(self) -> None:
            messages.append((self.author, self.content))

    async def task_start():
        emissions.append("start")

    async def task_end():
        emissions.append("end")

    monkeypatch.setattr(main.cl, "user_session", session)
    monkeypatch.setattr(main.cl, "Message", Message)
    monkeypatch.setattr(main.cl, "context", SimpleNamespace(emitter=SimpleNamespace(
        task_start=task_start, task_end=task_end,
    ), session=SimpleNamespace(current_task=None)))
    monkeypatch.setattr(main, "get_runtime_or_notify", get_runtime)
    monkeypatch.setattr(main, "coerce_settings", lambda *_args, **_kwargs: settings)
    monkeypatch.setattr(main, "current_mcp_session_id", lambda: "mcp-1")
    monkeypatch.setattr(main, "settings_reasoning_level_is_explicit", lambda *_args: False)
    monkeypatch.setattr(main, "_run_agent_turn", run_turn)

    await main.run_response_action(SimpleNamespace(
        forId="earlier",
        payload={"response_id": "earlier", "action_name": "summarize"},
    ))

    assert len(turns) == 1
    assert turns[0]["agent_prompt"] == "Summarize Original request: Earlier answer"
    assert turns[0]["display_prompt"] == ""
    assert turns[0]["export_label"] == "Summarize"
    assert emissions == ["start", "end"]
    assert messages == []
    assert session.get(main.SESSION_ACTIVE_TURN_KEY) is None

    fail_next = True
    await main.run_response_action(SimpleNamespace(
        forId="earlier",
        payload={"response_id": "earlier", "action_name": "summarize"},
    ))
    assert session.get(main.SESSION_ACTIVE_TURN_KEY) is None
    assert messages == [("System", "The response action failed. Please try again.")]
    await main.run_response_action(SimpleNamespace(
        forId="earlier",
        payload={"response_id": "earlier", "action_name": "summarize"},
    ))
    assert len(turns) == 3
    assert emissions == ["start", "end"] * 3


@pytest.mark.anyio
async def test_response_action_rejects_overlap_and_stop_cancels_active_turn(monkeypatch) -> None:
    session = Session()
    action_config = ChainlitResponseActionConfig(
        name="summarize", label="Summarize", prompt="Summarize {response}"
    )
    runtime = SimpleNamespace(config=SimpleNamespace(
        model_name="model", model_choices=("model",),
        extensions=SimpleNamespace(
            chainlit_reasoning_steps_enabled=True,
            chainlit_tool_steps_enabled=True,
            chainlit_response_actions=(action_config,),
        ),
    ))
    settings = SimpleNamespace(
        model_name="model", reasoning_level="medium", thread_id="thread-1",
        show_reasoning_stream=True, show_tool_calls=True,
    )
    entered = asyncio.Event()
    sent: list[str] = []
    ended: list[bool] = []
    lifecycle: list[str] = []

    async def get_runtime():
        return runtime

    async def run_turn(**_kwargs: Any) -> None:
        entered.set()
        await asyncio.Event().wait()

    class Message:
        def __init__(self, content: str, author: str = "Assistant") -> None:
            self.content = content

        async def send(self) -> None:
            sent.append(self.content)

    async def noop():
        return None

    async def task_end():
        ended.append(True)

    monkeypatch.setattr(main.cl, "user_session", session)
    monkeypatch.setattr(main.cl, "Message", Message)
    monkeypatch.setattr(main.cl, "context", SimpleNamespace(emitter=SimpleNamespace(
        task_start=noop, task_end=task_end,
    ), session=SimpleNamespace(current_task=None)))
    monkeypatch.setattr(main, "get_runtime_or_notify", get_runtime)
    monkeypatch.setattr(main, "coerce_settings", lambda *_args, **_kwargs: settings)
    monkeypatch.setattr(main, "current_mcp_session_id", lambda: "mcp-1")
    monkeypatch.setattr(main, "settings_reasoning_level_is_explicit", lambda *_args: False)
    monkeypatch.setattr(main, "_run_agent_turn", run_turn)
    action = SimpleNamespace(
        forId="earlier", payload={"response_id": "earlier", "action_name": "summarize"}
    )

    first = asyncio.create_task(main.run_response_action(action))
    await entered.wait()
    await main.run_response_action(action)
    class RenderedUserMessage:
        id = "already-rendered"
        content = "another request"

    async def process_message(_payload: Any) -> RenderedUserMessage:
        lifecycle.append("rendered")
        return RenderedUserMessage()

    monkeypatch.setattr(chainlit_socket, "init_ws_context", lambda _session: SimpleNamespace(
        emitter=SimpleNamespace(
            process_message=process_message,
            task_start=noop,
            task_end=noop,
        ),
    ))
    monkeypatch.setattr(chainlit_callbacks, "Step", lambda **_kwargs: lifecycle.append("run step"))
    await chainlit_socket.process_message(None, {})
    other_session = Session()
    monkeypatch.setattr(main.cl, "user_session", other_session)
    assert main._claim_active_turn()
    main._release_active_turn()
    monkeypatch.setattr(main.cl, "user_session", session)
    await main.on_stop()
    await first

    assert len(sent) == 2
    assert all("already responding" in notice for notice in sent)
    assert all("was not sent" in notice for notice in sent)
    assert lifecycle == ["rendered"]
    assert ended == [True]
    assert session.get(main.SESSION_ACTIVE_TURN_KEY) is None


@pytest.mark.anyio
async def test_saved_response_actions_are_sent_for_original_message(monkeypatch) -> None:
    session = Session()
    session.set(exports.RESPONSE_EXPORTS_SESSION_KEY, {})
    action_config = ChainlitResponseActionConfig(
        name="summarize", label="Summarize", prompt="Summarize {response}"
    )
    runtime = SimpleNamespace(config=SimpleNamespace(extensions=SimpleNamespace(
        chainlit_response_actions=(action_config,),
    )))
    sent: list[tuple[str, str]] = []

    async def send_action(self, for_id: str) -> None:
        sent.append((for_id, self.label))

    monkeypatch.setattr(main.cl, "user_session", session)
    monkeypatch.setattr(main.cl.Action, "send", send_action)
    monkeypatch.setattr(main.cl, "context", SimpleNamespace(session=SimpleNamespace()))
    thread = {"steps": [{
        "id": "saved", "type": "assistant_message", "output": "Prior response",
        "metadata": {exports.RESPONSE_CONTEXT_METADATA_KEY: {
            "version": 1, "prompt": "Prior request", "export_label": ""
        }},
    }]}

    await main._restore_saved_response_actions(thread, runtime)
    await main._restore_saved_response_actions(thread, runtime)

    assert sent == [
        ("saved", "Markdown"), ("saved", "PDF"), ("saved", "Summarize")
    ]
