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
