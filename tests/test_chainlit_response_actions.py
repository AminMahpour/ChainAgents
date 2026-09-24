"""Behavior of configured actions attached to Chainlit responses."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chainagents.exports import response as exports
from chainagents.runtime.types import ChainlitResponseActionConfig


class Session:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


@pytest.fixture
def session(monkeypatch) -> Session:
    session = Session()
    monkeypatch.setattr(exports.cl, "user_session", session)
    monkeypatch.setattr(exports.cl, "context", SimpleNamespace(session=SimpleNamespace()))
    return session


ACTIONS = (
    ChainlitResponseActionConfig(
        name="summarize", label="Summarize", prompt="Summarize {response}", icon="list"
    ),
    ChainlitResponseActionConfig(
        name="explain", label="Explain", prompt="Explain {prompt}: {response}"
    ),
)


def test_attach_response_actions_after_exports_and_save_context(session: Session) -> None:
    message = SimpleNamespace(id="answer-1", content="Answer", actions=[], elements=[], metadata={})

    exports.attach_response_export_actions(
        message,
        prompt="Question?",
        response_text="Answer",
        response_actions=ACTIONS,
    )

    assert [action.label for action in message.actions] == [
        "Markdown", "PDF", "Summarize", "Explain"
    ]
    assert message.actions[2].payload == {
        "response_id": "answer-1", "action_name": "summarize"
    }
    assert "Question?" not in str(message.actions[2].payload)
    assert message.metadata[exports.RESPONSE_CONTEXT_METADATA_KEY] == {
        "version": 1,
        "prompt": "Question?",
        "export_label": "",
    }
    assert session.get(exports.RESPONSE_EXPORTS_SESSION_KEY)["answer-1"]["response_text"] == "Answer"


def test_response_action_prompt_replaces_only_known_tokens_once() -> None:
    rendered = exports.expand_response_action_prompt(
        "Before {prompt}; body {response}; keep {unknown}",
        prompt="What is {response}?",
        response_text="The answer is {prompt}.",
    )

    assert rendered == (
        "Before What is {response}?; body The answer is {prompt}.; keep {unknown}"
    )


def test_resolve_response_action_targets_clicked_older_response(session: Session) -> None:
    for message_id, prompt, response in (
        ("first", "Original question", "Original answer"),
        ("second", "Later question", "Later answer"),
    ):
        exports.attach_response_export_actions(
            SimpleNamespace(id=message_id, content=response, actions=[], elements=[], metadata={}),
            prompt=prompt,
            response_text=response,
            response_actions=ACTIONS,
        )

    action = SimpleNamespace(
        forId="first", payload={"response_id": "first", "action_name": "explain"}
    )
    resolved = exports.resolve_response_action(action, ACTIONS)

    assert resolved is not None
    assert resolved.prompt == "Explain Original question: Original answer"
    assert resolved.label == "Explain"
    assert exports.resolve_response_action(
        SimpleNamespace(forId="second", payload=action.payload), ACTIONS
    ) is None
    assert exports.resolve_response_action(
        SimpleNamespace(forId="first", payload={"response_id": "first", "action_name": "missing"}),
        ACTIONS,
    ) is None


def test_restore_response_actions_from_saved_step_and_legacy_record(
    session: Session, monkeypatch
) -> None:
    session.set(
        exports.RESPONSE_EXPORTS_SESSION_KEY,
        {"legacy": {"prompt": "Old prompt", "response_text": "Old answer", "basename": "old"}},
    )
    thread = {
        "steps": [
            {
                "id": "saved", "type": "assistant_message", "output": "Saved answer",
                "metadata": {exports.RESPONSE_CONTEXT_METADATA_KEY: {
                    "version": 1, "prompt": "Saved prompt", "export_label": ""
                }},
            },
            {"id": "legacy", "type": "assistant_message", "output": "Old answer", "metadata": {}},
            {"id": "system", "type": "assistant_message", "output": "Status", "metadata": {}},
        ]
    }

    restored = exports.restore_response_export_actions(thread, response_actions=ACTIONS)
    repeated = exports.restore_response_export_actions(thread, response_actions=ACTIONS)
    session.set("restored_response_action_ids", [action.id for action in restored])
    monkeypatch.setattr(exports.cl.context, "session", SimpleNamespace())
    reopened = exports.restore_response_export_actions(thread, response_actions=ACTIONS)

    assert [(action.forId, action.label) for action in restored] == [
        ("saved", "Markdown"), ("saved", "PDF"), ("saved", "Summarize"), ("saved", "Explain"),
        ("legacy", "Markdown"), ("legacy", "PDF"), ("legacy", "Summarize"), ("legacy", "Explain"),
    ]
    assert repeated == []
    assert [(action.forId, action.label) for action in reopened] == [
        (action.forId, action.label) for action in restored
    ]
    assert exports.RESTORED_RESPONSE_ACTIONS_CONNECTION_ATTR not in session.values
    assert exports.resolve_response_action(
        SimpleNamespace(forId="saved", payload={"response_id": "saved", "action_name": "summarize"}),
        ACTIONS,
    ).prompt == "Summarize Saved answer"
    assert session.get(exports.RESPONSE_EXPORTS_SESSION_KEY).get("system") is None
