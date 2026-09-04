"""Exercise user-driven reflection retries with real, verified memory storage."""

import asyncio
from dataclasses import replace

import pytest
from deepagents.backends import StoreBackend
from deepagents.backends.protocol import WriteResult
from langgraph.store.memory import InMemoryStore

import main
from chainagents.runtime.core import AgentRuntime, AppSettings
from chainagents.runtime.reflection import ReflectionConfig, ReflectionProposal
from test_deepagent_runtime_rag import make_runtime_config


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("decision", "failure"),
    [
        ("retry", "error"),
        ("retry", "mismatch"),
        ("dismiss", "error"),
        ("timeout", "error"),
        ("cancel-request", "error"),
        ("cancel-save", "cancel"),
    ],
)
async def test_failed_reflection_keeps_same_proposal_for_user_retry(
    tmp_path, monkeypatch, decision, failure
):
    """Replace consumed actions after failure; never retry writes without a click."""
    config = make_runtime_config(tmp_path)
    config = replace(
        config,
        agent_state="stateful",
        extensions=replace(
            config.extensions,
            agent_memory_namespace="reflection-ui-test",
            agent_reflection=ReflectionConfig(enabled=True),
        ),
    )
    runtime = AgentRuntime(config, project_root=tmp_path)
    runtime._store = InMemoryStore()
    proposal = ReflectionProposal(
        reason="correction",
        memory_file="/memories/AGENTS.md",
        lesson="- Verify the generated output directory.",
        trigger="Wrong",
    )
    asks = []
    messages = []
    proposals = []
    writes = []
    verified_reads = []
    original_write = StoreBackend.awrite
    original_read = StoreBackend.aread
    original_save = runtime.save_reflection

    async def write_with_transient_failure(backend, path, content):
        writes.append(content)
        if len(writes) == 1:
            if failure == "cancel":
                raise asyncio.CancelledError()
            if failure == "mismatch":
                # A backend can acknowledge a write without actually persisting it.
                return WriteResult(path=path)
            raise OSError("temporary storage outage")
        return await original_write(backend, path, content)

    async def observe_real_read(backend, *args, **kwargs):
        result = await original_read(backend, *args, **kwargs)
        if result.file_data and proposal.lesson in result.file_data["content"]:
            verified_reads.append(result.file_data["content"])
        return result

    async def observe_real_save(value):
        proposals.append(value)
        await original_save(value)

    class AskActionMessage:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.actions = list(kwargs["actions"])
            asks.append(self)

        async def send(self):
            assert self.kwargs["timeout"] == 90
            assert self.kwargs["raise_on_timeout"] is False
            assert proposal.lesson in self.kwargs["content"]
            assert [action.label for action in self.actions] == (
                ["Save lesson", "Dismiss"] if len(asks) == 1 else ["Retry", "Dismiss"]
            )
            if len(asks) == 1:
                selected = self.actions[0]
            else:
                assert len(asks) == 2, "Unexpected extra confirmation"
                assert len(writes) == 1, "A write retried without user confirmation"
                assert not any(message.startswith("Saved") for message in messages)
                assert await runtime.store.aget(("reflection-ui-test",), "/AGENTS.md") is None
                if decision == "cancel-request":
                    raise asyncio.CancelledError()
                if decision == "timeout":
                    self.actions.clear()
                    return None
                selected = self.actions[0 if decision == "retry" else 1]
            # Chainlit removes actions before returning the user's selection.
            self.actions.clear()
            return {"payload": selected.payload, "label": selected.label}

    class Message:
        def __init__(self, content, **kwargs):
            self.content = content

        async def send(self):
            if self.content.startswith("Saved"):
                assert verified_reads, "Success preceded verified readback"
                item = await runtime.store.aget(("reflection-ui-test",), "/AGENTS.md")
                assert item.value["content"].count(proposal.lesson) == 1
            messages.append(self.content)

    monkeypatch.setattr(StoreBackend, "awrite", write_with_transient_failure)
    monkeypatch.setattr(StoreBackend, "aread", observe_real_read)
    monkeypatch.setattr(runtime, "save_reflection", observe_real_save)
    monkeypatch.setattr(main.cl, "AskActionMessage", AskActionMessage)
    monkeypatch.setattr(main.cl, "Message", Message)

    async def exercise():
        await main.ask_to_save_reflection_lesson(
            runtime=runtime,
            settings=AppSettings(
                model_name="fake-model", reasoning_level="medium", thread_id="t"
            ),
            proposal=proposal,
            reasoning_level="medium",
            model_name="fake-model",
            async_url_override=None,
            mcp_session_id=None,
        )

    if decision.startswith("cancel"):
        with pytest.raises(asyncio.CancelledError):
            await exercise()
    else:
        await exercise()

    assert len(asks) == (1 if decision == "cancel-save" else 2)
    assert all(value is proposal for value in proposals)
    assert len(proposals) == len(writes) == (2 if decision == "retry" else 1)
    assert sum(message.startswith("Saved") for message in messages) == (decision == "retry")
    item = await runtime.store.aget(("reflection-ui-test",), "/AGENTS.md")
    if decision == "retry":
        assert item.value["content"].count(proposal.lesson) == 1
    else:
        assert item is None
