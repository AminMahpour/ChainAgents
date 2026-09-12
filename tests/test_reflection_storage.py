"""Verify confirmed memory writes with the installed backend and an in-memory store."""

import asyncio
from dataclasses import replace

import pytest
from deepagents.backends import StoreBackend
from deepagents.backends.protocol import ReadResult, WriteResult
from langgraph.store.memory import InMemoryStore
from langgraph.graph import StateGraph, START, END

from chainagents.runtime import core
from chainagents.runtime.reflection import ReflectionConfig, ReflectionProposal
from test_deepagent_runtime_rag import make_runtime_config


@pytest.fixture
def runtime(tmp_path):
    config = make_runtime_config(tmp_path)
    config = replace(
        config,
        agent_state="stateful",
        extensions=replace(
            config.extensions,
            agent_memory_namespace="reflection-tests",
            agent_reflection=ReflectionConfig(enabled=True),
        ),
    )
    instance = core.AgentRuntime(config, project_root=tmp_path)
    instance._store = InMemoryStore()
    return instance


def proposal(
    lesson="- Verify before relying on assumptions.", path="/memories/AGENTS.md"
):
    return ReflectionProposal(
        reason="correction", memory_file=path, lesson=lesson, trigger="Wrong"
    )


def backend(runtime):
    return StoreBackend(namespace=lambda _: ("reflection-tests",), store=runtime.store)


def test_save_writes_once_to_actual_composite_memory_route(runtime, monkeypatch):
    async def no_model(*args, **kwargs):
        pytest.fail("Persistence must not invoke an agent")

    monkeypatch.setattr(runtime, "get_agent", no_model)

    async def exercise():
        await runtime.save_reflection(proposal())
        await runtime.save_reflection(proposal("Verify before relying on assumptions."))
        composite = core.build_deepagent_backend(
            project_root=runtime.project_root, memory_namespace="reflection-tests"
        )

        async def read_memory(state):
            result = await composite.aread("/memories/AGENTS.md")
            return {"content": result.file_data["content"]}

        graph = StateGraph(dict)
        graph.add_node("read", read_memory)
        graph.add_edge(START, "read")
        graph.add_edge("read", END)
        result = await graph.compile(store=runtime.store).ainvoke({})
        assert result["content"] == (
            "## Lessons learned from corrections\n\n- Verify before relying on assumptions.\n"
        )
        assert (
            await runtime.store.aget(("reflection-tests",), "/memories/AGENTS.md")
            is None
        )
        assert await runtime.store.aget(("other-agent",), "/AGENTS.md") is None

    asyncio.run(exercise())


def test_save_preserves_large_file_and_later_sections(runtime):
    original = (
        "# Memory\n"
        + "Existing context\n" * 2100
        + (
            "## Lessons learned from corrections\n\n- Old lesson.\n\n"
            "## Unrelated instructions\n\nKeep this exactly.\n"
        )
    )

    async def exercise():
        await backend(runtime).awrite("/AGENTS.md", original)
        await runtime.save_reflection(proposal())
        item = await runtime.store.aget(("reflection-tests",), "/AGENTS.md")
        assert item.value["content"] == original.replace(
            "## Unrelated instructions",
            "- Verify before relying on assumptions.\n\n## Unrelated instructions",
        )

    asyncio.run(exercise())


def test_concurrent_saves_preserve_each_unique_lesson(runtime):
    async def exercise():
        await asyncio.gather(
            *(
                runtime.save_reflection(proposal(f"- Lesson {n % 5}."))
                for n in range(30)
            )
        )
        item = await runtime.store.aget(("reflection-tests",), "/AGENTS.md")
        for n in range(5):
            assert item.value["content"].count(f"- Lesson {n}.") == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["read", "write", "mismatch", "exception"])
def test_failed_storage_never_reports_success_and_can_retry(
    runtime, monkeypatch, failure
):
    async def exercise():
        await backend(runtime).awrite("/AGENTS.md", "# Preserve me\n")
        with monkeypatch.context() as patch:
            if failure == "read":

                async def fail_read(*args, **kwargs):
                    return ReadResult(error="permission denied")

                patch.setattr(StoreBackend, "aread", fail_read)
            elif failure == "exception":

                async def fail_get(*args, **kwargs):
                    raise OSError("unavailable")

                patch.setattr(InMemoryStore, "aget", fail_get)
            else:

                async def fail_write(*args, **kwargs):
                    return (
                        WriteResult(error="denied")
                        if failure == "write"
                        else WriteResult(path="/AGENTS.md")
                    )

                patch.setattr(StoreBackend, "awrite", fail_write)
            with pytest.raises(RuntimeError, match="[Rr]eflection"):
                await runtime.save_reflection(proposal())
        item = await runtime.store.aget(("reflection-tests",), "/AGENTS.md")
        assert item.value["content"] == "# Preserve me\n"
        await runtime.save_reflection(proposal())
        item = await runtime.store.aget(("reflection-tests",), "/AGENTS.md")
        assert (
            item.value["content"].count("- Verify before relying on assumptions.") == 1
        )

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "state,enabled,path,lesson",
    [
        ("stateless", True, "/memories/AGENTS.md", "valid"),
        ("stateful", False, "/memories/AGENTS.md", "valid"),
        ("stateful", True, "/memories/other.md", "valid"),
        ("stateful", True, "/memories/AGENTS.md", "  -  "),
        ("stateful", True, "/memories/AGENTS.md", "x" * 701),
    ],
)
def test_save_enforces_runtime_configuration(runtime, state, enabled, path, lesson):
    runtime.config = replace(
        runtime.config,
        agent_state=state,
        extensions=replace(
            runtime.config.extensions,
            agent_reflection=ReflectionConfig(enabled=enabled),
        ),
    )

    async def exercise():
        with pytest.raises(ValueError):
            await runtime.save_reflection(proposal(lesson, path))
        assert await runtime.store.asearch(("reflection-tests",)) == []

    asyncio.run(exercise())


def test_save_preserves_legacy_content(runtime):
    async def exercise():
        await runtime.store.aput(
            ("reflection-tests",), "/AGENTS.md", {"content": ["# Memory", "Keep me."]}
        )
        await runtime.save_reflection(proposal())
        item = await runtime.store.aget(("reflection-tests",), "/AGENTS.md")
        assert item.value["content"].startswith("# Memory\nKeep me.\n\n")

    asyncio.run(exercise())


def test_save_preserves_existing_line_endings(runtime):
    original = "# Memory\r\nKeep these exact bytes.\r\n"

    async def exercise():
        await backend(runtime).awrite("/AGENTS.md", original)
        await runtime.save_reflection(proposal())
        item = await runtime.store.aget(("reflection-tests",), "/AGENTS.md")
        assert item.value["content"].startswith(original)

    asyncio.run(exercise())


def test_save_uses_custom_configured_nested_path_and_ignores_fenced_headings(runtime):
    path = "/memories/team/notes.md"
    runtime.config = replace(
        runtime.config,
        extensions=replace(
            runtime.config.extensions,
            agent_reflection=ReflectionConfig(enabled=True, memory_file=path),
        ),
    )
    original = (
        "# Memory\n\n```md\n## Lessons learned from corrections\n- Example only.\n```\n"
    )

    async def exercise():
        await backend(runtime).awrite("/team/notes.md", original)
        await runtime.save_reflection(proposal(path=path))
        item = await runtime.store.aget(("reflection-tests",), "/team/notes.md")
        assert item.value["content"] == original + (
            "\n## Lessons learned from corrections\n\n- Verify before relying on assumptions.\n"
        )
        assert await runtime.store.aget(("reflection-tests",), "/AGENTS.md") is None

    asyncio.run(exercise())


def test_existing_empty_list_item_does_not_block_saving(runtime):
    original = "## Lessons learned from corrections\n\n-\n"

    async def exercise():
        await backend(runtime).awrite("/AGENTS.md", original)
        await runtime.save_reflection(
            proposal("* - Verify before relying on assumptions.")
        )
        await runtime.save_reflection(proposal())
        item = await runtime.store.aget(("reflection-tests",), "/AGENTS.md")
        assert (
            item.value["content"]
            == original + "- Verify before relying on assumptions.\n"
        )

    asyncio.run(exercise())
