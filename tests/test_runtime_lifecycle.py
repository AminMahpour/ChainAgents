"""Exercise runtime resource ownership without live transports or models."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import anyio
import pytest

import deepagent_runtime as core
import chainagents.runtime.backends as runtime_backends
import chainagents.runtime.artifacts as runtime_artifacts
import chainagents.runtime.background_tasks as runtime_background_tasks
import chainagents.runtime.config as runtime_config
import chainagents.runtime.lifecycle as runtime_lifecycle
import chainagents.runtime.middleware as runtime_middleware
from chainagents.runtime.background_tasks import (
    BackgroundTaskManager,
    create_background_task_tools,
)
from chainagents.runtime.types import BackgroundSubagentConfig
from langchain.tools import ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from deepagents.backends.protocol import DeleteResult, FileUploadResponse
from test_deepagent_runtime_rag import make_runtime_config, make_extensions_config


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    config = replace(make_runtime_config(tmp_path), rag=None, rag_requested=False)
    instance = core.AgentRuntime(config, project_root=tmp_path)
    instance._store = InMemoryStore()
    instance._checkpointer = MemorySaver()
    monkeypatch.setattr(instance, "_build_model", lambda *a, **kw: object())
    monkeypatch.setattr(
        runtime_middleware,
        "create_deep_agent_with_configured_summarization",
        lambda *a, **kw: object(),
    )
    return instance


def test_stateful_context_closes_on_its_owner_task(runtime, monkeypatch):
    runtime.config = replace(
        runtime.config,
        extensions=make_extensions_config(
            mcp_stateful=True, agent_mcp_servers=("repo",)
        ),
    )
    events = []

    @asynccontextmanager
    async def session(server):
        async with anyio.create_task_group():
            events.append(("open", asyncio.current_task()))
            try:
                yield object()
            finally:
                events.append(("close", asyncio.current_task()))

    runtime._mcp_client = SimpleNamespace(
        session=session, callbacks=None, tool_interceptors=[]
    )

    async def load(*a, **kw):
        return []

    monkeypatch.setattr(runtime_lifecycle, "load_mcp_tools", load)

    async def exercise():
        await asyncio.create_task(
            runtime.get_agent("medium", thread_id="thread", mcp_session_id="session")
        )
        await runtime.close_mcp_session("session")
        assert len(events) == 2
        assert events[0][1] is events[1][1]

    asyncio.run(exercise())


def test_runtime_background_tasks_share_the_artifact_registry(runtime):
    """Runtime background jobs must inherit the registry used by their backend."""
    assert (
        runtime.background_tasks.artifact_registry
        is runtime.large_tool_result_artifacts
    )


@pytest.mark.parametrize("stateful", [False, True])
def test_conversation_close_evicts_graph_with_no_mcp_client(runtime, stateful):
    runtime.config = replace(
        runtime.config,
        extensions=replace(runtime.config.extensions, mcp_stateful=stateful),
    )

    async def exercise():
        first = await runtime.get_agent(
            "medium", thread_id="thread", mcp_session_id="session"
        )
        other = await runtime.get_agent(
            "medium", thread_id="other", mcp_session_id="other-session"
        )
        await runtime.close_conversation(thread_id="thread", mcp_session_id="session")
        assert first not in runtime._agents.values()
        assert (
            await runtime.get_agent(
                "medium", thread_id="other", mcp_session_id="other-session"
            )
            is other
        )
        assert (
            await runtime.get_agent(
                "medium", thread_id="thread", mcp_session_id="session"
            )
            is not first
        )
        await runtime.close()

    asyncio.run(exercise())


def test_conversation_close_cancels_background_tasks_before_mcp(runtime, monkeypatch):
    """Conversation resources must remain alive until background jobs stop."""
    runtime.background_tasks = BackgroundTaskManager(
        BackgroundSubagentConfig(enabled=True)
    )

    async def exercise():
        queue = runtime.background_tasks.subscribe("thread")

        async def runner(task_id):
            await asyncio.Event().wait()
            return task_id

        spawned = await runtime.background_tasks.spawn(
            session_id="thread",
            agent_name="worker",
            description="work",
            agent_path=("worker",),
            runner=runner,
        )
        events = []

        async def close_mcp_session(session_id):
            events.append((session_id, not queue.empty()))

        monkeypatch.setattr(runtime, "close_mcp_session", close_mcp_session)
        await runtime.close_conversation(thread_id="thread", mcp_session_id="mcp")

        terminal = queue.get_nowait()
        assert terminal.task_id == spawned.task_id
        assert terminal.status == "cancelled"
        assert events == [("mcp", True)]
        assert await runtime.background_tasks.list("thread") == []
        await runtime.background_tasks.close()

    asyncio.run(exercise())


def test_conversation_close_deletes_only_its_large_tool_results(runtime):
    """Conversation teardown must not delete another session's offloads."""

    async def exercise():
        backend = runtime_backends.build_deepagent_backend(
            project_root=runtime.project_root,
            include_memories=False,
            artifact_registry=runtime.large_tool_result_artifacts,
        )
        prefix = runtime_backends.deepagent_artifacts_route_prefix(runtime.project_root)
        first_path = f"{prefix}large_tool_results/first-call"
        second_path = f"{prefix}large_tool_results/second-call"
        first = runtime.large_tool_result_artifacts.open_session("thread")
        second = runtime.large_tool_result_artifacts.open_session("other")
        token = runtime.large_tool_result_artifacts.activate(first)
        assert backend.write(first_path, "first").error is None
        runtime.large_tool_result_artifacts.reset(token)
        token = runtime.large_tool_result_artifacts.activate(second)
        assert backend.write(second_path, "second").error is None
        runtime.large_tool_result_artifacts.reset(token)

        await runtime.close_conversation(
            thread_id="thread",
            mcp_session_id="session",
        )

        token = runtime.large_tool_result_artifacts.activate(first)
        assert backend.read(first_path).error is not None
        runtime.large_tool_result_artifacts.reset(token)
        token = runtime.large_tool_result_artifacts.activate(second)
        assert backend.read(second_path).error is None
        runtime.large_tool_result_artifacts.reset(token)
        await runtime.close()
        token = runtime.large_tool_result_artifacts.activate(second)
        assert backend.read(second_path).error is not None
        runtime.large_tool_result_artifacts.reset(token)

    asyncio.run(exercise())


def test_background_tasks_keep_the_foreground_artifact_generation(tmp_path):
    """An empty task context must reactivate its owning artifact generation."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        manager = BackgroundTaskManager(
            BackgroundSubagentConfig(enabled=True),
            artifact_registry=registry,
        )
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        path = f"{backend.artifacts_root}/large_tool_results/background-call"
        handle = registry.open_session("thread")
        token = registry.activate(handle)

        async def runner(task_id):
            assert backend.write(path, task_id).error is None
            return task_id

        spawned = await manager.spawn(
            session_id="thread",
            agent_name="worker",
            description="work",
            agent_path=("worker",),
            runner=runner,
        )
        registry.reset(token)
        finished = await manager.get("thread", spawned.task_id, wait_seconds=1)
        assert finished.status == "success"
        physical = (
            tmp_path
            / ".files"
            / "deepagent"
            / "session_tool_results"
            / handle.token
            / "background-call"
        )
        unscoped = tmp_path / ".files" / "deepagent" / "session_tool_results"
        assert physical.read_text(encoding="utf-8") == spawned.task_id
        assert not list(unscoped.glob("unscoped-*/background-call"))

        await asyncio.gather(
            registry.close_session("thread"),
            registry.close_session("thread"),
        )

        assert not physical.exists()
        await manager.close()
        await registry.close()

    asyncio.run(exercise())


def test_successful_artifact_close_releases_generation_state():
    """Closed sessions must not accumulate handles or per-session locks."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        first = registry.open_session("thread")

        await registry.close_session("thread")

        assert "thread" not in registry._current
        assert first not in registry._states
        assert "thread" not in registry._session_locks

        second = registry.open_session("thread")
        assert second != first
        await registry.close_session("thread")
        assert "thread" not in registry._current
        assert second not in registry._states
        assert "thread" not in registry._session_locks
        await registry.close()

    asyncio.run(exercise())


def test_artifact_context_is_scoped_to_its_registry(tmp_path):
    """Activating one registry must not replace another registry's session."""

    async def exercise():
        first_registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        second_registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        second_backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=second_registry,
        )
        second_handle = second_registry.open_session("second")
        second_token = second_registry.activate(second_handle)
        first_handle = first_registry.open_session("first")
        first_token = first_registry.activate(first_handle)
        logical = f"{second_backend.artifacts_root}/large_tool_results/result"

        try:
            assert second_backend.write(logical, "owned by second").error is None
            result = second_backend.read(logical)
            assert result.file_data is not None
            assert result.file_data["content"] == "owned by second"
        finally:
            first_registry.reset(first_token)
            second_registry.reset(second_token)

        physical = (
            tmp_path
            / ".files"
            / "deepagent"
            / "session_tool_results"
            / second_handle.token
            / "result"
        )
        assert physical.is_file()
        assert not list(
            (
                tmp_path
                / ".files"
                / "deepagent"
                / "session_tool_results"
                / first_handle.token
            ).glob("result")
        )
        await first_registry.close()
        await second_registry.close()

    asyncio.run(exercise())


def test_successful_artifact_close_removes_generation_directory(tmp_path):
    """Successful session cleanup must remove its physical token namespace."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        handle = registry.open_session("thread")
        token = registry.activate(handle)
        logical = f"{backend.artifacts_root}/large_tool_results/nested/result"
        assert backend.write(logical, "temporary").error is None
        registry.reset(token)
        generation_root = (
            tmp_path
            / ".files"
            / "deepagent"
            / "session_tool_results"
            / handle.token
        )
        assert generation_root.is_dir()

        await registry.close_session("thread")

        assert not generation_root.exists()
        await registry.close()

    asyncio.run(exercise())


def test_failed_artifact_mutations_remain_owned_until_session_close(
    tmp_path,
    monkeypatch,
):
    """Partial write and upload failures must not escape session cleanup."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        handle = registry.open_session("thread")
        token = registry.activate(handle)
        logical_root = f"{backend.artifacts_root}/large_tool_results"
        sync_write = backend.write(f"{logical_root}/sync-write", "\ud800")
        async_write = await backend.awrite(
            f"{logical_root}/async-write",
            "\ud800",
        )
        underlying = backend.backend
        original_upload = underlying.upload_files

        def partial_upload(files):
            original_upload(files)
            return [
                FileUploadResponse(path=path, error="disk full")
                for path, _content in files
            ]

        async def partial_async_upload(files):
            original_upload(files)
            return [
                FileUploadResponse(path=path, error="disk full")
                for path, _content in files
            ]

        monkeypatch.setattr(underlying, "upload_files", partial_upload)
        monkeypatch.setattr(underlying, "aupload_files", partial_async_upload)
        sync_upload = backend.upload_files(
            [(f"{logical_root}/sync-upload", b"partial")]
        )[0]
        async_upload = (
            await backend.aupload_files(
                [(f"{logical_root}/async-upload", b"partial")]
            )
        )[0]
        registry.reset(token)
        generation_root = (
            tmp_path
            / ".files"
            / "deepagent"
            / "session_tool_results"
            / handle.token
        )

        assert sync_write.error is not None
        assert async_write.error is not None
        assert sync_upload.error == "disk full"
        assert async_upload.error == "disk full"
        assert generation_root.is_dir()

        await registry.close_session("thread")

        assert not generation_root.exists()
        await registry.close()

    asyncio.run(exercise())


def test_physical_artifact_namespaces_are_hidden_from_filesystem_tools(tmp_path):
    """Only the active session's logical large-result namespace is accessible."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        handle = registry.open_session("first")
        token = registry.activate(handle)
        logical = f"{backend.artifacts_root}/large_tool_results/secret.txt"
        assert backend.write(logical, "session secret").error is None
        registry.reset(token)
        physical_root = f"{backend.artifacts_root}/session_tool_results"
        physical = f"{physical_root}/{handle.token}/secret.txt"
        alias_root = "/workspace/.files/deepagent/session_tool_results"
        alias = f"{alias_root}/{handle.token}/secret.txt"

        for path in (physical, alias):
            assert backend.read(path).error is not None
            assert (await backend.aread(path)).error is not None
            assert backend.write(path, "overwrite").error is not None
            assert (await backend.awrite(path, "overwrite")).error is not None
            assert backend.edit(path, "secret", "changed").error is not None
            assert (await backend.aedit(path, "secret", "changed")).error is not None
            assert backend.delete(path).error is not None
            assert (await backend.adelete(path)).error is not None
            assert backend.upload_files([(path, b"overwrite")])[0].error is not None
            assert (await backend.aupload_files([(path, b"overwrite")]))[
                0
            ].error is not None
            assert backend.download_files([path])[0].error is not None
            assert (await backend.adownload_files([path]))[0].error is not None

        for root in (physical_root, alias_root):
            assert backend.ls(root).error is not None
            assert (await backend.als(root)).error is not None
            assert backend.glob("**/*", root).error is not None
            assert (await backend.aglob("**/*", root)).error is not None
            assert backend.grep("secret", root).error is not None
            assert (await backend.agrep("secret", root)).error is not None

        for root in (backend.artifacts_root, "/workspace/.files/deepagent"):
            listed = backend.ls(root)
            async_listed = await backend.als(root)
            globbed = backend.glob("**/*", root)
            async_globbed = await backend.aglob("**/*", root)
            grepped = backend.grep("secret", root)
            async_grepped = await backend.agrep("secret", root)
            visible_paths = [
                item["path"]
                for result in (
                    listed.entries,
                    async_listed.entries,
                    globbed.matches,
                    async_globbed.matches,
                    grepped.matches,
                    async_grepped.matches,
                )
                for item in (result or [])
            ]
            assert all("session_tool_results" not in path for path in visible_paths)

        token = registry.activate(handle)
        assert backend.read(logical).file_data is not None
        registry.reset(token)
        await registry.close()

    asyncio.run(exercise())


def test_parent_searches_restore_only_the_active_logical_artifacts(tmp_path):
    """Parent discovery exposes the caller's logical namespace, not physical tokens."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        first = registry.open_session("first")
        first_token = registry.activate(first)
        absolute_logical = f"{backend.artifacts_root}/large_tool_results/first.txt"
        assert backend.write(absolute_logical, "first-secret").error is None
        registry.reset(first_token)
        second = registry.open_session("second")
        second_token = registry.activate(second)
        assert backend.write(
            f"{backend.artifacts_root}/large_tool_results/second.txt",
            "second-secret",
        ).error is None
        registry.reset(second_token)

        first_token = registry.activate(first)
        for root, logical_root in (
            (
                backend.artifacts_root,
                f"{backend.artifacts_root}/large_tool_results",
            ),
            (
                "/workspace/.files/deepagent",
                "/workspace/.files/deepagent/large_tool_results",
            ),
        ):
            listed = backend.ls(root)
            globbed = backend.glob("**/*", root)
            grepped = backend.grep("first-secret", root)
            async_globbed = await backend.aglob("**/*", root)
            assert listed.error is None
            assert globbed.error is None
            assert grepped.error is None
            assert async_globbed.error is None
            listed_paths = [item["path"].rstrip("/") for item in listed.entries or []]
            globbed_paths = [item["path"] for item in globbed.matches or []]
            grepped_paths = [item["path"] for item in grepped.matches or []]
            async_globbed_paths = [
                item["path"] for item in async_globbed.matches or []
            ]
            assert logical_root in listed_paths
            assert f"{logical_root}/first.txt" in globbed_paths
            assert f"{logical_root}/first.txt" in grepped_paths
            assert f"{logical_root}/first.txt" in async_globbed_paths
            assert all("second.txt" not in path for path in globbed_paths)
            assert all(
                "session_tool_results" not in path
                for path in [*listed_paths, *globbed_paths, *grepped_paths]
            )
        registry.reset(first_token)
        await registry.close()

    asyncio.run(exercise())


def test_parent_grep_isolates_before_limiting_and_hides_legacy_files(
    tmp_path,
    monkeypatch,
):
    """Foreign and pre-upgrade artifacts cannot consume or enter grep results."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        tokens = iter(("0000000000000000", "ffffffffffffffff"))
        monkeypatch.setattr(
            runtime_artifacts.uuid,
            "uuid4",
            lambda: SimpleNamespace(hex=next(tokens)),
        )
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        foreign = registry.open_session("foreign")
        foreign_token = registry.activate(foreign)
        assert backend.write(
            f"{backend.artifacts_root}/large_tool_results/foreign.txt",
            "shared-needle",
        ).error is None
        registry.reset(foreign_token)
        active = registry.open_session("active")
        active_token = registry.activate(active)
        assert backend.write(
            f"{backend.artifacts_root}/large_tool_results/active.txt",
            "shared-needle",
        ).error is None
        legacy = tmp_path / ".files" / "deepagent" / "large_tool_results" / "legacy.txt"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text("shared-needle", encoding="utf-8")

        result = backend.grep(
            "shared-needle",
            backend.artifacts_root,
            max_count=1,
        )
        globbed = backend.glob("**/*", backend.artifacts_root)

        assert result.error is None
        assert [item["path"] for item in result.matches or []] == [
            f"{backend.artifacts_root}/large_tool_results/active.txt"
        ]
        assert all(
            "legacy.txt" not in item["path"] for item in globbed.matches or []
        )
        registry.reset(active_token)
        await registry.close()

    asyncio.run(exercise())


def test_workspace_alias_uses_the_active_artifact_generation(tmp_path):
    """Workspace logical paths must remain session-scoped and teardown-owned."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        alias = "/workspace/.files/deepagent/large_tool_results/result.txt"
        first = registry.open_session("first")
        first_token = registry.activate(first)
        assert backend.write(alias, "first").error is None
        assert backend.read(alias).file_data["content"] == "first"
        registry.reset(first_token)
        second = registry.open_session("second")
        second_token = registry.activate(second)
        assert backend.read(alias).error is not None
        assert backend.write(alias, "second").error is None
        assert backend.read(alias).file_data["content"] == "second"
        registry.reset(second_token)

        await registry.close_session("first")

        first_physical = (
            tmp_path
            / ".files"
            / "deepagent"
            / "session_tool_results"
            / first.token
        )
        second_physical = (
            tmp_path
            / ".files"
            / "deepagent"
            / "session_tool_results"
            / second.token
            / "result.txt"
        )
        assert not first_physical.exists()
        assert second_physical.read_text(encoding="utf-8") == "second"
        await registry.close()

    asyncio.run(exercise())


def test_generic_runtime_agent_scopes_artifacts_from_invocation_config(
    runtime,
    monkeypatch,
):
    """A cached generic agent derives artifact ownership from each invocation."""

    class Agent:
        def __init__(self, backend):
            self.backend = backend

        async def ainvoke(self, payload, config):
            del config
            path = f"{self.backend.artifacts_root}/large_tool_results/result.txt"
            if "content" in payload:
                return self.backend.write(path, payload["content"])
            return self.backend.read(path)

    monkeypatch.setattr(
        runtime_middleware,
        "create_deep_agent_with_configured_summarization",
        lambda _config, **kwargs: Agent(kwargs["backend"]),
    )

    async def exercise():
        agent = await runtime.get_agent("medium", thread_id=None)
        first_config = {"configurable": {"thread_id": "first"}}
        second_config = {"configurable": {"thread_id": "second"}}
        assert (await agent.ainvoke({"content": "first"}, first_config)).error is None
        assert (await agent.ainvoke({}, second_config)).error is not None
        assert (await agent.ainvoke({"content": "second"}, second_config)).error is None
        first_read = await agent.ainvoke({}, first_config)
        second_read = await agent.ainvoke({}, second_config)
        assert first_read.file_data["content"] == "first"
        assert second_read.file_data["content"] == "second"

        await runtime.close_conversation(thread_id="first")

        assert (await agent.ainvoke({}, first_config)).error is not None
        assert (await agent.ainvoke({}, second_config)).file_data["content"] == "second"
        await runtime.close()

    asyncio.run(exercise())


def test_uploaded_artifacts_register_before_cancellation_propagates(
    tmp_path,
    monkeypatch,
):
    """Sync and cancelled async uploads remain owned by the active session."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        handle = registry.open_session("thread")
        token = registry.activate(handle)
        sync_path = f"{backend.artifacts_root}/large_tool_results/sync-upload"
        async_path = f"{backend.artifacts_root}/large_tool_results/async-upload"
        assert backend.upload_files([(sync_path, b"sync")])[0].error is None

        underlying = backend.backend
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking_upload(files):
            started.set()
            await release.wait()
            return underlying.upload_files(files)

        monkeypatch.setattr(underlying, "aupload_files", blocking_upload)
        upload_task = asyncio.create_task(
            backend.aupload_files([(async_path, b"async")])
        )
        registry.reset(token)
        await started.wait()
        upload_task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await upload_task

        physical_root = (
            tmp_path / ".files" / "deepagent" / "session_tool_results" / handle.token
        )
        assert (physical_root / "sync-upload").is_file()
        assert (physical_root / "async-upload").is_file()

        await registry.close_session("thread")

        assert not (physical_root / "sync-upload").exists()
        assert not (physical_root / "async-upload").exists()
        await registry.close()

    asyncio.run(exercise())


def test_registration_after_conversation_close_is_deleted_immediately(runtime):
    """A late tool result must not survive an already completed close."""

    async def exercise():
        backend = runtime_backends.build_deepagent_backend(
            project_root=runtime.project_root,
            include_memories=False,
            artifact_registry=runtime.large_tool_result_artifacts,
        )
        path = (
            f"{runtime_backends.deepagent_artifacts_route_prefix(runtime.project_root)}"
            "large_tool_results/late-call"
        )
        handle = runtime.large_tool_result_artifacts.open_session("thread")
        await runtime.close_conversation(
            thread_id="thread",
            mcp_session_id="session",
        )
        token = runtime.large_tool_result_artifacts.activate(handle)
        assert backend.write(path, "late").error is None

        assert backend.read(path).error is not None
        runtime.large_tool_result_artifacts.reset(token)
        await runtime.close()

    asyncio.run(exercise())


def test_artifact_cleanup_failure_does_not_skip_other_conversation_teardown(
    runtime,
    monkeypatch,
):
    """A deletion error must surface only after MCP and agent cleanup continue."""

    async def exercise():
        mcp_calls = []

        async def fail_artifact_cleanup(session_id):
            assert session_id == "thread"
            raise RuntimeError("artifact cleanup failed")

        async def close_mcp_session(session_id):
            mcp_calls.append(session_id)

        monkeypatch.setattr(
            runtime.large_tool_result_artifacts,
            "close_session",
            fail_artifact_cleanup,
        )
        monkeypatch.setattr(runtime, "close_mcp_session", close_mcp_session)

        with pytest.raises(RuntimeError, match="artifact cleanup failed"):
            await runtime.close_conversation(
                thread_id="thread",
                mcp_session_id="session",
            )

        assert mcp_calls == ["session"]
        await runtime.close()

    asyncio.run(exercise())


def test_session_cannot_reopen_while_artifact_cleanup_is_running():
    """An overlapping graph lookup must not let a late offload escape cleanup."""

    class Backend:
        def __init__(self) -> None:
            self.files = {"first", "late"}
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def adelete(self, path):
            self.started.set()
            await self.release.wait()
            self.files.discard(path)
            return SimpleNamespace(error=None)

        def delete(self, path):
            self.files.discard(path)
            return SimpleNamespace(error=None)

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        backend = Backend()
        handle = registry.open_session("thread")
        registry.register(handle, "first", backend)
        closing = asyncio.create_task(registry.close_session("thread"))
        await asyncio.wait_for(backend.started.wait(), timeout=1)

        with pytest.raises(RuntimeError, match="closing"):
            registry.open_session("thread")
        late_registration = asyncio.create_task(
            registry.aregister(handle, "late", backend)
        )
        await asyncio.sleep(0)
        backend.release.set()
        await asyncio.gather(closing, late_registration)

        assert backend.files == set()

    asyncio.run(exercise())


def test_cancelled_async_write_finishes_registration_and_cleanup(tmp_path, monkeypatch):
    """Cancellation cannot orphan a filesystem write still running in a worker."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        underlying = backend.backend
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocking_write(path, content):
            started.set()
            await release.wait()
            return underlying.write(path, content)

        monkeypatch.setattr(underlying, "awrite", blocking_write)
        handle = registry.open_session("thread")
        token = registry.activate(handle)
        path = f"{backend.artifacts_root}/large_tool_results/cancelled"
        task = asyncio.create_task(backend.awrite(path, "result"))
        registry.reset(token)
        await started.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert list((tmp_path / ".files" / "deepagent").rglob("cancelled"))
        await registry.close_session("thread")
        assert not list((tmp_path / ".files" / "deepagent").rglob("cancelled"))

    asyncio.run(exercise())


def test_conversation_close_preserves_generated_batch_outputs(
    runtime,
    tmp_path,
):
    """Conversation teardown removes temporary offloads but keeps deliverables."""

    async def exercise():
        registry = runtime.large_tool_result_artifacts
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        output_store = runtime_background_tasks.create_batch_result_output_store(
            backend,
            backend_prefix=runtime_backends.generated_outputs_route_prefix(tmp_path),
        )
        handle = registry.open_session("thread")
        token = registry.activate(handle)
        artifact_path = f"{backend.artifacts_root}/large_tool_results/temporary"
        artifact_result = await backend.awrite(artifact_path, "temporary")
        registry.reset(token)
        assert artifact_result.error is None

        relative_output = "subagent-batches/batch-call/01-researcher-task.md"
        output_result = await output_store.backend.awrite(
            output_store.backend_path(relative_output),
            "# researcher\n",
        )
        assert output_result.error is None
        physical_artifact = (
            tmp_path
            / ".files"
            / "deepagent"
            / "session_tool_results"
            / handle.token
            / "temporary"
        )
        physical_output = tmp_path / ".files" / "outputs" / relative_output
        assert physical_artifact.is_file()
        assert physical_output.is_file()

        await runtime.close_conversation(thread_id="thread")

        assert not physical_artifact.exists()
        assert physical_output.read_text(encoding="utf-8") == "# researcher\n"
        await runtime.close()

    asyncio.run(exercise())


def test_failed_artifact_delete_remains_retryable(tmp_path, monkeypatch):
    """A failed close retains ownership so the next close can retry."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        handle = registry.open_session("thread")
        token = registry.activate(handle)
        path = f"{backend.artifacts_root}/large_tool_results/retry"
        assert backend.write(path, "result").error is None
        registry.reset(token)
        underlying = backend.backend
        original_delete = underlying.adelete
        attempts = 0

        async def flaky_delete(file_path):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return DeleteResult(error="temporary failure")
            return await original_delete(file_path)

        monkeypatch.setattr(underlying, "adelete", flaky_delete)
        with pytest.raises(ExceptionGroup, match="cleanup failed"):
            await registry.close_session("thread")
        await registry.close_session("thread")

        assert attempts == 2
        assert not list((tmp_path / ".files" / "deepagent").rglob("retry"))

    asyncio.run(exercise())


def test_terminal_registry_deletes_late_writes_and_rejects_reopen(tmp_path):
    """Shutdown is terminal even for work retaining an old context handle."""

    async def exercise():
        registry = runtime_artifacts.LargeToolResultArtifactRegistry()
        backend = runtime_backends.build_deepagent_backend(
            project_root=tmp_path,
            include_memories=False,
            artifact_registry=registry,
        )
        handle = registry.open_session("thread")
        await registry.close()
        with pytest.raises(RuntimeError, match="closed"):
            registry.open_session("thread")

        token = registry.activate(handle)
        path = f"{backend.artifacts_root}/large_tool_results/late"
        assert backend.write(path, "result").error is None
        assert backend.read(path).error is not None
        registry.reset(token)

    asyncio.run(exercise())


def test_conversation_close_rejects_spawns_until_resource_teardown_finishes(
    runtime,
    monkeypatch,
):
    """A closing conversation cannot launch jobs while MCP resources close."""
    runtime.background_tasks = BackgroundTaskManager(
        BackgroundSubagentConfig(enabled=True)
    )

    async def exercise():
        mcp_close_started = asyncio.Event()
        allow_mcp_close = asyncio.Event()

        async def close_mcp_session(session_id):
            assert session_id == "mcp"
            mcp_close_started.set()
            await allow_mcp_close.wait()

        async def runner(task_id):
            return task_id

        monkeypatch.setattr(runtime, "close_mcp_session", close_mcp_session)
        close_task = asyncio.create_task(
            runtime.close_conversation(thread_id="thread", mcp_session_id="mcp")
        )
        await asyncio.wait_for(mcp_close_started.wait(), timeout=1)

        with pytest.raises(RuntimeError, match="session is closing"):
            await runtime.background_tasks.spawn(
                session_id="thread",
                agent_name="worker",
                description="late work",
                agent_path=("worker",),
                runner=runner,
            )

        allow_mcp_close.set()
        await asyncio.wait_for(close_task, timeout=1)
        await runtime.background_tasks.close()

    asyncio.run(exercise())


def test_cancelled_conversation_close_finishes_resource_teardown(
    runtime,
    monkeypatch,
):
    """Caller cancellation must not reopen a partially closed conversation."""
    runtime.background_tasks = BackgroundTaskManager(
        BackgroundSubagentConfig(enabled=True)
    )

    async def exercise():
        mcp_close_started = asyncio.Event()
        allow_mcp_close = asyncio.Event()

        async def close_mcp_session(session_id):
            assert session_id == "mcp"
            mcp_close_started.set()
            await allow_mcp_close.wait()

        async def runner(task_id):
            return task_id

        monkeypatch.setattr(runtime, "close_mcp_session", close_mcp_session)
        close_task = asyncio.create_task(
            runtime.close_conversation(thread_id="thread", mcp_session_id="mcp")
        )
        await asyncio.wait_for(mcp_close_started.wait(), timeout=1)
        close_task.cancel()
        await asyncio.sleep(0)

        assert not close_task.done()
        with pytest.raises(RuntimeError, match="session is closing"):
            await runtime.background_tasks.spawn(
                session_id="thread",
                agent_name="worker",
                description="late work",
                agent_path=("worker",),
                runner=runner,
            )

        allow_mcp_close.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(close_task, timeout=1)

        reopened = await runtime.background_tasks.spawn(
            session_id="thread",
            agent_name="worker",
            description="new work",
            agent_path=("worker",),
            runner=runner,
        )
        assert (
            await runtime.background_tasks.get(
                "thread",
                reopened.task_id,
                wait_seconds=1,
            )
        ).status == "success"
        await runtime.background_tasks.close()

    asyncio.run(exercise())


def test_cancelled_conversation_close_finishes_artifact_cleanup(
    runtime,
    monkeypatch,
):
    """Caller cancellation must not interrupt deletion of session offloads."""

    async def exercise():
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()

        async def close_session(session_id):
            assert session_id == "thread"
            cleanup_started.set()
            await allow_cleanup.wait()
            cleanup_finished.set()

        monkeypatch.setattr(
            runtime.large_tool_result_artifacts,
            "close_session",
            close_session,
        )
        close_task = asyncio.create_task(
            runtime.close_conversation(
                thread_id="thread",
                mcp_session_id="session",
            )
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        close_task.cancel()
        await asyncio.sleep(0)
        assert not close_task.done()

        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(close_task, timeout=1)
        assert cleanup_finished.is_set()
        await runtime.close()

    asyncio.run(exercise())


def test_conversation_close_invalidates_existing_background_tools(runtime):
    """A foreground run from before close cannot access a reopened session."""
    runtime.background_tasks = BackgroundTaskManager(
        BackgroundSubagentConfig(enabled=True)
    )

    async def exercise():
        generation = runtime.background_tasks.session_generation("thread")
        tools = create_background_task_tools(
            manager=runtime.background_tasks,
            subagents={"worker": object()},
            agent_path=(),
            recursion_limit=20,
            session_generation=generation,
        )
        spawn_tool = next(
            tool for tool in tools if tool.name == "spawn_background_task"
        )
        batch_tool = next(tool for tool in tools if tool.name == "run_subagent_batch")
        list_tool = next(tool for tool in tools if tool.name == "list_background_tasks")
        get_tool = next(tool for tool in tools if tool.name == "get_background_task")
        cancel_tool = next(
            tool for tool in tools if tool.name == "cancel_background_task"
        )
        tool_runtime = ToolRuntime(
            state={},
            context=None,
            config={"configurable": {"thread_id": "thread"}},
            stream_writer=lambda _: None,
            tool_call_id="stale-spawn",
            store=None,
        )

        await runtime.close_conversation(
            thread_id="thread",
            mcp_session_id="thread",
        )

        with pytest.raises(RuntimeError, match="session was closed"):
            await spawn_tool.coroutine("late work", "worker", tool_runtime)

        reopened = await runtime.background_tasks.spawn(
            session_id="thread",
            agent_name="worker",
            description="new work",
            agent_path=("worker",),
            runner=lambda task_id: asyncio.sleep(0, result=task_id),
        )
        for call in (
            lambda: batch_tool.coroutine(
                [{"description": "late work", "subagent_type": "worker"}],
                tool_runtime,
            ),
            lambda: list_tool.coroutine(tool_runtime),
            lambda: get_tool.coroutine(reopened.task_id, 0, tool_runtime),
            lambda: cancel_tool.coroutine(reopened.task_id, tool_runtime),
        ):
            with pytest.raises(RuntimeError, match="session was closed"):
                await call()
        await runtime.background_tasks.close()

    asyncio.run(exercise())


def test_runtime_close_cancels_background_tasks(tmp_path):
    """Runtime shutdown must not leave local subagent asyncio tasks alive."""
    config = replace(
        make_runtime_config(tmp_path),
        rag=None,
        rag_requested=False,
        extensions=replace(
            make_runtime_config(tmp_path).extensions,
            background_subagents=BackgroundSubagentConfig(enabled=True),
        ),
    )
    instance = core.AgentRuntime(config, project_root=tmp_path)

    async def exercise():
        queue = instance.background_tasks.subscribe("thread")
        cleaned = []

        teardown_events = []

        async def close_persistence_resource():
            teardown_events.append(("persistence", not queue.empty()))

        instance._exit_stack.push_async_callback(close_persistence_resource)

        async def runner(task_id):
            await asyncio.Event().wait()
            return task_id

        async def cleanup(task_id):
            cleaned.append(task_id)

        await instance.background_tasks.spawn(
            session_id="thread",
            agent_name="worker",
            description="work",
            agent_path=("worker",),
            runner=runner,
            cleanup=cleanup,
        )
        await instance.close()

        assert queue.get_nowait().status == "cancelled"
        assert len(cleaned) == 1
        assert teardown_events == [("persistence", True)]

    asyncio.run(exercise())


@pytest.mark.parametrize("factory", ["get", "create"])
@pytest.mark.parametrize("cancel", [False, True])
def test_factory_unwinds_resources_after_failed_or_cancelled_startup(
    runtime, monkeypatch, factory, cancel
):
    closed = []
    entered = asyncio.Event()

    @asynccontextmanager
    async def resource():
        try:
            yield object()
        finally:
            closed.append(True)

    async def initialize(self):
        await self._exit_stack.enter_async_context(resource())
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        raise ValueError("startup failed")

    monkeypatch.setattr(runtime_lifecycle.AgentRuntime, "_instance", None)
    monkeypatch.setattr(runtime_lifecycle.AgentRuntime, "_initialize", initialize)
    monkeypatch.setattr(
        runtime_config.RuntimeConfig, "from_env", lambda: runtime.config
    )

    async def exercise():
        task = asyncio.create_task(getattr(core.AgentRuntime, factory)())
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else ValueError):
            await task
        assert closed == [True]
        assert core.AgentRuntime.current() is None

    asyncio.run(exercise())


def test_stateless_mcp_tools_are_shared_across_ended_chats(runtime):
    runtime.config = replace(
        runtime.config, extensions=make_extensions_config(agent_mcp_servers=("repo",))
    )
    loads = []

    async def get_tools(**kwargs):
        loads.append(kwargs)
        return []

    runtime._mcp_client = SimpleNamespace(get_tools=get_tools)

    async def exercise():
        for index in range(10):
            thread, session = f"thread-{index}", f"session-{index}"
            await runtime.get_agent("medium", thread_id=thread, mcp_session_id=session)
            await runtime.close_conversation(thread_id=thread, mcp_session_id=session)
            assert not runtime._agents
            assert not runtime._mcp_sessions
        assert len(runtime._mcp_tools_cache) == 1
        assert len(loads) == 1
        await runtime.close()
        assert not runtime._mcp_tools_cache

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_mcp_tool_loading_unwinds_new_transport_on_error(runtime, monkeypatch, cancel):
    runtime.config = replace(
        runtime.config,
        extensions=make_extensions_config(
            mcp_stateful=True, agent_mcp_servers=("repo",)
        ),
    )
    closed = []
    entered = asyncio.Event()

    @asynccontextmanager
    async def session(server):
        async with anyio.create_task_group():
            try:
                yield object()
            finally:
                closed.append(True)

    runtime._mcp_client = SimpleNamespace(
        session=session, callbacks=None, tool_interceptors=[]
    )

    async def load(*a, **kw):
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        raise ValueError("tool loading failed")

    monkeypatch.setattr(runtime_lifecycle, "load_mcp_tools", load)

    async def exercise():
        task = asyncio.create_task(
            runtime.get_agent("medium", thread_id="thread", mcp_session_id="session")
        )
        await entered.wait()
        if cancel:
            task.cancel()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert closed == [True]
        assert not runtime._mcp_sessions
        assert not runtime._mcp_tools_cache
        assert not runtime._agents

        async def succeeds(*a, **kw):
            return []

        monkeypatch.setattr(runtime_lifecycle, "load_mcp_tools", succeeds)
        await runtime.get_agent("medium", thread_id="thread", mcp_session_id="session")
        await runtime.close()
        assert closed == [True, True]

    asyncio.run(exercise())


def test_mcp_discovery_keeps_healthy_tools_and_retries_failed_server(runtime):
    runtime.config = replace(
        runtime.config,
        extensions=make_extensions_config(agent_mcp_servers=("broken", "healthy")),
    )
    attempts = []
    failed = True

    async def get_tools(*, server_name):
        nonlocal failed
        attempts.append(server_name)
        if server_name == "broken" and failed:
            raise OSError("server is down")
        return [SimpleNamespace(name=f"{server_name}_tool")]

    runtime._mcp_client = SimpleNamespace(get_tools=get_tools)

    async def exercise():
        nonlocal failed
        first_tools, first_warnings = await runtime.get_mcp_tools_with_status(
            ("broken", "healthy"), thread_id="thread"
        )
        assert [tool.name for tool in first_tools] == ["healthy_tool"]
        assert first_warnings == ("broken",)
        failed = False
        second_tools, second_warnings = await runtime.get_mcp_tools_with_status(
            ("broken", "healthy"), thread_id="thread"
        )
        assert [tool.name for tool in second_tools] == ["broken_tool", "healthy_tool"]
        assert second_warnings == ()
        assert attempts == ["broken", "healthy", "broken"]

    asyncio.run(exercise())


def test_failed_mcp_discovery_retains_owner_when_transport_close_fails(runtime, monkeypatch):
    runtime.config = replace(
        runtime.config,
        extensions=make_extensions_config(
            mcp_stateful=True, agent_mcp_servers=("broken",)
        ),
    )

    @asynccontextmanager
    async def session(server):
        try:
            yield object()
        finally:
            raise RuntimeError("transport close failed")

    async def load(*args, **kwargs):
        raise OSError("tool listing failed")

    runtime._mcp_client = SimpleNamespace(
        session=session, callbacks=None, tool_interceptors=[]
    )
    monkeypatch.setattr(runtime_lifecycle, "load_mcp_tools", load)

    async def exercise():
        tools, failures = await runtime.get_mcp_tools_with_status(
            ("broken",), mcp_session_id="session"
        )
        assert tools == [] and failures == ("broken",)
        assert ("session", "broken") in runtime._mcp_session_owners

    asyncio.run(exercise())


def test_mcp_close_failure_keeps_owner_for_later_retry(runtime):
    runtime.config = replace(
        runtime.config,
        extensions=make_extensions_config(mcp_stateful=True),
    )

    class RetryableOwner:
        attempts = 0

        async def aclose(self):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("close failed")

    owner = RetryableOwner()
    runtime._mcp_session_owners[("session", "repo")] = owner

    async def exercise():
        with pytest.raises(OSError, match="close failed"):
            await runtime.close_mcp_session("session")
        assert runtime._mcp_session_owners[("session", "repo")] is owner
        await runtime.close_mcp_session("session")
        assert runtime._mcp_session_owners == {}
        assert owner.attempts == 2

    asyncio.run(exercise())


def test_mcp_runtime_close_retains_owner_if_transport_teardown_fails(runtime):
    class RetryableOwner:
        attempts = 0

        async def aclose(self):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("close failed")

    owner = RetryableOwner()
    runtime._mcp_session_owners[("session", "repo")] = owner

    async def exercise():
        with pytest.raises(OSError, match="close failed"):
            await runtime.close_all_mcp_sessions()
        assert runtime._mcp_session_owners[("session", "repo")] is owner
        await runtime.close_all_mcp_sessions()
        assert runtime._mcp_session_owners == {}

    asyncio.run(exercise())


def test_agent_build_reports_mcp_failure_and_retries_on_next_run(runtime):
    runtime.config = replace(
        runtime.config,
        extensions=make_extensions_config(agent_mcp_servers=("broken",)),
    )
    attempts = 0

    async def get_tools(*, server_name):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("down")
        return []

    runtime._mcp_client = SimpleNamespace(get_tools=get_tools)

    async def exercise():
        _, warnings = await runtime.get_agent_with_status("medium", thread_id="thread")
        assert warnings == ("broken",)
        assert runtime._agents == {}
        _, warnings = await runtime.get_agent_with_status("medium", thread_id="thread")
        assert warnings == ()
        assert attempts == 2
        assert len(runtime._agents) == 1

    asyncio.run(exercise())


@pytest.mark.parametrize("cancel", [False, True])
def test_mcp_session_startup_unwinds_owner_and_allows_retry(
    runtime, monkeypatch, cancel
):
    runtime.config = replace(
        runtime.config,
        extensions=make_extensions_config(
            mcp_stateful=True, agent_mcp_servers=("repo",)
        ),
    )
    closed = []
    entered = asyncio.Event()
    fail = True

    @asynccontextmanager
    async def session(server):
        async with anyio.create_task_group():
            try:
                entered.set()
                if fail:
                    if cancel:
                        await asyncio.Event().wait()
                    raise ValueError("transport startup failed")
                yield object()
            finally:
                closed.append(True)

    runtime._mcp_client = SimpleNamespace(
        session=session, callbacks=None, tool_interceptors=[]
    )

    async def load(*a, **kw):
        return []

    monkeypatch.setattr(runtime_lifecycle, "load_mcp_tools", load)

    async def exercise():
        nonlocal fail
        task = asyncio.create_task(
            runtime.get_agent("medium", thread_id="thread", mcp_session_id="session")
        )
        await entered.wait()
        if cancel:
            task.cancel()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert closed == [True]
        assert not runtime._mcp_sessions
        assert not runtime._mcp_session_owners
        fail = False
        await runtime.get_agent("medium", thread_id="thread", mcp_session_id="session")
        await runtime.close()
        assert closed == [True, True]

    asyncio.run(exercise())


def test_conversation_close_rebuilds_other_session_on_same_thread(runtime):
    runtime.config = replace(
        runtime.config, extensions=replace(runtime.config.extensions, mcp_stateful=True)
    )

    async def exercise():
        first = await runtime.get_agent(
            "medium", thread_id="thread", mcp_session_id="session"
        )
        other = await runtime.get_agent(
            "medium", thread_id="thread", mcp_session_id="other-session"
        )
        await runtime.close_conversation(thread_id="thread", mcp_session_id="session")
        assert first not in runtime._agents.values()
        assert other not in runtime._agents.values()
        assert (
            await runtime.get_agent(
                "medium", thread_id="thread", mcp_session_id="other-session"
            )
            is not other
        )
        await runtime.close()

    asyncio.run(exercise())
