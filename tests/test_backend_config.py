"""Tests for configurable native DeepAgents backends."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import re
import tomllib
from types import SimpleNamespace

import pytest

from chainagents.runtime.backends import (
    BackendMetadata,
    ContextHubBackendConfig,
    DeepAgentsBackendConfig,
    FilesystemBackendConfig,
    LangSmithSandboxBackendConfig,
    LocalShellBackendConfig,
    StateBackendConfig,
    StoreBackendConfig,
    build_backend_bundle,
    parse_backend_config,
)
from deepagents.backends import (
    CompositeBackend,
    ContextHubBackend,
    FilesystemBackend,
    LangSmithSandbox,
    LocalShellBackend,
    StateBackend,
    StoreBackend,
)
from langgraph.store.memory import InMemoryStore


def _parse(
    backend: dict[str, object] | None,
    tmp_path: Path,
    *,
    agent_state: str = "stateful",
    execute_tool_enabled: bool = False,
):
    raw_config = {} if backend is None else {"backend": backend}
    return parse_backend_config(
        raw_config,
        tmp_path / "deepagent.toml",
        agent_state=agent_state,
        execute_tool_enabled=execute_tool_enabled,
    )


def test_omitted_backend_config_preserves_legacy_mode(tmp_path: Path) -> None:
    assert _parse(None, tmp_path) is None


@pytest.mark.parametrize(
    ("raw_backend", "expected_type"),
    [
        ({"type": "state"}, StateBackendConfig),
        ({"type": "store", "namespace": ["agents", "shared"]}, StoreBackendConfig),
        (
            {"type": "filesystem", "root_dir": "workspace", "virtual_mode": True},
            FilesystemBackendConfig,
        ),
        (
            {
                "type": "local_shell",
                "root_dir": "workspace",
                "virtual_mode": True,
                "allow_unrestricted_host_execution": True,
            },
            LocalShellBackendConfig,
        ),
        (
            {"type": "context_hub", "identifier": "owner/agent-repo"},
            ContextHubBackendConfig,
        ),
        (
            {"type": "langsmith_sandbox", "sandbox_name": "existing-sandbox"},
            LangSmithSandboxBackendConfig,
        ),
    ],
)
def test_parses_every_native_backend_type(
    raw_backend: dict[str, object],
    expected_type: type[object],
    tmp_path: Path,
) -> None:
    if raw_backend["type"] == "local_shell":
        config = _parse(raw_backend, tmp_path, execute_tool_enabled=True)
    else:
        config = _parse(raw_backend, tmp_path)

    assert config is not None
    assert isinstance(config.default, expected_type)
    assert config.artifacts_root == "/workspace/.files/deepagent"


def test_resolves_local_backend_paths_relative_to_config(tmp_path: Path) -> None:
    config = _parse(
        {"type": "filesystem", "root_dir": "../shared", "virtual_mode": True},
        tmp_path / "config",
    )

    assert config is not None
    assert isinstance(config.default, FilesystemBackendConfig)
    assert config.default.root_dir == (tmp_path / "shared").resolve()


def test_normalizes_and_preserves_explicit_routes(tmp_path: Path) -> None:
    config = _parse(
        {
            "type": "state",
            "routes": [
                {
                    "path": "/reference",
                    "type": "context_hub",
                    "identifier": "owner/agent-repo",
                },
                {"path": "/workspace", "type": "state"},
            ],
        },
        tmp_path,
    )

    assert config is not None
    assert [route.path for route in config.routes] == ["/reference/", "/workspace/"]
    assert isinstance(config.routes[0].backend, ContextHubBackendConfig)


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        ({"type": "unknown"}, "unknown backend type"),
        ({"type": "state", "root_dir": "."}, "root_dir"),
        ({"type": "store", "namespace": []}, "namespace"),
        ({"type": "filesystem", "root_dir": ".", "virtual_mode": "yes"}, "virtual_mode"),
        ({"type": "context_hub", "identifier": ""}, "identifier"),
        ({"type": "langsmith_sandbox", "sandbox_name": "",}, "sandbox_name"),
        (
            {"type": "langsmith_sandbox", "sandbox_name": "sandbox", "client_timeout": 0},
            "client_timeout",
        ),
        (
            {"type": "langsmith_sandbox", "sandbox_name": "sandbox", "max_retries": -1},
            "max_retries",
        ),
        ({"type": "state", "artifacts_root": "relative"}, "artifacts_root"),
    ],
)
def test_rejects_invalid_backend_values(
    backend: dict[str, object],
    message: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=message):
        _parse(backend, tmp_path)


@pytest.mark.parametrize("client_timeout", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_langsmith_client_timeout(
    client_timeout: float,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="client_timeout"):
        _parse(
            {
                "type": "langsmith_sandbox",
                "sandbox_name": "sandbox",
                "client_timeout": client_timeout,
            },
            tmp_path,
        )


@pytest.mark.parametrize("segment", ["*", "?", "with/slash", "with space", "with.dot"])
def test_rejects_store_namespace_segments_invalid_for_native_stores(
    segment: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="namespace"):
        _parse({"type": "store", "namespace": ["agents", segment]}, tmp_path)


@pytest.mark.parametrize("path", ["/", "relative", "/a/../b", "/a/./b", "//server/share"])
def test_rejects_invalid_route_paths(path: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="route path"):
        _parse(
            {
                "type": "state",
                "routes": [{"path": path, "type": "state"}],
            },
            tmp_path,
        )


def test_rejects_duplicate_normalized_routes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate backend route"):
        _parse(
            {
                "type": "state",
                "routes": [
                    {"path": "/reference", "type": "state"},
                    {"path": "/reference/", "type": "state"},
                ],
            },
            tmp_path,
        )


def test_rejects_state_backends_inside_generated_output_route(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="generated-output route"):
        _parse(
            {
                "type": "state",
                "routes": [
                    {
                        "path": "/workspace/.files/outputs/private/",
                        "type": "state",
                    }
                ],
            },
            tmp_path,
        )


@pytest.mark.parametrize("backend_type", ["filesystem", "local_shell"])
def test_rejects_nonvirtual_host_backends_inside_generated_output_route(
    backend_type: str,
    tmp_path: Path,
) -> None:
    route: dict[str, object] = {
        "path": "/workspace/.files/outputs/private/",
        "type": backend_type,
        "root_dir": ".",
        "virtual_mode": False,
    }
    if backend_type == "filesystem":
        route["allow_unrestricted_host_filesystem"] = True
    else:
        route["allow_unrestricted_host_execution"] = True

    with pytest.raises(ValueError, match="generated-output route"):
        _parse(
            {"type": "state", "routes": [route]},
            tmp_path,
        )


@pytest.mark.parametrize(
    "artifacts_root",
    [
        "/workspace/.files/outputs",
        "/workspace/.files/outputs/internal",
    ],
)
def test_rejects_artifacts_roots_inside_generated_outputs(
    artifacts_root: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="artifacts_root"):
        _parse(
            {"type": "state", "artifacts_root": artifacts_root},
            tmp_path,
        )


@pytest.mark.parametrize(
    "backend",
    [
        {"type": "store", "namespace": ["agent"]},
        {
                "type": "state",
                "routes": [
                    {"path": "/reference/", "type": "store", "namespace": ["agent"]},
            ],
        },
        {
            "type": "state",
            "routes": [
                {"path": "/memories/", "type": "state"},
            ],
        },
    ],
)
def test_rejects_store_or_memory_overrides_in_stateless_mode(
    backend: dict[str, object],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="stateless"):
        _parse(backend, tmp_path, agent_state="stateless")


def test_requires_unrestricted_filesystem_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow_unrestricted_host_filesystem"):
        _parse(
            {"type": "filesystem", "root_dir": ".", "virtual_mode": False},
            tmp_path,
        )


@pytest.mark.parametrize("execute_tool_enabled", [False, True])
def test_requires_unrestricted_execution_acknowledgement(
    execute_tool_enabled: bool,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="allow_unrestricted_host_execution"):
        _parse(
            {"type": "local_shell", "root_dir": "."},
            tmp_path,
            execute_tool_enabled=execute_tool_enabled,
        )


def test_default_local_shell_requires_execute_tool(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="execute_tool_enabled"):
        _parse(
            {
                "type": "local_shell",
                "root_dir": ".",
                "allow_unrestricted_host_execution": True,
            },
            tmp_path,
        )


def test_routed_local_shell_does_not_require_execute_tool(tmp_path: Path) -> None:
    config = _parse(
        {
            "type": "state",
            "routes": [
                {
                    "path": "/build/",
                    "type": "local_shell",
                    "root_dir": ".",
                    "allow_unrestricted_host_execution": True,
                }
            ],
        },
        tmp_path,
    )

    assert config is not None
    assert isinstance(config.routes[0].backend, LocalShellBackendConfig)


def test_accepts_full_local_shell_configuration(tmp_path: Path) -> None:
    config = _parse(
        {
            "type": "local_shell",
            "root_dir": ".",
            "virtual_mode": True,
            "timeout": 45,
            "max_output_bytes": 4096,
            "max_file_size_mb": 4,
            "env": {"PATH": "/usr/bin:/bin"},
            "inherit_env": False,
            "allow_unrestricted_host_execution": True,
        },
        tmp_path,
        execute_tool_enabled=True,
    )

    assert config is not None
    assert isinstance(config.default, LocalShellBackendConfig)
    assert config.default.timeout == 45
    assert config.default.max_output_bytes == 4096
    assert config.default.max_file_size_mb == 4
    assert config.default.env == {"PATH": "/usr/bin:/bin"}
    assert config.default.inherit_env is False


def test_builds_native_nodes_and_replaces_managed_routes(tmp_path: Path) -> None:
    store = InMemoryStore()
    config = DeepAgentsBackendConfig(
        default=StoreBackendConfig(namespace=("default",)),
        routes=(
            # Replaces the managed local workspace route.
            _route("/workspace/", StateBackendConfig()),
            _route(
                "/reference/",
                FilesystemBackendConfig(root_dir=tmp_path / "reference"),
            ),
        ),
    )

    bundle = build_backend_bundle(
        config,
        project_root=tmp_path,
        include_memories=True,
        memory_namespace="memory",
        store=store,
    )

    assert isinstance(bundle.backend.default, StoreBackend)
    assert bundle.backend.default._store is store  # noqa: SLF001
    assert isinstance(bundle.backend.routes["/workspace/"], StateBackend)
    assert isinstance(bundle.backend.routes["/reference/"], FilesystemBackend)
    assert bundle.backend.routes["/reference/"].cwd == (tmp_path / "reference").resolve()
    assert bundle.backend.routes["/memories/"]._store is store  # noqa: SLF001
    assert bundle.backend.artifacts_root == "/workspace/.files/deepagent"
    assert bundle.metadata.workspace_local is False


def _route(path: str, backend: object):
    from chainagents.runtime.backends import BackendRouteConfig

    return BackendRouteConfig(path=path, backend=backend)


def test_builds_local_shell_with_all_configured_limits(tmp_path: Path) -> None:
    config = DeepAgentsBackendConfig(
        default=LocalShellBackendConfig(
            root_dir=tmp_path,
            virtual_mode=False,
            max_file_size_mb=7,
            timeout=13,
            max_output_bytes=2048,
            env={"PATH": "/bin"},
            inherit_env=False,
        ),
        execute_tool_enabled=True,
    )

    bundle = build_backend_bundle(
        config,
        project_root=tmp_path,
        include_memories=False,
        memory_namespace="unused",
    )

    backend = bundle.backend.default
    assert isinstance(backend, LocalShellBackend)
    assert backend.cwd == tmp_path.resolve()
    assert backend.virtual_mode is False
    assert backend.max_file_size_bytes == 7 * 1024 * 1024
    assert backend._default_timeout == 13  # noqa: SLF001
    assert backend._max_output_bytes == 2048  # noqa: SLF001
    assert backend._env == {"PATH": "/bin"}  # noqa: SLF001
    assert bundle.metadata.execution_capable is True
    assert bundle.metadata.execution_environment == "host"


def test_builds_context_hub_route_without_exposing_identifier_in_metadata(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    import chainagents.runtime.backends as backend_module

    client_kwargs: dict[str, object] = {}
    close_count = 0

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            client_kwargs.update(kwargs)

        def close(self) -> None:
            nonlocal close_count
            close_count += 1

    class FakeContextHubBackend(StateBackend):
        def __init__(self, identifier: str, *, client: object) -> None:
            self.identifier = identifier
            self.client = client

    monkeypatch.setattr(backend_module, "ContextHubBackend", FakeContextHubBackend)
    monkeypatch.setenv("LANGSMITH_API_KEY", "context-secret")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://context.example.test")
    config = DeepAgentsBackendConfig(
        default=StateBackendConfig(),
        routes=(
            _route(
                "/reference/",
                ContextHubBackendConfig(identifier="private/repository"),
            ),
        ),
    )

    bundle = build_backend_bundle(
        config,
        project_root=tmp_path,
        include_memories=False,
        memory_namespace="unused",
        context_client_factory=FakeClient,
    )
    with caplog.at_level(logging.WARNING, logger="deepagents.backends.context_hub"):
        logging.getLogger("deepagents.backends.context_hub").warning(
            "Hub pull failed for %r",
            "private/repository",
        )
    asyncio.run(bundle.close())
    asyncio.run(bundle.close())

    assert bundle.backend.routes["/reference/"].identifier == "private/repository"
    assert bundle.metadata.to_status()["routes"][-1] == {
        "path": "/reference/",
        "type": "context_hub",
    }
    assert "private" not in repr(bundle.metadata.to_status())
    assert "private/repository" not in caplog.text
    assert "[redacted]" in caplog.text
    assert client_kwargs == {
        "api_url": "https://context.example.test",
        "api_key": "context-secret",
    }
    assert close_count == 1


def test_attaches_existing_sandbox_and_closes_owned_resources_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import chainagents.runtime.backends as backend_module

    events: list[str] = []
    factory_kwargs: dict[str, object] = {}

    class FakeSandboxBackend(StateBackend):
        def __init__(self, sandbox: object) -> None:
            self.sandbox = sandbox

        async def aclose(self) -> None:
            events.append("backend")

        def execute(self, command: str):
            return command

        async def aexecute(self, command: str):
            return command

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            factory_kwargs.update(kwargs)

        def get_sandbox(self, name: str) -> object:
            events.append(f"attach:{name}")
            return object()

        def close(self) -> None:
            events.append("client")

    monkeypatch.setattr(backend_module, "LangSmithSandbox", FakeSandboxBackend)
    monkeypatch.setenv("LANGSMITH_API_KEY", "secret-key")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://langsmith.example.test")
    config = DeepAgentsBackendConfig(
        default=LangSmithSandboxBackendConfig(
            sandbox_name="existing-sandbox",
            client_timeout=7.5,
            max_retries=2,
        ),
        execute_tool_enabled=True,
    )

    bundle = build_backend_bundle(
        config,
        project_root=tmp_path,
        include_memories=False,
        memory_namespace="unused",
        sandbox_client_factory=FakeClient,
    )
    assert bundle.metadata.execution_capable is True
    assert bundle.metadata.execution_environment == "sandbox"
    asyncio.run(bundle.close())
    asyncio.run(bundle.close())

    assert events == ["attach:existing-sandbox", "backend", "client"]
    assert factory_kwargs == {
        "api_endpoint": "https://langsmith.example.test",
        "timeout": 7.5,
        "api_key": "secret-key",
        "max_retries": 2,
    }


def test_sandbox_execution_is_disabled_when_execute_tool_is_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import chainagents.runtime.backends as backend_module
    from chainagents.runtime.core import SYSTEM_PROMPT, system_prompt_for_backend

    class FakeSandboxBackend(StateBackend):
        def __init__(self, sandbox: object) -> None:
            pass

        def execute(self, command: str):
            return command

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def get_sandbox(self, name: str) -> object:
            return object()

        def close(self) -> None:
            pass

    monkeypatch.setattr(backend_module, "LangSmithSandbox", FakeSandboxBackend)
    bundle = build_backend_bundle(
        DeepAgentsBackendConfig(
            default=LangSmithSandboxBackendConfig(sandbox_name="existing-sandbox")
        ),
        project_root=tmp_path,
        include_memories=False,
        memory_namespace="unused",
        sandbox_client_factory=FakeClient,
    )

    prompt = system_prompt_for_backend(
        SYSTEM_PROMPT,
        bundle.metadata,
        project_root=tmp_path,
    )

    assert bundle.metadata.execution_capable is False
    assert bundle.metadata.execution_environment is None
    assert "Sandbox command execution is enabled" not in prompt
    assert "You do not have host shell execution" in prompt
    asyncio.run(bundle.close())


def test_sandbox_attachment_failure_is_sanitized_and_closes_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    close_count = 0

    class FailingClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def get_sandbox(self, name: str) -> object:
            raise RuntimeError(f"sandbox {name} rejected secret-token")

        def close(self) -> None:
            nonlocal close_count
            close_count += 1

    monkeypatch.setenv("LANGSMITH_API_KEY", "secret-token")
    config = DeepAgentsBackendConfig(
        default=LangSmithSandboxBackendConfig(sandbox_name="private-sandbox")
    )

    with pytest.raises(RuntimeError) as exc_info:
        build_backend_bundle(
            config,
            project_root=tmp_path,
            include_memories=False,
            memory_namespace="unused",
            sandbox_client_factory=FailingClient,
        )

    assert str(exc_info.value) == "Failed to attach the configured LangSmith sandbox."
    assert "private-sandbox" not in str(exc_info.value)
    assert "secret-token" not in str(exc_info.value)
    assert close_count == 1


def test_backend_build_rolls_back_resources_when_a_later_route_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import chainagents.runtime.backends as backend_module

    logger = logging.getLogger("deepagents.backends.context_hub")
    initial_filters = tuple(logger.filters)
    close_count = 0

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def close(self) -> None:
            nonlocal close_count
            close_count += 1

    class FakeContextHubBackend(StateBackend):
        def __init__(self, identifier: str, *, client: object) -> None:
            pass

    class FailingFilesystemBackend:
        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError("constructor rejected a private path")

    monkeypatch.setattr(backend_module, "ContextHubBackend", FakeContextHubBackend)
    monkeypatch.setattr(backend_module, "FilesystemBackend", FailingFilesystemBackend)
    config = DeepAgentsBackendConfig(
        default=ContextHubBackendConfig(identifier="private/repository")
    )

    with pytest.raises(RuntimeError, match="Failed to configure the DeepAgents backend"):
        build_backend_bundle(
            config,
            project_root=tmp_path,
            include_memories=False,
            memory_namespace="unused",
            context_client_factory=FakeClient,
        )

    assert close_count == 1
    assert tuple(logger.filters) == initial_filters


def test_backend_metadata_status_contains_only_safe_capabilities() -> None:
    metadata = BackendMetadata(
        default_type="langsmith_sandbox",
        routes=(),
        execution_capable=False,
        workspace_local=False,
    )

    assert metadata.to_status() == {
        "default_type": "langsmith_sandbox",
        "routes": [],
        "execution_capable": False,
        "workspace_local": False,
    }


def test_file_and_runtime_config_include_backend_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from chainagents.runtime.core import RuntimeConfig, RuntimeConfigOverrides, load_file_config

    config_path = tmp_path / "deepagent.toml"
    config_path.write_text(
        """
[agent]
execute_tool_enabled = true

[backend]
type = "local_shell"
root_dir = "workspace"
allow_unrestricted_host_execution = true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)

    file_config = load_file_config(config_path)
    runtime_config = RuntimeConfig.from_env(RuntimeConfigOverrides(config_path=config_path))

    assert isinstance(file_config.backend, DeepAgentsBackendConfig)
    assert isinstance(runtime_config.backend, DeepAgentsBackendConfig)
    assert isinstance(runtime_config.backend.default, LocalShellBackendConfig)
    assert runtime_config.backend.default.root_dir == (tmp_path / "workspace").resolve()


def test_missing_config_keeps_backend_unconfigured(tmp_path: Path) -> None:
    from chainagents.runtime.core import load_file_config

    assert load_file_config(tmp_path / "missing.toml").backend is None


def test_compatibility_backend_builder_preserves_legacy_routes(tmp_path: Path) -> None:
    from chainagents.runtime.core import build_deepagent_backend

    backend = build_deepagent_backend(
        project_root=tmp_path,
        include_memories=True,
        memory_namespace="agent",
    )

    artifacts_root = (tmp_path / ".files" / "deepagent").resolve()
    outputs_root = (tmp_path / ".files" / "outputs").resolve()
    assert backend.artifacts_root == artifacts_root.as_posix()
    assert f"{artifacts_root.as_posix()}/" in backend.routes
    assert f"{outputs_root.as_posix()}/" in backend.routes
    assert backend.routes["/workspace/"].cwd == tmp_path.resolve()
    assert backend.routes["/memories/"]._namespace(None) == ("agent",)  # noqa: SLF001


def test_compatibility_backend_builder_accepts_explicit_config(tmp_path: Path) -> None:
    from chainagents.runtime.core import build_deepagent_backend

    backend = build_deepagent_backend(
        project_root=tmp_path,
        include_memories=False,
        backend_config=DeepAgentsBackendConfig(default=StateBackendConfig()),
    )

    assert isinstance(backend.default, StateBackend)
    assert backend.artifacts_root == "/workspace/.files/deepagent"
    assert callable(backend.aclose_chainagents_resources)
    asyncio.run(backend.aclose_chainagents_resources())


def _runtime_config(
    tmp_path: Path,
    *,
    backend: DeepAgentsBackendConfig | None,
    with_subagent: bool = False,
):
    from chainagents.runtime.core import ExtensionsConfig, RuntimeConfig, SubagentConfig

    subagents = (
        (
            SubagentConfig(
                name="researcher",
                description="Research",
                system_prompt="Research carefully.",
            ),
        )
        if with_subagent
        else ()
    )
    return RuntimeConfig(
        database_url=None,
        model_provider="ollama",
        model_name="test-model",
        model_choices=("test-model",),
        model_base_url="http://127.0.0.1:11434",
        model_api_key=None,
        model_temperature=0.0,
        default_reasoning="medium",
        persistence_mode="memory",
        extensions=ExtensionsConfig(
            config_path=tmp_path / "deepagent.toml",
            subagents=subagents,
        ),
        backend=backend,
    )


def test_runtime_shares_one_backend_across_agents_subagents_and_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import chainagents.runtime.core as core

    catalog_backends: list[object] = []
    catalog_ran_off_event_loop: list[bool] = []
    middleware_backends: list[object] = []
    agent_kwargs: list[dict[str, object]] = []

    def fake_catalog(extensions, *, backend=None, project_root=None):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            catalog_ran_off_event_loop.append(True)
        catalog_backends.append(backend)
        return (), ()

    def fake_middleware(*, backend, **kwargs):
        middleware_backends.append(backend)
        return []

    def fake_create(config, **kwargs):
        agent_kwargs.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.setattr(core, "build_chainlit_command_catalog", fake_catalog)
    monkeypatch.setattr(core, "build_agent_middleware", fake_middleware)
    monkeypatch.setattr(core, "create_deep_agent_with_configured_summarization", fake_create)
    config = _runtime_config(
        tmp_path,
        backend=DeepAgentsBackendConfig(default=StateBackendConfig()),
        with_subagent=True,
    )

    async def exercise() -> None:
        runtime = await core.AgentRuntime.create(config, project_root=tmp_path)
        try:
            await runtime.get_agent("low")
            await runtime.get_agent("high")
            assert catalog_backends == [runtime.backend]
            assert catalog_ran_off_event_loop == [True]
            assert agent_kwargs
            assert all(kwargs["backend"] is runtime.backend for kwargs in agent_kwargs)
            assert middleware_backends
            assert all(backend is runtime.backend for backend in middleware_backends)
            assert runtime.backend_metadata.default_type == "state"
        finally:
            await runtime.close()

    asyncio.run(exercise())


def test_runtime_closes_backend_bundle_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import chainagents.runtime.core as core

    close_count = 0

    class FakeBundle:
        backend = object()
        metadata = BackendMetadata(
            default_type="state",
            routes=(),
            execution_capable=False,
            workspace_local=False,
        )

        async def close(self) -> None:
            nonlocal close_count
            close_count += 1

    monkeypatch.setattr(core, "build_runtime_backend_bundle", lambda **kwargs: FakeBundle())
    monkeypatch.setattr(
        core,
        "build_chainlit_command_catalog",
        lambda extensions, *, backend=None, project_root=None: ((), ()),
    )
    config = _runtime_config(tmp_path, backend=DeepAgentsBackendConfig(default=StateBackendConfig()))

    async def exercise() -> None:
        runtime = await core.AgentRuntime.create(config, project_root=tmp_path)
        await runtime.close()
        await runtime.close()

    asyncio.run(exercise())
    assert close_count == 1


def test_static_graph_factory_accepts_a_shared_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import chainagents.runtime.core as core

    captured: list[dict[str, object]] = []
    config = _runtime_config(
        tmp_path,
        backend=DeepAgentsBackendConfig(default=StateBackendConfig()),
    )
    shared_backend = build_backend_bundle(
        config.backend,
        project_root=tmp_path,
        include_memories=True,
        memory_namespace="memory",
    ).backend
    monkeypatch.setattr(
        core,
        "create_deep_agent_with_configured_summarization",
        lambda runtime_config, **kwargs: captured.append(kwargs) or object(),
    )
    monkeypatch.setattr(core, "build_agent_middleware", lambda **kwargs: [])

    core.create_configured_graph(
        include_async_subagents=False,
        config=config,
        backend=shared_backend,
    )
    core.create_configured_graph(
        include_async_subagents=True,
        config=config,
        backend=shared_backend,
    )

    assert len(captured) == 2
    assert all(kwargs["backend"] is shared_backend for kwargs in captured)


def test_static_graph_factory_exposes_cleanup_for_an_owned_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import chainagents.runtime.core as core

    close_count = 0
    backend = CompositeBackend(default=StateBackend(), routes={})

    class FakeBundle:
        metadata = BackendMetadata(
            default_type="state",
            routes=(),
            execution_capable=False,
            workspace_local=False,
        )

        def __init__(self) -> None:
            self.backend = backend

        async def close(self) -> None:
            nonlocal close_count
            close_count += 1

    monkeypatch.setattr(core, "build_runtime_backend_bundle", lambda **kwargs: FakeBundle())
    monkeypatch.setattr(
        core,
        "create_deep_agent_with_configured_summarization",
        lambda config, **kwargs: SimpleNamespace(kwargs=kwargs),
    )

    graph = core.create_configured_graph(
        include_async_subagents=False,
        config=_runtime_config(tmp_path, backend=DeepAgentsBackendConfig(default=StateBackendConfig())),
    )

    assert graph._chainagents_backend_bundle.backend is backend
    asyncio.run(graph.aclose_chainagents_resources())
    assert close_count == 1


def test_static_graph_factory_closes_owned_backend_when_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import chainagents.runtime.core as core

    close_count = 0

    class FakeBundle:
        backend = CompositeBackend(default=StateBackend(), routes={})
        metadata = BackendMetadata(
            default_type="state",
            routes=(),
            execution_capable=False,
            workspace_local=False,
        )

        async def close(self) -> None:
            nonlocal close_count
            close_count += 1

    monkeypatch.setattr(core, "build_runtime_backend_bundle", lambda **kwargs: FakeBundle())
    monkeypatch.setattr(
        core,
        "create_deep_agent_with_configured_summarization",
        lambda config, **kwargs: (_ for _ in ()).throw(RuntimeError("graph failed")),
    )

    with pytest.raises(RuntimeError, match="graph failed"):
        core.create_configured_graph(
            include_async_subagents=False,
            config=_runtime_config(
                tmp_path,
                backend=DeepAgentsBackendConfig(default=StateBackendConfig()),
            ),
        )

    assert close_count == 1


def test_runtime_create_closes_partial_startup_when_command_discovery_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import chainagents.runtime.core as core

    close_count = 0

    class FakeBundle:
        backend = CompositeBackend(default=StateBackend(), routes={})
        metadata = BackendMetadata(
            default_type="state",
            routes=(),
            execution_capable=False,
            workspace_local=False,
        )

        async def close(self) -> None:
            nonlocal close_count
            close_count += 1

    monkeypatch.setattr(core, "build_runtime_backend_bundle", lambda **kwargs: FakeBundle())
    monkeypatch.setattr(
        core,
        "build_chainlit_command_catalog",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("skills failed")),
    )

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="skills failed"):
            await core.AgentRuntime.create(
                _runtime_config(
                    tmp_path,
                    backend=DeepAgentsBackendConfig(default=StateBackendConfig()),
                ),
                project_root=tmp_path,
            )

    asyncio.run(exercise())
    assert close_count == 1


def test_langgraph_server_registers_same_loop_backend_lifespan() -> None:
    config_path = Path(__file__).parents[1] / "langgraph.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["http"]["app"] == "./chainagents/langgraph/webapp.py:app"


def test_system_prompt_reflects_remote_workspace_and_execution_capability(
    tmp_path: Path,
) -> None:
    from chainagents.runtime.core import SYSTEM_PROMPT, system_prompt_for_backend

    prompt = system_prompt_for_backend(
        SYSTEM_PROMPT,
        BackendMetadata(
            default_type="local_shell",
            routes=(),
            execution_capable=True,
            workspace_local=False,
            execution_environment="host",
        ),
        project_root=tmp_path,
    )

    assert str(tmp_path) not in prompt
    assert "`/workspace/` is provided by the configured backend" in prompt
    assert "Host shell execution is enabled" in prompt


def test_system_prompt_appends_backend_contract_to_arbitrary_prompts(tmp_path: Path) -> None:
    from chainagents.runtime.core import system_prompt_for_backend

    workspace_root = tmp_path / "relocated-workspace"
    outputs_root = tmp_path / "remote-outputs"
    prompt = system_prompt_for_backend(
        "Research carefully and cite sources.",
        BackendMetadata(
            default_type="langsmith_sandbox",
            routes=(),
            execution_capable=True,
            workspace_local=True,
            workspace_root=workspace_root,
            outputs_local=True,
            outputs_root=outputs_root,
            execution_environment="sandbox",
        ),
        project_root=tmp_path,
    )

    assert prompt.startswith("Research carefully and cite sources.")
    assert f"`/workspace/` for real project files. This route maps to `{workspace_root}`" in prompt
    assert f"maps to `{outputs_root}`" in prompt
    assert "Sandbox command execution is enabled" in prompt
    assert "Host shell execution" not in prompt


def test_remote_workspace_skill_commands_keep_virtual_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chainagents.runtime.core as core

    backend = core.CompositeBackend(
        default=StateBackend(),
        routes={"/workspace/": StateBackend()},
        artifacts_root="/workspace/.files/deepagent",
    )
    extensions = core.ExtensionsConfig(
        config_path=None,
        skills=("/workspace/skills/",),
    )
    monkeypatch.setattr(
        core,
        "_list_skills",
        lambda backend, source_path: [
            {
                "name": "remote-review",
                "description": "Review remote files",
                "path": "/workspace/skills/remote-review/SKILL.md",
            }
        ],
    )

    commands, notes = core.build_chainlit_command_catalog(
        extensions,
        backend=backend,
    )

    assert notes == ()
    assert commands[0].value == "/workspace/skills/remote-review/SKILL.md"


def test_child_remote_skill_commands_keep_virtual_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import chainagents.runtime.core as core

    backend = core.CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
            "/workspace/skills/remote/": StateBackend(),
        },
        artifacts_root="/workspace/.files/deepagent",
    )
    extensions = core.ExtensionsConfig(
        config_path=None,
        skills=("/workspace/skills/remote/",),
    )
    monkeypatch.setattr(
        core,
        "_list_skills",
        lambda backend, source_path: [
            {
                "name": "remote-review",
                "description": "Review remote files",
                "path": "/workspace/skills/remote/review/SKILL.md",
            }
        ],
    )

    commands, _notes = core.build_chainlit_command_catalog(
        extensions,
        backend=backend,
    )

    assert commands[0].value == "/workspace/skills/remote/review/SKILL.md"


def test_tool_resilience_keeps_remote_workspace_paths_virtual(tmp_path: Path) -> None:
    import chainagents.runtime.core as core

    remote_backend = core.CompositeBackend(
        default=StateBackend(),
        routes={"/workspace/": StateBackend()},
        artifacts_root="/workspace/.files/deepagent",
    )
    local_workspace = tmp_path / "configured-workspace"
    local_backend = core.CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(
                root_dir=local_workspace,
                virtual_mode=True,
            )
        },
        artifacts_root="/workspace/.files/deepagent",
    )
    config = _runtime_config(tmp_path, backend=None)
    remote_middleware = core.build_agent_middleware(
        backend=remote_backend,
        config=config,
    )[-1]
    local_middleware = core.build_agent_middleware(
        backend=local_backend,
        config=config,
    )[-1]
    remote_request = SimpleNamespace(
        tool_call={"name": "external_tool", "args": {"path": "/workspace/file.txt"}}
    )
    local_request = SimpleNamespace(
        tool_call={"name": "external_tool", "args": {"path": "/workspace/file.txt"}}
    )

    remote_middleware._map_workspace_path_args(remote_request)  # noqa: SLF001
    local_middleware._map_workspace_path_args(local_request)  # noqa: SLF001

    assert remote_request.tool_call["args"]["path"] == "/workspace/file.txt"
    assert local_request.tool_call["args"]["path"] == str(local_workspace / "file.txt")


def test_tool_resilience_preserves_native_and_child_remote_routes(tmp_path: Path) -> None:
    import chainagents.runtime.core as core

    local_workspace = tmp_path / "workspace"
    backend = core.CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(
                root_dir=local_workspace,
                virtual_mode=True,
            ),
            "/workspace/reference/": StateBackend(),
        },
        artifacts_root="/workspace/.files/deepagent",
    )
    middleware = core.build_agent_middleware(
        backend=backend,
        config=_runtime_config(tmp_path, backend=None),
    )[-1]
    native_request = SimpleNamespace(
        tool_call={"name": "ls", "args": {"path": "/workspace/local/"}}
    )
    child_route_request = SimpleNamespace(
        tool_call={"name": "external_tool", "args": {"path": "/workspace/reference/"}}
    )

    middleware._map_workspace_path_args(native_request)  # noqa: SLF001
    middleware._map_workspace_path_args(child_route_request)  # noqa: SLF001

    assert native_request.tool_call["args"]["path"] == "/workspace/local/"
    assert child_route_request.tool_call["args"]["path"] == "/workspace/reference/"


def test_backend_documentation_examples_are_valid_toml() -> None:
    docs_path = Path("backends.md")
    blocks = re.findall(
        r"```toml\n(.*?)```",
        docs_path.read_text(encoding="utf-8"),
        flags=re.DOTALL,
    )
    parsed_blocks = [tomllib.loads(block) for block in blocks]
    backend_blocks = [block for block in parsed_blocks if "backend" in block]
    documented_types = {
        str(node["type"])
        for block in backend_blocks
        for node in [block["backend"], *block["backend"].get("routes", [])]
    }

    assert len(backend_blocks) >= 6
    assert documented_types == {
        "state",
        "store",
        "filesystem",
        "local_shell",
        "context_hub",
        "langsmith_sandbox",
    }
    for index, block in enumerate(backend_blocks):
        agent = block.get("agent", {})
        parse_backend_config(
            block,
            Path("docs") / f"example-{index}.toml",
            agent_state=str(agent.get("state", "stateful")),
            execute_tool_enabled=bool(agent.get("execute_tool_enabled", False)),
        )


def test_deepagent_example_remains_parseable_with_commented_backend_reference() -> None:
    parsed = tomllib.loads(Path("deepagent.toml.example").read_text(encoding="utf-8"))

    assert parsed["model"]["provider"] == "ollama"
