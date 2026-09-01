"""Typed configuration and construction for native DeepAgents backends."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeAlias

from deepagents.backends import (
    BackendProtocol,
    CompositeBackend,
    ContextHubBackend,
    FilesystemBackend,
    LangSmithSandbox,
    LocalShellBackend,
    StateBackend,
    StoreBackend,
)


BackendType = Literal[
    "state",
    "store",
    "filesystem",
    "local_shell",
    "context_hub",
    "langsmith_sandbox",
]
AgentState = Literal["stateful", "stateless"]
DEFAULT_ARTIFACTS_ROOT = "/workspace/.files/deepagent"
WORKSPACE_ROUTE = "/workspace/"
MEMORIES_ROUTE = "/memories/"
GENERATED_OUTPUTS_ROUTE = "/workspace/.files/outputs/"
_BACKEND_TYPES = frozenset(
    {
        "state",
        "store",
        "filesystem",
        "local_shell",
        "context_hub",
        "langsmith_sandbox",
    }
)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STORE_NAMESPACE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_@+:~-]+$")
_CONTEXT_HUB_LOGGER = logging.getLogger("deepagents.backends.context_hub")


class _SensitiveValueFilter(logging.Filter):
    """Redact configured backend identifiers from dependency log records."""

    def __init__(self, *values: str) -> None:
        super().__init__()
        self._values = tuple(value for value in values if value)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            for sensitive in self._values:
                value = value.replace(sensitive, "[redacted]")
            return value
        if isinstance(value, tuple):
            return tuple(self._redact(item) for item in value)
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, dict):
            return {
                self._redact(key): self._redact(item)
                for key, item in value.items()
            }
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact messages, arguments, and exception text in place."""
        record.msg = self._redact(record.msg)
        record.args = self._redact(record.args)
        if record.exc_info is not None:
            _exc_type, _exc, traceback = record.exc_info
            sanitized = RuntimeError("Context Hub backend operation failed.")
            record.exc_info = (RuntimeError, sanitized, traceback)
        return True


@dataclass(frozen=True)
class StateBackendConfig:
    """Configuration for DeepAgents' state-backed virtual filesystem."""

    type: Literal["state"] = "state"


@dataclass(frozen=True)
class StoreBackendConfig:
    """Configuration for a LangGraph store-backed virtual filesystem."""

    namespace: tuple[str, ...]
    type: Literal["store"] = "store"


@dataclass(frozen=True)
class FilesystemBackendConfig:
    """Configuration for a host filesystem backend."""

    root_dir: Path
    virtual_mode: bool = True
    max_file_size_mb: int = 10
    type: Literal["filesystem"] = "filesystem"


@dataclass(frozen=True)
class LocalShellBackendConfig:
    """Configuration for a host filesystem plus shell execution backend."""

    root_dir: Path
    virtual_mode: bool = True
    max_file_size_mb: int = 10
    timeout: int = 120
    max_output_bytes: int = 100_000
    env: dict[str, str] = field(default_factory=dict)
    inherit_env: bool = False
    type: Literal["local_shell"] = "local_shell"


@dataclass(frozen=True)
class ContextHubBackendConfig:
    """Configuration for a LangChain Context Hub repository backend."""

    identifier: str
    type: Literal["context_hub"] = "context_hub"


@dataclass(frozen=True)
class LangSmithSandboxBackendConfig:
    """Configuration for an existing LangSmith sandbox backend."""

    sandbox_name: str
    api_endpoint: str | None = None
    client_timeout: float = 10.0
    max_retries: int = 3
    type: Literal["langsmith_sandbox"] = "langsmith_sandbox"


BackendNodeConfig: TypeAlias = (
    StateBackendConfig
    | StoreBackendConfig
    | FilesystemBackendConfig
    | LocalShellBackendConfig
    | ContextHubBackendConfig
    | LangSmithSandboxBackendConfig
)


@dataclass(frozen=True)
class BackendRouteConfig:
    """Bind a normalized virtual route prefix to a backend node."""

    path: str
    backend: BackendNodeConfig


@dataclass(frozen=True)
class DeepAgentsBackendConfig:
    """Configuration for the runtime's implicit CompositeBackend."""

    default: BackendNodeConfig
    routes: tuple[BackendRouteConfig, ...] = ()
    artifacts_root: str = DEFAULT_ARTIFACTS_ROOT
    execute_tool_enabled: bool = False


def _field_name(context: str, name: str) -> str:
    return f"{context}.{name}"


def _require_table(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"The '{field_name}' config must be a table/object.")
    return value


def _reject_unknown_fields(
    raw: dict[str, Any],
    *,
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown field(s) for {context}: {joined}.")


def _require_bool(raw: dict[str, Any], name: str, *, context: str, default: bool) -> bool:
    value = raw.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"The '{_field_name(context, name)}' config must be a boolean.")
    return value


def _require_positive_int(
    raw: dict[str, Any],
    name: str,
    *,
    context: str,
    default: int,
) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"The '{_field_name(context, name)}' config must be a positive integer."
        )
    return value


def _require_nonnegative_int(
    raw: dict[str, Any],
    name: str,
    *,
    context: str,
    default: int,
) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"The '{_field_name(context, name)}' config must be a non-negative integer."
        )
    return value


def _require_positive_number(
    raw: dict[str, Any],
    name: str,
    *,
    context: str,
    default: float,
) -> float:
    value = raw.get(name, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"The '{_field_name(context, name)}' config must be a positive number."
        )
    return float(value)


def _require_nonempty_string(
    raw: dict[str, Any],
    name: str,
    *,
    context: str,
) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"The '{_field_name(context, name)}' config must be non-empty.")
    return value.strip()


def _resolve_root_dir(raw: dict[str, Any], *, context: str, base_dir: Path) -> Path:
    value = raw.get("root_dir", ".")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"The '{_field_name(context, 'root_dir')}' config must be non-empty.")
    candidate = Path(value.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def normalize_backend_route_path(value: Any) -> str:
    """Normalize one absolute POSIX route prefix."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A backend route path must be a non-empty absolute POSIX path.")
    candidate = value.strip()
    if "\\" in candidate or not candidate.startswith("/") or candidate.startswith("//"):
        raise ValueError("A backend route path must be an absolute POSIX path.")
    path = PurePosixPath(candidate)
    if path == PurePosixPath("/"):
        raise ValueError("A backend route path cannot target the root route '/'.")
    if any(part in {".", ".."} for part in candidate.split("/")):
        raise ValueError("A backend route path cannot contain traversal segments.")
    return f"/{'/'.join(path.parts[1:])}/"


def _parse_namespace(raw: dict[str, Any], *, context: str) -> tuple[str, ...]:
    value = raw.get("namespace")
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"The '{_field_name(context, 'namespace')}' config must be a non-empty array."
        )
    namespace: list[str] = []
    for segment in value:
        if not isinstance(segment, str) or not segment.strip():
            raise ValueError(
                f"The '{_field_name(context, 'namespace')}' config must contain "
                "non-empty string segments."
            )
        normalized = segment.strip()
        if not _STORE_NAMESPACE_SEGMENT_RE.fullmatch(normalized):
            raise ValueError(
                f"The '{_field_name(context, 'namespace')}' config contains a segment "
                "that is not valid for native DeepAgents stores."
            )
        namespace.append(normalized)
    return tuple(namespace)


def _parse_environment(raw: dict[str, Any], *, context: str) -> dict[str, str]:
    value = raw.get("env", {})
    if not isinstance(value, dict):
        raise ValueError(f"The '{_field_name(context, 'env')}' config must be a table/object.")
    environment: dict[str, str] = {}
    for name, env_value in value.items():
        if not isinstance(name, str) or not _ENV_NAME_RE.fullmatch(name):
            raise ValueError(
                f"The '{_field_name(context, 'env')}' config contains an invalid variable name."
            )
        if not isinstance(env_value, str):
            raise ValueError(
                f"The '{_field_name(context, 'env')}' config values must be strings."
            )
        environment[name] = env_value
    return environment


def _parse_backend_node(
    raw: dict[str, Any],
    *,
    context: str,
    base_dir: Path,
    is_default: bool,
    execute_tool_enabled: bool,
) -> BackendNodeConfig:
    raw_type = raw.get("type")
    if not isinstance(raw_type, str) or raw_type not in _BACKEND_TYPES:
        raise ValueError(
            f"The '{_field_name(context, 'type')}' config has an unknown backend type. "
            f"Expected one of: {', '.join(sorted(_BACKEND_TYPES))}."
        )

    common = {"type"}
    if raw_type == "state":
        _reject_unknown_fields(raw, allowed=common, context=f"{context} state backend")
        return StateBackendConfig()

    if raw_type == "store":
        _reject_unknown_fields(
            raw,
            allowed=common | {"namespace"},
            context=f"{context} store backend",
        )
        return StoreBackendConfig(namespace=_parse_namespace(raw, context=context))

    if raw_type == "filesystem":
        allowed = common | {
            "root_dir",
            "virtual_mode",
            "max_file_size_mb",
            "allow_unrestricted_host_filesystem",
        }
        _reject_unknown_fields(raw, allowed=allowed, context=f"{context} filesystem backend")
        virtual_mode = _require_bool(raw, "virtual_mode", context=context, default=True)
        acknowledgement = _require_bool(
            raw,
            "allow_unrestricted_host_filesystem",
            context=context,
            default=False,
        )
        if not virtual_mode and not acknowledgement:
            raise ValueError(
                f"The '{_field_name(context, 'allow_unrestricted_host_filesystem')}' "
                "config must be true when virtual_mode is false."
            )
        return FilesystemBackendConfig(
            root_dir=_resolve_root_dir(raw, context=context, base_dir=base_dir),
            virtual_mode=virtual_mode,
            max_file_size_mb=_require_positive_int(
                raw,
                "max_file_size_mb",
                context=context,
                default=10,
            ),
        )

    if raw_type == "local_shell":
        allowed = common | {
            "root_dir",
            "virtual_mode",
            "max_file_size_mb",
            "timeout",
            "max_output_bytes",
            "env",
            "inherit_env",
            "allow_unrestricted_host_execution",
        }
        _reject_unknown_fields(raw, allowed=allowed, context=f"{context} local_shell backend")
        acknowledgement = _require_bool(
            raw,
            "allow_unrestricted_host_execution",
            context=context,
            default=False,
        )
        if not acknowledgement:
            raise ValueError(
                f"The '{_field_name(context, 'allow_unrestricted_host_execution')}' "
                "config must be true for local_shell backends."
            )
        if is_default and not execute_tool_enabled:
            raise ValueError(
                "A default local_shell backend requires "
                "'[agent].execute_tool_enabled = true'."
            )
        return LocalShellBackendConfig(
            root_dir=_resolve_root_dir(raw, context=context, base_dir=base_dir),
            virtual_mode=_require_bool(raw, "virtual_mode", context=context, default=True),
            max_file_size_mb=_require_positive_int(
                raw,
                "max_file_size_mb",
                context=context,
                default=10,
            ),
            timeout=_require_positive_int(raw, "timeout", context=context, default=120),
            max_output_bytes=_require_positive_int(
                raw,
                "max_output_bytes",
                context=context,
                default=100_000,
            ),
            env=_parse_environment(raw, context=context),
            inherit_env=_require_bool(raw, "inherit_env", context=context, default=False),
        )

    if raw_type == "context_hub":
        _reject_unknown_fields(
            raw,
            allowed=common | {"identifier"},
            context=f"{context} context_hub backend",
        )
        return ContextHubBackendConfig(
            identifier=_require_nonempty_string(raw, "identifier", context=context)
        )

    _reject_unknown_fields(
        raw,
        allowed=common
        | {"sandbox_name", "api_endpoint", "client_timeout", "max_retries"},
        context=f"{context} langsmith_sandbox backend",
    )
    api_endpoint = raw.get("api_endpoint")
    if api_endpoint is not None and (
        not isinstance(api_endpoint, str) or not api_endpoint.strip()
    ):
        raise ValueError(
            f"The '{_field_name(context, 'api_endpoint')}' config must be non-empty."
        )
    return LangSmithSandboxBackendConfig(
        sandbox_name=_require_nonempty_string(raw, "sandbox_name", context=context),
        api_endpoint=api_endpoint.strip() if isinstance(api_endpoint, str) else None,
        client_timeout=_require_positive_number(
            raw,
            "client_timeout",
            context=context,
            default=10.0,
        ),
        max_retries=_require_nonnegative_int(
            raw,
            "max_retries",
            context=context,
            default=3,
        ),
    )


def parse_backend_config(
    raw_config: dict[str, Any],
    config_path: Path,
    *,
    agent_state: AgentState,
    execute_tool_enabled: bool,
) -> DeepAgentsBackendConfig | None:
    """Parse optional strict backend configuration from ``deepagent.toml``."""
    if "backend" not in raw_config:
        return None
    raw_backend = _require_table(raw_config["backend"], field_name="backend")
    routes_value = raw_backend.get("routes", [])
    artifacts_root_value = raw_backend.get("artifacts_root", DEFAULT_ARTIFACTS_ROOT)
    node_raw = {
        name: value
        for name, value in raw_backend.items()
        if name not in {"routes", "artifacts_root"}
    }
    default = _parse_backend_node(
        node_raw,
        context="backend",
        base_dir=config_path.parent,
        is_default=True,
        execute_tool_enabled=execute_tool_enabled,
    )
    if not isinstance(artifacts_root_value, str):
        raise ValueError("The 'backend.artifacts_root' config must be a string.")
    try:
        artifacts_root = normalize_backend_route_path(artifacts_root_value).rstrip("/")
    except ValueError as exc:
        raise ValueError(f"Invalid 'backend.artifacts_root' config: {exc}") from exc
    artifacts_route = f"{artifacts_root}/"
    if artifacts_route.startswith(GENERATED_OUTPUTS_ROUTE):
        raise ValueError(
            "The 'backend.artifacts_root' config cannot equal or be nested beneath "
            "the generated-output route."
        )
    if not isinstance(routes_value, list):
        raise ValueError("The 'backend.routes' config must be an array of tables.")

    routes: list[BackendRouteConfig] = []
    seen_paths: set[str] = set()
    for index, raw_route_value in enumerate(routes_value, start=1):
        raw_route = _require_table(
            raw_route_value,
            field_name=f"backend.routes entry #{index}",
        )
        path = normalize_backend_route_path(raw_route.get("path"))
        if path in seen_paths:
            raise ValueError(f"A duplicate backend route is configured for '{path}'.")
        route_node_raw = {name: value for name, value in raw_route.items() if name != "path"}
        backend = _parse_backend_node(
            route_node_raw,
            context=f"backend.routes[{index}]",
            base_dir=config_path.parent,
            is_default=False,
            execute_tool_enabled=execute_tool_enabled,
        )
        routes.append(BackendRouteConfig(path=path, backend=backend))
        seen_paths.add(path)

    if any(
        route.path.startswith(GENERATED_OUTPUTS_ROUTE)
        and isinstance(route.backend, StateBackendConfig)
        for route in routes
    ):
        raise ValueError(
            "State backends cannot override the generated-output route because "
            "downloads run outside graph state."
        )
    if any(
        route.path.startswith(GENERATED_OUTPUTS_ROUTE)
        and isinstance(
            route.backend,
            (FilesystemBackendConfig, LocalShellBackendConfig),
        )
        and not route.backend.virtual_mode
        for route in routes
    ):
        raise ValueError(
            "Non-virtual host backends cannot override the generated-output route "
            "because composite path stripping would bypass output confinement."
        )

    if agent_state == "stateless":
        if isinstance(default, StoreBackendConfig) or any(
            isinstance(route.backend, StoreBackendConfig) for route in routes
        ):
            raise ValueError("Store backends cannot be configured in stateless mode.")
        if any(route.path == MEMORIES_ROUTE for route in routes):
            raise ValueError(
                "The '/memories/' backend route cannot be overridden in stateless mode."
            )

    return DeepAgentsBackendConfig(
        default=default,
        routes=tuple(routes),
        artifacts_root=artifacts_root,
        execute_tool_enabled=execute_tool_enabled,
    )


@dataclass(frozen=True)
class BackendRouteSummary:
    """Non-sensitive route metadata exposed by runtime status."""

    path: str
    type: BackendType


@dataclass(frozen=True)
class BackendMetadata:
    """Non-sensitive capability metadata for a constructed backend."""

    default_type: BackendType
    routes: tuple[BackendRouteSummary, ...]
    execution_capable: bool
    workspace_local: bool
    workspace_root: Path | None = None
    outputs_local: bool = False
    outputs_root: Path | None = None
    execution_environment: Literal["host", "sandbox"] | None = None

    def to_status(self) -> dict[str, Any]:
        """Return the safe JSON-compatible status payload."""
        return {
            "default_type": self.default_type,
            "routes": [
                {"path": route.path, "type": route.type} for route in self.routes
            ],
            "execution_capable": self.execution_capable,
            "workspace_local": self.workspace_local,
        }


@dataclass
class BackendBundle:
    """Own a shared CompositeBackend and resources opened while building it."""

    backend: CompositeBackend
    metadata: BackendMetadata
    _cleanup_callbacks: list[Callable[[], Any]] = field(default_factory=list)
    _closed: bool = False

    async def close(self) -> None:
        """Close owned clients exactly once without deleting remote resources."""
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []
        for callback in reversed(self._cleanup_callbacks):
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                errors.append(
                    RuntimeError(
                        f"Backend cleanup failed ({type(exc).__name__})."
                    )
                )
        if errors:
            raise ExceptionGroup("Backend cleanup failed.", errors)


SandboxClientFactory: TypeAlias = Callable[..., Any]
ContextClientFactory: TypeAlias = Callable[..., Any]


def _build_node(
    config: BackendNodeConfig,
    *,
    store: Any | None,
    cleanup_callbacks: list[Callable[[], Any]],
    sandbox_client_factory: SandboxClientFactory | None,
    context_client_factory: ContextClientFactory | None,
) -> BackendProtocol:
    """Construct one native backend node."""
    if isinstance(config, StateBackendConfig):
        return StateBackend()
    if isinstance(config, StoreBackendConfig):
        return StoreBackend(namespace=lambda _runtime: config.namespace, store=store)
    if isinstance(config, FilesystemBackendConfig):
        return FilesystemBackend(
            root_dir=config.root_dir,
            virtual_mode=config.virtual_mode,
            max_file_size_mb=config.max_file_size_mb,
        )
    if isinstance(config, LocalShellBackendConfig):
        backend = LocalShellBackend(
            root_dir=config.root_dir,
            virtual_mode=config.virtual_mode,
            timeout=config.timeout,
            max_output_bytes=config.max_output_bytes,
            env=dict(config.env),
            inherit_env=config.inherit_env,
        )
        backend.max_file_size_bytes = config.max_file_size_mb * 1024 * 1024
        return backend
    if isinstance(config, ContextHubBackendConfig):
        if context_client_factory is None:
            from langsmith import Client

            context_client_factory = Client
        client = context_client_factory(
            api_url=os.getenv("LANGSMITH_ENDPOINT") or None,
            api_key=os.getenv("LANGSMITH_API_KEY") or None,
        )
        close = getattr(client, "close", None)
        log_filter = _SensitiveValueFilter(config.identifier)
        _CONTEXT_HUB_LOGGER.addFilter(log_filter)
        try:
            backend = ContextHubBackend(config.identifier, client=client)
        except Exception:
            if callable(close):
                close()
            _CONTEXT_HUB_LOGGER.removeFilter(log_filter)
            raise RuntimeError("Failed to configure the Context Hub backend.") from None
        cleanup_callbacks.append(
            lambda: _CONTEXT_HUB_LOGGER.removeFilter(log_filter)
        )
        if callable(close):
            cleanup_callbacks.append(close)
        return backend

    if sandbox_client_factory is None:
        from langsmith.sandbox import SandboxClient

        sandbox_client_factory = SandboxClient
    client = sandbox_client_factory(
        api_endpoint=config.api_endpoint or os.getenv("LANGSMITH_ENDPOINT") or None,
        timeout=config.client_timeout,
        api_key=os.getenv("LANGSMITH_API_KEY") or None,
        max_retries=config.max_retries,
    )
    close = getattr(client, "close", None)
    try:
        sandbox = client.get_sandbox(config.sandbox_name)
    except Exception:
        if callable(close):
            close()
        raise RuntimeError(
            "Failed to attach the configured LangSmith sandbox."
        ) from None
    backend = LangSmithSandbox(sandbox)
    if callable(close):
        cleanup_callbacks.append(close)
    aclose = getattr(backend, "aclose", None)
    if callable(aclose):
        cleanup_callbacks.append(aclose)
    return backend


def _node_type(config: BackendNodeConfig) -> BackendType:
    return config.type


def _run_awaitable_for_rollback(awaitable: Any) -> None:
    """Finish an async cleanup while a synchronous factory unwinds."""

    def run() -> None:
        try:
            asyncio.run(awaitable)
        except Exception:
            return

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        run()
        return
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()


def _rollback_cleanup_callbacks(callbacks: list[Callable[[], Any]]) -> None:
    """Best-effort cleanup for resources opened before construction failed."""
    for callback in reversed(callbacks):
        try:
            result = callback()
            if inspect.isawaitable(result):
                _run_awaitable_for_rollback(result)
        except Exception:
            continue


def close_backend_bundle_after_failure(bundle: BackendBundle) -> None:
    """Synchronously roll back an owned bundle when graph creation aborts."""
    _run_awaitable_for_rollback(bundle.close())


def build_backend_bundle(
    config: DeepAgentsBackendConfig,
    *,
    project_root: Path,
    include_memories: bool,
    memory_namespace: str,
    store: Any | None = None,
    sandbox_client_factory: SandboxClientFactory | None = None,
    context_client_factory: ContextClientFactory | None = None,
) -> BackendBundle:
    """Build one configured CompositeBackend and its lifecycle metadata."""
    cleanup_callbacks: list[Callable[[], Any]] = []
    try:
        default = _build_node(
            config.default,
            store=store,
            cleanup_callbacks=cleanup_callbacks,
            sandbox_client_factory=sandbox_client_factory,
            context_client_factory=context_client_factory,
        )
        artifacts_local_root = (project_root / ".files" / "deepagent").resolve()
        outputs_local_root = (project_root / ".files" / "outputs").resolve()
        routes: dict[str, BackendProtocol] = {
            f"{config.artifacts_root.rstrip('/')}/": FilesystemBackend(
                root_dir=artifacts_local_root,
                virtual_mode=True,
            ),
            GENERATED_OUTPUTS_ROUTE: FilesystemBackend(
                root_dir=outputs_local_root,
                virtual_mode=True,
            ),
            WORKSPACE_ROUTE: FilesystemBackend(
                root_dir=project_root,
                virtual_mode=True,
            ),
        }
        route_types: dict[str, BackendType] = {
            f"{config.artifacts_root.rstrip('/')}/": "filesystem",
            GENERATED_OUTPUTS_ROUTE: "filesystem",
            WORKSPACE_ROUTE: "filesystem",
        }
        if include_memories:
            routes[MEMORIES_ROUTE] = StoreBackend(
                namespace=lambda _runtime: (memory_namespace,),
                store=store,
            )
            route_types[MEMORIES_ROUTE] = "store"
        for route in config.routes:
            routes[route.path] = _build_node(
                route.backend,
                store=store,
                cleanup_callbacks=cleanup_callbacks,
                sandbox_client_factory=sandbox_client_factory,
                context_client_factory=context_client_factory,
            )
            route_types[route.path] = _node_type(route.backend)

        backend = CompositeBackend(
            default=default,
            routes=routes,
            artifacts_root=config.artifacts_root,
        )
    except Exception as exc:
        _rollback_cleanup_callbacks(cleanup_callbacks)
        message = str(exc)
        if message in {
            "Failed to attach the configured LangSmith sandbox.",
            "Failed to configure the Context Hub backend.",
        }:
            raise RuntimeError(message) from None
        raise RuntimeError("Failed to configure the DeepAgents backend.") from None
    workspace_backend = routes[WORKSPACE_ROUTE]
    outputs_backend = routes[GENERATED_OUTPUTS_ROUTE]
    execution_capable = (
        config.execute_tool_enabled
        and callable(getattr(default, "execute", None))
    )
    execution_environment = (
        "host"
        if execution_capable and isinstance(default, LocalShellBackend)
        else "sandbox"
        if execution_capable and isinstance(default, LangSmithSandbox)
        else None
    )
    metadata = BackendMetadata(
        default_type=_node_type(config.default),
        routes=tuple(
            BackendRouteSummary(path=path, type=route_types[path])
            for path in routes
        ),
        execution_capable=execution_capable,
        workspace_local=isinstance(workspace_backend, FilesystemBackend),
        workspace_root=(
            workspace_backend.cwd
            if (
                isinstance(workspace_backend, FilesystemBackend)
                and workspace_backend.virtual_mode
            )
            else None
        ),
        outputs_local=isinstance(outputs_backend, FilesystemBackend),
        outputs_root=(
            outputs_backend.cwd
            if (
                isinstance(outputs_backend, FilesystemBackend)
                and outputs_backend.virtual_mode
            )
            else None
        ),
        execution_environment=execution_environment,
    )
    return BackendBundle(
        backend=backend,
        metadata=metadata,
        _cleanup_callbacks=cleanup_callbacks,
    )


def build_legacy_backend_bundle(
    *,
    project_root: Path,
    include_memories: bool,
    memory_namespace: str,
    store: Any | None = None,
) -> BackendBundle:
    """Build the historical local composite backend without changing its routes."""
    artifacts_root = (project_root / ".files" / "deepagent").resolve()
    outputs_root = (project_root / ".files" / "outputs").resolve()
    routes: dict[str, BackendProtocol] = {
        f"{artifacts_root.as_posix()}/": FilesystemBackend(
            root_dir=artifacts_root,
            virtual_mode=True,
        ),
        f"{outputs_root.as_posix()}/": FilesystemBackend(
            root_dir=outputs_root,
            virtual_mode=True,
        ),
        WORKSPACE_ROUTE: FilesystemBackend(
            root_dir=project_root,
            virtual_mode=True,
        ),
    }
    if include_memories:
        routes[MEMORIES_ROUTE] = StoreBackend(
            namespace=lambda _runtime: (memory_namespace,),
            store=store,
        )
    backend = CompositeBackend(
        default=StateBackend(),
        routes=routes,
        artifacts_root=artifacts_root.as_posix(),
    )
    route_summaries = [
        BackendRouteSummary(path=DEFAULT_ARTIFACTS_ROOT + "/", type="filesystem"),
        BackendRouteSummary(path=GENERATED_OUTPUTS_ROUTE, type="filesystem"),
        BackendRouteSummary(path=WORKSPACE_ROUTE, type="filesystem"),
    ]
    if include_memories:
        route_summaries.append(BackendRouteSummary(path=MEMORIES_ROUTE, type="store"))
    return BackendBundle(
        backend=backend,
        metadata=BackendMetadata(
            default_type="state",
            routes=tuple(route_summaries),
            execution_capable=False,
            workspace_local=True,
            workspace_root=project_root.resolve(),
            outputs_local=True,
            outputs_root=outputs_root,
        ),
    )


def build_runtime_backend_bundle(
    *,
    backend_config: DeepAgentsBackendConfig | None,
    project_root: Path,
    include_memories: bool,
    memory_namespace: str,
    store: Any | None = None,
    sandbox_client_factory: SandboxClientFactory | None = None,
    context_client_factory: ContextClientFactory | None = None,
) -> BackendBundle:
    """Build either the configured backend or the unchanged legacy backend."""
    if backend_config is None:
        return build_legacy_backend_bundle(
            project_root=project_root,
            include_memories=include_memories,
            memory_namespace=memory_namespace,
            store=store,
        )
    return build_backend_bundle(
        backend_config,
        project_root=project_root,
        include_memories=include_memories,
        memory_namespace=memory_namespace,
        store=store,
        sandbox_client_factory=sandbox_client_factory,
        context_client_factory=context_client_factory,
    )


def build_deepagent_backend(
    *,
    project_root: Path,
    include_memories: bool = True,
    memory_namespace: str = "filesystem",
    backend_config: DeepAgentsBackendConfig | None = None,
    store: Any | None = None,
    sandbox_client_factory: SandboxClientFactory | None = None,
    context_client_factory: ContextClientFactory | None = None,
) -> CompositeBackend:
    """Compatibility factory returning a backend with an explicit async closer."""
    bundle = build_runtime_backend_bundle(
        backend_config=backend_config,
        project_root=project_root,
        include_memories=include_memories,
        memory_namespace=memory_namespace,
        store=store,
        sandbox_client_factory=sandbox_client_factory,
        context_client_factory=context_client_factory,
    )
    setattr(bundle.backend, "_chainagents_backend_bundle", bundle)
    setattr(bundle.backend, "aclose_chainagents_resources", bundle.close)
    return bundle.backend


__all__ = [
    "BackendBundle",
    "BackendMetadata",
    "BackendNodeConfig",
    "BackendRouteConfig",
    "BackendRouteSummary",
    "ContextHubBackendConfig",
    "DEFAULT_ARTIFACTS_ROOT",
    "DeepAgentsBackendConfig",
    "FilesystemBackendConfig",
    "LangSmithSandboxBackendConfig",
    "LocalShellBackendConfig",
    "StateBackendConfig",
    "StoreBackendConfig",
    "build_deepagent_backend",
    "build_backend_bundle",
    "build_legacy_backend_bundle",
    "build_runtime_backend_bundle",
    "close_backend_bundle_after_failure",
    "normalize_backend_route_path",
    "parse_backend_config",
]
