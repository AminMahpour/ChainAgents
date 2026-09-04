"""Workspace path mapping and DeepAgents filesystem backends."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
    StoreBackend,
)

import chainagents.runtime.constants as runtime_constants
from chainagents.runtime.constants import (
    DEEPAGENT_ARTIFACTS_DIRECTORY,
    DEFAULT_AGENT_MEMORY_NAMESPACE,
    GENERATED_OUTPUTS_DIRECTORY,
)


def deepagent_artifacts_root(project_root: Path | None = None) -> Path:
    """Return the local directory used for stored tool artifacts.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The local directory used for stored tool artifacts.
    """
    root = (project_root or runtime_constants.PROJECT_ROOT).resolve()
    return root / DEEPAGENT_ARTIFACTS_DIRECTORY


def deepagent_artifacts_route_prefix(project_root: Path | None = None) -> str:
    """Return the URL route prefix for stored tool artifacts.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The URL route prefix for stored tool artifacts.
    """
    return f"{deepagent_artifacts_root(project_root).as_posix().rstrip('/')}/"


def generated_outputs_root(project_root: Path | None = None) -> Path:
    """Return the local directory used for downloadable generated outputs.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The local directory used for downloadable generated outputs.
    """
    root = (project_root or runtime_constants.PROJECT_ROOT).resolve()
    return root / GENERATED_OUTPUTS_DIRECTORY


def generated_outputs_route_prefix(project_root: Path | None = None) -> str:
    """Return the URL route prefix for downloadable generated outputs.

    Args:
        project_root: Project root used to resolve local paths.

    Returns:
        The URL route prefix for downloadable generated outputs.
    """
    return f"{generated_outputs_root(project_root).as_posix().rstrip('/')}/"


WORKSPACE_PATH_TOOL_ARG_KEYS = {
    "destination",
    "dest",
    "dst",
    "path",
    "paths",
    "source",
    "src",
}


def _map_workspace_tool_path_value(value: Any, project_root: Path) -> Any:
    """Map one virtual workspace path value to a local path.

    Args:
        value: Value to normalize, convert, or serialize.
        project_root: Project root used to resolve local paths.

    Returns:
        The mapped value.
    """
    if isinstance(value, str):
        return virtual_workspace_path_to_local(value, project_root)
    if isinstance(value, list):
        return [_map_workspace_tool_path_value(item, project_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_workspace_tool_path_value(item, project_root) for item in value)
    return value


def map_workspace_paths_in_tool_args(args: Any, project_root: Path | None = None) -> Any:
    """Map workspace paths in tool args.

    Args:
        args: Parsed command-line arguments.
        project_root: Project root used to resolve local paths.

    Returns:
        The mapped value.
    """
    if not isinstance(args, dict):
        return args

    root = (project_root or runtime_constants.PROJECT_ROOT).resolve()
    mapped = dict(args)
    for key, value in args.items():
        if str(key).lower() in WORKSPACE_PATH_TOOL_ARG_KEYS:
            mapped[key] = _map_workspace_tool_path_value(value, root)
    return mapped


def virtual_workspace_path_to_local(path_value: str, project_root: Path | None = None) -> str:
    """Convert a virtual workspace path into a local filesystem path.

    Args:
        path_value: The path value value.
        project_root: Project root used to resolve local paths.

    Returns:
        The virtual workspace path to local result.
    """
    normalized = path_value.strip().replace("\\", "/")
    workspace_prefix = "/workspace"
    if normalized != workspace_prefix and not normalized.startswith(f"{workspace_prefix}/"):
        return path_value

    root = (project_root or runtime_constants.PROJECT_ROOT).resolve()
    relative = PurePosixPath(normalized.removeprefix(workspace_prefix).lstrip("/"))
    local_path = (root / Path(*relative.parts)).resolve()
    try:
        local_path.relative_to(root)
    except ValueError:
        return path_value
    return str(local_path)


def build_deepagent_backend(
    *,
    project_root: Path | None = None,
    include_memories: bool = True,
    memory_namespace: str = DEFAULT_AGENT_MEMORY_NAMESPACE,
) -> CompositeBackend:
    """Build deepagent backend.

    Args:
        project_root: Project root used to resolve local paths.
        include_memories: Whether to expose the /memories/ store route.
        memory_namespace: Shared StoreBackend namespace for /memories/.

    Returns:
        The constructed deepagent backend.
    """
    resolved_project_root = project_root or runtime_constants.PROJECT_ROOT
    artifacts_root = deepagent_artifacts_root(resolved_project_root)
    outputs_root = generated_outputs_root(resolved_project_root)
    routes = {
        deepagent_artifacts_route_prefix(resolved_project_root): FilesystemBackend(
            root_dir=str(artifacts_root),
            virtual_mode=True,
        ),
        generated_outputs_route_prefix(resolved_project_root): FilesystemBackend(
            root_dir=str(outputs_root),
            virtual_mode=True,
        ),
        "/workspace/": FilesystemBackend(
            root_dir=str(resolved_project_root),
            virtual_mode=True,
        ),
    }
    if include_memories:
        routes["/memories/"] = StoreBackend(
            namespace=lambda _runtime: (memory_namespace,)
        )
    return CompositeBackend(
        default=StateBackend(),
        routes=routes,
        artifacts_root=str(artifacts_root),
    )
