"""Identify and describe generated workspace output files."""

from __future__ import annotations

import ast
import base64
import json
import mimetypes
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from deepagents.backends import FilesystemBackend


GENERATED_OUTPUTS_DIRECTORY = Path(".files/outputs")
GENERATED_FILE_TOOL_SUFFIXES = ("write_file", "edit_file", "create_file")
GENERATED_FILE_PATH_ARG_KEYS = (
    "path",
    "file_path",
    "destination",
    "dest",
    "output_path",
)
MAX_GENERATED_FILES = 12
MAX_REMOTE_GENERATED_FILE_BYTES = 25 * 1024 * 1024
GENERATED_OUTPUTS_VIRTUAL_ROOT = "/workspace/.files/outputs"
GENERATED_OUTPUTS_VIRTUAL_PREFIX = f"{GENERATED_OUTPUTS_VIRTUAL_ROOT}/"
GENERATED_FILE_PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:/workspace/|\.files/outputs/|/[^`'\"<>\s)]*/\.files/outputs/)"
    r"[^`'\"<>\s)]*"
    r")"
)


@dataclass(frozen=True)
class GeneratedFileDescriptor:
    """Renderer-safe metadata for one downloadable generated file."""

    name: str
    mime_type: str
    size_bytes: int
    download_url: str

    def to_payload(self) -> dict[str, str | int]:
        """Return this descriptor in its stable API wire shape."""
        return asdict(self)


@dataclass(frozen=True)
class GeneratedFileDownload:
    """Validated local path or remote bytes for one generated output."""

    name: str
    mime_type: str
    size_bytes: int
    relative_path: PurePosixPath
    local_path: Path | None = None
    content: bytes | None = None

    @property
    def download_url(self) -> str:
        """Return the stable API download URL for this output."""
        return "/api/generated-files/" + "/".join(
            quote(part, safe="") for part in self.relative_path.parts
        )


def generated_file_paths_from_tool_args(
    tool_name: str,
    raw_args: str,
) -> tuple[str, ...]:
    """Return candidate paths captured from write-style tool arguments."""
    name = tool_name.strip().lower()
    if not any(name.endswith(suffix) for suffix in GENERATED_FILE_TOOL_SUFFIXES):
        return ()

    parsed = _parse_tool_args(raw_args)
    if not isinstance(parsed, dict):
        return ()

    paths: list[str] = []
    for key in GENERATED_FILE_PATH_ARG_KEYS:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
        elif isinstance(value, list):
            paths.extend(
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            )
    return tuple(dict.fromkeys(paths))


def generated_file_paths_from_text(text: str) -> tuple[str, ...]:
    """Return generated output path tokens referenced by response text."""
    return tuple(
        dict.fromkeys(
            match.group("path")
            for match in GENERATED_FILE_PATH_RE.finditer(text)
            if match.group("path").strip()
        )
    )


def generated_file_descriptors(
    raw_paths: list[str],
    *,
    project_root: Path,
) -> list[GeneratedFileDescriptor]:
    """Resolve, validate, and deduplicate generated output paths."""
    output_root = _generated_outputs_root(project_root)
    if output_root is None:
        return []
    descriptors: list[GeneratedFileDescriptor] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        resolved = resolve_generated_output(raw_path, project_root=project_root)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        try:
            relative_path = resolved.relative_to(output_root)
            size_bytes = resolved.stat().st_size
        except (OSError, ValueError):
            continue
        mime_type, _encoding = mimetypes.guess_type(resolved.name)
        descriptors.append(
            GeneratedFileDescriptor(
                name=resolved.name,
                mime_type=mime_type or "application/octet-stream",
                size_bytes=size_bytes,
                download_url=(
                    "/api/generated-files/"
                    + "/".join(quote(part, safe="") for part in relative_path.parts)
                ),
            )
        )
        if len(descriptors) >= MAX_GENERATED_FILES:
            break
    return descriptors


async def generated_file_descriptors_for_backend(
    raw_paths: list[str],
    *,
    backend: Any,
    project_root: Path,
) -> list[GeneratedFileDescriptor]:
    """Describe safe outputs using local files or backend download bytes."""
    downloads = await generated_file_downloads_for_backend(
        raw_paths,
        backend=backend,
        project_root=project_root,
    )
    return [
        GeneratedFileDescriptor(
            name=download.name,
            mime_type=download.mime_type,
            size_bytes=download.size_bytes,
            download_url=download.download_url,
        )
        for download in downloads
    ]


async def generated_file_downloads_for_backend(
    raw_paths: list[str],
    *,
    backend: Any,
    project_root: Path,
) -> list[GeneratedFileDownload]:
    """Retrieve bounded generated outputs through the active backend."""
    local_root = _backend_generated_outputs_root(backend, project_root=project_root)
    normalized_paths: list[str] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        virtual_path = _normalize_generated_virtual_path(
            raw_path,
            local_root=local_root,
        )
        if (
            virtual_path is None
            or virtual_path in seen
            or _is_backend_artifact_path(backend, virtual_path)
        ):
            continue
        normalized_paths.append(virtual_path)
        seen.add(virtual_path)
        if len(normalized_paths) >= MAX_GENERATED_FILES:
            break

    downloads_by_path: dict[str, GeneratedFileDownload] = {}
    remote_paths: list[str] = []
    for virtual_path in normalized_paths:
        local_path = _local_output_path_for_backend(backend, virtual_path)
        if local_path is None:
            remote_paths.append(virtual_path)
            continue
        try:
            size_bytes = local_path.stat().st_size
        except OSError:
            continue
        relative_path = PurePosixPath(
            virtual_path.removeprefix(GENERATED_OUTPUTS_VIRTUAL_PREFIX)
        )
        mime_type, _encoding = mimetypes.guess_type(local_path.name)
        downloads_by_path[virtual_path] = GeneratedFileDownload(
            name=local_path.name,
            mime_type=mime_type or "application/octet-stream",
            size_bytes=size_bytes,
            relative_path=relative_path,
            local_path=local_path,
        )

    if remote_paths:
        approved_paths = await _preflight_remote_download_paths(
            remote_paths,
            backend=backend,
        )
        try:
            responses = await backend.adownload_files(approved_paths)
        except Exception:
            responses = []
        responses_by_path = {
            response.path: response
            for response in responses
            if getattr(response, "path", None) in seen
        }
        for virtual_path in approved_paths:
            response = responses_by_path.get(virtual_path)
            content = getattr(response, "content", None)
            error = getattr(response, "error", None)
            if error is not None or not isinstance(content, bytes):
                continue
            if len(content) > MAX_REMOTE_GENERATED_FILE_BYTES:
                continue
            try:
                relative_path = PurePosixPath(
                    virtual_path.removeprefix(GENERATED_OUTPUTS_VIRTUAL_PREFIX)
                )
            except ValueError:
                continue
            mime_type, _encoding = mimetypes.guess_type(relative_path.name)
            downloads_by_path[virtual_path] = GeneratedFileDownload(
                name=relative_path.name,
                mime_type=mime_type or "application/octet-stream",
                size_bytes=len(content),
                relative_path=relative_path,
                content=content,
            )
    return [
        downloads_by_path[path]
        for path in normalized_paths
        if path in downloads_by_path
    ]


def _is_backend_artifact_path(backend: Any, virtual_path: str) -> bool:
    """Reject private artifact trees accidentally placed inside public outputs."""
    artifacts_root = getattr(backend, "artifacts_root", None)
    if not isinstance(artifacts_root, str):
        return False
    artifacts_prefix = f"{artifacts_root.rstrip('/')}/"
    return (
        artifacts_prefix.startswith(GENERATED_OUTPUTS_VIRTUAL_PREFIX)
        and virtual_path.startswith(artifacts_prefix)
    )


async def _preflight_remote_download_paths(
    paths: list[str],
    *,
    backend: Any,
) -> list[str]:
    """Reject unknown, directory, or oversized remote files before download."""
    list_directory = getattr(backend, "als", None)
    if not callable(list_directory):
        return []
    paths_by_parent: dict[str, list[str]] = {}
    for path in paths:
        node, _backend_path = _backend_node_and_path(backend, path)
        if isinstance(node, FilesystemBackend) and not node.virtual_mode:
            continue
        parent = str(PurePosixPath(path).parent)
        if parent != "/":
            parent = f"{parent}/"
        paths_by_parent.setdefault(parent, []).append(path)

    approved: set[str] = set()
    for parent, candidates in paths_by_parent.items():
        try:
            result = await list_directory(parent)
        except Exception:
            continue
        if getattr(result, "error", None):
            continue
        entries = getattr(result, "entries", result)
        entries_by_path = {
            str(entry.get("path")): entry
            for entry in entries or []
            if isinstance(entry, dict) and entry.get("path")
        }
        for candidate in candidates:
            entry = entries_by_path.get(candidate)
            size = entry.get("size") if entry is not None else None
            if entry is not None and entry.get("is_dir") is False and size is None:
                size = await _probe_backend_file_size(backend, candidate)
            if (
                entry is not None
                and entry.get("is_dir") is False
                and isinstance(size, int)
                and not isinstance(size, bool)
                and 0 <= size <= MAX_REMOTE_GENERATED_FILE_BYTES
            ):
                approved.add(candidate)
    return [path for path in paths if path in approved]


async def _probe_backend_file_size(backend: Any, virtual_path: str) -> int | None:
    """Ask an execution-capable backend for metadata omitted by its listing."""
    node, backend_path = _backend_node_and_path(backend, virtual_path)
    execute = getattr(node, "aexecute", None)
    if not callable(execute):
        return None
    encoded_path = base64.b64encode(backend_path.encode("utf-8")).decode("ascii")
    script = (
        "import base64, os; "
        f"p=base64.b64decode('{encoded_path}').decode('utf-8'); "
        "print(os.lstat(p).st_size if os.path.isfile(p) and not os.path.islink(p) else -1)"
    )
    try:
        result = await execute(f"python3 -c {shlex.quote(script)}")
    except Exception:
        return None
    if getattr(result, "exit_code", None) != 0:
        return None
    try:
        size = int(str(getattr(result, "output", "")).strip())
    except ValueError:
        return None
    return size if size >= 0 else None


async def resolve_generated_download_for_backend(
    relative_path: str,
    *,
    backend: Any,
    project_root: Path,
) -> GeneratedFileDownload | None:
    """Resolve one API download request through the active backend."""
    requested = str(relative_path).strip()
    if (
        not requested
        or requested.startswith("/")
        or "\\" in requested
        or any(part in {"", ".", ".."} for part in requested.split("/"))
    ):
        return None
    downloads = await generated_file_downloads_for_backend(
        [f"{GENERATED_OUTPUTS_VIRTUAL_PREFIX}{requested}"],
        backend=backend,
        project_root=project_root,
    )
    return downloads[0] if downloads else None


def _backend_generated_outputs_root(backend: Any, *, project_root: Path) -> Path | None:
    """Return the local root used by the active output route, if any."""
    routes = getattr(backend, "routes", {})
    if GENERATED_OUTPUTS_VIRTUAL_PREFIX in routes:
        explicit_output_backend = routes[GENERATED_OUTPUTS_VIRTUAL_PREFIX]
        if (
            isinstance(explicit_output_backend, FilesystemBackend)
            and explicit_output_backend.virtual_mode
        ):
            return explicit_output_backend.cwd
        return None
    workspace_backend = routes.get("/workspace/")
    if isinstance(workspace_backend, FilesystemBackend) and workspace_backend.virtual_mode:
        return workspace_backend.cwd / GENERATED_OUTPUTS_DIRECTORY
    return None


def _local_output_path_for_backend(backend: Any, virtual_path: str) -> Path | None:
    """Resolve one output only when its most-specific route is virtual local FS."""
    node, relative = _backend_node_and_path(backend, virtual_path)
    if not isinstance(node, FilesystemBackend) or not node.virtual_mode:
        return None
    root = node.cwd
    if root.is_symlink():
        return None
    try:
        resolved_root = root.resolve()
        candidate = (resolved_root / Path(relative.lstrip("/"))).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not candidate.is_file() or not _is_relative_to(candidate, resolved_root):
        return None
    return candidate


def _backend_node_and_path(backend: Any, virtual_path: str) -> tuple[Any, str]:
    """Return the most-specific backend node and the path it receives."""
    routes = getattr(backend, "routes", {})
    matching_routes = [
        (prefix, node)
        for prefix, node in routes.items()
        if virtual_path.startswith(prefix)
    ]
    if matching_routes:
        route_prefix, node = max(matching_routes, key=lambda item: len(item[0]))
        return node, f"/{virtual_path.removeprefix(route_prefix).lstrip('/')}"
    else:
        node = getattr(backend, "default", backend)
        return node, virtual_path


def _normalize_generated_virtual_path(
    raw_path: str,
    *,
    local_root: Path | None,
) -> str | None:
    """Normalize a candidate to the one allowed generated-output subtree."""
    path_text = str(raw_path).strip().strip("`'\"<>[]()").rstrip(".,;:!?")
    if not path_text or "\\" in path_text:
        return None
    if local_root is not None:
        raw_local = Path(path_text)
        if raw_local.is_absolute() and not path_text.startswith("/workspace/"):
            try:
                relative_local = raw_local.resolve().relative_to(local_root.resolve())
            except (OSError, RuntimeError, ValueError):
                return None
            path_text = f"{GENERATED_OUTPUTS_VIRTUAL_PREFIX}{relative_local.as_posix()}"
    if path_text.startswith(f"{GENERATED_OUTPUTS_DIRECTORY.as_posix()}/"):
        path_text = f"/workspace/{path_text}"
    if not path_text.startswith(GENERATED_OUTPUTS_VIRTUAL_PREFIX):
        return None
    relative_text = path_text.removeprefix(GENERATED_OUTPUTS_VIRTUAL_PREFIX)
    if not relative_text or relative_text.startswith("/"):
        return None
    parts = relative_text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return f"{GENERATED_OUTPUTS_VIRTUAL_PREFIX}{'/'.join(parts)}"


def resolve_generated_output(raw_path: str, *, project_root: Path) -> Path | None:
    """Resolve one existing file only when it stays inside generated outputs."""
    path_text = str(raw_path).strip().strip("`'\"<>[]()").rstrip(".,;:!?")
    if not path_text:
        return None

    resolved_project_root = project_root.resolve()
    output_root = _generated_outputs_root(resolved_project_root)
    if output_root is None:
        return None

    raw_candidate = Path(path_text)
    if raw_candidate.is_absolute():
        candidates = [raw_candidate]
        if path_text.startswith("/workspace/"):
            candidates.append(
                resolved_project_root / path_text.removeprefix("/workspace/")
            )
    else:
        candidates = [resolved_project_root / path_text]

    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved.is_file() and _is_relative_to(resolved, output_root):
            return resolved
    return None


def resolve_generated_download(
    relative_path: str,
    *,
    project_root: Path,
) -> Path | None:
    """Resolve an output-relative download request without accepting traversal."""
    requested = Path(relative_path)
    if (
        not relative_path.strip()
        or requested.is_absolute()
        or ".." in requested.parts
    ):
        return None

    output_root = _generated_outputs_root(project_root)
    if output_root is None:
        return None
    try:
        resolved = (output_root / requested).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file() or not _is_relative_to(resolved, output_root):
        return None
    return resolved


def _parse_tool_args(raw_args: str) -> Any:
    text = raw_args.strip()
    if not text:
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
    return None


def _generated_outputs_root(project_root: Path) -> Path | None:
    """Return the canonical output root only when its route is not symlinked."""
    resolved_project_root = project_root.resolve()
    files_directory = resolved_project_root / GENERATED_OUTPUTS_DIRECTORY.parent
    output_directory = resolved_project_root / GENERATED_OUTPUTS_DIRECTORY
    if files_directory.is_symlink() or output_directory.is_symlink():
        return None
    try:
        output_root = output_directory.resolve()
    except OSError:
        return None
    if not _is_relative_to(output_root, resolved_project_root):
        return None
    return output_root


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
