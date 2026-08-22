"""Identify and describe generated workspace output files."""

from __future__ import annotations

import ast
import json
import mimetypes
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


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
    if raw_candidate.is_absolute() and not path_text.startswith("/workspace/"):
        candidate = raw_candidate
    elif path_text.startswith("/workspace/"):
        candidate = resolved_project_root / path_text.removeprefix("/workspace/")
    else:
        candidate = resolved_project_root / path_text

    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_file() or not _is_relative_to(resolved, output_root):
        return None
    return resolved


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
    except OSError:
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
