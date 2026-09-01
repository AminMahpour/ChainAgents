"""Test generated output discovery and download path confinement."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import chainagents.exports.generated_files as generated_files
from chainagents.exports.generated_files import (
    MAX_REMOTE_GENERATED_FILE_BYTES,
    MAX_GENERATED_FILES,
    generated_file_descriptors_for_backend,
    generated_file_descriptors,
    resolve_generated_download_for_backend,
    resolve_generated_download,
    resolve_generated_output,
)
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import FileDownloadResponse, LsResult


def test_generated_descriptors_reject_unsafe_or_unavailable_paths(
    tmp_path: Path,
) -> None:
    """Only existing regular files within the generated output root are eligible."""
    outputs = tmp_path / ".files" / "outputs"
    outputs.mkdir(parents=True)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (outputs / "folder").mkdir()
    (outputs / "escape.txt").symlink_to(outside)

    descriptors = generated_file_descriptors(
        [
            "/workspace/.files/outputs/missing.txt",
            "/workspace/.files/outputs/folder",
            "/workspace/.files/outputs/escape.txt",
            "/workspace/.files/outputs/../outputs/escape.txt",
            outside.as_posix(),
        ],
        project_root=tmp_path,
    )

    assert descriptors == []
    assert resolve_generated_download("../secret.txt", project_root=tmp_path) is None
    assert resolve_generated_download(outside.as_posix(), project_root=tmp_path) is None


def test_generated_output_rejects_cyclic_symlink(tmp_path: Path) -> None:
    """A cyclic output reference must be ignored instead of breaking the stream."""
    outputs = tmp_path / ".files" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "loop").symlink_to("loop")

    assert (
        resolve_generated_output(
            "/workspace/.files/outputs/loop",
            project_root=tmp_path,
        )
        is None
    )


def test_generated_download_rejects_cyclic_symlink(tmp_path: Path) -> None:
    """A cyclic download path must fail closed instead of returning a server error."""
    outputs = tmp_path / ".files" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "loop").symlink_to("loop")

    assert resolve_generated_download("loop", project_root=tmp_path) is None


def test_generated_output_preserves_physical_path_beneath_workspace(
    monkeypatch,
) -> None:
    """A physical project path under /workspace wins over virtual path mapping."""
    project_root = Path("/workspace/ChainAgents")
    output = project_root / ".files" / "outputs" / "report.csv"
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(Path, "is_file", lambda self: self == output)

    assert (
        resolve_generated_output(output.as_posix(), project_root=project_root)
        == output
    )


def test_generated_descriptors_deduplicate_encode_and_cap_paths(
    tmp_path: Path,
) -> None:
    """Metadata is stable, URL-safe, deduplicated, and bounded."""
    outputs = tmp_path / ".files" / "outputs"
    outputs.mkdir(parents=True)
    named = outputs / "résumé #1.txt"
    named.write_text("hello", encoding="utf-8")
    remaining = []
    for index in range(MAX_GENERATED_FILES + 2):
        path = outputs / f"report-{index}.txt"
        path.write_text(str(index), encoding="utf-8")
        remaining.append(f"/workspace/.files/outputs/{path.name}")

    descriptors = generated_file_descriptors(
        [
            "/workspace/.files/outputs/résumé #1.txt",
            named.as_posix(),
            *remaining,
        ],
        project_root=tmp_path,
    )

    assert len(descriptors) == MAX_GENERATED_FILES
    assert descriptors[0].download_url == (
        "/api/generated-files/r%C3%A9sum%C3%A9%20%231.txt"
    )
    assert [descriptor.name for descriptor in descriptors].count(named.name) == 1


def test_generated_descriptors_fail_closed_when_file_disappears(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A file removed after resolution must not fail the enclosing agent stream."""
    output = tmp_path / ".files" / "outputs" / "transient.txt"
    output.parent.mkdir(parents=True)
    output.write_text("temporary", encoding="utf-8")

    def disappearing_output(raw_path: str, *, project_root: Path) -> Path:
        del raw_path, project_root
        output.unlink()
        return output

    monkeypatch.setattr(
        generated_files,
        "resolve_generated_output",
        disappearing_output,
    )

    assert generated_file_descriptors(
        ["/workspace/.files/outputs/transient.txt"],
        project_root=tmp_path,
    ) == []


def test_generated_download_rejects_symlinked_output_root(tmp_path: Path) -> None:
    """The output directory itself cannot redirect downloads elsewhere."""
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret.txt").write_text("secret", encoding="utf-8")
    files_directory = tmp_path / ".files"
    files_directory.mkdir()
    (files_directory / "outputs").symlink_to(private, target_is_directory=True)

    assert resolve_generated_download("secret.txt", project_root=tmp_path) is None
    assert generated_file_descriptors(
        ["/workspace/.files/outputs/secret.txt"],
        project_root=tmp_path,
    ) == []


class _RemoteBackend:
    """Return controlled backend download responses for remote-output tests."""

    def __init__(self, responses: dict[str, FileDownloadResponse]) -> None:
        self.responses = responses
        self.requests: list[list[str]] = []

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        self.requests.append(paths)
        return [
            self.responses.get(
                path,
                FileDownloadResponse(path=path, error="file_not_found"),
            )
            for path in paths
        ]

    async def als(self, path: str) -> LsResult:
        prefix = path if path.endswith("/") else f"{path}/"
        entries = []
        for response_path, response in self.responses.items():
            if str(Path(response_path).parent).rstrip("/") != prefix.rstrip("/"):
                continue
            content = response.content
            entries.append(
                {
                    "path": response_path,
                    "is_dir": False,
                    "size": len(content) if isinstance(content, bytes) else 0,
                    "modified_at": "",
                }
            )
        return LsResult(entries=entries)


def test_remote_generated_descriptors_download_bytes_and_fail_closed(tmp_path: Path) -> None:
    valid_path = "/workspace/.files/outputs/report.txt"
    directory_path = "/workspace/.files/outputs/folder"
    oversized_path = "/workspace/.files/outputs/huge.bin"
    backend = _RemoteBackend(
        {
            valid_path: FileDownloadResponse(path=valid_path, content=b"report"),
            directory_path: FileDownloadResponse(path=directory_path, error="is_directory"),
            oversized_path: FileDownloadResponse(
                path=oversized_path,
                content=b"x" * (MAX_REMOTE_GENERATED_FILE_BYTES + 1),
            ),
        }
    )

    descriptors = asyncio.run(
        generated_file_descriptors_for_backend(
            [
                valid_path,
                ".files/outputs/report.txt",
                directory_path,
                oversized_path,
                "/workspace/.files/outputs/../secret.txt",
                "/workspace/other.txt",
            ],
            backend=backend,
            project_root=tmp_path,
        )
    )

    assert [descriptor.name for descriptor in descriptors] == ["report.txt"]
    assert descriptors[0].size_bytes == 6
    assert descriptors[0].download_url == "/api/generated-files/report.txt"
    assert backend.requests == [[valid_path, directory_path]]


def test_remote_generated_descriptors_reject_backend_errors(tmp_path: Path) -> None:
    class FailingBackend:
        async def adownload_files(self, paths: list[str]):
            raise RuntimeError("remote service unavailable")

    descriptors = asyncio.run(
        generated_file_descriptors_for_backend(
            ["/workspace/.files/outputs/report.txt"],
            backend=FailingBackend(),
            project_root=tmp_path,
        )
    )

    assert descriptors == []


def test_remote_sandbox_outputs_preflight_size_when_listing_omits_it(
    tmp_path: Path,
) -> None:
    virtual_path = "/workspace/.files/outputs/report.txt"

    class SandboxLikeBackend(_RemoteBackend):
        async def als(self, path: str) -> LsResult:
            return LsResult(
                entries=[{"path": virtual_path, "is_dir": False}]
            )

        async def aexecute(self, command: str):
            assert "python3" in command
            return SimpleNamespace(exit_code=0, output="11\n")

    backend = SandboxLikeBackend(
        {virtual_path: FileDownloadResponse(path=virtual_path, content=b"hello world")}
    )

    downloads = asyncio.run(
        generated_files.generated_file_downloads_for_backend(
            [virtual_path],
            backend=backend,
            project_root=tmp_path,
        )
    )

    assert len(downloads) == 1
    assert downloads[0].content == b"hello world"


def test_explicit_remote_output_route_overrides_local_workspace_fast_path(
    tmp_path: Path,
) -> None:
    virtual_path = "/workspace/.files/outputs/report.txt"
    stale_local = tmp_path / ".files" / "outputs" / "report.txt"
    stale_local.parent.mkdir(parents=True)
    stale_local.write_text("stale local data", encoding="utf-8")
    remote_outputs = _RemoteBackend(
        {"/report.txt": FileDownloadResponse(path="/report.txt", content=b"remote data")}
    )
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
            "/workspace/.files/outputs/": remote_outputs,
        },
        artifacts_root="/workspace/.files/deepagent",
    )

    downloads = asyncio.run(
        generated_files.generated_file_downloads_for_backend(
            [virtual_path],
            backend=backend,
            project_root=tmp_path,
        )
    )

    assert len(downloads) == 1
    assert downloads[0].content == b"remote data"
    assert downloads[0].local_path is None


def test_nested_remote_output_route_takes_precedence_over_local_workspace(
    tmp_path: Path,
) -> None:
    virtual_path = "/workspace/.files/outputs/private/report.txt"
    stale_local = tmp_path / ".files" / "outputs" / "private" / "report.txt"
    stale_local.parent.mkdir(parents=True)
    stale_local.write_text("stale local data", encoding="utf-8")
    remote_outputs = _RemoteBackend(
        {"/report.txt": FileDownloadResponse(path="/report.txt", content=b"remote data")}
    )
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
            "/workspace/.files/outputs/private/": remote_outputs,
        },
        artifacts_root="/workspace/.files/deepagent",
    )

    downloads = asyncio.run(
        generated_files.generated_file_downloads_for_backend(
            [virtual_path],
            backend=backend,
            project_root=tmp_path,
        )
    )

    assert len(downloads) == 1
    assert downloads[0].content == b"remote data"
    assert downloads[0].local_path is None


def test_nonvirtual_output_route_cannot_download_arbitrary_host_files(tmp_path: Path) -> None:
    outside = tmp_path / "outside-secret.txt"
    outside.write_bytes(b"secret")
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/.files/outputs/": FilesystemBackend(
                root_dir=tmp_path / "configured-root",
                virtual_mode=False,
            )
        },
        artifacts_root="/workspace/.files/deepagent",
    )
    virtual_path = f"/workspace/.files/outputs/{outside.as_posix().lstrip('/')}"

    downloads = asyncio.run(
        generated_files.generated_file_downloads_for_backend(
            [virtual_path],
            backend=backend,
            project_root=tmp_path,
        )
    )

    assert downloads == []


def test_generated_downloads_defensively_reject_backend_artifact_subtrees(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / ".files" / "deepagent"
    artifact_root.mkdir(parents=True)
    (artifact_root / "secret.txt").write_text("private artifact", encoding="utf-8")
    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/.files/outputs/internal/": FilesystemBackend(
                root_dir=artifact_root,
                virtual_mode=True,
            )
        },
        artifacts_root="/workspace/.files/outputs/internal",
    )

    downloads = asyncio.run(
        generated_files.generated_file_downloads_for_backend(
            ["/workspace/.files/outputs/internal/secret.txt"],
            backend=backend,
            project_root=tmp_path,
        )
    )

    assert downloads == []


def test_remote_generated_descriptors_keep_twelve_file_limit(tmp_path: Path) -> None:
    paths = [
        f"/workspace/.files/outputs/report-{index}.txt"
        for index in range(MAX_GENERATED_FILES + 2)
    ]
    backend = _RemoteBackend(
        {
            path: FileDownloadResponse(path=path, content=str(index).encode())
            for index, path in enumerate(paths)
        }
    )

    descriptors = asyncio.run(
        generated_file_descriptors_for_backend(
            paths,
            backend=backend,
            project_root=tmp_path,
        )
    )

    assert len(descriptors) == MAX_GENERATED_FILES
    assert backend.requests == [paths[:MAX_GENERATED_FILES]]


def test_remote_generated_download_normalizes_relative_paths(tmp_path: Path) -> None:
    virtual_path = "/workspace/.files/outputs/reports/result.csv"
    backend = _RemoteBackend(
        {
            virtual_path: FileDownloadResponse(
                path=virtual_path,
                content=b"name,value\na,1\n",
            )
        }
    )

    download = asyncio.run(
        resolve_generated_download_for_backend(
            "reports/result.csv",
            backend=backend,
            project_root=tmp_path,
        )
    )
    traversal = asyncio.run(
        resolve_generated_download_for_backend(
            "../secret.txt",
            backend=backend,
            project_root=tmp_path,
        )
    )

    assert download is not None
    assert download.name == "result.csv"
    assert download.content == b"name,value\na,1\n"
    assert download.local_path is None
    assert traversal is None
    assert backend.requests == [[virtual_path]]
