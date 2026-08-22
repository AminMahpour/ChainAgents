"""Test generated output discovery and download path confinement."""

from __future__ import annotations

from pathlib import Path

import chainagents.exports.generated_files as generated_files
from chainagents.exports.generated_files import (
    MAX_GENERATED_FILES,
    generated_file_descriptors,
    resolve_generated_download,
    resolve_generated_output,
)


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
