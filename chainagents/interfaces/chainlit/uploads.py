"""Handle Chainlit file/image uploads and RAG upload action buttons."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import chainlit as cl

from chainagents.interfaces.uploads import (
    RAG_UPLOAD_ACCEPT,
    NormalizedUpload,
    image_content_part,
    is_image_upload,
    provider_safe_image_mime_type,
    uploaded_file_mime_type as _shared_uploaded_file_mime_type,
)
from chainagents.rag.runtime import UploadedRagFile

REBUILD_RAG_INDEX_ACTION = "rebuild_knowledge_index"
UPLOAD_RAG_FILE_ACTION = "upload_rag_file"


def build_rag_action() -> cl.Action:
    """Build RAG action.

    Returns:
        The constructed rag action.
    """
    return cl.Action(
        name=REBUILD_RAG_INDEX_ACTION,
        payload={},
        label="Rebuild Knowledge Index",
        tooltip="Rebuild the local documentation RAG index.",
        icon="refresh-cw",
    )


def build_upload_rag_action() -> cl.Action:
    """Build upload RAG action.

    Returns:
        The constructed upload rag action.
    """
    return cl.Action(
        name=UPLOAD_RAG_FILE_ACTION,
        payload={},
        label="Upload File For RAG",
        tooltip="Upload a text file and add it to this chat thread's knowledge index.",
        icon="paperclip",
    )


def rag_actions() -> list[cl.Action]:
    """Return Chainlit action buttons for RAG workflows.

    Returns:
        Chainlit action buttons for RAG workflows.
    """
    return [build_rag_action(), build_upload_rag_action()]


def message_uploads(message: cl.Message) -> list[tuple[Path, str, str]]:
    """Return readable files attached to a Chainlit message.

    Args:
        message: Chainlit message or LangChain message to process.

    Returns:
        Tuples of path, display name, and MIME type for attached files.
    """
    uploads: list[tuple[Path, str, str]] = []
    for element in getattr(message, "elements", []) or []:
        raw_path = getattr(element, "path", None)
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.exists() or not path.is_file():
            continue
        name = str(getattr(element, "name", "") or path.name).strip() or path.name
        mime_type = uploaded_file_mime_type(element, path=path, name=name)
        uploads.append((path, name, mime_type))
    return uploads


def uploaded_file_mime_type(element: Any, *, path: Path, name: str) -> str:
    """Resolve the MIME type for an uploaded Chainlit file."""
    for attr in ("mime", "mime_type", "content_type"):
        raw_mime = getattr(element, attr, None)
        if isinstance(raw_mime, str) and "/" in raw_mime:
            return _shared_uploaded_file_mime_type(raw_mime, path=path, name=name)
    return _shared_uploaded_file_mime_type(None, path=path, name=name)


def message_uploaded_rag_files(message: cl.Message) -> list[UploadedRagFile]:
    """Build the message for uploaded RAG files.

    Args:
        message: Chainlit message or LangChain message to process.

    Returns:
        The constructed the message for uploaded rag files.
    """
    uploads: list[UploadedRagFile] = []
    for path, name, mime_type in message_uploads(message):
        if is_image_upload(path, mime_type):
            continue
        uploads.append(UploadedRagFile(path=path, name=name))
    return uploads


def message_uploaded_image_names(message: cl.Message) -> tuple[str, ...]:
    """Return names for image files attached to a Chainlit message."""
    return tuple(
        name
        for path, name, mime_type in message_uploads(message)
        if provider_safe_image_mime_type(path, mime_type) is not None
    )


def unsupported_uploaded_image_names(message: cl.Message) -> tuple[str, ...]:
    """Return names for image files that cannot be sent to vision providers."""
    return tuple(
        name
        for path, name, mime_type in message_uploads(message)
        if is_image_upload(path, mime_type)
        and provider_safe_image_mime_type(path, mime_type) is None
    )


def message_uploaded_image_parts(message: cl.Message) -> list[dict[str, Any]]:
    """Build multimodal content parts for uploaded Chainlit images.

    Args:
        message: Chainlit message or LangChain message to process.

    Returns:
        OpenAI-compatible image content parts backed by data URLs.
    """
    parts: list[dict[str, Any]] = []
    for path, _name, mime_type in message_uploads(message):
        image_mime_type = provider_safe_image_mime_type(path, mime_type)
        if image_mime_type is None:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        parts.append(
            image_content_part(
                NormalizedUpload(
                    name=path.name,
                    mime_type=image_mime_type,
                    kind="image",
                    data=data,
                )
            )
        )
    return parts


def unsupported_uploaded_images_message(image_names: tuple[str, ...]) -> str:
    """Build a user-facing note for unsupported image uploads."""
    names = ", ".join(f"`{name}`" for name in image_names)
    return (
        "Some uploaded images were not attached to the agent request because their "
        "formats are not supported by the configured vision providers.\n\n"
        f"- Unsupported: {names}\n"
        "- Supported image formats: PNG, JPEG, WEBP, GIF"
    )


def upload_result_prompt_note(added_files: tuple[str, ...]) -> str:
    """Build the prompt note describing uploaded RAG files.

    Args:
        added_files: The added files value.

    Returns:
        The constructed the prompt note describing uploaded rag files.
    """
    if not added_files:
        return ""
    file_list = ", ".join(f"`{name}`" for name in added_files)
    return (
        "\n\nUploaded files are available in this thread's knowledge index: "
        f"{file_list}. Use `search_workspace_knowledge` if the user refers to them."
    )


def upload_result_message(upload_result) -> str:
    """Build the Chainlit message for a RAG upload result.

    Args:
        upload_result: The upload result value.

    Returns:
        The constructed the chainlit message for a rag upload result.
    """
    if upload_result.added_files:
        added = ", ".join(f"`{name}`" for name in upload_result.added_files)
        content = (
            "Uploaded file(s) added to this thread's knowledge index.\n\n"
            f"- Added: {added}\n"
            f"- Uploaded files indexed for this thread: `{upload_result.indexed_files}`\n"
            f"- Uploaded chunks indexed for this thread: `{upload_result.chunk_count}`"
        )
        if upload_result.rejected_files:
            rejected = ", ".join(f"`{name}`" for name in upload_result.rejected_files)
            content += f"\n- Rejected: {rejected}"
        return content

    if upload_result.rejected_files:
        rejected = ", ".join(f"`{name}`" for name in upload_result.rejected_files)
        return f"No supported text files were added to RAG. Rejected: {rejected}"

    return upload_result.reason or "No files were added to RAG."


async def ask_for_rag_upload() -> list[UploadedRagFile]:
    """Ask for for RAG upload.

    Returns:
        The prompt or response used to ask the user.
    """
    files = await cl.AskFileMessage(
        content=(
            "Upload text-based files for this chat thread's knowledge index.\n\n"
            "Accepted examples: `.md`, `.txt`, `.rst`, `.json`, `.toml`, `.yaml`, `.yml`, `.csv`, `.log`, `.py`."
        ),
        accept=RAG_UPLOAD_ACCEPT,
        max_size_mb=25,
        max_files=5,
        timeout=300,
        raise_on_timeout=False,
    ).send()
    if not files:
        return []

    def _existing_uploads() -> list[UploadedRagFile]:
        return [
            UploadedRagFile(path=Path(file.path), name=file.name)
            for file in files
            if Path(file.path).exists()
        ]

    return await asyncio.to_thread(_existing_uploads)
