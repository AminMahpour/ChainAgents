"""Shared upload limits and supported content types for ChainAgents UIs."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chainagents.rag.runtime import ALLOWED_RAG_UPLOAD_EXTENSIONS


MAX_UPLOAD_FILES = 5
MAX_UPLOAD_FILE_BYTES = 25 * 1024 * 1024
SUPPORTED_IMAGE_MIME_TYPES = (
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
)
SUPPORTED_IMAGE_EXTENSIONS = (".gif", ".jpeg", ".jpg", ".png", ".webp")
SUPPORTED_RAG_EXTENSIONS = ALLOWED_RAG_UPLOAD_EXTENSIONS
IMAGE_UPLOAD_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VISION_IMAGE_MIME_TYPE_BY_EXTENSION = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
VISION_IMAGE_MIME_TYPES = frozenset(VISION_IMAGE_MIME_TYPE_BY_EXTENSION.values())
VISION_IMAGE_MIME_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
}
GENERIC_UPLOAD_MIME_TYPES = {"", "application/octet-stream"}
RAG_UPLOAD_MIME_TYPES = {
    "application/json",
    "application/octet-stream",
    "application/toml",
    "application/x-toml",
    "application/x-yaml",
    "application/yaml",
    "text/csv",
    "text/markdown",
    "text/plain",
    "text/x-python",
    "text/x-rst",
    "text/x-toml",
    "text/x-yaml",
}
RAG_UPLOAD_ACCEPT = {
    "text/plain": list(SUPPORTED_RAG_EXTENSIONS),
    "application/json": [".json"],
    "text/markdown": [".md"],
    "application/octet-stream": list(SUPPORTED_RAG_EXTENSIONS),
}


@dataclass(frozen=True)
class NormalizedUpload:
    """Validated uploaded bytes ready for image or RAG processing."""

    name: str
    mime_type: str
    kind: Literal["image", "rag"]
    data: bytes


def uploaded_file_mime_type(
    declared_mime: str | None,
    *,
    path: Path,
    name: str,
) -> str:
    """Resolve a normalized declared or inferred MIME type."""
    if isinstance(declared_mime, str) and "/" in declared_mime:
        return declared_mime.split(";", 1)[0].strip().lower()
    for candidate in (name, path.name):
        guessed_type, _ = mimetypes.guess_type(candidate)
        if guessed_type:
            return guessed_type.lower()
    return ""


def is_image_upload(path: Path, mime_type: str) -> bool:
    """Return whether an upload should be treated as an image candidate."""
    return mime_type.startswith("image/") or path.suffix.lower() in IMAGE_UPLOAD_EXTENSIONS


def provider_safe_image_mime_type(path: Path, mime_type: str) -> str | None:
    """Return a vision-provider-safe MIME type for an uploaded image."""
    normalized_mime = VISION_IMAGE_MIME_ALIASES.get(mime_type, mime_type)
    inferred_mime = VISION_IMAGE_MIME_TYPE_BY_EXTENSION.get(path.suffix.lower())
    if normalized_mime in VISION_IMAGE_MIME_TYPES:
        return normalized_mime if inferred_mime == normalized_mime else None
    if inferred_mime and normalized_mime in GENERIC_UPLOAD_MIME_TYPES:
        return inferred_mime
    return None


def normalize_upload(
    *,
    name: str,
    declared_mime: str | None,
    data: bytes,
) -> NormalizedUpload:
    """Validate one browser upload against ChainAgents' v1 allowlists."""
    upload_name = Path(name).name.strip()
    if not upload_name or upload_name in {".", ".."}:
        raise ValueError("Uploaded files must have a valid filename.")
    path = Path(upload_name)
    mime_type = uploaded_file_mime_type(declared_mime, path=path, name=upload_name)
    if len(data) > MAX_UPLOAD_FILE_BYTES:
        raise OverflowError(
            f"Uploaded file '{upload_name}' exceeds the 25 MB size limit."
        )
    if is_image_upload(path, mime_type):
        image_mime = provider_safe_image_mime_type(path, mime_type)
        if image_mime is None:
            raise ValueError(
                f"Uploaded image '{upload_name}' has an unsupported or mismatched format."
            )
        return NormalizedUpload(
            name=upload_name,
            mime_type=image_mime,
            kind="image",
            data=data,
        )
    if path.suffix.lower() not in SUPPORTED_RAG_EXTENSIONS:
        raise ValueError(f"Uploaded file '{upload_name}' has an unsupported extension.")
    if mime_type not in RAG_UPLOAD_MIME_TYPES and not mime_type.startswith("text/"):
        raise ValueError(f"Uploaded file '{upload_name}' has an unsupported MIME type.")
    return NormalizedUpload(
        name=upload_name,
        mime_type=mime_type,
        kind="rag",
        data=data,
    )


def image_content_part(upload: NormalizedUpload) -> dict[str, object]:
    """Build an OpenAI-compatible image data-URL part."""
    encoded = base64.b64encode(upload.data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{upload.mime_type};base64,{encoded}"},
    }


def prompt_with_images(
    content: str,
    *,
    image_names: tuple[str, ...],
    prompt_note: str = "",
) -> str:
    """Compose the shared image fallback and attachment note for an agent prompt."""
    if not image_names:
        return f"{content}{prompt_note}"
    prompt = content.strip() or "Extract any visible text from the attached image(s)."
    attached = ", ".join(f"`{name}`" for name in image_names)
    prompt = (
        f"{prompt}\n\n"
        f"Attached image file(s): {attached}. "
        "Use the image content directly when answering."
    )
    return f"{prompt}{prompt_note}"


def upload_result_prompt_note(added_files: tuple[str, ...]) -> str:
    """Build a prompt note describing newly indexed RAG files."""
    if not added_files:
        return ""
    names = ", ".join(f"`{name}`" for name in added_files)
    return f"\n\nThread knowledge uploaded for this request: {names}."
