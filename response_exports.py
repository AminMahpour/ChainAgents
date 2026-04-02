from __future__ import annotations

import re
import textwrap

import chainlit as cl
from chainlit.element import Element, File, Pdf


DOWNLOAD_MARKDOWN_ACTION = "download_response_markdown"
DOWNLOAD_PDF_ACTION = "download_response_pdf"
RESPONSE_EXPORTS_SESSION_KEY = "response_exports"
RESPONSE_EXPORT_ELEMENTS_SESSION_KEY = "response_export_elements"
DEFAULT_EXPORT_BASENAME = "response"
PDF_PAGE_WIDTH = 612
PDF_PAGE_HEIGHT = 792
PDF_MARGIN = 50
PDF_FONT_SIZE = 10
PDF_LINE_HEIGHT = 12
PDF_MAX_CHARS_PER_LINE = 86
PDF_MAX_LINES_PER_PAGE = 57


def attach_response_export_actions(
    message: cl.Message,
    *,
    prompt: str,
    response_text: str,
) -> None:
    message_id = str(getattr(message, "id", "") or "").strip()
    if not message_id or not response_text.strip():
        return

    exports = _get_response_exports()
    exports[message_id] = {
        "prompt": prompt,
        "response_text": response_text,
        "basename": suggested_export_basename(prompt, message_id),
    }
    cl.user_session.set(RESPONSE_EXPORTS_SESSION_KEY, exports)

    message.actions = [
        cl.Action(
            name=DOWNLOAD_MARKDOWN_ACTION,
            payload={"response_id": message_id},
            label="Markdown",
            tooltip="Download this response as Markdown.",
            icon="download",
        ),
        cl.Action(
            name=DOWNLOAD_PDF_ACTION,
            payload={"response_id": message_id},
            label="PDF",
            tooltip="Download this response as a PDF.",
            icon="download",
        ),
    ]


async def send_markdown_export(action: cl.Action) -> None:
    export = response_export_for_action(action)
    if export is None:
        await _send_export_unavailable_message()
        return

    element = File(
        name=f"{export['basename']}.md",
        content=export["response_text"].encode("utf-8"),
        display="inline",
        mime="text/markdown",
    )
    await _send_export_element(
        action=action,
        export_kind="markdown",
        element=element,
    )


async def send_pdf_export(action: cl.Action) -> None:
    export = response_export_for_action(action)
    if export is None:
        await _send_export_unavailable_message()
        return

    element = Pdf(
        name=f"{export['basename']}.pdf",
        content=build_pdf_bytes(export["response_text"]),
        display="inline",
    )
    await _send_export_element(
        action=action,
        export_kind="pdf",
        element=element,
    )


def response_export_for_action(action: cl.Action) -> dict[str, str] | None:
    message_id = response_message_id_from_action(action)
    if not message_id:
        return None

    export = _get_response_exports().get(message_id)
    if not isinstance(export, dict):
        return None

    prompt = str(export.get("prompt", ""))
    response_text = str(export.get("response_text", ""))
    basename = str(export.get("basename", ""))
    if not response_text.strip():
        return None

    return {
        "message_id": message_id,
        "prompt": prompt,
        "response_text": response_text,
        "basename": basename or suggested_export_basename(prompt, message_id),
    }


def response_message_id_from_action(action: cl.Action) -> str:
    payload = action.payload if isinstance(action.payload, dict) else {}
    message_id = str(action.forId or payload.get("response_id") or "").strip()
    return message_id


def suggested_export_basename(prompt: str, message_id: str) -> str:
    source = next((line.strip() for line in prompt.splitlines() if line.strip()), "")
    slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")
    if not slug:
        slug = DEFAULT_EXPORT_BASENAME
    slug = slug[:40].strip("-") or DEFAULT_EXPORT_BASENAME
    return f"{slug}-{message_id[:8]}"


def build_pdf_bytes(text: str) -> bytes:
    wrapped_lines = _wrap_pdf_lines(text)
    pages = _chunk_lines(wrapped_lines or [""], PDF_MAX_LINES_PER_PAGE)

    font_object_id = 3
    next_object_id = 4
    page_object_ids: list[int] = []
    objects: dict[int, bytes] = {}

    for page_lines in pages:
        content_object_id = next_object_id
        page_object_id = next_object_id + 1
        next_object_id += 2

        content_stream = _build_pdf_content_stream(page_lines)
        objects[content_object_id] = _pdf_stream_object(content_stream)
        objects[page_object_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PDF_PAGE_WIDTH} {PDF_PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_object_id} 0 R >> >> "
            f"/Contents {content_object_id} 0 R >>"
        ).encode("latin-1")
        page_object_ids.append(page_object_id)

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{page_object_id} 0 R" for page_object_id in page_object_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_object_ids)} >>".encode(
        "latin-1"
    )
    objects[font_object_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"

    return _serialize_pdf(objects)


async def _send_export_element(
    *,
    action: cl.Action,
    export_kind: str,
    element: Element,
) -> None:
    message_id = response_message_id_from_action(action)
    if not message_id:
        await _send_export_unavailable_message()
        return

    sent_elements = _get_sent_export_elements()
    previous_element = sent_elements.get(message_id, {}).get(export_kind)
    if isinstance(previous_element, Element):
        await previous_element.remove()

    await element.send(for_id=message_id)

    sent_elements.setdefault(message_id, {})[export_kind] = element
    cl.user_session.set(RESPONSE_EXPORT_ELEMENTS_SESSION_KEY, sent_elements)


async def _send_export_unavailable_message() -> None:
    await cl.Message(
        content="That response is no longer available for download in this session.",
        author="System",
    ).send()


def _get_response_exports() -> dict[str, dict[str, str]]:
    raw_exports = cl.user_session.get(RESPONSE_EXPORTS_SESSION_KEY)
    if isinstance(raw_exports, dict):
        return {
            str(message_id): value
            for message_id, value in raw_exports.items()
            if isinstance(value, dict)
        }
    return {}


def _get_sent_export_elements() -> dict[str, dict[str, Element]]:
    raw_elements = cl.user_session.get(RESPONSE_EXPORT_ELEMENTS_SESSION_KEY)
    if not isinstance(raw_elements, dict):
        return {}

    normalized: dict[str, dict[str, Element]] = {}
    for message_id, element_map in raw_elements.items():
        if not isinstance(element_map, dict):
            continue
        typed_map = {
            str(export_kind): element
            for export_kind, element in element_map.items()
            if isinstance(element, Element)
        }
        if typed_map:
            normalized[str(message_id)] = typed_map
    return normalized


def _wrap_pdf_lines(text: str) -> list[str]:
    wrapper = textwrap.TextWrapper(
        width=PDF_MAX_CHARS_PER_LINE,
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=True,
        break_on_hyphens=False,
    )

    lines: list[str] = []
    for raw_line in text.splitlines():
        normalized = raw_line.expandtabs(4).rstrip()
        safe_line = normalized.encode("latin-1", "replace").decode("latin-1")
        if not safe_line:
            lines.append("")
            continue
        lines.extend(wrapper.wrap(safe_line) or [""])

    return lines


def _chunk_lines(lines: list[str], size: int) -> list[list[str]]:
    return [lines[index : index + size] for index in range(0, len(lines), size)] or [[]]


def _build_pdf_content_stream(lines: list[str]) -> bytes:
    start_x = PDF_MARGIN
    start_y = PDF_PAGE_HEIGHT - PDF_MARGIN
    commands = [
        "BT",
        f"/F1 {PDF_FONT_SIZE} Tf",
        f"{PDF_LINE_HEIGHT} TL",
        f"{start_x} {start_y} Td",
    ]
    for line in lines:
        commands.append(f"({_escape_pdf_text(line)}) Tj")
        commands.append("T*")
    commands.append("ET")
    return "\n".join(commands).encode("latin-1")


def _pdf_stream_object(content: bytes) -> bytes:
    header = f"<< /Length {len(content)} >>\nstream\n".encode("latin-1")
    return header + content + b"\nendstream"


def _escape_pdf_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _serialize_pdf(objects: dict[int, bytes]) -> bytes:
    object_ids = sorted(objects)
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}

    for object_id in object_ids:
        offsets[object_id] = len(pdf)
        pdf.extend(f"{object_id} 0 obj\n".encode("latin-1"))
        pdf.extend(objects[object_id])
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    max_object_id = object_ids[-1] if object_ids else 0
    pdf.extend(f"xref\n0 {max_object_id + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for object_id in range(1, max_object_id + 1):
        offset = offsets.get(object_id, 0)
        generation = "00000"
        in_use = "n" if object_id in offsets else "f"
        pdf.extend(f"{offset:010d} {generation} {in_use} \n".encode("latin-1"))

    pdf.extend(
        (
            f"trailer\n<< /Size {max_object_id + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("latin-1")
    )
    return bytes(pdf)
