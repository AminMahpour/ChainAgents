"""Create Markdown and PDF exports for Chainlit response messages."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import chainlit as cl
from chainlit.element import Element, File, Pdf
from markdown_it import MarkdownIt


DOWNLOAD_MARKDOWN_ACTION = "download_response_markdown"
DOWNLOAD_PDF_ACTION = "download_response_pdf"
RESPONSE_EXPORTS_SESSION_KEY = "response_exports"
RESPONSE_EXPORT_ELEMENTS_SESSION_KEY = "response_export_elements"
DEFAULT_EXPORT_BASENAME = "response"
HOMEBREW_LIBRARY_PATH = Path("/opt/homebrew/lib")
PDF_EXPORT_DEPENDENCY_ERROR = (
    "PDF export requires WeasyPrint and its native runtime libraries. "
    "On macOS, install them with `brew install weasyprint`; on Linux, install "
    "the Pango packages listed in the WeasyPrint installation guide. Restart "
    "the app after installing the system libraries."
)
PDF_STYLES = """
@page {
  size: letter;
  margin: 0.75in;

  @bottom-center {
    color: #6b7280;
    content: "Page " counter(page) " of " counter(pages);
    font-family: Georgia, "Times New Roman", Times, serif;
    font-size: 8pt;
  }
}

body {
  color: #111827;
  font-family: Georgia, "Times New Roman", Times, serif;
  font-size: 9pt;
  line-height: 1.5;
}

h1,
h2,
h3 {
  line-height: 1.2;
  margin: 0 0 0.35em;
}

p,
ul,
ol,
pre,
blockquote {
  margin: 0 0 0.85em;
}

code,
pre {
  font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
}

pre {
  background: #f3f4f6;
  border: 1px solid #e5e7eb;
  border-radius: 4px;
  padding: 0.7em;
  white-space: pre-wrap;
}

blockquote {
  border-left: 3px solid #d1d5db;
  color: #4b5563;
  padding-left: 0.8em;
}

table {
  border-collapse: collapse;
  margin: 0 0 0.85em;
  width: 100%;
}

th,
td {
  border: 1px solid #d1d5db;
  padding: 0.35em 0.5em;
  text-align: left;
  vertical-align: top;
}

th {
  background: #f3f4f6;
  font-weight: 600;
}
"""
PDF_MARKDOWN_RENDERER = MarkdownIt("commonmark", {"html": False}).enable("table")


def attach_response_export_actions(
    message: cl.Message,
    *,
    prompt: str,
    response_text: str,
) -> None:
    """Attach response export actions.

    Args:
        message: Chainlit message or LangChain message to process.
        prompt: The prompt value.
        response_text: The response text value.
    """
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
    """Send markdown export.

    Args:
        action: The action value.
    """
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
    """Send PDF export.

    Args:
        action: The action value.
    """
    export = response_export_for_action(action)
    if export is None:
        await _send_export_unavailable_message()
        return

    try:
        pdf_content = build_pdf_bytes(export["response_text"])
    except RuntimeError as exc:
        await cl.Message(content=str(exc), author="System").send()
        return

    element = Pdf(
        name=f"{export['basename']}.pdf",
        content=pdf_content,
        display="inline",
    )
    await _send_export_element(
        action=action,
        export_kind="pdf",
        element=element,
    )


def response_export_for_action(action: cl.Action) -> dict[str, str] | None:
    """Return the stored response export for a Chainlit action.

    Args:
        action: The action value.

    Returns:
        The stored response export for a Chainlit action.
    """
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
    """Extract the response message ID from a Chainlit action.

    Args:
        action: The action value.

    Returns:
        The response message ID, or None when it cannot be determined.
    """
    payload = action.payload if isinstance(action.payload, dict) else {}
    message_id = str(action.forId or payload.get("response_id") or "").strip()
    return message_id


def suggested_export_basename(prompt: str, message_id: str) -> str:
    """Suggest export basename.

    Args:
        prompt: The prompt value.
        message_id: Message identifier.

    Returns:
        The suggested value.
    """
    source = next((line.strip() for line in prompt.splitlines() if line.strip()), "")
    slug = re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")
    if not slug:
        slug = DEFAULT_EXPORT_BASENAME
    slug = slug[:40].strip("-") or DEFAULT_EXPORT_BASENAME
    return f"{slug}-{message_id[:8]}"


def build_pdf_bytes(text: str) -> bytes:
    """Build PDF bytes.

    Args:
        text: Text content to process.

    Returns:
        The constructed pdf bytes.
    """
    _prepare_weasyprint_environment()
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as exc:
        raise RuntimeError(PDF_EXPORT_DEPENDENCY_ERROR) from exc

    try:
        return HTML(
            string=build_pdf_html_document(text),
            url_fetcher=_blocked_pdf_url_fetcher,
        ).write_pdf()
    except OSError as exc:
        raise RuntimeError(PDF_EXPORT_DEPENDENCY_ERROR) from exc


def build_pdf_html_document(text: str) -> str:
    """Build the HTML document rendered into PDF.

    Args:
        text: Markdown response text to process.

    Returns:
        The HTML document rendered by WeasyPrint.
    """
    body = PDF_MARKDOWN_RENDERER.render(text.strip() or "\u00a0")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>{PDF_STYLES}</style>
</head>
<body>
{body}
</body>
</html>
"""


def _blocked_pdf_url_fetcher(url: str, *_args: object, **_kwargs: object) -> dict[str, str]:
    """Reject external resource fetches while rendering response PDFs.

    Args:
        url: The URL WeasyPrint attempted to fetch.
        _args: Positional arguments from WeasyPrint.
        _kwargs: Keyword arguments from WeasyPrint.

    Returns:
        Nothing; this function always raises.
    """
    raise ValueError(f"External resources are disabled for response PDF exports: {url}")


def _prepare_weasyprint_environment() -> None:
    """Set macOS library lookup defaults before importing WeasyPrint."""
    if sys.platform != "darwin" or not HOMEBREW_LIBRARY_PATH.exists():
        return

    homebrew_lib = str(HOMEBREW_LIBRARY_PATH)
    current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    paths = [path for path in current.split(":") if path]
    if homebrew_lib in paths:
        return

    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join([homebrew_lib, *paths])


async def _send_export_element(
    *,
    action: cl.Action,
    export_kind: str,
    element: Element,
) -> None:
    """Send a generated export file as a Chainlit element.

    Args:
        action: The action value.
        export_kind: The export kind value.
        element: The element value.
    """
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
    """Notify the user that no exportable response is available."""
    await cl.Message(
        content="That response is no longer available for download in this session.",
        author="System",
    ).send()


def _get_response_exports() -> dict[str, dict[str, str]]:
    """Return the session response export registry.

    Returns:
        The session response export registry.
    """
    raw_exports = cl.user_session.get(RESPONSE_EXPORTS_SESSION_KEY)
    if isinstance(raw_exports, dict):
        return {
            str(message_id): value
            for message_id, value in raw_exports.items()
            if isinstance(value, dict)
        }
    return {}


def _get_sent_export_elements() -> dict[str, dict[str, Element]]:
    """Return IDs of export elements already sent this session.

    Returns:
        IDs of export elements already sent this session.
    """
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
