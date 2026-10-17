"""Create Markdown and PDF exports for Chainlit response messages."""

from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import math
import os
import re
import sys
import threading
import unicodedata
import uuid
import weakref
from collections.abc import Iterable, Mapping
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait as wait_for_futures,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chainlit as cl
from chainlit.element import Element, File, Pdf
from markdown_it import MarkdownIt

from chainagents.exports.generated_files import GeneratedFileDescriptor
from chainagents.exports.pdf_images import (
    PDF_IMAGE_MAX_BYTES,
    PDF_IMAGE_MAX_PIXELS,
    PdfImageDownloadBudget,
    PdfImageError,
    PdfImageHostLimiter,
    PdfImageHostSlot,
    PdfImageResource,
    _pdf_image_log_identifier as _pdf_image_log_identifier,
    decode_pdf_data_image as _decode_pdf_data_image,
    download_pdf_image as _download_pdf_image,
    pdf_image_host_key,
)
from chainagents.runtime.types import ChainlitResponseActionConfig


DOWNLOAD_MARKDOWN_ACTION = "download_response_markdown"
DOWNLOAD_PDF_ACTION = "download_response_pdf"
RESPONSE_EXPORTS_SESSION_KEY = "response_exports"
RESPONSE_CONTEXT_METADATA_KEY = "chainagents_response_context"
RESTORED_RESPONSE_ACTIONS_CONNECTION_ATTR = "_chainagents_restored_response_action_ids"
RUN_RESPONSE_ACTION = "run_configured_response_action"
RESPONSE_EXPORT_ELEMENTS_SESSION_KEY = "response_export_elements"
DEFAULT_EXPORT_BASENAME = "response"
MAX_PDF_REMOTE_IMAGES = 20
MAX_PDF_REMOTE_IMAGE_BYTES = 25 * 1024 * 1024
MAX_PDF_REMOTE_IMAGE_PIXELS = 50_000_000
PDF_IMAGE_DOWNLOAD_BUDGET_SECONDS = 30.0
PDF_IMAGE_DOWNLOAD_CONCURRENCY = 4
PDF_IMAGE_PER_HOST_CONCURRENCY = 2
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
  font-family: Georgia, "Times New Roman", "Noto Serif", "DejaVu Serif",
    "Liberation Serif", Times, serif;
  font-size: 9pt;
  line-height: 1.5;
}

sub,
sup {
  font-size: 75%;
  line-height: 0;
}

img {
  display: block;
  height: auto;
  margin: 0 auto 0.85em;
  max-height: 8.25in;
  max-width: 100%;
  object-fit: contain;
}

.pdf-image-unavailable {
  background: #f3f4f6;
  border: 1px solid #d1d5db;
  color: #4b5563;
  display: block;
  margin: 0 0 0.85em;
  padding: 0.6em;
}

sub {
  vertical-align: sub;
}

sup {
  vertical-align: super;
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
  font-family: "SFMono-Regular", "Menlo", "Consolas", "Noto Sans Mono",
    "DejaVu Sans Mono", monospace;
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

.pdf-toc {
  break-after: page;
}

.pdf-toc-title {
  font-size: 14pt;
  font-weight: 600;
  margin: 0 0 0.6em;
}

.pdf-toc ol {
  list-style: none;
  margin: 0;
  padding: 0;
}

.pdf-toc li {
  margin: 0 0 0.25em;
}

.pdf-toc-level-2 {
  padding-left: 1.2em;
}

.pdf-toc-level-3 {
  padding-left: 2.4em;
}

.pdf-toc a {
  color: inherit;
  text-decoration: none;
}

.pdf-toc a::after {
  content: leader(".") target-counter(attr(href), page);
}
"""
PDF_MARKDOWN_RENDERER = MarkdownIt("commonmark", {"html": False}).enable("table")
# Responses with at least this many level 1-3 headings get a contents section.
PDF_TOC_MIN_HEADINGS = 3
PDF_TOC_HEADING_RE = re.compile(r"<h([1-3])>(.*?)</h\1>", re.DOTALL)
PDF_TOC_IMAGE_ALT_RE = re.compile(r'<img\b[^>]*?\balt="([^"]*)"[^>]*>', re.IGNORECASE)
PDF_TEXT_REWRITE_SKIP_TAGS = frozenset({"code", "kbd", "pre", "samp"})
PDF_HTML_TAG_RE = re.compile(r"(<[^>]+>)")
PDF_HTML_TAG_NAME_RE = re.compile(r"^<\s*/?\s*([a-zA-Z][a-zA-Z0-9-]*)")
PDF_IMAGE_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
PDF_IMAGE_ATTRIBUTE_RE = re.compile(
    r"\b(?P<name>src|alt)\s*=\s*\"(?P<value>[^\"]*)\"",
    re.IGNORECASE,
)
logger = logging.getLogger(__name__)
_PDF_RENDER_LOCK = threading.Lock()
_CHAINLIT_PDF_RENDER_SEMAPHORES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Semaphore
] = weakref.WeakKeyDictionary()
MOJIBAKE_MARKERS = ("Â", "Ã", "â", "ð", "�")
PDF_SUBSCRIPT_CHARS = {
    **dict(
        zip("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789", strict=True)
    ),
    "\u208a": "+",
    "\u208b": "-",
    "\u208c": "=",
    "\u208d": "(",
    "\u208e": ")",
    "\u2090": "a",
    "\u2091": "e",
    "\u2095": "h",
    "\u1d62": "i",
    "\u2c7c": "j",
    "\u2096": "k",
    "\u2097": "l",
    "\u2098": "m",
    "\u2099": "n",
    "\u2092": "o",
    "\u209a": "p",
    "\u1d63": "r",
    "\u209b": "s",
    "\u209c": "t",
    "\u1d64": "u",
    "\u1d65": "v",
    "\u2093": "x",
}
PDF_SUPERSCRIPT_CHARS = {
    "\u2070": "0",
    "\u00b9": "1",
    "\u00b2": "2",
    "\u00b3": "3",
    "\u2074": "4",
    "\u2075": "5",
    "\u2076": "6",
    "\u2077": "7",
    "\u2078": "8",
    "\u2079": "9",
    "\u207a": "+",
    "\u207b": "-",
    "\u207c": "=",
    "\u207d": "(",
    "\u207e": ")",
    "\u1d43": "a",
    "\u1d47": "b",
    "\u1d9c": "c",
    "\u1d48": "d",
    "\u1d49": "e",
    "\u1da0": "f",
    "\u1d4d": "g",
    "\u02b0": "h",
    "\u2071": "i",
    "\u02b2": "j",
    "\u1d4f": "k",
    "\u02e1": "l",
    "\u1d50": "m",
    "\u207f": "n",
    "\u1d52": "o",
    "\u1d56": "p",
    "\u02b3": "r",
    "\u02e2": "s",
    "\u1d57": "t",
    "\u1d58": "u",
    "\u1d5b": "v",
    "\u02b7": "w",
    "\u02e3": "x",
    "\u02b8": "y",
    "\u1dbb": "z",
    "\u1d2c": "A",
    "\u1d2e": "B",
    "\u1d30": "D",
    "\u1d31": "E",
    "\u1d33": "G",
    "\u1d34": "H",
    "\u1d35": "I",
    "\u1d36": "J",
    "\u1d37": "K",
    "\u1d38": "L",
    "\u1d39": "M",
    "\u1d3a": "N",
    "\u1d3c": "O",
    "\u1d3e": "P",
    "\u1d3f": "R",
    "\u1d40": "T",
    "\u1d41": "U",
    "\u2c7d": "V",
    "\u1d42": "W",
}


def _mojibake_variants(character: str, encoding: str) -> tuple[str, ...]:
    """Return mojibake forms for one character under the given encoding."""
    try:
        mojibake = character.encode("utf-8").decode(encoding)
    except UnicodeError:
        return ()
    if mojibake == character:
        return ()
    return (mojibake,)


COMMON_MOJIBAKE_REPAIR_CHARS = (
    "".join(PDF_SUBSCRIPT_CHARS)
    + "".join(PDF_SUPERSCRIPT_CHARS)
    + "\u00a0\u00a9\u00ae\u00b0\u00b1\u00b5\u00b7"
    + "\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u2122"
)
COMMON_MOJIBAKE_REPLACEMENTS: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {
            mojibake: character
            for character in COMMON_MOJIBAKE_REPAIR_CHARS
            for encoding in ("windows-1252", "latin-1")
            for mojibake in _mojibake_variants(character, encoding)
        }.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def attach_response_export_actions(
    message: cl.Message,
    *,
    prompt: str,
    response_text: str,
    generated_files: Iterable[GeneratedFileDescriptor] = (),
    response_actions: Iterable[ChainlitResponseActionConfig] = (),
    export_label: str = "",
) -> None:
    """Attach response export actions.

    Args:
        message: Chainlit message or LangChain message to process.
        prompt: The prompt value.
        response_text: The response text value.
        generated_files: Validated generated files attached to the message.
    """
    message_id = str(getattr(message, "id", "") or "").strip()
    if not message_id or not response_text.strip():
        return

    exports = _get_response_exports()
    exports[message_id] = {
        "prompt": prompt,
        "response_text": response_text,
        "basename": suggested_export_basename(export_label or prompt, message_id),
    }
    cl.user_session.set(RESPONSE_EXPORTS_SESSION_KEY, exports)

    metadata = dict(getattr(message, "metadata", None) or {})
    metadata[RESPONSE_CONTEXT_METADATA_KEY] = {
        "version": 1,
        "prompt": prompt,
        "export_label": export_label,
    }
    message.metadata = metadata
    message.actions = response_actions_for_message(message_id, response_actions)

    generated_elements = generated_file_elements(generated_files)
    if generated_elements:
        existing_elements = list(getattr(message, "elements", []) or [])
        combined_elements: list[Element] = [*existing_elements, *generated_elements]
        # cl.Message.elements is typed List[ElementBased], an unbound TypeVar in
        # chainlit's stub; cast bridges that stub quirk without a runtime effect.
        message.elements = cast("list[Any]", combined_elements)


def response_actions_for_message(
    message_id: str,
    response_actions: Iterable[ChainlitResponseActionConfig],
) -> list[cl.Action]:
    """Build export and configured actions in their UI order."""
    actions = [
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
    actions.extend(
        cl.Action(
            name=RUN_RESPONSE_ACTION,
            payload={"response_id": message_id, "action_name": configured.name},
            label=configured.label,
            tooltip=configured.tooltip or "",
            icon=configured.icon,
        )
        for configured in response_actions
    )
    return actions


@dataclass(frozen=True)
class ResolvedResponseAction:
    """A configured request resolved against the clicked response."""

    prompt: str
    label: str


def expand_response_action_prompt(
    template: str, *, prompt: str, response_text: str
) -> str:
    """Expand supported tokens once, without interpreting inserted braces."""
    return re.sub(
        r"\{(prompt|response)\}",
        lambda match: prompt if match.group(1) == "prompt" else response_text,
        template,
    )


def resolve_response_action(
    action: cl.Action,
    response_actions: Iterable[ChainlitResponseActionConfig],
) -> ResolvedResponseAction | None:
    """Resolve the action using server-owned configuration and response text."""
    payload = action.payload if isinstance(action.payload, dict) else {}
    response_id = payload.get("response_id")
    action_name = payload.get("action_name")
    if not isinstance(response_id, str) or not response_id:
        return None
    if not isinstance(action_name, str) or not action_name:
        return None
    if action.forId and action.forId != response_id:
        return None
    export = _get_response_exports().get(response_id)
    if not isinstance(export, dict) or not str(export.get("response_text", "")).strip():
        return None
    for configured in response_actions:
        if configured.name == action_name:
            return ResolvedResponseAction(
                prompt=expand_response_action_prompt(
                    configured.prompt,
                    prompt=str(export.get("prompt", "")),
                    response_text=str(export["response_text"]),
                ),
                label=configured.label,
            )
    return None


def restore_response_export_actions(
    thread: Mapping[str, Any],
    *,
    response_actions: Iterable[ChainlitResponseActionConfig] = (),
) -> list[cl.Action]:
    """Restore actions for saved final responses with reliable context."""
    exports = _get_response_exports()
    # Chainlit persists user_session in thread metadata, so deduplicate per connection.
    connection = cl.context.session
    emitted = getattr(connection, RESTORED_RESPONSE_ACTIONS_CONNECTION_ATTR, None)
    if not isinstance(emitted, set):
        emitted = set()
    restored: list[cl.Action] = []
    for step in thread.get("steps", ()):
        if not isinstance(step, dict) or step.get("type") != "assistant_message":
            continue
        message_id = str(step.get("id") or "").strip()
        response_text = str(step.get("output") or "")
        if not message_id or not response_text.strip():
            continue
        metadata = step.get("metadata")
        context = (
            metadata.get(RESPONSE_CONTEXT_METADATA_KEY)
            if isinstance(metadata, dict)
            else None
        )
        if isinstance(context, dict) and context.get("version") == 1:
            prompt = context.get("prompt")
            export_label = context.get("export_label", "")
            if not isinstance(prompt, str) or not isinstance(export_label, str):
                continue
            exports[message_id] = {
                "prompt": prompt,
                "response_text": response_text,
                "basename": suggested_export_basename(export_label or prompt, message_id),
            }
        elif message_id not in exports:
            continue
        for action in response_actions_for_message(message_id, response_actions):
            action.id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{message_id}:{action.name}:{action.payload}"))
            if action.id in emitted:
                continue
            action.forId = message_id
            restored.append(action)
            emitted.add(action.id)
    cl.user_session.set(RESPONSE_EXPORTS_SESSION_KEY, exports)
    setattr(connection, RESTORED_RESPONSE_ACTIONS_CONNECTION_ATTR, emitted)
    return restored


def generated_file_elements(
    generated_files: Iterable[GeneratedFileDescriptor],
) -> list[File]:
    """Return downloadable Chainlit file elements for validated generated files.

    Args:
        generated_files: Descriptors resolved by the strict generated-file rule.

    Returns:
        Chainlit file elements for descriptors that carry a local path.
    """
    return [
        File(
            thread_id=_current_chainlit_thread_id(),
            name=descriptor.name,
            path=descriptor.path.as_posix(),
            display="inline",
            mime=descriptor.mime_type,
        )
        for descriptor in generated_files
        if descriptor.path is not None
    ]


def _current_chainlit_thread_id() -> str:
    """Return the active Chainlit thread ID when one is available."""
    try:
        return str(cl.context.session.thread_id)
    except Exception:
        return ""


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
        pdf_content = await _build_chainlit_pdf_bytes(export["response_text"])
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
    with _PDF_RENDER_LOCK:
        return _build_pdf_bytes_unlocked(text)


def _build_pdf_bytes_unlocked(text: str) -> bytes:
    """Render one PDF while the process-wide render lock is held."""
    _prepare_weasyprint_environment()
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as exc:
        raise RuntimeError(PDF_EXPORT_DEPENDENCY_ERROR) from exc

    try:
        document = build_pdf_html_document(text)
        document, resources = _prepare_pdf_image_resources(document)
        return HTML(string=document, url_fetcher=_pdf_url_fetcher(resources)).write_pdf()
    except OSError as exc:
        raise RuntimeError(PDF_EXPORT_DEPENDENCY_ERROR) from exc


def build_pdf_html_document(text: str) -> str:
    """Build the HTML document rendered into PDF.

    Args:
        text: Markdown response text to process.

    Returns:
        The HTML document rendered by WeasyPrint.
    """
    normalized_text = _normalize_pdf_markdown_text(text.strip() or "\u00a0")
    body = PDF_MARKDOWN_RENDERER.render(normalized_text)
    body = _rewrite_pdf_html_text_nodes(body)
    body = _with_pdf_table_of_contents(body)
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


def _with_pdf_table_of_contents(body: str) -> str:
    """Prefix a linked contents section when the body has enough headings.

    Level 1-3 headings get numbered ids (safe for duplicate and non-Latin
    titles); the contents entries link to them, WeasyPrint fills in their page
    numbers, and the ids also give the PDF a bookmark outline. The heading text
    is already HTML-escaped by the Markdown renderer, so only tags are stripped.
    """
    headings = PDF_TOC_HEADING_RE.findall(body)
    if len(headings) < PDF_TOC_MIN_HEADINGS:
        return body
    counter = iter(range(1, len(headings) + 1))
    entries: list[str] = []

    def add_id(match: re.Match[str]) -> str:
        level, inner = match.group(1), match.group(2)
        anchor = f"toc-{next(counter)}"
        # Images keep their (already escaped) alt text; other tags are dropped.
        title = re.sub(r"<[^>]+>", "", PDF_TOC_IMAGE_ALT_RE.sub(r"\1", inner)).strip()
        title = " ".join(title.split()) or f"Section {anchor.removeprefix('toc-')}"
        entries.append(
            f'<li class="pdf-toc-level-{level}"><a href="#{anchor}">{title}</a></li>'
        )
        return f'<h{level} id="{anchor}">{inner}</h{level}>'

    body = PDF_TOC_HEADING_RE.sub(add_id, body)
    contents = (
        '<nav class="pdf-toc"><p class="pdf-toc-title">Contents</p><ol>'
        + "".join(entries)
        + "</ol></nav>\n"
    )
    return contents + body


def _normalize_pdf_markdown_text(text: str) -> str:
    """Return Markdown text that is safer for WeasyPrint text shaping."""
    repaired_text = _repair_common_mojibake(text)
    normalized_text = unicodedata.normalize("NFC", repaired_text)
    return "".join(_pdf_safe_text_char(char) for char in normalized_text)


def _repair_common_mojibake(text: str) -> str:
    """Repair common UTF-8 text that was decoded as Windows-1252 or Latin-1."""
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text

    text = _repair_common_mojibake_sequences(text)
    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text

    original_score = _mojibake_score(text)
    candidates = []
    for encoding in ("windows-1252", "latin-1"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except UnicodeError:
            continue
        candidates.append(candidate)

    if not candidates:
        return text

    best_candidate = min(candidates, key=_mojibake_score)
    if _mojibake_score(best_candidate) >= original_score:
        return text
    return best_candidate


def _repair_common_mojibake_sequences(text: str) -> str:
    """Repair known mojibake sequences without rewriting the entire string."""
    repaired_text = text
    for mojibake, character in COMMON_MOJIBAKE_REPLACEMENTS:
        repaired_text = repaired_text.replace(mojibake, character)
    return repaired_text


def _mojibake_score(text: str) -> int:
    """Return a simple score for characters usually found in mojibake."""
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS)


def _rewrite_pdf_html_text_nodes(html: str) -> str:
    """Rewrite rendered HTML text nodes while preserving generated tags."""
    parts: list[str] = []
    skipped_tag_depth = 0
    for token in PDF_HTML_TAG_RE.split(html):
        if not token:
            continue
        if token.startswith("<") and token.endswith(">"):
            tag_name = _pdf_html_tag_name(token)
            if tag_name in PDF_TEXT_REWRITE_SKIP_TAGS:
                if token.startswith("</"):
                    skipped_tag_depth = max(0, skipped_tag_depth - 1)
                elif not token.rstrip().endswith("/>"):
                    skipped_tag_depth += 1
            parts.append(token)
            continue

        if skipped_tag_depth:
            parts.append(token)
        else:
            parts.append(_rewrite_pdf_text_segment(token))

    return "".join(parts)


def _pdf_html_tag_name(tag: str) -> str:
    """Return a lowercase tag name from a rendered HTML tag token."""
    match = PDF_HTML_TAG_NAME_RE.match(tag)
    if match is None:
        return ""
    return match.group(1).lower()


def _rewrite_pdf_text_segment(text: str) -> str:
    """Rewrite one rendered HTML text segment for PDF rendering."""
    parts: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in PDF_SUBSCRIPT_CHARS:
            index = _append_pdf_script_run(
                parts,
                text,
                index,
                tag="sub",
                replacements=PDF_SUBSCRIPT_CHARS,
            )
            continue
        if char in PDF_SUPERSCRIPT_CHARS:
            index = _append_pdf_script_run(
                parts,
                text,
                index,
                tag="sup",
                replacements=PDF_SUPERSCRIPT_CHARS,
            )
            continue

        parts.append(_pdf_safe_text_char(char))
        index += 1

    return "".join(parts)


def _append_pdf_script_run(
    parts: list[str],
    text: str,
    index: int,
    *,
    tag: str,
    replacements: dict[str, str],
) -> int:
    """Append one consecutive subscript or superscript text run."""
    replacement_chars: list[str] = []
    while index < len(text) and text[index] in replacements:
        replacement_chars.append(replacements[text[index]])
        index += 1

    replacement_text = "".join(replacement_chars)
    parts.append(f"<{tag}>{replacement_text}</{tag}>")
    return index


def _pdf_safe_text_char(char: str) -> str:
    """Return a character that will not become a PDF replacement glyph."""
    if char == "\ufeff":
        return ""
    if char == "\ufffd":
        return "?"

    codepoint = ord(char)
    if _is_unicode_noncharacter(codepoint):
        return "?"

    category = unicodedata.category(char)
    if category in {"Cs", "Cn"}:
        return "?"
    if category == "Cc" and char not in {"\n", "\r", "\t"}:
        return " "
    return char


def _is_unicode_noncharacter(codepoint: int) -> bool:
    """Return True when a codepoint is permanently reserved as a noncharacter."""
    return 0xFDD0 <= codepoint <= 0xFDEF or (codepoint & 0xFFFE) == 0xFFFE


def _prepare_pdf_image_resources(
    document: str,
) -> tuple[str, dict[str, PdfImageResource]]:
    """Download unique public images and replace failures with placeholders."""
    import time

    image_tags = list(PDF_IMAGE_TAG_RE.finditer(document))
    urls: list[str] = []
    failures: set[str] = set()
    for match in image_tags:
        attributes = _pdf_image_attributes(match.group(0))
        source = attributes.get("src", "")
        if source.lower().startswith(
            ("http://", "https://", "data:image/")
        ) and source not in urls:
            urls.append(source)
        elif not source.lower().startswith(("http://", "https://", "data:image/")):
            failures.add(source)

    selected = urls[:MAX_PDF_REMOTE_IMAGES]
    failures.update(urls[MAX_PDF_REMOTE_IMAGES:])
    deadline = time.monotonic() + PDF_IMAGE_DOWNLOAD_BUDGET_SECONDS
    budget = PdfImageDownloadBudget(
        deadline,
        MAX_PDF_REMOTE_IMAGE_BYTES,
        MAX_PDF_REMOTE_IMAGE_PIXELS,
    )
    remote_urls = [url for url in selected if not url.lower().startswith("data:image/")]
    outcomes: dict[str, PdfImageResource | BaseException] = {}
    # Scheduling happens here rather than in the workers: a URL is submitted
    # only when its host has a free slot, so queued images for a busy host
    # never hold a worker that another host's image could use.
    queued = list(remote_urls)
    host_limiter = PdfImageHostLimiter(PDF_IMAGE_PER_HOST_CONCURRENCY)
    in_flight: dict[Future[PdfImageResource], tuple[str, PdfImageHostSlot]] = {}
    # Worker threads start lazily on submit, so data-only documents spawn none.
    executor = ThreadPoolExecutor(
        max_workers=PDF_IMAGE_DOWNLOAD_CONCURRENCY,
        thread_name_prefix="pdf-image",
    )

    def submit_ready() -> None:
        index = 0
        while index < len(queued) and len(in_flight) < PDF_IMAGE_DOWNLOAD_CONCURRENCY:
            url = queued[index]
            host = pdf_image_host_key(url)
            # Non-blocking: a busy host (including one busy with redirected
            # downloads from other origins) is skipped, never waited on here.
            if not host_limiter.try_acquire(host):
                index += 1
                continue
            del queued[index]
            # The task owns the slot from here and releases it when it ends.
            slot = PdfImageHostSlot(host_limiter, host)
            future = executor.submit(
                _download_pdf_image_task,
                url,
                deadline=deadline,
                budget=budget,
                host_slot=slot,
            )
            in_flight[future] = (url, slot)

    try:
        submit_ready()
        for url in selected:
            if url.lower().startswith("data:image/"):
                try:
                    outcomes[url] = _decode_pdf_data_image_with_budget(url, budget)
                except PdfImageError as exc:
                    outcomes[url] = exc
        while in_flight:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                break
            done, _pending = wait_for_futures(
                in_flight,
                timeout=remaining_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                break
            for future in done:
                url, _slot = in_flight.pop(future)
                error = future.exception()
                outcomes[url] = future.result() if error is None else error
            submit_ready()
    finally:
        # Do not wait for stragglers: each stays bounded by its own image
        # deadline and none of them writes to the resources collected below.
        executor.shutdown(wait=False, cancel_futures=True)
        # A task cancelled before it started never took ownership of its slot.
        for future, (_url, slot) in in_flight.items():
            if future.cancelled():
                slot.release()
    for url in remote_urls:
        outcomes.setdefault(url, PdfImageError("image download budget expired"))

    resources: dict[str, PdfImageResource] = {}
    for url in selected:
        outcome = outcomes[url]
        if isinstance(outcome, PdfImageResource):
            resources[url] = outcome
            continue
        failures.add(url)
        if isinstance(outcome, PdfImageError):
            logger.warning(
                "Unable to include PDF image %s: %s",
                _pdf_image_log_identifier(url),
                outcome,
            )
        else:
            logger.warning(
                "Unable to include PDF image %s: unexpected %s",
                _pdf_image_log_identifier(url),
                type(outcome).__name__,
            )

    def replace_failed_image(match: re.Match[str]) -> str:
        tag = match.group(0)
        attributes = _pdf_image_attributes(tag)
        source = attributes.get("src", "")
        if source not in failures:
            return tag
        alt = attributes.get("alt", "image").strip() or "image"
        if source.lower().startswith("data:image/"):
            source_label = "(embedded image)"
        else:
            source_label = f'(<a href="{html.escape(source, quote=True)}">source</a>)'
        return (
            '<span class="pdf-image-unavailable">'
            f"Image unavailable: {html.escape(alt)} "
            f"{source_label}"
            "</span>"
        )

    return PDF_IMAGE_TAG_RE.sub(replace_failed_image, document), resources


def _pdf_image_attributes(tag: str) -> dict[str, str]:
    """Return decoded src and alt attributes from generated Markdown HTML."""
    return {
        match.group("name").lower(): html.unescape(match.group("value"))
        for match in PDF_IMAGE_ATTRIBUTE_RE.finditer(tag)
    }


def _decode_pdf_data_image_with_budget(
    url: str,
    budget: PdfImageDownloadBudget,
) -> PdfImageResource:
    """Decode one data URI and charge its retained bytes and pixels."""
    max_bytes = min(PDF_IMAGE_MAX_BYTES, budget.remaining_bytes)
    if max_bytes <= 0:
        raise PdfImageError("PDF image download budget exceeded")
    max_pixels = min(PDF_IMAGE_MAX_PIXELS, budget.remaining_pixels)
    if max_pixels <= 0:
        raise PdfImageError("PDF image pixel budget exceeded")
    resource = _decode_pdf_data_image(
        url,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
        budget=budget,
    )
    try:
        budget.consume(len(resource.content))
    except PdfImageError:
        budget.refund_pixels(resource.pixel_count)
        raise
    return resource


def _download_pdf_image_task(
    url: str,
    *,
    deadline: float,
    budget: PdfImageDownloadBudget,
    host_slot: PdfImageHostSlot,
) -> PdfImageResource:
    """Download one remote image, charge its retained bytes, release its slot."""
    try:
        return _download_pdf_image_in_slot(
            url, deadline=deadline, budget=budget, host_slot=host_slot
        )
    finally:
        host_slot.release()


def _download_pdf_image_in_slot(
    url: str,
    *,
    deadline: float,
    budget: PdfImageDownloadBudget,
    host_slot: PdfImageHostSlot,
) -> PdfImageResource:
    import time

    if time.monotonic() >= deadline:
        raise PdfImageError("image download budget expired")
    # Cap by bytes not yet transferred rather than by remaining_bytes, which
    # already excludes other images' active reservations: those may still be
    # handed back, and reserve()/consume() enforce the shared total anyway.
    max_bytes = min(PDF_IMAGE_MAX_BYTES, budget.unconsumed_bytes)
    if max_bytes <= 0:
        raise PdfImageError("PDF image download budget exceeded")
    if budget.remaining_pixels <= 0:
        raise PdfImageError("PDF image pixel budget exceeded")
    # The aggregate pixel total is enforced when validation reserves pixels,
    # so the per-image cap is not frozen at a start-time snapshot.
    max_pixels = PDF_IMAGE_MAX_PIXELS
    image_budget = _CountingPdfImageBudget(budget)
    resource = _download_pdf_image(
        url,
        deadline=deadline,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
        budget=image_budget,
        host_slot=host_slot,
    )
    # Decompression and normalization can retain more bytes than were
    # transferred; charge the difference against the shared budget.
    retained_difference = len(resource.content) - image_budget.consumed_bytes
    if retained_difference > 0:
        try:
            budget.consume(retained_difference)
        except PdfImageError:
            budget.refund_pixels(image_budget.consumed_pixels)
            raise
    return resource


class _CountingPdfImageBudget(PdfImageDownloadBudget):
    """Per-image view of the shared export budget that counts its own usage."""

    def __init__(self, shared: PdfImageDownloadBudget) -> None:
        super().__init__(shared.deadline, 0, 0)
        self._shared = shared
        self.consumed_bytes = 0
        self.consumed_pixels = 0

    @property
    def remaining_bytes(self) -> int:
        return self._shared.remaining_bytes

    @property
    def remaining_pixels(self) -> int:
        return self._shared.remaining_pixels

    def consume(self, size: int) -> None:
        self._shared.consume(size)
        self.consumed_bytes += size

    def reserve(self, size: int, *, deadline: float = math.inf) -> None:
        self._shared.reserve(size, deadline=deadline)
        self.consumed_bytes += size

    def draw_reservation(self, size: int) -> None:
        self._shared.draw_reservation(size)

    @property
    def unconsumed_bytes(self) -> int:
        return self._shared.unconsumed_bytes

    def settle_reservation(self, unused: int) -> None:
        self._shared.settle_reservation(unused)
        self.consumed_bytes -= unused

    def consume_pixels(self, count: int) -> None:
        self._shared.consume_pixels(count)
        self.consumed_pixels += count

    def refund_pixels(self, count: int) -> None:
        self._shared.refund_pixels(count)
        self.consumed_pixels -= count


def _pdf_url_fetcher(resources: dict[str, PdfImageResource]) -> object:
    """Return a WeasyPrint fetcher limited to validated in-memory images."""
    from weasyprint.urls import URLFetcher, URLFetcherResponse

    class PdfResourceFetcher(URLFetcher):
        def __init__(self) -> None:
            super().__init__(allowed_protocols=set(), fail_on_errors=False)

        def fetch(self, url: str, headers: object = None) -> object:
            resource = resources.get(url)
            if resource is not None:
                return URLFetcherResponse(
                    url,
                    resource.content,
                    {"Content-Type": resource.mime_type},
                )
            return super().fetch(url, headers)

    return PdfResourceFetcher()


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


async def _build_chainlit_pdf_bytes(text: str) -> bytes:
    """Gate Chainlit PDF work before submitting it to the shared executor."""
    loop = asyncio.get_running_loop()
    semaphore = _CHAINLIT_PDF_RENDER_SEMAPHORES.get(loop)
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)
        _CHAINLIT_PDF_RENDER_SEMAPHORES[loop] = semaphore
    async with semaphore:
        worker = asyncio.create_task(asyncio.to_thread(build_pdf_bytes, text))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await worker
            raise


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
