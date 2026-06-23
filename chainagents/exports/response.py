"""Create Markdown and PDF exports for Chainlit response messages."""

from __future__ import annotations

import mimetypes
import os
import re
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import chainlit as cl
from chainlit.element import Element, File, Pdf
from markdown_it import MarkdownIt


DOWNLOAD_MARKDOWN_ACTION = "download_response_markdown"
DOWNLOAD_PDF_ACTION = "download_response_pdf"
RESPONSE_EXPORTS_SESSION_KEY = "response_exports"
RESPONSE_EXPORT_ELEMENTS_SESSION_KEY = "response_export_elements"
DEFAULT_EXPORT_BASENAME = "response"
DEEPAGENT_ARTIFACTS_DIRECTORY = Path(".files/deepagent")
MAX_GENERATED_FILE_ATTACHMENTS = 12
HOMEBREW_LIBRARY_PATH = Path("/opt/homebrew/lib")
PDF_EXPORT_DEPENDENCY_ERROR = (
    "PDF export requires WeasyPrint and its native runtime libraries. "
    "On macOS, install them with `brew install weasyprint`; on Linux, install "
    "the Pango packages listed in the WeasyPrint installation guide. Restart "
    "the app after installing the system libraries."
)
GENERATED_FILE_PATH_RE = re.compile(
    r"(?P<path>"
    r"(?:/workspace/|\.files/deepagent/|/[^`'\"<>\s)]*/\.files/deepagent/)"
    r"[^`'\"<>\s)]*"
    r")"
)
GENERATED_FILE_TRAILING_PUNCTUATION = ".,;:!?"
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
"""
PDF_MARKDOWN_RENDERER = MarkdownIt("commonmark", {"html": False}).enable("table")
PDF_TEXT_REWRITE_SKIP_TAGS = frozenset({"code", "kbd", "pre", "samp"})
PDF_HTML_TAG_RE = re.compile(r"(<[^>]+>)")
PDF_HTML_TAG_NAME_RE = re.compile(r"^<\s*/?\s*([a-zA-Z][a-zA-Z0-9-]*)")
MOJIBAKE_MARKERS = ("Â", "Ã", "â", "ð", "�")
PDF_SUBSCRIPT_CHARS = {
    **dict(zip("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789")),
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
    generated_file_paths: Iterable[str | Path] = (),
    project_root: Path | None = None,
) -> None:
    """Attach response export actions.

    Args:
        message: Chainlit message or LangChain message to process.
        prompt: The prompt value.
        response_text: The response text value.
        generated_file_paths: Local, workspace, or artifact paths created during the run.
        project_root: Project root used to resolve virtual workspace paths.
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

    generated_elements = generated_file_elements_from_text(
        response_text,
        generated_file_paths=generated_file_paths,
        project_root=project_root,
    )
    if generated_elements:
        existing_elements = list(getattr(message, "elements", []) or [])
        message.elements = [*existing_elements, *generated_elements]


def generated_file_elements_from_text(
    text: str,
    *,
    generated_file_paths: Iterable[str | Path] = (),
    project_root: Path | None = None,
) -> list[File]:
    """Return Chainlit file elements for generated files mentioned in text.

    Args:
        text: Response text that may mention generated file paths.
        generated_file_paths: Additional file paths captured from successful tool calls.
        project_root: Project root used to resolve virtual workspace paths.

    Returns:
        Downloadable Chainlit file elements for safe, existing generated files.
    """
    raw_paths = [
        str(path)
        for path in generated_file_paths
        if str(path).strip()
    ]
    raw_paths.extend(
        match.group("path")
        for match in GENERATED_FILE_PATH_RE.finditer(text)
        if match.group("path").strip()
    )

    return generated_file_elements_from_paths(raw_paths, project_root=project_root)


def generated_file_elements_from_paths(
    raw_paths: Iterable[str | Path],
    *,
    project_root: Path | None = None,
) -> list[File]:
    """Return Chainlit file elements for existing files under allowed routes.

    Args:
        raw_paths: Candidate generated file paths.
        project_root: Project root used to resolve virtual workspace paths.

    Returns:
        Downloadable Chainlit file elements.
    """
    root = (project_root or Path.cwd()).resolve()
    elements: list[File] = []
    seen: set[Path] = set()
    for raw_path in raw_paths:
        path = _resolve_generated_file_path(raw_path, project_root=root)
        if path is None or path in seen:
            continue
        seen.add(path)
        mime_type, _encoding = mimetypes.guess_type(path.name)
        elements.append(
            File(
                thread_id=_current_chainlit_thread_id(),
                name=path.name,
                path=path.as_posix(),
                display="inline",
                mime=mime_type,
            )
        )
        if len(elements) >= MAX_GENERATED_FILE_ATTACHMENTS:
            break
    return elements


def _resolve_generated_file_path(raw_path: str | Path, *, project_root: Path) -> Path | None:
    """Resolve one generated file path if it points to a safe existing file."""
    path_text = _clean_generated_file_path(raw_path)
    if not path_text:
        return None

    if path_text == "/workspace":
        return None
    if path_text.startswith("/workspace/"):
        candidate = project_root / path_text.removeprefix("/workspace/")
    elif path_text == DEEPAGENT_ARTIFACTS_DIRECTORY.as_posix():
        return None
    elif path_text.startswith(f"{DEEPAGENT_ARTIFACTS_DIRECTORY.as_posix()}/"):
        candidate = project_root / path_text
    else:
        candidate = Path(path_text)
        if not candidate.is_absolute():
            candidate = project_root / candidate

    try:
        resolved = candidate.resolve()
    except OSError:
        return None

    if not _is_relative_to(resolved, project_root):
        return None
    if not resolved.is_file():
        return None
    return resolved


def _clean_generated_file_path(raw_path: str | Path) -> str:
    """Normalize one generated file path token from tool args or Markdown text."""
    return str(raw_path).strip().strip("`'\"<>[]()").rstrip(
        GENERATED_FILE_TRAILING_PUNCTUATION
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether path is inside parent."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


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
    normalized_text = _normalize_pdf_markdown_text(text.strip() or "\u00a0")
    body = PDF_MARKDOWN_RENDERER.render(normalized_text)
    body = _rewrite_pdf_html_text_nodes(body)
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
