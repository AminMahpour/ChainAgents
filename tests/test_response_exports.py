"""Tests for response export generation."""

from __future__ import annotations

import builtins
import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import response_exports


def test_build_pdf_bytes_uses_weasyprint_html_renderer(monkeypatch) -> None:
    """Verify that PDF exports are rendered through WeasyPrint."""
    html_calls: list[dict[str, Any]] = []

    class FakeHTML:
        def __init__(self, **kwargs: Any) -> None:
            html_calls.append(kwargs)

        def write_pdf(self) -> bytes:
            return b"%PDF-WEASYPRINT"

    monkeypatch.setitem(sys.modules, "weasyprint", SimpleNamespace(HTML=FakeHTML))

    pdf_bytes = response_exports.build_pdf_bytes("# Export\n\n- item")

    assert pdf_bytes == b"%PDF-WEASYPRINT"
    assert len(html_calls) == 1
    html = html_calls[0]["string"]
    assert "<h1>Export</h1>" in html
    assert "<li>item</li>" in html
    assert "@page" in html
    assert callable(html_calls[0]["url_fetcher"])
    with pytest.raises(ValueError, match="External resources are disabled"):
        html_calls[0]["url_fetcher"]("file:///etc/passwd")


def test_build_pdf_bytes_escapes_raw_html_before_rendering(monkeypatch) -> None:
    """Verify that response markdown cannot inject raw HTML into PDF exports."""
    html_calls: list[dict[str, Any]] = []

    class FakeHTML:
        def __init__(self, **kwargs: Any) -> None:
            html_calls.append(kwargs)

        def write_pdf(self) -> bytes:
            return b"%PDF-WEASYPRINT"

    monkeypatch.setitem(sys.modules, "weasyprint", SimpleNamespace(HTML=FakeHTML))

    response_exports.build_pdf_bytes("<script>alert('x')</script>\n\n**safe**")

    html = html_calls[0]["string"]
    assert "<script>" not in html
    assert "&lt;script&gt;alert" in html
    assert "<strong>safe</strong>" in html


def test_build_pdf_html_document_renders_pipe_tables() -> None:
    """Verify that response Markdown pipe tables render as HTML tables."""
    html = response_exports.build_pdf_html_document(
        "| Name | Value |\n"
        "| --- | --- |\n"
        "| Alpha | 1 |\n"
    )

    assert "<table>" in html
    assert "<th>Name</th>" in html
    assert "<td>Alpha</td>" in html
    assert "border-collapse" in html


def test_build_pdf_html_document_uses_compact_body_text() -> None:
    """Verify that PDF exports use compact body text."""
    html = response_exports.build_pdf_html_document("compact")

    assert "font-size: 9pt;" in html
    assert 'font-family: Georgia, "Times New Roman", Times, serif;' in html


def test_build_pdf_html_document_adds_page_number_footer() -> None:
    """Verify that PDF exports include page numbers in the page footer."""
    html = response_exports.build_pdf_html_document("page numbers")

    assert "@bottom-center" in html
    assert 'content: "Page " counter(page) " of " counter(pages);' in html
    assert "font-size: 8pt;" in html


def test_build_pdf_bytes_adds_homebrew_library_path_on_macos(
    monkeypatch,
    tmp_path,
) -> None:
    """Verify that macOS PDF exports can discover Homebrew native libraries."""
    html_calls: list[dict[str, Any]] = []
    homebrew_lib = tmp_path / "homebrew" / "lib"
    homebrew_lib.mkdir(parents=True)

    class FakeHTML:
        def __init__(self, **kwargs: Any) -> None:
            html_calls.append(kwargs)

        def write_pdf(self) -> bytes:
            return b"%PDF-WEASYPRINT"

    monkeypatch.setitem(sys.modules, "weasyprint", SimpleNamespace(HTML=FakeHTML))
    monkeypatch.setattr(response_exports.sys, "platform", "darwin")
    monkeypatch.setattr(response_exports, "HOMEBREW_LIBRARY_PATH", homebrew_lib)
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)

    response_exports.build_pdf_bytes("hello")

    assert html_calls
    assert os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == str(homebrew_lib)


def test_build_pdf_bytes_reports_missing_weasyprint_runtime(monkeypatch) -> None:
    """Verify that missing native WeasyPrint libraries produce an actionable error."""
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "weasyprint":
            raise OSError("cannot load library 'libgobject-2.0-0'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="brew install weasyprint"):
        response_exports.build_pdf_bytes("hello")


@pytest.mark.anyio
async def test_send_pdf_export_reports_generation_errors(monkeypatch) -> None:
    """Verify that PDF generation failures are sent to the Chainlit chat."""
    sent_messages: list[Any] = []

    class FakeUserSession:
        def __init__(self) -> None:
            self.values = {
                response_exports.RESPONSE_EXPORTS_SESSION_KEY: {
                    "message-1": {
                        "prompt": "hello",
                        "response_text": "answer",
                        "basename": "answer-message",
                    }
                }
            }

        def get(self, key: str) -> Any:
            return self.values.get(key)

        def set(self, key: str, value: Any) -> None:
            self.values[key] = value

    class FakeMessage:
        def __init__(self, content: str, author: str | None = None) -> None:
            self.content = content
            self.author = author

        async def send(self) -> None:
            sent_messages.append(self)

    def fail_pdf_generation(_text: str) -> bytes:
        raise RuntimeError("PDF export requires WeasyPrint native libraries.")

    monkeypatch.setattr(response_exports.cl, "user_session", FakeUserSession())
    monkeypatch.setattr(response_exports.cl, "Message", FakeMessage)
    monkeypatch.setattr(response_exports, "build_pdf_bytes", fail_pdf_generation)

    action = SimpleNamespace(forId=None, payload={"response_id": "message-1"})

    await response_exports.send_pdf_export(action)  # type: ignore[arg-type]

    assert len(sent_messages) == 1
    assert sent_messages[0].author == "System"
    assert "PDF export requires WeasyPrint native libraries" in sent_messages[0].content
