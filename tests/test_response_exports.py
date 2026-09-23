"""Tests for response export generation."""

from __future__ import annotations

import builtins
import gzip
from io import BytesIO
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from pypdf import PdfReader

import response_exports
import chainagents.exports.pdf_images as pdf_images


def _png_bytes(*, width: int = 48, height: int = 32) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), "#2563eb").save(output, format="PNG")
    return output.getvalue()


def test_generated_file_elements_from_text_includes_workspace_and_artifacts(
    tmp_path: Path,
) -> None:
    """Verify generated response file paths become downloadable Chainlit files."""
    report_path = tmp_path / "reports" / "summary.csv"
    chart_path = tmp_path / ".files" / "outputs" / "charts" / "plot.png"
    report_path.parent.mkdir(parents=True)
    chart_path.parent.mkdir(parents=True)
    report_path.write_text("name,value\nalpha,1\n", encoding="utf-8")
    chart_path.write_bytes(b"\x89PNG\r\n")

    elements = response_exports.generated_file_elements_from_text(
        "Created `/workspace/reports/summary.csv` and `.files/outputs/charts/plot.png`.",
        project_root=tmp_path,
    )

    assert [element.name for element in elements] == ["summary.csv", "plot.png"]
    assert [element.path for element in elements] == [
        report_path.as_posix(),
        chart_path.as_posix(),
    ]
    assert [element.mime for element in elements] == ["text/csv", "image/png"]


def test_generated_file_elements_from_text_resolves_absolute_workspace_artifacts(
    monkeypatch,
) -> None:
    """Verify absolute artifact paths under /workspace are not remapped twice."""
    project_root = Path("/workspace/ChainAgents")
    artifact_path = project_root / ".files" / "outputs" / "plot.png"

    def fake_is_file(path: Path) -> bool:
        return path == artifact_path

    monkeypatch.setattr(Path, "is_file", fake_is_file)

    elements = response_exports.generated_file_elements_from_text(
        "Created `/workspace/ChainAgents/.files/outputs/plot.png`.",
        project_root=project_root,
    )

    assert [element.name for element in elements] == ["plot.png"]
    assert [element.path for element in elements] == [artifact_path.as_posix()]


def test_generated_file_elements_from_text_ignores_unsafe_or_unavailable_paths(
    tmp_path: Path,
) -> None:
    """Verify only existing generated files under allowed routes are downloadable."""
    directory_path = tmp_path / "reports"
    directory_path.mkdir()
    outside_path = tmp_path.parent / "outside.txt"
    outside_path.write_text("secret", encoding="utf-8")

    elements = response_exports.generated_file_elements_from_text(
        "\n".join(
            [
                "`/workspace/reports`",
                "`/workspace/missing.txt`",
                "`/workspace/../outside.txt`",
                outside_path.as_posix(),
            ]
        ),
        project_root=tmp_path,
    )

    assert elements == []


def test_build_pdf_bytes_uses_weasyprint_html_renderer(monkeypatch) -> None:
    """Verify that PDF exports are rendered through WeasyPrint."""
    html_calls: list[dict[str, Any]] = []

    class FakeHTML:
        def __init__(self, **kwargs: Any) -> None:
            html_calls.append(kwargs)

        def write_pdf(self) -> bytes:
            return b"%PDF-WEASYPRINT"

    monkeypatch.setitem(sys.modules, "weasyprint", SimpleNamespace(HTML=FakeHTML))
    monkeypatch.setattr(
        response_exports,
        "_pdf_url_fetcher",
        lambda _resources: response_exports._blocked_pdf_url_fetcher,
    )

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


def test_build_pdf_bytes_embeds_downloaded_remote_image(monkeypatch) -> None:
    """Dropping remote image resources must not silently remove response figures."""
    image_url = "https://images.example.test/figure.png"
    monkeypatch.setattr(
        response_exports,
        "_download_pdf_image",
        lambda url, **_kwargs: response_exports.PdfImageResource(
            content=_png_bytes(), mime_type="image/png"
        ),
    )

    pdf_bytes = response_exports.build_pdf_bytes(
        f"# Result\n\n![Dose response]({image_url})"
    )

    page = PdfReader(BytesIO(pdf_bytes)).pages[0]
    assert len(page.images) == 1
    assert "Dose response" not in (page.extract_text() or "")


def test_build_pdf_bytes_downloads_repeated_image_once(monkeypatch) -> None:
    """Repeated Markdown references must reuse one bounded download."""
    calls: list[str] = []
    image_url = "https://images.example.test/repeated.png"

    def download(url: str, **_kwargs: Any) -> Any:
        calls.append(url)
        return response_exports.PdfImageResource(
            content=_png_bytes(), mime_type="image/png"
        )

    monkeypatch.setattr(response_exports, "_download_pdf_image", download)

    response_exports.build_pdf_bytes(
        f"![First]({image_url})\n\n![Second]({image_url})"
    )

    assert calls == [image_url]


def test_build_pdf_bytes_keeps_document_when_remote_image_fails(monkeypatch) -> None:
    """One unavailable image must not abort the whole response export."""
    def fail_download(_url: str, **_kwargs: Any) -> Any:
        raise response_exports.PdfImageError("network unavailable")

    monkeypatch.setattr(response_exports, "_download_pdf_image", fail_download)

    pdf_bytes = response_exports.build_pdf_bytes(
        "Before\n\n![Dose response](https://images.example.test/missing.png)\n\nAfter"
    )

    text = " ".join(
        (page.extract_text() or "") for page in PdfReader(BytesIO(pdf_bytes)).pages
    )
    assert "Before" in text
    assert "Image unavailable: Dose response" in text
    assert "After" in text


def test_pdf_image_download_rejects_private_destination(monkeypatch) -> None:
    """Remote response images must not turn PDF export into an SSRF primitive."""
    monkeypatch.setattr(
        pdf_images.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (pdf_images.socket.AF_INET, pdf_images.socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ],
    )
    monkeypatch.setattr(
        pdf_images,
        "_request_pdf_image_url",
        lambda *_args, **_kwargs: pytest.fail("private address must not be requested"),
    )

    with pytest.raises(pdf_images.PdfImageError, match="public internet"):
        pdf_images.download_pdf_image(
            "http://metadata.example.test/latest",
            deadline=float("inf"),
        )


def test_pdf_image_download_revalidates_redirect_destination(monkeypatch) -> None:
    """Redirects must pass the same public-address policy as original URLs."""
    def resolve(url: str, **_kwargs: Any) -> tuple[str, str, int, str]:
        if "public.example" in url:
            return url, "public.example", 443, "203.0.113.10"
        raise pdf_images.PdfImageError("image host resolved outside the public internet")

    monkeypatch.setattr(pdf_images, "_resolve_public_image_url", resolve)
    monkeypatch.setattr(
        pdf_images,
        "_request_pdf_image_url",
        lambda *_args, **_kwargs: pdf_images._PdfHttpResponse(
            302,
            {"location": "http://127.0.0.1/private.png"},
            b"",
        ),
    )

    with pytest.raises(pdf_images.PdfImageError, match="public internet"):
        pdf_images.download_pdf_image(
            "https://public.example/image.png",
            deadline=float("inf"),
        )


def test_prepare_pdf_images_limits_unique_downloads(monkeypatch) -> None:
    """Responses with many image tags must keep network work bounded."""
    calls: list[str] = []

    def download(url: str, **_kwargs: Any) -> Any:
        calls.append(url)
        return response_exports.PdfImageResource(
            content=_png_bytes(), mime_type="image/png"
        )

    monkeypatch.setattr(response_exports, "_download_pdf_image", download)
    document = "".join(
        f'<img src="https://images.example.test/{index}.png" alt="image {index}" />'
        for index in range(22)
    )

    rendered, resources = response_exports._prepare_pdf_image_resources(document)

    assert len(calls) == response_exports.MAX_PDF_REMOTE_IMAGES
    assert len(resources) == response_exports.MAX_PDF_REMOTE_IMAGES
    assert rendered.count("Image unavailable:") == 2


def test_prepare_pdf_images_limits_total_download_bytes(monkeypatch) -> None:
    """Several valid images must share one aggregate in-memory byte budget."""
    allowed_sizes: list[int] = []

    def download(_url: str, *, max_bytes: int, **_kwargs: Any) -> Any:
        allowed_sizes.append(max_bytes)
        return response_exports.PdfImageResource(
            content=bytes(max_bytes), mime_type="image/png"
        )

    monkeypatch.setattr(response_exports, "_download_pdf_image", download)
    document = "".join(
        f'<img src="https://images.example.test/{index}.png" alt="image {index}" />'
        for index in range(4)
    )

    rendered, resources = response_exports._prepare_pdf_image_resources(document)

    assert allowed_sizes == [10 * 1024 * 1024, 10 * 1024 * 1024, 5 * 1024 * 1024]
    assert len(resources) == 3
    assert rendered.count("Image unavailable:") == 1


def test_prepare_pdf_images_counts_failed_downloads_against_total(monkeypatch) -> None:
    """Invalid image bodies must still consume the export's transfer budget."""
    allowed_sizes: list[int] = []

    def download(
        _url: str,
        *,
        max_bytes: int,
        budget: pdf_images.PdfImageDownloadBudget,
        **_kwargs: Any,
    ) -> Any:
        allowed_sizes.append(max_bytes)
        budget.consume(max_bytes)
        raise response_exports.PdfImageError("invalid image")

    monkeypatch.setattr(response_exports, "_download_pdf_image", download)
    document = "".join(
        f'<img src="https://images.example.test/{index}.png" alt="image {index}" />'
        for index in range(4)
    )

    rendered, resources = response_exports._prepare_pdf_image_resources(document)

    assert allowed_sizes == [10 * 1024 * 1024, 10 * 1024 * 1024, 5 * 1024 * 1024]
    assert not resources
    assert rendered.count("Image unavailable:") == 4


def test_prepare_pdf_images_counts_expanded_resource_bytes(monkeypatch) -> None:
    """Retained decoded bytes must count toward the aggregate memory budget."""
    allowed_sizes: list[int] = []

    def download(
        _url: str,
        *,
        max_bytes: int,
        budget: pdf_images.PdfImageDownloadBudget,
        **_kwargs: Any,
    ) -> Any:
        allowed_sizes.append(max_bytes)
        budget.consume(1)
        return response_exports.PdfImageResource(bytes(max_bytes), "image/png")

    monkeypatch.setattr(response_exports, "_download_pdf_image", download)
    document = "".join(
        f'<img src="https://images.example.test/{index}.png" alt="image {index}" />'
        for index in range(4)
    )

    rendered, resources = response_exports._prepare_pdf_image_resources(document)

    assert allowed_sizes == [10 * 1024 * 1024, 10 * 1024 * 1024, 5 * 1024 * 1024]
    assert len(resources) == 3
    assert rendered.count("Image unavailable:") == 1


def test_prepare_pdf_images_accepts_case_insensitive_http_scheme(monkeypatch) -> None:
    """URL scheme casing must not bypass the prefetch step."""
    calls: list[str] = []
    url = "HTTPS://images.example.test/figure.png"

    def download(requested_url: str, **_kwargs: Any) -> Any:
        calls.append(requested_url)
        return response_exports.PdfImageResource(_png_bytes(), "image/png")

    monkeypatch.setattr(response_exports, "_download_pdf_image", download)

    _document, resources = response_exports._prepare_pdf_image_resources(
        f'<img src="{url}" alt="figure" />'
    )

    assert calls == [url]
    assert url in resources


def test_pdf_image_validation_uses_first_gif_frame() -> None:
    """Animated response images must become one deterministic PDF image."""
    output = BytesIO()
    frames = [
        Image.new("RGB", (10, 10), "red"),
        Image.new("RGB", (10, 10), "blue"),
    ]
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )

    resource = pdf_images._validate_pdf_image(output.getvalue())

    assert resource.mime_type == "image/png"
    with Image.open(BytesIO(resource.content)) as image:
        assert image.n_frames == 1
        assert image.getpixel((0, 0))[:3] == (255, 0, 0)


def test_pdf_image_validation_limits_normalized_gif_size() -> None:
    """GIF normalization must stay within the caller's in-memory byte budget."""
    output = BytesIO()
    Image.new("RGB", (10, 10), "red").save(output, format="GIF")

    with pytest.raises(pdf_images.PdfImageError, match="download size limit"):
        pdf_images._validate_pdf_image(output.getvalue(), max_bytes=10)


def test_pdf_image_validation_rejects_svg_external_resource() -> None:
    """An embedded SVG must not create a second unvalidated network request."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg">'
        b'<image href="https://private.example/image.png" />'
        b"</svg>"
    )

    with pytest.raises(pdf_images.PdfImageError, match="external resource"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_late_svg_doctype() -> None:
    """DTD rejection must inspect the complete bounded SVG payload."""
    svg = (
        b'<?xml version="1.0"?>'
        + b" " * 5000
        + b'<!DOCTYPE svg><svg xmlns="http://www.w3.org/2000/svg" />'
    )

    with pytest.raises(pdf_images.PdfImageError, match="declarations"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_decompression_enforces_expanded_size_limit() -> None:
    """Compressed responses must not bypass the per-image memory limit."""
    compressed = gzip.compress(b"x" * 1025)

    with pytest.raises(pdf_images.PdfImageError, match="download size limit"):
        pdf_images._decompress_pdf_image(
            compressed,
            encoding="gzip",
            max_bytes=1024,
        )


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


def test_build_pdf_html_document_rewrites_unicode_subscripts_and_superscripts() -> None:
    """Verify that PDF exports avoid font-dependent subscript/superscript glyphs."""
    html = response_exports.build_pdf_html_document(
        "H₂O, CO₂, x², 10⁻³ mol L⁻¹, C₆H₁₂O₆"
    )

    assert "H<sub>2</sub>O" in html
    assert "CO<sub>2</sub>" in html
    assert "x<sup>2</sup>" in html
    assert "10<sup>-3</sup>" in html
    assert "L<sup>-1</sup>" in html
    assert "C<sub>6</sub>H<sub>12</sub>O<sub>6</sub>" in html
    assert "₂" not in html
    assert "²" not in html
    assert "⁻" not in html


def test_build_pdf_html_document_removes_pdf_hostile_unicode() -> None:
    """Verify that invalid/replacement codepoints do not leak into PDF HTML."""
    html = response_exports.build_pdf_html_document("bad�\udcff\ufeff\x00text")

    assert "bad?? text" in html
    assert "�" not in html
    assert "\udcff" not in html
    assert "\ufeff" not in html
    assert "\x00" not in html


def test_build_pdf_html_document_repairs_common_mojibake() -> None:
    """Verify common UTF-8-as-Windows-1252 artifacts are repaired for PDFs."""
    html = response_exports.build_pdf_html_document("Hâ‚‚O and xÂ²")

    assert "H<sub>2</sub>O" in html
    assert "x<sup>2</sup>" in html
    assert "â" not in html
    assert "Â" not in html


def test_build_pdf_html_document_repairs_mojibake_with_unicode_text() -> None:
    """Verify mojibake repair still works when surrounding text is Unicode."""
    html = response_exports.build_pdf_html_document("Δ sample: Hâ‚‚O and xÂ²")

    assert "Δ sample: H<sub>2</sub>O" in html
    assert "x<sup>2</sup>" in html
    assert "â" not in html
    assert "Â" not in html


def test_build_pdf_html_document_uses_compact_body_text() -> None:
    """Verify that PDF exports use compact body text."""
    html = response_exports.build_pdf_html_document("compact")

    assert "font-size: 9pt;" in html
    assert 'font-family: Georgia, "Times New Roman", "Noto Serif", "DejaVu Serif",' in html
    assert '"Liberation Serif", Times, serif;' in html


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
    caller_thread = threading.get_ident()
    render_threads: list[int] = []

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
        render_threads.append(threading.get_ident())
        raise RuntimeError("PDF export requires WeasyPrint native libraries.")

    monkeypatch.setattr(response_exports.cl, "user_session", FakeUserSession())
    monkeypatch.setattr(response_exports.cl, "Message", FakeMessage)
    monkeypatch.setattr(response_exports, "build_pdf_bytes", fail_pdf_generation)

    action = SimpleNamespace(forId=None, payload={"response_id": "message-1"})

    await response_exports.send_pdf_export(action)  # type: ignore[arg-type]

    assert len(sent_messages) == 1
    assert render_threads and render_threads[0] != caller_thread
    assert sent_messages[0].author == "System"
    assert "PDF export requires WeasyPrint native libraries" in sent_messages[0].content
