"""Tests for response export generation."""

from __future__ import annotations

import asyncio
import builtins
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
from io import BytesIO
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from PIL import Image
from pypdf import PdfReader

import chainagents.exports.response as response_exports
from chainagents.exports.generated_files import (
    generated_file_descriptors,
    generated_file_paths_from_text,
    resolve_generated_output,
)
import chainagents.exports.pdf_images as pdf_images


def _png_bytes(*, width: int = 48, height: int = 32) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), "#2563eb").save(output, format="PNG")
    return output.getvalue()


def _descriptors_from_text(text: str, project_root: Path):
    return generated_file_descriptors(
        list(generated_file_paths_from_text(text)), project_root=project_root
    )


def test_generated_file_elements_for_response_text_paths_include_outputs(
    tmp_path: Path,
) -> None:
    """Verify generated response file paths become downloadable Chainlit files."""
    report_path = tmp_path / ".files" / "outputs" / "reports" / "summary.csv"
    chart_path = tmp_path / ".files" / "outputs" / "charts" / "plot.png"
    report_path.parent.mkdir(parents=True)
    chart_path.parent.mkdir(parents=True)
    report_path.write_text("name,value\nalpha,1\n", encoding="utf-8")
    chart_path.write_bytes(b"\x89PNG\r\n")

    elements = response_exports.generated_file_elements(
        _descriptors_from_text(
            "Created `/workspace/.files/outputs/reports/summary.csv` and"
            " `.files/outputs/charts/plot.png`.",
            tmp_path,
        )
    )

    assert [element.name for element in elements] == ["summary.csv", "plot.png"]
    assert [element.path for element in elements] == [
        report_path.as_posix(),
        chart_path.as_posix(),
    ]
    assert [element.mime for element in elements] == ["text/csv", "image/png"]


def test_generated_output_resolves_absolute_workspace_artifacts(
    monkeypatch,
) -> None:
    """Verify absolute artifact paths under /workspace are not remapped twice."""
    project_root = Path("/workspace/ChainAgents")
    artifact_path = project_root / ".files" / "outputs" / "plot.png"

    def fake_is_file(path: Path) -> bool:
        return path == artifact_path

    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(Path, "is_file", fake_is_file)

    resolved = resolve_generated_output(
        "/workspace/ChainAgents/.files/outputs/plot.png",
        project_root=project_root,
    )

    assert resolved == artifact_path


def test_generated_file_descriptors_ignore_unsafe_or_unavailable_paths(
    tmp_path: Path,
) -> None:
    """Verify only existing generated files under allowed routes are downloadable."""
    directory_path = tmp_path / "reports"
    directory_path.mkdir()
    outside_path = tmp_path.parent / "outside.txt"
    outside_path.write_text("secret", encoding="utf-8")

    descriptors = generated_file_descriptors(
        [
            "/workspace/reports",
            "/workspace/missing.txt",
            "/workspace/../outside.txt",
            outside_path.as_posix(),
        ],
        project_root=tmp_path,
    )

    assert descriptors == []


def test_generated_file_descriptors_reject_file_outside_outputs(
    tmp_path: Path,
) -> None:
    """A real, existing file outside `.files/outputs` must not be attached."""
    outside_path = tmp_path / "README.md"
    outside_path.write_text("# notes", encoding="utf-8")

    descriptors = generated_file_descriptors(
        [outside_path.as_posix()],
        project_root=tmp_path,
    )

    assert descriptors == []


def test_generated_file_descriptors_reject_symlinked_files_directory(
    tmp_path: Path,
) -> None:
    """A symlinked `.files` directory must fail closed instead of being followed."""
    real_target = tmp_path / "elsewhere"
    (real_target / "outputs").mkdir(parents=True)
    output_path = real_target / "outputs" / "plot.png"
    output_path.write_bytes(b"\x89PNG\r\n")
    (tmp_path / ".files").symlink_to(real_target, target_is_directory=True)

    descriptors = generated_file_descriptors(
        [(tmp_path / ".files" / "outputs" / "plot.png").as_posix()],
        project_root=tmp_path,
    )

    assert descriptors == []


def test_generated_file_elements_attach_descriptor_inside_outputs(
    tmp_path: Path,
) -> None:
    """An existing file under `.files/outputs` is attached."""
    output_path = tmp_path / ".files" / "outputs" / "plot.png"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"\x89PNG\r\n")

    elements = response_exports.generated_file_elements(
        generated_file_descriptors([output_path.as_posix()], project_root=tmp_path)
    )

    assert [element.name for element in elements] == ["plot.png"]
    assert [element.path for element in elements] == [output_path.as_posix()]


def test_build_pdf_bytes_uses_weasyprint_html_renderer(monkeypatch) -> None:
    """Verify that PDF exports are rendered through WeasyPrint."""
    html_calls: list[dict[str, Any]] = []

    class FakeHTML:
        def __init__(self, **kwargs: Any) -> None:
            html_calls.append(kwargs)

        def write_pdf(self) -> bytes:
            return b"%PDF-WEASYPRINT"

    def blocked_pdf_url_fetcher(url: str, *_args: object, **_kwargs: object) -> object:
        """Stand in for the restricted WeasyPrint fetcher; always raises."""
        raise ValueError(f"External resources are disabled for response PDF exports: {url}")

    monkeypatch.setitem(sys.modules, "weasyprint", SimpleNamespace(HTML=FakeHTML))
    monkeypatch.setattr(
        response_exports,
        "_pdf_url_fetcher",
        lambda _resources: blocked_pdf_url_fetcher,
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


def test_build_pdf_bytes_serializes_concurrent_renders(monkeypatch) -> None:
    """All interfaces must share one process-wide PDF render slot."""
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    def render(_text: str) -> bytes:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1
        return b"%PDF"

    monkeypatch.setattr(response_exports, "_build_pdf_bytes_unlocked", render)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(response_exports.build_pdf_bytes, ["one", "two"]))

    assert results == [b"%PDF", b"%PDF"]
    assert maximum_active == 1


@pytest.mark.anyio
async def test_chainlit_pdf_render_gates_before_thread_submission(monkeypatch) -> None:
    """Chainlit PDF requests must not occupy executor workers while waiting."""
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    def render(_text: str) -> bytes:
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1
        return b"%PDF"

    monkeypatch.setattr(response_exports, "build_pdf_bytes", render)

    results = await asyncio.gather(
        response_exports._build_chainlit_pdf_bytes("one"),
        response_exports._build_chainlit_pdf_bytes("two"),
    )

    assert results == [b"%PDF", b"%PDF"]
    assert maximum_active == 1


@pytest.mark.anyio
async def test_chainlit_pdf_render_keeps_gate_until_cancelled_worker_stops(
    monkeypatch,
) -> None:
    """Cancellation must not admit another render while its worker still runs."""
    first_started = threading.Event()
    first_release = threading.Event()
    second_started = threading.Event()
    calls = 0

    def render(_text: str) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            assert first_release.wait(timeout=2)
        else:
            second_started.set()
        return b"%PDF"

    monkeypatch.setattr(response_exports, "build_pdf_bytes", render)

    first = asyncio.create_task(response_exports._build_chainlit_pdf_bytes("one"))
    assert await asyncio.to_thread(first_started.wait, 1)
    first.cancel()
    second = asyncio.create_task(response_exports._build_chainlit_pdf_bytes("two"))
    await asyncio.sleep(0.05)

    assert not second_started.is_set()
    first_release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second == b"%PDF"
    assert second_started.is_set()


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


def test_build_pdf_bytes_embeds_valid_data_image() -> None:
    """Top-level data images must pass validation before PDF rendering."""
    encoded = base64.b64encode(_png_bytes()).decode("ascii")

    pdf_bytes = response_exports.build_pdf_bytes(
        f"Before\n\n![inline](data:image/png;base64,{encoded})\n\nAfter"
    )

    page = PdfReader(BytesIO(pdf_bytes)).pages[0]
    assert len(page.images) == 1
    assert "Before" in (page.extract_text() or "")
    assert "After" in (page.extract_text() or "")


def test_build_pdf_bytes_replaces_invalid_data_image() -> None:
    """Invalid data images must use the recoverable placeholder path."""
    encoded = base64.b64encode(b"not-an-image").decode("ascii")

    pdf_bytes = response_exports.build_pdf_bytes(
        f"Before\n\n![inline](data:image/png;base64,{encoded})\n\nAfter"
    )

    text = " ".join(
        (page.extract_text() or "") for page in PdfReader(BytesIO(pdf_bytes)).pages
    )
    assert "Before" in text
    assert "Image unavailable: inline" in text
    assert "After" in text


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


def test_pdf_image_download_rejects_multicast_destination(monkeypatch) -> None:
    """Public-only downloads must reject non-unicast global addresses."""
    monkeypatch.setattr(
        pdf_images.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                pdf_images.socket.AF_INET,
                pdf_images.socket.SOCK_STREAM,
                6,
                "",
                ("224.0.0.251", 80),
            )
        ],
    )

    with pytest.raises(pdf_images.PdfImageError, match="public internet"):
        pdf_images.download_pdf_image(
            "http://multicast.example.test/image.png",
            deadline=float("inf"),
        )


def test_pdf_image_download_revalidates_redirect_destination(monkeypatch) -> None:
    """Redirects must pass the same public-address policy as original URLs."""
    def resolve(url: str, **_kwargs: Any) -> tuple[str, str, int, tuple[str, ...]]:
        if "public.example" in url:
            return url, "public.example", 443, ("203.0.113.10",)
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


def test_pdf_image_download_tries_each_validated_address(monkeypatch) -> None:
    """An unreachable first DNS result must not hide a healthy public address."""
    attempts: list[str] = []
    image = _png_bytes()
    monkeypatch.setattr(
        pdf_images,
        "_resolve_public_image_url",
        lambda *_args, **_kwargs: (
            "https://images.example.test/figure.png",
            "images.example.test",
            443,
            ("2001:4860:4860::8888", "8.8.8.8"),
        ),
    )

    def request(*_args: Any, address: str, **_kwargs: Any) -> Any:
        attempts.append(address)
        if len(attempts) == 1:
            raise pdf_images._PdfImageConnectionError("image download failed")
        return pdf_images._PdfHttpResponse(200, {}, image)

    monkeypatch.setattr(pdf_images, "_request_pdf_image_url", request)

    resource = pdf_images.download_pdf_image(
        "https://images.example.test/figure.png",
        deadline=float("inf"),
    )

    assert attempts == ["2001:4860:4860::8888", "8.8.8.8"]
    assert resource.mime_type == "image/png"


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
    monkeypatch.setattr(response_exports, "PDF_IMAGE_DOWNLOAD_CONCURRENCY", 1)
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
    monkeypatch.setattr(response_exports, "PDF_IMAGE_DOWNLOAD_CONCURRENCY", 1)
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
    monkeypatch.setattr(response_exports, "PDF_IMAGE_DOWNLOAD_CONCURRENCY", 1)
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


def test_prepare_pdf_images_replaces_unsupported_sources() -> None:
    """Relative and unsupported image URLs must retain labeled placeholders."""
    document = (
        '<img src="../figure.png" alt="relative" />'
        '<img src="ftp://example.test/figure.png" alt="ftp" />'
    )

    rendered, resources = response_exports._prepare_pdf_image_resources(document)

    assert not resources
    assert "Image unavailable: relative" in rendered
    assert "Image unavailable: ftp" in rendered


def test_prepare_pdf_images_redacts_source_in_warning(monkeypatch, caplog) -> None:
    """Signed URL secrets and data payloads must not be copied into logs."""
    secret = "super-secret-token"
    url = f"https://images.example.test/private.png?token={secret}"

    def fail_download(*_args: Any, **_kwargs: Any) -> Any:
        raise response_exports.PdfImageError("invalid image")

    monkeypatch.setattr(response_exports, "_download_pdf_image", fail_download)

    response_exports._prepare_pdf_image_resources(
        f'<img src="{url}" alt="private" />'
    )

    assert secret not in caplog.text
    assert "https://images.example.test" in caplog.text


def test_prepare_pdf_images_limits_aggregate_pixels(monkeypatch) -> None:
    """Pixel exhaustion must stop later images before their pixels are decoded."""
    def download(
        *_args: Any,
        budget: pdf_images.PdfImageDownloadBudget,
        **_kwargs: Any,
    ) -> Any:
        budget.consume_pixels(25_000_000)
        return response_exports.PdfImageResource(
            _png_bytes(),
            "image/png",
            pixel_count=25_000_000,
        )

    monkeypatch.setattr(response_exports, "_download_pdf_image", download)
    document = "".join(
        f'<img src="https://images.example.test/{index}.png" alt="image {index}" />'
        for index in range(3)
    )

    rendered, resources = response_exports._prepare_pdf_image_resources(document)

    assert len(resources) == 2
    assert rendered.count("Image unavailable:") == 1


@pytest.fixture
def retry_waits(monkeypatch) -> list[float]:
    """Record retry sleeps instead of sleeping, with deterministic jitter."""
    waits: list[float] = []
    monkeypatch.setattr(pdf_images, "_sleep_before_retry", waits.append)
    monkeypatch.setattr(pdf_images, "_retry_jitter", lambda: 0.5)
    return waits


def _fake_public_resolution(
    monkeypatch,
    addresses: tuple[str, ...] = ("203.0.113.10",),
) -> list[str]:
    calls: list[str] = []

    def resolve(url: str, **_kwargs: Any) -> tuple[str, str, int, tuple[str, ...]]:
        calls.append(url)
        return url, "images.example.test", 443, addresses

    monkeypatch.setattr(pdf_images, "_resolve_public_image_url", resolve)
    return calls


def _scripted_requests(monkeypatch, outcomes: list[Any]) -> list[str]:
    calls: list[str] = []

    def request(*_args: Any, address: str, **_kwargs: Any) -> Any:
        calls.append(address)
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(pdf_images, "_request_pdf_image_url", request)
    return calls


def _image_response() -> Any:
    return pdf_images._PdfHttpResponse(200, {}, _png_bytes())


def _status_response(status: int, headers: dict[str, str] | None = None) -> Any:
    return pdf_images._PdfHttpResponse(status, headers or {}, b"")


def test_pdf_image_download_retries_rate_limited_response(
    monkeypatch, retry_waits
) -> None:
    """A 429 must be retried after the server's Retry-After delay."""
    _fake_public_resolution(monkeypatch)
    calls = _scripted_requests(
        monkeypatch,
        [_status_response(429, {"retry-after": "1"}), _image_response()],
    )

    resource = pdf_images.download_pdf_image(
        "https://images.example.test/figure.png",
        deadline=float("inf"),
    )

    assert resource.mime_type == "image/png"
    assert len(calls) == 2
    assert retry_waits == [1.0]


def test_pdf_image_parse_retry_after_accepts_seconds_and_dates() -> None:
    """Retry-After may be delta seconds or an HTTP-date."""
    from email.utils import formatdate

    now = 1_700_000_000.0
    assert pdf_images._parse_retry_after("3", now=now) == 3.0
    assert pdf_images._parse_retry_after(
        formatdate(now + 2, usegmt=True), now=now
    ) == pytest.approx(2.0, abs=1.0)
    assert pdf_images._parse_retry_after(
        formatdate(now - 120, usegmt=True), now=now
    ) == 0.0
    assert pdf_images._parse_retry_after("soon, maybe", now=now) is None
    assert pdf_images._parse_retry_after(None, now=now) is None


def test_pdf_image_download_gives_up_after_bounded_retries(
    monkeypatch, retry_waits
) -> None:
    """Persistent 503 responses must stop after the attempt limit."""
    _fake_public_resolution(monkeypatch)
    calls = _scripted_requests(monkeypatch, [_status_response(503)] * 3)

    with pytest.raises(pdf_images.PdfImageError, match="HTTP 503"):
        pdf_images.download_pdf_image(
            "https://images.example.test/figure.png",
            deadline=float("inf"),
        )

    assert len(calls) == 3
    assert retry_waits == [0.5, 1.0]


@pytest.mark.parametrize("status", [404, 403, 401, 400])
def test_pdf_image_download_does_not_retry_client_errors(
    monkeypatch, retry_waits, status: int
) -> None:
    """Permanent client errors must fail on the first response."""
    _fake_public_resolution(monkeypatch)
    calls = _scripted_requests(
        monkeypatch, [_status_response(status), _image_response()]
    )

    with pytest.raises(pdf_images.PdfImageError, match=f"HTTP {status}"):
        pdf_images.download_pdf_image(
            "https://images.example.test/figure.png",
            deadline=float("inf"),
        )

    assert len(calls) == 1
    assert retry_waits == []


def test_pdf_image_download_skips_retry_after_beyond_cap(
    monkeypatch, retry_waits
) -> None:
    """A long Retry-After must not be shortened into an early retry."""
    _fake_public_resolution(monkeypatch)
    calls = _scripted_requests(
        monkeypatch,
        [_status_response(429, {"retry-after": "60"}), _image_response()],
    )

    with pytest.raises(pdf_images.PdfImageError, match="HTTP 429"):
        pdf_images.download_pdf_image(
            "https://images.example.test/figure.png",
            deadline=float("inf"),
        )

    assert len(calls) == 1
    assert retry_waits == []


def test_pdf_image_download_skips_retry_past_image_time_cap(
    monkeypatch, retry_waits
) -> None:
    """A retry that cannot finish inside the per-image cap must not start."""
    monkeypatch.setattr(
        pdf_images,
        "time",
        SimpleNamespace(monotonic=lambda: 100.0, time=time.time, sleep=time.sleep),
    )
    _fake_public_resolution(monkeypatch)
    calls = _scripted_requests(
        monkeypatch, [_status_response(503), _image_response()]
    )

    with pytest.raises(pdf_images.PdfImageError, match="HTTP 503"):
        pdf_images.download_pdf_image(
            "https://images.example.test/figure.png",
            deadline=float("inf"),
            per_image_seconds=1.0,
        )

    assert len(calls) == 1
    assert retry_waits == []


def test_pdf_image_download_retries_connection_resets_on_each_address(
    monkeypatch, retry_waits
) -> None:
    """A retry must again try every validated address before giving up."""
    _fake_public_resolution(monkeypatch, ("2001:4860:4860::8888", "8.8.8.8"))
    calls = _scripted_requests(
        monkeypatch,
        [
            pdf_images._PdfImageConnectionError("image download failed"),
            pdf_images._PdfImageConnectionError("image download failed"),
            _image_response(),
        ],
    )

    resource = pdf_images.download_pdf_image(
        "https://images.example.test/figure.png",
        deadline=float("inf"),
    )

    assert resource.mime_type == "image/png"
    assert calls == ["2001:4860:4860::8888", "8.8.8.8", "2001:4860:4860::8888"]
    assert retry_waits == [0.5]


def test_pdf_image_download_does_not_retry_resolution_failure(
    monkeypatch, retry_waits
) -> None:
    """DNS and SSRF failures are final and must never reach a request."""
    resolutions: list[str] = []

    def resolve(url: str, **_kwargs: Any) -> Any:
        resolutions.append(url)
        raise pdf_images.PdfImageError("image host could not be resolved")

    monkeypatch.setattr(pdf_images, "_resolve_public_image_url", resolve)
    monkeypatch.setattr(
        pdf_images,
        "_request_pdf_image_url",
        lambda *_args, **_kwargs: pytest.fail("unresolved host must not be requested"),
    )

    with pytest.raises(pdf_images.PdfImageError, match="could not be resolved"):
        pdf_images.download_pdf_image(
            "https://images.example.test/figure.png",
            deadline=float("inf"),
        )

    assert len(resolutions) == 1
    assert retry_waits == []


def test_pdf_image_download_does_not_retry_size_limit(
    monkeypatch, retry_waits
) -> None:
    """Oversized bodies are final failures."""
    _fake_public_resolution(monkeypatch)
    calls = _scripted_requests(
        monkeypatch,
        [
            pdf_images.PdfImageError("image exceeds the download size limit"),
            _image_response(),
        ],
    )

    with pytest.raises(pdf_images.PdfImageError, match="size limit"):
        pdf_images.download_pdf_image(
            "https://images.example.test/figure.png",
            deadline=float("inf"),
        )

    assert len(calls) == 1
    assert retry_waits == []


def test_pdf_image_download_retry_revalidates_destination(
    monkeypatch, retry_waits
) -> None:
    """A retry must re-run DNS resolution and the public-address policy."""
    resolutions: list[str] = []

    def resolve(url: str, **_kwargs: Any) -> Any:
        resolutions.append(url)
        if len(resolutions) == 1:
            return url, "images.example.test", 443, ("203.0.113.10",)
        raise pdf_images.PdfImageError("image host resolved outside the public internet")

    monkeypatch.setattr(pdf_images, "_resolve_public_image_url", resolve)
    calls = _scripted_requests(
        monkeypatch, [_status_response(503), _image_response()]
    )

    with pytest.raises(pdf_images.PdfImageError, match="public internet"):
        pdf_images.download_pdf_image(
            "https://images.example.test/figure.png",
            deadline=float("inf"),
        )

    assert len(resolutions) == 2
    assert len(calls) == 1
    assert retry_waits == [0.5]


def test_pdf_image_download_logs_redacted_retry(
    monkeypatch, retry_waits, caplog
) -> None:
    """Retry logs must identify the host without leaking signed URL tokens."""
    import logging

    secret = "super-secret-token"
    caplog.set_level(logging.DEBUG, logger="chainagents.exports.pdf_images")
    _fake_public_resolution(monkeypatch)
    _scripted_requests(monkeypatch, [_status_response(503), _image_response()])

    pdf_images.download_pdf_image(
        f"https://images.example.test/figure.png?token={secret}",
        deadline=float("inf"),
    )

    assert "attempt 2/3" in caplog.text
    assert "https://images.example.test" in caplog.text
    assert secret not in caplog.text


class _FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def monotonic(self) -> float:
        return self.now


class _RecordingTimer:
    instances: ClassVar[list[_RecordingTimer]] = []

    def __init__(self, interval: float, function: Any, args: Any = (), kwargs: Any = None) -> None:
        self.interval = interval
        self.daemon = False
        _RecordingTimer.instances.append(self)

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        pass


class _FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)

    def shutdown(self, _how: int) -> None:
        pass


class _ScriptedHttpResponse:
    def __init__(
        self,
        clock: _FakeClock,
        chunks: list[bytes | BaseException],
        *,
        delay: float,
        status: int = 200,
        length: int | None = None,
        headers: list[tuple[str, str]] | None = None,
    ) -> None:
        self.status = status
        self.length = length
        self._headers = headers or []
        self._clock = clock
        self._chunks = list(chunks)
        self._delay = delay

    def getheaders(self) -> list[tuple[str, str]]:
        return self._headers

    def read1(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        self._clock.now += self._delay
        chunk = self._chunks.pop(0)
        if isinstance(chunk, BaseException):
            raise chunk
        return chunk[:size]


def _install_fake_http(
    monkeypatch,
    clock: _FakeClock,
    *,
    response: _ScriptedHttpResponse | None = None,
    request_error: BaseException | None = None,
    request_errors: list[BaseException] | None = None,
) -> list[Any]:
    connections: list[Any] = []
    pending_errors = list(request_errors or [])

    class FakeConnection:
        def __init__(self, host: str, port: int | None = None, timeout: float | None = None) -> None:
            self.host = host
            self.port = port
            self.timeout = timeout
            self.sock = _FakeSocket()
            connections.append(self)

        def request(self, *_args: Any, **_kwargs: Any) -> None:
            if request_error is not None:
                raise request_error
            if pending_errors:
                raise pending_errors.pop(0)

        def getresponse(self) -> Any:
            return response

        def close(self) -> None:
            pass

    _RecordingTimer.instances = []
    monkeypatch.setattr(pdf_images.http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(
        pdf_images,
        "time",
        SimpleNamespace(monotonic=clock.monotonic, time=time.time, sleep=time.sleep),
    )
    monkeypatch.setattr(
        pdf_images,
        "threading",
        SimpleNamespace(
            Timer=_RecordingTimer,
            Event=threading.Event,
            Lock=threading.Lock,
            Condition=threading.Condition,
            Thread=threading.Thread,
        ),
    )
    return connections


def _request_fake_image(
    clock: _FakeClock,
    *,
    deadline_seconds: float = 30.0,
    budget: pdf_images.PdfImageDownloadBudget | None = None,
) -> Any:
    deadline = clock.now + deadline_seconds
    return pdf_images._request_pdf_image_url(
        "http://images.example.test/figure.png",
        host="images.example.test",
        port=80,
        address="203.0.113.10",
        max_bytes=pdf_images.PDF_IMAGE_MAX_BYTES,
        deadline=deadline,
        budget=budget or pdf_images.PdfImageDownloadBudget(deadline, 25 * 1024 * 1024),
    )


def test_pdf_image_request_aborts_stalled_body(monkeypatch) -> None:
    """A trickling body must not consume the whole export budget."""
    clock = _FakeClock()
    _install_fake_http(
        monkeypatch,
        clock,
        response=_ScriptedHttpResponse(clock, [b"x" * 100] * 20, delay=3.0),
    )

    with pytest.raises(pdf_images.PdfImageError, match="stalled") as excinfo:
        _request_fake_image(clock)

    assert not isinstance(excinfo.value, pdf_images.PdfImageRetryableError)


def test_pdf_image_request_accepts_slow_steady_body(monkeypatch) -> None:
    """A slow body that keeps making progress must still complete."""
    clock = _FakeClock()
    _install_fake_http(
        monkeypatch,
        clock,
        response=_ScriptedHttpResponse(clock, [b"x" * 8192] * 5, delay=4.0),
    )

    result = _request_fake_image(clock)

    assert result.status == 200
    assert len(result.content) == 5 * 8192


def test_pdf_image_request_timer_uses_image_deadline(monkeypatch) -> None:
    """The hard-close timer must fire at the per-image deadline."""
    clock = _FakeClock()
    connections = _install_fake_http(
        monkeypatch,
        clock,
        response=_ScriptedHttpResponse(clock, [b"x" * 10], delay=0.0),
    )

    _request_fake_image(clock, deadline_seconds=12.5)

    assert [timer.interval for timer in _RecordingTimer.instances] == [12.5]
    assert connections[0].timeout == pdf_images.PDF_IMAGE_TIMEOUT_SECONDS


def test_pdf_image_request_treats_tls_failure_as_final(monkeypatch) -> None:
    """Certificate failures must not be retried as connection errors."""
    import ssl

    clock = _FakeClock()
    _install_fake_http(
        monkeypatch,
        clock,
        request_error=ssl.SSLCertVerificationError("certificate verify failed"),
    )

    with pytest.raises(pdf_images.PdfImageError, match="TLS") as excinfo:
        _request_fake_image(clock)

    assert not isinstance(excinfo.value, pdf_images.PdfImageRetryableError)


def test_pdf_image_request_treats_connection_reset_as_retryable(monkeypatch) -> None:
    """Connection resets must remain eligible for address fallback and retry."""
    clock = _FakeClock()
    _install_fake_http(
        monkeypatch,
        clock,
        request_error=ConnectionResetError("reset by peer"),
    )

    with pytest.raises(pdf_images._PdfImageConnectionError):
        _request_fake_image(clock)


def _image_document(urls: list[str]) -> str:
    return "".join(
        f'<img src="{url}" alt="image {index}" />' for index, url in enumerate(urls)
    )


def test_prepare_pdf_images_hung_image_does_not_block_others(monkeypatch) -> None:
    """One hung download must cost only its own image."""
    release = threading.Event()
    urls = [f"https://host{index}.example.test/{index}.png" for index in range(4)]

    def download(url: str, **_kwargs: Any) -> Any:
        if url == urls[1]:
            release.wait(10)
        return response_exports.PdfImageResource(_png_bytes(), "image/png")

    budget_seconds = 1.5
    monkeypatch.setattr(
        response_exports, "PDF_IMAGE_DOWNLOAD_BUDGET_SECONDS", budget_seconds
    )
    monkeypatch.setattr(response_exports, "_download_pdf_image", download)
    started = time.monotonic()
    try:
        rendered, resources = response_exports._prepare_pdf_image_resources(
            _image_document(urls)
        )
        elapsed = time.monotonic() - started
    finally:
        release.set()

    assert set(resources) == {urls[0], urls[2], urls[3]}
    assert rendered.count("Image unavailable:") == 1
    assert "Image unavailable: image 1" in rendered
    assert elapsed < budget_seconds + 1.5


def test_prepare_pdf_images_downloads_in_parallel(monkeypatch) -> None:
    """Independent images must download concurrently."""
    barrier = threading.Barrier(4)
    urls = [f"https://host{index}.example.test/{index}.png" for index in range(4)]

    def download(_url: str, **_kwargs: Any) -> Any:
        barrier.wait(timeout=2)
        return response_exports.PdfImageResource(_png_bytes(), "image/png")

    monkeypatch.setattr(response_exports, "_download_pdf_image", download)

    rendered, resources = response_exports._prepare_pdf_image_resources(
        _image_document(urls)
    )

    assert set(resources) == set(urls)
    assert "Image unavailable:" not in rendered


def test_prepare_pdf_images_limits_per_host_concurrency(monkeypatch) -> None:
    """One host must not receive more than two concurrent downloads."""
    lock = threading.Lock()
    active_by_host: dict[str, int] = {}
    max_by_host: dict[str, int] = {}
    active_total = 0
    max_total = 0
    urls = [
        f"https://host{index % 3}.example.test/{index}.png" for index in range(9)
    ]

    def download(url: str, **_kwargs: Any) -> Any:
        nonlocal active_total, max_total
        host = url.split("/")[2]
        with lock:
            active_by_host[host] = active_by_host.get(host, 0) + 1
            max_by_host[host] = max(max_by_host.get(host, 0), active_by_host[host])
            active_total += 1
            max_total = max(max_total, active_total)
        time.sleep(0.05)
        with lock:
            active_by_host[host] -= 1
            active_total -= 1
        return response_exports.PdfImageResource(_png_bytes(), "image/png")

    monkeypatch.setattr(response_exports, "_download_pdf_image", download)

    _rendered, resources = response_exports._prepare_pdf_image_resources(
        _image_document(urls)
    )

    assert len(resources) == 9
    assert max(max_by_host.values()) <= response_exports.PDF_IMAGE_PER_HOST_CONCURRENCY
    assert max_total <= response_exports.PDF_IMAGE_DOWNLOAD_CONCURRENCY
    assert max_total >= 2


def test_prepare_pdf_images_keeps_source_order(monkeypatch, caplog) -> None:
    """Resources and warnings must follow source order, not completion order."""
    urls = [f"https://host{index}.example.test/{index}.png" for index in range(4)]

    def download(url: str, **_kwargs: Any) -> Any:
        index = urls.index(url)
        time.sleep((len(urls) - index) * 0.05)
        if index % 2:
            raise response_exports.PdfImageError(f"failure {index}")
        return response_exports.PdfImageResource(_png_bytes(), "image/png")

    monkeypatch.setattr(response_exports, "_download_pdf_image", download)

    rendered, resources = response_exports._prepare_pdf_image_resources(
        _image_document(urls)
    )

    assert list(resources) == [urls[0], urls[2]]
    messages = [record.getMessage() for record in caplog.records]
    failures = [message for message in messages if "failure" in message]
    assert [message.rsplit(" ", 1)[-1] for message in failures] == ["1", "3"]
    assert rendered.index("image 1") < rendered.index("image 3")


def test_prepare_pdf_images_enforces_byte_budget_concurrently(monkeypatch) -> None:
    """Concurrent downloads must share the aggregate byte budget exactly."""
    created: list[pdf_images.PdfImageDownloadBudget] = []
    consumed_lock = threading.Lock()
    total_consumed = 0
    chunk = 1024 * 1024

    class RecordingBudget(pdf_images.PdfImageDownloadBudget):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    def download(
        _url: str,
        *,
        max_bytes: int,
        budget: pdf_images.PdfImageDownloadBudget,
        **_kwargs: Any,
    ) -> Any:
        nonlocal total_consumed
        consumed = 0
        target = min(max_bytes, 6 * chunk)
        while consumed < target:
            size = min(chunk, target - consumed)
            budget.consume(size)
            with consumed_lock:
                total_consumed += size
            consumed += size
            time.sleep(0.001)
        return response_exports.PdfImageResource(bytes(consumed), "image/png")

    monkeypatch.setattr(response_exports, "PdfImageDownloadBudget", RecordingBudget)
    monkeypatch.setattr(response_exports, "_download_pdf_image", download)
    urls = [f"https://host{index}.example.test/{index}.png" for index in range(6)]

    _rendered, resources = response_exports._prepare_pdf_image_resources(
        _image_document(urls)
    )

    cap = response_exports.MAX_PDF_REMOTE_IMAGE_BYTES
    retained = sum(len(resource.content) for resource in resources.values())
    assert resources
    assert retained <= total_consumed <= cap
    # Transferred bytes are charged exactly once, and retained bytes that were
    # already transferred are not charged again.
    assert created[0].remaining_bytes == cap - total_consumed


def test_prepare_pdf_images_slow_host_does_not_starve_other_hosts(monkeypatch) -> None:
    """Queued images for a busy host must not hold workers other hosts need."""
    release = threading.Event()
    slow_urls = [f"https://slow.example.test/{index}.png" for index in range(8)]
    fast_url = "https://fast.example.test/figure.png"

    def download(url: str, **_kwargs: Any) -> Any:
        if url in slow_urls:
            release.wait(5)
        return response_exports.PdfImageResource(_png_bytes(), "image/png")

    monkeypatch.setattr(response_exports, "PDF_IMAGE_DOWNLOAD_BUDGET_SECONDS", 1.0)
    monkeypatch.setattr(response_exports, "_download_pdf_image", download)
    try:
        _rendered, resources = response_exports._prepare_pdf_image_resources(
            _image_document([*slow_urls, fast_url])
        )
    finally:
        release.set()

    assert fast_url in resources


def test_prepare_pdf_images_contains_unexpected_worker_errors(
    monkeypatch, caplog
) -> None:
    """A bug in one download must become a placeholder, not break the export."""
    urls = [f"https://host{index}.example.test/{index}.png" for index in range(2)]

    def download(url: str, **_kwargs: Any) -> Any:
        if url == urls[0]:
            raise RuntimeError("boom")
        return response_exports.PdfImageResource(_png_bytes(), "image/png")

    monkeypatch.setattr(response_exports, "_download_pdf_image", download)

    rendered, resources = response_exports._prepare_pdf_image_resources(
        _image_document(urls)
    )

    assert list(resources) == [urls[1]]
    assert "Image unavailable: image 0" in rendered
    assert "unexpected RuntimeError" in caplog.text


def test_pdf_image_download_shares_attempts_across_redirects(
    monkeypatch, retry_waits
) -> None:
    """The attempt limit applies to the whole image, not to each redirect hop."""
    _fake_public_resolution(monkeypatch)
    calls = _scripted_requests(
        monkeypatch,
        [
            _status_response(503),
            _status_response(302, {"location": "https://images.example.test/next.png"}),
            _status_response(503),
            _image_response(),
        ],
    )

    with pytest.raises(pdf_images.PdfImageError, match="HTTP 503"):
        pdf_images.download_pdf_image(
            "https://images.example.test/figure.png",
            deadline=float("inf"),
        )

    assert len(calls) == 3
    assert retry_waits == [0.5]


def test_pdf_image_request_treats_timeout_after_partial_body_as_stall(
    monkeypatch,
) -> None:
    """Silence after part of the body arrived is a final stall."""
    clock = _FakeClock()
    _install_fake_http(
        monkeypatch,
        clock,
        response=_ScriptedHttpResponse(
            clock, [b"x" * 100, TimeoutError("timed out")], delay=1.0
        ),
    )

    with pytest.raises(pdf_images.PdfImageError, match="stalled") as excinfo:
        _request_fake_image(clock)

    assert not isinstance(excinfo.value, pdf_images.PdfImageRetryableError)


def test_pdf_image_request_reports_time_limit_after_timer_close(monkeypatch) -> None:
    """A body cut short by the hard-close timer must not be returned as content."""

    class FiringTimer(_RecordingTimer):
        def __init__(self, interval: float, function: Any, args: Any = (), kwargs: Any = None) -> None:
            super().__init__(interval, function, args, kwargs)
            self._function = function
            self._args = args

        def start(self) -> None:
            self._function(*self._args)

    clock = _FakeClock()
    _install_fake_http(
        monkeypatch,
        clock,
        response=_ScriptedHttpResponse(clock, [b"x" * 10], delay=0.0),
    )
    monkeypatch.setattr(pdf_images.threading, "Timer", FiringTimer)

    with pytest.raises(pdf_images.PdfImageError, match="time limit"):
        _request_fake_image(clock)


def test_pdf_image_request_retries_truncated_content_length_body(monkeypatch) -> None:
    """An early EOF before Content-Length is reached is a transient failure."""
    clock = _FakeClock()
    _install_fake_http(
        monkeypatch,
        clock,
        response=_ScriptedHttpResponse(clock, [b"x" * 10], delay=0.0, length=90),
    )

    with pytest.raises(pdf_images._PdfImageConnectionError):
        _request_fake_image(clock)


def test_pdf_image_request_treats_tls_eof_as_retryable(monkeypatch) -> None:
    """Non-certificate TLS errors must keep address fallback and retries."""
    import ssl

    clock = _FakeClock()
    _install_fake_http(
        monkeypatch,
        clock,
        request_error=ssl.SSLEOFError("EOF occurred in violation of protocol"),
    )

    with pytest.raises(pdf_images._PdfImageConnectionError):
        _request_fake_image(clock)


def test_pdf_image_download_falls_back_after_tls_eof(monkeypatch) -> None:
    """A TLS EOF on one address must not hide a healthy second address."""
    import ssl

    clock = _FakeClock()
    connections = _install_fake_http(
        monkeypatch,
        clock,
        response=_ScriptedHttpResponse(clock, [_png_bytes()], delay=0.0),
        request_errors=[ssl.SSLEOFError("EOF occurred in violation of protocol")],
    )
    monkeypatch.setattr(
        pdf_images,
        "_resolve_public_image_url",
        lambda url, **_kwargs: (
            url,
            "images.example.test",
            80,
            ("203.0.113.10", "203.0.113.11"),
        ),
    )

    resource = pdf_images.download_pdf_image(
        "http://images.example.test/figure.png",
        deadline=float("inf"),
    )

    assert resource.mime_type == "image/png"
    assert [connection.host for connection in connections] == [
        "203.0.113.10",
        "203.0.113.11",
    ]


def test_pdf_image_dns_wait_reports_image_time_limit(monkeypatch) -> None:
    """A resolver that outlives the image deadline reports the image time limit."""
    release = threading.Event()

    def slow_getaddrinfo(*_args: Any, **_kwargs: Any) -> Any:
        release.wait(5)
        return []

    monkeypatch.setattr(pdf_images.socket, "getaddrinfo", slow_getaddrinfo)
    try:
        with pytest.raises(pdf_images.PdfImageError, match="time limit"):
            pdf_images._getaddrinfo_before_deadline(
                "images.example.test", 443, deadline=time.monotonic() + 0.05
            )
    finally:
        release.set()


def test_pdf_image_request_releases_unused_reservation_after_failure(
    monkeypatch,
) -> None:
    """A failed attempt must return the unread part of its reservation."""
    clock = _FakeClock()
    _install_fake_http(
        monkeypatch,
        clock,
        response=_ScriptedHttpResponse(
            clock,
            [b"x" * 100],
            delay=0.0,
            length=900,
            headers=[("Content-Length", "1000")],
        ),
    )
    budget = pdf_images.PdfImageDownloadBudget(float("inf"), 5_000)

    with pytest.raises(pdf_images._PdfImageConnectionError):
        _request_fake_image(clock, budget=budget)

    assert budget.remaining_bytes == 5_000 - 100


def test_pdf_image_request_reservation_matches_transferred_bytes(monkeypatch) -> None:
    """A complete body consumes exactly its declared, reserved size."""
    clock = _FakeClock()
    _install_fake_http(
        monkeypatch,
        clock,
        response=_ScriptedHttpResponse(
            clock,
            [b"x" * 600, b"x" * 400],
            delay=0.0,
            headers=[("Content-Length", "1000")],
        ),
    )
    budget = pdf_images.PdfImageDownloadBudget(float("inf"), 5_000)

    result = _request_fake_image(clock, budget=budget)

    assert len(result.content) == 1000
    assert budget.remaining_bytes == 4_000


def test_pdf_image_request_rejects_declared_size_beyond_budget(monkeypatch) -> None:
    """A declared body the shared budget cannot hold must fail before reading."""
    clock = _FakeClock()
    response = _ScriptedHttpResponse(
        clock,
        [b"x" * 1000],
        delay=0.0,
        headers=[("Content-Length", "1000")],
    )
    _install_fake_http(monkeypatch, clock, response=response)
    budget = pdf_images.PdfImageDownloadBudget(float("inf"), 999)

    with pytest.raises(pdf_images.PdfImageError, match="budget exceeded") as excinfo:
        _request_fake_image(clock, budget=budget)

    assert not isinstance(excinfo.value, pdf_images.PdfImageRetryableError)
    assert len(response._chunks) == 1
    assert budget.remaining_bytes == 999


def test_pdf_image_retry_releases_reservation_between_attempts(
    monkeypatch, retry_waits
) -> None:
    """A retried download must not hold the first attempt's reservation."""
    clock = _FakeClock()
    image = _png_bytes()
    responses = [
        _ScriptedHttpResponse(
            clock,
            [image[:10]],
            delay=0.0,
            length=len(image) - 10,
            headers=[("Content-Length", str(len(image)))],
        ),
        _ScriptedHttpResponse(
            clock,
            [image],
            delay=0.0,
            headers=[("Content-Length", str(len(image)))],
        ),
    ]
    connections = _install_fake_http(monkeypatch, clock)
    monkeypatch.setattr(
        pdf_images.http.client.HTTPConnection,
        "getresponse",
        lambda _self: responses.pop(0),
    )
    monkeypatch.setattr(
        pdf_images,
        "_resolve_public_image_url",
        lambda url, **_kwargs: (url, "images.example.test", 80, ("203.0.113.10",)),
    )
    budget = pdf_images.PdfImageDownloadBudget(float("inf"), 1024 * 1024)

    pdf_images.download_pdf_image(
        "http://images.example.test/figure.png",
        deadline=float("inf"),
        budget=budget,
    )

    assert len(connections) == 2
    assert retry_waits == [0.5]
    assert budget.remaining_bytes == 1024 * 1024 - 10 - len(image)


def test_prepare_pdf_images_reserves_declared_sizes_concurrently(monkeypatch) -> None:
    """Parallel large images must not all fail by sharing the byte budget."""
    size = 10 * 1024 * 1024
    image = _png_bytes()
    body = image + bytes(size - len(image))
    barrier = threading.Barrier(4)
    reads: dict[str, int] = {}
    reads_lock = threading.Lock()
    created: list[pdf_images.PdfImageDownloadBudget] = []

    class RecordingBudget(pdf_images.PdfImageDownloadBudget):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    class StreamingResponse:
        status = 200

        def __init__(self, host: str) -> None:
            self.host = host
            self.length: int | None = size
            self._offset = 0

        def getheaders(self) -> list[tuple[str, str]]:
            return [("Content-Length", str(size)), ("Content-Type", "image/png")]

        def read1(self, amount: int) -> bytes:
            time.sleep(0)
            chunk = body[self._offset : self._offset + amount]
            self._offset += len(chunk)
            self.length = size - self._offset
            with reads_lock:
                reads[self.host] = reads.get(self.host, 0) + len(chunk)
            return chunk

    class FakeConnection:
        def __init__(self, host: str, port: int | None = None, timeout: float | None = None) -> None:
            self.sock = _FakeSocket()
            self.timeout = timeout
            self._host = ""

        def request(self, _method: str, _path: str, headers: dict[str, str]) -> None:
            self._host = headers["Host"]

        def getresponse(self) -> Any:
            barrier.wait(timeout=2)
            return StreamingResponse(self._host)

        def close(self) -> None:
            pass

    monkeypatch.setattr(pdf_images.http.client, "HTTPConnection", FakeConnection)
    monkeypatch.setattr(
        pdf_images,
        "_resolve_public_image_url",
        lambda url, **_kwargs: (
            url,
            url.split("/")[2],
            80,
            ("203.0.113.10",),
        ),
    )
    monkeypatch.setattr(response_exports, "PdfImageDownloadBudget", RecordingBudget)
    monkeypatch.setattr(response_exports, "_download_pdf_image", pdf_images.download_pdf_image)
    urls = [f"http://host{index}.example.test/{index}.png" for index in range(4)]

    rendered, resources = response_exports._prepare_pdf_image_resources(
        _image_document(urls)
    )

    assert len(resources) == 2
    assert rendered.count("Image unavailable:") == 2
    assert sorted(reads.values()) == [size, size]
    retained = sum(len(resource.content) for resource in resources.values())
    assert created[0].remaining_bytes == (
        response_exports.MAX_PDF_REMOTE_IMAGE_BYTES - retained
    )


@pytest.mark.parametrize(
    "headers",
    [
        [("Content-Length", "-1000000")],
        [("Content-Length", "1_000")],
        [("Content-Length", "+5")],
        [("Content-Length", "1 0")],
        [("Content-Length", "")],
        [("Content-Length", "10"), ("Content-Length", "20")],
        # Beyond Python's int-string conversion limit (4300 digits).
        [("Content-Length", "9" * 5000)],
    ],
)
def test_pdf_image_request_rejects_invalid_content_length(
    monkeypatch, headers: list[tuple[str, str]]
) -> None:
    """Malformed declared lengths must fail before any budget change or body read."""
    clock = _FakeClock()
    response = _ScriptedHttpResponse(
        clock, [b"x" * 1000], delay=0.0, headers=headers
    )
    _install_fake_http(monkeypatch, clock, response=response)
    budget = pdf_images.PdfImageDownloadBudget(float("inf"), 1_000)

    with pytest.raises(pdf_images.PdfImageError, match="invalid content length") as excinfo:
        _request_fake_image(clock, budget=budget)

    assert not isinstance(excinfo.value, pdf_images.PdfImageRetryableError)
    assert len(response._chunks) == 1
    assert budget.remaining_bytes == 1_000


def test_pdf_image_request_accepts_matching_duplicate_content_length(
    monkeypatch,
) -> None:
    """Repeated identical Content-Length headers describe one consistent body."""
    clock = _FakeClock()
    _install_fake_http(
        monkeypatch,
        clock,
        response=_ScriptedHttpResponse(
            clock,
            [b"x" * 10],
            delay=0.0,
            headers=[("Content-Length", "10"), ("content-length", " 10 ")],
        ),
    )
    budget = pdf_images.PdfImageDownloadBudget(float("inf"), 1_000)

    result = _request_fake_image(clock, budget=budget)

    assert len(result.content) == 10
    assert budget.remaining_bytes == 990


@pytest.mark.parametrize(
    "operation",
    ["consume", "reserve", "settle_reservation", "consume_pixels", "refund_pixels"],
)
def test_pdf_image_budget_rejects_negative_amounts(operation: str) -> None:
    """No caller may inflate the shared byte or pixel budget."""
    shared = pdf_images.PdfImageDownloadBudget(
        float("inf"), 1_000, remaining_pixels=1_000
    )
    view = response_exports._CountingPdfImageBudget(shared)

    for budget in (shared, view):
        with pytest.raises(ValueError):
            getattr(budget, operation)(-1)

    assert shared.remaining_bytes == 1_000
    assert shared.remaining_pixels == 1_000
    assert view.consumed_bytes == 0
    assert view.consumed_pixels == 0


def _png_header_only(width: int, height: int) -> bytes:
    """Return a PNG whose header declares a size without any pixel data."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


def test_pdf_image_validation_does_not_mutate_global_warning_filters(
    monkeypatch,
) -> None:
    """Validation must not rewrite the process-wide warning filter list."""
    import warnings

    original_filters = list(warnings.filters)
    seen_filters: list[list[Any]] = []
    real_open = pdf_images.Image.open

    def recording_open(*args: Any, **kwargs: Any) -> Any:
        seen_filters.append(list(warnings.filters))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(pdf_images.Image, "open", recording_open)

    pdf_images._validate_pdf_image(_png_bytes())

    assert seen_filters
    assert all(filters == original_filters for filters in seen_filters)
    assert list(warnings.filters) == original_filters


def test_pdf_image_concurrent_validation_keeps_filters_and_rejects_bombs() -> None:
    """Parallel validations leave warning state intact and still reject bombs."""
    import warnings

    original_filters = list(warnings.filters)
    payloads = [
        _png_bytes(),
        _png_header_only(10_000, 10_000),
        _png_header_only(20_000, 20_000),
    ] * 4
    results: list[str] = []
    results_lock = threading.Lock()

    def validate(content: bytes) -> None:
        try:
            pdf_images._validate_pdf_image(content)
            outcome = "ok"
        except pdf_images.PdfImageError:
            outcome = "rejected"
        with results_lock:
            results.append(outcome)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pdf_images.Image.DecompressionBombWarning)
        inner_filters = list(warnings.filters)
        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(validate, payloads))
        assert list(warnings.filters) == inner_filters

    assert list(warnings.filters) == original_filters
    assert sorted(results) == ["ok"] * 4 + ["rejected"] * 8


def _start_reservation(
    budget: pdf_images.PdfImageDownloadBudget,
    size: int,
    *,
    deadline: float,
) -> tuple[threading.Thread, list[Any]]:
    outcome: list[Any] = []

    def reserve() -> None:
        try:
            budget.reserve(size, deadline=deadline)
            outcome.append("reserved")
        except pdf_images.PdfImageError as exc:
            outcome.append(exc)

    thread = threading.Thread(target=reserve, daemon=True)
    thread.start()
    return thread, outcome


def test_pdf_image_budget_reservation_waits_for_active_reservations() -> None:
    """A reservation that may fit after others settle must wait, not fail."""
    budget = pdf_images.PdfImageDownloadBudget(float("inf"), 100)
    budget.reserve(60)
    budget.reserve(40)

    thread, outcome = _start_reservation(budget, 50, deadline=time.monotonic() + 5)
    time.sleep(0.05)
    assert outcome == []

    budget.settle_reservation(50)  # the first body used only 10 of its 60 bytes
    thread.join(timeout=2)

    assert outcome == ["reserved"]
    assert budget.remaining_bytes == 0
    budget.settle_reservation(0)
    budget.settle_reservation(0)
    assert budget.remaining_bytes == 0


def test_pdf_image_budget_reservation_fails_once_others_settle() -> None:
    """A reservation that can never fit must fail after others settle, not hang."""
    budget = pdf_images.PdfImageDownloadBudget(float("inf"), 100)
    budget.reserve(80)

    thread, outcome = _start_reservation(budget, 50, deadline=time.monotonic() + 5)
    time.sleep(0.05)
    assert outcome == []

    budget.settle_reservation(0)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert len(outcome) == 1
    assert isinstance(outcome[0], pdf_images.PdfImageError)
    assert "budget exceeded" in str(outcome[0])
    assert budget.remaining_bytes == 20


def test_pdf_image_budget_reservation_wait_stops_at_deadline() -> None:
    """Waiting for other reservations must end at the image deadline."""
    budget = pdf_images.PdfImageDownloadBudget(float("inf"), 100)
    budget.reserve(80)

    thread, outcome = _start_reservation(budget, 50, deadline=time.monotonic() + 0.1)
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert len(outcome) == 1
    assert "time limit" in str(outcome[0])
    budget.settle_reservation(80)
    assert budget.remaining_bytes == 100


def test_pdf_image_budget_reservation_fails_immediately_without_contention() -> None:
    """With no other reservation active, an oversized reservation fails at once."""
    budget = pdf_images.PdfImageDownloadBudget(float("inf"), 10)

    with pytest.raises(pdf_images.PdfImageError, match="budget exceeded"):
        budget.reserve(11, deadline=float("inf"))

    assert budget.remaining_bytes == 10


def test_prepare_pdf_images_limits_redirect_target_host(monkeypatch) -> None:
    """Origins that redirect to one CDN must share that CDN's host limit."""
    lock = threading.Lock()
    active_cdn = 0
    max_cdn = 0
    image = _png_bytes()
    urls = [f"https://origin{index}.example.test/{index}.png" for index in range(4)]

    monkeypatch.setattr(
        pdf_images,
        "_resolve_public_image_url",
        lambda url, **_kwargs: (url, url.split("/")[2], 443, ("203.0.113.10",)),
    )

    def request(url: str, **_kwargs: Any) -> Any:
        nonlocal active_cdn, max_cdn
        host = url.split("/")[2]
        if host.startswith("origin"):
            name = url.rsplit("/", 1)[-1]
            return _status_response(302, {"location": f"https://cdn.example.test/{name}"})
        with lock:
            active_cdn += 1
            max_cdn = max(max_cdn, active_cdn)
        time.sleep(0.1)
        with lock:
            active_cdn -= 1
        return pdf_images._PdfHttpResponse(200, {}, image)

    monkeypatch.setattr(pdf_images, "_request_pdf_image_url", request)

    _rendered, resources = response_exports._prepare_pdf_image_resources(
        _image_document(urls)
    )

    assert set(resources) == set(urls)
    assert max_cdn <= response_exports.PDF_IMAGE_PER_HOST_CONCURRENCY


def test_pdf_image_host_limiter_bounds_each_host() -> None:
    """The shared limiter must cap each host and time out blocked acquires."""
    limiter = pdf_images.PdfImageHostLimiter(1)

    assert limiter.try_acquire("cdn.example.test")
    assert not limiter.try_acquire("cdn.example.test")
    assert limiter.try_acquire("other.example.test")
    assert not limiter.acquire("cdn.example.test", deadline=time.monotonic() + 0.05)

    acquired: list[bool] = []
    waiter = threading.Thread(
        target=lambda: acquired.append(
            limiter.acquire("cdn.example.test", deadline=time.monotonic() + 5)
        ),
        daemon=True,
    )
    waiter.start()
    time.sleep(0.05)
    limiter.release("cdn.example.test")
    waiter.join(timeout=2)

    assert acquired == [True]


def test_pdf_image_redirect_waits_for_target_host_slot(monkeypatch) -> None:
    """A redirect to a saturated host fails at the image time limit."""
    limiter = pdf_images.PdfImageHostLimiter(1)
    assert limiter.try_acquire("origin.example.test")  # held by the scheduler
    assert limiter.try_acquire("cdn.example.test")  # held by another image
    monkeypatch.setattr(
        pdf_images,
        "_resolve_public_image_url",
        lambda url, **_kwargs: (url, url.split("/")[2], 443, ("203.0.113.10",)),
    )
    requested: list[str] = []

    def request(url: str, **_kwargs: Any) -> Any:
        requested.append(url)
        return _status_response(302, {"location": "https://cdn.example.test/a.png"})

    monkeypatch.setattr(pdf_images, "_request_pdf_image_url", request)

    with pytest.raises(pdf_images.PdfImageError, match="time limit"):
        pdf_images.download_pdf_image(
            "https://origin.example.test/a.png",
            deadline=float("inf"),
            per_image_seconds=0.1,
            host_limiter=limiter,
        )

    assert requested == ["https://origin.example.test/a.png"]
    limiter.release("cdn.example.test")
    # The download released nothing it did not acquire.
    assert not limiter.try_acquire("origin.example.test")
    assert limiter.try_acquire("cdn.example.test")


def test_pdf_image_redirect_releases_acquired_host_slots(monkeypatch) -> None:
    """Slots taken for redirect targets are returned when the image finishes."""
    limiter = pdf_images.PdfImageHostLimiter(1)
    assert limiter.try_acquire("origin.example.test")
    image = _png_bytes()
    monkeypatch.setattr(
        pdf_images,
        "_resolve_public_image_url",
        lambda url, **_kwargs: (url, url.split("/")[2], 443, ("203.0.113.10",)),
    )

    def request(url: str, **_kwargs: Any) -> Any:
        if "origin" in url:
            return _status_response(302, {"location": "https://cdn.example.test/a.png"})
        assert not limiter.try_acquire("cdn.example.test")
        return pdf_images._PdfHttpResponse(200, {}, image)

    monkeypatch.setattr(pdf_images, "_request_pdf_image_url", request)

    pdf_images.download_pdf_image(
        "https://origin.example.test/a.png",
        deadline=float("inf"),
        host_limiter=limiter,
    )

    assert limiter.try_acquire("cdn.example.test")
    assert not limiter.try_acquire("origin.example.test")


def test_pdf_image_budget_consume_is_atomic() -> None:
    """Concurrent consumers must never overdraw the shared byte budget."""
    budget = pdf_images.PdfImageDownloadBudget(float("inf"), 5_000)
    successes = 0
    successes_lock = threading.Lock()
    previous_interval = sys.getswitchinterval()

    def consume() -> None:
        nonlocal successes
        for _ in range(1_000):
            try:
                budget.consume(1)
            except pdf_images.PdfImageError:
                continue
            with successes_lock:
                successes += 1

    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=consume) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
    finally:
        sys.setswitchinterval(previous_interval)

    assert successes == 5_000
    assert budget.remaining_bytes == 0


def test_pdf_image_validation_reserves_pixels_before_decode(monkeypatch) -> None:
    """The aggregate pixel budget must be checked before pixels are decoded."""
    monkeypatch.setattr(
        pdf_images,
        "_normalize_pdf_raster_mode",
        lambda *_args, **_kwargs: pytest.fail("image must not be decoded"),
    )
    budget = pdf_images.PdfImageDownloadBudget(
        float("inf"), 1024 * 1024, remaining_pixels=100
    )

    with pytest.raises(pdf_images.PdfImageError, match="pixel budget"):
        pdf_images._validate_pdf_image(_png_bytes(width=48, height=32), budget=budget)

    assert budget.remaining_pixels == 100


def test_pdf_image_validation_charges_and_refunds_pixels() -> None:
    """Valid images keep their pixel reservation; invalid ones refund it."""
    budget = pdf_images.PdfImageDownloadBudget(
        float("inf"), 1024 * 1024, remaining_pixels=10_000
    )

    resource = pdf_images._validate_pdf_image(_png_bytes(width=48, height=32), budget=budget)

    assert resource.pixel_count == 48 * 32
    assert budget.remaining_pixels == 10_000 - 48 * 32

    output = BytesIO()
    Image.new("RGB", (10, 10), "#2563eb").save(output, format="JPEG")
    with pytest.raises(pdf_images.PdfImageError):
        pdf_images._validate_pdf_image(output.getvalue()[:-2], budget=budget)
    assert budget.remaining_pixels == 10_000 - 48 * 32

    encoded = base64.b64encode(_png_bytes(width=10, height=10)).decode("ascii")
    pdf_images.decode_pdf_data_image(f"data:image/png;base64,{encoded}", budget=budget)
    assert budget.remaining_pixels == 10_000 - 48 * 32 - 100
    with pytest.raises(pdf_images.PdfImageError, match="pixel budget"):
        budget.consume_pixels(10_000)
    budget.refund_pixels(100)
    assert budget.remaining_pixels == 10_000 - 48 * 32


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


def test_pdf_image_validation_rejects_truncated_jpeg() -> None:
    """Raster validation must decode pixels instead of checking headers alone."""
    output = BytesIO()
    Image.new("RGB", (64, 64), "red").save(output, format="JPEG")

    with pytest.raises(pdf_images.PdfImageError, match="image content is invalid"):
        pdf_images._validate_pdf_image(output.getvalue()[:-2])


def test_pdf_image_validation_normalizes_unsupported_raster_mode() -> None:
    """Raster modes unsupported by WeasyPrint must become safe 8-bit PNGs."""
    output = BytesIO()
    Image.new("I;16", (10, 10), 32_768).save(output, format="PNG")

    resource = pdf_images._validate_pdf_image(output.getvalue())

    assert resource.mime_type == "image/png"
    with Image.open(BytesIO(resource.content)) as image:
        assert image.mode in {"L", "LA", "RGB", "RGBA"}


def test_pdf_image_validation_rejects_svg_external_resource() -> None:
    """An embedded SVG must not create a second unvalidated network request."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg">'
        b'<image href="https://private.example/image.png" />'
        b"</svg>"
    )

    with pytest.raises(pdf_images.PdfImageError, match="external resource"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_svg_data_resource() -> None:
    """Nested data images must not bypass raster validation limits."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg">'
        b'<image href="data:image/png;base64,iVBORw0KGgo=" />'
        b"</svg>"
    )

    with pytest.raises(pdf_images.PdfImageError, match="external resource"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_wraps_unknown_xml_encoding() -> None:
    """Malformed SVG encodings must remain recoverable image failures."""
    svg = (
        b'<?xml version="1.0" encoding="not-a-real-encoding"?>'
        b'<svg xmlns="http://www.w3.org/2000/svg" />'
    )

    with pytest.raises(pdf_images.PdfImageError, match="SVG content is invalid"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_detects_svg_after_leading_comment() -> None:
    """Valid SVGs may contain comments before the document element."""
    svg = (
        b"<!-- generated image -->"
        b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" />'
    )

    resource = pdf_images._validate_pdf_image(svg)

    assert resource == pdf_images.PdfImageResource(svg, "image/svg+xml")


def test_pdf_image_validation_detects_utf16_svg() -> None:
    """A UTF-16 BOM must route SVG content through the XML validator."""
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" />'.encode(
        "utf-16"
    )

    resource = pdf_images._validate_pdf_image(svg)

    assert resource.mime_type == "image/svg+xml"


def test_pdf_image_validation_detects_bomless_utf16_svg_after_whitespace() -> None:
    """Encoded leading whitespace must not hide a BOM-less UTF-16 SVG."""
    svg = (
        ' \n<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" />'
    ).encode("utf-16-le")

    resource = pdf_images._validate_pdf_image(svg)

    assert resource.mime_type == "image/svg+xml"


def test_pdf_image_validation_rejects_utf16_doctype() -> None:
    """DTD detection must account for multibyte XML encodings."""
    svg = (
        '<!DOCTYPE svg [<!ENTITY payload "expanded">]>'
        '<svg xmlns="http://www.w3.org/2000/svg"><text>&payload;</text></svg>'
    ).encode("utf-16")

    with pytest.raises(pdf_images.PdfImageError, match="declarations"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_bomless_utf16_doctype() -> None:
    """DTD detection must recognize XML's BOM-less UTF-16 byte signature."""
    svg = (
        '<?xml version="1.0" encoding="utf-16"?>'
        '<!DOCTYPE svg [<!ENTITY payload "expanded">]>'
        '<svg xmlns="http://www.w3.org/2000/svg"><text>&payload;</text></svg>'
    ).encode("utf-16-le")

    with pytest.raises(pdf_images.PdfImageError, match="declarations"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_excessive_svg_structure() -> None:
    """SVG structure must be bounded before an ElementTree is materialized."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        + '<path d="M0 0" />' * (pdf_images.PDF_SVG_MAX_ELEMENTS + 1)
        + "</svg>"
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="structure limit"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_excessive_svg_path_complexity() -> None:
    """One path attribute must not hide an unbounded rendering workload."""
    path_data = "M0 0" + "L1 1" * (pdf_images.PDF_SVG_MAX_PATH_COMMANDS + 1)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg"><path d="{path_data}" /></svg>'
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="path complexity limit"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_excessive_svg_point_complexity() -> None:
    """Polyline point lists must have the same bounded rendering workload."""
    points = "0,0 " * (pdf_images.PDF_SVG_MAX_POINT_COORDINATES // 2 + 1)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg"><polyline points="{points}" /></svg>'
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="point complexity limit"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_excessive_svg_text() -> None:
    """Renderable SVG character data must be bounded before tree creation."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg"><text>'
        + "a" * (pdf_images.PDF_SVG_MAX_TEXT_CHARS + 1)
        + "</text></svg>"
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="character data limit"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_excessive_svg_css_cascade() -> None:
    """Stylesheet rule matching work must be bounded across SVG elements."""
    rules = "*{fill:red}" * 501
    rectangles = '<rect width="1" height="1" />' * 2_001
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg"><style>{rules}</style>'
        f"{rectangles}</svg>"
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="stylesheet complexity"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_multi_selector_svg_css_cascade() -> None:
    """Comma-separated selectors must each count toward cascade work."""
    selectors = ",".join("*" for _ in range(501))
    rectangles = '<rect width="1" height="1" />' * 2_001
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg"><style>{selectors}'
        f"{{fill:red}}</style>{rectangles}</svg>"
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="stylesheet complexity"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_repeating_gradient() -> None:
    """Repeating gradients must not create an unbounded renderer loop."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="100000000">'
        b'<defs><linearGradient id="gradient" spreadMethod="repeat" '
        b'gradientUnits="userSpaceOnUse" x1="0" x2="1">'
        b'<stop offset="0" stop-color="red" />'
        b'<stop offset="1" stop-color="blue" />'
        b'</linearGradient></defs>'
        b'<rect width="100000000" height="1" fill="url(#gradient)" /></svg>'
    )

    with pytest.raises(pdf_images.PdfImageError, match="repeating gradient"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_circular_svg_use() -> None:
    """Circular local references must not recurse during PDF rendering."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg">'
        b'<g id="loop"><use href=" #loop " /></g>'
        b"</svg>"
    )

    with pytest.raises(pdf_images.PdfImageError, match="circular"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_excessive_svg_use_expansion() -> None:
    """Acyclic local references must not expand exponentially at render time."""
    definitions = ['<g id="node-0"><rect width="1" height="1" /></g>']
    definitions.extend(
        f'<g id="node-{index}"><use href="#node-{index - 1}" />'
        f'<use href="#node-{index - 1}" /></g>'
        for index in range(1, 16)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg"><defs>'
        + "".join(definitions)
        + '</defs><use href="#node-15" /></svg>'
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="expansion limit"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_aggregate_root_svg_use_expansion() -> None:
    """Duplicate document-level references must share one expansion budget."""
    children = '<rect width="1" height="1" />' * 500
    references = '<use href="#shape" />' * 500
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        f'<defs><g id="shape">{children}</g></defs>{references}</svg>'
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="expansion limit"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_visible_id_svg_use_expansion() -> None:
    """References nested under a visible unreferenced ID count in the document."""
    children = '<rect width="1" height="1" />' * 400
    references = '<use href="#shape" />' * 400
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        f'<defs><g id="shape">{children}</g></defs>'
        f'<g id="visible">{references}</g></svg>'
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="expansion limit"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_deep_svg_inheritance_chain() -> None:
    """Gradient and pattern inheritance must stay below renderer recursion."""
    definitions = "".join(
        f'<linearGradient id="gradient-{index}" href="#gradient-{index + 1}" />'
        for index in range(pdf_images.PDF_SVG_MAX_INHERITANCE_DEPTH + 1)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg"><defs>'
        + definitions
        + '<linearGradient id="gradient-end" /></defs></svg>'
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="inheritance depth"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_circular_svg_inheritance() -> None:
    """Local non-use resource inheritance must reject cycles."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg"><defs>'
        b'<linearGradient id="first" href="#second" />'
        b'<linearGradient id="second" href="#first" />'
        b"</defs></svg>"
    )

    with pytest.raises(pdf_images.PdfImageError, match="circular inheritance"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_circular_svg_presentation_reference() -> None:
    """Mask, clip-path, and marker resources must not reference themselves."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg"><defs>'
        b'<mask id="loop" mask="url(#loop)"><rect width="1" height="1" /></mask>'
        b'</defs><rect width="10" height="10" mask="url(#loop)" /></svg>'
    )

    with pytest.raises(pdf_images.PdfImageError, match="circular"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_decodes_css_escaped_presentation_reference() -> None:
    """CSS escapes must not hide a circular local presentation reference."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg"><defs>'
        b'<mask id="loop" style="mask:u\\72l(#loop)">'
        b'<rect width="1" height="1" /></mask></defs></svg>'
    )

    with pytest.raises(pdf_images.PdfImageError, match="circular"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_stylesheet_presentation_reference() -> None:
    """Stylesheet presentation URLs must not bypass graph validation."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg"><style>'
        b'.loop { mask: url(#mask); }</style><defs>'
        b'<mask id="mask"><rect class="loop" width="1" height="1" /></mask>'
        b'</defs><rect class="loop" width="10" height="10" /></svg>'
    )

    with pytest.raises(
        pdf_images.PdfImageError,
        match="stylesheet presentation reference",
    ):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_normalizes_encoded_fragment_ids() -> None:
    """IRI-equivalent encoded IDs and references must form the same graph node."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<g id="%C3%A9"><use href="#é" /></g></svg>'
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="circular"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_rejects_marker_expansion() -> None:
    """Repeated markers must contribute every painted instance to the budget."""
    marker_children = '<rect width="1" height="1" />' * 100
    points = "0,0 " * 500
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg"><defs>'
        f'<marker id="dot">{marker_children}</marker></defs>'
        f'<polyline points="{points}" marker-mid="url(#dot)" /></svg>'
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="expansion limit"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_handles_deep_acyclic_svg_use_chain() -> None:
    """Reference validation must not depend on Python recursion depth."""
    definitions = "".join(
        f'<g id="node-{index}"><use href="#node-{index + 1}" /></g>'
        for index in range(1_100)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg"><defs>{definitions}'
        '<g id="node-1100"><rect width="1" height="1" /></g></defs>'
        '<use href="#node-0" /></svg>'
    ).encode()

    resource = pdf_images._validate_pdf_image(svg)

    assert resource.mime_type == "image/svg+xml"


def test_pdf_image_validation_rejects_renderer_unsafe_svg_depth() -> None:
    """Accepted nesting must remain below WeasyPrint's recursion ceiling."""
    depth = pdf_images.PDF_SVG_MAX_DEPTH + 1
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        + "".join(f'<g id="node-{index}">' for index in range(depth))
        + '<rect width="1" height="1" />'
        + "</g>" * depth
        + "</svg>"
    ).encode()

    with pytest.raises(pdf_images.PdfImageError, match="structure limit"):
        pdf_images._validate_pdf_image(svg)


def test_pdf_image_validation_preserves_svg_hyperlink() -> None:
    """Navigation links do not load resources and may remain in safe SVGs."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg">'
        b'<a href="https://example.com/docs"><text>Docs</text></a>'
        b"</svg>"
    )

    resource = pdf_images._validate_pdf_image(svg)

    assert resource.mime_type == "image/svg+xml"


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
    html = response_exports.build_pdf_html_document("Hâ‚‚O and xÂ²")  # noqa: RUF001 -- intentional mojibake fixture

    assert "H<sub>2</sub>O" in html
    assert "x<sup>2</sup>" in html
    assert "â" not in html
    assert "Â" not in html


def test_build_pdf_html_document_repairs_mojibake_with_unicode_text() -> None:
    """Verify mojibake repair still works when surrounding text is Unicode."""
    html = response_exports.build_pdf_html_document("Δ sample: Hâ‚‚O and xÂ²")  # noqa: RUF001 -- intentional mojibake fixture

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


# ---------------------------------------------------------- table of contents

TOC_MARKDOWN = """# Overview

Intro text.

## Setup

Install it.

### Configure `deepagent.toml`

Edit the file.

## Setup

A second section with the same title.

#### Deep detail

Not listed.
"""


def _toc_entries(html: str) -> list[tuple[str, str, str]]:
    """Return (level class, href, text) for each contents entry."""
    import re

    nav = re.search(r'<nav class="pdf-toc">(.*?)</nav>', html, re.S)
    assert nav, "no contents section"
    return re.findall(
        r'<li class="(pdf-toc-level-\d)"><a href="(#toc-\d+)">(.*?)</a></li>', nav.group(1)
    )


def test_pdf_html_lists_headings_in_a_contents_section() -> None:
    html = response_exports.build_pdf_html_document(TOC_MARKDOWN)

    assert _toc_entries(html) == [
        ("pdf-toc-level-1", "#toc-1", "Overview"),
        ("pdf-toc-level-2", "#toc-2", "Setup"),
        ("pdf-toc-level-3", "#toc-3", "Configure deepagent.toml"),
        ("pdf-toc-level-2", "#toc-4", "Setup"),
    ]
    assert '<h1 id="toc-1">Overview</h1>' in html
    assert '<h2 id="toc-4">Setup</h2>' in html
    assert "<h4>Deep detail</h4>" in html
    assert html.index('<nav class="pdf-toc">') < html.index('<h1 id="toc-1">')


def test_pdf_html_skips_contents_below_three_headings() -> None:
    html = response_exports.build_pdf_html_document("# One\n\ntext\n\n## Two\n\nmore")

    assert '<nav class="pdf-toc">' not in html
    assert "<h1>One</h1>" in html


def test_pdf_contents_escape_heading_text() -> None:
    html = response_exports.build_pdf_html_document(
        "# A & B\n\n## *Émphasis* <tag>\n\n## Ünïcode 日本語\n"
    )

    texts = [text for _, _, text in _toc_entries(html)]
    assert texts == ["A &amp; B", "Émphasis &lt;tag&gt;", "Ünïcode 日本語"]


def test_pdf_render_includes_contents_page_numbers_and_bookmarks() -> None:
    pdf = PdfReader(BytesIO(response_exports.build_pdf_bytes(TOC_MARKDOWN)))

    first_page = pdf.pages[0].extract_text()
    assert "Contents" in first_page
    assert "Overview" in first_page
    assert any(char.isdigit() for char in first_page)
    outline_titles = []
    stack = list(pdf.outline)
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack[:0] = item
        else:
            outline_titles.append(item.title)
    assert "Overview" in outline_titles
    assert "Configure deepagent.toml" in outline_titles


def test_pdf_contents_use_image_alt_text_for_image_headings() -> None:
    html = response_exports.build_pdf_html_document(
        "# ![Architecture](diagram.png)\n\n## Data ![flow](flow.png) path\n\n## ![](blank.png)\n"
    )

    texts = [text for _, _, text in _toc_entries(html)]
    assert texts == ["Architecture", "Data flow path", "Section 3"]
