"""Bounded, public-only image downloads for response PDF exports."""

from __future__ import annotations

import http.client
import ipaddress
import math
import queue
import socket
import ssl
import threading
import time
import warnings
import zlib
from dataclasses import dataclass
from io import BytesIO
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

from PIL import Image, UnidentifiedImageError


PDF_IMAGE_MAX_BYTES = 10 * 1024 * 1024
PDF_IMAGE_MAX_PIXELS = 25_000_000
PDF_IMAGE_MAX_REDIRECTS = 3
PDF_IMAGE_TIMEOUT_SECONDS = 5.0
PDF_IMAGE_USER_AGENT = "ChainAgents-PDF/1.0"
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_RASTER_MIME_TYPES = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class PdfImageError(ValueError):
    """An image cannot be safely included in a response PDF."""


class _PdfImageConnectionError(PdfImageError):
    """A validated address could not complete an HTTP request."""


@dataclass(frozen=True)
class PdfImageResource:
    """Validated image bytes ready for WeasyPrint."""

    content: bytes
    mime_type: str


@dataclass
class PdfImageDownloadBudget:
    """Shared wall-clock and transferred-byte budget for one PDF export."""

    deadline: float
    remaining_bytes: int

    def consume(self, size: int) -> None:
        if size > self.remaining_bytes:
            raise PdfImageError("PDF image download budget exceeded")
        self.remaining_bytes -= size


@dataclass(frozen=True)
class _PdfHttpResponse:
    status: int
    headers: dict[str, str]
    content: bytes


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to a resolved public address while authenticating the hostname."""

    def __init__(self, host: str, port: int, address: str, *, timeout: float) -> None:
        tls_context = ssl.create_default_context()
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            context=tls_context,
        )
        self._address = address
        self._tls_context = tls_context

    def connect(self) -> None:
        sock = socket.create_connection(
            (self._address, self.port),
            self.timeout,
        )
        self.sock = self._tls_context.wrap_socket(sock, server_hostname=self.host)


def download_pdf_image(
    url: str,
    *,
    deadline: float,
    max_bytes: int = PDF_IMAGE_MAX_BYTES,
    budget: PdfImageDownloadBudget | None = None,
) -> PdfImageResource:
    """Download and validate one public HTTP(S) image."""
    download_budget = budget or PdfImageDownloadBudget(deadline, max_bytes)
    current_url = url
    for redirect_count in range(PDF_IMAGE_MAX_REDIRECTS + 1):
        if time.monotonic() >= deadline:
            raise PdfImageError("image download budget expired")
        normalized_url, host, port, addresses = _resolve_public_image_url(
            current_url,
            deadline=download_budget.deadline,
        )
        response: _PdfHttpResponse | None = None
        last_connection_error: _PdfImageConnectionError | None = None
        for address in addresses:
            try:
                response = _request_pdf_image_url(
                    normalized_url,
                    host=host,
                    port=port,
                    address=address,
                    max_bytes=max_bytes,
                    budget=download_budget,
                )
            except _PdfImageConnectionError as exc:
                last_connection_error = exc
                if time.monotonic() >= download_budget.deadline:
                    raise PdfImageError("image download budget expired") from exc
                continue
            break
        if response is None:
            raise last_connection_error or PdfImageError("image download failed")
        if response.status in _REDIRECT_STATUSES:
            location = response.headers.get("location", "").strip()
            if not location or redirect_count >= PDF_IMAGE_MAX_REDIRECTS:
                raise PdfImageError("image redirect limit exceeded")
            current_url = urljoin(normalized_url, location)
            continue
        if not 200 <= response.status < 300:
            raise PdfImageError(f"image server returned HTTP {response.status}")
        return _validate_pdf_image(response.content, max_bytes=max_bytes)
    raise PdfImageError("image redirect limit exceeded")


def _resolve_public_image_url(
    url: str,
    *,
    deadline: float = float("inf"),
) -> tuple[str, str, int, tuple[str, ...]]:
    """Resolve one strict HTTP(S) URL to a public address."""
    if not url or any(char.isspace() or ord(char) < 33 for char in url):
        raise PdfImageError("invalid image URL")
    try:
        parsed = urlsplit(url)
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise PdfImageError("invalid image URL") from exc
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    if (
        scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise PdfImageError("image URL must be public HTTP or HTTPS")
    try:
        ascii_host = host.encode("idna").decode("ascii")
        records = _getaddrinfo_before_deadline(
            ascii_host,
            port,
            deadline=deadline,
        )
    except (OSError, UnicodeError) as exc:
        raise PdfImageError("image host could not be resolved") from exc

    addresses: list[str] = []
    for record in records:
        value = record[4][0]
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise PdfImageError("image host resolved to an invalid address") from exc
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_global:
            raise PdfImageError("image host resolved outside the public internet")
        normalized = str(address)
        if normalized not in addresses:
            addresses.append(normalized)
    if not addresses:
        raise PdfImageError("image host could not be resolved")

    netloc_host = f"[{ascii_host}]" if ":" in ascii_host else ascii_host
    default_port = 443 if scheme == "https" else 80
    netloc = netloc_host if port == default_port else f"{netloc_host}:{port}"
    path = quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~")
    query = quote(parsed.query, safe="%/?@!$&'()*+,;=:-._~")
    normalized_url = urlunsplit((scheme, netloc, path, query, ""))
    return normalized_url, ascii_host, port, tuple(addresses)


def _getaddrinfo_before_deadline(
    host: str,
    port: int,
    *,
    deadline: float,
) -> list[tuple[object, ...]]:
    """Resolve a host without holding the export worker past its deadline."""
    results: queue.Queue[object] = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            value: object = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            value = exc
        results.put(value)

    resolver = threading.Thread(target=resolve, daemon=True)
    resolver.start()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PdfImageError("image download budget expired")
    try:
        result = results.get() if math.isinf(remaining) else results.get(timeout=remaining)
    except queue.Empty as exc:
        raise PdfImageError("image download budget expired") from exc
    if isinstance(result, OSError):
        raise result
    return result  # type: ignore[return-value]


def _request_pdf_image_url(
    url: str,
    *,
    host: str,
    port: int,
    address: str,
    max_bytes: int,
    budget: PdfImageDownloadBudget,
) -> _PdfHttpResponse:
    """Request one already-resolved URL without another DNS lookup."""
    parsed = urlsplit(url)
    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = _PinnedHTTPSConnection(
            host,
            port,
            address,
            timeout=PDF_IMAGE_TIMEOUT_SECONDS,
        )
    else:
        connection = http.client.HTTPConnection(
            address,
            port=port,
            timeout=PDF_IMAGE_TIMEOUT_SECONDS,
        )
    default_port = 443 if parsed.scheme == "https" else 80
    header_host = f"[{host}]" if ":" in host else host
    host_header = header_host if port == default_port else f"{header_host}:{port}"
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    content = b""
    deadline_timer: threading.Timer | None = None
    try:
        remaining_seconds = budget.deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise PdfImageError("image download budget expired")
        connection.timeout = min(PDF_IMAGE_TIMEOUT_SECONDS, remaining_seconds)
        deadline_timer = threading.Timer(
            remaining_seconds,
            _close_expired_connection,
            args=(connection,),
        )
        deadline_timer.daemon = True
        deadline_timer.start()
        connection.request(
            "GET",
            path,
            headers={
                "Accept": "image/png,image/jpeg,image/webp,image/gif,image/svg+xml",
                "Host": host_header,
                "User-Agent": PDF_IMAGE_USER_AGENT,
            },
        )
        response = connection.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        if response.status in _REDIRECT_STATUSES or not 200 <= response.status < 300:
            return _PdfHttpResponse(response.status, headers, b"")
        declared_length = headers.get("content-length")
        if declared_length:
            try:
                content_length = int(declared_length)
            except ValueError as exc:
                raise PdfImageError("image has an invalid content length") from exc
            if content_length > max_bytes:
                raise PdfImageError("image exceeds the download size limit")
        chunks: list[bytes] = []
        transferred = 0
        while True:
            remaining_seconds = budget.deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise PdfImageError("image download budget expired")
            if connection.sock is not None:
                connection.sock.settimeout(
                    min(PDF_IMAGE_TIMEOUT_SECONDS, remaining_seconds)
                )
            chunk = response.read(min(64 * 1024, max_bytes + 1 - transferred))
            if not chunk:
                break
            budget.consume(len(chunk))
            chunks.append(chunk)
            transferred += len(chunk)
            if transferred > max_bytes:
                raise PdfImageError("image exceeds the download size limit")
        content = b"".join(chunks)
    except (OSError, http.client.HTTPException) as exc:
        raise _PdfImageConnectionError("image download failed") from exc
    finally:
        if deadline_timer is not None:
            deadline_timer.cancel()
        connection.close()
    if len(content) > max_bytes:
        raise PdfImageError("image exceeds the download size limit")
    content = _decompress_pdf_image(
        content,
        encoding=headers.get("content-encoding", "").lower(),
        max_bytes=max_bytes,
    )
    return _PdfHttpResponse(response.status, headers, content)


def _close_expired_connection(connection: http.client.HTTPConnection) -> None:
    """Interrupt a request that remains active past the export deadline."""
    sock = connection.sock
    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
    connection.close()


def _decompress_pdf_image(content: bytes, *, encoding: str, max_bytes: int) -> bytes:
    """Decode supported content encodings without unbounded expansion."""
    if encoding not in {"gzip", "deflate"}:
        return content
    window_bits = [zlib.MAX_WBITS | 16] if encoding == "gzip" else [zlib.MAX_WBITS, -15]
    last_error: zlib.error | None = None
    for bits in window_bits:
        decoder = zlib.decompressobj(bits)
        try:
            decoded = decoder.decompress(content, max_bytes + 1)
            if len(decoded) > max_bytes or decoder.unconsumed_tail:
                raise PdfImageError("image exceeds the download size limit")
            decoded += decoder.flush(max_bytes + 1 - len(decoded))
        except zlib.error as exc:
            last_error = exc
            continue
        if len(decoded) > max_bytes:
            raise PdfImageError("image exceeds the download size limit")
        if not decoder.eof or decoder.unused_data:
            raise PdfImageError("image compression is invalid")
        return decoded
    raise PdfImageError("image compression is invalid") from last_error


def _validate_pdf_image(
    content: bytes,
    *,
    max_bytes: int = PDF_IMAGE_MAX_BYTES,
) -> PdfImageResource:
    """Validate supported image bytes and normalize animated GIFs."""
    if not content:
        raise PdfImageError("image response was empty")
    stripped = content.removeprefix(b"\xef\xbb\xbf").lstrip()
    if stripped.startswith(b"<"):
        return _validate_svg_image(content)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                image_format = str(image.format or "").upper()
                if image_format not in _RASTER_MIME_TYPES:
                    raise PdfImageError("image format is not supported")
                if image.width * image.height > PDF_IMAGE_MAX_PIXELS:
                    raise PdfImageError("image dimensions exceed the pixel limit")
                if image_format == "GIF":
                    image.seek(0)
                    output = BytesIO()
                    image.convert("RGBA").save(output, format="PNG")
                    normalized = output.getvalue()
                    if len(normalized) > max_bytes:
                        raise PdfImageError("image exceeds the download size limit")
                    return PdfImageResource(normalized, "image/png")
                image.verify()
    except PdfImageError:
        raise
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise PdfImageError("image content is invalid") from exc
    return PdfImageResource(content, _RASTER_MIME_TYPES[image_format])


def _validate_svg_image(content: bytes) -> PdfImageResource:
    """Accept SVG only when it contains no external resource references."""
    upper = content.upper()
    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
        raise PdfImageError("SVG declarations are not supported")
    try:
        root = ElementTree.fromstring(content)
    except (ElementTree.ParseError, LookupError, ValueError) as exc:
        raise PdfImageError("SVG content is invalid") from exc
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        raise PdfImageError("image content is not SVG")
    for element in root.iter():
        for name, value in element.attrib.items():
            normalized = value.strip().lower()
            if name.lower().endswith("href") and normalized and not normalized.startswith(
                "#"
            ):
                raise PdfImageError("SVG contains an external resource")
            if _svg_css_has_external_resource(normalized):
                raise PdfImageError("SVG contains an external resource")
        if element.tag.rsplit("}", 1)[-1].lower() == "style" and (
            "@import" in (element.text or "").lower()
            or _svg_css_has_external_resource(element.text or "")
        ):
            raise PdfImageError("SVG contains an external resource")
    return PdfImageResource(content, "image/svg+xml")


def _svg_css_has_external_resource(value: str) -> bool:
    """Return whether a CSS value contains a non-local URL reference."""
    remaining = value.lower()
    while "url(" in remaining:
        remaining = remaining.split("url(", 1)[1]
        reference, separator, remaining = remaining.partition(")")
        if not separator:
            return True
        normalized = reference.strip(" \t\r\n'\"")
        if normalized and not normalized.startswith("#"):
            return True
    return False
