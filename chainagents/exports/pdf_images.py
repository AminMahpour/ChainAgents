"""Bounded, public-only image downloads for response PDF exports."""

from __future__ import annotations

import base64
import binascii
import contextlib
import email.utils
import http.client
import ipaddress
import logging
import math
import queue
import random
import re
import socket
import ssl
import threading
import time
import zlib
from dataclasses import dataclass
from datetime import UTC
from io import BytesIO
from typing import cast
from urllib.parse import quote, unquote, unquote_to_bytes, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree
from xml.parsers import expat

from PIL import Image, UnidentifiedImageError


PDF_IMAGE_MAX_BYTES = 10 * 1024 * 1024
PDF_IMAGE_MAX_PIXELS = 25_000_000
PDF_SVG_MAX_ELEMENTS = 10_000
PDF_SVG_MAX_DEPTH = 256
PDF_SVG_MAX_ATTRIBUTES = 100_000
PDF_SVG_MAX_ATTRIBUTE_VALUE_CHARS = 1_000_000
PDF_SVG_MAX_PATH_DATA_CHARS = 500_000
PDF_SVG_MAX_PATH_COMMANDS = 25_000
PDF_SVG_MAX_POINT_COORDINATES = 50_000
PDF_SVG_MAX_CHARACTER_DATA_CHARS = 1_000_000
PDF_SVG_MAX_TEXT_CHARS = 250_000
PDF_SVG_MAX_STYLE_CHARS = 100_000
PDF_SVG_MAX_STYLE_RULES = 1_000
PDF_SVG_MAX_STYLE_SELECTORS = 1_000
PDF_SVG_MAX_STYLE_DECLARATIONS = 10_000
PDF_SVG_MAX_STYLE_MATCH_WORK = 1_000_000
PDF_SVG_MAX_USE_EXPANSION = 4_096
PDF_SVG_MAX_TOTAL_EXPANSION = 25_000
PDF_SVG_MAX_INHERITANCE_DEPTH = 128
PDF_IMAGE_MAX_REDIRECTS = 3
PDF_IMAGE_TIMEOUT_SECONDS = 5.0
PDF_IMAGE_USER_AGENT = "ChainAgents-PDF/1.0"
PDF_IMAGE_PER_IMAGE_SECONDS = 15.0
PDF_IMAGE_MAX_ATTEMPTS = 3
PDF_IMAGE_RETRY_BASE_SECONDS = 0.5
PDF_IMAGE_RETRY_AFTER_MAX_SECONDS = 5.0
PDF_IMAGE_MIN_ATTEMPT_SECONDS = 1.0
PDF_IMAGE_STALL_WINDOW_SECONDS = 5.0
PDF_IMAGE_STALL_MIN_BYTES = 4096
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
_RASTER_MIME_TYPES = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
_SVG_PATH_COMMANDS = frozenset("MmZzLlHhVvCcSsQqTtAa")
_SVG_NUMBER_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)
_SVG_LOCAL_URL_RE = re.compile(
    r"url\(\s*['\"]?#([^'\"\s)]+)['\"]?\s*\)",
    re.IGNORECASE,
)
_SVG_PRESENTATION_STYLE_RE = re.compile(
    r"(?:^|;)\s*(mask|clip-path|filter|marker(?:-start|-mid|-end)?)\s*:\s*([^;]+)",
    re.IGNORECASE,
)
_CSS_ESCAPE_RE = re.compile(
    r"\\(?:([0-9a-fA-F]{1,6})(?:\r\n|[ \t\r\n\f])?|(\r\n|[\n\r\f])|(.))",
    re.DOTALL,
)
_SVG_STYLESHEET_PRESENTATION_RE = re.compile(
    r"(?:^|[;{])\s*(?:mask|clip-path|filter|marker(?:-start|-mid|-end)?)"
    r"\s*:[^;{}]*url\(\s*['\"]?#",
    re.IGNORECASE,
)
_PDF_RASTER_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})
_RETRY_AFTER_SECONDS_RE = re.compile(r"\d+", re.ASCII)
_CONTENT_LENGTH_RE = re.compile(r"[0-9]+")
_CONTENT_LENGTH_MAX_DIGITS = 18

logger = logging.getLogger(__name__)


class PdfImageError(ValueError):
    """An image cannot be safely included in a response PDF."""


class PdfImageRetryableError(PdfImageError):
    """A transient failure that a later attempt may not repeat."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class _PdfImageConnectionError(PdfImageRetryableError):
    """A validated address could not complete an HTTP request."""


@dataclass(frozen=True)
class PdfImageResource:
    """Validated image bytes ready for WeasyPrint."""

    content: bytes
    mime_type: str
    pixel_count: int = 0


def _require_non_negative(amount: int) -> None:
    if amount < 0:
        raise ValueError("PDF image budget amounts must not be negative")


class PdfImageDownloadBudget:
    """Thread-safe wall-clock, byte and pixel budget shared by one PDF export."""

    def __init__(
        self,
        deadline: float,
        remaining_bytes: int,
        remaining_pixels: int = PDF_IMAGE_MAX_PIXELS,
    ) -> None:
        self.deadline = deadline
        self._remaining_bytes = remaining_bytes
        self._remaining_pixels = remaining_pixels
        self._active_reservations = 0
        self._lock = threading.Lock()
        self._settled = threading.Condition(self._lock)

    @property
    def remaining_bytes(self) -> int:
        with self._lock:
            return self._remaining_bytes

    @property
    def remaining_pixels(self) -> int:
        with self._lock:
            return self._remaining_pixels

    def consume(self, size: int) -> None:
        _require_non_negative(size)
        with self._lock:
            if size > self._remaining_bytes:
                raise PdfImageError("PDF image download budget exceeded")
            self._remaining_bytes -= size

    def reserve(self, size: int, *, deadline: float = math.inf) -> None:
        """Set aside bytes for a declared body until settle_reservation.

        While other reservations are active they may still return unused
        bytes, so a reservation that does not fit yet waits for them to settle
        (until ``deadline``). It fails only when nothing else is reserved and
        the bytes still do not fit.
        """
        _require_non_negative(size)
        with self._settled:
            while size > self._remaining_bytes:
                if self._active_reservations == 0:
                    raise PdfImageError("PDF image download budget exceeded")
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise PdfImageError("image download time limit reached")
                self._settled.wait(
                    None if math.isinf(remaining_seconds) else remaining_seconds
                )
            self._remaining_bytes -= size
            self._active_reservations += 1

    def settle_reservation(self, unused: int) -> None:
        """End one reservation, returning the bytes its body did not use."""
        _require_non_negative(unused)
        with self._settled:
            if self._active_reservations == 0:
                raise RuntimeError("no active PDF image reservation to settle")
            self._remaining_bytes += unused
            self._active_reservations -= 1
            self._settled.notify_all()

    def consume_pixels(self, count: int) -> None:
        _require_non_negative(count)
        with self._lock:
            if count > self._remaining_pixels:
                raise PdfImageError("PDF image pixel budget exceeded")
            self._remaining_pixels -= count

    def refund_pixels(self, count: int) -> None:
        _require_non_negative(count)
        with self._lock:
            self._remaining_pixels += count


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
    max_pixels: int = PDF_IMAGE_MAX_PIXELS,
    budget: PdfImageDownloadBudget | None = None,
    per_image_seconds: float = PDF_IMAGE_PER_IMAGE_SECONDS,
) -> PdfImageResource:
    """Download and validate one public HTTP(S) image with bounded retries."""
    download_budget = (
        budget if budget is not None else PdfImageDownloadBudget(deadline, max_bytes)
    )
    export_deadline = min(deadline, download_budget.deadline)
    image_deadline = min(export_deadline, time.monotonic() + per_image_seconds)
    attempts = 0
    current_url = url
    for redirect_count in range(PDF_IMAGE_MAX_REDIRECTS + 1):
        while True:
            _raise_if_past_deadline(export_deadline, image_deadline)
            attempts += 1
            try:
                normalized_url, response = _fetch_pdf_image_hop(
                    current_url,
                    deadline=image_deadline,
                    max_bytes=max_bytes,
                    budget=download_budget,
                )
            except PdfImageRetryableError as exc:
                wait = (
                    _retry_wait(attempts, exc.retry_after)
                    if attempts < PDF_IMAGE_MAX_ATTEMPTS
                    else None
                )
                if (
                    wait is None
                    or time.monotonic() + wait + PDF_IMAGE_MIN_ATTEMPT_SECONDS
                    > image_deadline
                ):
                    raise
                logger.debug(
                    "Retrying PDF image %s (attempt %d/%d) after %.2fs: %s",
                    _pdf_image_log_identifier(current_url),
                    attempts + 1,
                    PDF_IMAGE_MAX_ATTEMPTS,
                    wait,
                    exc,
                )
                _sleep_before_retry(wait)
                continue
            break
        if response.status in _REDIRECT_STATUSES:
            location = response.headers.get("location", "").strip()
            if not location or redirect_count >= PDF_IMAGE_MAX_REDIRECTS:
                raise PdfImageError("image redirect limit exceeded")
            current_url = urljoin(normalized_url, location)
            continue
        return _validate_pdf_image(
            response.content,
            max_bytes=max_bytes,
            max_pixels=max_pixels,
            budget=download_budget,
        )
    raise PdfImageError("image redirect limit exceeded")


def _fetch_pdf_image_hop(
    url: str,
    *,
    deadline: float,
    max_bytes: int,
    budget: PdfImageDownloadBudget,
) -> tuple[str, _PdfHttpResponse]:
    """Resolve, validate and request one redirect hop across its addresses."""
    normalized_url, host, port, addresses = _resolve_public_image_url(
        url,
        deadline=deadline,
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
                deadline=deadline,
                budget=budget,
            )
        except _PdfImageConnectionError as exc:
            last_connection_error = exc
            if time.monotonic() >= deadline:
                raise PdfImageError("image download time limit reached") from exc
            continue
        break
    if response is None:
        raise last_connection_error or PdfImageError("image download failed")
    if response.status in _REDIRECT_STATUSES or 200 <= response.status < 300:
        return normalized_url, response
    message = f"image server returned HTTP {response.status}"
    if response.status in _RETRYABLE_STATUSES:
        raise PdfImageRetryableError(
            message,
            retry_after=_parse_retry_after(
                response.headers.get("retry-after"),
                now=time.time(),
            ),
        )
    raise PdfImageError(message)


def _raise_if_past_deadline(export_deadline: float, image_deadline: float) -> None:
    now = time.monotonic()
    if now >= export_deadline:
        raise PdfImageError("image download budget expired")
    if now >= image_deadline:
        raise PdfImageError("image download time limit reached")


def _retry_wait(attempt: int, retry_after: float | None) -> float | None:
    """Return the backoff before the next attempt, or None to stop retrying."""
    if retry_after is not None and retry_after > PDF_IMAGE_RETRY_AFTER_MAX_SECONDS:
        return None
    backoff = PDF_IMAGE_RETRY_BASE_SECONDS * 2 ** (attempt - 1) * (0.5 + _retry_jitter())
    return max(backoff, retry_after or 0.0)


def _retry_jitter() -> float:
    return random.random()


def _sleep_before_retry(seconds: float) -> None:
    time.sleep(seconds)


def _parse_retry_after(value: str | None, *, now: float) -> float | None:
    """Parse a Retry-After header as delta seconds or an HTTP-date."""
    if value is None:
        return None
    value = value.strip()
    if _RETRY_AFTER_SECONDS_RE.fullmatch(value):
        try:
            return float(int(value))
        except OverflowError:
            return math.inf
    try:
        retry_at = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, retry_at.timestamp() - now)


def _pdf_image_log_identifier(source: str) -> str:
    """Return a bounded image identifier without query tokens or data payloads."""
    if source.lower().startswith("data:"):
        return source.partition(",")[0][:80]
    try:
        parsed = urlsplit(source)
        hostname = parsed.hostname
    except ValueError:
        return "invalid-image-source"
    if parsed.scheme and hostname:
        return f"{parsed.scheme.lower()}://{hostname}"[:160]
    return "invalid-image-source"


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
        # getaddrinfo's 5th element is always a socket address tuple (2-tuple
        # for IPv4, 4-tuple for IPv6) whose first item is the address string.
        sockaddr = cast("tuple[str, ...]", record[4])
        value = sockaddr[0]
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise PdfImageError("image host resolved to an invalid address") from exc
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        if (
            not address.is_global
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
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
        raise PdfImageError("image download time limit reached")
    try:
        result = results.get() if math.isinf(remaining) else results.get(timeout=remaining)
    except queue.Empty as exc:
        raise PdfImageError("image download time limit reached") from exc
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
    deadline: float,
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
    transferred = 0
    reserved_bytes = 0
    holding_reservation = False
    expired = threading.Event()
    deadline_timer: threading.Timer | None = None
    try:
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise PdfImageError("image download time limit reached")
        connection.timeout = min(PDF_IMAGE_TIMEOUT_SECONDS, remaining_seconds)
        deadline_timer = threading.Timer(
            remaining_seconds,
            _close_expired_connection,
            args=(connection, expired),
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
        raw_headers = response.getheaders()
        headers = {key.lower(): value for key, value in raw_headers}
        if response.status in _REDIRECT_STATUSES or not 200 <= response.status < 300:
            return _PdfHttpResponse(response.status, headers, b"")
        content_length = _declared_content_length(raw_headers)
        if content_length is not None:
            if content_length > max_bytes:
                raise PdfImageError("image exceeds the download size limit")
            # Claim the declared transfer size before reading, so parallel
            # downloads cannot each stream part of a body the shared budget
            # cannot finish. Content-Length counts transferred (possibly
            # compressed) bytes, which is exactly what the budget charges;
            # decompression growth is charged by the caller afterwards.
            budget.reserve(content_length, deadline=deadline)
            holding_reservation = True
            reserved_bytes = content_length
        chunks: list[bytes] = []
        window_started = time.monotonic()
        window_bytes = 0
        while True:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise PdfImageError("image download time limit reached")
            if connection.sock is not None:
                connection.sock.settimeout(
                    min(PDF_IMAGE_TIMEOUT_SECONDS, remaining_seconds)
                )
            # read1 returns as soon as any data arrives, so a trickling server
            # cannot keep this loop blocked inside one large read.
            chunk = response.read1(min(64 * 1024, max_bytes + 1 - transferred))
            if not chunk:
                break
            from_reservation = min(reserved_bytes, len(chunk))
            reserved_bytes -= from_reservation
            if len(chunk) > from_reservation:
                budget.consume(len(chunk) - from_reservation)
            chunks.append(chunk)
            transferred += len(chunk)
            if transferred > max_bytes:
                raise PdfImageError("image exceeds the download size limit")
            # The rate is checked only when a read returns, so a trickle is
            # detected within about two windows; that stays inside the image cap.
            now = time.monotonic()
            window_bytes += len(chunk)
            if now - window_started >= PDF_IMAGE_STALL_WINDOW_SECONDS:
                if window_bytes < PDF_IMAGE_STALL_MIN_BYTES:
                    raise PdfImageError("image download stalled")
                window_started = now
                window_bytes = 0
        if expired.is_set():
            raise PdfImageError("image download time limit reached")
        if response.length:
            # read1 reports an early EOF as b"" instead of IncompleteRead, so a
            # body shorter than its Content-Length must be caught here.
            raise _PdfImageConnectionError("image download failed")
        content = b"".join(chunks)
    except (OSError, http.client.HTTPException) as exc:
        raise _classify_request_error(
            exc,
            expired=expired.is_set() or time.monotonic() >= deadline,
            received_body=transferred > 0,
        ) from exc
    finally:
        if holding_reservation:
            budget.settle_reservation(reserved_bytes)
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


def _declared_content_length(raw_headers: list[tuple[str, str]]) -> int | None:
    """Return the body's declared length, rejecting malformed or conflicting values.

    Only plain ASCII digits are accepted: a sign, separators or embedded spaces
    would let a hostile server declare a negative size and inflate the shared
    budget, and disagreeing duplicates could make the reservation differ from
    the length http.client actually reads.
    """
    values = {
        value.strip() for key, value in raw_headers if key.lower() == "content-length"
    }
    if not values:
        return None
    if len(values) != 1:
        raise PdfImageError("image has an invalid content length")
    value = values.pop()
    # Far more digits than any real image size would hit Python's int-string
    # conversion limit (a plain ValueError) before reaching the size check.
    if len(value) > _CONTENT_LENGTH_MAX_DIGITS or not _CONTENT_LENGTH_RE.fullmatch(value):
        raise PdfImageError("image has an invalid content length")
    return int(value)


def _classify_request_error(
    exc: OSError | http.client.HTTPException,
    *,
    expired: bool,
    received_body: bool,
) -> PdfImageError:
    """Split request failures into final errors and retryable connection errors."""
    if expired:
        return PdfImageError("image download time limit reached")
    if isinstance(exc, ssl.SSLCertVerificationError):
        return PdfImageError("image TLS verification failed")
    if isinstance(exc, (http.client.LineTooLong, http.client.InvalidURL)):
        return PdfImageError("image download failed")
    if isinstance(exc, TimeoutError) and received_body:
        # Silence after part of the body arrived is a stall, not a fresh
        # connection problem; retrying would only trickle again.
        return PdfImageError("image download stalled")
    return _PdfImageConnectionError("image download failed")


def _close_expired_connection(
    connection: http.client.HTTPConnection,
    expired: threading.Event | None = None,
) -> None:
    """Interrupt a request that remains active past the image deadline."""
    if expired is not None:
        expired.set()
    sock = connection.sock
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)
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


def decode_pdf_data_image(
    url: str,
    *,
    max_bytes: int = PDF_IMAGE_MAX_BYTES,
    max_pixels: int = PDF_IMAGE_MAX_PIXELS,
    budget: PdfImageDownloadBudget | None = None,
) -> PdfImageResource:
    """Decode and validate one bounded image data URI."""
    header, separator, payload = url.partition(",")
    if not separator or not header[:5].lower() == "data:":
        raise PdfImageError("invalid image data URI")
    metadata = header[5:].split(";")
    if not metadata[0].lower().startswith("image/"):
        raise PdfImageError("data URI must contain an image")
    encoded = unquote_to_bytes(payload)
    if any(parameter.lower() == "base64" for parameter in metadata[1:]):
        if len(encoded) > ((max_bytes + 2) // 3) * 4 + 4:
            raise PdfImageError("image exceeds the download size limit")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PdfImageError("image data URI is invalid") from exc
    else:
        content = encoded
    if len(content) > max_bytes:
        raise PdfImageError("image exceeds the download size limit")
    return _validate_pdf_image(
        content,
        max_bytes=max_bytes,
        max_pixels=max_pixels,
        budget=budget,
    )


def _validate_pdf_image(
    content: bytes,
    *,
    max_bytes: int = PDF_IMAGE_MAX_BYTES,
    max_pixels: int = PDF_IMAGE_MAX_PIXELS,
    budget: PdfImageDownloadBudget | None = None,
) -> PdfImageResource:
    """Validate supported image bytes and normalize animated GIFs.

    When a budget is given, raster pixels are reserved from it after the
    dimensions are read and before any pixel data is decoded, and refunded
    if validation then fails.
    """
    if not content:
        raise PdfImageError("image response was empty")
    stripped = content.removeprefix(b"\xef\xbb\xbf").lstrip()
    if stripped.startswith(b"<") or _has_multibyte_xml_signature(content):
        return _validate_svg_image(content)
    if content.startswith(
        (b"\xff\xfe", b"\xfe\xff", b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00")
    ):
        return _validate_svg_image(content)
    reserved_pixels = 0
    try:
        try:
            # No warnings.catch_warnings() here: it swaps the process-global
            # filter list and is not thread-safe, so parallel validations could
            # leave DecompressionBombWarning configured as an error. The explicit
            # pixel check below (25M) sits far under Pillow's bomb threshold
            # (about 89M warns, 179M raises DecompressionBombError), so every
            # image that is decoded has already passed a stricter limit.
            with Image.open(BytesIO(content)) as image:
                image_format = str(image.format or "").upper()
                if image_format not in _RASTER_MIME_TYPES:
                    raise PdfImageError("image format is not supported")
                if image.width * image.height > min(PDF_IMAGE_MAX_PIXELS, max_pixels):
                    raise PdfImageError("image dimensions exceed the pixel limit")
                pixel_count = image.width * image.height
                if budget is not None:
                    budget.consume_pixels(pixel_count)
                    reserved_pixels = pixel_count
                if image_format == "GIF":
                    image.seek(0)
                    output = BytesIO()
                    image.convert("RGBA").save(output, format="PNG")
                    normalized = output.getvalue()
                    if len(normalized) > max_bytes:
                        raise PdfImageError("image exceeds the download size limit")
                    return PdfImageResource(normalized, "image/png", pixel_count)
                image.verify()
            if image_format == "JPEG" and not content.rstrip().endswith(b"\xff\xd9"):
                raise PdfImageError("image content is invalid")
            with Image.open(BytesIO(content)) as decoded_image:
                decoded_image.load()
                normalized_raster = _normalize_pdf_raster_mode(
                    decoded_image,
                    max_bytes=max_bytes,
                )
                if normalized_raster is not None:
                    return PdfImageResource(normalized_raster, "image/png", pixel_count)
        except PdfImageError:
            raise
        except (
            Image.DecompressionBombError,
            # Still raised if the process runs with warnings turned into errors.
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
        ) as exc:
            raise PdfImageError("image content is invalid") from exc
        return PdfImageResource(content, _RASTER_MIME_TYPES[image_format], pixel_count)
    except BaseException:
        if budget is not None and reserved_pixels:
            budget.refund_pixels(reserved_pixels)
        raise


def _normalize_pdf_raster_mode(
    image: Image.Image,
    *,
    max_bytes: int,
) -> bytes | None:
    """Return an 8-bit PNG when a raster mode is unsafe for PDF rendering."""
    if image.mode in _PDF_RASTER_MODES:
        return None
    if image.mode.startswith("I;16"):
        normalized_image = image.point(lambda value: value / 257).convert("L")
    else:
        target_mode = "RGBA" if "A" in image.getbands() else "RGB"
        normalized_image = image.convert(target_mode)
    output = BytesIO()
    normalized_image.save(output, format="PNG")
    normalized = output.getvalue()
    if len(normalized) > max_bytes:
        raise PdfImageError("image exceeds the download size limit")
    return normalized


def _validate_svg_image(content: bytes) -> PdfImageResource:
    """Accept SVG only when it contains no external resource references."""
    _preflight_svg_structure(content)
    try:
        root = ElementTree.fromstring(content)
    except (ElementTree.ParseError, LookupError, ValueError) as exc:
        raise PdfImageError("SVG content is invalid") from exc
    if root.tag.rsplit("}", 1)[-1].lower() != "svg":
        raise PdfImageError("image content is not SVG")
    for element in root.iter():
        element_name = element.tag.rsplit("}", 1)[-1].lower()
        if element_name in {"lineargradient", "radialgradient"} and element.attrib.get(
            "spreadMethod",
            "",
        ).strip().lower() in {"repeat", "reflect"}:
            raise PdfImageError("SVG contains a repeating gradient")
        for name, value in element.attrib.items():
            normalized = _decode_css_escapes(value.strip()).lower()
            if name.lower().endswith("href") and normalized:
                if element_name == "a":
                    if not normalized.startswith(("#", "http://", "https://", "mailto:")):
                        raise PdfImageError("SVG contains an unsafe hyperlink")
                elif not normalized.startswith("#"):
                    raise PdfImageError("SVG contains an external resource")
            if _svg_css_has_external_resource(normalized):
                raise PdfImageError("SVG contains an external resource")
        if element_name == "style":
            stylesheet = _decode_css_escapes(element.text or "")
            if (
                "@import" in stylesheet.lower()
                or _svg_css_has_external_resource(stylesheet)
            ):
                raise PdfImageError("SVG contains an external resource")
            if _SVG_STYLESHEET_PRESENTATION_RE.search(stylesheet):
                raise PdfImageError(
                    "SVG stylesheet presentation references are not supported"
                )
    (
        use_graph,
        use_base_costs,
        referenced_ids,
        document_references,
        document_element_count,
    ) = _svg_use_graph(root)
    if _svg_graph_has_cycle(use_graph):
        raise PdfImageError("SVG contains a circular local reference")
    if _svg_graph_exceeds_expansion_limit(
        use_graph,
        use_base_costs,
        referenced_ids,
        document_references,
        document_element_count,
    ):
        raise PdfImageError("SVG local reference expansion limit exceeded")
    _validate_svg_inheritance(root)
    return PdfImageResource(content, "image/svg+xml")


def _has_multibyte_xml_signature(content: bytes) -> bool:
    """Return whether BOM-less UTF-16/32 starts with XML after whitespace."""
    signatures = (
        (b"<\x00", (b" \x00", b"\t\x00", b"\r\x00", b"\n\x00")),
        (b"\x00<", (b"\x00 ", b"\x00\t", b"\x00\r", b"\x00\n")),
        (
            b"<\x00\x00\x00",
            (b" \x00\x00\x00", b"\t\x00\x00\x00", b"\r\x00\x00\x00", b"\n\x00\x00\x00"),
        ),
        (
            b"\x00\x00\x00<",
            (b"\x00\x00\x00 ", b"\x00\x00\x00\t", b"\x00\x00\x00\r", b"\x00\x00\x00\n"),
        ),
    )
    for marker, whitespace_units in signatures:
        offset = 0
        unit_size = len(whitespace_units[0])
        while content[offset : offset + unit_size] in whitespace_units:
            offset += unit_size
        if content.startswith(marker, offset):
            return True
    return False


def _preflight_svg_structure(content: bytes) -> None:
    """Reject unsafe or excessive XML before building an in-memory tree."""
    element_count = 0
    attribute_count = 0
    attribute_value_chars = 0
    character_data_chars = 0
    text_chars = 0
    style_chars = 0
    style_rules = 0
    style_declarations = 0
    style_fragments: list[str] = []
    depth = 0
    element_stack: list[str] = []
    parser = expat.ParserCreate()

    def reject_declaration(*_args: object) -> None:
        raise PdfImageError("SVG declarations are not supported")

    def reject_external_entity(
        _context: str,
        _base: str | None,
        _system_id: str | None,
        _public_id: str | None,
    ) -> int:
        raise PdfImageError("SVG declarations are not supported")

    def start_element(_name: str, attributes: dict[str, str]) -> None:
        nonlocal element_count, attribute_count, attribute_value_chars, depth
        element_count += 1
        attribute_count += len(attributes)
        attribute_value_chars += sum(len(value) for value in attributes.values())
        depth += 1
        if (
            element_count > PDF_SVG_MAX_ELEMENTS
            or attribute_count > PDF_SVG_MAX_ATTRIBUTES
            or attribute_value_chars > PDF_SVG_MAX_ATTRIBUTE_VALUE_CHARS
            or depth > PDF_SVG_MAX_DEPTH
        ):
            raise PdfImageError("SVG structure limit exceeded")
        if _name.rsplit(":", 1)[-1].lower() == "path":
            path_data = next(
                (
                    value
                    for name, value in attributes.items()
                    if name.rsplit(":", 1)[-1].lower() == "d"
                ),
                "",
            )
            if len(path_data) > PDF_SVG_MAX_PATH_DATA_CHARS or sum(
                character in _SVG_PATH_COMMANDS for character in path_data
            ) > PDF_SVG_MAX_PATH_COMMANDS:
                raise PdfImageError("SVG path complexity limit exceeded")
        element_name = _name.rsplit(":", 1)[-1].lower()
        if element_name in {"polyline", "polygon"}:
            points = next(
                (
                    value
                    for name, value in attributes.items()
                    if name.rsplit(":", 1)[-1].lower() == "points"
                ),
                "",
            )
            if (
                sum(1 for _match in _SVG_NUMBER_RE.finditer(points))
                > PDF_SVG_MAX_POINT_COORDINATES
            ):
                raise PdfImageError("SVG point complexity limit exceeded")
        element_stack.append(element_name)

    def end_element(_name: str) -> None:
        nonlocal depth
        element_stack.pop()
        depth -= 1

    def character_data(data: str) -> None:
        nonlocal character_data_chars, style_chars, style_declarations
        nonlocal style_rules, text_chars
        size = len(data)
        character_data_chars += size
        current_element = element_stack[-1] if element_stack else ""
        if current_element in {"text", "textpath", "tspan"}:
            text_chars += size
        elif current_element == "style":
            style_chars += size
            style_rules += data.count("{")
            style_declarations += data.count(":")
            style_fragments.append(data)
        if (
            character_data_chars > PDF_SVG_MAX_CHARACTER_DATA_CHARS
            or text_chars > PDF_SVG_MAX_TEXT_CHARS
            or style_chars > PDF_SVG_MAX_STYLE_CHARS
            or style_rules > PDF_SVG_MAX_STYLE_RULES
            or style_declarations > PDF_SVG_MAX_STYLE_DECLARATIONS
        ):
            raise PdfImageError("SVG character data limit exceeded")

    parser.StartDoctypeDeclHandler = reject_declaration
    parser.EntityDeclHandler = reject_declaration
    parser.ExternalEntityRefHandler = reject_external_entity
    parser.StartElementHandler = start_element
    parser.EndElementHandler = end_element
    parser.CharacterDataHandler = character_data
    try:
        parser.Parse(content, True)
    except PdfImageError:
        raise
    except (expat.ExpatError, LookupError, ValueError) as exc:
        raise PdfImageError("SVG content is invalid") from exc
    style_selectors = style_rules + sum(
        fragment.count(",") for fragment in style_fragments
    )
    if (
        style_selectors > PDF_SVG_MAX_STYLE_SELECTORS
        or style_selectors * element_count > PDF_SVG_MAX_STYLE_MATCH_WORK
    ):
        raise PdfImageError("SVG stylesheet complexity limit exceeded")


def _svg_use_graph(
    root: ElementTree.Element,
) -> tuple[dict[str, list[str]], dict[str, int], set[str], list[str], int]:
    """Build a local-reference graph and direct element costs for SVG IDs."""
    elements_by_id: dict[str, ElementTree.Element] = {}
    for element in root.iter():
        raw_identifier = element.attrib.get("id", "").strip()
        if not raw_identifier:
            continue
        identifier = _normalize_svg_fragment(raw_identifier)
        if identifier in elements_by_id:
            raise PdfImageError("SVG contains ambiguous fragment identifiers")
        elements_by_id[identifier] = element
    graph: dict[str, list[str]] = {identifier: [] for identifier in elements_by_id}
    base_costs = dict.fromkeys(elements_by_id, 0)
    referenced_ids: set[str] = set()
    document_references: list[str] = []
    document_element_count = 0
    pending: list[tuple[ElementTree.Element, str | None, bool]] = [
        (root, None, True)
    ]
    while pending:
        element, owner, visible = pending.pop()
        document_element_count += 1
        element_name = element.tag.rsplit("}", 1)[-1].lower()
        identifier = _normalize_svg_fragment(element.attrib.get("id", "").strip())
        if identifier:
            if owner is not None:
                graph[owner].append(identifier)
            owner = identifier
        if owner is not None:
            base_costs[owner] += 1
        if element_name == "use":
            for name, value in element.attrib.items():
                normalized = value.strip()
                if name.lower().endswith("href") and normalized.startswith("#"):
                    target = _normalize_svg_fragment(normalized[1:])
                    if target in graph:
                        referenced_ids.add(target)
                        if owner is not None:
                            graph[owner].append(target)
                        if visible:
                            document_references.append(target)
        for target, repetitions in _svg_presentation_references(element):
            if target not in graph:
                continue
            referenced_ids.add(target)
            if owner is not None:
                graph[owner].extend([target] * repetitions)
            if visible:
                document_references.extend([target] * repetitions)
        children_visible = visible and element_name != "defs"
        pending.extend((child, owner, children_visible) for child in element)
    return (
        graph,
        base_costs,
        referenced_ids,
        document_references,
        document_element_count,
    )


def _svg_presentation_references(
    element: ElementTree.Element,
) -> list[tuple[str, int]]:
    """Return local mask, clipping, filter, and marker references with fan-out."""
    references: list[tuple[str, int]] = []
    presentation_attributes = {
        "mask",
        "clip-path",
        "filter",
        "marker",
        "marker-start",
        "marker-mid",
        "marker-end",
    }
    for name, value in element.attrib.items():
        property_name = name.rsplit("}", 1)[-1].lower()
        decoded_value = _decode_css_escapes(value)
        if property_name in presentation_attributes:
            references.extend(
                (
                    _normalize_svg_fragment(match.group(1)),
                    _svg_presentation_repetitions(element, property_name),
                )
                for match in _SVG_LOCAL_URL_RE.finditer(decoded_value)
            )
        elif property_name == "style":
            for style_match in _SVG_PRESENTATION_STYLE_RE.finditer(decoded_value):
                style_property = style_match.group(1).lower()
                references.extend(
                    (
                        _normalize_svg_fragment(url_match.group(1)),
                        _svg_presentation_repetitions(element, style_property),
                    )
                    for url_match in _SVG_LOCAL_URL_RE.finditer(
                        style_match.group(2)
                    )
                )
    return references


def _svg_presentation_repetitions(
    element: ElementTree.Element,
    property_name: str,
) -> int:
    """Estimate how many times one presentation resource is painted."""
    if property_name not in {"marker", "marker-mid"}:
        return 1
    attributes = {
        name.rsplit("}", 1)[-1].lower(): value
        for name, value in element.attrib.items()
    }
    if "points" in attributes:
        painted_positions = sum(
            1 for _match in _SVG_NUMBER_RE.finditer(attributes["points"])
        ) // 2
    elif "d" in attributes:
        painted_positions = sum(
            character in _SVG_PATH_COMMANDS for character in attributes["d"]
        )
    else:
        painted_positions = 1
    if property_name == "marker-mid":
        painted_positions = max(1, painted_positions - 2)
    return min(painted_positions, PDF_SVG_MAX_TOTAL_EXPANSION + 1)


def _svg_graph_has_cycle(graph: dict[str, list[str]]) -> bool:
    """Return whether a local SVG reference graph contains a cycle."""

    state: dict[str, int] = {}
    for start in graph:
        if state.get(start) == 2:
            continue
        state[start] = 1
        stack = [(start, iter(graph[start]))]
        while stack:
            identifier, targets = stack[-1]
            try:
                target = next(targets)
            except StopIteration:
                state[identifier] = 2
                stack.pop()
                continue
            target_state = state.get(target, 0)
            if target_state == 1:
                return True
            if target_state == 0:
                state[target] = 1
                stack.append((target, iter(graph[target])))
    return False


def _svg_graph_exceeds_expansion_limit(
    graph: dict[str, list[str]],
    base_costs: dict[str, int],
    referenced_ids: set[str],
    document_references: list[str],
    document_element_count: int,
) -> bool:
    """Return whether expanding any local SVG reference exceeds the limit."""
    expanded_costs: dict[str, int] = {}
    for start in referenced_ids:
        if start in expanded_costs:
            continue
        stack = [(start, False)]
        while stack:
            identifier, dependencies_visited = stack.pop()
            if identifier in expanded_costs:
                continue
            if not dependencies_visited:
                stack.append((identifier, True))
                stack.extend(
                    (target, False)
                    for target in graph[identifier]
                    if target not in expanded_costs
                )
                continue
            cost = base_costs[identifier]
            for target in graph[identifier]:
                cost += expanded_costs[target]
                if cost > PDF_SVG_MAX_USE_EXPANSION:
                    return True
            expanded_costs[identifier] = cost
    total_expansion = document_element_count
    for target in document_references:
        total_expansion += expanded_costs[target]
        if total_expansion > PDF_SVG_MAX_TOTAL_EXPANSION:
            return True
    return False


def _validate_svg_inheritance(root: ElementTree.Element) -> None:
    """Reject cyclic or renderer-unsafe local gradient and pattern chains."""
    graph: dict[str, str] = {}
    inheritable_elements = {"lineargradient", "radialgradient", "pattern"}
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1].lower() not in inheritable_elements:
            continue
        raw_identifier = element.attrib.get("id", "").strip()
        if not raw_identifier:
            continue
        identifier = _normalize_svg_fragment(raw_identifier)
        for name, value in element.attrib.items():
            reference = value.strip()
            if name.lower().endswith("href") and reference.startswith("#"):
                graph[identifier] = _normalize_svg_fragment(reference[1:])
                break

    depths: dict[str, int] = {}
    for start in graph:
        if start in depths:
            continue
        path: list[str] = []
        positions: dict[str, int] = {}
        current = start
        while current in graph and current not in depths:
            if current in positions:
                raise PdfImageError("SVG contains circular inheritance")
            positions[current] = len(path)
            path.append(current)
            current = graph[current]
        depth = depths.get(current, 0)
        for identifier in reversed(path):
            depth += 1
            if depth > PDF_SVG_MAX_INHERITANCE_DEPTH:
                raise PdfImageError("SVG inheritance depth limit exceeded")
            depths[identifier] = depth


def _decode_css_escapes(value: str) -> str:
    """Decode CSS escapes before inspecting resource-bearing values."""

    def replace(match: re.Match[str]) -> str:
        hexadecimal = match.group(1)
        if hexadecimal is not None:
            codepoint = int(hexadecimal, 16)
            if codepoint == 0 or codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
                return "\ufffd"
            return chr(codepoint)
        if match.group(2) is not None:
            return ""
        return match.group(3) or ""

    return _CSS_ESCAPE_RE.sub(replace, value)


def _normalize_svg_fragment(value: str) -> str:
    """Return the canonical URI encoding used for SVG fragment matching."""
    return quote(
        unquote(value),
        safe="!$&'()*+,;=:@/?-._~",
    )


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
