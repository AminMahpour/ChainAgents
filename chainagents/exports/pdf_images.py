"""Bounded, public-only image downloads for response PDF exports."""

from __future__ import annotations

import base64
import binascii
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
from urllib.parse import quote, unquote_to_bytes, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree
from xml.parsers import expat

from PIL import Image, UnidentifiedImageError


PDF_IMAGE_MAX_BYTES = 10 * 1024 * 1024
PDF_IMAGE_MAX_PIXELS = 25_000_000
PDF_SVG_MAX_ELEMENTS = 10_000
PDF_SVG_MAX_DEPTH = 4_096
PDF_SVG_MAX_ATTRIBUTES = 100_000
PDF_SVG_MAX_ATTRIBUTE_VALUE_CHARS = 1_000_000
PDF_SVG_MAX_PATH_DATA_CHARS = 500_000
PDF_SVG_MAX_PATH_COMMANDS = 25_000
PDF_SVG_MAX_USE_EXPANSION = 4_096
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
_SVG_PATH_COMMANDS = frozenset("MmZzLlHhVvCcSsQqTtAa")


class PdfImageError(ValueError):
    """An image cannot be safely included in a response PDF."""


class _PdfImageConnectionError(PdfImageError):
    """A validated address could not complete an HTTP request."""


@dataclass(frozen=True)
class PdfImageResource:
    """Validated image bytes ready for WeasyPrint."""

    content: bytes
    mime_type: str
    pixel_count: int = 0


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


def decode_pdf_data_image(
    url: str,
    *,
    max_bytes: int = PDF_IMAGE_MAX_BYTES,
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
    return _validate_pdf_image(content, max_bytes=max_bytes)


def _validate_pdf_image(
    content: bytes,
    *,
    max_bytes: int = PDF_IMAGE_MAX_BYTES,
) -> PdfImageResource:
    """Validate supported image bytes and normalize animated GIFs."""
    if not content:
        raise PdfImageError("image response was empty")
    stripped = content.removeprefix(b"\xef\xbb\xbf").lstrip()
    if stripped.startswith(b"<") or _has_multibyte_xml_signature(content):
        return _validate_svg_image(content)
    if content.startswith(
        (b"\xff\xfe", b"\xfe\xff", b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00")
    ):
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
                pixel_count = image.width * image.height
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
    return PdfImageResource(content, _RASTER_MIME_TYPES[image_format], pixel_count)


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
        for name, value in element.attrib.items():
            normalized = value.strip().lower()
            if name.lower().endswith("href") and normalized:
                if element_name == "a":
                    if not normalized.startswith(("#", "http://", "https://", "mailto:")):
                        raise PdfImageError("SVG contains an unsafe hyperlink")
                elif not normalized.startswith("#"):
                    raise PdfImageError("SVG contains an external resource")
            if _svg_css_has_external_resource(normalized):
                raise PdfImageError("SVG contains an external resource")
        if element.tag.rsplit("}", 1)[-1].lower() == "style" and (
            "@import" in (element.text or "").lower()
            or _svg_css_has_external_resource(element.text or "")
        ):
            raise PdfImageError("SVG contains an external resource")
    use_graph, use_base_costs, referenced_ids = _svg_use_graph(root)
    if _svg_graph_has_cycle(use_graph):
        raise PdfImageError("SVG contains a circular local reference")
    if _svg_graph_exceeds_expansion_limit(
        use_graph,
        use_base_costs,
        referenced_ids,
    ):
        raise PdfImageError("SVG local reference expansion limit exceeded")
    return PdfImageResource(content, "image/svg+xml")


def _has_multibyte_xml_signature(content: bytes) -> bool:
    """Return whether bytes begin with a BOM-less UTF-16/32 XML signature."""
    return content.startswith(
        (
            b"<\x00",
            b"\x00<",
            b"<\x00\x00\x00",
            b"\x00\x00\x00<",
        )
    )


def _preflight_svg_structure(content: bytes) -> None:
    """Reject unsafe or excessive XML before building an in-memory tree."""
    element_count = 0
    attribute_count = 0
    attribute_value_chars = 0
    depth = 0
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

    def end_element(_name: str) -> None:
        nonlocal depth
        depth -= 1

    parser.StartDoctypeDeclHandler = reject_declaration
    parser.EntityDeclHandler = reject_declaration
    parser.ExternalEntityRefHandler = reject_external_entity
    parser.StartElementHandler = start_element
    parser.EndElementHandler = end_element
    try:
        parser.Parse(content, True)
    except PdfImageError:
        raise
    except (expat.ExpatError, LookupError, ValueError) as exc:
        raise PdfImageError("SVG content is invalid") from exc


def _svg_use_graph(
    root: ElementTree.Element,
) -> tuple[dict[str, list[str]], dict[str, int], set[str]]:
    """Build a local-reference graph and direct element costs for SVG IDs."""
    elements_by_id = {
        identifier: element
        for element in root.iter()
        if (identifier := element.attrib.get("id", "").strip())
    }
    graph: dict[str, list[str]] = {identifier: [] for identifier in elements_by_id}
    base_costs = dict.fromkeys(elements_by_id, 0)
    referenced_ids: set[str] = set()
    pending: list[tuple[ElementTree.Element, str | None]] = [(root, None)]
    while pending:
        element, owner = pending.pop()
        identifier = element.attrib.get("id", "").strip()
        if identifier:
            if owner is not None:
                graph[owner].append(identifier)
            owner = identifier
        if owner is not None:
            base_costs[owner] += 1
        if element.tag.rsplit("}", 1)[-1].lower() == "use":
            for name, value in element.attrib.items():
                normalized = value.strip()
                if name.lower().endswith("href") and normalized.startswith("#"):
                    target = normalized[1:]
                    if target in graph:
                        referenced_ids.add(target)
                        if owner is not None:
                            graph[owner].append(target)
        pending.extend((child, owner) for child in element)
    return graph, base_costs, referenced_ids


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
    return False


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
