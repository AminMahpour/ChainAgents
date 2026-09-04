"""Single trusted owner HTTP policy; no per-user identity or proxy trust."""

from __future__ import annotations

import ipaddress
import os
import re
import secrets
from dataclasses import dataclass
from urllib.parse import urlsplit

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_TOKEN = re.compile(r"[A-Za-z0-9._~+/\-]+=*\Z")


@dataclass(frozen=True)
class ApiSecurityConfig:
    """Optional bearer credential; an explicitly configured empty value is invalid."""

    token: str | None = None

    def __post_init__(self) -> None:
        if self.token is not None and (
            not 1 <= len(self.token) <= 4096 or not _TOKEN.fullmatch(self.token)
        ):
            raise ValueError(
                "CHAINAGENTS_API_TOKEN must be a nonempty ASCII bearer token."
            )

    @classmethod
    def from_env(cls) -> ApiSecurityConfig:
        return cls(token=os.environ.get("CHAINAGENTS_API_TOKEN"))


def is_loopback(value: str) -> bool:
    """Accept numeric loopback addresses, including IPv4-mapped IPv6."""
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_loopback


def _authority(value: str, scheme: str) -> tuple[str, int] | None:
    """Parse strict HTTP authority without credentials, paths, or whitespace."""
    if not value or any(c.isspace() or ord(c) < 33 or ord(c) > 126 for c in value):
        return None
    if any(c in value for c in "/\\?#@%") or value.endswith(":"):
        return None
    try:
        parsed = urlsplit(f"{scheme}://{value}")
        host = parsed.hostname
        port = parsed.port
        if not host or parsed.username or parsed.password:
            return None
        # urlsplit accepts unbracketed IPv6 ambiguously; require bracketed literals.
        if value.count(":") > 1 and not value.startswith("["):
            return None
        return host, port if port is not None else (443 if scheme == "https" else 80)
    except ValueError:
        return None


class ApiTrustMiddleware:
    """Authorize before routing. Raw headers and peer are authoritative."""

    def __init__(
        self, app: ASGIApp, *, config: ApiSecurityConfig, public_static: bool = False
    ):
        self.app = app
        self.config = config
        self.public_static = public_static

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if scope["method"] in {"GET", "HEAD"} and (
            path == "/health"
            or (
                self.public_static
                and not path.startswith(("/api", "/docs", "/redoc", "/openapi.json"))
            )
        ):
            await self.app(scope, receive, send)
            return
        headers: dict[bytes, list[str]] = {}
        for key, value in scope.get("headers", []):
            headers.setdefault(key.lower(), []).append(value.decode("latin-1"))
        scheme = scope.get("scheme", "http")
        hosts = headers.get(b"host", [])
        authority = _authority(hosts[0], scheme) if len(hosts) == 1 else None
        authorization = headers.get(b"authorization", [])
        token = self.config.token
        if token is not None:
            valid = False
            if len(authorization) == 1:
                kind, separator, supplied = authorization[0].partition(" ")
                if (
                    kind.lower() == "bearer"
                    and separator
                    and _TOKEN.fullmatch(supplied)
                ):
                    valid = secrets.compare_digest(
                        supplied.encode("ascii"), token.encode("ascii")
                    )
            if not valid:
                await JSONResponse(
                    {"detail": "Bearer authentication required."},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )(scope, receive, send)
                return
            allowed = authority is not None
        else:
            peer = scope.get("client")
            allowed = bool(peer and is_loopback(peer[0]) and authority is not None)
            if authority is not None:
                allowed = allowed and (
                    authority[0] == "localhost" or is_loopback(authority[0])
                )
            allowed = allowed and len(authorization) <= 1
            if authorization:
                kind, separator, supplied = authorization[0].partition(" ")
                allowed = (
                    allowed
                    and kind.lower() == "bearer"
                    and bool(separator)
                    and bool(_TOKEN.fullmatch(supplied))
                )
            allowed = allowed and not any(
                k == b"forwarded" or k.startswith(b"x-forwarded-") for k in headers
            )
            sites = headers.get(b"sec-fetch-site", [])
            allowed = allowed and (
                not sites or (len(sites) == 1 and sites[0] in {"same-origin", "none"})
            )
            origins = headers.get(b"origin", [])
            if origins:
                try:
                    origin = urlsplit(origins[0])
                    origin_authority = _authority(origin.netloc, origin.scheme)
                    allowed = (
                        allowed
                        and len(origins) == 1
                        and origin.scheme == scheme
                        and origin_authority == authority
                    )
                    allowed = (
                        allowed
                        and bool(origin_authority)
                        and not (origin.path or origin.query or origin.fragment)
                    )
                except ValueError:
                    allowed = False
        if not allowed:
            await JSONResponse(
                {"detail": "Request is outside the API trust boundary."},
                status_code=403,
            )(scope, receive, send)
            return
        await self.app(scope, receive, send)


# Allow five 25 MiB images with base64 overhead, or five raw multipart uploads.
MAX_JSON_BODY_BYTES = 180 * 1024 * 1024
MAX_MULTIPART_BODY_BYTES = 130 * 1024 * 1024


class RequestBodyTooLarge(HTTPException, OSError):
    """HTTP 413 transport failure that triggers multipart parser file cleanup.

    Starlette closes in-progress multipart files for OSError; FastAPI preserves
    HTTPException status codes while converting other parser failures to 400.
    """

    def __init__(self) -> None:
        HTTPException.__init__(
            self, status_code=413, detail="Request body exceeds the size limit."
        )


class RequestBodyLimitMiddleware:
    """Count receive bytes before the JSON/multipart parser can buffer them."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers", [])
        lengths = [v for k, v in headers if k.lower() == b"content-length"]
        types = [v for k, v in headers if k.lower() == b"content-type"]
        if (
            len(lengths) > 1
            or len(types) > 1
            or (lengths and not re.fullmatch(rb"[0-9]{1,20}", lengths[0]))
        ):
            await JSONResponse({"detail": "Invalid request headers."}, status_code=400)(
                scope, receive, send
            )
            return
        multipart = bool(
            types
            and types[0].split(b";", 1)[0].strip().lower() == b"multipart/form-data"
        )
        limit = MAX_MULTIPART_BODY_BYTES if multipart else MAX_JSON_BODY_BYTES
        if lengths and int(lengths[0]) > limit:
            await JSONResponse(
                {"detail": "Request body exceeds the size limit."}, status_code=413
            )(scope, receive, send)
            return
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    # FastAPI preserves HTTPException from its body parser; a plain
                    # Exception would become 400 and hide the intended 413 contract.
                    raise RequestBodyTooLarge()
            return message

        await self.app(scope, bounded_receive, send)
