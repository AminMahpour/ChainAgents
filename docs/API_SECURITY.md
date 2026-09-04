# API access and request limits

The FastAPI service is for **one trusted owner**. Access permits running the
configured agent/tools and reading that instance's conversations, reflection
proposals, generated outputs, and runtime configuration. A thread ID selects
conversation state; it is not a user identity. Use separate instances with
separate storage, credentials, and filesystem access for mutually untrusted
users. The bearer token does not create separate-user isolation.

## Local default

With `CHAINAGENTS_API_TOKEN` unset, run:

```sh
uv run chainagents-api --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/api/status
```

Sensitive requests require a numeric loopback connection peer and a Host naming
`localhost` or a loopback IP. IPv6 `::1` and IPv4-mapped IPv6 loopback peers are
supported. Browser Origin, when present, must exactly match the request's scheme,
host, and effective port; opaque, foreign, and malformed Origins are rejected.
Use the same origin for the UI and API (including the port). Non-browser clients
may omit Origin. Cross-site Fetch Metadata, duplicate or malformed Host and
Authorization headers, and all `Forwarded` / `X-Forwarded-*` headers are rejected
in local mode. A syntactically valid bearer header does not bypass these checks.
Local processes are trusted; this is not a sandbox against other local users.

## Bearer token and remote access

Generate a strong random token and keep it in your private environment or secret
manager. For example:

```sh
export CHAINAGENTS_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run chainagents-api --host 0.0.0.0 --port 8000
```

Unset the variable to select local mode. An explicitly empty, whitespace-bearing,
non-ASCII, or otherwise malformed bearer token makes application creation fail.
Tokens must use the bearer alphabet (`A-Z`, `a-z`, digits, `-._~+/`, and trailing
`=` padding) and contain 1–4096 characters; the generated example is recommended.

Every sensitive request must carry `Authorization: Bearer <token>`. Comparison
uses a constant-time secret comparison. Chainlit credentials and cookies do not
authenticate the API. Tokens are never accepted in query parameters or download
URLs. For example, when deployed behind HTTPS:

```sh
curl https://agents.example/api/status \
  -H "Authorization: Bearer $CHAINAGENTS_API_TOKEN"
curl https://agents.example/api/generated-files/report.pdf \
  -H "Authorization: Bearer $CHAINAGENTS_API_TOKEN" -o report.pdf
```

The CLI refuses non-loopback binds without a token and sets Uvicorn
`proxy_headers=False`. Direct ASGI deployments use the same request middleware;
configure the ASGI server with proxy trust disabled (for Uvicorn,
`--no-proxy-headers`). A runner that rewrites connection peers before the app can
invalidate tokenless peer checks. Remote/proxied deployments must configure the
token even if a proxy connects over loopback. Use HTTPS at the remote endpoint;
the CLI's listener itself does not configure TLS. Preserve a single valid Host
header. No permissive CORS policy is installed.

`GET /health` stays public and returns only `{"status":"ok"}`. Agent invocation,
streaming and multipart requests, runtime status, reflection saves, PDF exports,
generated-file downloads, `/openapi.json`, `/docs`, and `/redoc` are protected.
Unknown `/api...` paths cannot bypass authentication via a static-file fallback.

## Browser clients and static UI

`--ui-dir` may serve a **built, public UI directory**. Its shell and assets can be
fetched before authentication; do not point it at a project, configuration, or
private output directory. API status/bootstrap calls still require authorization.

A token-protected deployment needs a client capable of adding Authorization to
all API requests, including status, streams, multipart uploads, exports, and
download fetches. For example:

```js
const response = await fetch('/api/status', {
  headers: { Authorization: `Bearer ${token}` },
});
```

Ordinary download links, browser navigation to the documentation, and a stock
SPA without token-header support cannot attach that header. Use an authenticating
reverse proxy that authorizes this single owner and injects the backend bearer
header, or adapt the client to authenticated fetches (and blobs for downloads).
Do not embed the server token in public JavaScript or URLs. This change adds no
browser sign-in UI. The default local same-origin browser flow needs no token.

## Server-owned request context

- `model` must match an entry in the server's configured `model_choices`.
- Async subagent destinations come from the server configuration. A supplied
  `async_subagent_url` must exactly match a configured destination; the API then
  discards this redundant override and lets each configured subagent use its own
  URL. A request cannot redirect a subagent or supply a URL to an unconfigured one.
- MCP scope is derived from the resolved `thread_id`. An omitted `mcp_session_id`
  is normal; a supplied value must equal that thread ID. This prevents independent
  scope selection, while retaining the single-owner conversation model.

## Resource and error boundaries

Constants in `chainagents/interfaces/api/security.py` bound actual streamed bytes
before JSON/base64 or multipart parsing can buffer beyond the limit, including
requests without Content-Length or with a false smaller length:

| Limit | Value |
| --- | --- |
| JSON/other request body | 180 MiB |
| Multipart request body | 130 MiB |
| Each uploaded file / decoded history image | 25 MiB |
| Uploaded files / total history images | 5 |
| Prompt, each history string or text part | 100,000 characters |
| Total history text | 1,000,000 characters |
| History messages / parts per message | 200 / 100 |
| Thread/model/command/scope identifiers and uploaded filenames | 256 characters |
| Async destination field | 2,048 characters |
| Reflection lesson/trigger/tool-result fields | 100,000 characters |

History image data URLs have an encoded-length bound before base64 decoding.
The body limits leave room for five existing 25 MiB uploads and base64 overhead.
The multipart parser also applies its own per-field limit (currently 1 MiB), so
large replay histories should use JSON. PDF export retains its 100,000-character,
2,000-line, and single-concurrent-render limits. Reflection configuration can
impose a smaller lesson limit.

Oversized bodies return 413; malformed/duplicate length or content-type headers
return 400. Streaming enforcement preserves HTTP 413 through FastAPI parsing and
closes multipart spool files already opened when the limit is crossed. The
middleware does not buffer or spawn a task and propagates cancellation.

Validation responses retain field locations, error types, and bounded local
correction messages, omitting submitted input and non-serializable exception
context; at most 20 errors are returned. Unexpected backend exceptions are logged
server-side and replaced by safe messages in HTTP/NDJSON responses. Reflection
fallbacks and RAG status errors also avoid relaying raw backend exception text.
This does not redact intentional model responses or tool-result content, which
are part of the trusted owner's agent output. These are per-request bounds, not
rate limits or a cap on aggregate concurrent requests or model/tool execution.
