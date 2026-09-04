"""Exercise the single-owner HTTP trust boundary without external services."""

import pytest
from fastapi.testclient import TestClient

from chainagents.interfaces.api import app as api
from test_chainagents_api import _FakeAgent, _FakeRuntime


def make_client(monkeypatch, token=None, *, peer="127.0.0.1", host="127.0.0.1"):
    if token is None:
        monkeypatch.delenv("CHAINAGENTS_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("CHAINAGENTS_API_TOKEN", token)
    runtime = _FakeRuntime(_FakeAgent([]))
    runtime.project_root = "/private/tmp/chainagents-security-no-output"
    return TestClient(
        api.create_app(runtime=runtime), client=(peer, 50000), base_url=f"http://{host}"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/api/status",
        "/api/generated-files/example.txt",
        "/openapi.json",
        "/docs",
        "/redoc",
    ],
)
def test_sensitive_routes_require_bearer(monkeypatch, path):
    with make_client(monkeypatch, "test-token") as client:
        assert client.get(path).status_code == 401
        assert (
            client.get(path, headers={"Authorization": "Bearer wrong"}).status_code
            == 401
        )
        assert (
            client.get(path, headers={"Authorization": "Bearer test-token"}).status_code
            != 401
        )
        assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "evil.example"},
        {"Origin": "https://evil.example"},
        {"Origin": "null"},
        {"Origin": "http://localhost"},
        {"Origin": "http://127.0.0.1:9999"},
        {"Origin": "http://127.0.0.1/path"},
        {"Forwarded": "for=127.0.0.1"},
        {"X-Forwarded-For": "127.0.0.1"},
        {"X-Forwarded-Host": "localhost"},
        {"Sec-Fetch-Site": "cross-site"},
        [("Host", "127.0.0.1"), ("Host", "evil.example")],
        [("Origin", "http://127.0.0.1"), ("Origin", "https://evil.example")],
        {"Host": "127.0.0.1:bad"},
        {"Host": "127.0.0.1@evil.example"},
    ],
)
def test_tokenless_rejects_rebinding_and_browser_spoofing(monkeypatch, headers):
    with make_client(monkeypatch) as client:
        assert client.get("/api/status", headers=headers).status_code == 403
        assert (
            client.post(
                "/api/agent/invoke",
                headers=headers,
                json={"prompt": "hi", "thread_id": "t"},
            ).status_code
            == 403
        )


@pytest.mark.parametrize("peer", ["192.0.2.1", "testclient"])
def test_remote_peer_cannot_spoof_local_headers(monkeypatch, peer):
    with make_client(monkeypatch, peer=peer) as client:
        assert client.get("/api/status").status_code == 403


@pytest.mark.parametrize(
    "peer,host",
    [
        ("127.0.0.1", "127.0.0.1"),
        ("::1", "localhost"),
        ("::ffff:127.0.0.1", "localhost"),
    ],
)
def test_valid_local_nonbrowser_and_same_origin_browser(monkeypatch, peer, host):
    with make_client(monkeypatch, peer=peer, host=host) as client:
        assert client.get("/api/status").status_code == 200
        assert (
            client.get(
                "/api/status",
                headers={"Origin": f"http://{host}", "Sec-Fetch-Site": "same-origin"},
            ).status_code
            == 200
        )


@pytest.mark.parametrize(
    "auth",
    [
        "Bearer",
        "Basic test-token",
        "Bearer test-token extra",
        "Bearer  test-token",
        "Bearer test-token, Bearer test-token",
    ],
)
def test_malformed_authorization_fails_closed(monkeypatch, auth):
    with make_client(monkeypatch, "test-token") as client:
        assert (
            client.get("/api/status", headers={"Authorization": auth}).status_code
            == 401
        )


def test_duplicate_authorization_and_chainlit_credentials_do_not_authenticate(
    monkeypatch,
):
    monkeypatch.setenv("CHAINLIT_AUTH_SECRET", "test-token")
    with make_client(monkeypatch, "test-token") as client:
        assert (
            client.get(
                "/api/status",
                headers=[
                    ("Authorization", "Bearer test-token"),
                    ("Authorization", "Bearer test-token"),
                ],
            ).status_code
            == 401
        )
        client.cookies.set("access_token", "test-token")
        assert client.get("/api/status").status_code == 401


@pytest.mark.parametrize("token", ["", " has-spaces", "has spaces", "x\n", "秘密"])
def test_invalid_configured_token_refuses_startup(monkeypatch, token):
    monkeypatch.setenv("CHAINAGENTS_API_TOKEN", token)
    with pytest.raises(ValueError, match="CHAINAGENTS_API_TOKEN"):
        api.create_app(runtime=_FakeRuntime(_FakeAgent([])))


def test_remote_bind_requires_token_and_disables_proxy_trust(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
    monkeypatch.delenv("CHAINAGENTS_API_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        api.main(["--host", "0.0.0.0"])
    assert not calls
    monkeypatch.setenv("CHAINAGENTS_API_TOKEN", "test-token")
    api.main(["--host", "0.0.0.0"])
    assert calls[0]["proxy_headers"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "arbitrary-provider:secret-model"),
        ("async_subagent_url", "http://169.254.169.254/metadata"),
        ("mcp_session_id", "someone-elses-session"),
    ],
)
@pytest.mark.parametrize("endpoint", ["invoke", "stream", "stream/multipart"])
def test_request_cannot_override_server_context(monkeypatch, field, value, endpoint):
    with make_client(monkeypatch) as client:
        payload = {"prompt": "hi", "thread_id": "t", field: value}
        kwargs = (
            {"data": payload} if endpoint.endswith("multipart") else {"json": payload}
        )
        assert client.post(f"/api/agent/{endpoint}", **kwargs).status_code == 422


def test_context_uses_thread_scope_and_allows_only_configured_destination():
    from types import SimpleNamespace

    runtime = _FakeRuntime(_FakeAgent([]))
    runtime.config.extensions.async_subagents = (
        SimpleNamespace(url="https://configured.example"),
    )
    context = api._run_context(
        runtime,
        api.AgentRunRequest(
            prompt="hi",
            thread_id="t",
            model="other-model",
            mcp_session_id="t",
            async_subagent_url="https://configured.example",
        ),
    )
    assert context.model_name == "other-model"
    assert context.mcp_session_id == "t"
    assert (
        context.async_subagent_url is None
    )  # Configured destinations remain server-owned.
    assert (
        api._run_context(
            runtime, api.AgentRunRequest(prompt="hi", thread_id="t")
        ).mcp_session_id
        == "t"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "x" * 100_001, "thread_id": "t"},
        {"prompt": "hi", "thread_id": "x" * 257},
        {
            "prompt": "hi",
            "thread_id": "t",
            "history": [{"role": "user", "content": "x" * 100_001}],
        },
        {
            "prompt": "hi",
            "thread_id": "t",
            "history": [{"role": "user", "content": "x" * 100_000}] * 11,
        },
    ],
)
def test_bounded_text_and_history(monkeypatch, payload):
    with make_client(monkeypatch) as client:
        client.app.state.runtime.config.agent_state = "stateless"
        response = client.post("/api/agent/invoke", json=payload)
        assert response.status_code == 422
        assert len(response.content) < 2000
        assert "input" not in response.json()["detail"][0]


def test_multipart_validation_is_small_and_serializable(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.post(
            "/api/agent/stream/multipart",
            data={
                "prompt": "hi",
                "thread_id": "t",
                "history": '[{"role":"user","content":" "}]',
            },
        )
        assert response.status_code == 422
        assert "input" not in response.json()["detail"][0]
        assert "ctx" not in response.json()["detail"][0]


@pytest.mark.parametrize(
    "content_type,path",
    [
        ("application/json", "/api/agent/invoke"),
        ("multipart/form-data; boundary=X", "/api/agent/stream/multipart"),
    ],
)
def test_streamed_oversize_body_is_413_before_parser(monkeypatch, content_type, path):
    import asyncio

    # Direct ASGI requests exercise separate receive chunks with no Content-Length.
    app = api.create_app(runtime=_FakeRuntime(_FakeAgent([])))
    chunks = [
        b'{"prompt":"'
        if content_type == "application/json"
        else b'--X\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n',
        b"x" * (192 * 1024 * 1024),
    ]
    sent = []

    async def receive():
        return {
            "type": "http.request",
            "body": chunks.pop(0),
            "more_body": bool(chunks),
        }

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1"), (b"content-type", content_type.encode())],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
    }
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 413


def test_unexpected_backend_valueerror_does_not_leak(monkeypatch):
    from types import SimpleNamespace

    with make_client(monkeypatch) as client:
        runtime = client.app.state.runtime
        runtime.commands["lookup"] = SimpleNamespace(
            name="lookup",
            description="Lookup",
            target="mcp_tool",
            value="lookup",
            template=None,
            mcp_server="docs",
        )
        runtime.command_error = ValueError(
            "credential=supersecret from http://internal/db"
        )
        response = client.post(
            "/api/agent/invoke", json={"prompt": "/lookup {}", "thread_id": "t"}
        )
        assert response.status_code == 422
        assert "supersecret" not in response.text


def test_unhandled_preparation_error_is_safe(monkeypatch):
    with make_client(monkeypatch) as client:

        async def fail(*a, **kw):
            raise RuntimeError("password=private")

        monkeypatch.setattr(api, "_validate_history_replay", fail)
        # TestClient stores this setting on its transport.
        client._transport.raise_server_exceptions = False
        response = client.post(
            "/api/agent/invoke", json={"prompt": "hi", "thread_id": "t"}
        )
        assert response.status_code == 500
        assert response.json() == {"detail": "Agent operation failed. Please retry."}


@pytest.mark.parametrize(
    "host",
    [
        "[::1]evil",
        "[::1]:80evil",
        "localhost:",
        "localhost:65536",
        "localhost,evil.example",
    ],
)
def test_invalid_host_authority_fails_closed(monkeypatch, host):
    with make_client(monkeypatch) as client:
        assert client.get("/api/status", headers={"Host": host}).status_code == 403


@pytest.mark.parametrize(
    "headers,expected",
    [
        ([("Content-Length", "999999999999999999")], 413),
        ([("Content-Length", "-1")], 400),
        ([("Content-Length", "5"), ("Content-Length", "5")], 400),
        (
            [
                ("Content-Type", "application/json"),
                ("Content-Type", "multipart/form-data"),
            ],
            400,
        ),
    ],
)
def test_request_header_limits(monkeypatch, headers, expected):
    with make_client(monkeypatch) as client:
        assert (
            client.post("/api/agent/invoke", content=b"{}", headers=headers).status_code
            == expected
        )


def test_spa_bootstrap_public_but_sensitive_fallbacks_protected(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAINAGENTS_API_TOKEN", "test-token")
    (tmp_path / "index.html").write_text("<html>Public SPA shell</html>")
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "missing").write_text("sensitive fallback")
    with TestClient(
        api.create_app(runtime=_FakeRuntime(_FakeAgent([])), ui_dir=tmp_path),
        client=("192.0.2.1", 1),
        base_url="https://api.example",
    ) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/missing").status_code == 401
        assert (
            client.get(
                "/api/status", headers={"Authorization": "Bearer test-token"}
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/api/status",
                headers=[
                    ("Host", "api.example"),
                    ("Host", "evil.example"),
                    ("Authorization", "Bearer test-token"),
                ],
            ).status_code
            == 403
        )


def test_stream_body_limit_preserves_cancellation():
    import asyncio
    from chainagents.interfaces.api.security import RequestBodyLimitMiddleware

    async def downstream(scope, receive, send):
        await receive()

    async def receive():
        raise asyncio.CancelledError

    async def send(message):
        pytest.fail("Cancellation must not be converted to an HTTP response")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            RequestBodyLimitMiddleware(downstream)(
                {"type": "http", "headers": []}, receive, send
            )
        )


def test_ipv6_literal_host_and_origin_direct_asgi():
    import asyncio
    from chainagents.interfaces.api.security import (
        ApiSecurityConfig,
        ApiTrustMiddleware,
    )

    reached = []

    async def downstream(scope, receive, send):
        reached.append(True)

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        pytest.fail("Valid IPv6 local request must reach the app")

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/status",
        "scheme": "http",
        "client": ("::1", 1),
        "headers": [(b"host", b"[::1]:8000"), (b"origin", b"http://[::1]:8000")],
    }
    asyncio.run(
        ApiTrustMiddleware(downstream, config=ApiSecurityConfig())(scope, receive, send)
    )
    assert reached == [True]


@pytest.mark.parametrize(
    "authorization", ["Bearer", "Basic unrelated", "Bearer  two-spaces"]
)
def test_tokenless_rejects_malformed_authorization(monkeypatch, authorization):
    with make_client(monkeypatch) as client:
        assert (
            client.get(
                "/api/status", headers={"Authorization": authorization}
            ).status_code
            == 403
        )


def test_reflection_fallback_does_not_echo_backend_exception(monkeypatch):
    from chainagents.events.stream import AgentStreamEvent
    from chainagents.runtime.reflection import ReflectionConfig

    async def failed_events(*args, **kwargs):
        collector = kwargs.get("reflection_collector")
        event = AgentStreamEvent(
            kind="tool_result",
            source="main-agent",
            tool_name="read_file",
            tool_result="",
            status="error",
        )
        if collector:
            collector.record_event(event)
        yield event
        raise RuntimeError("credential=supersecret")

    monkeypatch.setattr(api, "_iter_agent_events", failed_events)
    with make_client(monkeypatch) as client:
        client.app.state.runtime.config.extensions.agent_reflection = ReflectionConfig(
            enabled=True
        )
        response = client.post(
            "/api/agent/stream", json={"prompt": "try again", "thread_id": "t"}
        )
        assert "reflection_proposal" in response.text
        assert "supersecret" not in response.text


def test_rag_status_does_not_echo_backend_exception():
    from chainagents.rag.runtime import RagUploadResult

    payload = api._attachment_status_payload(
        RagUploadResult(thread_id="t", reason="credential=supersecret")
    )
    assert payload["status"] == "error"
    assert "supersecret" not in str(payload)


def test_oversize_multipart_closes_parser_files(monkeypatch):
    import asyncio
    import tempfile
    import starlette.formparsers
    from chainagents.interfaces.api import security

    monkeypatch.setattr(security, "MAX_MULTIPART_BODY_BYTES", 512)
    opened = []

    def recording_file(*args, **kwargs):
        file = tempfile.SpooledTemporaryFile(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", recording_file)
    app = api.create_app(runtime=_FakeRuntime(_FakeAgent([])))
    chunks = [
        b'--X\r\nContent-Disposition: form-data; name="files"; filename="test.txt"\r\nContent-Type: text/plain\r\n\r\nfirst',
        b"x" * 513,
    ]
    sent = []

    async def receive():
        return {
            "type": "http.request",
            "body": chunks.pop(0),
            "more_body": bool(chunks),
        }

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/agent/stream/multipart",
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1"),
            (b"content-type", b"multipart/form-data; boundary=X"),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
    }
    asyncio.run(app(scope, receive, send))
    assert sent[0]["status"] == 413
    assert opened and all(file.closed for file in opened)


def test_multipart_filename_is_bounded(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.post(
            "/api/agent/stream/multipart",
            data={"prompt": "hi", "thread_id": "t"},
            files={"files": ("x" * 257 + ".txt", b"text", "text/plain")},
        )
        assert response.status_code == 422
        assert len(response.content) < 500


def test_validation_preserves_actionable_local_message(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.post(
            "/api/agent/invoke", json={"prompt": "hi", "thread_id": "x" * 257}
        )
        error = response.json()["detail"][0]
        assert error["loc"] == ["body", "thread_id"]
        assert "256" in error["msg"]
        assert "input" not in error


@pytest.mark.parametrize("deployment", ["root_path", "mount"])
@pytest.mark.parametrize("token", [None, "test-token"])
@pytest.mark.parametrize(
    "path",
    [
        "/api/status",
        "/openapi.json",
        "/docs",
        "/redoc",
        "/api/generated-files/report.txt",
    ],
)
def test_prefixed_sensitive_routes_require_authorization(
    tmp_path, deployment, token, path
):
    from fastapi import FastAPI
    from chainagents.interfaces.api.security import ApiSecurityConfig

    ui = tmp_path / "ui"
    ui.mkdir()
    (ui / "index.html").write_text("public shell")
    output = tmp_path / ".files" / "outputs" / "report.txt"
    output.parent.mkdir(parents=True)
    output.write_text("private report")
    runtime = _FakeRuntime(_FakeAgent([]))
    runtime.project_root = tmp_path
    child = api.create_app(
        runtime=runtime, ui_dir=ui, security=ApiSecurityConfig(token=token)
    )
    # Mounted apps do not receive their parent's lifespan events. Supply the
    # injected fake runtime directly without constructing any runtime resources.
    child.state.runtime = runtime
    if deployment == "mount":
        app = FastAPI()
        app.mount("/prefix", child)
        root_path = ""
    else:
        app = child
        root_path = "/prefix"

    with TestClient(
        app,
        root_path=root_path,
        client=("192.0.2.1", 1),
        base_url="http://evil.example",
    ) as client:
        assert client.get(f"/prefix{path}").status_code == (401 if token else 403)
        assert client.get("/prefix/").text == "public shell"
        assert client.get("/prefix/health").json() == {"status": "ok"}
        if token:
            response = client.get(
                f"/prefix{path}", headers={"Authorization": "Bearer test-token"}
            )
            assert response.status_code == 200
            if path.endswith("report.txt"):
                assert response.text == "private report"

    if token is None:
        with TestClient(
            app,
            root_path=root_path,
            client=("127.0.0.1", 1),
            base_url="http://127.0.0.1",
        ) as client:
            response = client.get(
                f"/prefix{path}", headers={"Origin": "http://127.0.0.1"}
            )
            assert response.status_code == 200
            if path.endswith("report.txt"):
                assert response.text == "private report"


@pytest.mark.parametrize("token", [None, "test-token"])
def test_root_path_partial_prefix_keeps_sensitive_route_protected(tmp_path, token):
    from chainagents.interfaces.api.security import ApiSecurityConfig

    (tmp_path / "index.html").write_text("public shell")
    app = api.create_app(
        runtime=_FakeRuntime(_FakeAgent([])),
        ui_dir=tmp_path,
        security=ApiSecurityConfig(token=token),
    )
    # Starlette does not strip '/ap' from '/api/status': there is no slash boundary.
    with TestClient(
        app, root_path="/ap", client=("192.0.2.1", 1), base_url="http://evil.example"
    ) as client:
        assert client.get("/api/status").status_code == (401 if token else 403)
