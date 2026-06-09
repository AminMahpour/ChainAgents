#!/usr/bin/env python3
"""Provide FastAPI access to the ChainAgents runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_stream_events import AgentStreamEvent, AgentStreamEventAdapter
from deepagent_runtime import (
    AgentRuntime,
    ReasoningLevel,
    build_langgraph_run_config,
    normalize_reasoning_level,
)


AGENT_STREAM_MODES = ["messages", "updates", "custom"]
NDJSON_MEDIA_TYPE = "application/x-ndjson"


class AgentRunRequest(BaseModel):
    """HTTP request body for running the agent."""

    prompt: str = Field(..., min_length=1)
    thread_id: str = Field(..., min_length=1)
    model: str | None = None
    reasoning: ReasoningLevel | None = None
    async_subagent_url: str | None = None
    mcp_session_id: str | None = None


class AgentRunResponse(BaseModel):
    """HTTP response body for a completed agent run."""

    response: str
    thread_id: str
    model: str
    reasoning: ReasoningLevel


class RuntimeStatusResponse(BaseModel):
    """Resolved runtime configuration exposed by the API."""

    model: str
    model_provider: str
    model_choices: list[str]
    default_reasoning: ReasoningLevel
    agent_state: str
    recursion_limit: int
    persistence_mode: str


@dataclass(frozen=True)
class AgentRunContext:
    """Resolved request values used for one agent run."""

    prompt: str
    thread_id: str
    model_name: str
    reasoning_level: ReasoningLevel
    async_subagent_url: str | None
    mcp_session_id: str | None


def create_app(runtime: Any | None = None) -> FastAPI:
    """Create the FastAPI app.

    Args:
        runtime: Optional runtime test double or initialized AgentRuntime.

    Returns:
        The configured FastAPI app.
    """
    managed_runtime: AgentRuntime | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal managed_runtime
        if runtime is None:
            managed_runtime = await AgentRuntime.create()
            app.state.runtime = managed_runtime
        else:
            app.state.runtime = runtime

        try:
            yield
        finally:
            if managed_runtime is not None:
                await managed_runtime.close()

    app = FastAPI(
        title="ChainAgents API",
        version="1.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status", response_model=RuntimeStatusResponse)
    async def status(request: Request) -> RuntimeStatusResponse:
        active_runtime = _runtime_from_request(request)
        config = active_runtime.config
        return RuntimeStatusResponse(
            model=config.model_name,
            model_provider=config.model_provider,
            model_choices=list(config.model_choices),
            default_reasoning=config.default_reasoning,
            agent_state=config.agent_state,
            recursion_limit=config.recursion_limit,
            persistence_mode=config.persistence_mode,
        )

    @app.post("/api/agent/invoke", response_model=AgentRunResponse)
    async def invoke_agent(
        payload: AgentRunRequest,
        request: Request,
    ) -> AgentRunResponse:
        active_runtime = _runtime_from_request(request)
        context = _run_context(active_runtime, payload)
        response_parts: list[str] = []
        try:
            async for event in _iter_agent_events(active_runtime, context):
                if event.kind == "response_delta":
                    response_parts.append(event.text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise _agent_error(exc) from exc

        return AgentRunResponse(
            response="".join(response_parts),
            thread_id=context.thread_id,
            model=context.model_name,
            reasoning=context.reasoning_level,
        )

    @app.post("/api/agent/stream")
    async def stream_agent(
        payload: AgentRunRequest,
        request: Request,
    ) -> StreamingResponse:
        active_runtime = _runtime_from_request(request)
        context = _run_context(active_runtime, payload)

        async def lines() -> AsyncIterator[str]:
            try:
                async for event in _iter_agent_events(active_runtime, context):
                    yield _json_line(_event_payload(event, context))
                yield _json_line(
                    {
                        "kind": "done",
                        "thread_id": context.thread_id,
                        "model": context.model_name,
                        "reasoning": context.reasoning_level,
                    }
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield _json_line(
                    {
                        "kind": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "thread_id": context.thread_id,
                        "model": context.model_name,
                        "reasoning": context.reasoning_level,
                    }
                )

        return StreamingResponse(lines(), media_type=NDJSON_MEDIA_TYPE)

    return app


def _runtime_from_request(request: Request) -> Any:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Agent runtime is not initialized.")
    return runtime


def _run_context(runtime: Any, request: AgentRunRequest) -> AgentRunContext:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt must not be blank.")

    thread_id = _required_text(request.thread_id, "thread_id")
    model_name = _optional_text(request.model) or runtime.config.model_name
    try:
        reasoning_level = normalize_reasoning_level(
            request.reasoning,
            default=runtime.config.default_reasoning,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AgentRunContext(
        prompt=prompt,
        thread_id=thread_id,
        model_name=model_name,
        reasoning_level=reasoning_level,
        async_subagent_url=_optional_text(request.async_subagent_url),
        mcp_session_id=_optional_text(request.mcp_session_id),
    )


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _required_text(value: str, field_name: str) -> str:
    text = value.strip()
    if not text:
        raise HTTPException(status_code=422, detail=f"{field_name} must not be blank.")
    return text


async def _iter_agent_events(
    runtime: Any,
    context: AgentRunContext,
) -> AsyncIterator[AgentStreamEvent]:
    agent = await runtime.get_agent(
        context.reasoning_level,
        model_name=context.model_name,
        thread_id=context.thread_id,
        async_subagent_url_override=context.async_subagent_url,
        mcp_session_id=context.mcp_session_id,
    )
    payload = {"messages": [{"role": "user", "content": context.prompt}]}
    config = build_langgraph_run_config(runtime.config, thread_id=context.thread_id)
    adapter = AgentStreamEventAdapter(prompt=context.prompt)
    stream = agent.astream_events(
        payload,
        config=config,
        version="v2",
        stream_mode=AGENT_STREAM_MODES,
        subgraphs=True,
    )

    try:
        async for raw_event in stream:
            for event in adapter.events_from_raw_event(raw_event):
                yield event
    finally:
        with suppress(Exception):
            await stream.aclose()


def _event_payload(event: AgentStreamEvent, context: AgentRunContext) -> dict[str, Any]:
    payload = asdict(event)
    payload.update(
        {
            "thread_id": context.thread_id,
            "model": context.model_name,
            "reasoning": context.reasoning_level,
        }
    )
    return payload


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"


def _agent_error(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=500,
        detail=f"{type(exc).__name__}: {exc}",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the ChainAgents API command-line parser."""
    parser = argparse.ArgumentParser(
        prog="chainagents-api",
        description="Run the ChainAgents FastAPI server.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn reload for local development.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the API server from the console script."""
    args = build_parser().parse_args(argv)

    import uvicorn

    uvicorn.run(
        "chainagents_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


app = create_app()


if __name__ == "__main__":
    main()
