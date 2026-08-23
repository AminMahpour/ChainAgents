#!/usr/bin/env python3
"""Provide FastAPI access to the ChainAgents runtime."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import mimetypes
import os
import secrets
import tempfile
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from chainagents.commands.native import (
    dumps_tool_result,
    resolve_native_command,
    resolve_runtime_command,
)
from chainagents.events.stream import AgentStreamEvent, AgentStreamEventAdapter
from chainagents.exports.generated_files import (
    generated_file_descriptors,
    generated_file_paths_from_text,
    generated_file_paths_from_tool_args,
    resolve_generated_download,
)
from chainagents.exports.response import build_pdf_bytes
from chainagents.interfaces.uploads import (
    MAX_UPLOAD_FILE_BYTES,
    MAX_UPLOAD_FILES,
    NormalizedUpload,
    SUPPORTED_IMAGE_MIME_TYPES,
    SUPPORTED_RAG_EXTENSIONS,
    image_content_part,
    normalize_upload,
    prompt_with_images,
    upload_result_prompt_note,
)
from chainagents.rag.runtime import RagUploadResult, UploadedRagFile
from chainagents.runtime import (
    AgentRuntime,
    ReasoningLevel,
    build_langgraph_run_config,
    normalize_reasoning_level,
    resolve_runtime_model_profile,
)
from chainagents.runtime.reflection import (
    ReflectionCollector,
    ReflectionProposal,
    reflection_save_prompt,
)


AGENT_STREAM_MODES = ["messages", "updates", "custom"]
NDJSON_MEDIA_TYPE = "application/x-ndjson"
MAX_RESPONSE_PDF_CONTENT_LENGTH = 100_000
MAX_RESPONSE_PDF_LINES = 2_000
MAX_CONCURRENT_RESPONSE_PDF_EXPORTS = 1
MAX_HISTORY_MESSAGES = 200
MAX_PENDING_REFLECTIONS = 128


class AgentImageUrl(BaseModel):
    """Validated data URL used for replayable image history."""

    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Accept only bounded base64 data URLs for supported image formats."""
        prefix, separator, encoded = value.partition(",")
        if not separator or not prefix.startswith("data:") or not prefix.endswith(";base64"):
            raise ValueError("history images must use base64 data URLs")
        mime_type = prefix[5:-7].lower()
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError("history image MIME type is not supported")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("history image data must be valid base64") from exc
        if len(decoded) > MAX_UPLOAD_FILE_BYTES:
            raise ValueError("history image exceeds the upload size limit")
        return value


class AgentTextContentPart(BaseModel):
    """Text content in replayable message history."""

    type: Literal["text"]
    text: str = Field(..., min_length=1)


class AgentImageContentPart(BaseModel):
    """Image content in replayable message history."""

    type: Literal["image_url"]
    image_url: AgentImageUrl


AgentHistoryContentPart = Annotated[
    AgentTextContentPart | AgentImageContentPart,
    Field(discriminator="type"),
]


class AgentHistoryMessage(BaseModel):
    """Caller-owned user or assistant message on a selected local branch."""

    role: Literal["user", "assistant"]
    content: str | list[AgentHistoryContentPart]

    @model_validator(mode="after")
    def validate_content(self) -> "AgentHistoryMessage":
        """Reject blank text and empty multipart messages."""
        if isinstance(self.content, str):
            if not self.content.strip():
                raise ValueError("history message content must not be blank")
        elif not self.content:
            raise ValueError("history message content must not be empty")
        elif self.role == "assistant" and any(
            isinstance(part, AgentImageContentPart) for part in self.content
        ):
            raise ValueError("history images are allowed only in user messages")
        return self


class AgentRunRequest(BaseModel):
    """HTTP request body for running the agent."""

    prompt: str = Field(..., min_length=1)
    command: str | None = None
    thread_id: str = Field(..., min_length=1)
    model: str | None = None
    reasoning: ReasoningLevel | None = None
    async_subagent_url: str | None = None
    mcp_session_id: str | None = None
    history: list[AgentHistoryMessage] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_MESSAGES,
    )
    source_thread_id: str | None = None

    @model_validator(mode="after")
    def validate_history_image_count(self) -> "AgentRunRequest":
        """Bound replay images to the same aggregate limit as uploads."""
        image_count = sum(
            isinstance(part, AgentImageContentPart)
            for message in self.history
            if isinstance(message.content, list)
            for part in message.content
        )
        if image_count > MAX_UPLOAD_FILES:
            raise ValueError(
                f"history may contain at most {MAX_UPLOAD_FILES} image attachments"
            )
        return self


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
    ui_api_version: int = 1
    models: list["RuntimeModelOption"]
    reasoning_levels: list[ReasoningLevel]
    features: "RuntimeFeatureFlags"
    starters: list["RuntimeStarter"]
    commands: list["RuntimeCommand"]
    uploads: "RuntimeUploadCapabilities"


class RuntimeModelOption(BaseModel):
    """One selectable runtime model or named profile."""

    id: str
    provider: str
    default_reasoning: ReasoningLevel
    modalities: list[str]


class RuntimeFeatureFlags(BaseModel):
    """UI surfaces supported by the active ChainAgents configuration."""

    generated_files: bool
    reasoning: bool
    tools: bool
    generated_panels: bool
    reflection: bool
    rag: bool
    images: bool


class RuntimeStarter(BaseModel):
    """Configured starter prompt exposed by the UI."""

    label: str
    message: str
    command: str | None = None
    icon: str | None = None


class RuntimeCommand(BaseModel):
    """Configured native slash command exposed by the UI."""

    name: str
    description: str
    target: str


class RuntimeUploadCapabilities(BaseModel):
    """Upload limits and allowlists shared with browser clients."""

    max_files: int
    max_file_size_bytes: int
    image_mime_types: list[str]
    rag_extensions: list[str]


class ReflectionProposalRequest(BaseModel):
    """Client-confirmed reflection proposal emitted by an earlier stream."""

    reason: Literal["correction", "tool_failure"]
    memory_file: str = Field(..., min_length=1)
    lesson: str = Field(..., min_length=1)
    trigger: str = Field(..., min_length=1)
    tool_name: str = ""
    tool_result: str = ""
    confirmation_token: str = Field(..., min_length=1)


class ReflectionSaveRequest(BaseModel):
    """Request to save one validated reflection proposal."""

    thread_id: str = Field(..., min_length=1)
    proposal: ReflectionProposalRequest
    model: str | None = None
    reasoning: ReasoningLevel | None = None
    async_subagent_url: str | None = None
    mcp_session_id: str | None = None


class ReflectionSaveResponse(BaseModel):
    """Result of the hidden reflection save workflow."""

    saved: bool
    memory_file: str
    thread_id: str


class ResponsePdfExportRequest(BaseModel):
    """HTTP request body for exporting one assistant response as PDF."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_RESPONSE_PDF_CONTENT_LENGTH,
    )
    filename: str = Field(
        default="response",
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )


@dataclass(frozen=True)
class AgentRunContext:
    """Resolved request values used for one agent run."""

    prompt: str
    thread_id: str
    model_name: str
    reasoning_level: ReasoningLevel
    reasoning_level_is_explicit: bool
    async_subagent_url: str | None
    mcp_session_id: str | None
    history: tuple[dict[str, Any], ...]
    source_thread_id: str | None
    direct_response: str | None = None
    command_error: str | None = None
    command_error_status: int = 422
    image_parts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PendingReflection:
    """Bind one opaque confirmation token to a server-issued proposal."""

    thread_id: str
    proposal: ReflectionProposal


def create_app(
    runtime: Any | None = None,
    ui_dir: str | Path | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    Args:
        runtime: Optional runtime test double or initialized AgentRuntime.

    Returns:
        The configured FastAPI app.
    """
    managed_runtime: AgentRuntime | None = None
    configured_ui_directory = _resolve_ui_directory(ui_dir)
    pending_reflections: OrderedDict[str, PendingReflection] = OrderedDict()
    pdf_render_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RESPONSE_PDF_EXPORTS)

    def store_pending_reflection(
        token: str,
        pending: PendingReflection,
    ) -> None:
        """Store one bounded pending proposal, evicting the oldest if needed."""
        pending_reflections[token] = pending
        pending_reflections.move_to_end(token)
        while len(pending_reflections) > MAX_PENDING_REFLECTIONS:
            pending_reflections.popitem(last=False)

    def issue_reflection_token(
        thread_id: str,
        proposal: ReflectionProposal,
    ) -> str:
        """Issue an opaque token proving a proposal originated from this server."""
        token = secrets.token_urlsafe(32)
        while token in pending_reflections:
            token = secrets.token_urlsafe(32)
        store_pending_reflection(
            token,
            PendingReflection(thread_id=thread_id, proposal=proposal),
        )
        return token

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
        active_model = resolve_runtime_model_profile(config)
        model_options = [
            RuntimeModelOption(
                id=model_id,
                provider=profile.provider,
                default_reasoning=profile.reasoning_effort,
                modalities=list(profile.modalities),
            )
            for model_id in config.model_choices
            for profile in [resolve_runtime_model_profile(config, model_id)]
        ]
        extensions = getattr(config, "extensions", None)
        reflection = getattr(extensions, "agent_reflection", None)
        starters = getattr(extensions, "chainlit_starters", ())
        commands = getattr(extensions, "chainlit_commands", ())
        return RuntimeStatusResponse(
            model=config.model_name,
            model_provider=active_model.provider,
            model_choices=list(config.model_choices),
            default_reasoning=config.default_reasoning,
            agent_state=config.agent_state,
            recursion_limit=config.recursion_limit,
            persistence_mode=config.persistence_mode,
            models=model_options,
            reasoning_levels=["low", "medium", "high"],
            features=RuntimeFeatureFlags(
                generated_files=True,
                reasoning=bool(
                    getattr(extensions, "chainlit_reasoning_mode_enabled", True)
                ),
                tools=bool(getattr(extensions, "chainlit_tool_steps_enabled", True)),
                generated_panels=bool(
                    getattr(extensions, "chainlit_generative_ui_enabled", True)
                ),
                reflection=bool(getattr(reflection, "enabled", False)),
                rag=bool(config.rag_requested and config.rag is not None),
                images=any("image" in option.modalities for option in model_options),
            ),
            starters=[
                RuntimeStarter(
                    label=starter.label,
                    message=starter.message,
                    command=starter.command,
                    icon=starter.icon,
                )
                for starter in starters
            ],
            commands=[
                RuntimeCommand(
                    name=command.name,
                    description=command.description,
                    target=command.target,
                )
                for command in commands
            ],
            uploads=RuntimeUploadCapabilities(
                max_files=MAX_UPLOAD_FILES,
                max_file_size_bytes=MAX_UPLOAD_FILE_BYTES,
                image_mime_types=list(SUPPORTED_IMAGE_MIME_TYPES),
                rag_extensions=list(SUPPORTED_RAG_EXTENSIONS),
            ),
        )

    @app.get("/api/generated-files/{relative_path:path}")
    async def download_generated_file(
        relative_path: str,
        request: Request,
    ) -> FileResponse:
        """Download one existing file confined to generated outputs."""
        active_runtime = _runtime_from_request(request)
        path = resolve_generated_download(
            relative_path,
            project_root=Path(active_runtime.project_root),
        )
        if path is None:
            raise HTTPException(status_code=404, detail="Generated file not found.")
        mime_type, _encoding = mimetypes.guess_type(path.name)
        return FileResponse(
            path,
            filename=path.name,
            media_type=mime_type or "application/octet-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post(
        "/api/exports/pdf",
        response_class=Response,
        responses={
            200: {
                "description": "Rendered assistant response PDF.",
                "content": {
                    "application/pdf": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    async def export_response_pdf(payload: ResponsePdfExportRequest) -> Response:
        content = payload.content.strip()
        if not content:
            raise HTTPException(status_code=422, detail="content must not be blank.")
        if len(content.splitlines()) > MAX_RESPONSE_PDF_LINES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"content must contain at most {MAX_RESPONSE_PDF_LINES:,} lines."
                ),
            )

        try:
            async with pdf_render_semaphore:
                pdf_content = await run_in_threadpool(build_pdf_bytes, content)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return Response(
            content=pdf_content,
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{payload.filename}.pdf"'
                )
            },
            media_type="application/pdf",
        )

    @app.post("/api/agent/invoke", response_model=AgentRunResponse)
    async def invoke_agent(
        payload: AgentRunRequest,
        request: Request,
    ) -> AgentRunResponse:
        active_runtime = _runtime_from_request(request)
        context = await _prepare_run_context(active_runtime, payload)
        if context.command_error:
            raise HTTPException(
                status_code=context.command_error_status,
                detail=context.command_error,
            )
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
        context = await _prepare_run_context(active_runtime, payload)
        return StreamingResponse(
            _agent_stream_lines(
                active_runtime,
                context,
                issue_reflection_token=issue_reflection_token,
            ),
            media_type=NDJSON_MEDIA_TYPE,
        )

    @app.post("/api/agent/stream/multipart")
    async def stream_agent_multipart(
        request: Request,
        thread_id: str = Form(...),
        prompt: str = Form(""),
        command: str | None = Form(None),
        model: str | None = Form(None),
        reasoning: str | None = Form(None),
        source_thread_id: str | None = Form(None),
        history: str | None = Form(None),
        async_subagent_url: str | None = Form(None),
        mcp_session_id: str | None = Form(None),
        files: list[UploadFile] | None = File(None),
    ) -> StreamingResponse:
        active_runtime = _runtime_from_request(request)
        normalized_uploads = await _read_multipart_uploads(files or [])
        if not prompt.strip() and not normalized_uploads:
            raise HTTPException(
                status_code=422,
                detail="A prompt or at least one attachment is required.",
            )
        image_uploads = [upload for upload in normalized_uploads if upload.kind == "image"]
        rag_uploads = [upload for upload in normalized_uploads if upload.kind == "rag"]
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="chainagents-api-uploads-"
        )
        try:
            history_payload = _parse_history_form(history)
            context_request = AgentRunRequest.model_validate(
                {
                    "prompt": prompt.strip() or "__attachment_only__",
                    "command": command,
                    "thread_id": thread_id,
                    "model": model,
                    "reasoning": reasoning,
                    "source_thread_id": source_thread_id,
                    "history": history_payload,
                    "async_subagent_url": async_subagent_url,
                    "mcp_session_id": mcp_session_id,
                }
            )
            if prompt.strip() or image_uploads:
                context_request.prompt = prompt_with_images(
                    prompt,
                    image_names=tuple(upload.name for upload in image_uploads),
                )
                context = await _prepare_run_context(
                    active_runtime,
                    context_request,
                    has_current_images=bool(image_uploads),
                )
            elif _optional_text(command):
                context = await _prepare_run_context(
                    active_runtime,
                    context_request,
                    command_raw_text="",
                )
            else:
                context = replace(
                    _run_context(active_runtime, context_request),
                    prompt="",
                )
            stored_rag_uploads = _store_temporary_rag_uploads(
                Path(temporary_directory.name),
                rag_uploads,
            )
        except ValidationError as exc:
            temporary_directory.cleanup()
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except Exception:
            temporary_directory.cleanup()
            raise

        async def multipart_lines() -> AsyncIterator[str]:
            active_context = context
            try:
                if (
                    active_context.source_thread_id
                    and active_context.source_thread_id != active_context.thread_id
                ):
                    clone_uploads = getattr(active_runtime, "clone_rag_uploads", None)
                    if clone_uploads is not None:
                        clone_result = await clone_uploads(
                            source_thread_id=active_context.source_thread_id,
                            target_thread_id=active_context.thread_id,
                        )
                        if bool(getattr(clone_result, "conflict", False)):
                            raise RuntimeError(
                                str(getattr(clone_result, "reason", "")).strip()
                                or "Target branch thread is not fresh."
                            )
                    active_context = replace(active_context, source_thread_id=None)

                upload_result: RagUploadResult | None = None
                if stored_rag_uploads:
                    upload_result = await active_runtime.ingest_rag_uploads(
                        thread_id=active_context.thread_id,
                        uploads=stored_rag_uploads,
                    )
                    yield _json_line(_attachment_status_payload(upload_result))

                if not active_context.prompt and not image_uploads:
                    yield _json_line(_done_payload(active_context))
                    return

                prompt_note = (
                    upload_result_prompt_note(upload_result.added_files)
                    if upload_result is not None
                    else ""
                )
                final_prompt = f"{active_context.prompt}{prompt_note}"
                image_parts = tuple(image_content_part(upload) for upload in image_uploads)
                active_context = replace(
                    active_context,
                    prompt=final_prompt,
                    image_parts=image_parts,
                )
                async for line in _agent_stream_lines(
                    active_runtime,
                    active_context,
                    issue_reflection_token=issue_reflection_token,
                ):
                    yield line
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield _json_line(
                    {
                        "kind": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "thread_id": active_context.thread_id,
                        "model": active_context.model_name,
                        "reasoning": active_context.reasoning_level,
                    }
                )
            finally:
                temporary_directory.cleanup()

        return StreamingResponse(multipart_lines(), media_type=NDJSON_MEDIA_TYPE)

    @app.post("/api/reflections/save", response_model=ReflectionSaveResponse)
    async def save_reflection(
        payload: ReflectionSaveRequest,
        request: Request,
    ) -> ReflectionSaveResponse:
        active_runtime = _runtime_from_request(request)
        reflection_config = getattr(
            getattr(active_runtime.config, "extensions", None),
            "agent_reflection",
            None,
        )
        if not bool(getattr(reflection_config, "enabled", False)):
            raise HTTPException(
                status_code=409,
                detail="Reflection saving is not enabled for this runtime.",
            )

        proposal_payload = payload.proposal
        configured_memory_file = str(
            getattr(reflection_config, "memory_file", "")
        ).strip()
        if proposal_payload.memory_file.strip() != configured_memory_file:
            raise HTTPException(
                status_code=422,
                detail="Reflection proposal memory_file does not match runtime configuration.",
            )
        lesson = proposal_payload.lesson.strip()
        max_lesson_chars = int(getattr(reflection_config, "max_lesson_chars", 0))
        if not lesson or max_lesson_chars <= 0 or len(lesson) > max_lesson_chars:
            raise HTTPException(
                status_code=422,
                detail="Reflection proposal lesson exceeds runtime validation limits.",
            )

        context = _run_context(
            active_runtime,
            AgentRunRequest(
                prompt="Save confirmed reflection lesson.",
                thread_id=payload.thread_id,
                model=payload.model,
                reasoning=payload.reasoning,
                async_subagent_url=payload.async_subagent_url,
                mcp_session_id=payload.mcp_session_id,
            ),
        )
        proposal = ReflectionProposal(
            reason=proposal_payload.reason,
            memory_file=configured_memory_file,
            lesson=lesson,
            trigger=proposal_payload.trigger.strip(),
            tool_name=proposal_payload.tool_name.strip(),
            tool_result=proposal_payload.tool_result.strip(),
        )
        confirmation_token = proposal_payload.confirmation_token.strip()
        pending_reflection = pending_reflections.get(confirmation_token)
        if pending_reflection is None:
            raise HTTPException(
                status_code=409,
                detail="Reflection proposal is unknown, expired, or already saved.",
            )
        if (
            pending_reflection.thread_id != context.thread_id
            or pending_reflection.proposal != proposal
        ):
            raise HTTPException(
                status_code=422,
                detail="Reflection proposal does not match the server-issued proposal.",
            )

        pending_reflections.pop(confirmation_token)
        reflection_thread_id = f"{context.thread_id}:reflection"
        saved = False
        try:
            agent = await active_runtime.get_agent(
                context.reasoning_level,
                model_name=context.model_name,
                reasoning_level_is_explicit=context.reasoning_level_is_explicit,
                thread_id=reflection_thread_id,
                async_subagent_url_override=context.async_subagent_url,
                mcp_session_id=context.mcp_session_id,
            )
            await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": reflection_save_prompt(proposal),
                        }
                    ]
                },
                config=build_langgraph_run_config(
                    active_runtime.config,
                    thread_id=reflection_thread_id,
                ),
            )
            saved = True
        finally:
            if not saved:
                store_pending_reflection(confirmation_token, pending_reflection)
        return ReflectionSaveResponse(
            saved=True,
            memory_file=configured_memory_file,
            thread_id=reflection_thread_id,
        )

    if configured_ui_directory is not None:
        app.mount(
            "/",
            StaticFiles(directory=configured_ui_directory, html=True),
            name="sparxui",
        )

    return app


def _resolve_ui_directory(ui_dir: str | Path | None) -> Path | None:
    """Resolve and validate optional same-origin static UI hosting."""
    raw_value: str | Path | None = ui_dir
    if raw_value is None:
        raw_value = os.getenv("CHAINAGENTS_UI_DIR")
    if raw_value is None or not str(raw_value).strip():
        return None
    directory = Path(str(raw_value)).expanduser().resolve()
    index_path = directory / "index.html"
    if not directory.is_dir() or not index_path.is_file():
        raise ValueError(
            "Configured ChainAgents UI directory must exist and contain index.html: "
            f"{directory}"
        )
    return directory


async def _read_multipart_uploads(files: list[UploadFile]) -> list[NormalizedUpload]:
    """Read and validate a bounded set of FastAPI uploads."""
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"At most {MAX_UPLOAD_FILES} files may be uploaded per request.",
        )

    uploads: list[NormalizedUpload] = []
    for upload in files:
        try:
            data = await upload.read(MAX_UPLOAD_FILE_BYTES + 1)
            uploads.append(
                normalize_upload(
                    name=upload.filename or "upload",
                    declared_mime=upload.content_type,
                    data=data,
                )
            )
        except OverflowError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            await upload.close()
    return uploads


def _parse_history_form(history: str | None) -> list[dict[str, Any]]:
    """Parse optional JSON history supplied as one multipart form field."""
    if history is None or not history.strip():
        return []
    try:
        value = json.loads(history)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="history must be valid JSON") from exc
    if not isinstance(value, list):
        raise HTTPException(status_code=422, detail="history must be a JSON array")
    return value


def _store_temporary_rag_uploads(
    directory: Path,
    uploads: list[NormalizedUpload],
) -> list[UploadedRagFile]:
    """Materialize validated RAG bytes for the existing ingestion runtime."""
    stored: list[UploadedRagFile] = []
    for index, upload in enumerate(uploads):
        path = directory / f"{index}-{upload.name}"
        path.write_bytes(upload.data)
        stored.append(UploadedRagFile(path=path, name=upload.name))
    return stored


async def _agent_stream_lines(
    runtime: Any,
    context: AgentRunContext,
    *,
    issue_reflection_token: Callable[[str, ReflectionProposal], str],
) -> AsyncIterator[str]:
    """Yield the stable NDJSON stream contract for one resolved run."""
    if context.command_error:
        yield _json_line(
            {
                "kind": "error",
                "error": context.command_error,
                "thread_id": context.thread_id,
                "model": context.model_name,
                "reasoning": context.reasoning_level,
            }
        )
        return

    reflection_collector = ReflectionCollector.from_runtime_config(
        runtime.config,
        prompt=context.prompt,
    )
    generated_file_paths: list[str] = []
    generated_files_emitted = False
    response_parts: list[str] = []
    tool_calls: dict[str, tuple[str, str]] = {}
    try:
        async for event in _iter_agent_events(
            runtime,
            context,
            reflection_collector=reflection_collector,
        ):
            if event.kind == "tool_call":
                if event.previous_tool_call_id:
                    tool_calls.pop(event.previous_tool_call_id, None)
                tool_calls[event.tool_call_id] = (event.tool_name, event.tool_args)
            elif event.kind == "tool_result":
                tool_name, tool_args = tool_calls.pop(
                    event.tool_call_id,
                    (event.tool_name, ""),
                )
                if event.status.lower() != "error":
                    generated_file_paths.extend(
                        generated_file_paths_from_tool_args(tool_name, tool_args)
                    )
            elif event.kind == "response_delta":
                response_parts.append(event.text)
            yield _json_line(_event_payload(event, context))
        generated_files_line = _generated_files_line(
            runtime,
            context,
            generated_file_paths=generated_file_paths,
            response_text="".join(response_parts),
        )
        if generated_files_line is not None:
            yield generated_files_line
            generated_files_emitted = True
        proposal = reflection_collector.build_proposal()
        if proposal is not None:
            yield _json_line(
                _reflection_proposal_payload(
                    proposal,
                    context,
                    issue_reflection_token=issue_reflection_token,
                )
            )
        yield _json_line(_done_payload(context))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        reflection_collector.mark_run_failed(exc)
        if not generated_files_emitted:
            generated_files_line = _generated_files_line(
                runtime,
                context,
                generated_file_paths=generated_file_paths,
                response_text="".join(response_parts),
            )
            if generated_files_line is not None:
                yield generated_files_line
        proposal = reflection_collector.build_proposal()
        if proposal is not None:
            yield _json_line(
                _reflection_proposal_payload(
                    proposal,
                    context,
                    issue_reflection_token=issue_reflection_token,
                )
            )
        yield _json_line(
            {
                "kind": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "thread_id": context.thread_id,
                "model": context.model_name,
                "reasoning": context.reasoning_level,
            }
        )


def _generated_files_line(
    runtime: Any,
    context: AgentRunContext,
    *,
    generated_file_paths: list[str],
    response_text: str,
) -> str | None:
    """Build the generated-files event after final filesystem validation."""
    raw_paths = [
        *generated_file_paths,
        *generated_file_paths_from_text(response_text),
    ]
    if not raw_paths:
        return None
    descriptors = generated_file_descriptors(
        raw_paths,
        project_root=Path(runtime.project_root),
    )
    if not descriptors:
        return None
    return _json_line(
        {
            "kind": "generated_files",
            "source": "main-agent",
            "files": [descriptor.to_payload() for descriptor in descriptors],
            "thread_id": context.thread_id,
            "model": context.model_name,
            "reasoning": context.reasoning_level,
        }
    )


def _reflection_proposal_payload(
    proposal: ReflectionProposal,
    context: AgentRunContext,
    *,
    issue_reflection_token: Callable[[str, ReflectionProposal], str],
) -> dict[str, Any]:
    """Build a reflection event whose proposal can be confirmed exactly once."""
    proposal_payload = proposal.to_payload()
    proposal_payload["confirmation_token"] = issue_reflection_token(
        context.thread_id,
        proposal,
    )
    return {
        "kind": "reflection_proposal",
        "proposal": proposal_payload,
        "thread_id": context.thread_id,
        "model": context.model_name,
        "reasoning": context.reasoning_level,
    }


def _done_payload(context: AgentRunContext) -> dict[str, Any]:
    """Build the terminal stream event."""
    return {
        "kind": "done",
        "thread_id": context.thread_id,
        "model": context.model_name,
        "reasoning": context.reasoning_level,
    }


def _attachment_status_payload(result: RagUploadResult) -> dict[str, Any]:
    """Build a typed stream event for RAG attachment ingestion."""
    if result.added_files:
        count = len(result.added_files)
        noun = "file" if count == 1 else "files"
        message = f"Indexed {count} uploaded {noun} for this thread."
        status = "complete"
    else:
        message = result.reason or "No supported RAG files were indexed."
        status = "error"
    return {
        "kind": "attachment_status",
        "status": status,
        "message": message,
        "added_files": list(result.added_files),
        "rejected_files": list(result.rejected_files),
        "indexed_files": result.indexed_files,
        "chunk_count": result.chunk_count,
        "thread_id": result.thread_id,
    }


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
        reasoning_level_is_explicit=request.reasoning is not None,
        async_subagent_url=_optional_text(request.async_subagent_url),
        mcp_session_id=_optional_text(request.mcp_session_id),
        history=tuple(_history_message_payload(message) for message in request.history),
        source_thread_id=_optional_text(request.source_thread_id),
    )


async def _prepare_run_context(
    runtime: Any,
    request: AgentRunRequest,
    *,
    has_current_images: bool = False,
    command_raw_text: str | None = None,
) -> AgentRunContext:
    """Resolve validation and native command behavior for one request."""
    context = _run_context(runtime, request)
    await _validate_history_replay(runtime, context)
    _validate_image_modalities(
        runtime,
        context.model_name,
        has_images=has_current_images or _history_contains_images(request.history),
    )
    context = await _clone_branch_rag_before_commands(runtime, context)
    if context.command_error:
        return context
    parsed = resolve_native_command(
        raw_text=context.prompt if command_raw_text is None else command_raw_text,
        selected_command=request.command,
    )
    if parsed is None:
        return context

    try:
        result = await resolve_runtime_command(
            runtime=runtime,
            parsed=parsed,
            thread_id=context.thread_id,
            mcp_session_id=context.mcp_session_id,
        )
    except ValueError as exc:
        return replace(context, command_error=str(exc), command_error_status=422)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return replace(
            context,
            command_error=f"{type(exc).__name__}: {exc}",
            command_error_status=500,
        )
    if result.target == "unknown":
        return replace(
            context,
            command_error=f"Unknown command `/{result.command_name}`.",
        )
    if result.target == "mcp_tool":
        return replace(context, direct_response=dumps_tool_result(result.tool_result))
    return replace(context, prompt=(result.prompt or "").strip())


async def _clone_branch_rag_before_commands(
    runtime: Any,
    context: AgentRunContext,
) -> AgentRunContext:
    """Atomically reserve a branch's RAG scope before command side effects."""
    if (
        context.source_thread_id is None
        or context.source_thread_id == context.thread_id
    ):
        return context
    clone_uploads = getattr(runtime, "clone_rag_uploads", None)
    if clone_uploads is None:
        return replace(context, source_thread_id=None)
    try:
        result = await clone_uploads(
            source_thread_id=context.source_thread_id,
            target_thread_id=context.thread_id,
        )
    except ValueError as exc:
        return replace(
            context,
            command_error=str(exc),
            command_error_status=422,
            source_thread_id=None,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return replace(
            context,
            command_error=f"{type(exc).__name__}: {exc}",
            command_error_status=500,
            source_thread_id=None,
        )
    if bool(getattr(result, "conflict", False)):
        return replace(
            context,
            command_error=(
                str(getattr(result, "reason", "")).strip()
                or "Target branch thread is not fresh."
            ),
            command_error_status=409,
            source_thread_id=None,
        )
    return replace(context, source_thread_id=None)


def _history_contains_images(history: list[AgentHistoryMessage]) -> bool:
    """Return whether replay history contains user image content."""
    return any(
        isinstance(part, AgentImageContentPart)
        for message in history
        if isinstance(message.content, list)
        for part in message.content
    )


def _validate_image_modalities(
    runtime: Any,
    model_name: str,
    *,
    has_images: bool,
) -> None:
    """Reject image content before commands or model calls for text-only profiles."""
    if not has_images:
        return
    try:
        selected_model = resolve_runtime_model_profile(runtime.config, model_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if "image" not in selected_model.modalities:
        raise HTTPException(
            status_code=422,
            detail="The selected model does not declare image input support.",
        )


async def _validate_history_replay(runtime: Any, context: AgentRunContext) -> None:
    """Allow stateful history replay only into a fresh branch checkpoint."""
    if not context.history or runtime.config.agent_state != "stateful":
        return
    if (
        context.source_thread_id is None
        or context.source_thread_id == context.thread_id
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Stateful history replay requires a distinct source_thread_id and "
                "fresh target thread_id."
            ),
        )
    existing_checkpoint = await runtime.checkpointer.aget_tuple(
        {"configurable": {"thread_id": context.thread_id}}
    )
    if existing_checkpoint is not None:
        raise HTTPException(
            status_code=409,
            detail="History cannot be replayed into an existing stateful thread.",
        )


def _history_message_payload(message: AgentHistoryMessage) -> dict[str, Any]:
    """Convert validated history into the LangChain message wire shape."""
    content: str | list[dict[str, Any]]
    if isinstance(message.content, str):
        content = message.content.strip()
    else:
        content = [part.model_dump() for part in message.content]
    return {"role": message.role, "content": content}


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
    *,
    reflection_collector: ReflectionCollector | None = None,
) -> AsyncIterator[AgentStreamEvent]:
    if context.direct_response is not None:
        yield AgentStreamEvent(
            kind="response_delta",
            source="native-command",
            text=context.direct_response,
        )
        return

    agent = await runtime.get_agent(
        context.reasoning_level,
        model_name=context.model_name,
        reasoning_level_is_explicit=context.reasoning_level_is_explicit,
        thread_id=context.thread_id,
        async_subagent_url_override=context.async_subagent_url,
        mcp_session_id=context.mcp_session_id,
    )
    payload = {
        "messages": [
            *context.history,
            {
                "role": "user",
                "content": (
                    [{"type": "text", "text": context.prompt}, *context.image_parts]
                    if context.image_parts
                    else context.prompt
                ),
            },
        ]
    }
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
                if reflection_collector is not None:
                    reflection_collector.record_event(event)
                yield event
    finally:
        with suppress(Exception):
            await stream.aclose()


def _event_payload(event: AgentStreamEvent, context: AgentRunContext) -> dict[str, Any]:
    payload = asdict(event)
    if event.kind not in {"ui_message", "ui_remove"}:
        for key in ("ui_id", "ui_name", "ui_props", "ui_metadata"):
            payload.pop(key, None)
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
    parser.add_argument(
        "--ui-dir",
        default=None,
        help=(
            "Serve a built SparxUI directory after API routes. "
            "Defaults to CHAINAGENTS_UI_DIR when set."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the API server from the console script."""
    args = build_parser().parse_args(argv)

    if args.ui_dir is not None:
        os.environ["CHAINAGENTS_UI_DIR"] = args.ui_dir

    import uvicorn

    uvicorn.run(
        "chainagents.interfaces.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


app = create_app()


if __name__ == "__main__":
    main()
