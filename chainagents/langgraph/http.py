"""Custom Agent Server routes and lifecycle for process-local task managers."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from chainagents.runtime.graph import (
    close_static_background_tasks,
    close_static_background_session,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cancel exported-graph background jobs before Agent Server shutdown."""
    try:
        yield
    finally:
        await close_static_background_tasks()


app = FastAPI(lifespan=lifespan)


@app.delete("/background-tasks/sessions/{thread_id:path}")
async def close_background_session(thread_id: str) -> dict[str, object]:
    """Cancel and forget one session across exported graph managers."""
    session_id = thread_id.strip()
    if not session_id:
        raise HTTPException(status_code=422, detail="thread_id must not be blank.")
    await close_static_background_session(session_id)
    return {"closed": True, "thread_id": session_id}
