"""Agent Server HTTP application lifecycle for shared backend resources."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from chainagents.langgraph.app import _backend_bundle


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Close backend clients on the same serving loop used by graph requests."""
    try:
        yield
    finally:
        await _backend_bundle.close()


app = FastAPI(lifespan=lifespan)
