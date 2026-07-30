"""Aggregate request token usage and append it to a local JSON Lines log."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult


DEFAULT_TOKEN_USAGE_LOG_PATH = Path(".files/token-usage.jsonl")
logger = logging.getLogger(__name__)
_APPEND_LOCK = threading.Lock()


def _token_count(value: Any) -> int | None:
    """Return a valid token count or None for malformed values."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class TokenUsageFileCallbackHandler(BaseCallbackHandler):
    """Collect model usage for one request and write one aggregate record."""

    run_inline = True

    def __init__(self, *, thread_id: str, log_path: Path) -> None:
        """Initialize a request-scoped token usage callback."""
        super().__init__()
        self._thread_id = thread_id
        self._log_path = log_path
        self._state_lock = threading.Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._written = False

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Add normalized usage from a completed model call."""
        self._add_response_usage(response)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Add usage from a partial model response when one is available."""
        response = kwargs.get("response")
        if isinstance(response, LLMResult):
            self._add_response_usage(response)

    def _add_response_usage(self, response: LLMResult) -> None:
        """Normalize and add usage from one model result."""
        usage: dict[str, Any] | None = None
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None
        if isinstance(generation, ChatGeneration):
            message = generation.message
            if isinstance(message, AIMessage) and message.usage_metadata:
                usage = message.usage_metadata
        if usage is None and isinstance(response.llm_output, dict):
            provider_usage = response.llm_output.get("token_usage")
            if isinstance(provider_usage, dict):
                usage = provider_usage
        if usage is None:
            return
        input_tokens = (
            _token_count(usage.get("input_tokens", usage.get("prompt_tokens"))) or 0
        )
        output_tokens = (
            _token_count(
                usage.get("output_tokens", usage.get("completion_tokens"))
            )
            or 0
        )
        raw_total_tokens = usage.get("total_tokens")
        total_tokens = (
            input_tokens + output_tokens
            if raw_total_tokens is None
            else (_token_count(raw_total_tokens) or 0)
        )
        with self._state_lock:
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._total_tokens += total_tokens

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Append aggregate usage when the root request succeeds."""
        if parent_run_id is not None:
            return
        self._write_once(run_id=run_id, status="success")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Append aggregate usage when the root request fails."""
        if parent_run_id is not None:
            return
        self._write_once(run_id=run_id, status="error")

    def _write_once(self, *, run_id: UUID, status: str) -> None:
        """Append the terminal request record at most once."""
        with self._state_lock:
            if self._written:
                return
            self._written = True
            record = {
                "timestamp": datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
                "thread_id": self._thread_id,
                "request_id": str(run_id),
                "status": status,
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._total_tokens,
            }
        try:
            with _APPEND_LOCK:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(
                        json.dumps(
                            record,
                            ensure_ascii=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        except OSError:
            logger.warning(
                "Could not append token usage to %s.",
                self._log_path,
                exc_info=True,
            )
