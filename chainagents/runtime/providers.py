"""Provider-specific request and streaming adapters."""

from __future__ import annotations

import copy
import hashlib
from functools import cached_property
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI

from chainagents.runtime.constants import (
    OPENAI_COMPATIBLE_REASONING_DELTA_KEYS,
    SNOWFLAKE_CORTEX_CANONICAL_TOOL_CALL_ID_RE,
)


def _first_openai_compatible_delta(chunk: dict[str, Any]) -> dict[str, Any]:
    """Return the first delta object from an OpenAI-compatible chunk.

    Args:
        chunk: Streamed event chunk to normalize.

    Returns:
        The first delta object from an OpenAI-compatible chunk.
    """
    choices = chunk.get("choices", [])
    if not choices:
        nested_chunk = chunk.get("chunk")
        if isinstance(nested_chunk, dict):
            choices = nested_chunk.get("choices", [])

    if not isinstance(choices, list) or not choices:
        return {}

    choice = choices[0]
    if not isinstance(choice, dict):
        return {}

    delta = choice.get("delta")
    if isinstance(delta, dict):
        return delta
    return {}


def _openai_compatible_reasoning_delta(chunk: dict[str, Any]) -> Any:
    """Return reasoning content from an OpenAI-compatible delta.

    Args:
        chunk: Streamed event chunk to normalize.

    Returns:
        Reasoning content from an OpenAI-compatible delta.
    """
    delta = _first_openai_compatible_delta(chunk)
    for key in OPENAI_COMPATIBLE_REASONING_DELTA_KEYS:
        value = delta.get(key)
        if value not in (None, ""):
            return value
    return None


class OpenAICompatibleChatOpenAI(ChatOpenAI):
    """Adapt OpenAI-compatible chat chunks while preserving reasoning deltas."""

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict | None,
    ):
        """Convert provider chunks while preserving OpenAI-compatible reasoning.

        Args:
            chunk: Streamed event chunk to normalize.
            default_chunk_class: The default chunk class value.
            base_generation_info: The base generation info value.

        Returns:
            The convert chunk to generation chunk result.
        """
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None

        reasoning_delta = _openai_compatible_reasoning_delta(chunk)
        if reasoning_delta is None or not isinstance(
            generation_chunk.message,
            AIMessageChunk,
        ):
            return generation_chunk

        generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning_delta
        return generation_chunk


class SnowflakeCortexChatOpenAI(OpenAICompatibleChatOpenAI):
    """Adapt Snowflake Cortex Chat Completions tool-call IDs."""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Return a copied Chat Completions payload with canonical tool-call IDs."""
        payload = copy.deepcopy(
            super()._get_request_payload(input_, stop=stop, **kwargs)
        )
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        pending_ids: set[str] = set()
        pending_canonical_ids: set[str] = set()
        canonical_ids: dict[str, str] = {}
        for message in messages:
            if not isinstance(message, dict):
                if pending_ids:
                    raise ValueError("incomplete tool-call batch before a new non-tool message")
                continue
            role = message.get("role")
            tool_calls = message.get("tool_calls")
            if role == "assistant" and tool_calls:
                if pending_ids:
                    raise ValueError("incomplete tool-call batch before a new assistant batch")
                if not isinstance(tool_calls, list):
                    raise ValueError("assistant tool calls must be a list")
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        raise ValueError("assistant tool call must be an object")
                    raw_id = tool_call.get("id")
                    if not isinstance(raw_id, str) or not raw_id:
                        raise ValueError("empty tool call ID")
                    if raw_id in pending_ids:
                        raise ValueError("duplicate tool call ID within a batch")
                    canonical_id = (
                        raw_id
                        if SNOWFLAKE_CORTEX_CANONICAL_TOOL_CALL_ID_RE.fullmatch(raw_id)
                        else f"call_{hashlib.sha256(raw_id.encode('utf-8')).hexdigest()[:24]}"
                    )
                    if canonical_id in pending_canonical_ids:
                        raise ValueError("duplicate canonical tool call ID within a batch")
                    pending_ids.add(raw_id)
                    pending_canonical_ids.add(canonical_id)
                    canonical_ids[raw_id] = canonical_id
                    tool_call["id"] = canonical_id
                continue

            if role == "tool":
                raw_id = message.get("tool_call_id")
                if not isinstance(raw_id, str) or not raw_id:
                    raise ValueError("empty tool response ID")
                if raw_id not in pending_ids:
                    raise ValueError("unmatched tool response ID")
                message["tool_call_id"] = canonical_ids[raw_id]
                pending_ids.remove(raw_id)
                if not pending_ids:
                    pending_canonical_ids.clear()
                    canonical_ids.clear()
                continue

            if pending_ids:
                raise ValueError("incomplete tool-call batch before a new non-tool message")

        if pending_ids:
            raise ValueError("incomplete tool-call batch at payload end")
        return payload


class AnthropicDefaultQueryChatAnthropic(ChatAnthropic):
    """ChatAnthropic variant that forwards endpoint query params to the SDK."""

    default_query: dict[str, object] | None = None

    @cached_property
    def _client_params(self) -> dict[str, Any]:
        """Return Anthropic client params with optional default query values."""
        params = super()._client_params.copy()
        if self.default_query:
            params["default_query"] = self.default_query
        return params
