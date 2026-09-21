"""Provider-specific request and streaming adapters."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import AsyncIterator, Iterator
from functools import cached_property
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import AIMessageChunk, BaseMessage, ToolCallChunk
from langchain_core.outputs import ChatGenerationChunk
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


class _ToolCallChunkIndexRepair:
    """Repair a provider stream that reuses one index for distinct tool calls."""

    def __init__(self) -> None:
        self._call_indexes: dict[str, int] = {}
        self._index_owners: dict[int, str] = {}
        self._provider_index_calls: dict[int, list[str]] = {}
        self._active_index: int | None = None
        self._repairing = False

    def repair(self, chunk: ChatGenerationChunk) -> ChatGenerationChunk:
        """Return a chunk with unambiguous tool-call indexes when needed."""
        message = chunk.message
        if not isinstance(message, AIMessageChunk) or not message.tool_call_chunks:
            return chunk

        repaired_chunks: list[ToolCallChunk] = []
        changed = False
        for tool_call_chunk in message.tool_call_chunks:
            repaired = tool_call_chunk.copy()
            call_id = repaired.get("id")
            provider_index = repaired.get("index")
            if isinstance(call_id, str) and call_id:
                repaired_index = self._call_indexes.get(call_id)
                if repaired_index is None and isinstance(provider_index, int):
                    provider_calls = self._provider_index_calls.setdefault(
                        provider_index,
                        [],
                    )
                    owner = self._index_owners.get(provider_index)
                    if provider_calls or (owner is not None and owner != call_id):
                        self._repairing = True
                        repaired_index = self._next_index()
                    else:
                        repaired_index = provider_index
                    provider_calls.append(call_id)
                    self._call_indexes[call_id] = repaired_index
                    self._index_owners[repaired_index] = call_id
                if repaired_index is not None:
                    self._active_index = repaired_index
                    if provider_index != repaired_index:
                        repaired["index"] = repaired_index
                        changed = True
            elif self._repairing and self._active_index is not None:
                repaired_index = self._active_index
                if isinstance(provider_index, int):
                    provider_calls = self._provider_index_calls.get(
                        provider_index,
                        [],
                    )
                    if len(provider_calls) == 1:
                        repaired_index = self._call_indexes[provider_calls[0]]
                if provider_index != repaired_index:
                    repaired["index"] = repaired_index
                    changed = True
            repaired_chunks.append(repaired)

        if not changed:
            return chunk
        repaired_message = message.model_copy(deep=True)
        repaired_message.tool_call_chunks = repaired_chunks
        return chunk.model_copy(update={"message": repaired_message})

    def _next_index(self) -> int:
        """Return the first non-negative index not already owned by a call."""
        candidate = 0
        while candidate in self._index_owners:
            candidate += 1
        return candidate


class OpenAICompatibleChatOpenAI(ChatOpenAI):
    """Adapt OpenAI-compatible chat chunks while preserving reasoning deltas."""

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chunks with request-local repair for reused tool-call indexes."""
        repair = _ToolCallChunkIndexRepair()
        for chunk in super()._stream(
            messages,
            stop=stop,
            run_manager=None,
            stream_usage=stream_usage,
            **kwargs,
        ):
            repaired = repair.repair(chunk)
            if run_manager is not None:
                logprobs = (repaired.generation_info or {}).get("logprobs")
                run_manager.on_llm_new_token(
                    repaired.text,
                    chunk=repaired,
                    logprobs=logprobs,
                )
            yield repaired

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        *,
        stream_usage: bool | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously stream chunks with request-local index repair."""
        repair = _ToolCallChunkIndexRepair()
        async for chunk in super()._astream(
            messages,
            stop=stop,
            run_manager=None,
            stream_usage=stream_usage,
            **kwargs,
        ):
            repaired = repair.repair(chunk)
            if run_manager is not None:
                logprobs = (repaired.generation_info or {}).get("logprobs")
                await run_manager.on_llm_new_token(
                    repaired.text,
                    chunk=repaired,
                    logprobs=logprobs,
                )
            yield repaired

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
    """Adapt Snowflake Cortex Chat Completions tool calls and result turns."""

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Return a copied payload with canonical IDs and Cortex-safe tool turns."""
        payload = copy.deepcopy(
            super()._get_request_payload(input_, stop=stop, **kwargs)
        )
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        pending_ids: set[str] = set()
        pending_canonical_ids: set[str] = set()
        canonical_ids: dict[str, str] = {}
        pending_assistant: dict[str, Any] | None = None
        pending_call_order: list[str] = []
        pending_calls: dict[str, dict[str, Any]] = {}
        pending_results: dict[str, dict[str, Any]] = {}
        rewritten_messages: list[Any] = []
        for message in messages:
            if not isinstance(message, dict):
                if pending_ids:
                    raise ValueError("incomplete tool-call batch before a new non-tool message")
                rewritten_messages.append(message)
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
                    pending_call_order.append(raw_id)
                    pending_calls[raw_id] = tool_call
                if len(tool_calls) == 1:
                    rewritten_messages.append(message)
                else:
                    pending_assistant = message
                continue

            if role == "tool":
                raw_id = message.get("tool_call_id")
                if not isinstance(raw_id, str) or not raw_id:
                    raise ValueError("empty tool response ID")
                if raw_id not in pending_ids:
                    raise ValueError("unmatched tool response ID")
                message["tool_call_id"] = canonical_ids[raw_id]
                pending_ids.remove(raw_id)
                if pending_assistant is None:
                    rewritten_messages.append(message)
                else:
                    pending_results[raw_id] = message
                if not pending_ids:
                    if pending_assistant is not None:
                        for index, call_id in enumerate(pending_call_order):
                            replayed_assistant = copy.deepcopy(pending_assistant)
                            replayed_assistant["tool_calls"] = [pending_calls[call_id]]
                            if index:
                                replayed_assistant["content"] = None
                            rewritten_messages.extend(
                                [replayed_assistant, pending_results[call_id]]
                            )
                    pending_canonical_ids.clear()
                    canonical_ids.clear()
                    pending_assistant = None
                    pending_call_order.clear()
                    pending_calls.clear()
                    pending_results.clear()
                continue

            if pending_ids:
                raise ValueError("incomplete tool-call batch before a new non-tool message")
            rewritten_messages.append(message)

        if pending_ids:
            raise ValueError("incomplete tool-call batch at payload end")
        payload["messages"] = rewritten_messages
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
