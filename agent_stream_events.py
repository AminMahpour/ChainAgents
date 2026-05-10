"""Normalize LangGraph stream chunks for CLI, Chainlit, and TUI renderers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


StreamEventKind = Literal[
    "response_delta",
    "reasoning_delta",
    "tool_call",
    "tool_result",
    "summarization_status",
]

LANGGRAPH_STREAM_MODES = {
    "values",
    "updates",
    "custom",
    "messages",
    "checkpoints",
    "tasks",
    "debug",
}
SUMMARIZATION_STATUS_KIND = "summarization_status"


@dataclass(frozen=True)
class AgentStreamEvent:
    """A renderer-neutral event produced from LangGraph stream chunks."""

    kind: StreamEventKind
    source: str
    text: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    tool_args: str = ""
    tool_args_delta: str = ""
    tool_result: str = ""
    status: str = ""


def stringify_content(value: Any) -> str:
    """Convert LangChain message content into displayable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(stringify_content(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "reasoning", "content"):
            nested = value.get(key)
            if isinstance(nested, (str, list, dict)):
                return stringify_content(nested)
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    return str(value)


def langgraph_part_from_event_chunk(chunk: Any) -> dict[str, Any] | None:
    """Normalize LangGraph stream event chunks into part metadata."""
    if isinstance(chunk, dict):
        mode = chunk.get("type")
        if isinstance(mode, str) and mode in LANGGRAPH_STREAM_MODES and "data" in chunk:
            return chunk
        return None

    if not isinstance(chunk, tuple):
        return None

    if len(chunk) == 3:
        ns, mode, data = chunk
    elif len(chunk) == 2:
        first, data = chunk
        if isinstance(first, tuple):
            ns = first
            mode = "values"
        else:
            ns = ()
            mode = first
    else:
        return None

    if not isinstance(mode, str) or mode not in LANGGRAPH_STREAM_MODES:
        return None

    if isinstance(ns, tuple):
        namespace = ns
    elif ns in (None, ""):
        namespace = ()
    else:
        return None

    return {"type": mode, "ns": namespace, "data": data}


def reasoning_text_from_token(token: Any) -> str:
    """Extract reasoning text from a streamed model token."""
    if hasattr(token, "additional_kwargs"):
        text = stringify_content(token.additional_kwargs.get("reasoning_content"))
        if text:
            return text
    if hasattr(token, "reasoning_content"):
        return stringify_content(token.reasoning_content)
    return ""


def namespace_label(ns: tuple[str, ...], metadata: dict[str, Any]) -> str:
    """Return a display label for a streamed namespace."""
    agent_name = metadata.get("lc_agent_name")
    if agent_name:
        return str(agent_name)
    if not ns:
        return "main-agent"

    labels: list[str] = []
    for segment in ns:
        if segment.startswith("tools:"):
            labels.append(f"subagent {segment.split(':', 1)[1]}")
            continue
        labels.append(segment.split(":", 1)[0])
    return " / ".join(labels)


def iter_messages(value: Any) -> list[Any]:
    """Return message-like objects from nested LangGraph update payloads."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        if "messages" in value:
            return iter_messages(value["messages"])
        if "value" in value:
            return iter_messages(value["value"])
        return [value]
    if isinstance(value, (str, bytes)):
        return []

    for attr in ("value", "messages", "data"):
        if hasattr(value, attr):
            messages = iter_messages(getattr(value, attr))
            if messages:
                return messages

    try:
        return list(value)
    except TypeError:
        return [value]


def messages_from_node_data(data: Any) -> list[Any]:
    """Extract message objects from LangGraph node update payloads."""
    if data is None:
        return []
    if isinstance(data, dict):
        return iter_messages(data.get("messages"))
    return iter_messages(data)


def is_assistant_message(message: Any) -> bool:
    """Return whether a message is assistant-authored."""
    return getattr(message, "type", None) in {"ai", "AIMessageChunk"}


def message_text(message: Any) -> str:
    """Return normalized text from a LangChain-style message."""
    return stringify_content(getattr(message, "content", "")).strip()


def assistant_messages_for_current_prompt(messages: list[Any], prompt: str) -> list[Any]:
    """Return assistant messages produced after the current prompt began."""
    prompt_text = prompt.strip()
    current_prompt_index = -1
    for index, message in enumerate(messages):
        if getattr(message, "type", None) != "human":
            continue
        if message_text(message) == prompt_text:
            current_prompt_index = index

    if current_prompt_index < 0:
        return []

    return [
        message
        for message in messages[current_prompt_index + 1 :]
        if is_assistant_message(message)
    ]


class AgentStreamEventAdapter:
    """Convert raw LangGraph events into renderer-neutral stream events."""

    def __init__(self, *, prompt: str) -> None:
        self.prompt = prompt
        self.response_buffer = ""
        self.response_streamed_from_messages = False
        self.reasoning_buffers: dict[str, str] = {}
        self.tool_names: dict[str, str] = {}
        self.tool_args_buffers: dict[str, str] = {}
        self.tool_call_started: set[str] = set()
        self.completed_tool_results: set[tuple[str, str, str]] = set()

    def events_from_raw_event(self, event: dict[str, Any]) -> list[AgentStreamEvent]:
        """Return normalized events from one raw LangGraph stream event."""
        if event.get("event") != "on_chain_stream":
            return []
        if event.get("parent_ids"):
            return []

        data = event.get("data")
        if not isinstance(data, dict):
            return []

        part = langgraph_part_from_event_chunk(data.get("chunk"))
        if part is None:
            return []
        return self.events_from_part(part)

    def events_from_part(self, part: dict[str, Any]) -> list[AgentStreamEvent]:
        """Return normalized events from one LangGraph stream part."""
        kind = part.get("type")
        if kind == "messages":
            return self._events_from_message_chunk(part)
        if kind == "updates":
            return self._events_from_update_chunk(part)
        if kind == "custom":
            return self._events_from_custom_chunk(part)
        return []

    def _events_from_message_chunk(self, part: dict[str, Any]) -> list[AgentStreamEvent]:
        token, metadata = part["data"]
        metadata = metadata if isinstance(metadata, dict) else {}
        ns = tuple(part.get("ns", ()))
        source = namespace_label(ns, metadata)
        is_main_source = not ns
        events: list[AgentStreamEvent] = []

        reasoning_text = reasoning_text_from_token(token)
        if reasoning_text:
            event = self._reasoning_event(source, reasoning_text)
            if event is not None:
                events.append(event)

        tool_call_chunks = getattr(token, "tool_call_chunks", None) or []
        for chunk in tool_call_chunks:
            events.append(self._tool_call_event(source, chunk))

        if getattr(token, "type", None) == "tool":
            event = self._tool_result_event(source, token)
            return events + ([event] if event is not None else [])

        content_text = stringify_content(getattr(token, "content", ""))
        if is_main_source and content_text and not tool_call_chunks:
            self.response_streamed_from_messages = True
            event = self._response_event(source, content_text)
            if event is not None:
                events.append(event)
        return events

    def _events_from_update_chunk(self, part: dict[str, Any]) -> list[AgentStreamEvent]:
        ns = tuple(part.get("ns", ()))
        source = namespace_label(ns, {"lc_agent_name": None})
        data_by_node = part.get("data")
        if not isinstance(data_by_node, dict):
            return []

        events: list[AgentStreamEvent] = []
        for node_name, data in data_by_node.items():
            if node_name != "tools":
                if not ns and not self.response_streamed_from_messages:
                    assistant_messages = assistant_messages_for_current_prompt(
                        messages_from_node_data(data),
                        self.prompt,
                    )
                    if assistant_messages:
                        content_text = stringify_content(
                            getattr(assistant_messages[-1], "content", "")
                        )
                        if content_text:
                            event = self._response_event(source, content_text)
                            if event is not None:
                                events.append(event)
                continue

            for message in messages_from_node_data(data):
                if getattr(message, "type", None) == "tool":
                    event = self._tool_result_event(source, message)
                    if event is not None:
                        events.append(event)
        return events

    def _events_from_custom_chunk(self, part: dict[str, Any]) -> list[AgentStreamEvent]:
        data = part.get("data")
        if not isinstance(data, dict):
            return []
        if data.get("kind") != SUMMARIZATION_STATUS_KIND:
            return []

        return [
            AgentStreamEvent(
                kind="summarization_status",
                source=str(data.get("source") or "main-agent").strip() or "main-agent",
                status=str(data.get("status") or "triggered").strip() or "triggered",
                text=str(
                    data.get("message") or "Conversation summarization triggered."
                ).strip(),
            )
        ]

    def _response_event(self, source: str, text: str) -> AgentStreamEvent | None:
        delta = text[len(self.response_buffer) :] if text.startswith(self.response_buffer) else text
        if not delta:
            return None
        self.response_buffer += delta
        return AgentStreamEvent(kind="response_delta", source=source, text=delta)

    def _reasoning_event(self, source: str, text: str) -> AgentStreamEvent | None:
        previous = self.reasoning_buffers.get(source, "")
        delta = text[len(previous) :] if text.startswith(previous) else text
        if not delta:
            return None
        self.reasoning_buffers[source] = previous + delta
        return AgentStreamEvent(kind="reasoning_delta", source=source, text=delta)

    def _tool_call_event(self, source: str, chunk: dict[str, Any]) -> AgentStreamEvent:
        call_id = str(chunk.get("id") or f"{source}:{chunk.get('index', '0')}")
        tool_name = str(chunk.get("name") or self.tool_names.get(call_id) or "tool")
        self.tool_names[call_id] = tool_name

        args_delta = chunk.get("args")
        args_delta_text = ""
        if args_delta:
            args_delta_text = str(args_delta)
            self.tool_args_buffers[call_id] = (
                self.tool_args_buffers.get(call_id, "") + args_delta_text
            )

        status = "update" if call_id in self.tool_call_started else "start"
        self.tool_call_started.add(call_id)
        return AgentStreamEvent(
            kind="tool_call",
            source=source,
            tool_call_id=call_id,
            tool_name=tool_name,
            tool_args=self.tool_args_buffers.get(call_id, ""),
            tool_args_delta=args_delta_text,
            status=status,
        )

    def _tool_result_event(
        self,
        source: str,
        tool_message: Any,
    ) -> AgentStreamEvent | None:
        name = str(getattr(tool_message, "name", "") or "tool")
        status = str(getattr(tool_message, "status", "") or "done")
        content = stringify_content(getattr(tool_message, "content", ""))
        result_key = self._tool_result_key(
            source=source,
            name=name,
            tool_message=tool_message,
            content=content,
        )
        if result_key in self.completed_tool_results:
            return None
        self.completed_tool_results.add(result_key)

        return AgentStreamEvent(
            kind="tool_result",
            source=source,
            tool_call_id=str(
                getattr(tool_message, "tool_call_id", None)
                or getattr(tool_message, "id", None)
                or ""
            ),
            tool_name=name,
            tool_result=content,
            status=status,
        )

    @staticmethod
    def _tool_result_key(
        *,
        source: str,
        name: str,
        tool_message: Any,
        content: str,
    ) -> tuple[str, str, str]:
        stable_id = str(
            getattr(tool_message, "tool_call_id", None)
            or getattr(tool_message, "id", None)
            or ""
        ).strip()
        if stable_id:
            return (source, "id", stable_id)
        return (source, name, content)
