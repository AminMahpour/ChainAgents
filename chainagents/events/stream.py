"""Normalize LangGraph stream chunks for CLI, Chainlit, and TUI renderers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


StreamEventKind = Literal[
    "response_delta",
    "reasoning_delta",
    "tool_call",
    "tool_result",
    "summarization_status",
    "ui_message",
    "ui_remove",
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
ANTHROPIC_THINKING_BLOCK_TYPES = {"thinking", "redacted_thinking"}


@dataclass(frozen=True)
class AgentStreamEvent:
    """A renderer-neutral event produced from LangGraph stream chunks."""

    kind: StreamEventKind
    source: str
    text: str = ""
    tool_call_id: str = ""
    previous_tool_call_id: str = ""
    tool_name: str = ""
    tool_args: str = ""
    tool_args_delta: str = ""
    tool_result: str = ""
    status: str = ""
    ui_id: str = ""
    ui_name: str = ""
    ui_props: dict[str, Any] = field(default_factory=dict)
    ui_metadata: dict[str, Any] = field(default_factory=dict)


def stringify_content(value: Any) -> str:
    """Convert LangChain message content into displayable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(stringify_content(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") in ANTHROPIC_THINKING_BLOCK_TYPES:
            return ""
        for key in ("text", "reasoning", "content"):
            nested = value.get(key)
            if isinstance(nested, (str, list, dict)):
                return stringify_content(nested)
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    return str(value)


def anthropic_thinking_text(value: Any) -> str:
    """Extract Claude thinking text from LangChain Anthropic content blocks."""
    if isinstance(value, list):
        return "".join(anthropic_thinking_text(item) for item in value)
    if not isinstance(value, dict) or value.get("type") != "thinking":
        return ""
    return stringify_content(value.get("thinking"))


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
    if hasattr(token, "content"):
        return anthropic_thinking_text(token.content)
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


def assistant_messages_for_current_prompt(
    messages: list[Any], prompt: str
) -> list[Any]:
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
        self.tool_call_ids_by_index: dict[tuple[str, str], str] = {}
        self.previous_tool_call_ids: dict[str, str] = {}
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

    def _events_from_message_chunk(
        self, part: dict[str, Any]
    ) -> list[AgentStreamEvent]:
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
        if data.get("type") == "ui":
            return self._events_from_ui_message(data)
        if data.get("type") == "remove-ui":
            return self._events_from_ui_remove(data)
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

    @staticmethod
    def _events_from_ui_message(data: dict[str, Any]) -> list[AgentStreamEvent]:
        ui_id = str(data.get("id") or "").strip()
        ui_name = str(data.get("name") or "").strip()
        props = data.get("props")
        metadata = data.get("metadata")
        if not ui_id or not ui_name or not isinstance(props, dict):
            return []
        if not isinstance(metadata, dict):
            metadata = {}

        source = str(
            metadata.get("source") or metadata.get("name") or "main-agent"
        ).strip() or "main-agent"
        return [
            AgentStreamEvent(
                kind="ui_message",
                source=source,
                ui_id=ui_id,
                ui_name=ui_name,
                ui_props=props,
                ui_metadata=metadata,
            )
        ]

    @staticmethod
    def _events_from_ui_remove(data: dict[str, Any]) -> list[AgentStreamEvent]:
        ui_id = str(data.get("id") or "").strip()
        if not ui_id:
            return []
        return [
            AgentStreamEvent(
                kind="ui_remove",
                source="main-agent",
                ui_id=ui_id,
            )
        ]

    def _response_event(self, source: str, text: str) -> AgentStreamEvent | None:
        delta = (
            text[len(self.response_buffer) :]
            if text.startswith(self.response_buffer)
            else text
        )
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
        call_id = self._tool_call_id(source, chunk)
        previous_call_id = self.previous_tool_call_ids.pop(call_id, "")
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
            previous_tool_call_id=previous_call_id,
            tool_name=tool_name,
            tool_args=self.tool_args_buffers.get(call_id, ""),
            tool_args_delta=args_delta_text,
            status=status,
        )

    def _tool_call_id(self, source: str, chunk: dict[str, Any]) -> str:
        """Return a stable call id for streamed tool chunks."""
        raw_index = chunk.get("index")
        raw_id = chunk.get("id")
        if raw_id:
            call_id = str(raw_id)
            if raw_index is None:
                return call_id
            index_key = (source, str(raw_index))
            existing_id = self.tool_call_ids_by_index.get(index_key)
            if existing_id and existing_id != call_id:
                self.previous_tool_call_ids[call_id] = existing_id
                self._merge_tool_call_state(existing_id, call_id)
            self.tool_call_ids_by_index[index_key] = call_id
            return call_id

        index = str(raw_index if raw_index is not None else "0")
        index_key = (source, index)
        existing_id = self.tool_call_ids_by_index.get(index_key)
        if existing_id:
            return existing_id

        call_id = f"{source}:{index}"
        self.tool_call_ids_by_index[index_key] = call_id
        return call_id

    def _merge_tool_call_state(self, old_id: str, new_id: str) -> None:
        """Move buffered state from a synthetic chunk id onto the real call id."""
        old_name = self.tool_names.pop(old_id, "")
        if old_name and new_id not in self.tool_names:
            self.tool_names[new_id] = old_name

        old_args = self.tool_args_buffers.pop(old_id, "")
        if old_args:
            new_args = self.tool_args_buffers.get(new_id, "")
            if not new_args:
                self.tool_args_buffers[new_id] = old_args
            elif not new_args.startswith(old_args):
                self.tool_args_buffers[new_id] = old_args + new_args

        if old_id in self.tool_call_started:
            self.tool_call_started.remove(old_id)

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

        event = AgentStreamEvent(
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
        self._clear_tool_call_state(event.tool_call_id)
        return event

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

    def _clear_tool_call_state(self, call_id: str) -> None:
        """Clear streamed tool-call buffers after the matching result arrives."""
        if not call_id:
            return

        self.tool_names.pop(call_id, None)
        self.tool_args_buffers.pop(call_id, None)
        self.previous_tool_call_ids.pop(call_id, None)
        self.tool_call_started.discard(call_id)
        for index_key, mapped_call_id in list(self.tool_call_ids_by_index.items()):
            if mapped_call_id == call_id:
                self.tool_call_ids_by_index.pop(index_key, None)
