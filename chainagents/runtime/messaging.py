"""Bound, process-local mailboxes for opted-in agent invocations."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from contextlib import suppress

from chainagents.runtime.types import MessagingConfig
from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from collections.abc import AsyncIterator, Iterable
from typing import Any

from chainagents.runtime.background_tasks.context import current_background_session_id

_CURRENT_AGENT_ADDRESS: contextvars.ContextVar[str] = contextvars.ContextVar(
    "chainagents_messaging_address", default="main"
)
_CURRENT_MESSAGING_SESSION: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "chainagents_messaging_session", default=None
)

AGENT_MESSAGE_TOOL_NAMES = frozenset({
    "list_agent_recipients",
    "send_agent_message",
    "get_agent_message",
    "wait_for_agent_messages",
})


def validate_agent_message_tool_names(existing_tools: Iterable[object]) -> None:
    """Reject configured tools that would be shadowed by messaging actions."""
    collisions = sorted({
        name
        for candidate in existing_tools
        if (name := str(
            getattr(candidate, "name", None) or getattr(candidate, "__name__", "")
        ).strip()) in AGENT_MESSAGE_TOOL_NAMES
    })
    if collisions:
        raise ValueError(
            "Configured tools use reserved agent messaging tool names: "
            f"{', '.join(collisions)}. Rename or prefix the configured tools."
        )


@dataclass
class AgentMessage:
    id: str
    session_id: str
    sender: str
    recipient: str
    body: str
    created_at: str
    status: str = "pending"


@dataclass(frozen=True)
class AgentRecipient:
    address: str
    name: str
    parent: str | None


class MessageBroker:
    """Keep messages and participant lifetimes isolated by conversation."""

    def __init__(self, config: MessagingConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._participants: dict[str, dict[str, AgentRecipient]] = {}
        self._messages: dict[str, dict[str, AgentMessage]] = {}
        self._events: dict[
            tuple[str, str], list[tuple[asyncio.Event, asyncio.AbstractEventLoop]]
        ] = {}

    def open(
        self, session_id: str, address: str, *, name: str, parent: str | None
    ) -> None:
        with self._lock:
            self._participants.setdefault(session_id, {})[address] = AgentRecipient(
                address, name, parent
            )

    def close(self, session_id: str, address: str) -> None:
        with self._lock:
            self._participants.get(session_id, {}).pop(address, None)
            for message in self._messages.get(session_id, {}).values():
                if message.recipient == address and message.status in {
                    "pending",
                    "inflight",
                }:
                    message.status = "undeliverable"
            for event, loop in self._events.pop((session_id, address), []):
                if not loop.is_closed():
                    loop.call_soon_threadsafe(event.set)

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._participants.pop(session_id, None)
            self._messages.pop(session_id, None)
            for key in tuple(self._events):
                if key[0] == session_id:
                    for event, loop in self._events.pop(key, []):
                        if not loop.is_closed():
                            loop.call_soon_threadsafe(event.set)

    def recipients(self, session_id: str, sender: str) -> list[AgentRecipient]:
        with self._lock:
            if sender not in self._participants.get(session_id, {}):
                raise ValueError("Sender is unavailable in this conversation.")
            return [
                recipient
                for address, recipient in self._participants[session_id].items()
                if address != sender
            ]

    def resolve(self, session_id: str, sender: str, recipient: str) -> str:
        with self._lock:
            participants = self._participants.get(session_id, {})
            if sender not in participants:
                raise ValueError("Sender is unavailable in this conversation.")
            if recipient == "parent":
                recipient = participants[sender].parent or ""
            if recipient in participants and recipient != sender:
                return recipient
            matches = [
                item.address
                for item in participants.values()
                if item.name == recipient and item.address != sender
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    "Recipient name is ambiguous; use its invocation address."
                )
            raise ValueError("Recipient is unavailable in this conversation.")

    def send(
        self, session_id: str, sender: str, recipient: str, body: str
    ) -> AgentMessage:
        if not self.config.enabled:
            raise ValueError("Agent messaging is disabled.")
        if not isinstance(body, str) or not body.strip():
            raise ValueError("Message body must contain text.")
        if len(body) > self.config.max_message_chars:
            raise ValueError("Message exceeds max_message_chars.")
        with self._lock:
            address = self.resolve(session_id, sender, recipient)
            messages = self._messages.setdefault(session_id, {})
            if len(messages) >= self.config.max_messages_per_session:
                raise ValueError("Conversation message limit reached.")
            pending = sum(
                item.recipient == address and item.status in {"pending", "inflight"}
                for item in messages.values()
            )
            if pending >= self.config.max_pending_per_recipient:
                raise ValueError("Recipient pending limit reached.")
            message = AgentMessage(
                id=uuid.uuid4().hex,
                session_id=session_id,
                sender=sender,
                recipient=address,
                body=body,
                created_at=datetime.now(UTC).isoformat(),
            )
            messages[message.id] = message
            for event, loop in self._events.get((session_id, address), []):
                if not loop.is_closed():
                    loop.call_soon_threadsafe(event.set)
            return message

    def get(self, session_id: str, message_id: str) -> AgentMessage:
        with self._lock:
            message = self._messages.get(session_id, {}).get(message_id)
            if message is None:
                raise ValueError("Message is unavailable in this conversation.")
            return message

    def pending(self, session_id: str, address: str) -> list[AgentMessage]:
        with self._lock:
            if address not in self._participants.get(session_id, {}):
                raise ValueError("Recipient is unavailable in this conversation.")
            return [
                item
                for item in self._messages.get(session_id, {}).values()
                if item.recipient == address and item.status == "pending"
            ]

    def deliver(self, session_id: str, address: str) -> list[AgentMessage]:
        with self._lock:
            messages = self.pending(session_id, address)[
                : self.config.max_deliveries_per_step
            ]
            for message in messages:
                message.status = "delivered"
            if not self.pending(session_id, address):
                for event, loop in self._events.get((session_id, address), []):
                    if not loop.is_closed():
                        loop.call_soon_threadsafe(event.clear)
            return messages

    def claim(self, session_id: str, address: str) -> list[AgentMessage]:
        """Reserve input for a model step, retaining it if the step is cancelled."""
        with self._lock:
            self.recover(session_id, address)
            messages = self.pending(session_id, address)[
                : self.config.max_deliveries_per_step
            ]
            for message in messages:
                message.status = "inflight"
            return messages

    def acknowledge(self, session_id: str, address: str) -> None:
        with self._lock:
            for message in self._messages.get(session_id, {}).values():
                if message.recipient == address and message.status == "inflight":
                    message.status = "delivered"

    def recover(self, session_id: str, address: str) -> None:
        with self._lock:
            for message in self._messages.get(session_id, {}).values():
                if message.recipient == address and message.status == "inflight":
                    message.status = "pending"

    def take_late_user_steering(self, session_id: str) -> list[AgentMessage]:
        """Move only unconsumed user steering into a follow-up turn."""
        with self._lock:
            selected = [
                message
                for message in self._messages.get(session_id, {}).values()
                if message.recipient == "main"
                and message.sender == "user"
                and message.status == "pending"
            ]
            for message in selected:
                message.status = "delivered"
            return selected

    def pending_user_steering_count(self, session_id: str) -> int:
        with self._lock:
            return sum(
                message.recipient == "main"
                and message.sender == "user"
                and message.status in {"pending", "inflight"}
                for message in self._messages.get(session_id, {}).values()
            )

    async def wait(
        self, session_id: str, address: str, wait_seconds: float
    ) -> list[AgentMessage]:
        with self._lock:
            pending = self.pending(session_id, address)
            if pending or wait_seconds <= 0:
                return pending
            key = (session_id, address)
            event = asyncio.Event()
            self._events.setdefault(key, []).append((event, asyncio.get_running_loop()))
        try:
            with suppress(TimeoutError):
                await asyncio.wait_for(event.wait(), wait_seconds)
            return self.pending(session_id, address)
        finally:
            with self._lock:
                waiters = self._events.get(key, [])
                waiters[:] = [entry for entry in waiters if entry[0] is not event]
                if not waiters:
                    self._events.pop(key, None)

    def send_user(self, session_id: str, body: str) -> AgentMessage:
        """Add a user steering note to the main agent's next model step."""
        with self._lock:
            self.open(session_id, "user", name="user", parent=None)
            try:
                return self.send(session_id, "user", "main", body)
            finally:
                self._participants[session_id].pop("user", None)


def _session_id(runtime: Any, fixed_session_id: str | None = None) -> str:
    configured = getattr(runtime, "config", {}) or {}
    session = (
        _CURRENT_MESSAGING_SESSION.get()
        or current_background_session_id()
        or fixed_session_id
        or configured.get("configurable", {}).get("thread_id")
    )
    if not session:
        raise ValueError("Agent messaging requires a conversation identity.")
    return str(session)


def create_agent_message_tools(
    broker: MessageBroker, *, fixed_session_id: str | None = None
) -> list[Any]:
    """Expose participant-scoped mailbox actions to DeepAgents."""

    @tool
    def list_agent_recipients(runtime: ToolRuntime) -> list[dict[str, str | None]]:
        """List live agents that can receive a message in this conversation."""
        session = _session_id(runtime, fixed_session_id)
        return [
            vars(item)
            for item in broker.recipients(session, _CURRENT_AGENT_ADDRESS.get())
        ]

    @tool
    def send_agent_message(
        recipient: str, message: str, runtime: ToolRuntime
    ) -> dict[str, str]:
        """Send text to a live agent; delivery occurs at its next model step."""
        session = _session_id(runtime, fixed_session_id)
        sent = broker.send(session, _CURRENT_AGENT_ADDRESS.get(), recipient, message)
        return {"id": sent.id, "status": sent.status, "recipient": sent.recipient}

    @tool
    def get_agent_message(message_id: str, runtime: ToolRuntime) -> dict[str, str]:
        """Get a message's delivery status and text in this conversation."""
        session = _session_id(runtime, fixed_session_id)
        item = broker.get(session, message_id)
        actor = _CURRENT_AGENT_ADDRESS.get()
        if actor not in (item.sender, item.recipient):
            raise ValueError("Message is unavailable to this agent.")
        return vars(item)

    @tool
    async def wait_for_agent_messages(
        timeout_seconds: float, runtime: ToolRuntime
    ) -> list[dict[str, str]]:
        """Wait briefly for new messages without starting another agent turn."""
        if not 0 <= timeout_seconds <= 30:
            raise ValueError("timeout_seconds must be between 0 and 30.")
        session = _session_id(runtime, fixed_session_id)
        address = _CURRENT_AGENT_ADDRESS.get()
        await broker.wait(session, address, timeout_seconds)
        items = broker.deliver(session, address)
        return [vars(item) for item in items]

    return [
        list_agent_recipients,
        send_agent_message,
        get_agent_message,
        wait_for_agent_messages,
    ]


class AgentMessageMiddleware(AgentMiddleware[Any, Any, Any]):
    """Insert pending messages into the next model input for one participant."""

    def __init__(self, broker: MessageBroker, fixed_session_id: str | None = None):
        self.broker = broker
        self.fixed_session_id = fixed_session_id

    def _before(self, runtime: Any) -> dict[str, Any] | None:
        session = _session_id(runtime, self.fixed_session_id)
        address = _CURRENT_AGENT_ADDRESS.get()
        if address == "main":
            self.broker.open(session, "main", name="main", parent=None)
        messages = self.broker.claim(session, address)
        if not messages:
            return None
        return {
            "messages": [
                HumanMessage(
                    content=f"Message from {item.sender} (id {item.id}):\n{item.body}",
                    id=f"agent-message-{item.id}",
                )
                for item in messages
            ]
        }

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._before(runtime)

    async def abefore_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        return self._before(runtime)

    def after_model(self, state: Any, runtime: Any) -> None:
        self.broker.acknowledge(
            _session_id(runtime, self.fixed_session_id), _CURRENT_AGENT_ADDRESS.get()
        )

    async def aafter_model(self, state: Any, runtime: Any) -> None:
        self.after_model(state, runtime)


class MessagingScopedRunnable(Runnable[Any, Any]):
    """Register one configured subagent invocation for its whole lifetime."""

    def __init__(
        self,
        runnable: Any,
        broker: MessageBroker,
        name: str,
        fixed_session_id: str | None = None,
        *,
        participant: bool = True,
    ) -> None:
        self.runnable = runnable
        self.broker = broker
        self.agent_name = name
        self.fixed_session_id = fixed_session_id
        self.participant = participant

    def __getattr__(self, name: str) -> Any:
        return getattr(self.runnable, name)

    def _enter(self, config: RunnableConfig | None) -> tuple[str, str, Any, Any]:
        session = _session_id(
            type("Run", (), {"config": config or {}})(), self.fixed_session_id
        )
        parent = _CURRENT_AGENT_ADDRESS.get() or None
        address = f"{self.agent_name}:{uuid.uuid4().hex[:12]}"
        self.broker.open(session, "main", name="main", parent=None)
        if self.participant:
            self.broker.open(session, address, name=self.agent_name, parent=parent)
        return (
            session,
            address,
            _CURRENT_AGENT_ADDRESS.set(address if self.participant else ""),
            _CURRENT_MESSAGING_SESSION.set(session),
        )

    def _leave(self, scope: tuple[str, str, Any, Any]) -> None:
        session, address, address_token, session_token = scope
        if self.participant:
            self.broker.close(session, address)
        _CURRENT_AGENT_ADDRESS.reset(address_token)
        _CURRENT_MESSAGING_SESSION.reset(session_token)

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        scope = self._enter(config)
        try:
            return self.runnable.invoke(input, config, **kwargs)
        finally:
            self._leave(scope)

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        scope = self._enter(config)
        try:
            return await self.runnable.ainvoke(input, config, **kwargs)
        finally:
            self._leave(scope)

    async def astream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        scope = self._enter(config)
        try:
            async for chunk in self.runnable.astream(input, config, **kwargs):
                yield chunk
        finally:
            self._leave(scope)
