"""Mailbox worker that drives one live agent's turns from its broker inbox."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from chainagents.runtime.context_usage import (
    AGENT_INSTANCE_KEY,
    AGENT_KIND_KEY,
    AGENT_NAME_KEY,
    SESSION_ID_KEY,
)
from chainagents.runtime.live_agents.events import LiveAgentEvent, LiveAgentEventBus
from chainagents.runtime.live_agents.registry import LiveAgentRecord
from chainagents.runtime.messaging import (
    _CURRENT_AGENT_ADDRESS,
    _CURRENT_MESSAGING_SESSION,
    AgentMessage,
    MessageBroker,
)

logger = logging.getLogger(__name__)

NO_REPLY_TEXT = "(no reply)"
MAX_CONSECUTIVE_FAILURES = 3
_WAIT_SECONDS = 30
_USER_SENDER = "user"


def _final_text(result: Any) -> str:
    """Return the text of the last AIMessage in a graph result, if any."""
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        content = message.content
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content or []:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


class LiveAgentWorker:
    """Process one inbox batch per turn on the live agent's own thread."""

    def __init__(
        self,
        record: LiveAgentRecord,
        graph: Any,
        broker: MessageBroker,
        events: LiveAgentEventBus,
        *,
        recursion_limit: int,
        max_turns: int,
        clock: Callable[[], float],
        wall_clock: Callable[[], float] = time.time,
        turn_context: Callable[[LiveAgentRecord], AbstractContextManager[Any]]
        | None = None,
        on_finished: Callable[[LiveAgentRecord, str], Awaitable[None]],
        artifact_registry: Any | None = None,
        artifact_handle: Any | None = None,
        turn_callbacks: Callable[[LiveAgentRecord], list[Any]] | None = None,
    ) -> None:
        self.record = record
        self.graph = graph
        self.broker = broker
        self.events = events
        self.recursion_limit = recursion_limit
        self.max_turns = max_turns
        self.clock = clock
        self.wall_clock = wall_clock
        self.turn_context = turn_context
        self.on_finished = on_finished
        # The worker runs in a fresh context (see LiveAgentManager.spawn), so the
        # conversation's artifact scope and tracing callbacks are reapplied here.
        self.artifact_registry = artifact_registry
        self.artifact_handle = artifact_handle
        self.turn_callbacks = turn_callbacks

    async def run(self) -> None:
        record = self.record
        session, address = record.session_id, record.address
        try:
            while True:
                try:
                    await self.broker.wait(session, address, _WAIT_SECONDS)
                    batch = self.broker.claim(session, address)
                except ValueError:
                    # The address was closed underneath us; nothing left to serve.
                    logger.debug("Live agent %s inbox closed; worker exiting", record.id)
                    return
                if not batch:
                    continue
                self._publish_inbound(batch)
                reason = await self._process(batch)
                if reason is not None:
                    await self.on_finished(record, reason)
                    return
        except asyncio.CancelledError:
            self.broker.recover(session, address)
            raise

    async def _process(self, batch: list[AgentMessage]) -> str | None:
        """Run one turn for a claimed batch; return a finish reason, if any."""
        record = self.record
        record.status = "running"
        record.mark_active(self.clock(), self.wall_clock())
        self._publish("turn_started", {})
        try:
            text = await self.run_turn(batch)
            ok = True
        except Exception as exc:  # reported to senders
            logger.warning("Live agent %s turn failed", record.id, exc_info=True)
            text = f"[error] {type(exc).__name__}: {exc}"
            ok = False
        self.broker.acknowledge(record.session_id, record.address)
        record.turns += 1
        record.consecutive_failures = 0 if ok else record.consecutive_failures + 1
        self._reply(batch, text)
        record.status = "idle"
        record.mark_active(self.clock(), self.wall_clock())
        self._publish("turn_finished", {"turns": record.turns, "ok": ok})
        if record.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            record.status = "failed"
            return "failed"
        if record.turns >= self.max_turns:
            return "max_turns"
        return None

    async def run_turn(self, messages: list[AgentMessage]) -> str:
        record = self.record
        address_token = _CURRENT_AGENT_ADDRESS.set(record.address)
        session_token = _CURRENT_MESSAGING_SESSION.set(record.session_id)
        artifacts = self.artifact_registry
        artifact_token = (
            artifacts.activate(self.artifact_handle)
            if artifacts is not None and self.artifact_handle is not None
            else None
        )
        try:
            context = (
                self.turn_context(record) if self.turn_context else nullcontext()
            )
            with context:
                result = await self.graph.ainvoke(
                    {
                        "messages": [
                            HumanMessage(
                                content=(
                                    f"Message from {m.sender} (id {m.id}):\n{m.body}"
                                )
                            )
                            for m in messages
                        ]
                    },
                    self._run_config(),
                )
        finally:
            if artifacts is not None and artifact_token is not None:
                artifacts.reset(artifact_token)
            _CURRENT_MESSAGING_SESSION.reset(session_token)
            _CURRENT_AGENT_ADDRESS.reset(address_token)
        text = _final_text(result)
        return text if text.strip() else NO_REPLY_TEXT

    def _run_config(self) -> dict[str, Any]:
        record = self.record
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": record.thread_id,
                SESSION_ID_KEY: record.session_id,
                AGENT_KIND_KEY: "live",
                AGENT_NAME_KEY: record.name,
                AGENT_INSTANCE_KEY: record.id,
            },
            "recursion_limit": self.recursion_limit,
        }
        callbacks = self.turn_callbacks(record) if self.turn_callbacks else []
        if callbacks:
            config["callbacks"] = callbacks
            config["metadata"] = {"langfuse_session_id": record.session_id}
        return config

    def _reply(self, batch: list[AgentMessage], text: str) -> None:
        record = self.record
        body = text[: self.broker.config.max_message_chars]
        senders = list(dict.fromkeys(m.sender for m in batch))
        for sender in senders:
            if sender == _USER_SENDER:
                # Users only see replies as events; keep the latest for polling.
                record.set_last_reply(body, self.clock(), self.wall_clock())
            else:
                try:
                    self.broker.send(record.session_id, record.address, sender, body)
                except ValueError:
                    logger.warning(
                        "Live agent %s could not reply to %s",
                        record.id,
                        sender,
                        exc_info=True,
                    )
                    continue
            self._publish("message", {"from": record.name, "to": sender, "body": body})

    def _publish_inbound(self, batch: list[AgentMessage]) -> None:
        """Publish claimed messages so observers see every inbound message.

        User messages are skipped: ``LiveAgentManager`` already publishes them
        when they are sent.
        """
        for message in batch:
            if message.sender == _USER_SENDER:
                continue
            self._publish(
                "message",
                {"from": message.sender, "to": self.record.name, "body": message.body},
            )

    def _publish(self, kind: Any, payload: dict[str, object]) -> None:
        record = self.record
        self.events.publish(
            LiveAgentEvent(
                kind=kind,
                session_id=record.session_id,
                agent_id=record.id,
                agent_name=record.name,
                payload=payload,
            )
        )
