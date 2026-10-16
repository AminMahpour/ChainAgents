"""Context window usage snapshots, token estimation and a per-session registry."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.config import get_config, get_stream_writer

from chainagents.runtime.constants import SUMMARIZATION_STATUS_EVENT_KIND
from chainagents.runtime.summarization_budget import context_window, effective_threshold_tokens
from chainagents.runtime.types import ExtensionsConfig

logger = logging.getLogger(__name__)

CATEGORY_NAMES: tuple[str, ...] = ("system", "tools", "summary", "messages", "tool_results")
MAX_AGENTS_PER_SESSION = 50

SESSION_ID_KEY = "chainagents_session_id"
AGENT_KIND_KEY = "chainagents_agent_kind"
AGENT_NAME_KEY = "chainagents_agent_name"
AGENT_INSTANCE_KEY = "chainagents_agent_instance"

CONTEXT_USAGE_EVENT_KIND = "context_usage"
SUMMARY_NOTICE_MESSAGE = "Conversation summarized to fit the context window."


def is_summary_message(message: Any) -> bool:
    """Return True when the message was produced by conversation summarization."""
    kwargs = getattr(message, "additional_kwargs", None)
    return isinstance(kwargs, dict) and kwargs.get("lc_source") == "summarization"


def estimate_categories(
    system_message: Any | None, tools: Sequence[Any], messages: Sequence[Any]
) -> dict[str, int]:
    """Estimate token usage per category with the approximate token counter."""
    summary: list[Any] = []
    tool_results: list[Any] = []
    others: list[Any] = []
    for message in messages:
        if is_summary_message(message):
            summary.append(message)
        elif isinstance(message, ToolMessage):
            tool_results.append(message)
        else:
            others.append(message)
    return {
        "system": count_tokens_approximately([system_message]) if system_message else 0,
        "tools": count_tokens_approximately([], tools=list(tools)) if tools else 0,
        "summary": count_tokens_approximately(summary) if summary else 0,
        "messages": count_tokens_approximately(others) if others else 0,
        "tool_results": count_tokens_approximately(tool_results) if tool_results else 0,
    }


def scale_to_total(estimates: dict[str, int], total: int) -> dict[str, int]:
    """Make the categories sum to ``total``, with ``system`` as the remainder.

    Non-system categories keep their estimates unless together they exceed
    ``total``; then they are scaled down with largest-remainder rounding.
    """
    others = [name for name in CATEGORY_NAMES if name != "system"]
    values = {name: max(int(estimates.get(name, 0)), 0) for name in others}
    total = max(total, 0)
    others_sum = sum(values.values())
    if others_sum > total:
        floors = {name: values[name] * total // others_sum for name in others}
        remainders = {name: values[name] * total % others_sum for name in others}
        leftover = total - sum(floors.values())
        for name in sorted(others, key=lambda item: -remainders[item])[:leftover]:
            floors[name] += 1
        values = floors
    return {"system": total - sum(values.values()), **values}


@dataclass(frozen=True)
class ContextUsageSnapshot:
    """Context window usage of one agent at the time of its last model call."""

    agent: str
    kind: str
    model: str
    window: int | None
    used: int
    categories: dict[str, int]
    trigger: int | None
    keep: int | None
    exact: bool
    at_utc: str
    last_summarized_at_utc: str | None
    status: str

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe dict with the snapshot's fields."""
        payload = asdict(self)
        payload["categories"] = dict(self.categories)
        return payload


class ContextUsageRegistry:
    """Process-local store of the latest snapshot per session and agent.

    Not locked: all methods run on the event loop thread.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, OrderedDict[str, ContextUsageSnapshot]] = {}
        self._digests: dict[str, dict[str, tuple[str, str]]] = {}

    def record(self, session_id: str, key: str, snapshot: ContextUsageSnapshot) -> None:
        """Store a snapshot, evicting the least recently updated beyond the cap."""
        entries = self._snapshots.setdefault(session_id, OrderedDict())
        entries[key] = snapshot
        entries.move_to_end(key)
        while len(entries) > MAX_AGENTS_PER_SESSION:
            evicted, _ = entries.popitem(last=False)
            # Summary state is per agent too; drop it so the cap bounds memory.
            self._digests.get(session_id, {}).pop(evicted, None)

    def note_summary(self, session_id: str, key: str, digest: str) -> bool:
        """Record a summary digest; return True when it differs from the last one."""
        digests = self._digests.setdefault(session_id, {})
        known = digests.get(key)
        if known is not None and known[0] == digest:
            return False
        now = datetime.now(UTC).isoformat(timespec="seconds")
        digests[key] = (digest, now)
        return True

    def last_summarized(self, session_id: str, key: str) -> str | None:
        """Return when a new summary was last noted for the agent, if ever."""
        known = self._digests.get(session_id, {}).get(key)
        return known[1] if known else None

    def mark_ended(self, session_id: str, key: str, status: str) -> None:
        """End an agent instance and its nested children, keeping eviction order.

        Nested subagents of a background task or live agent are stored under
        ``<key>:<subagent>``; they cannot run once their owner has ended.
        """
        entries = self._snapshots.get(session_id)
        if entries is None:
            return
        for entry_key, snapshot in entries.items():
            if entry_key == key or entry_key.startswith(f"{key}:"):
                entries[entry_key] = replace(snapshot, status=status)

    def list(self, session_id: str) -> list[ContextUsageSnapshot]:
        """Return the session's snapshots: main first, then newest first."""
        snapshots = list(self._snapshots.get(session_id, {}).values())
        snapshots.sort(key=lambda item: item.at_utc, reverse=True)
        snapshots.sort(key=lambda item: item.kind != "main")
        return snapshots

    def clear(self, session_id: str) -> None:
        """Drop all snapshots and summary digests of the session."""
        self._snapshots.pop(session_id, None)
        self._digests.pop(session_id, None)


@dataclass(frozen=True)
class _CallIdentity:
    """Who is calling the model: session, registry key, label and kind."""

    session_id: str
    key: str
    agent: str
    kind: str


def _resolve_identity() -> _CallIdentity | None:
    """Resolve the calling agent from the LangGraph run config, if any."""
    try:
        config = get_config()
    except RuntimeError:
        return None
    configurable = config.get("configurable") or {}
    metadata = config.get("metadata") or {}
    session_id = configurable.get(SESSION_ID_KEY) or configurable.get("thread_id")
    if not session_id:
        return None
    subagent_name = metadata.get("lc_agent_name")
    run_agent_name = configurable.get(AGENT_NAME_KEY)
    if run_agent_name and subagent_name and subagent_name != run_agent_name:
        # LangGraph passes the run's identity keys on to nested `task` calls.
        instance = configurable.get(AGENT_INSTANCE_KEY) or run_agent_name
        return _CallIdentity(
            session_id=str(session_id),
            key=f"{instance}:{subagent_name}",
            agent=str(subagent_name),
            kind="subagent",
        )
    kind = configurable.get(AGENT_KIND_KEY) or ("subagent" if subagent_name else "main")
    agent = run_agent_name or ("main" if kind == "main" else subagent_name)
    agent = str(agent or kind)
    key = configurable.get(AGENT_INSTANCE_KEY) or f"{kind}:{agent}"
    return _CallIdentity(session_id=str(session_id), key=str(key), agent=agent, kind=str(kind))


def _model_name(model: Any) -> str:
    """Return the model's name, falling back to its class name."""
    # ChatBedrockConverse names its model in `model_id`.
    name = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "model_id", None)
    )
    return name if isinstance(name, str) and name else type(model).__name__


def _last_ai_message(response: Any) -> AIMessage | None:
    """Return the last AIMessage of a model call's response, if any."""
    if isinstance(response, AIMessage):
        return response
    response = getattr(response, "model_response", response)
    for message in reversed(getattr(response, "result", None) or []):
        if isinstance(message, AIMessage):
            return message
    return None


def _input_tokens(response: Any) -> int | None:
    """Return the server-reported input tokens when they are a positive int."""
    message = _last_ai_message(response)
    usage = getattr(message, "usage_metadata", None) if message is not None else None
    value = usage.get("input_tokens") if isinstance(usage, dict) else None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _summary_digest(messages: Sequence[Any]) -> str | None:
    """Return the SHA-256 of the leading summary message's text, if it has one."""
    if not messages or not is_summary_message(messages[0]):
        return None
    content = messages[0].content
    if isinstance(content, list):
        parts = [
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
            if isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")
        ]
        content = "".join(parts)
    return hashlib.sha256(str(content).encode("utf-8")).hexdigest()


class ContextUsageMiddleware(AgentMiddleware[Any, Any, Any]):
    """Measure each model call's context usage and publish it per agent."""

    def __init__(self, *, registry: ContextUsageRegistry, extensions: ExtensionsConfig) -> None:
        """Initialize the middleware with the registry and summarization settings."""
        super().__init__()
        self.registry = registry
        self.extensions = extensions

    @property
    def name(self) -> str:
        """Return the middleware name exposed to LangGraph."""
        return "ChainAgentsContextUsage"

    def wrap_model_call(self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], Any]) -> Any:
        """Measure a synchronous model call; the handler's result passes through."""
        estimates = self._estimate(request)
        response = handler(request)
        self._observe(request, response, estimates)
        return response

    async def awrap_model_call(
        self, request: ModelRequest[Any], handler: Callable[[ModelRequest[Any]], Awaitable[Any]]
    ) -> Any:
        """Measure an asynchronous model call; the handler's result passes through."""
        estimates = self._estimate(request)
        response = await handler(request)
        self._observe(request, response, estimates)
        return response

    def _estimate(self, request: ModelRequest[Any]) -> dict[str, int] | None:
        """Estimate category tokens of the request, or None when that fails."""
        try:
            return estimate_categories(request.system_message, request.tools or [], request.messages)
        except Exception:
            logger.debug("Failed to estimate context usage", exc_info=True)
            return None

    def _observe(self, request: ModelRequest[Any], response: Any, estimates: dict[str, int] | None) -> None:
        """Record the call's snapshot and emit stream events; never raises."""
        if estimates is None:
            return
        try:
            identity = _resolve_identity()
            if identity is None:
                return
            total = _input_tokens(response)
            categories = scale_to_total(estimates, total) if total is not None else dict(estimates)
            digest = _summary_digest(request.messages)
            summarized = digest is not None and self.registry.note_summary(
                identity.session_id, identity.key, digest
            )
            trigger, keep = effective_threshold_tokens(request.model, self.extensions)
            snapshot = ContextUsageSnapshot(
                agent=identity.agent,
                kind=identity.kind,
                model=_model_name(request.model),
                window=context_window(request.model),
                used=total if total is not None else sum(categories.values()),
                categories=categories,
                trigger=trigger,
                keep=keep,
                exact=total is not None,
                at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
                last_summarized_at_utc=self.registry.last_summarized(identity.session_id, identity.key),
                status="active",
            )
            self.registry.record(identity.session_id, identity.key, snapshot)
            if identity.kind != "main":
                return
            writer = get_stream_writer()
            writer({"kind": CONTEXT_USAGE_EVENT_KIND, "snapshot": snapshot.to_payload()})
            if summarized:
                writer(
                    {
                        "kind": SUMMARIZATION_STATUS_EVENT_KIND,
                        "status": "completed",
                        "source": identity.agent,
                        "message": SUMMARY_NOTICE_MESSAGE,
                    }
                )
        except Exception:
            logger.debug("Failed to record context usage", exc_info=True)
