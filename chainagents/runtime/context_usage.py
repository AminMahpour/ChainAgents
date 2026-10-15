"""Context window usage snapshots, token estimation and a per-session registry."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

CATEGORY_NAMES: tuple[str, ...] = ("system", "tools", "summary", "messages", "tool_results")
MAX_AGENTS_PER_SESSION = 50

SESSION_ID_KEY = "chainagents_session_id"
AGENT_KIND_KEY = "chainagents_agent_kind"
AGENT_NAME_KEY = "chainagents_agent_name"
AGENT_INSTANCE_KEY = "chainagents_agent_instance"


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
            entries.popitem(last=False)

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
        """Set the status of a stored snapshot without changing its eviction order."""
        entries = self._snapshots.get(session_id)
        if entries is not None and key in entries:
            entries[key] = replace(entries[key], status=status)

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
