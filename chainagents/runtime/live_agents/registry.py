"""In-memory registry of live agent records."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from chainagents.runtime.live_agents.config import ResolvedLiveAgentSpec
from chainagents.runtime.types import LiveAgentsConfig

LiveAgentStatus = Literal["idle", "running", "stopping", "stopped", "failed"]
TERMINAL_STATUSES: frozenset[str] = frozenset({"stopped", "failed"})
# Stopped/failed records kept per session for listings (the panel shows them);
# older ones are evicted so repeated spawn/stop cannot grow memory unbounded.
MAX_RETAINED_TERMINAL_RECORDS = 50


@dataclass
class LiveAgentRecord:
    """Runtime state of one live agent."""

    id: str
    session_id: str
    name: str
    template: str | None
    model: str | None
    address: str
    thread_id: str
    status: LiveAgentStatus
    created_at: float
    last_active_at: float
    spawned_by: str
    turns: int = 0
    consecutive_failures: int = 0
    stop_reason: str | None = None
    parent_trace: Any = None
    # Latest reply to the user (``{"body": str, "at": float}``), for API polling.
    last_reply: dict[str, object] | None = None

    def to_payload(
        self, *, pending: int = 0, wall_offset: float | None = None
    ) -> dict[str, object]:
        """Return the record for clients.

        Times are on the manager's monotonic clock. With ``wall_offset`` (wall
        clock minus monotonic clock, see ``LiveAgentManager.wall_offset``) the
        payload also carries ISO 8601 UTC ``*_utc`` fields clients can display.
        """
        payload: dict[str, object] = {
            "id": self.id,
            "name": self.name,
            "template": self.template,
            "model": self.model,
            "status": self.status,
            "turns": self.turns,
            "created_at": self.created_at,
            "last_active_at": self.last_active_at,
            "pending": pending,
            "stop_reason": self.stop_reason,
            "last_reply": dict(self.last_reply) if self.last_reply else None,
        }
        if wall_offset is not None:
            payload["created_at_utc"] = _utc_iso(self.created_at + wall_offset)
            payload["last_active_at_utc"] = _utc_iso(self.last_active_at + wall_offset)
            if self.last_reply:
                reply = dict(self.last_reply)
                reply["at_utc"] = _utc_iso(float(reply["at"]) + wall_offset)  # type: ignore[arg-type]
                payload["last_reply"] = reply
        return payload


def _utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="seconds")


class LiveAgentLimitError(ValueError):
    """Raised when a spawn would exceed a configured live agent limit."""


class LiveAgentRegistry:
    """Track live agent records per session."""

    def __init__(
        self,
        config: LiveAgentsConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._records: dict[str, list[LiveAgentRecord]] = {}
        self._sequence: dict[str, int] = {}

    def _active(self, session_id: str) -> list[LiveAgentRecord]:
        return [
            r
            for r in self._records.get(session_id, [])
            if r.status not in TERMINAL_STATUSES
        ]

    def reserve(
        self, session_id: str, spec: ResolvedLiveAgentSpec, *, spawned_by: str
    ) -> LiveAgentRecord:
        if len(self._active(session_id)) >= self.config.max_live_per_session:
            raise LiveAgentLimitError(
                "Live agent limit reached for this session "
                f"({self.config.max_live_per_session})."
            )
        total = sum(len(self._active(sid)) for sid in self._records)
        if total >= self.config.max_live_total:
            raise LiveAgentLimitError(
                f"Total live agent limit reached ({self.config.max_live_total})."
            )
        agent_id = uuid.uuid4().hex[:12]
        now = self._clock()
        record = LiveAgentRecord(
            id=agent_id,
            session_id=session_id,
            name=spec.name,
            template=spec.template,
            model=spec.model,
            address=f"live:{spec.name}:{agent_id}",
            thread_id=f"live-{session_id}-{agent_id}",
            status="idle",
            created_at=now,
            last_active_at=now,
            spawned_by=spawned_by,
        )
        self._records.setdefault(session_id, []).append(record)
        self._sequence[session_id] = self._sequence.get(session_id, 0) + 1
        return record

    def discard(self, record: LiveAgentRecord) -> None:
        records = self._records.get(record.session_id, [])
        if record in records:
            records.remove(record)

    def evict_terminal(self, session_id: str) -> list[LiveAgentRecord]:
        """Drop the oldest terminal records beyond ``MAX_RETAINED_TERMINAL_RECORDS``.

        Age is the time the record stopped (``last_active_at``), then spawn
        order. Live records are never evicted. Evicted records disappear from
        listings and ``lookup`` by id raises ``KeyError`` for them.

        Returns:
            The evicted records.
        """
        records = self._records.get(session_id, [])
        terminal = [r for r in records if r.status in TERMINAL_STATUSES]
        excess = len(terminal) - MAX_RETAINED_TERMINAL_RECORDS
        if excess <= 0:
            return []
        terminal.sort(key=lambda r: r.last_active_at)  # stable: ties keep spawn order
        evicted = terminal[:excess]
        dropped = {id(r) for r in evicted}
        self._records[session_id] = [r for r in records if id(r) not in dropped]
        return evicted

    def lookup(self, session_id: str, ref: str) -> LiveAgentRecord:
        """Find a live record by name, else any retained record by id.

        Raises:
            KeyError: No match, including terminal records already evicted by
                ``evict_terminal``.
        """
        records = self._records.get(session_id, [])
        for record in records:
            if record.status not in TERMINAL_STATUSES and record.name == ref:
                return record
        for record in records:
            if record.id == ref:
                return record
        raise KeyError(ref)

    def records(
        self, session_id: str, *, include_stopped: bool = False
    ) -> list[LiveAgentRecord]:
        records = self._records.get(session_id, [])
        if include_stopped:
            return list(records)
        return self._active(session_id)

    def live_names(self, session_id: str) -> set[str]:
        return {r.name for r in self._active(session_id)}

    def next_sequence(self, session_id: str) -> int:
        return self._sequence.get(session_id, 0) + 1

    def forget_session(self, session_id: str) -> None:
        self._records.pop(session_id, None)
        self._sequence.pop(session_id, None)
