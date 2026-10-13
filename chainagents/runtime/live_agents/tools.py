"""Main-agent tools, prompt and roster middleware for live agents."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime, tool

from chainagents.runtime.live_agents.config import SpawnRequest
from chainagents.runtime.live_agents.manager import LiveAgentManager
from chainagents.runtime.messaging import _session_id
from chainagents.runtime.tracing import LangSmithTracing

logger = logging.getLogger(__name__)

LIVE_AGENT_TOOL_NAMES = frozenset({
    "list_live_agent_templates",
    "spawn_live_agent",
    "list_live_agents",
    "teardown_live_agent",
})

LIVE_AGENTS_PROMPT = """## Live agents

You can spawn live agents: long-lived helpers that persist across turns of this
conversation until they are torn down.
- Call `list_live_agent_templates` before spawning to see the available
  templates and what may be added to them, then call `spawn_live_agent`.
- Message a live agent with `send_agent_message`, addressing it by its name.
- Replies arrive asynchronously: they appear as messages on a later step, or
  you can wait for them with `wait_for_agent_messages`. Do not assume a reply
  is immediate.
- Use `list_live_agents` to check their status.
- Tear agents down with `teardown_live_agent` when their work is done.
- Messages from `live-agents` are FYI notices about what the user did with
  live agents directly; they need no reply and cannot be replied to."""


def validate_live_agent_tool_names(existing_tools: Iterable[object]) -> None:
    """Reject configured tools that would be shadowed by live agent actions."""
    collisions = sorted({
        name
        for candidate in existing_tools
        if (name := str(
            getattr(candidate, "name", None) or getattr(candidate, "__name__", "")
        ).strip()) in LIVE_AGENT_TOOL_NAMES
    })
    if collisions:
        raise ValueError(
            "Configured tools use reserved live agent tool names: "
            f"{', '.join(collisions)}. Rename or prefix the configured tools."
        )


def _error(exc: BaseException) -> dict[str, object]:
    if isinstance(exc, KeyError) and exc.args:
        return {"error": f"Unknown live agent '{exc.args[0]}'."}
    return {"error": str(exc) or type(exc).__name__}


def create_live_agent_tools(
    manager: LiveAgentManager,
    *,
    fixed_session_id: str | None = None,
    langsmith_tracing: LangSmithTracing | None = None,
) -> list[Any]:
    """Expose live agent lifecycle actions to the main agent."""

    @tool
    def list_live_agent_templates(runtime: ToolRuntime) -> dict[str, object]:
        """List live agent templates and what a spawn may add to them."""
        return manager.templates_payload()

    @tool
    async def spawn_live_agent(
        runtime: ToolRuntime,
        template: str | None = None,
        name: str | None = None,
        instructions: str | None = None,
        model: str | None = None,
        add_tools: list[str] | None = None,
        remove_tools: list[str] | None = None,
        add_skills: list[str] | None = None,
        add_mcp_servers: list[str] | None = None,
        initial_message: str | None = None,
    ) -> dict[str, object]:
        """Spawn a long-lived live agent from a template (or custom instructions).

        Message it afterwards with send_agent_message using its name; replies
        arrive asynchronously. Tear it down when its work is done.
        """
        try:
            session = _session_id(runtime, fixed_session_id)
            request = SpawnRequest(
                template=template,
                name=name,
                instructions=instructions,
                model=model,
                add_tools=tuple(add_tools or ()),
                remove_tools=tuple(remove_tools or ()),
                add_skills=tuple(add_skills or ()),
                add_mcp_servers=tuple(add_mcp_servers or ()),
                initial_message=initial_message,
            )
            parent_trace = (
                langsmith_tracing.capture_parent(runtime.config)
                if langsmith_tracing is not None
                else None
            )
            record, spec = await manager.spawn(
                session, request, parent_trace=parent_trace
            )
        except (ValueError, RuntimeError) as exc:
            # LiveAgentSpawnError / LiveAgentLimitError are ValueErrors; a
            # RuntimeError means the manager or conversation is shutting down.
            return _error(exc)
        except Exception as exc:
            logger.warning("Live agent spawn failed", exc_info=True)
            return _error(exc)
        return {
            "id": record.id,
            "name": record.name,
            "address": record.address,
            "status": record.status,
            "resolved": {
                "model": spec.model,
                "tools": list(spec.tools),
                "skills": list(spec.skills),
                "mcp_servers": list(spec.mcp_servers),
            },
        }

    @tool
    def list_live_agents(
        runtime: ToolRuntime, include_stopped: bool = False
    ) -> list[dict[str, object]]:
        """List this conversation's live agents with status and pending messages."""
        session = _session_id(runtime, fixed_session_id)
        return manager.list(session, include_stopped=include_stopped)

    @tool
    async def teardown_live_agent(
        agent: str, runtime: ToolRuntime, reason: str | None = None
    ) -> dict[str, object]:
        """Stop a live agent by name or id and return its final status."""
        try:
            session = _session_id(runtime, fixed_session_id)
            record = await manager.teardown(
                session, agent, reason=reason or "requested"
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            return _error(exc)
        return record.to_payload()

    return [
        list_live_agent_templates,
        spawn_live_agent,
        list_live_agents,
        teardown_live_agent,
    ]


class LiveAgentRosterMiddleware(AgentMiddleware[Any, Any, Any]):
    """Append a one-line roster of live agents to each main-agent model request."""

    def __init__(
        self, manager: LiveAgentManager, fixed_session_id: str | None = None
    ) -> None:
        self.manager = manager
        self.fixed_session_id = fixed_session_id

    def _with_roster(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        try:
            session = _session_id(request.runtime, self.fixed_session_id)
        except ValueError:
            return request
        roster = self.manager.roster(session)
        if not roster:
            return request
        return request.override(
            system_message=append_to_system_message(request.system_message, roster)
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(self._with_roster(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(self._with_roster(request))
