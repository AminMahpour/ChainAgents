"""Slash commands for live agents: /agents, /spawn, /msg and /kill."""

from __future__ import annotations

from chainagents.runtime.live_agents.config import SpawnRequest
from chainagents.runtime.live_agents.manager import LiveAgentManager

_USAGE = {
    "agents": "Usage: /agents",
    "spawn": "Usage: /spawn <template> [name] [-- instructions]",
    "msg": "Usage: /msg <agent> <text>",
    "kill": "Usage: /kill <agent|all>",
}
_SPAWN_SEPARATOR = " -- "

LIVE_AGENT_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "agents": "List live agents in this conversation",
    "spawn": "Spawn a live agent: /spawn <template> [name] [-- instructions]",
    "msg": "Message a live agent: /msg <agent> <text>",
    "kill": "Stop a live agent: /kill <agent|all>",
}


def format_idle(seconds: float) -> str:
    """Format an idle duration compactly (``42s``, ``3m``, ``1h5m``)."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes = total // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes}m" if minutes else f"{hours}h"


def _agents_table(manager: LiveAgentManager, session_id: str) -> str:
    rows = manager.list(session_id)
    if not rows:
        return "No live agents."
    now = manager.now()
    lines = [
        "| name | template | status | turns | idle |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        idle = format_idle(now - float(row["last_active_at"]))  # type: ignore[arg-type]
        lines.append(
            f"| {row['name']} | {row['template'] or '-'} | {row['status']} "
            f"| {row['turns']} | {idle} |"
        )
    return "\n".join(lines)


def _notify_main(manager: LiveAgentManager, session_id: str, text: str) -> None:
    if manager.config.notify_main_on_user_actions:
        manager.notify_main(session_id, text)


async def handle_live_agent_command(
    manager: LiveAgentManager,
    *,
    command_name: str,
    raw_args: str,
    session_id: str,
) -> str:
    """Run one live agent slash command and return Markdown output.

    Raises:
        ValueError: On usage errors, unknown agents and manager rejections; the
            message includes the command's usage line.
    """
    usage = _USAGE.get(command_name)
    if usage is None:
        raise ValueError(f"Unknown live agent command `/{command_name}`.")
    args = raw_args.strip()
    try:
        if command_name == "agents":
            return _agents_table(manager, session_id)
        if command_name == "spawn":
            return await _spawn(manager, session_id, args, usage)
        if command_name == "msg":
            return _msg(manager, session_id, args, usage)
        return await _kill(manager, session_id, args, usage)
    except KeyError as exc:
        name = exc.args[0] if exc.args else ""
        raise ValueError(f"Unknown live agent `{name}`. {usage}") from exc
    except ValueError as exc:
        if usage in str(exc):
            raise
        raise ValueError(f"{exc} {usage}") from exc


async def _spawn(
    manager: LiveAgentManager, session_id: str, args: str, usage: str
) -> str:
    left, sep, instructions = f" {args} ".partition(_SPAWN_SEPARATOR)
    tokens = left.split()
    if not tokens or len(tokens) > 2:
        raise ValueError(usage)
    request = SpawnRequest(
        template=tokens[0],
        name=tokens[1] if len(tokens) > 1 else None,
        instructions=instructions.strip() or None if sep else None,
    )
    record, _spec = await manager.spawn(session_id, request, spawned_by="user")
    template = record.template or "custom"
    _notify_main(
        manager,
        session_id,
        f"FYI: user spawned `{record.name}` (template {template}).",
    )
    return f"Spawned `{record.name}` ({template}, {record.model or 'default'})."


def _msg(manager: LiveAgentManager, session_id: str, args: str, usage: str) -> str:
    parts = args.split(None, 1)
    if len(parts) < 2:
        raise ValueError(usage)
    record = manager.get(session_id, parts[0])
    manager.send_from_user(session_id, record.id, parts[1])
    _notify_main(
        manager,
        session_id,
        f"FYI: user sent a message to `{record.name}`: {parts[1]}",
    )
    return f"Sent to `{record.name}`. The reply will appear when ready."


async def _kill(
    manager: LiveAgentManager, session_id: str, args: str, usage: str
) -> str:
    if len(args.split()) != 1:
        raise ValueError(usage)
    if args == "all":
        rows = manager.list(session_id)
        for row in rows:
            await manager.teardown(session_id, str(row["id"]))
        if rows:
            _notify_main(
                manager, session_id, f"FYI: user stopped {len(rows)} live agents."
            )
        return f"Stopped {len(rows)} agents."
    record = manager.get(session_id, args)
    await manager.teardown(session_id, record.id)
    _notify_main(manager, session_id, f"FYI: user stopped `{record.name}`.")
    return f"Stopped `{record.name}`."
