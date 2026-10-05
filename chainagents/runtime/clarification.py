"""Let the main agent pause a run to ask the user a clarifying question."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, ToolCall
from langchain_core.tools import tool
from langgraph.types import interrupt

from chainagents.runtime.background_tasks.context import current_background_task_id

ASK_USER_TOOL_NAME = "ask_user"
CLARIFICATION_INTERRUPT_KIND = "clarification"
MAX_CLARIFICATION_OPTIONS = 6
# Checkpoint write channel LangGraph records interrupts on. Its constant is
# private since LangGraph 1.0; test_pending_questions_are_read_from_the_
# checkpointer fails if the channel name ever changes.
_INTERRUPT_WRITE_CHANNEL = "__interrupt__"
CLARIFICATION_SYSTEM_PROMPT = (
    "Clarifying questions:\n"
    "- If a request is ambiguous in ways that would change what subagents do "
    "(scope, sources, output), call `ask_user` once with one focused question "
    "before delegating. Otherwise proceed without asking.\n"
    "- Offer short `options` when the likely answers are a few clear choices."
)


@dataclass(frozen=True)
class PendingClarification:
    """One clarification question a paused run is waiting on."""

    interrupt_id: str
    question: str
    options: tuple[str, ...] = ()


def normalize_clarification_options(options: list[str] | None) -> list[str]:
    """Return trimmed, de-duplicated options capped at the supported count."""
    normalized: list[str] = []
    for option in options or []:
        text = " ".join(str(option).split())
        if text and text not in normalized:
            normalized.append(text)
    return normalized[:MAX_CLARIFICATION_OPTIONS]


@tool(ASK_USER_TOOL_NAME)
def ask_user(question: str, options: list[str] | None = None) -> str:
    """Pause and ask the user one clarifying question; returns their answer.

    Use this before delegating to subagents when the request is ambiguous in
    ways that would change the work. `options` are optional suggested answers.
    """
    text = " ".join(question.split())
    if not text:
        return "ask_user needs a non-empty question."
    if current_background_task_id() is not None:
        # A background run has no user attached and cannot be resumed.
        return (
            "ask_user is unavailable in background tasks; proceed with your "
            "best judgment and state your assumptions."
        )
    answer = interrupt(
        {
            "kind": CLARIFICATION_INTERRUPT_KIND,
            "question": text,
            "options": normalize_clarification_options(options),
        }
    )
    return str(answer)


class ClarificationMiddleware(AgentMiddleware[Any, Any, Any]):
    """Give only the agent it is attached to the ``ask_user`` tool.

    DeepAgents copies the main agent's ``tools`` argument into subagents that
    declare no tools of their own, but never middleware tools, so attaching
    the tool here keeps it main-agent only.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tools = [ask_user]

    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Make ``ask_user`` the only tool call in its message.

        Tool calls in one message run concurrently, so a ``task`` emitted next
        to ``ask_user`` would delegate before the question is answered. Other
        calls are dropped; the model can issue them again after the answer.
        """
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        message = messages[-1]
        asks = [call for call in message.tool_calls if call.get("name") == ASK_USER_TOOL_NAME]
        if not asks or len(message.tool_calls) == 1 or not message.id:
            return None
        return {"messages": [_keep_only_tool_call(message, asks[0])]}


def validate_ask_user_tool_name(existing_tools: Any) -> None:
    """Reject a configured tool that would share the reserved ``ask_user`` name.

    ``ClarificationMiddleware`` recognizes calls by name, so a configured
    tool named ``ask_user`` would be shadowed or turned into a pause.
    """
    for candidate in existing_tools or ():
        name = str(
            getattr(candidate, "name", None) or getattr(candidate, "__name__", "")
        ).strip()
        if name == ASK_USER_TOOL_NAME:
            raise ValueError(
                f"A configured tool uses the reserved name '{ASK_USER_TOOL_NAME}', "
                "which agent.clarification needs. Rename or prefix the configured "
                "tool, or disable agent.clarification."
            )


async def pending_clarifications_from_checkpointer(
    checkpointer: Any, thread_id: str
) -> list[PendingClarification]:
    """Read a thread's pending questions straight from its latest checkpoint.

    This avoids building the agent (models, MCP sessions) just to learn
    whether a reopened conversation is waiting on an answer.
    """
    checkpoint = await checkpointer.aget_tuple(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    )
    if checkpoint is None:
        return []
    interrupts: list[Any] = []
    for write in checkpoint.pending_writes or ():
        if len(write) >= 3 and write[1] == _INTERRUPT_WRITE_CHANNEL and isinstance(write[2], (list, tuple)):
            interrupts.extend(write[2])
    return pending_clarifications(interrupts)


def pending_clarifications(interrupts: Any) -> list[PendingClarification]:
    """Return the clarification questions among a state's pending interrupts."""
    pending: list[PendingClarification] = []
    for item in interrupts or ():
        value = getattr(item, "value", None)
        interrupt_id = getattr(item, "id", None)
        if (
            not isinstance(value, dict)
            or value.get("kind") != CLARIFICATION_INTERRUPT_KIND
            or not interrupt_id
        ):
            continue
        pending.append(
            PendingClarification(
                interrupt_id=str(interrupt_id),
                question=str(value.get("question") or ""),
                options=tuple(normalize_clarification_options(value.get("options"))),
            )
        )
    return pending


def _keep_only_tool_call(message: AIMessage, kept: ToolCall) -> AIMessage:
    """Return ``message`` (same id, so it replaces the original) with one call."""
    kept_id = kept.get("id")
    content = message.content
    if isinstance(content, list):
        # Anthropic-style content repeats each call as a tool_use block; a
        # block without a matching result would be rejected by the provider.
        content = [
            block
            for block in content
            if not (
                isinstance(block, dict)
                and block.get("type") in {"tool_use", "function_call"}
                and block.get("id") != kept_id
            )
        ]
    additional_kwargs = dict(message.additional_kwargs)
    raw_calls = additional_kwargs.get("tool_calls")
    if isinstance(raw_calls, list):
        additional_kwargs["tool_calls"] = [
            call for call in raw_calls if isinstance(call, dict) and call.get("id") == kept_id
        ]
    return message.model_copy(
        update={
            "content": content,
            "tool_calls": [kept],
            "invalid_tool_calls": [],
            "additional_kwargs": additional_kwargs,
        }
    )
