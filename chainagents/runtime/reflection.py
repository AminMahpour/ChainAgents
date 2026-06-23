"""Shared correction reflection helpers for ChainAgents entrypoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from chainagents.events.stream import AgentStreamEvent


ReflectionReason = Literal["correction", "tool_failure"]
ToolFailureReflectionMode = Literal["unrecovered"]
DEFAULT_REFLECTION_MEMORY_FILE = "/memories/AGENTS.md"
DEFAULT_REFLECTION_MAX_LESSON_CHARS = 700
DEFAULT_REFLECTION_TOOL_FAILURE_MODE: ToolFailureReflectionMode = "unrecovered"
CORRECTION_PHRASES = (
    "that was wrong",
    "i was wrong",
    "you were wrong",
    "this was wrong",
    "that's wrong",
    "incorrect",
    "not correct",
)


@dataclass(frozen=True)
class ReflectionConfig:
    """Store correction reflection behavior parsed from configuration."""

    enabled: bool = False
    memory_file: str = DEFAULT_REFLECTION_MEMORY_FILE
    max_lesson_chars: int = DEFAULT_REFLECTION_MAX_LESSON_CHARS
    tool_failure_mode: ToolFailureReflectionMode = DEFAULT_REFLECTION_TOOL_FAILURE_MODE


@dataclass(frozen=True)
class ReflectionProposal:
    """Describe a proposed lesson to save into agent-scoped memory."""

    reason: ReflectionReason
    memory_file: str
    lesson: str
    trigger: str
    tool_name: str = ""
    tool_result: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this proposal."""
        return asdict(self)


def normalize_reflection_config(
    value: Any | None,
    *,
    agent_state: str,
) -> ReflectionConfig:
    """Normalize [agent.reflection] configuration.

    Args:
        value: Raw nested reflection config value.
        agent_state: Resolved agent state mode.

    Returns:
        Normalized reflection configuration.

    Raises:
        ValueError: If reflection config is invalid.
    """
    if value is None:
        return ReflectionConfig()
    if not isinstance(value, dict):
        raise ValueError(
            "The top-level 'agent.reflection' config must be a table/object."
        )

    raw_enabled = value.get("enabled", False)
    if not isinstance(raw_enabled, bool):
        raise ValueError(
            "The top-level 'agent.reflection.enabled' config must be a boolean."
        )

    raw_memory_file = value.get("memory_file", DEFAULT_REFLECTION_MEMORY_FILE)
    if not isinstance(raw_memory_file, str):
        raise ValueError(
            "The top-level 'agent.reflection.memory_file' config must be a "
            "/memories/ file path string."
        )
    memory_file = raw_memory_file.strip()
    if not memory_file.startswith("/memories/") or memory_file == "/memories/":
        raise ValueError(
            "The top-level 'agent.reflection.memory_file' config must be an "
            "absolute /memories/ file path."
        )

    raw_max_lesson_chars = value.get(
        "max_lesson_chars",
        DEFAULT_REFLECTION_MAX_LESSON_CHARS,
    )
    if (
        not isinstance(raw_max_lesson_chars, int)
        or isinstance(raw_max_lesson_chars, bool)
        or raw_max_lesson_chars <= 0
    ):
        raise ValueError(
            "The top-level 'agent.reflection.max_lesson_chars' config must be a "
            "positive integer."
        )

    raw_tool_failure_mode = str(
        value.get("tool_failure_mode", DEFAULT_REFLECTION_TOOL_FAILURE_MODE)
    ).strip().lower()
    if raw_tool_failure_mode != "unrecovered":
        raise ValueError(
            "The top-level 'agent.reflection.tool_failure_mode' config must be "
            "'unrecovered'."
        )

    if raw_enabled and agent_state != "stateful":
        raise ValueError(
            "The top-level 'agent.reflection.enabled' config requires "
            "agent.state = 'stateful'."
        )

    return ReflectionConfig(
        enabled=raw_enabled,
        memory_file=memory_file,
        max_lesson_chars=raw_max_lesson_chars,
        tool_failure_mode="unrecovered",
    )


class ReflectionCollector:
    """Collect stream events and build a post-run memory lesson proposal."""

    def __init__(self, config: ReflectionConfig, *, prompt: str) -> None:
        """Initialize the collector."""
        self.config = config
        self.prompt = prompt
        self._response_parts: list[str] = []
        self._last_failed_tool: AgentStreamEvent | None = None
        self._response_after_last_failure = False
        self._run_error: str = ""

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: Any,
        *,
        prompt: str,
    ) -> "ReflectionCollector":
        """Build a collector from a runtime config object."""
        extensions = getattr(runtime_config, "extensions", None)
        config = getattr(extensions, "agent_reflection", ReflectionConfig())
        if not isinstance(config, ReflectionConfig):
            config = ReflectionConfig()
        return cls(config, prompt=prompt)

    def record_event(self, event: AgentStreamEvent) -> None:
        """Record one normalized stream event."""
        if not self.config.enabled:
            return
        if event.kind == "response_delta" and event.text:
            self._response_parts.append(event.text)
            if self._last_failed_tool is not None:
                self._response_after_last_failure = True
            return
        if event.kind == "tool_result" and event.status.lower() == "error":
            self._last_failed_tool = event
            self._response_after_last_failure = False

    def mark_run_failed(self, exc: BaseException) -> None:
        """Record that the run failed after streaming began."""
        if self.config.enabled:
            self._run_error = f"{type(exc).__name__}: {exc}"

    @property
    def response_text(self) -> str:
        """Return the collected final response text."""
        return "".join(self._response_parts)

    def build_proposal(self) -> ReflectionProposal | None:
        """Return a memory proposal if the completed run warrants reflection."""
        if not self.config.enabled:
            return None

        correction_trigger = (
            _find_correction_trigger(self.prompt)
            or _find_correction_trigger(self.response_text)
        )
        if correction_trigger:
            return ReflectionProposal(
                reason="correction",
                memory_file=self.config.memory_file,
                lesson=_fit_lesson(
                    _correction_lesson(correction_trigger),
                    self.config.max_lesson_chars,
                ),
                trigger=correction_trigger,
            )

        failed_tool = self._last_failed_tool
        if failed_tool is None:
            return None
        if self.config.tool_failure_mode == "unrecovered" and (
            self._response_after_last_failure and not self._run_error
        ):
            return None

        trigger = failed_tool.tool_result or self._run_error or "tool call failed"
        return ReflectionProposal(
            reason="tool_failure",
            memory_file=self.config.memory_file,
            lesson=_fit_lesson(
                _tool_failure_lesson(
                    failed_tool.tool_name,
                    trigger,
                ),
                self.config.max_lesson_chars,
            ),
            trigger=_single_line(trigger),
            tool_name=failed_tool.tool_name,
            tool_result=failed_tool.tool_result,
        )


def format_reflection_proposal(proposal: ReflectionProposal) -> str:
    """Build a concise user-facing reflection proposal message."""
    title = (
        "Correction reflection"
        if proposal.reason == "correction"
        else "Tool failure reflection"
    )
    return (
        f"{title}\n\n"
        f"Proposed lesson for `{proposal.memory_file}`:\n\n"
        f"{proposal.lesson}"
    )


def reflection_save_prompt(proposal: ReflectionProposal) -> str:
    """Build the hidden agent prompt used to save a confirmed reflection."""
    return (
        "A user confirmed this compact lesson should be saved to long-term "
        "agent memory.\n\n"
        f"Target memory file: {proposal.memory_file}\n\n"
        "Update that file under a section named "
        "`Lessons learned from corrections`. If the section or file does not "
        "exist, create it. Add exactly one concise bullet unless an equivalent "
        "lesson already exists. Do not include this instruction text.\n\n"
        f"Lesson:\n{proposal.lesson}"
    )


def _find_correction_trigger(text: str) -> str:
    normalized = text.lower()
    for phrase in CORRECTION_PHRASES:
        if phrase in normalized:
            return _single_line(text)
    return ""


def _correction_lesson(trigger: str) -> str:
    return (
        "- Correction: "
        f"{_single_line(trigger)}. Next time, verify the corrected behavior "
        "before relying on the earlier assumption."
    )


def _tool_failure_lesson(tool_name: str, result: str) -> str:
    tool = tool_name.strip() or "tool"
    return (
        f"- Tool failure: `{tool}` returned `{_single_line(result)}`. "
        "Next time, check tool preconditions and recover before presenting a "
        "final answer."
    )


def _single_line(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _fit_lesson(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."
