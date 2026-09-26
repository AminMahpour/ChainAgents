"""Define the renderer contract each front end implements for one agent turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from chainagents.commands.native import RuntimeCommandResult
    from chainagents.events.stream import AgentStreamEvent
    from chainagents.exports.generated_files import GeneratedFileDescriptor
    from chainagents.runtime.reflection import ReflectionProposal
    from chainagents.turns.runner import TurnCommandError, TurnResult


class TurnRenderer(Protocol):
    """Receive the output of one ``TurnRunner.run`` call.

    Call order for a turn that is not cancelled:

    - command error: ``on_command_error`` then ``on_complete``;
    - MCP-tool command: ``on_command_result``, then the finish steps;
    - agent run: ``on_event`` for every stream event, then the finish steps.

    Finish steps are ``on_generated_files`` (only when files exist),
    ``on_reflection`` (only when a proposal exists), ``on_error`` (only when the
    agent run failed) and finally ``on_complete``. On cancellation only
    ``on_cancelled`` is called and the runner re-raises ``CancelledError``.
    """

    async def on_event(self, event: AgentStreamEvent) -> None:
        """Render one normalized agent stream event."""
        ...

    async def on_command_result(self, result: RuntimeCommandResult) -> None:
        """Render the direct output of an MCP-tool native command."""
        ...

    async def on_command_error(self, exc: TurnCommandError, status: int) -> None:
        """Render a failed or unknown native command (see ``TurnCommandError``)."""
        ...

    async def on_generated_files(self, files: list[GeneratedFileDescriptor]) -> None:
        """Render validated generated output files for the turn."""
        ...

    async def on_reflection(self, proposal: ReflectionProposal) -> None:
        """Render a reflection proposal for the turn."""
        ...

    async def on_cancelled(self) -> None:
        """React to the turn being cancelled."""
        ...

    async def on_error(self, exc: Exception) -> None:
        """Render an agent run failure (the raw exception; sanitise as needed)."""
        ...

    async def on_complete(self, result: TurnResult) -> None:
        """Handle the final result of a turn that was not cancelled."""
        ...


class BaseTurnRenderer:
    """No-op ``TurnRenderer`` for front ends that only need a few callbacks."""

    async def on_event(self, event: AgentStreamEvent) -> None:
        """Ignore the event."""

    async def on_command_result(self, result: RuntimeCommandResult) -> None:
        """Ignore the command result."""

    async def on_command_error(self, exc: TurnCommandError, status: int) -> None:
        """Ignore the command error."""

    async def on_generated_files(self, files: list[GeneratedFileDescriptor]) -> None:
        """Ignore the generated files."""

    async def on_reflection(self, proposal: ReflectionProposal) -> None:
        """Ignore the reflection proposal."""

    async def on_cancelled(self) -> None:
        """Ignore cancellation."""

    async def on_error(self, exc: Exception) -> None:
        """Ignore the failure."""

    async def on_complete(self, result: TurnResult) -> None:
        """Ignore completion."""
