"""Render one shared agent turn into Chainlit messages through the event bridge."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import chainlit as cl
from chainlit.element import File

from chainagents.commands.native import RuntimeCommandResult, dumps_tool_result
from chainagents.events.stream import AgentStreamEvent
from chainagents.exports.generated_files import GeneratedFileDescriptor
from chainagents.exports.response import generated_file_elements_from_paths
from chainagents.interfaces.chainlit.bridge import ChainlitEventBridge
from chainagents.turns import BaseTurnRenderer, TurnCommandError, TurnResult


def native_command_output_message(result: RuntimeCommandResult) -> str:
    """Return the System message content for an MCP-tool native command."""
    return (
        f"Ran `/{result.command_name}` ({result.description}).\n\n"
        "Tool result:\n```json\n"
        f"{dumps_tool_result(result.tool_result)}\n"
        "```"
    )


class ChainlitTurnRenderer(BaseTurnRenderer):
    """Render a ``TurnRunner`` turn with a ``ChainlitEventBridge``.

    The bridge is built once the agent is ready, from the final agent prompt,
    so command outcomes never touch the task list.
    """

    def __init__(
        self,
        bridge_factory: Callable[[str], ChainlitEventBridge],
        *,
        prompt: str,
        project_root: Path | None = None,
    ) -> None:
        """Bind the renderer to a bridge factory.

        Args:
            bridge_factory: Builds the bridge for the final agent prompt.
            prompt: Raw user text, used when the agent fails before it starts.
            project_root: Project root used to resolve generated files.
        """
        self.bridge_factory = bridge_factory
        self.prompt = prompt
        self.project_root = project_root
        self.bridge: ChainlitEventBridge | None = None
        self.command_result: RuntimeCommandResult | None = None
        self.generated_files: list[GeneratedFileDescriptor] = []

    async def on_agent_start(self, prompt: str) -> None:
        await self._start_bridge(prompt)

    async def on_event(self, event: AgentStreamEvent) -> None:
        if event.kind == "mcp_status":
            await cl.Message(content=event.text).send()
            return
        bridge = self.bridge or await self._start_bridge(self.prompt)
        await bridge.handle_stream_event(event)

    async def on_command_result(self, result: RuntimeCommandResult) -> None:
        self.command_result = result

    async def on_command_error(self, exc: TurnCommandError, status: int) -> None:
        if exc.unknown:
            content = (
                f"{exc.message}\n"
                "Use a configured command from startup or send a normal prompt."
            )
        else:
            content = f"Native command `/{exc.command_name}` failed: {exc.message}"
        await cl.Message(author="System", content=content).send()

    async def on_generated_files(self, files: list[GeneratedFileDescriptor]) -> None:
        self.generated_files = files

    async def on_cancelled(self) -> None:
        if self.bridge is not None:
            with suppress(Exception):
                await self.bridge.cancel()

    async def on_error(self, exc: Exception) -> None:
        details = "".join(traceback.format_exception(exc, limit=10))
        with suppress(Exception):
            bridge = self.bridge or await self._start_bridge(self.prompt)
            await bridge.fail(exc, details, elements=self._file_elements())

    async def on_complete(self, result: TurnResult) -> None:
        if result.status != "completed":
            return
        if self.command_result is not None:
            await cl.Message(
                author="System",
                content=native_command_output_message(self.command_result),
                elements=self._file_elements(),
            ).send()
        elif self.bridge is not None:
            # The bridge attaches generated files to the final response itself.
            await self.bridge.finish()

    async def _start_bridge(self, prompt: str) -> ChainlitEventBridge:
        self.bridge = self.bridge_factory(prompt)
        await self.bridge.start()
        return self.bridge

    def _file_elements(self) -> list[File]:
        return generated_file_elements_from_paths(
            [file.path for file in self.generated_files if file.path is not None],
            project_root=self.project_root,
        )
