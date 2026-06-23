"""Textual terminal UI for ChainAgents."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from langchain_mcp_adapters import sessions as mcp_sessions
from rich.panel import Panel
from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Markdown, RichLog, Static

from chainagents.commands.native import (
    dumps_tool_result,
    parse_native_command,
    resolve_native_command,
    resolve_runtime_command,
)
from chainagents.events.stream import AgentStreamEvent, AgentStreamEventAdapter
from chainagents.runtime.reflection import (
    ReflectionCollector,
    format_reflection_proposal,
)
from chainagents.runtime import (
    AgentRuntime,
    PROJECT_ROOT,
    ReasoningLevel,
    build_langgraph_run_config,
    normalize_reasoning_level,
)


DEFAULT_TUI_THREAD_ID = "tui"
TUI_SIDE_PANEL_WIDTH = 56
TUI_STDERR_LOG = Path(".files/tui-stderr.log")


def tui_stderr_log_path(runtime: AgentRuntime) -> Path:
    """Return the log path used for stderr emitted during TUI mode."""
    project_root = Path(getattr(runtime, "project_root", PROJECT_ROOT))
    return project_root / TUI_STDERR_LOG


@contextmanager
def capture_mcp_stdio_stderr(log_path: Path):
    """Route stdio MCP server stderr to a log file during TUI mode."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdio_client = mcp_sessions.stdio_client
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log_file:

            def stdio_client_with_log(server, errlog=None):
                return original_stdio_client(server, errlog=log_file)

            mcp_sessions.stdio_client = stdio_client_with_log
            yield
    finally:
        mcp_sessions.stdio_client = original_stdio_client


class ChainAgentsTuiApp(App[int]):
    """Interactive Textual app for the configured ChainAgents runtime."""

    TITLE = "ChainAgents"

    CSS = """
    Screen {
        layout: vertical;
    }

    #status {
        height: 1;
        padding: 0 1;
        background: $boost;
        color: $text;
    }

    #main {
        height: 1fr;
    }

    #conversation {
        width: 2fr;
        height: 100%;
        border: solid $primary;
        padding: 0 1;
    }

    .conversation-label {
        height: 1;
        margin: 1 0 0 0;
        color: $text-muted;
    }

    .conversation-user {
        margin: 0 0 1 0;
    }

    .conversation-assistant {
        margin: 0 0 1 0;
        padding: 0 1;
        border: solid $success;
    }

    #side {
        width: __SIDE_PANEL_WIDTH__;
        height: 100%;
    }

    #reasoning {
        height: 1fr;
        border: solid $secondary;
    }

    #tools {
        height: 1fr;
        border: solid $accent;
    }

    #prompt {
        height: 3;
    }

    #commands {
        height: auto;
        max-height: 6;
        padding: 0 1;
        border: solid $warning;
        display: none;
    }
    """.replace("__SIDE_PANEL_WIDTH__", str(TUI_SIDE_PANEL_WIDTH))

    BINDINGS = [
        ("ctrl+c", "cancel_or_quit", "Cancel/Quit"),
        ("ctrl+l", "clear_conversation", "Clear"),
        ("tab", "complete_slash_command", "Complete command"),
    ]

    def __init__(self, *, runtime: AgentRuntime, args: Any) -> None:
        super().__init__()
        self.runtime = runtime
        self.args = args
        self.thread_id = str(getattr(args, "thread_id", "") or DEFAULT_TUI_THREAD_ID)
        self.reasoning_level: ReasoningLevel = normalize_reasoning_level(
            getattr(args, "reasoning", None),
            default=runtime.config.default_reasoning,
        )
        self.model_name = getattr(args, "model", None) or runtime.config.model_name
        self.async_subagent_url = getattr(args, "async_subagent_url", None)
        self.mcp_session_id = getattr(args, "mcp_session_id", None)
        self.active_task: asyncio.Task[None] | None = None
        self.status_message = "Ready."
        self.conversation_entries: list[tuple[str, str]] = []
        self.assistant_markdown_widgets: dict[int, Markdown] = {}
        self.reasoning_entries: list[tuple[str, str]] = []
        self.reasoning_entry_indexes: dict[str, int] = {}
        self.tool_entries: list[str] = []
        self.command_help_text = ""
        self.command_help_visible = False
        self.visible_command_names: list[str] = []

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header(show_clock=True)
        yield Static(self.status_message, id="status")
        with Horizontal(id="main"):
            yield VerticalScroll(id="conversation")
            with Vertical(id="side"):
                yield RichLog(id="reasoning", wrap=True, markup=True, highlight=False)
                yield RichLog(id="tools", wrap=True, markup=True, highlight=False)
        yield Static("", id="commands")
        yield Input(placeholder="Prompt ChainAgents...", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        """Focus the prompt when the TUI starts."""
        self.query_one("#prompt", Input).focus()
        self._set_status(
            f"Ready. thread={self.thread_id} model={self.model_name} "
            f"reasoning={self.reasoning_level}"
        )

    def on_key(self, event: events.Key) -> None:
        """Handle prompt-level slash command completion keys."""
        if event.key == "tab" and self.command_help_visible:
            event.prevent_default()
            event.stop()
            self.action_complete_slash_command()

    @on(Input.Submitted, "#prompt")
    async def on_prompt_submitted(self, event: Input.Submitted) -> None:
        """Send a submitted prompt to the agent."""
        event.stop()
        prompt = event.value.strip()
        if not prompt or self.active_task is not None:
            return

        event.input.value = ""
        self.hide_command_help()
        self.active_task = asyncio.create_task(self._run_prompt(prompt))

    @on(Input.Changed, "#prompt")
    def on_prompt_changed(self, event: Input.Changed) -> None:
        """Refresh slash command help as the user types."""
        self.refresh_command_help(event.value)

    async def action_cancel_or_quit(self) -> None:
        """Cancel an active run, or exit when idle."""
        if self.active_task is not None and not self.active_task.done():
            self.active_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.active_task
            return
        self.exit(0)

    def action_clear_conversation(self) -> None:
        """Clear the visible conversation panes."""
        if self.active_task is not None and not self.active_task.done():
            return
        self.conversation_entries.clear()
        self.assistant_markdown_widgets.clear()
        self.reasoning_entries.clear()
        self.reasoning_entry_indexes.clear()
        self.tool_entries.clear()
        self.query_one("#conversation", VerticalScroll).remove_children()
        self.query_one("#reasoning", RichLog).clear()
        self.query_one("#tools", RichLog).clear()
        self._set_status("Cleared.")

    def action_complete_slash_command(self) -> None:
        """Complete the first visible slash command into the prompt."""
        if not self.command_help_visible or not self.visible_command_names:
            return
        prompt = self.query_one("#prompt", Input)
        command_name = self.visible_command_names[0]
        prompt.value = f"/{command_name} "
        prompt.cursor_position = len(prompt.value)
        self.hide_command_help()

    async def _run_prompt(self, raw_prompt: str) -> None:
        prompt_input = self.query_one("#prompt", Input)
        prompt_input.disabled = True
        self._set_status("Running...")
        await self._append_conversation("You", raw_prompt)
        self.reasoning_entry_indexes.clear()

        try:
            resolved_prompt = await self._resolve_prompt(raw_prompt)
            if resolved_prompt is None:
                return
            await self._stream_agent_prompt(resolved_prompt)
            self._set_status("Ready.")
        except asyncio.CancelledError:
            self._set_status("Cancelled.")
            raise
        except Exception as exc:
            self._append_tool_entry(f"runtime error {type(exc).__name__}: {exc}")
            self._set_status(f"{type(exc).__name__}: {exc}")
        finally:
            prompt_input.disabled = False
            prompt_input.focus()
            self.active_task = None

    def refresh_command_help(self, prompt_value: str) -> None:
        """Show matching slash commands when the prompt starts with slash."""
        text = prompt_value.lstrip()
        if not text.startswith("/") or " " in text:
            self.hide_command_help()
            return

        query = text[1:].lower()
        commands = [
            command
            for command in getattr(self.runtime, "chainlit_commands", ())
            if str(getattr(command, "name", "")).lower().startswith(query)
        ]
        commands = sorted(commands, key=lambda command: str(getattr(command, "name", "")))
        self.visible_command_names = [
            str(getattr(command, "name", ""))
            for command in commands
            if str(getattr(command, "name", ""))
        ]

        if not commands:
            self.command_help_text = "No matching slash commands."
        else:
            self.command_help_text = "\n".join(
                self._format_command_help_line(command) for command in commands
            )

        panel = self.query_one("#commands", Static)
        panel.update(self.command_help_text)
        panel.display = True
        self.command_help_visible = True

    def hide_command_help(self) -> None:
        """Hide the slash command help panel."""
        self.command_help_text = ""
        self.command_help_visible = False
        self.visible_command_names = []
        panel = self.query_one("#commands", Static)
        panel.update("")
        panel.display = False

    @staticmethod
    def _format_command_help_line(command: Any) -> str:
        name = str(getattr(command, "name", "")).strip()
        description = str(getattr(command, "description", "")).strip()
        if description:
            return f"/{name} - {description}"
        return f"/{name}"

    async def _resolve_prompt(self, raw_prompt: str) -> str | None:
        parsed_command = resolve_native_command(raw_text=raw_prompt, selected_command=None)
        slash_command_from_text = parse_native_command(raw_prompt)
        if parsed_command is None:
            return raw_prompt

        try:
            command_result = await resolve_runtime_command(
                runtime=self.runtime,
                parsed=parsed_command,
                thread_id=self.thread_id,
                mcp_session_id=self.mcp_session_id,
            )
        except Exception as exc:
            self._append_tool_entry(
                f"command /{parsed_command.command_name} failed: {exc}"
            )
            self._set_status(f"Command /{parsed_command.command_name} failed.")
            return None

        if command_result.target == "unknown":
            if slash_command_from_text is not None:
                self._append_tool_entry(f"unknown command /{parsed_command.command_name}")
                self._set_status(f"Unknown command /{parsed_command.command_name}.")
                return None
            return raw_prompt

        if command_result.target == "mcp_tool":
            self._append_tool_entry(dumps_tool_result(command_result.tool_result))
            self._set_status(f"Command /{parsed_command.command_name} finished.")
            return None

        prompt = command_result.prompt or ""
        return prompt if prompt.strip() else None

    async def _stream_agent_prompt(self, prompt: str) -> None:
        agent = await self.runtime.get_agent(
            self.reasoning_level,
            model_name=self.model_name,
            thread_id=self.thread_id,
            async_subagent_url_override=self.async_subagent_url,
            mcp_session_id=self.mcp_session_id,
        )
        payload = {"messages": [{"role": "user", "content": prompt}]}
        config = build_langgraph_run_config(
            self.runtime.config,
            thread_id=self.thread_id,
        )
        adapter = AgentStreamEventAdapter(prompt=prompt)
        reflection_collector = ReflectionCollector.from_runtime_config(
            self.runtime.config,
            prompt=prompt,
        )
        stream = agent.astream_events(
            payload,
            config=config,
            version="v2",
            stream_mode=["messages", "updates", "custom"],
            subgraphs=True,
        )

        stream_error: Exception | None = None
        try:
            while True:
                try:
                    event = await anext(stream)
                except StopAsyncIteration:
                    break
                for stream_event in adapter.events_from_raw_event(event):
                    reflection_collector.record_event(stream_event)
                    await self._handle_stream_event(stream_event)
        except Exception as exc:
            stream_error = exc
            reflection_collector.mark_run_failed(exc)
        finally:
            with suppress(Exception):
                await stream.aclose()

        proposal = reflection_collector.build_proposal()
        if proposal is not None:
            self._append_tool_entry(format_reflection_proposal(proposal))
        if stream_error is not None:
            raise stream_error

    async def _handle_stream_event(self, event: AgentStreamEvent) -> None:
        if event.kind == "response_delta":
            await self._append_response_delta(event.text)
        elif event.kind == "reasoning_delta":
            self._append_reasoning(event.source, event.text)
        elif event.kind == "tool_call":
            self._append_tool_entry(
                " ".join(
                    part
                    for part in (
                        event.source,
                        event.tool_name,
                        event.status,
                        event.tool_args,
                    )
                    if part
                )
            )
        elif event.kind == "tool_result":
            self._append_tool_entry(
                " ".join(
                    part
                    for part in (
                        event.source,
                        event.tool_name,
                        event.status,
                        self._preview(event.tool_result),
                    )
                    if part
                )
            )
        elif event.kind == "summarization_status":
            self._append_tool_entry(
                f"{event.source} summarization {event.status}: {event.text}"
            )

    async def _append_conversation(self, role: str, text: str) -> None:
        self.conversation_entries.append((role, text))
        widget = await self._mount_conversation_entry(
            len(self.conversation_entries) - 1,
            role,
            text,
        )
        if role == "Assistant" and text and widget is not None:
            await widget.append(text)
            self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)

    async def _append_response_delta(self, text: str) -> None:
        if not text:
            return
        if self.conversation_entries and self.conversation_entries[-1][0] == "Assistant":
            role, current = self.conversation_entries[-1]
            updated_text = current + text
            entry_index = len(self.conversation_entries) - 1
            self.conversation_entries[-1] = (role, updated_text)
            widget = self.assistant_markdown_widgets.get(entry_index)
            if widget is not None:
                await widget.append(text)
        else:
            self.conversation_entries.append(("Assistant", text))
            widget = await self._mount_conversation_entry(
                len(self.conversation_entries) - 1,
                "Assistant",
            )
            if widget is not None:
                await widget.append(text)
        self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)

    async def _mount_conversation_entry(
        self,
        index: int,
        role: str,
        text: str = "",
    ) -> Markdown | None:
        conversation = self.query_one("#conversation", VerticalScroll)
        if role == "Assistant":
            markdown = Markdown("", classes="conversation-assistant")
            markdown.code_indent_guides = False
            self.assistant_markdown_widgets[index] = markdown
            await conversation.mount_all(
                (
                    Static("Assistant", classes="conversation-label"),
                    markdown,
                )
            )
            conversation.scroll_end(animate=False)
            return markdown
        else:
            await conversation.mount(
                Static(
                    Panel(
                        Text(text, style="white"),
                        title=role,
                        title_align="left",
                        border_style="cyan",
                    ),
                    classes="conversation-user",
                )
            )
        conversation.scroll_end(animate=False)
        return None

    def _append_reasoning(self, source: str, text: str) -> None:
        if not text:
            return
        existing_index = self.reasoning_entry_indexes.get(source)
        if existing_index is not None:
            current_source, current_text = self.reasoning_entries[existing_index]
            self.reasoning_entries[existing_index] = (current_source, current_text + text)
        else:
            self.reasoning_entries.append((source, text))
            self.reasoning_entry_indexes[source] = len(self.reasoning_entries) - 1
        self._render_reasoning()

    def _render_reasoning(self) -> None:
        log = self.query_one("#reasoning", RichLog)
        log.clear()
        for source, text in self.reasoning_entries:
            log.write(
                Panel(
                    Text(text, style="magenta"),
                    title=f"{source} reasoning",
                    title_align="left",
                    border_style="magenta",
                )
            )

    def _append_tool_entry(self, text: str) -> None:
        self.tool_entries.append(text)
        self.query_one("#tools", RichLog).write(Text(text, style="yellow"))

    def _set_status(self, message: str) -> None:
        self.status_message = message
        status = self.query_one("#status", Static)
        status.update(message)

    @staticmethod
    def _preview(text: str, limit: int = 200) -> str:
        compact = text.strip()
        if len(compact) <= limit:
            return compact
        return compact[:limit]


async def run_tui(runtime: AgentRuntime, args: Any) -> int:
    """Run the Textual TUI and return a process-style exit code."""
    app = ChainAgentsTuiApp(runtime=runtime, args=args)
    with capture_mcp_stdio_stderr(tui_stderr_log_path(runtime)):
        result = await app.run_async()
    return int(result or 0)
