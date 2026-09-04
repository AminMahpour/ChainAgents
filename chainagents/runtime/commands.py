"""Generated UI tools and native skill command discovery."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.middleware.skills import _list_skills
from langchain_core.tools import tool
from langgraph.graph.ui import push_ui_message
from pydantic import BaseModel, Field

import chainagents.runtime.backends as runtime_backends
import chainagents.runtime.constants as runtime_constants
from chainagents.runtime.constants import GENERATIVE_UI_COMPONENT_NAME
from chainagents.runtime.types import (
    ChainlitCommandConfig,
    ExtensionsConfig,
    SkillCommandMetadata,
)

logger = logging.getLogger("chainagents.runtime.core")


class RenderChainlitUIInput(BaseModel):
    """Define the schema for generated Chainlit UI panel requests."""

    title: str = Field(
        ...,
        min_length=1,
        description="Short title shown at the top of the generated UI panel.",
    )
    summary: str | None = Field(
        default=None,
        description="Optional concise markdown-style summary for the panel body.",
    )
    facts: dict[str, Any] | None = Field(
        default=None,
        description="Optional key-value facts to display in a compact grid.",
    )
    items: list[Any] | None = Field(
        default=None,
        description=(
            "Optional short list items to display in order. Prefer strings; "
            "use actions, not items, for prompt buttons."
        ),
    )
    table: dict[str, Any] | None = Field(
        default=None,
        description="Optional small table with columns and rows.",
    )
    actions: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional prompt buttons with label and prompt values.",
    )
    id: str | None = Field(
        default=None,
        description="Optional stable panel id. Reusing it updates the existing panel.",
    )


def _normalize_generated_ui_props(
    *,
    title: str,
    summary: str | None = None,
    facts: dict[str, Any] | None = None,
    items: list[Any] | None = None,
    table: dict[str, Any] | None = None,
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return props accepted by the bundled GeneratedPanel custom element."""
    props: dict[str, Any] = {"title": title.strip()}
    normalized_actions: list[dict[str, str]] = []
    action_keys: set[tuple[str, str]] = set()

    def action_parts(value: Any) -> tuple[str, str] | None:
        if not isinstance(value, dict):
            return None
        label = str(value.get("label") or "").strip()
        prompt = str(value.get("prompt") or "").strip()
        if label and prompt:
            return label, prompt
        return None

    def append_action(value: Any) -> None:
        parts = action_parts(value)
        if parts is None:
            return
        label, prompt = parts
        if parts not in action_keys:
            normalized_actions.append({"label": label, "prompt": prompt})
            action_keys.add(parts)

    def item_text(value: Any) -> str:
        if isinstance(value, dict):
            for key in ("label", "text", "title", "value", "name"):
                text = str(value.get(key) or "").strip()
                if text:
                    return text
            return ""
        return str(value).strip()

    if summary is not None and summary.strip():
        props["summary"] = summary.strip()
    if facts:
        props["facts"] = {str(key): value for key, value in facts.items()}
    if items:
        normalized_items: list[str] = []
        for item in items:
            append_action(item)
            if action_parts(item) is not None:
                continue
            text = item_text(item)
            if text:
                normalized_items.append(text)
        if normalized_items:
            props["items"] = normalized_items
    if table:
        props["table"] = table
    if actions:
        for action in actions:
            append_action(action)
    if normalized_actions:
        props["actions"] = normalized_actions
    return props


def create_render_chainlit_ui_tool() -> Any:
    """Create the built-in tool that emits LangGraph UI messages for Chainlit."""

    @tool(
        "render_chainlit_ui",
        args_schema=RenderChainlitUIInput,
        return_direct=False,
    )
    def render_chainlit_ui(
        title: str,
        summary: str | None = None,
        facts: dict[str, Any] | None = None,
        items: list[Any] | None = None,
        table: dict[str, Any] | None = None,
        actions: list[dict[str, Any]] | None = None,
        id: str | None = None,
    ) -> dict[str, Any]:
        """Render a whitelisted Chainlit generated UI panel for the current answer."""
        props = _normalize_generated_ui_props(
            title=title,
            summary=summary,
            facts=facts,
            items=items,
            table=table,
            actions=actions,
        )
        ui_message = push_ui_message(
            GENERATIVE_UI_COMPONENT_NAME,
            props,
            id=(id.strip() if isinstance(id, str) and id.strip() else None),
            metadata={"source": "main-agent"},
            state_key=None,
        )
        return {
            "rendered": True,
            "component": ui_message["name"],
            "id": ui_message["id"],
        }

    return render_chainlit_ui


def normalize_chainlit_command_name(value: str) -> str:
    """Normalize chainlit command name.

    Args:
        value: Value to normalize, convert, or serialize.

    Returns:
        The normalized value.
    """
    return value.strip().lstrip("/").lower()


def _load_skill_command_bucket(
    *,
    backend: CompositeBackend,
    source_paths: tuple[str, ...],
    source: Literal["agent_skill", "subagent_skill"],
    project_root: Path | None = None,
    owner: str | None = None,
) -> tuple[SkillCommandMetadata, ...]:
    """Load skill command metadata from one configured source bucket.

    Args:
        backend: The backend value.
        source_paths: Paths to the source.
        source: The source value.
        project_root: Project root used to resolve local paths.
        owner: The owner value.

    Returns:
        The loaded value.
    """
    commands_by_name: dict[str, SkillCommandMetadata] = {}
    for source_path in source_paths:
        try:
            source_skills = _list_skills(backend, source_path)
        except Exception as exc:
            logger.warning(
                "Failed to load skills from '%s' for Chainlit command generation: %s",
                source_path,
                exc,
            )
            continue

        for skill in source_skills:
            command_name = normalize_chainlit_command_name(str(skill["name"]))
            if not command_name or " " in command_name:
                logger.warning(
                    "Skipping skill '%s' from %s because it is not slash-command compatible.",
                    skill["name"],
                    skill["path"],
                )
                continue

            metadata = SkillCommandMetadata(
                name=command_name,
                description=str(skill["description"]).strip(),
                path=runtime_backends.virtual_workspace_path_to_local(str(skill["path"]), project_root),
                source=source,
                owner=owner,
            )
            previous = commands_by_name.pop(command_name, None)
            if previous is not None:
                logger.warning(
                    "Auto skill command '/%s' from %s overrides %s.",
                    command_name,
                    metadata.label,
                    previous.label,
                )
            commands_by_name[command_name] = metadata
    return tuple(commands_by_name.values())


def _resolve_chainlit_project_root(
    *,
    backend: CompositeBackend | None,
    project_root: Path | None,
) -> Path:
    """Resolve the project root used for Chainlit command discovery.

    Args:
        backend: The backend value.
        project_root: Project root used to resolve local paths.

    Returns:
        The resolved the project root used for chainlit command discovery.
    """
    if project_root is not None:
        return project_root

    if backend is not None:
        workspace_backend = backend.routes.get("/workspace/")
        if isinstance(workspace_backend, FilesystemBackend):
            return workspace_backend.cwd

    return runtime_constants.PROJECT_ROOT


def build_chainlit_command_catalog(
    extensions: ExtensionsConfig,
    *,
    backend: CompositeBackend | None = None,
    project_root: Path | None = None,
) -> tuple[tuple[ChainlitCommandConfig, ...], tuple[str, ...]]:
    """Build chainlit command catalog.

    Args:
        extensions: The extensions value.
        backend: The backend value.
        project_root: Project root used to resolve local paths.

    Returns:
        The constructed chainlit command catalog.
    """
    resolved_project_root = _resolve_chainlit_project_root(
        backend=backend,
        project_root=project_root,
    )
    backend = backend or runtime_backends.build_deepagent_backend(project_root=resolved_project_root)
    notes: list[str] = []
    merged_commands = list(extensions.chainlit_commands)
    explicit_names = {command.name: command for command in extensions.chainlit_commands}

    main_skill_commands = _load_skill_command_bucket(
        backend=backend,
        source_paths=extensions.skills,
        source="agent_skill",
        project_root=resolved_project_root,
    )
    subagent_commands_by_name: dict[str, SkillCommandMetadata] = {}
    for subagent in extensions.subagents:
        for metadata in _load_skill_command_bucket(
            backend=backend,
            source_paths=subagent.skills,
            source="subagent_skill",
            project_root=resolved_project_root,
            owner=subagent.name,
        ):
            previous = subagent_commands_by_name.pop(metadata.name, None)
            if previous is not None:
                logger.warning(
                    "Auto skill command '/%s' from %s overrides %s.",
                    metadata.name,
                    metadata.label,
                    previous.label,
                )
            subagent_commands_by_name[metadata.name] = metadata
    subagent_skill_commands = tuple(subagent_commands_by_name.values())

    winner_by_name: dict[str, ChainlitCommandConfig | SkillCommandMetadata] = {
        command.name: command for command in merged_commands
    }

    for metadata in main_skill_commands:
        explicit = explicit_names.get(metadata.name)
        if explicit is not None:
            note = (
                f"`/{metadata.name}` from {metadata.label} is hidden by explicit "
                f"Chainlit command `/{explicit.name}`."
            )
            notes.append(note)
            logger.warning(note)
            continue
        merged_commands.append(metadata.to_chainlit_command())
        winner_by_name[metadata.name] = metadata

    for metadata in subagent_skill_commands:
        winner = winner_by_name.get(metadata.name)
        if winner is None:
            merged_commands.append(metadata.to_chainlit_command())
            winner_by_name[metadata.name] = metadata
            continue

        if isinstance(winner, ChainlitCommandConfig):
            note = (
                f"`/{metadata.name}` from {metadata.label} is hidden by explicit "
                f"Chainlit command `/{winner.name}`."
            )
        else:
            note = f"`/{metadata.name}` from {metadata.label} is hidden by {winner.label}."
        notes.append(note)
        logger.warning(note)

    return tuple(merged_commands), tuple(notes)
