"""Implement the interactive ``--configure`` subcommand for deepagent.toml."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO

from chainagents.runtime import (
    DEFAULT_EXTENSIONS_CONFIG,
    PROJECT_ROOT,
    format_model_provider,
    normalize_model_provider,
    normalize_snowflake_cortex_endpoint_url,
    resolve_local_path,
)


@dataclass(frozen=True)
class ConfigPrompt:
    """Describe one interactive TOML configuration prompt."""

    section: str
    key: str
    label: str
    kind: str
    default: Any | None = None
    choices: tuple[str, ...] = ()
    optional: bool = False


CONFIGURE_PROMPTS = (
    ConfigPrompt(
        section="model",
        key="provider",
        label="Model provider",
        kind="choice",
        default="ollama",
        choices=(
            "ollama",
            "openai_compatible",
            "snowflake_cortex",
            "anthropic",
            "claude",
        ),
    ),
    ConfigPrompt(
        section="model",
        key="base_url",
        label="Model base URL",
        kind="str",
        default="http://127.0.0.1:11434",
    ),
    ConfigPrompt(
        section="model",
        key="name",
        label="Model name",
        kind="str",
        default="gpt-oss:20b",
    ),
    ConfigPrompt(
        section="model",
        key="reasoning_effort",
        label="Reasoning effort",
        kind="choice",
        default="medium",
        choices=("low", "medium", "high"),
    ),
    ConfigPrompt(
        section="model",
        key="temperature",
        label="Temperature",
        kind="float",
        default=0,
    ),
    ConfigPrompt(
        section="agent",
        key="state",
        label="Agent state",
        kind="choice",
        default="stateful",
        choices=("stateful", "stateless"),
    ),
    ConfigPrompt(
        section="agent",
        key="recursion_limit",
        label="Recursion limit",
        kind="int",
        default=200,
    ),
    ConfigPrompt(
        section="rag",
        key="enabled",
        label="Enable RAG",
        kind="bool",
        default=False,
    ),
    ConfigPrompt(
        section="rag.embedding",
        key="provider",
        label="RAG embedding provider",
        kind="choice",
        default="auto",
        choices=("auto", "ollama", "openai_compatible"),
    ),
    ConfigPrompt(
        section="rag.embedding",
        key="model",
        label="RAG embedding model",
        kind="str",
        optional=True,
    ),
    ConfigPrompt(
        section="rag.embedding",
        key="base_url",
        label="RAG embedding base URL",
        kind="str",
        optional=True,
    ),
    ConfigPrompt(
        section="langfuse",
        key="enabled",
        label="Enable Langfuse",
        kind="bool",
        default=False,
    ),
    ConfigPrompt(
        section="chainlit",
        key="model_mode_enabled",
        label="Show Chainlit model selector",
        kind="bool",
        default=True,
    ),
    ConfigPrompt(
        section="chainlit",
        key="reasoning_mode_enabled",
        label="Show Chainlit reasoning modes",
        kind="bool",
        default=True,
    ),
    ConfigPrompt(
        section="chainlit",
        key="reasoning_steps_enabled",
        label="Show Chainlit reasoning steps",
        kind="bool",
        default=True,
    ),
    ConfigPrompt(
        section="chainlit",
        key="tool_steps_enabled",
        label="Show Chainlit tool steps",
        kind="bool",
        default=True,
    ),
    ConfigPrompt(
        section="chainlit",
        key="startup_status_enabled",
        label="Show Chainlit startup status",
        kind="bool",
        default=True,
    ),
)


def nested_config_value(config: dict[str, Any], *, section: str, key: str) -> Any | None:
    """Return a value from a dotted TOML section path."""
    current: Any = config
    for part in section.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if not isinstance(current, dict):
        return None
    return current.get(key)


def prompt_default_text(value: Any | None, *, optional: bool) -> str:
    """Format a current/default value for an interactive prompt."""
    if value is None:
        return "skip" if optional else ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def parse_config_prompt_value(raw_value: str, prompt: ConfigPrompt) -> Any:
    """Parse and validate one interactive config answer."""
    value = raw_value.strip()
    if prompt.kind == "choice":
        normalized = value.lower().replace("-", "_")
        if normalized == "snowflake_cortex" and value != normalized:
            choices = ", ".join(prompt.choices)
            raise ValueError(f"Choose one of: {choices}.")
        if normalized not in prompt.choices:
            choices = ", ".join(prompt.choices)
            raise ValueError(f"Choose one of: {choices}.")
        return normalized
    if prompt.kind == "bool":
        normalized = value.lower()
        if normalized in {"true", "yes", "y", "1", "on"}:
            return True
        if normalized in {"false", "no", "n", "0", "off"}:
            return False
        raise ValueError("Enter yes or no.")
    if prompt.kind == "int":
        try:
            parsed_int = int(value)
        except ValueError as exc:
            raise ValueError("Enter a whole number.") from exc
        if parsed_int < 1:
            raise ValueError("Enter a number greater than zero.")
        return parsed_int
    if prompt.kind == "float":
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("Enter a number.") from exc
    return value


def read_config_prompt_value(
    prompt: ConfigPrompt,
    *,
    current_value: Any | None,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
) -> tuple[Any | None, bool]:
    """Read and validate one interactive config prompt value."""
    fallback = current_value if current_value is not None else prompt.default
    while True:
        default_text = prompt_default_text(fallback, optional=prompt.optional)
        print(f"{prompt.label} [{default_text}]: ", end="", file=stdout, flush=True)
        raw_value = stdin.readline()
        if raw_value == "":
            raw_value = "\n"
        candidate = raw_value.strip()
        if not candidate:
            if fallback is not None:
                return fallback, True
            return None, False
        try:
            return parse_config_prompt_value(candidate, prompt), True
        except ValueError as exc:
            print(f"{prompt.label}: {exc}", file=stderr)


def toml_scalar(value: Any) -> str:
    """Serialize a scalar value for the supported config fields."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    return json.dumps(str(value), ensure_ascii=True)


def toml_section_ranges(lines: list[str]) -> dict[str, tuple[int, int]]:
    """Return line ranges for non-array TOML sections."""
    headers: list[tuple[str | None, int]] = []
    for index, line in enumerate(lines):
        array_match = re.match(r"^\s*\[\[([^\[\]]+)]]\s*(?:#.*)?$", line)
        if array_match:
            headers.append((None, index))
            continue
        table_match = re.match(r"^\s*\[([^\[\]]+)]\s*(?:#.*)?$", line)
        if table_match:
            headers.append((table_match.group(1).strip(), index))

    ranges: dict[str, tuple[int, int]] = {}
    for offset, (section, start) in enumerate(headers):
        if section is None:
            continue
        end = headers[offset + 1][1] if offset + 1 < len(headers) else len(lines)
        ranges[section] = (start, end)
    return ranges


def apply_toml_updates(
    original: str,
    updates: dict[tuple[str, str], Any],
    *,
    removals: set[tuple[str, str]] | None = None,
) -> str:
    """Apply known TOML scalar updates while preserving unrelated text."""
    removals = removals or set()
    lines = original.splitlines()
    section_order = list(
        dict.fromkeys(
            [
                *(section for section, _ in updates),
                *(section for section, _ in removals),
            ]
        )
    )
    ranges = toml_section_ranges(lines)

    for section in section_order:
        section_updates = {
            key: value
            for (update_section, key), value in updates.items()
            if update_section == section
        }
        if section not in ranges:
            if not section_updates:
                continue
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(f"[{section}]")
            for key, value in section_updates.items():
                lines.append(f"{key} = {toml_scalar(value)}")
            ranges = toml_section_ranges(lines)
            continue

        start, end = ranges[section]
        section_removals = {
            key
            for removal_section, key in removals
            if removal_section == section and key not in section_updates
        }
        for key in section_removals:
            key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
            for index in range(start + 1, end):
                if key_pattern.match(lines[index]):
                    del lines[index]
                    end -= 1
                    ranges = toml_section_ranges(lines)
                    break
        for key, value in section_updates.items():
            replacement = f"{key} = {toml_scalar(value)}"
            key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
            for index in range(start + 1, end):
                if key_pattern.match(lines[index]):
                    lines[index] = replacement
                    break
            else:
                lines.insert(end, replacement)
                end += 1
                ranges = toml_section_ranges(lines)

    return "\n".join(lines).rstrip() + "\n"


def resolve_configure_config_path(config_path: str | Path | None) -> Path:
    """Resolve the config path that the interactive command should edit."""
    config_name = (
        str(config_path).strip()
        if config_path is not None
        else os.getenv("DEEPAGENT_CONFIG", DEFAULT_EXTENSIONS_CONFIG).strip()
    )
    return resolve_local_path(
        config_name or DEFAULT_EXTENSIONS_CONFIG,
        PROJECT_ROOT,
    )


def run_configure_command(
    *,
    config_path: Path,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    """Interactively configure a deepagent.toml file."""
    config_path = config_path.expanduser()
    original = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    try:
        current_config = tomllib.loads(original) if original.strip() else {}
    except tomllib.TOMLDecodeError as exc:
        print(f"configure: could not parse {config_path}: {exc}", file=stderr)
        return 1

    print("Configure ChainAgents. Press Enter to keep the current value.", file=stdout)
    updates: dict[tuple[str, str], Any] = {}
    removals: set[tuple[str, str]] = set()
    raw_current_model_provider = nested_config_value(
        current_config,
        section="model",
        key="provider",
    )
    try:
        current_model_provider = normalize_model_provider(raw_current_model_provider)
    except ValueError:
        current_model_provider = None
    current_model_endpoint_url = nested_config_value(
        current_config,
        section="model",
        key="endpoint_url",
    )
    selected_model_provider = current_model_provider
    for prompt in CONFIGURE_PROMPTS:
        current_value = nested_config_value(
            current_config,
            section=prompt.section,
            key=prompt.key,
        )
        effective_prompt = prompt
        provider_changed = selected_model_provider != current_model_provider
        is_model_base_url = prompt.section == "model" and prompt.key == "base_url"
        is_model_name = prompt.section == "model" and prompt.key == "name"
        is_cortex_base_url = bool(
            is_model_base_url
            and selected_model_provider == "snowflake_cortex"
        )
        is_openai_compatible_base_url = bool(
            is_model_base_url
            and selected_model_provider == "openai_compatible"
        )
        requires_explicit_model_name = bool(
            is_model_name
            and selected_model_provider
            in {"openai_compatible", "snowflake_cortex", "anthropic"}
        )
        if is_model_base_url and provider_changed:
            current_value = None
        if is_model_name and provider_changed:
            current_value = None
        if is_model_base_url and selected_model_provider != "ollama":
            effective_prompt = replace(prompt, default=None)
        if is_cortex_base_url:
            if current_model_provider != "snowflake_cortex":
                current_value = None
            elif current_model_endpoint_url is not None and current_value is not None:
                try:
                    normalize_snowflake_cortex_endpoint_url(
                        current_model_endpoint_url,
                        full_endpoint=True,
                    )
                except ValueError:
                    try:
                        normalize_snowflake_cortex_endpoint_url(
                            current_value,
                            full_endpoint=False,
                        )
                    except ValueError:
                        pass
                    else:
                        removals.add(("model", "endpoint_url"))
                else:
                    try:
                        normalize_snowflake_cortex_endpoint_url(
                            current_value,
                            full_endpoint=False,
                        )
                    except ValueError:
                        current_value = None
                        removals.add(("model", "base_url"))
        elif requires_explicit_model_name:
            effective_prompt = replace(prompt, default=None)
        value, should_write = read_config_prompt_value(
            effective_prompt,
            current_value=current_value,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
        )
        if prompt.section == "model" and prompt.key == "provider" and should_write:
            selected_model_provider = normalize_model_provider(value)
            if selected_model_provider != current_model_provider:
                removals.add(("model", "base_url"))
                removals.add(("model", "endpoint_url"))
                removals.add(("model", "api_key"))
        if is_openai_compatible_base_url and not should_write:
            if (
                current_model_provider == "openai_compatible"
                and str(current_model_endpoint_url or "").strip()
            ):
                continue
            print(
                "Model base URL: OpenAI-compatible providers require an explicit URL.",
                file=stderr,
            )
            return 1
        if is_cortex_base_url:
            if not should_write:
                if (
                    current_model_provider == "snowflake_cortex"
                    and current_model_endpoint_url is not None
                ):
                    try:
                        normalize_snowflake_cortex_endpoint_url(
                            current_model_endpoint_url,
                            full_endpoint=True,
                        )
                    except ValueError as exc:
                        print(f"Model endpoint URL: {exc}", file=stderr)
                        return 1
                    continue
                print(
                    "Model base URL: Snowflake Cortex requires an explicit account URL.",
                    file=stderr,
                )
                return 1
            try:
                value, _ = normalize_snowflake_cortex_endpoint_url(
                    value,
                    full_endpoint=False,
                )
            except ValueError as exc:
                print(f"Model base URL: {exc}", file=stderr)
                return 1
            normalized_current_value = None
            if current_value is not None:
                with contextlib.suppress(ValueError):
                    normalized_current_value, _ = normalize_snowflake_cortex_endpoint_url(
                        current_value,
                        full_endpoint=False,
                    )
            if (
                current_model_endpoint_url is not None
                and value != normalized_current_value
            ):
                removals.add(("model", "endpoint_url"))
        if requires_explicit_model_name and not should_write:
            provider_label = format_model_provider(selected_model_provider)
            print(
                f"Model name: {provider_label} requires an explicit model name.",
                file=stderr,
            )
            return 1
        if should_write:
            updates[(prompt.section, prompt.key)] = value

    updated = apply_toml_updates(original, updates, removals=removals)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(updated, encoding="utf-8")
    print(f"Configuration written to {config_path}", file=stdout)
    return 0
