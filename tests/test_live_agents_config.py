from pathlib import Path

import pytest

from chainagents.runtime.extension_config import (
    normalize_skill_source_path,
    parse_extensions_config,
)
from chainagents.runtime.live_agents.config import (
    LIVE_AGENT_COMMAND_NAMES,
    live_agent_filesystem_tools,
    parse_live_agents_config,
    validate_live_agent_models,
)
from chainagents.runtime.types import (
    LiveAgentAllowlist,
    LiveAgentsConfig,
    LiveAgentTemplate,
)

TOOLS = ["read_file", "ls", "grep", "glob", "write_file", "edit_file"]


def parse(value, tmp_path, **kwargs):
    options = {
        "base_dir": tmp_path,
        "mcp_servers": {"docs": {}},
        "messaging_enabled": True,
        "delete_tool_enabled": False,
        "execute_tool_enabled": False,
    }
    options.update(kwargs)
    return parse_live_agents_config(value, **options)


def template(**overrides):
    base = {"name": "researcher", "description": "Reads.", "system_prompt": "Be brief."}
    base.update(overrides)
    return base


def test_defaults_when_section_missing(tmp_path):
    assert parse(None, tmp_path) == LiveAgentsConfig()


def test_enabled_requires_messaging(tmp_path):
    with pytest.raises(ValueError, match="agent.messaging"):
        parse({"enabled": True}, tmp_path, messaging_enabled=False)
    assert parse({"enabled": True}, tmp_path).enabled is True


def test_rejects_unknown_fields(tmp_path):
    with pytest.raises(ValueError, match="Unknown 'agent.live_agents'"):
        parse({"bogus": 1}, tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "max_live_per_session",
        "max_live_total",
        "max_turns_per_agent",
        "turn_recursion_limit",
        "idle_timeout_seconds",
    ],
)
@pytest.mark.parametrize("bad", [0, -1, "4"])
def test_limits_must_be_positive_ints(tmp_path, field, bad):
    if field == "idle_timeout_seconds" and bad == 0:
        assert parse({field: 0}, tmp_path).idle_timeout_seconds == 0
        return
    with pytest.raises(ValueError, match=f"agent.live_agents.{field}"):
        parse({field: bad}, tmp_path)


def test_applies_to_must_be_all_or_overrides(tmp_path):
    with pytest.raises(ValueError, match="applies_to"):
        parse({"allow": {"applies_to": "some"}}, tmp_path)
    assert parse({"allow": {"applies_to": "overrides"}}, tmp_path).allow.applies_to == "overrides"


def test_template_requires_name_and_description(tmp_path):
    with pytest.raises(ValueError, match="name.*description"):
        parse({"templates": [template(description="")]}, tmp_path)
    with pytest.raises(ValueError, match="name.*description"):
        parse({"templates": [template(name="")]}, tmp_path)


def test_template_names_unique(tmp_path):
    with pytest.raises(ValueError, match="more than once"):
        parse({"templates": [template(), template()]}, tmp_path)


def test_template_prompt_and_prompt_file_exclusive(tmp_path):
    (tmp_path / "p.md").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="both"):
        parse({"templates": [template(system_prompt_file="p.md")]}, tmp_path)


def test_template_prompt_file_loaded(tmp_path):
    (tmp_path / "p.md").write_text("  From file.\n", encoding="utf-8")
    raw = template(system_prompt_file="p.md")
    del raw["system_prompt"]
    config = parse({"templates": [raw]}, tmp_path)
    assert config.templates == (
        LiveAgentTemplate(name="researcher", description="Reads.", system_prompt="From file."),
    )


def test_template_prompt_file_missing_raises_value_error(tmp_path):
    raw = template(system_prompt_file="missing.md")
    del raw["system_prompt"]
    with pytest.raises(ValueError, match="'researcher' system_prompt_file.*not readable"):
        parse({"templates": [raw]}, tmp_path)


def test_template_name_must_be_string(tmp_path):
    with pytest.raises(ValueError, match="must be strings"):
        parse({"templates": [template(name=5)]}, tmp_path)


def test_unknown_mcp_server_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown MCP server 'nope'"):
        parse({"allow": {"mcp_servers": ["nope"]}}, tmp_path)
    with pytest.raises(ValueError, match="unknown MCP server 'nope'"):
        parse({"templates": [template(mcp_servers=["nope"])]}, tmp_path)


def test_unknown_tool_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown tool 'delete'"):
        parse({"allow": {"tools": ["delete"]}}, tmp_path)
    with pytest.raises(ValueError, match="unknown tool 'bogus'"):
        parse({"templates": [template(tools=["bogus"])], "allow": {"applies_to": "overrides"}}, tmp_path)
    config = parse({"allow": {"tools": ["delete"]}}, tmp_path, delete_tool_enabled=True)
    assert config.allow.tools == ("delete",)


def test_filesystem_tools_helper():
    base = live_agent_filesystem_tools(delete_tool_enabled=False, execute_tool_enabled=False)
    assert set(base) == set(TOOLS) and "delete" not in base and "execute" not in base
    both = live_agent_filesystem_tools(delete_tool_enabled=True, execute_tool_enabled=True)
    assert both == (*base, "delete", "execute")


def test_template_outside_allowlist_rejected_when_applies_to_all(tmp_path):
    with pytest.raises(ValueError, match="outside allowlist"):
        parse(
            {
                "allow": {"tools": ["read_file"]},
                "templates": [template(tools=["write_file"])],
            },
            tmp_path,
        )


def test_template_outside_allowlist_accepted_when_overrides(tmp_path):
    config = parse(
        {
            "allow": {"applies_to": "overrides", "tools": ["read_file"]},
            "templates": [template(tools=["write_file"])],
        },
        tmp_path,
    )
    assert config.templates[0].tools == ("write_file",)


def test_skills_normalized_in_allow_and_template(tmp_path):
    normalized = normalize_skill_source_path("/skills/a", tmp_path)
    config = parse(
        {"allow": {"skills": ["/skills/a"]}, "templates": [template(skills=["/skills/a"])]},
        tmp_path,
    )
    assert config.allow.skills == (normalized,)
    assert config.templates[0].skills == (normalized,)


def test_validate_models_rejects_unknown_profile(tmp_path):
    config = parse({"allow": {"models": ["nope"]}}, tmp_path)
    with pytest.raises(ValueError, match="unknown model profile 'nope'"):
        validate_live_agent_models(config, {})
    config = parse(
        {
            "allow": {"applies_to": "overrides"},
            "templates": [template(model="nope")],
        },
        tmp_path,
    )
    with pytest.raises(ValueError, match="unknown model profile 'nope'"):
        validate_live_agent_models(config, {})


def test_validate_models_all_mode_requires_allowlisted_template_model(tmp_path):
    config = parse({"templates": [template(model="fast")]}, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        validate_live_agent_models(config, {"fast": object()})
    config = parse(
        {"allow": {"models": ["fast"]}, "templates": [template(model="fast")]}, tmp_path
    )
    validate_live_agent_models(config, {"fast": object()})
    overrides = parse(
        {"allow": {"applies_to": "overrides"}, "templates": [template(model="fast")]},
        tmp_path,
    )
    validate_live_agent_models(overrides, {"fast": object()})


def test_reserved_chainlit_command_rejected(tmp_path):
    assert LIVE_AGENT_COMMAND_NAMES == {"agents", "spawn", "msg", "kill"}
    raw = {
        "chainlit": {
            "commands": [
                {"name": "kill", "description": "x", "target": "prompt", "value": "v"}
            ]
        }
    }
    with pytest.raises(ValueError, match="reserved"):
        parse_extensions_config(raw, tmp_path / "deepagent.toml")


def test_extensions_config_exposes_live_agents_and_panel_flag(tmp_path):
    default = parse_extensions_config({}, tmp_path / "deepagent.toml")
    assert default.live_agents == LiveAgentsConfig()
    assert default.chainlit_live_agents_panel_enabled is True
    assert default.enabled is False

    raw = {
        "agent": {
            "messaging": {"enabled": True},
            "live_agents": {
                "enabled": True,
                "allow": {"tools": ["read_file"]},
                "templates": [template(tools=["read_file"])],
            },
        },
        "chainlit": {"live_agents_panel_enabled": False},
    }
    config = parse_extensions_config(raw, tmp_path / "deepagent.toml")
    assert config.live_agents.enabled is True
    assert config.live_agents.allow == LiveAgentAllowlist(tools=("read_file",))
    assert config.chainlit_live_agents_panel_enabled is False
    assert config.enabled is True

    with pytest.raises(ValueError, match="live_agents_panel_enabled"):
        parse_extensions_config(
            {"chainlit": {"live_agents_panel_enabled": "no"}}, tmp_path / "deepagent.toml"
        )
