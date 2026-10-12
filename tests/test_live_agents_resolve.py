from __future__ import annotations

import pytest

from chainagents.runtime.live_agents import config as live_agents_config
from chainagents.runtime.types import (
    LiveAgentAllowlist,
    LiveAgentsConfig,
    LiveAgentTemplate,
)

SpawnRequest = live_agents_config.SpawnRequest
LiveAgentSpawnError = live_agents_config.LiveAgentSpawnError
resolve_spawn = live_agents_config.resolve_spawn

TEMPLATE_PROMPT = "You are a researcher."


def _config(
    *,
    allow: LiveAgentAllowlist | None = None,
    templates: tuple[LiveAgentTemplate, ...] | None = None,
) -> LiveAgentsConfig:
    if templates is None:
        templates = (
            LiveAgentTemplate(
                name="researcher",
                description="Researches",
                system_prompt=TEMPLATE_PROMPT,
                model="fast",
                tools=("read_file", "grep"),
                skills=("/skills/web/",),
                mcp_servers=("docs",),
            ),
        )
    if allow is None:
        allow = LiveAgentAllowlist(
            models=("fast", "smart"),
            mcp_servers=("docs", "search"),
            skills=("/skills/web/", "/skills/code/"),
            tools=("read_file", "grep", "write_file"),
        )
    return LiveAgentsConfig(allow=allow, templates=templates)


def _resolve(request, config=None, *, existing=(), sequence=1):
    return resolve_spawn(
        request, config or _config(), existing_names=set(existing), sequence=sequence
    )


def test_template_only_copies_template_fields():
    spec = _resolve(SpawnRequest(template="researcher"))
    assert spec.name == "researcher-1"
    assert spec.template == "researcher"
    assert spec.system_prompt == TEMPLATE_PROMPT
    assert spec.model == "fast"
    assert spec.tools == ("read_file", "grep")
    assert spec.skills == ("/skills/web/",)
    assert spec.mcp_servers == ("docs",)


def test_add_tool_in_allowlist():
    spec = _resolve(
        SpawnRequest(template="researcher", add_tools=("write_file", "grep"))
    )
    assert spec.tools == ("read_file", "grep", "write_file")


def test_add_tool_outside_allowlist_lists_permitted():
    with pytest.raises(LiveAgentSpawnError) as err:
        _resolve(SpawnRequest(template="researcher", add_tools=("execute",)))
    text = str(err.value)
    assert "execute" in text
    for tool in ("read_file", "grep", "write_file"):
        assert tool in text


def test_add_tool_outside_allowlist_checked_when_applies_to_overrides():
    config = _config(allow=LiveAgentAllowlist(applies_to="overrides", tools=("grep",)))
    with pytest.raises(LiveAgentSpawnError):
        _resolve(SpawnRequest(template="researcher", add_tools=("execute",)), config)


def test_remove_tool_always_allowed():
    spec = _resolve(
        SpawnRequest(template="researcher", remove_tools=("grep", "not_in_template"))
    )
    assert spec.tools == ("read_file",)


def test_model_override_must_be_allowed():
    assert _resolve(SpawnRequest(template="researcher", model="smart")).model == "smart"
    with pytest.raises(LiveAgentSpawnError) as err:
        _resolve(SpawnRequest(template="researcher", model="huge"))
    assert "huge" in str(err.value)
    assert "smart" in str(err.value)


def test_add_skill_and_mcp_server_checked():
    spec = _resolve(
        SpawnRequest(
            template="researcher",
            add_skills=("/skills/code/", "/skills/web"),
            add_mcp_servers=("search",),
        )
    )
    assert spec.skills == ("/skills/web/", "/skills/code/")
    assert spec.mcp_servers == ("docs", "search")
    with pytest.raises(LiveAgentSpawnError) as err:
        _resolve(SpawnRequest(template="researcher", add_skills=("/skills/evil/",)))
    assert "/skills/evil/" in str(err.value)
    assert "/skills/code/" in str(err.value)
    with pytest.raises(LiveAgentSpawnError) as err:
        _resolve(SpawnRequest(template="researcher", add_mcp_servers=("nope",)))
    assert "nope" in str(err.value)
    assert "search" in str(err.value)


def test_instructions_appended_under_heading():
    spec = _resolve(SpawnRequest(template="researcher", instructions="Find X."))
    assert (
        spec.system_prompt == f"{TEMPLATE_PROMPT}\n\n## Task from supervisor\nFind X."
    )


def test_templateless_uses_instructions_as_prompt():
    spec = _resolve(SpawnRequest(instructions="Be helpful."))
    assert spec.system_prompt == "Be helpful."
    assert spec.template is None
    assert spec.tools == ()
    assert spec.skills == ()
    assert spec.mcp_servers == ()
    assert spec.model is None


def test_templateless_requires_instructions():
    with pytest.raises(LiveAgentSpawnError):
        _resolve(SpawnRequest())
    with pytest.raises(LiveAgentSpawnError):
        _resolve(SpawnRequest(instructions="   "))


def test_templateless_rejected_when_custom_prompt_disallowed():
    config = _config(allow=LiveAgentAllowlist(allow_custom_prompt=False))
    with pytest.raises(LiveAgentSpawnError) as err:
        _resolve(SpawnRequest(instructions="Hi"), config)
    assert "researcher" in str(err.value) or "template" in str(err.value)


def test_instructions_with_template_accepted_when_custom_prompt_disallowed():
    config = _config(allow=LiveAgentAllowlist(allow_custom_prompt=False))
    spec = _resolve(SpawnRequest(template="researcher", instructions="Go."), config)
    assert spec.system_prompt.endswith("## Task from supervisor\nGo.")


def test_unknown_template_lists_templates():
    with pytest.raises(LiveAgentSpawnError) as err:
        _resolve(SpawnRequest(template="ghost"))
    assert "ghost" in str(err.value)
    assert "researcher" in str(err.value)


def test_default_name_uses_template_and_sequence():
    assert (
        _resolve(SpawnRequest(template="researcher"), sequence=3).name == "researcher-3"
    )


def test_default_name_without_template():
    assert _resolve(SpawnRequest(instructions="x"), sequence=3).name == "agent-3"


@pytest.mark.parametrize("bad", ["Bad Name", "-x", "a" * 41, ""])
def test_invalid_name_rejected(bad):
    with pytest.raises(LiveAgentSpawnError) as err:
        _resolve(SpawnRequest(template="researcher", name=bad))
    assert "name" in str(err.value)


def test_max_length_name_accepted():
    name = "a" * 40
    assert _resolve(SpawnRequest(template="researcher", name=name)).name == name


def test_duplicate_name_rejected():
    with pytest.raises(LiveAgentSpawnError) as err:
        _resolve(SpawnRequest(template="researcher", name="r1"), existing={"r1"})
    assert "r1" in str(err.value)
