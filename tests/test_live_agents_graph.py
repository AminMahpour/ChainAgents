"""Graph construction for live agents."""

import asyncio
from dataclasses import replace

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from pydantic import Field

from chainagents.runtime.artifacts import LargeToolResultArtifactRegistry
from chainagents.runtime.background_tasks import BACKGROUND_TASK_TOOL_NAMES
from chainagents.runtime.backends import build_deepagent_backend
from chainagents.runtime.constants import DEFAULT_DEEPAGENT_FILESYSTEM_TOOLS
from chainagents.runtime.graph import build_live_agent_graph
from chainagents.runtime.live_agents.config import ResolvedLiveAgentSpec
from chainagents.runtime.messaging import MessageBroker
from chainagents.runtime.middleware import build_agent_middleware
from chainagents.runtime.types import ExtensionsConfig, MessagingConfig
from test_deepagent_runtime_rag import make_runtime_config

LIVE_AGENT_TOOL_NAMES = {
    "list_live_agent_templates",
    "spawn_live_agent",
    "list_live_agents",
    "teardown_live_agent",
}


class RecordingModel(FakeListChatModel):
    bound: list = Field(default_factory=list)
    seen_messages: list = Field(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        for item in tools:
            if isinstance(item, dict):
                name = item.get("name") or item.get("function", {}).get("name")
            else:
                name = getattr(item, "name", None)
            self.bound.append(name)
        return self

    def _call(self, messages, *args, **kwargs):
        self.seen_messages.append(messages)
        return super()._call(messages, *args, **kwargs)


def _config(tmp_path):
    return replace(
        make_runtime_config(
            tmp_path,
            extensions=ExtensionsConfig(
                config_path=None, messaging=MessagingConfig(enabled=True)
            ),
        ),
        rag=None,
        rag_requested=False,
        agent_state="stateless",
    )


def _backend(tmp_path):
    return build_deepagent_backend(
        project_root=tmp_path,
        include_memories=False,
        artifact_registry=LargeToolResultArtifactRegistry(),
    )


def _run(tmp_path, tools=("read_file", "ls"), prompt="You are the live agent."):
    config = _config(tmp_path)
    model = RecordingModel(responses=["done"])
    broker = MessageBroker(config.extensions.messaging)
    broker.open("s", "main", name="main", parent=None)
    spec = ResolvedLiveAgentSpec(
        name="worker",
        template=None,
        system_prompt=prompt,
        model=None,
        tools=tuple(tools),
        skills=(),
        mcp_servers=(),
    )
    graph = build_live_agent_graph(
        config,
        spec=spec,
        model=model,
        model_name="fake",
        reasoning_level="medium",
        mcp_tools=[],
        backend=_backend(tmp_path),
        project_root=tmp_path,
        broker=broker,
        session_id="s",
    )
    asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="hi")]},
            {"configurable": {"thread_id": "t"}},
        )
    )
    return model


def test_live_graph_exposes_only_resolved_filesystem_tools(tmp_path):
    names = set(_run(tmp_path).bound)
    assert {"read_file", "ls"} <= names
    assert not names & {"write_file", "edit_file", "glob", "grep"}


def test_live_graph_has_messaging_tools_and_no_management_tools(tmp_path):
    names = set(_run(tmp_path).bound)
    assert "send_agent_message" in names
    assert not names & set(BACKGROUND_TASK_TOOL_NAMES)
    assert not names & LIVE_AGENT_TOOL_NAMES
    assert "task" not in names


def test_live_graph_uses_spec_system_prompt(tmp_path):
    model = _run(tmp_path, prompt="You are the zebra specialist.")
    assert "zebra specialist" in str(model.seen_messages)


def test_build_agent_middleware_default_tools_unchanged(tmp_path):
    def fs_tools(**kwargs):
        mw = build_agent_middleware(backend=_backend(tmp_path), **kwargs)
        fs = next(m for m in mw if type(m).__name__ == "FilesystemMiddleware")
        return sorted(t.name for t in fs.tools)

    assert fs_tools() == sorted(DEFAULT_DEEPAGENT_FILESYSTEM_TOOLS)
    assert fs_tools(filesystem_tools=["read_file"]) == ["read_file"]


def test_live_graph_with_no_filesystem_tools_builds_and_binds_none(tmp_path):
    names = set(_run(tmp_path, tools=()).bound)
    assert not names & {
        "read_file", "ls", "write_file", "edit_file", "glob", "grep"
    }
    assert "send_agent_message" in names


def test_live_graph_without_read_file_hides_it(tmp_path):
    names = set(_run(tmp_path, tools=("ls",)).bound)
    assert "ls" in names
    assert "read_file" not in names
