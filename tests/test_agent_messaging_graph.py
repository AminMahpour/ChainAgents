"""Exercise message insertion through a real compiled DeepAgents graph."""

import asyncio
from dataclasses import replace

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from chainagents.runtime.artifacts import LargeToolResultArtifactRegistry
from chainagents.runtime.graph import build_agent_kwargs
from chainagents.runtime.messaging import MessageBroker
from chainagents.runtime.middleware import (
    create_deep_agent_with_configured_summarization,
)
from chainagents.runtime.types import (
    ExtensionsConfig,
    MessagingConfig,
    ModelDefaults,
    SubagentConfig,
)
from test_deepagent_runtime_rag import make_runtime_config


class ToolAwareFakeModel(FakeListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def test_main_model_receives_steering_before_its_next_step(tmp_path):
    config = replace(
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
    broker = MessageBroker(config.extensions.messaging)
    broker.open("s", "main", name="main", parent=None)
    sent = broker.send_user("s", "Please focus on the tests")
    model = ToolAwareFakeModel(responses=["done"])
    kwargs = build_agent_kwargs(
        config,
        tools=[],
        model_profile=ModelDefaults(provider="ollama", name="fake"),
        reasoning_level="medium",
        reasoning_level_is_explicit=False,
        system_prompt="You are helpful.",
        custom_instruction=None,
        rag_enabled=False,
        project_root=tmp_path,
        artifact_registry=LargeToolResultArtifactRegistry(),
        include_async_subagents=False,
        build_model=lambda _level, _profile: model,
        session_id="s",
        messaging_broker=broker,
    )
    graph = create_deep_agent_with_configured_summarization(config, **kwargs)

    async def exercise():
        return await graph.ainvoke(
            {"messages": [HumanMessage(content="Original request")]},
            {"configurable": {"thread_id": "s"}},
        )

    result = asyncio.run(exercise())
    assert broker.get("s", sent.id).status == "delivered"
    assert any(
        isinstance(message, HumanMessage)
        and "Please focus on the tests" in message.content
        for message in result["messages"]
    )


def test_opted_in_leaf_subagent_runs_as_compiled_participant(tmp_path):
    config = replace(
        make_runtime_config(
            tmp_path,
            extensions=ExtensionsConfig(
                config_path=None,
                messaging=MessagingConfig(enabled=True),
                subagents=(
                    SubagentConfig(
                        name="reviewer",
                        description="Reviews",
                        system_prompt="Review.",
                        messaging=True,
                    ),
                ),
            ),
        ),
        rag=None,
        rag_requested=False,
        agent_state="stateless",
    )
    broker = MessageBroker(config.extensions.messaging)
    model = ToolAwareFakeModel(responses=["reviewed"])
    kwargs = build_agent_kwargs(
        config,
        tools=[],
        model_profile=ModelDefaults(provider="ollama", name="fake"),
        reasoning_level="medium",
        reasoning_level_is_explicit=False,
        system_prompt="Main.",
        custom_instruction=None,
        rag_enabled=False,
        project_root=tmp_path,
        artifact_registry=LargeToolResultArtifactRegistry(),
        include_async_subagents=False,
        build_model=lambda _level, _profile: model,
        session_id="s",
        messaging_broker=broker,
    )
    subagent = kwargs["subagents"][0]["runnable"]

    async def exercise():
        return await subagent.ainvoke(
            {"messages": [HumanMessage(content="Review")]},
            {"configurable": {"thread_id": "s"}},
        )

    result = asyncio.run(exercise())
    assert result["messages"][-1].content == "reviewed"
    assert broker.recipients("s", "main") == []


def test_unopted_subagent_graph_does_not_receive_message_tools(tmp_path):
    config = replace(
        make_runtime_config(
            tmp_path,
            extensions=ExtensionsConfig(
                config_path=None,
                messaging=MessagingConfig(enabled=True),
                subagents=(SubagentConfig("plain", "Plain", "Plain"),),
            ),
        ),
        rag=None,
        rag_requested=False,
        agent_state="stateless",
    )
    model = ToolAwareFakeModel(responses=["ok"])
    kwargs = build_agent_kwargs(
        config,
        tools=[],
        model_profile=ModelDefaults(provider="ollama", name="fake"),
        reasoning_level="medium",
        reasoning_level_is_explicit=False,
        system_prompt="Main.",
        custom_instruction=None,
        rag_enabled=False,
        project_root=tmp_path,
        artifact_registry=LargeToolResultArtifactRegistry(),
        include_async_subagents=False,
        build_model=lambda _level, _profile: model,
        session_id="s",
        messaging_broker=MessageBroker(config.extensions.messaging),
    )
    graph = kwargs["subagents"][0]["runnable"].runnable
    tool_names = set(graph.nodes["tools"].bound.tools_by_name)
    assert "send_agent_message" not in tool_names
    assert "list_agent_recipients" not in tool_names
