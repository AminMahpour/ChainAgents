ChainAgents Documentation
=========================

ChainAgents is a local-first `LangChain Deep Agent
<https://github.com/langchain-ai/deepagents>`_ you can drive through a
Chainlit web UI, a command-line interface, a full-screen terminal UI, or a
FastAPI HTTP server. Everything runs on your machine by default.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   configuration
   interfaces

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   architecture
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Operational Notes

   API_SECURITY.md
   orchestration_instructions.md
   rag-upload-migration.md

Highlights
----------

- Configurable model backends: Ollama, OpenAI-compatible servers
  (LM Studio, vLLM, OpenAI), Anthropic, and Snowflake Cortex.
- Sub-agents with isolated context windows, configurable as synchronous
  or asynchronous background tasks.
- Filesystem tools over local, sandboxed, or remote backends, with repo
  files mounted for the agent under ``/workspace/``.
- Context management: long-thread summarization and tool-output offload.
- Persistent memory via pluggable LangGraph checkpoint and store backends
  (Postgres when ``DATABASE_URL`` is set, in-memory otherwise).
- Skills, MCP servers, and bring-your-own tools.
- Optional Langfuse and LangSmith tracing.
- Per-response Markdown and PDF export buttons in Chainlit.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
