---
name: repo-guide
description: Use this skill when the user wants a repo walkthrough, file summary, architecture explanation, or implementation starting point.
---

# repo-guide

Use this skill when the user asks for any of the following:

- a quick walkthrough of the current repository
- a summary of one or more files
- the main entrypoints and how the app is wired together
- where to make a change for a requested feature

Working rules:

1. Read the relevant files in `/workspace/` before answering.
2. Prefer concrete file references over vague summaries.
3. Start with the highest-signal files first:
   `README.md`, `main.py`, `deepagent_runtime.py`, `chainlit_bridge.py`, and config files.
4. If the user asks how to change behavior, identify the likely edit points and explain why.
5. Keep the answer concise and implementation-oriented.

Expected output style:

- Start with a 1-2 sentence summary of what the repo does.
- Then list the key files and what each one is responsible for.
- If relevant, end with the next file(s) to edit for the user’s request.

Example requests where this skill should be used:

- "Give me a quick tour of this project."
- "Summarize how the Chainlit UI is connected to the deep agent."
- "Where would I add a new MCP-backed subagent?"
- "Which files should I change to add a new UI behavior?"
