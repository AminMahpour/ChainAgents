# Nested Subagents README Tutorial Design

## Goal

Add a longer, task-oriented nested-subagents tutorial to `README.md` so users can
choose the correct configuration form, copy a complete example, and understand
the runtime constraints without reading the parser or tests.

## Scope

The change is documentation-only. It will:

- add a `### Nested Subagents` subsection beneath `## Add Subagents`
- retain the existing general subagent field reference
- explain inline private children and reusable top-level children separately
- document supported behavior and configuration errors verified by the runtime
- avoid changing runtime code, configuration defaults, or tests

## Tutorial Structure

1. Introduce nested subagents as synchronous child agents available to a parent.
2. Explain when to use each configuration form:
   - `[[subagents.subagents]]` for an inline child private to its parent
   - `nested_subagents = ["name"]` for a top-level child that remains available
     to the main agent and can be reused by a parent
3. Provide a complete, copyable TOML example for an inline private child.
4. Provide a complete, copyable TOML example for a reusable top-level child.
5. Explain that nested children use the normal synchronous-subagent fields,
   including prompts, skills, MCP servers, and models.
6. Document the synchronous-only constraint and clarify that Agent Protocol
   subagents remain top-level `[[async_subagents]]` entries.
7. List configurations rejected during loading:
   - references to unknown top-level subagents
   - duplicate direct child names
   - reference cycles
   - nested entries that define `graph_id`
8. End with a short selection rule: use inline children for parent-private roles
   and references for shared roles.

## Placement

The tutorial will follow the supported subagent fields in `README.md`. The current
short nested-subagent paragraph will be replaced by the tutorial so the README
has one authoritative explanation instead of duplicated guidance.

## Accuracy and Verification

Tutorial claims and examples will be checked against:

- recursive parsing and validation in `chainagents/runtime/core.py`
- nested-subagent regression coverage in
  `tests/test_deepagent_runtime_rag.py`
- the current configuration example in `deepagent.toml.example`

Verification will include reviewing the rendered Markdown structure, checking
the TOML examples for valid syntax, and running `git diff --check`.

## Success Criteria

- A reader can distinguish private inline children from reusable referenced
  children without consulting source code.
- Both supported forms have independent, copyable examples.
- The synchronous-only limitation and rejected configurations are explicit.
- Existing general subagent documentation remains intact and non-duplicative.
