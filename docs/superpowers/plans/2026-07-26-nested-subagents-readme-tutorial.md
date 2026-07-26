# Nested Subagents README Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a longer nested-subagents tutorial to the README with separate
examples for private inline children and reusable referenced children.

**Architecture:** Keep the existing `## Add Subagents` overview and supported
field list, then replace its short nested-subagent paragraph with one focused
`### Nested Subagents` tutorial. Derive all behavioral claims from the current
runtime parser, nested-agent builders, tests, and `deepagent.toml.example`.

**Tech Stack:** Markdown, TOML, Python 3.12 `tomllib`

## Global Constraints

- This is a documentation-only change.
- Do not change runtime code, configuration defaults, or tests.
- Retain the existing general subagent field reference.
- Document only behavior verified in `chainagents/runtime/core.py` and
  `tests/test_deepagent_runtime_rag.py`.
- Do not create a Git commit unless the user explicitly requests one.

---

### Task 1: Add and verify the nested-subagents tutorial

**Files:**

- Modify: `README.md:618-633`
- Reference: `chainagents/runtime/core.py:2360-2535`
- Reference: `chainagents/runtime/core.py:4231-4415`
- Reference: `tests/test_deepagent_runtime_rag.py:4752-4940`
- Reference: `deepagent.toml.example:137-163`

**Interfaces:**

- Consumes: the existing `## Add Subagents` overview and supported-field list
- Produces: a standalone `### Nested Subagents` README tutorial

- [ ] **Step 1: Replace the short nested-subagent paragraph**

Use `apply_patch` to replace the paragraph beginning with “Nested subagents are
synchronous only” with this exact Markdown:

````markdown
### Nested Subagents

Nested subagents let a synchronous subagent delegate work to its own synchronous
child agents. They are useful when a coordinator needs focused helpers but the
main agent should not necessarily see every helper directly.

ChainAgents supports two nesting patterns:

- use `[[subagents.subagents]]` for a private inline child available only to its
  parent
- use `nested_subagents = ["name"]` to reuse a top-level synchronous subagent as
  a child while keeping it available to the main agent

#### Add a Private Inline Child

Place `[[subagents.subagents]]` immediately after its parent
`[[subagents]]` entry:

```toml
[[subagents]]
name = "research-manager"
description = "Coordinates repository research and planning."
system_prompt = "Delegate focused work and synthesize the results."

[[subagents.subagents]]
name = "repo-planner"
description = "Turns repository findings into an implementation plan."
system_prompt = "Produce a concise, actionable implementation plan."
```

In this configuration:

- `repo-planner` is available to `research-manager`
- `repo-planner` is not exposed directly to the main agent
- the child can use the normal synchronous subagent fields, including `skills`,
  `mcp_servers`, and `model`

Use an inline child when it is an implementation detail of one parent.

#### Reuse a Top-Level Subagent as a Child

Define the child as a normal top-level `[[subagents]]` entry and reference its
name from the parent:

```toml
[[subagents]]
name = "research-manager"
description = "Coordinates repository research and review."
system_prompt = "Delegate research and ask the reviewer to check the result."
nested_subagents = ["reviewer"]

[[subagents]]
name = "reviewer"
description = "Reviews proposed changes for bugs and regressions."
system_prompt = "Return concise findings with actionable file references."
```

In this configuration:

- `reviewer` remains available directly to the main agent
- `research-manager` can also delegate to `reviewer`
- more than one parent can reference the same top-level subagent
- each referenced name must exactly match a top-level synchronous subagent

Use a reference when a role should be shared by the main agent or multiple
parents. A parent can expose several shared children, for example
`nested_subagents = ["planner", "reviewer"]`.

#### Configuration and Inheritance

Inline children accept the same fields as other synchronous subagents:
`name`, `description`, `system_prompt` or `system_prompt_file`, `skills`,
`mcp_servers`, `model`, and their own nested children. A referenced child uses
the configuration from its top-level `[[subagents]]` entry wherever it is
reused. When `model` is omitted, the child continues with its parent/default
model configuration.

#### Limitations and Validation

Nested subagents are synchronous only. Async Agent Protocol subagents must
remain top-level `[[async_subagents]]` entries.

Configuration loading rejects:

- a `nested_subagents` name that does not match a top-level synchronous
  subagent
- duplicate direct child names, including an inline child and a referenced
  child with the same name
- reference cycles such as `manager -> reviewer -> manager`
- a nested child that defines `graph_id`, because that represents an async
  subagent

As a rule of thumb, use an inline child for a parent-private specialist, a
referenced child for a shared synchronous role, and `[[async_subagents]]` for
remote or background Agent Protocol work.
````

- [ ] **Step 2: Parse both tutorial TOML examples**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
import re
import tomllib

readme = Path("README.md").read_text(encoding="utf-8")
section = readme.split("### Nested Subagents", 1)[1]
section = section.split("Main `[agent]` additions:", 1)[0]
examples = re.findall(r"```toml\n(.*?)\n```", section, flags=re.DOTALL)
assert len(examples) == 2, f"expected 2 TOML examples, found {len(examples)}"
for example in examples:
    tomllib.loads(example)
print("validated 2 nested-subagent TOML examples")
PY
```

Expected output:

```text
validated 2 nested-subagent TOML examples
```

- [ ] **Step 3: Review the focused README diff**

Run:

```bash
git diff -- README.md
```

Confirm that:

- the new section appears once beneath the supported-field list
- both configuration forms have independent examples
- the prior five-line summary paragraph is gone
- the following `Main [agent] additions` content is unchanged

- [ ] **Step 4: Check formatting and repository state**

Run:

```bash
git diff --check
git status --short
```

Expected result:

- `git diff --check` produces no output and exits successfully
- `git status --short` lists `README.md` plus the approved design and plan
  documents, with no unrelated changes
