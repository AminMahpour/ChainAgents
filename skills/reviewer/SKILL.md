---
name: reviewer
description: Use this skill when the user wants a code review, bug hunt, regression check, or test-gap assessment for changes in the current repository.
---

# reviewer

Use this skill when the user asks for any of the following:

- a review of code changes
- a bug hunt or regression check
- missing tests or risky edge cases
- a second pass on whether an implementation is safe to ship

Working rules:

1. Read the relevant files first and inspect the diff when one is available.
2. Focus on correctness, behavior changes, edge cases, and failure handling.
3. Prefer concrete findings with file references over broad commentary.
4. Flag missing or weak tests when behavior changed or a bug fix is not verified.
5. If there are no findings, say that explicitly and note any remaining risk or test gap.

Expected output style:

- List findings first, ordered by severity.
- Each finding should explain the impact and point to the file to inspect.
- Keep summaries short and implementation-oriented.

Example requests where this skill should be used:

- "Review these changes for bugs."
- "Do a regression pass on the Chainlit bridge."
- "What tests are missing for this feature?"
- "Is this safe to merge?"
