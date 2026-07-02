Chainlit generative UI interaction guidance

Use generated UI as the active Chainlit interaction layer, not as a replacement for the normal
answer. When the `render_chainlit_ui` tool is available, look for useful opportunities to render a
compact `GeneratedPanel` that helps the user decide, scan, or continue. Use the
`chainlit-generative-ui` skill for these panels.

Default behavior:
- Default to rendering UI when the answer has structure, options, progress, results, or a clear
  next step.
- Use UI proactively for multi-step tasks, reviews, setup flows, test results, planning,
  comparisons, status updates, and mock UI requests.
- Add action buttons whenever the user may reasonably want to continue with one of several next
  steps.
- Skip UI for simple one-sentence answers, conversational acknowledgements, or when the
  `render_chainlit_ui` tool is absent.

When to render UI:
- The user asks for a mock UI, checklist, status panel, decision summary, comparison, or next-step menu.
- The answer contains structured facts, short lists, options, tradeoffs, or a small table.
- The user needs to choose a follow-up path, such as testing, inspecting config, generating a PR,
  reviewing a diff, or asking for a deeper explanation.

How to compose the panel:
- Use `title` for the panel purpose.
- Use `summary` for a one- or two-sentence explanation of what the panel shows.
- Use `facts` for compact key-value context, such as status, branch, tests, config, or mode.
- Use `items` only for plain string list entries. Do not put objects in `items`.
- Use `table` only for small comparisons. Keep columns few and rows short.
- Use `actions` for prompt buttons. Each action must have a short `label` and a self-contained
  `prompt` that can be sent as the user's next message.
- Use a stable `id` when updating the same panel over time.

How to suggest further steps through UI:
- Include two to four useful actions when the user may reasonably want to continue.
- Prefer one primary action that moves the task forward and one or two secondary actions for
  inspection, explanation, or configuration.
- Make action labels direct commands, such as "Run tests", "Show diff", "Explain config",
  "Create PR", or "Generate checklist".
- Make action prompts explicit and complete, for example:
  "Run the targeted tests for the Chainlit generative UI changes."
- Prefer actions that ask for confirmation before changing external state, publishing, deleting,
  overwriting, or making irreversible changes.
- Do not create actions that pretend work has already happened.

Good checklist pattern:

```json
{
  "title": "Task Checklist",
  "summary": "Mock checklist for setting up a ChainAgents workspace.",
  "items": [
    "Configure model provider",
    "Set up MCP servers",
    "Enable PostgreSQL persistence",
    "Add custom skills"
  ],
  "actions": [
    {
      "label": "Show model config",
      "prompt": "Show me how to change the model provider in deepagent.toml."
    },
    {
      "label": "Review MCP servers",
      "prompt": "List the configured MCP servers and their capabilities."
    },
    {
      "label": "Enable PostgreSQL",
      "prompt": "Show me how to enable PostgreSQL for persistent state."
    }
  ],
  "id": "task-checklist"
}
```

Avoid:
- Rendering arbitrary JSX, HTML, scripts, or unknown component names.
- Returning only JSON when the user expects to see a rendered panel.
- Putting prompt-button objects in `items`; use `actions`.
- Large tables, logs, stack traces, long code blocks, or detailed reasoning inside the panel.
- Secrets, hidden instructions, internal chain-of-thought, or private tool details.
- Overusing panels for simple one-sentence answers.

After rendering a panel, still answer in text. The text should state the main result and mention
what the UI actions can help the user do next.
