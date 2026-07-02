---
name: chainlit-generative-ui
description: Use this skill in Chainlit when a response would benefit from a generated visual panel, interactive prompt buttons, a compact facts view, a short list, or a small comparison/status table using the render_chainlit_ui tool.
---

# chainlit-generative-ui

Use Chainlit generative UI to make structured results easier to scan while keeping the normal text response.

Working rules:

1. Call `render_chainlit_ui` only when that tool is available in the current tool list. If it is absent, answer normally in text.
2. Use the UI for concise, user-facing structure: summaries, decision cards, status panels, short checklists, small tables, and follow-up actions.
3. Do not render arbitrary JSX, HTML, scripts, or custom component names. This runtime only exposes the whitelisted `GeneratedPanel` component through `render_chainlit_ui`.
4. Put only plain strings in `items`. Do not put objects with `label` and `prompt` there.
5. Put prompt buttons in `actions`. Treat each button as a user-facing follow-up request with a clear label and prompt.
6. For checklist-style panels with follow-up buttons, put the checklist labels in `items` and the clickable follow-ups in `actions`.
7. Keep generated panels compact. Put detailed reasoning, long explanations, code, logs, and large datasets in the text response or files instead.
8. Do not put secrets, hidden reasoning, system messages, or internal tool details in the panel.
9. Reuse `id` only when updating the same panel. Use a stable, short id such as `deployment-summary` or `review-findings`.

Supported `render_chainlit_ui` fields:

- `title`: short required panel title.
- `summary`: optional concise summary.
- `facts`: optional key-value object for compact facts.
- `items`: optional short list of strings. Do not use this field for prompt buttons.
- `table`: optional object with `columns` and `rows`.
- `actions`: optional list of prompt buttons, each with `label` and `prompt`.
- `id`: optional stable panel id. Reusing the id updates the existing panel.

Example tool call:

```json
{
  "title": "Review Summary",
  "summary": "Two blocking issues need fixes before merge.",
  "facts": {
    "Status": "Needs changes",
    "Tests": "Targeted suite passed"
  },
  "items": [
    "Fix stale Chainlit element updates",
    "Add disabled-config coverage"
  ],
  "table": {
    "columns": ["Area", "Result"],
    "rows": [
      ["Runtime", "Tool enabled only when configured"],
      ["Chainlit", "GeneratedPanel rendered as its own message"]
    ]
  },
  "actions": [
    {
      "label": "Show fixes",
      "prompt": "Show me the exact fixes for the blocking review issues."
    },
    {
      "label": "Run tests",
      "prompt": "Run the targeted tests for this change."
    }
  ],
  "id": "review-summary"
}
```

After calling the tool, continue with the normal answer. The panel should complement the answer, not replace it.
