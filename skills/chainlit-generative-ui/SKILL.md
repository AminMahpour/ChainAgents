---
name: chainlit-generative-ui
description: Use this skill in Chainlit when a response would benefit from a generated visual panel, interactive prompt buttons, a compact facts view, a short list, or a small comparison/status table using the render_chainlit_ui tool.
---

# chainlit-generative-ui

Use Chainlit generative UI as the active interaction layer for structured Chainlit turns while
keeping the normal text response.

Working rules:

1. Call `render_chainlit_ui` only when that tool is available in the current tool list. If it is absent, answer normally in text.
2. Default to rendering UI for concise, user-facing structure: summaries, decision cards, status panels, short checklists, small tables, and follow-up actions.
3. Include `actions` when the user may want next steps. Prefer useful continuation buttons such as "Run tests", "Show diff", "Explain config", or "Create PR".
4. Skip UI for simple one-sentence answers, conversational acknowledgements, and cases where a panel would duplicate the text without adding interaction.
5. Do not render arbitrary JSX, HTML, scripts, or custom component names. This runtime only exposes the whitelisted `GeneratedPanel` component through `render_chainlit_ui`.
6. Put only plain strings in `items`. Do not put objects with `label` and `prompt` there.
7. Put prompt buttons in `actions`. Treat each button as a user-facing follow-up request with a clear label and prompt.
8. For checklist-style panels with follow-up buttons, put the checklist labels in `items` and the clickable follow-ups in `actions`.
9. Do not duplicate the same option in `items` and `actions`. If it is clickable, put it only in `actions`.
10. Keep generated panels compact. Put detailed reasoning, long explanations, code, logs, and large datasets in the text response or files instead.
11. Do not put secrets, hidden reasoning, system messages, or internal tool details in the panel.
12. Reuse `id` only when updating the same panel. Use a stable, short id such as `deployment-summary` or `review-findings`.

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
