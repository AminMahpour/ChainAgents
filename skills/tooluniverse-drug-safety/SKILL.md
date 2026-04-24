---
name: tooluniverse-drug-safety
description: Drug safety, pharmacovigilance, label, adverse event, boxed warning, contraindication, and side-effect review using ToolUniverse FAERS, SIDER, openFDA, FDA label, DailyMed, and clinical trial adverse-event tools. Use when asked about medication risks, side effects, adverse reactions, safety signals, drug comparisons, or serious-event reports.
---

# ToolUniverse Drug Safety

## Overview

Use ToolUniverse safety tools to combine official labeling, known label-derived side effects, spontaneous adverse event reports, and trial adverse events. Always separate what is established in official labeling from what is a signal in reports.

For exact schemas, call `get_tool_info` before `execute_tool` when needed. See `references/tool-map.md` for the core tool list.

## Tool Call Rule

Run ToolUniverse domain tools only through `execute_tool`. Do not call a tool named `execute`, and do not call domain tool names such as `FDA_get_drug_label` directly unless they are actually exposed as standalone MCP tools.

If `list_tools`, `get_tool_info`, or `execute_tool` are unavailable in the current tool list, stop and report that the ToolUniverse MCP server is not attached or not running for this agent.

Use this shape:

```json
{
  "tool_name": "FDA_get_drug_label",
  "arguments": {
    "drug_name": "warfarin"
  }
}
```

## Workflow

1. Normalize the request.
   - Identify generic name, brand name, drug class, combination products, route, dose context, patient group, adverse event terms, and comparator drugs.
   - Prefer generic drug names for FAERS, SIDER, labels, and comparisons.
   - Use MedDRA preferred terms where available. For openFDA, remember reaction terms often use British spelling, such as `haemorrhage`.

2. Start with authoritative labeling.
   - Use `FDA_get_drug_label` or `FDA_search_drug_labels` for official prescribing information.
   - Use `FDA_get_drug_label_info_by_field_value` when specific sections are needed.
   - Return fields such as `boxed_warning`, `contraindications`, `warnings`, `warnings_and_cautions`, `precautions`, `adverse_reactions`, `drug_interactions`, `use_in_specific_populations`, and `recent_major_changes`.
   - Use `FDA_get_boxed_warning_info_by_drug_name`, `FDA_get_precautions_by_drug_name`, or `FDA_get_recent_changes_by_drug_name` for targeted checks.
   - Use `DailyMed_parse_contraindications` when structured contraindications are needed.

3. Add known side-effect context.
   - Use `SIDER_search_drug` to resolve a drug if needed.
   - Use `SIDER_get_drug_side_effects` for label-derived side effects and frequency ranges.
   - Use `SIDER_search_side_effect` and `SIDER_get_drugs_for_side_effect` for reverse lookup by adverse effect.

4. Analyze FAERS and openFDA reports carefully.
   - Use `FAERS_count_seriousness_by_drug_event`, `FAERS_count_outcomes_by_drug_event`, `FAERS_count_patient_age_distribution`, and country or route count tools for aggregate views.
   - Use `FAERS_search_adverse_event_reports`, `FAERS_search_serious_reports_by_drug`, or reaction/indication/outcome-specific FAERS tools for case-level summaries.
   - Use `OpenFDA_search_drug_events` when Lucene queries, reaction counts, or specific openFDA report fields are useful.
   - Use `FAERS_compare_drugs` for ROR, PRR, or IC comparisons of two drugs for the same adverse event.
   - Use filters sparingly; too many filters can create false zero-result interpretations.

5. Add trial and literature context when needed.
   - If NCT IDs are available, use `extract_clinical_trial_adverse_events`.
   - If the user asks for published evidence, use the literature-search skill to retrieve PubMed, PMC, and citation context.
   - Compare trial adverse events, label adverse reactions, and spontaneous reports as different evidence types.

## Common Calls

Call `execute_tool` to fetch official FDA label information:

```json
{
  "tool_name": "FDA_get_drug_label",
  "arguments": {
    "drug_name": "warfarin"
  }
}
```

Call `execute_tool` to fetch selected label sections by generic name:

```json
{
  "tool_name": "FDA_get_drug_label_info_by_field_value",
  "arguments": {
    "field": "openfda.generic_name",
    "field_value": "WARFARIN",
    "return_fields": [
      "openfda.brand_name",
      "openfda.generic_name",
      "boxed_warning",
      "contraindications",
      "warnings",
      "adverse_reactions",
      "drug_interactions"
    ],
    "limit": 5
  }
}
```

Call `execute_tool` to count common FAERS reactions through openFDA:

```json
{
  "tool_name": "OpenFDA_search_drug_events",
  "arguments": {
    "drug_name": "warfarin",
    "count": "patient.reaction.reactionmeddrapt.exact",
    "limit": 25
  }
}
```

Call `execute_tool` to retrieve SIDER label-derived side effects:

```json
{
  "tool_name": "SIDER_get_drug_side_effects",
  "arguments": {
    "operation": "get_side_effects",
    "drug_name": "metformin",
    "limit": 100
  }
}
```

## Output Template

Use this structure for safety reviews:

- Drug and scope: generic/brand name, route or population if relevant, event terms used.
- Official label: boxed warnings, contraindications, warnings, precautions, adverse reactions, interactions.
- Known side effects: SIDER or label-derived frequencies when available.
- FAERS/openFDA: report counts, seriousness, outcomes, demographics, top reactions, and case examples if requested.
- Trial adverse events: NCT IDs, arms, serious and other adverse events if posted.
- Interpretation: what is established, what is a signal, what is uncertain, and what needs clinician review.
- Caveats: FAERS/openFDA reports are spontaneous reports and cannot establish incidence or causality.

## Safety

Do not provide dosing, stopping, rechallenge, or medication substitution advice as a directive. For severe symptoms, suspected serious adverse reactions, pregnancy, pediatrics, complex comorbidity, or medication changes, recommend urgent clinician, pharmacist, poison control, or emergency care as appropriate.
