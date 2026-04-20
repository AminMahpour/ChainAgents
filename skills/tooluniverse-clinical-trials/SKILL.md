---
name: tooluniverse-clinical-trials
description: ClinicalTrials.gov discovery and trial evidence extraction using ToolUniverse tools. Use when asked to find trials for a condition, drug, sponsor, biomarker, or intervention; retrieve NCT protocols, status, phase, eligibility, outcomes, locations, references, posted results, or trial adverse events.
---

# ToolUniverse Clinical Trials

## Overview

Use ToolUniverse ClinicalTrials.gov tools to move from a condition, intervention, sponsor, or NCT ID to a traceable trial summary. Distinguish registered protocol information from posted results and extracted outcomes.

For exact schemas, call `get_tool_info` before `execute_tool` when needed. See `references/tool-map.md` for the core tool list.

## Tool Call Rule

Run ToolUniverse domain tools only through `execute_tool`. Do not call a tool named `execute`, and do not call domain tool names such as `ClinicalTrials_search_studies` directly unless they are actually exposed as standalone MCP tools.

If `list_tools`, `get_tool_info`, or `execute_tool` are unavailable in the current tool list, stop and report that the ToolUniverse MCP server is not attached or not running for this agent.

Use this shape:

```json
{
  "tool_name": "ClinicalTrials_search_studies",
  "arguments": {
    "query_cond": "condition",
    "page_size": 10
  }
}
```

## Workflow

1. Choose the entry point.
   - If the user provides NCT IDs, skip search and fetch details directly.
   - If the user provides a disease or condition, use `ClinicalTrials_search_studies` or `search_clinical_trials` with `query_cond` or `condition`.
   - If the user provides a drug, device, biologic, or procedure, use `ClinicalTrials_search_by_intervention` or `search_clinical_trials` with `intervention`.
   - If the user provides a sponsor, use `ClinicalTrials_search_by_sponsor`.

2. Search with conservative filters.
   - Use exact ClinicalTrials.gov status values such as `RECRUITING`, `NOT_YET_RECRUITING`, `ACTIVE_NOT_RECRUITING`, `COMPLETED`, `TERMINATED`, `WITHDRAWN`, and `SUSPENDED`.
   - Use phase values such as `EARLY_PHASE1`, `PHASE1`, `PHASE2`, `PHASE3`, `PHASE4`, and `NA`.
   - Avoid over-filtering on the first pass; broad searches are better for missing fewer trials.
   - Preserve `next_page_token` or `pageToken` when pagination is needed.

3. Retrieve protocol details.
   - Use `ClinicalTrials_get_study` for a complete protocol record.
   - Use `get_clinical_trial_descriptions` for brief or full descriptions across multiple NCT IDs.
   - Use `get_clin_tria_cond_and_inte` for conditions, interventions, arms, and groups.
   - Use `get_clinical_trial_eligibility_criteria` for inclusion and exclusion criteria.
   - Use `get_clinical_trial_outcome_measures` before extracting results so the measure names are clear.
   - Use `get_clinical_trial_locations` for recruiting site geography.
   - Use `get_clinical_trial_references` for linked publications or citations.

4. Extract results only when appropriate.
   - Use `extract_clinical_trial_outcomes` for posted outcome results.
   - Use `extract_clinical_trial_adverse_events` for posted adverse event results.
   - If no results are returned, say that posted results were not available from the tool output; do not imply the outcome did not occur.
   - For adverse events, first query `adverse_event_type: "serious"` as a sanity check, then specific MedDRA-like event names if needed.

5. Cross-check important claims.
   - Use linked references or PMIDs with the literature-search skill when publications are needed.
   - Use drug-safety skill tools when trial adverse events need label, FAERS, or SIDER context.
   - Call `ClinicalTrials_get_field_values` when unsure which field values are valid for status, phase, sponsor class, or intervention type.

## Common Calls

Call `execute_tool` to search active or recruiting trials for a drug and condition:

```json
{
  "tool_name": "ClinicalTrials_search_studies",
  "arguments": {
    "query_cond": "non-small cell lung cancer",
    "query_intr": "osimertinib",
    "filter_status": "RECRUITING,ACTIVE_NOT_RECRUITING",
    "filter_phase": "PHASE2,PHASE3",
    "page_size": 25
  }
}
```

Call `execute_tool` to get full protocol details:

```json
{
  "tool_name": "ClinicalTrials_get_study",
  "arguments": {
    "nct_id": "NCT04280705"
  }
}
```

Call `execute_tool` to extract trial adverse events:

```json
{
  "tool_name": "extract_clinical_trial_adverse_events",
  "arguments": {
    "nct_ids": ["NCT04280705"],
    "adverse_event_type": "serious"
  }
}
```

## Output Template

Use a table for multi-trial comparisons:

- NCT ID and title
- condition and intervention
- phase and study type
- overall status
- enrollment and sponsor
- arms or groups
- primary outcomes
- key eligibility constraints
- locations, if relevant
- posted results and adverse event availability
- linked references

Then summarize the practical takeaways and evidence gaps. Clearly label protocol-only information, posted ClinicalTrials.gov results, and publication-derived evidence.

## Safety

Do not present trial eligibility as medical advice or a guarantee of enrollment. For patient-specific trial matching, state that trial staff or a clinician must confirm eligibility, location availability, and current recruitment status.
