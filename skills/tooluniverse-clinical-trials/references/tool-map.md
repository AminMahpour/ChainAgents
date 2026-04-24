# ToolUniverse Clinical Trials Tool Map

Use these exact tool names with `execute_tool`. Call `get_tool_info` first when a parameter detail is uncertain. Do not call `execute`; the execution wrapper is named `execute_tool`.

## Discovery

- `find_tools`: discover newer or more specific ToolUniverse tools.
- `list_tools`: list ToolUniverse tools or categories when the available surface needs to be enumerated.
- `get_tool_info`: retrieve current descriptions and schemas.
- `execute_tool`: run the selected domain tool by exact name. Put the domain tool name in `tool_name` and its parameters in `arguments`.

## Search Tools

- `ClinicalTrials_search_studies`: flexible ClinicalTrials.gov search by condition, intervention, keyword, status, phase, and type.
- `search_clinical_trials`: primary discovery tool for condition, intervention, keyword, status, and pagination.
- `ClinicalTrials_search_by_intervention`: intervention-focused search.
- `ClinicalTrials_search_by_sponsor`: sponsor-focused search.
- `ClinicalTrials_get_field_values`: inspect valid or common values for status, phase, study type, sponsor class, or intervention type.
- `ClinicalTrials_get_database_stats`: database-level aggregate counts.

## Detail Tools

- `ClinicalTrials_get_study`: full study protocol by NCT ID.
- `get_clinical_trial_descriptions`: brief or full descriptions for NCT IDs.
- `get_clin_tria_cond_and_inte`: conditions, interventions, arms, and groups.
- `get_clinical_trial_eligibility_criteria`: inclusion and exclusion criteria.
- `get_clinical_trial_outcome_measures`: primary, secondary, or all outcome measures.
- `get_clinical_trial_locations`: study locations.
- `get_clinical_trial_references`: linked references and publications.

## Results Tools

- `extract_clinical_trial_outcomes`: posted outcome results for NCT IDs.
- `extract_clinical_trial_adverse_events`: posted adverse event results for NCT IDs.

## Query Notes

- Status values commonly include `RECRUITING`, `NOT_YET_RECRUITING`, `ACTIVE_NOT_RECRUITING`, `COMPLETED`, `ENROLLING_BY_INVITATION`, `SUSPENDED`, `TERMINATED`, and `WITHDRAWN`.
- Phase values commonly include `EARLY_PHASE1`, `PHASE1`, `PHASE2`, `PHASE3`, `PHASE4`, and `NA`.
- Preserve page tokens when a result set is paginated.
- Absence of posted results in ClinicalTrials.gov output is not evidence of no effect or no adverse event.
