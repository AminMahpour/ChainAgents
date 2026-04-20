# ToolUniverse Drug Safety Tool Map

Use these exact tool names with `execute_tool`. Call `get_tool_info` first when a parameter detail is uncertain. Do not call `execute`; the execution wrapper is named `execute_tool`.

## Discovery

- `find_tools`: discover newer or more specific ToolUniverse tools.
- `list_tools`: list ToolUniverse tools or categories when the available surface needs to be enumerated.
- `get_tool_info`: retrieve current descriptions and schemas.
- `execute_tool`: run the selected domain tool by exact name. Put the domain tool name in `tool_name` and its parameters in `arguments`.

## FDA Labels and DailyMed

- `FDA_get_drug_label`: complete FDA-approved prescribing information by drug name.
- `FDA_search_drug_labels`: label search by drug or indication.
- `FDA_get_drug_label_info_by_field_value`: fielded label lookup with selected returned sections.
- `OpenFDA_search_drug_labels`: openFDA SPL label search with Lucene syntax.
- `FDA_get_boxed_warning_info_by_drug_name`: boxed warning and adverse effects by drug.
- `FDA_get_drug_names_by_boxed_warning`: drugs matching boxed warning text.
- `FDA_get_precautions_by_drug_name`: precautions by drug.
- `FDA_get_general_precautions_by_drug_name`: general precautions by drug.
- `FDA_get_recent_changes_by_drug_name`: recent major label changes.
- `FDA_get_effective_time_by_drug_name`: label effective time.
- `DailyMed_get_spl_by_setid`: full SPL XML by set ID.
- `DailyMed_parse_contraindications`: structured contraindications by set ID or drug name.
- `fda_pharmacogenomic_biomarkers`: FDA pharmacogenomic biomarker table by drug or biomarker.

## SIDER

- `SIDER_search_drug`: find a SIDER drug ID by name.
- `SIDER_get_drug_side_effects`: label-derived side effects and frequency data.
- `SIDER_search_side_effect`: find a MedDRA/UMLS side-effect code by name.
- `SIDER_get_drugs_for_side_effect`: reverse lookup drugs associated with an effect.

## FAERS and openFDA Events

- `FAERS_search_adverse_event_reports`: case-level FAERS reports by drug.
- `FAERS_search_serious_reports_by_drug`: serious case-level reports by drug.
- `FAERS_search_reports_by_drug_and_reaction`: case-level reports for a drug and MedDRA preferred term.
- `FAERS_search_reports_by_drug_and_indication`: reports for a drug and indication.
- `FAERS_search_reports_by_drug_and_outcome`: reports for a drug and outcome.
- `FAERS_search_reports_by_drug_combination`: reports involving multiple drugs.
- `FAERS_count_seriousness_by_drug_event`: serious versus non-serious counts.
- `FAERS_count_outcomes_by_drug_event`: outcome counts.
- `FAERS_count_patient_age_distribution`: age distribution.
- `FAERS_count_country_by_drug_event`: occurrence country counts.
- `FAERS_count_reportercountry_by_drug_event`: reporter country counts.
- `FAERS_count_drug_routes_by_event`: route counts.
- `FAERS_count_death_related_by_drug`: death-related count.
- `FAERS_count_additive_adverse_reactions`: reaction counts across multiple drugs.
- `FAERS_count_additive_administration_routes`: route counts across multiple drugs.
- `FAERS_count_additive_event_reports_by_country`: country counts across multiple drugs.
- `FAERS_compare_drugs`: compare two drugs for the same adverse event using disproportionality metrics.
- `OpenFDA_search_drug_events`: openFDA FAERS search or counts with convenience drug and reaction parameters.

## Clinical Trial Safety

- `extract_clinical_trial_adverse_events`: serious, other, all, or specific adverse events by NCT IDs.
- `get_clinical_trial_outcome_measures`: inspect outcome measures before extracting outcomes or events.

## Interpretation Notes

- FDA label and DailyMed sections are authoritative for approved labeling, not necessarily the latest clinical evidence.
- SIDER is label-derived and useful for known adverse reactions and frequencies when available.
- FAERS and openFDA are spontaneous reports. They are useful for signal detection and case examples, not incidence or causality.
- Compare drugs only for the same adverse event term, and report the exact term used.
