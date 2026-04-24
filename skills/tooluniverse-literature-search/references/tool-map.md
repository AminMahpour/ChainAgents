# ToolUniverse Literature Tool Map

Use these exact tool names with `execute_tool`. Call `get_tool_info` first when a parameter detail is uncertain. Do not call `execute`; the execution wrapper is named `execute_tool`.

## Discovery

- `find_tools`: discover newer or more specific ToolUniverse tools.
- `list_tools`: list ToolUniverse tools or categories when the available surface needs to be enumerated.
- `get_tool_info`: retrieve current descriptions and schemas.
- `execute_tool`: run the selected domain tool by exact name. Put the domain tool name in `tool_name` and its parameters in `arguments`.

## Core Literature Tools

- `PubMed_search_articles`: search PubMed; supports query, limit, dates, sort, and abstracts.
- `PubMed_get_article`: retrieve full metadata and abstract for one or more PMIDs.
- `PubMed_get_related`: find articles related to a PMID.
- `PubMed_get_cited_by`: find PubMed articles citing a PMID.
- `PubMed_get_links`: retrieve PubMed LinkOut and full-text links.
- `PubMed_Guidelines_Search`: search guideline and practice guideline publication types.
- `PMC_search_papers`: search PubMed Central full-text archive.
- `EuropePMC_get_full_text`: retrieve structured full text for open-access PMC articles.
- `EPMC_get_citations`: find Europe PMC citing articles for a PMID.
- `EPMC_get_references`: retrieve bibliography entries cited by a PMID.
- `EPMC_get_text_mined_annotations`: retrieve Europe PMC annotations from a PMID or PMCID.
- `PubTator3_get_annotations`: extract normalized biomedical entities from PMIDs.
- `MeSH_search_descriptors`: find MeSH descriptors and controlled vocabulary.
- `Crossref_search_works`: search scholarly works and verify DOI metadata.

## Query Notes

- PubMed supports Boolean operators and field tags, for example `Smith J[Author]`, `diabetes[MeSH]`, and `Nature[Journal]`.
- Use `include_abstract: true` when the answer depends on study details.
- Use `sort: "pub_date"` for latest work and `sort: "relevance"` for best match.
- Use `mindate`, `maxdate`, and `datetype: "pdat"` for publication-date filters.
- Use multiple searches when synonyms, biomarkers, brand names, or old terminology may split the literature.
