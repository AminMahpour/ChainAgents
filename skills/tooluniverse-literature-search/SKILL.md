---
name: tooluniverse-literature-search
description: Biomedical literature discovery and evidence synthesis using ToolUniverse PubMed, PubMed Central, Europe PMC, Crossref, MeSH, and PubTator tools. Use when asked to find papers, summarize current evidence, build citation lists, inspect abstracts/full text, find guidelines, map citations, or extract biomedical entities from articles.
---

# ToolUniverse Literature Search

## Overview

Use ToolUniverse as the primary source for biomedical and scholarly literature. Prefer structured literature tools over generic web search when the user asks for papers, PMIDs, abstracts, guidelines, citation trails, or evidence summaries.

For exact tool schemas, call `get_tool_info` before `execute_tool` when arguments are not already clear. See `references/tool-map.md` for the core tool list.

## Tool Call Rule

Run ToolUniverse domain tools only through `execute_tool`. Do not call a tool named `execute`, and do not call domain tool names such as `PubMed_search_articles` directly unless they are actually exposed as standalone MCP tools.

If `list_tools`, `get_tool_info`, or `execute_tool` are unavailable in the current tool list, stop and report that the ToolUniverse MCP server is not attached or not running for this agent.

Use this shape:

```json
{
  "tool_name": "PubMed_search_articles",
  "arguments": {
    "query": "topic",
    "limit": 10
  }
}
```

## Workflow

1. Frame the question.
   - Convert broad requests into search concepts: population, condition, intervention or exposure, comparator, outcome, date range, and study type.
   - Use synonyms, brand/generic names, MeSH terms, and abbreviations where appropriate.
   - State the search date in the answer when recency matters.

2. Search broadly, then narrow.
   - Start with `PubMed_search_articles` for biomedical literature.
   - Set `include_abstract: true` for evidence synthesis unless the user only needs identifiers.
   - Use `mindate`, `maxdate`, and `datetype: "pdat"` for date-bounded searches.
   - Use `MeSH_search_descriptors` when the concept may have controlled vocabulary variants.
   - Use `PMC_search_papers` or `EuropePMC_get_full_text` when full text is needed and likely open access.
   - Use `Crossref_search_works` for non-biomedical scholarly coverage or DOI verification.

3. Deepen high-value records.
   - Use `PubMed_get_article` for full metadata, abstracts, MeSH terms, affiliations, grants, and references.
   - Use `PubMed_get_related` to expand from a key PMID.
   - Use `PubMed_get_cited_by` or `EPMC_get_citations` to find newer papers that cite a key article.
   - Use `EPMC_get_references` to inspect foundational sources cited by a key paper.
   - Use `PubTator3_get_annotations` for genes, diseases, chemicals, species, mutations, or cell lines discussed in PMIDs.

4. Screen results explicitly.
   - Prioritize systematic reviews, randomized trials, clinical guidelines, large observational studies, and recent high-quality primary research.
   - Keep negative, null, or conflicting studies when they affect the conclusion.
   - Do not treat abstracts as full-text evidence when methods or results detail is needed.
   - Do not infer clinical recommendations from preclinical, in vitro, animal, or case-report evidence without saying so.

5. Report with traceability.
   - Include the search query, sources searched, date range, and major filters.
   - Cite PMIDs, DOIs, PMCIDs, or URLs returned by the tools.
   - Separate findings by evidence type and confidence.
   - Include limitations: database coverage, missing full text, small sample sizes, heterogeneity, and recency gaps.

## Common Calls

Call `execute_tool` to search recent PubMed records with abstracts:

```json
{
  "tool_name": "PubMed_search_articles",
  "arguments": {
    "query": "(semaglutide OR GLP-1) AND cardiovascular outcomes",
    "limit": 20,
    "mindate": "2020",
    "datetype": "pdat",
    "include_abstract": true,
    "sort": "pub_date"
  }
}
```

Call `execute_tool` to fetch a full PubMed record:

```json
{
  "tool_name": "PubMed_get_article",
  "arguments": {
    "pmid": "12345678"
  }
}
```

Call `execute_tool` to extract biomedical entities from articles:

```json
{
  "tool_name": "PubTator3_get_annotations",
  "arguments": {
    "pmids": "33205991,34234088",
    "concepts": "gene,disease,chemical,mutation"
  }
}
```

## Output Template

Use this structure for literature reviews unless the user asks for another format:

- Search strategy: sources, query strings, date filters, and result counts.
- Key studies: title, year, design, population/model, intervention/exposure, main result, PMID/DOI.
- Synthesis: what the evidence supports, what conflicts, and what remains uncertain.
- Applicability: whether evidence is human, clinical, preclinical, observational, or mechanistic.
- Limitations: missing full text, incomplete abstracts, database coverage, and recency.

## Safety

For medical questions, provide evidence summaries rather than personal medical advice. Recommend clinician input for diagnosis, treatment decisions, medication changes, pregnancy, pediatrics, severe symptoms, or urgent safety concerns.
