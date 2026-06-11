---
name: pharma-pipeline
description: Use this skill when the user wants to search for drug pipelines, investigational compounds, or medicine information from pharmaceutical and biotech company websites.
---

# pharma-pipeline

Use this skill when the user asks for any of the following:

- pipeline data (investigational drugs, clinical-stage assets) from a specific company
- approved medicines / commercial products list from a pharma/biotech site
- drug information by indication or therapeutic area across companies
- competitive landscape comparison between multiple pharma pipelines
- "What is [Company] working on?" or "[Drug name] pipeline" queries

## Working rules

### 1. Identify the correct URL first

Pharmaceutical company sites use many different URL patterns for their pipeline pages:

| Common pattern | Example |
|---|---|
| `/science/pipeline.html` | AbbVie, Merck-style nav |
| `/research-development/pipeline/` or `R&D/pipeline/` | Alkermes style |
| `/pipeline/index.jsp` (older sites) | Legacy JSP-based pipelines |
| `/our-science/pipeline/` | Some biotechs |

**Strategy:** Fetch the homepage first to find pipeline links in navigation menus. Look for:
- "R&D", "Science", "Research & Development" nav sections
- Links labeled "Pipeline", "Our Pipeline", "Drug Discovery"
- Sitemap or footer quick-links mentioning "pipeline"

If no clear link exists, try these fallback URLs (in order):
1. `https://www.[company].com/science/pipeline.html`
2. `https://www.[company].com/research-development/pipeline/`
3. `https://www.[company].com/R&D/pipeline/index.jsp`
4. `https://www.[company].com/pipeline/`

### 2. Use stealthy fetcher for all requests

Pharmaceutical sites often have bot protection (Cloudflare, Akamai). Always use the **stealthy_fetch** tool:

```python
# Parameters to always include:
extraction_type="markdown"       # Cleanest output format
main_content_only=true           # Skip nav/footer noise
solve_cloudflare=true            # Bypass Cloudflare challenges
network_idle=true                # Wait for JS-rendered content (pipeline tables load dynamically)
timeout=45000                    # 45s — pipeline pages are heavy with data
```

### 3. Fetch supporting pages in parallel

After finding the main pipeline page, also fetch these to get complete coverage:

| Page | Why it matters |
|---|---|
| `/medicines/` or `proprietary-medicines/` | Commercial products not on the R&D pipeline |
| `/disease-areas/` | Therapeutic focus areas and context |
| Sitemap (`/sitemap`) | Discover hidden sub-pages with product data |

Use **bulk_stealthy_fetch** when fetching multiple URLs simultaneously.

### 4. Extract structured information from raw HTML/markdown

Pipeline pages are typically interactive tables rendered by JavaScript. From the markdown output, extract:

- **Molecule / Asset name** (code + trade name if available)
- **Target mechanism** (e.g., PD-1, IL-23, orexin 2 receptor agonist)
- **Molecule type** (Biologic, Small Molecule, Antibody, ADC, Gene Therapy, Device)
- **Indication(s)** — list all listed indications per molecule
- **Phase / Status** (Discovery, Preclinical, Phase 1/2/3, Submitted, Approved, Launched)
- **Region markers** (US, EU, JA = Japan, OUS = Other US regions or Worldwide)

### 5. Present results in a structured format

Organize output by:
1. **Company overview stats** — total compounds, R&D investment, late-stage count
2. **Pipeline table grouped by therapeutic area / disease focus** (Immunology, Oncology, Neuroscience, etc.)
3. **Commercial medicines section** separately from investigational pipeline
4. **Key highlights** at the end

Use markdown tables for each molecule group with columns: Molecule | Type | Target | Indication(s) | Phase/Status

### 6. Note data freshness and limitations

Always include in your output:
- The date stated on the page (e.g., "Updated May 11, 2026")
- A disclaimer that pipeline status changes frequently
- Any regions/markets noted for each product's approval or trial phase

## Expected output style

```markdown
## [Company Name] Pipeline Summary

**~X compounds** | **$YB R&D investment (YEAR)** | **Updated DATE**

---

### Therapeutic Area (~N entries)

| Molecule | Type | Target | Indication(s) | Phase/Status |
|----------|------|--------|---------------|--------------|
| ...      | ...  | ...    | ...           | ...          |

### Commercial Medicines (~M products)

| Product | Active Ingredient | Indication(s) | Notes |
|---------|-------------------|---------------|-------|
| ...     | ...               | ...           | ...   |

**Last updated: DATE from source site.** Pipeline status is subject to change.
```

## Example requests where this skill should be used

- "What's AbbVie's pipeline?"
- "Show me all drugs in Phase 3 for Merck"
- "Compare the neuroscience pipelines of Alkermes and Biogen"
- "Find approved medicines from Gilead Sciences"
- "What is Pfizer researching for Alzheimer's?"
- "Pull the oncology pipeline from Novartis"

## Common pitfalls to avoid

1. **Don't assume URL patterns** — always verify by checking navigation on the homepage first. Sites like AbbVie (`/our-innovation/drug-discovery-development/pipeline/index.jsp`) use non-standard paths that fail with 404s if guessed wrong.
2. **Always set `network_idle=true`** — pipeline data is loaded via JavaScript APIs; without it you'll only get the shell page structure.
3. **Don't miss commercial products** — many companies list approved medicines on a separate `/medicines/` or `/products/` page, not in the R&D pipeline section. Always fetch both.
4. **Handle pagination/filtering** — some sites (like AbbVie) have filter buttons for Phase/Molecule Type that dynamically load data. The main unfiltered view typically shows all entries; if it doesn't, try fetching with common query params like `?pipelineType=all&pipelinePhase=ALL`.
