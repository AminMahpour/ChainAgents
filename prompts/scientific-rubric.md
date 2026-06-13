# ToolUniverse Scientific Tool-Use Rubric

A response is satisfactory only if it is scientifically grounded, tool-aware,
transparent about evidence quality, and appropriate for research support. The
response must not present tool outputs as definitive clinical, regulatory, or
experimental truth without human expert review.

## 1. Task Understanding

- Correctly identifies the scientific task type: literature review, target or
  tractability assessment, clinical development, translational analysis,
  biomarker review, drug safety, trial design, or related task.
- States the biological, chemical, clinical, or translational question being
  answered.
- Identifies the relevant entity types: disease, target, pathway, gene,
  variant, compound, modality, indication, patient population, endpoint,
  comparator, or trial phase.
- Does not answer a broader or narrower question than the user asked unless
  explicitly justified.

## 2. Tool Selection and Use

- Selects tools that match the task objective and explains why those tools or
  data sources are relevant.
- Uses multiple independent sources when the question requires evidence
  triangulation.
- Distinguishes retrieval tools, prediction tools, annotation tools, clinical
  tools, and summarization tools.
- Reports tool names, key inputs, key outputs, and any failed or unavailable
  tool calls.
- Does not fabricate tool results, citations, database entries, identifiers,
  scores, or study outcomes.
- If a tool result conflicts with another source, surfaces the conflict instead
  of hiding it.

## 3. Evidence Quality and Provenance

- Cites specific evidence sources where possible: papers, databases, clinical
  trials, regulatory labels, guidelines, or tool outputs.
- Separates primary evidence from reviews, predictions, preprints, abstracts,
  and inferred claims.
- Notes evidence recency and whether the answer may require updated database or
  literature checks.
- Evaluates strength of evidence using clear categories such as strong,
  moderate, weak, conflicting, or insufficient.
- Does not overstate findings from single studies, animal models, in vitro
  assays, association studies, or computational predictions.
- Clearly marks hypotheses, predictions, and extrapolations as
  non-confirmatory.

## 4. Literature Review Quality

- Defines the scope of the review: entities, time range if relevant, databases
  searched, inclusion focus, and exclusion rationale.
- Summarizes the most relevant studies rather than listing papers without
  synthesis.
- Compares agreement and disagreement across studies.
- Identifies key mechanisms, populations, endpoints, assays, and limitations.
- Highlights gaps in the literature and what evidence would reduce uncertainty.
- Avoids cherry-picking favorable studies while ignoring negative, null, or
  contradictory findings.

## 5. Target Tractability and Druggability

- Assesses biological rationale: disease relevance, genetics, pathway position,
  expression, causal evidence, and phenotype links.
- Assesses therapeutic tractability: known ligands, binding pockets, modality
  fit, antibody accessibility, degrader feasibility, RNA or gene therapy
  feasibility, and other relevant modality constraints.
- Reviews available chemical, structural, omics, perturbation, and phenotypic
  evidence.
- Discusses safety risks: tissue expression, essentiality, paralogs, pathway
  liabilities, on-target toxicity, and patient stratification.
- Distinguishes target validation from target tractability.
- Provides a clear confidence level and the main reasons for that confidence.

## 6. Clinical Development Assessment

- Identifies the intended indication, line of therapy, target population, and
  clinical context.
- Reviews current standard of care, unmet need, competitor landscape, and
  differentiation hypothesis.
- Discusses feasible endpoints, biomarkers, inclusion and exclusion criteria,
  and trial phase considerations.
- Notes safety, tolerability, drug-drug interaction, contraindication, and
  monitoring considerations when relevant.
- Distinguishes preclinical promise from clinical evidence.
- Does not give patient-specific medical advice or treatment recommendations
  unless explicitly framed as informational and requiring clinician review.

## 7. Quantitative and Predictive Claims

- Reports units, thresholds, confidence intervals, sample sizes, effect sizes,
  model scores, or assay conditions when available.
- Explains what prediction scores mean and what they do not mean.
- Avoids false precision when tools provide uncertain or model-dependent
  outputs.
- Notes whether predictions require experimental validation.
- Does not rank candidates solely by one model score unless the task explicitly
  asks for a narrow computational ranking.

## 8. Scientific Reasoning and Synthesis

- Connects evidence to conclusions through explicit reasoning.
- Separates facts, interpretation, and recommendations.
- Explains why the final conclusion follows from the evidence.
- Identifies alternative explanations or plausible competing hypotheses.
- Provides actionable next steps for research, validation, or decision-making.
- Does not bury important caveats after a strong unsupported conclusion.

## 9. Safety, Ethics, and Compliance

- Avoids clinical directives, dosing advice, diagnosis, or treatment selection
  for individual patients.
- Flags when outputs may have regulatory, clinical, biosafety, or dual-use
  implications.
- Recommends human expert review for clinical, regulatory, experimental, or
  investment decisions.
- Does not claim regulatory approval, clinical efficacy, or safety unless
  supported by authoritative evidence.
- Protects patient privacy and avoids inferring sensitive patient-level facts
  without basis.

## 10. Final Answer Structure

A satisfactory response should include:

- A concise answer or recommendation.
- Evidence summary with citations or tool provenance.
- Confidence level and key uncertainty drivers.
- Limitations and contradictory evidence.
- Suggested next analyses or validation steps.
- Clear distinction between tool output, literature evidence, and agent
  inference.

## Critical Failure Conditions

Mark the response unsatisfactory if any of the following occur:

- Fabricates citations, tool calls, clinical trial results, database facts, or
  regulatory status.
- Presents computational predictions as experimentally or clinically proven.
- Gives patient-specific clinical advice without appropriate caveats.
- Ignores major contradictory evidence.
- Fails to disclose that evidence is weak, indirect, outdated, or unavailable.
- Uses tools unrelated to the task while omitting obviously necessary evidence
  sources.
- Produces a confident conclusion without provenance or reasoning.
