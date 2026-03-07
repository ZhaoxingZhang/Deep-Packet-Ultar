# Spec: Research Log Citation Rewrite

## ADDED Requirements

### Requirement: Research log citation elimination
All citations to research logs, private notes, or unpublished documents SHALL be replaced with detailed inline descriptions that make the content self-contained.

#### Scenario: Research log reference with experimental result
- **WHEN** a sentence contains "详见研究日志" or similar followed by a quantitative result (e.g., "Macro-F1 from 0.72 to 0.31")
- **THEN** the citation SHALL be replaced with:
  - What was attempted (technique, method, approach)
  - Experimental setup details
  - Quantitative results (preserving all metrics)
  - Analysis of why the result occurred
- **AND** the rewritten sentence SHALL be 1-2 sentences long, not verbose

#### Scenario: Multiple research log citations in same section
- **WHEN** multiple research log citations appear in the same section (e.g., Line 118 has two)
- **THEN** each SHALL be rewritten independently
- **AND** redundant phrases SHALL be consolidated if they refer to the same experiment

### Requirement: Experimental detail preservation
When rewriting research log citations, all quantitative metrics and technical details SHALL be preserved exactly.

#### Scenario: Performance metrics preserved
- **WHEN** a research log citation contains performance metrics (e.g., Macro-F1, accuracy, FPR@TPR95)
- **THEN** the rewritten description SHALL preserve all exact values
- **AND** SHALL include before/after comparisons if present in original

#### Scenario: Sample size information preserved
- **WHEN** a research log citation mentions sample sizes or ratios (e.g., "1:146", "9476 samples")
- **THEN** the rewritten description SHALL preserve this information
- **AND** SHALL follow principle 8 (use ratios not absolute numbers when appropriate)

### Requirement: Academic tone maintenance
Rewritten research log citations SHALL maintain formal academic tone and use objective language.

#### Scenario: Informal to formal translation
- **WHEN** a research log might contain informal descriptions
- **THEN** the rewritten description SHALL use formal academic language
- **AND** SHALL use phrases like "实验表明" (experiments show), "分析发现" (analysis reveals)
- **AND** SHALL avoid narrative or storytelling language

#### Scenario: Result interpretation language
- **WHEN** explaining why an experiment failed or succeeded
- **THEN** the explanation SHALL use technical, analytical language
- **AND** SHALL focus on mechanism-level explanations, not author intentions

### Requirement: Causal explanation inclusion
Rewritten research log citations SHALL include not just results, but also causal explanations for why those results occurred.

#### Scenario: Failed experiment explanation
- **WHEN** an experiment produced negative results (e.g., performance degradation)
- **THEN** the rewritten description SHALL explain the likely cause
- **AND** SHALL connect the result to technical mechanisms (e.g., "在高维空间中拟合分布极其困难")

#### Scenario: Successful experiment explanation
- **WHEN** an experiment produced positive results
- **THEN** the rewritten description SHALL explain what technical factors contributed to success
- **AND** SHALL avoid claiming success without technical rationale

### Requirement: No external references
Rewritten content SHALL not include references to external documents, appendices, or supplementary materials.

#### Scenario: Self-contained verification
- **WHEN** reviewing a rewritten research log citation
- **THEN** a reader SHALL be able to understand the complete experiment without accessing any external materials
- **AND** SHALL not need to consult research logs, appendices, or supplementary documents

#### Scenario: No footnote dependencies
- **WHEN** rewriting research log citations
- **THEN** footnotes SHALL NOT be used as alternative to inline descriptions
- **AND** all critical information SHALL appear in the main text
