# Spec: Thesis Chapter Improvement Framework

## ADDED Requirements

### Requirement: Self-contained content
Thesis chapters SHALL be completely self-contained and understandable without access to external materials such as research logs, private notes, or unpublished documents.

#### Scenario: Reviewer reads thesis without research logs
- **WHEN** a thesis reviewer reads any chapter
- **THEN** all references, experimental results, and technical claims SHALL be fully explained inline
- **AND** the reviewer SHALL NOT need to consult external sources to understand the content

#### Scenario: Research log citation detected
- **WHEN** a citation to research logs (e.g., "详见研究日志10月24日") is found
- **THEN** the citation SHALL be replaced with a detailed description of the experimental content
- **AND** the description SHALL include experimental setup, results, and analysis

### Requirement: Historical accuracy
All historical claims in the thesis SHALL be accurate and properly attribute contributions without overstating originality.

#### Scenario: Earliest proposal claims verified
- **WHEN** a claim states that a problem was "first systematically proposed" or "earliest" by specific authors
- **THEN** the claim SHALL be verified against historical context
- **AND** if inaccurate, SHALL be reframed to acknowledge the actual contribution (e.g., "systematic framework establishment" rather than "earliest proposal")

#### Scenario: Conservative historical language
- **WHEN** precise historical origins cannot be determined without extensive research
- **THEN** the thesis SHALL use conservative, qualified language (e.g., "can be traced back to", "has roots in")
- **AND** SHALL avoid making specific earliest-claim assertions

### Requirement: Research methodology separation
Core research methodology choices SHALL be separated from literature review into independent sections.

#### Scenario: Literature vs. methodology distinction
- **WHEN** a paragraph discusses both existing methods (literature) and this study's approach (methodology)
- **THEN** these SHALL be separated into distinct sections
- **AND** the methodology section SHALL explain WHY this approach was chosen over alternatives

#### Scenario: Independent methodology subsection
- **WHEN** a core research decision (e.g., discriminative vs. generative models) is currently mixed within literature review
- **THEN** a new subsection SHALL be created (e.g., 2.1.4 "本研究的思路选择")
- **AND** the subsection SHALL provide rationale for the chosen approach

### Requirement: Failed attempts introduction
All approaches mentioned in summary sections (e.g., "failed attempts") SHALL be introduced and explained in earlier sections before being referenced in summaries.

#### Scenario: Failed techniques mentioned in summary
- **WHEN** a summary section mentions that certain techniques "did not achieve ideal results"
- **THEN** those techniques SHALL be introduced in detail in an earlier section
- **AND** the introduction SHALL include what was tried, why it was tried, and what the results were

#### Scenario: Chronological narrative coherence
- **WHEN** failed attempts are introduced
- **THEN** they SHALL be placed before the successful solution to maintain narrative flow
- **AND** a concluding sentence SHALL explain how failures motivated the eventual successful approach

### Requirement: Technical content sufficiency
Technical background sections SHALL provide sufficient depth for readers to understand key concepts, balancing conciseness with completeness.

#### Scenario: Architecture descriptions
- **WHEN** a technical architecture (e.g., LSTM, Transformer) is mentioned
- **THEN** the description SHALL include:
  - Full name/expansion
  - Core mechanism or innovation
  - Why it's relevant to this domain
  - 1-2 sentences on how it's applied in this field

#### Scenario: Shallow technical content detected
- **WHEN** a technical concept is mentioned in only one sentence or less
- **THEN** the content SHALL be expanded to 2-3 sentences
- **AND** SHALL maintain focus on aspects most relevant to the thesis

### Requirement: Principle-based modification
Thesis modifications SHALL be guided by extracted principles from examples, not just fixing the specific examples provided.

#### Scenario: Systematic review using principles
- **WHEN** improving a thesis chapter based on modification examples
- **THEN** principles SHALL be extracted from the examples
- **AND** the entire chapter SHALL be systematically reviewed using those principles
- **AND** issues beyond the original examples SHALL be identified and fixed

#### Scenario: Principle consistency across chapters
- **WHEN** modifying multiple chapters
- **THEN** ALL principles from previous chapters SHALL be applied to subsequent chapters
- **AND** new principles MAY be extracted from new examples
- **AND** the combined principle set SHALL be used for systematic review
