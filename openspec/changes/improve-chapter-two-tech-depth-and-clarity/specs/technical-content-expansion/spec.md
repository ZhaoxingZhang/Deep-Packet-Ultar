# Spec: Technical Content Expansion

## ADDED Requirements

### Requirement: Sufficient technical depth
Technical background sections SHALL provide sufficient depth for readers to understand key concepts, following the principle of "concise but complete."

#### Scenario: Architecture introduction completeness
- **WHEN** introducing a technical architecture (e.g., CNN, LSTM, Transformer)
- **THEN** the introduction SHALL include:
  - Full name and common abbreviation
  - Core innovation or mechanism (what makes it different)
  - Key technical contribution to the field
  - Why it's relevant to this thesis's domain
- **AND** the description SHALL be 2-3 sentences long

#### Scenario: Insufficient depth detected
- **WHEN** a technical concept is mentioned in only one sentence or listed alongside others without elaboration
- **THEN** the content SHALL be expanded to meet the completeness criteria
- **AND** SHALL maintain focus on aspects most relevant to the thesis

### Requirement: Balanced breadth vs. depth
Technical content SHALL balance coverage breadth (number of topics) with depth (detail per topic).

#### Scenario: Multiple architectures overview
- **WHEN** covering multiple architectures in one section (e.g., CNN, RNN, LSTM, Transformer)
- **THEN** each SHALL receive approximately equal attention
- **AND** foundational architectures (e.g., CNN, RNN) MAY receive slightly less detail if well-known
- **AND** novel architectures (e.g., Transformer) SHALL receive sufficient explanation of their innovation

#### Scenario: Paragraph-level organization
- **WHEN** expanding technical content
- **THEN** each major architecture or concept SHALL have its own paragraph
- **AND** paragraphs SHALL be organized logically (foundational → advanced, or chronological)
- **AND** transitions between paragraphs SHALL explain relationships

### Requirement: Domain-specific application context
Technical descriptions SHALL include how general architectures or methods are applied to the specific domain of this thesis (network traffic classification).

#### Scenario: General architecture in traffic classification
- **WHEN** describing a general-purpose architecture (e.g., LSTM, Transformer)
- **THEN** the description SHALL include how it's applied to traffic classification
- **AND** SHALL mention domain-specific benefits (e.g., "处理流量包的时序特征" for LSTM)
- **AND** MAY mention representative works in traffic classification if relevant

#### Scenario: Mechanism-to-domain mapping
- **WHEN** explaining an architecture's core mechanism
- **THEN** the explanation SHALL connect the mechanism to domain-specific advantages
- **EXAMPLE**: For LSTM gating: "门控机制控制信息流" → "处理流量包的时序特征"

### Requirement: Conciseness preservation
While expanding technical content, the overall section SHALL remain concise and avoid unnecessary verbosity.

#### Scenario: Target length guidelines
- **WHEN** expanding a technical background subsection
- **THEN** each subsection SHALL be approximately 2-3 paragraphs total
- **AND** each paragraph SHALL be 2-4 sentences long
- **AND** the entire section SHALL not exceed what's necessary for understanding

#### Scenario: Avoiding over-expansion
- **WHEN** adding technical details
- **THEN** details SHALL be included only if they:
  - Are necessary for understanding the architecture's core innovation
  - Explain why it's relevant to this thesis
  - Help readers understand later chapters
- **AND** historical details, mathematical derivations, or implementation details SHALL be excluded unless critical

### Requirement: Representative work references
Technical content MAY include references to representative works in the field, but SHALL NOT become a literature survey.

#### Scenario: Representative citation inclusion
- **WHEN** introducing an architecture or method
- **THEN** 1-2 representative citations MAY be included if they demonstrate the concept's application to traffic classification
- **AND** these citations SHALL be well-known or foundational works, not exhaustive surveys

#### Scenario: Avoiding literature drift
- **WHEN** expanding technical content
- **THEN** the section SHALL NOT become a literature review
- **AND** citations SHALL support understanding, not comprehensiveness
- **AND** if more than 2-3 citations are needed, consider creating a separate literature subsection

### Requirement: Principle 8 consistency (data presentation)
All technical content expansions SHALL follow Principle 8: use ratios/proportions instead of absolute sample numbers, unless absolute numbers are necessary.

#### Scenario: Sample size references
- **WHEN** technical content mentions dataset sizes or imbalances
- **THEN** ratios or proportions SHALL be used (e.g., "1:146", "3.3:1")
- **AND** absolute sample counts SHALL NOT be used unless comparing absolute scale across studies
