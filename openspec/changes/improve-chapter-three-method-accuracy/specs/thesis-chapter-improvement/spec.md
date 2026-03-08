# Spec: Thesis Chapter Improvement (Extension)

## MODIFIED Requirements

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

## ADDED Requirements

### Requirement: Method comparison accuracy (Principle 14)
Technical writing MUST accurately distinguish between "improvements to existing methods" and "new architectural design choices" when describing proposed approaches.

#### Scenario: Detecting inaccurate method comparison
- **WHEN** a text describes a new method as "improving" or "compared to" its "traditional" version
- **THEN** verify whether the "traditional" version actually exists as an established method
- **IF** the traditional version does NOT exist: This is inaccurate method comparison
- **AND** SHALL be rewritten to frame it as a design choice between alternatives

#### Scenario: Rewriting inaccurate comparisons
- **WHEN** inaccurate method comparison is detected (e.g., "相比传统的固定加权平均，GEE...")
- **THEN** identify the actual comparison: Is it "Strategy A vs Strategy B" or "New vs Old"?
- **IF** it's a choice between strategies: Rewrite as "GEE采用策略A，而非策略B"
- **IF** it's truly an improvement: Keep as "相比传统的X，新方法Y改进了..."

#### Scenario: Comparison focus alignment
- **WHEN** rewriting method comparisons
- **THEN** ensure the focus is on technical distinction, not temporal relationship
- **EXAMPLE**: "固定加权平均 vs 可学习融合" (fixed vs learnable) - focus on technical difference
- **NOT**: "传统 vs 现代" (traditional vs modern) - focus on temporal relationship unless accurate
