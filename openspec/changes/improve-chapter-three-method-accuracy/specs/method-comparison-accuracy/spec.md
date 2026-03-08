# Spec: Method Comparison Accuracy

## ADDED Requirements

### Requirement: Distinguish improvement from design choice
Technical writing SHALL accurately distinguish between "improving existing methods" and "new design choices" when describing proposed approaches.

#### Scenario: Improving existing method
- **WHEN** a proposed method is an improvement over an existing, established method
- **THEN** the text MAY use comparative language like "传统的" (traditional), "现有的" (existing), "之前的" (previous)
- **AND** SHALL clearly explain what limitations of the existing method are being addressed
- **EXAMPLE**: "相比传统的CNN，ResNet通过残差连接解决了梯度消失问题" is acceptable

#### Scenario: New design choice
- **WHEN** a proposed method is a new architectural choice with no prior implementation
- **THEN** the text SHALL NOT use comparative language implying a "traditional version" exists
- **AND** SHALL present it as a choice between alternative strategies
- **EXAMPLE**: "GEE采用可学习融合策略，而非固定加权平均" is correct
- **COUNTER-EXAMPLE**: "相比传统的固定加权平均，GEE使用可学习融合" is incorrect

#### Scenario: Ambiguous case detection
- **WHEN** it is unclear whether a method represents an improvement or a new choice
- **THEN** the text SHALL use neutral comparative language
- **AND** SHALL focus on "方案A vs 方案B" (Approach A vs Approach B) rather than "新 vs 旧" (new vs traditional)

### Requirement: Accurate method comparison framing
Method comparisons SHALL focus on the technical difference between approaches, not temporal or evolutionary relationships.

#### Scenario: Strategy comparison
- **WHEN** comparing two different technical strategies
- **THEN** the comparison SHALL highlight the technical distinction (e.g., "固定 vs 可学习", "集中式 vs 分布式")
- **AND** SHALL avoid implying one is a "replacement" for the other unless clearly true

#### Scenario: Feature comparison
- **WHEN** comparing features or capabilities
- **THEN** the text SHALL describe what each approach can/cannot do
- **AND** SHALL not use temporal language unless there is a clear historical progression

### Requirement: Novelty clarity
When presenting novel methods, the text SHALL make their novelty clear without misrepresenting existing work.

#### Scenario: Novel architecture
- **WHEN** introducing a newly proposed architecture
- **THEN** the text SHALL clearly state it is novel
- **AND** SHALL compare it to alternative design choices, not "traditional versions" of itself
- **AND** MAY compare it to existing methods that solve similar problems

#### Scenario: Novel component in existing framework
- **WHEN** introducing a novel component within an established framework
- **THEN** the text SHALL clarify what is novel vs what is established
- **AND** SHALL not misrepresent the established framework as "traditional" when only the component is new

### Requirement: Comparison language accuracy
All comparative language (相比, 对比, vs, 等) SHALL accurately reflect the relationship between methods.

#### Scenario: Verification of comparison
- **WHEN** using comparative language
- **THEN** the author SHALL verify: does method B actually exist as a "traditional" or "established" version of method A?
- **IF** NO: Use neutral strategy comparison instead
- **IF** YES: Comparative language is acceptable

#### Scenario: Alternative strategies
- **WHEN** describing design choices between alternatives
- **THEN** the text SHALL use "采用X，而非Y" (adopts X, rather than Y) framing
- **AND** SHALL avoid "相比传统的Y，X更优" (compared to traditional Y, X is better) framing if Y is not traditional
