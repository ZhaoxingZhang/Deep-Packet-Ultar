# Design: Chapter 3 Enhancement - Method Comparison Accuracy

## Context

Chapter 3 (GEE架构设计) contains a method comparison error where the text describes GEE's gating network as an improvement over "传统的固定加权平均" (traditional fixed weighted average). This is inaccurate because:

1. **GEE is a novel architecture**: There is no "traditional" version of GEE that uses fixed weighted averaging
2. **Misrepresentation**: The comparison suggests GEE evolved from an existing traditional method, when it's actually a new architectural design
3. **Confusion**: Readers may misunderstand whether GEE is completely new or an improvement of existing approaches

**Current State**:
- Text says: "相比传统的固定加权平均（如0.85×基准+0.15×专家），门控网络能够根据样本特性动态调整融合权重"
- Implies: Traditional GEE used fixed weighting → New GEE uses learnable gating
- Reality: GEE is entirely new; no prior "traditional GEE" exists

**Constraints**:
- Must apply ALL 14 principles (8 from Ch1 + 5 from Ch2 + 1 new from Ch3)
- Must maintain academic rigor and technical accuracy
- Changes localized to Chapter 3 (approximately lines 236-280)
- Must preserve technical content while fixing comparison language

**Stakeholders**:
- Thesis reviewers: Need accurate understanding of GEE's novelty
- Research community: Should not misrepresent existing work
- Future readers: Must clearly understand what is new vs established

## Goals / Non-Goals

**Goals**:
1. Remove all inaccurate "传统的" (traditional) comparisons when GEE is novel
2. Reframe comparisons as "Strategy A vs Strategy B" design choices
3. Ensure all method comparisons accurately reflect actual relationships
4. Scan entire Chapter 3 for similar comparison issues
5. Maintain readability and flow after corrections

**Non-Goals**:
- Rewriting Chapter 3 structure or technical content
- Changing the technical description of GEE architecture
- Modifying other chapters (only Chapter 3)
- Removing accurate comparisons (e.g., "相比传统的CNN" is correct if CNN is traditional)

## Decisions

### Decision 1: Comparison Rewriting Strategy

**Choice**: Replace "相比传统的X，新方法Y" with "新方法Y采用X，而非Z" framing

**Rationale**:
- "相比传统的X" implies X is a traditional version of Y (incorrect for GEE)
- "采用X，而非Z" (adopts X, not Z) presents it as a design choice between alternatives
- Focuses on technical distinction (fixed vs learnable) not temporal relationship

**Alternatives considered**:
- Keep "相比传统的X" but add footnote explaining GEE is novel: Rejected because footnote doesn't fix inaccuracy
- Remove comparison entirely: Rejected because comparison is useful for understanding design choices
- Change to "与可能的替代方案X相比": Rejected as too verbose and weak

### Decision 2: Scope of Scanning

**Choice**: Scan entire Chapter 3 for all comparison language (相比, 对比, vs, 传统, 现有)

**Rationale**:
- The identified example may not be the only instance
- Consistency is important across the entire chapter
- Other comparison types might have similar issues

**Alternatives considered**:
- Only fix the identified example: Rejected because other issues likely exist
- Scan entire thesis: Rejected as out of scope (focus on Chapter 3)

### Decision 3: Verification Criteria

**Choice**: Use the "existence test" - does the "traditional" method actually exist?

**Rationale**:
- Clear, objective criterion
- Can be applied consistently
- Prevents similar errors in future writing

**Verification process**:
1. When text says "相比传统的X"
2. Ask: Is there an established "traditional X" in literature?
3. If NO: Rewrite as design choice
4. If YES: Keep comparison, verify accuracy

**Alternatives considered**:
- Remove all "传统的" regardless: Rejected as overcorrection
- Only remove if explicitly incorrect: Too subjective, existence test is better

### Decision 4: Comparison Classification

**Choice**: Classify all comparisons into two types before rewriting:

**Type A: Improvement of existing methods**
- Criteria: Method B exists as established approach
- Language allowed: "传统的", "现有的", "之前的"
- Example: "相比传统的CNN，ResNet通过残差连接..." ✓

**Type B: New design choice**
- Criteria: No prior version exists; choosing between alternatives
- Language required: "采用X，而非Y" or "X vs Y"
- Example: "GEE采用可学习融合，而非固定加权平均" ✓

**Rationale**:
- Clear framework for making editing decisions
- Prevents future errors by establishing categories
- Makes verification systematic

### Decision 5: Principle Integration

**Choice**: Explicitly add Principle 14 to the existing 13-principle framework

**Rationale**:
- Maintains consistency with Ch1 and Ch2 improvements
- Provides clear guidance for this and future chapters
- Principle-based approach has proven effective

**Principle 14 statement**:
"准确的方法对比关系：当提出新方法时，区分'改进现有方法'和'新设计选择'。前者可以使用'传统的'等比较语言，后者应呈现为不同策略的选择，而非'新 vs 旧'的进化关系。"

## Risks / Trade-offs

### Risk 1: Overcorrection

**Risk**: Removing accurate comparisons that use "传统的" correctly

**Mitigation**:
- Apply existence test before removing
- If uncertain, keep comparison and verify with literature
- When in doubt, prefer conservative approach (keep if possibly accurate)

### Risk 2: Inconsistent Detection

**Risk**: Missing similar comparison errors elsewhere in Chapter 3

**Mitigation**:
- Systematic scan using grep for comparison keywords
- Manual review of all comparison sentences
- Cross-check with Principle 14 requirements

### Risk 3: Flow Disruption

**Risk**: Rewritten comparisons may disrupt reading flow

**Mitigation**:
- Preserve technical meaning and key information
- Use natural Chinese phrasing
- Read revised sentences aloud to check flow
- If flow is awkward, try alternative rewording

### Risk 4: Scope Creep

**Risk**: Expanding to fix other unrelated writing issues in Chapter 3

**Mitigation**:
- Focus only on method comparison accuracy
- Document other issues separately for future review
- Stick to Principle 14 scope only

## Migration Plan

### Phase 1: Locate and Classify
1. Grep Chapter 3 for comparison keywords: 相比, 对比, vs, 传统, 现有, 之前
2. Extract all sentences with these keywords
3. Classify each as Type A (improvement) or Type B (design choice) using existence test

### Phase 2: Rewrite Type B Comparisons
1. For each Type B comparison:
   - Remove "传统的", "现有的" modifiers
   - Rephrase as "采用X，而非Y" or "X策略 vs Y策略"
   - Ensure technical distinction is clear
2. Preserve Type A comparisons (verify they are accurate)

### Phase 3: Verification
1. Read all rewritten sentences in context
2. Verify flow and readability
3. Apply existence test to all comparisons
4. Ensure no new inaccuracies introduced

### Phase 4: Documentation
1. Update tasks.md with all changes made
2. Record modifications in research log using log-paper-changes skill
3. Create git commit

### Rollback Strategy
- Git commit before each phase
- Keep original sentences in comments if uncertain
- Easy to revert specific rewrites if needed

## Open Questions

1. **Existence test edge cases**: What if a "similar but not identical" traditional method exists?
   - **Resolution**: If not directly comparable as "traditional version of new method", treat as Type B (design choice)

2. **Multiple comparisons**: How to handle sentences with multiple comparisons?
   - **Resolution**: Break into separate sentences or restructure for clarity

3. **Consistent terminology**: Should we establish standard terms for "design choice" vs "improvement"?
   - **Resolution**: Use "设计选择" (design choice) and "改进" (improvement) consistently

4. **Chapter 3 scope**: Are there other Chapter 3-specific principles beyond comparison accuracy?
   - **Resolution**: Focus on Principle 14 for now; user can provide more examples if other issues exist
