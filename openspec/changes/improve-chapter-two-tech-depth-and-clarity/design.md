# Design: Chapter 2 Enhancement - Technical Depth and Clarity

## Context

Chapter 2 (相关理论与技术) currently has 5 major issues that compromise academic quality:

1. **Research log citations**: Multiple references to "详见研究日志10月24日" that reviewers cannot access
2. **OSR historical accuracy**: Claims OSR was "systematically proposed by Scheirer et al. 2013", but OSR as a field predates this
3. **Research思路混排**: Core research methodology (discriminative model + weighted loss) mixed with generative model comparison
4. **Insufficient technical depth**: Section 2.2.1 only mentions CNN, RNN, LSTM, Transformer in one sentence each
5. **Missing context**: Failed attempts (SEBlock, SMOTE, Focal Loss) mentioned in summary but not introduced earlier

**Constraints:**
- Must apply ALL 13 principles (8 from Chapter 1 + 5 new from Chapter 2)
- Must maintain academic rigor and logical coherence
- Changes localized to Chapter 2 (lines ~106-177)
- Cannot rely on external materials (research logs, papers not cited)

**Stakeholders:**
- Thesis reviewers: Need complete, self-contained explanations
- Research advisor: Expects accurate historical attribution
- Future readers: Should understand technical depth without external references

## Goals / Non-Goals

**Goals:**
1. Eliminate all research log citations by replacing with detailed experimental descriptions
2. Correct OSR historical attribution from "earliest systematic proposal" to "systematic framework establishment"
3. Extract research methodology (discriminative vs. generative) into independent subsection (2.1.4)
4. Expand technical content in 2.2.1 to properly cover LSTM and Transformer architectures
5. Add "单一模块优化尝试" subsection to 2.1.3 to provide context for failed approaches

**Non-Goals:**
- Rewriting Chapter 2 structure beyond identified issues
- Adding new technical content not relevant to this thesis
- Changing Chapter 2's overall scope or purpose
- Modifying other chapters

## Decisions

### Decision 1: Research Log Citation Strategy

**Choice**: Expand inline descriptions rather than adding footnotes or appendices

**Rationale:**
- Reviewers should understand content without flipping to appendices
- Inline descriptions maintain flow and readability
- Footnotes would still require external reference

**Alternatives considered:**
- Footnotes: Rejected because still externalizes information
- Appendix section: Rejected because breaks reading flow
- Citations to external logs: Rejected because logs aren't accessible to reviewers

### Decision 2: OSR Historical Correction Approach

**Choice**: Reframe Scheirer 2013 as "systematic framework establishment" not "earliest proposal"

**Rationale:**
- OSR concepts (zero-day detection, anomaly detection) existed before 2013
- Scheirer 2013's contribution was standardization and terminology
- Avoids historical inaccuracy while acknowledging their work

**Alternatives considered:**
- Remove Scheirer reference entirely: Rejected because they did make key contributions
- Add extensive historical survey: Rejected because beyond scope of this thesis
- Keep "earliest proposal" claim: Rejected because historically inaccurate

### Decision 3: Research Methodology Section Structure

**Choice**: Create new subsection 2.1.4 "本研究的思路选择" after 2.1.2

**Rationale:**
- Separates methodology choice from literature review
- Provides clear rationale for discriminative vs. generative approach
- Maintains logical flow: background → literature → methodology choice → technical details

**Alternatives considered:**
- Keep mixed in 2.1.2: Rejected because conflates two different dimensions (literature vs. research choice)
- Add to Chapter 1 introduction: Rejected because lacks technical context from 2.1.2
- Create entirely new Chapter 2 section: Rejected because too disruptive to structure

### Decision 4: Technical Content Expansion Strategy

**Choice**: Expand 2.2.1 into 3 paragraphs (CNN, LSTM, Transformer) with 2-3 sentences each

**Rationale:**
- Maintains conciseness while providing sufficient depth
- Follows principle 8: technical content should be sufficient but not verbose
- Balances breadth (3 architectures) with depth (key mechanisms)

**Alternatives considered:**
- Single paragraph listing all architectures: Rejected as too shallow
- Multiple subsections for each architecture: Rejected as too verbose
- Remove RNN/CNN focus only on LSTM/Transformer: Rejected because CNN/RNN are foundational

### Decision 5: Failed Attempts Placement

**Choice**: Add "单一模块优化尝试" subsection to 2.1.3 before existing literature review

**Rationale:**
- Provides chronological narrative: failed attempts → GEE solution
- Justifies why architecture-level innovation was necessary
- Maintains logical coherence with summary section

**Alternatives considered:**
- Add to 2.1.1 (OSR methods): Rejected because these are class imbalance methods, not OSR
- Add entirely new section 2.1.5: Rejected because breaks flow
- Merge into 2.3 summary: Rejected because summary should summarize, not introduce

## Risks / Trade-offs

### Risk 1: Over-expansion of Chapter 2

**Risk**: Adding technical content and new subsections could make Chapter 2 too long

**Mitigation**: Keep additions concise and focused
- LSTM: 2-3 sentences on gating mechanism and traffic classification
- Transformer: 2-3 sentences on self-attention and interpretability
- Failed attempts: 4 sentences total (3 techniques + 1 conclusion)

### Risk 2: Inconsistent Academic Tone

**Risk**: Expanding research log citations could introduce informal language

**Mitigation**: Use formal academic language
- Replace "详见研究日志" with "实验表明" or "分析发现"
- Maintain passive voice and objective tone
- Focus on technical explanations not narrative description

### Risk 3: OSR Historical Accuracy Without Extensive Research

**Risk**: Correcting OSR history without thorough literature review may introduce new inaccuracies

**Mitigation**: Use conservative, qualified language
- "OSR问题的概念最早可以追溯到" (can be traced back to)
- Avoid claiming specific earliest works
- Focus on what Scheirer 2013 actually contributed (framework, terminology)

### Risk 4: Research Methodology Section Redundancy

**Risk**: New 2.1.4 could repeat content from 2.1.2 literature review

**Mitigation**: Clear separation of concerns
- 2.1.2: What OSR methods exist (literature survey)
- 2.1.4: Why this study chose discriminative approach (research rationale)
- Different content, different purpose

## Migration Plan

### Phase 1: Research Log Citation Rewriting (Lines 118, 144)
1. Read research log entries for 10月24日
2. Extract experimental details for SMOTE, Focal Loss
3. Rewrite citations as detailed inline descriptions
4. Verify all quantitative metrics (Macro-F1 scores) are preserved

### Phase 2: OSR Historical Correction (Line 122)
1. Locate existing OSR history paragraph
2. Rewrite opening sentence to use "可以追溯到" framing
3. Preserve Scheirer 2013 attribution as "systematic framework"
4. Verify no other historical claims need correction

### Phase 3: Research Methodology Extraction (After 2.1.2)
1. Locate discriminative model sentence in generative model paragraph
2. Remove from 2.1.2 fourth method
3. Create new 2.1.4 subsection with proposed content
4. Verify smooth transition to 2.2 technical background

### Phase 4: Technical Content Expansion (2.2.1)
1. Read current 2.2.1 section
2. Expand CNN paragraph if needed (1-2 sentences)
3. Add LSTM paragraph (2-3 sentences): gating mechanism, traffic classification
4. Add Transformer paragraph (2-3 sentences): self-attention, interpretability
5. Verify smooth flow between paragraphs

### Phase 5: Failed Attempts Addition (2.1.3)
1. Locate 2.1.3 section start
2. Insert new subsection before existing literature
3. Add SEBlock, SMOTE, Focal Loss descriptions
4. Add concluding sentence linking to GEE innovation
5. Verify summary section (2.3) still makes sense

### Rollback Strategy
- All changes are additive except citation rewrites (replacement)
- Git commit before each phase for easy rollback
- Keep original text in comments if uncertain

## Open Questions

1. **OSR Historical References**: Should we add specific early OSR works (pre-2013) beyond general fields (zero-day detection, anomaly detection)?
   - **Resolution**: No, keep general references to avoid historical inaccuracies without extensive research

2. **LSTM/Transformer Representative Works**: Should we cite specific traffic classification papers using LSTM/Transformer?
   - **Resolution**: Yes, add 1-2 representative citations if available, otherwise describe generically

3. **Failed Attempts Metrics**: Are the Macro-F1 scores (0.72→0.31, 0.72→0.52, 0.09) accurate and verifiable?
   - **Resolution**: Verify against research log before including in failed attempts section

4. **Research Methodology Section Number**: Should it be 2.1.3 (shifting existing) or 2.1.4 (new subsection)?
   - **Resolution**: Use 2.1.4 as new subsection to avoid renumbering entire chapter

5. **Technical Content Balance**: Is 2-3 sentences per architecture sufficient, or should we go deeper?
   - **Resolution**: Start with 2-3 sentences, expand only if reviewer feedback indicates insufficient depth
