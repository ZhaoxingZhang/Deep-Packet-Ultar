# Tasks: Chapter 3 Enhancement - Method Comparison Accuracy

## 1. Preparation and Context Gathering

- [ ] 1.1 Read Chapter 3 content (lines ~236-280) to understand full context
- [ ] 1.2 Create Git commit before starting modifications for rollback safety
- [ ] 1.3 Review Principle 14: 准确的方法对比关系

## 2. Phase 1: Locate and Classify Comparisons

- [ ] 2.1 Grep Chapter 3 for comparison keywords: 相比, 对比, vs, 传统, 现有, 之前
- [ ] 2.2 Extract all sentences containing these keywords with line numbers
- [ ] 2.3 For each comparison, apply existence test: Does the "traditional" method actually exist?
- [ ] 2.4 Classify each comparison as Type A (improvement) or Type B (design choice)
- [ ] 2.5 Document classification results in a checklist

## 3. Phase 2: Rewrite Type B Comparisons

- [ ] 3.1 Locate identified example: "相比传统的固定加权平均（如0.85×基准+0.15×专家）"
- [ ] 3.2 Rewrite as: "GEE架构采用可学习融合策略，而非固定加权平均。固定加权平均使用人为设定的统一权重..."
- [ ] 3.3 Verify technical meaning is preserved (fixed vs learnable distinction)
- [ ] 3.4 Verify flow and readability of rewritten sentence
- [ ] 3.5 Process any other Type B comparisons found during scan
- [ ] 3.6 For each Type B comparison: Remove "传统的" modifiers and rephrase as design choice
- [ ] 3.7 Ensure all rewritten sentences use "采用X，而非Y" or similar neutral framing

## 4. Phase 3: Verification and Quality Assurance

- [ ] 4.1 Read all rewritten sentences in full context
- [ ] 4.2 Verify no new inaccuracies introduced
- [ ] 4.3 Verify flow and readability (sentences should sound natural in Chinese)
- [ ] 4.4 Apply existence test to all comparisons (including Type A kept as-is)
- [ ] 4.5 Verify all Type A comparisons are accurate (e.g., "相比传统的CNN" - does traditional CNN exist?)
- [ ] 4.6 Check for consistency in terminology (设计选择 vs 改进)
- [ ] 4.7 Verify Chapter 3 still maintains academic rigor and technical accuracy
- [ ] 4.8 Ensure no "traditional GEE" implications remain in text

## 5. Cross-Chapter Principle Application

- [ ] 5.1 Apply Principle 1 (Technical background before problem statements): Review all technical descriptions
- [ ] 5.2 Apply Principle 2 (Qualitative claims need quantitative support): Check all qualitative statements
- [ ] 5.3 Apply Principle 3 (Traditional methods before innovations): Ensure traditional approaches introduced first
- [ ] 5.4 Apply Principle 4 (Proper innovation attribution): Verify all "proposed" claims
- [ ] 5.5 Apply Principle 5 (Technical precision): Check technical terminology
- [ ] 5.6 Apply Principle 6 (Academic language precision): Replace informal expressions
- [ ] 5.7 Apply Principle 7 (Accurate scope of work): Ensure claims are accurate
- [ ] 5.8 Apply Principle 8 (Consistent data presentation): Check ratios vs absolute numbers
- [ ] 5.9 Apply Principle 9 (Research log citation rewriting): Verify no external references
- [ ] 5.10 Apply Principle 10 (Historical accuracy): Verify historical claims
- [ ] 5.11 Apply Principle 11 (Research思路 separation): Check methodology sections
- [ ] 5.12 Apply Principle 12 (Technical content sufficiency): Verify sufficient depth
- [ ] 5.13 Apply Principle 13 (Failed attempts introduction): Verify前置介绍
- [ ] 5.14 Apply Principle 14 (Method comparison accuracy): Verify all comparisons are accurate

## 6. Final Review

- [ ] 6.1 Read entire modified Chapter 3 to check logical flow and coherence
- [ ] 6.2 Verify no "传统的" modifiers incorrectly used for novel methods
- [ ] 6.3 Verify all comparisons accurately reflect relationships between methods
- [ ] 6.4 Verify GEE is clearly presented as novel architecture, not improvement of traditional method
- [ ] 6.5 Verify comparison focus is on technical distinction (fixed vs learnable, etc.)
- [ ] 6.6 Verify academic tone is consistent throughout (formal, objective, precise)
- [ ] 6.7 Check chapter length has not expanded significantly (target: <5% increase)
- [ ] 6.8 Verify all 14 principles have been applied consistently

## 7. Documentation and Logging

- [ ] 7.1 Create skill invocation for log-paper-changes to record modifications
- [ ] 7.2 Prepare changes array with all modification details (location, title, before, after)
- [ ] 7.3 Execute log-paper-changes skill to append to research log
- [ ] 7.4 Verify research log entry follows standard format (date, theme, numbered changes)
- [ ] 7.5 Confirm all modifications are recorded in research log

## 8. Git Commit and Finalization

- [ ] 8.1 Create Git commit after all modifications with descriptive message
- [ ] 8.2 Verify commit message clearly describes Principle 14 application
- [ ] 8.3 Review git diff to ensure only intended changes were made
- [ ] 8.4 Verify no unintended modifications to other chapters or sections
- [ ] 8.5 Ready for implementation completion with `/opsx:archive`
