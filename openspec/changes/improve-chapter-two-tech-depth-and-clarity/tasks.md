# Tasks: Chapter 2 Enhancement - Technical Depth and Clarity

## 1. Preparation and Context Gathering

- [x] 1.1 Read research log entries for 10月24日 to extract SMOTE and Focal Loss experimental details
- [x] 1.2 Read current Chapter 2 content (lines ~106-177) to understand full context
- [x] 1.3 Verify all quantitative metrics mentioned in proposal (Macro-F1 scores, FPR@TPR95) against research logs
- [ ] 1.4 Create Git commit before starting modifications for rollback safety

## 2. Research Log Citation Rewriting (Phase 1)

- [x] 2.1 Locate Line 118 with "详见研究日志10月24日" reference to SMOTE
- [x] 2.2 Rewrite SMOTE citation: Replace "详见研究日志10月24日，说明SMOTE在小样本高维数据上效果不佳" with detailed description including experimental result (Macro-F1: 0.72→0.31) and technical explanation (overfitting in high-dimensional space)
- [x] 2.3 Locate Line 118 with "详见研究日志10月24日" reference to Focal Loss
- [x] 2.4 Rewrite Focal Loss citation: Replace "详见研究日志10月24日，说明Focal Loss在极度不均衡场景下可能导致优化困难" with detailed description including experimental result (Macro-F1: 0.09) and technical explanation (optimization direction deviation in 1:146 imbalance scenario)
- [x] 2.5 Locate Line 144 "详见研究日志" and "详见研究日志10月24日" references
- [x] 2.6 Rewrite all Line 144 citations with detailed experimental descriptions following same pattern
- [x] 2.7 Verify all rewritten citations use formal academic language ("实验表明", "分析发现") not narrative tone
- [x] 2.8 Verify all quantitative metrics are preserved exactly in rewritten descriptions
- [ ] 2.9 Scan entire Chapter 2 for any additional "详见研究日志" or similar external references
- [ ] 2.10 Rewrite any additional research log citations found during scan

## 3. OSR Historical Correction (Phase 2)

- [x] 3.1 Locate Line 122 OSR history paragraph with claim "OSR问题最早由Scheirer等人于2013年系统性地提出"
- [x] 3.2 Rewrite opening sentence to use conservative framing: "OSR问题的概念最早可以追溯到早期的零日攻击检测和异常检测研究"
- [x] 3.3 Preserve Scheirer 2013 attribution with corrected framing: "Scheirer等人于2013年系统性地提出了OSR的理论框架和标准化评估方法，为该领域奠定了理论基础"
- [ ] 3.4 Verify no other historical claims in Chapter 2 make "earliest" or "first" assertions without proper qualification
- [ ] 3.5 Scan entire chapter for other potentially inaccurate historical attributions

## 4. Research Methodology Section Creation (Phase 3)

- [x] 4.1 Locate Section 2.1.2 and find the discriminative model sentence mixed in generative model paragraph: "本研究采用判别模型（ResNet）+ 加权损失和门控网络融合..."
- [x] 4.2 Remove this sentence from the fourth method (generative models) paragraph
- [x] 4.3 Create new subsection 2.1.4 "本研究的思路选择" after 2.1.2, before 2.2
- [x] 4.4 Write subsection introduction: "在众多OSR方法中，本研究选择了基于判别模型的路径，而非生成模型方法。这一选择基于以下考虑："
- [x] 4.5 Write first paragraph explaining discriminative model feasibility: Describe generative model challenges (training difficulty, mode collapse, curse of dimensionality) in 2-3 sentences
- [x] 4.6 Write second paragraph describing this study's discriminative approach: ResNet backbone, weighted loss, gated network fusion, FPR@TPR95 improvement (0.47→0.03) in 2-3 sentences
- [x] 4.7 Write third paragraph explaining design philosophy: How it combines method type 3 (discriminative info) and type 4 (distribution fitting) advantages in 2 sentences
- [ ] 4.8 Verify smooth transition from new 2.1.4 to 2.2 technical background section
- [ ] 4.9 Verify no redundancy between 2.1.2 literature review and new 2.1.4 methodology choice

## 5. Technical Content Expansion (Phase 4)

- [x] 5.1 Locate Section 2.2.1 深度学习分类方法 (approximately lines 158-162)
- [x] 5.2 Read current CNN paragraph content and assess if expansion needed
- [ ] 5.3 Expand CNN paragraph to 2-3 sentences if currently only 1 sentence: Add core mechanism (convolutional layers, feature hierarchy) and relevance to traffic classification
- [x] 5.4 Locate or create RNN/LSTM paragraph (currently merged with RNN)
- [x] 5.5 Write LSTM paragraph (2-3 sentences): Full name (Long Short-Term Memory), gating mechanism (controlling information flow), solving long sequence dependency, application to traffic packet temporal features
- [x] 5.6 Add 1-2 representative citations for LSTM in traffic classification if available
- [x] 5.7 Locate or create Transformer paragraph (currently merged with RNN)
- [x] 5.8 Write Transformer paragraph (2-3 sentences): Self-attention mechanism, success in NLP (BERT, GPT), application to traffic classification (attention weight interpretability, capturing long-range dependencies)
- [x] 5.9 Add 1-2 representative citations for Transformer in traffic classification if available
- [ ] 5.10 Verify smooth logical flow between CNN → LSTM → Transformer paragraphs
- [ ] 5.11 Verify each architecture description follows principle 8: use ratios not absolute sample numbers if mentioning data
- [ ] 5.12 Check that total section length remains concise (target: 3 paragraphs, 2-4 sentences each)

## 6. Failed Attempts Section Addition (Phase 5)

- [x] 6.1 Locate Section 2.1.3 类别不均衡学习研究
- [x] 6.2 Identify where "现有的类别不均衡学习方法" subsection begins
- [x] 6.3 Insert new subsection "#### 单一模块优化方法的尝试" before existing literature review
- [x] 6.4 Write subsection introduction: "在采用架构层面的GEE方案之前，本研究尝试了多种单一模块的优化方法，试图在现有基准模型的基础上提升少数类识别性能。"
- [x] 6.5 Write SEBlock attempt paragraph: Describe mechanism (Squeeze-and-Excitation attention), experimental results (accuracy: 90.8%→87.2%, Macro-F1: 0.72→0.52), technical analysis (attention mechanism强化多数类，忽略少数类) in 2-3 sentences
- [x] 6.6 Write SMOTE attempt paragraph: Describe method (synthetic minority oversampling in feature space), experimental results (Macro-F1: 0.72→0.31), technical analysis (synthetic samples don't reflect real distribution in high-dimensional space) in 2-3 sentences
- [x] 6.7 Write Focal Loss attempt paragraph: Describe method (reduce easy sample weights, focus on hard samples), experimental results (Macro-F1: 0.09), technical analysis (optimization direction deviation in 1:146 extreme imbalance) in 2-3 sentences
- [x] 6.8 Write concluding sentence: "这些单一模块的优化尝试均未取得理想效果，这提示我们需要从架构层面进行创新，而非依赖于单一模块的修补。"
- [ ] 6.9 Verify all Macro-F1 scores match research log data (0.72→0.52 for SEBlock, 0.72→0.31 for SMOTE, 0.09 for Focal Loss)
- [ ] 6.10 Verify summary section 2.3 reference to failed attempts now makes sense with earlier introduction

## 7. Cross-Chapter Principle Application

- [ ] 7.1 Apply Principle 1 (Technical background before problem statements): Review all technical descriptions to ensure concepts are explained before being used
- [ ] 7.2 Apply Principle 2 (Qualitative claims need quantitative support): Check all qualitative statements have backing metrics or data
- [ ] 7.3 Apply Principle 3 (Traditional methods before innovations): Ensure traditional approaches are introduced before novel methods
- [ ] 7.4 Apply Principle 4 (Proper innovation attribution): Verify all "proposed" or "developed" claims are actually novel applications, not existing methods
- [ ] 7.5 Apply Principle 5 (Technical precision): Check all technical descriptions use precise terminology
- [ ] 7.6 Apply Principle 6 (Academic language precision): Replace informal expressions with formal academic language
- [ ] 7.7 Apply Principle 7 (Accurate scope of work): Ensure claims about contributions are accurate and not overstated
- [ ] 7.8 Apply Principle 8 (Consistent data presentation): Replace all absolute sample numbers with ratios throughout Chapter 2 (e.g., 9476→146倍, 215 and 65→3.3:1)

## 8. Verification and Quality Assurance

- [ ] 8.1 Read entire modified Chapter 2 to check logical flow and coherence
- [ ] 8.2 Verify no "详见研究日志" or external references remain
- [ ] 8.3 Verify OSR historical claims are accurate and qualified
- [ ] 8.4 Verify research methodology section (2.1.4) clearly separates literature review from research choices
- [ ] 8.5 Verify technical content (2.2.1) provides sufficient depth for CNN, LSTM, Transformer
- [ ] 8.6 Verify failed attempts section (2.1.3) provides context for summary section references
- [ ] 8.7 Verify all quantitative metrics are accurate and consistent with research logs
- [ ] 8.8 Verify all 13 principles (8 from Ch1 + 5 from Ch2) have been applied
- [ ] 8.9 Check chapter length has not expanded excessively (target: <20% increase)
- [ ] 8.10 Verify academic tone is consistent throughout (formal, objective, precise)
- [ ] 8.11 Create Git commit after all modifications with descriptive message

## 9. Documentation and Logging

- [ ] 9.1 Create skill invocation for log-paper-changes to record modifications
- [ ] 9.2 Prepare changes array with all modification details (location, title, before, after)
- [ ] 9.3 Execute log-paper-changes skill to append to research log
- [ ] 9.4 Verify research log entry follows standard format (date, theme, numbered changes with location tags)
- [ ] 9.5 Confirm all 5 major modifications are recorded in research log

## 10. Final Review

- [ ] 10.1 Compare modified Chapter 2 against proposal's 8 success criteria
- [ ] 10.2 Verify all research log citations rewritten (success criterion 1)
- [ ] 10.3 Verify OSR history accurate (success criterion 2)
- [ ] 10.4 Verify research思路独立成段 (success criterion 3)
- [ ] 10.5 Verify 2.2.1 expanded to include LSTM, Transformer (success criterion 4)
- [ ] 10.6 Verify 单一模块优化尝试 introduced upfront (success criterion 5)
- [ ] 10.7 Verify Chapter 2 maintains technical depth and academic quality (success criterion 6)
- [ ] 10.8 Verify modifications are logically coherent and not redundant (success criterion 7)
- [ ] 10.9 Verify reviewers can understand all content without external materials (success criterion 8)
- [ ] 10.10 Ready for implementation with `/opsx:apply`
