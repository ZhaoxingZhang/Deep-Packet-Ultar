# Chapter Content Outline Specification

## ADDED Requirements

### Requirement: Paragraph-Level Writing Instructions
The chapter content outline SHALL provide writing instructions at the paragraph level for each section.

**Description**:
For each section in the thesis, the outline shall specify what content each paragraph should contain, which materials to use (from research logs or experiment results), and which innovation point the content supports. This enables AI-assisted writing with clear guidance.

#### Scenario: Writing instructions format
- **WHEN** creating writing instructions for a section
- **THEN** each paragraph SHALL have the following structure:
  - **段落序号** (Paragraph Number): e.g., 1.1节第1段
  - **写作要点** (Writing Points): 2-3 bullet points describing what this paragraph should say
  - **素材来源** (Source Material): Reference to `.agent/研究日志.md` (e.g., "参考10月24日日志") or experiment results
  - **支撑创新点** (Supported Innovation): Which innovation claim this paragraph supports (e.g., "支撑OSR性能提升")
  - **预估字数** (Estimated Words): Target word count for this paragraph
- **AND** multiple paragraphs SHALL form a coherent section
- **AND** transitions between paragraphs SHALL be specified

#### Scenario: Example writing instruction for Section 1.1.1
- **WHEN** creating writing instructions for section 1.1.1 (课题来源)
- **THEN** paragraph 1 SHALL specify:
  - **写作要点**:
    - Describe the increasing use of encryption in network traffic
    - State that traditional deep learning methods fail with unknown traffic types
    - Introduce the two core problems: OSR and IL
  - **素材来源**: Background knowledge, reference papers
  - **支撑创新点**: Problem motivation
  - **预估字数**: 200-300 words

### Requirement: Research Log Mapping
The chapter content outline SHALL establish a mapping from research log entries to thesis paragraphs.

**Description**:
The research log (`.agent/研究日志.md`) contains the complete research history. The outline shall specify which log entries provide material for which thesis sections, ensuring all key findings are incorporated.

#### Scenario: Mapping research log to Chapter 3
- **WHEN** mapping research log entries to Chapter 3 (GEE架构设计)
- **THEN** the following mappings SHALL be established:
  - 3.1节 (GEE整体框架): 参考11月7日日志（门控网络设计思路）
  - 3.2节 (基准与专家模型): 参考11月1日日志（专家数据集构建）
  - 3.3节 (门控网络设计): 参考11月7日日志（门控网络架构）
  - 3.4节 (加权损失策略): 参考11月9日和11月16日日志（加权CE实现与结果）
  - 3.5节 (垃圾类机制): 参考12月6日日志（垃圾类策略与实验结果）
- **AND** for each mapping, extract key findings, data, and quotes from the log
- **AND** synthesize multiple log entries if needed for a section

#### Scenario: Mapping research log to Chapter 4
- **WHEN** mapping research log entries to Chapter 4 (实验设计与分析)
- **THEN** the following mappings SHALL be established:
  - 4.2节 (OSR实验): 参考12月6日日志（GEE OSR结果）和11月22日日志（基线OSR结果）
  - 4.3节 (IL实验): 参考12月19日日志（CNN-GEE结果）和11月16日日志（ResNet-GEE结果）
  - 4.4节 (消融实验): 参考11月16日日志（加权CE vs 简单加权）
  - 4.5节 (多模型验证): 参考12月19日日志（CNN-GEE验证）
- **AND** experimental data (tables, figures) SHALL be extracted from log files

#### Scenario: Mapping research log to Chapter 5
- **WHEN** mapping research log entries to Chapter 5 (理论分析与讨论)
- **THEN** the following mappings SHALL be established:
  - 5.1节 (门控网络决策模式): Based on experimental analysis (not directly in log)
  - 5.2节 (加权损失影响): Based on 11月9日和11月16日实验结果
  - 5.3节 (垃圾类作用机制): Based on 12月6日实验结果
- **AND** theoretical analysis SHALL be derived from experimental results in the log

### Requirement: Content Emphasis and Depth
The chapter content outline SHALL specify the emphasis and depth for each chapter.

**Description**:
Not all chapters should have the same depth. Core chapters (3, 4, 5) should be detailed, while background chapters (1, 2) can be more concise. The outline shall specify word counts and detail levels for each chapter.

#### Scenario: Content depth for Chapter 1 (绪论)
- **WHEN** defining content depth for Chapter 1
- **THEN** the chapter SHALL be concise (5-8 pages)
- **AND** SHALL focus on:
  - Clearly stating the two core problems
  - Motivating the need for GEE architecture
  - Previewing the thesis organization
- **AND** SHALL NOT go into technical details (those belong to Chapter 3)
- **AND** SHALL be easy for non-experts to understand

#### Scenario: Content depth for Chapter 2 (相关理论与技术)
- **WHEN** defining content depth for Chapter 2
- **THEN** the chapter SHALL provide comprehensive background (8-12 pages)
- **AND** SHALL cover:
  - Open-set recognition: definitions, metrics (AUROC, FPR@TPR95), existing methods
  - Incremental learning: definitions, metrics (Macro-F1), existing methods
  - Mixture-of-Experts: architecture, training strategies, applications
  - Encrypted traffic classification: challenges, existing approaches
- **AND** SHALL cite 15-20 key references
- **AND** SHALL identify gaps that motivate GEE (transition to Chapter 3)

#### Scenario: Content depth for Chapter 3 (GEE架构设计)
- **WHEN** defining content depth for Chapter 3
- **THEN** the chapter SHALL be highly detailed (12-18 pages) as it's a core contribution
- **AND** SHALL provide:
  - Complete architecture diagrams (figures)
  - Mathematical formulations (equations)
  - Algorithm descriptions (pseudocode if needed)
  - Design rationale for each component
- **AND** SHALL be detailed enough for a researcher to reimplement GEE

#### Scenario: Content depth for Chapter 4 (实验设计与分析)
- **WHEN** defining content depth for Chapter 4
- **THEN** the chapter SHALL be the most detailed (15-22 pages) as it contains primary results
- **AND** SHALL include:
  - Complete experimental setup (datasets, baselines, metrics)
  - Detailed result tables (at least 4 main tables)
  - Visualization figures (at least 6 main figures)
  - Ablation study results
  - Multi-model validation results
- **AND** SHALL present sufficient data to support all claims

#### Scenario: Content depth for Chapter 5 (理论分析与讨论)
- **WHEN** defining content depth for Chapter 5
- **THEN** the chapter SHALL provide deep analysis (8-12 pages)
- **AND** SHALL go beyond empirical results to explain mechanisms:
  - Parameter distribution visualizations
  - Decision boundary comparisons
  - Gradient direction analyses
  - Feature space separations
- **AND** SHALL include theoretical derivations where appropriate
- **AND** SHALL validate theoretical claims against experimental results

### Requirement: Experiment Result Integration
The chapter content outline SHALL specify how to integrate experimental results into the narrative.

**Description**:
Experimental results should not be presented in isolation but should be woven into the narrative to support specific claims. The outline shall specify which results support which claims and how to present them.

#### Scenario: Integrating OSR results into Chapter 4
- **WHEN** integrating OSR experimental results into Chapter 4
- **THEN** for the claim "GEE significantly improves OSR performance":
  - Present Table 4-1: AUROC and FPR@TPR95 for ResNet-Baseline vs ResNet-GEE
  - Highlight that for hard folds (classes 8, 9, 10 as unknown), FPR@TPR95 improves from 0.8+ to 0.03
  - Discuss the implication: GEE can reliably detect unknown traffic
  - Reference Section 3.5 (garbage class mechanism) as the enabler
- **AND** follow the pattern: claim → evidence → discussion → linkage to design

#### Scenario: Integrating IL results into Chapter 4
- **WHEN** integrating IL experimental results into Chapter 4
- **THEN** for the claim "GEE significantly improves IL performance":
  - Present Table 4-2: Macro-F1 and per-class F1 for Baseline vs GEE
  - Highlight that Macro-F1 improves from 0.63 to 0.83
  - Highlight that minority class F1 improves from 0 to 0.92 (class 7)
  - Discuss the implication: GEE can learn from few-shot samples
  - Reference Section 3.4 (weighted loss) as the enabler
- **AND** follow the pattern: claim → evidence → discussion → linkage to design

#### Scenario: Integrating ablation results into Chapter 4
- **WHEN** integrating ablation study results into Chapter 4
- **THEN** for each ablation comparison:
  - State the hypothesis: "Component X contributes to performance"
  - Present results: Table 4-3 showing metrics with/without component X
  - Interpret results: Does performance drop when X is removed?
  - Conclude: Component X is necessary/sufficient for Y improvement
- **AND** present ablations in logical order: baseline → simple weighted → learned gating (standard CE) → learned gating (weighted CE) → with garbage class

### Requirement: Failed Attempts Discussion
The chapter content outline SHALL specify where and how to discuss failed optimization attempts.

**Description**:
The research log contains several failed attempts (SEBlock, SMOTE, Focal Loss, Macro-F1 optimization). These should be discussed to show the research process, but they should not dominate the narrative.

#### Scenario: Discussing failed attempts in Chapter 2
- **WHEN** discussing failed attempts in Chapter 2 (相关理论与技术)
- **THEN** these attempts MAY be mentioned as "existing optimization methods that were tried":
  - Briefly mention attention mechanisms (SEBlock) and why they failed (overfitting)
  - Briefly mention data augmentation (SMOTE) and why it failed (severe overfitting)
  - Briefly mention Focal Loss and why it failed (performance collapse)
  - Conclude: These single-module optimizations are insufficient; need architectural changes
- **AND** this discussion SHALL motivate the need for GEE (architectural solution)

#### Scenario: Discussing failed attempts in Chapter 6
- **WHEN** discussing failed attempts in Chapter 6 (总结与展望)
- **THEN** these attempts MAY be summarized as "lessons learned":
  - Tried single-module optimizations (attention, augmentation, loss functions) - all failed
  - Realized need for architectural innovation (GEE)
  - This insight guided the research direction
- **AND** this summary SHALL provide research wisdom for future work

### Requirement: Figure and Table References
The chapter content outline SHALL specify where to place figures and tables in the narrative.

**Description**:
Figures and tables should be placed near where they are first referenced in the text. The outline shall specify which figures/tables appear in which sections.

#### Scenario: Figure placement in Chapter 3
- **WHEN** placing figures in Chapter 3 (GEE架构设计)
- **THEN** the following figures SHALL be included:
  - Figure 3-1: GEE overall architecture diagram (in Section 3.1)
  - Figure 3-2: Gating network internal architecture (in Section 3.3)
  - Figure 3-3: Weight calculation formula visualization (in Section 3.4)
  - Figure 3-4: Garbage class mechanism diagram (in Section 3.5)
- **AND** each figure SHALL be referenced in the text before it appears
- **AND** each figure SHALL have a descriptive caption explaining key insights

#### Scenario: Table placement in Chapter 4
- **WHEN** placing tables in Chapter 4 (实验设计与分析)
- **THEN** the following tables SHALL be included:
  - Table 4-1: OSR performance comparison (ResNet-Baseline vs ResNet-GEE)
  - Table 4-2: IL performance comparison (Baseline vs GEE)
  - Table 4-3: Ablation study results (all configurations)
  - Table 4-4: CNN-GEE validation results
- **AND** each table SHALL be referenced in the text before it appears
- **AND** each table SHALL have a descriptive title and clear column headers

### Requirement: Transition Sentences
The chapter content outline SHALL specify transition sentences between paragraphs and sections.

**Description**:
Good academic writing has clear transitions between ideas. The outline shall specify transition sentences to ensure smooth flow.

#### Scenario: Transition from Section 3.4 to 3.5
- **WHEN** transitioning from Section 3.4 (加权损失) to Section 3.5 (垃圾类机制)
- **THEN** the transition paragraph SHALL state:
  - Recap: Weighted loss addresses IL (few-shot learning) by balancing gradient contributions
  - Motivation: But GEE must also address OSR (unknown traffic detection)
  - Preview: Next section introduces the garbage class mechanism for OSR
- **AND** this transition SHALL connect the two core innovation threads (IL and OSR)

#### Scenario: Transition from Chapter 4 to Chapter 5
- **WHEN** transitioning from Chapter 4 (experiments) to Chapter 5 (theoretical analysis)
- **THEN** the final paragraph of Chapter 4 SHALL state:
  - Recap: Experiments demonstrate GEE's effectiveness (empirical results)
  - Question: But WHY does GEE work? What are the underlying mechanisms?
  - Preview: Chapter 5 provides theoretical analysis to explain GEE's success
- **AND** this transition SHALL set up the theoretical depth of Chapter 5
