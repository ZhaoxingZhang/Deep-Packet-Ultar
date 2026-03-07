# Thesis Framework Specification

## ADDED Requirements

### Requirement: Complete Chapter Structure
The thesis framework SHALL define a complete 6-chapter structure following the reference paper format in `.agent/参考论文结构.md`.

**Description**:
Based on the reference thesis structure, the thesis shall have 6 main chapters plus auxiliary sections (abstract, references, appendices, acknowledgment). Each chapter shall be divided into sections, and each section shall be divided into subsections as needed.

#### Scenario: Chapter-level structure
- **WHEN** establishing the thesis chapter structure
- **THEN** the thesis SHALL contain the following chapters in order:
  1. 第一章 绪论 (Introduction)
  2. 第二章 相关理论与技术 (Related Theory and Technology)
  3. 第三章 GEE架构设计 (GEE Architecture Design)
  4. 第四章 实验设计与分析 (Experiment Design and Analysis)
  5. 第五章 理论分析与讨论 (Theoretical Analysis and Discussion)
  6. 第六章 总结与展望 (Conclusion and Future Work)
- **AND** each chapter SHALL start on a new page
- **AND** chapters SHALL be numbered using Chinese numerals (第一章, 第二章, etc.)

#### Scenario: Section-level structure for Chapter 1
- **WHEN** defining the structure for Chapter 1 (绪论)
- **THEN** the chapter SHALL contain the following sections:
  - 1.1 课题来源及研究意义 (Topic Origin and Research Significance)
    - 1.1.1 课题来源 (Topic Origin)
    - 1.1.2 研究意义 (Research Significance)
  - 1.2 研究目标及研究内容 (Research Objectives and Content)
    - 1.2.1 研究目标 (Research Objectives)
    - 1.2.2 研究内容 (Research Content)
  - 1.3 论文组织结构 (Thesis Organization)
  - 1.4 本章小结 (Chapter Summary)
- **AND** each section SHALL have a clear heading with hierarchical numbering

#### Scenario: Section-level structure for Chapter 3
- **WHEN** defining the structure for Chapter 3 (GEE架构设计)
- **THEN** the chapter SHALL contain the following sections:
  - 3.1 GEE架构整体框架 (GEE Overall Framework)
  - 3.2 基准模型与专家模型设计 (Baseline and Expert Model Design)
  - 3.3 门控网络设计 (Gating Network Design)
  - 3.4 加权交叉熵损失训练策略 (Weighted Cross-Entropy Training Strategy)
  - 3.5 垃圾类机制用于开放集识别 (Garbage Class Mechanism for OSR)
  - 3.6 本章小结 (Chapter Summary)
- **AND** each section SHALL correspond to a key component of the GEE architecture

### Requirement: Page Count Allocation
The thesis framework SHALL allocate page counts to each chapter to ensure the total is within 50-90 pages.

**Description**:
To manage the 50-90 page constraint, each chapter shall have a target page count with minimum and maximum bounds. Core chapters (3, 4, 5) shall receive the majority of pages.

#### Scenario: Page count distribution
- **WHEN** allocating page counts to chapters
- **THEN** the distribution SHALL be as follows:
  - 第一章 绪论: 5-8 pages (约15-24页，考虑到参考文献和摘要等)
  - 第二章 相关理论与技术: 8-12 pages
  - 第三章 GEE架构设计: 12-18 pages (核心章节)
  - 第四章 实验设计与分析: 15-22 pages (核心章节)
  - 第五章 理论分析与讨论: 8-12 pages (核心章节)
  - 第六章 总结与展望: 2-4 pages
- **AND** total SHALL be 50-76 pages for main content
- **AND** with abstract, references, appendices: 50-90 pages total

#### Scenario: Page count adjustment during writing
- **WHEN** a chapter exceeds its maximum page count during writing
- **THEN** the writer SHALL identify less critical sections to condense
- **AND** SHALL maintain logical coherence while reducing length
- **AND** MAY move detailed content to appendices if appropriate
- **AND** SHALL rebalance by expanding other chapters that are under minimum

### Requirement: Word Count Estimation
The thesis framework SHALL estimate word counts for each chapter based on page allocations.

**Description**:
Chinese academic theses typically have 400-600 words per page. The framework shall estimate word counts to guide writing progress and ensure balanced content density.

#### Scenario: Word count calculation
- **WHEN** estimating word counts for each chapter
- **THEN** system SHALL assume 400-600 words per page (average 500 words/page)
- **AND** SHALL calculate target word counts:
  - 第一章 绪论: 2,500-4,000 words
  - 第二章 相关理论与技术: 4,000-6,000 words
  - 第三章 GEE架构设计: 6,000-9,000 words
  - 第四章 实验设计与分析: 7,500-11,000 words
  - 第五章 理论分析与讨论: 4,000-6,000 words
  - 第六章 总结与展望: 1,000-2,000 words
- **AND** total SHALL be approximately 25,000-38,000 words for main content

### Requirement: Cross-Chapter Logical Flow
The thesis framework SHALL define the logical flow and dependencies between chapters.

**Description**:
Each chapter shall have a clear purpose and build upon previous chapters. The framework shall explicitly state how chapters connect to ensure a coherent narrative.

#### Scenario: Logical flow from Chapter 1 to Chapter 2
- **WHEN** transitioning from Chapter 1 (绪论) to Chapter 2 (相关理论与技术)
- **THEN** Chapter 1 SHALL state the two core problems (OSR and IL)
- **AND** Chapter 2 SHALL provide background on these problems:
  - Open-set recognition existing methods and their limitations
  - Incremental learning existing methods and their limitations
  - Mixture-of-Experts models as the foundation for GEE
- **AND** this background SHALL motivate the need for the GEE architecture in Chapter 3

#### Scenario: Logical flow from Chapter 2 to Chapter 3
- **WHEN** transitioning from Chapter 2 (相关理论与技术) to Chapter 3 (GEE架构设计)
- **THEN** Chapter 2 SHALL have identified gaps in existing methods:
  - Existing OSR methods (Softmax threshold) have high FPR@TPR95
  - Existing IL methods (full fine-tuning) are inefficient and cause catastrophic forgetting
- **AND** Chapter 3 SHALL present GEE as the solution to address these gaps
- **AND** Chapter 3 SHALL reference the limitations identified in Chapter 2

#### Scenario: Logical flow from Chapter 3 to Chapter 4
- **WHEN** transitioning from Chapter 3 (GEE架构设计) to Chapter 4 (实验设计与分析)
- **THEN** Chapter 3 SHALL have presented the complete GEE architecture
- **AND** Chapter 4 SHALL validate this architecture through experiments:
  - Does GEE actually improve OSR performance? (Test with 6-fold CV)
  - Does GEE actually improve IL performance? (Test with Macro-F1)
  - Which components contribute most? (Ablation studies)
- **AND** Chapter 4 experiments SHALL directly correspond to design decisions in Chapter 3

#### Scenario: Logical flow from Chapter 4 to Chapter 5
- **WHEN** transitioning from Chapter 4 (实验设计与分析) to Chapter 5 (理论分析与讨论)
- **THEN** Chapter 4 SHALL have demonstrated that GEE works (performance improvements)
- **AND** Chapter 5 SHALL explain WHY GEE works (theoretical mechanisms):
  - How does the gating network decide between baseline and expert?
  - How does weighted loss affect gradient directions?
  - How does the garbage class separate unknown samples in feature space?
- **AND** Chapter 5 SHALL provide depth beyond the empirical results in Chapter 4

### Requirement: Two Core Innovation Threads
The thesis framework SHALL ensure two core innovation threads (OSR and IL) run consistently through all chapters.

**Description**:
The thesis addresses two core problems: open-set recognition and incremental learning. The framework shall ensure both problems are introduced, designed, evaluated, and analyzed consistently across chapters.

#### Scenario: OSR thread across chapters
- **WHEN** tracing the OSR innovation thread through the thesis
- **THEN** the thread SHALL appear as:
  - Chapter 1: Problem statement - models misclassify unknown traffic with high confidence
  - Chapter 2: Background - existing OSR methods and their limitations (Softmax threshold fails)
  - Chapter 3: Solution - garbage class mechanism in GEE architecture
  - Chapter 4: Validation - OSR experiments show FPR@TPR95 improvement from 0.8+ to 0.03
  - Chapter 5: Analysis - why garbage class works (feature space separation)
  - Chapter 6: Summary - OSR capability is a key contribution

#### Scenario: IL thread across chapters
- **WHEN** tracing the IL innovation thread through the thesis
- **THEN** the thread SHALL appear as:
  - Chapter 1: Problem statement - models cannot learn from few-shot minority classes (F1=0)
  - Chapter 2: Background - existing IL methods (full fine-tuning is inefficient)
  - Chapter 3: Solution - minority expert + weighted loss in GEE
  - Chapter 4: Validation - IL experiments show Macro-F1 improvement from 0.63 to 0.83
  - Chapter 5: Analysis - why weighted loss works (gradient direction balancing)
  - Chapter 6: Summary - IL capability is a key contribution

### Requirement: Hierarchical Heading Numbering
The thesis framework SHALL define a consistent hierarchical heading numbering system.

**Description**:
To ensure clarity and navigability, all headings shall follow a consistent numbering scheme: chapters (一), sections (一、1), subsections (一、1、(1)).

#### Scenario: Heading numbering format
- **WHEN** numbering headings in the thesis
- **THEN** the format SHALL be:
  - Chapter level: 第一章, 第二章, etc.
  - Section level: 1.1, 1.2, 3.1, 3.2, etc.
  - Subsection level: 1.1.1, 1.1.2, 3.1.1, 3.1.2, etc.
  - Sub-subsection level: 1.1.1.1, etc. (if needed)
- **AND** numbering SHALL be automatic in LaTeX or manual in Word
- **AND** all headings SHALL be included in the table of contents

### Requirement: Auxiliary Sections Structure
The thesis framework SHALL define the structure of auxiliary sections (abstract, references, appendices).

**Description**:
In addition to the 6 main chapters, the thesis shall include an abstract, references, appendices (if needed), and acknowledgment. These sections shall follow standard academic format.

#### Scenario: Abstract structure
- **WHEN** writing the Chinese abstract (中文摘要)
- **THEN** it SHALL be 300-500 words
- **AND** SHALL include:
  1. Background and problem statement (1-2 sentences)
  2. Proposed approach (GEE architecture)
  3. Key innovations (garbage class, weighted loss, etc.)
  4. Main results (FPR@TPR95 and Macro-F1 improvements)
  5. Conclusions and impact
- **AND** SHALL be placed at the beginning of the thesis
- **AND** MAY include an English abstract if required

#### Scenario: References structure
- **WHEN** organizing references
- **THEN** references SHALL follow GB/T 7714-2015 format
- **AND** SHALL be cited in the text using [1], [2] numbering
- **AND** SHALL include:
  - Papers on encrypted traffic classification
  - Papers on open-set recognition
  - Papers on incremental learning
  - Papers on mixture-of-experts models
  - Papers on baseline methods (Deep-Packet, etc.)
- **AND** SHALL have 30-50 references for a master's thesis

#### Scenario: Appendices structure
- **WHEN** including appendices
- **THEN** appendices SHALL contain supplementary material that supports the main text:
  - Appendix A: Detailed experimental results tables (e.g., CNN-GEE results, per-class F1 breakdowns)
  - Appendix B: Additional ablation study results
  - Appendix C: Algorithm pseudocode (if appropriate)
- **AND** each appendix SHALL be referenced in the main text
- **AND** appendices SHALL be labeled 附录A, 附录B, etc.
