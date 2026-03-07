# Spec: Thesis Chapter Enhancement

## ADDED Requirements

### Requirement: Add encryption technology background to Section 1.1.1

The thesis Chapter 1, Section 1.1.1 SHALL include technical explanations of encryption technologies (HTTPS, VPN, Tor) immediately after the statement about encryption adoption rates. The explanations SHALL provide sufficient technical depth for readers from non-security backgrounds while maintaining conciseness.

#### Scenario: HTTPS technical explanation
- **GIVEN** the reader is at line 31 of main.md where encryption technologies are mentioned
- **WHEN** reading the background section
- **THEN** reader SHALL find explanation of HTTPS/TLS protocol including:
  - TLS/SSL handshake process (ClientHello, ServerHello, certificate exchange)
  - Encryption principles (symmetric and asymmetric cryptography)
  - Why payload content becomes invisible to DPI
  - Quantitative adoption rate (80%+ of Internet traffic) with citation

#### Scenario: VPN technical explanation
- **GIVEN** the reader is learning about encrypted traffic challenges
- **WHEN** reading the VPN background section
- **THEN** reader SHALL find explanation of VPN technology including:
  - Tunneling concept (encapsulation in outer packets)
  - Encryption layers (point-to-point or end-to-end)
  - Common protocols (IPsec, OpenVPN, WireGuard)
  - How VPN hides application-layer signatures
  - Impact on traffic classification methods

#### Scenario: Tor network explanation
- **GIVEN** the reader needs to understand anonymity networks
- **WHEN** reading the Tor background section
- **THEN** reader SHALL find explanation of Tor network including:
  - Onion routing principle (multi-layer encryption)
  - Circuit-based routing through relay nodes
  - How each relay only knows immediate predecessor and successor
  - Anonymity properties achieved
  - Challenges for traffic classification

#### Scenario: Connection to DPI failure
- **GIVEN** the reader has learned about encryption technologies
- **WHEN** reading the transition to deep learning solutions
- **THEN** the text SHALL explicitly explain:
  - Why port-based identification fails (dynamic ports, port hopping)
  - Why DPI fails (payload encryption, signature hiding)
  - The logical necessity of learning from non-payload features
  - Smooth transition to deep learning as solution

---

### Requirement: Integrate quantitative data for "fundamental flaws" claims

The thesis Chapter 1, Section 1.1.1 SHALL support all claims about baseline model's "fundamental flaws" with specific quantitative data from experimental results in Chapters 4-5. Each claim SHALL include exact metrics with references to experimental sections.

#### Scenario: OSR failure quantification
- **GIVEN** the text states baseline model cannot recognize unknown classes
- **WHEN** examining the supporting evidence
- **THEN** the text SHALL include:
  - FPR@TPR95 values for different unknown class scenarios (minority vs majority classes as unknown)
  - Specific values: FPR@TPR95 > 0.8 when majority classes (8, 9, 10) are unknown
  - Comparison: FPR@TPR95 < 0.1 when minority classes (5, 7) are unknown
  - Reference to Chapter 4, Section 4.2 for detailed experiments
  - Interpretation: "高置信度错误分类" (high-confidence misclassification)

#### Scenario: Class imbalance quantification
- **GIVEN** the text states baseline model performs poorly on minority classes
- **WHEN** examining the quantitative support
- **THEN** the text SHALL include:
  - Sample count disparity: 9476 (class 10) vs 65 (class 7) = 146x difference
  - Performance metrics: F1=0 for classes 5 and 7 under baseline training
  - Overall accuracy paradox: 96.33% accuracy but Macro-F1=0.63
  - Reference to Chapter 4, Section 4.1 for dataset analysis
  - Clear statement: gradient contribution dominated by majority classes

#### Scenario: Confidence distribution analysis
- **GIVEN** the text mentions overconfidence on unknown samples
- **WHEN** reviewing supporting data
- **THEN** the text SHALL include:
  - Softmax entropy comparison: baseline (0.35) vs GEE (1.18) on unknown samples
  - Interpretation: low entropy = high confidence = wrong
  - Reference to Chapter 5, Section 5.4 for entropy analysis
  - Explanation of why this is catastrophic in security scenarios

#### Scenario: GEE improvement quantification
- **GIVEN** the text states GEE solves these problems
- **WHEN** examining the quantitative evidence
- **THEN** the text SHALL include:
  - OSR improvement: FPR@TPR95 from 0.47 to 0.03 (93% reduction)
  - IL improvement: Macro-F1 from 0.63 to 0.83 (32% increase)
  - Minority class improvement: F1 from 0 to 0.92 (infinite improvement)
  - Reference to specific experimental sections for each claim

---

### Requirement: Add MoE background before GEE innovation description

The thesis Chapter 1, Section 1.1.2 SHALL introduce traditional Mixture-of-Experts (MoE) architecture BEFORE presenting GEE's innovation. This establishes the "before" state so readers can appreciate the innovation.

#### Scenario: Traditional MoE architecture introduction
- **GIVEN** the reader is in Section 1.1.2 where innovations are listed
- **WHEN** reading about GEE's first innovation
- **THEN** the text SHALL first explain traditional MoE:
  - Multiple expert networks (typically 8-32 experts in literature)
  - Gating network for dynamic expert selection
  - Sparse activation mechanism (only top-k experts activated)
  - Load balancing challenges (ensuring all experts are utilized)
  - Expert specialization (different experts learn different patterns)
  - Training complexity (balancing multiple loss functions)
  - Computational overhead (multiple forward passes, large parameter count)
  - Citation to seminal MoE papers (Shazeer et al., 2017)

#### Scenario: MoE challenges in encrypted traffic classification
- **GIVEN** the reader understands traditional MoE
- **WHEN** learning why MoE isn't widely used in traffic classification
- **THEN** the text SHALL explain:
  - Complexity of training multiple experts on limited data
  - Difficulty in achieving expert specialization
  - Computational requirements for real-time traffic classification
  - The gap between MoE theory and practical deployment

#### Scenario: GEE's two-expert innovation
- **GIVEN** the reader understands traditional MoE complexity
- **WHEN** reading about GEE's innovation
- **THEN** the text SHALL present:
  - The hypothesis: Can two-expert MoE achieve excellent results?
  - The simplification: baseline (majority expert) + minority expert + gating
  - The proof: Experimental results showing competitive performance
  - The advantage: Lower training/inference cost, simpler architecture
  - Clear positioning: Innovation is in proving simplicity works, not in complexity

#### Scenario: Narrative flow from background to innovation
- **GIVEN** the reader follows the logical progression
- **WHEN** reading the complete section
- **THEN** the flow SHALL be:
  1. Traditional MoE exists and is powerful but complex
  2. MoE has challenges in traffic classification domain
  3. GEE proves two-expert MoE can work well
  4. This simplification enables practical deployment

---

### Requirement: Correct weighted cross-entropy loss innovation attribution

The thesis Chapter 1, Section 1.1.2 SHALL correctly describe the weighted cross-entropy loss contribution as an "innovative application" of an existing technique to a new problem context, NOT as the invention of weighted loss itself.

#### Scenario: Correct innovation formulation
- **GIVEN** the original text incorrectly states "提出了加权交叉熵损失训练策略"
- **WHEN** reading the corrected text
- **THEN** the formulation SHALL be:
  - "创新性地将加权交叉熵损失应用于门控网络训练" (innovatively applied weighted CE to gating network training)
  - NOT "提出了加权交叉熵损失" (proposed weighted CE loss)
  - Clear distinction: Weighted CE existed; innovation is in WHERE it's applied

#### Scenario: Specify the innovation components
- **GIVEN** the reader needs to understand what's truly novel
- **WHEN** reading the innovation description
- **THEN** the text SHALL clarify:
  - Original strategy (XX): Weighted CE for class imbalance in classification
  - Modification (ZZ): Applied to gating network, not just final classification
  - Problem solved (YY): Preventing gating network from ignoring minority expert
  - Results achieved: Macro-F1 from 0.63 to 0.83, minority F1 from 0 to 0.92

#### Scenario: Academic integrity maintained
- **GIVEN** the thesis committee evaluates innovation claims
- **WHEN** reviewing the weighted CE contribution
- **THEN** the text SHALL:
  - Attribute weighted CE to existing literature if applicable
  - Position the contribution as "application/adaptation" not "invention"
  - Use precise language: "应用于" (applied to), "迁移到" (migrated to)
  - Avoid claiming priority for well-established techniques

#### Scenario: Results-based justification
- **GIVEN** the innovation claim needs validation
- **WHEN** examining the evidence
- **THEN** the text SHALL include:
  - Quantitative results: Macro-F1 improvement from 0.63 to 0.83
  - Ablation study comparison: Standard CE gating (Config 3) vs Weighted CE gating (Config 4)
  - Reference to Chapter 4, Section 4.4 for detailed experiments
  - Clear causal link: Weighted CE enables gating network to work

---

### Requirement: Clarify transferability claim about CNN model

The thesis Chapter 1 SHALL clarify that "transferability to CNN model" means replacing the backbone network from ResNet to standard CNN while keeping the GEE architecture design unchanged.

#### Scenario: Establish backbone network context
- **GIVEN** the reader first encounters GEE architecture description
- **WHEN** learning about the base model
- **THEN** the text SHALL establish:
  - Deep-Packet framework uses ResNet1d as default backbone
  - ResNet1d provides residual connections for deep networks
  - This is the "backbone network" choice, not the GEE architecture itself
  - GEE is an architecture overlay that works with different backbones

#### Scenario: Explain the transferability experiment
- **GIVEN** the text claims GEE has good transferability
- **WHEN** reading the experimental validation
- **THEN** the text SHALL specify:
  - What was changed: Backbone from ResNet1d to standard CNN
  - What stayed constant: GEE architecture (baseline + expert + gating)
  - What was measured: Performance improvement of GEE over baseline
  - Results: Similar performance gains observed with CNN backbone

#### Scenario: Clear reformulation of ambiguous phrase
- **GIVEN** the original phrase "在CNN模型上也取得了类似的性能提升" is unclear
- **WHEN** reading the corrected version
- **THEN** the text SHALL state:
  - "当将Deep-Packet框架的基础骨干网络从ResNet替换为标准CNN后"
  - "GEE架构仍然展现出显著的性能提升优势"
  - "证明了该架构设计不依赖于特定的骨干网络选择"
  - "具有广泛的适用性"

#### Scenario: Context maintained from earlier sections
- **GIVEN** the transferability claim appears in Section 1.1.2
- **WHEN** the reader encounters it
- **THEN** the context SHALL be clear because:
  - Backbone network was established earlier in the section
  - The distinction between GEE architecture and backbone is clear
  - The transition from ResNet results to CNN results is natural
  - The validation purpose is explicit (testing architecture generality)

#### Scenario: Cross-reference to experimental details
- **GIVEN** the reader wants detailed experimental setup
- **WHEN** following the reference
- **THEN** the text SHALL point to:
  - Chapter 4, Section 4.x (CNN-GEE experiments)
  - Specific performance metrics for CNN baseline vs CNN-GEE
  - Comparison table showing ResNet vs CNN results
  - Discussion of why transferability matters (real-world deployment)

---

### Requirement: Refine incremental learning wording for precision

The thesis Chapter 1, Section 1.2.1 SHALL replace imprecise phrasing "不遗忘旧类别" (not forget old categories) with more academically precise "保持旧类别的识别精度" (maintain recognition accuracy on old categories).

#### Scenario: Replace imprecise wording in research goal
- **GIVEN** the original text states "从少量样本中学习新类别且不遗忘旧类别"
- **WHEN** reading the corrected version
- **THEN** the text SHALL state:
  - "从少量样本中学习新类别且保持旧类别的识别精度"
  - Alternative: "在学习新类别的同时，保持对旧类别的识别性能"
  - Emphasis on positive "maintain accuracy" not negative "not forget"

#### Scenario: Consistency throughout the section
- **GIVEN** the concept of incremental learning appears multiple times
- **WHEN** reading the complete Section 1.2.1
- **THEN** all instances SHALL use precise terminology:
  - "保持识别精度" (maintain recognition accuracy)
  - "维持性能" (maintain performance)
  - "避免性能下降" (avoid performance degradation)
  - NOT "不遗忘" (not forget) or informal variations

#### Scenario: Academic writing standard maintained
- **GIVEN** Chinese academic writing conventions
- **WHEN** reviewing the phrasing
- **THEN** the language SHALL be:
  - Formal: "保持" not "不丢" or "没忘"
  - Precise: "识别精度" not "认识" or "知道"
  - Professional: "性能维持" not "还是好的"
  - Consistent with terminology in later chapters (Chapter 2, 3, 5)

#### Scenario: Connection to catastrophic forgetting problem
- **GIVEN** the text mentions "灾难性遗忘" (catastrophic forgetting)
- **WHEN** explaining what GEE prevents
- **THEN** the formulation SHALL be:
  - "避免灾难性遗忘问题" (avoid catastrophic forgetting problem)
  - "在学习新类别时，保持旧类别的识别精度" (maintain old class accuracy while learning new classes)
  - Link to experimental evidence: Reference Chapter 4 incremental learning results

---

### Requirement: Expand validation description to multi-dimensional

The thesis Chapter 1, Section 1.2.2 SHALL expand the brief phrase "多模型验证与理论分析" to accurately reflect the comprehensive, multi-dimensional nature of experimental validation.

#### Scenario: Comprehensive enumeration of validation dimensions
- **GIVEN** the original text only mentions "多模型验证"
- **WHEN** reading the expanded description
- **THEN** the text SHALL enumerate:
  - **Multiple backbone models**: ResNet, CNN (testing architecture generality)
  - **Different gating architectures**: Various MLP designs (testing robustness)
  - **Multiple datasets**: VPN and potentially others (testing generalizability)
  - **Ablation studies**: Systematic removal of components (testing contribution)
  - **Theoretical analysis**: Weight distribution, decision boundaries, gradients (testing understanding)

#### Scenario: Proper terminology update
- **GIVEN** the phrase "多模型验证" is insufficient
- **WHEN** reading the corrected version
- **THEN** the text SHALL use:
  - "多维度的验证实验与理论分析" (multi-dimensional validation experiments and theoretical analysis)
  - NOT just "多模型验证" (multi-model validation)
  - Accurately reflects the breadth of validation work

#### Scenario: Detailed breakdown in research content summary
- **GIVEN** Section 1.2.2 lists six aspects of research work
- **WHEN** reading the sixth aspect
- **THEN** the text SHALL state:
  - "第六，多维度的验证实验与理论分析。"
  - Followed by detailed enumeration:
    - "本研究不仅验证了不同骨干网络（ResNet、CNN）下的架构有效性"
    - "还探索了不同门控网络架构、不同数据集场景下的性能表现"
    - "通过控制变量的消融实验，系统地验证了GEE架构各组件的贡献"
    - "此外，从门控网络权重分布、决策边界几何特性、梯度贡献机制等多个角度进行了深入的理论分析"
    - "揭示了GEE架构有效的深层机制"

#### Scenario: Connection to experimental chapters
- **GIVEN** the reader wants to see detailed results
- **WHEN** following the validation discussion
- **THEN** the text SHALL provide clear pointers:
  - Reference Chapter 4 for experimental results
  - Reference Chapter 5 for theoretical analysis
  - Specific sections for each validation dimension
  - Logical flow from claim → evidence → detailed analysis

#### Scenario: Academic rigor conveyed
- **GIVEN** thesis committees evaluate thoroughness
- **WHEN** reviewing the validation description
- **THEN** the text SHALL demonstrate:
  - Systematic approach (not just random experiments)
  - Control of variables (ablation studies)
  - Multiple angles of analysis (experimental + theoretical)
  - Comprehensive coverage (architecture, datasets, components)
  - Depth of analysis (mechanisms, not just performance)

---

## Success Criteria

### Overall Quality Criteria

1. **Completeness**: All 7 modifications implemented in Chapter 1
2. **Technical Accuracy**: All technical explanations are correct and citable
3. **Quantitative Support**: All claims have supporting numbers with references
4. **Academic Integrity**: Innovation claims properly attributed
5. **Coherence**: Changes maintain logical flow across sections
6. **Consistency**: Terminology aligned across all chapters
7. **Writing Quality**: Formal Chinese academic style throughout
8. **Length Balance**: Chapter 1 not disproportionately long
9. **Reference Integrity**: All cross-references remain valid
10. **Citation Properness**: New content has appropriate citations

### Modification-Specific Criteria

**Modification 1 (Encryption Background)**:
- [ ] 2-3 paragraphs per technology (HTTPS, VPN, Tor)
- [ ] Technical depth appropriate for introduction (not too shallow, not too deep)
- [ ] Clear connection to DPI failure explained
- [ ] Citations included for protocol details and adoption statistics
- [ ] Smooth transition to deep learning solutions

**Modification 2 (Quantitative Data)**:
- [ ] FPR@TPR95 values included with specific numbers
- [ ] Class imbalance metrics (146x, F1=0) present
- [ ] Entropy comparisons included
- [ ] All numbers reference specific experimental sections
- [ ] Interpretation provided for why numbers matter

**Modification 3 (MoE Background)**:
- [ ] Traditional MoE explained before GEE innovation
- [ ] MoE components and challenges covered
- [ ] Two-expert innovation positioned clearly
- [ ] Citations to seminal MoE papers included
- [ ] Logical flow: background → gap → innovation

**Modification 4 (Innovation Correction)**:
- [ ] "提出了" changed to "应用了" or similar
- [ ] Weighted CE properly attributed to existing work
- [ ] Innovation framed as application to gating network
- [ ] Results still mentioned to justify contribution
- [ ] Academic integrity maintained

**Modification 5 (Transferability Clarity)**:
- [ ] Backbone network context established early
- [ ] "CNN model" clarified as "backbone from ResNet to CNN"
- [ ] What changed vs. what stayed constant is clear
- [ ] Cross-reference to detailed experiments included
- [ ] Generality claim properly qualified

**Modification 6 (Wording Refinement)**:
- [ ] "不遗忘" replaced with "保持识别精度"
- [ ] All instances of imprecise wording updated
- [ ] Consistent terminology throughout section
- [ ] Professional academic tone maintained
- [ ] Alignment with later chapters verified

**Modification 7 (Validation Expansion)**:
- [ ] "多模型验证" expanded to "多维度验证"
- [ ] All validation dimensions enumerated
- [ ] Theoretical analysis mentioned alongside experiments
- [ ] Ablation studies explicitly noted
- [ ] Comprehensive nature accurately conveyed
