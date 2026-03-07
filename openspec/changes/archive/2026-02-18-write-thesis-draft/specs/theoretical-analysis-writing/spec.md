# Theoretical Analysis Writing Specification

## ADDED Requirements

### Requirement: Theoretical Analysis Depth and Methodology
The theoretical analysis writing SHALL define the depth and methodology for analyzing WHY GEE is effective.

**Description**:
Chapter 5 (理论分析与讨论) goes beyond empirical results ("it works") to explain mechanisms ("why it works"). This requirement specifies the analytical methods and depth for each analysis.

#### Scenario: Analysis depth requirements
- **WHEN** determining the depth of theoretical analysis
- **THEN** each analysis section SHALL include:
  1. **Hypothesis**: What mechanism do we expect to observe?
  2. **Method**: How do we analyze it (what data, what visualization)?
  3. **Observation**: What do the data/visualizations actually show?
  4. **Interpretation**: What does this mean for GEE's effectiveness?
  5. **Validation**: Does this align with experimental results from Chapter 4?
- **AND** each analysis SHALL be 1.5-2 pages (with figures)
- **AND** total Chapter 5 SHALL be 8-12 pages
- **AND** analysis SHALL be rigorous enough for master's thesis standards

#### Scenario: Mathematical formalization where appropriate
- **WHEN** theoretical analysis involves mathematical relationships
- **THEN** system SHALL provide mathematical formulations:
  - Weighted cross-entropy loss function: L = -Σ w_i * log(p_i)
  - Gradient of weighted loss: ∂L/∂θ = -Σ w_i * (∂log(p_i)/∂θ)
  - Show how w_i scales gradient contribution for class i
- **AND** equations SHALL be typeset in LaTeX using `equation` environments
- **AND** equations SHALL be numbered and referenced in text
- **AND** derivations SHALL be clear and step-by-step

### Requirement: Gating Network Decision Pattern Analysis
The theoretical analysis writing SHALL analyze the gating network's learned decision patterns.

**Description**:
By examining the gating network's internal parameters and outputs, we can understand whether it learned a meaningful fusion strategy (trust expert for minority classes) or defaulted to always trusting the baseline.

#### Scenario: Weight distribution analysis writing
- **WHEN** writing Section 5.1 (门控网络决策模式分析)
- **THEN** the section SHALL include:
  - **Paragraph 1**: Motivation - Why analyze gating network weights?
  - **Paragraph 2**: Method - Extracted weights from which layer? How to visualize?
  - **Paragraph 3**: Observation - Describe Figure 5-1 (weight heatmap)
  - **Paragraph 4**: Interpretation - Are expert weights larger than baseline weights for minority classes?
  - **Paragraph 5**: Validation - Link to Chapter 4 results (Macro-F1 improvement)
- **AND** reference Figure 5-1 in Paragraph 3
- **AND** use specific values from analysis (e.g., "Expert weights are 2.3x larger for class 7")

#### Scenario: Output contribution analysis writing
- **WHEN** analyzing how much the gating network trusts baseline vs expert
- **THEN** the section SHALL include:
  - **Method**: For each test sample, record the gating network's weighted contribution from baseline inputs vs expert inputs
  - **Observation**: For majority class samples, baseline contribution > expert contribution; for minority class samples, expert contribution > baseline contribution
  - **Visualization**: Bar chart (Figure 5-1b) showing average contribution per class
  - **Interpretation**: Gating network learned "when to trust expert"
- **AND** this analysis SHALL directly support the claim: "GEE leverages expert for minority classes"

### Requirement: Expert Complementarity Analysis
The theoretical analysis writing SHALL analyze and visualize the complementarity between baseline and expert models.

**Description**:
GEE's effectiveness relies on baseline and expert having complementary strengths. This analysis visualizes their decision boundaries to show how GEE combines them.

#### Scenario: Decision boundary visualization writing
- **WHEN** writing Section 5.2 (专家决策边界分析)
- **THEN** the section SHALL include:
  - **Paragraph 1**: Motivation - Why analyze decision boundaries?
  - **Paragraph 2**: Method - How to project high-dimensional features to 2D? (PCA or t-SNE)
  - **Paragraph 3**: Observation - Describe Figure 5-2 (three subplots: baseline, expert, GEE)
  - **Paragraph 4**: Interpretation - Baseline separates majority classes well, Expert separates minority classes well, GEE combines both
  - **Paragraph 5**: Validation - Link to Macro-F1 improvement (GEE > both individual models)
- **AND** reference Figure 5-2 in Paragraph 3
- **AND** quantify complementarity if possible (e.g., "Baseline accuracy on minority classes: 0%, Expert: 95%, GEE: 92%")

#### Scenario: Prediction agreement analysis writing
- **WHEN** analyzing when baseline and expert agree vs disagree
- **THEN** the section SHALL include:
  - **Method**: For each test sample, categorize as:
    - Both correct
    - Both wrong
    - Baseline correct, expert wrong
    - Baseline wrong, expert correct
  - **Observation**: Table or bar chart showing percentage of samples in each category
  - **Interpretation**: For minority classes, "Baseline wrong, expert correct" is high → Expert is essential
  - **Validation**: This explains why GEE's Macro-F1 is higher than baseline alone
- **AND** quantify percentages (e.g., "For class 7, baseline correct expert wrong: 5%, baseline wrong expert correct: 85%")

### Requirement: Weighted Loss Gradient Analysis
The theoretical analysis writing SHALL analyze how weighted cross-entropy changes gradient directions during training.

**Description**:
Weighted cross-entropy is key to preventing the gating network from ignoring the expert. This analysis shows HOW it works (gradient scaling) rather than just THAT it works (better Macro-F1).

#### Scenario: Gradient contribution analysis writing
- **WHEN** writing Section 5.3 (加权损失影响机制)
- **THEN** the section SHALL include:
  - **Paragraph 1**: Motivation - Why is weighted loss necessary? (Recall: Standard CE caused gating network to ignore expert, Macro-F1 dropped to 0.63)
  - **Paragraph 2**: Method - How to calculate gradient contribution per class? (Record gradients during training, calculate L2 norm per class)
  - **Paragraph 3**: Observation - Describe Figure 5-4 (bar chart: gradient contribution per class, unweighted vs weighted)
  - **Paragraph 4**: Interpretation - Unweighted: Majority classes dominate gradients; Weighted: Balanced across classes
  - **Paragraph 5**: Validation - Link to ablation results (Macro-F1: 0.67 simple weighted → 0.83 weighted CE)
- **AND** reference Figure 5-4 in Paragraph 3
- **AND** use mathematical formulation to explain gradient scaling

#### Scenario: Training trajectory comparison writing
- **WHEN** comparing training with and without class weighting
- **THEN** the section SHALL include:
  - **Method**: Train two gating networks (same architecture, same data, different loss functions)
  - **Visualization**: Side-by-side training curves (Figure 4-5 can be referenced here)
  - **Observation**: Unweighted CE converges quickly but to poor Macro-F1; Weighted CE converges slower but to much higher Macro-F1
  - **Interpretation**: Weighted CE forces the network to learn from minority classes, even though it's harder
- **AND** this explains why training takes longer but achieves better performance

### Requirement: Garbage Class Mechanism Analysis
The theoretical analysis writing SHALL analyze how the garbage class enables open-set recognition.

**Description**:
The garbage class adds an (N+1)th output for unknown samples. This analysis shows how it changes the feature space representation to separate known from unknown.

#### Scenario: Feature space separation analysis writing
- **WHEN** writing Section 5.4 (垃圾类作用机制)
- **THEN** the section SHALL include:
  - **Paragraph 1**: Motivation - How does adding an (N+1)th output help?
  - **Paragraph 2**: Method - Extract penultimate layer features for GEE with and without garbage class, project to 2D
  - **Paragraph 3**: Observation - Describe Figure 5-3 (two subplots: without vs with garbage class)
  - **Paragraph 4**: Interpretation - Without garbage class: Unknown samples intermingle with known; With garbage class: Unknown samples form distinct cluster
  - **Paragraph 5**: Validation - Link to OSR results (FPR@TPR95: 0.8+ → 0.03)
- **AND** reference Figure 5-3 in Paragraph 3
- **AND** quantify separation if possible (e.g., "Overlap coefficient: 0.82 → 0.15")

#### Scenario: Garbage neuron activation analysis writing
- **WHEN** analyzing the garbage class neuron's behavior
- **THEN** the section SHALL include:
  - **Method**: Record garbage class neuron activation (P_garbage) for all test samples
  - **Visualization**: Histogram (Figure 5-3b) showing P_garbage for known vs unknown samples
  - **Observation**: Known samples: P_garbage concentrated near 0; Unknown samples: P_garbage concentrated near 1
  - **Interpretation**: Garbage neuron learned to activate specifically for unknown samples
  - **Validation**: This explains why FPR@TPR95 is so low (unknown samples are confidently identified)
- **AND** quantify separability (e.g., "Mean P_garbage: known=0.08, unknown=0.89")

### Requirement: Theory-Experiment Alignment Validation
The theoretical analysis writing SHALL validate all theoretical claims against experimental results.

**Description**:
Every theoretical explanation must be supported by experimental evidence. If theory and experiment conflict, we must acknowledge the discrepancy.

#### Scenario: Alignment check template
- **WHEN** writing each analysis section in Chapter 5
- **THEN** the final paragraph SHALL follow this template:
  - **Theoretical Claim**: [What we claim the mechanism is]
  - **Experimental Support**: [Which experiment from Chapter 4 supports this]
  - **Consistency Check**: [Do theory and experiment align? Yes/No]
  - **If Yes**: This confirms our understanding of the mechanism
  - **If No**: Possible explanations are [X, Y, Z]; further investigation needed
- **AND** this alignment check SHALL be explicit in each section
- **AND** discrepancies SHALL be honestly acknowledged rather than hidden

#### Scenario: Example alignment check for gating network analysis
- **WHEN** validating the gating network decision pattern analysis
- **THEN** the alignment check SHALL state:
  - **Theoretical Claim**: Gating network learned to trust expert for minority classes
  - **Experimental Support**: Macro-F1 improved from 0.63 (baseline) to 0.83 (GEE), with minority class F1 improving from 0 to 0.92
  - **Consistency Check**: Yes, aligned. Weight analysis shows expert weights are 2-3x larger for minority classes
  - **Conclusion**: This confirms that the gating network's learned decision pattern directly causes the IL performance improvement

### Requirement: Theoretical Analysis Limitations Discussion
The theoretical analysis writing SHALL acknowledge limitations and boundary conditions.

**Description**:
Not all aspects of GEE may be fully explainable with current analyses. We should acknowledge limitations and identify open questions for future research.

#### Scenario: Limitations paragraph at chapter end
- **WHEN** concluding Chapter 5 (理论分析与讨论)
- **THEN** a limitations paragraph SHALL include:
  - **Acknowledged Limitations**:
    - Decision boundary visualization uses 2D projection, may lose high-dimensional information
    - Gradient analysis assumes convex loss landscape, actual training may be more complex
    - Analyses are post-hoc (based on trained models), may not capture all training dynamics
  - **Open Questions**:
    - How does GEE scale to more than 2 experts?
    - What is the theoretical upper bound on GEE's improvement?
    - Can we derive theoretical guarantees on Macro-F1 improvement given class imbalance?
  - **Future Work**: These questions could be explored in future research
- **AND** this honesty enhances academic credibility

### Requirement: Theoretical Contribution Summary
The theoretical analysis writing SHALL culminate in a clear summary of key insights.

**Description**:
Chapter 5 should conclude with a concise summary of what we learned about WHY GEE works, not just THAT it works.

#### Scenario: Chapter 5 conclusion summary
- **WHEN** writing the final section of Chapter 5
- **THEN** the summary SHALL articulate:
  - **Key Insight 1**: GEE's effectiveness comes from complementary specialization (baseline for majority, expert for minority) combined via learnable gating
  - **Key Insight 2**: Weighted loss is essential - without it, gating network defaults to always trusting baseline (Macro-F1 drops to 0.63)
  - **Key Insight 3**: Garbage class enables OSR by learning to map unknown samples to a distinct feature region (FPR@TPR95 drops from 0.8+ to 0.03)
  - **Overall Contribution**: We have explained the mechanisms behind GEE's empirical success, providing both theoretical understanding and practical guidance
- **AND** this summary SHALL transition naturally to Chapter 6 (总结与展望)
