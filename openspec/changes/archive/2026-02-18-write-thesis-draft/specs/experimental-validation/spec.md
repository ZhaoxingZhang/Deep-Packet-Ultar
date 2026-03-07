# Experimental Validation Specification

## ADDED Requirements

### Requirement: Existing Experimental Data Audit
The experimental validation SHALL first audit existing experimental data to identify gaps and completeness.

**Description**:
Before running new experiments, we need to understand what data we already have from the research logs and what gaps need to be filled for a complete thesis.

#### Scenario: Inventorying existing OSR data
- **WHEN** auditing existing OSR experimental data
- **THEN** system SHALL check the following sources:
  - `.local/logs/` directory for evaluation logs
  - Research log entries dated 11/22 (baseline OSR) and 12/6 (GEE OSR)
  - Script outputs from `scripts/run_*_open_set_evaluation.sh`
- **AND** system SHALL identify what data exists:
  - ResNet-Baseline: AUROC and FPR@TPR95 for 6 folds (classes 5-10)
  - ResNet-GEE: AUROC and FPR@TPR95 for 6 folds
  - CNN-Baseline: AUROC and FPR@TPR95 for 6 folds
  - CNN-GEE: AUROC and FPR@TPR95 for 6 folds
- **AND** system SHALL identify gaps:
  - Are all 6 folds present for each model?
  - Are metrics consistent (AUROC, FPR@TPR95)?
  - Are detailed per-class results available?

#### Scenario: Inventorying existing IL data
- **WHEN** auditing existing IL experimental data
- **THEN** system SHALL check research log entries dated 11/16 and 12/19
- **AND** system SHALL identify what data exists:
  - ResNet-Baseline: Macro-F1, per-class F1
  - ResNet-GEE: Macro-F1, per-class F1 (with weighted CE)
  - Simple weighted: Macro-F1, per-class F1
  - CNN-Baseline and CNN-GEE: Macro-F1, per-class F1
- **AND** system SHALL identify gaps:
  - Are ablation results complete (baseline → simple → gating standard CE → gating weighted CE)?
  - Are per-class F1 breakdowns available for minority classes?
  - Are training time and parameter counts recorded?

#### Scenario: Inventorying existing visualization assets
- **WHEN** auditing existing figures and tables
- **THEN** system SHALL check what has been generated:
  - ROC curve plots (6 folds × 4 models = 24 plots)
  - Confusion matrices (baseline vs GEE)
  - Training curves (loss over epochs)
  - Decision boundary visualizations (if any)
- **AND** system SHALL identify gaps:
  - Which plots are missing or need regeneration?
  - Which plots need better formatting or labeling?
  - Which additional plots would strengthen the thesis?

### Requirement: Supplementary Experiment Planning
The experimental validation SHALL plan supplementary experiments to fill identified gaps.

**Description**:
Based on the audit, some experiments may be missing or incomplete. This requirement defines which supplementary experiments are needed and how to run them.

#### Scenario: Missing ablation experiments
- **WHEN** ablation study data is incomplete
- **THEN** system SHALL identify which comparisons are missing:
  - Baseline only vs Baseline + Simple weighted?
  - Simple weighted vs Gating network (standard CE)?
  - Gating network (standard CE) vs Gating network (weighted CE)?
  - GEE (no garbage class) vs GEE (with garbage class)?
- **AND** system SHALL plan to run missing experiments
- **AND** system SHALL create or modify automation scripts (e.g., `scripts/run_ablation_study.sh`)
- **AND** priority SHALL be given to experiments that support key claims

#### Scenario: Missing multi-model validation
- **WHEN** CNN-GEE results are incomplete
- **THEN** system SHALL plan to run:
  - CNN-Baseline OSR evaluation (6 folds)
  - CNN-Baseline IL evaluation
  - CNN-GEE OSR evaluation (6 folds, with garbage class)
  - CNN-GEE IL evaluation (with weighted CE)
- **AND** system SHALL use existing scripts: `scripts/run_open_set_cnn_*.sh` and `scripts/run_incremental_cnn_*.sh`
- **AND** results SHALL be compared to ResNet-GEE to demonstrate architectural portability

#### Scenario: Missing theoretical analysis data
- **WHEN** data for theoretical analysis (Chapter 5) is missing
- **THEN** system SHALL plan to extract or generate:
  - Gating network weight distributions (from saved model checkpoints)
  - Expert decision boundary visualizations (2D projections of penultimate layer features)
  - Gradient direction comparisons (weighted vs unweighted loss)
  - Garbage class neuron activations (for known vs unknown samples)
- **AND** system SHALL create analysis scripts (e.g., `analysis/extract_weights.py`, `analysis/visualize_boundaries.py`)
- **AND** priority SHALL be given to visualizations that clearly show mechanisms

### Requirement: Data Visualization Strategy
The experimental validation SHALL define a comprehensive strategy for visualizing experimental results.

**Description**:
Tables and figures are critical for conveying results efficiently. This requirement defines what visualizations are needed and how they should be formatted.

#### Scenario: Result table design for OSR
- **WHEN** designing tables for OSR results
- **THEN** the following tables SHALL be created:
  - **Table 4-1**: OSR Performance Comparison
    - Columns: Fold (Unknown Class), AUROC (Baseline), AUROC (GEE), Improvement, FPR@TPR95 (Baseline), FPR@TPR95 (GEE), Improvement
    - Rows: 6 folds (classes 5, 6, 7, 8, 9, 10 as unknown) + Average
    - Highlight: Worst folds for baseline (classes 8, 9, 10) show largest GEE improvements
  - **Table 4-2**: CNN-GEE OSR Performance (same format as Table 4-1)
- **AND** tables SHALL use LaTeX booktabs for professional formatting
- **AND** significant improvements SHALL be bolded or marked with asterisks

#### Scenario: Result table design for IL
- **WHEN** designing tables for IL results
- **THEN** the following tables SHALL be created:
  - **Table 4-3**: IL Performance Comparison
    - Columns: Class, Sample Count, Baseline F1, GEE F1, Improvement
    - Rows: All classes (5-10) + Macro-average
    - Highlight: Minority classes (5, 7) show largest improvements (0 → 0.28, 0 → 0.92)
  - **Table 4-4**: Ablation Study Results
    - Columns: Configuration, Macro-F1, Class 5 F1, Class 7 F1, Training Time
    - Rows: Baseline, Simple Weighted, Gating (Standard CE), Gating (Weighted CE)
    - Highlight: Progressive improvement as components are added
- **AND** tables SHALL clearly show the contribution of each component

#### Scenario: Figure design for OSR
- **WHEN** designing figures for OSR results
- **THEN** the following figures SHALL be created:
  - **Figure 4-1**: ROC Curves (ResNet-Baseline vs ResNet-GEE)
    - Subplots for 6 folds (classes 5-10 as unknown)
    - Each subplot: X-axis = FPR, Y-axis = TPR, two curves (baseline and GEE)
    - Shade AUROC regions
  - **Figure 4-2**: FPR@TPR95 Bar Chart
    - X-axis: Fold (Unknown Class), Y-axis: FPR@TPR95
    - Grouped bars: Baseline vs GEE
    - Highlight: Large improvements for folds 8, 9, 10
  - **Figure 4-3**: Confidence Score Histograms (Baseline vs GEE)
    - For a representative fold (e.g., class 8 as unknown)
    - Two subplots: Baseline confidence distribution, GEE confidence distribution
    - Each subplot: Histogram for known samples (blue), unknown samples (red)
    - Show that GEE separates known/unknown better

#### Scenario: Figure design for IL
- **WHEN** designing figures for IL results
- **THEN** the following figures SHALL be created:
  - **Figure 4-4**: Per-Class F1 Comparison
    - X-axis: Class (5-10), Y-axis: F1 Score
    - Grouped bars: Baseline vs GEE
    - Highlight: Large improvements for classes 5 and 7
  - **Figure 4-5**: Training Curves
    - Subplots: Training Loss, Validation Loss, Macro-F1
    - X-axis: Epoch, Y-axis: Loss/F1
    - Show convergence for gating network with weighted CE
  - **Figure 4-6**: Confusion Matrix Heatmaps
    - Three subplots: Baseline, Simple Weighted, GEE (Weighted CE)
    - Color intensity: Number of samples (log scale if imbalanced)
    - Show that GEE correctly classifies minority classes

#### Scenario: Figure design for theoretical analysis
- **WHEN** designing figures for Chapter 5 (理论分析)
- **THEN** the following figures SHALL be created:
  - **Figure 5-1**: Gating Network Weight Distribution
    - Heatmap of first layer weights
    - Annotate: Weights for expert inputs vs baseline inputs
    - Show that expert weights are larger for minority classes
  - **Figure 5-2**: Decision Boundaries (Baseline vs Expert vs GEE)
    - 2D projections (PCA/t-SNE) of penultimate layer features
    - Three subplots: Baseline, Expert, GEE
    - Color points by true class
    - Show complementarity: Baseline separates majority, Expert separates minority
  - **Figure 5-3**: Feature Space Separation with Garbage Class
    - Two subplots: GEE without garbage class, GEE with garbage class
    - 2D projections with unknown class highlighted
    - Show that garbage class creates distinct unknown cluster
  - **Figure 5-4**: Gradient Contribution per Class
    - Bar chart: Average gradient L2 norm per class
    - Grouped bars: Unweighted CE vs Weighted CE
    - Show that weighted CE balances gradient contributions

### Requirement: Baseline Definition
The experimental validation SHALL explicitly define baseline methods for fair comparison.

**Description**:
To claim that GEE improves performance, we must compare it against well-defined baselines. This requirement specifies what those baselines are and how to implement them.

#### Scenario: OSR baseline definition
- **WHEN** defining the OSR baseline for comparison
- **THEN** the baseline SHALL be:
  - **Model**: Deep-Packet ResNet1d (trained on known classes only)
  - **Decision mechanism**: Softmax confidence threshold
    - Confidence = max(P_softmax)
    - If confidence > 0.5: classify as "known", predicted class = argmax(P_softmax)
    - If confidence ≤ 0.5: classify as "unknown"
  - **Training strategy**: Train on K-1 classes (excluding the unknown class for that fold)
  - **Evaluation**: Calculate AUROC and FPR@TPR95 on test set with all K classes
- **AND** this baseline represents the simplest OSR method
- **AND** GEE SHALL be compared against this baseline on the same datasets

#### Scenario: IL baseline definition
- **WHEN** defining the IL baseline for comparison
- **THEN** the baseline SHALL be:
  - **Model**: Deep-Packet ResNet1d (retrained on all classes)
  - **Training strategy**: Full fine-tuning - add new class data to training set and retrain entire model from scratch
  - **Evaluation**: Calculate Macro-F1 and per-class F1 on test set with all classes
- **AND** this baseline represents the traditional IL method
- **AND** GEE SHALL be compared against this baseline in terms of:
  - Macro-F1 (similar or better)
  - Training time (much faster)
  - Catastrophic forgetting (GEE avoids this)

#### Scenario: Ablation baseline definitions
- **WHEN** defining baselines for ablation studies
- **THEN** the following configurations SHALL be compared:
  - **Config 1 (Baseline)**: Single ResNet model on all classes
  - **Config 2 (Simple Weighted)**: Baseline (0.85) + Expert (0.15), fixed weighted average
  - **Config 3 (Gating Standard CE)**: Baseline + Expert + Gating Network with standard CE
  - **Config 4 (Gating Weighted CE)**: Baseline + Expert + Gating Network with weighted CE
  - **Config 5 (GEE Full)**: Config 4 + Garbage Class (for OSR only)
- **AND** progressive improvement from Config 1 to 5 SHALL demonstrate each component's contribution

### Requirement: Experiment Reproducibility
The experimental validation SHALL ensure all experiments are reproducible with clear documentation.

**Description**:
For academic rigor and to enable verification, all experiments must be reproducible. This requirement specifies what documentation and scripts are needed.

#### Scenario: Experiment script documentation
- **WHEN** documenting experiment automation scripts
- **THEN** each script in `scripts/` SHALL have:
  - Clear filename describing its purpose (e.g., `run_incremental_resnet_gee.sh`)
  - Usage comments at the top: Purpose, Prerequisites, Parameters, Example commands
  - Step-by-step comments explaining each phase (data generation, training, evaluation)
  - Config file references (e.g., `config/incremental_resnet.yaml`)
- **AND** scripts SHALL be tested end-to-end before documentation
- **AND** README files SHALL explain how to run experiments

#### Scenario: Experiment configuration documentation
- **WHEN** documenting experiment configurations
- **THEN** each config file in `config/` SHALL specify:
  - Dataset paths (e.g., `train_test_data/traffic_imbalanced/`)
  - Model hyperparameters (learning rate, batch size, epochs)
  - Evaluation settings (metrics to calculate, output paths)
  - Random seeds for reproducibility
- **AND** config files SHALL be version-controlled
- **AND** thesis SHALL reference config files in experimental setup sections

#### Scenario: Result logging and archiving
- **WHEN** logging and archiving experimental results
- **THEN** each experiment run SHALL produce:
  - Log file with complete output (e.g., `.local/logs/exp_20250115.txt`)
  - Model checkpoints (`.pth` files) for reproducibility
  - Evaluation metrics in machine-readable format (JSON or CSV)
  - Timestamp and config file used
- **AND** results SHALL be archived with descriptive names
- **AND** thesis SHALL reference specific log files for each presented result

### Requirement: Experimental Validation Timeline
The experimental validation SHALL define a timeline for completing supplementary experiments and visualizations.

**Description**:
To fit within the 24-hour timebox for thesis writing, experiments must be completed efficiently. This requirement defines the order and priority of experimental work.

#### Scenario: Phase 1 - Critical path experiments (Week 1)
- **WHEN** prioritizing experiments for thesis writing
- **THEN** Phase 1 SHALL focus on data needed for core chapters:
  - Complete ablation study (if incomplete)
  - Generate all required tables (Tables 4-1 to 4-4)
  - Generate key figures (Figures 4-1 to 4-6)
- **AND** these experiments SHALL be completed before Chapter 4 writing begins
- **AND** priority: P0 (must have) for thesis completion

#### Scenario: Phase 2 - Theoretical analysis data (Week 2)
- **WHEN** generating data for Chapter 5 (理论分析)
- **THEN** Phase 2 SHALL focus on:
  - Extract gating network weights from checkpoints
  - Generate decision boundary visualizations
  - Calculate gradient contributions
  - Analyze garbage class activations
- **AND** these analyses SHALL be completed before Chapter 5 writing begins
- **AND** priority: P0 (must have) for thesis depth

#### Scenario: Phase 3 - Supplementary validation (Week 2, parallel with writing)
- **WHEN** running supplementary experiments
- **THEN** Phase 3 SHALL focus on:
  - CNN-GEE validation (if incomplete)
  - Additional visualizations that enhance presentation
  - Robustness checks (different random seeds, data splits)
- **AND** these experiments MAY run in parallel with thesis writing
- **AND** priority: P1 (nice to have) for strengthening the thesis
