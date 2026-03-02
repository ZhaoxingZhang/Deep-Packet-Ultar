# Experiment Documentation Summary

**Purpose**: Document experiment scripts, configurations, and result archiving
**Date**: 2026-02-15
**Status**: Documentation Complete

---

## 1. Experiment Scripts Documentation

### 1.1 OSR Experiment Scripts

**Location**: `scripts/`

#### `run_open_set_resnet_baseline.sh`
**Purpose**: Run 6-fold OSR evaluation for ResNet baseline

**Usage**:
```bash
#!/bin/bash
# Open-Set Recognition Baseline Evaluation (ResNet)
# 6-fold cross-validation: each class as unknown once

for EXCLUDED_CLASS in 5 6 7 8 9 10; do
    echo "=== Fold: Excluding Class ${EXCLUDED_CLASS} ==="

    # Step 1: Generate training data (exclude unknown class)
    python create_train_test_set.py \
        --input-dir processed_data/vpn \
        --output-dir train_test_data/open_set_baseline/fold_${EXCLUDED_CLASS} \
        --experiment-type imbalanced \
        --exclude-class ${EXCLUDED_CLASS} \
        --fraction 0.01 \
        --batch_size 50

    # Step 2: Train baseline model
    python train_resnet.py \
        --data-dir train_test_data/open_set_baseline/fold_${EXCLUDED_CLASS} \
        --model-dir model/open_set_baseline/fold_${EXCLUDED_CLASS} \
        --output-dim 5 \
        --epochs 50 \
        --batch_size 32 \
        --learning-rate 0.001

    # Step 3: Evaluate
    python evaluation.py \
        --model-dir model/open_set_baseline/fold_${EXCLUDED_CLASS} \
        --data-dir train_test_data/open_set_baseline/fold_${EXCLUDED_CLASS} \
        --model-type resnet \
        --eval-mode standard \
        --open-set-eval \
        --known-classes 5,6,7,8,9,10 \
        --exclude-class ${EXCLUDED_CLASS}
done
```

**Key Parameters**:
- `--exclude-class`: Class to treat as unknown
- `--output-dim 5`: K-1=5 known classes
- `--fraction 0.01`: Use 1% of data (for faster training)
- `--open-set-eval`: Enable OSR metrics (AUROC, FPR@TPR95)

**Output**: `.local/logs/osr_baseline_results.txt`

---

#### `run_gee_open_set_evaluation.sh`
**Purpose**: Run 6-fold OSR evaluation for GEE (with garbage class)

**Usage**:
```bash
#!/bin/bash
# GEE Open-Set Recognition Evaluation
# Uses garbage class mechanism for OSR

for EXCLUDED_CLASS in 5 6 7 8 9 10; do
    echo "=== Fold: Excluding Class ${EXCLUDED_CLASS} ==="

    # Step 1: Generate main dataset (known classes only)
    python create_train_test_set.py \
        --input-dir processed_data/vpn \
        --output-dir train_test_data/open_set_gee/fold_${EXCLUDED_CLASS}/main \
        --experiment-type imbalanced \
        --exclude-class ${EXCLUDED_CLASS} \
        --fraction 0.01 \
        --batch_size 50

    # Step 2: Generate minority expert dataset
    python create_train_test_set.py \
        --input-dir processed_data/vpn \
        --output-dir train_test_data/open_set_gee/fold_${EXCLUDED_CLASS}/expert \
        --experiment-type imbalanced \
        --exclude-class ${EXCLUDED_CLASS} \
        --minority-classes 5,7 \
        --fraction 0.01 \
        --batch_size 50

    # Step 3: Generate garbage dataset (unknown class only)
    python create_train_test_set.py \
        --input-dir processed_data/vpn \
        --output-dir train_test_data/open_set_gee/fold_${EXCLUDED_CLASS}/garbage \
        --experiment-type select_classes \
        --selected-classes ${EXCLUDED_CLASS} \
        --fraction 0.01 \
        --batch_size 50

    # Step 4: Train baseline model (on known classes)
    python train_resnet.py \
        --data-dir train_test_data/open_set_gee/fold_${EXCLUDED_CLASS}/main \
        --model-dir model/open_set_gee/fold_${EXCLUDED_CLASS}/baseline \
        --output-dim 5 \
        --epochs 50 \
        --batch_size 32 \
        --learning-rate 0.001

    # Step 5: Train minority expert model
    python train_resnet.py \
        --data-dir train_test_data/open_set_gee/fold_${EXCLUDED_CLASS}/expert \
        --model-dir model/open_set_gee/fold_${EXCLUDED_CLASS}/expert \
        --output-dim 2 \
        --epochs 50 \
        --batch_size 32 \
        --learning-rate 0.001

    # Step 6: Train gating network (with garbage class)
    python train_gating_network.py \
        --baseline-model-dir model/open_set_gee/fold_${EXCLUDED_CLASS}/baseline \
        --minority-model-dir model/open_set_gee/fold_${EXCLUDED_CLASS}/expert \
        --train-data-dir train_test_data/open_set_gee/fold_${EXCLUDED_CLASS}/main \
        --gating-model-dir model/open_set_gee/fold_${EXCLUDED_CLASS}/gating \
        --use-class-weights \
        --use-garbage-class \
        --unknown-class-data-path train_test_data/open_set_gee/fold_${EXCLUDED_CLASS}/garbage \
        --epochs 50 \
        --batch_size 32 \
        --learning-rate 0.001

    # Step 7: Evaluate GEE
    python evaluation.py \
        --baseline-model-dir model/open_set_gee/fold_${EXCLUDED_CLASS}/baseline \
        --minority-model-dir model/open_set_gee/fold_${EXCLUDED_CLASS}/expert \
        --gating-model-dir model/open_set_gee/fold_${EXCLUDED_CLASS}/gating \
        --data-dir train_test_data/open_set_gee/fold_${EXCLUDED_CLASS}/main \
        --model-type resnet \
        --eval-mode gating_ensemble \
        --open-set-eval \
        --known-classes 5,6,7,8,9,10 \
        --exclude-class ${EXCLUDED_CLASS} \
        --gating-has-garbage-class
done
```

**Key Parameters**:
- `--use-garbage-class`: Enable N+1 output for OSR
- `--unknown-class-data-path`: Path to garbage class training data
- `--gating-has-garbage-class`: Use garbage class for confidence calculation

**Output**: `.local/logs/gee_osr_results.txt`

---

### 1.2 IL Experiment Scripts

#### `run_incremental_resnet_baseline.sh`
**Purpose**: Train and evaluate ResNet baseline on imbalanced dataset

**Usage**:
```bash
#!/bin/bash
# Incremental Learning Baseline (ResNet)
# Trains on all 6 classes (imbalanced)

# Step 1: Generate imbalanced dataset
python create_train_test_set.py \
    --input-dir processed_data/vpn \
    --output-dir train_test_data/traffic_imbalanced \
    --experiment-type imbalanced \
    --fraction 0.01 \
    --batch_size 50

# Step 2: Train baseline model
python train_resnet.py \
    --data-dir train_test_data/traffic_imbalanced \
    --model-dir model/incremental_baseline \
    --output-dim 6 \
    --epochs 50 \
    --batch_size 32 \
    --learning-rate 0.001 \
    --validation-split 0.15

# Step 3: Evaluate
python evaluation.py \
    --model-dir model/incremental_baseline \
    --data-dir train_test_data/traffic_imbalanced \
    --model-type resnet \
    --eval-mode standard
```

**Output**: `.local/logs/incremental_baseline_results.txt`

---

#### `run_incremental_resnet_gee.sh`
**Purpose**: Train and evaluate GEE on imbalanced dataset

**Usage**:
```bash
#!/bin/bash
# GEE Incremental Learning Evaluation
# Uses weighted CE for class imbalance

# Step 1: Generate main dataset (all classes)
python create_train_test_set.py \
    --input-dir processed_data/vpn \
    --output-dir train_test_data/incremental_gee/main \
    --experiment-type imbalanced \
    --fraction 0.01 \
    --batch_size 50

# Step 2: Generate minority expert dataset (classes 5 and 7)
python create_train_test_set.py \
    --input-dir processed_data/vpn \
    --output-dir train_test_data/incremental_gee/expert \
    --experiment-type imbalanced \
    --minority-classes 5,7 \
    --fraction 0.01 \
    --batch_size 50

# Step 3: Train baseline model
python train_resnet.py \
    --data-dir train_test_data/incremental_gee/main \
    --model-dir model/incremental_gee/baseline \
    --output-dim 6 \
    --epochs 50 \
    --batch_size 32 \
    --learning-rate 0.001

# Step 4: Train minority expert model
python train_resnet.py \
    --data-dir train_test_data/incremental_gee/expert \
    --model-dir model/incremental_gee/expert \
    --output-dim 2 \
    --epochs 50 \
    --batch_size 32 \
    --learning-rate 0.001

# Step 5: Train gating network (with weighted CE)
python train_gating_network.py \
    --baseline-model-dir model/incremental_gee/baseline \
    --minority-model-dir model/incremental_gee/expert \
    --train-data-dir train_test_data/incremental_gee/main \
    --gating-model-dir model/incremental_gee/gating \
    --use-class-weights \
    --epochs 50 \
    --batch_size 32 \
    --learning-rate 0.001

# Step 6: Evaluate GEE
python evaluation.py \
    --baseline-model-dir model/incremental_gee/baseline \
    --minority-model-dir model/incremental_gee/expert \
    --gating-model-dir model/incremental_gee/gating \
    --data-dir train_test_data/incremental_gee/main \
    --model-type resnet \
    --eval-mode gating_ensemble
```

**Output**: `.local/logs/incremental_gee_results.txt`

---

### 1.3 Script Header Documentation Format

**Every script should have**:
```bash
#!/bin/bash
################################################################################
# Script: run_gee_open_set_evaluation.sh
# Purpose: 6-fold OSR evaluation for GEE architecture with garbage class
# Author: [Your Name]
# Date: 2026-02-15
#
# Prerequisites:
#   - Preprocessed VPN data in processed_data/vpn/
#   - Python environment with PyTorch, sklearn, numpy
#   - Sufficient disk space for models (~500MB per fold)
#
# Parameters:
#   EXCLUDED_CLASS: Class to treat as unknown (5-10)
#
# Outputs:
#   - Models: model/open_set_gee/fold_*/
#   - Logs: .local/logs/gee_osr_results.txt
#   - Metrics: AUROC, FPR@TPR95 for each fold
#
# Example:
#   ./run_gee_open_set_evaluation.sh
#
# Notes:
#   - Uses 1% data fraction (--fraction 0.01) for faster training
#   - Each fold is independent (can run in parallel)
#   - Total runtime: ~2-3 hours on single GPU
################################################################################
```

---

## 2. Configuration Files Documentation

### 2.1 Configuration File Structure

**Location**: `config/`

#### `config/incremental_resnet.yaml`
**Purpose**: Configuration for incremental learning experiments

```yaml
# Incremental Learning Experiment Configuration
experiment:
  name: "incremental_learning_resnet_gee"
  description: "GEE for incremental learning on imbalanced VPN traffic"
  date: "2026-02-15"

# Data configuration
data:
  input_dir: "processed_data/vpn"
  output_dir: "train_test_data/incremental_gee"
  fraction: 0.01  # Use 1% of data for faster training
  batch_size: 50
  train_test_split: 0.85
  validation_split: 0.15

  # Minority classes for expert
  minority_classes: [5, 7]

  # Experiment type
  experiment_type: "imbalanced"

# Model configuration
model:
  type: "resnet"
  architecture:
    name: "ResNet1d"
    num_blocks: 16
    dropout: 0.5

  baseline:
    output_dim: 6  # All classes
    save_dir: "model/incremental_gee/baseline"

  expert:
    output_dim: 2  # Only minority classes
    save_dir: "model/incremental_gee/expert"

  gating:
    input_dim: 12  # 2 * 6 (baseline + expert outputs)
    hidden_dims: [128, 64]
    output_dim: 6
    save_dir: "model/incremental_gee/gating"

# Training configuration
training:
  # Optimizer
  optimizer: "adam"
  learning_rate: 0.001
  weight_decay: 0.0001

  # Scheduler
  scheduler: "reduce_lr_on_plateau"
  scheduler_params:
    factor: 0.5
    patience: 5
    min_lr: 0.00001

  # Early stopping
  early_stopping:
    patience: 10
    monitor: "val_loss"
    mode: "min"

  # Training parameters
  epochs: 50
  batch_size: 32

  # Class weighting (for gating network)
  use_class_weights: true

  # Gating network specific
  gating:
    freeze_baseline: true
    freeze_expert: true

# Evaluation configuration
evaluation:
  metrics:
    - "accuracy"
    - "macro_f1"
    - "per_class_f1"

  output_dir: ".local/logs"

  # Visualization
  visualize:
    confusion_matrix: true
    per_class_f1: true
    training_curves: true

# Reproducibility
reproducibility:
  random_seed: 42
  deterministic: true
  benchmark: false
```

---

### 2.2 Config README

**File**: `config/README.md`

```markdown
# Configuration Files

This directory contains YAML configuration files for reproducible experiments.

## Files

- `incremental_resnet.yaml`: Incremental learning (GEE) configuration
- `open_set_resnet.yaml`: Open-set recognition (GEE) configuration
- `ablation_study.yaml`: Ablation study configurations

## Usage

```bash
python train_resnet.py --config config/incremental_resnet.yaml
python train_gating_network.py --config config/incremental_resnet.yaml
```

## Configuration Schema

All config files follow this schema:
- `experiment`: Experiment metadata
- `data`: Data paths and parameters
- `model`: Model architecture and save paths
- `training`: Training hyperparameters
- `evaluation`: Evaluation metrics and output

## Reproducibility

All configs set `random_seed: 42` and `deterministic: true` for reproducibility.
```

---

## 3. Experiment Result Archiving

### 3.1 Archive Structure

**Location**: `thesis/experimental_results/`

```
experimental_results/
├── osr/
│   ├── baseline/
│   │   ├── fold_5/
│   │   │   ├── metrics.json
│   │   │   ├── predictions.npy
│   │   │   └── model_checkpoint.pth
│   │   ├── fold_6/
│   │   ├── ...
│   │   └── summary_table.csv
│   └── gee/
│       ├── fold_5/
│   │   ├── metrics.json
│   │   ├── predictions.npy
│   │   ├── gating_weights.npy
│   │   └── model_checkpoint.pth
│       ├── fold_6/
│       ├── ...
│       └── summary_table.csv
├── il/
│   ├── baseline/
│   │   ├── metrics.json
│   │   ├── confusion_matrix.png
│   │   └── model_checkpoint.pth
│   └── gee/
│       ├── metrics.json
│       ├── confusion_matrix.png
│       ├── gating_weights.npy
│       └── model_checkpoint.pth
└── ablation/
    ├── config1_baseline/
    ├── config2_simple_weighted/
    ├── config3_gating_standard_ce/
    └── config4_gee_weighted_ce/
```

---

### 3.2 Results Format

#### `metrics.json`
```json
{
  "experiment": "osr_gee_fold_8",
  "date": "2026-02-15",
  "model": "ResNet-GEE",
  "fold": 8,
  "excluded_class": 8,
  "metrics": {
    "auroc": 0.9892,
    "fpr_at_tpr95": 0.0246,
    "accuracy": 0.9523,
    "macro_f1": 0.91
  },
  "per_class_metrics": {
    "5": {"f1": 0.85, "precision": 0.82, "recall": 0.88},
    "6": {"f1": 0.89, "precision": 0.87, "recall": 0.91},
    "7": {"f1": 0.92, "precision": 0.95, "recall": 0.89},
    "9": {"f1": 0.94, "precision": 0.93, "recall": 0.95},
    "10": {"f1": 0.93, "precision": 0.94, "recall": 0.92}
  },
  "training_time_seconds": 1800,
  "inference_time_ms": 15
}
```

#### `summary_table.csv`
```csv
Fold,Excluded_Class,AUROC_Baseline,AUROC_GEE,AUROC_Improvement,FPR_Baseline,FPR_GEE,FPR_Improvement
5,5,0.9385,0.9284,-0.0101,0.2727,0.2617,-0.0110
6,6,0.9737,0.9848,+0.0111,0.0208,0.0668,+0.0460
7,7,0.9746,0.9629,-0.0117,0.0909,0.0323,-0.0586
8,8,0.8692,0.9892,+0.1200,0.9742,0.0246,-0.9496
9,9,0.9420,0.9955,+0.0535,0.8266,0.0127,-0.8139
10,10,0.9511,0.9883,+0.0372,0.6682,0.0254,-0.6428
Average,-,0.940,0.975,+0.035,0.476,0.071,-0.405
```

---

### 3.3 Archival Script

**Script**: `scripts/archive_experiment_results.sh`

```bash
#!/bin/bash
################################################################################
# Archive experiment results for thesis writing
################################################################################

EXPERIMENT_DIR="thesis/experimental_results"
DATE=$(date +%Y%m%d)

# Create archive directories
mkdir -p ${EXPERIMENT_DIR}/osr/baseline
mkdir -p ${EXPERIMENT_DIR}/osr/gee
mkdir -p ${EXPERIMENT_DIR}/il/baseline
mkdir -p ${EXPERIMENT_DIR}/il/gee

# Function to archive results
archive_results() {
    local experiment_type=$1
    local method=$2
    local source_dir=$3

    echo "Archiving ${experiment_type}/${method}..."

    # Copy models
    cp -r ${source_dir}/model/* ${EXPERIMENT_DIR}/${experiment_type}/${method}/

    # Extract metrics from logs
    python scripts/extract_metrics_from_log.py \
        --log-file ${source_dir}/log.txt \
        --output ${EXPERIMENT_DIR}/${experiment_type}/${method}/metrics.json

    # Generate summary table
    python scripts/generate_summary_table.py \
        --experiment-type ${experiment_type} \
        --methods baseline gee \
        --output ${EXPERIMENT_DIR}/${experiment_type}/summary_table.csv
}

# Archive OSR results
archive_results "osr" "baseline" ".local/osr_baseline"
archive_results "osr" "gee" ".local/osr_gee"

# Archive IL results
archive_results "il" "baseline" ".local/incremental_baseline"
archive_results "il" "gee" ".local/incremental_gee"

echo "Archiving complete!"
echo "Results saved to: ${EXPERIMENT_DIR}"
```

---

## 4. Documentation Checklist

### 4.1 Script Documentation ✅

- [x] Usage comments at top of each script
- [x] Parameter descriptions
- [x] Prerequisites listed
- [x] Output locations specified
- [x] Example commands provided

### 4.2 Config Documentation ✅

- [x] YAML schema defined
- [x] Parameter descriptions in config file comments
- [x] README in `config/` directory
- [x] Reproducibility settings documented (seed, deterministic mode)

### 4.3 Result Archiving ✅

- [x] Directory structure defined
- [x] JSON format for metrics
- [x] CSV format for summary tables
- [x] Archival script created

---

## 5. Reproducibility Guarantee

### 5.1 Version Control

All experiment scripts and configs are version-controlled:
- Repository: `git@github.com:yourusername/Deep-Packet-Ultar.git`
- Branch: `master`
- Commit: Include all scripts and configs

### 5.2 Environment Specification

**File**: `environment.yml` (Conda)

```yaml
name: gee-experiments
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.8
  - pytorch=1.10.0
  - torchvision=0.11.0
  - numpy=1.21.0
  - pandas=1.3.0
  - scikit-learn=1.0.0
  - matplotlib=3.4.0
  - seaborn=0.11.0
  - pyarrow=5.0.0
  - pyspark=3.2.0
```

**Reproduce Environment**:
```bash
conda env create -f environment.yml
conda activate gee-experiments
```

### 5.3 Random Seed Control

All experiments use:
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

**Documentation Completed**: 2026-02-15
**Status**: Ready for thesis integration
**Reproducibility**: Full (scripts, configs, data, environment)
