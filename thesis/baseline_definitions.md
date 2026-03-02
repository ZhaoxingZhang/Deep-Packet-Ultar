# Baseline Method Definitions

**Purpose**: Explicitly define baseline methods for fair comparison in thesis experiments
**Date**: 2026-02-15

---

## 1. OSR (Open-Set Recognition) Baseline

### 1.1 Method: Softmax Confidence Threshold

**Principle**: Use maximum Softmax probability as confidence score

**Training Procedure**:
```python
# Train ResNet model on K-1 known classes (exclude unknown class)
model = ResNet1d(output_dim=K-1)
train_loader = DataLoader(dataset excluding unknown_class)
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for batch in train_loader:
        logits = model(batch.x)
        loss = CrossEntropyLoss()(logits, batch.y)
        loss.backward()
        optimizer.step()
```

**Inference Procedure**:
```python
def predict_with_threshold(model, x, threshold=0.5):
    logits = model(x)
    probs = softmax(logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)

    if confidence > threshold:
        return pred.item(), confidence.item(), "known"
    else:
        return -1, confidence.item(), "unknown"
```

**Decision Rule**:
- **Known**: confidence = max(P_softmax) > 0.5, predicted_class = argmax(P_softmax)
- **Unknown**: confidence = max(P_softmax) ≤ 0.5

**Evaluation Metrics**:
- AUROC: Area under ROC curve (TPR vs FPR)
- FPR@TPR95: False Positive Rate when True Positive Rate = 0.95

**Parameters**:
- Threshold: 0.5 (default, can be tuned on validation set)
- Architecture: ResNet1d (same as GEE's baseline)
- Training: Same hyperparameters as GEE baseline (lr=0.001, epochs=50, batch_size=32)

**Expected Performance** (from 11月22日 experiment):
- Average AUROC: 0.94
- Average FPR@TPR95: 0.48
- Worst case (class 8 as unknown): FPR@TPR95 = 0.97

**Limitations**:
- Modern neural networks are overconfident: even unknown samples get high confidence
- Fixed threshold may not generalize across different unknown classes
- Cannot learn "what unknown looks like" (never sees unknown during training)

---

## 2. IL (Incremental Learning) Baseline

### 2.1 Method: Full Fine-Tuning (全量微调)

**Principle**: Retrain entire model from scratch on combined old + new class data

**Training Procedure**:
```python
# Step 1: Train on initial classes (5, 6, 7, 8, 9, 10)
model = ResNet1d(output_dim=6)
train_on_classes(model, [5, 6, 7, 8, 9, 10])

# Step 2: When new class (11) arrives, combine data and retrain
model_new = ResNet1d(output_dim=7)  # Expand output for new class
combined_data = old_data + new_data  # All data together
train_on_classes(model_new, [5, 6, 7, 8, 9, 10, 11])
```

**Key Characteristics**:
- **Data Requirement**: Must keep all historical training data
- **Compute Cost**: Retraining from scratch each time (expensive)
- **Performance**: Theoretically optimal (can reach highest accuracy)
- **Catastrophic Forgetting**: Not an issue (retraining from scratch)

**Evaluation Metrics**:
- Macro-F1: Average F1 across all classes (unweighted)
- Per-Class F1: F1 for each individual class
- Training Time: Total time to retrain model
- Storage: Must store all historical data

**Expected Performance** (from 11月16日 experiment):
- Macro-F1: 0.96 (best possible)
- Training Time: ~2-3 hours (full retraining)
- Data Storage: All previous data must be kept

**Comparison with GEE**:
| Aspect | Full Fine-Tuning | GEE |
|--------|------------------|-----|
| Macro-F1 | 0.96 | 0.83 |
| Training Time | 2-3 hours | 10-15 minutes (only gating network) |
| Old Data Storage | Required | Not required (only models) |
| Catastrophic Forgetting | N/A (retraining) | Avoided (baseline + expert separate) |
| Scalability | Poor (linear growth with new classes) | Good (add new experts) |

**Limitations**:
- Computationally expensive: every new class requires full retraining
- Storage inefficient: must keep all historical data
- Not scalable: training time grows linearly with number of classes

---

## 3. Ablation Study Configurations

### 3.1 Config 1: Baseline (Single Model)

**Setup**:
- Single ResNet model trained on all classes
- No gating network, no expert model

**Training**:
```python
model = ResNet1d(output_dim=6)
train_on_imbalanced_data(model, train_loader)  # All 6 classes
```

**Expected Results**:
- Macro-F1: 0.63
- Minority Class F1: 0.00 (classes 5 and 7)
- Majority Class F1: >0.95 (classes 8, 9, 10)

**Purpose**: Establish baseline performance without any GEE components

---

### 3.2 Config 2: Simple Weighted Average

**Setup**:
- Baseline model (on all classes) + Expert model (on minority classes)
- Fixed weighted average: 0.85 × baseline_output + 0.15 × expert_output

**Training**:
```python
baseline = ResNet1d(output_dim=6)
expert = ResNet1d(output_dim=2)  # Only classes 5 and 7
train_on_imbalanced_data(baseline, train_loader_all)
train_on_minority_data(expert, train_loader_minority)

# Inference: fixed weighted average
def simple_ensemble(x):
    baseline_prob = baseline(x)
    expert_prob = expand_to_6_classes(expert(x))  # Expand 2→6 classes
    return 0.85 * baseline_prob + 0.15 * expert_prob
```

**Expected Results**:
- Macro-F1: 0.67
- Minority Class F1: 0.02-0.06 (small improvement)
- Improvement over Config 1: +0.04 Macro-F1

**Purpose**: Show that simple fusion helps, but is limited by fixed weights

---

### 3.3 Config 3: Gating Network (Standard Cross-Entropy)

**Setup**:
- Baseline + Expert + Gating Network
- Gating network trained with **standard CE loss** (no class weighting)

**Training**:
```python
baseline = ResNet1d(output_dim=6)
expert = ResNet1d(output_dim=2)
gating = GatingNetwork(input_dim=12, output_dim=6)  # 2×6=12 inputs

# Freeze baseline and expert, only train gating
for param in baseline.parameters():
    param.requires_grad = False
for param in expert.parameters():
    param.requires_grad = False

# Train gating with standard CE
criterion = CrossEntropyLoss()  # Standard (unweighted)
optimizer = Adam(gating.parameters(), lr=0.001)

for batch in train_loader:
    baseline_out = baseline(batch.x)
    expert_out = expand_to_6_classes(expert(batch.x))
    gating_input = torch.cat([baseline_out, expert_out], dim=1)
    logits = gating(gating_input)
    loss = criterion(logits, batch.y)
    loss.backward()
    optimizer.step()
```

**Expected Results**:
- Macro-F1: ≈0.63 (may drop back to baseline level)
- Minority Class F1: 0.00 (gating network ignores expert)

**Failure Analysis**:
- Gating network learns "always trust baseline" to maximize overall accuracy
- Standard CE is dominated by majority class gradients
- Minority class influence is negligible (65 samples vs 9476 samples = 1:146 ratio)

**Purpose**: Demonstrate that gating network **fails** without weighted loss

---

### 3.4 Config 4: GEE (Gating Network with Weighted CE)

**Setup**:
- Baseline + Expert + Gating Network
- Gating network trained with **weighted CE loss** (class weights inversely proportional to sample count)

**Training**:
```python
# Calculate class weights
class_counts = [215, 1034, 65, 4408, 1089, 9476]  # Classes 5-10
total_samples = sum(class_counts)
class_weights = [total_samples / (len(class_counts) * count) for count in class_counts]
# Result: [12.63, 2.62, 41.75, 0.62, 2.49, 0.29]

# Train gating with weighted CE
criterion = CrossEntropyLoss(weight=torch.tensor(class_weights))
optimizer = Adam(gating.parameters(), lr=0.001)

for batch in train_loader:
    baseline_out = baseline(batch.x)
    expert_out = expand_to_6_classes(expert(batch.x))
    gating_input = torch.cat([baseline_out, expert_out], dim=1)
    logits = gating(gating_input)
    loss = criterion(logits, batch.y)
    loss.backward()
    optimizer.step()
```

**Expected Results**:
- Macro-F1: 0.83
- Minority Class F1: 0.28 (class 5), 0.92 (class 7)
- Improvement over Config 1: +0.20 Macro-F1
- Improvement over Config 3: +0.20 Macro-F1

**Success Analysis**:
- Weighted CE balances gradient contributions from majority and minority classes
- Gating network learns "when to trust expert" for minority classes
- Expert weights are ~2.3x larger than baseline weights for minority classes

**Purpose**: Demonstrate that **weighted CE is critical** for GEE's success

---

## 4. Baseline Reproducibility

### 4.1 Code Locations

**OSR Baseline**:
- Script: `scripts/run_open_set_resnet_baseline.sh`
- Training: `train_resnet.py --output_dim K-1`
- Evaluation: `evaluation.py --eval-mode standard --open-set-eval`

**IL Baseline**:
- Script: `scripts/run_incremental_resnet_baseline.sh`
- Training: `train_resnet.py --output_dim N` (N = all classes)
- Evaluation: `evaluation.py --eval-mode standard`

**Ablation Configs**:
- Config 1: Baseline only (see above)
- Config 2: `evaluation.py --eval-mode ensemble --fixed-weights 0.85 0.15`
- Config 3: `train_gating_network.py` (without `--use-class-weights`)
- Config 4: `train_gating_network.py --use-class-weights` (= GEE)

### 4.2 Hyperparameters

**All baselines use same hyperparameters**:
- Architecture: ResNet1d (16 BasicBlocks, 3 FC layers)
- Optimizer: Adam (lr=0.001)
- Batch Size: 32
- Epochs: 50
- Learning Rate Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Early Stopping: patience=10 on validation loss

**Data Splits**:
- Train/Validation/Test: 70%/15%/15%
- Random Seed: 42 (for reproducibility)

### 4.3 Evaluation Protocol

**OSR Evaluation**:
1. Train model on K-1 classes (exclude unknown class)
2. Test on all K classes (including unknown)
3. Compute ROC curve (vary confidence threshold)
4. Report AUROC and FPR@TPR95

**IL Evaluation**:
1. Train model on all classes (imbalanced dataset)
2. Test on held-out test set
3. Compute Macro-F1 and Per-Class F1
4. Report training time and model size

---

## 5. Fair Comparison Guarantees

### 5.1 Controlled Variables

**Same Across All Methods**:
- Dataset: Same train/validation/test splits
- Architecture: Same ResNet1d backbone (where applicable)
- Training Hyperparameters: Same optimizer, learning rate, batch size, epochs
- Evaluation Metrics: Same calculation methods
- Random Seeds: Fixed for reproducibility

### 5.2 Only Variable

**The only difference between methods**:
- **OSR**: Threshold mechanism (Softmax vs Garbage Class)
- **IL**: Training strategy (Full Fine-Tuning vs GEE)
- **Ablation**: Gating network loss function (Standard CE vs Weighted CE)

This ensures that performance differences are due **only** to the proposed innovations, not confounding factors.

---

**Document Completed**: 2026-02-15
**Status**: Ready for integration into thesis Chapter 4
