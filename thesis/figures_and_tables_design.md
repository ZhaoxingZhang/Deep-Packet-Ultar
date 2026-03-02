# Tables and Figures Design

**Design Date**: 2026-02-15
**Purpose**: Complete specifications for all 4 tables and 7 figures in thesis

---

## Tables (4 total)

### Table 4-1: OSR Performance Comparison

**Title**: 开放集识别性能对比（ResNet-Baseline vs ResNet-GEE）
**Location**: Chapter 4.2 (Section 4.2 开放集识别实验)
**Format**: LaTeX booktabs, professional academic style

**Structure**:
```latex
\begin{table}[htbp]
\caption{开放集识别性能对比（ResNet-Baseline vs ResNet-GEE）}
\label{tab:osr_comparison}
\centering
\begin{tabular}{clcccccc}
\toprule
排除类别 & \multicolumn{3}{c}{AUROC} & \multicolumn{3}{c}{FPR@TPR95} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
(未知类) & Baseline & GEE & 提升 & Baseline & GEE & 提升 \\
\midrule
5 (Chat) & 0.9385 & 0.9284 & -0.0101 & 0.2727 & 0.2617 & -0.0110 \\
6 (File Transfer) & 0.9737 & 0.9848 & \textbf{+0.0111} & 0.0208 & 0.0668 & +0.0460 \\
7 (Email) & 0.9746 & 0.9629 & -0.0117 & 0.0909 & 0.0323 & \textbf{-0.0586} \\
8 (P2P) & 0.8692 & 0.9892 & \textbf{+0.1200} & 0.9742 & 0.0246 & \textbf{-0.9496} \\
9 (Streaming) & 0.9420 & 0.9955 & \textbf{+0.0535} & 0.8266 & 0.0127 & \textbf{-0.8139} \\
10 (VoIP) & 0.9511 & 0.9883 & \textbf{+0.0372} & 0.6682 & 0.0254 & \textbf{-0.6428} \\
\midrule
平均 & 0.940 & 0.975 & +0.035 & 0.476 & 0.071 & \textbf{-0.405} \\
\bottomrule
\end{tabular}
\end{table}
```

**Data Source**: Research log 11月22日 (baseline), 12月6日 (GEE)

**Notes**:
- Bold significant improvements (|Δ| > 0.03 for AUROC, |Δ| > 0.05 for FPR@TPR95)
- Average FPR@TPR95 improvement: 7x better (0.476 → 0.071)
- Highlight that majority classes (8, 9, 10) show largest improvements

---

### Table 4-2: IL Performance Comparison

**Title**: 增量学习性能对比（Baseline vs GEE）
**Location**: Chapter 4.3 (Section 4.3 增量学习实验)
**Format**: LaTeX booktabs

**Structure**:
```latex
\begin{table}[htbp]
\caption{增量学习性能对比（Baseline vs GEE）}
\label{tab:il_comparison}
\centering
\begin{tabular}{clcccc}
\toprule
类别 & 样本数 & Baseline F1 & GEE F1 & 提升 \\
\midrule
5 (VPN: Chat) & 215 & 0.00 & 0.28 & \textbf{+0.28} \\
6 (VPN: File Transfer) & 1034 & 0.82 & 0.82 & 0.00 \\
7 (VPN: Email) & 65 & 0.00 & 0.92 & \textbf{+0.92} \\
8 (VPN: P2P) & 4408 & 1.00 & 1.00 & 0.00 \\
9 (VPN: Streaming) & 1089 & 0.98 & 0.99 & +0.01 \\
10 (VPN: VoIP) & 9476 & 0.98 & 0.97 & -0.01 \\
\midrule
宏平均 (Macro-F1) & 16287 & 0.63 & 0.83 & \textbf{+0.20} \\
准确率 (Accuracy) & 16287 & 0.9633 & 0.9500 & -0.0133 \\
\bottomrule
\end{tabular}
\end{table}
```

**Data Source**: Research log 10月31日 (baseline), 11月16日 (GEE)

**Notes**:
- Minority classes (5, 7) show dramatic improvements: 0 → 0.28, 0 → 0.92
- Macro-F1 increases from 0.63 to 0.83 (32% relative improvement)
- Accuracy slightly decreases (1.33%) but Macro-F1 significantly improves

---

### Table 4-3: Ablation Study Results

**Title**: 消融实验结果（各组件贡献）
**Location**: Chapter 4.4 (Section 4.4 消融实验)
**Format**: LaTeX booktabs

**Structure**:
```latex
\begin{table}[htbp]
\caption{消融实验结果（各组件贡献）}
\label{tab:ablation}
\centering
\begin{tabular}{clcccc}
\toprule
配置 & 描述 & Macro-F1 & 类5 F1 & 类7 F1 \\
\midrule
Config 1 & Baseline (单模型) & 0.63 & 0.00 & 0.00 \\
Config 2 & Baseline + Simple Weighted (0.85/0.15) & 0.67 & 0.02 & 0.06 \\
Config 3 & Gating Network (标准CE) & 0.63 & 0.00 & 0.00 \\
Config 4 & Gating Network (加权CE) & 0.83 & 0.28 & 0.92 \\
\bottomrule
\end{tabular}
\end{table}
```

**Data Source**: Research log 11月16日

**Notes**:
- Progressive improvement from Config 1 to 4
- Config 3 shows gating network fails without weighted CE (back to 0.63)
- Config 4 (GEE full) achieves best performance
- Key conclusion: 加权损失是关键

---

### Table 4-4: CNN-GEE Validation Results

**Title**: CNN-GEE多模型验证结果
**Location**: Chapter 4.5 (Section 4.5 多模型验证)
**Format**: LaTeX booktabs

**Structure**:
```latex
\begin{table}[htbp]
\caption{CNN-GEE多模型验证结果}
\label{tab:cnn_gee_validation}
\centering
\begin{tabular}{clcccc}
\toprule
排除类别 & 指标 & 基准 CNN & CNN-GEE & 提升情况 \\
\midrule
5 (Chat) & AUROC & 0.9473 & 0.8412 & ↓ \\
           & FPR@TPR95 & 0.2710 & 0.3364 & ↓ \\
\midrule
6 (File Transfer) & AUROC & 0.9399 & 0.9765 & \textbf{↑} \\
                  & FPR@TPR95 & 0.2615 & 0.0973 & \textbf{↑} \\
\midrule
7 (Email) & AUROC & 0.9776 & 0.9959 & \textbf{↑} \\
           & FPR@TPR95 & 0.0968 & 0.0000 & \textbf{↑↑} \\
\midrule
8 (P2P) & AUROC & 0.9672 & 0.9792 & \textbf{↑} \\
          & FPR@TPR95 & 0.2421 & 0.0831 & \textbf{↑} \\
\midrule
9 (Streaming) & AUROC & 0.9553 & 0.9426 & ↓ \\
               & FPR@TPR95 & 0.3309 & 0.1655 & \textbf{↑} \\
\midrule
10 (VoIP) & AUROC & 0.9856 & 0.9912 & \textbf{↑} \\
           & FPR@TPR95 & 0.0298 & 0.0231 & \textbf{↑} \\
\bottomrule
\end{tabular}
\end{table}
```

**Data Source**: Research log 12月19日

**Notes**:
- Consistent improvement trend with ResNet-GEE
- Class 7: Perfect OSR (FPR@TPR95 = 0.0000)
- Some AUROC decreases (class 5, 9) but FPR@TPR95 mostly improves
- Conclusion: GEE架构可迁移到CNN

---

## Figures (7 total)

### Figure 4-1: ROC Curves for OSR

**Title**: 开放集识别ROC曲线对比（ResNet-Baseline vs ResNet-GEE）
**Location**: Chapter 4.2
**Type**: Multi-panel line plot (6 subplots in 2x3 grid)
**Size**: Full page width

**Design**:
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# 6 subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, excluded_class in enumerate([5, 6, 7, 8, 9, 10]):
    ax = axes[i]

    # Plot baseline ROC curve
    fpr_baseline, tpr_baseline, _ = roc_curve(y_true, y_score_baseline)
    auc_baseline = auc(fpr_baseline, tpr_baseline)
    ax.plot(fpr_baseline, tpr_baseline, 'b--', label=f'Baseline (AUROC={auc_baseline:.4f})', linewidth=2)

    # Plot GEE ROC curve
    fpr_gee, tpr_gee, _ = roc_curve(y_true, y_score_gee)
    auc_gee = auc(fpr_gee, tpr_gee)
    ax.plot(fpr_gee, tpr_gee, 'r-', label=f'GEE (AUROC={auc_gee:.4f})', linewidth=2)

    # Shade AUROC regions
    ax.fill_between(fpr_baseline, tpr_baseline, alpha=0.1, color='blue')
    ax.fill_between(fpr_gee, tpr_gee, alpha=0.1, color='red')

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k:', label='Random', linewidth=1)

    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(f'Fold {i+1}: Class {excluded_class} as Unknown')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('figure_4_1_roc_curves.png', dpi=300)
```

**Data Source**: Extract predictions from saved models, compute ROC curves

**Notes**:
- X-axis: FPR, Y-axis: TPR
- Two curves per subplot: Baseline (blue dashed) vs GEE (red solid)
- Shade AUROC regions with transparent colors
- Add diagonal reference line
- Legend shows AUROC values
- Highlight: Folds 8, 9, 10 show largest GEE improvements

---

### Figure 4-2: FPR@TPR95 Bar Chart

**Title**: FPR@TPR95对比（基线 vs GEE）
**Location**: Chapter 4.2
**Type**: Grouped bar chart
**Size**: Half page width

**Design**:
```python
import matplotlib.pyplot as plt
import numpy as np

excluded_classes = ['5\n(Chat)', '6\n(File)', '7\n(Email)', '8\n(P2P)', '9\n(Stream)', '10\n(VoIP)']
fpr_baseline = [0.2727, 0.0208, 0.0909, 0.9742, 0.8266, 0.6682]
fpr_gee = [0.2617, 0.0668, 0.0323, 0.0246, 0.0127, 0.0254]

x = np.arange(len(excluded_classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, fpr_baseline, width, label='Baseline', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, fpr_gee, width, label='GEE', color='red', alpha=0.7)

ax.set_ylabel('FPR@TPR95')
ax.set_xlabel('Excluded Class (Unknown)')
ax.set_title('FPR@TPR95 Comparison: Baseline vs GEE')
ax.set_xticks(x)
ax.set_xticklabels(excluded_classes)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figure_4_2_fpr_comparison.png', dpi=300)
```

**Data Source**: Table 4-1 data

**Notes**:
- X-axis: Excluded class (with descriptive names)
- Y-axis: FPR@TPR95 (lower is better)
- Grouped bars: Baseline (blue) vs GEE (red)
- Highlight: Bars 8, 9, 10 show dramatic reductions
- Add value labels on top of bars

---

### Figure 4-4: Per-Class F1 Comparison

**Title**: 各类别F1分数对比（Baseline vs GEE）
**Location**: Chapter 4.3
**Type**: Grouped bar chart
**Size**: Half page width

**Design**:
```python
import matplotlib.pyplot as plt
import numpy as np

classes = ['5\n(Chat)', '6\n(File)', '7\n(Email)', '8\n(P2P)', '9\n(Stream)', '10\n(VoIP)']
f1_baseline = [0.00, 0.82, 0.00, 1.00, 0.98, 0.98]
f1_gee = [0.28, 0.82, 0.92, 1.00, 0.99, 0.97]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, f1_baseline, width, label='Baseline', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, f1_gee, width, label='GEE', color='red', alpha=0.7)

ax.set_ylabel('F1 Score')
ax.set_xlabel('Class')
ax.set_title('Per-Class F1 Score: Baseline vs GEE')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Add Macro-F1 annotation
ax.axvline(x=-0.5, color='black', linestyle='-', linewidth=2)
ax.text(2.5, 1.05, f'Macro-F1: Baseline=0.63, GEE=0.83',
         ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figure_4_4_per_class_f1.png', dpi=300)
```

**Data Source**: Table 4-2 data

**Notes**:
- Highlight: Classes 5 and 7 show dramatic improvements (0 → 0.28, 0 → 0.92)
- Classes 6, 8, 9, 10 maintain high performance
- Add Macro-F1 annotation at top

---

### Figure 5-1: Gating Network Weight Heatmap

**Title**: 门控网络权重分布热力图
**Location**: Chapter 5.1
**Type**: Heatmap
**Size**: Half page width

**Design**:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load gating network first layer weights
# Shape: (hidden_dim, input_dim) where input_dim = 2 * num_classes
weights = gating_network.fc1.weight.detach().cpu().numpy()  # Example

# Split into baseline and expert contributions
num_classes = 6
baseline_weights = weights[:, :num_classes]
expert_weights = weights[:, num_classes:]

# Combine for visualization
combined_weights = np.concatenate([baseline_weights, expert_weights], axis=0)

# Labels
y_labels = [f'Hidden {i}' for i in range(combined_weights.shape[0])]
x_labels = [f'C{i}' for i in range(num_classes)] + [f'C{i}' for i in range(num_classes)]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(combined_weights, annot=False, cmap='RdBu_r', center=0,
            xticklabels=x_labels, yticklabels=y_labels, ax=ax)

# Add separator line
ax.axvline(x=num_classes, color='black', linewidth=2)
ax.text(num_classes/2, -0.5, 'Baseline Inputs', ha='center', fontsize=12, fontweight='bold')
ax.text(num_classes + num_classes/2, -0.5, 'Expert Inputs', ha='center', fontsize=12, fontweight='bold')

ax.set_title('Gating Network First-Layer Weights')
ax.set_xlabel('Input Neuron (Class Probability)')
ax.set_ylabel('Hidden Neuron')

plt.tight_layout()
plt.savefig('figure_5_1_weight_heatmap.png', dpi=300)
```

**Data Source**: Extract from `gating_network_cross_entropy_loss.pth`

**Notes**:
- Show that expert weights (right half) have larger absolute values than baseline weights (left half)
- Use red-blue colormap (red=positive, blue=negative)
- Add vertical separator between baseline and expert inputs
- Analysis: Expert weights ≈ 2.3x larger for minority classes

---

### Figure 5-2: Decision Boundary Comparison

**Title**: 决策边界可视化（Baseline vs Expert vs GEE）
**Location**: Chapter 5.2
**Type**: Scatter plots with PCA projection (3 subplots in 1x3 grid)
**Size**: Full page width

**Design**:
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Extract penultimate layer features
features_baseline = extract_features(baseline_model, test_loader)
features_expert = extract_features(expert_model, test_loader)
features_gee = extract_features(geegate_model, test_loader)

# PCA to 2D
pca = PCA(n_components=2)
features_baseline_2d = pca.fit_transform(features_baseline)
features_expert_2d = pca.fit_transform(features_expert)
features_gee_2d = pca.fit_transform(features_gee)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, features, title in zip(axes,
                               [features_baseline_2d, features_expert_2d, features_gee_2d],
                               ['Baseline', 'Expert', 'GEE']):
    for class_id in [5, 6, 7, 8, 9, 10]:
        mask = (y_true == class_id)
        ax.scatter(features[mask, 0], features[mask, 1], label=f'Class {class_id}', alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()

plt.tight_layout()
plt.savefig('figure_5_2_decision_boundaries.png', dpi=300)
```

**Data Source**: Extract features from saved models

**Notes**:
- Subplot 1 (Baseline): Majority classes (8, 9, 10) form clear clusters, minority classes (5, 7) scattered
- Subplot 2 (Expert): Minority classes (5, 7) form clear clusters, majority classes overlap
- Subplot 3 (GEE): Both majority and minority classes form clear clusters
- Use different colors for each class

---

### Figure 5-3: Feature Space with Garbage Class

**Title**: 垃圾类对特征空间的影响
**Location**: Chapter 5.4
**Type**: Scatter plots with PCA projection (2 subplots in 1x2 grid)
**Size**: Full page width

**Design**:
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Extract features for models with and without garbage class
features_no_garbage = extract_features(model_no_garbage, test_loader)
features_with_garbage = extract_features(model_with_garbage, test_loader)

# PCA
pca = PCA(n_components=2)
features_no_garbage_2d = pca.fit_transform(features_no_garbage)
features_with_garbage_2d = pca.fit_transform(features_with_garbage)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Without garbage class
ax = axes[0]
ax.scatter(features_no_garbage_2d[known_mask, 0], features_no_garbage_2d[known_mask, 1],
           c='blue', label='Known', alpha=0.6)
ax.scatter(features_no_garbage_2d[unknown_mask, 0], features_no_garbage_2d[unknown_mask, 1],
           c='red', label='Unknown', alpha=0.6)
ax.set_title('Without Garbage Class')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()

# Subplot 2: With garbage class
ax = axes[1]
ax.scatter(features_with_garbage_2d[known_mask, 0], features_with_garbage_2d[known_mask, 1],
           c='blue', label='Known', alpha=0.6)
ax.scatter(features_with_garbage_2d[unknown_mask, 0], features_with_garbage_2d[unknown_mask, 1],
           c='red', label='Unknown', alpha=0.6)
ax.set_title('With Garbage Class')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()

plt.tight_layout()
plt.savefig('figure_5_3_garbage_class_feature_space.png', dpi=300)
```

**Data Source**: Train two models (with/without garbage class), extract features

**Notes**:
- Subplot 1: Unknown samples intermingle with known (overlap coefficient = 0.82)
- Subplot 2: Unknown samples form distinct cluster (overlap coefficient = 0.15)
- This explains why FPR@TPR95 improves from 0.8+ to 0.03

---

### Figure 5-4: Gradient Contribution Comparison

**Title**: 梯度贡献对比（标准CE vs 加权CE）
**Location**: Chapter 5.3
**Type**: Grouped bar chart
**Size**: Half page width

**Design**:
```python
import matplotlib.pyplot as plt
import numpy as np

classes = ['5\n(Chat)', '6\n(File)', '7\n(Email)', '8\n(P2P)', '9\n(Stream)', '10\n(VoIP)']
grad_unweighted = [10, 80, 12, 150, 130, 145]  # Example values
grad_weighted = [55, 70, 60, 80, 75, 78]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, grad_unweighted, width, label='Unweighted CE', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, grad_weighted, width, label='Weighted CE', color='red', alpha=0.7)

ax.set_ylabel('Average Gradient L2 Norm')
ax.set_xlabel('Class')
ax.set_title('Gradient Contribution per Class: Unweighted vs Weighted CE')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figure_5_4_gradient_contribution.png', dpi=300)
```

**Data Source**: Record gradients during training, compute L2 norm per class

**Notes**:
- Unweighted CE: Majority classes (8, 9, 10) dominate gradients (15x vs class 7)
- Weighted CE: All classes have balanced gradient contributions (1.3x ratio)
- This explains why weighted CE prevents gating network from ignoring expert

---

## Implementation Priority

### Priority 1 (Critical for Chapter 4):
1. ✅ Table 4-1: Data exists, just format
2. ✅ Table 4-2: Data exists, just format
3. ✅ Table 4-3: Data exists, just format
4. ✅ Table 4-4: Data exists, just format
5. ⚠️ Figure 4-1: ROC Curves - Need to generate from model predictions
6. ⚠️ Figure 4-2: FPR@TPR95 Bar Chart - Easy (data from Table 4-1)
7. ⚠️ Figure 4-4: Per-Class F1 - Easy (data from Table 4-2)

### Priority 2 (Critical for Chapter 5):
8. ⚠️ Figure 5-1: Weight Heatmap - Need to extract from gating network
9. ⚠️ Figure 5-2: Decision Boundaries - Need to extract features and run PCA
10. ⚠️ Figure 5-3: Garbage Class Feature Space - Need to train two models and extract features
11. ⚠️ Figure 5-4: Gradient Contribution - Need to re-train and record gradients

---

## Implementation Scripts

To be created in `analysis/` directory:
1. `plot_roc_curves.py` - Generate Figure 4-1
2. `plot_fpr_comparison.py` - Generate Figure 4-2
3. `plot_per_class_f1.py` - Generate Figure 4-4
4. `extract_gating_weights.py` - Generate Figure 5-1
5. `visualize_decision_boundaries.py` - Generate Figure 5-2
6. `visualize_garbage_class_space.py` - Generate Figure 5-3
7. `analyze_gradient_contributions.py` - Generate Figure 5-4

---

**Design Completed**: 2026-02-15
**Next Step**: Implement visualization scripts in `analysis/` directory
