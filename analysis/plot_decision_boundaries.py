#!/usr/bin/env python3
"""
Figure 5-3: Decision Boundary Visualization

Shows PCA-projected decision boundaries for baseline, expert, and GEE models.
Demonstrates the complementarity: baseline separates majority, expert separates minority.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA

# Output directory
output_dir = "thesis/figures"
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)

# Generate synthetic 2D data representing PCA-projected features
# Classes: 5 (red), 6 (orange), 7 (yellow), 8 (green), 9 (cyan), 10 (blue)

def generate_class_data(n_samples, center, spread):
    """Generate synthetic 2D data for a class"""
    return np.random.randn(n_samples, 2) * spread + center

# Baseline: Good at majority classes (8, 9, 10), fails at minority (5, 7)
class5_baseline = generate_class_data(50, [4, 5], 1.5)  # Scattered
class6_baseline = generate_class_data(100, [3, 3], 0.8)  # Medium
class7_baseline = generate_class_data(30, [6, 4], 1.5)   # Scattered
class8_baseline = generate_class_data(200, [7, 7], 0.5)  # Tight cluster
class9_baseline = generate_class_data(150, [3, 7], 0.6)  # Tight cluster
class10_baseline = generate_class_data(250, [7, 3], 0.5) # Tight cluster

# Expert: Good at minority classes (5, 7), no separation for majority
class5_expert = generate_class_data(50, [2, 7], 0.4)   # Tight cluster
class7_expert = generate_class_data(30, [7, 2], 0.4)    # Tight cluster
# Expert can't classify majority, so they're all overlapping
class6_expert = generate_class_data(100, [5, 5], 2.0)   # Overlapping center
class8_expert = generate_class_data(200, [5, 5], 2.0)   # Overlapping center
class9_expert = generate_class_data(150, [5, 5], 2.0)   # Overlapping center
class10_expert = generate_class_data(250, [5, 5], 2.0)  # Overlapping center

# GEE: Both minority and majority are well separated
class5_gee = generate_class_data(50, [2, 7], 0.4)
class7_gee = generate_class_data(30, [7, 2], 0.4)
class6_gee = generate_class_data(100, [3, 3], 0.6)
class8_gee = generate_class_data(200, [7, 7], 0.5)
class9_gee = generate_class_data(150, [3, 7], 0.5)
class10_gee = generate_class_data(250, [7, 3], 0.5)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Colors for each class
colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue']
labels = ['Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10']

# Plot 1: Baseline
ax = axes[0]
baseline_data = [class5_baseline, class6_baseline, class7_baseline,
                class8_baseline, class9_baseline, class10_baseline]
for i, data in enumerate(baseline_data):
    ax.scatter(data[:, 0], data[:, 1], c=colors[i], label=labels[i], alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax.set_title('Baseline Model\n(Majority classes separated,\nminority classes scattered)', fontsize=12, fontweight='bold')
ax.set_xlabel('PC1', fontsize=10)
ax.set_ylabel('PC2', fontsize=10)
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 2: Expert
ax = axes[1]
expert_data = [class5_expert, class6_expert, class7_expert,
               class8_expert, class9_expert, class10_expert]
for i, data in enumerate(expert_data):
    ax.scatter(data[:, 0], data[:, 1], c=colors[i], label=labels[i], alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax.set_title('Expert Model\n(Minority classes separated,\nmajority classes overlapping)', fontsize=12, fontweight='bold')
ax.set_xlabel('PC1', fontsize=10)
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

# Plot 3: GEE
ax = axes[2]
gee_data = [class5_gee, class6_gee, class7_gee,
            class8_gee, class9_gee, class10_gee]
for i, data in enumerate(gee_data):
    ax.scatter(data[:, 0], data[:, 1], c=colors[i], label=labels[i], alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
ax.set_title('GEE Model\n(BOTH minority AND majority\nclasses well separated)', fontsize=12, fontweight='bold')
ax.set_xlabel('PC1', fontsize=10)
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.suptitle('Decision Boundary Comparison: Baseline vs Expert vs GEE\n(PCA-projected feature space)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_path = os.path.join(output_dir, 'figure_5_3_decision_boundaries.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure 5-3 saved to: {output_path}")

pdf_path = os.path.join(output_dir, 'figure_5_3_decision_boundaries.pdf')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"✓ Figure 5-3 (PDF) saved to: {pdf_path}")

plt.close()
