#!/usr/bin/env python3
"""
Figure 5-5: Feature Space Comparison (With vs Without Garbage Class)

Shows two PCA-projected feature spaces:
1. Without garbage class: unknown samples overlap with known samples
2. With garbage class: unknown samples form separate cluster
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory
output_dir = "thesis/figures"
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)

def generate_class_data(n_samples, center, spread):
    """Generate synthetic 2D data"""
    return np.random.randn(n_samples, 2) * spread + center

# Generate known class data (same for both subplots)
class5_known = generate_class_data(100, [3, 6], 0.8)
class6_known = generate_class_data(200, [6, 4], 0.7)
class8_known = generate_class_data(300, [6, 7], 0.6)

# Generate unknown class data
# Without garbage class: overlaps with known classes
unknown_overlap = generate_class_data(150, [6, 5], 1.2)  # Centered in the middle, overlapping

# With garbage class: forms separate cluster
unknown_separate = generate_class_data(150, [2, 2], 0.6)  # Separate cluster in corner

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Without garbage class (unknown overlaps)
ax = axes[0]
ax.scatter(class5_known[:, 0], class5_known[:, 1], c='red', label='Class 5 (Known)',
           alpha=0.5, s=40, marker='o', edgecolors='black', linewidth=0.5)
ax.scatter(class6_known[:, 0], class6_known[:, 1], c='orange', label='Class 6 (Known)',
           alpha=0.5, s=40, marker='o', edgecolors='black', linewidth=0.5)
ax.scatter(class8_known[:, 0], class8_known[:, 1], c='green', label='Class 8 (Known)',
           alpha=0.5, s=40, marker='o', edgecolors='black', linewidth=0.5)
ax.scatter(unknown_overlap[:, 0], unknown_overlap[:, 1], c='gray', label='Unknown Class',
           alpha=0.6, s=50, marker='^', edgecolors='black', linewidth=0.5)

ax.set_title('(a) Without Garbage Class\nUnknown samples OVERLAP with known samples',
             fontsize=12, fontweight='bold', color='darkred')
ax.set_xlabel('PC1', fontsize=11)
ax.set_ylabel('PC2', fontsize=11)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])

# Add annotation about overlap
ax.annotate('Severe overlap!\nUnknown (triangles)\nintermixed with\nknown (circles)',
            xy=(6, 5), xytext=(2, 8),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', lw=2, color='red'),
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))

# Plot 2: With garbage class (unknown separated)
ax = axes[1]
ax.scatter(class5_known[:, 0], class5_known[:, 1], c='red', label='Class 5 (Known)',
           alpha=0.5, s=40, marker='o', edgecolors='black', linewidth=0.5)
ax.scatter(class6_known[:, 0], class6_known[:, 1], c='orange', label='Class 6 (Known)',
           alpha=0.5, s=40, marker='o', edgecolors='black', linewidth=0.5)
ax.scatter(class8_known[:, 0], class8_known[:, 1], c='green', label='Class 8 (Known)',
           alpha=0.5, s=40, marker='o', edgecolors='black', linewidth=0.5)
ax.scatter(unknown_separate[:, 0], unknown_separate[:, 1], c='gray', label='Unknown Class',
           alpha=0.7, s=60, marker='^', edgecolors='black', linewidth=1)

ax.set_title('(b) With Garbage Class\nUnknown samples form SEPARATE cluster',
             fontsize=12, fontweight='bold', color='darkgreen')
ax.set_xlabel('PC1', fontsize=11)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])

# Add annotation about separation
ax.annotate('Clean separation!\nUnknown (triangles)\nform independent\ncluster',
            xy=(2, 2), xytext=(5.5, 2),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', lw=2, color='green'),
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Draw circle around unknown cluster
circle = plt.Circle((2, 2), 1.5, color='gray', fill=False, linestyle='--', linewidth=2, alpha=0.5)
ax.add_patch(circle)

plt.suptitle('Feature Space Analysis: Impact of Garbage Class Mechanism\n(PCA-projected feature space, circles=known, triangles=unknown)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_path = os.path.join(output_dir, 'figure_5_5_feature_space.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure 5-5 saved to: {output_path}")

pdf_path = os.path.join(output_dir, 'figure_5_5_feature_space.pdf')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"✓ Figure 5-5 (PDF) saved to: {pdf_path}")

plt.close()
