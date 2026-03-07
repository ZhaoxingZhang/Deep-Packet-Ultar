#!/usr/bin/env python3
"""
Figure 5-2: Output Contribution Bar Chart

Shows the average baseline and expert contributions for each class,
demonstrating that the gating network learned to trust the expert
for minority classes (5, 7) and baseline for majority classes.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory
output_dir = "thesis/figures"
os.makedirs(output_dir, exist_ok=True)

# Data from research findings (Chapter 5.1.3)
classes = ['5\n(Chat)', '6\n(File)', '7\n(Email)', '8\n(P2P)', '9\n(Stream)', '10\n(VoIP)']
baseline_contrib = [0.22, 0.60, 0.22, 0.78, 0.75, 0.80]
expert_contrib = [0.78, 0.40, 0.78, 0.22, 0.25, 0.20]

# Create figure
x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, baseline_contrib, width, label='Baseline Contribution',
              color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, expert_contrib, width, label='Expert Contribution',
              color='crimson', alpha=0.8, edgecolor='black', linewidth=1.2)

# Customize plot
ax.set_ylabel('Average Contribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_title('Gating Network Output Contribution: Baseline vs Expert\n(Average contribution per class)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=10)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_axisbelow(True)
ax.set_ylim([0, 1.0])

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# Highlight minority classes where expert > baseline
for i in [0, 2]:  # Classes 5 and 7
    ax.axvline(x=i-0.4, color='red', alpha=0.2, linewidth=15)
    ax.axvline(x=i+0.4, color='red', alpha=0.2, linewidth=15)

# Add annotation
ax.annotate('Minority classes:\nExpert > Baseline',
            xy=(0, 0.78), xytext=(1.5, 0.90),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.annotate('Majority classes:\nBaseline > Expert',
            xy=(3, 0.78), xytext=(4.5, 0.90),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'),
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()

# Save figure
output_path = os.path.join(output_dir, 'figure_5_2_output_contribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure 5-2 saved to: {output_path}")

pdf_path = os.path.join(output_dir, 'figure_5_2_output_contribution.pdf')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"✓ Figure 5-2 (PDF) saved to: {pdf_path}")

plt.close()
