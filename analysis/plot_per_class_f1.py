#!/usr/bin/env python3
"""
Figure 4-4: Per-Class F1 Score Comparison

Generates a grouped bar chart comparing per-class F1 scores between Baseline and GEE.
Highlights dramatic improvements for minority classes (5 and 7).

Data Source: thesis/figures_and_tables_design.md (Table 4-2 data)
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory
output_dir = "thesis/figures"
os.makedirs(output_dir, exist_ok=True)

# Data from experimental results (research log 11月16日)
classes = ['5\n(Chat)', '6\n(File)', '7\n(Email)', '8\n(P2P)', '9\n(Stream)', '10\n(VoIP)']
class_names = ['Chat', 'File Transfer', 'Email', 'P2P', 'Streaming', 'VoIP']
f1_baseline = [0.00, 0.82, 0.00, 1.00, 0.98, 0.98]
f1_gee = [0.28, 0.82, 0.92, 1.00, 0.99, 0.97]

# Create figure
x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, f1_baseline, width, label='Baseline',
              color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, f1_gee, width, label='GEE',
              color='crimson', alpha=0.8, edgecolor='black', linewidth=1.2)

# Customize plot
ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_title('Per-Class F1 Score: Baseline vs GEE', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=10)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_axisbelow(True)

# Add horizontal line at y=0.5 for reference
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# Highlight minority class improvements
for i in [0, 2]:  # Classes 5 and 7 (indices 0 and 2)
    ax.annotate(f'{class_names[i]}\nF1: {f1_baseline[i]:.2f}→{f1_gee[i]:.2f}',
                xy=(i, max(f1_baseline[i], f1_gee[i]) + 0.02),
                xytext=(i, 1.15),
                fontsize=9, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

# Add Macro-F1 annotation
textstr = f'Macro-F1:\nBaseline: 0.63\nGEE: 0.83\nImprovement: +32%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right', bbox=props)

# Add annotation for minority class breakthrough
ax.annotate('Minority classes (5, 7)\nshow dramatic improvements\n(0 → 0.28, 0 → 0.92)',
            xy=(1, 0.5), xytext=(3.5, 0.3),
            fontsize=10, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))

plt.tight_layout()

# Save figure
output_path = os.path.join(output_dir, 'figure_4_4_per_class_f1.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure 4-4 saved to: {output_path}")

# Also save as PDF for LaTeX
pdf_path = os.path.join(output_dir, 'figure_4_4_per_class_f1.pdf')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"✓ Figure 4-4 (PDF) saved to: {pdf_path}")

plt.close()
