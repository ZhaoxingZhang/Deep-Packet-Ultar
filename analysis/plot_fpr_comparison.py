#!/usr/bin/env python3
"""
Figure 4-2: FPR@TPR95 Comparison Bar Chart

Generates a grouped bar chart comparing FPR@TPR95 between Baseline and GEE
for all 6 folds (classes 5-10 as unknown).

Data Source: thesis/figures_and_tables_design.md (Table 4-1 data)
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory
output_dir = "thesis/figures"
os.makedirs(output_dir, exist_ok=True)

# Data from experimental results (research log 12月6日)
excluded_classes = ['5\n(Chat)', '6\n(File)', '7\n(Email)', '8\n(P2P)', '9\n(Stream)', '10\n(VoIP)']
class_names = ['Chat', 'File Transfer', 'Email', 'P2P', 'Streaming', 'VoIP']
fpr_baseline = [0.2727, 0.0208, 0.0909, 0.9742, 0.8266, 0.6682]
fpr_gee = [0.2617, 0.0668, 0.0323, 0.0246, 0.0127, 0.0254]

# Create figure
x = np.arange(len(excluded_classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, fpr_baseline, width, label='Baseline',
              color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, fpr_gee, width, label='GEE',
              color='crimson', alpha=0.8, edgecolor='black', linewidth=1.2)

# Customize plot
ax.set_ylabel('FPR@TPR95', fontsize=12, fontweight='bold')
ax.set_xlabel('Excluded Class (Unknown)', fontsize=12, fontweight='bold')
ax.set_title('FPR@TPR95 Comparison: Baseline vs GEE', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(excluded_classes, fontsize=10)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_axisbelow(True)

# Set y-axis limit to show all bars clearly
ax.set_ylim([0, 1.0])

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height < 0.1:  # Only show labels for small values
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)

# Add annotation for key improvements
ax.annotate('Worst cases show\ndramatic improvement',
            xy=(2.5, 0.85), xytext=(3.5, 0.95),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))

# Add average values as text box
avg_baseline = np.mean(fpr_baseline)
avg_gee = np.mean(fpr_gee)
improvement = avg_baseline - avg_gee

textstr = f'Average FPR@TPR95:\nBaseline: {avg_baseline:.3f}\nGEE: {avg_gee:.3f}\nImprovement: {improvement:.3f} (7x better)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

plt.tight_layout()

# Save figure
output_path = os.path.join(output_dir, 'figure_4_2_fpr_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Figure 4-2 saved to: {output_path}")

# Also save as PDF for LaTeX
pdf_path = os.path.join(output_dir, 'figure_4_2_fpr_comparison.pdf')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
print(f"✓ Figure 4-2 (PDF) saved to: {pdf_path}")

plt.close()
