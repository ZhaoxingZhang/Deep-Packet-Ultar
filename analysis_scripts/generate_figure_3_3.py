#!/usr/bin/env python3
"""
图3-3：决策边界偏移示意图
展示标准CE和加权CE的决策边界差异
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Ellipse
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ==================== 左图：标准CE的决策边界偏移 ====================

# 绘制多数类（蓝色）的决策区域
majority_region = Ellipse((0, 0), width=3.5, height=3.0,
                          angle=0, color='steelblue', alpha=0.3, label='多数类决策区域')
ax1.add_patch(majority_region)

# 绘制少数类（红色）的真实区域（被侵占）
minority_true = Ellipse((-1.5, 0), width=1.2, height=1.0,
                       angle=0, color='crimson', alpha=0.5, label='少数类真实区域')
ax1.add_patch(minority_true)

# 绘制标准CE的决策边界（虚线）
# 边界向少数类偏移
boundary_line = plt.Line2D([-0.5, 1.5], [-1.2, 1.2],
                           color='black', linewidth=2.5, linestyle='--', label='决策边界')
ax1.add_line(boundary_line)

# 添加标注
ax1.annotate('决策边界\n向少数类偏移', xy=(0.5, 0), xytext=(1.5, 1.5),
            fontsize=11, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', lw=2, color='black'),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax1.annotate('多数类\n过度扩张', xy=(0, 0.5), xytext=(0, -1.5),
            fontsize=11, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

ax1.annotate('少数类区域\n被侵占', xy=(-1.5, 0), xytext=(-2.5, 0),
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))

ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')
ax1.set_title('(a) 标准CE：决策边界偏移', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# ==================== 右图：加权CE的中性决策边界 ====================

# 绘制多数类（蓝色）的决策区域（收缩）
majority_region_balanced = Ellipse((0.2, 0), width=3.0, height=2.5,
                                   angle=0, color='steelblue', alpha=0.3, label='多数类决策区域')
ax2.add_patch(majority_region_balanced)

# 绘制少数类（红色）的决策区域（得到保护）
minority_protected = Ellipse((-1.5, 0), width=1.4, height=1.2,
                           angle=0, color='crimson', alpha=0.5, label='少数类决策区域')
ax2.add_patch(minority_protected)

# 绘制加权CE的决策边界（虚线，中性位置）
boundary_line_balanced = plt.Line2D([-0.2, 1.0], [-1.0, 1.0],
                                    color='black', linewidth=2.5, linestyle='--', label='决策边界')
ax2.add_line(boundary_line_balanced)

# 添加标注
ax2.annotate('决策边界\n更加中性', xy=(0.4, 0), xytext=(1.8, 1.5),
            fontsize=11, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', lw=2, color='black'),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

ax2.annotate('多数类区域\n适当收缩', xy=(0.2, 0.5), xytext=(0.2, -1.5),
            fontsize=11, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

ax2.annotate('少数类区域\n得到保护', xy=(-1.5, 0), xytext=(-2.5, 0),
            fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))

ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-2, 2)
ax2.set_aspect('equal')
ax2.set_title('(b) 加权CE：中性决策边界', fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# 整体标题
fig.suptitle('图3-3：决策边界偏移示意图', fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/Users/weisman/Documents/repo_root/graduate/Deep-Packet-Ultar/thesis/figures/figure_3_3_decision_boundary_shift.png',
            dpi=300, bbox_inches='tight')
print("✅ Figure 3-3 generated successfully!")
print("📍 Saved to: thesis/figures/figure_3_3_decision_boundary_shift.png")
print("\n📊 图示内容:")
print("   (a) 标准CE：决策边界向少数类偏移，多数类过度扩张")
print("   (b) 加权CE：决策边界中性，少数类得到保护")
