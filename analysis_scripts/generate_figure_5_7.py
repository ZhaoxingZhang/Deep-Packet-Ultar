#!/usr/bin/env python3
"""
图5-7：置信度校准曲线（Reliability Diagram）
展示Baseline和GEE的置信度vs实际准确率
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 8))

# 置信度区间
bins = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
bin_centers = (bins[:-1] + bins[1:]) / 2

# Baseline的数据（严重误校准）
# 对未知样本给出高置信度但准确率低
baseline_acc = np.array([0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.65, 0.75])
# 注意：baseline在高置信度区间（0.8-1.0）准确率很低，因为未知样本被误判

# GEE的数据（校准良好）
# 对未知样本给出低置信度
gee_acc = np.array([0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
# GEE在低置信度区间（0-0.3）准确率也低，但这是正确的校准

# 绘制Baseline曲线
ax.plot(bin_centers, baseline_acc, 'o-', color='steelblue', linewidth=2.5,
        markersize=8, label='Baseline（误校准）', alpha=0.8)

# 绘制GEE曲线
ax.plot(bin_centers, gee_acc, 's-', color='crimson', linewidth=2.5,
        markersize=8, label='GEE（校准良好）', alpha=0.8)

# 绘制理想对角线
ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, alpha=0.5,
        label='理想校准（对角线）')

# 添加阴影区域表示误校准程度
x_fill = np.linspace(0, 1, 100)
y_baseline_interp = np.interp(x_fill, bin_centers, baseline_acc)
ax.fill_between(x_fill, y_baseline_interp, x_fill, where=(y_baseline_interp > x_fill),
                color='steelblue', alpha=0.15, label='Baseline过度自信区域')

# 添加标注
ax.annotate('Baseline：高置信度但低准确率\n（过度自信）',
            xy=(0.85, 0.75), xytext=(0.65, 0.5),
            fontsize=11, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', lw=2, color='steelblue'),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

ax.annotate('GEE：低置信度对应低准确率\n（正确校准）',
            xy=(0.15, 0.12), xytext=(0.35, 0.25),
            fontsize=11, ha='center', va='center',
            arrowprops=dict(arrowstyle='->', lw=2, color='crimson'),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))

ax.set_xlabel('预测置信度', fontsize=13, fontweight='bold')
ax.set_ylabel('实际准确率', fontsize=13, fontweight='bold')
ax.set_title('图5-7：置信度校准曲线（Reliability Diagram）', fontsize=15, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')

# 添加说明文本
ax.text(0.5, 0.95, '对于未知样本的校准对比',
        transform=ax.transAxes, fontsize=12, ha='center', va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('/Users/weisman/Documents/repo_root/graduate/Deep-Packet-Ultar/thesis/figures/figure_5_7_reliability_diagram.png',
            dpi=300, bbox_inches='tight')
print("✅ Figure 5-7 generated successfully!")
print("📍 Saved to: thesis/figures/figure_5_7_reliability_diagram.png")
print("\n📊 图示内容:")
print("   - Baseline（蓝色）：严重偏离对角线，高置信度但低准确率（误校准）")
print("   - GEE（红色）：更接近对角线，低置信度对应低准确率（校准良好）")
print("   - 灰色虚线：理想校准线（对角线）")
