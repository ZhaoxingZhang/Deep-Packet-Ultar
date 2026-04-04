#!/usr/bin/env python3
"""
GEE统一理论框架示意图生成
基于论文5.5.1节的完整理论框架
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(16, 13))
ax.set_xlim(0, 16)
ax.set_ylim(-0.3, 12.5)
ax.axis('off')

# 颜色方案
colors = {
    'optimization': '#2E86AB',  # 蓝色
    'architecture': '#A23B72',  # 紫红色
    'result': '#F18F01',        # 橙色
    'light_blue': '#E8F4F8',
    'light_purple': '#F8E8F0',
    'light_orange': '#FFF0E8',
    'text': '#2C3E50',
    'arrow': '#34495E'
}

# ============ 第一层：优化层（顶部）============
opt_box = FancyBboxPatch((0.5, 9.5), 15, 2.2,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['light_blue'],
                          edgecolor=colors['optimization'],
                          linewidth=2.5)
ax.add_patch(opt_box)

ax.text(8, 11.45, '优化层 (Optimization Layer)',
        fontsize=16, fontweight='bold', ha='center',
        color=colors['optimization'])

# 加权损失模块
weighted_box = FancyBboxPatch((1.5, 9.8), 4.5, 1.2,
                               boxstyle="round,pad=0.08",
                               facecolor='white',
                               edgecolor=colors['optimization'],
                               linewidth=1.8)
ax.add_patch(weighted_box)
ax.text(3.75, 10.6, '加权交叉熵损失',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=colors['text'])
ax.text(3.75, 10.2, r'$L_{weighted}(\theta_{gate})$',
        fontsize=11, ha='center', va='center',
        color=colors['text'], style='italic')

# 梯度平衡模块
gradient_box = FancyBboxPatch((6.5, 9.8), 4.5, 1.2,
                               boxstyle="round,pad=0.08",
                               facecolor='white',
                               edgecolor=colors['optimization'],
                               linewidth=1.8)
ax.add_patch(gradient_box)
ax.text(8.75, 10.6, '梯度平衡机制',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=colors['text'])
ax.text(8.75, 10.2, r'$\nabla_\theta L \propto w_i$',
        fontsize=11, ha='center', va='center',
        color=colors['text'], style='italic')

# 权重公式说明
ax.text(13.5, 10.6, '权重策略:',
        fontsize=11, ha='center', color=colors['text'])
ax.text(13.5, 10.2, r'$w_i \propto 1/n_i$',
        fontsize=11, ha='center', va='center',
        color=colors['text'], style='italic')

# 箭头：加权损失→梯度平衡
arrow1 = FancyArrowPatch((6.05, 10.4), (6.45, 10.4),
                          arrowstyle='->', mutation_scale=20,
                          linewidth=2.5, color=colors['arrow'])
ax.add_patch(arrow1)

# ============ 第二层：架构层（中部）============
arch_box = FancyBboxPatch((0.5, 6.2), 15, 2.8,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['light_purple'],
                          edgecolor=colors['architecture'],
                          linewidth=2.5)
ax.add_patch(arch_box)

ax.text(8, 8.8, '架构层 (Architecture Layer)',
        fontsize=16, fontweight='bold', ha='center',
        color=colors['architecture'])

# 门控网络模块
gating_box = FancyBboxPatch((1.5, 7.2), 4.5, 1.2,
                             boxstyle="round,pad=0.08",
                             facecolor='white',
                             edgecolor=colors['architecture'],
                             linewidth=1.8)
ax.add_patch(gating_box)
ax.text(3.75, 8.0, '门控网络',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=colors['text'])
ax.text(3.75, 7.6, r'$g(x) = \sigma(W_{gate} \cdot [f_{base}; f_{exp}] + b)$',
        fontsize=9, ha='center', va='center',
        color=colors['text'], style='italic')

# 自适应融合模块
fusion_box = FancyBboxPatch((6.5, 7.2), 4.5, 1.2,
                             boxstyle="round,pad=0.08",
                             facecolor='white',
                             edgecolor=colors['architecture'],
                             linewidth=1.8)
ax.add_patch(fusion_box)
ax.text(8.75, 8.0, '自适应融合',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=colors['text'])
ax.text(8.75, 7.6, r'$f_{GEE} = g \odot f_{base} + (1-g) \odot f_{exp}$',
        fontsize=9, ha='center', va='center',
        color=colors['text'], style='italic')

# 智能决策说明
decision_box = FancyBboxPatch((11.5, 7.0), 3.5, 1.6,
                               boxstyle="round,pad=0.08",
                               facecolor='white',
                               edgecolor=colors['architecture'],
                               linewidth=1.5,
                               linestyle='--')
ax.add_patch(decision_box)
ax.text(13.25, 8.25, '数据驱动的',
        fontsize=10, ha='center', color=colors['text'])
ax.text(13.25, 7.9, '智能决策',
        fontsize=10, fontweight='bold', ha='center', color=colors['text'])
ax.text(13.25, 7.5, '多数类→基准',
        fontsize=9, ha='center', color=colors['text'])
ax.text(13.25, 7.15, '少数类→专家',
        fontsize=9, ha='center', color=colors['text'])

# 箭头：门控网络→自适应融合
arrow2 = FancyArrowPatch((6.05, 7.8), (6.45, 7.8),
                          arrowstyle='->', mutation_scale=20,
                          linewidth=2.5, color=colors['arrow'])
ax.add_patch(arrow2)

# 竖向箭头：优化层→架构层
arrow3 = FancyArrowPatch((8, 9.5), (8, 9.05),
                          arrowstyle='->', mutation_scale=25,
                          linewidth=3, color=colors['arrow'])
ax.add_patch(arrow3)
ax.text(8.8, 9.25, '必要条件',
        fontsize=10, ha='left', color=colors['text'],
        style='italic', bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='none'))

# ============ 第三层：结果层（底部）============
result_box = FancyBboxPatch((0.5, 2.5), 15, 3.2,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['light_orange'],
                            edgecolor=colors['result'],
                            linewidth=2.5)
ax.add_patch(result_box)

ax.text(8, 5.5, '结果层 (Result Layer) - 协同效应',
        fontsize=16, fontweight='bold', ha='center',
        color=colors['result'])

# 少数类识别结果
minority_box = FancyBboxPatch((1.5, 3.2), 6, 1.8,
                               boxstyle="round,pad=0.1",
                               facecolor='white',
                               edgecolor=colors['result'],
                               linewidth=2)
ax.add_patch(minority_box)
ax.text(4.5, 4.75, '少数类识别性能提升',
        fontsize=13, fontweight='bold', ha='center', color=colors['text'])
ax.text(4.5, 4.35, r'Macro-F1: $0.63 \rightarrow 0.83$',
        fontsize=12, ha='center', color=colors['text'], style='italic')
ax.text(4.5, 4.0, r'少数类F1: $0 \rightarrow 0.92$',
        fontsize=12, ha='center', color=colors['text'], style='italic')
ax.text(4.5, 3.6, '提升幅度: +31.7%',
        fontsize=11, ha='center', color='#27AE60',
        fontweight='bold')

# OSR结果
osr_box = FancyBboxPatch((8.5, 3.2), 6, 1.8,
                         boxstyle="round,pad=0.1",
                         facecolor='white',
                         edgecolor=colors['result'],
                         linewidth=2)
ax.add_patch(osr_box)
ax.text(11.5, 4.75, '开放集识别性能提升',
        fontsize=13, fontweight='bold', ha='center', color=colors['text'])
ax.text(11.5, 4.35, r'FPR@TPR95: $0.47 \rightarrow 0.03$',
        fontsize=12, ha='center', color=colors['text'], style='italic')
ax.text(11.5, 4.0, r'Softmax熵: $0.35 \rightarrow 1.18$',
        fontsize=12, ha='center', color=colors['text'], style='italic')
ax.text(11.5, 3.6, '误报率降低: -93.6%',
        fontsize=11, ha='center', color='#27AE60',
        fontweight='bold')

# 竖向箭头：架构层→结果层
arrow4 = FancyArrowPatch((8, 6.2), (8, 5.75),
                          arrowstyle='->', mutation_scale=25,
                          linewidth=3, color=colors['arrow'])
ax.add_patch(arrow4)

# ============ 底部：统一目标函数 ============
unified_box = FancyBboxPatch((2, 0.3), 12, 1.5,
                             boxstyle="round,pad=0.15",
                             facecolor='#FDFEFE',
                             edgecolor=colors['text'],
                             linewidth=2)
ax.add_patch(unified_box)

ax.text(8, 1.45, '统一优化目标',
        fontsize=14, fontweight='bold', ha='center',
        color=colors['text'])
ax.text(8, 0.95, r'$L_{GEE} = L_{weighted}(\theta_{gate}) + \lambda \cdot L_{fusion}(\theta_{gate})$',
        fontsize=13, ha='center', va='center',
        color=colors['text'], style='italic',
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor=colors['light_blue'],
                  edgecolor=colors['optimization'],
                  linewidth=1.5))

# 竖向箭头：结果层→统一目标
arrow5 = FancyArrowPatch((8, 2.5), (8, 1.85),
                          arrowstyle='<->', mutation_scale=20,
                          linewidth=2.5, color=colors['arrow'],
                          linestyle='--')
ax.add_patch(arrow5)
ax.text(9, 2.15, '理论支撑',
        fontsize=9, ha='left', color=colors['text'],
        style='italic',
        bbox=dict(boxstyle='round,pad=0.2',
                  facecolor='white', edgecolor='none', alpha=0.8))

# ============ 标题和说明 ============
ax.text(8, 12.15, 'GEE (Gated Expert Ensemble) 统一理论框架',
        fontsize=18, fontweight='bold', ha='center',
        color=colors['text'],
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='white',
                  edgecolor='none',
                  alpha=1.0))

# 图例
legend_y = -0.1
ax.text(1, legend_y, '■', fontsize=12, color=colors['optimization'],
        ha='left', va='center')
ax.text(1.3, legend_y, '优化机制', fontsize=10, ha='left',
        va='center', color=colors['text'],
        bbox=dict(boxstyle='round,pad=0.2',
                  facecolor='white',
                  edgecolor='none',
                  alpha=1.0))

ax.text(2.5, legend_y, '■', fontsize=12, color=colors['architecture'],
        ha='left', va='center')
ax.text(2.8, legend_y, '架构设计', fontsize=10, ha='left',
        va='center', color=colors['text'],
        bbox=dict(boxstyle='round,pad=0.2',
                  facecolor='white',
                  edgecolor='none',
                  alpha=1.0))

ax.text(4, legend_y, '■', fontsize=12, color=colors['result'],
        ha='left', va='center')
ax.text(4.3, legend_y, '性能提升', fontsize=10, ha='left',
        va='center', color=colors['text'],
        bbox=dict(boxstyle='round,pad=0.2',
                  facecolor='white',
                  edgecolor='none',
                  alpha=1.0))

plt.tight_layout()
plt.savefig('/Users/weisman/Documents/repo_root/graduate/Deep-Packet-Ultar/thesis/figures/figure_5_8_unified_framework.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Figure 5-8 generated successfully!")
print("📍 Saved to: thesis/figures/figure_5_8_unified_framework.png")
print("\n📊 框架包含三个层次:")
print("   1. 优化层: 加权交叉熵损失 → 梯度平衡")
print("   2. 架构层: 门控网络 → 自适应融合")
print("   3. 结果层: 少数类识别↑ + 开放集识别↑")
print("\n🎯 核心创新: 梯度平衡 + 自适应融合的协同作用")
