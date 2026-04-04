#!/usr/bin/env python3
"""
生成门控网络权重热力图 - 完全符合论文描述

论文描述(第1012行):
- X轴: 12个维度，前6维(C1-C6)基准模型输出，后6维(C1-C6)专家模型输出
- Y轴: H1-H64隐藏层神经元
- 红色=正权重，蓝色=负权重
- 后6维(专家)颜色普遍深于前6维(基准)
- 类别5和7对应列(第6列和第8列)权重深红或深蓝

说明: 如果实际模型的hidden_dim不是64，则显示实际的神经元数
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def load_and_analyze_model(model_path):
    """加载门控网络模型并分析权重"""
    checkpoint = torch.load(model_path, map_location='cpu')

    # 处理不同的checkpoint格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 获取第一层权重
    if 'network.0.weight' not in state_dict:
        raise ValueError(f"模型中未找到 'network.0.weight'。可用的键: {list(state_dict.keys())}")

    weight = state_dict['network.0.weight']  # shape: [hidden_dim, input_dim]

    print(f"原始权重矩阵: {weight.shape}")
    print(f"  - 隐藏神经元数: {weight.shape[0]}")
    print(f"  - 输入维度: {weight.shape[1]}")

    return weight.numpy()

def plot_weight_heatmap_correct(model_path, output_path_pdf, output_path_png,
                                highlight_minority_classes=True):
    """绘制门控网络第一层权重热力图"""

    # 加载权重
    weight = load_and_analyze_model(model_path)

    # 转置: [hidden_dim, input_dim] -> [input_dim, hidden_dim]
    # 这样X轴是输入维度，Y轴是神经元
    weight = weight.T
    input_dim, hidden_dim = weight.shape

    # 推断类别数
    num_classes = input_dim // 2

    print(f"\n转置后权重矩阵: {weight.shape}")
    print(f"  - X轴(输入维度): {input_dim}")
    print(f"  - Y轴(隐藏神经元): {hidden_dim}")
    print(f"  - 类别数: {num_classes}")

    # 如果不是6类(12输入维度)，需要重新映射
    # 论文中提到的类别5和7是全局标签，在6类任务中可能是类别0和2
    if num_classes == 6:
        # VPN数据集: 全局标签5,6,7,8,9,10 -> 内部标签0,1,2,3,4,5
        # 少数类: 全局5->内部0, 全局7->内部2
        minority_internal_indices = [0, 2]  # 对应论文中的类别5和7
        minority_global_labels = [5, 7]
    elif num_classes == 11:
        # traffic数据集: 11个类别
        minority_internal_indices = [4, 6]  # 假设类别4和6是少数类
        minority_global_labels = [4, 6]
    else:
        minority_internal_indices = []
        minority_global_labels = []

    # 创建X轴标签
    x_labels = []
    for i in range(num_classes):
        x_labels.append(f'C{i+1}')
    x_labels = ['Baseline: ' + label for label in x_labels] + ['Expert: ' + label for label in x_labels]

    # 创建Y轴标签
    y_labels = [f'H{i+1}' for i in range(hidden_dim)]

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 12 if hidden_dim > 20 else 10))

    # 绘制热力图 (红色=正, 蓝色=负, 居中)
    vmax = max(abs(weight.min()), abs(weight.max()))
    im = ax.imshow(weight, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax)

    # 设置坐标轴
    ax.set_xticks(np.arange(input_dim))
    ax.set_yticks(np.arange(hidden_dim))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(y_labels, fontsize=9 if hidden_dim > 20 else 10)

    # 添加分隔线
    ax.axvline(x=num_classes - 0.5, color='black', linewidth=3, linestyle='--')

    # 高亮少数类列
    if highlight_minority_classes and len(minority_internal_indices) > 0:
        for idx in minority_internal_indices:
            # 高亮基准输入中的少数类
            ax.axvline(x=idx - 0.5, color='red', linewidth=1, alpha=0.3, linestyle='-')
            ax.axvline(x=idx + 0.5, color='red', linewidth=1, alpha=0.3, linestyle='-')

            # 高亮专家输入中的少数类
            expert_idx = num_classes + idx
            ax.axvline(x=expert_idx - 0.5, color='red', linewidth=2, alpha=0.5, linestyle='-')
            ax.axvline(x=expert_idx + 0.5, color='red', linewidth=2, alpha=0.5, linestyle='-')

    # 添加标题和标签
    title = '门控网络第一层权重分布热力图\n'
    title += f'(红色=正权重, 蓝色=负权重, 颜色越深权重越大)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.set_xlabel('输入维度 (Input Dimensions)', fontsize=14, fontweight='bold')
    ylabel = f'隐藏层神经元 (Hidden Neurons H1-H{hidden_dim})'
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('权重值 (Weight Value)', fontsize=12, rotation=270, labelpad=20)

    # 添加文本标注
    ax.text(num_classes // 2 - 0.5, hidden_dim + 0.5, '基准模型输出\n(Baseline Model Outputs)',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='blue')

    ax.text(num_classes + num_classes // 2 - 0.5, hidden_dim + 0.5, '专家模型输出\n(Expert Model Outputs)',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

    # 统计分析
    baseline_weights = weight[:num_classes, :]
    expert_weights = weight[num_classes:, :]

    baseline_abs_mean = np.abs(baseline_weights).mean()
    expert_abs_mean = np.abs(expert_weights).mean()

    # 分析少数类列的权重
    stats_text = f'统计分析:\n'
    stats_text += f'基准模型输入权重绝对值平均: {baseline_abs_mean:.4f}\n'
    stats_text += f'专家模型输入权重绝对值平均: {expert_abs_mean:.4f}\n'
    stats_text += f'专家/基准权重比: {expert_abs_mean/(baseline_abs_mean+1e-8):.2f}x\n'

    if len(minority_internal_indices) > 0:
        stats_text += f'\n少数类(全局标签{minority_global_labels})权重:\n'
        for i, idx in enumerate(minority_internal_indices):
            global_label = minority_global_labels[i]
            baseline_col = weight[idx, :]
            expert_col = weight[num_classes + idx, :]
            stats_text += f'  类别{global_label}: 基准={np.abs(baseline_col).mean():.3f}, '
            stats_text += f'专家={np.abs(expert_col).mean():.3f}\n'

    # 添加统计文本框
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_path_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
    print(f"\n✓ 热力图已保存:")
    print(f"  PDF: {output_path_pdf}")
    print(f"  PNG: {output_path_png}")

    plt.close()

    # 打印详细统计
    print(f"\n详细统计:")
    print(f"  基准模型权重: abs_mean={baseline_abs_mean:.4f}, max={np.abs(baseline_weights).max():.4f}")
    print(f"  专家模型权重: abs_mean={expert_abs_mean:.4f}, max={np.abs(expert_weights).max():.4f}")
    print(f"  专家/基准比: {expert_abs_mean/(baseline_abs_mean+1e-8):.2f}x")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='生成门控网络权重热力图')
    parser.add_argument('--model', type=str,
                       default='model/exp_traffic/incremental/gee_resnet/gating_network_resnet.pt',
                       help='门控网络模型路径')
    parser.add_argument('--output-pdf', type=str,
                       default='thesis/figures/figure_5_1_weight_heatmap.pdf',
                       help='输出PDF路径')
    parser.add_argument('--output-png', type=str,
                       default='thesis/figures/figure_5_1_weight_heatmap.png',
                       help='输出PNG路径')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)

    # 生成热力图
    plot_weight_heatmap_correct(args.model, args.output_pdf, args.output_png)

    print("\n完成!")
