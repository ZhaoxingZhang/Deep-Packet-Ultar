#!/usr/bin/env python3
"""
生成门控网络权重热力图 - 符合论文描述
论文描述: X轴12维(前6维基准模型C1-C6, 后6维专家模型C1-C6), Y轴H1-H64神经元
红色表示正权重,蓝色表示负权重
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

def load_gating_model(model_path):
    """加载门控网络模型"""
    checkpoint = torch.load(model_path, map_location='cpu')

    # 处理不同的checkpoint格式
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    return state_dict

def plot_weight_heatmap(model_path, output_path_pdf, output_path_png):
    """绘制门控网络第一层权重热力图"""

    # 加载模型权重
    state_dict = load_gating_model(model_path)

    # 获取第一层权重
    if 'network.0.weight' not in state_dict:
        print("错误: 模型中未找到 'network.0.weight'")
        print(f"可用的键: {list(state_dict.keys())}")
        return

    weight = state_dict['network.0.weight']  # shape: [hidden_dim, input_dim]

    print(f"原始权重矩阵形状: {weight.shape}")
    print(f"  - Y轴(隐藏神经元): {weight.shape[0]}")
    print(f"  - X轴(输入维度): {weight.shape[1]}")

    # 转置为 [input_dim, hidden_dim] 以便 X轴是输入维度, Y轴是神经元
    weight = weight.T.numpy()  # shape: [input_dim, hidden_dim]

    input_dim, hidden_dim = weight.shape
    print(f"\n转置后权重矩阵形状: {weight.shape}")
    print(f"  - X轴(输入维度): {input_dim}")
    print(f"  - Y轴(隐藏神经元): {hidden_dim}")

    # 如果输入维度是12，说明是6类任务 (2 * 6 = 12)
    if input_dim == 12:
        num_classes = 6
    elif input_dim == 24:
        num_classes = 12
    else:
        num_classes = input_dim // 2

    print(f"\n检测到的类别数: {num_classes}")

    # 创建X轴标签
    x_labels = []
    for i in range(num_classes):
        x_labels.append(f'C{i+1}')
    x_labels = ['Baseline: ' + label for label in x_labels] + ['Expert: ' + label for label in x_labels]

    # 创建Y轴标签
    y_labels = [f'H{i+1}' for i in range(hidden_dim)]

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))

    # 绘制热力图 (红色=正, 蓝色=负)
    im = ax.imshow(weight, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(weight).max(), vmax=np.abs(weight).max())

    # 设置坐标轴
    ax.set_xticks(np.arange(input_dim))
    ax.set_yticks(np.arange(hidden_dim))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(y_labels, fontsize=9)

    # 添加分隔线区分基准模型输入和专家模型输入
    ax.axvline(x=num_classes - 0.5, color='black', linewidth=3, linestyle='--')

    # 添加标题和标签
    ax.set_xlabel('Input Dimensions', fontsize=14, fontweight='bold')
    ax.set_ylabel('Hidden Neurons', fontsize=14, fontweight='bold')
    ax.set_title('Gating Network First-Layer Weight Distribution\n(Red=Positive Weight, Blue=Negative Weight)',
                 fontsize=16, fontweight='bold', pad=20)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Weight Value', fontsize=12, rotation=270, labelpad=20)

    # 添加文本标注说明
    ax.text(num_classes // 2 - 0.5, hidden_dim + 0.5, 'Baseline Model Outputs',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='blue')
    ax.text(num_classes + num_classes // 2 - 0.5, hidden_dim + 0.5, 'Expert Model Outputs',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='red')

    # 统计分析
    baseline_weights = weight[:num_classes, :]
    expert_weights = weight[num_classes:, :]

    baseline_abs_mean = np.abs(baseline_weights).mean()
    expert_abs_mean = np.abs(expert_weights).mean()

    print(f"\n统计分析:")
    print(f"  基准模型输入权重绝对值平均: {baseline_abs_mean:.4f}")
    print(f"  专家模型输入权重绝对值平均: {expert_abs_mean:.4f}")
    print(f"  专家/基准权重比: {expert_abs_mean/(baseline_abs_mean+1e-8):.2f}x")

    # 在图上添加统计信息
    stats_text = f'Statistics:\n'
    stats_text += f'Baseline Input Abs Mean: {baseline_abs_mean:.3f}\n'
    stats_text += f'Expert Input Abs Mean: {expert_abs_mean:.3f}\n'
    stats_text += f'Expert/Baseline Ratio: {expert_abs_mean/(baseline_abs_mean+1e-8):.2f}x'

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # 保存图像
    plt.savefig(output_path_pdf, bbox_inches='tight', dpi=300)
    plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
    print(f"\n热力图已保存到:")
    print(f"  - {output_path_pdf}")
    print(f"  - {output_path_png}")

    plt.close()

if __name__ == "__main__":
    # 模型路径 - 使用增量学习实验的门控网络
    # 该模型来自 exp_traffic/incremental/gee_resnet 实验
    # 包含6个类别(5, 6, 7, 8, 9, 10)，其中5和7是少数类
    model_path = os.path.join(project_root, "model/exp_traffic/incremental/gee_resnet/gating_network_resnet.pt")

    # 输出路径
    output_dir = os.path.join(project_root, "thesis/figures")
    os.makedirs(output_dir, exist_ok=True)

    output_pdf = os.path.join(output_dir, "figure_5_1_weight_heatmap.pdf")
    output_png = os.path.join(output_dir, "figure_5_1_weight_heatmap.png")

    # 生成热力图
    plot_weight_heatmap(model_path, output_pdf, output_png)

    print("\n完成!")
