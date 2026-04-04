#!/usr/bin/env python3
"""
检查门控网络模型的权重矩阵形状
"""
import torch
import sys
import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ml.model import GatingNetwork

# 检查几个模型文件
model_files = [
    "model/gating_network.pth",
    "model/gating_network_cross_entropy_loss_1115.pth",
]

for model_file in model_files:
    if not os.path.exists(model_file):
        print(f"跳过不存在的文件: {model_file}")
        continue

    print(f"\n{'='*60}")
    print(f"检查模型: {model_file}")
    print('='*60)

    # 加载模型权重
    checkpoint = torch.load(model_file, map_location='cpu')

    # 打印keys
    print("Keys in checkpoint:")
    for key in checkpoint.keys():
        print(f"  - {key}")

    # 查找第一层权重
    if 'network.0.weight' in checkpoint:
        first_layer_weight = checkpoint['network.0.weight']
        print(f"\n第一层权重矩阵形状: {first_layer_weight.shape}")
        print(f"  - Y轴(神经元数): {first_layer_weight.shape[0]}")
        print(f"  - X轴(输入维度): {first_layer_weight.shape[1]}")
        print(f"\n权重范围: [{first_layer_weight.min().item():.4f}, {first_layer_weight.max().item():.4f}]")
        print(f"权重绝对值平均: {first_layer_weight.abs().mean().item():.4f}")

        # 分析前6维和后6维的权重差异
        if first_layer_weight.shape[1] == 12:
            baseline_weights = first_layer_weight[:, :6]
            expert_weights = first_layer_weight[:, 6:]

            print(f"\n基准模型输入权重统计(前6维):")
            print(f"  绝对值平均: {baseline_weights.abs().mean().item():.4f}")
            print(f"  绝对值最大: {baseline_weights.abs().max().item():.4f}")

            print(f"\n专家模型输入权重统计(后6维):")
            print(f"  绝对值平均: {expert_weights.abs().mean().item():.4f}")
            print(f"  绝对值最大: {expert_weights.abs().max().item():.4f}")

            ratio = expert_weights.abs().mean() / (baseline_weights.abs().mean() + 1e-8)
            print(f"\n专家/基准权重比: {ratio.item():.2f}x")
    else:
        print("未找到 'network.0.weight'")
        print("可用的键:", list(checkpoint.keys()))

print("\n" + "="*60)
print("检查GatingNetwork类定义的默认参数")
print("="*60)

# 创建一个示例模型来查看默认参数
test_model = GatingNetwork(num_classes=6, use_garbage_class=False)
print(f"\nnum_classes=6时的网络结构:")
for name, param in test_model.named_parameters():
    print(f"  {name}: shape={param.shape}")

print(f"\n总参数数: {sum(p.numel() for p in test_model.parameters())}")
