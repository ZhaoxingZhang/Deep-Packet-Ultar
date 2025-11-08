#!/usr/bin/env python3
"""
自动化专家系统测试脚本

用于验证auto_expert_system.py的核心功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_expert_system import AutoExpertSystem, AnalysisResults

def test_with_sample_data():
    """使用示例数据测试自动化专家系统"""

    print("🧪 测试自动化专家系统...")

    # 模拟类别分布数据 (基于之前的VPN数据集)
    sample_distribution = {
        5: 215,   # VPN: Chat (少数类)
        6: 1034,  # VPN: File Transfer
        7: 65,    # VPN: Email (少数类)
        8: 4408,  # VPN: Streaming
        9: 1089,  # VPN: Torrent
        10: 9476  # VPN: Voip
    }

    # 创建系统实例
    system = AutoExpertSystem("processed_data/vpn", "test_results")

    print("\n📊 测试数据分析...")
    stats = system.calculate_distribution_statistics(sample_distribution)

    print("\n🎯 测试少数类识别...")
    minority_classes = system.identify_minority_classes(sample_distribution, stats)

    print("\n🧠 测试专家策略设计...")
    strategy = system.design_expert_strategy(sample_distribution, minority_classes)

    print("\n✅ 测试完成!")
    print(f"识别的少数类: {minority_classes}")
    print(f"专家策略类型: {strategy['type']}")
    print(f"专家数量: {len(strategy['experts'])}")

    for expert in strategy['experts']:
        print(f"  - {expert.name}: {expert.description}")
        print(f"    目标类别: {expert.target_classes}")

    return True

def test_system_integration():
    """测试系统集成 (需要实际数据)"""

    print("\n🔧 测试系统集成...")

    # 检查数据目录是否存在
    data_dir = "processed_data/vpn"
    if not os.path.exists(data_dir):
        print(f"⚠️  数据目录不存在: {data_dir}")
        print("请先运行数据预处理脚本")
        return False

    try:
        # 创建系统实例
        system = AutoExpertSystem(data_dir, "integration_test_results")

        # 只测试分析部分，不进行实际训练
        print("📊 执行数据分析...")
        class_distribution = system.analyze_class_distribution()
        stats = system.calculate_distribution_statistics(class_distribution)
        minority_classes = system.identify_minority_classes(class_distribution, stats)
        strategy = system.design_expert_strategy(class_distribution, minority_classes)

        print("✅ 系统集成测试通过!")
        return True

    except Exception as e:
        print(f"❌ 系统集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("自动化专家系统测试")
    print("="*60)

    # 测试1: 示例数据测试
    test1_passed = test_with_sample_data()

    # 测试2: 系统集成测试
    test2_passed = test_system_integration()

    print("\n" + "="*60)
    print("测试结果汇总:")
    print(f"示例数据测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"系统集成测试: {'✅ 通过' if test2_passed else '❌ 失败'}")

    if test1_passed:
        print("\n🎯 核心算法验证成功!")
        print("可以尝试运行完整的自动化流程:")
        print("python auto_expert_system.py --data_source processed_data/vpn")

    print("="*60)

if __name__ == "__main__":
    main()