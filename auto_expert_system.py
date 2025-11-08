#!/usr/bin/env python3
"""
自动化专家选择系统 (Auto Expert Selection System)

该系统能够自动分析数据集分布，识别少数类，设计最优专家策略，
并完成端到端的训练与评估流程。

作者: Deep-Packet Research Team
日期: 2025年11月7日
"""

import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, ArrayType, LongType, DoubleType

# 导入项目模块
from utils import ID_TO_TRAFFIC

@dataclass
class ExpertConfig:
    """专家配置数据类"""
    name: str
    target_classes: List[int]
    description: str
    dataset_name: str
    model_path: Optional[str] = None

@dataclass
class AnalysisResults:
    """数据分析结果"""
    class_distribution: Dict[int, int]
    total_samples: int
    total_classes: int
    statistics: Dict[str, float]
    minority_classes: List[int]
    expert_strategy: Dict

class AutoExpertSystem:
    """自动化专家选择系统主类"""

    def __init__(self, data_source_dir: str, output_base_dir: str = "auto_expert_results"):
        """
        初始化自动化专家系统

        Args:
            data_source_dir: 预处理数据源目录
            output_base_dir: 输出基础目录
        """
        self.data_source_dir = data_source_dir
        self.output_base_dir = output_base_dir
        self.spark = None

        # 创建输出目录
        os.makedirs(output_base_dir, exist_ok=True)

        # 系统配置
        self.config = {
            'minority_detection_strategies': ['magnitude', 'logarithmic', 'tier'],
            'magnitude_ratio': 5,  # 数量级差异倍数 (5倍差异)
            'log_std_threshold': 0.75,  # 对数标准差阈值 (适中标准)
            'tier_ratio': 3,  # 梯队差异倍数 (3倍差异，更严格)
            'max_minority_experts': 3,  # 最大少数类专家数
            'min_expert_samples': 50  # 专家训练最小样本数
        }

    def _init_spark(self):
        """初始化Spark会话"""
        if self.spark is None:
            os.environ["PYSPARK_PYTHON"] = sys.executable
            os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
            self.spark = (
                SparkSession.builder.master("local[*]")
                .config("spark.driver.host", "127.0.0.1")
                .getOrCreate()
            )

    def analyze_class_distribution(self) -> Dict[int, int]:
        """
        分析数据集类别分布

        Returns:
            类别分布字典 {class_id: sample_count}
        """
        print("🔍 分析数据集类别分布...")

        self._init_spark()

        schema = StructType([
            StructField("app_label", LongType(), True),
            StructField("traffic_label", LongType(), True),
            StructField("feature", ArrayType(DoubleType()), True),
        ])

        df = self.spark.read.schema(schema).json(f"{self.data_source_dir}/*.json.gz")

        # 统计traffic_label分布
        traffic_label_counts = (df.filter(col("traffic_label").isNotNull())
                              .groupBy("traffic_label")
                              .count()
                              .orderBy("traffic_label")
                              .toPandas())

        # 转换为字典
        class_distribution = dict(zip(traffic_label_counts['traffic_label'],
                                    traffic_label_counts['count']))

        print(f"✅ 发现 {len(class_distribution)} 个类别，总计 {sum(class_distribution.values())} 个样本")

        # 打印详细分布
        print("\n📊 类别分布详情:")
        for class_id, count in sorted(class_distribution.items()):
            class_name = ID_TO_TRAFFIC.get(class_id, f"Unknown-{class_id}")
            print(f"   类别 {class_id} ({class_name}): {count:,} 样本")

        return class_distribution

    def calculate_distribution_statistics(self, class_distribution: Dict[int, int]) -> Dict[str, float]:
        """
        计算分布统计特征

        Args:
            class_distribution: 类别分布字典

        Returns:
            统计特征字典
        """
        counts = list(class_distribution.values())

        stats = {
            'mean': np.mean(counts),
            'std': np.std(counts),
            'median': np.median(counts),
            'min': np.min(counts),
            'max': np.max(counts),
            'q25': np.percentile(counts, 25),
            'q75': np.percentile(counts, 75),
            'iqr': np.percentile(counts, 75) - np.percentile(counts, 25),
            'cv': np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0  # 变异系数
        }

        print(f"\n📈 分布统计特征:")
        print(f"   平均样本数: {stats['mean']:,.0f}")
        print(f"   中位数: {stats['median']:,.0f}")
        print(f"   标准差: {stats['std']:,.0f}")
        print(f"   最小值: {stats['min']:,}")
        print(f"   最大值: {stats['max']:,}")
        print(f"   变异系数: {stats['cv']:.3f}")

        return stats

    def identify_minority_classes(self, class_distribution: Dict[int, int],
                                stats: Dict[str, float]) -> List[int]:
        """
        基于长尾分布特征的多策略少数类识别

        Args:
            class_distribution: 类别分布
            stats: 统计特征 (用于兼容性，实际不使用)

        Returns:
            少数类ID列表
        """
        print("\n🎯 基于长尾分布特征的多策略少数类识别...")

        minority_candidates = set()
        strategy_results = {}

        # 策略1: 数量级差异检测 (5倍差异原则)
        if 'magnitude' in self.config['minority_detection_strategies']:
            max_count = max(class_distribution.values())
            magnitude_threshold = max_count / self.config['magnitude_ratio']  # 5倍差异
            magnitude_minority = [cls for cls, count in class_distribution.items()
                                 if count < magnitude_threshold]
            minority_candidates.update(magnitude_minority)
            strategy_results['magnitude'] = magnitude_minority
            print(f"   数量级差异检测 (<{magnitude_threshold:,.0f}, 1/{self.config['magnitude_ratio']}倍): {magnitude_minority}")

        # 策略2: 对数尺度检测 (基于对数标准差)
        if 'logarithmic' in self.config['minority_detection_strategies']:
            log_counts = np.log10(list(class_distribution.values()))
            log_mean = np.mean(log_counts)
            log_std = np.std(log_counts)
            log_threshold = log_mean - self.config['log_std_threshold']  # 低于均值1个标准差
            log_minority = [cls for cls, count in class_distribution.items()
                           if np.log10(count) < log_threshold]
            minority_candidates.update(log_minority)
            strategy_results['logarithmic'] = log_minority
            print(f"   对数尺度检测 (log<{log_threshold:.2f}): {log_minority}")

        # 策略3: 梯队差异检测 (相邻类别3倍差异)
        if 'tier' in self.config['minority_detection_strategies']:
            tier_minority = self._detect_tier_gaps(class_distribution)
            minority_candidates.update(tier_minority)
            strategy_results['tier'] = tier_minority
            print(f"   梯队差异检测 (>={self.config['tier_ratio']}倍差异): {tier_minority}")

        # 统计各策略的交集和并集
        all_strategies = list(strategy_results.values())
        if all_strategies:
            union = set().union(*all_strategies)

            # 计算交集（用于分析）
            if len(all_strategies) > 1:
                intersection = set(all_strategies[0]).intersection(*all_strategies[1:])
            else:
                intersection = set(all_strategies[0])
        else:
            union = set()
            intersection = set()

        print(f"\n📋 策略汇总:")
        print(f"   各策略交集: {list(intersection)}")
        print(f"   各策略并集: {list(union)}")

        # 决策逻辑：至少被2个策略识别的类别才被认为是真正的少数类
        final_minority = []
        if union:
            for cls in union:
                vote_count = sum(1 for strategy_classes in strategy_results.values() if cls in strategy_classes)
                if vote_count >= 2:  # 至少被2个策略识别
                    final_minority.append(cls)

        # 如果没有类别被多个策略识别，说明数据集中没有明显的少数类
        if not final_minority:
            print("   📌 结论: 没有类别被多个策略一致识别为少数类")
            print("   📌 建议: 可能需要调整策略参数或使用单一基准模型")
            return []

        # 按样本数量排序
        final_minority.sort(key=lambda x: class_distribution[x])

        print(f"🎉 最终确定的少数类: {final_minority}")
        for cls in final_minority:
            class_name = ID_TO_TRAFFIC.get(cls, f"Unknown-{cls}")
            print(f"   类别 {cls} ({class_name}): {class_distribution[cls]:,} 样本")

            # 显示每个类别的投票情况
            votes = []
            for strategy_name, strategy_classes in strategy_results.items():
                if cls in strategy_classes:
                    votes.append(strategy_name)
            print(f"     支持策略: {', '.join(votes)}")

        return final_minority

    def _detect_tier_gaps(self, class_distribution: Dict[int, int]) -> List[int]:
        """
        检测梯队差异 - 识别相邻类别间的显著差异

        Args:
            class_distribution: 类别分布字典

        Returns:
            被识别为少数类的类别列表
        """
        # 按样本数量排序
        sorted_items = sorted(class_distribution.items(), key=lambda x: x[1])

        tier_minority = []

        # 检查所有相邻类别之间的差异，收集所有可能的少数类
        for i in range(1, len(sorted_items)):
            prev_cls, prev_count = sorted_items[i-1]
            curr_cls, curr_count = sorted_items[i]

            # 如果发现显著差异
            if curr_count >= prev_count * self.config['tier_ratio']:
                # 前面的类别可能是少数类
                tier_minority.append(prev_cls)
                print(f"     发现梯队差异: 类别{prev_cls}({prev_count:,}) -> 类别{curr_cls}({curr_count:,})")

        # 如果没有发现梯队差异，检查最小的类别是否明显小于其他类别
        if not tier_minority and len(sorted_items) >= 2:
            smallest_cls, smallest_count = sorted_items[0]
            second_smallest_cls, second_smallest_count = sorted_items[1]

            # 如果最小类别明显小于第二小类别
            if second_smallest_count >= smallest_count * self.config['tier_ratio']:
                tier_minority.append(smallest_cls)
                print(f"     最小类别显著较小: 类别{smallest_cls}({smallest_count:,})")

        return sorted(list(set(tier_minority)))  # 去重并排序

    def design_expert_strategy(self, class_distribution: Dict[int, int],
                             minority_classes: List[int]) -> Dict:
        """
        设计专家策略

        Args:
            class_distribution: 类别分布
            minority_classes: 少数类列表

        Returns:
            专家策略配置
        """
        print("\n🧠 设计专家策略...")

        total_classes = len(class_distribution)
        minority_count = len(minority_classes)

        # 策略决策树
        if minority_count == 0:
            print("   📌 策略: 无明显少数类，使用单一基准模型")
            return {
                'type': 'single',
                'experts': [
                    ExpertConfig(
                        name='baseline',
                        target_classes=list(class_distribution.keys()),
                        description='基准模型，处理所有类别',
                        dataset_name='baseline_all_classes'
                    )
                ]
            }

        elif minority_count <= 2:
            print(f"   📌 策略: 少量少数类({minority_count}个)，使用基准模型 + 单一少数类专家")
            return {
                'type': 'baseline_plus_minority',
                'experts': [
                    ExpertConfig(
                        name='baseline',
                        target_classes=list(class_distribution.keys()),
                        description='基准模型，处理所有类别',
                        dataset_name='baseline_all_classes'
                    ),
                    ExpertConfig(
                        name='minority_expert',
                        target_classes=minority_classes,
                        description=f'少数类专家，专门处理类别 {minority_classes}',
                        dataset_name='minority_expert'
                    )
                ]
            }

        elif minority_count <= total_classes // 2:
            print(f"   📌 策略: 中等数量少数类({minority_count}个)，使用基准模型 + 分组少数类专家")

            # 对少数类进行聚类分组
            expert_groups = self._cluster_minority_classes(minority_classes, class_distribution)

            experts = [
                ExpertConfig(
                    name='baseline',
                    target_classes=list(class_distribution.keys()),
                    description='基准模型，处理所有类别',
                    dataset_name='baseline_all_classes'
                )
            ]

            for i, group in enumerate(expert_groups):
                experts.append(
                    ExpertConfig(
                        name=f'minority_expert_{i+1}',
                        target_classes=group,
                        description=f'少数类专家{i+1}，处理类别 {group}',
                        dataset_name=f'minority_expert_{i+1}'
                    )
                )

            return {
                'type': 'baseline_plus_grouped',
                'experts': experts
            }

        else:
            print(f"   📌 策略: 大量少数类({minority_count}个)，使用多层专家架构")
            return self._design_hierarchical_experts(class_distribution, minority_classes)

    def _cluster_minority_classes(self, minority_classes: List[int],
                                class_distribution: Dict[int, int]) -> List[List[int]]:
        """
        对少数类进行聚类分组

        Args:
            minority_classes: 少数类列表
            class_distribution: 类别分布

        Returns:
            分组后的少数类列表
        """
        minority_counts = {cls: class_distribution[cls] for cls in minority_classes}
        sorted_classes = sorted(minority_counts.items(), key=lambda x: x[1])

        # 动态确定聚类数量
        if len(sorted_classes) <= 3:
            return [[cls for cls, _ in sorted_classes]]
        else:
            # 按数量相似性分组
            groups = []
            current_group = []
            current_count = None

            for cls, count in sorted_classes:
                if current_count is None or abs(count - current_count) / max(current_count, 1) < 0.5:
                    current_group.append(cls)
                    current_count = count
                else:
                    groups.append(current_group)
                    current_group = [cls]
                    current_count = count

            if current_group:
                groups.append(current_group)

            print(f"   🔀 少数类分组结果: {groups}")
            return groups

    def _design_hierarchical_experts(self, class_distribution: Dict[int, int],
                                   minority_classes: List[int]) -> Dict:
        """
        设计多层专家架构

        Args:
            class_distribution: 类别分布
            minority_classes: 少数类列表

        Returns:
            多层专家策略配置
        """
        # 按样本数量将少数类分为3组
        minority_counts = {cls: class_distribution[cls] for cls in minority_classes}
        sorted_classes = sorted(minority_counts.items(), key=lambda x: x[1])

        n = len(sorted_classes)
        if n <= 3:
            groups = [[cls for cls, _ in sorted_classes]]
        else:
            # 三等分
            third = n // 3
            groups = [
                [cls for cls, _ in sorted_classes[:third]],
                [cls for cls, _ in sorted_classes[third:2*third]],
                [cls for cls, _ in sorted_classes[2*third:]]
            ]

        experts = [
            ExpertConfig(
                name='baseline',
                target_classes=list(class_distribution.keys()),
                description='基准模型，处理所有类别',
                dataset_name='baseline_all_classes'
            )
        ]

        for i, group in enumerate(groups):
            if group:  # 确保组不为空
                experts.append(
                    ExpertConfig(
                        name=f'specialist_expert_{i+1}',
                        target_classes=group,
                        description=f'专项专家{i+1}，处理类别 {group}',
                        dataset_name=f'specialist_expert_{i+1}'
                    )
                )

        return {
            'type': 'hierarchical',
            'experts': experts
        }

    def create_expert_datasets(self, expert_strategy: Dict) -> Dict[str, str]:
        """
        为每个专家创建专用数据集

        Args:
            expert_strategy: 专家策略配置

        Returns:
            专家数据集路径字典 {expert_name: dataset_path}
        """
        print("\n📦 创建专家专用数据集...")

        dataset_paths = {}

        for expert_config in expert_strategy['experts']:
            print(f"   🔄 创建数据集: {expert_config.name}")

            # 构建数据集创建命令
            output_dir = f"train_test_data/{expert_config.dataset_name}"

            cmd = [
                "python", "create_train_test_set.py",
                "--source_dir", self.data_source_dir,
                "--output_dir", output_dir,
                "--experiment_type", "imbalanced"  # 保持原始分布
            ]

            # 如果是少数类专家，指定目标类别
            if "minority" in expert_config.name or "specialist" in expert_config.name:
                classes_str = ",".join(map(str, expert_config.target_classes))
                cmd.extend(["--minority-classes", classes_str])

            try:
                print(f"   执行命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"   ✅ 数据集创建成功: {output_dir}")
                dataset_paths[expert_config.name] = output_dir

            except subprocess.CalledProcessError as e:
                print(f"   ❌ 数据集创建失败: {e}")
                print(f"   错误输出: {e.stderr}")
                continue

        return dataset_paths

    def train_expert_models(self, expert_strategy: Dict,
                          dataset_paths: Dict[str, str]) -> Dict[str, str]:
        """
        训练专家模型

        Args:
            expert_strategy: 专家策略配置
            dataset_paths: 数据集路径字典

        Returns:
            专家模型路径字典 {expert_name: model_path}
        """
        print("\n🤖 训练专家模型...")

        model_paths = {}

        for expert_config in expert_strategy['experts']:
            if expert_config.name not in dataset_paths:
                print(f"   ⚠️  跳过 {expert_config.name}: 缺少数据集")
                continue

            print(f"   🔄 训练模型: {expert_config.name}")

            dataset_path = dataset_paths[expert_config.name]
            model_dir = f"trained_models/{expert_config.name}"

            # 构建训练命令
            cmd = [
                "python", "train_resnet.py",
                "--train_data", f"{dataset_path}/train.parquet",
                "--test_data", f"{dataset_path}/test.parquet",
                "--output_dir", model_dir,
                "--max_epochs", "50",
                "--task", "traffic"
            ]

            try:
                print(f"   执行命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                # 查找生成的模型文件
                if os.path.exists(model_dir):
                    ckpt_files = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
                    if ckpt_files:
                        model_path = os.path.join(model_dir, ckpt_files[0])
                        model_paths[expert_config.name] = model_path
                        print(f"   ✅ 模型训练成功: {model_path}")
                    else:
                        print(f"   ⚠️  未找到模型文件在 {model_dir}")
                else:
                    print(f"   ⚠️  模型目录不存在: {model_dir}")

            except subprocess.CalledProcessError as e:
                print(f"   ❌ 模型训练失败: {e}")
                print(f"   错误输出: {e.stderr}")
                continue

        return model_paths

    def optimize_ensemble_weights(self, expert_strategy: Dict,
                                model_paths: Dict[str, str]) -> Dict:
        """
        优化集成权重

        Args:
            expert_strategy: 专家策略配置
            model_paths: 模型路径字典

        Returns:
            优化后的配置
        """
        print("\n⚖️  优化集成权重...")

        # 简单的权重分配策略
        # 基准模型获得基础权重，专家模型根据其专业性获得额外权重

        total_experts = len(model_paths)
        if total_experts == 1:
            weights = {name: 1.0 for name in model_paths.keys()}
        elif total_experts == 2:
            # 基准模型 + 单一专家
            baseline_weight = 0.8
            expert_weight = 0.2
            weights = {}
            for name in model_paths.keys():
                if 'baseline' in name:
                    weights[name] = baseline_weight
                else:
                    weights[name] = expert_weight
        else:
            # 多个专家的权重分配
            baseline_weight = 0.6
            remaining_weight = 0.4
            expert_weight = remaining_weight / (total_experts - 1)

            weights = {}
            for name in model_paths.keys():
                if 'baseline' in name:
                    weights[name] = baseline_weight
                else:
                    weights[name] = expert_weight

        print("   权重分配:")
        for name, weight in weights.items():
            print(f"     {name}: {weight:.3f}")

        return {
            'strategy': expert_strategy,
            'model_paths': model_paths,
            'weights': weights
        }

    def run_full_pipeline(self) -> AnalysisResults:
        """
        运行完整的自动化专家系统流程

        Returns:
            分析结果
        """
        print("=" * 80)
        print("🚀 启动自动化专家选择系统")
        print("=" * 80)

        try:
            # Phase 1: 数据分布分析
            print("\n" + "="*50)
            print("📊 Phase 1: 数据分布分析")
            print("="*50)

            class_distribution = self.analyze_class_distribution()
            stats = self.calculate_distribution_statistics(class_distribution)

            # Phase 2: 少数类识别
            print("\n" + "="*50)
            print("🎯 Phase 2: 少数类自动识别")
            print("="*50)

            minority_classes = self.identify_minority_classes(class_distribution, stats)

            # Phase 3: 专家策略设计
            print("\n" + "="*50)
            print("🧠 Phase 3: 专家策略设计")
            print("="*50)

            expert_strategy = self.design_expert_strategy(class_distribution, minority_classes)

            # Phase 4: 数据集创建
            print("\n" + "="*50)
            print("📦 Phase 4: 专家数据集创建")
            print("="*50)

            dataset_paths = self.create_expert_datasets(expert_strategy)

            # Phase 5: 模型训练
            print("\n" + "="*50)
            print("🤖 Phase 5: 专家模型训练")
            print("="*50)

            model_paths = self.train_expert_models(expert_strategy, dataset_paths)

            # Phase 6: 权重优化
            print("\n" + "="*50)
            print("⚖️  Phase 6: 集成权重优化")
            print("="*50)

            final_config = self.optimize_ensemble_weights(expert_strategy, model_paths)

            # 保存结果
            results = AnalysisResults(
                class_distribution=class_distribution,
                total_samples=sum(class_distribution.values()),
                total_classes=len(class_distribution),
                statistics=stats,
                minority_classes=minority_classes,
                expert_strategy=final_config
            )

            # 保存配置文件
            self.save_results(results)

            print("\n" + "="*80)
            print("🎉 自动化专家选择系统执行完成!")
            print("="*80)

            return results

        except Exception as e:
            print(f"\n❌ 系统执行失败: {e}")
            raise
        finally:
            if self.spark:
                self.spark.stop()

    def save_results(self, results: AnalysisResults):
        """
        保存分析结果

        Args:
            results: 分析结果
        """
        print("\n💾 保存分析结果...")

        # 保存详细配置
        config_file = os.path.join(self.output_base_dir, "auto_expert_config.json")

        # 转换为可序列化的格式
        serializable_config = {
            'strategy': results.expert_strategy['strategy'].type if hasattr(results.expert_strategy['strategy'], 'type') else 'unknown',
            'experts': [
                {
                    'name': expert.name,
                    'target_classes': expert.target_classes,
                    'description': expert.description,
                    'dataset_name': expert.dataset_name,
                    'model_path': expert.model_path
                }
                for expert in results.expert_strategy['strategy'].experts if hasattr(results.expert_strategy['strategy'], 'experts')
            ],
            'model_paths': results.expert_strategy['model_paths'],
            'weights': results.expert_strategy['weights']
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)

        # 保存分析报告
        report_file = os.path.join(self.output_base_dir, "analysis_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("自动化专家选择系统分析报告\n")
            f.write("="*50 + "\n\n")

            f.write(f"数据源目录: {self.data_source_dir}\n")
            f.write(f"总样本数: {results.total_samples:,}\n")
            f.write(f"总类别数: {results.total_classes}\n")
            f.write(f"少数类: {results.minority_classes}\n\n")

            f.write("类别分布详情:\n")
            for class_id, count in sorted(results.class_distribution.items()):
                class_name = ID_TO_TRAFFIC.get(class_id, f"Unknown-{class_id}")
                marker = " [少数类]" if class_id in results.minority_classes else ""
                f.write(f"  类别 {class_id} ({class_name}){marker}: {count:,} 样本\n")

            f.write(f"\n分布统计特征:\n")
            for key, value in results.statistics.items():
                f.write(f"  {key}: {value:,.3f}\n")

            f.write(f"\n专家策略配置:\n")
            f.write(f"  模型路径: {results.expert_strategy['model_paths']}\n")
            f.write(f"  权重分配: {results.expert_strategy['weights']}\n")

        print(f"   ✅ 配置文件已保存: {config_file}")
        print(f"   ✅ 分析报告已保存: {report_file}")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="自动化专家选择系统")
    parser.add_argument("--data_source", type=str,
                       default="processed_data/vpn",
                       help="预处理数据源目录")
    parser.add_argument("--output_dir", type=str,
                       default="auto_expert_results",
                       help="输出基础目录")
    parser.add_argument("--config", type=str,
                       help="系统配置文件路径 (JSON格式)")

    args = parser.parse_args()

    # 创建系统实例
    auto_system = AutoExpertSystem(args.data_source, args.output_dir)

    # 加载自定义配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            auto_system.config.update(custom_config)
        print(f"✅ 已加载自定义配置: {args.config}")

    # 运行完整流程
    results = auto_system.run_full_pipeline()

    print("\n🎯 接下来可以使用以下命令进行模型评估:")
    print("python evaluation.py --eval-mode ensemble --data_path <test_data> \\")
    print("                    --baseline_model_path <baseline_model> \\")
    print("                    --minority_model_path <minority_model> \\")
    print("                    --minority_classes <minority_classes>")

if __name__ == "__main__":
    main()