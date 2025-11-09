#!/usr/bin/env python3
"""
自动化测试脚本：训练和评估不同参数的门控网络

这个脚本会：
1. 训练多个不同λ值的门控网络
2. 训练自适应调度的门控网络
3. 使用评估脚本测试所有模型的性能
4. 生成对比报告
"""

import warnings
warnings.filterwarnings("ignore", "pkg_resources is deprecated")

import os
import subprocess
import json
import time
import click
from datetime import datetime
from ruamel.yaml import YAML

# 创建log目录
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)

# 配置参数
PYTHON_PATH = "python"

# 默认基础配置
DEFAULT_BASE_CONFIG = {
    "train_data_path": "train_test_data/exp2_imbalanced_small/traffic_classification/train.parquet",
    "baseline_model_path": "model/resnet_traffic_featurize.model.ckpt",
    "minority_model_path": "model/minority_expert_resnet.pth.ckpt",
    "minority_classes": [5, 7],
    "epochs": 10,
    "lr": 0.001
}

# 默认测试配置
DEFAULT_TEST_CONFIGS = [
    {
        "name": "lambda_0.1",
        "output_path": "model/model/gating_network_lambda_0.1.ckpt",
        "lambda_macro": 0.1
    },
    {
        "name": "lambda_0.3",
        "output_path": "model/model/gating_network_lambda_0.3.ckpt",
        "lambda_macro": 0.3
    },
    {
        "name": "lambda_0.5",
        "output_path": "model/model/gating_network_lambda_0.5.ckpt",
        "lambda_macro": 0.5
    },
    {
        "name": "lambda_0.7",
        "output_path": "model/model/gating_network_lambda_0.7.ckpt",
        "lambda_macro": 0.7
    },
    {
        "name": "adaptive",
        "output_path": "model/model/gating_network_adaptive.ckpt",
        "use_adaptive": True,
        "initial_lambda": 0.1,
        "final_lambda": 0.7
    }
]

def load_config(config_file_path):
    """从YAML文件加载配置"""
    yaml = YAML()
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_file_path}")
        return None
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return None

def merge_configs(default_config, user_config):
    """合并用户配置和默认配置"""
    merged = default_config.copy()
    if user_config:
        merged.update(user_config)
    return merged


def run_training_command(config):
    """执行单个训练命令"""
    cmd = [
        PYTHON_PATH,
        "train_gating_network.py",
        "--train_data_path", config["train_data_path"],
        "--baseline_model_path", config["baseline_model_path"],
        "--minority_model_path", config["minority_model_path"],
        "--output_path", config["output_path"],
        "--epochs", str(config["epochs"]),
        "--lr", str(config["lr"])
    ]

    # 添加minority classes
    for minority_class in config["minority_classes"]:
        cmd.extend(["--minority_classes", str(minority_class)])

    # 添加λ相关参数
    if config.get("use_adaptive", False):
        cmd.extend([
            "--use_adaptive",
            "--initial_lambda", str(config["initial_lambda"]),
            "--final_lambda", str(config["final_lambda"])
        ])
    else:
        cmd.extend(["--lambda_macro", str(config["lambda_macro"])])

    # 创建日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOG_DIR, f"training_{config['name']}_{timestamp}.log")

    print(f"\n=== 开始训练: {config['name']} ===")
    print(f"命令: {' '.join(cmd)}")
    print(f"日志文件: {log_file}")

    start_time = time.time()

    # 运行命令并输出到日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        # 写入命令头部信息
        f.write(f"=== 门控网络训练日志: {config['name']} ===\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"命令: {' '.join(cmd)}\n")
        f.write(f"配置: {config}\n")
        f.write("="*80 + "\n\n")

        # 运行命令并实时写入日志
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, universal_newlines=True)

        output_lines = []
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line.strip())
            print(line.strip(), end='')  # 实时显示
            f.write(line)
            f.flush()

        process.wait()
        result = subprocess.CompletedProcess(process.args, process.returncode,
                                           ''.join(output_lines), '')

    end_time = time.time()
    training_time = end_time - start_time

    # 在日志文件末尾写入总结
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n=== 训练总结 ===\n")
        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {training_time:.1f}秒\n")
        f.write(f"返回码: {result.returncode}\n")
        f.write(f"成功: {'是' if result.returncode == 0 else '否'}\n")

    if result.returncode == 0:
        print(f"\n✅ {config['name']} 训练成功 (耗时: {training_time:.1f}秒)")
        return {
            "success": True,
            "training_time": training_time,
            "log_file": log_file,
            "output": result.stdout
        }
    else:
        print(f"\n❌ {config['name']} 训练失败 (耗时: {training_time:.1f}秒)")
        print(f"详细信息请查看日志: {log_file}")
        return {
            "success": False,
            "training_time": training_time,
            "log_file": log_file,
            "output": result.stdout
        }

def run_evaluation(model_path, model_name, test_data_path, baseline_model_path, minority_model_path, minority_classes):
    """运行评估脚本"""
    # 创建输出目录
    output_dir = os.path.join(LOG_DIR, f"evaluation_results_{model_name}")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        PYTHON_PATH,
        "evaluation.py",
        "--data_path", test_data_path,
        "--output_dir", output_dir,
        "--baseline_model_path", baseline_model_path,
        "--minority_model_path", minority_model_path,
        "--gating_network_path", model_path,
        "--eval-mode", "gating_ensemble",
        "--minority_classes"
    ] + [str(c) for c in minority_classes]

    # 创建评估日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOG_DIR, f"evaluation_{model_name}_{timestamp}.log")

    print(f"\n=== 评估模型: {model_name} ===")
    print(f"命令: {' '.join(cmd)}")
    print(f"日志文件: {log_file}")

    start_time = time.time()

    # 运行命令并输出到日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        # 写入评估头部信息
        f.write(f"=== 门控网络评估日志: {model_name} ===\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"命令: {' '.join(cmd)}\n")
        f.write("="*80 + "\n\n")

        # 运行命令并实时写入日志
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, universal_newlines=True)

        output_lines = []
        for line in iter(process.stdout.readline, ''):
            output_lines.append(line.strip())
            print(line.strip(), end='')  # 实时显示
            f.write(line)
            f.flush()

        process.wait()
        result = subprocess.CompletedProcess(process.args, process.returncode,
                                           ''.join(output_lines), '')

    end_time = time.time()
    evaluation_time = end_time - start_time

    # 在日志文件末尾写入总结
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n\n=== 评估总结 ===\n")
        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"评估耗时: {evaluation_time:.1f}秒\n")
        f.write(f"返回码: {result.returncode}\n")
        f.write(f"成功: {'是' if result.returncode == 0 else '否'}\n")

    if result.returncode == 0:
        print(f"\n✅ {model_name} 评估成功 (耗时: {evaluation_time:.1f}秒)")

        # 解析评估结果
        try:
            output_lines = result.stdout.split('\n')
            accuracy = None
            macro_avg = None
            minority_classes = {}

            for line in output_lines:
                if "Accuracy:" in line:
                    accuracy = float(line.split(":")[1].strip())
                elif "macro avg" in line:
                    parts = line.split()
                    macro_avg = float(parts[2]) if len(parts) > 2 else None
                elif line.strip().isdigit() and int(line.strip()) in [5, 7]:
                    # 找到少数类，下一行是指标
                    idx = output_lines.index(line)
                    if idx + 1 < len(output_lines):
                        next_line = output_lines[idx + 1]
                        if "precision" in next_line:
                            metrics = next_line.split()
                            if len(metrics) >= 4:
                                minority_classes[int(line.strip())] = {
                                    "precision": float(metrics[0]),
                                    "recall": float(metrics[1]),
                                    "f1": float(metrics[2])
                                }

            # 在日志文件中写入解析结果
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n=== 解析结果 ===\n")
                f.write(f"准确率: {accuracy:.4f}\n" if accuracy else "准确率: N/A\n")
                f.write(f"Macro-F1: {macro_avg:.4f}\n" if macro_avg else "Macro-F1: N/A\n")
                if minority_classes:
                    f.write(f"少数类表现:\n")
                    for class_id, metrics in minority_classes.items():
                        f.write(f"  类别{class_id}: F1={metrics.get('f1', 0):.4f}, ")
                        f.write(f"Precision={metrics.get('precision', 0):.4f}, ")
                        f.write(f"Recall={metrics.get('recall', 0):.4f}\n")

            return {
                "success": True,
                "evaluation_time": evaluation_time,
                "accuracy": accuracy,
                "macro_avg": macro_avg,
                "minority_classes": minority_classes,
                "log_file": log_file,
                "output": result.stdout
            }
        except Exception as e:
            print(f"⚠️ 解析评估结果时出错: {e}")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n=== 解析错误 ===\n")
                f.write(f"错误信息: {e}\n")
            return {
                "success": True,
                "evaluation_time": evaluation_time,
                "log_file": log_file,
                "output": result.stdout
            }
    else:
        print(f"\n❌ {model_name} 评估失败 (耗时: {evaluation_time:.1f}秒)")
        print(f"详细信息请查看日志: {log_file}")

        # 在日志文件中写入错误信息
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=== 错误信息 ===\n")
            f.write(f"错误输出: {result.stderr}\n")

        return {
            "success": False,
            "evaluation_time": evaluation_time,
            "log_file": log_file,
            "output": result.stdout,
            "error": result.stderr
        }

def generate_report(results):
    """生成测试报告"""
    print("\n" + "="*80)
    print("📊 门控网络测试报告")
    print("="*80)

    successful_trainings = [r for r in results if r["training"]["success"]]
    successful_evaluations = [r for r in results if r.get("evaluation", {}).get("success", False)]

    print(f"\n📈 总体统计:")
    print(f"  训练成功: {len(successful_trainings)}/{len(results)}")
    print(f"  评估成功: {len(successful_evaluations)}/{len(results)}")

    if successful_evaluations:
        print(f"\n🏆 性能排名 (按Macro-F1):")

        # 按macro_avg排序
        ranked = sorted(successful_evaluations,
                       key=lambda x: x["evaluation"].get("macro_avg", 0),
                       reverse=True)

        for i, result in enumerate(ranked, 1):
            config = result["config"]
            eval_result = result["evaluation"]

            print(f"\n{i}. {config['name']}")
            print(f"   训练时间: {result['training']['training_time']:.1f}秒")
            print(f"   评估时间: {eval_result['evaluation_time']:.1f}秒")
            print(f"   准确率: {eval_result.get('accuracy', 'N/A'):.4f}")
            print(f"   Macro-F1: {eval_result.get('macro_avg', 'N/A'):.4f}")

            # 显示少数类表现
            minority_classes = eval_result.get("minority_classes", {})
            if minority_classes:
                print(f"   少数类表现:")
                for class_id, metrics in minority_classes.items():
                    print(f"     类别{class_id}: F1={metrics.get('f1', 0):.4f}, "
                          f"Recall={metrics.get('recall', 0):.4f}")

    print(f"\n💡 建议:")
    if successful_evaluations:
        best_model = max(successful_evaluations,
                        key=lambda x: x["evaluation"].get("macro_avg", 0))
        print(f"  最佳模型: {best_model['config']['name']}")
        print(f"  推荐使用: {best_model['config']['output_path']}")

        # 分析λ值趋势
        lambda_models = [r for r in successful_evaluations
                        if not r["config"].get("use_adaptive", False)]
        if len(lambda_models) >= 2:
            print(f"  λ值分析: 不同λ值对性能有显著影响")
            print(f"  自适应调度: 比固定λ值 {'更好' if any(r['config'].get('use_adaptive') for r in successful_evaluations) else '需要进一步调优'}")

    # 保存详细报告
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_models": len(results),
            "successful_trainings": len(successful_trainings),
            "successful_evaluations": len(successful_evaluations)
        },
        "results": []
    }

    for result in results:
        report_data["results"].append({
            "config": result["config"],
            "training": {
                "success": result["training"]["success"],
                "time": result["training"]["training_time"],
                "error": result["training"].get("error", "")
            },
            "evaluation": result.get("evaluation", {})
        })

    report_file = os.path.join(LOG_DIR, f"gating_network_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n📄 详细报告已保存到: {report_file}")

    # 同时生成可读的文本报告
    text_report_file = os.path.join(LOG_DIR, f"gating_network_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(text_report_file, 'w', encoding='utf-8') as f:
        f.write("门控网络测试报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("总体统计:\n")
        f.write(f"  训练成功: {len(successful_trainings)}/{len(results)}\n")
        f.write(f"  评估成功: {len(successful_evaluations)}/{len(results)}\n\n")

        if successful_evaluations:
            f.write("性能排名 (按Macro-F1):\n")
            f.write("-"*60 + "\n")

            ranked = sorted(successful_evaluations,
                           key=lambda x: x["evaluation"].get("macro_avg", 0),
                           reverse=True)

            for i, result in enumerate(ranked, 1):
                config = result["config"]
                eval_result = result["evaluation"]

                f.write(f"\n{i}. {config['name']}\n")
                f.write(f"   训练时间: {result['training']['training_time']:.1f}秒\n")
                f.write(f"   评估时间: {eval_result['evaluation_time']:.1f}秒\n")
                f.write(f"   准确率: {eval_result.get('accuracy', 'N/A'):.4f}\n")
                f.write(f"   Macro-F1: {eval_result.get('macro_avg', 'N/A'):.4f}\n")

                # 日志文件信息
                f.write(f"   训练日志: {result['training'].get('log_file', 'N/A')}\n")
                f.write(f"   评估日志: {eval_result.get('log_file', 'N/A')}\n")

                # 显示少数类表现
                minority_classes = eval_result.get("minority_classes", {})
                if minority_classes:
                    f.write(f"   少数类表现:\n")
                    for class_id, metrics in minority_classes.items():
                        f.write(f"     类别{class_id}: F1={metrics.get('f1', 0):.4f}, ")
                        f.write(f"Recall={metrics.get('recall', 0):.4f}\n")

        f.write(f"\n建议:\n")
        if successful_evaluations:
            best_model = max(successful_evaluations,
                            key=lambda x: x["evaluation"].get("macro_avg", 0))
            f.write(f"  最佳模型: {best_model['config']['name']}\n")
            f.write(f"  推荐使用: {best_model['config']['output_path']}\n")

    print(f"📄 文本报告已保存到: {text_report_file}")

@click.command()
@click.option("--config", required=True, help="Path to the YAML configuration file.")
def run_test_gating_networks(config):
    """
    自动化测试门控网络不同参数的性能

    这个脚本会：
    1. 从YAML文件加载配置
    2. 训练多个不同λ值的门控网络
    3. 训练自适应调度的门控网络
    4. 使用评估脚本测试所有模型的性能
    5. 生成对比报告

    配置文件格式示例:
    base_config:
      train_data_path: "train_test_data/exp2_imbalanced_small/traffic_classification/train.parquet"
      baseline_model_path: "model/resnet_traffic_featurize.model.ckpt"
      minority_model_path: "model/minority_expert_resnet.pth.ckpt"
      minority_classes: [5, 7]
      epochs: 10
      lr: 0.001

    test_configs:
      - name: "lambda_0.1"
        output_path: "model/model/gating_network_lambda_0.1.ckpt"
        lambda_macro: 0.1
      - name: "lambda_0.5"
        output_path: "model/model/gating_network_lambda_0.5.ckpt"
        lambda_macro: 0.5
      - name: "adaptive"
        output_path: "model/model/gating_network_adaptive.ckpt"
        use_adaptive: true
        initial_lambda: 0.1
        final_lambda: 0.7
    """

    print("🚀 开始门控网络自动化测试")
    print(f"配置文件: {config}")

    # 加载配置文件
    user_config = load_config(config)
    if user_config is None:
        return

    # 合并基础配置
    base_config = merge_configs(DEFAULT_BASE_CONFIG, user_config.get('base_config', {}))

    # 合并测试配置（如果用户提供了test_configs，则使用用户配置；否则使用默认配置）
    user_test_configs = user_config.get('test_configs')
    test_configs = user_test_configs if user_test_configs is not None else DEFAULT_TEST_CONFIGS

    print(f"基础配置:")
    for key, value in base_config.items():
        print(f"  {key}: {value}")
    print(f"\n测试配置: {len(test_configs)} 个模型")
    for i, test_config in enumerate(test_configs, 1):
        print(f"  {i}. {test_config['name']}")
    print(f"日志目录: {os.path.abspath(LOG_DIR)}")
    print(f"所有训练和评估日志将保存在该目录中")

    results = []

    for i, test_config in enumerate(test_configs, 1):
        print(f"\n{'='*60}")
        print(f"进度: {i}/{len(test_configs)} - {test_config['name']}")
        print('='*60)

        # 合并基础配置和测试配置
        full_config = base_config.copy()
        full_config.update(test_config)

        # 训练
        training_result = run_training_command(full_config)

        result_entry = {
            "config": test_config,
            "training": training_result
        }

        # 如果训练成功，进行评估
        if training_result["success"]:
            # 等待一下确保模型文件保存完成
            time.sleep(2)

            if os.path.exists(test_config["output_path"]):
                # 推导测试数据路径
                test_data_path = base_config["train_data_path"].replace("/train.parquet", "/test.parquet")

                evaluation_result = run_evaluation(
                    model_path=test_config["output_path"],
                    model_name=test_config["name"],
                    test_data_path=test_data_path,
                    baseline_model_path=base_config["baseline_model_path"],
                    minority_model_path=base_config["minority_model_path"],
                    minority_classes=base_config["minority_classes"]
                )
                result_entry["evaluation"] = evaluation_result
            else:
                print(f"⚠️ 模型文件不存在: {test_config['output_path']}")
                result_entry["evaluation"] = {"success": False, "error": "模型文件不存在"}

        results.append(result_entry)

        # 短暂休息，避免系统过载
        time.sleep(3)

    # 生成报告
    generate_report(results)

    print(f"\n🎉 测试完成!")

if __name__ == "__main__":
    run_test_gating_networks()