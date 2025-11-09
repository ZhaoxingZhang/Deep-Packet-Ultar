# 门控网络测试配置

本目录包含了门控网络自动化测试的配置文件。

## 配置文件说明

### `gating_network_test.yaml`
完整的测试配置，包含5个不同的门控网络配置：
- λ=0.1, 0.3, 0.5, 0.7 的固定值测试
- 自适应调度测试

### `gating_network_quick_test.yaml`
快速测试配置，只包含3个代表性配置，训练轮数较少：
- λ=0.3, 0.5 的固定值测试
- 自适应调度测试

## 使用方法

### 运行完整测试
```bash
/usr/local/Caskroom/miniconda/base/envs/deep_packet/bin/python test_gating_networks.py --config config/gating_network_test.yaml
```

### 运行快速测试
```bash
/usr/local/Caskroom/miniconda/base/envs/deep_packet/bin/python test_gating_networks.py --config config/gating_network_quick_test.yaml
```

## 配置文件格式

```yaml
# 基础配置（必需）
base_config:
  train_data_path: "训练数据路径"
  baseline_model_path: "基准模型路径"
  minority_model_path: "专家模型路径"
  minority_classes: [5, 7]  # 少数类标签列表
  epochs: 10                # 训练轮数
  lr: 0.001                 # 学习率

# 测试配置（必需）
test_configs:
  - name: "配置名称"
    output_path: "模型保存路径"
    lambda_macro: 0.5       # 固定lambda值（可选）
    use_adaptive: true       # 使用自适应调度（可选）
    initial_lambda: 0.1      # 初始lambda值（自适应时必需）
    final_lambda: 0.7        # 最终lambda值（自适应时必需）
```

## 参数说明

### lambda_macro
- 范围：0.0 - 1.0
- 作用：控制Macro-F1损失在总损失中的权重
- 0.0：纯CrossEntropy损失（只关注准确率）
- 1.0：纯Macro-F1损失（只关注类别平衡）
- 0.5：两者平衡

### use_adaptive
- true：使用自适应调度，lambda值从initial_lambda逐渐增加到final_lambda
- false：使用固定的lambda_macro值

### 输出文件
所有日志和报告都会保存在 `log/` 目录下：
- 训练日志：`training_{配置名}_{时间戳}.log`
- 评估日志：`evaluation_{配置名}_{时间戳}.log`
- 测试报告：`gating_network_test_report_{时间戳}.json/.txt`