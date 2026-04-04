# GEE Model Analysis Scripts

分析GEE模型参数和特征,验证3.5节的理论猜想

## 目录结构

```
analysis_scripts/
├── README.md                      # 本文件
├── requirements.txt                # Python依赖
├── config.yaml                     # 配置文件
├── utils.py                        # 工具函数
├── extract_features.py             # 特征提取脚本(核心)
├── visualize_results.py            # 可视化生成脚本
└── download_test_data.sh           # 下载测试数据
```

## 使用流程

### Step 1: 安装依赖

```bash
pip install -r requirements.txt
```

### Step 2: 下载测试数据

从远端服务器下载测试数据:

```bash
bash download_test_data.sh
```

这会将以下数据下载到 `train_test_data/` 目录:
- `exp_traffic/incremental/test.parquet`
- `exp_traffic/openset/test.parquet`
- `exp_traffic_v2/incremental/test.parquet`
- `exp_traffic_v3/incremental/test.parquet`

**注意**: 需要安装 `sshpass`:
- macOS: `brew install sshpass`
- Ubuntu/Debian: `sudo apt-get install sshpass`

### Step 3: 修改配置文件

根据实际情况修改 `config.yaml`:

```yaml
experiments:
  traffic:
    incremental:
      baseline_model: "model/model/exp_traffic/incremental/resnet_baseline_all.pt.ckpt"
      expert_model: "model/model/exp_traffic/incremental/resnet_minority_expert.pt.ckpt"
      gating_network: "model/model/exp_traffic/incremental/gating_network_resnet.pt"
      test_data: "train_test_data/exp_traffic/incremental/test.parquet"
      minority_classes: [5, 7]
      model_type: "resnet"
```

### Step 4: 提取特征和参数

运行特征提取脚本(使用CPU,可能需要较长时间):

```bash
python extract_features.py --config config.yaml --experiment traffic --scenario incremental --device cpu
```

**输出**:
- `analysis_results/features/baseline_features.npz` - Baseline模型的倒数第二层特征
- `analysis_results/features/expert_features.npz` - Expert模型的倒数第二层特征
- `analysis_results/features/gee_features.npz` - GEE模型的倒数第二层特征
- `analysis_results/features/baseline_probs.npz` - Baseline的Softmax概率
- `analysis_results/features/gating_probs.npz` - GEE的Softmax概率
- `analysis_results/weights/gating_network_weights.json` - 门控网络权重分析
- `analysis_results/weights/weight_matrix.npz` - 门控网络权重矩阵
- `analysis_results/weights/output_contribution.json` - 输出贡献分析

**注意**:
- 本地没有GPU,使用CPU进行分析
- 特征提取可能需要30分钟到1小时(取决于数据大小)
- 可以看到进度条显示执行进度

### Step 5: 生成可视化图表

运行可视化脚本生成所有图表:

```bash
python visualize_results.py --results_dir ./analysis_results --output_dir ../thesis/figures
```

**生成的图表**:
- `figure_5_1_weight_heatmap.png/pdf` - 门控网络权重分布热力图
- `figure_5_2_output_contribution.png/pdf` - 输出贡献柱状图
- `figure_5_3_decision_boundaries.png/pdf` - 决策边界可视化
- `figure_5_6_entropy_distribution.png/pdf` - Softmax熵分布对比

## 分析其他实验

要分析其他数据集或场景,修改 `--experiment` 和 `--scenario` 参数:

```bash
# Dataset: traffic_v2
python extract_features.py --config config.yaml --experiment traffic_v2 --scenario incremental --device cpu

# Dataset: traffic_v3
python extract_features.py --config config.yaml --experiment traffic_v3 --scenario incremental --device cpu

# Open Set scenario
python extract_features.py --config config.yaml --experiment traffic --scenario openset --device cpu
```

## 预期输出

### 1. 权重分析验证

**期望结果**:
- 少数类(5, 7)维度的expert权重 > baseline权重
- 例如: Class 5的weight_ratio ≈ 2.3, Class 7的weight_ratio ≈ 2.5

### 2. 输出贡献验证

**期望结果**:
- 多数类(8, 9, 10): baseline贡献 > expert贡献
- 少数类(5, 7): expert贡献 > baseline贡献
- 例如: Class 7的expert_mean ≈ 0.78, baseline_mean ≈ 0.22

### 3. 熵分析验证

**期望结果**:
- GEE对未知样本的平均熵 > Baseline对未知样本的平均熵
- 例如: GEE熵 ≈ 1.18, Baseline熵 ≈ 0.35

## 故障排除

### 问题1: 模型加载失败

**错误**: `FileNotFoundError: Model not found`

**解决**:
- 检查 `model/model/` 目录是否有模型文件
- 检查 `config.yaml` 中的路径是否正确

### 问题2: 测试数据不存在

**错误**: `FileNotFoundError: Test data not found`

**解决**:
```bash
bash download_test_data.sh  # 先下载测试数据
```

### 问题3: 内存不足

**错误**: `MemoryError` 或程序崩溃

**解决**:
- 修改 `config.yaml` 中的 `batch_size: 64` 为更小的值(如32)
- 关闭其他应用程序释放内存

### 问题4: 执行速度慢

**原因**: 使用CPU进行特征提取

**解决**:
- 这是正常的,使用CPU比GPU慢10-50倍
- 耐心等待,可以查看进度条
- 或者只分析最重要的traffic数据集

## 与3.5节理论的对应关系

| 3.5节理论猜想 | 验证方法 | 输出文件 |
|--------------|---------|---------|
| 门控网络学会"在少数类上信任专家" | 权重分析+输出贡献 | weight_heatmap, output_contribution |
| 加权损失→梯度均衡 | 梯度分析(需要训练时记录) | gradient_comparison |
| 中性决策边界 | 决策边界可视化 | decision_boundaries |
| GEE对未知样本高熵 | 熵分析 | entropy_distribution |

## 下一步

1. **确认数据和模型**: 检查 `model/model/` 和 `train_test_data/` 目录
2. **下载测试数据**: 运行 `bash download_test_data.sh`
3. **运行特征提取**: `python extract_features.py --config config.yaml --experiment traffic --scenario incremental --device cpu`
4. **生成可视化**: `python visualize_results.py`
5. **更新第五章**: 将生成的图表整合到论文第五章中

## 时间估算

- 下载测试数据: 10-30分钟(取决于网络速度)
- 特征提取: 30分钟-2小时(使用CPU,取决于数据大小)
- 生成可视化: 5-10分钟
- 总计: 1-3小时
