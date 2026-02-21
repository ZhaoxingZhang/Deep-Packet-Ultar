# 实验脚本配置总结

## 概述

为三个数据集（Dataset 1, 2, 3）创建了完整的实验脚本，每个数据集包含4个核心实验：
1. 增量学习 - Baseline
2. 增量学习 - GEE
3. 开放集识别 - Baseline
4. 开放集识别 - GEE

---

## 数据集配置对比

### Dataset 1: `processed_data/traffic` (原有)
**目录**: `scripts/exp_traffic/`

| 配置项 | 值 |
|--------|-----|
| Traffic类型 | 0, 2, 4, 6, 7, 8, 9, 10 (8种) |
| Minority Classes | 4 (VoIP), 7 (VPN:Email) |
| 全部类别 | 0 2 4 6 7 8 9 10 |
| 数据采样率 | 增量: 0.01, 开放集: 0.005 |

**类别说明**:
- 0: Chat (QQ/Weixin聊天)
- 2: File Transfer (QQ/Weixin文件传输)
- 4: VoIP (QQ/Weixin语音)
- 6: VPN: File Transfer
- 7: VPN: Email
- 8: VPN: Streaming
- 9: VPN: Torrent
- 10: VPN: VoIP

---

### Dataset 2: `processed_data/traffic_v2` (新增)
**目录**: `scripts/exp_traffic_v2/`

| 配置项 | 值 |
|--------|-----|
| Traffic类型 | 0, 2, 4, 5, 6, 8 (6种) |
| Minority Classes | 5 (VPN:Chat), 8 (VPN:Streaming) |
| 全部类别 | 0 2 4 5 6 8 |
| 数据采样率 | 增量: 0.01, 开放集: 0.01 |

**类别说明**:
- 0: Chat (QQ/Weixin聊天)
- 2: File Transfer (QQ/Weixin文件传输)
- 4: VoIP (QQ/Weixin语音)
- **5: VPN: Chat** (AIM/Skype/ICQ聊天) - *新增*
- 6: VPN: File Transfer (SFTP/Skype文件)
- **8: VPN: Streaming** (Spotify/Vimeo/YouTube) - *新增*

**少数类选择理由**:
- 类型5 (VPN:Chat): 样本量少（仅3个文件）
- 类型8 (VPN:Streaming): 样本量相对较少（44个文件）

---

### Dataset 3: `processed_data/traffic_v3` (新增)
**目录**: `scripts/exp_traffic_v3/`

| 配置项 | 值 |
|--------|-----|
| Traffic类型 | 0, 2, 4, 7, 9, 10 (6种) |
| Minority Classes | 7 (VPN:Email), 9 (VPN:Torrent) |
| 全部类别 | 0 2 4 7 9 10 |
| 数据采样率 | 增量: 0.01, 开放集: 0.01 |

**类别说明**:
- 0: Chat (QQ/Weixin聊天)
- 2: File Transfer (QQ/Weixin文件传输)
- 4: VoIP (QQ/Weixin语音)
- **7: VPN: Email** - *保留*
- **9: VPN: Torrent** (BitTorrent) - *保留*
- **10: VPN: VoIP** (Skype/Facebook/VoIPbuster) - *保留*

**少数类选择理由**:
- 类型7 (VPN:Email): 样本量少（仅3个文件）
- 类型9 (VPN:Torrent): 样本量相对较少（43个文件）

---

## 实验脚本说明

### 1. 增量学习 - Baseline (`run_incremental_resnet_baseline.sh`)

**目的**: 训练一个标准的ResNet基线模型作为对比

**流程**:
1. 生成数据集（experiment_type=imbalanced）
2. 训练ResNet模型（50 epochs, 20%验证集）
3. 评估模型性能

**输出**:
- 模型: `model/{exp_name}/incremental/resnet_baseline_all.pt.ckpt`
- 评估结果: `evaluation_results/{exp_name}/incremental/baseline_resnet/`

---

### 2. 增量学习 - GEE (`run_incremental_resnet_gee.sh`)

**目的**: 训练GEE架构，验证其对少数类的改进效果

**流程**:
1. 生成主数据集（imbalanced）
2. 生成少数类数据集（exp8_minority，只包含少数类）
3. 训练Baseline ResNet
4. 训练Minority Expert ResNet
5. 训练Gating Network（100 epochs, lr=0.001，使用加权交叉熵）
6. 评估完整GEE架构

**关键参数**:
- `--minority-classes`: 指定少数类标签
- 加权交叉熵损失：自动计算类别权重

**输出**:
- Baseline模型: `{model_dir}/resnet_baseline.pt.ckpt`
- Minority模型: `{model_dir}/resnet_minority_expert.pt.ckpt`
- Gating模型: `{model_dir}/gating_network_resnet.pt`
- 评估结果: `evaluation_results/{exp_name}/incremental/gee_resnet/`

---

### 3. 开放集识别 - Baseline (`run_open_set_resnet_baseline.sh`)

**目的**: 使用6折交叉验证评估Baseline的开放集识别能力

**流程**:
对每个类别作为"未知类"：
1. 生成数据集（排除该类）
2. 训练ResNet模型
3. 评估开放集性能（AUROC, FPR@TPR95）
4. 聚合所有折的结果

**输出**:
- 每折模型: `model/{exp_name}/openset/baseline_resnet/exclude_{class}.pt.ckpt`
- 评估结果: `evaluation_results/{exp_name}/openset/baseline_resnet/exclude_{class}/`
- 聚合指标: 打印在控制台

---

### 4. 开放集识别 - GEE (`run_open_set_resnet_gee.sh`)

**目的**: 使用6折交叉验证评估GEE的开放集识别能力

**流程**:
对每个类别作为"未知类"：
1. 生成三个数据集：
   - Main: 排除未知类的已知类数据
   - Minority: 排除未知类的少数类数据
   - Unknown/Garbage: 仅包含未知类（用于训练垃圾类别）
2. 训练Baseline ResNet
3. 训练Minority Expert ResNet
4. 训练Gating Network（带垃圾类别）
5. 评估GEE的开放集性能

**关键特性**:
- 垃圾类别策略：将未知类作为第N+1个类别训练
- 动态少数类：如果某个少数类被排除为未知类，自动调整

**输出**:
- 模型: `model/{exp_name}/openset/gee_resnet/exclude_{class}/`
- 评估结果: `evaluation_results/{exp_name}/openset/gee_resnet/exclude_{class}/`

---

## 运行实验

### 单独运行某个实验

```bash
# Dataset 1
bash scripts/exp_traffic/run_incremental_resnet_baseline.sh
bash scripts/exp_traffic/run_incremental_resnet_gee.sh
bash scripts/exp_traffic/run_open_set_resnet_baseline.sh
bash scripts/exp_traffic/run_open_set_resnet_gee.sh

# Dataset 2
bash scripts/exp_traffic_v2/run_incremental_resnet_baseline.sh
bash scripts/exp_traffic_v2/run_incremental_resnet_gee.sh
bash scripts/exp_traffic_v2/run_open_set_resnet_baseline.sh
bash scripts/exp_traffic_v2/run_open_set_resnet_gee.sh

# Dataset 3
bash scripts/exp_traffic_v3/run_incremental_resnet_baseline.sh
bash scripts/exp_traffic_v3/run_incremental_resnet_gee.sh
bash scripts/exp_traffic_v3/run_open_set_resnet_baseline.sh
bash scripts/exp_traffic_v3/run_open_set_resnet_gee.sh
```

### 批量运行（后台执行）

```bash
# Dataset 1
bash scripts/exp_traffic/exe.sh

# Dataset 2
bash scripts/exp_traffic_v2/exe.sh

# Dataset 3
bash scripts/exp_traffic_v3/exe.sh
```

日志文件位置：
- `log/exp_traffic.log`
- `log/exp_traffic_v2.log`
- `log/exp_traffic_v3.log`

---

## 脚本生成工具

如需重新生成或修改脚本，可使用：

```bash
# 重新生成所有脚本
bash scripts/generate_complete_exp_scripts.sh

# 仅生成Dataset 2
# 编辑 generate_complete_exp_scripts.sh 中的调用部分
```

---

## 实验结果目录结构

```
train_test_data/
├── exp_traffic/          # Dataset 1
│   ├── incremental/
│   └── openset/
├── exp_traffic_v2/       # Dataset 2
│   ├── incremental/
│   └── openset/
└── exp_traffic_v3/       # Dataset 3
    ├── incremental/
    └── openset/

model/
├── exp_traffic/
├── exp_traffic_v2/
└── exp_traffic_v3/

evaluation_results/
├── exp_traffic/
├── exp_traffic_v2/
└── exp_traffic_v3/
```

---

## 论文实验建议

### 实验设计

1. **主数据集**: Dataset 1 (traffic) - 8种类型，最全面
2. **验证数据集**: Dataset 2 (traffic_v2) - 6种类型，含VPN Chat
3. **验证数据集**: Dataset 3 (traffic_v3) - 6种类型，含Torrent

### 对比维度

| 对比项 | Dataset 1 | Dataset 2 | Dataset 3 |
|--------|-----------|-----------|-----------|
| 类型数量 | 8 | 6 | 6 |
| 少数类 | 4,7 | 5,8 | 7,9 |
| 特色 | 全类型 | VPN Chat | VPN Torrent |

### 评估指标

**增量学习**:
- Macro-F1 (主要)
- 每个类的F1分数
- 准确率

**开放集识别**:
- AUROC
- FPR@TPR95
- 每个类作为未知类的表现

---

## 注意事项

1. **数据采样率**: Dataset 2和3使用1%采样率（比Dataset 1的0.5%稍高），因为文件总数较少
2. **少数类选择**: 基于文件数量选择，确保样本量差异明显
3. **GPU内存**: 如果OOM，可调整 `--fraction` 参数减小数据量
4. **并行执行**: 三个数据集的实验可以同时运行，互不干扰
