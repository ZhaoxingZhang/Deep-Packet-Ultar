# AI Agent Unified Instruction File

## 1. Meta-Instructions (How to Use This File)

### 1.1. File Structure
*   **Section 1 (This Section):** Defines how the AI agent should interact with this document.
*   **Section 2: Core Research Plan:** High-level research goals, background, and methodology.
*   **Section 3: Project Changelog:** A reverse-chronological log of significant code and file system changes.
*   **Section 4: Progress Reports:** A collection of formatted progress reports suitable for stakeholders.
*   **Section 5: Conversation Memory:** A log of the most recent conversation turns to provide immediate context.

### 1.2. AI Reading Protocol
*   **On Session Start:** Always read this entire file to gain full context.
*   **Before Taking Action:** Refer to the relevant sections. For example, when planning an experiment, refer to Section 2. When modifying code, refer to Section 3 to maintain conventions.

### 1.3. AI Writing/Update Protocol
*   **[CRITICAL] Append-Only with Timestamps:** With the exception of **Section 5 (Conversation Memory)**, all new content must be **appended** to the end of the relevant section. Do not modify or delete existing content. Every new block of appended content must begin with a timestamp, e.g., `**--- Update (YYYY-MM-DD HH:MM) ---**`.
*   **Centralized Updates:** All updates should be made to this single file.
*   **Changelog:** When logging a code change, append a new entry under **Section 3**.
*   **Progress Reports:** When a new report is requested, append it as a new sub-section under **Section 4**.
*   **Conversation Memory:** This is the only exception. This section should be **overwritten** at the end of a session with the latest context summary.

---

## 2. Core Research Plan

# AI 实验设计与执行代理

## 1. 角色与目标

你是一名专业的机器学习研究员和实验设计专家。

你的核心目标是根据以下提供的研究背景和实验思路，设计、实施并评估一系列实验，以验证“面向开放集识别的增量式加密流量分类技术”的有效性。

## 2. 研究背景

*   **核心课题**: 面向开放集识别的增量式加密流量分类技术的研究与实现。
*   **关键挑战**: 
    1.  **开放集识别**: 模型需要有效识别并归类从未见过的新流量类别（即“未知类”）。
    2.  **数据不均衡**: 在真实网络环境中，各类应用的流量数据通常是不均衡的，模型需要在此情况下依然保持稳健的性能。

## 3. 实验设计方案

**--- Update (2025-09-30) ---**

### **实验八：混合专家模型 (MoE) 方案 (已完成)**

*   **核心思想**: 放弃“全才”模型，组建“专家团队”。将识别所有15个类别的复杂任务，分解为更简单的子任务，交给不同的专家模型来处理，以期解决单一模型无法兼顾多数类和少数类性能的根本矛盾。

*   **架构设计**:
    *   **多数类专家 (Expert A)**: 采用 `exp5` 的基准模型，作为多数类分类器。
    *   **少数类专家 (Expert B)**: 采用 `exp8` 新训练的模型，专注于识别稀有类别。
    *   **门控网络 (Gating Network)**: 一个轻量级的CNN，作为“调度员”，判断输入样本应由哪个专家处理。
    *   **MoE主模型**: 一个更高层次的PyTorch Lightning模块，负责组合三个子网络，并将专家们的局部预测结果，“拼接”成覆盖全局类别的最终预测。

*   **训练流程**:
    1.  **阶段一：专家预训练 (已完成)**
        *   **决策**: 采用 `exp5` 基准模型作为多数类专家，`exp8` 新模型作为少数类专家。
        *   **状态**: **已完成**。
    2.  **阶段二：门控网络训练 (已完成)**
        *   **目标**: 冻结专家权重，只训练门控网络。
        *   **动作**:
            1.  修复了 `ml/model.py` 中缺失的 `MixtureOfExperts` 类定义。
            2.  在 `train_moe.py` 中加入了冻结专家权重的逻辑。
            3.  训练门控网络，其验证准确率 (`val_gate_acc`) 达到了 **98%**，表现优异。
            4.  修复了 `MixtureOfExperts` 类中 `validation_step` 的推理逻辑，解决了“结果拼接”问题。
        *   **状态**: **已完成**。
    3.  **阶段三：端到端联合微调 (已完成)**
        *   **目标**: 解冻所有权重，用 `1e-5` 的低学习率对整个系统进行微调。
        *   **动作**:
            1.  修改 `train_moe.py` 以支持微调，并从第二阶段的最佳模型恢复训练。
            2.  修改 `ml/model.py` 以使用更低的学习率。
            3.  执行微调。
        *   **状态**: **已完成**。
    4.  **阶段四：软路由探索 (已完成)**
        *   **目标**: 将“硬路由”替换为“软路由”，探索性能是否能进一步提升。
        *   **动作**:
            1.  修改 `MixtureOfExperts` 的 `forward` 方法以实现软路由。
            2.  在实现过程中发现并修复了因填充值错误导致的性能崩溃问题。
            3.  使用修正后的软路由逻辑重新评估了最终模型。
        *   **状态**: **已完成**。

*   **最终成果**:
    *   **性能**: 最终的 MoE 模型（无论是硬路由还是软路由）在测试集上取得了 **0.60** 的 Macro F1-Score，相较于 `exp5` 基准模型的 **0.49**，实现了决定性的性能飞跃。
    *   **结论**: 实验成功。MoE 架构被证明是解决此场景下数据不均衡问题的有效方案。由于门控网络决策高度自信，软路由在此场景下未带来额外收益，简单的硬路由已足够高效。

---

## 3. Project Changelog

**--- Update (2025-10-01) ---**

*   **`create_train_test_set.py`**:
    *   **Added `exp_open_set` mode**: Creates a training set with 10 specific "known" classes and a test set with all 15 classes to simulate an open-set environment.
    *   **Added `exp_open_set_majority` mode**: Creates a dataset containing only the 3 "majority" classes from the known set (`[2, 3, 5]`)
    *   **Added `exp_open_set_minority` mode**: Creates a dataset containing only the 7 "minority" classes from the known set (`[1, 7, 8, 9, 11, 13, 14]`)
*   **`evaluate_open_set.py`**:
    *   **Created new file**: A dedicated script to evaluate open-set recognition performance. It implements three distinct rejection strategies (Baseline-Softmax, MoE-Softmax, MoE-Gate) and generates data for "Accuracy-Rejection" curves.

---

## 4. Progress Reports

# 研究进展报告
**--- Update (2025-10-01) ---**
    
### **实验A (开放集识别) 进展与设计详述**

此部分详细阐述我们为验证论文第一个核心创新点（开放集识别）所设计的实验A的完整逻辑和当前进展。

#### **核心创新的深化**

在初步讨论后，我们一致认为，简单地在模型最终输出上应用一个Softmax阈值作为“拒绝选项”，虽然有一定效果，但技术创新性不足，且未能利用MoE架构的独特性。

为此，我们将核心创新点深化为：**发掘并验证MoE架构的门控网络（Gating Network）作为一种高级“新颖性信号”（Novelty Signal）的潜力。**

我们的核心假设是：当一个模型从未见过的“未知类”样本输入时，门控网络的不确定性（表现为输出概率分布的熵增高或最大置信度降低）是比最终分类层更灵敏、更准确的“未知”指示器。这是因为最终分
    类层的不确定性通常表现为在多个“已知类”之间混淆，而门控网络的不确定性则更可能表示样本不属于任何一个已知的元类别（例如“多数类”或“少数类”），是更高维度的“无知”信号。

#### **实验设计**

为验证上述假设，实验A的核心目标是**对比不同新颖性信号的质量**。

*   **对比策略**:
    1.  **基准策略 (Baseline-Softmax)**: 在标准的ResNet基准模型上，使用其最终Softmax输出的置信度进行拒绝。
    2.  **MoE-Softmax策略**: 在我们训练的MoE模型上，同样使用其最终Softmax输出的置信度进行拒绝。
    3.  **MoE-Gate策略 (核心验证)**: 在MoE模型上，使用其**门控网络**输出的置信度进行拒绝。

*   **数据集**:
    *   **已知类 (10个)**: `[2, 3, 5, 1, 7, 8, 9, 11, 13, 14]`
    *   **未知类 (5个)**: `[10, 0, 4, 6, 12]`
    *   **设计考量**: 我们特意将一个“多数类”（类别10）放入“未知类”中。如果模型能成功将在训练中常见的类别识别为“未知”，将极大增强我们结论的说服力。

*   **评估指标**: 为上述三种策略分别绘制“准确率-拒绝率”曲线，并进行对比。我们期望证明“MoE-Gate策略”的曲线显著优于其他两者。

**--- Update (2025-10-01) ---**

### **实验A (开放集识别) 进展**

我们已经正式启动了针对论文**创新点一：开放集识别**的验证实验。

*   **核心创新点深化**: 我们将创新点从简单的“阈值拒绝”深化为“**发掘并验证MoE门控网络作为高级新颖性信号的潜力**”。实验将对比三种拒绝策略（基准模型Softmax、MoE模型Softmax、MoE门控网络不确定性），以证明MoE架构的内在优势。

*   **数据集设计**:
    *   **已知类 (10个)**: `[2, 3, 5, 1, 7, 8, 9, 11, 13, 14]`
    *   **未知类 (5个)**: `[10, 0, 4, 6, 12]` (包含一个多数类`10`以增强实验说服力)

*   **脚本准备**:
    *   `create_train_test_set.py` 已被修改，增加了`exp_open_set`, `exp_open_set_majority`, `exp_open_set_minority`三种模式以生成所需数据集。
    *   `evaluate_open_set.py` 已作为新脚本创建，用于执行我们定制的开放集评估逻辑。

*   **当前状态**: 正在等待数据集生成。一旦数据就绪，我们将按计划分阶段训练基准模型、两个新专家模型，并最终微调MoE模型，然后使用新脚本进行评估。

**--- Update (2025-10-01) ---**

  ### **论文蓝图与核心创新点规划**

  此部分作为备忘录，用于指导后续的实验设计，确保所有工作都服务于最终的学术论文目标。

  **创新点一：基于置信度拒绝的开放集识别 (Open-Set Recognition, OSR)**

   * 核心论点: 证明模型通过引入“拒绝选项”，可以在牺牲对少量不确定流量覆盖率的前提下，换取在其余流量上分类准确率的显著提升，从而提高现实应用价值。我们将此机制称为“基于置信度的拒绝机制”。

   * 实验设计 (实验A：开放集拒绝能力验证):
       1. 数据集: 将总类别（如15类）划分为“已知类”（如10类，用于训练）和“未知类”（如5类，仅用于测试）。
       2. 模型训练: 仅在“已知类”数据上训练模型。
       3. 评估方法:
           * 在包含全部“已知类”和“未知类”的测试集上进行评估。
           * 根据模型输出的Softmax最大概率（置信度），设定一个可变的置信度阈值 `τ`。
           * 任何置信度低于 τ 的样本，无论来自已知还是未知类，都被归类为“拒绝”（或“未知”）。
           * 通过调整 τ，获得一系列“准确率-拒绝率”数据点。

   * 论文所需关键图表:
       * 图：准确率-拒绝率权衡曲线:
           * X轴: 拒绝率 (被拒绝的样本 / 总样本)。
           * Y轴: 剩余样本准确率 (在未被拒绝的样本中，分类正确的比例)。
           * 目的: 直观展示拒绝不确定样本对提升核心准确率的效果。
       * 表：最优阈值下的混淆矩阵: 展示在选定的最佳 τ 值下，模型对已知、未知、被拒绝样本的详细分类情况。

  **创新点二：基于模块化更新的抗灾难性遗忘增量学习**

   * 核心论点: 证明MoE架构的模块化特性允许通过“外科手术式”的更新（即只更新部分专家网络），在学习新流量类别的同时，有效缓解“灾难性遗忘”问题，实现比传统模型更高效、低成本的增量学习。

   * 实验设计 (实验B：抗灾难性遗忘能力验证):
       1. 初始训练: 在一批“旧类别”（如10类）上，同时训练基准模型和MoE模型，记录其初始性能 Acc_old_phase1。
       2. 增量更新: 引入一批“新类别”（如2类）。
           * 基准模型: 在包含全部12个类别的数据上进行全局微调。
           * MoE模型: 仅微调与新类别相关的专家（如“少数类专家”）及门控网络，冻结其他专家。
       3. 最终评估: 在包含全部12个类别的测试集上，评估两个模型更新后的性能。

   * 论文所需关键图表:
       * 表：增量学习性能对比表:
           * 列: 初始性能 (对10个旧类), 更新后性能 (对10个旧类), 遗忘率 (性能下降值), 更新后性能 (对2个新类)。
           * 行: 基准模型, MoE模型。
           * 目的: 通过对比遗忘率，强有力地证明MoE模型在保持旧知识方面的优越性。

**--- Update (2025-09-13) ---**

### **实验八：MoE专家预训练阶段总结**

此阶段的核心目标是验证通过“分治”思想，独立训练的专家模型是否能在其各自的领域内超越“全才”的基准模型。我们依次生成了专属数据集，并训练、评估了两个专家。

#### **1. 多数类专家 (Expert A) 评估**

*   **训练流程**: 使用了优化后的训练机制（`ModelCheckpoint` + `ReduceLROnPlateau` + `patience=20`）。
*   **核心发现**: 与`exp5`基准模型相比，专门训练的多数类专家，在其负责的4个类别上，**性能并无优势，基本持平甚至略差**。
*   **结论**: 证明了多数类模型的学习，同样需要少数类作为“反例”来辅助定义决策边界。在“无菌环境”中训练并不能带来性能提升。

| 原始类别 | `exp5` 基准模型 F1 | `exp8` 多数类专家 F1 (新) | 性能变化 |
| :--- | :--- | :--- | :--- |
| 2 | 0.84 | 0.83 | **持平** |
| 3 | 0.99 | 0.98 | **持平** |
| 5 | 0.90 | 0.91 | **持平** |
| 10 | 0.90 | 0.83 | **略微变差** |

#### **2. 少数类专家 (Expert B) 评估**

*   **训练流程**: 额外使用了“类别感知采样” (`class_aware`) 策略。
*   **核心发现**: 少数类专家在识别其负责的11个类别时，**性能远超基准模型**。多个在基准模型中F1分数几乎为0的类别，得到了有效的识别。
*   **结论**: 将少数类从多数类的干扰中隔离出来进行专门训练，是极其有效的策略。

| 原始类别 | `exp5` 基准模型 F1 | `exp8` 少数类专家 F1 | 性能变化 |
| :--- | :--- | :--- | :--- |
| 1 | 0.00 | **0.66** | **巨大成功** |
| 7 | 0.75 | **0.88** | **显著提升** |
| 8 | 0.79 | **0.95** | **显著提升** |
| 9 | 0.76 | **0.99** | **巨大成功** |
| ... | ... | ... | ... |

#### **3. 阶段性战略决策**

基于以上结论，我们完成了MoE架构的**第一阶段（专家预训练）**，并确定了最终的专家人选：

*   **多数类专家**: **沿用`exp5`的基准模型** (`model/application_classification.resnet.exp5.baseline.model`)。它是在最完整、最多样的数据上训练的，事实证明它依然是我们最好的多数类分类器。
*   **少数类专家**: **采用`exp8`新训练的模型** (`model/expert_minor.model`)。它在识别疑难类别上表现卓越。

**--- Update (2025-09-30) ---**

### **实验八：MoE 最终实验报告**

此报告总结了混合专家模型（MoE）从门控网络训练到最终评估的完整流程与结论。

#### **1. 门控网络训练与评估**

*   **流程**: 我们首先冻结了两位预训练专家的权重，专门训练门控网络。在训练过程中，我们发现并修复了 `MixtureOfExperts` 类中端到端准确率 (`val_moe_acc`) 计算不正确的问题。
*   **核心发现**:
    *   门控网络本身性能卓越，验证准确率 (`val_gate_acc`) 超过 **98%**。
    *   在修复推理逻辑后，未经微调的“硬路由” MoE 模型在测试集上的 **Macro F1 分数达到了 0.59**，远超基准模型 (`exp5`) 的 **0.49**。
*   **结论**: “分而治之”的 MoE 架构取得了巨大成功。门控网络可以精确地路由，而专家各司其职，大幅提升了对少数类的识别能力。

#### **2. 端到端联合微调**

*   **流程**: 在训练好门控网络后，我们解冻了所有参数，并使用 `1e-5` 的低学习率对整个模型进行微调。
*   **核心发现**: 微调后的最终模型在测试集上的 **Macro F1 分数达到了 0.60**，与微调前相比有 **+0.01** 的微小提升。
*   **结论**: 联合微调带来的性能增益很小，表明模型在第二阶段结束后已非常接近其性能上限。

#### **3. 软路由（Soft Routing）探索**

*   **动机**: “硬路由”模型的性能上限受限于门控网络的准确率。理论上，“软路由”通过加权融合两位专家的意见，可以突破这一上限。
*   **实现与修正**: 我们在 `MixtureOfExperts` 类中实现了软路由逻辑。在发现并修复了因填充值错误导致的性能崩溃问题后，我们使用修正后的逻辑重新评估了最终模型。
*   **核心发现**: 修正后的“软路由”模型，其各项性能指标（Macro F1: 0.60）与“硬路由”模型**几乎完全相同**。

#### **4. 实验八最终结论**

**MoE 实验取得了决定性成功**。我们最终得到的模型在处理不均衡数据的核心指标（Macro F1-Score）上，实现了从 **0.49** 到 **0.60** 的飞跃。

| 类别 | 基准模型 F1 | MoE 最终模型 F1 | 性能变化 |
| :--- | :--- | :--- | :--- |
| 7 | 0.39 | **0.83** | **+0.44** |
| 8 | 0.74 | **0.90** | **+0.16** |
| 12 | 0.12 | **0.77** | **+0.65** |
| 14 | 0.73 | **0.92** | **+0.19** |
| **Macro Avg** | **0.49** | **0.60** | **+0.11** |

实验证明，由于我们的门控网络决策高度自信（准确率>98%），更复杂的“软路由”无法带来额外收益，**简单的“硬路由”已是当前架构的最优解**。MoE 架构本身是本次实验成功的关键。

---

## 5. Conversation Memory

# 实验记忆与上下文总结

**报告日期**: 2025-09-13

--- 

(...omitted for brevity...)