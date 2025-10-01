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

### **实验八：混合专家模型 (MoE) 方案 (进行中)**

*   **核心思想**: 放弃“全才”模型，组建“专家团队”。将识别所有15个类别的复杂任务，分解为更简单的子任务，交给不同的专家模型来处理，以期解决单一模型无法兼顾多数类和少数类性能的根本矛盾。

*   **架构设计**:
    *   **多数类专家 (Expert A)**: 一个只专注于识别常见流量类型的ResNet模型。
    *   **少数类专家 (Expert B)**: 一个只专注于识别稀有、疑难流量类型的ResNet模型。
    *   **门控网络 (Gating Network)**: 一个轻量级的CNN，作为“调度员”，判断输入样本应由哪个专家处理。
    *   **MoE主模型**: 一个更高层次的PyTorch Lightning模块，负责组合三个子网络，并将专家们的局部预测结果，“拼接”成覆盖全局类别的最终预测。

*   **训练流程**: 一个分阶段的、更稳健的训练策略。
    1.  **阶段一：专家预训练 (已完成)**
        *   **目标**: 独立训练两个专家，使其在各自的领域内达到最佳性能。
        *   **动作**: 
            1.  分析`exp5`数据集的类别分布，定义“多数类” (`{2, 3, 5, 10}`) 和“少数类” (其余11个)。
            2.  修改`create_train_test_set.py`，使其能生成两个专家专属的数据集。
            3.  独立训练`expert_major.model`和`expert_minor.model`。
        *   **状态**: **已完成**。我们已获得两个预训练好的专家模型。
    2.  **阶段二：门控网络训练 (下一步)**
        *   **目标**: 冻结预训练好的专家权重，只训练门控网络，让它学会如何精准地“分诊”。
        *   **实现**: 门控网络的训练目标是一个二分类问题（判断样本属于多数类还是少数类），并将采用**类别感知采样**来保证训练的公平性。
    3.  **阶段三：端到端联合微调 (可选)**
        *   **目标**: 在门控网络收敛后，解冻所有权重，用一个极小的学习率对整个系统进行微调，让所有组件更好地协同工作。

*   **关键技术点**:
    *   **标签反向映射**: 在MoE主模型的`forward`方法中，将通过预定义的映射关系，把两个专家基于局部标签的预测结果，重新“拼接”回全局的15个类别标签空间，以产生最终输出。
    *   **早停机制优化**: 训练流程已集成`ModelCheckpoint`和带有更高耐心值（20）的`EarlyStopping`，并引入了`ReduceLROnPlateau`学习率调度器，以确保能捕获到模型的最佳性能点，并避免训练过早停止。


---

## 4. Progress Reports

# 研究进展报告

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

**下一步**: 我们已经拥有了两位强大的、预训练好的专家。现在将进入MoE实施的**第二阶段：训练门控网络 (Gating Network)**。

---

## 5. Conversation Memory

# 实验记忆与上下文总结

**报告日期**: 2025-09-13

---

### **第六阶段：系统性解决数据不均衡问题**

在`exp5`确立了强大的性能基准（标准ResNet, Macro F1: 0.72）后，我们将核心矛盾聚焦于**解决数据不均衡问题**。

#### 1. 两种主流方法的证伪

我们依次尝试了两种主流方案，但均以失败告终，并得到了宝贵的否定性结论：

*   **`exp6` - 数据中心方法 (SMOTE)**: 尝试通过过采样技术创造一个均衡的数据集。
    *   **结果**: **失败** (Macro F1: 0.31)。模型严重过拟合于合成数据，泛化能力大幅下降。
    *   **结论**: 简单的过采样不适用于此任务。

*   **`exp7` - 算法层方法 (Focal Loss)**: 尝试通过修改损失函数，使其关注少数类和难分样本。
    *   **结果**: **灾难性失败** (Macro F1: 0.09)。训练过程极不稳定，模型性能完全崩溃。
    *   **结论**: 简单的静态或动态加权损失方案，在我们的极端数据分布下亦不可行。

#### 2. 代码库清理与演进

*   **并行工作**: 我们完成了对无效的**注意力机制**和**Focal Loss**相关代码的清理与回退。
*   **优化训练框架**: 在讨论中，您提出了关于**早停机制（EarlyStopping）**可能错失最优解的深刻见解。作为回应，我们通过引入`ModelCheckpoint`和`ReduceLROnPlateau`（学习率动态调整）对训练框架进行了优化，确保能捕获并保存模型在验证集上的历史最佳性能点。

### **第七阶段：转向混合专家 (MoE) 架构**

#### 1. 核心洞察与共识

*   **核心洞察**: `exp6`和`exp7`的连续失败，让我们达成共识——依赖单一的“全才”模型已无法突破性能瓶颈。
*   **当前共识**: 采纳了您提出的、极具前瞻性的建议，我们将研究方向转向一个更强大的高级架构：**混合专家网络 (MoE)**。

#### 2. MoE 实施方案设计与澄清

我们共同设计了一个分阶段的实施方案：

*   **架构**: 一个基础的、可解释的2-专家MoE模型（多数类专家 + 少数类专家 + 门控网络）。
*   **关键技术点澄清**: 
    *   **标签反向映射**: 您提出了“专家如何将局部标签映射回全局”的关键问题。我们澄清了该过程将在MoE主模型的`forward`方法中，通过“结果拼接”机制实现。
    *   **门控网络训练**: 您提出了“为何需要训练门控”以及“多数/少数类划分是否确定”的问题。我们澄清了门控网络是一个**元分类器**，需要通过在岗“实习”（即端到端训练）来学习如何根据样本特征，动态地做出最优的路由决策，而不是依赖静态规则。

#### 3. 专家预训练阶段 (`exp8`) 完成

*   **数据准备**: 我们修正了`create_train_test_set.py`中的一个bug，成功生成了专家专属的数据集。
*   **专家评估**: 我们分别训练并评估了两个专家，得出了“**多数类专家独立训练效果不佳，少数类专家独立训练效果极好**”的重要结论。
*   **战略决策**: 基于此结论，我们决定采用**`exp5`的基准模型**作为多数类专家，**`exp8`的新模型**作为少数类专家。

#### 4. 当前状态

*   MoE架构的**第一阶段（专家预训练）**已完成，两位专家已准备就绪。
*   **待执行动作**: 开始**第二阶段：训练门控网络**。