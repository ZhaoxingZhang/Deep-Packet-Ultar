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

### **实验一：模型对未知流量类型的适应性 (开放集识别能力) - 最终方案**

*   **基准模型**: **ResNet**
*   **实验假设**: 通过在一个包含“已知类”和“部分未知类”的数据集上进行训练，ResNet模型能够学习到区分“已知”与“未知”的通用边界，并能将训练中从未出现过的、全新的未知类也正确地识别为“未知”。
*   **执行步骤**:
    1.  **三类别划分**: 将总的应用类别（15类）随机划分为三组：
        *   **已知类 (30%, 约4-5类)**: 在训练和测试中都作为独立的、已知的类别存在。
        *   **训练期未知类 (30%, 约4-5类)**: 在训练时，它们的数据被统一映射到一个新增的“未知”标签下，用于教会模型什么是“未知”。
        *   **测试期未知类 (40%, 约6类)**: 这些类别的数据**不在训练中出现**。在测试时，它们也被映射到“未知”标签下，用于检验模型对“未知”概念的泛化能力。
    2.  **模型适配**: ResNet模型的输出层神经元数量需要动态设置为 `(已知类数量 + 1)`。
    3.  **训练与评估**: 在新生成的数据集上训练ResNet模型，并在新的测试集上评估其表现，重点关注其对“测试期未知类”的识别准确率。

### **实验二：模型在数据不均衡场景下的表现 - 暂停**

*   **状态**: 待实验一完成后继续。

## 4. 核心脚本分析

*   `create_train_test_set.py`: 从 `processed_data` 创建训练集和测试集。
*   `train_resnet.py`: 训练ResNet模型。
*   `evaluation.py`: 对训练好的模型进行评估。

## 5. 具体执行计划

### **实验一 (ResNet Baseline) - 执行中**

*   **当前状态**: 已确定最终实验方案，即将开始修改代码。
*   **下一步计划**:
    1.  **修改 `create_train_test_set.py`**: 实现上述的三类别划分（已知/训练期未知/测试期未知）和标签映射逻辑。
    2.  **修改 `ml/utils.py`**: 使 `train_application_classification_resnet_model` 函数的 `output_dim` 参数变为动态可配置。
    3.  **生成数据集**: 运行修改后的脚本，生成实验一的最终数据集。
    4.  **训练模型**: 运行 `train_resnet.py` 训练新的基准模型。
    5.  **评估模型**: 运行 `evaluation.py` (如有必要则进行微调) 对新模型进行评估。

## 6. 实验进展报告

**报告日期:** 2025年8月30日

*   **当前状态**: 已根据您的指示，确定了实验一的最终方案，并将基准模型替换为ResNet。
*   **下一步**: 开始执行新方案的第一步：修改 `create_train_test_set.py`。

**报告日期:** 2025年8月31日

### **实验一：开放集识别能力的初步验证**

#### **阶段1: ResNet测试方案 - 已完成测试**
*   在processed_data 中建立了一个small文件夹，我将每一类数据取一小部分复制进来，每一类的数据量差不多一样，这样我可以加快验证的速度。据此生成了`train_test_data/exp1_small_test/`
*   **数据生成**: `train_test_data/exp1_small_test/` ✅
    - 采用三类别划分策略：已知类(30%) + 训练期未知类(30%) + 测试期未知类(40%)
    - 实现了标签重映射，确保模型学习"未知"概念
*   **模型训练**: ResNet模型已训练完成 ✅
    - 模型文件: `model/exp1_small_test.model`
    - 训练日志: `train_results/ResNet_small_epoch40.txt`
    - 完成40个epoch，final loss: 0.0612
*   **已完成**: 评估 ✅
    ```bash
    python evaluation.py \
      -m model/exp1_small_test.model \
      -d train_test_data/exp1_small_test/application_classification/test.parquet \
      --model_type resnet
    ```
    评估结果输出在`evaluation_results/exp1_small_test`  
---


### **实验二：数据不均衡适应性**

#### **当前状态**: 待重新生成数据集，当前exp2_imbalanced数据集样本种类太少
*   **数据生成**: `train_test_data/exp2_imbalanced/` 
*   **配置**: 多数类保留100%，少数类采样10%
*   **待完成**: 
    1. **训练模型** ⏳
        ```bash
        python train_cnn.py \
          -d train_test_data/exp2_imbalanced/application_classification/train.parquet \
          -m model/exp2_imbalanced.cnn.model \
          -t app
        ```
    2. **评估模型** ⏳
        ```bash
        python evaluation.py \
          -m model/exp2_imbalanced.cnn.model \
          -d train_test_data/exp2_imbalanced/application_classification/test.parquet \
          --model_type cnn
        ```

---


### **小型测试实验 (调试与验证)**

#### **状态**: 已完成 ✅
*   **数据生成**: `train_test_data/exp1_small_test/` ✅
*   **模型训练**: 已完成40个epoch ✅
    - 模型文件: `model/exp1_small_test.model`
    - 用于快速验证训练流程的正确性

---


### **性能优化发现与改进**

#### **已完成的优化** ✅
*   **评估脚本优化**: `evaluation.py` 已优化
    - 输出目录改为 `evaluation_results/` 而非 `model/`
    - Tensor创建性能优化：使用 `torch.from_numpy(np.array())` 替代 `torch.tensor()`
    - 消除了PyTorch性能警告

#### **待完成的优化** ⏳
*   **训练脚本优化**: `train_resnet.py` 中仍存在相同性能警告
    - 文件: `ml/model.py:555`
    - 建议: 应用与 `evaluation.py` 相同的优化策略

---
**报告日期:** 2025年8月31日 (第二次更新)

### **实验二：模型在数据不均衡场景下的表现**

#### **阶段1: ResNet在不平衡数据集上的基准测试 - 已完成**

*   **数据生成**: `train_test_data/exp2_imbalanced_small/` ✅
    *   **方法**: 为了快速验证，创建了一个小型的不平衡数据集。保留了10个Facebook应用的所有数据，而其他每个类别只保留一个文件，以此来模拟数据倾斜。
    *   **脚本修改**: `create_train_test_set.py` 已被修改，增加 `--experiment_type` 参数以区分不同的实验数据生成逻辑。

*   **模型训练**: ResNet模型已在不平衡数据集上训练完成 ✅
    *   **模型文件**: `model/application_classification.resnet.exp2.model`
    *   **脚本修改**: `train_resnet.py` 已被修改，使其能够根据训练数据动态计算模型的 `output_dim`，解决了因标签不连续导致的 `IndexError`。

*   **模型评估**: 已完成 ✅
    *   **命令**:
        ```bash
        python evaluation.py \
          -m model/application_classification.resnet.exp2.model \
          -d train_test_data/exp2_imbalanced_small/application_classification/test.parquet \
          --model_type resnet
        ```
    *   **评估结果**:
        *   **准确率**: ~75.5%
        *   **保存路径**: `evaluation_results/application_classification.resnet.exp2/`

*   **核心结论**:
    *   **准确率的误导性**: 尽管准确率看似尚可，但它并不能真实反映模型性能，因为模型严重偏向于预测样本量巨大的多数类。
    *   **少数类识别失败**: 模型在识别少数类时表现极差，例如，对类别 `11` 的召回率为0，意味着完全无法识别该类别。类别 `6` 和 `14` 的F1分数也接近于0。
    *   **根本原因**: 数据不平衡是导致模型性能问题的根本原因。

*   **下一步建议**:
    *   **数据层面**: 尝试过采样（如SMOTE）或欠采样来平衡数据集。
    *   **算法层面**: 采用代价敏感学习，为少数类的错误分类施加更高的惩罚。
    *   **评估指标**: 更多地关注F1分数（特别是宏平均）和AUC-ROC，而不是单一的准确率。

---


### **实验三：模型在均衡数据场景下的基准表现**

#### **阶段1: ResNet在平衡数据集上的基准测试 - 已完成**

*   **目标**: 建立一个在均衡数据集上的性能基准，以便与实验二（不均衡数据）的结果进行直接对比，从而量化数据不平衡对模型性能的影响。

*   **数据生成**: `train_test_data/exp3_balanced_small/` ✅
    *   **方法**: 创建一个小型但类别均衡的数据集，确保每个类别都有相同数量的样本。
    *   **关键修复**: 在执行过程中，发现并修复了 `create_train_test_set.py` 中的一个关键bug。原有的 `split_train_test` 函数在处理小数据集时，可能导致测试集中某些类别缺失。已通过确保每个类别都按比例分配到训练集和测试集中来解决此问题。

*   **模型训练**: ResNet模型已在新的平衡数据集上训练完成 ✅
    *   **模型文件**: `model/application_classification.resnet.exp3.model`

*   **模型评估**: 已完成 ✅
    *   **命令**:
        ```bash
        python evaluation.py \
          -m model/application_classification.resnet.exp3.model \
          -d train_test_data/exp3_balanced_small/application_classification/test.parquet \
          --model_type resnet
        ```
    *   **评估结果保存路径**: `evaluation_results/application_classification.resnet.exp3/`

*   **当前状态与下一步**:
    *   实验三的评估已执行完毕。
    *   **下一步**: 分析 `evaluation_results/application_classification.resnet.exp3/evaluation_summary.txt` 和 `confusion_matrix.png`，总结模型在均衡数据下的真实性能，并与实验二的结果进行深入对比。

---


### **模型改进计划：集成注意力机制 (v1)**

*   **目标**: 针对实验三暴露出的基准模型性能不足（准确率~59%）的问题，通过引入注意力机制，增强模型的特征表示能力，以期提高分类准确率和F1分数。

*   **理论依据**: 采纳 `@.agent/proposal.md` 中 4.2.1 节的建议，即通过引入“注意力机制”来解决CNN模型“特征表示能力不足”的问题，并将该思路应用到我们正在使用的ResNet模型上。

*   **实现思路**:
    1.  **修改文件**: `ml/model.py`。
    2.  **修改模型**: `ResNet` 类。
    3.  **新增模块**: 定义一个轻量级的 `SEBlock` (Squeeze-and-Excitation Block) 注意力模块。该模块通过“压缩(Squeeze)”全局空间信息到通道描述符，和“激励(Excitation)”学习通道间的非线性关系，来动态地重新校准(recalibrate)每个特征通道的重要性。
    4.  **集成位置**: 将 `SEBlock` 实例插入到 `ResNet` 的最后一个残差块之后，全局平均池化层之前。
    5.  **增加控制参数**: 为 `ResNet` 类的 `__init__` 方法增加一个布尔类型的参数 `use_attention=False`。在 `forward` 方法中，通过 `if self.use_attention:` 条件来控制是否激活SE注意力模块。
    6.  **修改训练脚本**: 修改 `train_resnet.py`，添加一个命令行参数（如 `--use_attention`），用于控制在实例化`ResNet`模型时是否启用注意力机制。

*   **验证手段**:
    1.  **模型训练**: 使用 `train_resnet.py` 脚本，在 `exp3_balanced_small` 数据集上训练两个模型：
        *   **基准模型 (v2)**: 运行脚本时不添加 `--use_attention` 参数。
        *   **增强模型 (v1)**: 运行脚本时添加 `--use_attention` 参数。
    2.  **模型评估**: 分别使用 `evaluation.py` 对上述两个模型进行评估。
    3.  **结果对比**: 重点对比两个模型的准确率、宏平均F1分数以及各类别F1分数的差异。预期“增强模型”的各项指标应显著优于“基准模型”。

*   **当前状态**:
    *   **计划已更新**: 已根据您的反馈更新计划。
    *   **下一步**: 开始执行代码修改，首先是 `ml/model.py`。

---


### **模型改进计划：集成注意力机制 (v2) - 在中等规模数据集上重新验证 (进行中)**

*   **背景**: 在 `exp3_balanced_small` 小型均衡数据集上进行的注意力机制实验失败（性能反而下降）。分析认为，小数据集不足以支撑注意力机制的有效学习，可能导致过拟合或学到伪特征。

*   **新策略**: 为了更可靠地评估模型架构的改进，决定放弃小型数据集，转向一个规模更大、更具代表性的中等规模数据集。

*   **数据生成 (`exp4_medium_balanced`)**:
    *   **挑战**: 直接使用 `create_train_test_set.py` 对10%的全量数据进行采样和平衡处理时，遭遇了严重的 `SparkOutOfMemoryError`。原因是 `repartition` 和 `orderBy` 操作在local模式下对内存消耗巨大。
    *   **解决方案**: 调整了数据生成策略。不再在Spark内部进行数据平衡，而是在加载数据之前，从文件系统层面进行随机采样（抽取10%的 `.json.gz` 文件）。这创建了一个中等规模但**不均衡**的数据集 `train_test_data/exp4_medium_balanced/`。这个数据集将作为未来所有模型改进的**新基准**。
    *   **代码修改**: `create_train_test_set.py` 中增加了 `exp4_medium_balanced` 类型，实现了文件级别的采样。

*   **模型训练与评估**:
    *   **基准模型评估 (已完成)**:
        *   **模型**: `model/application_classification.resnet.exp4.baseline.model`
        *   **数据集**: `train_test_data/exp4_medium_balanced/application_classification/test.parquet`
        *   **评估结果**: 准确率 ~38.5%。这符合在不均衡数据集上的预期，将作为后续改进的基准线。
        *   **正确评估脚本**:
            ```bash
            python evaluation.py \
              -m model/application_classification.resnet.exp4.baseline.model \
              -d train_test_data/exp4_medium_balanced/application_classification/test.parquet \
              --model_type resnet
            ```

    *   **注意力模型训练 (进行中)**:
        *   **模型**: `model/application_classification.resnet.exp4.attention.model`
        *   **数据集**: `train_test_data/exp4_medium_balanced/application_classification`
        *   **当前状态**: 在修正了脚本参数的错误后，即将开始训练。
        *   **正确训练脚本**:
            ```bash
            python train_resnet.py \
              -d train_test_data/exp4_medium_balanced/application_classification \
              -m model/application_classification.resnet.exp4.attention.model \
              -t app \
              --use_attention
            ```

    *   **下一步**:
        1.  **[进行中]** 运行正确的训练脚本，训练带注意力机制的ResNet模型。
        2.  **[待办]** 使用 `evaluation.py` 评估刚刚训练好的注意力模型。
        3.  **[待办]** 对比基准模型和注意力模型的评估结果（准确率、F1分数等），以判断注意力机制是否带来了有效的性能提升。

---

---

## 3. Project Changelog

# 项目更新日志 (Changelog) 

本文档以结构化形式，记录了项目在AI代理协助下进行的所有重要改进、功能实现和问题修复。

---

### **实验设计 (Experiment Design)**

*   **`[改进]`** 开放集识别实验方案已重构。从一个简单的基线测试（训练集包含部分类别，测试集包含全部类别），演进为一个更严谨、更科学的三类别划分方案（已知类、训练期未知类、测试期未知类），以更准确地评估模型的泛化能力。
*   **`[改进]`** 实验一的基准模型已从 `CNN` 升级为更先进的 `ResNet` 模型，以获取性能更强的基线。

### **数据流水线 (`create_train_test_set.py`)**

*   **`[功能]`** 初步实现了通过 `--known_classes_ratio` 参数创建开放集识别数据集的功能。
*   **`[功能]`** 初步实现了通过 `--imbalance_ratio` 和 `--majority_class` 参数创建类别不均衡数据集的功能。
*   **`[重构]`** **(核心)** 根据最终实验方案，彻底重构了数据处理逻辑，废弃了旧的参数，改为采用 `--known_ratio` 和 `--unknown_train_ratio` 实现三类别划分。
*   **`[修复]`** 修复了在分配“未知类”标签时，新标签可能会与一个已存在的原始标签发生冲突的严重bug。修正后的逻辑确保了“未知类”标签永远是一个全新的、唯一的标签。
*   **`[修复]`** 新增了**标签重映射**逻辑。确保最终输出给模型的数据标签永远是从0开始的连续整数（如 `[0, 1, 2, ...]`)，解决了因标签不连续导致的 `IndexError: Target out of bounds` 训练错误。
*   **`[修复]`** 修复了在代码重构过程中意外引入的 `IndentationError` (缩进错误)。

### **数据完整性 (Data Integrity)**

*   **`[功能]`** 集成了 `check_gzip_files.py` 脚本，用于检测损坏的 `.gz` 压缩文件。
*   **`[修复]`** 定位并删除了两个由于损坏（提前结束的流）而导致 Spark `EOFException` 错误的 `.json.gz` 文件。
*   **`[改进]`** 创建了一个小型的、类别均衡的数据子集 (`processed_data/small/`)，用于快速验证整个数据处理和训练流程的正确性，极大地提升了调试效率。

### **模型训练 (`train_*.py`, `ml/utils.py`, `ml/model.py`)**

*   **`[重构]`** **(核心)** 彻底移除了对 `datasets` 库的依赖。为了解决一系列棘手的 `FileNotFoundError` 和路径解析问题，重构了 `ml/model.py` 中的数据加载逻辑，改为使用 `pyarrow` 直接读取 Parquet 文件目录，并手动创建 `torch.utils.data.TensorDataset`，从而获得了对数据加载过程更强的确定性和控制力。
*   **`[修复]`** 相应地，适配了模型中的 `training_step` 方法，使其能够正确处理由 `TensorDataset` 输出的元组(tuple)格式的batch数据，而不是旧的字典格式。
*   **`[重构]`** 将 `ml/utils.py` 中训练函数（如 `train_application_classification_resnet_model`）的 `output_dim` 参数从硬编码的固定值修改为可动态传入的参数。
*   **`[重构]`** 相应地，修改了 `train_resnet.py`，使其在训练开始前先读取训练集，动态计算出正确的 `output_dim` (已知类数量 + 1)，再将其传递给训练函数。
*   **`[修复]`** 修复了在 `train_resnet.py` 中因误用Pandas的 `.nunique()` 方法操作PyArrow对象而导致的 `AttributeError`。

### **环境与依赖 (Environment & Dependencies)**

*   **`[修复]`** 解决了多组库之间的版本冲突问题，包括 `pytorch-lightning` vs `torchmetrics` 和 `datasets` vs `pyarrow`。
*   **`[修复]`** **(关键)** 通过将不稳定的 PyTorch Nightly 开发版本替换为官方的稳定发行版，成功解决了导致程序崩溃的 `Segmentation fault: 11` (段错误) 问题。

### **评估 (`evaluation.py`)**

*   **`[功能]`** 创建了一个全新的、独立的评估脚本 `evaluation.py`，将评估流程与训练流程解耦。
*   **`[改进]`** 评估脚本现在可以计算并输出完整的评估报告，包括总体准确率、各类别精确率/召回率/F1分数，以及一个可视化的混淆矩阵图，并能将结果保存到文件中。

---

## 4. Progress Reports

# 研究进展报告

## 1. 总体研究路径概览

| 阶段 | 实验/任务 | 核心目的 | 关键发现/产出 |
| :--- | :--- | :--- | :--- |
| **第一阶段** | `exp1`, `exp2`, `exp3` | 在小型、快速迭代的数据集上，分别对“开放集识别”、“数据不均衡”等核心研究问题进行可行性验证。 | **洞察**：证明了小型、人工构造的数据集对于模型评估存在严重误导性，其上的实验结论不可靠。 |
| **第二阶段** | `exp4`, `exp5` | 演进至更大、更真实的（不均衡）数据集，并建立可靠的实验基准。 | **洞察**：发现标准ResNet性能远超添加了注意力机制的复杂模型，确立了新的基准模型和性能瓶颈（少数类识别）。 |
| **&nbsp;** | *工程重构* | 解决大数据集带来的内存溢出和训练耗时问题。 | **产出**：实现了支持分批处理的数据脚本和带有周期验证的训练流程，保障了后续研究的可行性。 |
| **当前阶段** | `exp6` | 采用数据中心方法，直接应对“数据不均衡”这一核心挑战。 | **产出**：已成功利用SMOTE算法生成了类别均衡的`exp6_smote`训练集，即将开始模型训练与评估。 |

---

## 2. 研究背景与数据集

*   **数据集**: 所有实验均基于 **ISCXVPN2016 NonVPN 数据集**。
*   **核心研究问题**: 
    1.  **开放集识别 (Open-Set Recognition)**: 模型在面对训练期间从未见过的、全新的流量类别时，应如何有效地将其识别为“未知”，而非错误地归类到任何一个已知类别中。
    2.  **数据不均衡适应性 (Imbalanced Data Adaptability)**: 在类别样本数量分布悬殊的真实网络环境下，模型应如何保持其分类性能的稳健性。

---

## 3. 详细实验迭代过程

### 3.1. 第一阶段：基于小型数据集的可行性与基线探索

此阶段的目标是在一系列小规模、可快速处理的数据集上，分别对上述两个核心研究问题进行初步探索，并建立一个“控制变量”基准。

#### 3.1.1. 实验设计

此阶段设计的三个关联子实验及其核心思路如下表所示：

| 实验代号 | 关联研究问题 | 核心目的 | 数据集构建方法 |
| :--- | :--- | :--- | :--- |
| `exp1_small_test` | 开放集识别 | 验证开放集识别方案的技术可行性。 | 每类流量选取一个文件，构建小型均衡数据集。 |
| `exp2_imbalanced_small` | 数据不均衡适应性 | 观察不均衡数据对模型评估指标的影响。 | 人工构造一个类别数量差异巨大的小型数据集。 |
| `exp3_balanced_small` | 基准性能/控制变量 | 检验模型在无干扰的“标准”均衡场景下的真实性能。 | 构建一个与`exp2`规模相仿但类别完全均衡的数据集。 |

#### 3.1.2. 实验结果与分析

| 实验代号 | 关联研究问题 | 关键结果与结论 |
| :--- | :--- | :--- |
| `exp1_small_test` | 开放集识别 | 验证了数据处理、模型训练、评估等整个技术管线的通畅性。但因数据集过小，评估结果无统计学意义。 |
| `exp2_imbalanced_small` | 数据不均衡适应性 | 获得了具有高度误导性的**75.5%**准确率。深入分析发现模型严重偏向多数类，对少数类缺乏有效的识别能力。 |
| `exp3_balanced_small` | 控制实验/基准性能 | 准确率仅为**59%**。这个结果为我们提供了一个极其宝贵的、无偏的性能基线。它证明了`exp2`的高准确率是虚假的，并促使我们思考如何提升模型的基础能力。 |

#### 3.1.3. 阶段性洞察

| 洞察类别 | 具体内容 |
| :--- | :--- |
| **关于实验基准** | **小型、人工构造的数据集是不可靠的**。其规模不足以为模型（特别是较复杂的模型）提供足够的学习信号，其上的实验结论不具备推广性，甚至会得出与在真实数据分布下截然相反的结论（如在`exp3`上错误地否定了注意力机制）。 |
| **关于评估指标** | 在不均衡数据集上，总体准确率（Accuracy）是一个高度不可靠的“虚荣指标”。**宏平均F1分数（Macro F1-Score）**等能够平等对待每个类别的指标，是评估模型真实性能的关键。 |

---

### 3.2. 第二阶段：向可靠基准的演进

基于第一阶段的洞察，此阶段的核心目标是建立一个规模和分布都更接近真实场景的、可靠的实验基准，并在此之上重新进行模型选型。

#### 3.2.1. 面临的挑战与工程实践

在转向更大规模数据集（全量数据的1/600，约2.4万训练样本）的过程中，遇到了两个关键的工程挑战：

1.  **数据生成瓶颈**: 直接使用Spark对大规模数据进行全局操作，会导致`SparkOutOfMemoryError`。
    *   **解决方案**: 对`create_train_test_set.py`进行重构，实现了**分批处理与精确比例采样**功能。新流程通过循环处理小批量文件并合并采样结果，在避免内存溢出的同时，实现了对最终数据集大小的精确控制。
2.  **训练效率瓶颈**: 新数据集使单次训练（epoch）耗时超过1.5小时，严重影响研究效率。
    *   **解决方案**: 对训练框架（PyTorch Lightning）的调用方式进行重构，为训练流程引入了**周期性验证（per-epoch validation）机制**。通过在每个epoch后评估模型在独立验证集上的性能，并利用TensorBoard进行可视化，为科学地判断模型收敛点、缩减不必要的训练时长提供了数据支持。

#### 3.2.2. 模型性能对比 (`exp5`)

在新的`exp5_fractional_1_600`不均衡数据集上，对标准ResNet模型及添加了SEBlock注意力机制的ResNet模型进行了性能对比。

| 模型 | 准确率 | 宏平均F1 | 宏平均Precision | 宏平均Recall |
| :--- | :--- | :--- | :--- | :--- |
| **标准 ResNet** | **<font color='green'>90.8%</font>** | **<font color='green'>0.72</font>** | **<font color='green'>0.73</font>** | **<font color='green'>0.72</font>** |
| ResNet + Attention | 86.2% | 0.52 | 0.52 | 0.55 |

#### 3.2.3. 阶段性洞察

| 洞察类别 | 具体内容 |
| :--- | :--- |
| **关于模型选型** | 标准的ResNet架构在当前任务中表现出强大的鲁棒性和学习能力，是一个非常高的性能基准。 |
| **关于注意力机制** | SEBlock注意力机制在当前不均衡的数据分布下，对模型性能有显著的负面影响。推断其可能被多数类信号主导，从而抑制了对少数类特征的有效学习。 |
| **关于性能瓶颈** | 模型架构已不再是性能的主要瓶颈。当前的核心挑战已明确为**数据不均衡**导致的极端少数类（如类别`0`, `11`）识别失败问题。 |

---

## 4. 当前阶段与下一步计划

### 4.1. 当前最佳模型与核心挑战

*   **当前最佳模型**: 标准ResNet（无注意力机制）。
*   **当前最佳性能**: 在`exp5`不均衡测试集上，总体准确率90.8%，宏平均F1分数**0.72**。
*   **核心挑战**: 提升对极端少数类的识别能力，将宏平均F1分数从0.72的水平进一步提高。

### 4.2. 下一步计划：基于SMOTE的数据中心方法

*   **方向**: 既然核心挑战是数据问题，下一步计划将采用数据中心（Data-Centric）的方法，通过优化训练集来提升模型性能。
*   **具体任务**: 采用SMOTE（合成少数类过采样技术）对`exp5`的训练集进行处理，人工生成少数类的样本，从而创建一个类别完全均衡的训练集。
*   **当前状态**: 
    *   `apply_smote.py`脚本已开发并调试完毕。
    *   已成功生成平衡后的训练数据集`exp6_smote`。
*   **待执行动作**: 
    1.  使用当前最佳模型（标准ResNet），在`exp6_smote`数据集上进行训练。
    2.  在原始的、不均衡的`exp5`测试集上进行评估，检验SMOTE方法对少数类识别能力的提升效果。

---

## 5. 经验与教训总结

| 方面 | 经验与教训 |
| :--- | :--- |
| **实验基准** | **小型或人工构造的数据集是不可靠的**。它们可能无法提供足够的信号来评估复杂模型，甚至会得出与在真实数据分布下截然相反的结论。建立一个规模充足、分布真实的基准数据集是进行科学模型研究的必要前提。 |
| **评估指标** | **宏平均F1分数远比总体准确率更重要**。在类别不均衡的任务中，必须使用能够平等对待所有类别的指标，才能真实地衡量模型的泛化能力，避免被多数类带来的虚高准确率所误导。 |
| **模型复杂度** | **奥卡姆剃刀原理同样适用于此**：更复杂的模型不总是更好。在当前任务中，一个更简洁的ResNet架构表现比增加了注意力机制的复杂版本更为鲁棒。必须警惕复杂组件在特定数据分布下可能带来的负面效应。 |
| **工程实践** | **可扩展的数据管线和可观测的训练流程是研究的加速器**。通过对数据生成脚本和训练脚本的重构，解决了内存瓶颈和效率瓶颈，使得在更大规模数据集上进行快速、可靠的实验成为可能。 |

---

## 5. Conversation Memory

# 实验记忆与上下文总结

**文档目的**: 本文档旨在详细记录“面向开放集识别的增量式加密流量分类技术”项目在AI代理协助下所进行的全部实验步骤、遇到的问题、调试过程以及最终确定的方案，以供随时回顾和无缝衔接后续工作。

**报告日期**: 2025年8月30日

---

## 初始目标与规划

我们最初的目标是验证一个基线分类模型在两个关键场景下的表现：

1.  **实验一：开放集识别能力**：测试模型面对从未见过的“未知”应用类别的识别能力。
2.  **实验二：数据不均衡适应性**：测试模型在训练数据类别极不均衡时的性能表现。

---

## 第一阶段：实验一的首次尝试 (CNN模型)

此阶段充满了挑战，我们通过一系列的失败和修复，最终得到了初步但关键的结论。

### 1. 首次执行与环境修复

*   **遇到的问题**: 在尝试训练`CNN`模型的过程中，我们遭遇了一连串的环境依赖和代码错误：
    1.  **`ImportError`**: `pytorch_lightning` 与 `torchmetrics` 版本不兼容。
        *   **解决方案**: 将 `torchmetrics` 降级到 `0.10.3`。
    2.  **`AttributeError`**: `datasets` 库与 `pyarrow` 版本不兼容。
        *   **解决方案**: 将 `pyarrow` 降级到 `8.0.0`。
    3.  **`Segmentation fault: 11`**: 底层库二进制不兼容，根本原因是使用了不稳定的 PyTorch Nightly 版本。
        *   **解决方案**: 将 `torch`, `torchvision`, `torchaudio` 从dev版本替换为匹配的稳定版本 (`1.12.1`等)。
    4.  **`FileNotFoundError`**: `datasets` 库在解析文件路径时出现问题，无法正确找到数据文件。
        *   **解决方案 (最终方案)**: 彻底放弃使用 `datasets` 库加载数据。重构了 `ml/model.py` 中的 `train_dataloader` 方法，改用 `pyarrow` 直接读取 Parquet 文件，并手动创建 `torch.utils.data.TensorDataset`。同时，适配了 `training_step` 以处理新的数据格式。

### 2. 初步评估与结论

*   **评估**: 在解决了所有环境和代码问题后，我们成功训练了CNN模型。为了评估它，我们创建了一个新的 `evaluation.py` 脚本。
*   **结论**: **实验假设被证伪**。评估结果显示，标准的CNN模型完全无法识别任何“未知类”，它将所有未知样本都强行归类到了它唯一认识的那个“已知类”中。这个结果虽然是“失败”的，但它清晰地证明了**采用更专业的开放集识别方案的必要性**。

---

## 第二阶段：实验一的最终方案 (ResNet模型)

在您指出初始实验设计的缺陷后，我们共同制定了一套更严谨、更科学的实验方案。

### 1. 最终方案设计

*   **核心思想**: 不再依赖模型的“不确定性”，而是直接训练模型去“认识”什么叫“未知”。
*   **基准模型**: 采纳您的建议，将基准模型从 `CNN` 升级为 `ResNet`。
*   **数据划分**: 将所有类别随机划分为三组：
    *   **已知类 (30%)**: 用于训练和测试的核心类别。
    *   **训练期未知类 (30%)**: 在训练时，它们的标签被统一映射到一个新增的“未知”标签，用于教会模型识别“未知”这一概念。
    *   **测试期未知类 (40%)**: 完全不参与训练，在测试时也映射到“未知”标签，用于检验模型的泛化能力。

### 2. 代码实现与调试

*   **`create_train_test_set.py` 重构**: 对该脚本进行了大规模修改，以支持上述的三类别划分和标签重映射逻辑。
*   **`ml/utils.py` 和 `train_resnet.py` 适配**: 修改了这两个文件，使模型输出维度可以根据训练数据动态确定。
*   **遇到的问题**: 在运行重构后的数据生成脚本时，我们再次遇到了一系列新的问题：
    1.  **`ValueError: max() arg is an empty sequence`**: 原因是当时 `processed_data` 目录中只有3个类别的数据，按比例划分后“已知类”列表为空。
        *   **解决方案**: 您向 `processed_data` 中补充了更多类别的数据（从3个增加到5个，后增加到7个），使划分可以继续。
    2.  **标签冲突**: 发现“未知类”的新标签可能会与某个原始标签重复。
        *   **解决方案**: 修正了代码，确保“未知类”的新标签永远是 `max(所有原始标签) + 1`。
    3.  **`IndentationError`**: 修复上一个bug时引入了错误的缩进。
        *   **解决方案**: 修复了缩进。
    4.  **`java.io.EOFException`**: Spark底层报错，指示有损坏的 `.gz` 文件。
        *   **解决方案**: 运行您提供的 `check_gzip_files.py` 脚本，定位到两个损坏文件，并使用 `rm` 命令将其删除。

---

## 第三阶段：小型数据集上的完整基准测试

为了快速迭代和验证整个流程，我们决定先在一系列小型的、可控的数据集上进行实验。

### 1. 实验一 (开放集) 在小型数据集上的验证 (`exp1_small_test`)

*   **数据**: 创建了一个包含所有类别、但每个类别只有少量均衡样本的小型数据集。
*   **结果**: 成功在该数据集上完整地执行了开放集识别实验的**数据生成、模型训练和模型评估**三个步骤。
*   **结论**: 整个流程技术上是通畅的。但由于数据集过小，评估结果（如混淆矩阵）的参考价值有限。

### 2. 实验二 (不均衡) 在小型数据集上的验证 (`exp2_imbalanced_small`)

*   **数据**: 创建了一个小型但不均衡的数据集，其中Facebook应用的样本量远超其他类别。
*   **结果**: 成功训练并评估了ResNet模型。
*   **结论**: 评估结果（准确率75.5%）具有**高度误导性**。混淆矩阵显示，模型严重偏向于预测多数类，而对少数类的识别能力极差（多个类别的F1分数为0）。这清晰地证明了**在不均衡数据下，只看准确率是不可靠的，必须关注宏平均F1分数等指标**。

### 3. 实验三 (均衡) 在小型数据集上的验证 (`exp3_balanced_small`)

*   **数据**: 创建了一个与`exp2`规模相仿，但所有类别样本数完全均衡的数据集。
*   **结果**: 成功训练并评估了ResNet模型。
*   **结论**: 在这个“理想”的均衡环境下，模型的**真实性能暴露无遗，准确率仅为59%**。这个结果本身虽然不佳，但它为我们提供了一个**极其宝贵的、无偏的性能基线**。它证明了`exp2`中的高准确率是虚假的，并促使我们思考如何提升模型的基础能力。

---

## 第四阶段：模型改进的首次尝试 (注意力机制)

基于实验三暴露出的模型性能不足问题，我们决定尝试对模型架构进行改进。

### 1. 方案设计与实现

*   **思路**: 采纳`proposal.md`中的建议，为ResNet模型引入**SEBlock（Squeeze-and-Excitation Block）注意力机制**，期望能增强模型的特征表示能力。
*   **实现**:
    *   修改了 `ml/model.py`，为 `ResNet` 类增加了 `SEBlock` 模块和 `use_attention` 控制参数。
    *   修改了 `train_resnet.py`，增加了 `--use_attention` 命令行标志，用于控制是否启用注意力机制。

### 2. 在小型均衡数据集 (`exp3`) 上的评估

*   **结果**: **实验失败**。增加了注意力机制后，模型的性能**反而从59%下降到了56%**。
*   **核心洞察**: 这次失败引出了一个至关重要的结论——**我们一直在使用的小型数据集是不可靠的**。它可能太小，无法为更复杂的注意力机制提供足够的学习信号，甚至可能导致过拟合，学到一些伪特征。因此，任何基于该数据集的实验结论都可能是无效的。

---

## 当前状态与下一步

*   **当前状态**: 我们已经意识到，所有后续的实验都必须在一个规模更大、更具代表性的数据集上进行，才能得出可靠的结论。
*   **下一步**: 
    1.  **生成中等规模数据集**: 放弃小型数据集，从全量数据中采样10%，创建一个中等规模的数据集 (`exp4_medium_balanced`)。
    2.  **重新建立基准**: 在这个新数据集上，重新训练一个不带注意力的ResNet模型，以获取一个新的、可靠的性能基线。
    3.  **重新评估注意力**: 同样在新数据集上，训练一个带注意力的ResNet模型。
    4.  **最终对比**: 对比上述两个模型的结果，以最终判断注意力机制的有效性。

在执行第一步（生成中等规模数据集）的过程中，我们遇到了严重的 `SparkOutOfMemoryError`，这表明我们的数据处理流程也需要针对大数据集进行优化。

---
**报告日期**: 2025年9月6日

## 第五阶段：向大数据集和更优数据流的演进

### 1. 解决数据生成瓶颈

*   **挑战**: 直接使用Spark对10%的全量数据进行`repartition`和`orderBy`操作，在本地模式下会导致严重的内存溢出。
*   **解决方案**:
    1.  **初步方案**: 放弃在Spark内做全局平衡，改为在文件系统层面随机抽样10%的`.json.gz`文件。这成功生成了`exp4`数据集，但后续发现该数据集被错误地进行了“欠采样”处理，依然是均衡的。
    2.  **最终方案 (脚本重构)**: 为了实现对数据集大小的精确控制并从根本上解决内存问题，对`create_train_test_set.py`进行了大规模重构。
        *   **核心改进**: 实现了**分批处理**和**精确比例采样**功能。脚本现在可以逐批读取源文件，对每批数据进行行采样，最后将结果合并。
        *   **成果**: 新的流程彻底解决了内存溢出问题，并能按任意比例（如1/600）创建大小可控、分布真实的不均衡数据集。

### 2. 解决训练效率瓶颈

*   **挑战**: 在新的、更大的数据集上，每个epoch的训练时间长达1小时40分钟，严重影响迭代效率。
*   **解决方案 (训练流程优化)**:
    *   **核心改进**: 对`ml/model.py`和`ml/utils.py`进行了重构，为训练流程引入了**周期性验证 (per-epoch validation)**机制。
    *   **成果**: 现在可以在每个epoch结束后，通过独立的验证集来评估模型性能，并使用**TensorBoard**将`val_loss`等指标可视化。这为科学地判断模型收敛点、缩减不必要的训练epoch数提供了数据支持。

### 3. 在新基准 (`exp5`) 上的最终模型对决

*   **数据**: 使用重构后的脚本，创建了`exp5_fractional_1_600`数据集（约占全量数据的1/600）。
*   **结果**:
    *   **标准ResNet**: 表现出色，取得了**90.8%**的准确率和**0.72**的宏平均F1分数。
    *   **ResNet+注意力**: 表现糟糕，宏平均F1分数仅为**0.52**。
*   **最终结论**:
    1.  **注意力机制不适用**: SEBlock注意力机制在当前任务的真实数据分布下，会产生负面作用。
    2.  **确立新基准**: 标准ResNet模型被确立为新的、性能强大的基准模型。
    3.  **明确核心矛盾**: 模型在极端少数类上的识别能力差，是限制整体性能（宏平均F1分数）的主要瓶颈。

---

## 当前状态与下一步

*   **当前状态**:
    1.  拥有了一套健壮、可扩展的数据生成与训练验证流程。
    2.  确立了一个性能强大的基准模型（标准ResNet）和基准分数（0.72 Macro F1-Score）。
    3.  明确了当前的核心挑战是**数据不均衡**导致的少数类识别问题。
*   **下一步**:
    *   **A轨 (数据中心)**: 采用**SMOTE**（合成少数类过采样技术）对`exp5`的训练集进行数据增强，以期在不改变优秀模型架构的前提下，专门提升对少数类的识别能力。
    *   **当前进展**: `apply_smote.py`脚本已开发调试完成，并成功生成了平衡后的`exp6_smote`训练集。
    *   **即将执行**: 在`exp6_smote`上重新训练标准ResNet模型，并进行最终评估。