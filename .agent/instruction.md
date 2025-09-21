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

## 5. Conversation Memory

# 实验记忆与上下文总结

**文档目的**: 本文档旨在详细记录“面向开放集识别的增量式加密流量分类技术”项目在AI代理协助下所进行的全部实验步骤、遇到的问题、调试过程以及最终确定的方案，以供随时回顾和无缝衔接后续工作。

**报告日期**: 2025年9月13日

---

## 第六阶段：系统性解决数据不均衡问题

在`exp5`确立了强大的性能基准（标准ResNet, Macro F1: 0.72）后，我们将核心矛盾聚焦于**解决数据不均衡问题**。

### 1. 两种主流方法的证伪

我们依次尝试了两种主流方案，但均以失败告终，并得到了宝贵的否定性结论：

*   **`exp6` - 数据中心方法 (SMOTE)**: 尝试通过过采样技术创造一个均衡的数据集。
    *   **结果**: **失败** (Macro F1: 0.31)。模型严重过拟合于合成数据，泛化能力大幅下降。
    *   **结论**: 简单的过采样不适用于此任务。

*   **`exp7` - 算法层方法 (Focal Loss)**: 尝试通过修改损失函数，使其关注少数类和难分样本。
    *   **结果**: **灾难性失败** (Macro F1: 0.09)。训练过程极不稳定，模型性能完全崩溃。
    *   **结论**: 简单的静态或动态加权损失方案，在我们的极端数据分布下亦不可行。

### 2. 代码库清理与回退

在实验间隙，我们完成了两项重要的代码维护工作：
1.  **移除了被证伪的注意力机制 (SEBlock)** 的全部相关代码。
2.  在`exp7`失败后，**回退了Focal Loss**的全部相关代码。

当前，代码库处于一个移除了所有无效尝试的、干净的基线状态。

### 3. 当前共识与下一阶段：混合专家 (MoE)

*   **核心洞察**: 两次关键实验的失败，让我们达成共识——依赖单一的“全才”模型已无法突破性能瓶颈。
*   **下一步 (`exp8`)**: 采纳了您提出的、极具前瞻性的建议，我们将研究方向转向一个更强大的高级架构：**混合专家网络 (MoE)**。
*   **目标**: 设计并实现一个包含“门控网络”和多个“专家网络”的MoE模型，让不同的专家处理不同类型的流量（如多数类 vs 少数类），从根本上解决单一模型无法兼顾的矛盾。

```json