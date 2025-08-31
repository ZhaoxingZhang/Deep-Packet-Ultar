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
