# Thesis Revision Checklist
**Generated**: 2026-02-15
**Purpose**: Specific actionable fixes for thesis content issues

---

## 🔧 High Priority Fixes

### Fix 1: Chapter 5 Figure Numbering

**Issue**: Figure 5-1 appears twice, causing numbering confusion

**Location**: `thesis/chapters/chapter5_theoretical_analysis.md`

**Changes needed**:

1. **Line 15**: Change to Figure 5-1
   ```
   OLD: **[图5-1: 门控网络权重分布热力图（第一层权重矩阵）- 将插入此处]**
   NEW: **[图5-1: 门控网络权重分布热力图（第一层权重矩阵）- 将插入此处]**
   (Keep as-is, this is the first 5-1)
   ```

2. **Line 33**: Change to Figure 5-2
   ```
   OLD: **[图5-1: 门控网络权重热力图 + 输出贡献柱状图 - 将插入此处，展示两个子图]**
   NEW: **[图5-2: 输出贡献柱状图（基准模型 vs 专家模型在不同类别上的贡献）- 将插入此处]**
   ```

3. **Line 47**: Change to Figure 5-3
   ```
   OLD: **[图5-2: 决策边界可视化（Baseline vs Expert vs GEE）- 将插入此处，展示三个子图]**
   NEW: **[图5-3: 决策边界可视化（Baseline vs Expert vs GEE）- 将插入此处，展示三个子图]**
   ```

4. **Line 79**: Change to Figure 5-4
   ```
   OLD: **[图5-4: 梯度贡献对比 + 训练曲线（标准CE vs 加权CE）- 将插入此处，展示两个子图]**
   NEW: **[图5-4: 梯度贡献对比 + 训练曲线（标准CE vs 加权CE）- 将插入此处，展示两个子图]**
   ```

5. **Line 99**: Change to Figure 5-5
   ```
   OLD: **[图5-3: 特征空间对比（无垃圾类 vs 有垃圾类）- 将插入此处，展示两个子图]**
   NEW: **[图5-5: 特征空间对比（无垃圾类 vs 有垃圾类）- 将插入此处，展示两个子图]**
   ```

**Check**: Verify no text references to old numbers need updating (search for "图5-1", "图5-2", "图5-3" in text)

---

### Fix 2: Standardize "Baseline" Terminology

**Issue**: Inconsistent use of "基线模型" vs "基准模型"

**Primary term**: "基准模型" (used in Ch1, Ch3, Ch6)
**To be replaced**: "基线模型" → "基准模型"

**Location 1**: `thesis/chapters/chapter4_experiments.md`

**Find and replace**:
```
Line 41: ### 4.2.2 ResNet-Baseline基线结果
       → ### 4.2.2 ResNet基准模型实验结果

Line 43: 表4-1展示了ResNet-Baseline在6折交叉验证中的OSR性能。
       → 表4-1展示了ResNet基准模型在6折交叉验证中的OSR性能。

Line 45: **[表4-1: 开放集识别性能对比（ResNet-Baseline vs ResNet-GEE）- 将插入此处]**
        → **[表4-1: 开放集识别性能对比（ResNet基准模型 vs ResNet-GEE）- 将插入此处]**
```

**Additional replacements throughout Chapter 4**:
- "Baseline" (when referring to model) → "基准模型"
- "ResNet-Baseline" → "ResNet基准模型"
- "CNN基线" → "CNN基准模型" (if found)

**Note**: Keep "Baseline" in English when:
- It's part of a proper name (e.g., "Baseline methods")
- It's in figure/table titles for consistency
- After first mention when alternating Chinese/English

---

### Fix 3: Add Citations to Chapter 5

**Issue**: Chapter 5 lacks citations for theoretical foundations

**Location**: `thesis/chapters/chapter5_theoretical_analysis.md`

**Citations to add** (using existing reference numbers):

#### Section 5.1: Gating Network Decision Patterns (after line 20)
```
ADD: 门控网络作为混合专家模型的核心组件，最早由Bengio等人[19]提出，用于实现模型的条件计算。Shazeer等人[18]进一步提出了稀疏门控的混合专家层，通过门控机制实现专家的选择性激活。
```

#### Section 5.1: Weight Distribution Analysis (after line 30)
```
ADD: 相关研究表明，门控网络能够学习到复杂的融合策略[18, 21]。本实验观察到...
```

#### Section 5.3: Gradient Analysis (after line 70)
```
ADD: 加权交叉熵损失是处理类别不平衡问题的经典方法，其核心思想是为不同类别的损失分配不同权重[14]。类别不平衡会导致标准梯度下降算法偏向多数类，这一现象已被广泛研究和验证[7]。
```

#### Section 5.3: Training Trajectory (after line 75)
```
ADD: 优化理论表明，损失函数的梯度方向直接决定了模型参数的更新路径[45]。本研究中的梯度对比分析揭示了...
```

#### Section 5.4: Feature Space with Garbage Class (after line 90)
```
ADD: 开放集识别的核心挑战在于如何学习有效的特征空间以区分已知类和未知类[20, 31]。Bendale和Boult[20]首次系统性地定义了开放集识别问题，并提出将未知样本映射到特征空间边界的思路。垃圾类机制与这一思想高度契合...
```

#### Section 5.4: Clustering Analysis (after line 100)
```
ADD: 特征空间的聚类性质对于开放集识别至关重要[31]。本实验通过t-SNE可视化发现...
```

**Estimated citations to add**: ~10-15
**Time estimate**: 30-45 minutes

---

### Fix 4: Add Citations to Chapter 6

**Issue**: Chapter 6 lacks citations for future work

**Location**: `thesis/chapters/chapter6_conclusion.md`

**Citations to add**:

#### Section 6.3: Future Work - More Experts (after line 29)
```
ADD: 扩展到更多专家模型的理论基础已在混合专家模型的研究中奠定[18, 55]。然而，如何避免专家坍塌(expert collapse)[18]问题、保持训练稳定性，仍需深入研究。
```

#### Section 6.3: More Datasets (after line 31)
```
ADD: 加密流量分类领域已有多个公开数据集，如ISCX VPN-nonVPN[51]、ISCX Tor-nonTor[52]、USTC-TFC2016[56]等。在这些数据集上验证GEE架构的泛化能力，是未来工作的重要方向。
```

#### Section 6.3: Theoretical Analysis (after line 33)
```
ADD: 从信息论角度解释模型泛化能力的研究已有较多积累[30, 44]。如何将信息瓶颈理论[30]、互信息分析等方法应用于GEE架构，从理论上解释其有效性，是值得深入探索的方向。
```

#### Section 6.3: Joint Training (after line 35)
```
ADD: 联合训练多模块系统已在计算机视觉领域得到广泛应用[25, 28]。Li等人[28]提出的"Learning Without Forgetting"框架为多模型协同优化提供了思路。然而，GEE架构的特殊性在于需要同时处理开放集识别和增量学习两大任务，这为联合训练带来了新的挑战。
```

**Estimated citations to add**: ~5-8
**Time estimate**: 20-30 minutes

---

## 🔧 Medium Priority Fixes

### Fix 5: Handle Chapter 3 Figure Placeholders

**Issue**: Three figure placeholders in Chapter 3 not in original plan

**Location**: `thesis/chapters/chapter3_gee_architecture.md`

**Options**:

#### Option A: Remove Placeholders (Recommended)
```
Line 7: **[图3-1: GEE架构整体框架图 - 占据本节核心位置，展示三大组件及其连接关系]**
→ Replace with 2-3 sentences describing the architecture:
  "GEE架构由三大核心组件构成：基准模型、少数类专家模型和门控网络。基准模型在全部已知类上训练，擅长处理多数类；专家模型仅在少数类上训练，专注于优化少数类决策边界；门控网络通过可学习的融合策略，动态决定在特定样本上信任哪个模型。"

Line 41: **[图3-2: 门控网络内部架构 - 展示MLP结构、输入输出设计、与基准模型和专家模型的连接]**
→ Replace with:
  "门控网络采用MLP(Multi-Layer Perceptron)结构，包含两个隐藏层(128→64神经元)，使用ReLU激活函数。输入层接收基准模型和专家模型的拼接概率输出(维度为2×N，N为类别数)，输出层产生N维的融合概率向量。"

Line 69: **[图3-3: 垃圾类机制示意图 - 展示N+1输出、未知类样本训练、置信度计算方式]**
→ Replace with:
  "垃圾类机制将门控网络的输出维度从N扩展到N+1，第N+1个神经元专门用于识别"未知"类别。训练时，将未知类样本统一标记为第N+1类，与已知类样本共同训练。测试时，通过'1-垃圾类概率'计算置信度，而非传统的'最大Softmax概率'。"
```

#### Option B: Create Simple Diagrams
- Use draw.io or similar tool to create 3 simple architecture diagrams
- Estimate time: 2-3 hours

#### Option C: Merge into Single Figure
- Combine all 3 into one comprehensive "GEE架构" figure
- Update placeholder to: `[图3-1: GEE架构完整示意图（包含三大组件、门控网络结构、垃圾类机制）]`
- Estimate time: 1-2 hours

**Recommendation**: Option A (remove placeholders and describe in text)

---

## ✅ Verification Checklist

After completing all fixes, verify:

- [ ] All figures in Chapter 5 numbered sequentially (5-1 through 5-5)
- [ ] No duplicate figure numbers anywhere in thesis
- [ ] "基线模型" replaced with "基准模型" throughout
- [ ] Chapter 5 has 10-15 new citations added
- [ ] Chapter 6 has 5-8 new citations added
- [ ] All citations match reference list (numbers 1-65)
- [ ] All metric values consistent across chapters (FPR@TPR95, Macro-F1, etc.)
- [ ] No broken section numbering
- [ ] Figure/table references in text match actual figure/table numbers
- [ ] Chapter transitions smooth and logical

---

## 📊 Fix Summary

| Fix | Priority | Files Affected | Est. Time |
|-----|----------|----------------|-----------|
| Fix 1: Chapter 5 figure numbering | 🔴 High | chapter5_theoretical_analysis.md | 30 min |
| Fix 2: Baseline terminology | 🔴 High | chapter4_experiments.md | 15 min |
| Fix 3: Add citations to Ch5 | 🔴 High | chapter5_theoretical_analysis.md | 45 min |
| Fix 4: Add citations to Ch6 | 🔴 High | chapter6_conclusion.md | 30 min |
| Fix 5: Chapter 3 figures | 🟡 Medium | chapter3_gee_architecture.md | 15 min (Option A) |
| **Total** | | | **2-2.5 hours** |

---

## 🎯 After Fixes Complete

1. **Re-run consistency check** (optional)
2. **Merge all chapters** into `thesis/main.md`
3. **Generate figures** using Python scripts
4. **Insert figures** into document
5. **Final formatting** (headers, page numbers, etc.)
6. **Generate PDF** via Pandoc/LaTeX
7. **Final review** before submission

---

**Next step**: Would you like me to:
1. Apply Fix 1 (Chapter 5 figure numbering) now?
2. Apply Fix 2 (baseline terminology) now?
3. Apply all high-priority fixes at once?
4. Create the merged `thesis/main.md` document first?
