# Proposal: Thesis Chapter 1 Enhancement - Systematic Application of Academic Writing Principles

## Why

The user provided 7 modification examples that demonstrate key academic writing principles. However, these are EXAMPLES - the actual task is to apply these principles SYSTEMATICALLY to review and improve ALL of Chapter 1, not just mechanically fix the 7 example points.

### Core Academic Writing Principles (Extracted from Examples)

1. **技术背景先于问题陈述** - 在说"加密让DPI失效"前，必须先解释加密技术原理
2. **定性主张需要定量支撑** - "存在根本性缺陷"必须用具体数据（FPR@TPR95、F1分数等）支撑
3. **创新前先交代传统方法** - 说GEE创新应用MoE前，先解释传统MoE是什么、有什么挑战
4. **正确归属创新** - 是"创新性地应用"现有技术，不是"发明"现有技术
5. **技术表述精确性** - "CNN模型"不清楚 → "骨干网络从ResNet替换为CNN"
6. **学术语言精确性** - "不遗忘"太口语 → "保持识别精度"
7. **准确描述工作范围** - "多模型验证"不足以概括 → "多维度验证实验与理论分析"
8. **数据表达一致性** - 全文统一使用比例/百分比，避免使用绝对样本数

### Why These Principles Matter

These principles ensure academic rigor by:
- Providing necessary context for readers to appreciate innovations
- Making claims verifiable through quantitative evidence
- Properly attributing contributions to maintain academic integrity
- Using precise language that accurately conveys technical meaning
- Maintaining consistency in data presentation throughout the thesis

## What Changes - Systematic Review Results

After applying the 8 principles to review ALL of Chapter 1 (lines 25-90), the following issues were identified:

### Section 1.1.1 课题来源

#### Issue 1.1: Missing Encryption Technology Principles (原则1: 技术背景先于问题)

**Location:** Line 31
**Current:** "随着网络安全意识的提升，HTTPS、VPN、Tor等加密技术的使用率大幅提高。根据相关统计显示，目前互联网流量中超过80%为加密流量。传统的网络流量分类方法主要依赖于端口号识别和深度包检测（Deep Packet Inspection, DPI），但这些方法在加密流量面前已逐渐失效。"

**Problem:** Mentions encryption technologies and immediately states DPI fails, without explaining WHY encryption makes DPI ineffective.

**Required Fix:**
- Before "但这些方法在加密流量面前已逐渐失效"，插入加密技术原理说明：
  - 加密技术的基本原理：通过对数据进行加密转换，使得只有持有密钥的接收方才能解密
  - 对称加密和非对称加密的基本概念
  - 为什么DPI失效：DPI依赖读取载荷内容来识别应用，但加密后载荷内容不可见
  - 端口号识别为什么失效：现代应用使用动态端口、端口混淆技术
- 保持简洁（2-3段），重点是让非安全背景的读者理解问题本质

#### Issue 1.2: Sample Count Absolute Values (原则8: 使用比例代替绝对数量)

**Location:** Line 53
**Current:** "VPN数据集中类别10有9476个样本，而类别7仅有65个样本，样本数差距达146倍。"

**Problem:** Uses absolute sample counts (9476, 65), inconsistent with thesis principle of using ratios.

**Required Fix:**
- 改为："VPN数据集中，最大类别（类别10）的样本数是最小类别（类别7）的146倍。"
- 或者："类别10与类别7的样本数量比为146:1，呈现极端不均衡分布。"
- 全文统一：不再出现具体的9476、65等绝对数字，只用比例/百分比

#### Issue 1.3: Absolute Sample Counts in Research Content (原则8)

**Location:** Line 63
**Current:** "样本数分别为215和65。"

**Required Fix:**
- 改为："分别占总样本的X%和Y%"或"样本数比为215:65 ≈ 3.3:1"
- 检查全文所有出现样本数量的地方，统一改为比例表达

### Section 1.1.2 研究意义

#### Issue 2.1: Missing MoE Background (原则3: 创新前先交代传统)

**Location:** Line 41
**Current:** "该架构创新性地将混合专家模型（Mixture-of-Experts, MoE）思想应用于加密流量分类领域，并通过可学习的门控网络实现了模型融合..."

**Problem:** Jumps directly to GEE's innovation without explaining what traditional MoE is.

**Required Fix:**
在"该架构创新性地将MoE思想..."之前，添加传统MoE背景：
- 传统MoE架构包含多个专家网络（通常8-32个）
- 通过门控机制动态选择专家
- 训练复杂度高：需要平衡负载、确保专家专业化
- 计算开销大：多个前向传播、参数量大
- 然后引出GEE创新：证明了两个专家的MoE也能达到很好效果，大幅简化了架构

#### Issue 2.2: Incorrect Innovation Attribution (原则4: 正确归属创新)

**Location:** Line 41
**Current:** "第三，提出了加权交叉熵损失训练策略，解决了门控网络在类别极度不均衡数据上的失效问题。"

**Problem:** Claims to "propose" weighted CE loss, which is an established technique.

**Required Fix:**
- 改为："第三，创新性地将加权交叉熵损失应用于门控网络训练。加权损失是处理类别不均衡的现有技术，本研究将其成功迁移到MoE门控机制中，解决了门控网络在极端不均衡数据下的失效问题。"
- 明确：创新在于"应用"而非"发明"

#### Issue 2.3: Ambiguous Transferability Claim (原则5: 技术表述精确性)

**Location:** Line 43
**Current:** "...且该架构具有良好的可迁移性，在CNN模型上也取得了类似的性能提升。"

**Problem:** "CNN模型"指代不明 - 是什么换成CNN？

**Required Fix:**
- 首先在前文中确立：Deep-Packet使用ResNet1d作为骨干网络
- 然后改为："当将Deep-Packet框架的基础骨干网络从ResNet替换为标准CNN后，GEE架构仍然展现出显著的性能提升优势，证明了该架构设计不依赖于特定的骨干网络选择，具有广泛的适用性。"
- 确保读者理解：GEE架构不变，只是骨干网络换了

### Section 1.2.1 研究目标

#### Issue 3.1: Imprecise Wording (原则6: 学术语言精确性)

**Location:** Line 53
**Current:** "...设计能够从少量样本中学习新类别且不遗忘旧类别的IL架构..."

**Problem:** "不遗忘"过于口语化，不够精确。

**Required Fix:**
- 改为："...设计能够从少量样本中学习新类别且保持旧类别识别精度的IL架构..."
- 或："...在学习新类别的同时，维持对旧类别的识别性能..."

#### Issue 3.2: More Absolute Sample Counts (原则8)

**Location:** Line 53
**Current:** "VPN数据集中类别10有9476个样本，而类别7仅有65个样本，样本数差距达146倍。"

**Required Fix:** (见 Issue 1.2)

### Section 1.2.2 研究内容

#### Issue 4.1: Incomplete Validation Description (原则7: 准确描述工作范围)

**Location:** Line 69
**Current:** "第六，多模型验证与理论分析。"

**Problem:** 不足以概括全面的实验验证工作。

**Required Fix:**
- 改为："第六，多维度的验证实验与理论分析。本研究不仅验证了不同骨干网络（ResNet、CNN）下的架构有效性，还探索了不同门控网络架构、不同数据集场景下的性能表现。通过控制变量的消融实验，系统地验证了GEE架构各组件的贡献。此外，从门控网络权重分布、决策边界几何特性、梯度贡献机制等多个角度进行了深入的理论分析，揭示了GEE架构有效的深层机制。"
- 准确反映：多骨干网络、多门控架构、多数据集、消融实验、多角度理论分析

#### Issue 4.2: More Absolute Sample Counts (原则8)

**Location:** Line 61, 63
**Current:** "样本数量较多（均在1000以上）"、"样本数分别为215和65"

**Required Fix:**
- Line 61: 改为"样本占比超过X%"
- Line 63: 改为"分别占总样本的X%和Y%"或使用比例

#### Issue 4.3: Quantify "Small Sample" (原则2: 定量支撑)

**Location:** Line 63
**Current:** "...专门用于处理小样本类别。"

**Problem:** "小样本"是定性的，没有定量说明。

**Required Fix:**
- 明确什么算"小样本"：如"样本数少于总样本5%的类别"
- 或者直接给出比例："少数类样本仅占总样本的X%"

### Section 1.3 论文组织结构

#### Issue 5.1: Missing Backbone Context (原则5)

**Location:** Throughout section
**Problem:** When referencing ResNet, CNN, should clarify these are "backbone networks" (骨干网络).

**Required Fix:**
- 确保提到ResNet、CNN时，明确说明是"骨干网络"
- 建立术语：GEE架构 + 骨干网络 = 完整模型

### Section 1.4 本章小结

#### Issue 6.1: Review for Consistency (所有原则)

**Required Actions:**
- 检查是否使用了绝对样本数，改为比例
- 检查技术表述是否精确
- 检查创新归属是否正确
- 检查是否有定性主张缺少定量支撑

---

## Summary of Required Modifications

Based on systematic review, **6 major sections** with **12 specific issues** identified:

1. **Section 1.1.1**: 3 issues (encryption principles, 2× absolute counts)
2. **Section 1.1.2**: 3 issues (MoE background, innovation attribution, transferability)
3. **Section 1.2.1**: 2 issues (wording precision, absolute counts)
4. **Section 1.2.2**: 3 issues (validation description, absolute counts, quantify "small")
5. **Section 1.3**: 1 issue (backbone terminology)
6. **Section 1.4**: 1 issue (consistency review)

**Key Changes:**
- Add encryption technology principles (Issue 1.1)
- Add MoE background (Issue 2.1)
- Correct innovation attribution (Issue 2.2)
- Clarify transferability (Issue 2.3)
- Refine wording (Issue 3.1)
- Expand validation description (Issue 4.1)
- **Replace ALL absolute sample counts with ratios/percentages** (Issues 1.2, 1.3, 3.2, 4.2)

## Capabilities

### New Capabilities
- `thesis-chapter-enhancement`: Framework for systematically improving thesis chapters by applying academic writing principles
- `principle-based-review`: Methodology for reviewing academic text against established writing principles
- `data-consistency-check`: Systematic replacement of absolute values with ratios/percentages

### Modified Capabilities
- None (this is content-specific to the thesis)

## Impact

### Affected Files
- `/thesis/main.md` - Chapter 1 (lines 25-90, approximately 65 lines of content)
  - Section 1.1.1: Lines 31-35
  - Section 1.1.2: Lines 37-43
  - Section 1.2.1: Lines 47-55
  - Section 1.2.2: Lines 57-69
  - Section 1.3: Lines 71-85
  - Section 1.4: Lines 87-89

### Documentation Changes
- Technical principles added: Encryption technology background (2-3 paragraphs)
- Background added: Traditional MoE architecture description
- All innovation claims reformulated for proper attribution
- Technical statements clarified with precise terminology
- Academic language refined for precision and formality
- **CRITICAL: All absolute sample counts replaced with ratios/percentages**

### Dependencies
- Need to verify ratios match actual experimental data from Chapters 4-5
- May require citation of encryption technology reference materials
- Should align terminology with Chapters 2-3 (MoE, GEE, backbone, etc.)
- Cross-references to experimental sections must be accurate
- **Must maintain consistency with any mentions of sample ratios in later chapters**

### Success Criteria

#### Completeness Criteria
1. ✅ All 12 identified issues addressed across 6 sections
2. ✅ Encryption technology principles added before DPI failure explanation
3. ✅ Traditional MoE background added before GEE innovation description
4. ✅ All innovation claims properly attributed (application, not invention)
5. ✅ Transferability claims clarified with backbone network context
6. ✅ All imprecise wording refined to academic standards
7. ✅ Validation description expanded to reflect comprehensive work
8. ✅ **ALL absolute sample counts replaced with ratios/percentages**

#### Quality Criteria
9. ✅ Technical explanations accurate yet concise (encryption ~200-300 words)
10. ✅ All qualitative claims have quantitative support
11. ✅ Background → Innovation narrative flows logically
12. ✅ Academic integrity maintained (proper attributions)
13. ✅ Technical precision achieved (clear, unambiguous statements)
14. ✅ Formal academic language used throughout
15. ✅ Validation description accurately reflects scope of work
16. ✅ **Consistent data presentation: no absolute sample counts anywhere**
17. ✅ Changes maintain coherence with rest of thesis
18. ✅ No contradictions or inconsistencies introduced

#### Validation Criteria
19. ✅ All cross-references point to correct sections
20. ✅ Terminology consistent across all chapters
21. ✅ Chinese academic writing conventions followed
22. ✅ No grammatical or stylistic errors introduced

## Important Implementation Notes

### Global Principle: Data Presentation Consistency

**CRITICAL:** This thesis uses ratios/percentages, NOT absolute sample counts.

**Why:** Ratios are more robust across datasets, more interpretable, and focus on relative disparity rather than absolute scale.

**Examples:**
- ❌ "类别10有9476个样本，类别7有65个样本"
- ✅ "类别10的样本数是类别7的146倍"
- ❌ "样本数分别为215和65"
- ✅ "两类别样本数比约为3.3:1"或"分别占总样本的X%和Y%"

**Action:** Search entire Chapter 1 for any occurrence of absolute sample numbers and replace with ratios.

### Modification Priority Order

1. **Highest Priority:** Encryption principles (Issue 1.1) - foundational for understanding entire problem
2. **High Priority:** MoE background (Issue 2.1) - required for innovation narrative
3. **High Priority:** Innovation attribution (Issue 2.2) - academic integrity critical
4. **Medium Priority:** Transferability clarity (Issue 2.3) - technical precision
5. **Medium Priority:** Wording refinement (Issue 3.1) - language quality
6. **Medium Priority:** Validation description (Issue 4.1) - accurate scope
7. **Ongoing:** Replace absolute counts with ratios (Issues 1.2, 1.3, 3.2, 4.2) - consistency

### Cross-Chapter Consistency Checks

After modifications, verify:
- Encryption principles don't duplicate Chapter 2's technical content
- MoE terminology matches Chapter 2's MoE introduction
- "Backbone network" terminology consistent with Chapter 3
- All quantitative claims match Chapters 4-5 experimental results
- No new inconsistencies introduced
