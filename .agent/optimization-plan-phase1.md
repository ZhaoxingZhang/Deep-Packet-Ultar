# 阶段一：基础规范化优化（优先级：高）

## 1.1 符号系统规范化

### 目标
建立全章统一的数学符号体系，确保概念定义清晰、一致。

### 具体任务
1. **统一概率符号**
   - 当前问题：混用 $p_i$, $p_{y_i}$, $P(\text{各类} \mid \text{未知样本})$
   - 规范方案：
     - $p_{i,j}$ 表示第 $i$ 个样本属于第 $j$ 类的概率
     - $p_{y_i}$ 表示第 $i$ 个样本属于其真实类 $y_i$ 的概率
     - 明确定义域：$\forall i \in \{1, \ldots, N\}, j \in \{1, \ldots, K\}$

2. **定义核心概念的形式化表达**
   - **中性决策边界**：定义为几何度量
     $$ \mathcal{B}_{\text{neutral}} = \{x \mid \|f(x) - \mu_1\| = \|f(x) - \mu_2\| = \ldots = \|f(x) - \mu_K\|\} $$
     其中 $\mu_j$ 是第 $j$ 类的原型向量
   - **Softmax熵**：
     $$ H(p(x)) = -\sum_{j=1}^{K} p_j(x) \log p_j(x) $$

## 1.2 损失函数的规范化表达

### 目标
将损失函数从单样本形式改为严格的经验风险形式。

### 具体任务
1. **重写加权交叉熵损失**
   当前形式（不规范）：
   $$ L_{GEE} = -\sum_i w_{y_i} \log(p_{y_i}) $$

   规范形式（经验风险）：
   $$ \hat{R}_{WCE}(\theta) = \frac{1}{N} \sum_{i=1}^{N} w_{y_i} \cdot \ell_{CE}(f_\theta(x_i), y_i) $$

   其中：
   $$ \ell_{CE}(f_\theta(x_i), y_i) = -\log\left(\frac{\exp(f_{y_i}(x_i))}{\sum_{j=1}^{K}\exp(f_j(x_i))}\right) $$

2. **明确各变量定义**
   - $N$：训练样本总数
   - $K$：已知类别数
   - $\theta$：模型参数
   - $f_j(x_i)$：样本 $x_i$ 在第 $j$ 类的 logit
   - $w_{y_i}$：真实类别 $y_i$ 的权重

## 1.3 门控融合机制的澄清

### 目标
明确门控权重的数学形式，确保输出是合法的概率分布。

### 具体任务
1. **确定门控权重形式（推荐方案：标量门控）**

   标准Mixture of Experts形式：
   $$ y_{GEE}(x) = g_{baseline}(x) \cdot y_{baseline}(x) + g_{expert}(x) \cdot y_{expert}(x) $$

   约束条件：
   $$ g_{baseline}(x) + g_{expert}(x) = 1 $$
   $$ g_{baseline}(x), g_{expert}(x) \in [0, 1] $$

   实现方式：
   $$ g_{baseline}(x) = \sigma(v^T \cdot h(x) + b) $$
   $$ g_{expert}(x) = 1 - g_{baseline}(x) $$

   其中 $\sigma$ 是sigmoid函数，$h(x)$是门控网络，$v, b$是可学习参数。

2. **分析凸组合对熵的影响**

   理论性质：两个概率分布的凸组合的熵不低于各分布熵的加权和：
   $$ H(y_{GEE}) \geq g_{baseline} \cdot H(y_{baseline}) + g_{expert} \cdot H(y_{expert}) $$

   （需补充证明或引用相关不等式）

## 预期成果
- 符号定义清晰、统一、无歧义
- 损失函数表达符合机器学习理论规范
- 门控融合机制数学上自洽
- 为后续理论深化打下坚实基础

## 实施检查清单
- [ ] 创建符号定义表（添加到论文附录或3.5节开头）
- [ ] 重写所有损失函数公式
- [ ] 修正门控融合公式及说明文字
- [ ] 逐行检查文中数学表达的一致性
