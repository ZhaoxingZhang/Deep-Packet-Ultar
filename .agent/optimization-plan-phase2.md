# 阶段二：理论深化与形式化（优先级：中）

## 2.1 建立能量函数理论框架

### 目标
使用Energy Score将"中性决策边界"与"未知样本高熵"联系起来，提供形式化的理论桥梁。

### 理论背景
Energy Score与Softmax置信度的关系：
$$ E(x; f) = -T \cdot \log \sum_{j=1}^K \exp(f_j(x)/T) $$

其中 $T$ 是温度参数。关键性质：
- 低能量 $\rightarrow$ 高置信度（属于已知类）
- 高能量 $\rightarrow$ 低置信度（可能属于未知类）

### 具体任务
1. **推导加权CE对能量分布的影响**

   目标命题：加权CE损失倾向于
   - 降低已知类训练样本的能量
   - 提高决策边界外区域（潜在未知样本区域）的能量

   形式化论证思路：
   - 分析梯度更新方向：$\theta \leftarrow \theta - \eta \nabla_\theta \hat{R}_{WCE}$
   - 证明加权项抑制了多数类对边界的"过度扩张"
   - 这导致边界外区域的logit范数减小，能量升高

2. **连接能量与Softmax熵**

   利用关系：
   $$ p_j(x) = \frac{\exp(f_j(x)/T)}{\sum_{k=1}^K \exp(f_k(x)/T)} = \frac{\exp(-E(x; f))}{Z} $$

   证明：高能量样本具有更平坦的Softmax分布（高熵）

3. **补充能量函数的可视化**
   - 绘制Baseline vs GEE的能量分布直方图
   - 展示已知类和未知类样本的能量分布差异

## 2.2 Logit几何分析

### 目标
从logit向量的几何性质分析加权损失的影响。

### 具体任务
1. **分析logit范数分布**

   研究问题：加权CE如何影响 $\|f(x)\|$ 的分布？

   实验验证：
   - 统计已知类样本的平均logit范数：$\mathbb{E}_{x \in \text{known}}[\|f(x)\|]$
   - 统计未知类样本的平均logit范数：$\mathbb{E}_{x \in \text{unknown}}[\|f(x)\|]$
   - 对比Baseline和GEE的差异

   预期结果：
   - GEE对已知类保持高范数（高置信度）
   - GEE对未知类产生低范数（低置信度）
   - Baseline可能对未知类也产生高范数（过度自信）

2. **分析类别间margin分布**

   定义第 $i$ 个样本的margin：
   $$ m_i = \min_{j \neq y_i} (f_{y_i}(x_i) - f_j(x_i)) $$

   研究假设：
   - 加权CE通过平衡梯度，使margin分布更均匀
   - 未知样本的margin更接近0（各类别得分相近）$\rightarrow$ 高熵

   实验验证：绘制Baseline vs GEE的margin累积分布函数（CDF）

3. **添加数学引理或定理**

   尝试证明或引用：
   > **引理**：给定Softmax分类器，样本的logit范数与其Softmax熵呈负相关。即 $\|f(x)\|$ 越小，$H(p(x))$ 越大。

   （若无法严格证明，可作为实验观察并引用相关文献）

## 2.3 梯度动力学的严格分析

### 目标
替换原文中的"梯度近似"直觉说明，使用更严谨的分析方法。

### 具体任务
1. **分析类别不均衡对梯度分布的影响**

   标准CE的梯度期望（理论分析）：
   $$ \mathbb{E}[\nabla_\theta \ell_{CE}] = \sum_{j=1}^K \pi_j \cdot \mathbb{E}_{x \sim \mathcal{D}_j}[\nabla_\theta \ell_{CE}(f_\theta(x), j)] $$

   其中 $\pi_j$ 是第 $j$ 类的先验概率。

   问题：当 $\pi_j$ 分布不均时，多数类主导梯度方向。

2. **加权CE的平衡作用**

   加权后的梯度期望：
   $$ \mathbb{E}[\nabla_\theta \ell_{WCE}] = \sum_{j=1}^K \frac{w_j \pi_j}{\sum_k w_k \pi_k} \cdot \mathbb{E}_{x \sim \mathcal{D}_j}[\nabla_\theta \ell_{CE}(f_\theta(x), j)] $$

   证明：选择合适的权重 $w_j$（如 $w_j \propto 1/\pi_j$）可以平衡各类贡献。

3. **使用理论界代替近似**

   引用PAC-Bayes或Rademacher复杂度理论，分析加权CE如何影响：
   - 经验风险与真实风险的gap
   - 模型对分布外样本的泛化界

   （这属于优先级较低的探索性工作，可选择性添加）

## 预期成果
- 建立"加权CE → 能量分布 → Softmax熵"的严格理论链路
- 从logit几何角度提供额外的理论解释
- 用更严谨的数学分析替换原有的直觉说明
- 增强理论的数学深度和可信度

## 实施检查清单
- [ ] 查阅Energy Score相关文献（如Liu et al., "Energy-based Out-of-distribution Detection"）
- [ ] 推导能量与熵的关系式并添加到论文
- [ ] 进行logit范数和margin分布的实验分析
- [ ] 绘制能量分布和margin分布的可视化图表
- [ ] 重写梯度分析部分，使用期望形式或理论界
