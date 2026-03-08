# Proposal: Chapter 3 Enhancement - Method Comparison Accuracy

## Why

Chapter 3 (GEE架构设计) contains inaccurate method comparison language that misrepresents the novelty of the proposed GEE architecture. Specifically, the text compares GEE to "传统的固定加权平均" (traditional fixed weighted average), but GEE is a newly proposed architecture with no prior "traditional" implementation. This creates confusion about whether GEE is an improvement over existing methods or a new architectural choice.

## What Changes

### 1. 修正方法对比表述（全文）

**问题位置**:
- 第三章中提到："相比传统的固定加权平均（如0.85×基准+0.15×专家）"
- 类似的对比表述可能存在于多个位置

**问题分析**:
- "传统的固定加权平均" 暗示存在传统的GEE架构使用固定加权平均
- 但GEE是本研究新提出的架构，不存在"传统"版本
- 这不是"新方法改进传统方法"，而是"新方法选择不同的设计策略"

**修改方案**:
**原文**:
> "相比传统的固定加权平均（如0.85×基准+0.15×专家），门控网络能够根据样本特性动态调整融合权重。"

**修改为**:
> "固定加权平均策略使用人为设定的权重对所有样本进行融合（如0.85×基准+0.15×专家），但这些权重是固定的、无法适应样本差异。相比之下，GEE架构的门控网络采用可学习融合策略，能够根据每个样本的特性动态调整融合权重。"

**或者更简洁的版本**:
> "GEE架构采用可学习融合策略，而非固定加权平均。固定加权平均使用人为设定的统一权重（如0.85×基准+0.15×专家），而门控网络能够根据每个样本的特性动态调整融合权重。"

**核心改动**:
- 去掉"传统的"这个不准确的修饰语
- 明确这是两种不同的"融合策略"设计选择
- 对比的重点是"固定 vs 可学习"，而非"传统 vs 新"

### 2. 全文扫描类似表述

扫描第三章中所有涉及方法对比的表述，确保：
- 如果确实是改进现有方法，保留"传统的"、"现有的"等表述
- 如果是新方法的不同设计选择，去掉"传统的"，改为中性对比
- 对比焦点应该是"方案A vs 方案B"，而非"新方案 vs 旧方案"

## Capabilities

### New Capabilities
- `method-comparison-accuracy`: Ensuring accurate representation of method comparisons in technical writing

### Modified Capabilities
- `thesis-chapter-improvement`: Extending existing thesis improvement framework with new principle on method comparison accuracy

## Impact

### Affected Files
- `/thesis/main.md` - Chapter 3 (GEE架构设计), approximately lines 236-280

### Documentation Changes
- 修正方法对比表述，去除"传统的"等不准确修饰语
- 明确新方法是"设计选择"而非"改进传统方法"
- 保持学术严谨性，避免读者误解

### Dependencies
- 需要扫描第三章全文，找出所有类似表述
- 需要区分"改进现有方法"和"新设计选择"两种情况

### Success Criteria
1. ✅ 去除所有不准确的"传统的"表述（当GEE是新方法时）
2. ✅ 方法对比焦点明确为"方案A vs 方案B"的设计选择
3. ✅ 保持第三章的学术严谨性和技术准确性
4. ✅ 读者能够清晰理解GEE是新的架构设计，而非改进版
5. ✅ 修改后逻辑连贯、表述自然
