---
name: hif4-policy
description: Create a fixed-structure pre-implementation policy.md for a HiF4 optimization branch, covering its parent, problem evidence, related-solution research, theory, selected changes, expected result, and acceptance criteria. Use before modifying a new solution version.
---

# HiF4 Optimization Policy

Create `solution/<version>/policy.md` before changing the copied `solution.py`. Do not create one for the existing root baseline `v0_hessian_repair`. A policy describes what will be tested and must not claim unobserved results.

Use every heading below, in this order. Do not rename, omit, or reorder sections.

```markdown
# <version> 优化策略

## 父版本

## 实现基础

## 固定输入边界

## 问题分析

## 相关方案调研

## 理论分析

## 选定修改方案

### 核心算法

### 修改目标

### 修改范围

### 保持不变

## 算法内部超参数计划

## 实施步骤

## 预期结果

## 验收标准
```

## Section requirements

### Evidence standard

Every policy must separate three evidence levels:

- `已验证事实`: directly observable in repository code, task documents, or recorded evaluator output. Cite a file plus function/line, or a run/config plus exact metric.
- `理论推导`: derived from equations or an external method. Cite the primary paper, standard, or official documentation with title, author/organization, year, and direct URL; state the assumptions needed for the derivation to apply here.
- `待验证假设`: a falsifiable prediction for this version. State the target metric, expected direction, comparison baseline, and a condition that would reject the hypothesis.

For every proposed modification include one evidence-chain row:

| 问题证据 | 理论依据 | 算法动作 | 目标指标 | 否证条件 |
|---|---|---|---|---|
| <可定位事实> | <来源与适用假设> | <具体代码/流程变化> | <Linear/Attention MSE 或得分> | <什么结果说明假设不成立> |

The chain must be causal, not merely topical. A citation about quantization in general does not justify a specific transform. Theory may justify a mechanism or expected direction, but only measured official cases establish local effectiveness. Never invent citations, source details, formulas, or results.

- `父版本`：写出准确版本名、父版本 Linear MSE、Attention MSE 和最终得分。这些值是本分支唯一比较基线。
- `实现基础`：声明 `parent`、`v0` 或 `scratch`，并写出实际代码来源、复用模块、重写模块及不直接继承父版本的原因。父版本只定义比较关系，不强制代码继承。
- `固定输入边界`：明确 NVFP4 固定按 E2M1 值乘 E4M3 scale 反量化，每 16 个值共享一个输入 scale；禁止把输入反量化方式作为优化方案。
- `问题分析`：结合父版本 report、代码和指标，定位一个可验证问题。每项事实给出文件/函数或评测配置/指标定位，说明证据及其影响的是 Linear、Attention 还是格式搜索。
- `相关方案调研`：比较与问题直接相关的已有方法。项目知识不足时检索论文或官方文档，优先引用原始论文和官方资料并给出直达链接。说明每个方法可借鉴什么、为什么适合或不适合本题，避免堆砌无关文献。
- `理论分析`：先用白话解释，再给必要公式或参数分析。术语首次出现时解释；公式注明符号、适用假设和来源，随后说明它在本题中的含义。建立“问题证据 → 理论依据 → 算法动作 → 误差传播 → 目标 MSE → 否证条件”的逻辑链。
- `选定修改方案`：只保留本分支要验证的一个结构级算法假设。写明误差模型、等价变换、HiF4 层级选择或误差补偿如何改变；纯参数变化不得成为版本。
- `算法内部超参数计划`：最多列出两到三组有理论依据的配置，解释参数作用、候选值、选择指标和停止条件。所有配置保留在同一版本，禁止创建纯调参分支。
- `实施步骤`：写成 Agent 可直接执行的顺序操作，包含参数候选、回退条件和需要检查的边界。
- `预期结果`：只写有依据的方向性预期，例如“Attention MSE 降低、Linear MSE 基本不变”。可以给目标阈值，但不得伪造测试值。
- `验收标准`：要求官方格式检查通过；目标 MSE 与父版本比较；最终得分与父版本比较；未修改路径不能出现无法解释的回退。

每个父版本最多规划三个独立算法子策略。local-agent 任务统一由 `.agent/runner.py` 创建和调度。
