---
name: hif4-report
description: Create a fixed-structure post-evaluation report.md for a HiF4 solution version, placing local results first, then analyzing the parent comparison and policy execution, and ending with reusable takeaways. Use only after evaluation succeeds.
---

# HiF4 Version Report

Create `solution/<version>/report.md` only after `$hif4-evaluate` succeeds. `policy.md` records the plan; `report.md` records observed facts and lessons.

Use every heading below, in this order. Do not rename, omit, or reorder sections.

```markdown
# <version>

## 本地结果

| 指标 | 结果 |
|---|---:|
| Linear Output MSE | <value> |
| Attention Output MSE | <value> |
| 最终得分 | **<value>** |

## 算法与超参数

### 实现基础

### 核心算法

### 超参数说明

| 超参数 | 作用 | 测试配置 | 最终取值 | 选择依据 |
|---|---|---|---|---|
| <name> | <作用和影响路径> | <最多两到三个配置> | <value> | <实际评测依据> |

### 配置评测对照

| 配置 | Linear Output MSE | Attention Output MSE | 最终得分 | 是否选用 |
|---|---:|---:|---:|---|
| <config> | <value> | <value> | <value> | <是/否> |

## 结果分析

### 与父版本对比

### Policy 执行情况

### 原因分析

### 证据链复核

## Take Away

### 有效经验

### 失败经验与边界

### 可复用结论
```

## Section requirements

- `本地结果`：只能写评测器实际输出的两项 MSE 和最终得分，不添加 NMSE、SQNR、最大误差、耗时或估算平台分数。
- `实现基础`：记录名义父版本，以及实际使用 parent、v0 或 scratch；说明复用和重写范围。
- `核心算法`：用白话说明本版本新增或替换的量化算法、数据流和误差目标。明确它与父版本的算法差异；不能把超参数变化描述成核心算法。
- `超参数说明`：列出该算法实际使用的关键超参数。对每项解释它控制什么、增大或减小会带来什么影响、实际测试的两到三个配置、最终取值及选择依据。没有进行多配置实验的固定参数也要说明其理论来源或继承来源，不得伪造测试结果。
- 同一算法最多进行两到三次有理论依据的配置试验，配置结果保留在同一个版本报告中，不为纯参数变化创建额外版本。表格只保留影响算法行为的关键参数，不罗列设备、随机种子或无关实现常量。
- `配置评测对照`：逐项填写 runner 实际执行的官方 10 例结果。主配置也必须列出；未运行的候选不得出现。明确算法相对父版本的收益与内部调参带来的增量。
- `与父版本对比`：写出父版本名，并分别说明两项 MSE 和最终得分上升、下降或不变。不要只说“效果更好”。
- `Policy 执行情况`：说明实际修改了哪些函数和参数、哪些计划未实现，以及与 policy 偏离的原因。
- `原因分析`：判断结果是否支持 policy 的理论假设，解释收益或退化如何产生。区分有数据支持的结论与推测。
- `证据链复核`：逐条复核 policy 中的证据链，标记为 `结果支持`、`结果否证` 或 `证据不足`。引用实际配置名和精确指标；说明理论机制是否被结果直接验证，还是仅为合理解释。不得从官方本地 10 例外推隐藏集或其他模型上的普遍有效性。
- `有效经验`：提炼本轮已被结果支持、以后可以继续使用的做法。
- `失败经验与边界`：记录无效做法、适用条件、异常情况和不能推广的结论，防止后续分支重复试错。
- `可复用结论`：给出一到三条简短、可操作、能影响后续分支决策的经验。这是报告必须提供的核心 Take Away，不写空泛总结。

若版本无法完成评测，不生成正常 report；保留 policy，并在版本树中记录失败状态。报告末尾说明本地标准编码器与平台隐藏集评分的边界。
