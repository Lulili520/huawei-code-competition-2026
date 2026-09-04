---
name: hif4-report
description: Create a fixed-structure post-evaluation report.md for a flat HiF4 solution version, placing local results first, then analyzing its comparison reference and policy execution, and ending with reusable takeaways. Use only after evaluation succeeds.
---

# HiF4 Version Report

Create `solution/<version>/report.md` only after `$hif4-evaluate` succeeds. `policy.md` records the plan; `report.md` records observed facts and lessons.

The report stage is also the only stage allowed to propose follow-up algorithms. It receives the completed fixed-evaluation metrics and diagnostics, writes `report.md`, and returns exactly three structured proposals to Runner: two `explore` and one `exploit`. Those proposals are not guessed during implementation and do not have to use the current version as `based_on`.

Use every heading below, in this order. Do not rename, omit, or reorder sections.

```markdown
# <version>

## 本地结果

| 指标 | 结果 |
|---|---:|
| Linear Output MSE | <value> |
| Attention Output MSE | <value> |
| 最终得分 | **<value>** |

## 配置评测

| 配置 | Linear MSE | Attention MSE | 正式得分 | 耗时 | 是否选用 |
|---|---:|---:|---:|---:|---|

## 算法与超参数

### 实现基础

### 核心算法

### 超参数说明

| 超参数 | 作用 | 测试配置 | 最终取值 | 选择依据 |
|---|---|---|---|---|
| <name> | <作用和影响路径> | <最多两到三个配置> | <value> | <实际评测依据> |

## 结果分析

### 与基准版本对比

### Policy 执行情况

### 原因分析

### 证据链复核

### 稳定性与性能诊断

## Take Away

### 有效经验

### 失败经验与边界

### 可复用结论
```

## Section requirements

- `本地结果`：顶部主表只能写最终选中配置实际运行 50 Linear + 250 Attention 后的两项 MSE 和最终得分，不添加估算平台分数。
- `配置评测`：分开列出 60 例内部筛选和完整 300 例正式评测。只有后者能决定版本得分与最终选择；不得混入 smoke test 或估算结果。
- `实现基础`：记录 `based_on` 对比版本，以及实际使用 `based_on`、`v0_hessian_repair` 或 `scratch`；说明复用和重写范围，不表达父子关系。
- `核心算法`：用白话说明本版本新增或替换的量化算法、数据流和误差目标。明确它与基准版本的算法差异；不能把超参数变化描述成核心算法。
- `超参数说明`：列出该算法实际使用的关键超参数。对每项解释它控制什么、增大或减小会带来什么影响、实际测试的两到三个配置、最终取值及选择依据。没有进行多配置实验的固定参数也要说明其理论来源或继承来源，不得伪造测试结果。
- 同一算法最多进行两到三次有理论依据的配置试验，配置结果保留在同一个版本报告中，不为纯参数变化创建额外版本。表格只保留影响算法行为的关键参数，不罗列设备、随机种子或无关实现常量。
- `正式评测`：只比较 Runner 实际执行的完整 50 + 250 结果。明确算法相对基准版本的收益与内部调参带来的增量；不得用任何部分评测补位或外推。
- `与基准版本对比`：写出基准版本名，并分别说明两项 MSE 和最终得分上升、下降或不变。不要只说“效果更好”。
- `Policy 执行情况`：说明实际修改了哪些函数和参数、哪些计划未实现，以及与 policy 偏离的原因。
- `原因分析`：判断结果是否支持 policy 的理论假设，解释收益或退化如何产生，并区分有数据支持的结论与推测。回到 policy 声明的根因分解：Linear 分别检查 `ΔXWᵀ`、`XΔWᵀ` 和 `ΔXΔWᵀ`；Attention 分别检查中心化 logits、Softmax Jacobian 敏感方向和 V 路径；格式策略区分 clipping 与 resolution。
- `证据链复核`：逐条复核 policy 中的证据链，标记为 `结果支持`、`结果否证` 或 `证据不足`。引用实际配置名和精确指标；说明理论机制是否被结果直接验证，还是仅为合理解释。无法由本轮诊断区分的根因项必须标为 `证据不足`，不能用最终总分反推机制已经成立，也不得从本地 300 例外推隐藏集或其他模型上的普遍有效性。
- `稳定性与性能诊断`：报告两类 score 分布、负收益 case 数、最差代表 case、总耗时和各阶段耗时。若实现提供搜索诊断，报告候选接受率与回退次数。区分“均匀改善”和“少数样例拉高均值”。
- `有效经验`：提炼本轮已被结果支持、以后可以继续使用的做法。
- `失败经验与边界`：记录无效做法、适用条件、异常情况和不能推广的结论，防止后续分支重复试错。
- `可复用结论`：给出一到三条简短、可操作、能影响后续分支决策的经验。这是报告必须提供的核心 Take Away，不写空泛总结。

若版本无法完成评测，不生成正常 report；保留 policy，并在扁平账本中记录失败状态。报告末尾说明本地标准编码器与平台隐藏集评分的边界。

## Structured follow-up requirements

Only after the selected configuration has a valid full 50 Linear + 250 Attention result, return exactly three structural directions through the report-result schema:

1. two `explore` directions that test distinct root causes, cross algorithm families, or use `scratch` when warranted;
2. one `exploit` direction backed by a measured positive result or Pareto component advantage, with evidence strength sufficient for exploitation;
3. for every direction: an already fully evaluated `based_on`, `search_mode`, root-cause decomposition, algorithm action, evidence and its strength, target metric, expected cost, and falsification condition.

An `exploit` direction may use another evaluated Pareto version when the current hypothesis was rejected. None of the three may be a pure alpha, threshold, gain, multiplier, or candidate-count change. Runner deduplicates and dynamically reranks these proposals into the global portfolio; it targets six active slots in a 4 `explore` + 2 `exploit` mix.

The full-300 local score target is `20000`, but it does not automatically stop iteration. Never describe a compact self-check, 60-case screening result, smoke test or manual partial diagnostic as a formal result.
