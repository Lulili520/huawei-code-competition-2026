# v31_output_gated_stack_distillation

## 本地结果

| 指标 | 结果 |
|---|---:|
| Linear Output MSE | `0.002541630606880181` |
| Attention Output MSE | `0.00014752816557994677` |
| 最终得分 | **`21923.919748653454`** |

以上是 Runner 对 `main` 在固定 5 个 Linear 和 5 个 Attention 样例上的正式结果；Linear/Attention 分组索引分别为 `5` 和 `25`。`300` 只是兼容历史口径的分数折算尺度，不是本轮实测样例数。本地结果超过 `20000` 参考线，但不等于平台隐藏集成绩。

## 当前版本做出的变化

- **实现来源**：`v22_reciprocal_consumer_stack`，完整保留其 Linear 和 Attention teacher 路径。
- **本版新增**：对 Q/K/V teacher 残差拟合 rank-4 student，使用 `3` 个拟合 shard 和 `2` 个独立门禁 shard，按 KV group 进行完整 Attention 输出验证后选择 student 或 teacher。
- **保持不变**：NVFP4 固定解码、HiF4 合法编码和 v22 的完整回退路径。
- **实际生效情况**：`0/8` group 通过门禁，15 次动态调用全部走 teacher；因此精度和 v22 完全一致，本版只验证了“安全拒绝并回退”，没有验证蒸馏加速。

## 配置评测

| 配置 | Linear MSE | Attention MSE | 正式得分 | 耗时 | 是否选用 |
|---|---:|---:|---:|---:|---|
| `main` | `0.002541630606880181` | `0.00014752816557994677` | `21923.919748653454` | `47.70191890001297 s` | 是；唯一实际配置 |

本版本没有 trial，筛选阶段也未启用，因此不存在可登记的内部调参增量。`main` 是唯一正式配置，选择它是因为它完成了固定 5+5 正式评测，而不是因为它在多配置比较中胜出。候选文件 SHA-256 为 `74341a78803e770537b05287ea19832c9c37f0464e0fc6d41b8ad3ab00bec42d`。

## 算法与超参数

### 实现基础

- 对比版本和实际实现来源均为已评测的 `v22_reciprocal_consumer_stack`，即 `implementation_base=based_on`；这里只表示代码来源与对照关系，不形成父子树。
- 完整复用了 v22 的 Linear 路径、NVFP4 解码、合法 HiF4 编码、Attention 校准统计、Q/K 平滑与互逆变换，以及 Q/K/V 完整动态修复链作为 teacher。`dequantize_nvfp4()` 仍按末维每 16 值乘对应输入 scale 后恢复末维（`solution.py:304-315`），合法 64 值层级编码位于 `solution.py:336-470`。
- 新增范围是 Attention 校准后的残差蒸馏与留出门禁，以及动态期逐 head 的 student/teacher 分派。`hif4_calibration_attention()` 在构造原 teacher state 后调用蒸馏状态生成器（`solution.py:3632-4175`）；Linear 两个公开入口未改为蒸馏路径。

### 核心算法

本版本试图把 v22 的昂贵多阶段 Attention repair 链蒸馏成一次低秩残差映射，但只允许通过完整输出验证的 KV group 使用 student：

1. 校准期先用 v22 完整动态栈生成 Q/K/V teacher 合法重建；teacher 的 Q/K Hessian/码字下降和 V 链/直流/低秩修复仍完整保留（`solution.py:7333-7382`）。
2. 前 3 个 shard 用于拟合。每个 shard 最多确定性选 32 个 token；每个 head 对 `teacher reconstruction - prepared input` 做 rank-4 SVD 输出基和带 ridge 的对偶最小二乘输入投影（`solution.py:7459-7491`、`solution.py:7639-7738`）。student 在 v22 的等价坐标预处理后只做一次残差映射和一次合法消费者加权编码（`solution.py:7385-7456`）。
3. 后 2 个独立 shard 用于门禁。Q 同组 GQA heads、K、V 被原子地组合，直接比较完整 `softmax(QKᵀ/√d)V` 相对浮点参考的 SSE；每个留出 shard和聚合损失都必须落入 teacher 的 `0.25%` 相对预算才接受（`solution.py:7494-7636`、`solution.py:7893-8050`）。
4. 动态期若某些 head 被接受则走 student，其余 head 走完整 teacher，并在合法 HiF4 字段上合并；状态、shape、非有限值或合并异常均 fail closed 到完整 teacher（`solution.py:8147-8438`）。本轮门禁最终接受 `0/8` group，所以该混合路径存在于实现中，但正式执行全部落在 teacher 分支。

与 v22 的结构差异不是参数微调，而是新增“拟合 student—独立 exact-output 门禁—逐组冻结 dispatch”。不过，由于本轮 student 覆盖为零，正式输出没有包含这条新增算子的实际贡献。

### 超参数说明

| 超参数 | 作用 | 测试配置 | 最终取值 | 选择依据 |
|---|---|---|---|---|
| `_DISTILL_RANK` | 控制每 head 残差子空间容量；增大可表达更多 teacher 残差但增加拟合与动态乘法成本，减小则更快但偏差更大 | 仅 `main`，未作参数对照 | `4` | 与 v22 已使用的少量消费者主方向量级一致；本轮 `0/8` 接受说明该取值未得到输出门禁支持，不能称为最优 |
| `_DISTILL_TOKEN_LIMIT` | 限制每 shard 的确定性 token 数；增大可增加拟合/门禁覆盖但提高 teacher 校准和线性系统成本，减小则相反 | 仅 `main`，未作参数对照 | `32` | policy 的有界校准成本设计；没有多配置结果证明 32 优于其他值 |
| `_DISTILL_FIT_SHARDS` / `_DISTILL_GATE_SHARDS` | 在拟合数据与独立输出验证之间分配 shard；更多 fit 可能改善拟合，更多 gate 提高验证覆盖但减少拟合样本 | 仅 `main` | `3 / 2` | 保证 gate shard 未参与拟合；诊断确认实际记录 `3/2`，且无校准回退 |
| `_DISTILL_RIDGE` / `_DISTILL_RIDGE_FLOOR` | 稳定小样本对偶线性系统；正则增大降低方差但增加偏差，减小则可能病态 | 仅 `main` | `1e-3 / 1e-8` | 理论稳定项，源码按每 head Gram 对角均值缩放；本轮只证明 8/8 group 拟合状态有限，未证明预测有效 |
| `_DISTILL_GATE_RELATIVE_BUDGET` | 控制 student 相对 teacher 的 exact-output 容许退化；放宽会提高覆盖但增加精度风险，收紧则更易全部回退 | 仅 `main` | `0.0025`（`0.25%`） | 严于 policy 的正式 Attention MSE `1%` 否证线；本轮 student SSE 明显更差，故即使该门槛存在也无 group 通过 |

关键常量与只读诊断定义见 `solution.py:118-169`。以上数值均为唯一 `main` 的实际取值，没有伪造未运行的配置比较。

## 结果分析

### 与基准版本对比

同口径基准为 `v22_reciprocal_consumer_stack` 的固定 5+5 正式结果。

| 指标 | `v22_reciprocal_consumer_stack` | `v31 main` | 变化 |
|---|---:|---:|---:|
| Linear MSE | `0.002541630606880181` | `0.002541630606880181` | 不变，差值 `0` |
| Attention MSE | `0.00014752816557994677` | `0.00014752816557994677` | 不变，差值 `0`，相对变化 `0%` |
| 最终得分 | `21923.919748653454` | `21923.919748653454` | 不变，差值 `0` |
| 总耗时 | `48.718902699649334 s` | `47.70191890001297 s` | 减少 `1.01698379963636 s`（`2.08745218648712%`） |

精度完全保持不是 student 的算法收益：诊断显示 8 个候选 group 全部被拒绝，15 次动态 Q/K/V 调用全部执行 teacher，新增 student 的正式有效覆盖为零。总耗时虽单次测量少 `2.09%`，但没有同口径 v22 阶段耗时配对，也没有 student 执行，因而不能归因于蒸馏加速；它远不足以证明 policy 的“Attention 动态阶段降低至少 25%”。历史 v22 的 `642.32570909895 s` Attention 动态耗时来自 50+250 评测，不能与本轮 5+5 的 `23.87327240034938 s` 作配对比例。

本轮只有 `main`，所以算法内部调参收益为 `0`（没有对照配置）；新增算法相对 v22 的可归因精度收益也为 `0`，可验证的作用仅是门禁安全拒绝和 teacher 回退。

### Policy 执行情况

- 已实现：rank-4 Q/K/V 残差拟合、32-token 上限、确定性 `3 fit / 2 gate` 切分、逐 KV-group 完整 Attention 输出门禁、Q/K/V 原子 mask、student/teacher/head 合并分派和 fail-closed 回退，分别位于 `solution.py:7385-8143` 与 `solution.py:8147-8438`。
- 已实现：中心化 logit、参考 Softmax Jacobian 一阶传播、隔离 `P_refΔV` 路径 SSE，以及校准/覆盖/回退诊断；诊断快照为只读字典副本（`solution.py:130-169`、`solution.py:7542-7636`、`solution.py:8052-8115`）。
- 已保持：NVFP4 的 E2M1 值乘输入 E4M3 scale、16 值共享 scale、shape 恢复和合法 HiF4 五字段；三个公开 Attention 动态入口均汇入统一分派函数（`solution.py:304-315`、`solution.py:8441-8473`）。
- 没有发现静态实现缺项；计划中的 mixed/student-only 分支已经写入，但由于 `distill_group_accepted=0`，正式评测没有执行这两条分支。这里是实验假设失败，不是实现偏离。
- policy 要求的非零 student coverage 和至少 25% 动态降时未达到。唯一配置与 policy 一致，没有额外 trial，也没有将 smoke test 混作正式结果。

### 原因分析

- **有数据支持的直接原因**：8/8 group 的低秩状态均拟合有效，但留出 exact gate 中 student 总 SSE 为 `1632610679802499×10^-12`，teacher 为 `332984315748653×10^-12`，student 是 teacher 的 `4.90296570314995×`。因此全部 group 被拒绝，正式动态期自然退化为 v22 teacher-only 执行。这直接解释了为什么精度与 v22 完全相同、student 没有加速贡献。
- **Attention 路径诊断**：student/teacher 的中心化-logit SSE 比为 `17.1771315104436×`（`20149712357035316 / 1173054554818094`），Softmax Jacobian 一阶 SSE 比为 `15.3796183870999×`（`53296871202360 / 3465422214056`），隔离 V 路径 SSE 比为 `2.23779632323516×`（`573820061625629 / 256421934233971`）。这些配对数值支持“当前 student 在三个被观测路径上都比 teacher 差”，其中 Q/K 相关代理的倍率尤其大；但三种 SSE 位于不同表示空间且仍有高阶与耦合项，不能据此证明中心化 logits、Jacobian 或 V 中哪一项因果主导最终输出。
- **理论解释，不是已验证事实**：静态、逐 head 的线性 rank-4 残差图可能没有表达跨 token 的 Q/K 双线性交互、Softmax 状态依赖和 Q/K/V 联合修复。该解释与三个 student 路径 SSE 都变大的现象一致，但本轮没有结构消融，不能把“rank 不足”或“模型形式错误”单独确认为根因，也不应只围绕 rank 做纯参数搜索。
- **Linear 三项**：`ΔXWᵀ`、`XΔWᵀ` 与 `ΔXΔWᵀ` 没有独立诊断。Linear MSE 与 v22 精确相同只支持本轮汇总输出未变，不能证明三项数值逐位相同或判定主导项，结论为证据不足。
- **格式两项**：本轮没有 clipping 次数/SSE 与 resolution SSE 的分解。student 和 teacher 都使用同一合法 HiF4 编码器，只能说明格式语义保持；削顶与离散分辨率对门禁拒绝的相对贡献仍为证据不足。

### 证据链复核

| Policy 假设或证据链 | 结论 | 正式结果与诊断依据 |
|---|---|---|
| v22 是可安全复用的高精度 teacher，拒绝 group 应继续走完整 teacher | **结果支持** | `main` 的两项 MSE 和得分与 v22 同口径结果逐值一致；动态 `teacher_only=15/15`、`teacher_heads=400`、`no_policy/state/shape/nonfinite/merge fallback` 全为 `0` |
| rank-4 student 至少能让一个 KV group 通过独立 exact-output 门禁，并取得正式动态覆盖 | **结果否证** | `group_fit_valid=8/8`，但 `group_accepted=0/8`、`dynamic_effective_calls=0/15`、`student_heads=0`；student 完整输出 SSE 是 teacher 的 `4.90296570314995×` |
| 蒸馏会使同口径 Attention 动态阶段相对 v22 降低至少 25% | **结果否证** | student 有效覆盖为 `0`，已触发 policy 明示否证条件；总耗时仅减少 `2.08745218648712%`。Runner 未提供 v22 当前 5+5 的阶段明细，故 25% 动态比例本身不能作精确配对计算，但零覆盖已足以否证蒸馏加速机制 |
| fail-closed dispatch 将 Attention MSE 增幅限制在 1% 内、得分下降限制在 100 内 | **结果支持** | Attention MSE 增幅 `0%`，得分差值 `0`；这是“拒绝全部 student 后的安全性”证据，不是 accepted student 的精度证据 |
| 独立 exact gate 比未经验证的整体低秩替换更安全 | **结果支持** | gate 在校准成功且无异常回退时拒绝 `8/8` 个明显劣于 teacher 的 group，最终保持 v22 指标；该结果只支持 fail-closed 安全性，不支持当前 student 形式可用 |
| Linear 路径不变可推出 `ΔXWᵀ`、`XΔWᵀ`、`ΔXΔWᵀ` 各项不变或某项主导 | **证据不足** | 只有汇总 Linear MSE `0.002541630606880181`，无三项独立数值或配对路径消融 |
| 中心化 logits、Softmax Jacobian、V 路径中的某一项主导门禁失败 | **证据不足** | 三项 student/teacher SSE 都已观测且倍率分别为 `17.1771×`、`15.3796×`、`2.2378×`，但量纲、耦合和高阶项不同，无法据此完成因果排序 |
| clipping 或 resolution 是当前 student 失败的主因 | **证据不足** | Runner 未提供两类格式误差分解或固定其他路径的消融 |
| 当前结构可泛化到隐藏集、其他层或其他模型 | **证据不足** | 所有正式结论仅来自固定本地 5+5 样例和本地标准编码器近似 |

因此，policy 的整体联合假设按其明示条件判为**结果否证**：虽然精度安全条件成立，但 student 有效覆盖为零，且加速机制没有运行证据。

### 稳定性与性能诊断

| 类别 | 平均得分点 | 最低 | P10 | 中位数 | P90 | 最高 | 负收益 case |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear | `89.7848881733684` | `83.7538802002903` | `85.59616862882486` | `91.93696447292129` | `92.440817890933` | `92.45609961065752` | `0/5` |
| Attention | `69.73870135994014` | `67.1128385012707` | `67.5240548535039` | `69.27636824286316` | `72.44845148913056` | `73.91541609822647` | `0/5` |

- 最差 Linear 是 group `5`、sample `0`：标准 MSE `0.027874629944562912`，player MSE `0.0045285457745194435`，仍有 `83.7538802002903` 个正收益百分点。
- 最差 Attention 是 group `25`、sample `3`：标准 MSE `0.00042736256727948785`，player MSE `0.00014054741768632084`，仍有 `67.1128385012707` 个正收益百分点。
- 两类 10 个 case 均为正收益，Linear/Attention 的 P10、median 与 mean 均同向为高正值，没有“单个正收益 case 拉高整体均值”的迹象。但由于结果与 v22 相同，这种稳定性属于 teacher 路径，不能归给 student；样例数也不足以外推隐藏集。
- 阶段耗时：Linear 校准 `7.524306499399245 s`、Linear 动态样例 `6.343614600598812 s`、Attention 校准 `9.835784199647605 s`、Attention 动态样例 `23.87327240034938 s`；四阶段合计 `47.576977699995 s`，其余记录开销约 `0.124941200017929 s`，总耗时 `47.70191890001297 s`。
- 搜索/路由诊断：校准 `1/1` 成功、异常回退 `0`；fit/gate shard 为 `3/2`；候选 `8`、拟合有效 `8`、接受 `0`、拒绝 `8`，接受率 `0%`、拒绝率 `100%`。动态 `15` 次调用中 effective/student-only/mixed 均为 `0`，teacher-only 为 `15`；student/teacher heads 为 `0/400`。no-policy、state、shape、nonfinite、merge 回退均为 `0`，说明零覆盖来自门禁主动拒绝，而非异常旁路。

## Take Away

### 有效经验

- **独立完整输出门禁应保留**：8/8 group 的数值拟合都有效，但 student 完整输出 SSE 是 teacher 的 `4.90296570314995×`；exact gate 将其全部拒绝，使两项 MSE 和得分与 v22 完全一致。
- **有效覆盖必须与可拟合性分开报告**：`fit_valid=8/8` 并未转化为任何正式调用，`effective_calls=0/15`、`student_heads=0/400`。只有非零 applied coverage 才能支持蒸馏或加速声明。
- **逐路径诊断能定位模型错位而不能单独给出因果排序**：当前 student 相对 teacher 的中心化-logit、Jacobian、V 路径 SSE 分别为 `17.1771×`、`15.3796×`、`2.2378×`，足以否定“当前 student 已近似 teacher”，但不足以判定单一路径主导。

### 失败经验与边界

- 静态逐 head rank-4 张量残差蒸馏在本轮 `0/8` 接受，不应沿该结构只调整 rank、门槛、token 数或 shard 比例继续建立版本；需要改变 student 的表示或路由对象。
- `47.70191890001297 s` 比 v22 少 `2.0875%` 不是可归因的算法加速：student 从未执行，且缺少当前 5+5 的 v22 阶段配对时间。
- 精度守住来自 teacher-only 回退，不能写成 student 达到了 v22 精度。反过来，本轮也没有 accepted group，因此不能判断 student 一旦通过门禁后的正式泛化。
- Linear 三传播项和格式 clipping/resolution 均缺少独立诊断；不得从汇总 MSE、路径代理倍率或单次本地结果反推普遍机理。

### 可复用结论

1. 后续压缩 v22 应优先复用“独立 exact-output 验证 + 完整 teacher fail-closed”，但候选必须报告 `accepted/effective/student_heads`，零覆盖只能算安全回退，不能算算法收益。
2. 若继续做快速 Attention 近似，应从逐 head 静态张量残差转向能够表示 Q/K 双线性交互、Softmax 状态依赖或已评测快速 expert 的结构；不得只微调 rank 或 gate 预算。
3. 性能验收必须在相同固定样例上配对阶段耗时，并把校准成本、动态成本和有效覆盖一起报告；历史 50+250 阶段时间不能与当前 5+5 直接相除。

本报告只陈述本地固定 5 Linear + 5 Attention、由本地标准编码器近似得到的正式结果；这些结论不保证平台隐藏集、其他层或其他模型上的分数与排序一致。
