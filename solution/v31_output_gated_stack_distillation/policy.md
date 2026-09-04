# v31_output_gated_stack_distillation 优化策略

## 基准版本

- 唯一比较基准为 `v22_reciprocal_consumer_stack`；`based_on` 只表示实现来源与实验对照，不形成版本树。
- **已验证事实**：当前固定 5 Linear + 5 Attention 正式结果为 Linear MSE `0.002541630606880181`、Attention MSE `0.00014752816557994677`、折算最终得分 `21923.919748653454`、总耗时 `48.718902699649334 s`（`solution/v22_reciprocal_consumer_stack/report.md:5-21`）。这些数值是本版本在相同固定样例上的唯一验收基线。
- **已验证事实（补充高覆盖证据，不与上述 MSE 作配对差值）**：历史 50 Linear + 250 Attention 结果为 Linear MSE `0.004031384951498092`、Attention MSE `0.00023155507926627506`、得分 `21673.257105856967`、耗时 `1142.3838615007699 s`；其中 Attention 动态阶段 `642.32570909895 s`（同一 report 的 `:21`、`:84`、`:107`）。

## 实现基础

- `search_mode=exploit`，实现基础为 `based_on=v22_reciprocal_consumer_stack`。目标目录的初始 `solution.py` 与 v22 的 SHA-256 均为 `E04C2A59507727D22C0951ACCC001299E106E76C998630E9459D28BDC5910F81`，因此实际代码来源可复查。
- **复用**：完整 Linear 路径、NVFP4 解码、合法 HiF4 编码、Attention 校准统计、Q/K 平滑与互逆变换，以及原 Q/K/V 动态修复链全部保留为 teacher 和 fail-closed 回退。
- **重写**：在 Attention 校准返回前增加“teacher 输出蒸馏、独立留出 exact 输出门禁、逐 KV-group 冻结 dispatch”；动态 Q/K/V 入口增加逐 group 的 student/teacher 分派。student 只执行既有等价 Q/K 预处理、一次低秩残差算子和一次合法 HiF4 编码。
- **exploit 依据**：v22 相对 v6 在历史统一 300 例上得分增加 `5113.941609096524`，两类 MSE 同时下降，属于精度 Pareto；但其耗时约为 v6 的 `2.05×`（`solution/v22_reciprocal_consumer_stack/report.md:61-68`）。本版本不以未经验证的快速近似替换所有 group，而只蒸馏 exact 门禁通过的 group。

## 固定输入边界

- NVFP4 输入反量化严格保持 `E2M1 value × 对应输入 E4M3 scale`，每 16 个连续值共享一个输入 scale，最后恢复原 shape；实现定位为 `solution/v22_reciprocal_consumer_stack/solution.py:250-261`。不得重新估计或替换输入 scale，也不得改变分组或数值语义。
- HiF4 输出仍由既有 64 值层级编码器产生合法 `scale_factor/scale_lv2/scale_lv3/sign/mant`；student 不直接构造越界码字。
- 仅使用 CPU 张量运算；不读取数据集、评测器元数据或 case 标识，不修改评分口径。

## 问题分析

1. **已验证事实—性能根因**：v22 的 Attention 动态入口在 `_attention_dynamic()` 中，Q 串联敏感编码、Hessian repair 和固定 scale code descent；K 还串联 wide descent、双坐标求解与 breakpoint overlay；V 串联 hierarchy chain、chain repair、global-DC 与 low-rank repair（`solution/v22_reciprocal_consumer_stack/solution.py:7216-7295`）。历史阶段记录显示 Attention 动态耗时 `642.32570909895 s`，是首先需要压缩的路径。
2. **已验证事实—正例**：v22 的完整组合在当前 5+5 结果中得到 `21923.919748653454`，在历史高覆盖结果中同时降低 Linear 与 Attention MSE；因此它适合作为安全 teacher，而不能仅凭总分把某一个内部模块宣称为独立有效。
3. **已验证事实—反例**：v27 用统一低秩商空间路径直接替换 Q/K 链，校准 `0/8` group 接受、动态有效覆盖 `0/10`，且固定 5+5 得分为 `19146.707211539724`（`solution/v27_softmax_quotient_solver/report.md:7-9,66,84`）。这排除了“未通过输出验证也可以整体替换 teacher”的做法；它没有排除“逐 group 蒸馏并保留真实 teacher 回退”。
4. **Linear 输出误差分解**：写 `X̂=X+ΔX`、`Ŵ=W+ΔW`，则 `X̂Ŵᵀ-XWᵀ=ΔXWᵀ+XΔWᵀ+ΔXΔWᵀ`。本版本不改变 v22 的 Linear weight/activation 选择，因此三项均应保持代码路径一致；Runner 尚无三项独立数值，任何一项主导都属于**待验证假设**，不能从 Linear 汇总 MSE 推断。
5. **Attention 输出误差分解**：令中心化投影 `C=I-11ᵀ/T`，logit 误差的有效部分为 `ΔL_c=ΔL C`；概率一阶误差为 `J_softmax(L)ΔL_c`；最终还叠加 `PΔV` 及概率误差与 `ΔV` 的耦合项。v22 的 Q/K 多阶段修复针对前两项，V 链针对 `PΔV`。本版本的 exact gate 直接比较完整 `softmax(QKᵀ/√d)V`，因此三条路径及其耦合共同进入接受判据，但门禁结果本身不能识别哪一条主导。
6. **格式误差边界**：固定 HiF4 编码同时可能产生超出最大可表示幅值的削顶误差，以及未削顶值落在离散网格之间的分辨率误差。student 只改变送入同一编码器的低秩目标，不改变 E6M2/LV2/LV3 网格；本版本不声称其中任一格式误差主导，二者对最终输出的相对贡献为**证据不足**。

## 相关方案调研

- Geoffrey Hinton、Oriol Vinyals、Jeff Dean，*Distilling the Knowledge in a Neural Network*，2015，[原始论文](https://arxiv.org/abs/1503.02531)。可借鉴点是以 teacher 响应监督较小 student；本题不是分类蒸馏，因此只采用“拟合 teacher 输出”的机制，不采用温度或类别概率损失。
- Carl Eckart、Gale Young，*The Approximation of One Matrix by Another of Lower Rank*，1936，[原始论文 DOI](https://doi.org/10.1007/BF02288367)。Eckart–Young 结论说明截断 SVD 在 Frobenius 范数下给出固定秩矩阵的最优近似；本题仅把它用于 teacher 残差的输出子空间，量化后的最终优劣仍须 exact gate 判断。
- Guangxuan Xiao 等，*SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*，2022，[原始论文](https://arxiv.org/abs/2211.10438)。其等价缩放思想支持保留 v22 的 Q/K 平滑与互逆基变换作为 student 的输入坐标系；低秩算子不取代固定 NVFP4 解码。
- 项目内 v27 表明局部低秩代理的接受率不能替代完整输出验证。本版本借鉴其按 KV group 计算 Attention 输出的方式，但关键差异是：拒绝 group 继续执行 v22 完整 teacher，而不是落到被简化的 parent 编码。

## 理论分析

白话上，teacher 已经知道如何用多轮离散搜索得到高精度输出，但每个动态样例重复这条长链。校准时可观察同一输入经 teacher 后的合法重建。student 先保留 v22 的浮点等价坐标整理，再用一个“输入投影 × 输出基”的低秩残差算子预测 teacher 修复方向，最后只编码一次。留出的校准样本不用来拟合，只用于完整 Attention 输出比较；每个 KV group 独立冻结为 student 或 teacher。

对某角色和某个 head，把等价预处理后的输入记为 `Z∈R^{n×d}`，teacher 合法重建记为 `Y`，残差 `R=Y-Z`。在拟合集上先取 `R` 的前 `r` 个右奇异方向 `B∈R^{r×d}`，再用带正则的最小二乘拟合 `A∈R^{d×r}`：

`A = Z_cᵀ (Z_c Z_cᵀ + λI)^{-1} (R_c Bᵀ)`，`S(Z)=Z + (Z-Z̄)AB + R̄`。

其中 `Z_c/R_c` 是去均值张量，`r` 是固定低秩，`λ>0` 保证小样本线性系统稳定。Eckart–Young 只保证 `B` 对校准残差的 Frobenius 近似性质；它不保证经过 HiF4 离散编码后，或在未见样本上的 Attention 输出更好，所以必须使用留出 exact gate。

对每个 KV group `g`，门禁累计 teacher 与 student 相对原浮点输出的完整 SSE：`E_t,g=Σ||O_t,g-O_ref,g||²`、`E_s,g=Σ||O_s,g-O_ref,g||²`。只有拟合状态有限、每个留出样本均满足预算、且聚合 `E_s,g≤(1+ε)E_t,g` 时才接受。Q 对应的所有 GQA heads 与同组 K/V 原子切换，防止只替换一侧破坏消费者关系。

| 问题证据 | 理论依据 | 算法动作 | 目标指标 | 否证条件 |
|---|---|---|---|---|
| v22 Attention 动态长链耗时占比高（v22 report `:84,107`） | teacher distillation；截断 SVD 提取固定秩残差子空间，适用假设是校准与动态输入共享可复用方向 | 校准期拟合 Q/K/V 的单次低秩残差算子；运行期 accepted group 跳过多阶段 repair | Attention 动态阶段耗时相对 v22 降低至少 25%，student 有效 group 覆盖非零 | 动态耗时降低不足 25%，或有效覆盖为 0 |
| v27 低秩替换 `0/8` 接受且未保留完整 teacher 动态链 | Softmax 输出按 KV group 可独立计算；直接计算比局部二次代理更贴近最终消费者 | 用独立留出样本比较完整 `softmax(QKᵀ/√d)V`，Q/K/V 原子门禁；拒绝 group 执行原 v22 teacher | Attention MSE 不高于 v22 的 `1.01×`，综合分下降不超过 100 | Attention MSE 增加超过 1%，或得分下降超过 100，或拒绝 group 未走 teacher |

## 选定修改方案

### 核心算法

唯一结构假设是“**输出门禁的逐组 teacher-stack 蒸馏**”：先完整构造 v22 teacher state；前三个校准 shard 在有界 token 子集上同时取得 teacher 合法输出并拟合 Q/K/V 的 residual rank student；剩余两个 shard 同时运行 teacher 和 student，按 KV group 比较完整消费者输出；随后把同组 Q heads、K 和 V 原子冻结到 student 或 teacher。student 只做一次低秩残差映射和一次既有合法编码，不执行 v22 的 Hessian/wide/two-coordinate/gauge-overlay 或 V 多阶段 repair。

### 修改目标

- 首要目标：在保持 v22 精度 Pareto 的约束下减少 Attention 动态耗时；正式目标为动态阶段降低至少 25%。
- 安全目标：相同固定 5+5 样例上，综合分相对 `21923.919748653454` 下降不超过 100，Attention MSE 相对 `0.00014752816557994677` 增加不超过 1%。
- 可执行性目标：student accepted group 数和动态 effective coverage 均非零；否则机制没有运行证据。

### 修改范围

- 只修改目标版本的 Attention calibration 后处理、Attention 动态 dispatch、低秩拟合/执行、参数拼接和诊断；Linear 六接口中的两个函数保持算法路径不变。
- 设校准样本 token 总数为 `N_c`、head 数为 `H`、head dimension 为 `d`、rank 为常数 `r`。teacher 校准验证仍承担有界多阶段代价；新增拟合主项为 `O(H(N_c²d+N_c³))` 的对偶 ridge/SVD，且通过每 shard token 上限约束。accepted group 的动态 student 为 `O(NHdr + encode(NHd))`；rejected group 保留 v22 teacher 成本。混合拼接为线性 `O(NHd)`。
- 校准回退条件：校准 shard 不足、shape 不匹配、teacher/student 输出非有限、SVD/solve 失败或无独立 gate shard。动态回退条件：schema、head mask、factor、shape 或拼接不合法。所有这些情况 fail closed 到原 v22 teacher。

### 保持不变

- NVFP4 的 E2M1×E4M3、16 值共享 scale、shape 恢复；HiF4 合法字段与 64 值层级编码。
- v22 完整 teacher、Q/K 互逆变换、未接受 group 的所有 repair；Linear weight 与 activation 路径。
- 官方六接口的名称、参数顺序和返回结构；不读取 case 身份，不执行正式评测。

## 算法内部超参数计划

本版本只保留一个主配置，避免把 rank/门槛微调伪装成新算法：

| 配置 | residual rank | 每 shard token 上限 | fit/gate | exact 相对预算 `ε` | 选择依据 |
|---|---:|---:|---:|---:|---|
| `main` | `4` | `32` | `3/2` | `0.25%` | rank 4 与 v22 已使用的少量消费者主方向量级一致；32 token 控制 teacher 校准开销；3/2 保证 gate 不参与拟合；0.25% 留出误差预算严于正式 1% 否证线 |

该配置由 Runner 在相同固定 5 Linear + 5 Attention 样例上正式评测。停止条件为完整评测结束；本阶段不依据非正式 smoke test 选择参数，也不创建 trial 或后续版本。

## 实施步骤

1. 保留并调用 v22 原 Attention 校准，得到 teacher 的 Q/K/V state；不给 teacher repair 改写目标或阈值。
2. 将校准样本确定性切为前三个 fit shard 与后两个 gate shard；每个 shard 仅按 shape 等距选取最多 32 个 token，不读取样例标识。
3. fit shard 上对 Q/K/V 同时运行 teacher；Q/K student 输入保留 smooth、reciprocal transform 与 K 的等价 gauge 预处理，V 使用原坐标；拟合每 head 的 rank-4 残差算子。
4. gate shard 上分别运行完整 teacher 与单次 student，计算原浮点参考、teacher、student 的逐 KV-group 完整 Attention 输出 SSE；原子生成 accepted mask，并写回三个 role state。
5. 动态期按 mask 分离 heads：accepted 子集执行 student，rejected 子集执行未经简化的 teacher，再按原 head 顺序拼接五个 HiF4 字段；任一状态异常时整次调用回退 teacher。
6. 实现只读 `hif4_get_diagnostics()`，累计 calibration success/fallback、fit/gate shard、group candidate/accepted/rejected、完整输出与 student-teacher SSE，并分别记录逐行中心化 logit、参考 Softmax Jacobian 一阶传播及隔离 `P_refΔV` 路径的 SSE；另记录动态 student/teacher/mixed 调用、head 覆盖和 state/shape/nonfinite/merge 回退。重复读取诊断不得改变计数或算法输出。
7. 只做语法、静态结构和人工小张量的六接口/shape/finite/JSON 诊断自检；不访问 `datasets/`/`reference/`，不运行 compact 或正式评分，不生成 `report.md`。

## 预期结果

- **待验证假设**：至少一个 KV group 通过独立 exact 输出门禁，并在正式动态样例中实际走 student；相对 v22，Attention 动态耗时下降至少 25%。
- **待验证假设**：逐组 fail-closed dispatch 将 Attention MSE 增幅限制在 1% 内，并使综合分下降不超过 100；理论与校准门禁只能支持方向，固定 5+5 正式结果才可确认本地有效性。
- **已验证边界**：Linear 算法路径保持不变，但没有 `ΔXWᵀ/XΔWᵀ/ΔXΔWᵀ` 独立诊断，因此尚不能宣称 Linear 数值逐位相同或任一传播项主导。

## 验收标准

1. 官方六接口、shape、dtype、合法 HiF4 字段和有限性检查全部通过；NVFP4 固定语义不变；`hif4_get_diagnostics()` 可被严格 JSON 序列化且只读。
2. Runner 的固定 5 Linear + 5 Attention 正式评测中，综合分不低于 `21823.919748653454`，Attention MSE 不高于 `0.00014900344723574624`；Linear 出现变化时必须由相同源码路径或评测噪声解释，不能归因于本 Attention 机制。
3. 同口径 Attention 动态阶段耗时相对 v22 降低至少 25%；`group_accepted>0`、`dynamic_student_heads>0`，并同时报告 `dynamic_teacher_heads`，以区分 shape 可表示性与实际覆盖。
4. rejected group 必须继续执行完整 v22 teacher；任一 schema/shape/factor/非有限异常必须计数并 fail closed，不能静默改走简化 parent。
5. 若得分下降超过 100、Attention MSE 增加超过 1%、动态耗时降低不足 25% 或 student 有效覆盖为 0，则整体假设判为否证；不得从本地 5+5 结果外推隐藏集。
