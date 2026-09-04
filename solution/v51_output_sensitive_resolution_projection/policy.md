# v51_output_sensitive_resolution_projection 优化策略

## 基准版本

- 唯一比较基准为 `v37_heldout_qk_rescue_cascade`；`based_on` 只表示比较与代码来源，不形成版本树。
- **已验证事实**：`solution/v37_heldout_qk_rescue_cascade/report.md:5-10` 记录 Runner 固定 5 Linear + 5 Attention 正式结果：Linear MSE=`0.002541630606880181`，Attention MSE=`0.0001513888368008864`，按 300 例尺度折算的最终得分=`21765.67385377308`，总耗时=`47.68690699990839 s`。这些数值是本版本唯一正式比较线，不能外推平台隐藏集。
- **已验证事实**：同报告 `:64-68` 记录 QK 静态接受 `7/8`、动态有效改写 `315/315`，冻结 V 静态选择 `15/16`、动态有效改写 `75/75`，路由相关回退为 `0`。因此本版本保留这条已有正式正证据的 QK/V 路由，不用新的格式假设替换它。

## 实现基础

- `search_mode=explore`，实现方式为 `based_on=v37_heldout_qk_rescue_cascade`。Runner 已把来源 `solution.py` 原样复制到本目录；实施前目标文件与来源文件 SHA-256 同为 `C2736583AE8BBE16EE7F7A6424A26288A265B6D6085FC1AB183C6875BF633F92`。
- 复用来源的六个公开接口、Linear 路径、Attention 平滑/互逆变换、QK 两折救援、冻结 V 路由、合法 HiF4 编码器和安全回退。只新增 V 路径的 resolution 专用相邻格点求解、两折发布门禁、动态应用和诊断。
- 选择该来源是因为其 Attention MSE 与零负收益已有正式正证据，且 `solution/v37_heldout_qk_rescue_cascade/report.md:78` 又提供了尚未因果验证的格式分解；探索的信息增益是固定其他路径后检验 resolution，而不是继续调整 v37 的预算、折数、阈值或候选数。

## 固定输入边界

- NVFP4 输入反量化严格保持来源 `dequantize_nvfp4()`（`solution/v37_heldout_qk_rescue_cascade/solution.py:384-395`）：输入 E2M1 数值逐项乘对应输入 E4M3 scale，最后一维每 16 个连续值共享一个输入 scale，随后恢复原 shape。
- 不重估、替换、移动输入 scale，不改变 16 值分组、输入数值语义或 shape。新增算法只在输出 HiF4 的既有五字段 `scale_factor/scale_lv2/scale_lv3/sign/mant` 合法域内选择。
- 输出 HiF4 继续采用来源编码域：每 64 值一个 outer E6M2 `scale_factor`，8 个 LV2、16 个 LV3，以及 `mant∈{0,0.25,…,1.75}` 与零值规范 sign。本文不把该输出 mant 字段称为输入 E2M1，也不改变任何输入解码规则。

## 问题分析

### 已验证事实

1. 来源 `_router_format_error_split()`（`solution/v37_heldout_qk_rescue_cascade/solution.py:7669-7687`）以当前合法层级最大值 `1.75·scale_factor·scale_lv2·scale_lv3` 区分削顶与范围内重建误差；`_router_pilot_state()`（`:7849-7954`）只累计这两个张量 SSE，未据其发布格式动作。
2. `solution/v37_heldout_qk_rescue_cascade/report.md:78` 的校准诊断为 clipping SSE=`345.646861076355`、resolution SSE=`3364.0738677978516`；后者是前者的 `9.7326903456378` 倍，占两者合计 `90.6826716527190%`。这是“来源校准代理中 resolution 较大”的事实，不是“resolution 主导正式 Attention MSE”的事实。
3. **相关正例**：同报告 `:74-76,87-89` 记录中心化 logit SSE 与完整输出 loss 同向下降，Attention MSE 相对 v23 下降 `0.00004734089435820363`，说明消费者输出对齐的候选门禁在当前来源上可实际执行；但没有格式开关，不能把收益归给 resolution。
4. **相关反例**：Runner 正式记录 `v33_joint_scale_code_dp` 相对 v25 得分下降 `1294.7654807541876`，Linear/Attention MSE 分别增加 `47.18695806543649%`/`6.553605662631177%`；其完整代理虽改善 `4429.626092725142`，resolution SSE 反而增加 `10956.111709594727`。这否定“局部代理接受或联合 scale-code 搜索本身足够”，要求本版本固定 clipping 并由 exact output 留出门禁发布。
5. **结构失败边界**：Runner 对 `v45_resolution_clipping_frontier_solver` 的审查指出 raw Pareto 剪枝与消费者加权目标不对应、缺 heldout 时未 fail-closed、复杂度漏掉二次支配张量；对 `v48_causal_format_dual_code_router` 的审查指出理论码域与实现不一致、Grouped-Query Attention 成本漏乘 Q-head 倍率、缺少逐例门禁诊断。本版本不做 raw Pareto 剪枝，缺任一留出折即发布空动作，并显式计入 `G=Hq/Hkv`。

### 输出误差分解

- **Linear**：令 `X̂=X+ΔX`、`Ŵ=W+ΔW`，则 `X̂Ŵᵀ-XWᵀ=ΔXWᵀ+XΔWᵀ+ΔXΔWᵀ`。本版本不修改两个 Linear 公开接口及其调用链；这只能支持“代码路径不变”，Runner 未给出三项独立数值，故三项大小仍为**证据不足**。
- **Attention 中心化 logits**：令 `L=QKᵀ/√d`、`C=I-11ᵀ/T`，则 `ΔL_c=(ΔQKᵀ+QΔKᵀ+ΔQΔKᵀ)C/√d`。来源 QK 路由保持不变；新增格式动作不改 Q/K，因此不会新增该项，但不能宣称来源该项已最优。
- **Softmax Jacobian**：对一行概率 `p=softmax(l)`，`J(p)=diag(p)-ppᵀ`，有效一阶响应为 `J(p)Δl_c`。本版本不以该一阶式验收；heldout 门禁重算 exact Softmax 输出。
- **V 路径**：`Y=PV`，完整误差为 `(P̂-P)V + PΔV + (P̂-P)ΔV`。冻结 QK 后，输出对 `V̂` 是严格线性的：`δY=P̂δV`。因此可用 `P̂` 的 JVP 和局部曲率 `P̂ᵀP̂` 精确计算单个 V 格点动作对当前输出 SSE 的增量，再由 exact-output 门禁处理多动作交互。
- **格式 clipping/resolution**：clipping 是 `|v|>1.75·s_eff` 的饱和残差；resolution 是 `|v|≤1.75·s_eff` 时落到离散格点的残差。新增动作只允许改写在旧、新层级下都未削顶的元素；mant 使用相邻有符号码，LV2/LV3 仅在 `{1,2}` 间翻转并重新就近编码受影响的 8/4 个值。因而算法动作针对 resolution；任何非零 clipping 增量都是实现不合格而非优化收益。

### 可证伪根因

- **待验证假设**：在保持 v37 QK/V 路由和 clipping 分类不变时，消费者 `P̂ᵀP̂` 加权的合法相邻格点动作若能在两个 fit 折和两个发布折逐折降低输出损失，则 `main` 的正式 Attention MSE 将低于 `0.0001513888368008864`，且 5 个 Attention case 的负收益数保持 `0`。
- 当前证据可排除“只增大候选集或降低阈值即可证明格式根因”；尚不能排除所有动作被门禁拒绝、静态通道动作无法迁移到正式样例、resolution 代理与正式输出错位，或新增成本破坏时间 Pareto。

## 相关方案调研

- Yuanyong Luo 等，*HiFloat4 Format for Language Model Inference*，2026，[原始论文](https://arxiv.org/abs/2602.11287)。论文定义 64 元素三级共享尺度格式；本题借鉴其层级约束，只在既有 outer/LV2/LV3/mant 合法表示内移动，不据论文宣称当前代码一定增益。
- Markus Nagel、Rana Ali Amjad、Mart van Baalen、Christos Louizos、Tijmen Blankevoort，*Up or Down? Adaptive Rounding for Post-Training Quantization*，ICML 2020，[PMLR 原文](https://proceedings.mlr.press/v119/nagel20a.html)。其二阶任务损失说明最近舍入在存在消费者交叉项时未必最优；本版本采用离散相邻动作，但以 Attention 输出的严格 V 路径二次增量而非论文的软松弛来排序。
- Bolin Gao、Lacra Pavel，*On the Properties of the Softmax Function with Application in Game Theory and Reinforcement Learning*，2017，[原始论文](https://arxiv.org/abs/1704.00805)。其 Softmax Jacobian与平移不变结构用于解释 QK 误差传播；因本版本优化 V，`P̂` 直接作为输出对 V 的线性算子，最终发布仍重算 exact Softmax。
- M. Stone，*Cross-Validatory Choice and Assessment of Statistical Predictions*，1974，[原始论文 DOI](https://doi.org/10.1111/j.2517-6161.1974.tb00994.x)。借鉴点是把动作提议折与发布评价折分开。既有平滑、Hessian、QK mask 已读取完整 calibration list，所以本版本只声称“对新增格式动作的选择留出”，不声称统计上完全独立或可外推隐藏集。

## 理论分析

白话解释：来源先完成 Q/K 与 V 的已有修复。本版本随后只观察 V 的范围内格点误差。对于一个 V H64 块，枚举 152 个结构动作：64 个位置各向相邻有符号码的两个方向、16 个 LV3 四值组翻转、8 个 LV2 八值组翻转。层级翻转后只对受影响值重新舍入；旧或新层级中发生削顶的行保持原字段。每个 fit 折都用量化后的 `Q̂,K̂` 得到 `P̂`，直接计算动作输出增量，要求两折严格改善；每块最多保留一个跨 fit 稳定动作。发布时按最小 fit 增益降序逐块叠加，在两个未参与该动作提议的折上重算完整 Attention，只有两个折对应 KV group 都严格改善才写入静态动作表。

对折 `s`，令来源量化输出误差为 `R_s=Ŷ_s-Y_s`。某合法格式动作产生 `δV_{s,i}`，冻结 QK 后 `δY_{s,i}=P̂_sδV_{s,i}`，归一化损失增量为

`Δℓ_{s,i}=[2⟨R_s,P̂_sδV_{s,i}⟩+||P̂_sδV_{s,i}||²]/max(||Y_s||²,ε)`。

这里第一项是沿当前输出残差的 JVP，第二项是由消费者局部曲率 `P̂ᵀP̂` 给出的步长代价；对固定 QK 和单个 V 动作该式是精确二次式，不是 Taylor 截断。多个块可能在输出空间相互抵消或增强，因此 heldout 阶段在当前已接受组合上重算等式左侧的 exact 输出损失，不把独立 fit 分数相加当作发布依据。

| 问题证据 | 理论依据 | 算法动作 | 目标指标 | 否证条件 |
|---|---|---|---|---|
| v37 resolution SSE=`3364.0738677978516`，占格式两项 `90.6826716527190%`，但无格式消融 | HiF4 三级合法域（Luo 等，2026）；范围内误差与削顶需分开 | 只在旧/新层级均未削顶处枚举 mant/LV3/LV2 相邻动作，clipping 非零变化即拒绝 | resolution SSE 增量、clipping 增量、字段 changed | 无合法改写，或任一发布动作改变 clipping SSE |
| v33 代理改善而 resolution 与正式 MSE 同时退化 | 消费者二次舍入目标（Nagel 等，2020），且 V 路径对固定 `P̂` 严格线性 | 用 `2⟨R,P̂δV⟩+||P̂δV||²` 逐折评分，不按 raw 张量 SSE/Pareto 剪枝 | fit 两折 gain、动作族接受率、Attention MSE | fit 接受但 heldout 方向不稳，或正式 Attention MSE 不降 |
| v37 已有 QK/V 正收益，格式因果仍不可识别 | 单路径消融才能归因；Softmax 输出需 exact 验收 | 固定 v37 路由，在其输出后叠加格式动作；提供 `format_off` 同版本正式消融 | main-off 配对 Attention MSE/得分、QK/V覆盖一致性 | off 不复现来源路径，或 main 不优于 off |
| 聚合门禁可能掩盖单样例退化；v45/v48 曾因 fail-open/诊断不足被拒 | Stone（1974）的选择/评价分离，适用边界如上 | 2 个 fit 折逐折提议，2 个 heldout 折逐折严格发布；缺折/异常一律空动作 | 每折 parent/trial loss、接受/拒绝、fallback、负收益 case | 缺 heldout 仍发布、任一已发布候选在某门禁折不改善，或正式负收益数>0 |

## 选定修改方案

### 核心算法

在 v37 的静态 QK mask 与冻结 V mask 完成后新增 `output-sensitive V resolution projection`：按 H64 块在合法相邻 mant/LV3/LV2 动作中，用两折 exact V-output 二次增量选一个提议，再以两个选择留出折的完整 Attention 输出逐级门禁发布。动态 V 先执行未改的 v37 路由，再按静态动作表对当前未削顶元素做相同合法投影；任何状态、shape、非有限值或 clipping 不变量失败均保持 v37 输出。

### 修改目标

- 首要目标：正式 Attention MSE 严格低于 v37 的 `0.0001513888368008864`，且 Attention negative_count=`0/5`。
- 因果目标：`main` 在完全相同固定 5+5 样例上优于 `format_off` 的 Attention MSE；Linear MSE 保持 `0.002541630606880181`，QK/V 原路由覆盖与回退不因格式开关改变。
- 次要目标：最终得分严格高于 `21765.67385377308`；总耗时记录为 Pareto 指标，期望不超过 `65 s`，但不以时间达标替代精度因果条件。

### 修改范围

- 只修改本版本 `policy.md`、主 `solution.py`，并增加 `trials/format_off/solution.py`。不修改共享文件、数据、评测器、Runner、其他版本或 report。
- 新增格式诊断、合法动作应用、V 输出上下文/二次评分、两折提议与两折 exact gate；在 `_attention_build_repair_router()` 中附加动作表，在 `_attention_routed_v_repair()` 的来源结果后应用。
- 记 `S=F+H=4`、pilot token 上限 `T=16`、`G=Hq/Hkv`、QK 窗口 `W≤256`、QK 单元数 `N=Hkv·d/W`、V 块数 `B=Hkv·d/64`、每块动作常数 `A=2·64+16+8=152`。为重建 v37 基线，新路径顺序重放至多 `N` 个 QK 专家和 `B` 个 V 专家；按来源专家的有界 H64/窗口坐标下降，其上界分别计为 `O(SNT(G+1)W²)` 与 `O(SB(T·64²+64³))`，不能从格式成本中省略。
- 加上基线重放后，新校准总时间上界为 `O(SNT(G+1)W² + SB(T·64²+64³) + SHqT²d + FBA(GT²·64+T·64) + HB(HqT²d+T(Hq+2Hkv)d))`；最后一项包含每个 heldout 候选的完整 QK/Softmax/V 输出重算、五字段参数副本和解码。动作逐个处理，不构造 `A×A` 支配张量。
- 新校准峰值空间为 `O(S[T(Hq+2Hkv)d+HqT²+HqTd] + T(Hq+2Hkv)d + (G+1)W² + T Hkv d + GT·64 + B)`；它显式包含四个 pilot/context、紧凑专家状态、概率/残差、一次完整 V 参数副本、单动作输出和动作表。动态新增时间为 `O(THkv d+B_selected·T·64)`，峰值新增空间为一次五字段副本 `O(THkv d)`；不缓存正式数据集。
- 回退条件：少于 4 个可用校准样例、非法 head/shape、动作表不匹配、非有限上下文/损失、候选改变 clipping、候选无字段变化或任一留出折不严格改善。校准级异常发布全零动作表；动态级异常返回已经生成的 v37 V 参数，不丢弃来源路由。

### 保持不变

- 六个公开接口的签名与返回字段不变；NVFP4 固定解码、Linear 全链、Q/K 动态链、v37 QK/V mask 求解、平滑和互逆变换不变。
- 不更改 outer `scale_factor`；mant 只走相邻有符号码，LV2/LV3 只在合法 `{1,2}` 间相邻翻转；所有未选块与被 clipping 保护的元素逐字段保持。
- `format_off` 只关闭新增动作拟合/应用，输出路径回到来源 v37；不改变其他超参数。

## 算法内部超参数计划

| 配置 | 新格式路径 | 结构依据 | 选择与停止规则 |
|---|---|---|---|
| `main`（`solution.py`） | 开启 128 个 mant 相邻动作、16 个 LV3 翻转、8 个 LV2 翻转；2 fit + 2 heldout | 同时覆盖码字分辨率与共享微尺度分辨率，且固定 clipping | 每块取两 fit 折均严格改善的最大最小增益动作；heldout 两折逐级严格改善才发布；候选耗尽即止 |
| `format_off`（`trials/format_off/solution.py`） | 关闭新增拟合和动态投影，保留 v37 全部路径 | 单路径正式消融，用于识别格式因果，不是另一版本 | 不搜索格式动作；作为 paired control 接受同一固定 5+5 正式评测 |

不增加第三组参数配置，避免把阈值、候选数量或动作数扫描伪装成结构版本。Runner 应在同一评测进程、同一固定样例索引上比较两配置；正式选择以 Attention MSE、负收益数、综合分为主，同时记录耗时。

## 实施步骤

1. 保持 `dequantize_nvfp4()` 字节不变；扩展只读有限标量诊断并保证 `hif4_get_diagnostics()` 返回副本，`json.dumps(..., allow_nan=False)` 可序列化且读取不改变状态。
2. 实现 152 个有界动作的统一解码/应用器：先检查旧网格 in-range；mant 只移动一个有符号码，LV3/LV2 翻转后重编码 4/8 值；新网格发生 clipping 时该行原样保留，并累计 before/after clipping 与 resolution SSE。
3. 为四个 pilot 真实应用 v37 已发布 QK mask 与冻结 V mask；构造 `P̂`、浮点参考输出、来源残差与归一化能量。格式路径不得复用 heldout 损失来选择动作。
4. 在两个 fit pilot 上逐块逐动作计算严格 V-JVP 二次增量；记录动作总数、逐折 improved/equal/worse、mant/LV3/LV2 提议、字段 changed、clipping/resolution 增量。诊断将 `resolution_candidate_total/output_gain` 与 `clipping_candidate_total/output_gain` 分列；本专用求解器的后者按设计为 `0`，任何 clipping 不变量破坏另计 rejected，不能混入 resolution 收益。两折不都严格改善则不进入发布池。
5. 在两个 heldout pilot 的当前组合上按 fit 最小增益降序逐候选重算 exact Attention 输出；两个折对应 group 都严格改善且 clipping 增量为零才接受，否则保持当前参数。缺折或异常时动作表全零。
6. 把动作表只附到 V state。动态期先完成 v37 V repair，再按当前实际值逐块应用；记录 selected/effective、各动作族应用、受 clipping 保护数、动态 before/after resolution、clipping delta 和分类回退。诊断必须区分静态可表示动作数与动态实际 effective blocks。
7. 增加 `format_off` 配置，仅覆盖新增模式常量；做 AST/导入、六接口签名、合法字段、shape、有限性、诊断只读性和写入范围检查。不得运行正式评测，也不得生成 report。

## 预期结果

- **待验证假设**：`main` 相对 v37/`format_off` 降低正式 Attention MSE并提高综合分，Linear MSE基本不变，Attention 负收益保持 `0/5`。
- **待验证假设**：至少一个 resolution 动作通过两折 fit 与两折 exact-output gate，并在动态正式调用中实际改变合法字段；其 clipping 增量为零、resolution SSE方向与输出增益可分别观察。
- 允许全部动作被 fail-closed 拒绝；这会保留 v37 数值路径，但否证“当前 resolution 根因可由该静态合法投影利用”。不从固定本地 5+5 外推隐藏集。

## 验收标准

1. 官方六接口格式检查通过；返回字段、shape、dtype 与值域合法，数值有限；NVFP4 固定输入语义与每 16 值 scale 映射未改变。
2. `hif4_get_diagnostics()` 调用前后算法状态相同且严格 JSON 可序列化；能计算 format 候选接受率、fit/heldout 逐折方向、fallback、动作族、静态 selected、动态 effective、clipping/resolution before/after 与阶段耗时。
3. 结构一致性：每个 mant 动作最多相邻一格；LV2/LV3 只在 1/2 间翻转；未削顶保护失败时字段不变；任一发布动作的 clipping SSE 增量必须为 `0`。缺少两个 fit 或两个 heldout 折时静态 selected 必须为 `0`。
4. 核心否证线采用任务给定口径：若 `main` 的正式 Attention MSE 不低于 `0.0001513888368008864`，或 Attention negative_count 大于 `0`，则否证当前格式根因的可利用性；若 `main` 不优于 `format_off`，不得把任何组合变化归因于新增格式路径。
5. 定位条件：Linear MSE 应与 v37 的 `0.002541630606880181` 一致；QK/V 静态与动态覆盖不得因格式开关出现无法解释的变化。动作全空或 dynamic effective=`0` 时，正式差异不得归因于 resolution 投影。
6. 记录总耗时并与 v37 的 `47.68690699990839 s` 比较；超过 `65 s` 记为性能目标失败，即使精度改善也只能作为精度 Pareto 证据。正式结果只由 Runner 固定 5+5 评测产生。
