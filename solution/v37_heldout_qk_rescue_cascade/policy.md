# v37_heldout_qk_rescue_cascade 优化策略

## 基准版本

- **唯一比较基准**：`v23_budgeted_repair_router`。`based_on` 只表示本版本的代码来源和比较对象，不形成父子树关系。
- **已验证事实**：`solution/v23_budgeted_repair_router/report.md:5-11` 记录 Runner 当前统一固定 5 Linear + 5 Attention 正式结果：Linear MSE=`0.002541630606880181`，Attention MSE=`0.00019872973115909003`，按 300 例尺度折算的最终得分=`19778.137445442084`；`:15-19` 记录总耗时=`29.82529370021075 s`。这些是本分支唯一基准值，不能外推平台隐藏集。
- **已验证事实**：同报告 `:54`、`:87-91` 记录基准得分距阶段参考线 `20000` 尚差 `221.86255455791616`，Attention 负收益样例=`0/5`；Attention 校准/正式样例阶段分别耗时 `8.136203499510884 s`、`7.263448999263346 s`。
- **旁证而非基准**：`solution/v27_softmax_quotient_solver/report.md:15-18` 的同口径总耗时为 `40.392862499691546 s`。任务把该数值指定为速度否证线，但精度比较仍只对 v23。

## 实现基础

- **任务角色**：`search_mode=exploit`；**实现方式**：`based_on=v23_budgeted_repair_router`。
- **实际代码来源**：Runner 复制的 `solution/v23_budgeted_repair_router/solution.py`；实施前目标与来源 SHA-256 同为 `7D269EC84AC69714...`，共 `8554` 行。
- **复用且冻结的模块**：完整 Linear 两接口、NVFP4 解码、合法 HiF4 六字段编码、Q/K 互逆平滑与变换、Q/K 完整专家、V 完整消费者修复专家，以及 v23 的风险、pilot、`0.30` 一级预算求解和 V 动态 gather/expert/scatter 链。
- **重写模块**：只重写 Attention 路由发布阶段：一级仍调用 v23 原求解器并原样保留其 V mask；忽略一级 QK mask，新增“V 已修复输出残差 → QK 单元精确输出候选 → 不重叠留出折逐级门禁”的二级救援层。动态 Q/K 仍读取一个冻结的共享 mask，不引入跨公开接口的运行时通信。
- **exploit 正证据**：`solution/v23_budgeted_repair_router/report.md:58-60` 记录 V 静态选择 `15/16`、动态选择与有效改写 `75/75`，所有状态、shape 和异常回退均为 `0`；`:90-91` 记录总耗时 `29.82529370021075 s`，且 Q/K 动态专家耗时为 `0`。因此先保持已实际执行且处于速度 Pareto 的 V 路由，再只验证未获配额的 QK 边际贡献。
- **证据边界**：共享平滑、Hessian 和专家状态仍由完整 calibration list 构造；“留出”严格指二级救援候选的排序/发布不读取后两样例的损失，而不宣称共享 v23 状态与留出样例统计独立。这个泄漏边界会在诊断和报告中保留，不能把两折通过解释成隐藏集保证。

## 固定输入边界

- NVFP4 输入反量化严格保持来源 `dequantize_nvfp4()`（`solution/v23_budgeted_repair_router/solution.py:347-358`）：输入 E2M1 值逐项乘对应输入 E4M3 scale，最后一维每 16 个连续值共享一个 scale，随后恢复原 shape。
- 不重新估计、替换、重排输入 scale，不改变数值语义、16 值分组或 shape 还原；输入反量化不是优化变量。
- 输出仍只发布现有合法 E6M2 外层 scale、LV2、LV3、sign 与 mant 字段。QK 救援只复用来源专家生成的合法字段并按 H64 块散射，不发明格式外码字。
- 六个公开接口、CPU 边界、评分公式、评测器和数据保持不变。本实现阶段不读取 `datasets/`/`reference/`，不运行 compact、筛选或正式评测，也不生成 `report.md`。

## 问题分析

### 已验证事实

1. v23 在 `_attention_build_repair_router()`（`solution/v23_budgeted_repair_router/solution.py:8073-8244`）把 QK 与 V 放入同一个收益密度预算；`_solve_repair_router()`（`:8033-8070`）按 `benefit/cost` 单层排序，没有为已通过 pilot 的 QK 保留第二级可行域。
2. `solution/v23_budgeted_repair_router/report.md:41-44` 记录 QK/V pilot 都是 `2/2` 折正改善，汇总改善比例分别为 `33.0298340359%`、`29.3411848446%`；但 `:58-59`、`:91` 记录 QK 最终静态选择 `0/8`、动态有效覆盖 `0/360`，V 则为静态 `15/16`、动态有效 `75/75`。所以“QK 无正式执行证据”是事实，“QK 无用”不是事实。
3. v23 的 QK pilot `_router_pilot_qk_expert()`（来源 `:7919-7985`）只比较一个窗口的 Jacobian 一阶风险；它没有以已修复 V 重算完整 Softmax 输出，也没有用未参与路由排序的样例验收。
4. **相关正例**：`v0_softmax_aware_qk` 的正式记录显示相对 `v0_hessian_repair` 得分增加 `114.946483006621`、Attention MSE 降低 `4.5817035527663e-06`；这支持输出敏感 Q/K 方向存在局部正收益可能，但不证明 v23 专家在当前样例上有效。
5. **相关反例**：`solution/v32_joint_qk_output_gate/report.md:62-68` 记录底层代理接受率 `83.4325979838163%`，但 exact-output gate 接受 `0/8`、动态有效覆盖 `0/10`；`:73-76` 还记录正式得分与 Attention 汇总 MSE方向相反。因此本版本必须 fail-closed，并同时保留逐折精确损失、正式逐例得分和 MSE，不能用代理接受率替代输出证据。

### 输出误差分解

- **Linear（实现保持不变）**：令 `X̂=X+ΔX`、`Ŵ=W+ΔW`，则 `X̂Ŵᵀ-XWᵀ=ΔXWᵀ+XΔWᵀ+ΔXΔWᵀ`。本版本不修改 `hif4_calibration_and_quantize_weight()` 或 `hif4_dynamic_quantize_activation()`；源码不变只能支持“算法路径不变”，Runner 未提供三项独立数值，故三项主导关系仍为**证据不足**。
- **Attention 中心化 logit**：令 `L=QKᵀ/√d`，`C=I-11ᵀ/T_k`。Q/K 量化引入的有效误差为 `ΔL_c=(ΔQKᵀ+QΔKᵀ+ΔQΔKᵀ)C/√d`。v23 的 `_router_attention_risks()`（来源 `:7731-7748`）已先合并三项再中心化和平方；本版本保留该诊断，但候选发布改用完整 Softmax 输出。
- **Softmax Jacobian 敏感路径**：逐行 `p=softmax(l)` 的 Jacobian 为 `J(p)=diag(p)-ppᵀ`，一阶 `Δp≈J(p)Δl_c`。v23 已记录中心化能量和 Jacobian 输出风险（报告 `:65-67`），却因 QK mask 为空未验证其动态贡献；因此“剩余 QK 风险可补足得分”是**待验证假设**。
- **V 与交叉路径**：`Y=PV`，完整误差为 `(P̂-P)V + P(V̂-V) + (P̂-P)(V̂-V)`。一级先冻结 v23 的 V mask，二级以该 `V̂` 下的完整输出残差 `r=Y-Y_V` 衡量 QK 增量，因而 exact gate 同时看到 Softmax 高阶项和 `(P̂-P)(V̂-V)`，不会把 V 路径当作未修复常量。
- **格式误差**：削顶是目标幅值超出当前层级最大合法值后的饱和残差；分辨率误差是在可表示范围内落到离散格点的残差。v23 报告 `:68` 的 pilot 中 clipping SSE=`150.027503490448`、resolution SSE=`1466.228858947754`，但无输出级格式消融。本版本不改 scale/LV2/LV3 候选域，只复用专家；二者谁主导正式误差仍是**证据不足**。

### 可证伪根因

- **待验证假设**：v23 的一级预算已经保留了高覆盖 V 修复，但把两个 proxy fold 均正向的 QK 路径完全挤出；若在冻结 V 输出后按剩余精确输出残差筛选 QK，并要求两个未参与排序的留出折都同向改善，则少量 QK 单元可提高综合分，同时保持 Attention `0` 个负收益和相对 v27 的速度优势。
- 已能排除“仅提高 0.30 预算即可构成本版本”，因为任务和 v23 报告 `:103-105` 明确禁止沿纯预算比例调参；尚不能排除 QK exact 候选全部被拒、校准折方向不稳定、多个窗口发生非加性交互，或新增校准/动态成本超过 `10.567568799480796 s` 的速度余量。

## 相关方案调研

- Bolin Gao、Lacra Pavel，*On the Properties of the Softmax Function with Application in Game Theory and Reinforcement Learning*，2017，[原始论文](https://arxiv.org/abs/1704.00805)。论文从 log-sum-exp 梯度推导 Softmax 的光滑性；本题使用其 Jacobian/平移不变结构解释中心化一阶敏感性，但最终接受使用 exact Softmax，避免把一阶近似当成局部有效性的事实。
- M. Stone，*Cross-Validatory Choice and Assessment of Statistical Predictions*，1974，[原始论文 DOI](https://doi.org/10.1111/j.2517-6161.1974.tb00994.x)。可借鉴点是用未参与规则选择的数据评估预测处方；本版本把前两样例用于候选排序、后两样例只用于发布门禁。共享 v23 状态仍看过完整列表，所以这里只能称“对救援选择留出”，不能宣称无泄漏统计估计。
- H. Isermann，*Linear Lexicographic Optimization*，1982，[原始论文 DOI](https://doi.org/10.1007/BF01782758)。字典序思想要求高优先级可行性先固定，再优化低优先级目标；本版本据此先冻结 V 路由和逐折输出不劣约束，再按拟合期实测增量成本确定 QK 检查顺序，不把质量与成本压成可被尺度支配的单一密度。
- Elias Frantar、Saleh Ashkboos、Torsten Hoefler、Dan Alistarh，*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*，2022，[原始论文](https://arxiv.org/abs/2210.17323)。它支持复用二阶消费者状态生成离散候选；v32 的项目反例说明二阶候选仍须经 exact-output 留出门禁，本版本不把 GPTQ 方向性依据写成当前收益保证。

## 理论分析

白话解释：V 已经是 v23 中唯一实际执行的 Attention 深修复路径，所以先把它当作不可回退的第一层。QK 候选不再与 V 争同一个预算，也不再只问局部二次风险是否下降；算法先真正量化并修复 V，计算此时离浮点 Attention 输出还差多少，再尝试一个 QK 窗口能否沿这个“剩余误差”方向补偿。候选排序只看前两折及其真实专家耗时，后两折完全不参与排序；发布时按低成本顺序逐个叠加，任何候选只要在一个留出折上不严格改善，就保持当前 mask 不变。

对校准样例 `s`，令浮点输出为 `Y_s`，冻结 V 路由后的基础量化输出为 `Y_{V,s}`，剩余残差为 `r_s=Y_s-Y_{V,s}`。加入 QK 单元 `i` 后输出增量为 `δy_{s,i}=Y_{V+QK_i,s}-Y_{V,s}`，则精确 SSE 改善为

`G_{s,i}=||r_s||²-||r_s-δy_{s,i}||²=2⟨r_s,δy_{s,i}⟩-||δy_{s,i}||²`。

实现直接重算等式左侧的完整 Softmax 输出，不用右侧近似做接受；按每个 KV group 的浮点输出能量归一化，避免大能量样例单独支配：`ℓ_{s,g}=||Ŷ_{s,g}-Y_{s,g}||²/max(||Y_{s,g}||²,ε)`。候选先要求两个 fit fold 的 `G>0`，再按 `max_f cost_{f,i}` 递增、最小 fit 增益递减和固定索引排序。留出折在当前已接受组合上重算 `ℓ`；只有所有留出折对对应 KV group 都严格下降且字段确实改变才接受。便宜候选先进入会让后续冗余昂贵候选失去严格改善资格，这是“输出可行性优先、可行域内成本次优先”的有界贪心实现，不宣称全局字典序最优。

每个核心动作的闭环如下：

| 问题证据 | 理论依据 | 算法动作 | 目标指标 | 否证条件 |
|---|---|---|---|---|
| v23 V 动态有效改写 `75/75`、Attention 负收益 `0/5`，而 QK 为 `0/360` | 字典序优化先固定高优先级可行解（Isermann，1982） | 原样调用 v23 一级求解，冻结其 V mask；一级 QK mask 不发布 | V mask/effective 数与 Linear/Attention 稳定性 | V mask 相对来源流程改变、V effective 变为 0，或出现新增 V 回退 |
| QK pilot `2/2` 正向却因单层密度得到 `0/8` 配额 | 二阶消费者模型可生成候选，但不能代替输出验证（GPTQ，2022；项目 v32 反例） | 在 V-repaired 基线上对原 `8` 个同构 QK 单元逐一运行原专家，不改预算比例或候选数 | QK fit eligible、字段 changed、剩余输出损失 | 所有 fit 候选不改善，或候选未实际改变合法字段 |
| v23 pilot 只比较局部 Jacobian 风险，未看完整 V 交叉项 | 完整 `Ŷ-Y` 同时包含 `(P̂-P)V`、`PΔV` 与交叉项 | 用 exact Softmax+已路由 `V̂` 计算逐折归一化输出损失，并记录中心化 logit SSE | Attention MSE、逐例得分、门禁 margin | gate 通过但正式 Attention MSE/得分逆向，或诊断非有限 |
| v32 高代理接受率但 output gate `0/8` | 选择与评估分开可检验规则外推（Stone，1974；适用边界如上） | 前 2 折排序、后 2 折逐级同向门禁；失败只清空 QK rescue | crossfold accepted/total、fallback、Attention negative count | 任一已发布单元在某留出折不改善，或无留出时仍发布 QK |
| v23 比 v27 快 `10.567568799480796 s` | 质量约束满足后以实测增量成本作为次级顺序 | 低成本安全候选先进入 cascade；动态仅 gather 已发布窗口 | 总耗时 `<40.392862499691546 s`、Q/K expert ns | 总耗时达到或超过 v27，或 QK 选择与实测成本排序不一致 |

## 选定修改方案

### 核心算法

实现一个**冻结 V 路由的留出 QK 残差救援级联**，只验证这一项结构假设：

1. 保持 `_REPAIR_ROUTER_BUDGET_FRACTION=0.30`、原 pilot 数、原 QK/V 候选集合与 `_solve_repair_router()`；保存其 V mask 作为一级冻结决策，不允许二级失败把 V 路由回退为 full-v22 或基础编码。
2. 使用 v23 的两个 pilot 样例作为 fit folds，并从后续样例确定性取两个不重叠 heldout folds；所有折仍最多等距抽取 16 token。少于 4 个可用样例时 QK fail-closed，V mask 继续发布。
3. 对每折先按 V mask 真实运行 V 专家并得到 `V̂_route`；随后对每个原 QK 原子窗口运行与动态路径同构的 Q/K 完整专家，重算完整 Softmax 输出和逐 KV group 归一化损失。只保留两个 fit fold 都改善且有合法字段变化的单元。
4. 候选顺序只由 fit 期最大实测成本、最小 fit 改善和固定索引决定。对两个 heldout folds 从 V-only 参数开始逐个叠加；候选必须在当前组合上两折都严格降低对应 group 的 exact loss才加入共享 Q/K mask，否则拒绝。异常时二级 mask 全空并记录分类回退，一级 V mask不变。

### 修改目标

- 首要目标：固定 5+5 正式得分 `≥20000` 且严格高于 v23 的 `19778.137445442084`。
- 稳定性目标：Attention 负收益样例保持 `0/5`；定位 Attention MSE 是否不高于 v23 的 `0.00019872973115909003`，但不以校准 loss 代替正式 MSE。
- 性能目标：总耗时 `<40.392862499691546 s`，保留 v23 相对 v27 的当前速度优势。
- 因果定位：Linear MSE应与 v23 相同；诊断分别返回一级 legacy QK/V 选择、二级 fit eligible、heldout 比较/接受/拒绝、V-only 与候选归一化损失、实际字段改变、候选/选中成本及各类回退。

### 修改范围

- 只修改 `solution/v37_heldout_qk_rescue_cascade/policy.md` 与 `solution/v37_heldout_qk_rescue_cascade/solution.py`；不创建 trial 或 report。
- 渐近复杂度：设每折 token 上限 `P=16`、fit/heldout 折数各 `F=H=2`、QK 单元数 `N=H_k·d/W`、每 KV 组 Q 头数 `G=H_q/H_k`、窗口 `W≤256`。一级 v23 复杂度不变；新增校准专家调用上界为 `O((F+H)·N·P·(G+1)·W²)`，exact Attention 重算为 `O((F+H)·N·P²·H_q·d)`，候选排序为 `O(N log N)`。峰值额外内存按折和候选顺序复用，为 `O(P·(H_q+2H_k)·d + P²·H_q + P·(G+1)·W)`，不缓存全部数据集。
- 动态期 V 成本逐位沿用 v23；若救援选择 `S_qk` 个单元，只新增原有紧凑 Q/K 专家 `O(T·S_qk·(G+1)·W²)`、gather/scatter `O(T·S_qk·(G+1)·W)`。不发布时动态 Q/K 与 v23 一样走基础合法编码。
- 校准回退分层：一级 v23 状态/shape/pilot 失败仍走来源安全路径；二级无两折留出、非有限 exact loss、候选字段/shape错误或专家异常，只清空 QK rescue 并保留已求出的 V mask。动态 mask/schema 异常沿用来源 full-repair 安全路径并分类计数。

### 保持不变

- NVFP4 E2M1×E4M3 输入语义、连续 16 值分组与原 shape恢复。
- Linear 全部代码、常量、状态和两个公开接口。
- Q/K 互逆浮点等价变换、基础 K gauge、合法 HiF4 字段及原完整 Q/K 专家内部算法。
- v23 一级预算比例、候选个数、V mask求解与 V 动态修复链。
- 评测器、评分公式、数据、Runner、其他版本和正式评测流程。

## 算法内部超参数计划

只保留一组主配置；本任务验证的是二阶段误差目标和留出发布流程，不以折数、阈值、预算或候选数做内部调参：

| 配置 | 一级路由 | QK fit / heldout | 质量约束 | 次级成本规则 | 停止条件 |
|---|---|---:|---|---|---|
| `main`（`solution.py`） | v23 原 `0.30` 预算并冻结所得 V mask | `2 / 2` 个不重叠样例，每折最多 16 token | fit 两折及 cascade 当前状态下 heldout 两折均严格降低逐 group 归一化 exact loss | 按 fit 最大实测 ns 递增；同成本按最小 fit 增益递减、索引递增 | 候选耗尽；无留出或异常则 QK 全拒绝但保留 V |

所有正式判断由 Runner 对同一 `main` 执行固定 5 Linear + 5 Attention；没有 `trials/`，也不声称 `2/2` 或任何成本顺序已由本轮调优。

## 实施步骤

1. 先固化本 policy，再编辑复制的 `solution.py`；不触碰目标目录外文件。
2. 扩展只读累计诊断：一级 legacy mask、V 冻结一致性、fit/heldout 样例数、候选/eligible/accepted/rejected、逐折 improve/equal/worse、精确归一化损失、中心化 logit SSE、实际字段变化、候选/选中成本和分类回退；`hif4_get_diagnostics()` 继续只返回普通 `int/float` 副本。
3. 增加 V-mask pilot 发布、单 QK 原子候选构造及完整 Attention 输出损失 helper；所有 helper 复用固定解码和现有合法专家，不修改公开接口。
4. 在 `_attention_build_repair_router()` 中先执行原 v23 pilot、族收益和 `_solve_repair_router()`，冻结返回的 V mask；保留一级选择诊断。
5. 以前两 pilot 为 fit、后两样例为 heldout，在每折先真实应用冻结 V mask；逐原 QK 单元计算 fit 剩余残差改善和实测成本，按预定字典序排序。
6. 在 heldout 当前组合上逐候选重算 exact output；两折严格改善才同步发布 Q/K mask。任一二级异常时发布空 QK mask而非破坏 V mask。
7. 动态 Q/K/V 继续使用来源 gather/expert/scatter；检查 selected 与 effective coverage、Q/K mask一致、V mask有效、schema与字段 shape。
8. 只运行静态语法、导入、六接口签名、合成小张量 shape/dtype/有限性、NVFP4 16 值 scale 对应及诊断 JSON 序列化检查；不访问官方数据，不运行任何正式评测，不生成报告或后续方向。

## 预期结果

- **待验证假设**：V mask、V 动态有效覆盖、Linear 输出和 v23 保持；QK rescue 获得非零但稀疏的 accepted/effective coverage，Attention exact residual 和正式综合分提高。
- **待验证假设**：跨两个留出折的逐级严格门禁保持固定 5 个 Attention 样例负收益为 `0`，并使 Attention MSE方向不劣于 v23。
- **待验证假设**：按实测成本优先的稀疏 QK 救援把新增端到端耗时控制在相对 v27 的 `10.567568799480796 s` 余量内。
- QK 全拒绝是允许且可诊断的 fail-closed 结果，但会否证“补足得分缺口”的核心假设；不从任何本地结果外推隐藏集。

## 验收标准

1. 官方六接口格式检查通过；输出字段、shape、dtype 和数值有限；NVFP4 固定语义未改变；`hif4_get_diagnostics()` 可由 `json.dumps(..., allow_nan=False)` 序列化且调用前后状态相同。
2. Runner 正式评测必须恰好使用固定 5 Linear + 5 Attention；合成检查、compact 或校准损失不得登记为正式收益。
3. 一级 V mask 来自未改的 v23 求解结果；二级失败不得使 V mask丢失。诊断须区分候选、fit eligible、cross-heldout accepted、动态 selected/effective 及 fallback；无两折留出时 QK accepted 必须为 `0`。
4. 核心联合成功线：正式得分 `≥20000` 且 `>19778.137445442084`，总耗时 `<40.392862499691546 s`，Attention 负收益样例数=`0`。任一不满足即按任务给定条件否证整体假设。
5. 定位比较：Linear MSE 应与 v23 的 `0.002541630606880181` 一致且无新增 Linear 回退；记录 Attention MSE相对 `0.00019872973115909003` 的方向、逐例分布和 QK gate margin。若 QK mask为空，正式差异不得归因于 QK；若非空但 dynamic effective 为 0，执行机制不成立。
6. 只有报告阶段能依据真实固定 5+5 指标把假设标记为“结果支持 / 结果否证 / 证据不足”；本阶段不生成 `report.md`，不提出后续版本。
