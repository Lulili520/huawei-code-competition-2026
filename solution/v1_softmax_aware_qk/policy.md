# v1_softmax_aware_qk 优化策略

## 父版本与实现基础

- 名义父版本：`v0_hessian_repair`。父版本固定 5 个 Linear、5 个 Attention 样例的已记录结果为 Linear Output MSE `2.533227e-3`、Attention Output MSE `3.321124e-4`、总分 `5.983274`（`solution/v0_hessian_repair/report.md` 的“本地结果”表）。
- 实现基础：`parent`。完整复制父版本六接口与 Linear、V、HiF4 编码路径，仅重写 Attention 校准中的 Q/K Hessian 误差模型和相应状态标识。

## 固定输入边界

NVFP4 反量化保持父版本 `dequantize_nvfp4` 的规则：最后一维按连续 16 值展开，每组 E2M1 值逐项乘输入 E4M3 scale，随后恢复原 shape。实现不估计或替换输入 scale，也不改变数值语义与分组。

## 问题分析与证据分类

### 已验证事实

1. 父版本 `solution/v0_hessian_repair/solution.py::_attention_cross_hessians` 用对侧张量的未加权二阶矩构造 Q/K Hessian；它没有计算 attention logits、softmax 概率或 softmax 行方向的投影。
2. 父版本同文件 `hif4_calibration_attention` 以逐通道对侧二阶矩产生 importance，并由 `_attention_hessian_repair` 用二次型比较修复候选。故替换 Hessian 的统计定义会实际改变 Q/K 误差补偿流程，而不是只调整阈值。
3. 官方本地评测在 `.agent/skills/hif4-evaluate/scripts/evaluate.py:159-160` 以 `softmax(QK^T/sqrt(d)) @ V` 比较输出；目标并非 Q、K 各自的重构 SSE。

### 理论推导

Scaled dot-product attention 使用 `L=QK^T/sqrt(d)` 与 `P=softmax(L)`（Vaswani 等，《Attention Is All You Need》，2017，https://arxiv.org/abs/1706.03762）。对一行概率 `p`，softmax 的 Jacobian 为 `J(p)=diag(p)-pp^T`。因此 `J(p)1=0`：给整行 logits 加常数不改变概率；小扰动的局部二次代理为 `δl^T J(p) δl`，它自动消除该无效方向。

固定 K 时，Q 扰动满足 `δl=Kδq/sqrt(d)`，得到 Q 局部矩阵 `Hq=K^T J(p) K/d`。固定 Q 时，对第 j 个 key 的扰动可用 `sum_i J(p_i)[j,j] q_i q_i^T/d` 的正半定对角-token近似构造 `Hk`；忽略不同 key 之间的交叉块是为了让每个动态 K 行可独立编码。softmax Jacobian的协方差形式亦可由 Martins 与 Astudillo《From Softmax to Sparsemax》，2016，https://arxiv.org/abs/1602.02068 复核。

上述推导只支持“代理更贴近概率敏感方向”的机制，不证明最终 Attention output MSE 必然下降；`@V`、有限扰动高阶项及 Q/K 同时量化的交叉项均未被完整建模。

### 待验证假设

- 校准概率的 Fisher/softmax Jacobian 权重能比无条件对侧协方差更好地排序父候选与 Hessian 修复候选。
- 校准序列的敏感方向能泛化到固定测试序列。
- 对 K 使用 token 对角近似所损失的交叉信息小于其带来的独立动态量化可实现性收益。

## 闭环与选定方案

| 问题证据 | 理论依据 | 算法动作 | 目标指标 | 否证条件 |
|---|---|---|---|---|
| 父 Q Hessian 未建模 softmax，评测目标经过 softmax | `Hq=K^T(diag(p)-pp^T)K/d` 且常数 logit 方向为零空间 | 校准逐 query/head 计算稳定 softmax，以中心化后的 K 构造 Q block Hessian | Attention Output MSE 低于父版本；Linear MSE应不变 | Attention MSE不降，或候选产生非有限值/接口失败 |
| 父 K Hessian 对各 key 行使用同一无条件 Q 协方差 | `J_ii=p_i(1-p_i)` 给出 key 对每个 query 的局部敏感度 | 用 `p(1-p)` 加权 Q 外积，按校准 key 行聚合 K block Hessian | Attention Output MSE 与总分方向改善 | 正式 10 例不优于父版本即否证当前近似 |
| Q/K repair 已有合法 HiF4候选与回退 | PSD Fisher矩阵可继续用于阻尼 Cholesky 和二次型选择 | 保留父 scale/LV2/LV3 搜索，只替换 repair Hessian；异常时整组回退父统计路径 | 格式、CPU运行、数值有限性 | 任一结构/finite检查失败 |

## 修改范围与保持不变

- 修改 `_attention_cross_hessians`：从对侧无条件协方差改为 softmax-aware Q Fisher block 与 K token-diagonal Fisher block；继续经 `_cross_hessian_inverse` 阻尼求逆。
- 修改 Attention state 的 `proxy_role`/`rotation_reason`，明确实际代理；`_attention_hessian_repair` 的合法 HiF4编码和候选接受逻辑保持不变。
- Linear 两接口、V 动态量化、公开六接口签名、NVFP4解析、HiF4 shape/层级格式全部保持父版本。

## 超参数配置计划

只提供主配置 `solution.py`，不建立 trial。核心变化由精确 softmax Jacobian 定义决定；温度固定为评测公式的 `1/sqrt(head_dim)`。沿用父版本阻尼与候选接受阈值以隔离误差模型变化。矩阵不可逆、shape不符或非有限时回退普通合法 HiF4量化。

## 实施与验收

1. 校验 heads、head_dim、序列 Q/K 行数和有限性；按 GQA 映射每个 Q head 到 KV head。
2. 对每个校准样例计算变换后的 Q/K、scaled logits 和稳定 softmax。
3. Q：用概率加权中心化 K 的外积累加 `K^T J K/d`；K：用 `p(1-p)` 权重累加 Q 外积，最后按 key 行数归一化。
4. 截取每个 64 维 HiF4块的对角块，阻尼求逆并写入 Q/K state；失败则返回安全 fallback state。
5. 仅运行语法、六接口、自检/有限性检查，不运行正式评测，不生成 `report.md`。

预期仅为可证伪方向：Attention MSE可能下降，Linear MSE应与父版本一致。runner 的官方 10 例结果才可将各假设标记为结果支持、结果否证或证据不足。
