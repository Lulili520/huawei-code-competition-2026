# v0_alternating_joint_fit 策略

## 版本关系与固定边界

- 当前统一对比基准：`v0_hessian_repair`。该实现历史上源自现已清理的 `v3_output_aware_linear`，复用了 NVFP4 解析、reciprocal preconditioning、HiF4 合法编码、Hessian 修复和 Attention 全路径，仅重写 Linear 输出感知拟合与激活编码的耦合流程。
- `solution.py::dequantize_nvfp4` 保持固定语义：E2M1 值乘输入 E4M3 scale，每 16 个连续值共享一个 scale，最后恢复原 shape。六个公开接口不变。
- 本 Agent 不运行正式评测、不生成 `report.md`。

## 已验证事实

1. 当前 `solution.py::_alternating_joint_target` 明确执行激活编码、等效权重拟合、importance 更新和再次拟合，而不是固定激活编码后只求解一次。
2. 当前 `solution.py::hif4_dynamic_quantize_activation` 会读取校准阶段最后一轮产生的 `activation_importance`，使在线激活使用与末轮权重相匹配的规则。
3. `v0_hessian_repair` 的统一 300 例 Linear MSE 为 `0.005876365662724178`；本策略必须在相同数据上低于该值，才能支持交替拟合方向。

## 外部理论与理论推导

- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023, https://openreview.net/forum?id=tcbBPnfwxS ：校准输入二阶统计可近似层输出重构误差，支持以输出误差而非逐元素误差选择量化表示；它不证明本交替过程必然提升 HiF4。
- Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models*, ICML 2023, https://proceedings.mlr.press/v202/xiao23c.html ：支持权重/激活之间的等价 reciprocal scaling；本版本保留该变换，不从论文外推离散 HiF4 scale 的最优性。
- Bezdek and Hathaway, *Some Notes on Alternating Optimization*, AFSS 2002, https://doi.org/10.1007/3-540-45675-9_38 ：交替优化通过逐块更新变量处理耦合目标；其收敛结论依赖精确子问题和连续性，本实现的离散候选、阻尼与后续权重量化不满足全部条件，只提供机制依据。

令预条件后原激活为 `X`，HiF4 反量化激活为 `A(s)`，等效权重为 `U`，目标为 `||XW^T-A(s)U^T||_F^2`。固定 `s` 时，基准版本以分块阻尼最小二乘更新 `U`。固定 `U` 时，输出误差的对角近似给输入通道 `j` 的权重 `||U[:,j]||_2^2`；以该权重进入现有合法 scale/LV2/LV3 搜索，相当于让激活编码优先保护对当前输出更敏感的通道。随后用新 `A(s)` 再拟合 `U`。

## 修改闭环

| 问题证据 | 理论依据 | 算法动作 | 目标指标 | 否证条件 |
|---|---|---|---|---|
| 固定激活编码后只拟合一次权重，激活与权重可能失配 | 耦合目标的坐标交替；`U^TU` 对角给出通道输出敏感度 | 初始编码/权重拟合后，用拟合权重列能量归一化为新 importance，重新选择激活 scale/LV2/LV3，再拟合权重 | Linear MSE 低于当前基线，Attention MSE 保持 | 统一 300 例 Linear MSE 不低于 `0.005876365662724178` 或总分不高于 `16383.511407092246` |
| 连续最小二乘收益可能被不匹配的推理激活编码抵消 | 校准与推理必须共享同一离散编码规则 | 将最终轮 importance 写入 activation state，动态接口复用它 | 校准代理与推理路径一致、结果有限 | 六接口失败、非有限值、或推理使用的编码状态与末轮不一致 |
| 离散候选与病态 Gram 不保证交替单调 | 阻尼最小二乘只在固定编码下稳定 | 固定两轮；任一形状、有限性或 Cholesky 检查失败即回退到父式目标/状态 | CPU 时间有限且不破坏格式 | 超时、异常或正式样例无法完成 |

## 核心算法与实施顺序

1. 按基准版本计算 reciprocal precondition 与初始 activation importance。
2. 用完整动态 HiF4 编码路径生成初始 `A0`，分 64 通道块求解输出感知权重 `U0`。
3. 由 `U0` 的列平方和构造输出敏感 importance，按均值归一化并裁剪到父编码器接受的稳定范围。
4. 用该 importance 重新执行每个 64 块的合法 scale/LV2/LV3 离散选择，得到 `A1`，再求解 `U1`。
5. 对 `U1` 执行基准版本权重 scale 搜索和 Hessian/cross-block 修复；把末轮 importance 写入动态激活 state。Attention 不改。
6. 任一步失败时使用上一有效轮；若首轮失败则退回原 transformed weight 与初始 importance。

## 超参数配置

仅提供主配置 `solution.py`：两次权重拟合（初始轮加一次输出敏感重编码轮），阻尼 `0.01`、最多 `2048` 校准行沿用基准版本。迭代数是本结构流程的有界预算，不拆成新版本；当前没有证据支持额外配置，故不创建 trials。

## 待验证假设与验收

- 待验证假设：末轮激活编码与权重拟合相互匹配，可以降低统一 300 例的 Linear MSE；理论本身不保证平台隐藏集分数提升。
- 待验证假设：Attention 路径算法不变，其官方 MSE 应在数值精度内保持。
- 本地先检查导入、六接口契约、shape、字段和有限性；最终由 Runner 的串行统一 300 例结果判为结果支持、结果否证或证据不足。
