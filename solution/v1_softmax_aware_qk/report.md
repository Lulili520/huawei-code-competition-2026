# v1_softmax_aware_qk

## 正式评测结果

runner 对所选 `main` 配置完成固定 10 例正式评测（Linear 5 例、Attention 5 例）。下表数值均直接来自 runner 提供的配置结果；评测文件 SHA-256 为 `1a5d576b2a2c74e8826e747596adb4ae1ab1b8ec0d69f84a029ff494ee9bb2b8`。

| 配置 | Linear Output MSE | Attention Output MSE | Linear 得分 | Attention 得分 | 最终得分 | 是否选用 |
|---|---:|---:|---:|---:|---:|---|
| `main` | 0.0025332271610921716 | 0.0003291976436547344 | 3.817470491117524 | 2.191655555754403 | **6.009126046871927** | 是 |

仅有一个实际配置，没有未运行的候选，也不存在可报告的算法内部调参对照。

## 算法与超参数

### 实现基础

名义父版本为 `v0_hessian_repair`，实现基础为 `parent`。本版本保留父版本的六个公开接口、Linear 路径、V 路径、HiF4 编码与 Q/K repair 接受逻辑，只把 Attention 校准阶段的 Q/K Hessian 误差模型替换为 softmax-aware Fisher 代理。NVFP4 反量化仍为连续 16 个 E2M1 值逐项乘对应输入 E4M3 scale，再恢复原 shape；没有重估 scale 或改变分组与数值语义。

### 核心算法

对每个校准样例，本版本先在 Q/K 互逆平滑后的坐标中计算 `L=QK^T/sqrt(head_dim)` 和逐行 softmax 概率 `p`。

- Q 路径使用 `K^T(diag(p)-pp^T)K/head_dim` 的 64 维对角块作为局部 Fisher/Hessian 代理。实现通过概率加权的 K 二阶矩减去概率均值 K 的外积完成，因而理论上消除了整行 logit 平移方向。
- K 路径使用 `p(1-p)` 加权 Q 外积，并只保留 key-token 对角近似，使每个运行时 K 行仍可独立编码。
- 两类矩阵继续使用父版本的阻尼 Cholesky 求逆和自然顺序误差传播；候选只有在对应二次代理至少达到既有收益阈值时才替换父候选。结构、shape 或有限性检查失败时走安全回退。

这里的“更贴近 softmax 敏感方向”是理论解释，不是由 10 例评测直接测得的机理事实；评测只观测到最终输出 MSE。

### 超参数说明与最终取值

| 超参数/结构量 | 作用 | 实际测试取值 | 最终取值 | 选择依据 |
|---|---|---:|---:|---|
| softmax 温度 | 构造 scaled logits | `1/sqrt(head_dim)` | `1/sqrt(head_dim)` | 与评测 attention 公式及 scaled dot-product attention 定义一致，不作为自由调参项 |
| `_BLOCK` | Fisher/Hessian 对角块及 HiF4 修复宽度 | `64` | `64` | HiF4 64 值层级块约束；继承父版本 |
| `_HESSIAN_DAMPING` | 稳定 Cholesky 与矩阵求逆 | `0.003` | `0.003` | 继承父版本，以隔离 Hessian 统计定义的结构变化；无配置对照证明其最优 |
| `_HESSIAN_GAIN` | Hessian repair 候选的最小代理收益 | `0.0005` | `0.0005` | 继承父版本；无内部调参结果 |
| `_ATTENTION_ALPHA` | Q/K 互逆平滑尺度指数 | `0.75` | `0.75` | 继承父版本；无内部调参结果 |
| `_ATTENTION_GAIN` | Attention scale 候选的最小代理收益 | `0.05` | `0.05` | 继承父版本；无内部调参结果 |
| K Fisher 结构 | 是否保留不同 key 的交叉块 | token 对角近似 | token 对角近似 | 保持动态 K 行可独立量化；其近似损失尚未单独验证 |

最终选择 `main` 的依据是：它是唯一实际提交并由 runner 完成正式评测的配置，最终得分为 `6.009126046871927`。没有第二组配置，因此不能声称这些继承超参数经过本版本调优或已达到最优。

## 结果分析

### 与父版本对比：算法收益

父版本报告记录的同一固定 10 例结果为 Linear MSE `0.002533227`、Attention MSE `0.0003321124`、最终得分 `5.983274`。按这些已记录精度比较：

| 指标 | 父版本 | 本版本 `main` | 变化 |
|---|---:|---:|---:|
| Linear Output MSE | 0.002533227 | 0.0025332271610921716 | 在父报告精度内不变 |
| Attention Output MSE | 0.0003321124 | 0.0003291976436547344 | 降低 0.0000029147563452656，约 0.878% |
| 最终得分 | 5.983274 | 6.009126046871927 | 增加约 0.025852 |

这是一项结构级版本对比：本版本没有额外 trial，且 policy 指定保留 Linear、V 和既有阈值，因此观测到的 Attention 改善可记为 **softmax-aware Q/K 分支相对父版本的版本级算法收益**。但该单一综合对照不能进一步拆分 Q Fisher、K token 对角近似各自贡献，也不能排除两者交互；不能据此宣称其中任一组件单独必然有效。

### 算法内部调参收益

算法内部调参收益为“无可计算对照”。runner 的 `all_configs` 只有 `main`，没有其他实际超参数配置，因此不存在从候选配置切换到最终配置所带来的 MSE 或得分增益。继承的 `alpha`、damping、gain 等数值均不得解释为本版本调优收益。

### Policy 证据链逐条核对

| Policy 闭环/假设 | 实现与结果核对 | 结论 |
|---|---|---|
| 父 Q Hessian 未建模 softmax；以 `K^T(diag(p)-pp^T)K/d` 替换后应降低 Attention MSE，Linear 应不变 | `solution.py::_attention_cross_hessians` 实际计算 scaled logits、softmax、概率加权 K 二阶矩与均值外积差；正式结果中 Attention MSE 较父版本降低约 0.878%，Linear 在父报告精度内不变 | **结果支持**（支持本版本综合方向，不足以隔离 Q 项的独立贡献） |
| 父 K Hessian 对 key 行使用同一无条件 Q 协方差；以 `p(1-p)` 加权 Q 外积的 token 对角近似后应改善 Attention MSE 与总分 | 实现确有 `p(1-p)` 权重与按 GQA 聚合的 Q 外积；Attention MSE下降且总分提高 | **结果支持**（仅支持 Q/K 联合版本；K 近似的独立因果贡献证据不足） |
| PSD Fisher 可沿用阻尼 Cholesky、二次型候选选择及异常回退，并保持结构/有限性 | 实现继续调用 `_cross_hessian_inverse`，正式 10 例完整产出有限 MSE 和得分 | **结果支持**（只验证这 10 例可运行，不证明所有输入均不会回退或失败） |
| softmax Jacobian 的行平移零空间能使代理更贴近概率敏感方向 | 公式与实现一致，这是理论推导；runner 没有记录代理相关性、候选接受率或消融 | **证据不足**（理论解释，未被本次评测直接验证） |
| 校准序列的敏感方向能泛化到固定测试序列 | 固定 5 个 Attention 测试样例上端到端 MSE改善 | **结果支持**（范围仅限本次固定本地样例；对隐藏集、其他层或模型仍证据不足） |
| K token 对角近似丢失的交叉信息小于其可实现性收益 | 没有 full-token Fisher、关闭 K Fisher 或其他近似的对照配置 | **证据不足** |
| softmax-aware 代理必然降低 Attention output MSE | policy 已明确高阶项、`@V` 与 Q/K 同时量化交叉项未完整建模；本次仅一个数据点方向改善 | **证据不足**，不得从本地结果推导必然性或普遍性 |

本次没有出现“结果否证”的 policy 目标：目标 Attention MSE 与总分均按预期方向改善，Linear 也保持稳定。不过，多项机理和组件级因果假设仍因缺少消融而证据不足。

## Take Away

### 有效经验

- 在这组固定 5 个 Attention 样例上，将 Q/K repair 的无条件对侧二阶矩替换为 softmax-aware Fisher 代理，与 Attention MSE 从 `0.0003321124` 降至 `0.0003291976436547344` 同时出现。
- Linear 路径未改，Linear MSE 在父报告精度内保持不变，使本次总分提升可定位到 Attention 分支，而不是 Linear 分支。
- 保留既有 HiF4 候选、阻尼和接受阈值，有助于把版本差异限制在 Q/K Hessian 统计结构。

### 边界与失败经验

- 只有 `main` 一个配置，无法衡量内部调参收益，也无法证明 `alpha=0.75`、damping 或 gain 最优。
- Q Fisher 与 K token 对角 Fisher 同时改变，缺少逐组件消融，不能把全部改善分配给某一个动作。
- 没有 full-token K Fisher 对照，无法验证 token 对角近似的精度—可实现性权衡。
- 结果仅来自 runner 的固定 5 个 Linear 与 5 个 Attention 样例，不外推到隐藏集、其他层、其他模型或平台最终成绩。

### 可复用结论

后续若继续该方向，应优先建立 Q-only、K-only 或 K 近似结构消融，并保持 Linear、V 与继承阈值不变；只有这样才能区分组件贡献。当前可复用的最强结论仅是：`main` 在指定 10 例上以相同 Linear MSE 精度取得更低 Attention MSE 和更高总分。
