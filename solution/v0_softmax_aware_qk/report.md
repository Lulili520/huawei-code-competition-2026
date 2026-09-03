# v0_softmax_aware_qk

## 本地结果

统一评测集包含 50 个 Linear 和 250 个 Attention case。单例按赛题公式计算 MSE 提升百分比，最终分为 300 个 case 的百分比总和。

| 指标 | 结果 |
|---|---:|
| Linear Output MSE | 5.876366e-3 |
| Attention Output MSE | 3.740420e-4 |
| 最终得分 | **16498.457890** |
| 本地评测耗时 | 626.466 s |
| 格式检查 | 22/22 通过 |

## 核心设计

### 问题与修改方向

普通 Q/K Hessian 主要衡量 Q、K 自身的数值误差，却没有区分哪些误差会真正改变 softmax。由于 softmax 对整行 logits 的常数平移不敏感，相同大小的 Q/K 误差对最终 Attention 输出的破坏可能完全不同。

本版本保留 `v0_hessian_repair` 的 Linear、V、HiF4 编码和候选接受逻辑，只把 Q/K 的误差度量替换为 softmax-aware Fisher/Hessian 代理：先根据校准 Q/K 得到 attention 概率，再提高对 softmax 敏感方向的保护程度。

### 主要超参数

| 参数 | 取值 | 含义 |
|---|---:|---|
| `_BLOCK` | `64` | HiF4 一级块以及局部 Hessian 的处理宽度 |
| `_HESSIAN_DAMPING` | `0.003` | 稳定 Hessian 分解，避免病态矩阵导致数值异常 |
| `_HESSIAN_GAIN` | `0.0005` | 接受 Hessian 修复候选所需的最低代理收益 |
| `_ATTENTION_ALPHA` | `0.75` | Q/K 互逆平滑时的尺度分配指数 |
| `_ATTENTION_GAIN` | `0.05` | 接受 Attention scale 候选所需的最低代理收益 |

这些数值继承自原实现，本轮没有内部参数对照，因此只能把收益归因到完整算法版本，不能证明单个参数最优。

## 结果分析

相对 `v0_hessian_repair`：

- Linear MSE 完全相同，说明 Linear 路径没有受到修改影响。
- Attention MSE 从 `3.786237e-4` 降至 `3.740420e-4`，降低约 **1.210%**。
- 最终得分增加 `114.946483`，相对提高约 **0.702%**。

结果支持“softmax-aware Q/K 误差度量在当前 300 例上有效”，但不能分离 Q Fisher、K token 对角近似各自的独立贡献，也不能直接外推平台隐藏集。

## Take Away

- Attention 优化应衡量误差对 softmax 输出的影响，而不只是 Q/K 张量重构误差。
- 该版本是当前三个保留版本中的本地综合最优版本。
- 本地评测耗时超过五分钟，尚不能据此证明满足平台时限；后续需要单独测量六个量化接口的执行时间。

`solution.py` SHA-256：`F4F91EDB1559139FDCC54911A3FDA4CAE570681D7296751149E259A183C3938C`。
