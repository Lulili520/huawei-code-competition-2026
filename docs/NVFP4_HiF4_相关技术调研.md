# NVFP4 → HiF4 量化相关技术调研

> 调研日期：2026-08-28  
> 目标：为华为算法大赛 NVFP4 → HiF4 转换任务及当前 `../src/baseline.py` 的后续优化提供理论依据和实现方向。

## 1. 当前问题

比赛输入为 NVFP4 的数值载体和 block scale。参赛算法先按固定规则恢复其 BF16 参考值，再输出合法的 HiF4 参数。

需要优化两类算子：

```text
Linear：Weight + Activation → 矩阵乘输出
Attention：Q + K + V → Attention 输出
```

评分关注最终算子输出 MSE，而不是单个 Tensor 的重构 MSE：

$$
Score=\frac{MSE_{STD}-MSE_{PLAYER}}{MSE_{STD}}
$$

这意味着一个元素即使没有量化到离自身最近的值，只要能与其他误差相互抵消并降低最终算子误差，也可能是更好的选择。

当前 `baseline.py` 已经采用两层方法：

1. 在 HiF4 合法空间内搜索 E6M2、lv2、lv3 和 mant；
2. 用 calibration 数据的低秩二阶统计，联合调整 mant 的向上/向下舍入。

---

## 2. HiF4：目标格式与直接转换基线

### 2.1 格式结构

HiF4 每 64 个元素组成一个 block：

| 参数 | 数量 | 作用范围 |
|---|---:|---:|
| E6M2 `scale_factor` | 1 | 64 个元素 |
| E1 `scale_lv2` | 8 | 每个控制 8 个元素 |
| E1 `scale_lv3` | 16 | 每个控制 4 个元素 |
| S1P2 `sign/mant` | 64 | 每元素一个 |

反量化公式为：

$$
\hat{x}_i=sign_i\cdot mant_i\cdot scale\_lv3_i
\cdot scale\_lv2_i\cdot scale\_factor
$$

S1P2 的非负值集是：

$$
\{0,0.25,0.5,0.75,1,1.25,1.5,1.75\}
$$

两层微指数均取 1 或 2，因此单个元素相对基础 scale 的最大倍率为：

$$
1.75\times2\times2=7
$$

### 2.2 官方转换流程

HiF4 论文给出的直接转换包含三步：

1. 对每 4、8、64 个元素依次求局部最大绝对值；
2. 由 `max64/7` 生成 E6M2，再根据局部最大值生成 lv2/lv3；
3. 除以有效 scale，舍入并截断到 S1P2。

该流程适合硬件实现，但它主要由最大值驱动，不保证给定数据上的 MSE 最小。

### 2.3 对当前 baseline 的启示

当前 `_encode()` 在官方方案上增加了：

- `max/7` 附近的多个 E6M2 候选；
- 遵守树状共享约束的 lv2/lv3 局部搜索；
- 给定离散参数后的最小二乘 scale 回归。

这是一条合理路线。下一步可补充：

- 更对称的 E6M2 邻域；
- 基于 P99/P99.5 的 clipping 候选；
- “离散参数优化 ↔ scale 回归”的 1～2 轮交替迭代。

资料：[HiFloat4 Format for Language Model Inference](https://arxiv.org/abs/2602.11287)、[HiF4 官方格式说明](https://hifloat.gccorg.com/docs/en/hifloat4/white_paper/hifloat4_format_for_language_model_inference.html)

---

## 3. NVFP4：输入格式及其 outlier 处理

标准 NVFP4 使用 E2M1 元素、每 16 个元素一个 E4M3 block scale，以及整个 Tensor 的 FP32 global scale：

$$
\hat{x}_i=q_i^{E2M1}s_b^{E4M3}s_{global}^{FP32}
$$

E2M1 的非负值集为：

$$
\{0,0.5,1,1.5,2,3,4,6\}
$$

NVIDIA 的 NVFP4 训练 recipe 还包括：

- 权重使用二维 block scaling；
- 对部分训练矩阵乘输入使用 Random Hadamard Transform；
- 梯度采用随机舍入；
- 少量敏感层保留高精度。

这些方法说明：在 4-bit 下，异常值分布和量化轴非常关键。但比赛输入已经是展开后的 NVFP4 数据，因此不需要重新实现标准 NVFP4 编码，只需使用固定反量化函数。

资料：[NVIDIA Transformer Engine NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)、[Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)

---

## 4. GPTQ：二阶信息与误差补偿

### 4.1 为什么普通 Tensor MSE 不够

Linear 层中设权重量化误差为：

$$
D=W-\hat W
$$

校准输入为 $A$，则输出误差为：

$$
\|AD^T\|_F^2
=\sum_r d_r^T(A^TA)d_r
$$

因此 $H=A^TA$ 决定哪些通道误差重要，以及不同通道的误差能否相互抵消。

GPTQ 使用近似二阶信息逐步量化权重，并将当前量化误差补偿到尚未量化的权重。论文表明，这种方法在 3/4-bit 权重量化上明显优于简单 round-to-nearest。

### 4.2 与当前 baseline 的关系

当前 baseline 没有保存完整 $C\times C$ Hessian，而是对 calibration 矩阵做截断 SVD：

$$
A\approx U_k\Sigma_kV_k^T
$$

于是：

$$
A^TA\approx V_k\Sigma_k^2V_k^T
$$

再用 `_schwarz()` 每次联合枚举一组元素的 floor/ceil 组合。这与 GPTQ 共享同一个核心思想：

> 根据最终算子敏感度分配量化误差，而不是让每个元素独立最近邻舍入。

### 4.3 建议检查

如果使用“低秩非对角项 + 精确对角项”，近似矩阵应考虑：

$$
H_{approx}=V_w^TV_w+
diag(an_{full}-an_{svd})
$$

当前 baseline 的 Schwarz 代价使用 `VwᵀVw + diag(an_full)`，可能重复计算低秩部分的对角线，应当做消融验证。

资料：[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)

---

## 5. SmoothQuant 与 AWQ：利用 Activation 迁移量化难度

### 5.1 SmoothQuant

SmoothQuant 观察到 Activation outlier 往往持续出现在固定通道，并使用数学等价变换：

$$
XW^T=(X/S)(WS)^T
$$

将一部分 Activation 的量化难度迁移到 Weight。论文指出迁移强度需要平衡；在其 INT8 实验中，$\alpha$ 约 0.4～0.6 是常见有效区间，但 HiF4 需要重新搜索，不能直接照搬。

### 5.2 AWQ

AWQ 同样使用 Activation 统计识别重要权重通道，并通过等价缩放保护这些通道。它强调：权重的重要性不能只看权重自身，应参考输入 Activation 分布。

### 5.3 对本任务的启示

当前 baseline 只用 Activation 统计优化 Weight 的 mant 舍入，尚未实现正式的动态 Activation 量化。可以增加：

1. calibration 阶段搜索 per-channel 平滑尺度 $S$；
2. 将 $WS$ 量化为 HiF4 Weight；
3. 把 $S$ 放入 `activation_state`；
4. 在线阶段量化 $X/S$。

这种变换在量化前严格保持 Linear 结果不变，并能缓解 64 元素 HiF4 block 内的通道异常值。

资料：[SmoothQuant](https://arxiv.org/abs/2211.10438)、[AWQ](https://arxiv.org/abs/2306.00978)

---

## 6. QuaRot 与 SpinQuant：利用旋转分散异常值

### 6.1 基本原理

若 $R$ 为正交矩阵：

$$
RR^T=I
$$

则矩阵乘可以改写为：

$$
AB=(AR)(R^TB)
$$

旋转不会改变高精度计算结果，却能把集中在少数通道的异常值分散到更多通道，使低精度量化更容易。

QuaRot 使用随机 Hadamard 旋转；SpinQuant 则学习更适合量化的旋转，并报告学习旋转可优于随机旋转。

### 6.2 对本任务的适用性

Linear 可对 Weight/Activation 使用成对旋转；Q/K 也可在每个 head 内使用相同正交变换：

$$
(QR)(KR)^T=QK^T
$$

但需要注意：

- dynamic 阶段必须承担旋转计算成本；
- GQA 中 Q head 与共享 KV head 的旋转必须一致；
- V 若单独旋转，Attention 输出也会旋转，除非接口允许在输出端抵消，否则不能直接使用；
- 比赛总时限五分钟，完整旋转可能不如轻量平滑划算。

资料：[QuaRot](https://arxiv.org/abs/2404.00456)、[SpinQuant](https://arxiv.org/abs/2405.16406)

---

## 7. Attention 与 KV：Q、K、V 不应共用同一策略

### 7.1 Q/K 的误差目标

Attention logits 为：

$$
L=QK^T/\sqrt{d}
$$

一阶误差近似：

$$
\Delta L\approx\Delta QK^T+Q\Delta K^T
$$

因此：

- 量化 Q 时，K 的协方差决定 Q 误差的敏感方向；
- 量化 K 时，Q 的协方差决定 K 误差的敏感方向。

当前 baseline 使用 K 校准 Q、使用 Q 校准 K，方向正确。

### 7.2 GQA 中的一个问题

一个 KV head 对应多个 Q head 时，K 的合理敏感度近似应为：

$$
H_K=\sum_h Q_h^TQ_h
$$

当前 baseline 先对多个 Q head 求均值，再构造协方差，可能因正负抵消而低估重要方向。建议把对应 Q heads 沿样本维堆叠，而不是求平均。

### 7.3 KIVI 的启示

KIVI 对 KV Cache 分布的研究发现：

- K 中常有持续存在的固定通道 outlier，适合 per-channel 处理；
- V 没有相同的通道模式，per-token 量化更合适；
- K 的量化误差影响 Attention score；
- V 的量化误差应根据 $PV$ 的最终输出评估，而不只是 V 的重构误差。

HiF4 的 block 轴已经由接口固定，无法直接复制 KIVI 的量化布局，但可以分别设计 K 与 V 的 calibration 目标。

当前 `v_state = SVD(V)` 缺乏直接的算子误差推导。更合理的候选包括：

- V 使用普通局部 MSE，避免错误相关性；
- 用 calibration Q/K 计算 Attention 权重统计，再估计 V token 的重要性；
- 通过消融确认 `SVD(V)` 是否确有收益。

资料：[KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)

---

## 8. 对当前 baseline 的综合评价

### 值得保留

1. HiF4 格式内的 E6M2 和 lv2/lv3 搜索；
2. 用 SVD 压缩 calibration 二阶信息；
3. 用全局误差状态决定 mant floor/ceil；
4. Q/K 使用交叉 calibration；
5. Schwarz 小组精确枚举，降低逐元素贪心的顺序依赖。

### 需要优先修正

1. 改成比赛规定的六个公开接口；
2. Linear calibration 返回 `weight_params` 和 `activation_state`；
3. 实现动态 Activation 量化；
4. 检查 Schwarz 对角线是否重复计权；
5. GQA 的 Q 统计由均值改为样本堆叠；
6. 重新设计或消融 V calibration；
7. 在鲲鹏 CPU 上验证 SVD/Schwarz 的五分钟时限。

---

## 9. 推荐实验矩阵

不要一次叠加所有方法，建议逐项消融：

| 实验 | E6M2 搜索 | 二阶舍入 | 平滑 | Q/K 交叉统计 | V 策略 |
|---|---|---|---|---|---|
| A0 | 官方 max/7 | 无 | 无 | 无 | RTN |
| A1 | 邻域搜索 | 无 | 无 | 无 | RTN |
| A2 | 邻域搜索 | SVD+GPTQ | 无 | 无 | RTN |
| A3 | 邻域搜索 | SVD+Schwarz | 无 | 无 | RTN |
| A4 | 邻域搜索 | SVD+Schwarz | Smooth | 无 | RTN |
| A5 | 邻域搜索 | SVD+Schwarz | Smooth | 修正后 Q/K | RTN |
| A6 | 邻域搜索 | SVD+Schwarz | Smooth | 修正后 Q/K | Attention-aware V |

每项记录：

```text
格式合法性
Tensor 重构 MSE
Linear 输出 MSE
Attention logits MSE
Attention 最终输出 MSE
calibration 时间
dynamic 时间
state 大小
```

---

## 10. 推荐研发路线

### 第一阶段：可提交版本

- 补齐六个接口；
- 使用 `_encode()` 作为所有 Tensor 的统一基础量化器；
- 确保 self-check 全部通过；
- 建立真实 mini sample 的耗时和 MSE 基线。

### 第二阶段：修正二阶优化

- 验证 Schwarz 对角残差；
- 对比 RTN、逐元素 GPTQ、Schwarz；
- 测试 rank 8/16/32 对精度、state 和时间的影响。

### 第三阶段：算子专用优化

- Linear 加入 SmoothQuant/AWQ 式可逆缩放；
- Q/K 修正 GQA 统计并尝试对偶缩放；
- V 比较 RTN、SVD(V) 与 Attention-aware 统计。

### 第四阶段：性能优化

- calibration 阶段可使用较复杂搜索；
- dynamic 阶段减少候选和 Schwarz 轮数；
- 全部使用 reshape、广播和矩阵乘；
- 避免完整 calibration 矩阵常驻内存；
- 在目标 CPU 上进行端到端五分钟压力测试。

---

## 11. 参考资料索引

1. [HiFloat4 Format for Language Model Inference](https://arxiv.org/abs/2602.11287)
2. [HiFloat4 官方格式说明](https://hifloat.gccorg.com/docs/en/hifloat4/white_paper/hifloat4_format_for_language_model_inference.html)
3. [NVIDIA Transformer Engine：NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
4. [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)
5. [GPTQ](https://arxiv.org/abs/2210.17323)
6. [SmoothQuant](https://arxiv.org/abs/2211.10438)
7. [AWQ](https://arxiv.org/abs/2306.00978)
8. [QuaRot](https://arxiv.org/abs/2404.00456)
9. [SpinQuant](https://arxiv.org/abs/2405.16406)
10. [KIVI](https://arxiv.org/abs/2402.02750)

## 12. 结论

当前 baseline 的“SVD 二阶统计 + Schwarz 分组舍入”不是孤立思路，它与 GPTQ 的算子感知误差补偿高度一致；HiF4 格式内搜索提供了合法且较强的底座。下一步最值得投入的不是继续堆叠复杂优化，而是先完成接口闭环、修正二阶代价和 GQA 统计，再通过消融确认每个模块对真实 Linear/Attention MSE 的贡献。

若这些基础修正有效，再引入 SmoothQuant 式平滑和轻量 Q/K 对偶缩放；旋转方法应放在后期，并以运行时间收益比作为是否保留的依据。
