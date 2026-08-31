# NVFP4 与 HiF4 算法流程总结

## 1. 赛题背景与任务

大模型推理需要搬运和计算大量 Weight、Activation 和 KV Cache。将 BF16/FP16 压缩到 4 bit，可以降低存储和带宽开销，但也会产生量化误差。

NVFP4 和 HiF4 都是 4-bit 格式，但结构不同：

```text
NVFP4：每 16 个数一个 block，元素使用 E2M1
HiF4 ：每 64 个数一个 block，元素使用 S1P2 和三级 scale
```

因此本赛题需要完成：

```text
NVFP4 数据 → 反量化为 BF16 → 重新量化为 HiF4
```

需要处理两类算子：

- Linear：量化 Weight 和在线 Activation；
- Attention：分别量化在线 Q、K、V。

最终比较算子输出误差，而不是只比较单个 Tensor：

$$
MSE(X_{NVFP4}W_{NVFP4}^{T},X_{HiF4}W_{HiF4}^{T})
$$

以及：

$$
MSE(Attention(Q_{NVFP4},K_{NVFP4},V_{NVFP4}),
Attention(Q_{HiF4},K_{HiF4},V_{HiF4}))
$$

单个用例得分：

$$
Score=\frac{MSE_{STD}-MSE_{PLAYER}}{MSE_{STD}}
$$

选手误差低于标准算法时得正分，高于标准算法时得负分。

必须实现六个接口：

```python
# Linear
hif4_calibration_and_quantize_weight(...)
hif4_dynamic_quantize_activation(...)

# Attention
hif4_calibration_attention(...)
hif4_dynamic_quantize_q(...)
hif4_dynamic_quantize_k(...)
hif4_dynamic_quantize_v(...)
```

重要限制：

- HiF4 输出必须严格合法；
- calibration state 只能包含规定的基础类型和 CPU Tensor；
- 单次提交总运行时间不超过五分钟；
- 禁止计算 `A @ W` 后反向拟合 `Q(A)`；
- `self_check.py` 只检查接口和格式，不检查精度。

---

## 2. 必要基础

### 2.1 量化、scale 和 block

4 bit 只有 16 种编码，无法准确表示所有实数。量化就是把原数映射到附近的合法低精度值，两者之差就是量化误差。

`scale` 是恢复真实数量级的倍率：

```text
量化值 = 1.5
scale   = 100
恢复值   = 1.5 × 100 = 150
```

一组共享 scale 的数字叫 block。block 越小，scale 越贴合局部数据，但额外开销越大。

### 2.2 ExMy 是什么

- `E`：Exponent，指数，控制数值范围；
- `M`：Mantissa，尾数，控制精度；
- 数字表示相应字段占用的 bit 数。

```text
E2M1：2 bit 指数 + 1 bit 尾数
E4M3：4 bit 指数 + 3 bit 尾数
E6M2：6 bit 指数 + 2 bit 尾数
```

浮点数可理解为二进制科学计数法：

$$
数值=符号\times有效数字\times2^{指数}
$$

E 越多，范围越大；M 越多，相邻可表示值越密。

### 2.3 E2M1 的值集

正规浮点有效数字写成 `1.M`，开头的 `1` 是隐藏位。M1 只有一个 bit：

```text
M = 0：1.0₂ = 1
M = 1：1.1₂ = 1.5
```

乘以不同的 2 的幂后得到 `1、1.5、2、3、4、6`。加上 0 和次正规数 0.5，非负值集合为：

$$
\{0,0.5,1,1.5,2,3,4,6\}
$$

---

## 3. NVFP4 流程

标准 NVFP4 的反量化公式为：

$$
\hat{x}_i=q_i^{E2M1}\cdot s_b^{E4M3}\cdot s_{global}^{FP32}
$$

其中：

- `q`：4-bit E2M1 元素；
- `s_block`：每 16 个元素共享一个 E4M3 scale；
- `s_global`：整个 Tensor 共享一个 FP32 scale。

标准量化步骤：

1. 计算整个 Tensor 的最大绝对值 $a_{global}$。
2. 计算全局 scale：

   $$s_{global}=\frac{a_{global}}{448\times6}=\frac{a_{global}}{2688}$$

3. 每 16 个元素计算局部最大值 $a_b$，生成 E4M3 block scale：

   $$s_b=Q_{E4M3}\left(\frac{a_b/6}{s_{global}}\right)$$

4. 归一化并舍入到 E2M1：

   $$q_i=Q_{E2M1}\left(\frac{x_i}{s_{global}s_b}\right)$$

5. 反量化：

   $$\hat{x}_i=q_i s_b s_{global}$$

本赛题已经把尺度关系整理到 `scale_float` 中，直接提供：

```python
(quant_float, scale_float)
```

反量化只需：

```python
x = quant_float.unflatten(-1, (-1, 16))
x = x * scale_float.unsqueeze(-1)
x = x.flatten(-2, -1).to(torch.bfloat16)
```

因此不需要解析原始 4-bit bitstream，也不需要单独恢复 global scale。

---

## 4. HiF4 流程

### 4.1 数据结构

HiF4 每 64 个元素组成一个 block：

| 参数 | 数量 | 作用范围 |
|---|---:|---:|
| E6M2 `scale_factor` | 1 | 64 个元素 |
| `scale_lv2` | 8 | 每个控制 8 个元素 |
| `scale_lv3` | 16 | 每个控制 4 个元素 |
| S1P2 `sign/mant` | 64 | 每个元素一个 |

反量化公式：

$$
\hat{x}_i=sign_i\cdot mant_i\cdot scale\_lv3_i
\cdot scale\_lv2_i\cdot scale\_factor
$$

合法值要求：

```text
scale_factor：E6M2
scale_lv2   ：1 或 2
scale_lv3   ：1 或 2
sign        ：-1、0 或 1
mant        ：0～1.75，步长 0.25
```

S1P2 非负值集：

$$
\{0,0.25,0.5,0.75,1,1.25,1.5,1.75\}
$$

E6M2 的合法形式：

$$
scale=2^E\times\{1,1.25,1.5,1.75\}
$$

两层微尺度都能取 2，因此单个元素相对基础 scale 的最大倍数为：

$$
1.75\times2\times2=7
$$

### 4.2 官方转换步骤

第一步，对 64 个元素进行三级最大值归约：

```text
64 个元素 → 16 个四元素 max → 8 个八元素 max → 1 个全局 max
```

第二步，生成基础 scale：

$$
scale\_factor=Q_{E6M2}\left(\frac{max_{64}}{7}\right)
$$

每个 8 元素组根据局部范围选择 `scale_lv2=1` 或 `2`；每个 4 元素组再选择 `scale_lv3=1` 或 `2`。

第三步，对每个元素计算有效尺度：

$$
s_i=scale\_factor\cdot scale\_lv2\cdot scale\_lv3
$$

然后量化 mant：

$$
mant_i=clip\left(round\left(4\frac{|x_i|}{s_i}\right)/4,0,1.75\right)
$$

再保存符号，即得到完整 HiF4 参数。

---

## 5. 两种格式对比

| 特征 | NVFP4 | HiF4 |
|---|---|---|
| 元素格式 | E2M1 | S1P2 |
| 非负值集 | 0、0.5、1、1.5、2、3、4、6 | 0～1.75，步长 0.25 |
| block 大小 | 16 | 64 |
| scale | E4M3 block + FP32 global | E6M2 + 两层微尺度 |
| 特点 | block 小、元素范围大 | 尾数更细、适合 64 长度点积 |

白话理解：

- NVFP4：每 16 个数放进一个小盒子，贴一个较精确的倍率标签；
- HiF4：每 64 个数放进一个大盒子，先用总倍率，再让每 8 个和每 4 个数选择是否额外乘 2。

---

## 6. 比赛中的实际转换

```text
quant_float + scale_float
           ↓
反量化得到 BF16 参考值
           ↓
每 64 个元素分组
           ↓
生成或搜索 E6M2、lv2、lv3
           ↓
生成 sign、mant
           ↓
返回合法 HiF4Params
```

若输入形状为 `(*prefix, C)`，输出形状必须为：

```text
scale_factor : (*prefix, C // 64, 1, 1, 1)
scale_lv2    : (*prefix, C // 64, 8, 1, 1)
scale_lv3    : (*prefix, C // 64, 8, 2, 1)
sign         : (*prefix, C // 64, 8, 2, 4)
mant         : (*prefix, C // 64, 8, 2, 4)
```

实现时适合 reshape 为：

```python
x64 = x.unflatten(-1, (-1, 8, 2, 4))
```

---

## 7. 优化方向

官方 HiF4 转换主要由最大值驱动，速度快，但不保证 MSE 最低。可改为搜索：

$$
\min_{s,e_2,e_3,m}\sum_i\omega_i(x_i-\hat{x}_i)^2
$$

通用优化：

- 搜索 `max/7` 附近的多个 E6M2 候选；
- 对 lv2/lv3 组合进行局部精确搜索；
- 搜索 clipping ratio，降低异常值影响；
- 使用广播和 reshape，避免逐元素 Python 循环。

Linear 可根据 calibration activation 的通道能量加权 Weight 误差，并利用：

$$
XW^T=(X/S)(WS)^T
$$

在 Weight 与 Activation 之间重新分配异常值压力。

Q/K 可使用对偶缩放：

$$
Q'=Q/S,\qquad K'=KS,\qquad Q'K'^T=QK^T
$$

V 不宜采用无法在输出端抵消的旋转或缩放，应优先使用局部搜索和 clipping。

---

## 8. 推荐实现顺序

1. 实现严格合法的 E6M2、lv2、lv3、sign 和 mant，通过 self-check。
2. 实现 E6M2 邻域搜索和 lv2/lv3 局部精确搜索。
3. 加入 Linear 的 Activation-RMS 加权和可逆平滑。
4. 加入 Q/K 对偶缩放及交叉能量加权。
5. 控制 dynamic 阶段的搜索规模，确保总耗时不超过五分钟。

## 9. 参考资料

1. [NVIDIA Transformer Engine：NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
2. [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)
3. [HiFloat4 官方格式说明](https://hifloat.gccorg.com/docs/en/hifloat4/white_paper/hifloat4_format_for_language_model_inference.html)
4. [HiFloat4 Format for Language Model Inference](https://arxiv.org/abs/2602.11287)
