# HiF4 Local-Agent 迭代规范

## 任务模型

`.agent/runner.py` 使用固定 6 个本地 Codex Agent 槽位；候选充足时持续占满，空槽不等待同批任务结束。全局组合目标固定为 4 个 `explore`（探索新根因或跨算法族机制）和 2 个 `exploit`（沿已有正式正证据或 Pareto 分项优势深化）。正在运行的任务也计入 4:2 配额；某一角色暂时没有合法候选时，另一角色可以借用空槽，但 Runner 不得伪造任务或用纯调参凑数。

每个 Agent 独立完成一个完整算法版本：问题分析与调研、`policy.md`、算法实现、最多三组内部超参数配置、结构审查、60 例内部配置筛选、300 例正式评测、`report.md` 和扁平版本账本登记。

不存在单独的“发现 Agent”和“实现 Agent”。一个版本完成后立即按真实得分更新全局优先队列，无须等待同批任务结束。

## 固定边界

NVFP4 反量化固定为 E2M1 值乘对应 E4M3 scale，每 16 个连续值共享一个 scale，最后恢复原 shape。不得修改数值语义、scale 对应关系、16 值分组，或用重新估计的 scale 替换输入 scale。

Runner 启动任务时记录评测器、评分公式、配置、schema 与数据清单的 SHA-256。实现 Agent 和只读 Reviewer 都在轻量独立快照中运行；任何越过目标 `solution/<version>/` 的写入都会使任务失败。正式评测前后再次核对受保护文件和候选 `solution.py`，评测收益不能来自评测口径或数据被修改。

## 扁平版本

- 当前筛选保留的起始算法统一使用 `v0_<方法概括>`，表示并列的初始方法库。后续 Agent 新实验继续使用 `vN_<方法概括>`，其中 `N` 从 1 起全局连续分配。
- 不存在父子树、代数和每个节点三个子版本的限制。
- `v0_hessian_repair` 是固定比较基线；现存的其他 `v0_*` 是历史并列初始方法。新版本号由 Runner 在全局锁内分配为当前最大编号加一。
- 每个版本可记录 `based_on`，表示比较基准或代码来源；它只是溯源信息，不形成父子关系，也不限制后续从其他版本继续研究。
- 实现来源为 `based_on`、`v0_hessian_repair` 或 `scratch`。`policy.md` 必须解释为什么选择该来源。
- 方法概括写入目录后缀；完整算法族、问题方向和来源同时写入扁平账本 `.agent/versions.json`。

## 算法版本

- 正式版本必须对应可证伪的结构级算法策略；纯 alpha、gain、阈值、系数、候选数或倍率修改不得建立版本。
- 算法确定后，同一版本内部允许最多 2～3 组有理论依据的超参数配置。主配置放在 `solution.py`，其余放在 `trials/<config>/solution.py`。
- 新策略可以基于任意已评测版本，也可以从零实现；选择依据必须是问题证据和算法适配性，而不是目录层级。

## 证据标准

- 明确区分 `已验证事实`、`理论推导` 和 `待验证假设`，不得把推测写成事实。
- 项目事实定位到文件与函数/行号，或真实评测配置与精确指标。
- 外部依据优先使用原始论文、标准和官方资料，记录标题、作者/机构、年份与直达链接。
- 每项算法动作必须有完整闭环：`问题证据 → 理论依据 → 算法动作 → 目标指标 → 否证条件`。
- 报告逐项把假设标记为 `结果支持`、`结果否证` 或 `证据不足`，不得从本地样例直接外推隐藏集。
- 根因分析必须落到可观测的误差传播项，而不是只写“量化误差较大”：Linear 至少区分 `ΔXWᵀ`、`XΔWᵀ` 与 `ΔXΔWᵀ`；Attention 至少区分中心化 logits、Softmax Jacobian 敏感方向与 V 路径；格式方向至少区分 clipping 误差与离散分辨率误差。若现有数据不能分解某一项，应明确标记为待验证假设。

## 评分与优化重点

- 正式本地评测必须实际运行合并数据集的 50 个 Linear 和 250 个 Attention，共 300 个样例。
- 单例得分为 `(MSE_STD-MSE_PLAYER)/MSE_STD×100%`；综合分先分别求两类平均分，再按 Linear : Attention = 1 : 5 加权并乘 300。这与直接累加 300 个单例百分比得分等价，不是小样本外推。
- 因 Attention 权重为 Linear 的五倍，全局排序以综合分为主；同时保留 Linear MSE 和 Attention MSE 用于定位问题。
- 任何候选选择算法都应记录候选接受率、回退次数和逐例指标。只有汇总 MSE、没有中间诊断的版本不得直接据此扩展同类复杂策略。
- 允许缓存与候选无关的浮点参考输出和标准编码误差，缓存键必须绑定数据清单、PyTorch 版本与评测算法版本；候选量化、全部 300 例及计分仍须逐版本真实执行。缓存启用前必须以同一筛选集验证分数和两类 MSE 完全一致。
- 评测器统一记录每类逐例得分的最小值、P10、中位数、P90、最大值、负收益 case 数与阶段耗时。含内部搜索的算法还应通过只读 `hif4_get_diagnostics()` 返回接受数、候选总数和回退次数。
- 综合分仍是正式排序主指标；同时维护得分最大、两类 MSE 最小和耗时最小的 Pareto 非支配档案。Linear 专项、Attention 专项或速度版本可以成为后续实现基础，即使它不是综合最高分。
- `20000` 只作为阶段目标和参考线，不自动停止 Runner。Runner 持续迭代，直到用户明确暂停或结束。

## 固定评测

- 官方 compact 5 Linear + 5 Attention 用于六接口、shape 和输出格式自检，不登记算法得分。
- 最多三组内部配置先在确定性等距抽取的 10 Linear + 50 Attention 上筛选，再让前两名进入完整 50 + 250 评测；一至两组配置可直接完整评测。同一阶段的配置共享一次数据加载。
- 只有实际完成 300 例的结果能写入版本账本并参与全局排序；compact 自检、60 例筛选、局部 smoke test 或估算分均不得登记为正式结果。
- 正式数值评测共用全局串行锁，避免多个进程同时加载大数据造成内存与 I/O 抖动。

## 扁平调度

- `max_agents=6`，全局目标组合为 4 个 `explore` + 2 个 `exploit`；配额按正在运行和本轮待派发任务合计，不是每个来源版本各自分配。不存在每个版本最多三个子节点的约束。
- 每个 Agent 只能在完整正式评测与诊断完成后，由报告阶段提出恰好 3 个结构性后续方向：2 个 `explore`、1 个 `exploit`。实现阶段不得提前生成后续方向。方向进入全局候选池，不保证从本版本继续；被否证版本的 `exploit` 也可选择其他已评测 Pareto 版本作为 `based_on`。
- 候选任务记录 `based_on`、算法族、目标问题、证据强度、新颖性、不确定性、预计成本、目标指标与否证条件。全局按“来源质量 + 证据 + 新颖性 + 不确定性 + 算法族历史收益 + 探索奖励 + 停滞转向奖励 - 成本 - 失败惩罚”动态重排；不再把来源版本总分直接当作静态优先级。
- 空槽从全局优先队列直接补位；禁止用纯调参任务凑数。
- 每次并发派发先满足 4:2 角色组合，再在每种角色内覆盖不同 focus 和算法族，最后才按优先级补满或借用剩余槽位，避免高相似候选同时占满 6 个 Agent。
- 正式评测全局串行。Agent 在独立快照中工作，Runner 只导入目标数字版本文件，并独占写入 queue、versions 和正式结果。
- Runner 持续迭代，直到用户明确暂停或结束。

## 研究记忆与防止局部最优

- `.agent/knowledge/experiments.json` 同时保存成功、退化、超时和工作流失败，不只记最高分；每条正式实验带基线、分数差、两类 MSE 差、耗时、假设判定和可复用经验。
- 下一轮提示只注入少量同方向正例、反例、Pareto 起点和停滞长度，防止全历史堆积造成上下文污染。
- `.agent/knowledge/principles.json` 保存研究原则。报告可以 `add`、`reinforce` 或 `challenge` 原则，但新增原则至少需要两个不同算法族的正式结果支持才会激活；固定输入语义和评测完整性规则不可自动撤销。
- 连续没有新综合最优时，提高高新颖性、跨算法族和 `scratch` 方案的优先级；已有收益的算法族仍可继续利用，但其优先级由实际平均增益而不是目录层级决定。

## 单任务状态

```text
queued → running → isolated policy/implementation → structural review
       → at most 3 internal configs → optional 60-case config screening
       → serialized full 300-case evaluation
       → report + 2 explore/1 exploit proposals
       → flat registry record → completed
```

失败任务保留日志。调度器意外终止后用 `recover` 恢复未完成任务：若实现目录已有足够产物，则进入 `implementation_finalize` 完成结构化输出与审查；若完整评测已完成而报告阶段失败，则依赖 `checkpoint.json` 只恢复报告、后续方向与登记，不重复昂贵评测。`environment_failed`、`evaluation_timeout`、`workflow_failed` 与算法结构失败必须分开记录。

Runner 启动前必须通过 `codex --version`，并直接执行目标 Conda 环境内 Python 的绝对路径与 `--version`。正式评测不得再经过 `conda run` 包装层。Python/Codex/评测子进程统一强制 UTF-8。基础设施错误记为 `environment_failed`，不得写成算法失败或更新算法指标。

每个 Codex 子进程必须使用 `--ignore-user-config --ephemeral` 隔离用户级 MCP、提示和会话状态，同时显式指定模型、角色对应 reasoning effort 与 service tier。保留内置 code-mode host；长时间没有 stdout/stderr 事件时按空闲超时终止并进入可恢复失败路径。若已输出合法结构化结果和 `turn.completed` 但 Windows 进程未退出，宽限期后允许回收进程并采用事件流中的结果。

Windows 受限沙箱若触发 `CreateProcessWithLogonW 1385`，允许 Codex 使用 `danger-full-access`，但每个任务仍必须位于独立快照，共享状态仍由 Runner 独占写入。快照不复制 `datasets/` 和 `reference/`；Agent 禁止绕过快照自行访问根数据或执行正式评分。

## 文件结构

- `solution/v0_hessian_repair/`：固定根基线，只包含 `solution.py` 和 `report.md`。
- `solution/v0_<方法概括>/`：保留的其他初始候选，包含 `policy.md`、`solution.py` 和 `report.md`。
- `solution/vN_<方法概括>/`：包含 `policy.md`、`solution.py`、`report.md`，以及可选的 `trials/<config>/solution.py`。
- `.agent/versions.json`：扁平版本账本。
- `.agent/runtime/queue.json`：全局候选任务队列。
- `.agent/knowledge/principles.json`：带证据门禁的研究原则。
- `.agent/knowledge/experiments.json`：正负实验记忆。
- `.agent/knowledge/pareto.json`：综合精度、分项误差和耗时的非支配档案。
- `.agent/knowledge/process_metrics.json`：筛选一致率、正收益率、停滞与单位评测时间收益。

状态命令为 `doctor`、`init`、`seed`、`run`、`recover`、`status`、`audit`、`pause` 和 `resume`。`doctor --deep` 额外校验约 3.7 GB 数据文件的 SHA-256；`status --explain` 展示 Pareto、停滞长度和优先级分解；`audit` 衡量流程自身的筛选一致率与单位评测时间收益；`run --dry-run --once` 最多预览 6 个派发且不修改任务状态。
