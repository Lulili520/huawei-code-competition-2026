# HiF4 Local-Agent 迭代规范

## 任务模型

`.agent/runner.py` 最多并发 6 个本地 Codex Agent。每个 Agent 独立完成一个完整算法版本：调研与问题分析、`policy.md`、算法实现、最多三组内部超参数配置、官方 10 例评测、`report.md` 和版本树记录。

不存在单独的“发现 Agent”和“实现 Agent”。一个版本完成后立即按真实得分重排队列，无须等待同批任务全部结束。

## 固定边界

NVFP4 反量化固定为 E2M1 值乘对应 E4M3 scale，每 16 个连续值共享一个 scale，最后恢复原 shape。不得修改数值语义、scale 对应关系、16 值分组，或用重新估计的 scale 替换输入 scale。

## 算法版本

- 正式版本必须对应可证伪的结构级算法策略；纯 alpha、gain、阈值、系数、候选数或倍率修改不得建立版本。
- 算法确定后，同一版本内部允许最多 2～3 组有理论依据的超参数配置。主配置放在 `solution.py`，其余放在 `trials/<config>/solution.py`。
- 父版本定义比较关系，不强制继承代码。实现基础可以是 `parent`、`v0` 或 `scratch`，但 `policy.md` 必须解释选择原因。

## 证据标准

- 明确区分 `已验证事实`、`理论推导` 和 `待验证假设`，不得把推测写成事实。
- 项目事实定位到文件与函数/行号，或真实评测配置与精确指标。
- 外部依据优先使用原始论文、标准和官方资料，记录标题、作者/机构、年份与直达链接。
- 每项算法动作必须有完整闭环：`问题证据 → 理论依据 → 算法动作 → 目标指标 → 否证条件`。
- 理论依据只证明机制或方向时，不得宣称必然提升；最终结论以官方 10 例实测为准。
- 报告逐项把假设标记为 `结果支持`、`结果否证` 或 `证据不足`，不得从本地样例直接外推隐藏集。

## 树与调度

- `max_agents=6`；每个父版本最多 3 个直接算法子版本。
- 初始仅有 v0，因此 `seed` 只创建 3 个完整算法任务，剩余槽位等待。
- 每个完成评测的 Agent 必须经过调研提出恰好 3 个结构性后续方向。提升、持平或退化只影响队列优先级，不阻断后续研究。
- 子任务优先级使用父版本的实测总分；每次派发前重新排序，高分版本优先。
- Runner 持续迭代直到用户明确暂停或结束。空槽优先从高分版本的结构方向补充；禁止用纯调参任务凑数。
- 正式评测全局串行，仅使用固定 5 个 Linear 和 5 个 Attention 样例。
- Agent 在独立快照中工作；runner 只导入目标版本文件，并独占写入 queue、version-tree 和正式结果。

## 单任务状态

```text
queued → running → isolated policy/implementation → structural review
       → at most 3 internal configs → serialized official evaluation
       → report → tree record → completed
```

失败任务保留日志。调度器意外终止后用 `recover` 将未完成的 `running` 任务重新排队。

Runner 启动前必须通过 `codex --version`，并直接执行目标 Conda 环境内 Python 的绝对路径与 `--version`。正式评测不得再经过 `conda run` 包装层。Python/Codex/评测子进程统一强制 UTF-8。可执行文件缺失等基础设施错误记为 `environment_failed`，不得写成算法失败、不得更新算法指标；修复环境后由 `recover` 重新排队。

Windows 上受限沙箱若触发 `CreateProcessWithLogonW 1385`，允许 Codex 使用 `danger-full-access`，但不得削弱项目自身隔离：每个任务仍只在独立仓库快照中工作，runner 只导入该版本的 `policy.md`、`solution.py`、`trials/` 和 `report.md`，共享状态仍由 runner 独占写入。

## 文件与命令

根版本 `solution/v0_hessian_repair/` 只包含 `solution.py` 和 `report.md`。新算法版本包含 `policy.md`、`solution.py`、`report.md`，以及可选的 `trials/<config>/solution.py`。

```powershell
conda run -n huawei_competition_2026 python .agent/runner.py doctor
conda run -n huawei_competition_2026 python .agent/runner.py init
conda run -n huawei_competition_2026 python .agent/runner.py seed
conda run -n huawei_competition_2026 python .agent/runner.py run
conda run -n huawei_competition_2026 python .agent/runner.py recover
```

状态命令：`status`、`pause`、`resume`。`run --dry-run --once` 只验证派发顺序。
