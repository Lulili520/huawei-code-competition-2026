# Huawei HiF4 Competition

本项目研究如何把固定 NVFP4 输入转换为合法 HiF4 参数，并降低 Linear 与
Attention 的最终输出 MSE。自主流程只以完整 50 Linear + 250 Attention 的
300 例正式得分作为进展依据；目标是达到 `20000`，compact 自检和 60 例筛选
都不算正式成绩。

## 项目结构

```text
huawei_competition/
├── AGENTS.md                    # Agent 迭代、证据和评测规范
├── .agent/
│   ├── runner.py               # 扁平版本调度器
│   ├── config.json             # 6并发、4:2搜索组合、解释器和正式评测配置
│   ├── versions.json           # vN_<方法概括> 扁平版本账本
│   ├── core/                   # 完整性门禁、Pareto与自适应优先级
│   ├── knowledge/              # 研究原则、正负实验记忆和Pareto档案
│   ├── prompts/                # 实现、只读审查和报告提示
│   ├── scripts/
│   │   └── build_combined_dataset.py
│   ├── skills/                 # policy、评测、report和版本登记能力
│   └── tests/                  # 调度、评分、诊断和防篡改单元测试
├── datasets/
│   └── combined/               # 统一300例本地回归集（不提交Git）
│       ├── linear.pt           # 10组、50例
│       ├── attn.pt             # 50组、250例
│       └── manifest.json       # 数量、大小和SHA-256
├── solution/
│   ├── v0_hessian_repair/      # 已完成300例评测的固定基线
│   ├── v0_softmax_aware_qk/    # 当前300例综合最优
│   ├── v0_alternating_joint_fit/ # 当前300例Linear MSE最优
│   └── vN_<method>/            # 后续实验版本
├── docs/                       # 项目调研与数学分析
└── reference/                  # 任务书、官方资料和原始测试分卷（不提交Git）
    └── test_sample/
```

## 评测入口

手工复核正式版本时固定使用统一 300 例（50 个 Linear、250 个 Attention）：

```powershell
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/skills/hif4-evaluate/scripts/evaluate.py solution/<version>/solution.py --datasets-dir datasets/combined
```

Runner 对同一算法的 3 个内部配置采用分级评测：先在等距抽取的 10 个
Linear + 50 个 Attention 上筛选，再让前 2 名进入完整 300 例。一到两组配置
直接完整评测。一个阶段的全部配置共用一次大数据加载；只有完整结果进入版本账本。

从 `reference/test_sample/` 重新生成统一数据：

```powershell
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/scripts/build_combined_dataset.py
```

## Agent命令

```powershell
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/runner.py doctor
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/runner.py init
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/runner.py seed
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/runner.py status
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/runner.py status --explain
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/runner.py audit
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/runner.py run --dry-run --once
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/runner.py run
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/runner.py pause
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/runner.py resume
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/runner.py recover
```

Runner 数值评测始终串行，版本使用全局连续的 `vN_<方法概括>`，不存在父子树。
Runner 维护 6 个工作槽位：全局目标组合是 4 个 `explore` 与 2 个 `exploit`，
并把正在运行的任务计入配额。探索任务验证新根因或跨算法族机制；利用任务必须
沿已有正式正证据或 Pareto 分项优势深化。候选充足时空槽立即补位；某一角色
候选不足时允许另一角色借槽，但不会用纯调参任务凑数。轻量快照不复制约 3.7 GB
的 `datasets/`，且 Runner 会阻止目标版本目录以外的写入。

每个版本只有在完整正式评测和诊断完成后，报告阶段才生成 3 个后续结构方向：
2 个 `explore`、1 个 `exploit`。每个方向都要给出
`问题证据 → 理论依据 → 算法动作 → 目标指标 → 否证条件`，并把根因落实到
Linear 的三项误差传播、Attention 的中心化 logits/Softmax Jacobian/V 路径，
或 HiF4 格式的 clipping/分辨率矛盾。实现阶段不提前猜测后续方向。

队列不是按来源分数静态排序。它综合来源质量、证据强度、新颖性、不确定性、
预计成本、算法族历史收益和停滞信号动态排队。`status --explain` 可查看每项
优先级分解；`.agent/knowledge/experiments.json` 保存成功和失败经验，
`.agent/knowledge/pareto.json` 保留综合分、分项 MSE 与耗时上的非支配版本。
`audit` 进一步报告筛选冠军一致率、正收益率、停滞长度和单位正式评测小时的正向分数增量，用真实运行数据评估流程本身。

Codex 工作进程使用 `--ignore-user-config --ephemeral`，显式指定模型、角色推理
强度和 service tier，避免用户级 MCP 或历史会话干扰；600 秒没有事件即按空闲
超时处理。若实现已留下足够产物，`recover` 从 `implementation_finalize` 继续；
若完整评测已写入 checkpoint，则只恢复报告、后续方向和登记，不重复跑 300 例。
当且仅当某个完整 300 例正式得分达到 `20000` 时，Runner 自动停止新派发，
已经开始的任务会安全收尾。

项目当前保留暂停标记 `.agent/STOP`。`seed` 只建立初始任务，`resume` 后执行
`run` 才会开始持续迭代。
