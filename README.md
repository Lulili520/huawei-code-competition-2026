# Huawei HiF4 Competition

本项目研究如何把固定 NVFP4 输入转换为合法 HiF4 参数，并降低 Linear 与
Attention 的最终输出 MSE。

## 项目结构

```text
huawei_competition/
├── AGENTS.md                    # Agent 迭代、证据和评测规范
├── .agent/
│   ├── runner.py               # 扁平版本调度器
│   ├── config.json             # 并发数、解释器和正式评测配置
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
conda run -n huawei_competition_2026 python .agent/skills/hif4-evaluate/scripts/evaluate.py solution/<version>/solution.py --datasets-dir datasets/combined
```

Runner 对同一算法的 3 个内部配置采用分级评测：先在等距抽取的 10 个
Linear + 50 个 Attention 上筛选，再让前 2 名进入完整 300 例。一到两组配置
直接完整评测。一个阶段的全部配置共用一次大数据加载；只有完整结果进入版本账本。

从 `reference/test_sample/` 重新生成统一数据：

```powershell
conda run -n huawei_competition_2026 python .agent/scripts/build_combined_dataset.py
```

## Agent命令

```powershell
conda run -n huawei_competition_2026 python .agent/runner.py doctor
conda run -n huawei_competition_2026 python .agent/runner.py init
conda run -n huawei_competition_2026 python .agent/runner.py seed
conda run -n huawei_competition_2026 python .agent/runner.py status
conda run -n huawei_competition_2026 python .agent/runner.py status --explain
conda run -n huawei_competition_2026 python .agent/runner.py audit
conda run -n huawei_competition_2026 python .agent/runner.py run --dry-run --once
conda run -n huawei_competition_2026 python .agent/runner.py run
conda run -n huawei_competition_2026 python .agent/runner.py pause
conda run -n huawei_competition_2026 python .agent/runner.py resume
conda run -n huawei_competition_2026 python .agent/runner.py recover
```

Runner 数值评测始终串行，版本使用全局连续的 `vN_<方法概括>`，不存在父子树。
最多 6 个 Agent 可并行完成调研、policy 和实现；其轻量快照不复制约 3.7 GB
的 `datasets/`，且 Runner 会阻止目标版本目录以外的写入。

队列不是按来源分数静态排序。它综合来源质量、证据强度、新颖性、不确定性、
预计成本、算法族历史收益和停滞信号动态排队。`status --explain` 可查看每项
优先级分解；`.agent/knowledge/experiments.json` 保存成功和失败经验，
`.agent/knowledge/pareto.json` 保留综合分、分项 MSE 与耗时上的非支配版本。
`audit` 进一步报告筛选冠军一致率、正收益率、停滞长度和单位正式评测小时的正向分数增量，用真实运行数据评估流程本身。

项目当前保留暂停标记 `.agent/STOP`。`seed` 只建立初始任务，`resume` 后执行
`run` 才会开始持续迭代。
