# Huawei Competition Workspace

## 目录结构

```text
huawei_competition/
├── .agent/
│   ├── version-tree.json
│   └── skills/
│       ├── hif4-branch/
│       ├── hif4-evaluate/
│       │   ├── SKILL.md
│       │   └── scripts/evaluate.py
│       ├── hif4-policy/
│       └── hif4-report/
├── README.md
├── docs/
│   ├── assets/
│   │   ├── nvfp4_quantization.svg
│   │   └── hif4_quantization.svg
│   ├── 深度调研.md
│   ├── 深度调研.pdf
│   ├── 项目调研.md
│   └── 项目调研.pdf
├── solution/
│   └── v0_hessian_repair/
│       ├── report.md
│       └── solution.py
├── reference/
│   ├── 2026年华为算法大赛-初赛任务书-0819-V2.docx
│   ├── 2026+Huawei+Algorithm+Competition+-+Preliminary+Round+Task+Document-0819-V2.docx
│   ├── 本地调试参考-0818.zip
```

## 说明

- `docs/`：项目调研、网络深度调研及对应 PDF。
- `.agent/skills/`：不同方案版本共享的分支管理、策略、评测与报告能力。
- `solution/`：统一维护算法原型、参考方案、正式版本和评测环境。
- `solution/v0_hessian_repair/`：当前可运行根版本，核心为完整 Hessian 与误差修复。
- `reference/`：任务书、官方本地调试资料和他人提交方案，仅用于研究与对照。

每个版本目录内部固定使用文件名 `solution.py`，便于直接检查和打包。当前版本树从 `v0_hessian_repair` 开始。

## Agent 迭代入口

项目的版本迭代规范见 `AGENTS.md`。单轮实验统一执行：

```powershell
conda run -n huawei_competition_2026 python .agent/skills/hif4-evaluate/scripts/evaluate.py solution/v0_hessian_repair/solution.py solution/<version>/solution.py
```

评测脚本会先运行官方输出格式自检，再输出 Linear/Attention MSE 与任务书最终得分。Report 结构由 `AGENTS.md` 约束。
