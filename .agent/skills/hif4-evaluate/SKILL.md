---
name: hif4-evaluate
description: Validate and score one or more Huawei HiF4 competition solution.py versions with the project's official-format self-check and local Linear/Attention MSE scorer. Use when testing, comparing, accepting, or rejecting solution versions.
---

# HiF4 Solution Evaluation

Use this skill's maintained script at `.agent/skills/hif4-evaluate/scripts/evaluate.py`; do not recreate scoring logic in temporary scripts.

## Run

From the repository root, run:

```powershell
conda run -n huawei_competition_2026 python .agent/skills/hif4-evaluate/scripts/evaluate.py solution/v0_hessian_repair/solution.py solution/<candidate>/solution.py
```

For local-agent orchestration, request machine-readable evidence:

```powershell
conda run -n huawei_competition_2026 python .agent/skills/hif4-evaluate/scripts/evaluate.py solution/<candidate>/solution.py --json-output .agent/runtime/runs/<run>/evaluation.json
```

The evaluator first runs the official `self_check.py` embedded in `reference/本地调试参考-0818.zip`, then calculates:

- Linear Output MSE;
- Attention Output MSE;
- final local score from the task formula, summed over test cases.

The evaluator uses only the ten samples supplied by the official local-debug package: five Linear cases and five Attention cases. Do not append synthetic cases or silently replace these tensors. Every version report and version-tree metric must be produced from this same fixed ten-case suite.

Only `.agent/runner.py` may run and register formal local-agent evaluations. Worker Agents may prepare at most three configurations but must not write formal metrics themselves.

Use `--skip-self-check` only for repeated profiling after the same unchanged candidate has already passed the official check in the current experiment.

## Accepting a result

A valid experiment must satisfy all of the following:

1. The candidate is stored at `solution/<version>/solution.py`.
2. Official output-format checks pass with no failed case.
3. The evaluator exits with code 0 and prints both MSE values and the final score.
4. Compare versions by final score first; use Linear and Attention MSE to identify which path caused the change.

The local standard HiF4 encoder is a reproducible approximation because the platform does not publish its internal standard encoder or hidden `MSE_STD`. Never present the local score as the official platform score.

After execution, remove generated `__pycache__` directories from `solution/`; do not retain copied mini-sample data or temporary evaluators in the repository.

When the version belongs to `.agent/version-tree.json`, use `$hif4-branch` to record the displayed MSE and score after updating its report.
