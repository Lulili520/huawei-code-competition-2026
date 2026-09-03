---
name: hif4-evaluate
description: Validate and score one or more Huawei HiF4 competition solution.py versions with the project's official-format self-check and local Linear/Attention MSE scorer. Use when testing, comparing, accepting, or rejecting solution versions.
---

# HiF4 Solution Evaluation

Use this skill's maintained script at `.agent/skills/hif4-evaluate/scripts/evaluate.py`; do not recreate scoring logic in temporary scripts.

## Run

From the repository root, run:

```powershell
conda run -n huawei_competition_2026 python .agent/skills/hif4-evaluate/scripts/evaluate.py solution/v0_hessian_repair/solution.py solution/<candidate>/solution.py --datasets-dir datasets/combined
```

For local-agent orchestration, request machine-readable evidence:

```powershell
conda run -n huawei_competition_2026 python .agent/skills/hif4-evaluate/scripts/evaluate.py solution/<candidate>/solution.py --datasets-dir datasets/combined --json-output .agent/runtime/runs/<run>/evaluation.json
```

The evaluator first runs the official `self_check.py` on its compact 10-case
suite to validate interfaces and shapes. A single-config manual invocation then
calculates metrics once on the unified 300-case dataset.

Runner-managed versions use a two-fidelity cascade when three internal configs
exist. One or two configs go directly to full evaluation because the promotion
capacity is already two:

1. evaluate every config on a deterministic, evenly spaced 10 Linear + 50
   Attention screening subset;
2. promote the top two screening configs;
3. evaluate only those configs on the full 50 + 250 set;
4. register only the full result. Screening and full scores are different
   scales and must never be compared as absolute values.

All configs in one stage are passed to one evaluator process so the 3.7 GB
dataset is loaded once per stage instead of once per config.

- Linear Output MSE;
- Attention Output MSE;
- final local score as the sum of per-case MSE-improvement percentages. Since
  the unified set contains 50 Linear and 250 Attention cases, averaging each
  category with weights 1:5 and multiplying by 300 and 100 percentage points
  is algebraically identical to directly summing all 300 case percentages.

Every formal local result uses the immutable unified dataset at
`datasets/combined`: 50 Linear cases and 250 Attention cases. Reports and the
flat registry must record the two observed case counts from evaluator JSON.
The original official mini sample remains reference material and an interface
compatibility source; it is not the current ranking dataset.

Only `.agent/runner.py` may run and register formal local-agent evaluations. Worker Agents may prepare at most three configurations but must not write formal metrics themselves.

The JSON output also records per-case standard/player MSE and percentage-point
gain, score distribution (minimum, P10, median, P90, maximum and negative case
count), phase timings, and selected group indices. Algorithms with internal
candidate search may expose a read-only `hif4_get_diagnostics()` function; its
JSON-serializable counters are captured under `implementation_diagnostics`.

For a non-registering screening diagnostic, use:

```powershell
D:\Miniconda3\envs\huawei_competition_2026\python.exe .agent/skills/hif4-evaluate/scripts/evaluate.py solution/<version>/solution.py --datasets-dir datasets/combined --linear-groups 2 --attention-groups 10
```

Use `--skip-self-check` only for repeated profiling after the same unchanged candidate has already passed the official check in the current experiment.

## Accepting a result

A valid experiment must satisfy all of the following:

1. The candidate is stored at `solution/<version>/solution.py`.
2. Official output-format checks pass with no failed case.
3. The evaluator exits with code 0 and prints both MSE values and the final score.
4. Compare versions by final score first; use Linear and Attention MSE, negative-case count and runtime to identify which path and trade-off caused the change.

The local standard HiF4 encoder is a reproducible approximation because the platform does not publish its internal standard encoder or hidden `MSE_STD`. Never present the local score as the official platform score.

After execution, remove generated `__pycache__` directories from `solution/`; do not retain copied mini-sample data or temporary evaluators in the repository.

When the version belongs to `.agent/versions.json`, use `$hif4-version` to record the displayed MSE and score after updating its report.
